"""Benchmark the three decoders. Every row carries its full configuration.

Usage:
  uv run python bench/bench.py --device mps --target gpt2-xl --draft gpt2 \
      --prompt-tokens 64 --new-tokens 128 --gammas 2 4 6 8 --repeats 3
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from specdecode.model import GPT  # noqa: E402
from specdecode.generate import generate_no_cache, generate_kv, generate_speculative, generate_lookup  # noqa: E402

PROMPT = (
    "The Industrial Revolution began in Britain in the late eighteenth century and "
    "spread across Europe and North America over the following decades. It transformed "
    "agriculture, manufacturing, mining, and transport, and it changed where people "
    "lived and how they worked. Historians still debate why it started where it did. "
    "Some point to cheap coal and expensive labour, which made machines worth building. "
    "Others emphasise property rights, patent law, and a culture of practical "
    "experiment. A third group stresses access to colonial markets and raw materials. "
    "Whatever the cause, the consequences were enormous. Cities grew rapidly, "
    "and new social classes emerged. The factory system imposed a new discipline of "
    "time. Wages rose only slowly at first, and living conditions in the new "
    "industrial towns were often grim. Over the nineteenth century, however, real "
    "incomes began a sustained climb that has continued, with interruptions, to the "
    "present day. "
)


def hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if sys.platform == "darwin":
        info["chip"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        info["memory_gb"] = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) // 2**30
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def timed(fn, device):
    sync(device)
    t0 = time.perf_counter()
    out, stats = fn()
    sync(device)
    return time.perf_counter() - t0, out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--target", default="gpt2-xl")
    ap.add_argument("--draft", default="gpt2")
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--gammas", type=int, nargs="*", default=[2, 4, 6, 8])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--skip-no-cache", action="store_true")
    ap.add_argument("--lookup-gammas", type=int, nargs="*", default=[], help="also run prompt-lookup decoding (no draft model) at these gammas")
    ap.add_argument("--ngram", type=int, default=3)
    ap.add_argument("--out", default="results/results.jsonl")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ids = tok.encode(PROMPT)[: args.prompt_tokens]
    assert len(ids) == args.prompt_tokens
    prompt = torch.tensor([ids], device=args.device)

    print(f"loading {args.target} and {args.draft} on {args.device} {args.dtype}", file=sys.stderr)
    target = GPT.from_pretrained(args.target, device=args.device, dtype=dtype)
    draft = GPT.from_pretrained(args.draft, device=args.device, dtype=dtype)
    hw = hardware()
    base = {
        **hw,
        "device": args.device,
        "dtype": args.dtype,
        "target": args.target,
        "target_params_M": round(target.num_params() / 1e6, 1),
        "draft": args.draft,
        "draft_params_M": round(draft.num_params() / 1e6, 1),
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "temperature": args.temperature,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "kv_bytes_per_token_target": target.new_cache(1, 1).bytes_per_token(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    configs = []
    if not args.skip_no_cache:
        configs.append(("no_cache", None, lambda: generate_no_cache(target, prompt, args.new_tokens, args.temperature)))
    configs.append(("kv_cache", None, lambda: generate_kv(target, prompt, args.new_tokens, args.temperature)))
    for g in args.gammas:
        configs.append((f"speculative", g, (lambda g=g: generate_speculative(target, draft, prompt, args.new_tokens, gamma=g, temperature=args.temperature))))
    for g in args.lookup_gammas:
        configs.append((f"lookup_n{args.ngram}", g, (lambda g=g: generate_lookup(target, prompt, args.new_tokens, gamma=g, ngram=args.ngram, temperature=args.temperature))))

    # Greedy reference every method must reproduce exactly.
    ref_ids = generate_kv(target, prompt, args.new_tokens, 0.0)[0] if args.temperature == 0.0 else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out_path.open("a") as f:
        for name, gamma, fn in configs:
            for _ in range(args.warmup):
                timed(fn, args.device)
            times, stats = [], None
            for _ in range(args.repeats):
                t, out, stats = timed(fn, args.device)
                times.append(t)
            identical = bool(torch.equal(out, ref_ids)) if ref_ids is not None else None
            row = {
                **base,
                "method": name,
                "gamma": gamma,
                "wall_s_median": round(statistics.median(times), 4),
                "wall_s_all": [round(t, 4) for t in times],
                "tokens_per_s": round(args.new_tokens / statistics.median(times), 2),
                "target_forwards": stats.target_forwards,
                "draft_forwards": stats.draft_forwards,
                "acceptance_rate": round(stats.acceptance_rate, 4) if gamma else None,
                "mean_accepted_per_round": round(stats.mean_accepted_per_round, 3) if gamma else None,
                "output_identical_to_kv_greedy": identical,
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            label = f"{name}" + (f" γ={gamma}" if gamma else "")
            acc = f" acc={row['acceptance_rate']:.2f} ({row['mean_accepted_per_round']:.2f}/round)" if gamma else ""
            print(f"{label:18s} {row['tokens_per_s']:7.2f} tok/s  {row['wall_s_median']:.2f}s  "
                  f"target_fwd={row['target_forwards']:3d}{acc}  identical={identical}")
    kv = next(r for r in rows if r["method"] == "kv_cache")["tokens_per_s"]
    print(f"\nspeedup vs kv_cache: " + ", ".join(
        f"{r['method']} γ={r['gamma']}: {r['tokens_per_s']/kv:.2f}x" for r in rows if r["gamma"]))


if __name__ == "__main__":
    main()
