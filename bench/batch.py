"""Batched decoding: total tokens/sec across B identical-length prompts, KV vs speculative.

  uv run python bench/batch.py --device mps --target gpt2-xl --draft gpt2 --batches 1 2 4 8 --gamma 4
"""
from __future__ import annotations

import argparse, json, statistics, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from specdecode.model import GPT  # noqa: E402
from specdecode.generate import generate_kv, generate_speculative_batched  # noqa: E402
from bench.bench import hardware, timed, PROMPT  # noqa: E402

PROMPTS = [
    PROMPT,
    "The Treaty of Westphalia in 1648 ended the Thirty Years' War and is often described as the "
    "origin of the modern state system. Whether that description is accurate has been argued over "
    "by historians and political scientists for decades, and the argument says as much about the "
    "present as about the seventeenth century. The treaty itself was a pair of agreements signed in "
    "two towns in Westphalia after years of negotiation among dozens of delegations. " * 2,
    "A sourdough starter is a stable culture of wild yeast and lactic acid bacteria living in a "
    "paste of flour and water. Kept at room temperature and fed daily, it will raise bread "
    "indefinitely. The yeast produce carbon dioxide, which inflates the dough, and the bacteria "
    "produce the acids that give the bread its tang and keep competing microbes out. " * 2,
    "In computer architecture, the memory wall refers to the growing gap between processor speed "
    "and the speed at which data can be moved from main memory. Caches, prefetching, and "
    "out-of-order execution exist largely to hide this gap. For workloads that stream through more "
    "data than any cache can hold, none of those tricks help, and performance is set by bandwidth. " * 2,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--target", default="gpt2-xl")
    ap.add_argument("--draft", default="gpt2")
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--batches", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results/batch.jsonl")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    base_prompts = [tok.encode(p)[: args.prompt_tokens] for p in PROMPTS]
    assert all(len(p) == args.prompt_tokens for p in base_prompts)
    target = GPT.from_pretrained(args.target, device=args.device, dtype=dtype)
    draft = GPT.from_pretrained(args.draft, device=args.device, dtype=dtype)
    hw = hardware()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        for B in args.batches:
            prompts = torch.tensor([base_prompts[i % len(base_prompts)] for i in range(B)], device=args.device)
            refs = [generate_kv(target, prompts[b:b + 1], args.new_tokens)[0][0] for b in range(B)]
            for name, fn in [
                ("kv_cache", lambda: generate_kv(target, prompts, args.new_tokens)),
                ("speculative", lambda: generate_speculative_batched(target, draft, prompts, args.new_tokens, gamma=args.gamma)),
            ]:
                timed(fn, args.device)  # warmup
                times, stats = [], None
                for _ in range(args.repeats):
                    t, out, stats = timed(fn, args.device)
                    times.append(t)
                outs = [out[b] for b in range(B)] if isinstance(out, list) else [out[b] for b in range(B)]
                identical = all(torch.equal(o, r) for o, r in zip(outs, refs))
                total = B * args.new_tokens
                med = statistics.median(times)
                row = {**hw, "device": args.device, "dtype": args.dtype, "target": args.target, "draft": args.draft,
                       "target_params_M": round(target.num_params() / 1e6, 1), "draft_params_M": round(draft.num_params() / 1e6, 1),
                       "prompt_tokens": args.prompt_tokens, "new_tokens": args.new_tokens, "temperature": 0.0,
                       "warmup": 1, "repeats": args.repeats, "batch": B, "method": name,
                       "gamma": args.gamma if name == "speculative" else None,
                       "wall_s_median": round(med, 4), "tokens_per_s": round(total / med, 2),
                       "tokens_per_s_per_row": round(args.new_tokens / med, 2),
                       "target_forwards": stats.target_forwards, "draft_forwards": stats.draft_forwards,
                       "acceptance_rate": round(stats.acceptance_rate, 4) if name == "speculative" else None,
                       "mean_accepted_per_round": round(stats.accepted / stats.rounds / B, 3) if name == "speculative" else None,
                       "output_identical_to_kv_greedy": identical,
                       "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
                f.write(json.dumps(row) + "\n")
                print(f"B={B} {name:12s} {row['tokens_per_s']:8.2f} tok/s total  {row['tokens_per_s_per_row']:7.2f}/row  "
                      f"target_fwd={stats.target_forwards:3d}  acc={row['acceptance_rate']}  identical={identical}")


if __name__ == "__main__":
    main()
