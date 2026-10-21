"""Is decoding bandwidth-bound? Time one cached target forward over k new
positions for k in 1..64. If the curve is flat, scoring gamma+1 draft tokens
costs the same as scoring one, which is the whole premise of speculative
decoding. Where it bends is where longer gamma stops paying.

  uv run python bench/scaling.py --device mps --model gpt2-xl
"""
from __future__ import annotations

import argparse, json, statistics, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from specdecode.model import GPT  # noqa: E402
from bench.bench import hardware, sync  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--model", default="gpt2-xl")
    ap.add_argument("--prefix", type=int, default=128, help="tokens already in the cache")
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default="results/scaling.jsonl")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    model = GPT.from_pretrained(args.model, device=args.device, dtype=dtype)
    hw = hardware()
    weight_bytes = model.num_params() * torch.tensor([], dtype=dtype).element_size()
    prefix = torch.randint(0, model.cfg.vocab_size, (1, args.prefix), device=args.device)
    rows = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        for k in args.ks:
            cache = model.new_cache(1, args.prefix + k)
            model(prefix, cache)
            new = torch.randint(0, model.cfg.vocab_size, (1, k), device=args.device)
            times = []
            for i in range(args.repeats + 3):
                cache.rollback(0)
                cache.lengths[:] = args.prefix
                sync(args.device)
                t0 = time.perf_counter()
                model(new, cache)
                sync(args.device)
                if i >= 3:
                    times.append(time.perf_counter() - t0)
            ms = statistics.median(times) * 1e3
            row = {**hw, "device": args.device, "dtype": args.dtype, "model": args.model,
                   "params_M": round(model.num_params() / 1e6, 1), "prefix_tokens": args.prefix,
                   "k_positions": k, "ms_median": round(ms, 3), "ms_per_position": round(ms / k, 3),
                   "ms_p10": round(sorted(times)[len(times) // 10] * 1e3, 3),
                   "ms_p90": round(sorted(times)[-max(1, len(times) // 10)] * 1e3, 3),
                   "implied_GBps_if_bandwidth_bound": round(weight_bytes / (ms / 1e3) / 1e9, 1),
                   "repeats": args.repeats}
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            print(f"k={k:3d}  {ms:8.2f} ms   {ms/k:7.2f} ms/pos   x{ms/rows[0]['ms_median']:.2f} vs k=1")


if __name__ == "__main__":
    main()
