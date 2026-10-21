# Dead ends, one line each

- **v1 speculative decoder**: two target forwards per round, 0.85x of KV baseline. Rows kept in `results/v1-two-target-forwards-*.jsonl`. Fixed in v2.
- **Synthetic repeated prompt**: a sentence repeated eight times as the benchmark prompt. Acceptance rate 0.49 vs 0.79 on prose. Replaced with a paragraph of ordinary text; the v1 rows still carry the old prompt so they aren't acceptance-comparable.
- **gpt2-large as target on the laptop**: download died on a full disk. Measured on the 4090 instead.
- **A100 on RunPod**: create refused, no stock on the host it picked. Used a 4090.
- **bf16 / fp16 on MPS**: the bf16 gpt2-xl run produced no row in 7 minutes. Micro-measurement on gpt2 small, single cached token on the M2 Pro: fp32 6.8 ms, bf16 25.4 ms, fp16 26.8 ms. Half precision is 3.7x slower than fp32 on this backend for these shapes, so the half-precision rows come from the 4090 only.
- **v2 on CUDA**: speculative γ=4 at 0.79x of KV on the 4090 with the forward pass nearly flat in position count. Host overhead, not the algorithm. v3 fixed it; v2 rows in `results/v2-cuda/`.
- **bf16 no_cache beats KV at 128 tokens on the 4090**: 150 vs 114 tok/s. The per-layer slice copies in `KVCache.append` (96 small kernels per step) cost more than one flash-attention prefill when compute is cheap. Cache wins at 512 tokens. A fused append or CUDA graphs would fix it.
- **v3 on MPS**: 10% slower than v2 at γ=4 while 2.3x faster on CUDA. Not profiled.
