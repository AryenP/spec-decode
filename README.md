# spec-decode

KV-cache and speculative decoding, implemented from scratch on GPT-2 in under 400
lines of PyTorch, with tokens/sec measured on named hardware.

HuggingFace supplies only the weights and the tokenizer. The transformer
forward pass, the pre-allocated KV cache with rollback, and all three decoders
(full recompute, cached, speculative) are in [`specdecode/`](specdecode/) and
every line can be defended in an interview.

## What it proves

| claim | how it is checked |
|---|---|
| The from-scratch GPT-2 is correct | logits match `transformers` to 1e-3 ([test](tests/test_correctness.py)) |
| The KV cache is exact, not approximate | cached logits match full recompute to 1e-4, both chunked and one token at a time |
| Speculative greedy output is the target's greedy output | `torch.equal` on every benchmark row (`identical` column) |
| Speculative sampling (T>0) preserves the target distribution | 400-draw frequency test against the target's own softmax, within 4 standard errors |
| Numbers are reproducible | every JSONL row carries chip, memory, torch version, dtype, model sizes, prompt/new tokens, warmup, repeats |

## Results

All numbers measured on an **Apple M2 Pro, 32 GB unified memory**, PyTorch 2.14.1,
MPS backend, fp32, median of 3 runs after 1 warmup. Raw rows with every
configuration field are in [`results/`](results/). Reproduce with the commands
under [Run it](#run-it).

### Greedy decoding, gpt2-xl target (1.56 B) with gpt2 draft (124 M)

**gpt2-xl** (1557.6 M) target, **gpt2** (124.4 M) draft · Apple M2 Pro, 32 GB · mps float32 · torch 2.14.1 · prompt 64 tok, 128 new tok, T=0.0 · median of 3 after 1 warmup

| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |
|---|---|---|---|---|---|---|---|---|---|
| no_cache | — | 8.4 | 0.43× | 15.14 | 128 | 0 | — | — | yes |
| kv_cache | — | 19.8 | 1.00× | 6.48 | 128 | 0 | — | — | yes |
| speculative | 2 | 34.9 | 1.77× | 3.66 | 51 | 101 | 0.79 | 1.58 | yes |
| speculative | 4 | 41.7 | 2.11× | 3.07 | 35 | 134 | 0.71 | 2.79 | yes |
| speculative | 6 | 37.9 | 1.92× | 3.37 | 30 | 173 | 0.58 | 3.45 | yes |
| speculative | 8 | 34.8 | 1.76× | 3.68 | 27 | 203 | 0.51 | 3.96 | yes |

- The KV cache alone is **2.3×** over full recompute at 128 new tokens; the gap
  grows with sequence length because recompute is O(n²).
- Speculative decoding on top is **2.11× over the KV baseline at γ=4**
  (41.7 vs 19.8 tok/s) with the output bit-identical to target greedy on every
  row. Target forwards drop from 128 to 35.
- γ=4 is the sweet spot here. Longer drafts raise accepted tokens per round
  (2.8 → 4.0) but the per-token acceptance rate falls (0.71 → 0.51), so extra
  draft forwards are increasingly wasted on tokens that get rejected.

### Sampling at temperature 0.8, same models

**gpt2-xl** (1557.6 M) target, **gpt2** (124.4 M) draft · Apple M2 Pro, 32 GB · mps float32 · torch 2.14.1 · prompt 64 tok, 128 new tok, T=0.8 · median of 3 after 1 warmup

| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |
|---|---|---|---|---|---|---|---|---|---|
| kv_cache | — | 19.3 | 1.00× | 6.64 | 128 | 0 | — | — | n/a |
| speculative | 4 | 32.3 | 1.67× | 3.97 | 44 | 172 | 0.50 | 1.98 | n/a |

Sampling lowers the acceptance rate (0.71 → 0.50 at γ=4) because the draft and
target now have to agree on a random draw rather than an argmax, so the speedup
falls to **1.67×**. The output distribution is still exactly the target's; the
test suite checks this empirically.

### The bug the first version had, kept as evidence

**gpt2-xl** (1557.6 M) target, **gpt2** (124.4 M) draft · Apple M2 Pro, 32 GB · mps float32 · torch 2.14.1 · prompt 64 tok, 128 new tok, T=0.0 · median of 3 after 1 warmup

| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |
|---|---|---|---|---|---|---|---|---|---|
| no_cache | — | 8.4 | 0.43× | 15.18 | 128 | 0 | — | — | yes |
| kv_cache | — | 19.6 | 1.00× | 6.54 | 128 | 0 | — | — | yes |
| speculative | 2 | 16.7 | 0.85× | 7.68 | 130 | 130 | 0.49 | 0.98 | yes |
| speculative | 4 | 17.3 | 0.88× | 7.40 | 110 | 218 | 0.34 | 1.34 | yes |
| speculative | 6 | 16.1 | 0.82× | 7.95 | 102 | 302 | 0.26 | 1.53 | yes |
| speculative | 8 | 14.1 | 0.72× | 9.09 | 100 | 392 | 0.20 | 1.58 | yes |

Version 1 rolled both caches back to the committed prefix after each round and
re-fed the accepted tokens to the target, so every round cost **two** target
forwards. At γ=2 that is 130 target forwards for 128 tokens, more than plain
decoding, and speculative ran at 0.72–0.88× of the KV baseline. The fix is to
feed the previous round's last committed token together with the new drafts in
the single verify pass and roll back only over the rejected positions. (The v1
run also used a repetitive synthetic prompt, which explains its lower
acceptance rate; the doubled forward count is independent of the prompt.)

### Not measured

gpt2-large as a target, bf16, CPU, and a CUDA GPU. The gpt2-large download
failed on a full disk; the others need hardware this machine does not have.
Speedups will differ on CUDA, where per-kernel launch overhead is lower and the
draft's γ forwards are relatively cheaper.

## How speculative decoding works here

Decoding is memory-bandwidth bound: each token reads every parameter once.
A 1.5 B target reads ~6 GB per step in fp32; a 124 M draft reads ~0.5 GB.
Speculative decoding (Leviathan, Kalman & Matias 2023) lets the draft propose
γ tokens, then the target scores all γ+1 positions in **one** forward pass,
which costs about the same as scoring one. Accepted draft tokens are free
target tokens.

```
round:  draft ──γ small forwards──▶ [d1 … dγ]
        target ─1 forward over γ positions─▶ p(·|prefix,d1..di) for every i
        accept d_i while  rand < min(1, p_i(d_i) / q_i(d_i))      (greedy: argmax match)
        first rejection → resample from  norm(max(0, p_i − q_i)),  stop
        all accepted  → bonus token from the target's last position
        roll both KV caches back to the committed length
```

The accept/reject rule is the whole trick: it is a rejection sampler whose
stationary output is exactly the target distribution, so the draft can be
arbitrarily bad without changing *what* comes out, only *how fast*.

Expected speedup per round is (1 − α^(γ+1)) / (1 − α) target-tokens for the
price of one target forward plus γ draft forwards, where α is the per-token
acceptance rate. When the draft is not much smaller than the target (see the
gpt2-medium row) the draft forwards eat the gain.

## Run it

```bash
uv sync
uv run pytest                                   # ~100 s on CPU, downloads gpt2 + gpt2-medium
uv run python bench/bench.py --device mps --target gpt2-xl --draft gpt2 \
    --prompt-tokens 64 --new-tokens 128 --gammas 2 4 6 8 --repeats 3
uv run python bench/report.py results/*.jsonl   # markdown tables
```

## Layout

```
specdecode/model.py     GPTConfig, KVCache (append / commit / rollback), GPT with HF weight loader
specdecode/generate.py  generate_no_cache, generate_kv, generate_speculative, GenStats
tests/                  correctness: HF parity, cache exactness, greedy identity, sampling distribution
bench/bench.py          timed runs, JSONL rows with full configuration
bench/report.py         JSONL → markdown
results/                raw rows from the runs quoted above
```

## Failure modes and limits

- Batch size 1 only for the speculative path. Batched speculative decoding
  needs per-sequence acceptance lengths and ragged cache rollback.
- fp32 only in the quoted runs. GPT-2 in fp16 overflows in the attention
  logits of later layers; bf16 on MPS is supported by the code but not measured.
- `torch.multinomial` has no generator support on MPS, so sampling happens on
  CPU. Negligible at batch 1; it would matter for large batches.
- Sequence length is capped by GPT-2's 1024 learned positions.
- MPS kernel-launch overhead dominates for small models, which is why the
  draft's γ forwards are relatively expensive here compared with a CUDA GPU.

## References

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. arXiv:2211.17192
- Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* arXiv:2302.01318
- Radford et al. *Language Models are Unsupervised Multitask Learners.* 2019 (GPT-2)
