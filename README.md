# spec-decode

GPT-2 decoding written from scratch to find out what speculative decoding actually buys, measured on a laptop and on a GPU. On an RTX 4090 it's 1.5x over a plain KV cache with a draft model, and 2.7x with no draft model at all once the generation is 512 tokens long. On my M2 Pro it's 1.9x. In fp32 the output is byte-identical to the big model decoding on its own.

HuggingFace is only used for the weights and the tokenizer. The forward pass, the KV cache with per-row rollback, and the decoders are in `specdecode/`, about 600 lines. Every number below has its full configuration in a row under `results/`, and the two bugs that the numbers caught are kept there too.

## Numbers

gpt2-xl (1.56B) is the target, gpt2 (124M) the draft. 64-token prompt, greedy, fp32, median of 3 runs after a warmup. M2 Pro is 32 GB, PyTorch 2.14.1 on MPS. The 4090 is a RunPod secure-cloud pod, PyTorch 2.8.0+cu128.

| tok/s, 128 new tokens | M2 Pro | RTX 4090 |
|---|---|---|
| recompute everything | 7.8 | 52.6 |
| KV cache | 19.6 | 106.6 |
| speculative, best γ | 37.2 (γ=4, **1.90x**) | 156.5 (γ=6, **1.47x**) |
| prompt lookup, no draft model, γ=4 | 30.2 (1.55x) | 161.1 (1.51x) |

| tok/s, 512 new tokens | M2 Pro | RTX 4090 |
|---|---|---|
| KV cache | 18.9 | 94.5 |
| speculative γ=4 | 37.9 (**2.01x**) | 184.7 (**1.95x**) |
| prompt lookup γ=4 | 49.8 (**2.63x**) | 254.5 (**2.69x**) |

Longer generations help both methods for the same reason: the text starts repeating itself, acceptance climbs to 0.91, and lookup finds its n-grams. Prompt lookup is Saxena's trick of drafting from the sequence so far instead of a model. It costs nothing per draft, so on the 4090 it beats the 124M draft model even at 128 tokens.

Full tables per run, including γ sweeps, are in `results/tables.md`.

## Is decoding really bandwidth-bound here?

The premise of speculative decoding is that scoring γ+1 positions costs about the same as scoring one. `bench/scaling.py` measures one cached forward over k new positions:

| k | M2 Pro ms | RTX 4090 ms | 4090 vs k=1 |
|---|---|---|---|
| 1 | 56.3 | 9.5 | 1.00x |
| 2 | 70.0 | 13.4 | 1.42x |
| 8 | 84.7 | 14.2 | 1.50x |
| 64 | 91.2 | 15.9 | 1.68x |

Flat enough: 64 positions for 1.7x the price of one. Two things I didn't expect. The jump from k=1 to k=2 is the attention mask: the single-token path skips building one, every other path pays for it. And the k=1 step on the 4090 implies 659 GB/s against a 1008 GB/s card, while bf16 is only 1.2x faster than fp32, so at 1.5B parameters a single step on this GPU is partly kernel-launch bound, not purely bandwidth bound. CUDA graphs would be the next thing to try; I haven't.

## The two bugs the tables caught

**v1: two target forwards per round.** After verifying, I rolled both caches back to the committed prefix and re-fed the accepted tokens to get the next distribution. Speculative ran at 0.85x of the KV baseline. The `target forwards` column said 130 for 128 tokens. v2 feeds the last committed token in with the next round's drafts, so the one verify pass does both jobs.

**v2: the Python loop was slower than the model.** The first 4090 run had speculative at 0.79x of KV even though target forwards had dropped to 35, and the scaling test said a forward costs ~9 ms whatever k is. KV decoding was running at 22 ms per token. The other 13 ms was host overhead: a `.item()` per draft token in the accept loop (each a CUDA sync), `torch.multinomial` on the CPU, a 50k-wide one-hot built for every "probability" in greedy mode, `torch.cat` growing the sequence every token. On the M2 Pro the forward is slow enough that none of it showed. v3 takes the argmax directly, finds the accepted prefix with a cumprod and reads one integer per round, samples on the device, and writes into a preallocated buffer. KV decoding on the 4090 went from 45.8 to 106.6 tok/s from that alone.

Rows from both versions are under `results/v1-two-target-forwards-*` and `results/v2-cuda/`. More in `DECISIONS.md`.

## Batching

Batched speculative decoding needs per-row acceptance and per-row cache rollback, which is what the `lengths` tensor in `KVCache` is for. On the 4090, γ=4, 128 new tokens per row:

| B | KV total tok/s | speculative total tok/s | speculative vs KV |
|---|---|---|---|
| 1 | 106.5 | 139.3 | 1.31x |
| 2 | 194.1 | 202.0 | 1.04x |
| 4 | 354.8 | 330.2 | 0.93x |
| 8 | 660.3 | 518.5 | 0.79x |
| 16 | 1147.6 | 839.4 | 0.73x |

Past B=2 it loses. Batching already amortises the weight reads that speculation was saving, and finished rows keep riding along until the slowest one is done. That is the standard result and the reason production servers use speculation at low load and turn it off under high load. On the M2 Pro the crossover never comes within B=8 (1.85x at B=1, 1.22x at B=8): the forward pass there is slow enough that the draft's work stays cheap by comparison. My batched decoder also does one extra draft forward per round to keep every row's pending input to one token; the reasons are in `DECISIONS.md`.

## Sampling, bf16, a smaller target

With temperature 0.8 the draft and target have to agree on a random draw, not an argmax. Acceptance falls to about 0.4 and the gain mostly disappears: 1.20x on the M2 Pro, 1.01x on the 4090.

In bf16 on the 4090, speculative γ=4 reaches 184 tok/s, 1.61x over bf16 KV. But those rows are not bit-identical to the single-token run. The verify pass computes logits for a block of positions with a different reduction order than the one-token path, and in bf16 that flips near-tie argmaxes. Still a valid decode of the bf16 model, not the same one. Also in bf16 at 128 tokens, recomputing everything (150 tok/s) beats my KV cache (114 tok/s): once compute is cheap, the per-layer slice copies in `KVCache.append` cost more than a 192-token flash-attention prefill. The cache wins again at 512 tokens. On the M2 Pro, half precision is 3.7x slower than fp32 for these shapes, so there are no MPS bf16 rows.

gpt2-large (774M) as the target with the same draft: 1.31x at γ=4 on the 4090. The 6x size ratio is enough; the 3x ratio of gpt2-medium was not, in an early run.

## Things I don't know

- Why v3 is about 10% slower than v2 on the M2 Pro at γ=4 (41.7 vs 37.2 tok/s) while being 2.3x faster on CUDA. The device-side cumprod and the buffer writes are suspects. Not profiled.
- What an HBM card does to these ratios. The A100 create was refused for stock on RunPod that night, so every CUDA number here is from one RTX 4090.
- Whether CUDA graphs close the gap between the 659 GB/s implied by the k=1 step and the card's 1008 GB/s.
- Whether the bf16 non-identity is one flipped token that cascades or many. I haven't diffed the outputs.

## How speculative decoding works, in one screen

Each decode step reads every weight once, so a 1.5B model in fp32 moves ~6 GB per token and is bound by memory bandwidth, not compute. Scoring five positions in one pass costs about the same as scoring one. So: let a small draft guess γ tokens, verify all of them in one target pass, keep the ones the target agrees with.

The accept rule from Leviathan et al. (2023): accept draft token `d` with probability `min(1, p(d)/q(d))`, where `p` is the target's distribution and `q` the draft's. On the first rejection, sample from `max(0, p − q)` renormalised, and stop. If everything is accepted, take one bonus token from the target's last position. That's a rejection sampler whose output is exactly `p`, so a bad draft can't change what comes out, only how fast. Greedy is the case where accept means argmax match, which is why `torch.equal` holds on every fp32 row. Prompt lookup is the same rule with `q` a one-hot on the n-gram continuation.

## Tests

`uv run pytest`, about two minutes on CPU, downloads gpt2 and gpt2-medium:

- logits match `transformers` to 1e-3
- cached logits match full recompute to 1e-4, chunked, one token at a time, and with rows at different lengths
- speculative greedy equals target greedy exactly, with one target pass per round
- the batched decoder gives every row exactly its single-row greedy output
- prompt-lookup greedy equals target greedy exactly
- with T=1.0, 400 draws of the first token match the target's own softmax within 4 standard errors

## Run it

```bash
uv sync
uv run python bench/bench.py --device cuda --target gpt2-xl --draft gpt2 --gammas 2 4 6 8 --lookup-gammas 4
uv run python bench/scaling.py --device cuda --model gpt2-xl
uv run python bench/batch.py --device cuda --batches 1 2 4 8 16
uv run python bench/report.py results/*.jsonl
```

`results/run_cuda.sh` is the exact sequence that produced the 4090 rows, `results/run_local.sh` the M2 Pro ones. `bench/bench.py` refuses to write a row without the hardware and config fields.

## Files

`specdecode/model.py` is the model and `KVCache` (append, commit, per-row rollback, mask and position ids derived from `lengths`). `specdecode/generate.py` has `generate_no_cache`, `generate_kv`, `generate_speculative`, `generate_speculative_batched`, `generate_lookup`, and the shared `_verify`. `DECISIONS.md` has the reasoning and two reversals. `experiments/` lists the dead ends.
