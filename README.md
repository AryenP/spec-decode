# spec-decode

GPT-2 decoding written from scratch to see how much speculative decoding buys on a laptop. On my M2 Pro it's 2.1x over a plain KV cache, and the output is byte-identical to what the big model would have produced on its own.

HuggingFace is only used for the weights and the tokenizer. The forward pass, the KV cache and the three decoders are in `specdecode/`, about 390 lines.

## Numbers

Apple M2 Pro, 32 GB, PyTorch 2.14.1 on MPS, fp32. gpt2-xl (1.56B) is the target, gpt2 (124M) is the draft. 64-token prompt, 128 new tokens, greedy, median of 3 runs after a warmup. Raw rows with the full config are in `results/`.

| method | γ | tok/s | vs KV | target forwards | accepted per round |
|---|---|---|---|---|---|
| recompute everything | | 8.4 | 0.43x | 128 | |
| KV cache | | 19.8 | 1.00x | 128 | |
| speculative | 2 | 34.9 | 1.77x | 51 | 1.58 |
| speculative | 4 | **41.7** | **2.11x** | 35 | 2.79 |
| speculative | 6 | 37.9 | 1.92x | 30 | 3.45 |
| speculative | 8 | 34.8 | 1.76x | 27 | 3.96 |

γ=4 wins. Past that the draft keeps proposing tokens the target rejects, acceptance drops from 0.71 to 0.51 per token, and the extra draft passes eat the gain.

With sampling at T=0.8 instead of greedy, γ=4 gets 32.3 tok/s, 1.67x. Draft and target now have to agree on a random draw rather than an argmax, so fewer tokens get accepted.

## The version that was slower than no speculation at all

My first implementation ran at 0.85x of the KV baseline. The `target forwards` column gave it away: 130 forwards for 128 tokens at γ=2. After every round I rolled both caches back to the committed prefix and re-fed the accepted tokens to the target, so each round cost two target passes instead of one. The fix is to carry the last committed token into the next verify pass along with the new drafts, and roll back only over rejected positions. Those rows are kept in `results/v1-two-target-forwards-*.jsonl`. I also changed the prompt between v1 and v2 (the first one was a sentence repeated eight times, which hurt acceptance), so the acceptance rates aren't comparable across the two. The forward count is.

## Things I haven't measured

I don't have a CUDA GPU. MPS kernel launches are slow, which makes the draft's γ small passes relatively expensive here. On a real GPU I'd expect the sweet spot to move to a longer γ, but that's a guess.

gpt2-large as a target: the download died on a full disk. bf16 works in the code but isn't in the table. Batch size is 1 for the speculative path; batched acceptance needs ragged cache rollback and I haven't written it.

## How speculative decoding works, in one screen

Each decode step reads every weight once, so a 1.5B model is bandwidth-bound and reads ~6 GB per token in fp32. Scoring 5 positions in one pass costs about the same as scoring 1. So: let a 124M draft guess γ tokens, verify all of them in one target pass, keep the ones the target agrees with.

The accept rule from Leviathan et al. (2023): accept draft token `d` with probability `min(1, p(d)/q(d))` where `p` is the target's distribution and `q` the draft's. On the first rejection, sample from `max(0, p − q)` renormalised, and stop. If everything is accepted, take one bonus token from the target's last position. That's a rejection sampler whose output is exactly `p`, so a bad draft can't change what comes out, only how fast. Greedy is the special case where accept means argmax match, which is why `torch.equal` holds on every row above.

## Tests

`uv run pytest` (about 2 minutes on CPU, downloads gpt2 and gpt2-medium):

- logits match `transformers` to 1e-3
- cached logits match full recompute to 1e-4, both chunked and one token at a time
- speculative greedy equals target greedy exactly, and does one target pass per round
- with T=1.0, 400 draws of the first token match the target's own softmax within 4 standard errors

## Run it

```bash
uv sync
uv run python bench/bench.py --device mps --target gpt2-xl --draft gpt2 \
    --prompt-tokens 64 --new-tokens 128 --gammas 2 4 6 8 --repeats 3
uv run python bench/report.py results/*.jsonl
```

`specdecode/model.py` has the model and `KVCache` (append, commit, rollback). `specdecode/generate.py` has the three decoders. `bench/bench.py` refuses to write a row without the hardware and config fields.
