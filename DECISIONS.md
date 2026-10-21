# Decisions

Why things are the way they are, including the ones I'd do differently. Newest at the bottom.

## Write GPT-2 from scratch instead of using `transformers` generate

The point is to be able to defend the KV cache and the verify pass line by line. HF's cache is a tree of `DynamicCache` objects I didn't write and can't roll back per row. Cost: GPT-2 only, 1024 positions, no flash-attention kernels. Fine for a decoding-algorithm study; wrong for a serving system.

## GPT-2 family as target and draft

Same tokenizer for draft and target is a hard requirement of speculative decoding, and GPT-2 small / xl is the cheapest pair that satisfies it at a 12x size ratio. Llama 3.2 1B / 8B would be a more current pair but 8B in fp32 doesn't fit the laptop and I wanted the first numbers on hardware I own.

## Pre-allocated cache with a length counter, not a growing tensor

`torch.cat` per step copies the whole cache every token, which is O(n^2) in memory traffic and would make the "KV cache" row look worse than it is. Pre-allocating `(layers, B, heads, max_len, hd)` and tracking `lengths` costs one upfront allocation and makes rollback a subtraction.

## Rollback = decrement the length, leave the bytes

Rejected positions are never read again because the mask and position ids are derived from `lengths`. Zeroing them would be a kernel launch per round for nothing.

## fp32 on MPS for the headline numbers

GPT-2 in fp16 overflows in the attention logits of late layers. bf16 is safe and supported, and it is in the tables, but fp32 is what every row shares across MPS and CUDA so the comparison stays apples to apples.

## Reversal: one target forward per round, not two

v1 rolled both caches back to the committed prefix after each round and then re-fed the accepted tokens to the target to get the next distribution. That is a second target pass per round and it put speculative at 0.85x of plain KV decoding. The `target_forwards` column (130 for 128 tokens) is what caught it. v2 feeds the last committed token in with the next round's drafts so the verify pass does both jobs. Rows from v1 are kept in `results/v1-two-target-forwards-*.jsonl`.

## Batched decoder: the draft also processes its own last proposal

In the batch-1 decoder the draft never computes KV for its final draft token (it isn't needed until the next round, and then only if accepted). With a batch, rows accept different amounts, so the number of tokens each row's draft cache lacks would vary between 1 and 2, and ragged pending input is a mess. Spending one extra cheap draft forward per round makes every row's pending input exactly one token. The batch-1 path keeps the cheaper version.

## Finished rows keep computing in the batched decoder

Once a row hits `max_new` it still rides along in every forward until the slowest row finishes. Dropping it means compacting both caches mid-run. The waste is real and shows up in the batch tables as a smaller per-row gain at larger B. A production system would compact or schedule around it; this one reports it.

## Sampling on CPU

`torch.multinomial` has no `generator` support on MPS. At batch size 1 the transfer is a few microseconds. It would matter at large batch and I haven't fixed it.

## RTX 4090 for the CUDA numbers, not an A100

The A100 SXM create was refused for stock on RunPod at the time; the L40S create was blocked by a tooling permission. The 4090 (1 TB/s GDDR6X, 24 GB) was available at $0.89/hr and is a reasonable consumer-GPU datapoint. An HBM card would show different ratios; that row is still open.

## Reversal: one host sync per round, sampling on the device (v3)

The first 4090 run was the second time the `target_forwards` column told me something the
tok/s column hid: forwards had dropped to 35 but speculative was slower than KV. The scaling
test showed a cached forward at ~9 ms whether it scored 1 or 64 positions, and KV decoding ran
at ~22 ms per token. The missing 13 ms per token was the Python loop: a `.item()` per draft in
the accept loop (each one a CUDA sync), `torch.multinomial` on the CPU, a 50k-wide one-hot
built for every "probability" in greedy mode, and `torch.cat` growing the sequence each token.
On the M2 Pro the forward pass is slow enough that none of this showed. v3 takes the argmax
directly in greedy mode, computes the accepted prefix with a cumprod and reads one integer per
round, samples with a device generator where the backend supports it, and writes into a
preallocated buffer. v2 rows from the 4090 are in `results/v2-cuda/`.

## bf16 rows are reported as not identical, on purpose

In bf16 the verify pass scores a block of positions and the one-token path scores one; the matmul reduction order differs and a near-tie argmax flips. I could have made the `identical` column compare against a bf16 block-scored reference and shown "yes". That would hide the thing an inference engineer needs to know: low-precision speculative decoding is distribution-preserving, not bitwise-reproducing. The column stays honest.
