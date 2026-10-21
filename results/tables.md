<!-- results/rtx4090-batch.jsonl -->
**gpt2-xl** target, **gpt2** draft · NVIDIA GeForce RTX 4090 · cuda float32 · prompt 64 tok, 128 new tok per row, greedy, γ=4

| B | method | total tok/s | tok/s per row | vs KV at same B | target fwds | accept rate | identical |
|---|---|---|---|---|---|---|---|
| 1 | kv_cache | 106.5 | 106.5 | 1.00× | 128 | — | yes |
| 1 | speculative | 139.3 | 139.3 | 1.31× | 35 | 0.72 | yes |
| 2 | kv_cache | 194.1 | 97.0 | 1.00× | 128 | — | yes |
| 2 | speculative | 202.0 | 101.0 | 1.04× | 40 | 0.61 | yes |
| 4 | kv_cache | 354.8 | 88.7 | 1.00× | 128 | — | yes |
| 4 | speculative | 330.2 | 82.6 | 0.93× | 44 | 0.55 | yes |
| 8 | kv_cache | 660.3 | 82.5 | 1.00× | 128 | — | yes |
| 8 | speculative | 518.5 | 64.8 | 0.79× | 44 | 0.55 | yes |
| 16 | kv_cache | 1147.6 | 71.7 | 1.00× | 128 | — | yes |
| 16 | speculative | 839.4 | 52.5 | 0.73× | 44 | 0.55 | yes |

<!-- results/rtx4090-gpt2large-gpt2.jsonl -->
**gpt2-large** (774.0 M) target, **gpt2** (124.4 M) draft · NVIDIA GeForce RTX 4090 · cuda float32 · torch 2.8.0+cu128 · prompt 64 tok, 128 new tok, T=0.0 · median of 3 after 1 warmup

| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |
|---|---|---|---|---|---|---|---|---|---|
| no_cache | — | 102.0 | 0.70× | 1.25 | 128 | 0 | — | — | yes |
| kv_cache | — | 146.7 | 1.00× | 0.87 | 128 | 0 | — | — | yes |
| speculative | 2 | 170.5 | 1.16× | 0.75 | 52 | 102 | 0.77 | 1.53 | yes |
| speculative | 4 | 192.9 | 1.31× | 0.66 | 36 | 139 | 0.68 | 2.69 | yes |
| speculative | 6 | 172.4 | 1.18× | 0.74 | 30 | 173 | 0.58 | 3.45 | yes |

<!-- results/rtx4090-gpt2xl-512.jsonl -->
**gpt2-xl** (1557.6 M) target, **gpt2** (124.4 M) draft · NVIDIA GeForce RTX 4090 · cuda float32 · torch 2.8.0+cu128 · prompt 64 tok, 512 new tok, T=0.0 · median of 2 after 1 warmup

| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |
|---|---|---|---|---|---|---|---|---|---|
| no_cache | — | 27.4 | 0.29× | 18.71 | 512 | 0 | — | — | yes |
| kv_cache | — | 94.5 | 1.00× | 5.42 | 512 | 0 | — | — | yes |
| speculative | 4 | 184.7 | 1.95× | 2.77 | 111 | 441 | 0.91 | 3.65 | yes |
| lookup_n3 | 4 | 254.5 | 2.69× | 2.01 | 154 | 0 | 0.92 | 2.35 | yes |

<!-- results/rtx4090-gpt2xl-gpt2-T0.8.jsonl -->
**gpt2-xl** (1557.6 M) target, **gpt2** (124.4 M) draft · NVIDIA GeForce RTX 4090 · cuda float32 · torch 2.8.0+cu128 · prompt 64 tok, 128 new tok, T=0.8 · median of 3 after 1 warmup

| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |
|---|---|---|---|---|---|---|---|---|---|
| kv_cache | — | 103.0 | 1.00× | 1.24 | 128 | 0 | — | — | n/a |
| speculative | 4 | 104.3 | 1.01× | 1.23 | 52 | 204 | 0.38 | 1.51 | n/a |

<!-- results/rtx4090-gpt2xl-gpt2-bf16.jsonl -->
**gpt2-xl** (1557.6 M) target, **gpt2** (124.4 M) draft · NVIDIA GeForce RTX 4090 · cuda bfloat16 · torch 2.8.0+cu128 · prompt 64 tok, 128 new tok, T=0.0 · median of 3 after 1 warmup

| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |
|---|---|---|---|---|---|---|---|---|---|
| no_cache | — | 150.3 | 1.31× | 0.85 | 128 | 0 | — | — | **NO** |
| kv_cache | — | 114.4 | 1.00× | 1.12 | 128 | 0 | — | — | yes |
| speculative | 2 | 142.8 | 1.25× | 0.90 | 50 | 99 | 0.82 | 1.63 | **NO** |
| speculative | 4 | 184.0 | 1.61× | 0.70 | 34 | 132 | 0.73 | 2.91 | **NO** |
| speculative | 6 | 173.0 | 1.51× | 0.74 | 26 | 151 | 0.69 | 4.12 | **NO** |
| speculative | 8 | 167.4 | 1.46× | 0.76 | 25 | 190 | 0.56 | 4.38 | **NO** |
| lookup_n3 | 4 | 179.2 | 1.57× | 0.71 | 73 | 0 | 0.83 | 0.79 | **NO** |

<!-- results/rtx4090-gpt2xl-gpt2.jsonl -->
**gpt2-xl** (1557.6 M) target, **gpt2** (124.4 M) draft · NVIDIA GeForce RTX 4090 · cuda float32 · torch 2.8.0+cu128 · prompt 64 tok, 128 new tok, T=0.0 · median of 3 after 1 warmup

| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |
|---|---|---|---|---|---|---|---|---|---|
| no_cache | — | 52.6 | 0.49× | 2.43 | 128 | 0 | — | — | yes |
| kv_cache | — | 106.6 | 1.00× | 1.20 | 128 | 0 | — | — | yes |
| speculative | 2 | 122.9 | 1.15× | 1.04 | 51 | 101 | 0.79 | 1.58 | yes |
| speculative | 4 | 147.3 | 1.38× | 0.87 | 35 | 134 | 0.71 | 2.79 | yes |
| speculative | 6 | 156.5 | 1.47× | 0.82 | 30 | 173 | 0.58 | 3.45 | yes |
| speculative | 8 | 145.4 | 1.36× | 0.88 | 27 | 203 | 0.51 | 3.96 | yes |
| lookup_n3 | 4 | 161.1 | 1.51× | 0.79 | 74 | 0 | 0.72 | 0.75 | yes |

<!-- results/rtx4090-scaling-bf16.jsonl -->
**gpt2-xl** (1557.6 M) · NVIDIA GeForce RTX 4090 · cuda bfloat16 · one cached forward over k new positions after a 128-token prefix · median of 20

| k positions | ms | vs k=1 | ms per position | implied GB/s if bandwidth-bound |
|---|---|---|---|---|
| 1 | 8.65 | 1.00× | 8.65 | 360 |
| 2 | 12.01 | 1.39× | 6.00 | 260 |
| 4 | 11.63 | 1.34× | 2.91 | 268 |
| 8 | 11.29 | 1.31× | 1.41 | 276 |
| 16 | 11.58 | 1.34× | 0.72 | 269 |
| 32 | 12.48 | 1.44× | 0.39 | 250 |
| 64 | 11.62 | 1.34× | 0.18 | 268 |

<!-- results/rtx4090-scaling.jsonl -->
**gpt2-xl** (1557.6 M) · NVIDIA GeForce RTX 4090 · cuda float32 · one cached forward over k new positions after a 128-token prefix · median of 20

| k positions | ms | vs k=1 | ms per position | implied GB/s if bandwidth-bound |
|---|---|---|---|---|
| 1 | 9.46 | 1.00× | 9.46 | 659 |
| 2 | 13.40 | 1.42× | 6.70 | 465 |
| 4 | 13.65 | 1.44× | 3.41 | 456 |
| 8 | 14.20 | 1.50× | 1.77 | 439 |
| 16 | 13.93 | 1.47× | 0.87 | 447 |
| 32 | 15.01 | 1.59× | 0.47 | 415 |
| 64 | 15.88 | 1.68× | 0.25 | 392 |

<!-- results/mps-batch.jsonl -->
**gpt2-xl** target, **gpt2** draft · Apple M2 Pro, 32 GB · mps float32 · prompt 64 tok, 128 new tok per row, greedy, γ=4

| B | method | total tok/s | tok/s per row | vs KV at same B | target fwds | accept rate | identical |
|---|---|---|---|---|---|---|---|
| 1 | kv_cache | 19.5 | 19.5 | 1.00× | 128 | — | yes |
| 1 | speculative | 36.1 | 36.1 | 1.85× | 35 | 0.72 | yes |
| 2 | kv_cache | 37.8 | 18.9 | 1.00× | 128 | — | yes |
| 2 | speculative | 46.7 | 23.4 | 1.23× | 40 | 0.61 | yes |
| 4 | kv_cache | 70.9 | 17.7 | 1.00× | 128 | — | yes |
| 4 | speculative | 77.5 | 19.4 | 1.09× | 44 | 0.55 | yes |
| 8 | kv_cache | 110.9 | 13.9 | 1.00× | 128 | — | yes |
| 8 | speculative | 135.6 | 16.9 | 1.22× | 44 | 0.55 | yes |

<!-- results/mps-gpt2xl-512.jsonl -->
**gpt2-xl** (1557.6 M) target, **gpt2** (124.4 M) draft · Apple M2 Pro, 32 GB · mps float32 · torch 2.14.1 · prompt 64 tok, 512 new tok, T=0.0 · median of 2 after 1 warmup

| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |
|---|---|---|---|---|---|---|---|---|---|
| no_cache | — | 3.3 | 0.18× | 154.07 | 512 | 0 | — | — | yes |
| kv_cache | — | 18.9 | 1.00× | 27.07 | 512 | 0 | — | — | yes |
| speculative | 4 | 37.9 | 2.01× | 13.49 | 111 | 441 | 0.91 | 3.65 | yes |
| lookup_n3 | 4 | 49.8 | 2.63× | 10.28 | 154 | 0 | 0.92 | 2.35 | yes |

<!-- results/mps-gpt2xl-gpt2-T0.8.jsonl -->
**gpt2-xl** (1557.6 M) target, **gpt2** (124.4 M) draft · Apple M2 Pro, 32 GB · mps float32 · torch 2.14.1 · prompt 64 tok, 128 new tok, T=0.8 · median of 3 after 1 warmup

| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |
|---|---|---|---|---|---|---|---|---|---|
| kv_cache | — | 18.3 | 1.00× | 6.98 | 128 | 0 | — | — | n/a |
| speculative | 4 | 22.0 | 1.20× | 5.81 | 51 | 198 | 0.40 | 1.56 | n/a |

<!-- results/mps-gpt2xl-gpt2.jsonl -->
**gpt2-xl** (1557.6 M) target, **gpt2** (124.4 M) draft · Apple M2 Pro, 32 GB · mps float32 · torch 2.14.1 · prompt 64 tok, 128 new tok, T=0.0 · median of 3 after 1 warmup

| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |
|---|---|---|---|---|---|---|---|---|---|
| no_cache | — | 7.8 | 0.40× | 16.38 | 128 | 0 | — | — | yes |
| kv_cache | — | 19.6 | 1.00× | 6.55 | 128 | 0 | — | — | yes |
| speculative | 2 | 30.5 | 1.56× | 4.19 | 51 | 101 | 0.79 | 1.58 | yes |
| speculative | 4 | 37.2 | 1.90× | 3.44 | 35 | 134 | 0.71 | 2.79 | yes |
| speculative | 6 | 36.0 | 1.84× | 3.55 | 30 | 173 | 0.58 | 3.45 | yes |
| speculative | 8 | 34.0 | 1.74× | 3.76 | 27 | 203 | 0.51 | 3.96 | yes |
| lookup_n3 | 4 | 30.2 | 1.55× | 4.23 | 74 | 0 | 0.72 | 0.75 | yes |

<!-- results/mps-scaling.jsonl -->
**gpt2-xl** (1557.6 M) · Apple M2 Pro, 32 GB · mps float32 · one cached forward over k new positions after a 128-token prefix · median of 20

| k positions | ms | vs k=1 | ms per position | implied GB/s if bandwidth-bound |
|---|---|---|---|---|
| 1 | 56.30 | 1.00× | 56.30 | 111 |
| 2 | 70.02 | 1.24× | 35.01 | 89 |
| 4 | 72.48 | 1.29× | 18.12 | 86 |
| 8 | 84.67 | 1.50× | 10.58 | 74 |
| 16 | 79.91 | 1.42× | 4.99 | 78 |
| 32 | 86.19 | 1.53× | 2.69 | 72 |
| 64 | 91.23 | 1.62× | 1.43 | 68 |

<!-- results/v2-cuda/rtx4090-gpt2xl-gpt2.jsonl -->
**gpt2-xl** (1557.6 M) target, **gpt2** (124.4 M) draft · NVIDIA GeForce RTX 4090 · cuda float32 · torch 2.8.0+cu128 · prompt 64 tok, 128 new tok, T=0.0 · median of 3 after 1 warmup

| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |
|---|---|---|---|---|---|---|---|---|---|
| no_cache | — | 41.6 | 0.91× | 3.08 | 128 | 0 | — | — | yes |
| kv_cache | — | 45.8 | 1.00× | 2.80 | 128 | 0 | — | — | yes |
| speculative | 2 | 54.0 | 1.18× | 2.37 | 51 | 101 | 0.79 | 1.58 | yes |
| speculative | 4 | 36.2 | 0.79× | 3.53 | 35 | 134 | 0.71 | 2.79 | yes |
| speculative | 6 | 42.1 | 0.92× | 3.04 | 30 | 173 | 0.58 | 3.45 | yes |
| speculative | 8 | 32.0 | 0.70× | 4.01 | 27 | 203 | 0.51 | 3.96 | yes |
| lookup_n3 | 4 | 93.2 | 2.04× | 1.37 | 74 | 0 | 0.72 | 0.75 | yes |

<!-- results/v1-two-target-forwards-mps-gpt2xl-gpt2.jsonl -->
**gpt2-xl** (1557.6 M) target, **gpt2** (124.4 M) draft · Apple M2 Pro, 32 GB · mps float32 · torch 2.14.1 · prompt 64 tok, 128 new tok, T=0.0 · median of 3 after 1 warmup

| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |
|---|---|---|---|---|---|---|---|---|---|
| no_cache | — | 8.4 | 0.43× | 15.18 | 128 | 0 | — | — | yes |
| kv_cache | — | 19.6 | 1.00× | 6.54 | 128 | 0 | — | — | yes |
| speculative | 2 | 16.7 | 0.85× | 7.68 | 130 | 130 | 0.49 | 0.98 | yes |
| speculative | 4 | 17.3 | 0.88× | 7.40 | 110 | 218 | 0.34 | 1.34 | yes |
| speculative | 6 | 16.1 | 0.82× | 7.95 | 102 | 302 | 0.26 | 1.53 | yes |
| speculative | 8 | 14.1 | 0.72× | 9.09 | 100 | 392 | 0.20 | 1.58 | yes |

