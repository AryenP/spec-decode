**gpt2-xl** (1557.6 M) target, **gpt2** (124.4 M) draft · Apple M2 Pro, 32 GB · mps float32 · torch 2.14.1 · prompt 64 tok, 128 new tok, T=0.0 · median of 3 after 1 warmup

| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |
|---|---|---|---|---|---|---|---|---|---|
| no_cache | — | 8.4 | 0.43× | 15.14 | 128 | 0 | — | — | yes |
| kv_cache | — | 19.8 | 1.00× | 6.48 | 128 | 0 | — | — | yes |
| speculative | 2 | 34.9 | 1.77× | 3.66 | 51 | 101 | 0.79 | 1.58 | yes |
| speculative | 4 | 41.7 | 2.11× | 3.07 | 35 | 134 | 0.71 | 2.79 | yes |
| speculative | 6 | 37.9 | 1.92× | 3.37 | 30 | 173 | 0.58 | 3.45 | yes |
| speculative | 8 | 34.8 | 1.76× | 3.68 | 27 | 203 | 0.51 | 3.96 | yes |

**gpt2-xl** (1557.6 M) target, **gpt2** (124.4 M) draft · Apple M2 Pro, 32 GB · mps float32 · torch 2.14.1 · prompt 64 tok, 128 new tok, T=0.8 · median of 3 after 1 warmup

| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |
|---|---|---|---|---|---|---|---|---|---|
| kv_cache | — | 19.3 | 1.00× | 6.64 | 128 | 0 | — | — | n/a |
| speculative | 4 | 32.3 | 1.67× | 3.97 | 44 | 172 | 0.50 | 1.98 | n/a |

**gpt2-xl** (1557.6 M) target, **gpt2** (124.4 M) draft · Apple M2 Pro, 32 GB · mps float32 · torch 2.14.1 · prompt 64 tok, 128 new tok, T=0.0 · median of 3 after 1 warmup

| method | γ | tok/s | vs KV | wall s | target fwds | draft fwds | accept rate | accepted/round | identical |
|---|---|---|---|---|---|---|---|---|---|
| no_cache | — | 8.4 | 0.43× | 15.18 | 128 | 0 | — | — | yes |
| kv_cache | — | 19.6 | 1.00× | 6.54 | 128 | 0 | — | — | yes |
| speculative | 2 | 16.7 | 0.85× | 7.68 | 130 | 130 | 0.49 | 0.98 | yes |
| speculative | 4 | 17.3 | 0.88× | 7.40 | 110 | 218 | 0.34 | 1.34 | yes |
| speculative | 6 | 16.1 | 0.82× | 7.95 | 102 | 302 | 0.26 | 1.53 | yes |
| speculative | 8 | 14.1 | 0.72× | 9.09 | 100 | 392 | 0.20 | 1.58 | yes |

