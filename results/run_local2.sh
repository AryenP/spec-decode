#!/bin/sh
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1
uv run python bench/scaling.py --device mps --model gpt2-xl > results/mps-scaling.log 2>&1
uv run python bench/bench.py --device mps --target gpt2-xl --draft gpt2 --gammas 4 --lookup-gammas 4 8 --repeats 3 --skip-no-cache --out results/mps-gpt2xl-lookup.jsonl > results/mps-gpt2xl-lookup.log 2>&1
uv run python bench/bench.py --device mps --target gpt2-xl --draft gpt2 --new-tokens 512 --gammas 4 --lookup-gammas 4 --repeats 2 --out results/mps-gpt2xl-512.jsonl > results/mps-gpt2xl-512.log 2>&1
uv run python bench/batch.py --device mps --target gpt2-xl --draft gpt2 --batches 1 2 4 8 --gamma 4 --repeats 2 --out results/mps-batch.jsonl > results/mps-batch.log 2>&1
echo ALL_DONE > results/local.done
