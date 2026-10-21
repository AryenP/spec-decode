#!/bin/bash
# runs on a RunPod pod. usage: bash run_cuda.sh <tag>
set -e
TAG=${1:-cuda}
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"
mkdir -p results
python bench/scaling.py --device cuda --model gpt2-xl --out results/$TAG-scaling.jsonl 2>&1 | tee results/$TAG-scaling.log
python bench/scaling.py --device cuda --dtype bfloat16 --model gpt2-xl --out results/$TAG-scaling-bf16.jsonl 2>&1 | tee results/$TAG-scaling-bf16.log
python bench/bench.py --device cuda --target gpt2-xl --draft gpt2 --gammas 2 4 6 8 --lookup-gammas 4 --repeats 3 --out results/$TAG-gpt2xl-gpt2.jsonl 2>&1 | tee results/$TAG-gpt2xl-gpt2.log
python bench/bench.py --device cuda --dtype bfloat16 --target gpt2-xl --draft gpt2 --gammas 2 4 6 8 --lookup-gammas 4 --repeats 3 --out results/$TAG-gpt2xl-gpt2-bf16.jsonl 2>&1 | tee results/$TAG-gpt2xl-gpt2-bf16.log
python bench/bench.py --device cuda --target gpt2-xl --draft gpt2 --gammas 4 --temperature 0.8 --repeats 3 --skip-no-cache --out results/$TAG-gpt2xl-gpt2-T0.8.jsonl 2>&1 | tee results/$TAG-gpt2xl-gpt2-T0.8.log
python bench/bench.py --device cuda --target gpt2-large --draft gpt2 --gammas 2 4 6 --repeats 3 --out results/$TAG-gpt2large-gpt2.jsonl 2>&1 | tee results/$TAG-gpt2large-gpt2.log
python bench/bench.py --device cuda --target gpt2-xl --draft gpt2 --new-tokens 512 --gammas 4 --lookup-gammas 4 --repeats 2 --out results/$TAG-gpt2xl-512.jsonl 2>&1 | tee results/$TAG-gpt2xl-512.log
python bench/batch.py --device cuda --target gpt2-xl --draft gpt2 --batches 1 2 4 8 16 --gamma 4 --repeats 2 --out results/$TAG-batch.jsonl 2>&1 | tee results/$TAG-batch.log
echo CUDA_DONE > results/$TAG.done
