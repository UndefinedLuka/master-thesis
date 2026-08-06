#!/bin/bash
#SBATCH --job-name=thesis-benchmark
#SBATCH --partition=frida
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out

# Run one benchmark slice:
#
#   sbatch deploy/run_benchmark.sbatch <dataset> <pipeline> <model.gguf> [extra args...]
#   sbatch deploy/run_benchmark.sbatch spider direct Qwen3.5-9B-Q4_K_M.gguf
#   sbatch deploy/run_benchmark.sbatch bird text_to_sql Qwen3.5-9B-Q4_K_M.gguf --limit 10
#
# Resumable: requeue the same command after a timeout/preemption and it continues.
#
# Keep --gres on H100 even if the queue is busy: TTFT and tokens/s are only comparable across
# models measured on the same GPU type. Waiting is cheaper than re-running the campaign.

set -euo pipefail

DATASET="${1:?usage: sbatch run_benchmark.sbatch <dataset> <pipeline> <model.gguf>}"
PIPELINE="${2:?missing pipeline (direct|text_to_sql)}"
MODEL="${3:?missing model .gguf filename}"
shift 3

PROJECT="$WORK/thesis"

srun --container-image "$PROJECT/containers/thesis-eval.sqsh" \
     --container-mounts "$PROJECT:/work" \
     --container-workdir /work/code \
     python -m cli.run_benchmark \
       --dataset "$DATASET" --pipeline "$PIPELINE" --model "$MODEL" "$@"
