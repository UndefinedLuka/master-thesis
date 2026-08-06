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
#   sbatch code/deploy/run_benchmark.sbatch <dataset> <pipeline> <model.gguf> [extra args...]
#   sbatch code/deploy/run_benchmark.sbatch spider direct Qwen3.5-9B-Q4_K_M.gguf
#   sbatch code/deploy/run_benchmark.sbatch bird text_to_sql Qwen3.5-9B-Q4_K_M.gguf --limit 10
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

_SUBMIT="${SLURM_SUBMIT_DIR:-$PWD}"
_ENV="$_SUBMIT/deploy/job_env.sh"; [[ -f "$_ENV" ]] || _ENV="$_SUBMIT/code/deploy/job_env.sh"
[[ -f "$_ENV" ]] || { echo "ERROR: no deploy/job_env.sh under $_SUBMIT or $_SUBMIT/code -- sbatch from the thesis/ or thesis/code/ directory." >&2; exit 1; }
source "$_ENV"

srun --container-image "$IMAGE" \
     --container-mounts "$PROJECT:/work" \
     --container-workdir "/work/$CODE_NAME" \
     python -m cli.run_benchmark \
       --dataset "$DATASET" --pipeline "$PIPELINE" --model "$MODEL" "$@"
