#!/bin/bash
#SBATCH --job-name=thesis-score
#SBATCH --partition=frida
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

# Score all benchmark results (normalized Exact Match + ROUGE-L + BERTScore + execution accuracy).
#
#   sbatch code/deploy/score_results.sbatch
#   sbatch code/deploy/score_results.sbatch --no-bertscore   # CPU-cheap rescore
#
# BERTScore loads roberta-large from HF_HOME (downloaded once, then cached in ../hf-cache).

set -euo pipefail

_SUBMIT="${SLURM_SUBMIT_DIR:-$PWD}"
_ENV="$_SUBMIT/deploy/job_env.sh"; [[ -f "$_ENV" ]] || _ENV="$_SUBMIT/code/deploy/job_env.sh"
[[ -f "$_ENV" ]] || { echo "ERROR: no deploy/job_env.sh under $_SUBMIT or $_SUBMIT/code -- sbatch from the thesis/ or thesis/code/ directory." >&2; exit 1; }
source "$_ENV"

mkdir -p "$PROJECT/hf-cache"

srun --container-image "$IMAGE" \
     --container-mounts "$PROJECT:/work" \
     --container-workdir "/work/$CODE_NAME" \
     --export=ALL,HF_HOME=/work/hf-cache \
     python -m cli.score_results --results results/ "$@"
