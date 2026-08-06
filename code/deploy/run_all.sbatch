#!/bin/bash
#SBATCH --job-name=thesis-run-all
#SBATCH --partition=frida
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=%x-%j.out

# Run every GGUF in models/ across both datasets and both pipelines (sequential, one GPU).
#
#   sbatch code/deploy/run_all.sbatch            # full run
#   sbatch code/deploy/run_all.sbatch --limit 50 # smoke
#
# Fully resumable — requeue after the 7-day walltime and it picks up where it stopped.

set -euo pipefail

_SUBMIT="${SLURM_SUBMIT_DIR:-$PWD}"
_ENV="$_SUBMIT/deploy/job_env.sh"; [[ -f "$_ENV" ]] || _ENV="$_SUBMIT/code/deploy/job_env.sh"
[[ -f "$_ENV" ]] || { echo "ERROR: no deploy/job_env.sh under $_SUBMIT or $_SUBMIT/code -- sbatch from the thesis/ or thesis/code/ directory." >&2; exit 1; }
source "$_ENV"

srun --container-image "$IMAGE" \
     --container-mounts "$PROJECT:/work" \
     --container-workdir "/work/$CODE_NAME" \
     python -m cli.run_all "$@"
