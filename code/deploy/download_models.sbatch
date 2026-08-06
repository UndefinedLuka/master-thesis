#!/bin/bash
#SBATCH --job-name=thesis-download-models
#SBATCH --partition=frida
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out

# Download the GGUF models in deploy/models.txt into code/models/.
#
#   sbatch code/deploy/download_models.sbatch
#   sbatch code/deploy/download_models.sbatch other-list.txt
#
# Prefer running `bash code/deploy/download_models.sh` straight on the login node — it needs only
# curl, finishes in the foreground, and does not queue. Use this job when login-node egress
# is blocked or the download is large enough that you want it detached.
#
# No --gres here: the work is pure network I/O, so a GPU-free job schedules almost instantly
# instead of waiting behind the benchmark queue.

set -euo pipefail

_SUBMIT="${SLURM_SUBMIT_DIR:-$PWD}"
_ENV="$_SUBMIT/deploy/job_env.sh"; [[ -f "$_ENV" ]] || _ENV="$_SUBMIT/code/deploy/job_env.sh"
[[ -f "$_ENV" ]] || { echo "ERROR: no deploy/job_env.sh under $_SUBMIT or $_SUBMIT/code -- sbatch from the thesis/ or thesis/code/ directory." >&2; exit 1; }
source "$_ENV"

srun --container-image "$IMAGE" \
     --container-mounts "$PROJECT:/work" \
     --container-workdir "/work/$CODE_NAME" \
     bash deploy/download_models.sh "$@"
