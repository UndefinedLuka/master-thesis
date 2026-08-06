#!/bin/bash
#SBATCH --job-name=thesis-download-models
#SBATCH --partition=frida
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out

# Download the GGUF models in deploy/models.txt onto $WORK.
#
#   sbatch deploy/download_models.sbatch
#   sbatch deploy/download_models.sbatch other-list.txt
#
# Prefer running `bash deploy/download_models.sh` straight on the login node — it needs only
# curl, finishes in the foreground, and does not queue. Use this job when login-node egress
# is blocked or the download is large enough that you want it detached.
#
# No --gres here: the work is pure network I/O, so a GPU-free job schedules almost instantly
# instead of waiting behind the benchmark queue.

set -euo pipefail

PROJECT="$WORK/thesis"

srun --container-image "$PROJECT/containers/thesis-eval.sqsh" \
     --container-mounts "$PROJECT:/work" \
     --container-workdir /work/code \
     bash deploy/download_models.sh "$@"
