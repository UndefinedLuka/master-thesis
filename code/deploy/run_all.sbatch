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
#   sbatch deploy/run_all.sbatch            # full run
#   sbatch deploy/run_all.sbatch --limit 50 # smoke
#
# Fully resumable — requeue after the 7-day walltime and it picks up where it stopped.

set -euo pipefail

PROJECT="$WORK/thesis"

srun --container-image "$PROJECT/containers/thesis-eval.sqsh" \
     --container-mounts "$PROJECT:/work" \
     --container-workdir /work/code \
     python -m cli.run_all "$@"
