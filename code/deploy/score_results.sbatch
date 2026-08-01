#!/bin/bash
#SBATCH --job-name=thesis-score
#SBATCH --partition=frida
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

# Score all benchmark results (ROUGE-L + BERTScore + execution accuracy).
#
#   sbatch deploy/score_results.sbatch
#   sbatch deploy/score_results.sbatch --no-bertscore   # CPU-cheap rescore
#
# BERTScore loads roberta-large from HF_HOME (downloaded once, then cached in $WORK).

set -euo pipefail

PROJECT="$WORK/thesis"

srun --container-image "$PROJECT/containers/thesis-eval.sqsh" \
     --container-mounts "$PROJECT:/work" \
     --container-workdir /work/code \
     --export=ALL,HF_HOME=/work/hf-cache \
     python -m cli.score_results --results results/ "$@"
