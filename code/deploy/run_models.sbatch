#!/bin/bash
#SBATCH --job-name=thesis-run-model
#SBATCH --partition=frida
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%A_%a.out

# Run several models CONCURRENTLY as a Slurm job array — one array task, on its own dedicated
# H100, per model. Each task loads its model once and runs all four of its slices (spider|bird
# x direct|text_to_sql), so wall-clock drops roughly linearly in the number of models versus
# the sequential deploy/run_all.sbatch.
#
#   sbatch --array=0-2 deploy/run_models.sbatch \
#       Qwen3.5-0.8B-Q4_K_M.gguf Qwen3.5-2B-Q4_K_M.gguf Qwen3.5-4B-Q4_K_M.gguf
#
#   # smoke test first (extra cli.run_all flags go through RUN_ARGS, not positionally,
#   # because the positional args are the model list):
#   sbatch --export=ALL,RUN_ARGS="--limit 10" --array=0-2 deploy/run_models.sbatch <models...>
#
#   # throttle to 2 concurrent tasks if you hit a per-user GPU limit:
#   sbatch --array=0-2%2 deploy/run_models.sbatch <models...>
#
#   # re-run just one model (index 1 of the same list):
#   sbatch --array=1-1 deploy/run_models.sbatch <models...>
#
# ALL models run on the SAME GPU type, each on a card of its own. The benchmark records TTFT
# and tokens/s, so a model measured on an A100 is not comparable to one measured on an H100,
# and two models sharing one card contend for HBM bandwidth — either would corrupt the very
# metrics being collected. Don't override --gres for individual tasks; if H100s are backed up,
# move the WHOLE array to another type and re-run anything already measured elsewhere.
#
# --array indices are 0-based and index into the positional model list. Logs are %A_%a
# (array job id + task id) so concurrent tasks do not collide on one file.
#
# Parallel-safe: each task only ever writes results/<dataset>/<pipeline>/<format>/<model>.ndjson
# and the filename is derived from the model name, so no two tasks touch the same file.
# Resumable per task: requeue the same --array index after a timeout and it continues.

set -euo pipefail

MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "usage: sbatch --array=0-N deploy/run_models.sbatch <model1.gguf> [model2.gguf ...]" >&2
  exit 1
fi

INDEX="${SLURM_ARRAY_TASK_ID:?not an array job — submit with --array=0-$((${#MODELS[@]} - 1))}"
if [[ $INDEX -ge ${#MODELS[@]} ]]; then
  echo "array index $INDEX is out of range for ${#MODELS[@]} model(s): ${MODELS[*]}" >&2
  echo "use --array=0-$((${#MODELS[@]} - 1))" >&2
  exit 1
fi
MODEL="${MODELS[$INDEX]}"

PROJECT="$WORK/thesis"

echo "array task $INDEX/$((${#MODELS[@]} - 1)) -> model=$MODEL on $(hostname)"

srun --container-image "$PROJECT/containers/thesis-eval.sqsh" \
     --container-mounts "$PROJECT:/work" \
     --container-workdir /work/code \
     python -m cli.run_all --models "$MODEL" ${RUN_ARGS:-}
