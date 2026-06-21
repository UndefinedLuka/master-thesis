#!/bin/bash
# Single-job benchmark on FRIDA (recommended).
#
# Launches the vLLM OpenAI server in the background inside an Enroot container,
# waits for it to become healthy, then runs the full benchmark client in the
# same allocation over localhost. Keeping server + client in one job avoids
# cross-job hostname discovery (the common SLURM failure mode).
#
#   sbatch slurm/run_benchmark.sbatch
#   tail -f logs/bench-<jobid>.out
#
#SBATCH --job-name=bench
#SBATCH --partition=frida
#SBATCH --gres=gpu:1                 # bump (e.g. gpu:H100:2) for larger models
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bench-%j.out

set -euo pipefail
mkdir -p logs
source "$(dirname "$0")/env.sh"

QUANT_ARG=""
[ -n "$BENCH_QUANT" ] && QUANT_ARG="--quantization $BENCH_QUANT"

MOUNTS="${BENCH_REPO}:/work,${HF_HOME}:/hf"
COMMON_CONTAINER_ARGS="--container-image=${BENCH_IMAGE} --container-mounts=${MOUNTS} --container-workdir=/work"

# --- 1) Start the vLLM server in the background -----------------------------
srun $COMMON_CONTAINER_ARGS --no-container-remap-root \
  bash -lc "
    set -e
    export HF_HOME=/hf
    pip install --no-cache-dir 'vllm>=0.6' >/dev/null 2>&1 || true
    python -m vllm.entrypoints.openai.api_server \
      --model '${BENCH_MODEL}' ${QUANT_ARG} \
      --port ${VLLM_PORT} \
      --max-model-len ${VLLM_MAX_MODEL_LEN}
  " &
VLLM_PID=$!

# --- 2) Wait for health, then run the client --------------------------------
srun $COMMON_CONTAINER_ARGS \
  bash -lc "
    set -e
    export BENCH_ROOT=/work HF_HOME=/hf
    cd /work

    # Wait up to 10 minutes for the server (model load can be slow).
    for i in \$(seq 1 120); do
      if curl -sf http://localhost:${VLLM_PORT}/health >/dev/null; then echo 'server healthy'; break; fi
      sleep 5
    done

    # Install the client deps with uv (or pip if uv is unavailable in the image).
    if command -v uv >/dev/null 2>&1; then
      uv sync
      RUN='uv run python -m'
    else
      pip install --no-cache-dir -e . >/dev/null
      RUN='python -m'
    fi

    BASE=http://localhost:${VLLM_PORT}/v1
    for ds in spider bird; do
      for fmt in json xml yaml csv markdown toon; do
        \$RUN benchmark.cli.run --model '${BENCH_MODEL}' --dataset \$ds \
          --pipeline direct --format \$fmt --base-url \$BASE
      done
      \$RUN benchmark.cli.run --model '${BENCH_MODEL}' --dataset \$ds \
        --pipeline sql --base-url \$BASE
    done
  "

# --- 3) Tear down the server ------------------------------------------------
kill "$VLLM_PID" 2>/dev/null || true
echo "benchmark job complete"
