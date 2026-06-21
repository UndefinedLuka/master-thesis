#!/bin/bash
# Two-job alternative: a long-lived vLLM server.
#
# Use this when you want to keep the server up and point several client runs at
# it. It prints the node hostname; submit your client against
#   --base-url http://<node>:${VLLM_PORT}/v1
# A client job must be able to reach that node. The simplest way is to attach to
# this allocation from an interactive shell:
#
#   sbatch slurm/serve_vllm.sbatch          # note the JOBID and printed hostname
#   srun --jobid=<JOBID> --overlap --pty bash
#   # inside: cd /work && uv run python -m benchmark.cli.run \
#   #           --model <model> --dataset spider --pipeline sql \
#   #           --base-url http://localhost:${VLLM_PORT}/v1
#
# For most users the single-job slurm/run_benchmark.sbatch is simpler.
#
#SBATCH --job-name=vllm-serve
#SBATCH --partition=frida
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm-%j.out

set -euo pipefail
mkdir -p logs
source "$(dirname "$0")/env.sh"

QUANT_ARG=""
[ -n "$BENCH_QUANT" ] && QUANT_ARG="--quantization $BENCH_QUANT"

echo "vLLM server node: $(hostname)"
echo "Point clients at: http://$(hostname):${VLLM_PORT}/v1"

srun --container-image="${BENCH_IMAGE}" \
     --container-mounts="${BENCH_REPO}:/work,${HF_HOME}:/hf" \
     --container-workdir=/work --no-container-remap-root \
  bash -lc "
    set -e
    export HF_HOME=/hf
    pip install --no-cache-dir 'vllm>=0.6' >/dev/null 2>&1 || true
    python -m vllm.entrypoints.openai.api_server \
      --model '${BENCH_MODEL}' ${QUANT_ARG} \
      --host 0.0.0.0 --port ${VLLM_PORT} \
      --max-model-len ${VLLM_MAX_MODEL_LEN}
  "
