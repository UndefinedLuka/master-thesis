#!/bin/bash
# Shared configuration sourced by the FRIDA SLURM scripts.
# Edit these values for your run, then `sbatch slurm/run_benchmark.sbatch`.

# Container image: an NGC image with CUDA/PyTorch, or a prebuilt vLLM squashfs
# (recommended for reproducibility: `enroot import` a vllm/vllm-openai image and
# point this at the .sqsh file). The NGC pytorch image below pip-installs vLLM at
# runtime as a convenience for first experiments.
export BENCH_IMAGE="${BENCH_IMAGE:-nvcr.io#nvidia/pytorch:24.10-py3}"

# Model served by vLLM. Passed verbatim as --model and used as the OpenAI `model`
# string by the client, so full-precision vs. quantized never changes test code.
export BENCH_MODEL="${BENCH_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

# Optional vLLM quantization (e.g. awq, gptq, fp8). Empty = full precision.
export BENCH_QUANT="${BENCH_QUANT:-}"

# vLLM server port (client talks to localhost in the single-job layout).
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"

# Absolute path to the code_v2 checkout on the cluster filesystem.
export BENCH_REPO="${BENCH_REPO:-$HOME/master-thesis/code_v2}"

# HuggingFace cache (mounted so model weights persist between jobs).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
