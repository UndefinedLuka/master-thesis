# FRIDA deployment

Slurm job templates for running the benchmarks in Enroot/Pyxis containers on
[FRIDA](https://docs.rdc.si/). See the cluster documentation for account access and
SSH setup, and the [code README](../README.md) for the benchmark overview.

## Setup

Create a project directory on the cluster with this layout. The jobs create
`containers/` and `hf-cache/` automatically:

```text
thesis/
  code/
  containers/
  hf-cache/
```

From the local repository root, copy the code (replace the workspace placeholders):

```bash
REMOTE="login-frida.rdc.si:/shared/workspace/<account>/<user>/thesis"
rsync -avP --exclude .venv --exclude results --exclude models --exclude data \
  code/ "$REMOTE/code/"
```

Configure experiments in [`code/src/config/config.py`](../src/config/config.py) and
select model download URLs in [`models.txt`](models.txt).

## Run on the cluster

Submit from the project directory containing `code/`. Complete each stage successfully
before submitting the next; `sbatch` returns before its job finishes.

```bash
# Download selected GGUF models into code/models/
bash code/deploy/download_models.sh

# Build the CUDA-enabled container (repeat after dependency changes)
sbatch code/deploy/build_container.sbatch

# Download datasets, sample BIRD, and build reference answers
sbatch code/deploy/build_data.sbatch

# Run selected models across both datasets and pipelines, one H100 per task
sbatch --array=0-2%2 code/deploy/run_models.sbatch \
  Qwen3.5-0.8B-Q4_K_M.gguf \
  Qwen3.5-2B-Q4_K_M.gguf \
  Qwen3.5-4B-Q4_K_M.gguf

# Score after all benchmark tasks finish
sbatch code/deploy/score_results.sbatch
```

- Array indices are zero-based positions in the model list; `%2` limits concurrency.
  Adjust the range to match your models and preserve their order when resubmitting.
- Monitor jobs with `squeue -u "$USER"`. Logs appear in the submission directory;
  array logs use `thesis-run-model-<job>_<task>.out`.
- Benchmark jobs resume automatically when resubmitted. Keep GPU type and context
  window consistent across compared runs.
- After changing BIRD sampling settings, run `build_data.sbatch --overwrite` and
  move previous BIRD results outside `code/results/` before rerunning benchmarks.

For a single dataset and pipeline:

```bash
sbatch code/deploy/run_benchmark.sbatch spider direct Qwen3.5-0.8B-Q4_K_M.gguf
```

## Retrieve results

Results and scores are stored under `code/results/`. From the local repository root,
using the same `REMOTE` value as above:

```bash
rsync -avP "$REMOTE/code/results/" code/results/
```

See the [analysis instructions](../README.md#thesis-analysis) to generate thesis outputs.
