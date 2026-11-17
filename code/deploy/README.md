# FRIDA Cluster Deployment

Slurm and Enroot/Pyxis job scripts for running benchmarks and scoring on the [FRIDA](https://docs.rdc.si/) cluster. See [`../README.md`](../README.md) for package documentation.

## Files

| File | Purpose |
| --- | --- |
| [`models.txt`](models.txt) | HuggingFace download URLs for the 11 `Q4_K_M` / `UD-Q4_K_M` GGUF models |
| `download_models.sh` | Parallel `curl` downloader (`JOBS=3` default) into `code/models/` with `.part` resume and GGUF header check |
| `download_models.sbatch` | CPU Slurm job wrapper around `download_models.sh` (`8` CPUs, `16G` RAM, `4h`) |
| `job_env.sh` | Shared environment helper sourced by all `.sbatch` jobs; resolves `PROJECT`, `CODE_DIR`, and `IMAGE` |
| `build_container.sbatch` | Builds `containers/thesis-eval.sqsh` from `nvcr.io#nvidia/pytorch:25.06-py3` (`A100_80GB:1`, CUDA `llama-cpp-python` + `.[metrics]`) |
| `build_data.sbatch` | CPU job (`8` CPUs, `64G` RAM, `8h`) running `download_datasets`, `build_sampled_dbs`, and `build_references` |
| `run_models.sbatch` | H100 job array (`H100:1`, `8` CPUs, `48G` RAM, `24h`) running one model per array task across all datasets and pipelines |
| `run_all.sbatch` | Sequential single-GPU job (`H100:1`, `16` CPUs, `96G` RAM, `7d`) running all GGUF models in `code/models/` |
| `run_benchmark.sbatch` | Single-slice job (`H100:1`, `16` CPUs, `96G` RAM, `24h`) taking `<dataset> <pipeline> <model.gguf>` |
| `score_results.sbatch` | GPU scoring job (`gpu:1`, `16` CPUs, `64G` RAM, `12h`) writing `*.scored.ndjson` under `code/results/` |

## Workspace Layout & Sync

Create a project directory on FRIDA with the following layout (`containers/` and `hf-cache/` are created automatically by the jobs):

```text
thesis/
├── code/          # synced from local code/
├── containers/    # thesis-eval.sqsh
└── hf-cache/      # HuggingFace cache (HF_HOME for BERTScore)
```

From the local repository root, sync `code/` to the cluster:

```bash
REMOTE="login-frida.rdc.si:/shared/workspace/<account>/<user>/thesis"
rsync -avP --exclude .venv --exclude results --exclude models --exclude data \
  code/ "$REMOTE/code/"
```

Adjust experiment settings in [`../src/config/config.py`](../src/config/config.py) and uncomment desired model URLs in [`models.txt`](models.txt) before running. Path resolution in `job_env.sh` can be overridden via `THESIS_PROJECT`, `THESIS_CODE`, and `THESIS_IMAGE`.

## Execution Workflow

Submit jobs from `thesis/` (or `thesis/code/`). Wait for each stage to finish before starting the next.

### 1. Download GGUF Models

Run directly on the login node (or submit `sbatch code/deploy/download_models.sbatch`):

```bash
bash code/deploy/download_models.sh
# Optional overrides: JOBS=1, MODELS_DIR=/path, HF_TOKEN=hf_...
```

### 2. Build the Container Image

Run once, and re-run after changing Python dependencies in `pyproject.toml`:

```bash
sbatch code/deploy/build_container.sbatch
```

### 3. Prepare Datasets and References

```bash
sbatch code/deploy/build_data.sbatch
```

- `--overwrite`: rebuild sampled BIRD databases (`data/bird_sampled/`) and references after changing `BIRD_SAMPLE_ROWS` or `SAMPLE_SEED` (remove existing `code/results/bird/` files before rerunning benchmarks).
- `--skip-download`: skip archive downloads if `code/data/` is already populated.

### 4. Run Benchmarks

**Parallel job array (one model per H100 task):**

```bash
sbatch --array=0-2%2 code/deploy/run_models.sbatch \
  Qwen3.5-0.8B-Q4_K_M.gguf \
  Qwen3.5-2B-Q4_K_M.gguf \
  Qwen3.5-4B-Q4_K_M.gguf
```

- `--array=0-N%K`: zero-based indices into the positional model arguments; `%K` caps concurrent tasks.
- Pass extra `cli.run_all` flags via `RUN_ARGS`:
  ```bash
  sbatch --export=ALL,RUN_ARGS="--limit 10" --array=0-2 code/deploy/run_models.sbatch <models...>
  ```

**Alternatives:**

```bash
# Single dataset + pipeline slice
sbatch code/deploy/run_benchmark.sbatch spider direct Qwen3.5-0.8B-Q4_K_M.gguf

# All models in code/models/ sequentially on one H100
sbatch code/deploy/run_all.sbatch
```

Monitor jobs with `squeue -u "$USER"`. Logs are written to the submission directory (`<job-name>-<jobid>.out`, or `thesis-run-model-<jobid>_<taskid>.out` for arrays). Timed-out or preempted jobs resume automatically when resubmitted.

### 5. Score Results

```bash
sbatch code/deploy/score_results.sbatch
# Optional flags: --overwrite, --no-bertscore
```

## Retrieve Results

Sync raw and scored NDJSON files back to the local repository root:

```bash
rsync -avP "$REMOTE/code/results/" code/results/
```

Generate thesis tables and figures locally following [`../README.md`](../README.md#4-generate-thesis-tables-figures-and-audits).
