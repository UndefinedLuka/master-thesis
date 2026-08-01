# Running the benchmark on FRIDA (docs.rdc.si)

FRIDA is a Slurm cluster with **no module system, no conda, and no user installs on the
login node** — everything runs inside Enroot/Pyxis containers. This directory contains the
sbatch templates and the runbook to get from zero to full benchmark runs.

## 0. Access (one-time)

Access is granted via your supervisor (they email the request to `frida@rdc.si`).
Authentication is MFA via Teleport:

```bash
# local machine
tsh login --proxy=rdc.si          # browser-based MFA
tsh config >> ~/.ssh/config       # makes plain ssh/scp/rsync work
ssh login-frida.rdc.si            # login node (port 3022 via the Teleport proxy)
```

VS Code Remote-SSH works with the generated Host entry ("Use Local Server" disabled).
Accounts idle for 6 months are disabled and their data archived — keep a local copy of
everything important anyway ($WORK has **no backups**).

## 1. Storage layout

| Variable   | Path                                | Use                                        |
|------------|-------------------------------------|--------------------------------------------|
| `$HOME`    | `/shared/home/$USER`                | dotfiles, small scripts                    |
| `$WORK`    | `/shared/workspace/<account>`       | repo, datasets, GGUF models, results, HF cache |
| `$SCRATCH` | per-job local RAID0                 | temp files; purged when the job ends       |

Suggested project layout under `$WORK`:

```
$WORK/thesis/
  code/           # this repo (rsync or git clone)
  containers/        # saved .sqsh container images
  hf-cache/          # HF_HOME (BERTScore downloads roberta-large here once)
```

Copy code + models + datasets from your machine (datasets are easier to download locally —
Spider comes from Google Drive via gdown, which is awkward from a batch node):

```bash
rsync -avP --exclude .venv --exclude results code/ login-frida.rdc.si:\$WORK/thesis/code/
rsync -avP code/models/*.gguf login-frida.rdc.si:\$WORK/thesis/code/models/
rsync -avP code/data/ login-frida.rdc.si:\$WORK/thesis/code/data/
```

If `data/` was not synced: run `python -m cli.download_datasets`, then
`python -m cli.build_sampled_dbs`, then `python -m cli.build_references` inside a container
job (compute nodes have outbound internet).

## 2. Build the container (one-time per dependency change)

```bash
sbatch deploy/build_container.sbatch
```

This starts from an NGC PyTorch image (CUDA + torch preinstalled), installs the project with
`uv` against the container's Python (the project needs >=3.12), **compiles llama-cpp-python
with CUDA** (`GGML_CUDA=on` — the PyPI wheel is CPU-only on Linux), and saves the result as
`$WORK/thesis/containers/thesis-eval.sqsh`. Benchmark jobs then start instantly from that
squashfs instead of recompiling.

Home is NOT auto-mounted in containers; every job template mounts `$WORK` explicitly.

## 3. Smoke test on the dev partition (interactive)

```bash
srun -p dev --gres=gpu:A100_80GB:1 -c 16 --mem=64G --time=2:00:00 \
     --container-image $WORK/thesis/containers/thesis-eval.sqsh \
     --container-mounts $WORK/thesis:/work --container-workdir /work/code \
     --pty bash

# inside the container:
python -m cli.run_benchmark --dataset spider --pipeline text_to_sql \
       --model Qwen3.5-0.8B-Q4_K_M.gguf --limit 10
```

Two things to verify on the first smoke run:

1. **GPU offload** — the llama.cpp load log must show layers assigned to CUDA. If everything
   sits on CPU, the wheel was built without CUDA; rebuild the container.
2. **KV-cache clearing still works** — `LlamaCppBackend._hard_reset()` calls the private
   `_ctx.kv_cache_clear()`, which is version-sensitive. Run the same question twice and check
   the second TTFT is NOT near zero; a collapsed TTFT means prefix caching leaked back and
   the llama-cpp-python version needs pinning/adapting.

## 4. Production runs (frida partition)

```bash
# one slice: <dataset> <pipeline> <model.gguf from models/>
sbatch deploy/run_benchmark.sbatch spider direct Qwen3.5-0.8B-Q4_K_M.gguf
sbatch deploy/run_benchmark.sbatch bird text_to_sql Ministral-3-3B-Instruct-2512-Q4_K_M.gguf

# or everything: all models in models/ x both datasets x both pipelines
sbatch deploy/run_all.sbatch
```

Results are **resumable**: if a job hits its walltime or is preempted, `sbatch` it again and
it continues where it stopped (nothing is lost on a normal timeout/kill; a hard node crash
re-runs at most one fsync interval of examples — see `FSYNC_INTERVAL` in
`src/results/store.py`). Walltime on `frida` maxes out at 7 days; size `--time` per slice
and requeue as needed.

Config knobs (edit `src/config/config.py` before syncing, or in `$WORK` directly):
- `N_THREADS` — set to the job's `--cpus-per-task` (templates use 16).
- `N_GPU_LAYERS = -1` (all layers on GPU) and `N_CTX = 131072` stay as-is; one H100/A100-80GB
  fits any model in `models/` plus the 128K KV cache comfortably for a single sequence, but
  if model load OOMs, lower `N_CTX` first.

## 5. Scoring

```bash
sbatch deploy/score_results.sbatch
```

Runs `score_results` over `results/` on a GPU (BERTScore/roberta-large; downloads once into
`HF_HOME=$WORK/thesis/hf-cache`). Re-running skips already-scored slices.

Then `rsync` `results/` back to your machine for analysis:

```bash
rsync -avP login-frida.rdc.si:\$WORK/thesis/code/results/ code/results/
```
