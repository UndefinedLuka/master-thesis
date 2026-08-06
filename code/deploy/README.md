# Running the benchmark on FRIDA (docs.rdc.si)

FRIDA is a Slurm cluster with **no module system, no conda, and no user installs on the
login node** — everything runs inside Enroot/Pyxis containers. This directory contains the
sbatch templates and the runbook to get from zero to full benchmark runs.

## TL;DR — the whole sequence

Everything below runs from the project dir on the login node, with code + `data/` already synced:

```bash
cd /shared/workspace/linf/<user>/thesis     # dir holding code/ containers/ hf-cache/

# 1. models (login node, no job — a few minutes)
bash code/deploy/download_models.sh

# 2. container (one-time, ~20-40 min)
sbatch code/deploy/build_container.sbatch
#    wait for it to finish, then confirm the image exists:
ls -lh containers/thesis-eval.sqsh

# 3. run the models in parallel — one H100 each, all 4 slices per task
sbatch --array=0-2 code/deploy/run_models.sbatch \
    Qwen3.5-0.8B-Q4_K_M.gguf \
    Qwen3.5-2B-Q4_K_M.gguf \
    Qwen3.5-4B-Q4_K_M.gguf

squeue -u "$USER"                           # watch; logs are thesis-run-model-<A>_<a>.out

# 4. score once the array is done
sbatch code/deploy/score_results.sbatch
```

Each step is idempotent/resumable — re-submit anything that times out. Details, GPU rationale and
smoke-test instructions follow.

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
everything important anyway (the workspace filesystem has **no backups**).

## 1. Layout

The templates use **no site environment variables** (`$WORK` and friends are not defined on the
login node). Every path is derived from the directory you submit from, so the only requirement
is this relative layout:

```
thesis/               # the project dir — mounted into the container as /work
  code/               # this repo: deploy/ src/ models/ data/ results/
  containers/         # thesis-eval.sqsh
  hf-cache/           # HF_HOME (BERTScore's roberta-large; created automatically)
```

Submit from either `thesis/` or `thesis/code/` — both are detected. **Every command below assumes
you are in `thesis/`**, hence the `code/deploy/...` prefixes; from `thesis/code/` drop the `code/`:

```bash
cd /shared/workspace/linf/<user>/thesis        # <- the working dir used throughout this README
sbatch code/deploy/run_models.sbatch ...
```

The shared derivation lives in `deploy/job_env.sh`, sourced by every template. It prints the
resolved `project=` / `code=` / `image=` at the top of each job log, and fails fast with a clear
message if you submit from somewhere else or the container image is missing. `THESIS_CODE`,
`THESIS_PROJECT` and `THESIS_IMAGE` override the detection if the layout ever changes.

For reference, the cluster's own storage: `$HOME` = `/shared/home/$USER` (dotfiles),
`/shared/workspace/<account>` (the project lives here; **no backups**), and a per-job `$SCRATCH`
on local RAID0 that is purged when the job ends.

Copy code + datasets from your machine (datasets are easier to download locally — Spider comes
from Google Drive via gdown, which is awkward from a batch node). Models are better fetched
directly on FRIDA, see §2:

```bash
REMOTE=login-frida.rdc.si:/shared/workspace/linf/<user>/thesis
rsync -avP --exclude .venv --exclude results --exclude models code/ "$REMOTE/code/"
rsync -avP code/data/ "$REMOTE/code/data/"
```

If `data/` was not synced: run `python -m cli.download_datasets`, then
`python -m cli.build_sampled_dbs`, then `python -m cli.build_references` inside a container
job (compute nodes have outbound internet).

## 2. Models

Pull the GGUFs straight onto the cluster rather than rsyncing tens of GB over the Teleport proxy.
`deploy/models.txt` holds one HuggingFace download URL per line (blank lines and `#` comments
ignored — uncomment the models you want); `deploy/download_models.sh` needs only bash + curl,
so it runs on the login node with no container:

```bash
# on login-frida, in thesis/
bash code/deploy/download_models.sh                 # every uncommented URL in deploy/models.txt
JOBS=6 bash code/deploy/download_models.sh          # 6 concurrent downloads (default 3)
JOBS=1 bash code/deploy/download_models.sh          # sequential, with a live progress bar
HF_TOKEN=hf_... bash code/deploy/download_models.sh # gated repos
```

(The script resolves `models/` from its own location, so it always writes to `code/models/`
regardless of where you invoke it from.)

Each file lands as `models/<basename>.gguf`, where the basename is the URL's last path segment
minus `?download=true` — which is exactly the name you later pass to `--model`. A single file
by hand is just:

```bash
curl -L --fail --retry 5 --retry-delay 5 -C - \
  -o code/models/Qwen3.5-2B-Q4_K_M.gguf \
  "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf?download=true"
```

`-L` because HF redirects to a CDN, `--fail` so a 404 does not silently write an HTML error
page as a `.gguf`, `-C -` to resume, and the URL quoted so the shell keeps `?download=true`.

The script is idempotent and crash-safe: present models are skipped, interrupted downloads
resume from the `.part` file, and the final name appears only after the bytes are complete and
verified to start with the GGUF magic. So just re-run it after adding a line or losing a
connection. If login-node egress is ever blocked, `sbatch code/deploy/download_models.sbatch` does
the same thing from a compute node (no `--gres`, so it schedules almost immediately).

## 3. Build the container (one-time per dependency change)

```bash
sbatch code/deploy/build_container.sbatch
```

This starts from an NGC PyTorch image (CUDA + torch preinstalled), installs the project with
`uv` against the container's Python (the project needs >=3.12), **compiles llama-cpp-python
with CUDA** (`GGML_CUDA=on` — the PyPI wheel is CPU-only on Linux), and saves the result as
`thesis/containers/thesis-eval.sqsh`. Benchmark jobs then start instantly from that
squashfs instead of recompiling.

Home is NOT auto-mounted in containers; every job template mounts the project dir explicitly.

## 4. Smoke test on the dev partition (interactive)

`srun` is not an sbatch template, so it does not pick up `job_env.sh` — spell the paths out:

```bash
PROJECT="$PWD"          # .../thesis — the dir containing code/ and containers/

srun -p dev --gres=gpu:A100_80GB:1 -c 16 --mem=64G --time=2:00:00 \
     --container-image "$PROJECT/containers/thesis-eval.sqsh" \
     --container-mounts "$PROJECT:/work" --container-workdir /work/code \
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

The `dev` partition only offers L4/A100, so timings from a smoke run are **not** comparable with
production results — it is here to check that things work, not to measure. Delete any smoke
`results/` before the real run (`--limit` output is otherwise indistinguishable from a full slice
on resume).

## 5. Production runs (frida partition)

**Several models at once — the fast path.** `deploy/run_models.sbatch` is a job array with one
task per model, each on its own dedicated H100. Each task loads its model once and runs all four
of its slices (spider|bird × direct|text_to_sql), so wall-clock drops roughly linearly in the
number of models:

```bash
sbatch --array=0-2 code/deploy/run_models.sbatch \
    Qwen3.5-0.8B-Q4_K_M.gguf \
    Qwen3.5-2B-Q4_K_M.gguf \
    Qwen3.5-4B-Q4_K_M.gguf
```

`--array` indices are 0-based and index into the positional model list, so `--array=0-2` runs
all three concurrently. Variations:

```bash
# smoke test first — extra cli.run_all flags go through RUN_ARGS, since the positional
# arguments are the model list
sbatch --export=ALL,RUN_ARGS="--limit 10" --array=0-2 code/deploy/run_models.sbatch <models...>

sbatch --array=0-2%2 code/deploy/run_models.sbatch <models...>  # at most 2 running at a time
sbatch --array=1-1   code/deploy/run_models.sbatch <models...>  # re-run just the 2B (index 1)
```

Keep `--gres` as the template sets it. Overriding it per task would measure different models on
different hardware — see "Which GPU to ask for" below.

This is safe to parallelize because each task only ever writes
`results/<dataset>/<pipeline>/<format>/<model>.ndjson` and the filename is derived from the
model name — no two tasks touch the same file. Logs are `%A_%a` (array id + task id), one per
task. Under the hood it is `python -m cli.run_all --models <one model>`.

**Single slices, or everything sequentially:**

```bash
# one slice: <dataset> <pipeline> <model.gguf from models/>
sbatch code/deploy/run_benchmark.sbatch spider direct Qwen3.5-0.8B-Q4_K_M.gguf
sbatch code/deploy/run_benchmark.sbatch bird text_to_sql Ministral-3-3B-Instruct-2512-Q4_K_M.gguf

# all models in models/ x both datasets x both pipelines, one after another on one GPU
sbatch code/deploy/run_all.sbatch
```

Results are **resumable**: if a job hits its walltime or is preempted, `sbatch` it again and
it continues where it stopped (nothing is lost on a normal timeout/kill; a hard node crash
re-runs at most one fsync interval of examples — see `FSYNC_INTERVAL` in
`src/results/store.py`). Walltime on `frida` maxes out at 7 days; size `--time` per slice
and requeue as needed.

### Which GPU to ask for

**All models run on H100 (`--gres=gpu:H100:1`), each on a card of its own.** This is not only
the fastest option, it is required for the results to mean anything: the benchmark records TTFT
and tokens/s, so a model measured on an A100 is not comparable to one measured on an H100. Two
models sharing a single card would be just as bad — they would contend for HBM bandwidth and
distort both models' timings. Keep the GPU type uniform across the whole campaign; if H100s are
backed up, move the *entire* array to another type rather than mixing, and re-run anything
already measured on the old one.

Why H100 specifically: the workload is single-sequence greedy decoding (`TEMPERATURE = 0.0`),
so it is **memory-bandwidth bound, not compute bound** — HBM bandwidth sets tokens/s and the
compute class barely matters. H100 ≈ 3.35 TB/s vs A100-80GB ≈ 2.0 TB/s (so ~1.7× faster
decoding) vs L4 ≈ 0.3 TB/s. VRAM is dominated by the **KV cache**, not the weights: a 4B model
at Q4_K_M is only ~2.5 GB of weights, but `N_CTX = 131072` makes the cache the big allocation
and it barely shrinks with model size — so 80 GB is the right target regardless of model size,
and L4 (24 GB) is both an OOM risk and ~10× slower.

H100 on FRIDA is a single DGX node with 8×80GB, so up to 8 tasks can run at once; it is also
the most contended partition resource. To keep the queue wait down:

1. **Check what's free before submitting:**
   ```bash
   sinfo -p frida -O "nodehost,gres:40,gresused:40,statelong"
   squeue -p frida -t R -O "gres,timeleft"
   ```
2. **Ask for little besides the GPU.** With `N_GPU_LAYERS = -1` the CPU does almost nothing, so
   `run_models.sbatch` requests 8 CPUs / 48 GB (vs 16 / 96 GB in the older templates) — much
   easier to backfill. Keep `N_THREADS` at 8 to match.
3. **Ask for a shorter `--time`.** 1 day per model rather than the 7 days `run_all.sbatch` needs
   for all of them sequentially. Short jobs backfill into gaps, and runs are resumable, so
   requeue instead of over-requesting.
4. **Submit the array in one go** so all models queue together and start as GPUs free up. Add
   `%N` (e.g. `--array=0-2%2`) only if you hit a per-user GPU cap.

Config knobs (edit `src/config/config.py` before syncing, or on the cluster directly):
- `N_THREADS` — set to the job's `--cpus-per-task` (8 in `run_models.sbatch`, 16 in the others).
- `N_GPU_LAYERS = -1` (all layers on GPU) and `N_CTX = 131072` stay as-is; one H100/A100-80GB
  fits any model in `models/` plus the 128K KV cache comfortably for a single sequence, but
  if model load OOMs, lower `N_CTX` first.

## 6. Scoring

```bash
sbatch code/deploy/score_results.sbatch
```

Runs `score_results` over `results/` on a GPU (BERTScore/roberta-large; downloads once into
`HF_HOME=thesis/hf-cache`, which the job creates). Re-running skips already-scored slices.

Then `rsync` `results/` back to your machine for analysis:

```bash
rsync -avP login-frida.rdc.si:/shared/workspace/linf/<user>/thesis/code/results/ code/results/
```
