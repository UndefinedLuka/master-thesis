#!/bin/bash
#SBATCH --job-name=thesis-build-data
#SBATCH --partition=frida
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out

# Prepare the datasets the benchmark reads: download -> sample BIRD -> reference answers.
#
#   sbatch code/deploy/build_data.sbatch                  # download (if needed) + sample + refs
#   sbatch code/deploy/build_data.sbatch --overwrite      # REBUILD the BIRD sampled DBs + refs
#   sbatch code/deploy/build_data.sbatch --skip-download  # data/ already downloaded
#
# No --gres: this is pure CPU/IO (sqlite + zip extraction), so a GPU-free job schedules almost
# immediately instead of queueing behind the benchmark. Compute nodes have outbound internet,
# so the downloads work from here.
#
# Pass --overwrite whenever BIRD_SAMPLE_ROWS or SAMPLE_SEED in src/config/config.py changes.
# Without it, build_sampled_dbs SKIPS every database that already exists and you keep the OLD
# sample while believing you regenerated it. References are always rewritten, since they must
# describe the rows now on disk.
#
# The three steps in order, and why the order matters:
#   1. cli.download_datasets  - Spider (gdrive) + BIRD (http zip); idempotent, skips if present.
#   2. cli.build_sampled_dbs  - writes data/bird_sampled/<db>/<db>.sqlite, BIRD_SAMPLE_ROWS rows
#                               per table at SAMPLE_SEED. Everything downstream reads THESE.
#   3. cli.build_references   - executes gold SQL against the sampled DBs -> data/*.ndjson.
#                               Must run after (2) or the reference rows describe data the
#                               models never see.
#
# WARNING: regenerating the BIRD sample invalidates existing BIRD results and their scores —
# they were produced against the old rows. Delete results/bird/ (and its .scored.ndjson files)
# before re-running the benchmark, or resume will keep the stale records.

set -euo pipefail

OVERWRITE=0
SKIP_DOWNLOAD=0
for arg in "$@"; do
  case "$arg" in
    --overwrite) OVERWRITE=1 ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    *) echo "unknown flag: $arg (accepts --overwrite, --skip-download)" >&2; exit 1 ;;
  esac
done

_SUBMIT="${SLURM_SUBMIT_DIR:-$PWD}"
_ENV="$_SUBMIT/deploy/job_env.sh"; [[ -f "$_ENV" ]] || _ENV="$_SUBMIT/code/deploy/job_env.sh"
[[ -f "$_ENV" ]] || { echo "ERROR: no deploy/job_env.sh under $_SUBMIT or $_SUBMIT/code -- sbatch from the thesis/ or thesis/code/ directory." >&2; exit 1; }
source "$_ENV"

run_in_container() {
  srun --container-image "$IMAGE" \
       --container-mounts "$PROJECT:/work" \
       --container-workdir "/work/$CODE_NAME" \
       "$@"
}

if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
  echo "=== 1/3 download datasets ==="
  run_in_container python -m cli.download_datasets
else
  echo "=== 1/3 download skipped (--skip-download) ==="
fi

echo "=== 2/3 materialize sampled BIRD databases (overwrite=$OVERWRITE) ==="
if [[ $OVERWRITE -eq 1 ]]; then
  run_in_container python -m cli.build_sampled_dbs --overwrite
else
  run_in_container python -m cli.build_sampled_dbs
fi

echo "=== 3/3 build reference answers (both datasets) ==="
run_in_container python -m cli.build_references

echo "=== done ==="
