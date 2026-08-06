# Shared path setup for the sbatch templates. Sourced, not executed:
#
#   source "${SLURM_SUBMIT_DIR:-$PWD}/<...>/job_env.sh"
#
# Relies on NO site variables ($WORK, $SCRATCH, ... are not defined here). Everything is
# derived from the directory you submit from, which must be either the project dir or the
# repo checkout inside it:
#
#   thesis/               <- PROJECT: submit from here; mounted into the container as /work
#     code/               <- CODE_DIR: the repo (deploy/ src/ models/ data/ results/)
#     containers/         <- thesis-eval.sqsh
#     hf-cache/           <- HF_HOME for BERTScore's roberta-large
#
# So both of these work:
#
#   cd thesis      && sbatch code/deploy/run_models.sbatch ...
#   cd thesis/code && sbatch deploy/run_models.sbatch ...
#
# Anchoring on $SLURM_SUBMIT_DIR (rather than BASH_SOURCE) is deliberate: Slurm copies the
# batch script into a spool directory before running it, so inside a job BASH_SOURCE points at
# that throwaway copy, not at deploy/ in the repo. Slurm always sets SLURM_SUBMIT_DIR for
# batch jobs; the $PWD fallback covers running a template directly as a shell script.
#
# Overrides, if the layout ever differs: THESIS_CODE, THESIS_PROJECT, THESIS_IMAGE.

_looks_like_checkout() { [[ -d "$1/deploy" && -d "$1/src" ]]; }

_submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -n "${THESIS_CODE:-}" ]]; then
  CODE_DIR="$THESIS_CODE"
elif _looks_like_checkout "$_submit_dir"; then
  CODE_DIR="$_submit_dir"                    # submitted from the repo root
elif _looks_like_checkout "$_submit_dir/code"; then
  CODE_DIR="$_submit_dir/code"               # submitted from the project dir
else
  echo "ERROR: cannot locate the repo checkout from $_submit_dir" >&2
  echo "Submit from the project dir (containing code/) or from code/ itself, e.g." >&2
  echo "  cd /shared/workspace/<...>/thesis && sbatch code/deploy/<job>.sbatch" >&2
  echo "Or set THESIS_CODE=/path/to/code." >&2
  exit 1
fi

CODE_DIR="$(cd "$CODE_DIR" && pwd)"                                  # absolute, symlinks resolved
PROJECT="${THESIS_PROJECT:-$(cd "$CODE_DIR/.." && pwd)}"
CODE_NAME="$(basename "$CODE_DIR")"                                  # container workdir: /work/$CODE_NAME
IMAGE="${THESIS_IMAGE:-$PROJECT/containers/thesis-eval.sqsh}"

echo "project=$PROJECT  code=/work/$CODE_NAME  image=$IMAGE"

# Jobs other than the container build need the image to exist already; a missing .sqsh
# otherwise surfaces as an opaque Pyxis error deep in the job.
if [[ "${REQUIRE_IMAGE:-1}" == "1" && ! -f "$IMAGE" ]]; then
  echo "ERROR: container image not found: $IMAGE" >&2
  echo "Build it first: sbatch code/deploy/build_container.sbatch" >&2
  exit 1
fi
