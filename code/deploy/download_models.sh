#!/usr/bin/env bash
# Download the GGUF models listed in deploy/models.txt into models/.
#
#   bash deploy/download_models.sh                   # all uncommented URLs in deploy/models.txt
#   bash deploy/download_models.sh other-list.txt    # a different URL list
#   JOBS=1 bash deploy/download_models.sh            # sequential, with a live progress bar
#   HF_TOKEN=hf_... bash deploy/download_models.sh   # gated repos
#
# Only needs bash + curl, so it runs on the FRIDA login node, inside a container job, or
# locally — no Python environment required.
#
# Idempotent: a model already present in models/ is skipped, so re-running after adding a
# line to models.txt downloads only the new file. Interrupted downloads resume from the
# partial ".part" file; the final name appears only once the bytes are complete and verified
# to start with the GGUF magic (a 404 or gated-repo HTML page must never masquerade as a
# model). Same convention as src/datasets/download.py.

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
JOBS="${JOBS:-3}"

# ── worker mode ───────────────────────────────────────────────────────────────
# Re-invoked once per URL by the xargs fan-out below (keeps the parallel path a plain
# subprocess instead of an exported shell function).
if [[ "${1:-}" == "--fetch-one" ]]; then
  url="${2:?--fetch-one needs a URL}"
  name="${url##*/}"   # strip the path
  name="${name%%\?*}" # strip the ?download=true query
  dest="$MODELS_DIR/$name"
  part="$dest.part"

  if [[ -s "$dest" ]]; then
    echo "[$name] already present ($(du -h "$dest" | cut -f1)); skipping."
    exit 0
  fi

  # A progress meter is only readable when one download runs at a time on a terminal;
  # otherwise three curls interleave into noise (and Slurm logs fill with control codes).
  meter=(--no-progress-meter)
  [[ "$JOBS" == "1" && -t 1 ]] && meter=(--progress-bar)

  # ${arr[@]+...} guards the empty-array expansion, which is an unbound-variable error
  # under `set -u` on bash 3.2 (macOS).
  auth=()
  [[ -n "${HF_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer $HF_TOKEN")

  # -L: HF redirects to a CDN. --fail: turn a 404/403 into a non-zero exit instead of
  # silently writing the error page. -C -: resume a partial .part.
  fetch() {
    curl -L --fail --retry 5 --retry-delay 5 --retry-connrefused \
      "${meter[@]}" ${auth[@]+"${auth[@]}"} "$@" -o "$part" "$url"
  }

  echo "[$name] downloading -> $dest"
  if [[ -s "$part" ]]; then
    # A stale .part can be un-resumable (already complete -> HTTP 416, or bytes from an
    # older revision). Fall back to a clean restart rather than failing the whole run.
    if ! fetch -C -; then
      echo "[$name] resume failed; restarting from scratch." >&2
      rm -f "$part"
      fetch
    fi
  else
    fetch
  fi

  if [[ "$(head -c 4 "$part")" != "GGUF" ]]; then
    rm -f "$part"
    echo "[$name] ERROR: download does not start with the GGUF magic bytes." >&2
    echo "[$name] Check the URL — the repo/quant name is probably wrong or gated." >&2
    exit 1
  fi

  mv "$part" "$dest"
  echo "[$name] done ($(du -h "$dest" | cut -f1))."
  exit 0
fi

# ── driver mode ───────────────────────────────────────────────────────────────
LIST="${1:-$REPO_ROOT/deploy/models.txt}"
[[ -f "$LIST" ]] || { echo "URL list not found: $LIST" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is required but not installed." >&2; exit 1; }

urls=()
while IFS= read -r line; do
  line="${line%%#*}"                     # drop trailing comments
  line="$(echo "$line" | tr -d '[:space:]')"
  [[ -n "$line" ]] && urls+=("$line")
done <"$LIST"

if [[ ${#urls[@]} -eq 0 ]]; then
  echo "No URLs in $LIST (every line is blank or commented out)." >&2
  exit 1
fi

mkdir -p "$MODELS_DIR"
echo "Downloading ${#urls[@]} model(s) from $LIST into $MODELS_DIR (JOBS=$JOBS)."

# Each --fetch-one worker is a fresh bash process; hand it the resolved settings.
export MODELS_DIR JOBS

# Invoked via `bash <path>` rather than as an executable so a missing exec bit (e.g. after a
# checkout on a filesystem that drops it) does not break the fan-out.
status=0
printf '%s\n' "${urls[@]}" \
  | xargs -P "$JOBS" -I{} bash "$SCRIPT_PATH" --fetch-one {} \
  || status=$?

echo
if [[ $status -ne 0 ]]; then
  echo "One or more downloads FAILED (see the errors above). Re-run to retry;" >&2
  echo "completed models are skipped and partial ones resume." >&2
else
  echo "All models present in $MODELS_DIR:"
fi
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "  (no .gguf files)"
exit $status
