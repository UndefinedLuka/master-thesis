"""NDJSON result persistence with crash-safe resumability.

Results are split into one NDJSON file per ``(dataset, pipeline, format, model)`` slice, laid
out as ``<results_dir>/<dataset>/<pipeline>/<format>/<model>.ndjson`` so no single file grows
huge and a given slice is easy to locate and inspect. Text-to-SQL is schema-only (no
serialization format); its files live under a ``schema`` folder. Completion is keyed on
``(example_id, pipeline, format)`` so re-running skips finished work.

Crash safety (so re-running continues from the last processed request even after a hard kill):
  * each record is flushed to the OS before the run proceeds, so a process kill (SIGKILL,
    scheduler preemption) loses nothing; an ``fsync`` every ``_FSYNC_INTERVAL`` records bounds
    what a hard OS/power crash can lose to at most one interval (per-record fsync would stall
    on APFS/network filesystems);
  * if a previous process died mid-write leaving a partial trailing line, the next append first
    terminates that line with a newline so records never get glued together;
  * the loader tolerates a corrupt/partial line (skips it with a warning) instead of raising,
    so resume always succeeds — unfinished examples are simply re-run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from config.settings import RunConfig
from pipelines.types import PipelineResult
from serialization.types import SerializationFormat

CompletedKey = tuple[str, str, str | None]

#: Folder standing in for the (absent) format dimension of the schema-only Text-to-SQL run.
NO_FORMAT_DIR = "schema"


def _slug(text: str) -> str:
    return text.replace("/", "-").replace(".", "-").replace(":", "-")


def result_dir(cfg: RunConfig, fmt: SerializationFormat | None) -> Path:
    """Folder for one slice: ``<results_dir>/<dataset>/<pipeline>/<format>``."""
    fmt_dir = fmt if fmt is not None else NO_FORMAT_DIR
    return cfg.results_dir / cfg.dataset.name / cfg.pipeline / fmt_dir


def result_path(cfg: RunConfig, fmt: SerializationFormat | None) -> Path:
    """Output file for one slice: ``<dataset>/<pipeline>/<format>/<model>.ndjson``."""
    return result_dir(cfg, fmt) / f"{_slug(cfg.backend.model_id)}.ndjson"


def completed_key(result: PipelineResult) -> CompletedKey:
    """Identity used for resume: example + pipeline + format (format is None for SQL)."""
    return (result["example_id"], result["pipeline"], result["format"])


FSYNC_INTERVAL = 50  # fsync once every N records per slice file


class ResultWriter:
    """Appends results to one NDJSON slice file, tracking its own fsync cadence.

    Each writer owns the append count for its file (no shared module state), so fsync
    behavior is explicit and testable. Create one writer per output file per run.
    """

    def __init__(self, path: Path, *, fsync_interval: int = FSYNC_INTERVAL) -> None:
        self._path = path
        self._fsync_interval = fsync_interval
        self._count = 0

    @property
    def path(self) -> Path:
        return self._path

    def append(self, result: PipelineResult) -> None:
        """Append one result as a JSON line (flushed always; fsynced every interval).

        Repairs a partial trailing line left by a previously killed process (so records are
        never glued together). The interval fsync bounds hard-OS-crash loss to at most one
        interval of records; a plain process kill loses nothing because every record is
        flushed.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        _repair_trailing_newline(self._path)
        line = json.dumps(result, ensure_ascii=False) + "\n"
        self._count += 1
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            if self._count % self._fsync_interval == 0:
                os.fsync(handle.fileno())


def _repair_trailing_newline(path: Path) -> None:
    """If the file ends without a newline (an interrupted write), append one.

    This isolates the truncated final record on its own line so the loader can drop just that
    line and the next append starts cleanly on a fresh line.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return
    with path.open("ab") as handle:
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_results(path: Path, results: list[PipelineResult]) -> None:
    """Write a full list of results as NDJSON (overwrites). Used by the scoring pass."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def load_results(path: Path) -> list[PipelineResult]:
    """Read all results from an NDJSON file (empty list if it doesn't exist).

    A malformed line (e.g. a truncated final record from a killed process) is skipped with a
    warning rather than raising, so resume never fails because of one unfinished example — that
    example simply gets re-run.
    """
    if not path.exists():
        return []
    results: list[PipelineResult] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"warning: skipping malformed line {line_no} in {path} "
                    f"(likely a truncated record from an interrupted run)",
                    file=sys.stderr,
                )
    return results


def load_completed_keys(path: Path) -> set[CompletedKey]:
    """Set of already-completed keys for resuming a run."""
    return {completed_key(result) for result in load_results(path)}
