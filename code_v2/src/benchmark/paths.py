"""Path resolution anchored on the project root.

All data/result paths in the config are relative to BENCH_ROOT. We resolve it
from the ``BENCH_ROOT`` environment variable (set by the SLURM container to
``/work``) and otherwise fall back to the ``code_v2`` directory inferred from
this file's location. This keeps the same code working both locally and inside
the Enroot container where the checkout is mounted at a different absolute path.
"""

from __future__ import annotations

import os
from pathlib import Path

# This file lives at <code_v2>/src/benchmark/paths.py -> parents[2] is <code_v2>.
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def bench_root() -> Path:
    """Return the project root (``code_v2`` dir or ``$BENCH_ROOT``)."""
    env = os.environ.get("BENCH_ROOT")
    return Path(env).resolve() if env else _DEFAULT_ROOT


def resolve(path: str | Path) -> Path:
    """Resolve ``path`` against :func:`bench_root` unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else bench_root() / p


def processed_dir(dataset: str) -> Path:
    """Directory holding serialized databases for ``dataset``."""
    return bench_root() / "data" / "processed" / dataset


def processed_db_dir(dataset: str, db_id: str) -> Path:
    return processed_dir(dataset) / db_id


def manifest_file(dataset: str) -> Path:
    return processed_dir(dataset) / "manifest.json"


def reference_file(dataset: str) -> Path:
    return bench_root() / "data" / "reference" / f"reference_{dataset}.json"


def results_dir() -> Path:
    return bench_root() / "results"


def results_shard(model_name: str, dataset: str, pipeline: str, fmt: str | None) -> Path:
    """NDJSON shard path for one (model, dataset, pipeline[, format]) run."""
    stem = f"{model_name}_{dataset}_{pipeline}"
    if fmt:
        stem += f"_{fmt}"
    return results_dir() / f"{stem}.ndjson"
