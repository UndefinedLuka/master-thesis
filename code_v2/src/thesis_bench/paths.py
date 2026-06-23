"""Centralized path resolution.

``project_root`` resolves to the ``code_v2`` directory regardless of where the
process is launched from (this file lives at ``src/thesis_bench/paths.py``, so we
walk up two parents). All other paths are derived from it or from configuration,
keeping path logic out of the rest of the codebase.
"""

from __future__ import annotations

from pathlib import Path

# code_v2/src/thesis_bench/paths.py -> code_v2/
project_root: Path = Path(__file__).resolve().parent.parent.parent

# Default on-disk locations. Datasets live under ``data/`` and are overridable via
# DatasetConfig; results default to ``results/``.
default_data_dir: Path = project_root / "data"
default_results_dir: Path = project_root / "results"

# Prompt templates ship inside the package.
templates_dir: Path = Path(__file__).resolve().parent / "prompts" / "templates"
