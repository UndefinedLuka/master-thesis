"""Download and extract all datasets (Spider + BIRD) into ``data/``.

python -m cli.download_datasets
"""

from __future__ import annotations

from config.paths import ProjectPaths
from config.registry import DATASET_NAMES, get_download_spec
from datasets.download import ensure_dataset


def main(argv: list[str] | None = None) -> int:
    """Download and extract every registered dataset (idempotent)."""
    paths = ProjectPaths.discover()
    for name in DATASET_NAMES:
        ensure_dataset(get_download_spec(name), paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
