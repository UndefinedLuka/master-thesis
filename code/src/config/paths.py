"""Project path resolution.

Replaces v1's single global ``base_path`` with an explicit, injectable value object so
modules no longer depend on import-time global state (and tests can point at a temp dir).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """All directories the pipeline reads from / writes to, derived from one root."""

    root: Path
    data_dir: Path
    models_dir: Path
    results_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> ProjectPaths:
        root = root.resolve()
        return cls(
            root=root,
            data_dir=root / "data",
            models_dir=root / "models",
            results_dir=root / "results",
        )

    @classmethod
    def discover(cls, start: Path | None = None) -> ProjectPaths:
        """Locate the project root.

        Walks upward from ``start`` (default: this file) looking for a ``pyproject.toml``.
        Falls back to the directory two levels above this file.
        """
        here = (start or Path(__file__)).resolve()
        for candidate in (here, *here.parents):
            if (candidate / "pyproject.toml").is_file():
                return cls.from_root(candidate)
        # src/config/paths.py -> project root (2 levels up)
        return cls.from_root(Path(__file__).resolve().parents[2])

    def dataset_dir(self, name: str) -> Path:
        """Root directory for a dataset's downloaded files, e.g. ``data/spider``."""
        return self.data_dir / name

    def processed_dir(self, name: str) -> Path:
        """Where serialized format files are cached, e.g. ``data/spider_processed``."""
        return self.data_dir / f"{name}_processed"
