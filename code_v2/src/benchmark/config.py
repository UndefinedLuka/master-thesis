"""Central typed configuration, loaded once from ``config.toml`` and passed explicitly.

No module-global singletons: everything the pipelines need is carried on the
:class:`Config` object so multiple models/datasets can be driven cleanly.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from benchmark import paths

DEFAULT_FORMATS: tuple[str, ...] = ("json", "xml", "yaml", "csv", "markdown", "toon")


@dataclass(frozen=True)
class ModelSpec:
    """One open-weight model. ``model_id`` is passed verbatim to vLLM and the API."""

    name: str  # filesystem-safe label used in result filenames
    model_id: str  # HF id or local path: vLLM --model and OpenAI `model`
    quantization: str | None = None  # informational; real value is the vLLM launch arg


@dataclass(frozen=True)
class DatasetSpec:
    """A Text-to-SQL dataset (Spider or BIRD)."""

    name: str
    questions_file: Path
    db_root: Path
    tables_file: Path | None = None
    subsample_rows: int | None = None  # None => use full tables (Spider)
    has_evidence: bool = False  # BIRD ships external-knowledge "evidence"

    def question_db_path(self, db_id: str) -> Path:
        """Path to ``<db_root>/<db_id>/<db_id>.sqlite``."""
        return self.db_root / db_id / f"{db_id}.sqlite"


@dataclass(frozen=True)
class Config:
    datasets: dict[str, DatasetSpec]
    models: tuple[ModelSpec, ...]
    formats: tuple[str, ...] = DEFAULT_FORMATS
    bird_sample_n: int = 100
    sql_retry_cap: int = 3
    seed: int = 42
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    request_timeout_s: float = 600.0
    max_output_tokens: int = 2048
    temperature: float = 0.0

    def model(self, name: str) -> ModelSpec:
        for m in self.models:
            if m.name == name or m.model_id == name:
                return m
        raise KeyError(f"Model '{name}' not in config (have: {[m.name for m in self.models]})")

    def dataset(self, name: str) -> DatasetSpec:
        try:
            return self.datasets[name]
        except KeyError:
            raise KeyError(
                f"Dataset '{name}' not in config (have: {list(self.datasets)})"
            ) from None


def _dataset_from_toml(name: str, raw: dict, bird_sample_n: int) -> DatasetSpec:
    sub = raw.get("subsample_rows", 0) or 0
    # A dataset flagged with evidence (BIRD) with no explicit sub-sample falls
    # back to the global default; Spider keeps full tables (None).
    subsample: int | None
    if sub > 0:
        subsample = sub
    elif raw.get("has_evidence", False):
        subsample = bird_sample_n
    else:
        subsample = None
    tables = raw.get("tables_file")
    return DatasetSpec(
        name=name,
        questions_file=paths.resolve(raw["questions_file"]),
        db_root=paths.resolve(raw["db_root"]),
        tables_file=paths.resolve(tables) if tables else None,
        subsample_rows=subsample,
        has_evidence=bool(raw.get("has_evidence", False)),
    )


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate ``config.toml`` (defaults to ``<BENCH_ROOT>/config.toml``)."""
    cfg_path = Path(path) if path else paths.bench_root() / "config.toml"
    with open(cfg_path, "rb") as f:
        raw = tomllib.load(f)

    bird_sample_n = int(raw.get("bird_sample_n", 100))

    datasets = {
        name: _dataset_from_toml(name, draw, bird_sample_n)
        for name, draw in raw.get("datasets", {}).items()
    }
    models = tuple(
        ModelSpec(
            name=m["name"],
            model_id=m["model_id"],
            quantization=(m.get("quantization") or None),
        )
        for m in raw.get("models", [])
    )

    formats = tuple(raw.get("formats", DEFAULT_FORMATS))

    return Config(
        datasets=datasets,
        models=models,
        formats=formats,
        bird_sample_n=bird_sample_n,
        sql_retry_cap=int(raw.get("sql_retry_cap", 3)),
        seed=int(raw.get("seed", 42)),
        base_url=str(raw.get("base_url", "http://localhost:8000/v1")),
        api_key=str(raw.get("api_key", "EMPTY")),
        request_timeout_s=float(raw.get("request_timeout_s", 600.0)),
        max_output_tokens=int(raw.get("max_output_tokens", 2048)),
        temperature=float(raw.get("temperature", 0.0)),
    )


def with_overrides(cfg: Config, **overrides) -> Config:
    """Return a copy of ``cfg`` with the given non-None top-level fields replaced."""
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(cfg, **clean) if clean else cfg
