"""Typed run configuration.

Plain frozen dataclasses (no pydantic) keep dependencies minimal.
``DatasetName`` / ``BackendKind`` / ``PipelineKind`` are defined here because they are
fundamentally configuration choices; downstream packages import them from here to avoid
import cycles. All default values come from ``config.config``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast, get_args

from config.config import (
    API_KEY,
    BACKEND,
    BASE_URL,
    BERTSCORE_LANG,
    BERTSCORE_MODEL,
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    N_CTX,
    N_GPU_LAYERS,
    N_THREADS,
    SEED,
    TEMPERATURE,
)
from serialization.types import FORMATS, SerializationFormat

# These Literals are configuration choices, so they live here (a dependency-free leaf the
# whole package can import without cycles). pipelines.types re-imports PipelineKind from here.
BackendKind = Literal["llama_cpp", "openai_compat"]
DatasetName = Literal["spider", "bird"]
PipelineKind = Literal["direct", "text_to_sql"]

# MAX_OUTPUT_TOKENS = 0 means "uncapped"; backends expect None for that (0 would request
# zero-token generations from llama.cpp). Normalize once, here.
_MAX_TOKENS: int | None = MAX_OUTPUT_TOKENS if MAX_OUTPUT_TOKENS > 0 else None


def _validated_backend_kind(value: str) -> BackendKind:
    """Fail at import with a clear message if config.py holds a typo'd BACKEND."""
    valid = get_args(BackendKind)
    if value not in valid:
        raise ValueError(
            f"Invalid BACKEND {value!r} in config/config.py; valid values: {', '.join(valid)}"
        )
    return cast(BackendKind, value)


_BACKEND_KIND: BackendKind = _validated_backend_kind(BACKEND)


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Everything needed to construct an LLM backend."""

    kind: BackendKind = _BACKEND_KIND
    model_id: str = "model.gguf"  # gguf filename (llama_cpp) or served model name
    model_dir: Path | None = None  # llama_cpp: directory holding the .gguf
    base_url: str | None = BASE_URL  # openai_compat: e.g. http://localhost:8080/v1
    api_key: str | None = API_KEY
    n_ctx: int = N_CTX
    n_gpu_layers: int = N_GPU_LAYERS
    n_threads: int = N_THREADS
    temperature: float = TEMPERATURE
    seed: int = SEED
    max_tokens: int | None = _MAX_TOKENS


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Which dataset to run and where its databases live.

    BIRD sub-sampling parameters (BIRD_SAMPLE_ROWS / SAMPLE_SEED) are consumed by the
    build_sampled_dbs CLI at materialization time, not carried here — by run time the
    sampled databases already exist on disk under ``db_root``.
    """

    name: DatasetName
    dev_json: Path  # questions + gold SQL (dev split)
    db_root: Path  # directory containing <db_id>/<db_id>.sqlite


@dataclass(frozen=True, slots=True)
class RunConfig:
    """A single benchmark invocation."""

    backend: BackendConfig
    dataset: DatasetConfig
    pipeline: PipelineKind
    formats: tuple[SerializationFormat, ...] = FORMATS  # direct only; text_to_sql ignores
    max_retries: int = MAX_RETRIES
    results_dir: Path = field(default_factory=lambda: Path("results"))
    resume: bool = True


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Options for the score_results pass."""

    with_bertscore: bool = True
    model_type: str = BERTSCORE_MODEL
    lang: str = BERTSCORE_LANG
    force_order_sensitive: bool | None = None
    overwrite: bool = False


def default_db_root(name: DatasetName, data_dir: Path) -> Path:
    """Default ``<db_id>/<db_id>.sqlite`` parent dir per the layout datasets.download creates."""
    if name == "spider":
        return data_dir / "spider" / "spider_data" / "database"
    return data_dir / "bird" / "dev_20240627" / "dev_databases"


def sampled_db_root(data_dir: Path) -> Path:
    """Where materialized (sub-sampled) BIRD databases live (built by thesis-sample-dbs)."""
    return data_dir / "bird_sampled"


def effective_db_root(name: DatasetName, data_dir: Path) -> Path:
    """The database root an experiment must read from.

    Spider runs on the downloaded databases as-is. BIRD runs on the MATERIALIZED sampled
    copies so that serialization, reference answers, and Text-to-SQL execution all see the
    exact same rows (Chapter 3's level playing field). Build them with thesis-sample-dbs.
    """
    if name == "bird":
        return sampled_db_root(data_dir)
    return default_db_root(name, data_dir)


def default_dev_json(name: DatasetName, data_dir: Path) -> Path:
    """Default path to the dataset's dev split (questions + gold SQL)."""
    if name == "spider":
        return data_dir / "spider" / "spider_data" / "dev.json"
    return data_dir / "bird" / "dev_20240627" / "dev.json"
