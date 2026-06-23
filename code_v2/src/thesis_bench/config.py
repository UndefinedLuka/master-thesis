"""Frozen configuration dataclasses — the typed contract for a benchmark run.

Everything here is ``frozen=True, slots=True`` and uses ``tuple`` (never ``list``)
for collections so configs are immutable and hashable. The point is the user's
stated goal: *"typed so we are not passing wrong parameters."* A misspelled
generation knob cannot reach the backend because ``GenerationOptions`` is the only
path to it; an invalid format/pipeline/dataset cannot be constructed because those
are ``Enum`` members (see ``enums.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .enums import Backend, DatasetKind, PipelineKind, SerializationFormat

# Maximum number of *additional* generation attempts after the first one in the
# Text-to-SQL error-correction loop (ch-03: "up to three compilation retries").
DEFAULT_MAX_SQL_RETRIES = 3


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Decoding knobs forwarded to the backend.

    Defaults are tuned for reproducibility: greedy decoding (``temperature=0``)
    with a fixed ``seed``. These map onto Ollama's ``options`` payload; other
    backends translate them in their adapter.
    """

    temperature: float = 0.0
    seed: int = 0
    num_ctx: int = 128_000
    num_predict: int = -1  # -1 = until natural stop / context limit
    top_p: float = 1.0
    stop: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """A single model to benchmark.

    ``param_scale_b`` and ``quantization`` are recorded with every result so the
    analysis can group by parameter scale (an explicit experimental variable in
    ch-03) and confirm the quantization level was held constant.
    """

    backend: Backend
    model_id: str  # e.g. "qwen2.5:7b-instruct-q8_0" for Ollama
    param_scale_b: float  # billions of parameters, for analysis grouping
    quantization: str  # e.g. "Q8_0"; held constant across the model set
    options: GenerationOptions = field(default_factory=GenerationOptions)
    host: str | None = None  # backend endpoint override (e.g. OLLAMA_HOST)


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Where a dataset lives on disk and how to sample it.

    ``sample_rows is None`` means use full tables (Spider). An ``int`` caps each
    table to ``n`` rows (BIRD), sampled deterministically from ``sample_seed``.
    """

    kind: DatasetKind
    questions_path: Path
    db_root: Path
    sample_rows: int | None = None
    sample_seed: int = 42


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One cell of the experiment matrix: (model x dataset x format x pipeline).

    The orchestrator iterates the cartesian product of a ``BenchmarkConfig`` into
    these cells, each of which maps to exactly one results file.
    """

    model: ModelConfig
    dataset: DatasetConfig
    fmt: SerializationFormat
    pipeline: PipelineKind

    @property
    def cell_id(self) -> str:
        """Stable identifier used to name this cell's results file."""
        safe_model = self.model.model_id.replace(":", "-").replace("/", "-")
        return f"{self.dataset.kind.value}__{self.fmt.value}__{safe_model}__{self.pipeline.value}"


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """The whole experiment: every model x dataset x format x pipeline combination.

    ``expand()`` materializes the matrix into individual ``RunConfig`` cells.
    """

    models: tuple[ModelConfig, ...]
    datasets: tuple[DatasetConfig, ...]
    formats: tuple[SerializationFormat, ...]
    pipelines: tuple[PipelineKind, ...]
    results_dir: Path
    max_sql_retries: int = DEFAULT_MAX_SQL_RETRIES
    question_limit: int | None = None  # cap questions per db for quick runs
    bertscore_enabled: bool = False

    def expand(self) -> tuple[RunConfig, ...]:
        cells: list[RunConfig] = []
        for dataset in self.datasets:
            for fmt in self.formats:
                for model in self.models:
                    for pipeline in self.pipelines:
                        cells.append(
                            RunConfig(
                                model=model,
                                dataset=dataset,
                                fmt=fmt,
                                pipeline=pipeline,
                            )
                        )
        return tuple(cells)
