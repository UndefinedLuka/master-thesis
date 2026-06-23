"""Default experiment definition.

This is the concrete, editable description of the experiment matrix — the models,
datasets, formats, and pipelines to run. It is kept separate from ``config.py``
(which defines the *types*) so the schema and a particular experiment don't get
tangled. Edit ``default_benchmark()`` to change what runs, or build a
``BenchmarkConfig`` programmatically.

The model list mirrors ch-03's design: open-weight models at several parameter
scales, all at the same quantization level (``Q8_0``) so quantization is held
constant as a variable.
"""

from __future__ import annotations

from .config import (
    BenchmarkConfig,
    DatasetConfig,
    GenerationOptions,
    ModelConfig,
)
from .enums import Backend, DatasetKind, PipelineKind, SerializationFormat
from .paths import default_data_dir, default_results_dir

# Greedy, fixed-seed decoding for reproducibility (see ch-03 reproducibility goal).
_DETERMINISTIC = GenerationOptions(temperature=0.0, seed=0, num_ctx=128_000)


def _model(tag: str, scale_b: float) -> ModelConfig:
    return ModelConfig(
        backend=Backend.OLLAMA,
        model_id=tag,
        param_scale_b=scale_b,
        quantization="Q8_0",
        options=_DETERMINISTIC,
    )


def default_benchmark() -> BenchmarkConfig:
    """The default full matrix. Trim models/datasets/formats for quicker runs."""
    models = (
        _model("qwen2.5:7b-instruct-q8_0", 7.0),
        _model("qwen2.5:14b-instruct-q8_0", 14.0),
        _model("qwen2.5:32b-instruct-q8_0", 32.0),
    )

    datasets = (
        DatasetConfig(
            kind=DatasetKind.SPIDER,
            questions_path=default_data_dir / "spider" / "dev.json",
            db_root=default_data_dir / "spider" / "database",
            sample_rows=None,  # full tables
        ),
        DatasetConfig(
            kind=DatasetKind.BIRD,
            questions_path=default_data_dir / "bird" / "dev.json",
            db_root=default_data_dir / "bird" / "dev_databases",
            sample_rows=20,  # sub-sample large tables
            sample_seed=42,
        ),
    )

    formats = tuple(SerializationFormat)  # all six
    pipelines = (PipelineKind.DIRECT, PipelineKind.TEXT_TO_SQL)

    return BenchmarkConfig(
        models=models,
        datasets=datasets,
        formats=formats,
        pipelines=pipelines,
        results_dir=default_results_dir,
        bertscore_enabled=False,
    )
