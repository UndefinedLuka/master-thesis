"""Config matrix expansion and prompt-template validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from thesis_bench.config import (
    BenchmarkConfig,
    DatasetConfig,
    GenerationOptions,
    ModelConfig,
)
from thesis_bench.enums import Backend, DatasetKind, PipelineKind, SerializationFormat
from thesis_bench.errors import ThesisBenchError
from thesis_bench.prompts.loader import PromptTemplate, build_prompt


def test_matrix_expands_to_cartesian_product() -> None:
    config = BenchmarkConfig(
        models=(
            ModelConfig(Backend.OLLAMA, "a", 7.0, "Q8_0"),
            ModelConfig(Backend.OLLAMA, "b", 14.0, "Q8_0"),
        ),
        datasets=(
            DatasetConfig(DatasetKind.SPIDER, Path("q"), Path("db")),
        ),
        formats=(SerializationFormat.JSON, SerializationFormat.CSV),
        pipelines=(PipelineKind.DIRECT, PipelineKind.TEXT_TO_SQL),
        results_dir=Path("out"),
    )
    cells = config.expand()
    assert len(cells) == 2 * 1 * 2 * 2  # models x datasets x formats x pipelines
    # cell_id is unique per cell.
    assert len({c.cell_id for c in cells}) == len(cells)


def test_generation_options_defaults_are_deterministic() -> None:
    opts = GenerationOptions()
    assert opts.temperature == 0.0
    assert isinstance(opts.stop, tuple)


def test_format_family_property() -> None:
    from thesis_bench.enums import FormatFamily

    assert SerializationFormat.JSON.family is FormatFamily.STRUCTURAL
    assert SerializationFormat.CSV.family is FormatFamily.POSITIONAL
    assert SerializationFormat.TOON.family is FormatFamily.TOKEN_OPTIMIZED


def test_prompt_requires_exact_placeholders() -> None:
    assert "schema text" in build_prompt(PromptTemplate.SQL_ANSWER, schema="schema text")
    with pytest.raises(ThesisBenchError):
        build_prompt(PromptTemplate.SQL_ANSWER, data="wrong key")
    with pytest.raises(ThesisBenchError):
        build_prompt(PromptTemplate.SQL_RETRY, schema="s")  # missing keys
