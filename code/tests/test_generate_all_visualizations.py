"""Tests for the all-analysis-artifacts orchestration command."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

import pandas as pd

import visualizations.generate_all as generate_module
from visualizations.generate_all import (
    ALL_COLUMNS,
    GenerationContext,
    GenerationStage,
    generate_all,
)


def _context(tmp_path: Path) -> GenerationContext:
    return GenerationContext(
        records=pd.DataFrame(),
        completed_answers=pd.DataFrame(),
        datasets=("bird", "spider"),
        figures_dir=tmp_path / "figures",
        tables_dir=tmp_path / "tables",
    )


def test_all_snapshot_columns_are_unique() -> None:
    assert len(ALL_COLUMNS) == len(set(ALL_COLUMNS))
    assert {
        "dataset",
        "pipeline",
        "format",
        "model_id",
        "example_id",
        "input_tokens",
        "total_time_s",
        "execution.result_match",
        "scores.normalized_exact_match",
        "scores.rouge_l_f",
        "scores.bertscore_f",
    }.issubset(ALL_COLUMNS)


def test_generate_all_runs_stages_in_order_and_collects_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def stage(name: str, filename: str):
        def run(context: GenerationContext) -> list[Path]:
            calls.append(name)
            return [context.figures_dir / filename]

        return run

    monkeypatch.setattr(
        generate_module,
        "GENERATION_STAGES",
        (
            GenerationStage("first", stage("first", "one.png")),
            GenerationStage("second", stage("second", "two.png")),
        ),
    )

    outputs = generate_all(_context(tmp_path))

    assert calls == ["first", "second"]
    assert outputs == [tmp_path / "figures" / "one.png", tmp_path / "figures" / "two.png"]


def test_generate_all_names_the_failing_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_context: GenerationContext) -> list[Path]:
        raise ValueError("bad input")

    monkeypatch.setattr(
        generate_module,
        "GENERATION_STAGES",
        (GenerationStage("accuracy heatmaps", fail),),
    )

    with pytest.raises(RuntimeError, match="accuracy heatmaps") as error:
        generate_all(_context(tmp_path))
    assert isinstance(error.value.__cause__, ValueError)


def test_main_loads_one_snapshot_and_passes_shared_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    captured_contexts: list[GenerationContext] = []

    def fake_load(target: Path, **kwargs) -> pd.DataFrame:
        calls.append({"target": target, **kwargs})
        row = {column: None for column in ALL_COLUMNS}
        row.update(
            {
                "dataset": "bird",
                "pipeline": "direct",
                "format": "csv",
                "model_id": "model.gguf",
                "example_id": "bird:db:1",
                "error": None,
                "has_answer": True,
            }
        )
        return pd.DataFrame([row], columns=ALL_COLUMNS)

    def fake_generate(context: GenerationContext) -> list[Path]:
        captured_contexts.append(context)
        return [context.figures_dir / "one.png", context.tables_dir / "one.tex"]

    monkeypatch.setattr(generate_module, "load_scored_results", fake_load)
    monkeypatch.setattr(generate_module, "generate_all", fake_generate)

    result = generate_module.main(
        [
            "--results",
            str(tmp_path / "results"),
            "--figures-dir",
            str(tmp_path / "figures"),
            "--tables-dir",
            str(tmp_path / "tables"),
            "--dataset",
            "bird",
            "--model",
            "model.gguf",
        ]
    )

    assert result == 0
    assert len(calls) == 1
    assert calls[0]["target"] == tmp_path / "results"
    assert calls[0]["columns"] == ALL_COLUMNS
    assert calls[0]["datasets"] == ("bird",)
    assert calls[0]["models"] == ["model.gguf"]
    assert len(captured_contexts) == 1
    context = captured_contexts[0]
    assert context.datasets == ("bird",)
    assert len(context.records) == 1
    assert len(context.completed_answers) == 1
