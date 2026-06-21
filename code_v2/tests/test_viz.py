"""Visualization config loading and figure rendering smoke tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark.viz import add_approach_column, render_all
from benchmark.viz_config import load_viz_config


def test_viz_config_loads_from_repo() -> None:
    cfg = load_viz_config()
    assert cfg.theme.formats  # at least one output format
    # Every study format plus sql should have a fixed color.
    for key in ("json", "xml", "yaml", "csv", "markdown", "toon", "sql"):
        assert key in cfg.palette.by_format
    assert cfg.primary_metric in {"bertscore", "rouge_l"}


def test_palette_hues_length() -> None:
    cfg = load_viz_config()
    assert len(cfg.palette.hues(5)) == 5
    assert len(cfg.palette.hues(12)) == 12  # cycles past the sequence length


def test_add_approach_column() -> None:
    df = pd.DataFrame(
        {
            "pipeline": ["direct", "sql"],
            "format": ["json", None],
        }
    )
    out = add_approach_column(df)
    assert out["approach"].tolist() == ["json", "sql"]


def _mini_frame() -> pd.DataFrame:
    rows = []
    for model in ("m-small", "m-big"):
        for approach, pipeline, fmt in [
            ("json", "direct", "json"),
            ("sql", "sql", None),
        ]:
            rows.append(
                {
                    "model": model,
                    "dataset": "spider",
                    "pipeline": pipeline,
                    "format": fmt,
                    "input_tokens": 1000,
                    "output_tokens": 40,
                    "ttft_s": 0.2,
                    "total_generation_s": 1.0,
                    "rouge_l": 0.7,
                    "bertscore": 0.8,
                    "retry_count": 1,
                    "execution_accuracy": True,
                    "result_set_match": True,
                    "definitive_failure": False,
                    "error": None,
                }
            )
    return pd.DataFrame(rows)


def test_render_all_writes_files(tmp_path: Path) -> None:
    cfg = load_viz_config()
    drawn = render_all(_mini_frame(), cfg, tmp_path)
    assert drawn  # something was rendered
    produced = list(tmp_path.glob("*.png")) + list(tmp_path.glob("*.pdf"))
    assert produced, "no figure files written"
