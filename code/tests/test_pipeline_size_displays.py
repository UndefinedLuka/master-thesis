"""Publication displays pool both datasets and retain every method and size tier."""

from __future__ import annotations

import numpy as np
import pandas as pd

from visualizations.config import (
    COMMON_COHORT_METHOD_ORDER,
    DATASET_ORDER,
    DIRECT_ACCURACY_METRIC_ORDER,
    SERIALIZATION_FORMAT_ORDER,
    TOKENIZER_FAMILY_ORDER,
)
from visualizations.pipeline_retrieval_by_database_size import (
    SIZE_TIER_ORDER,
    calculate_pipeline_size_retrieval_comparison,
)
from visualizations.pipeline_retrieval_by_database_size_figure import (
    generate_pipeline_size_retrieval_figure,
    plot_pipeline_size_retrieval_comparison,
)
from visualizations.pipeline_size_cohort_table import generate_pipeline_size_cohort_table


def _analysis():
    tokens, runs = [], []
    for dataset in DATASET_ORDER:
        for tier, size in zip(SIZE_TIER_ORDER, (1000, 5000, 20000), strict=True):
            if dataset == "bird" and tier != "large":
                continue
            for database in range(2):
                db_id = f"{tier}_{database}"
                for fmt in SERIALIZATION_FORMAT_ORDER:
                    for family in TOKENIZER_FAMILY_ORDER:
                        tokens.append({
                            "dataset": dataset, "database_id": db_id, "format": fmt,
                            "tokenizer_family": family, "tokens": size,
                        })
                for method_index, method in enumerate(COMMON_COHORT_METHOD_ORDER):
                    sql = method == "text_to_sql"
                    runs.append({
                        "dataset": dataset, "db_id": db_id, "example_id": db_id,
                        "model_id": "Qwen3.5-2B", "pipeline": "text_to_sql" if sql else "direct",
                        "format": None if sql else method, "error": None,
                        "execution_success": True if sql else None,
                        "scores.normalized_exact_match": int(sql),
                        "scores.bertscore_f": 0.9 - 0.05 * method_index,
                        "scores.rouge_l_f": 0.8 - 0.05 * method_index,
                    })
    return calculate_pipeline_size_retrieval_comparison(pd.DataFrame(runs), pd.DataFrame(tokens))


def test_figure_covers_every_metric_method_and_tier_without_dataset_panels(tmp_path) -> None:
    import matplotlib.pyplot as plt

    analysis = _analysis()
    figure = plot_pipeline_size_retrieval_comparison(analysis)
    assert len(figure.axes) == 3
    for index, axis in enumerate(figure.axes):
        metric = DIRECT_ACCURACY_METRIC_ORDER[index]
        panel = analysis.summary.loc[analysis.summary["metric"].eq(metric)]
        assert len(axis.get_xticklabels()) == 3
        points = [line for line in axis.lines if line.get_marker() in ("o", "D")]
        expected = []
        for tier in SIZE_TIER_ORDER:
            tier_rows = panel.loc[panel["size_tier"].eq(tier)].set_index("method")
            if not tier_rows.empty:
                expected.extend(tier_rows.loc[list(COMMON_COHORT_METHOD_ORDER), "mean_score"])
        np.testing.assert_allclose([line.get_ydata()[0] for line in points], expected)
        assert len(axis.collections) == len(expected)
        assert not any(text.get_text() == "No databases" for text in axis.texts)
        assert "Spider" not in axis.get_ylabel()
        assert "BIRD" not in axis.get_ylabel()
    assert [text.get_text() for text in figure.legends[0].get_texts()] == [
        "Text-to-SQL", "CSV", "TOON", "Markdown", "YAML", "JSON", "XML",
    ]
    plt.close(figure)
    path = generate_pipeline_size_retrieval_figure(analysis, tmp_path)
    assert path.name == "pipeline_retrieval_by_database_size.pdf"
    assert path.read_bytes().startswith(b"%PDF")


def test_coverage_table_pools_counts_and_keeps_explicit_thresholds(tmp_path) -> None:
    path = generate_pipeline_size_cohort_table(_analysis(), tmp_path)
    table = path.read_text()
    assert path.name == "pipeline_size_cohorts.tex"
    assert r"\label{tab:pipeline-size-cohorts}" in table
    assert table.count(r"\num{1500}") == 2
    assert table.count(r"\num{10000}") == 2
    assert "Dataset" not in table and "Spider" not in table and "BIRD" not in table
    assert table.count(r"\num{2} & \num{2}") == 2
    assert table.count(r"\num{4} & \num{4}") == 1


def test_empty_pooled_tiers_are_explicit_without_fabricated_scores() -> None:
    import matplotlib.pyplot as plt

    analysis = _analysis()
    analysis.summary.drop(
        analysis.summary.index[analysis.summary["size_tier"].eq("medium")], inplace=True,
    )
    figure = plot_pipeline_size_retrieval_comparison(analysis)
    for axis in figure.axes:
        assert sum(text.get_text() == "No databases" for text in axis.texts) == 1
    plt.close(figure)
