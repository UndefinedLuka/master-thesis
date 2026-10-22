"""Chapter 5 CLI validation and dispatch across the two question cohorts."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

import pandas as pd

from config.paths import ProjectPaths
from visualizations import generate_all as orchestration
from visualizations.cohorts import DIRECT_FORMATS, RETAINED_MODELS


def _write_records(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


@pytest.fixture
def campaign(tmp_path: Path) -> tuple[list[str], list[dict], Path]:
    """Two datasets, each with answerable, empty-reference, and context-rejected questions."""
    references_dir = tmp_path / "references"
    references_dir.mkdir()
    records = []
    for dataset in ("spider", "bird"):
        references = []
        for question in ("answerable", "empty", "overflow"):
            example_id = f"{dataset}:{question}"
            rows = [] if question == "empty" else [{"value": 0, "nullable": None}]
            references.append(
                {
                    "dataset": dataset,
                    "example_id": example_id,
                    "db_id": "db",
                    "rows": rows,
                    "row_count": len(rows),
                    "error": None,
                }
            )
            for model in RETAINED_MODELS:
                for fmt in (*DIRECT_FORMATS, None):
                    sql = fmt is None
                    overflow = (
                        question == "overflow" and model == RETAINED_MODELS[0] and fmt == "xml"
                    )
                    # Failed SQL still has recorded inference; blank direct output also fits.
                    records.append(
                        {
                            "dataset": dataset,
                            "example_id": example_id,
                            "db_id": "db",
                            "model_id": model,
                            "pipeline": "text_to_sql" if sql else "direct",
                            "format": fmt,
                            "predicted_sql": "SELECT missing" if sql else None,
                            "raw_output": "SELECT missing" if sql else "",
                            "error": "context_exceeded" if overflow else None,
                            "execution_success": False if sql else None,
                            "definitive_failure": True if sql else None,
                            "retry_count": 0,
                            "input_tokens": 0 if overflow else 100,
                            "output_tokens": 0 if overflow else 10,
                            "time_to_first_token_s": 0 if overflow else 1.0,
                            "generation_time_s": 0 if overflow else 3.0,
                            "query_execution_time_s": 0.1 if sql else 0.0,
                            "total_time_s": 0 if overflow else (3.1 if sql else 3.0),
                            "scores": {
                                "normalized_exact_match": 0.0,
                                "bertscore_f": 0.0,
                                "rouge_l_f": 0.0,
                            },
                            "attempts": [
                                {
                                    "attempt_index": 0,
                                    "error": "no such column: missing",
                                    "llm_metrics": {
                                        "error": None,
                                        "input_tokens": 100,
                                        "output_tokens": 10,
                                        "time_to_first_token_s": 1.0,
                                        "total_generation_time_s": 3.0,
                                    },
                                }
                            ]
                            if sql
                            else [],
                        }
                    )
        _write_records(references_dir / f"{dataset}_references.ndjson", references)
    results_path = tmp_path / "campaign.scored.ndjson"
    _write_records(results_path, records)
    report_path = tmp_path / "tokens.csv"
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "database_id": "db",
                "format": fmt,
                "tokenizer_family": family,
                "tokenizer_label": label,
                "tokenizer_model": f"{family}.gguf",
                "table_count": 1,
                "row_count": 1,
                "tokens": 100,
            }
            for dataset in ("spider", "bird")
            for fmt in DIRECT_FORMATS
            for family, label in (
                ("qwen-3.5", "Qwen 3.5"),
                ("ministral-3", "Ministral 3"),
                ("gemma-4", "Gemma 4"),
            )
        ]
    ).to_csv(report_path, index=False)
    args = [
        "--results",
        str(results_path),
        "--references-dir",
        str(references_dir),
        "--serialization-report",
        str(report_path),
        "--tables-dir",
        str(tmp_path / "tables"),
        "--figures-dir",
        str(tmp_path / "figures"),
        "--audit-dir",
        str(tmp_path / "audit"),
        "--allow-incomplete",
    ]
    return args, records, results_path


@pytest.mark.parametrize("defect", ["missing_model", "missing_configuration", "unknown_model"])
def test_cli_allow_incomplete_still_rejects_invalid_cohort_matrix(
    campaign: tuple[list[str], list[dict], Path],
    defect: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args, records, results_path = campaign
    if defect == "missing_model":
        records = [row for row in records if row["model_id"] != RETAINED_MODELS[-1]]
    elif defect == "missing_configuration":
        records.pop()
    else:
        records[0]["model_id"] = "unknown-model.gguf"
    _write_records(results_path, records)
    generate = Mock(return_value=[])
    monkeypatch.setattr(orchestration, "generate_all", generate)

    with pytest.raises(SystemExit) as exc:
        orchestration.main(args)

    assert exc.value.code == 2
    assert any(
        word in capsys.readouterr().err.lower()
        for word in (
            "incomplete",
            "missing",
            "unexpected model",
        )
    )
    generate.assert_not_called()


def test_cli_dispatches_answerable_and_context_fit_populations(
    campaign: tuple[list[str], list[dict], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _, _ = campaign
    calculators = {}
    for name in (
        "calculate_direct_efficiency",
        "calculate_text_to_sql_analysis",
        "calculate_text_to_sql_retry_analysis",
        "calculate_common_cohort_analysis",
        "calculate_pipeline_retrieval_comparison",
    ):
        calculators[name] = Mock(return_value=object())
        monkeypatch.setattr(orchestration, name, calculators[name])
    generators = {}
    for name in (
        "generate_experiment_outcomes_table",
        "generate_serialization_efficiency_table",
        "generate_tokenizer_comparison_figure",
        "generate_direct_accuracy_by_format_figure",
        "generate_direct_accuracy_by_model_size_figure",
        "generate_direct_operational_table",
        "generate_direct_token_usage_figure",
        "generate_direct_latency_figure",
        "generate_text_to_sql_retrieval_table",
        "generate_text_to_sql_accuracy_figure",
        "generate_text_to_sql_operational_table",
        "generate_text_to_sql_token_figure",
        "generate_text_to_sql_latency_figure",
        "generate_error_correction_table",
        "generate_error_correction_figure",
        "generate_pipeline_retrieval_comparison_figure",
        "generate_common_cohort_table",
        "generate_common_cohort_latency_figure",
        "generate_common_cohort_token_figure",
        "generate_database_size_accuracy_figures",
    ):
        output = tmp_path / name
        generators[name] = Mock(return_value=[output] if name.endswith("figures") else output)
        monkeypatch.setattr(orchestration, name, generators[name])
    evidence = Mock()
    monkeypatch.setattr(orchestration, "save_chapter5_evidence", evidence)

    assert orchestration.main(args) == 0

    answerable = calculators["calculate_text_to_sql_analysis"].call_args.args[0]
    context_fit = calculators["calculate_direct_efficiency"].call_args.args[0]
    assert set(answerable["example_id"]) == {
        f"{dataset}:{question}"
        for dataset in ("spider", "bird")
        for question in ("answerable", "overflow")
    }
    assert set(context_fit["example_id"]) == {
        f"{dataset}:{question}"
        for dataset in ("spider", "bird")
        for question in ("answerable", "empty")
    }
    assert len(answerable) == len(context_fit) == 280
    assert len(answerable.loc[answerable["error"].eq("context_exceeded")]) == 2
    sql = context_fit.loc[context_fit["pipeline"].eq("text_to_sql")]
    assert len(sql) == 40
    assert not sql["execution_success"].any()
    assert sql["input_tokens"].sum() == 4_000
    assert not context_fit.loc[context_fit["pipeline"].eq("direct"), "has_answer"].any()
    assert calculators["calculate_common_cohort_analysis"].call_args.args[0] is context_fit
    assert calculators["calculate_pipeline_retrieval_comparison"].call_args.args[0] is answerable
    for name in (
        "generate_direct_accuracy_by_format_figure",
        "generate_direct_accuracy_by_model_size_figure",
        "generate_database_size_accuracy_figures",
    ):
        assert generators[name].call_args.args[0] is answerable

    for calculator, outputs in (
        (
            "calculate_direct_efficiency",
            (
                "generate_direct_operational_table",
                "generate_direct_token_usage_figure",
                "generate_direct_latency_figure",
            ),
        ),
        (
            "calculate_text_to_sql_analysis",
            (
                "generate_text_to_sql_retrieval_table",
                "generate_text_to_sql_accuracy_figure",
                "generate_text_to_sql_operational_table",
                "generate_text_to_sql_token_figure",
                "generate_text_to_sql_latency_figure",
            ),
        ),
        (
            "calculate_text_to_sql_retry_analysis",
            (
                "generate_error_correction_table",
                "generate_error_correction_figure",
            ),
        ),
        (
            "calculate_common_cohort_analysis",
            (
                "generate_common_cohort_table",
                "generate_common_cohort_latency_figure",
                "generate_common_cohort_token_figure",
            ),
        ),
        (
            "calculate_pipeline_retrieval_comparison",
            ("generate_pipeline_retrieval_comparison_figure",),
        ),
    ):
        for name in outputs:
            assert generators[name].call_args.args[0] is calculators[calculator].return_value
    assert calculators["calculate_text_to_sql_retry_analysis"].call_args.args[0] is (
        calculators["calculate_text_to_sql_analysis"].return_value
    )
    # Coverage and serialization retain their independent, unfiltered source populations.
    assert len(generators["generate_experiment_outcomes_table"].call_args.args[0]) == 420
    assert (
        generators["generate_experiment_outcomes_table"].call_args.kwargs["coverage_validated"]
        is False
    )
    report = generators["generate_serialization_efficiency_table"].call_args.args[0]
    assert len(report) == 36
    assert generators["generate_tokenizer_comparison_figure"].call_args.args[0] is report
    assert evidence.call_args.args[0] is answerable
    assert evidence.call_args.args[-1] == tmp_path / "audit"

    manifest = pd.read_csv(tmp_path / "audit" / "analysis_cohort_membership.csv")
    assert len(manifest) == 6
    assert manifest["answerable"].sum() == manifest["context_fit"].sum() == 4
    assert (manifest["answerable"] & manifest["context_fit"]).sum() == 2
    audit = json.loads((tmp_path / "audit" / "analysis_cohort_audit.json").read_text())
    assert all(source["sha256"] for source in audit["reference_sources"])
    assert (tmp_path / "tables" / "analysis_cohorts.tex").is_file()


def test_parser_defaults_use_stored_references_and_separate_audit_directory(tmp_path: Path) -> None:
    args = orchestration.build_parser(ProjectPaths.from_root(tmp_path)).parse_args([])

    assert args.references_dir == tmp_path / "data"
    assert args.audit_dir == tmp_path / "analysis" / "chapter5"


def test_cli_cannot_override_confirmed_model_roster() -> None:
    with pytest.raises(SystemExit) as exc:
        orchestration.build_parser().parse_args(["--exclude-model", "Qwen3.5-9B"])
    assert exc.value.code == 2
