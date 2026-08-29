"""End-to-end tests for the score_results CLI (file mode and directory mode).

Uses --no-bertscore so the heavy BERTScore model is never loaded. Normalized Exact Match and
ROUGE-L still require the optional metrics extra, so the module skips without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rouge_score", reason="requires the [metrics] extra")
pytest.importorskip("transformers", reason="requires the [metrics] extra")

from cli import score_results  # noqa: E402
from datasets.types import DatasetExample  # noqa: E402
from pipelines.shared import base_result  # noqa: E402


def _ex(idx: int) -> DatasetExample:
    return DatasetExample(
        dataset="spider",
        db_id="db",
        question="q",
        gold_sql="SELECT name FROM t",
        evidence=None,
        difficulty=None,
        example_id=f"spider:db:{idx}",
    )


def _ref(rows: list[dict] | None, gold_sql: str = "SELECT name FROM t") -> dict:
    return {"example_id": "spider:db:0", "rows": rows, "gold_sql": gold_sql}


def _write_ndjson(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _scored_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_cli_single_file(tmp_path: Path) -> None:
    direct = base_result(_ex(0), pipeline="direct", fmt="json", model_id="m")
    direct["raw_output"] = "Alice"
    in_path = tmp_path / "model.ndjson"
    _write_ndjson(in_path, [direct])

    refs = tmp_path / "refs.ndjson"
    _write_ndjson(refs, [_ref([{"name": "Alice"}])])

    rc = score_results.main(
        ["--results", str(in_path), "--references", str(refs), "--no-bertscore"]
    )
    assert rc == 0

    out = in_path.with_suffix("").with_suffix(".scored.ndjson")
    assert out.exists()
    rows = _scored_lines(out)
    assert rows[0]["scores"]["normalized_exact_match"] == 1.0
    assert rows[0]["scores"]["rouge_l_f"] == pytest.approx(1.0)  # "Alice" vs "Alice"
    assert rows[0]["scores"]["bertscore_f"] is None  # disabled -> null, not a fake 0.0 score


def test_cli_directory_mode_with_execution(tmp_path: Path) -> None:
    results = tmp_path / "results"
    # direct slice
    direct = base_result(_ex(0), pipeline="direct", fmt="json", model_id="m")
    direct["raw_output"] = "Alice"
    _write_ndjson(results / "spider" / "direct" / "json" / "m.ndjson", [direct])
    # text_to_sql slice
    sql = base_result(_ex(0), pipeline="text_to_sql", fmt=None, model_id="m")
    sql["predicted_rows"] = [{"name": "Alice"}]
    sql["execution_success"] = True
    _write_ndjson(results / "spider" / "text_to_sql" / "schema" / "m.ndjson", [sql])

    # Use an explicit --references file to keep the test hermetic (vs auto-resolving from
    # data/<dataset>_references.ndjson under the discovered project root).
    refs = tmp_path / "refs.ndjson"
    _write_ndjson(refs, [_ref([{"name": "Alice"}])])

    rc = score_results.main(
        ["--results", str(results), "--references", str(refs), "--no-bertscore"]
    )
    assert rc == 0

    direct_out = results / "spider" / "direct" / "json" / "m.scored.ndjson"
    sql_out = results / "spider" / "text_to_sql" / "schema" / "m.scored.ndjson"
    assert direct_out.exists() and sql_out.exists()

    direct_rows = _scored_lines(direct_out)
    assert direct_rows[0]["scores"]["normalized_exact_match"] == 1.0
    assert direct_rows[0]["scores"]["rouge_l_f"] == pytest.approx(1.0)
    assert direct_rows[0]["execution"] is None

    sql_rows = _scored_lines(sql_out)
    assert sql_rows[0]["scores"]["normalized_exact_match"] == 1.0
    assert sql_rows[0]["execution"]["execution_accuracy"] == 1.0


def test_cli_skips_already_scored_files(tmp_path: Path) -> None:
    results = tmp_path / "results"
    direct = base_result(_ex(0), pipeline="direct", fmt="json", model_id="m")
    direct["raw_output"] = "Alice"
    slice_dir = results / "spider" / "direct" / "json"
    _write_ndjson(slice_dir / "m.ndjson", [direct])

    refs = tmp_path / "refs.ndjson"
    _write_ndjson(refs, [_ref([{"name": "Alice"}], gold_sql="x")])

    score_results.main(["--results", str(results), "--references", str(refs), "--no-bertscore"])
    out = slice_dir / "m.scored.ndjson"
    first = out.read_text()

    # Re-running should skip (out exists) unless --overwrite — discovery excludes scored files.
    inputs = score_results._discover_inputs(results, overwrite=False)
    assert out not in inputs
    assert (slice_dir / "m.ndjson") not in inputs  # already has a scored sibling
    assert out.read_text() == first
