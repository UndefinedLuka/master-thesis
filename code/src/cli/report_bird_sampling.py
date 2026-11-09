"""Compare BIRD row caps using reference-query emptiness and payload token counts.

Run ``python -m cli.report_bird_sampling`` from ``code/``. The report uses the
benchmark sampler, serializers, and three GGUF tokenizers, without model inference.
Sampled databases, query records, and token measurements stay in memory. Only the
completed summary is written under ``analysis/summaries/``.
"""

from __future__ import annotations

import json
from collections import Counter
from contextlib import ExitStack, closing
from datetime import UTC, datetime

from config.config import BIRD_SAMPLE_ROWS, N_CTX, SAMPLE_SEED
from config.paths import ProjectPaths
from config.settings import default_db_root, default_dev_json, sampled_db_root
from datasets.bird import load_bird_examples
from datasets.loader import load_db_contents
from reference.answers import load_reference_file
from serialization.tokens import (
    TOKENIZER_FAMILIES,
    LlamaCppTokenCounter,
    discover_databases,
    measure_contents,
    resolve_tokenizer_models,
)
from serialization.types import FORMATS
from sqlite.connection import query_connection_safe
from sqlite.sampling import build_sampled_connection, load_connection_contents

ROW_CAPS = (100, 500, 1000)


def calculate_bird_sampling(
    paths: ProjectPaths | None = None,
    *,
    row_caps: tuple[int, ...] = ROW_CAPS,
) -> dict:
    """Compare row caps entirely in memory and return the completed sensitivity summary."""
    paths = paths or ProjectPaths.discover()
    if BIRD_SAMPLE_ROWS not in row_caps or any(cap <= 0 for cap in row_caps):
        raise ValueError("Row caps must be positive and include the benchmark sample size")
    examples = load_bird_examples(default_dev_json("bird", paths.data_dir))
    db_ids = sorted({example["db_id"] for example in examples})
    references = load_reference_file(paths.data_dir / "bird_references.ndjson")
    reference_by_id = {reference["example_id"]: reference for reference in references}
    if len(reference_by_id) != len(references) or set(reference_by_id) != {
        example["example_id"] for example in examples
    }:
        raise ValueError("Reference IDs do not match the BIRD development split")

    databases = discover_databases(default_db_root("bird", paths.data_dir), db_ids)
    benchmarks = {
        database.database_id: database.path
        for database in discover_databases(sampled_db_root(paths.data_dir), db_ids)
    }
    selections = resolve_tokenizer_models(paths.models_dir, TOKENIZER_FAMILIES)
    report = {
        "configuration": {
            "release": "dev_20240627",
            "executed_at_utc": datetime.now(UTC).isoformat(),
            "row_caps": list(row_caps),
            "sample_seed": SAMPLE_SEED,
            "context_window_tokens": N_CTX,
            "formats": list(FORMATS),
            "tokenizer_models": {
                selection.family.key: selection.model_path.name for selection in selections
            },
            "empty_definition": (
                "Successful gold SQL returning zero rows; NULL/0 rows are non-empty"
            ),
            "context_fit_definition": (
                "Serialized payload tokens <= context window; no prompt/output"
            ),
            "scope": "Single-seed post-hoc sensitivity analysis; no model inference",
        },
        "total_questions": len(examples),
        "total_databases": len(db_ids),
        "samples": [],
    }

    with ExitStack() as stack:
        counters = []
        for selection in selections:
            counter = LlamaCppTokenCounter.create(selection)
            stack.callback(counter.close)
            counters.append(counter)
        for cap in row_caps:
            tokens = []
            baseline_tokens = []
            query_records = []
            baseline_rows_verified = 0
            baseline_queries_verified = 0
            baseline_row_count_changes = []
            for database in databases:
                db_id = database.database_id
                print(f"Sampling and executing gold SQL: cap={cap}, database={db_id}", flush=True)
                with closing(
                    build_sampled_connection(database.path, cap, seed=SAMPLE_SEED)
                ) as conn:
                    conn.execute("PRAGMA temp_store = MEMORY")
                    contents = load_connection_contents(conn)
                    for example in examples:
                        if example["db_id"] != db_id:
                            continue
                        result, error = query_connection_safe(conn, example["gold_sql"])
                        row_count = len(result.rows) if result is not None else None
                        query_records.append(
                            {
                                "sample_rows_per_table": cap,
                                "example_id": example["example_id"],
                                "database_id": db_id,
                                "difficulty": example["difficulty"],
                                "row_count": row_count,
                                "error": error,
                            }
                        )
                        if cap == BIRD_SAMPLE_ROWS:
                            baseline = reference_by_id[example["example_id"]]
                            if (error, row_count == 0) != (
                                baseline["error"],
                                baseline["row_count"] == 0,
                            ):
                                raise ValueError(
                                    f"Reference empty/error status changed: {example['example_id']}"
                                )
                            if row_count != baseline["row_count"]:
                                # Time-dependent SQL may change counts without changing emptiness.
                                baseline_row_count_changes.append(
                                    {
                                        "example_id": example["example_id"],
                                        "gold_sql": example["gold_sql"],
                                        "stored_row_count": baseline["row_count"],
                                        "current_row_count": row_count,
                                    }
                                )
                            baseline_queries_verified += 1
                if cap == BIRD_SAMPLE_ROWS:
                    benchmark_contents = load_db_contents(benchmarks[db_id])
                    if contents != benchmark_contents:
                        raise ValueError(
                            f"Rebuilt {cap}-row sample differs from benchmark: {db_id}"
                        )
                    baseline_rows_verified += 1
                    baseline_tokens.extend(
                        measure_contents(
                            "bird",
                            db_id,
                            benchmark_contents,
                            FORMATS,
                            counters,
                        )
                    )
                tokens.extend(measure_contents("bird", db_id, contents, FORMATS, counters))
            expected_measurements = len(db_ids) * len(FORMATS) * len(counters)
            if len(tokens) != expected_measurements:
                raise ValueError("Incomplete token measurements")
            if cap == BIRD_SAMPLE_ROWS:
                if tokens != baseline_tokens:
                    raise ValueError(f"{cap}-row token counts do not reproduce benchmark payloads")

            empty = [row for row in query_records if row["error"] is None and row["row_count"] == 0]
            failures = sum(row["error"] is not None for row in query_records)
            summary = {
                "sample_rows_per_table": cap,
                "empty_results": len(empty),
                "empty_percentage": 100 * len(empty) / len(examples),
                "non_empty_results": len(examples) - len(empty) - failures,
                "gold_sql_failures": failures,
                "by_database": {
                    db_id: {
                        "total_questions": sum(
                            row["database_id"] == db_id for row in query_records
                        ),
                        "empty_results": sum(row["database_id"] == db_id for row in empty),
                        "sampled_rows": next(
                            row.row_count for row in tokens if row.database_id == db_id
                        ),
                    }
                    for db_id in db_ids
                },
                "empty_by_difficulty": dict(Counter(row["difficulty"] for row in empty)),
                "payload_measurements": len(tokens),
                "payloads_exceeding_context": sum(row.tokens > N_CTX for row in tokens),
                "by_format": {
                    fmt: {
                        "mean_tokens": sum(row.tokens for row in tokens if row.format == fmt)
                        / (len(db_ids) * len(counters)),
                        "fitting_databases_by_tokenizer": {
                            counter.family.key: [
                                row.database_id
                                for row in tokens
                                if row.format == fmt
                                and row.tokenizer_family == counter.family.key
                                and row.tokens <= N_CTX
                            ]
                            for counter in counters
                        },
                    }
                    for fmt in FORMATS
                },
            }
            if cap == BIRD_SAMPLE_ROWS:
                summary["baseline_verification"] = {
                    "databases_with_identical_contents": baseline_rows_verified,
                    "references_with_identical_error_and_empty_status": baseline_queries_verified,
                    "non_empty_row_count_changes": baseline_row_count_changes,
                    "identical_payload_token_counts": len(tokens),
                }
            report["samples"].append(summary)
            print(
                f"cap={cap}: {len(empty)}/{len(examples)} empty; {failures} SQL errors; "
                f"{summary['payloads_exceeding_context']}/{len(tokens)} payloads exceed context",
                flush=True,
            )
    return report


def main() -> int:
    paths = ProjectPaths.discover()
    report = calculate_bird_sampling(paths)
    report_path = paths.root / "analysis/summaries/bird_sampling_sensitivity.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
