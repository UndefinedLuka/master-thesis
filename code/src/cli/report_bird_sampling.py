"""Compare BIRD row caps using reference-query emptiness and payload token counts.

Run ``python -m cli.report_bird_sampling`` from ``code/``. The report uses the
benchmark sampler, serializers, and three GGUF tokenizers, without model inference.
Temporary databases are removed after measurement. Detailed CSVs are written under
``results/bird_sampling_sensitivity`` and a summary under ``analysis/summaries/``.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from cli.report_serialization_tokens import (
    TOKENIZER_FAMILIES,
    DatabaseInput,
    LlamaCppTokenCounter,
    measure_databases,
    resolve_tokenizer_models,
)
from config.config import BIRD_SAMPLE_ROWS, N_CTX, SAMPLE_SEED
from config.paths import ProjectPaths
from config.settings import default_db_root, default_dev_json, sampled_db_root
from datasets.bird import load_bird_examples
from datasets.loader import load_db_contents
from reference.answers import generate_reference_answer, load_reference_file
from serialization.types import FORMATS
from sqlite.materialize import materialize_sampled_db

ROW_CAPS = (100, 500, 1000)


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    paths = ProjectPaths.discover()
    examples = load_bird_examples(default_dev_json("bird", paths.data_dir))
    db_ids = sorted({example["db_id"] for example in examples})
    references = load_reference_file(paths.data_dir / "bird_references.ndjson")
    reference_by_id = {reference["example_id"]: reference for reference in references}
    if len(reference_by_id) != len(references) or set(reference_by_id) != {
        example["example_id"] for example in examples
    }:
        raise ValueError("Reference IDs do not match the BIRD development split")

    baseline_csv = paths.results_dir / "input_token_overhead/serialization_tokens_by_database.csv"
    with baseline_csv.open(encoding="utf-8", newline="") as handle:
        baseline_tokens = {
            (row["database_id"], row["format"], row["tokenizer_family"]): int(row["tokens"])
            for row in csv.DictReader(handle)
            if row["dataset"] == "bird"
        }

    output_dir = paths.results_dir / "bird_sampling_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    selections = resolve_tokenizer_models(paths.models_dir, TOKENIZER_FAMILIES)
    report = {
        "configuration": {
            "release": "dev_20240627",
            "executed_at_utc": datetime.now(UTC).isoformat(),
            "row_caps": list(ROW_CAPS),
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
        temporary_root = Path(
            stack.enter_context(TemporaryDirectory(prefix="bird-sensitivity-", dir=paths.data_dir))
        )
        for cap in ROW_CAPS:
            databases = []
            query_records = []
            baseline_rows_verified = 0
            baseline_queries_verified = 0
            baseline_row_count_changes = []
            for db_id in db_ids:
                print(f"Sampling and executing gold SQL: cap={cap}, database={db_id}", flush=True)
                src = default_db_root("bird", paths.data_dir) / db_id / f"{db_id}.sqlite"
                dest = temporary_root / str(cap) / db_id / f"{db_id}.sqlite"
                materialize_sampled_db(src, dest, cap, seed=SAMPLE_SEED)
                databases.append(DatabaseInput(database_id=db_id, path=dest))
                if cap == BIRD_SAMPLE_ROWS:
                    benchmark = sampled_db_root(paths.data_dir) / db_id / f"{db_id}.sqlite"
                    if load_db_contents(dest) != load_db_contents(benchmark):
                        raise ValueError(
                            f"Rebuilt {cap}-row sample differs from benchmark: {db_id}"
                        )
                    baseline_rows_verified += 1

                for example in examples:
                    if example["db_id"] != db_id:
                        continue
                    reference = generate_reference_answer(example, dest)
                    query_records.append(
                        {
                            "sample_rows_per_table": cap,
                            "example_id": example["example_id"],
                            "database_id": db_id,
                            "difficulty": example["difficulty"],
                            "row_count": reference["row_count"],
                            "error": reference["error"],
                        }
                    )
                    if cap == BIRD_SAMPLE_ROWS:
                        baseline = reference_by_id[example["example_id"]]
                        if (reference["error"], reference["row_count"] == 0) != (
                            baseline["error"],
                            baseline["row_count"] == 0,
                        ):
                            raise ValueError(
                                f"Reference empty/error status changed: {example['example_id']}"
                            )
                        if reference["row_count"] != baseline["row_count"]:
                            # Time-dependent gold SQL (e.g. JULIANDAY('now')) may
                            # change a non-empty row count without changing emptiness.
                            baseline_row_count_changes.append(
                                {
                                    "example_id": example["example_id"],
                                    "gold_sql": example["gold_sql"],
                                    "stored_row_count": baseline["row_count"],
                                    "current_row_count": reference["row_count"],
                                }
                            )
                        baseline_queries_verified += 1

            write_csv(output_dir / f"gold_queries_{cap}.csv", query_records)
            tokens = measure_databases("bird", databases, FORMATS, counters)
            write_csv(
                output_dir / f"serialization_tokens_{cap}.csv", [asdict(row) for row in tokens]
            )
            expected_measurements = len(db_ids) * len(FORMATS) * len(counters)
            if len(tokens) != expected_measurements:
                raise ValueError("Incomplete token measurements")
            if cap == BIRD_SAMPLE_ROWS:
                observed = {
                    (row.database_id, row.format, row.tokenizer_family): row.tokens
                    for row in tokens
                }
                if observed != baseline_tokens:
                    raise ValueError("500-row token counts do not reproduce the benchmark report")

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
            report_path = paths.root / "analysis/summaries/bird_sampling_sensitivity.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(
                f"cap={cap}: {len(empty)}/{len(examples)} empty; {failures} SQL errors; "
                f"{summary['payloads_exceeding_context']}/{len(tokens)} payloads exceed context",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
