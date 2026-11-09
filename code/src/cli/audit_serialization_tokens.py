"""Audit serialization token counts and format comparisons from in-memory measurements.

Run from code/: python -m cli.audit_serialization_tokens
Only the completed audit JSON is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median

from config.paths import ProjectPaths
from serialization.tokens import TokenRecord, measure_serialization_tokens
from visualizations.config import (
    CONTEXT_WINDOW_TOKENS,
    DATASET_ORDER,
    EXPECTED_DATABASE_COUNTS,
    SERIALIZATION_FORMAT_ORDER,
    TOKENIZER_FAMILY_ORDER,
)


def audit_serialization(records: Sequence[TokenRecord]) -> dict:
    """Use raw integer sums and paired minima, independent of display calculations."""
    rows = [asdict(record) for record in records]
    counts = {}
    models = defaultdict(set)
    databases = defaultdict(set)
    for row in rows:
        key = tuple(row[name] for name in ("dataset", "database_id", "format", "tokenizer_family"))
        if key in counts:
            raise ValueError(f"Duplicate serialization measurement: {key}")
        tokens = int(row["tokens"])
        if tokens <= 0:
            raise ValueError(f"Non-positive payload count: {key}")
        counts[key] = tokens
        models[row["tokenizer_family"]].add(row["tokenizer_model"])
        databases[row["dataset"]].add(row["database_id"])
    if {dataset: len(dbs) for dataset, dbs in databases.items()} != EXPECTED_DATABASE_COUNTS:
        raise ValueError("Unexpected serialization database coverage")
    expected = {
        (dataset, db, fmt, family)
        for dataset, dbs in databases.items()
        for db in dbs
        for fmt in SERIALIZATION_FORMAT_ORDER
        for family in TOKENIZER_FAMILY_ORDER
    }
    if set(counts) != expected or any(len(values) != 1 for values in models.values()):
        raise ValueError("Incomplete serialization matrix or inconsistent tokenizer provenance")

    summaries, tokenizer_rows, csv_exceptions = [], [], []
    for dataset in DATASET_ORDER:
        dbs = sorted(databases[dataset])
        csv_values = [
            counts[dataset, db, "csv", family] for db in dbs for family in TOKENIZER_FAMILY_ORDER
        ]
        for db in dbs:
            for family in TOKENIZER_FAMILY_ORDER:
                minimum = min(
                    counts[dataset, db, fmt, family] for fmt in SERIALIZATION_FORMAT_ORDER
                )
                if counts[dataset, db, "csv", family] != minimum:
                    csv_exceptions.append(
                        {"dataset": dataset, "database_id": db, "tokenizer_family": family}
                    )
        for fmt in SERIALIZATION_FORMAT_ORDER:
            values = [
                counts[dataset, db, fmt, family] for db in dbs for family in TOKENIZER_FAMILY_ORDER
            ]
            summaries.append(
                {
                    "dataset": dataset,
                    "format": fmt,
                    "observation_count": len(values),
                    "token_total": sum(values),
                    "mean_tokens": mean(values),
                    "ratio_of_means_to_csv": sum(values) / sum(csv_values),
                    "mean_of_individual_ratios_to_csv": mean(
                        value / baseline for value, baseline in zip(values, csv_values, strict=True)
                    ),
                }
            )
            for family in TOKENIZER_FAMILY_ORDER:
                family_values = [counts[dataset, db, fmt, family] for db in dbs]
                best = [
                    min(counts[dataset, db, fmt, other] for other in TOKENIZER_FAMILY_ORDER)
                    for db in dbs
                ]
                tokenizer_rows.append(
                    {
                        "dataset": dataset,
                        "format": fmt,
                        "tokenizer_family": family,
                        "database_count": len(dbs),
                        "mean_tokens": mean(family_values),
                        "median_excess_pct": median(
                            (value / minimum - 1) * 100
                            for value, minimum in zip(family_values, best, strict=True)
                        ),
                        "best_or_tied_database_count": sum(
                            value == minimum
                            for value, minimum in zip(family_values, best, strict=True)
                        ),
                        "payload_fitting_databases": sum(
                            v <= CONTEXT_WINDOW_TOKENS for v in family_values
                        ),
                    }
                )
    spider_csv_total = sum(
        v for (dataset, _, fmt, _), v in counts.items() if dataset == "spider" and fmt == "csv"
    )
    wta_total = sum(
        v
        for (dataset, db, fmt, _), v in counts.items()
        if dataset == "spider" and db == "wta_1" and fmt == "csv"
    )
    return {
        "source": "In-memory payload measurements from benchmark databases and GGUF tokenizers",
        "measurements_sha256": hashlib.sha256(
            json.dumps(
                sorted(
                    rows,
                    key=lambda row: (
                        row["dataset"],
                        row["database_id"],
                        row["format"],
                        row["tokenizer_family"],
                    ),
                ),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "reproduce": "cd code && .venv/bin/python -m cli.audit_serialization_tokens",
        "record_count": len(rows),
        "database_counts": {dataset: len(databases[dataset]) for dataset in DATASET_ORDER},
        "representative_tokenizers": {family: sorted(values) for family, values in models.items()},
        "payload_threshold_tokens": CONTEXT_WINDOW_TOKENS,
        "payload_threshold_inclusive": True,
        "format_means_and_ratios": summaries,
        "tokenizer_medians_and_payload_fit": tokenizer_rows,
        "csv_smallest_for_every_database_tokenizer": not csv_exceptions,
        "csv_not_smallest_exceptions": csv_exceptions,
        "wta_1_spider_csv": {
            "tokens": wta_total,
            "pooled_tokens": spider_csv_total,
            "share_pct": wta_total / spider_csv_total * 100,
        },
        "interpretation": [
            "558 repeated measurements of 31 databases; no inference-model exclusion applies.",
            "Format means pool database-family observations; ratios are ratios of means.",
            "Tokenizer excess is paired to the smallest family count for each database-format.",
            "Zero median excess does not imply best on every database.",
            "Payload fit excludes instructions, question, chat-template tokens and "
            "output headroom.",
            "Arithmetic means need not describe a typical database; wta_1 dominates Spider CSV.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    paths = ProjectPaths.discover()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=paths.models_dir,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.root / "analysis/summaries/serialization_token_audit.json",
    )
    args = parser.parse_args(argv)
    audit = audit_serialization(measure_serialization_tokens(paths, models_dir=args.models_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: audit[key]
                for key in (
                    "record_count",
                    "database_counts",
                    "wta_1_spider_csv",
                    "csv_smallest_for_every_database_tokenizer",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
