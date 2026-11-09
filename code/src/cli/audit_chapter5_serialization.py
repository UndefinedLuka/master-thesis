"""Independently reproduce Chapter 5 serialization evidence from the source CSV.

Run from code/: python -m cli.audit_chapter5_serialization
Only the new audit JSON is written; existing serialization artifacts are read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from visualizations.config import (
    CONTEXT_WINDOW_TOKENS,
    DATASET_ORDER,
    EXPECTED_DATABASE_COUNTS,
    SERIALIZATION_FORMAT_ORDER,
    TOKENIZER_FAMILY_ORDER,
)


def audit_serialization(source: Path) -> dict:
    """Use raw integer sums and paired minima, independent of display calculations."""
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
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
    expected = {(dataset, db, fmt, family)
                for dataset, dbs in databases.items() for db in dbs
                for fmt in SERIALIZATION_FORMAT_ORDER for family in TOKENIZER_FAMILY_ORDER}
    if set(counts) != expected or any(len(values) != 1 for values in models.values()):
        raise ValueError("Incomplete serialization matrix or inconsistent tokenizer provenance")

    summaries, tokenizer_rows, csv_exceptions = [], [], []
    for dataset in DATASET_ORDER:
        dbs = sorted(databases[dataset])
        csv_values = [counts[dataset, db, "csv", family]
                      for db in dbs for family in TOKENIZER_FAMILY_ORDER]
        for db in dbs:
            for family in TOKENIZER_FAMILY_ORDER:
                minimum = min(
                    counts[dataset, db, fmt, family] for fmt in SERIALIZATION_FORMAT_ORDER
                )
                if counts[dataset, db, "csv", family] != minimum:
                    csv_exceptions.append({"dataset": dataset, "database_id": db,
                                           "tokenizer_family": family})
        for fmt in SERIALIZATION_FORMAT_ORDER:
            values = [counts[dataset, db, fmt, family]
                      for db in dbs for family in TOKENIZER_FAMILY_ORDER]
            summaries.append({
                "dataset": dataset, "format": fmt, "observation_count": len(values),
                "token_total": sum(values), "mean_tokens": mean(values),
                "ratio_of_means_to_csv": sum(values) / sum(csv_values),
                "mean_of_individual_ratios_to_csv": mean(
                    value / baseline for value, baseline in zip(values, csv_values, strict=True)
                ),
            })
            for family in TOKENIZER_FAMILY_ORDER:
                family_values = [counts[dataset, db, fmt, family] for db in dbs]
                best = [min(counts[dataset, db, fmt, other] for other in TOKENIZER_FAMILY_ORDER)
                        for db in dbs]
                tokenizer_rows.append({
                    "dataset": dataset, "format": fmt, "tokenizer_family": family,
                    "database_count": len(dbs), "mean_tokens": mean(family_values),
                    "median_excess_pct": median(
                        (value / minimum - 1) * 100
                        for value, minimum in zip(family_values, best, strict=True)
                    ),
                    "best_or_tied_database_count": sum(
                        value == minimum for value, minimum in zip(family_values, best, strict=True)
                    ),
                    "payload_fitting_databases": sum(
                        v <= CONTEXT_WINDOW_TOKENS for v in family_values
                    ),
                })
    spider_csv_total = sum(v for (dataset, _, fmt, _), v in counts.items()
                           if dataset == "spider" and fmt == "csv")
    wta_total = sum(v for (dataset, db, fmt, _), v in counts.items()
                    if dataset == "spider" and db == "wta_1" and fmt == "csv")
    return {
        "source": "code/results/input_token_overhead/serialization_tokens_by_database.csv",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reproduce": "cd code && .venv/bin/python -m cli.audit_chapter5_serialization",
        "record_count": len(rows),
        "database_counts": {dataset: len(databases[dataset]) for dataset in DATASET_ORDER},
        "representative_tokenizers": {family: sorted(values) for family, values in models.items()},
        "payload_threshold_tokens": CONTEXT_WINDOW_TOKENS,
        "payload_threshold_inclusive": True,
        "format_means_and_ratios": summaries,
        "tokenizer_medians_and_payload_fit": tokenizer_rows,
        "csv_smallest_for_every_database_tokenizer": not csv_exceptions,
        "csv_not_smallest_exceptions": csv_exceptions,
        "wta_1_spider_csv": {"tokens": wta_total, "pooled_tokens": spider_csv_total,
                              "share_pct": wta_total / spider_csv_total * 100},
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


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=root / "results/input_token_overhead/serialization_tokens_by_database.csv",
    )
    parser.add_argument("--output", type=Path,
                        default=root / "analysis/summaries/chapter5_serialization_audit.json")
    args = parser.parse_args()
    audit = audit_serialization(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in (
        "record_count", "database_counts", "wta_1_spider_csv",
        "csv_smallest_for_every_database_tokenizer",
    )}, indent=2))


if __name__ == "__main__":
    main()
