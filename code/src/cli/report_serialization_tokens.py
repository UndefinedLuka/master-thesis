"""Measure serialized input tokens in memory and print the report to standard output.

Run ``python -m cli.report_serialization_tokens`` or ``thesis-report-serialization-tokens``.
Measurements use vocabulary-only GGUF tokenizers; no inference or file output is performed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import get_args

from config.paths import ProjectPaths
from config.settings import DatasetName
from serialization.tokens import FAMILY_BY_KEY, TokenRecord, measure_serialization_tokens
from serialization.types import FORMATS


def build_parser(paths: ProjectPaths | None = None) -> argparse.ArgumentParser:
    """Build the standalone command-line interface."""
    project_paths = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description="Measure serialized database payload tokens and print CSV to stdout",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=get_args(DatasetName),
        help="dataset to measure; repeat for both (default: spider and bird)",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=FORMATS,
        help="serialization format to measure; repeat as needed (default: all six)",
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=tuple(FAMILY_BY_KEY),
        help="tokenizer family to measure; repeat as needed (default: all three)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=project_paths.models_dir,
        help="directory containing experiment GGUF files (default: models/)",
    )
    parser.add_argument(
        "--tokenizer-model",
        action="append",
        type=_parse_tokenizer_override,
        metavar="FAMILY=GGUF",
        help="override a representative tokenizer GGUF; repeat per family",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="measure only the first N databases per dataset (smoke tests only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Measure payloads and print CSV without creating checkpoints or intermediate files."""
    paths = ProjectPaths.discover()
    parser = build_parser(paths)
    args = parser.parse_args(argv)
    family_keys = tuple(dict.fromkeys(args.family or FAMILY_BY_KEY))
    overrides: dict[str, Path] = {}
    for key, model_path in args.tokenizer_model or []:
        if key in overrides:
            parser.error(f"--tokenizer-model was supplied more than once for {key}")
        if key not in family_keys:
            parser.error(f"tokenizer override for unselected family {key!r}")
        overrides[key] = model_path
    records = measure_serialization_tokens(
        paths,
        datasets=tuple(dict.fromkeys(args.dataset or get_args(DatasetName))),
        formats=tuple(dict.fromkeys(args.formats or FORMATS)),
        families=tuple(FAMILY_BY_KEY[key] for key in family_keys),
        models_dir=args.models_dir,
        overrides=overrides,
        limit=args.limit,
    )
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[field.name for field in fields(TokenRecord)],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(asdict(record) for record in records)
    return 0


def _parse_tokenizer_override(raw: str) -> tuple[str, Path]:
    try:
        family, raw_path = raw.split("=", 1)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--tokenizer-model must use FAMILY=/path/to/model.gguf"
        ) from None
    family, raw_path = family.strip(), raw_path.strip()
    if family not in FAMILY_BY_KEY:
        raise argparse.ArgumentTypeError(
            f"unknown tokenizer family {family!r}; valid: {', '.join(FAMILY_BY_KEY)}"
        )
    if not raw_path:
        raise argparse.ArgumentTypeError("--tokenizer-model requires a GGUF path after '='")
    return family, Path(raw_path)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
