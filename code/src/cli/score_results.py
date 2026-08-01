"""Score benchmark results against reference answers (the retrieval-accuracy pass).

Reads benchmark NDJSON produced by ``run_benchmark`` (a single file, or a directory tree of
the split layout), computes ROUGE-L + BERTScore for every record with a usable gold reference,
adds a Spider-style execution-accuracy verdict for Text-to-SQL records, and writes a
``<name>.scored.ndjson`` next to each input — preserving the per-slice folder layout.

    # score one slice file
    python -m cli.score_results --results results/spider/direct/json/model.ndjson

    # score an entire run tree, auto-resolving references per dataset
    python -m cli.score_results --results results/

    # score against an explicit reference file (instead of data/<dataset>_references.ndjson)
    python -m cli.score_results --results results/ --references data/spider_references.ndjson

    # skip BERTScore (ROUGE-L + execution accuracy only)
    python -m cli.score_results --results results/ --no-bertscore

Requires the optional metrics dependencies: ``uv sync --extra metrics``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config.paths import ProjectPaths
from config.settings import ScoringConfig
from metrics.scoring import (
    ReferenceLookup,
    ScoringStats,
    references_from_rows,
    score_records,
)
from pipelines.types import PipelineResult
from reference.answers import load_reference_file
from results.store import load_results, write_results

SCORED_SUFFIX = ".scored.ndjson"


def main(argv: list[str] | None = None) -> int:
    """Score benchmark NDJSON file(s) and write ``.scored.ndjson`` siblings."""
    parser = argparse.ArgumentParser(description="Score benchmark results")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="benchmark NDJSON file, or a directory scored recursively",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=None,
        help=(
            "explicit reference NDJSON file; default: auto-resolve "
            "data/<dataset>_references.ndjson per dataset"
        ),
    )
    parser.add_argument(
        "--no-bertscore",
        dest="with_bertscore",
        action="store_false",
        help="skip BERTScore (ROUGE-L + execution accuracy only)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-score files that already have a .scored.ndjson",
    )
    parser.set_defaults(with_bertscore=True)
    args = parser.parse_args(argv)

    scoring = ScoringConfig(with_bertscore=args.with_bertscore, overwrite=args.overwrite)
    inputs = _discover_inputs(args.results, scoring.overwrite)
    if not inputs:
        print(f"No benchmark NDJSON files to score under {args.results}")
        return 0

    explicit_refs: ReferenceLookup | None = None
    if args.references is not None:
        explicit_refs = references_from_rows(load_reference_file(args.references))

    paths = ProjectPaths.discover() if explicit_refs is None else None
    ref_cache: dict[Path, ReferenceLookup] = {}
    print(f"Scoring {len(inputs)} file(s); BERTScore={'on' if scoring.with_bertscore else 'off'}")
    for in_path in inputs:
        records = load_results(in_path)
        if not records:
            continue
        if explicit_refs is not None:
            references = explicit_refs
        else:
            assert paths is not None
            references = _references_for(records, paths, ref_cache)
        scored, stats = score_records(
            records,
            references,
            with_bertscore=scoring.with_bertscore,
            force_order_sensitive=scoring.force_order_sensitive,
            model_type=scoring.model_type,
            lang=scoring.lang,
        )
        out_path = _scored_path(in_path)
        write_results(out_path, scored)
        _print_summary(in_path, out_path, scored, stats)
    return 0


def _discover_inputs(target: Path, overwrite: bool) -> list[Path]:
    if target.is_file():
        if target.name.endswith(SCORED_SUFFIX):
            # Scoring a .scored.ndjson would overwrite it with itself, double-scoring records.
            raise ValueError(f"--results points at an already-scored file: {target}")
        return [target]
    if target.is_dir():
        return sorted(
            p
            for p in target.rglob("*.ndjson")
            if not p.name.endswith(SCORED_SUFFIX) and (overwrite or not _scored_path(p).exists())
        )
    raise FileNotFoundError(f"--results path does not exist: {target}")


def _references_for(
    records: list[PipelineResult],
    paths: ProjectPaths,
    cache: dict[Path, ReferenceLookup],
) -> ReferenceLookup:
    lookup: ReferenceLookup = {}
    for dataset in {record["dataset"] for record in records}:
        ref_path = paths.data_dir / f"{dataset}_references.ndjson"
        if ref_path not in cache:
            cache[ref_path] = references_from_rows(load_reference_file(ref_path))
        lookup.update(cache[ref_path])
    return lookup


def _scored_path(in_path: Path) -> Path:
    return in_path.with_suffix("").with_suffix(SCORED_SUFFIX)


def _print_summary(
    in_path: Path, out_path: Path, scored: list[PipelineResult], stats: ScoringStats
) -> None:
    rouge_vals = [r["scores"]["rouge_l_f"] for r in scored if r["scores"] is not None]
    exec_vals = [r["execution"]["execution_accuracy"] for r in scored if r["execution"] is not None]
    parts = [f"scored {stats['scored']}/{stats['total']}"]
    if stats["skipped_no_reference"]:
        parts.append(f"{stats['skipped_no_reference']} no-ref")
    if rouge_vals:
        parts.append(f"mean ROUGE-L {sum(rouge_vals) / len(rouge_vals):.3f}")
    if exec_vals:
        parts.append(f"exec-acc {sum(exec_vals) / len(exec_vals):.3f}")
    print(f"  {in_path.name}: {', '.join(parts)} -> {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
