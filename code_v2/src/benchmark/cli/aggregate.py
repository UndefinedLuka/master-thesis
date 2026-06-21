"""Aggregate NDJSON result shards into an enriched dataset and summary table.

This is where BERTScore is computed: in one batched pass over all rows (it loads
a transformer, so doing it here keeps the GPU free for vLLM during the run). The
per-row scores are written back into ``all_results.ndjson`` and grouped summaries
are emitted as ``summary.csv``. Rich figures are produced separately by
``python -m benchmark.cli.visualize`` (which reads ``all_results.ndjson``); this
keeps the data step and the styling step independent.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import polars as pl

from benchmark import paths
from benchmark.metrics import compute_bertscore


def load_all_rows(results_dir: Path) -> pl.DataFrame:
    """Load and concatenate every result shard, skipping derived/empty files."""
    files = sorted(glob.glob(str(results_dir / "*.ndjson")))
    # Exclude the enriched output so re-running aggregate is idempotent.
    files = [f for f in files if Path(f).name != "all_results.ndjson"]
    frames = [
        pl.read_ndjson(f, infer_schema_length=None)
        for f in files
        if Path(f).stat().st_size > 0
    ]
    if not frames:
        raise SystemExit(f"No NDJSON shards found in {results_dir}")
    return pl.concat(frames, how="diagonal_relaxed")


def _fill_bertscore(df: pl.DataFrame, model_type: str | None) -> pl.DataFrame:
    """Compute BERTScore for rows that need it (scored, no API error)."""
    needs = (
        df.with_row_index("_ri")
        .filter(
            pl.col("bertscore").is_null()
            & ~pl.col("reference_unavailable")
            & pl.col("error").is_null()
        )
        .select("_ri", "output_text", "reference_text")
    )
    if needs.height == 0:
        return df.with_columns(pl.col("bertscore").fill_null(0.0))

    print(f"Computing BERTScore for {needs.height} rows ...")
    scores = compute_bertscore(
        needs["output_text"].to_list(),
        needs["reference_text"].to_list(),
        model_type=model_type,
    )
    score_by_ri = dict(zip(needs["_ri"].to_list(), scores))
    return (
        df.with_row_index("_ri")
        .with_columns(
            pl.col("_ri")
            .map_elements(lambda ri: score_by_ri.get(ri), return_dtype=pl.Float64)
            .alias("_bs")
        )
        .with_columns(
            pl.when(pl.col("_bs").is_not_null())
            .then(pl.col("_bs"))
            .otherwise(pl.col("bertscore").fill_null(0.0))
            .alias("bertscore")
        )
        .drop("_ri", "_bs")
    )


def _summary(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["model", "dataset", "pipeline", "format"])
        .agg(
            pl.len().alias("n"),
            pl.col("input_tokens").mean().alias("mean_input_tokens"),
            pl.col("output_tokens").mean().alias("mean_output_tokens"),
            pl.col("ttft_s").mean().alias("mean_ttft_s"),
            pl.col("total_generation_s").mean().alias("mean_total_gen_s"),
            pl.col("rouge_l").mean().alias("mean_rouge_l"),
            pl.col("bertscore").mean().alias("mean_bertscore"),
            pl.col("execution_accuracy").mean().alias("execution_accuracy"),
            pl.col("result_set_match").mean().alias("result_set_match"),
            pl.col("retry_count").mean().alias("mean_retry"),
            pl.col("definitive_failure").mean().alias("failure_rate"),
            pl.col("error").is_not_null().mean().alias("api_error_rate"),
        )
        .sort(["dataset", "model", "pipeline", "format"])
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate result shards.")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--bertscore-model", default=None, help="HF model_type for BERTScore")
    ap.add_argument("--no-bertscore", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else paths.results_dir()
    df = load_all_rows(results_dir)

    if not args.no_bertscore:
        df = _fill_bertscore(df, args.bertscore_model)
    else:
        df = df.with_columns(pl.col("bertscore").fill_null(0.0))

    enriched = results_dir / "all_results.ndjson"
    df.write_ndjson(enriched)
    print(f"Enriched rows -> {enriched} ({df.height} rows)")

    summary = _summary(df)
    csv_path = results_dir / "summary.csv"
    summary.write_csv(csv_path)
    print(f"Summary -> {csv_path}")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(summary)
    print("Next: render figures with  python -m benchmark.cli.visualize")


if __name__ == "__main__":
    main()
