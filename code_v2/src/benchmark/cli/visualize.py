"""Render the figure suite from aggregated results.

Reads ``results/all_results.ndjson`` (produced by ``benchmark.cli.aggregate``);
if that is missing it falls back to loading the raw shards directly so you can
preview figures before running the full BERTScore pass. All styling comes from
``viz_config.toml`` — edit that file and re-run to restyle every figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from benchmark import paths
from benchmark.cli.aggregate import load_all_rows
from benchmark.viz import render_all
from benchmark.viz_config import load_viz_config


def _load_dataframe(results_dir: Path):
    enriched = results_dir / "all_results.ndjson"
    if enriched.exists():
        # infer_schema_length=None scans all rows: direct rows have null
        # final_sql/attempts that would otherwise pin a NULL-typed column.
        df = pl.read_ndjson(enriched, infer_schema_length=None)
        print(f"Loaded {df.height} enriched rows from {enriched}")
    else:
        print("all_results.ndjson not found; loading raw shards (run aggregate for BERTScore).")
        df = load_all_rows(results_dir)
        if "bertscore" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("bertscore"))
        df = df.with_columns(pl.col("bertscore").fill_null(0.0))
    # seaborn works on pandas; convert once here.
    return df.to_pandas()


def main() -> None:
    ap = argparse.ArgumentParser(description="Render benchmark figures.")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--viz-config", default=None, help="path to viz_config.toml")
    ap.add_argument("--out-dir", default=None, help="figure output dir (default results/figures)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else paths.results_dir()
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "figures"

    cfg = load_viz_config(args.viz_config)
    df = _load_dataframe(results_dir)

    drawn = render_all(df, cfg, out_dir)
    print(f"Rendered {len(drawn)} figure group(s): {', '.join(drawn)}")
    print(f"Figures -> {out_dir}  (formats: {', '.join(cfg.theme.formats)})")


if __name__ == "__main__":
    main()
