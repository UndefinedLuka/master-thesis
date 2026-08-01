"""Materialize the sub-sampled BIRD databases to disk.

python -m cli.build_sampled_dbs

Writes ``data/bird_sampled/<db_id>/<db_id>.sqlite`` — a copy of each BIRD dev database
containing only ``BIRD_SAMPLE_ROWS`` deterministic rows per table (seed ``SAMPLE_SEED``).
Every downstream step (serialization for Direct Context Inference, reference answers, and
Text-to-SQL query execution) reads these files, so all of them see the exact same rows
(Chapter 3's level playing field). Idempotent: existing files are skipped; pass --overwrite
to rebuild (do that whenever BIRD_SAMPLE_ROWS or SAMPLE_SEED changes).
"""

from __future__ import annotations

import argparse

from tqdm import tqdm

from config.config import BIRD_SAMPLE_ROWS, SAMPLE_SEED
from config.paths import ProjectPaths
from config.settings import default_db_root, sampled_db_root
from sqlite.materialize import materialize_sampled_db


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize sampled BIRD databases")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rebuild databases that already exist (needed after changing sample size/seed)",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.discover()
    src_root = default_db_root("bird", paths.data_dir)
    out_root = sampled_db_root(paths.data_dir)
    if not src_root.is_dir():
        raise FileNotFoundError(
            f"BIRD databases not found at {src_root}. Run cli.download_datasets first."
        )

    db_ids = sorted(p.name for p in src_root.iterdir() if p.is_dir())
    print(
        f"Materializing {len(db_ids)} BIRD databases "
        f"(n_rows={BIRD_SAMPLE_ROWS}, seed={SAMPLE_SEED}) -> {out_root}"
    )
    built = skipped = 0
    for db_id in tqdm(db_ids, desc="Sampling BIRD", unit="db"):
        src = src_root / db_id / f"{db_id}.sqlite"
        if not src.exists():
            continue
        dest = out_root / db_id / f"{db_id}.sqlite"
        if dest.exists() and not args.overwrite:
            skipped += 1
            continue
        materialize_sampled_db(src, dest, BIRD_SAMPLE_ROWS, seed=SAMPLE_SEED)
        built += 1
    print(f"Done: {built} built, {skipped} already present (use --overwrite to rebuild).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
