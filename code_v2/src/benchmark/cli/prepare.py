"""Prepare step: build reference answers and serialize every database.

For each requested dataset this:
  1. executes the gold SQL for every question -> reference_<dataset>.json
  2. serializes every database to all six formats + the SQL schema
  3. writes a manifest.json recording each artifact's path and character length
     (the char length flags databases that may overflow the model context).

Idempotent: existing serialized files are skipped unless --force.
"""

from __future__ import annotations

import argparse
import json

from tqdm import tqdm

from benchmark import paths
from benchmark.config import Config, DatasetSpec, load_config
from benchmark.datasets import load_questions
from benchmark.reference import build_references
from benchmark.serialize import serialize_db_all_formats


def _databases_for(spec: DatasetSpec) -> list[str]:
    """Distinct db_ids referenced by the dataset's questions (in first-seen order)."""
    seen: dict[str, None] = {}
    for q in load_questions(spec):
        seen.setdefault(q.db_id, None)
    return list(seen)


def prepare_dataset(cfg: Config, dataset: str, force: bool, skip_reference: bool) -> None:
    spec = cfg.dataset(dataset)

    if not skip_reference:
        ref_path = paths.reference_file(dataset)
        if ref_path.exists() and not force:
            print(f"[{dataset}] reference exists ({ref_path}), skipping (use --force)")
        else:
            build_references(spec, ref_path)
            print(f"[{dataset}] reference -> {ref_path}")

    manifest: dict[str, dict] = {}
    db_ids = _databases_for(spec)
    sample_n = spec.subsample_rows  # None for Spider, N for BIRD

    for db_id in tqdm(db_ids, desc=f"Serialize [{dataset}]", unit="db"):
        db_path = spec.question_db_path(db_id)
        out_dir = paths.processed_db_dir(dataset, db_id)
        # Skip if all artifacts already exist and not forcing.
        done = out_dir.exists() and (out_dir / f"{db_id}.sql").exists()
        if done and not force:
            continue
        if not db_path.exists():
            tqdm.write(f"[{dataset}] WARNING missing database: {db_path}")
            continue
        info = serialize_db_all_formats(
            db_path, out_dir, tuple(cfg.formats), sample_n, cfg.seed
        )
        manifest[db_id] = info

    manifest_path = paths.manifest_file(dataset)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Merge with any previous manifest so partial runs accumulate.
    if manifest_path.exists():
        prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        prev.update(manifest)
        manifest = prev
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{dataset}] manifest -> {manifest_path} ({len(manifest)} databases)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build references and serialize databases.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--datasets", nargs="+", choices=["spider", "bird"], default=None)
    ap.add_argument("--bird-sample-n", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-reference", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.bird_sample_n is not None and "bird" in cfg.datasets:
        bird = cfg.datasets["bird"]
        cfg.datasets["bird"] = DatasetSpec(
            name=bird.name,
            questions_file=bird.questions_file,
            db_root=bird.db_root,
            tables_file=bird.tables_file,
            subsample_rows=args.bird_sample_n,
            has_evidence=bird.has_evidence,
        )

    datasets = args.datasets or list(cfg.datasets)
    for ds in datasets:
        prepare_dataset(cfg, ds, force=args.force, skip_reference=args.skip_reference)


if __name__ == "__main__":
    main()
