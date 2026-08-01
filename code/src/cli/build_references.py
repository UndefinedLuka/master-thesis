"""Generate reference answers (gold SQL executed against the DB) for all datasets.

python -m cli.build_references

BIRD gold SQL runs against the materialized sampled databases (data/bird_sampled), so run
cli.build_sampled_dbs first — the reference rows must match the data the models see.
"""

from __future__ import annotations

from typing import get_args

from cli.runner import dataset_config_for
from config.paths import ProjectPaths
from config.settings import DatasetName
from datasets.loader import load_dataset
from reference.answers import build_reference_set, write_reference_file


def main(argv: list[str] | None = None) -> int:
    """Build and write the reference-answer NDJSON for every dataset."""
    paths = ProjectPaths.discover()
    for name in get_args(DatasetName):
        dataset = dataset_config_for(name, paths)
        examples = load_dataset(dataset)
        print(f"Loaded {len(examples)} examples for {name}")
        references = build_reference_set(examples, dataset)

        out_path = paths.data_dir / f"{name}_references.ndjson"
        write_reference_file(out_path, references)

        failures = sum(1 for r in references if r["error"] is not None)
        print(f"Wrote {len(references)} references ({failures} gold-SQL failures) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
