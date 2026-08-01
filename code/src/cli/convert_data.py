"""Serialize all databases for all datasets into all six formats on disk.

python -m cli.convert_data

Writes ``data/<dataset>_processed/<db_id>/<db_id>.{json,xml,yaml,csv,md,toon}`` plus the
schema DDL as ``<db_id>.sql``. The benchmark itself serializes on the fly, so this step is
optional — useful to eyeball formats or to pre-materialize them.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args

from tqdm import tqdm

from config.paths import ProjectPaths
from config.settings import DatasetConfig, DatasetName, effective_db_root
from datasets.loader import db_path_for, load_db_contents
from serialization.dispatch import serialize
from serialization.types import FORMAT_EXTENSIONS, FORMATS
from sqlite.schema import get_schema_ddl


def _iter_db_ids(db_root: Path) -> list[str]:
    if not db_root.is_dir():
        raise FileNotFoundError(f"Database root not found: {db_root}")
    return sorted(p.name for p in db_root.iterdir() if p.is_dir())


def main(argv: list[str] | None = None) -> int:
    """Serialize every database of every dataset into all formats on disk."""
    paths = ProjectPaths.discover()
    for name in get_args(DatasetName):
        db_root = effective_db_root(name, paths.data_dir)
        out_root = paths.processed_dir(name)
        cfg = DatasetConfig(name=name, dev_json=Path(), db_root=db_root)

        db_ids = _iter_db_ids(db_root)
        pbar = tqdm(db_ids, desc=f"Converting {name}", unit="db")
        for db_id in pbar:
            pbar.set_postfix(db=db_id)
            db_path = db_path_for(db_id, cfg)
            if not db_path.exists():
                continue
            contents = load_db_contents(db_path)
            db_out = out_root / db_id
            db_out.mkdir(parents=True, exist_ok=True)
            (db_out / f"{db_id}.sql").write_text(get_schema_ddl(db_path), encoding="utf-8")
            for fmt in FORMATS:
                ext = FORMAT_EXTENSIONS[fmt]
                (db_out / f"{db_id}.{ext}").write_text(serialize(contents, fmt), encoding="utf-8")
        print(f"Converted {len(db_ids)} databases -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
