"""Download the Spider and BIRD datasets from their official public sources.

Spider:  Google Drive archive (via ``gdown``).
BIRD:    dev.zip / train.zip from the official OSS bucket (via ``requests``).

Both are idempotent: existing extracted files are left in place unless --force.
The archives bundle the SQLite databases, the question JSON and the gold SQL.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from benchmark import paths

SPIDER_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
BIRD_TRAIN_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def _unzip(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)


def download_spider(force: bool = False) -> None:
    import gdown

    root = paths.bench_root() / "data" / "spider"
    if (root / "dev.json").exists() and not force:
        print(f"[spider] already present at {root}, skipping (use --force to re-download)")
        return
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "spider.zip"
    print("[spider] downloading from Google Drive ...")
    gdown.download(id=SPIDER_GDRIVE_ID, output=str(archive), quiet=False)
    print("[spider] extracting ...")
    _unzip(archive, root)
    # The archive extracts into a nested 'spider' or 'spider_data' folder; flatten.
    _flatten_spider(root)
    print(f"[spider] done -> {root}")


def _flatten_spider(root: Path) -> None:
    """Move files up if the zip created a single nested directory."""
    if (root / "dev.json").exists():
        return
    for nested_name in ("spider", "spider_data"):
        nested = root / nested_name
        if nested.is_dir() and (nested / "dev.json").exists():
            for item in nested.iterdir():
                target = root / item.name
                if not target.exists():
                    item.rename(target)
            break


def download_bird(force: bool = False, with_train: bool = False) -> None:
    root = paths.bench_root() / "data" / "bird"
    dev_marker = root / "dev" / "dev.json"
    if dev_marker.exists() and not force:
        print(f"[bird] dev already present at {root/'dev'}, skipping (use --force)")
    else:
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "dev.zip"
        print("[bird] downloading dev.zip ...")
        _download(BIRD_DEV_URL, archive)
        print("[bird] extracting dev.zip ...")
        _unzip(archive, root)
        _normalize_bird_dev(root)

    if with_train:
        archive = root / "train.zip"
        if not (root / "train").exists() or force:
            print("[bird] downloading train.zip ...")
            _download(BIRD_TRAIN_URL, archive)
            _unzip(archive, root)
    print(f"[bird] done -> {root}")


def _normalize_bird_dev(root: Path) -> None:
    """BIRD dev.zip nests under 'dev_20240627/' and the DBs may be a nested zip.

    Normalize so that ``data/bird/dev/dev.json`` and
    ``data/bird/dev/dev_databases/<db>/<db>.sqlite`` exist, matching config.toml.
    """
    dev = root / "dev"
    # Find the directory that actually contains dev.json.
    src: Path | None = None
    if (dev / "dev.json").exists():
        src = dev
    else:
        for cand in root.glob("**/dev.json"):
            src = cand.parent
            break
    if src is None:
        return
    if src != dev:
        dev.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dev / item.name
            if not target.exists():
                item.rename(target)
    # Some releases ship the databases as a nested dev_databases.zip.
    nested_zip = dev / "dev_databases.zip"
    if nested_zip.exists() and not (dev / "dev_databases").exists():
        _unzip(nested_zip, dev)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Spider and BIRD datasets.")
    ap.add_argument(
        "--datasets", nargs="+", choices=["spider", "bird"], default=["spider", "bird"]
    )
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--bird-train", action="store_true", help="also fetch BIRD train.zip")
    args = ap.parse_args()

    if "spider" in args.datasets:
        download_spider(force=args.force)
    if "bird" in args.datasets:
        download_bird(force=args.force, with_train=args.bird_train)


if __name__ == "__main__":
    main()
