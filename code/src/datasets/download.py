"""Download + extract datasets, idempotently.

``ensure_dataset`` is a no-op when the dataset's ``expected_marker`` already exists, so it is
safe to re-run. Spider comes from Google Drive (``gdown``); BIRD from a direct HTTPS zip
(streamed via ``httpx``) whose outer archive nests ``dev_databases.zip``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import httpx
from tqdm import tqdm

from config.paths import ProjectPaths
from config.registry import DownloadSpec

_HTTP_TIMEOUT_S = 120.0
_CHUNK = 1 << 20  # 1 MiB


def ensure_dataset(spec: DownloadSpec, paths: ProjectPaths) -> Path:
    """Ensure ``spec``'s dataset is present under ``paths.data_dir/<name>``; return that dir.

    Idempotent: if ``expected_marker`` already exists, returns immediately.
    """
    dataset_dir = paths.dataset_dir(spec.name)
    marker = dataset_dir / spec.expected_marker
    if marker.exists():
        print(f"[{spec.name}] already present ({marker}); skipping download.")
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_dir / spec.archive_name

    if not archive_path.exists():
        download_archive(spec, archive_path)
    else:
        print(f"[{spec.name}] archive already downloaded ({archive_path}); extracting.")

    extract_archive(archive_path, dataset_dir, recurse_inner_zip=spec.recurse_inner_zip)

    if not marker.exists():
        raise FileNotFoundError(
            f"[{spec.name}] extraction finished but expected marker is missing: {marker}. "
            f"The dataset layout may have changed — verify the download spec in config.registry."
        )
    print(f"[{spec.name}] ready at {dataset_dir}")
    return dataset_dir


def download_archive(spec: DownloadSpec, dest: Path) -> Path:
    """Fetch ``spec`` to ``dest`` (Google Drive or HTTP). Returns ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if spec.source == "gdrive":
        _download_gdrive(spec.url_or_id, dest)
    elif spec.source == "http":
        _download_http(spec.url_or_id, dest)
    else:  # pragma: no cover - guarded by SourceKind Literal
        raise ValueError(f"Unknown source kind: {spec.source!r}")
    return dest


def _download_gdrive(file_id: str, dest: Path) -> None:
    import gdown  # lazy: only needed for Spider

    print(f"Downloading from Google Drive (id={file_id}) -> {dest}")
    # gdown returns None on failure (e.g. quota page) instead of raising.
    if gdown.download(id=file_id, output=str(dest), quiet=False) is None:
        raise RuntimeError(
            f"Google Drive download failed for id={file_id} (quota exceeded or file moved). "
            f"Override the id via the SPIDER_GDRIVE_ID environment variable."
        )


def _download_http(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=_HTTP_TIMEOUT_S) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) or None
        with (
            tmp.open("wb") as handle,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in response.iter_bytes(_CHUNK):
                handle.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def extract_archive(archive: Path, dest_dir: Path, *, recurse_inner_zip: bool = False) -> Path:
    """Unzip ``archive`` into ``dest_dir``. If ``recurse_inner_zip``, also extract any nested
    ``.zip`` files produced by the first extraction (BIRD's ``dev_databases.zip``)."""
    print(f"Extracting {archive.name} -> {dest_dir}")
    _safe_extract_zip(archive, dest_dir)

    if recurse_inner_zip:
        for inner in sorted(dest_dir.rglob("*.zip")):
            if inner == archive:
                continue
            print(f"Extracting nested {inner.relative_to(dest_dir)}")
            _safe_extract_zip(inner, inner.parent)
    return dest_dir


def _safe_extract_zip(archive: Path, dest_dir: Path) -> None:
    """Extract a zip, rejecting path-traversal entries (zip-slip)."""
    dest_root = dest_dir.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = (dest_root / member).resolve()
            if not _is_within(dest_root, target):
                raise ValueError(f"Unsafe path in archive {archive.name}: {member!r}")
        zf.extractall(dest_dir)


def _is_within(root: Path, target: Path) -> bool:
    return root == target or root in target.parents
