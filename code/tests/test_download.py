"""Tests for archive extraction: zip-slip rejection and nested-zip recursion (no network)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from datasets.download import _safe_extract_zip, extract_archive


def _make_zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def test_safe_extract_zip_extracts_normal_entries(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path / "ok.zip", {"a/b.txt": b"hello"})
    dest = tmp_path / "out"
    dest.mkdir()
    _safe_extract_zip(archive, dest)
    assert (dest / "a" / "b.txt").read_bytes() == b"hello"


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path / "evil.zip", {"../escape.txt": b"pwn"})
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ValueError, match="Unsafe path"):
        _safe_extract_zip(archive, dest)
    assert not (tmp_path / "escape.txt").exists()


def test_extract_archive_recurses_into_nested_zip(tmp_path: Path) -> None:
    # BIRD nests dev_databases.zip inside the outer archive.
    inner = _make_zip(tmp_path / "inner.zip", {"db/data.sqlite": b"bytes"})
    outer = _make_zip(
        tmp_path / "outer.zip",
        {"dev/inner.zip": inner.read_bytes(), "dev/dev.json": b"[]"},
    )
    dest = tmp_path / "out"
    dest.mkdir()
    extract_archive(outer, dest, recurse_inner_zip=True)
    assert (dest / "dev" / "dev.json").exists()
    assert (dest / "dev" / "db" / "data.sqlite").read_bytes() == b"bytes"


def test_extract_archive_without_recursion_leaves_inner_zip(tmp_path: Path) -> None:
    inner = _make_zip(tmp_path / "inner.zip", {"x.txt": b"x"})
    outer = _make_zip(tmp_path / "outer.zip", {"inner.zip": inner.read_bytes()})
    dest = tmp_path / "out"
    dest.mkdir()
    extract_archive(outer, dest, recurse_inner_zip=False)
    assert (dest / "inner.zip").exists()
    assert not (dest / "x.txt").exists()
