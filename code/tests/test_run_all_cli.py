"""Tests for model discovery in the run_all CLI (the --models allow-list).

Model selection is what deploy/run_models.sbatch relies on to give each Slurm array task
exactly one model, so an unknown name must fail loudly instead of running zero slices.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cli.run_all import _find_models, _parse_models


def _models_dir(tmp_path: Path, *names: str) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    for name in names:
        (models / name).write_bytes(b"GGUF")
    return models


def test_find_models_returns_all_sorted(tmp_path: Path) -> None:
    models = _models_dir(tmp_path, "b.gguf", "a.gguf", "notes.txt")
    assert _find_models(models) == ["a.gguf", "b.gguf"]  # .txt ignored


def test_find_models_honors_filter_and_request_order(tmp_path: Path) -> None:
    models = _models_dir(tmp_path, "a.gguf", "b.gguf", "c.gguf")
    # Requested order wins over sort order, so --array indices map to a stable model.
    assert _find_models(models, ["c.gguf", "a.gguf"]) == ["c.gguf", "a.gguf"]


def test_find_models_deduplicates_filter(tmp_path: Path) -> None:
    models = _models_dir(tmp_path, "a.gguf", "b.gguf")
    assert _find_models(models, ["a.gguf", "a.gguf"]) == ["a.gguf"]


def test_find_models_rejects_unknown_name_and_lists_available(tmp_path: Path) -> None:
    models = _models_dir(tmp_path, "a.gguf", "b.gguf")
    with pytest.raises(FileNotFoundError) as excinfo:
        _find_models(models, ["a.gguf", "typo.gguf"])
    message = str(excinfo.value)
    assert "typo.gguf" in message
    assert "a.gguf" in message and "b.gguf" in message  # available names are shown


def test_find_models_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Models directory not found"):
        _find_models(tmp_path / "nope")


def test_find_models_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No .gguf files"):
        _find_models(_models_dir(tmp_path))


def test_parse_models_splits_and_strips() -> None:
    assert _parse_models(" a.gguf , b.gguf ") == ["a.gguf", "b.gguf"]


def test_parse_models_rejects_empty() -> None:
    with pytest.raises(Exception, match="at least one filename"):
        _parse_models(" , ")
