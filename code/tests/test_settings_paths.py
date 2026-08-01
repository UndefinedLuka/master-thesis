"""Tests for path helpers, project-root discovery, and the download registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.paths import ProjectPaths
from config.registry import get_download_spec
from config.settings import default_db_root, default_dev_json, effective_db_root, sampled_db_root


def test_effective_db_root_spider_uses_downloaded_dbs(tmp_path: Path) -> None:
    assert effective_db_root("spider", tmp_path) == default_db_root("spider", tmp_path)
    assert "spider" in str(effective_db_root("spider", tmp_path))


def test_effective_db_root_bird_uses_sampled_copies(tmp_path: Path) -> None:
    # BIRD must read the materialized sampled databases, never the raw download.
    assert effective_db_root("bird", tmp_path) == sampled_db_root(tmp_path)
    assert effective_db_root("bird", tmp_path) == tmp_path / "bird_sampled"


def test_default_dev_json_per_dataset(tmp_path: Path) -> None:
    assert default_dev_json("spider", tmp_path).name == "dev.json"
    assert "spider" in str(default_dev_json("spider", tmp_path))
    assert "bird" in str(default_dev_json("bird", tmp_path))


def test_project_paths_from_root(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    assert paths.data_dir == tmp_path.resolve() / "data"
    assert paths.models_dir == tmp_path.resolve() / "models"
    assert paths.results_dir == tmp_path.resolve() / "results"


def test_project_paths_discover_walks_up_to_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    nested = tmp_path / "src" / "deep"
    nested.mkdir(parents=True)
    paths = ProjectPaths.discover(start=nested)
    assert paths.root == tmp_path.resolve()


def test_dataset_and_processed_dirs(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    assert paths.dataset_dir("spider") == paths.data_dir / "spider"
    assert paths.processed_dir("bird") == paths.data_dir / "bird_processed"


# --- download registry ---------------------------------------------------------------------


def test_registry_specs() -> None:
    spider = get_download_spec("spider")
    assert spider.source == "gdrive"
    bird = get_download_spec("bird")
    assert bird.source == "http"
    assert bird.recurse_inner_zip is True


def test_registry_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="No download spec"):
        get_download_spec("imdb")


def test_registry_env_override_read_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    # Overrides must take effect without re-importing the module.
    monkeypatch.setenv("BIRD_DEV_URL", "https://example.test/dev.zip")
    assert get_download_spec("bird").url_or_id == "https://example.test/dev.zip"
    monkeypatch.setenv("SPIDER_GDRIVE_ID", "override-id")
    assert get_download_spec("spider").url_or_id == "override-id"
