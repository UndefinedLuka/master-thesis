"""Tests for the benchmark runner loops, format parsing, and the LRU cache."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from cli.run_benchmark import _parse_formats
from cli.runner import LRUCache, run_direct_slices
from config.settings import BackendConfig, DatasetConfig, RunConfig
from llm.types import BackendInfo, LLMRequest, LLMResponse, empty_metrics
from results.store import load_results
from serialization.types import FORMATS


@dataclass(slots=True)
class FakeBackend:
    """Fixed-answer backend with a controllable count_tokens (to force context skips)."""

    prompt_tokens: int = 10
    calls: int = field(default=0)

    def info(self) -> BackendInfo:
        return BackendInfo(name="fake", model_id="fake-model")

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text="There are 5 singers.", metrics=empty_metrics())

    def count_tokens(self, text: str) -> int:
        return self.prompt_tokens

    def close(self) -> None:
        pass


def _example(idx: int) -> dict:
    return {
        "dataset": "spider",
        "db_id": "tiny",
        "question": "How many singers?",
        "gold_sql": "SELECT count(*) FROM singer",
        "evidence": None,
        "difficulty": None,
        "example_id": f"spider:tiny:{idx}",
    }


def _run_config(tmp_path: Path, tiny_db: Path, **backend_overrides) -> RunConfig:
    # runner resolves DBs as <db_root>/<db_id>/<db_id>.sqlite
    db_root = tmp_path / "dbs"
    (db_root / "tiny").mkdir(parents=True, exist_ok=True)
    shutil.copy(tiny_db, db_root / "tiny" / "tiny.sqlite")
    return RunConfig(
        backend=BackendConfig(model_id="fake-model", **backend_overrides),
        dataset=DatasetConfig(name="spider", dev_json=tmp_path / "dev.json", db_root=db_root),
        pipeline="direct",
        formats=("json",),
        results_dir=tmp_path / "results",
    )


def test_run_direct_writes_results_and_resumes(tmp_path: Path, tiny_db: Path) -> None:
    cfg = _run_config(tmp_path, tiny_db)
    backend = FakeBackend()
    examples = [_example(0), _example(1)]

    run_direct_slices(backend, cfg, examples)
    out = cfg.results_dir / "spider" / "direct" / "json" / "fake-model.ndjson"
    assert len(load_results(out)) == 2
    assert backend.calls == 2

    # Re-running resumes: nothing is re-generated, nothing is duplicated.
    run_direct_slices(backend, cfg, examples)
    assert backend.calls == 2
    assert len(load_results(out)) == 2


def test_run_direct_skips_oversized_database(tmp_path: Path, tiny_db: Path) -> None:
    cfg = _run_config(tmp_path, tiny_db, n_ctx=1024, max_tokens=1000)
    backend = FakeBackend(prompt_tokens=500)  # 500 + 1000 headroom >= 1024 -> skip
    run_direct_slices(backend, cfg, [_example(0)])

    out = cfg.results_dir / "spider" / "direct" / "json" / "fake-model.ndjson"
    records = load_results(out)
    assert backend.calls == 0  # never generated
    assert records[0]["error"] == "context_exceeded"
    assert records[0]["raw_output"] == ""


def test_parse_formats_all_and_subset() -> None:
    assert _parse_formats("all") == FORMATS
    assert _parse_formats("json,markdown") == ("json", "markdown")
    assert _parse_formats("json, markdown") == ("json", "markdown")  # whitespace tolerated
    assert _parse_formats("json,json") == ("json",)  # duplicates collapse


def test_parse_formats_rejects_unknown() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="Unknown format"):
        _parse_formats("json,protobuf")


def test_lru_cache_evicts_least_recently_used() -> None:
    cache = LRUCache(2)
    cache["a"] = 1
    cache["b"] = 2
    _ = cache["a"]  # refresh recency of "a"
    cache["c"] = 3  # evicts "b", not "a"
    assert "a" in cache
    assert "b" not in cache
    assert "c" in cache
