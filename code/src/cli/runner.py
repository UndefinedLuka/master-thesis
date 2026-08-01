"""The benchmark loops shared by run_benchmark (one slice) and run_all (every slice).

Each loop streams results to a resumable NDJSON slice file via :class:`ResultWriter`,
skipping examples whose ``(example_id, pipeline, format)`` key is already on disk.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from tqdm import tqdm

from config.paths import ProjectPaths
from config.settings import (
    DatasetConfig,
    DatasetName,
    RunConfig,
    default_dev_json,
    effective_db_root,
)
from datasets.loader import db_path_for, load_db_contents, resolve_db_path
from datasets.types import DatasetExample
from llm.backend import LLMBackend
from pipelines.direct import run_direct_inference
from pipelines.shared import base_result, model_id_of
from pipelines.text_to_sql import run_text_to_sql
from pipelines.types import PipelineResult
from prompts import load_prompt, render_prompt
from results.store import (
    CompletedKey,
    ResultWriter,
    completed_key,
    load_completed_keys,
    result_path,
)
from serialization.dispatch import serialize
from serialization.types import SerializationFormat
from sqlite.schema import get_schema_ddl

#: Generation headroom (tokens) the prompt must leave free when no output cap is set.
DEFAULT_HEADROOM_TOKENS = 1024


class LRUCache(OrderedDict[Any, Any]):
    """Minimal LRU cache backed by OrderedDict (recency refreshed on ``[]`` access)."""

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self._maxsize = maxsize

    def __getitem__(self, key: Any) -> Any:
        self.move_to_end(key)
        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self._maxsize:
            self.popitem(last=False)


def dataset_config_for(name: DatasetName, paths: ProjectPaths) -> DatasetConfig:
    """Build the DatasetConfig for ``name``, verifying BIRD's sampled DBs exist."""
    db_root = effective_db_root(name, paths.data_dir)
    if name == "bird" and not db_root.is_dir():
        raise FileNotFoundError(
            f"Materialized BIRD databases not found at {db_root}. "
            f"Run `python -m cli.build_sampled_dbs` first."
        )
    return DatasetConfig(
        name=name,
        dev_json=default_dev_json(name, paths.data_dir),
        db_root=db_root,
    )


def run_direct_slices(
    backend: LLMBackend,
    cfg: RunConfig,
    examples: list[DatasetExample],
) -> None:
    """Run the Direct Context Inference pipeline for every configured format."""
    template = load_prompt("direct")
    out_paths = {fmt: result_path(cfg, fmt) for fmt in cfg.formats}
    completed: set[CompletedKey] = set()
    if cfg.resume:
        for path in out_paths.values():
            completed |= load_completed_keys(path)

    # Generation headroom the prompt must leave free: the output cap when set, otherwise a
    # fixed safety margin so the model can still answer near the context limit.
    headroom = (
        cfg.backend.max_tokens if cfg.backend.max_tokens is not None else DEFAULT_HEADROOM_TOKENS
    )

    # Outer loop over formats so each format pass is contiguous: the LRU(2) contents
    # and serialized caches stay warm across all questions that share a database.
    for fmt in cfg.formats:
        writer = ResultWriter(out_paths[fmt])
        # LRU(2): one serialized string per database; evicts as db_id changes.
        serialized_cache = LRUCache(2)
        prompt_tokens_cache = LRUCache(2)
        skipped_dbs: set[str] = set()
        for example in tqdm(examples, desc=f"direct/{cfg.dataset.name}/{fmt}", unit="q"):
            if (example["example_id"], "direct", fmt) in completed:
                continue
            db_id = example["db_id"]
            if db_id not in serialized_cache:
                db_path = db_path_for(db_id, cfg.dataset)
                serialized_cache[db_id] = serialize(load_db_contents(db_path), fmt)
            if db_id not in prompt_tokens_cache:
                # Tokenizer-accurate count of the fixed prompt part (template + data). The
                # per-question suffix is tiny; the generation headroom absorbs it.
                system_prompt = render_prompt(template, serialized_cache[db_id])
                prompt_tokens_cache[db_id] = backend.count_tokens(system_prompt)
            if db_id not in skipped_dbs and _exceeds_context(
                prompt_tokens_cache[db_id], headroom, cfg.backend.n_ctx
            ):
                skipped_dbs.add(db_id)
                tqdm.write(
                    f"skipping db {db_id!r} for format {fmt!r}: "
                    f"~{prompt_tokens_cache[db_id]} prompt tokens + {headroom} headroom "
                    f">= n_ctx {cfg.backend.n_ctx}"
                )
            if db_id in skipped_dbs:
                result = _context_exceeded_result(example, fmt, model_id_of(backend))
            else:
                result = run_direct_inference(
                    backend,
                    example,
                    serialized_cache[db_id],
                    fmt,
                    template,
                    max_tokens=cfg.backend.max_tokens,
                    temperature=cfg.backend.temperature,
                    seed=cfg.backend.seed,
                )
            writer.append(result)
            completed.add(completed_key(result))


def run_sql_slice(
    backend: LLMBackend,
    cfg: RunConfig,
    examples: list[DatasetExample],
) -> None:
    """Run the Text-to-SQL pipeline (schema-only; no serialization format dimension)."""
    template = load_prompt("sql")
    schema_cache = LRUCache(2)
    writer = ResultWriter(result_path(cfg, None))
    completed = load_completed_keys(writer.path) if cfg.resume else set()

    for example in tqdm(examples, desc=f"text_to_sql/{cfg.dataset.name}", unit="q"):
        if (example["example_id"], "text_to_sql", None) in completed:
            continue
        db_path = resolve_db_path(example, cfg.dataset)
        db_id = example["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = get_schema_ddl(db_path)
        result = run_text_to_sql(
            backend,
            example,
            schema_cache[db_id],
            db_path,
            template,
            max_retries=cfg.max_retries,
            max_tokens=cfg.backend.max_tokens,
            temperature=cfg.backend.temperature,
            seed=cfg.backend.seed,
        )
        writer.append(result)
        completed.add(completed_key(result))


def _exceeds_context(prompt_tokens: int, headroom: int, n_ctx: int) -> bool:
    """Whether a prompt of ``prompt_tokens`` leaves less than ``headroom`` in the window."""
    return prompt_tokens + headroom >= n_ctx


def _context_exceeded_result(
    example: DatasetExample, fmt: SerializationFormat, model_id: str
) -> PipelineResult:
    """A zeroed result recording that the serialized database cannot fit the context.

    Note: this record marks the example as completed for resume purposes — after raising
    ``N_CTX``, delete the slice file (or its ``context_exceeded`` lines) to re-run them.
    """
    result = base_result(example, pipeline="direct", fmt=fmt, model_id=model_id)
    result["error"] = "context_exceeded"
    return result
