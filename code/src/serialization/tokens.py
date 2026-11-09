"""Measure serialized database payloads in memory using vocabulary-only GGUF tokenizers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

from config.paths import ProjectPaths
from config.settings import DatasetConfig, DatasetName, default_dev_json, effective_db_root
from datasets.loader import load_dataset, load_db_contents
from serialization.dispatch import serialize
from serialization.types import FORMATS, DatabaseContents, SerializationFormat


@dataclass(frozen=True, slots=True)
class TokenizerFamily:
    """One tokenizer shared by several parameter sizes in an experiment family."""

    key: str
    label: str
    model_glob: str
    preferred_model: str


TOKENIZER_FAMILIES: tuple[TokenizerFamily, ...] = (
    TokenizerFamily("qwen-3.5", "Qwen 3.5", "Qwen3.5-*.gguf", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    TokenizerFamily(
        "ministral-3",
        "Ministral 3",
        "Ministral-3-*.gguf",
        "Ministral-3-8B-Instruct-2512-Q4_K_M.gguf",
    ),
    TokenizerFamily("gemma-4", "Gemma 4", "gemma-4-*.gguf", "gemma-4-E2B-it-Q4_K_M.gguf"),
)
FAMILY_BY_KEY = {family.key: family for family in TOKENIZER_FAMILIES}


@dataclass(frozen=True, slots=True)
class TokenizerSelection:
    """The representative GGUF chosen for a tokenizer family."""

    family: TokenizerFamily
    model_path: Path
    covered_models: tuple[str, ...]


class TokenCounter(Protocol):
    """Minimal tokenizer interface used by the measurement loop."""

    family: TokenizerFamily
    model_path: Path

    def count_tokens(self, text: str) -> int: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class LlamaCppTokenCounter:
    """Tokenizer-only GGUF wrapper; it never loads model tensors for inference."""

    family: TokenizerFamily
    model_path: Path
    _llm: Any

    @classmethod
    def create(cls, selection: TokenizerSelection) -> LlamaCppTokenCounter:
        from llama_cpp import Llama

        llm = Llama(
            model_path=str(selection.model_path),
            vocab_only=True,
            n_gpu_layers=0,
            verbose=False,
        )
        return cls(selection.family, selection.model_path, llm)

    def count_tokens(self, text: str) -> int:
        # Match raw-text benchmark tokenization: no synthetic BOS, special-token aware.
        return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))

    def close(self) -> None:
        close = getattr(self._llm, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True, slots=True)
class DatabaseInput:
    """A database identifier and its conventional SQLite path."""

    database_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class TokenRecord:
    """Token count for one database, serialization format, and tokenizer family."""

    dataset: str
    database_id: str
    format: str
    tokenizer_family: str
    tokenizer_label: str
    tokenizer_model: str
    table_count: int
    row_count: int
    characters: int
    utf8_bytes: int
    tokens: int


def discover_databases(
    db_root: Path,
    database_ids: Iterable[str] | None = None,
) -> tuple[DatabaseInput, ...]:
    """Resolve SQLite paths, optionally restricted to the IDs referenced by a dev split."""
    if not db_root.is_dir():
        raise FileNotFoundError(f"Database root not found: {db_root}")
    ids = (
        sorted(set(database_ids))
        if database_ids is not None
        else sorted(path.name for path in db_root.iterdir() if path.is_dir())
    )
    databases: list[DatabaseInput] = []
    missing: list[Path] = []
    for database_id in ids:
        db_path = db_root / database_id / f"{database_id}.sqlite"
        if db_path.is_file():
            databases.append(DatabaseInput(database_id, db_path))
        else:
            missing.append(db_path)
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        suffix = f" (and {len(missing) - 3} more)" if len(missing) > 3 else ""
        raise FileNotFoundError(
            f"{len(missing)} requested database(s) lack the expected SQLite file: {preview}{suffix}"
        )
    if not databases:
        raise FileNotFoundError(f"No <db_id>/<db_id>.sqlite databases found under {db_root}")
    return tuple(databases)


def resolve_tokenizer_models(
    models_dir: Path,
    families: Sequence[TokenizerFamily],
    overrides: Mapping[str, Path] | None = None,
) -> tuple[TokenizerSelection, ...]:
    """Choose one available GGUF per family, preferring its smallest experiment model."""
    overrides = overrides or {}
    selections: list[TokenizerSelection] = []
    for family in families:
        available = (
            tuple(sorted(models_dir.glob(family.model_glob), key=lambda path: path.name))
            if models_dir.is_dir()
            else ()
        )
        override = overrides.get(family.key)
        if override is not None:
            model_path = override.expanduser().resolve()
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Tokenizer model override for {family.label} not found: {model_path}"
                )
        else:
            preferred = models_dir / family.preferred_model
            if preferred.is_file():
                model_path = preferred.resolve()
            elif available:
                model_path = min(
                    available,
                    key=lambda path: (path.stat().st_size, path.name),
                ).resolve()
            else:
                raise FileNotFoundError(
                    f"No {family.label} GGUF found in {models_dir} "
                    f"(expected pattern {family.model_glob!r}). Download the experiment models or "
                    f"pass --tokenizer-model {family.key}=/path/to/model.gguf."
                )
        covered = {path.name for path in available}
        covered.add(model_path.name)
        selections.append(TokenizerSelection(family, model_path, tuple(sorted(covered))))
    return tuple(selections)


def measure_contents(
    dataset: str,
    database_id: str,
    contents: DatabaseContents,
    formats: Sequence[SerializationFormat],
    counters: Sequence[TokenCounter],
) -> list[TokenRecord]:
    """Count each payload once per tokenizer, retaining only measurements in memory."""
    records: list[TokenRecord] = []
    table_count = len(contents)
    row_count = sum(len(rows) for rows in contents.values())
    for serialization_format in formats:
        try:
            payload = serialize(contents, serialization_format)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to serialize {dataset}/{database_id} as {serialization_format}: {exc}"
            ) from exc
        characters = len(payload)
        utf8_bytes = len(payload.encode("utf-8"))
        for counter in counters:
            try:
                token_count = counter.count_tokens(payload)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to tokenize {dataset}/{database_id}/{serialization_format} "
                    f"with {counter.family.label} ({counter.model_path.name}): {exc}"
                ) from exc
            records.append(
                TokenRecord(
                    dataset=dataset,
                    database_id=database_id,
                    format=serialization_format,
                    tokenizer_family=counter.family.key,
                    tokenizer_label=counter.family.label,
                    tokenizer_model=counter.model_path.name,
                    table_count=table_count,
                    row_count=row_count,
                    characters=characters,
                    utf8_bytes=utf8_bytes,
                    tokens=token_count,
                )
            )
    return records


def measure_databases(
    dataset: str,
    databases: Sequence[DatabaseInput],
    formats: Sequence[SerializationFormat],
    counters: Sequence[TokenCounter],
    *,
    show_progress: bool = True,
) -> list[TokenRecord]:
    """Read one database at a time and return its payload measurements without caching."""
    records: list[TokenRecord] = []
    for database in tqdm(
        databases,
        desc=f"Measuring {dataset}",
        unit="db",
        disable=not show_progress,
    ):
        records.extend(
            measure_contents(
                dataset,
                database.database_id,
                load_db_contents(database.path),
                formats,
                counters,
            )
        )
    return records


def measure_serialization_tokens(
    paths: ProjectPaths | None = None,
    *,
    datasets: Sequence[DatasetName] = ("spider", "bird"),
    formats: Sequence[SerializationFormat] = FORMATS,
    families: Sequence[TokenizerFamily] = TOKENIZER_FAMILIES,
    models_dir: Path | None = None,
    overrides: Mapping[str, Path] | None = None,
    limit: int | None = None,
    show_progress: bool = True,
) -> list[TokenRecord]:
    """Measure dev-split databases with one tokenizer per family; write no files.

    Spider uses full databases and BIRD uses the benchmark's materialized samples.
    Prompt templates, questions, chat-template tokens, and output tokens are excluded.
    Tokenizers are released on completion or failure; records belong to the caller.
    """
    if limit is not None and limit <= 0:
        raise ValueError("Database limit must be positive")
    paths = paths or ProjectPaths.discover()
    selections = resolve_tokenizer_models(models_dir or paths.models_dir, families, overrides)
    inputs = {}
    for dataset in dict.fromkeys(datasets):
        root = effective_db_root(dataset, paths.data_dir)
        config = DatasetConfig(dataset, default_dev_json(dataset, paths.data_dir), root)
        examples = load_dataset(config)
        inputs[dataset] = discover_databases(root, {example["db_id"] for example in examples})
        if limit is not None:
            inputs[dataset] = inputs[dataset][:limit]

    records: list[TokenRecord] = []
    with ExitStack() as stack:
        counters = []
        for selection in selections:
            counter = LlamaCppTokenCounter.create(selection)
            stack.callback(counter.close)
            counters.append(counter)
        for dataset, databases in inputs.items():
            records.extend(
                measure_databases(
                    dataset,
                    databases,
                    formats,
                    counters,
                    show_progress=show_progress,
                )
            )
    return records
