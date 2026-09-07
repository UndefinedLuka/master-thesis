"""Measure tokenizer-specific token counts for every serialized input database.

The report isolates the serialized database payload: prompt templates, questions, chat-template
control tokens, BOS tokens, and output tokens are deliberately excluded. It measures every unique
database referenced by each benchmark dev split. Spider databases are read in full; BIRD is read
from the same materialized sampled databases used by the benchmark.

Run independently with either form::

    python -m cli.report_serialization_tokens
    thesis-report-serialization-tokens

Only one GGUF is loaded per model family because all benchmark sizes within a family share a
tokenizer. ``llama.cpp`` is opened with ``vocab_only=True`` so no inference weights or GPU are
needed. The command atomically checkpoints one detailed CSV under ``results/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Protocol, cast, get_args

from tqdm import tqdm

from config.paths import ProjectPaths
from config.settings import (
    DatasetConfig,
    DatasetName,
    default_dev_json,
    effective_db_root,
)
from datasets.loader import load_dataset, load_db_contents
from serialization.dispatch import serialize
from serialization.types import FORMATS, SerializationFormat

REPORT_FILENAME = "serialization_tokens_by_database.csv"


@dataclass(frozen=True, slots=True)
class TokenizerFamily:
    """One tokenizer shared by several parameter sizes in an experiment family."""

    key: str
    label: str
    model_glob: str
    preferred_model: str


TOKENIZER_FAMILIES: tuple[TokenizerFamily, ...] = (
    TokenizerFamily(
        key="qwen-3.5",
        label="Qwen 3.5",
        model_glob="Qwen3.5-*.gguf",
        preferred_model="Qwen3.5-0.8B-Q4_K_M.gguf",
    ),
    TokenizerFamily(
        key="ministral-3",
        label="Ministral 3",
        model_glob="Ministral-3-*.gguf",
        preferred_model="Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
    ),
    TokenizerFamily(
        key="gemma-4",
        label="Gemma 4",
        model_glob="gemma-4-*.gguf",
        preferred_model="gemma-4-E2B-it-Q4_K_M.gguf",
    ),
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
        return cls(
            family=selection.family,
            model_path=selection.model_path,
            _llm=llm,
        )

    def count_tokens(self, text: str) -> int:
        # Match the benchmark's raw-text tokenization: no synthetic BOS, special-token aware.
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
class DatasetInput:
    """Resolved database inputs for one benchmark dataset."""

    name: DatasetName
    example_count: int
    databases: tuple[DatabaseInput, ...]


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


TokenRecordKey = tuple[str, str, str, str, str]


def discover_databases(
    db_root: Path,
    database_ids: Iterable[str] | None = None,
) -> tuple[DatabaseInput, ...]:
    """Resolve conventional SQLite paths for all requested benchmark databases.

    With no explicit ids, every database directory under ``db_root`` is discovered. The CLI
    passes the unique ids referenced by the dataset dev split so unrelated Spider training
    databases are not included in the report.
    """
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
            databases.append(DatabaseInput(database_id=database_id, path=db_path))
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
        selections.append(
            TokenizerSelection(
                family=family,
                model_path=model_path,
                covered_models=tuple(sorted(covered)),
            )
        )
    return tuple(selections)


def measure_databases(
    dataset: str,
    databases: Sequence[DatabaseInput],
    formats: Sequence[SerializationFormat],
    counters: Sequence[TokenCounter],
    *,
    completed: set[TokenRecordKey] | None = None,
    on_record: Callable[[TokenRecord], None] | None = None,
    show_progress: bool = True,
) -> list[TokenRecord]:
    """Serialize and count every missing database/format/family combination."""
    completed = completed if completed is not None else set()
    records: list[TokenRecord] = []
    progress: Iterable[DatabaseInput] = tqdm(
        databases,
        desc=f"Measuring {dataset}",
        unit="db",
        disable=not show_progress,
    )
    for database in progress:
        pending_by_format: dict[SerializationFormat, list[TokenCounter]] = {
            serialization_format: [
                counter
                for counter in counters
                if _target_key(dataset, database.database_id, serialization_format, counter)
                not in completed
            ]
            for serialization_format in formats
        }
        if not any(pending_by_format.values()):
            continue

        contents = load_db_contents(database.path)
        table_count = len(contents)
        row_count = sum(len(rows) for rows in contents.values())
        for serialization_format, pending_counters in pending_by_format.items():
            if not pending_counters:
                continue
            try:
                payload = serialize(contents, serialization_format)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to serialize {dataset}/{database.database_id} as "
                    f"{serialization_format}: {exc}"
                ) from exc
            characters = len(payload)
            utf8_bytes = len(payload.encode("utf-8"))
            for counter in pending_counters:
                try:
                    token_count = counter.count_tokens(payload)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to tokenize {dataset}/{database.database_id}/"
                        f"{serialization_format} with {counter.family.label} "
                        f"({counter.model_path.name}): {exc}"
                    ) from exc
                record = TokenRecord(
                    dataset=dataset,
                    database_id=database.database_id,
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
                records.append(record)
                completed.add(_record_key(record))
                if on_record is not None:
                    on_record(record)
    return records


def build_parser(paths: ProjectPaths | None = None) -> argparse.ArgumentParser:
    """Build the standalone command-line interface."""
    project_paths = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description=(
            "Serialize every Spider/BIRD input database in every format and write exact "
            "payload token counts for each experiment tokenizer family to CSV"
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=get_args(DatasetName),
        default=None,
        help="dataset to measure; repeat for both (default: spider and bird)",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=FORMATS,
        default=None,
        help="serialization format to measure; repeat as needed (default: all six)",
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=tuple(FAMILY_BY_KEY),
        default=None,
        help="tokenizer family to measure; repeat as needed (default: all three)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=project_paths.models_dir,
        help="directory containing experiment GGUF files (default: models/)",
    )
    parser.add_argument(
        "--tokenizer-model",
        action="append",
        type=_parse_tokenizer_override,
        default=None,
        metavar="FAMILY=GGUF",
        help=(
            "override a representative tokenizer GGUF; repeat per family, e.g. "
            "qwen-3.5=models/Qwen3.5-0.8B-Q4_K_M.gguf"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "CSV directory (default: results/input_token_overhead/ for a full run; "
            "scoped runs use results/input_token_overhead/subsets/<scope>/)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="measure only the first N databases per dataset (smoke tests only)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help=(
            "ignore an existing CSV checkpoint and recompute selected counts "
            "(use after changing data or serializers)"
        ),
    )
    parser.set_defaults(resume=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Resolve inputs/tokenizers, measure every combination, and write the CSV report."""
    paths = ProjectPaths.discover()
    parser = build_parser(paths)
    args = parser.parse_args(argv)

    datasets = cast(
        tuple[DatasetName, ...],
        tuple(dict.fromkeys(args.dataset or get_args(DatasetName))),
    )
    formats = cast(
        tuple[SerializationFormat, ...],
        tuple(dict.fromkeys(args.formats or FORMATS)),
    )
    family_keys: tuple[str, ...] = tuple(dict.fromkeys(args.family or tuple(FAMILY_BY_KEY)))
    families = tuple(FAMILY_BY_KEY[key] for key in family_keys)

    overrides: dict[str, Path] = {}
    for key, model_path in args.tokenizer_model or []:
        if key in overrides:
            parser.error(f"--tokenizer-model was supplied more than once for {key}")
        if key not in family_keys:
            parser.error(f"tokenizer override for unselected family {key!r}")
        overrides[key] = model_path

    selections = resolve_tokenizer_models(args.models_dir, families, overrides)
    output_dir = args.output_dir or _default_output_dir(
        paths,
        datasets=datasets,
        formats=formats,
        family_keys=family_keys,
        limit=args.limit,
        selections=selections,
        custom_tokenizers=bool(overrides),
    )

    dataset_inputs: list[DatasetInput] = []
    for dataset in datasets:
        dev_json = default_dev_json(dataset, paths.data_dir)
        db_root = effective_db_root(dataset, paths.data_dir)
        dataset_config = DatasetConfig(name=dataset, dev_json=dev_json, db_root=db_root)
        examples = load_dataset(dataset_config)
        databases = discover_databases(db_root, {example["db_id"] for example in examples})
        if args.limit is not None:
            databases = databases[: args.limit]
        dataset_inputs.append(
            DatasetInput(
                name=dataset,
                example_count=len(examples),
                databases=databases,
            )
        )

    if args.limit is not None:
        print(
            f"*** LIMIT ACTIVE: first {args.limit} database(s) per dataset; not a full report ***"
        )
    print("Tokenizer representatives (vocabulary only; no inference weights):")
    for selection in selections:
        print(f"  {selection.family.label}: {selection.model_path.name}")
    print("Database inputs:")
    for dataset_input in dataset_inputs:
        print(
            f"  {dataset_input.name}: {len(dataset_input.databases)} databases referenced by "
            f"{dataset_input.example_count} dev examples"
        )

    report_path = output_dir / REPORT_FILENAME
    target_keys = _target_keys(dataset_inputs, formats, selections)
    if args.resume:
        records = [
            record
            for record in load_token_records(report_path)
            if _record_key(record) in target_keys
        ]
        records = list({_record_key(record): record for record in records}.values())
        if records:
            print(f"Resuming from {len(records)}/{len(target_keys)} cached token counts.")
    else:
        records = []
        write_token_report(report_path, records)
    completed = {_record_key(record) for record in records}

    def checkpoint(record: TokenRecord) -> None:
        records.append(record)
        write_token_report(report_path, records)

    with ExitStack() as stack:
        counters: list[TokenCounter] = []
        for selection in selections:
            print(
                f"Loading {selection.family.label} tokenizer from {selection.model_path.name} ..."
            )
            counter = LlamaCppTokenCounter.create(selection)
            stack.callback(counter.close)
            counters.append(counter)
        for dataset_input in dataset_inputs:
            measure_databases(
                dataset_input.name,
                dataset_input.databases,
                formats,
                counters,
                completed=completed,
                on_record=checkpoint,
            )

    records = _sort_token_records(
        records,
        dataset_inputs=dataset_inputs,
        formats=formats,
        selections=selections,
    )
    if len(records) != len(target_keys):
        raise RuntimeError(
            f"Token report is incomplete: expected {len(target_keys)} rows, found {len(records)}"
        )
    write_token_report(report_path, records)
    print(f"Wrote {len(records)} token counts to {report_path}")
    return 0


def _parse_tokenizer_override(raw: str) -> tuple[str, Path]:
    try:
        family, raw_path = raw.split("=", 1)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--tokenizer-model must use FAMILY=/path/to/model.gguf"
        ) from None
    family = family.strip()
    raw_path = raw_path.strip()
    if family not in FAMILY_BY_KEY:
        raise argparse.ArgumentTypeError(
            f"unknown tokenizer family {family!r}; valid: {', '.join(FAMILY_BY_KEY)}"
        )
    if not raw_path:
        raise argparse.ArgumentTypeError("--tokenizer-model requires a GGUF path after '='")
    return family, Path(raw_path)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def _default_output_dir(
    paths: ProjectPaths,
    *,
    datasets: Sequence[str],
    formats: Sequence[str],
    family_keys: Sequence[str],
    limit: int | None,
    selections: Sequence[TokenizerSelection],
    custom_tokenizers: bool,
) -> Path:
    root = paths.results_dir / "input_token_overhead"
    is_full_scope = (
        tuple(datasets) == tuple(get_args(DatasetName))
        and tuple(formats) == FORMATS
        and tuple(family_keys) == tuple(FAMILY_BY_KEY)
        and limit is None
        and not custom_tokenizers
    )
    if is_full_scope:
        return root

    parts = [
        "datasets-" + "-".join(datasets),
        "formats-" + ("all" if tuple(formats) == FORMATS else "-".join(formats)),
        "families-"
        + ("all" if tuple(family_keys) == tuple(FAMILY_BY_KEY) else "-".join(family_keys)),
    ]
    if limit is not None:
        parts.append(f"limit-{limit}")
    if custom_tokenizers:
        names = "\n".join(selection.model_path.name for selection in selections)
        digest = hashlib.sha256(names.encode("utf-8")).hexdigest()[:8]
        parts.append(f"custom-tokenizers-{digest}")
    return root / "subsets" / "__".join(parts)


def _record_key(record: TokenRecord) -> TokenRecordKey:
    return (
        record.dataset,
        record.database_id,
        record.format,
        record.tokenizer_family,
        record.tokenizer_model,
    )


def _target_key(
    dataset: str,
    database_id: str,
    serialization_format: str,
    counter: TokenCounter,
) -> TokenRecordKey:
    return (
        dataset,
        database_id,
        serialization_format,
        counter.family.key,
        counter.model_path.name,
    )


def _target_keys(
    dataset_inputs: Sequence[DatasetInput],
    formats: Sequence[str],
    selections: Sequence[TokenizerSelection],
) -> set[TokenRecordKey]:
    return {
        (
            dataset.name,
            database.database_id,
            serialization_format,
            selection.family.key,
            selection.model_path.name,
        )
        for dataset in dataset_inputs
        for database in dataset.databases
        for serialization_format in formats
        for selection in selections
    }


def _sort_token_records(
    records: Sequence[TokenRecord],
    *,
    dataset_inputs: Sequence[DatasetInput],
    formats: Sequence[str],
    selections: Sequence[TokenizerSelection],
) -> list[TokenRecord]:
    dataset_positions = {dataset.name: index for index, dataset in enumerate(dataset_inputs)}
    database_positions = {
        (dataset.name, database.database_id): index
        for dataset in dataset_inputs
        for index, database in enumerate(dataset.databases)
    }
    format_positions = {value: index for index, value in enumerate(formats)}
    family_positions = {selection.family.key: index for index, selection in enumerate(selections)}
    return sorted(
        records,
        key=lambda record: (
            dataset_positions[record.dataset],
            database_positions[(record.dataset, record.database_id)],
            format_positions[record.format],
            family_positions[record.tokenizer_family],
        ),
    )


def load_token_records(path: Path) -> list[TokenRecord]:
    """Load a detailed CSV checkpoint written by :func:`write_token_report`."""
    if not path.is_file():
        return []
    expected = [field.name for field in fields(TokenRecord)]
    records: list[TokenRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected:
            raise ValueError(f"Token report schema mismatch in {path}; run with --no-resume")
        for line_number, row in enumerate(reader, start=2):
            try:
                records.append(
                    TokenRecord(
                        dataset=row["dataset"],
                        database_id=row["database_id"],
                        format=row["format"],
                        tokenizer_family=row["tokenizer_family"],
                        tokenizer_label=row["tokenizer_label"],
                        tokenizer_model=row["tokenizer_model"],
                        table_count=int(row["table_count"]),
                        row_count=int(row["row_count"]),
                        characters=int(row["characters"]),
                        utf8_bytes=int(row["utf8_bytes"]),
                        tokens=int(row["tokens"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid token report row {line_number} in {path}; run with --no-resume: {exc}"
                ) from exc
    return records


def write_token_report(path: Path, records: Sequence[TokenRecord]) -> None:
    """Atomically persist the CSV, including its header when it has no records yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    fieldnames = [field.name for field in fields(TokenRecord)]
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
