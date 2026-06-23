"""Closed-set enumerations the code branches on.

Using ``Enum`` (rather than bare strings or ``Literal``) for the domain's closed
sets means an invalid value cannot be constructed silently: ``SerializationFormat("jsom")``
raises immediately. Every enum derives from ``str`` so values serialize cleanly
to JSON via their ``.value`` while still being comparable as enum members.
"""

from __future__ import annotations

from enum import StrEnum


class FormatFamily(StrEnum):
    """Grouping of serialization formats used for analysis (see ch-03 §formats).

    - ``STRUCTURAL`` formats can represent deeply nested data.
    - ``POSITIONAL`` formats rely on column order and cannot nest.
    - ``TOKEN_OPTIMIZED`` formats minimize token overhead but are under-represented
      in LLM training corpora.
    """

    STRUCTURAL = "structural"
    POSITIONAL = "positional"
    TOKEN_OPTIMIZED = "token_optimized"


class SerializationFormat(StrEnum):
    """The six serialization formats compared by the benchmark."""

    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    CSV = "csv"
    MARKDOWN = "markdown"
    TOON = "toon"

    @property
    def family(self) -> FormatFamily:
        return _FORMAT_FAMILY[self]

    @property
    def file_extension(self) -> str:
        """Conventional file extension for this format (handy for filenames)."""
        return _FORMAT_EXTENSION[self]


_FORMAT_FAMILY: dict[SerializationFormat, FormatFamily] = {
    SerializationFormat.JSON: FormatFamily.STRUCTURAL,
    SerializationFormat.XML: FormatFamily.STRUCTURAL,
    SerializationFormat.YAML: FormatFamily.STRUCTURAL,
    SerializationFormat.CSV: FormatFamily.POSITIONAL,
    SerializationFormat.MARKDOWN: FormatFamily.POSITIONAL,
    SerializationFormat.TOON: FormatFamily.TOKEN_OPTIMIZED,
}

_FORMAT_EXTENSION: dict[SerializationFormat, str] = {
    SerializationFormat.JSON: "json",
    SerializationFormat.XML: "xml",
    SerializationFormat.YAML: "yaml",
    SerializationFormat.CSV: "csv",
    SerializationFormat.MARKDOWN: "md",
    SerializationFormat.TOON: "toon",
}


class PipelineKind(StrEnum):
    """The two information-retrieval pipelines under comparison."""

    DIRECT = "direct"
    TEXT_TO_SQL = "text_to_sql"


class DatasetKind(StrEnum):
    """Supported Text-to-SQL datasets."""

    SPIDER = "spider"
    BIRD = "bird"


class Backend(StrEnum):
    """LLM inference backends. Only ``OLLAMA`` is implemented; the others are
    placeholders that the factory rejects with a clear message until wired up.
    """

    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    LLAMA_SERVER = "llama_server"


class SqlAttemptOutcome(StrEnum):
    """Outcome of a single Text-to-SQL generate-and-execute attempt."""

    EXECUTED_OK = "executed_ok"
    EXECUTION_ERROR = "execution_error"
    GENERATION_ERROR = "generation_error"


class TextToSqlStatus(StrEnum):
    """Terminal status of the full Text-to-SQL attempt sequence."""

    SUCCESS = "success"
    FAILED_AFTER_RETRIES = "failed_after_retries"
    FAILED_GENERATION = "failed_generation"
