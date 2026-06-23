"""Typed exception hierarchy.

Pipelines and the orchestrator catch these to populate structured result fields
rather than stringifying arbitrary exceptions. A single base, ``ThesisBenchError``,
lets callers catch everything this package raises without swallowing unrelated
errors (``KeyboardInterrupt``, programming bugs, etc.).
"""

from __future__ import annotations


class ThesisBenchError(Exception):
    """Base class for all errors raised by thesis_bench."""


class ConfigError(ThesisBenchError):
    """Invalid or inconsistent configuration."""


class LLMError(ThesisBenchError):
    """The LLM backend failed to produce a response (network, daemon, model)."""


class SqlExecutionError(ThesisBenchError):
    """A SQL query failed to execute against the database.

    The original database error message is preserved in ``db_message`` so it can
    be fed back to the model during the Text-to-SQL error-correction loop.
    """

    def __init__(self, db_message: str) -> None:
        super().__init__(db_message)
        self.db_message = db_message


class SqlExtractionError(ThesisBenchError):
    """The model's response did not contain an extractable SQL statement."""


class SerializationError(ThesisBenchError):
    """A database snapshot could not be serialized into the requested format."""


class SamplingError(ThesisBenchError):
    """Deterministic row sampling failed (e.g. missing table, bad seed config)."""


class DatasetError(ThesisBenchError):
    """A dataset could not be loaded or is malformed."""
