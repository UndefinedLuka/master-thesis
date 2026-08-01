"""TOON (Token-Oriented Object Notation) serialization."""

from __future__ import annotations

import toon

from serialization.types import DatabaseContents


def to_toon(contents: DatabaseContents) -> str:
    """Encode via python-toon, which compresses redundant keys across records."""
    return toon.encode(contents)
