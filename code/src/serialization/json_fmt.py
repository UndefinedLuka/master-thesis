"""JSON serialization."""

from __future__ import annotations

import json

from serialization.types import DatabaseContents


def to_json(contents: DatabaseContents) -> str:
    """Indented JSON, preserving non-ASCII characters."""
    return json.dumps(contents, indent=2, ensure_ascii=False)
