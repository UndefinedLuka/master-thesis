"""YAML serialization."""

from __future__ import annotations

import yaml

from serialization.types import DatabaseContents


def to_yaml(contents: DatabaseContents) -> str:
    """Block-style YAML, keeping insertion order and non-ASCII characters."""
    return yaml.safe_dump(
        contents,
        indent=2,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
