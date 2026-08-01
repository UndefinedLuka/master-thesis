"""XML serialization."""

from __future__ import annotations

from dict2xml import dict2xml

from serialization.types import DatabaseContents


def to_xml(contents: DatabaseContents) -> str:
    """Wrap the database in a ``<root>`` element; tab-indented (matches v1)."""
    return dict2xml(contents, wrap="root", indent="\t")
