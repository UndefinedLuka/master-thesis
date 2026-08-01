"""Prompt template loading and rendering.

Templates are Markdown files containing a single ``<data>`` placeholder (the serialized
database for Direct Context Inference, or the schema DDL for Text-to-SQL).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

PromptKind = Literal["direct", "sql"]

_FILENAMES: dict[PromptKind, str] = {
    "direct": "direct_answer.md",
    "sql": "sql_answer.md",
}

_PLACEHOLDER = "<data>"


def load_prompt(kind: PromptKind, prompts_dir: Path | None = None) -> str:
    """Read the raw prompt template for ``kind``.

    Defaults to the templates shipped alongside this module.
    """
    base = prompts_dir or Path(__file__).parent
    return (base / _FILENAMES[kind]).read_text(encoding="utf-8")


def render_prompt(template: str, data: str) -> str:
    """Substitute the serialized data/schema into the template's ``<data>`` placeholder."""
    return template.replace(_PLACEHOLDER, data)


__all__ = ["PromptKind", "load_prompt", "render_prompt"]
