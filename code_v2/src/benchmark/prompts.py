"""Prompt assembly: inject serialized data/schema and the optional BIRD evidence.

Templates live in ``prompts_md/`` and contain two placeholders: ``<data>`` (the
serialized DB or SQL schema) and ``<evidence>`` (replaced by a hint block for
BIRD, or removed entirely for Spider so its prompts are unchanged).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts_md"


@lru_cache(maxsize=None)
def load_template(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


def _apply_evidence(template: str, evidence: str | None) -> str:
    if evidence:
        block = f"\n# EXTERNAL KNOWLEDGE\nUse the following hint to interpret the data:\n{evidence}\n"
    else:
        block = ""
    return template.replace("<evidence>", block)


def build_direct_prompt(data: str, evidence: str | None = None) -> str:
    """System prompt for the Direct Context Inference pipeline."""
    return _apply_evidence(load_template("direct_answer.md"), evidence).replace("<data>", data)


def build_sql_prompt(schema: str, evidence: str | None = None) -> str:
    """System prompt for the Text-to-SQL pipeline."""
    return _apply_evidence(load_template("sql_answer.md"), evidence).replace("<data>", schema)
