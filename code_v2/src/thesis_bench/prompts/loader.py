"""Typed prompt builder.

Templates use ``{{placeholder}}`` markers. The loader knows the exact set of
placeholders each template requires and fails loudly if the caller supplies the
wrong set — a missing or misspelled placeholder is a programming error we want to
surface immediately, not silently leave an empty hole in the prompt (v1 used a
bare ``str.replace`` with no validation).

Templates are cached after first read since they are small and immutable.
"""

from __future__ import annotations

import re
from enum import StrEnum

from ..errors import ThesisBenchError
from ..paths import templates_dir

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


class PromptTemplate(StrEnum):
    """The available system-prompt templates."""

    DIRECT_ANSWER = "direct_answer.md"
    SQL_ANSWER = "sql_answer.md"
    SQL_RETRY = "sql_retry.md"


# Required placeholders per template — the contract the builder enforces.
_REQUIRED: dict[PromptTemplate, frozenset[str]] = {
    PromptTemplate.DIRECT_ANSWER: frozenset({"data"}),
    PromptTemplate.SQL_ANSWER: frozenset({"schema"}),
    PromptTemplate.SQL_RETRY: frozenset({"schema", "previous_sql", "error"}),
}

_cache: dict[PromptTemplate, str] = {}


def _load_raw(template: PromptTemplate) -> str:
    if template not in _cache:
        path = templates_dir / template.value
        _cache[template] = path.read_text(encoding="utf-8")
    return _cache[template]


def build_prompt(template: PromptTemplate, **values: str) -> str:
    """Render ``template`` with ``values``, validating the placeholder set.

    Raises ``ThesisBenchError`` if the supplied keys don't exactly match the
    template's declared placeholders, or if the template body and declaration
    disagree (guards against template edits that drift from this module).
    """
    raw = _load_raw(template)
    declared = _REQUIRED[template]
    in_body = frozenset(_PLACEHOLDER_RE.findall(raw))

    if in_body != declared:
        raise ThesisBenchError(
            f"Template {template.value!r} body placeholders {set(in_body)} do not "
            f"match declared {set(declared)}; update prompts/loader.py."
        )

    supplied = frozenset(values)
    if supplied != declared:
        missing = declared - supplied
        extra = supplied - declared
        raise ThesisBenchError(
            f"build_prompt({template.value!r}) placeholder mismatch: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )

    def replace(match: re.Match[str]) -> str:
        return values[match.group(1)]

    return _PLACEHOLDER_RE.sub(replace, raw)
