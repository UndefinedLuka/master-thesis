"""Tests for prompt template loading and <data> placeholder rendering."""

from __future__ import annotations

from pathlib import Path

from prompts import load_prompt, render_prompt


def test_load_shipped_prompts_contain_placeholder() -> None:
    for kind in ("direct", "sql"):
        template = load_prompt(kind)  # type: ignore[arg-type]
        assert "<data>" in template, f"{kind} template is missing the <data> placeholder"


def test_load_prompt_from_custom_dir(tmp_path: Path) -> None:
    (tmp_path / "direct_answer.md").write_text("Answer using: <data>", encoding="utf-8")
    assert load_prompt("direct", prompts_dir=tmp_path) == "Answer using: <data>"


def test_render_prompt_substitutes_data() -> None:
    rendered = render_prompt("schema:\n<data>\nend", '{"t": []}')
    assert rendered == 'schema:\n{"t": []}\nend'
    assert "<data>" not in rendered
