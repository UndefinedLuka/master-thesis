"""Small helpers for rendering numeric values with siunitx in LaTeX tables."""

from __future__ import annotations


def num(value: object, decimals: int | None = None) -> str:
    """Render a numeric value as a canonical siunitx ``\\num`` argument."""
    raw = str(value) if decimals is None else f"{float(value):.{decimals}f}"
    return rf"\num{{{raw}}}"


def percent(value: object, decimals: int | None = None) -> str:
    """Render a percentage using siunitx's percent unit."""
    raw = str(value) if decimals is None else f"{float(value):.{decimals}f}"
    return rf"\qty{{{raw}}}{{\percent}}"
