"""Normalized Exact Match via Transformers' established SQuAD metric.

The SQuAD normalization lowercases text, removes punctuation and English articles, and
collapses all whitespace before exact comparison. This makes formatting-only differences such
as capitalization, comma separators, and newlines score as equal while retaining a strict
binary result after normalization.

The optional dependency is imported lazily so the package remains importable without the
``[metrics]`` extra.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import cast


def normalized_exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 iff SQuAD-normalized prediction and reference are identical, else 0.0."""
    try:
        module = import_module("transformers.data.metrics.squad_metrics")
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
        message = "Normalized Exact Match requires the optional 'metrics' dependencies. "
        message += "Install them with: uv sync --extra metrics"
        raise ModuleNotFoundError(message) from exc

    compute_exact = cast(Callable[[str, str], int], module.compute_exact)
    return float(compute_exact(reference, prediction))
