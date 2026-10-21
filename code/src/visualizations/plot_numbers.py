"""siunitx number rendering and PGF PDF output for visualization figures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


_NUMBER_FORMATTING_TEX = (
    Path(__file__).resolve().parents[3] / "thesis" / "style" / "number_formatting.tex"
)


def number(value: object, decimals: int | None = None) -> str:
    """Return raw siunitx markup, with a canonical decimal-dot input."""
    if decimals is None:
        raw = f"{float(value):.12g}"
    else:
        raw = f"{float(value):.{decimals}f}"
    return rf"\num{{{raw}}}"


def quantity(value: object, unit: str, decimals: int | None = None) -> str:
    """Return a value and siunitx unit (for example, ``quantity(95, r'\\percent')``)."""
    if decimals is None:
        raw = f"{float(value):.12g}"
    else:
        raw = f"{float(value):.{decimals}f}"
    return rf"\qty{{{raw}}}{{{unit}}}"


def number_formatter(
    decimals: int | None = None,
):
    """Build a Matplotlib tick formatter that delegates display to siunitx."""
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda value, _position: number(value, decimals))


def _pgf_rc_context():
    import matplotlib as mpl

    if not _NUMBER_FORMATTING_TEX.is_file():
        raise FileNotFoundError(
            f"Shared siunitx configuration not found: {_NUMBER_FORMATTING_TEX}"
        )
    return mpl.rc_context(
        {
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": rf"\input{{{_NUMBER_FORMATTING_TEX.as_posix()}}}",
        }
    )


def tight_layout(figure: Figure, **kwargs) -> None:
    """Measure layout with the PGF/LaTeX text renderer, then restore the original canvas."""
    from matplotlib.backends.backend_pgf import FigureCanvasPgf

    original_canvas = figure.canvas
    try:
        with _pgf_rc_context():
            FigureCanvasPgf(figure)
            figure.tight_layout(**kwargs)
    finally:
        figure.set_canvas(original_canvas)


def save_pgf_figure(figure: Figure, path: Path) -> None:
    """Save a PDF through PGF/LaTeX without enabling usetex for normal canvas rendering."""
    with _pgf_rc_context():
        figure.savefig(path, backend="pgf", dpi=300, bbox_inches="tight", facecolor="white")
