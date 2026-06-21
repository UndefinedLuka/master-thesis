"""Typed loader for ``viz_config.toml`` — the single file controlling all figures.

Mirrors the style of :mod:`benchmark.config`: load once, pass the frozen object
around. Everything visual (theme, palette, which figures to draw) is data here,
so restyling never requires touching plotting code.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from benchmark import paths


@dataclass(frozen=True)
class Theme:
    style: str = "whitegrid"
    context: str = "talk"
    font_family: str = "DejaVu Sans"
    formats: tuple[str, ...] = ("pdf", "png")
    dpi: int = 200
    transparent: bool = False
    fig_width: float = 11.0
    fig_height: float = 6.0
    despine: bool = True
    annotate_bars: bool = True
    value_format: str = "{:.2f}"


@dataclass(frozen=True)
class Palette:
    palette_name: str = ""
    sequence: tuple[str, ...] = ()
    by_format: dict[str, str] = field(default_factory=dict)
    by_pipeline: dict[str, str] = field(default_factory=dict)
    semantic: dict[str, str] = field(default_factory=dict)

    def color_for_approach(self, approach: str, fallback: str = "#8C8C8C") -> str:
        return self.by_format.get(approach, fallback)

    def hues(self, n: int) -> list[str]:
        """Return ``n`` categorical colors (named palette or explicit sequence)."""
        import seaborn as sns

        if self.palette_name:
            return sns.color_palette(self.palette_name, n).as_hex()
        if self.sequence:
            seq = list(self.sequence)
            return [seq[i % len(seq)] for i in range(n)]
        return sns.color_palette(n_colors=n).as_hex()


@dataclass(frozen=True)
class VizConfig:
    theme: Theme
    palette: Palette
    figures: dict[str, bool]
    primary_metric: str
    model_params_b: dict[str, float]
    approach_order: tuple[str, ...]

    def enabled(self, name: str) -> bool:
        return bool(self.figures.get(name, False))


def load_viz_config(path: str | Path | None = None) -> VizConfig:
    cfg_path = Path(path) if path else paths.bench_root() / "viz_config.toml"
    with open(cfg_path, "rb") as f:
        raw = tomllib.load(f)

    t = raw.get("theme", {})
    theme = Theme(
        style=t.get("style", "whitegrid"),
        context=t.get("context", "talk"),
        font_family=t.get("font_family", "DejaVu Sans"),
        formats=tuple(t.get("formats", ["pdf", "png"])),
        dpi=int(t.get("dpi", 200)),
        transparent=bool(t.get("transparent", False)),
        fig_width=float(t.get("fig_width", 11.0)),
        fig_height=float(t.get("fig_height", 6.0)),
        despine=bool(t.get("despine", True)),
        annotate_bars=bool(t.get("annotate_bars", True)),
        value_format=t.get("value_format", "{:.2f}"),
    )

    p = raw.get("palette", {})
    palette = Palette(
        palette_name=p.get("palette_name", ""),
        sequence=tuple(p.get("sequence", [])),
        by_format=dict(p.get("by_format", {})),
        by_pipeline=dict(p.get("by_pipeline", {})),
        semantic=dict(p.get("semantic", {})),
    )

    figs = raw.get("figures", {})
    primary_metric = str(figs.get("primary_metric", "bertscore"))
    figures = {k: bool(v) for k, v in figs.items() if isinstance(v, bool)}

    labels = raw.get("labels", {})
    return VizConfig(
        theme=theme,
        palette=palette,
        figures=figures,
        primary_metric=primary_metric,
        model_params_b={k: float(v) for k, v in raw.get("model_params_b", {}).items()},
        approach_order=tuple(labels.get("approach_order", [])),
    )
