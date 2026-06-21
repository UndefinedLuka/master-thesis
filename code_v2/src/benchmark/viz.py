"""Figure suite for the benchmark results.

seaborn + matplotlib, fully driven by :class:`benchmark.viz_config.VizConfig`, so
every color, style and toggle lives in ``viz_config.toml``. Figures are written
as vector PDF (for the thesis) and PNG (for quick viewing) into ``results/figures``.

The single derived dimension used throughout is **approach**: the serialization
format for the direct pipeline, or the literal ``"sql"`` for Text-to-SQL. This
lets a single x-axis compare all seven retrieval strategies side by side.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from benchmark.viz_config import VizConfig  # noqa: E402

METRIC_LABELS = {"bertscore": "BERTScore F1", "rouge_l": "ROUGE-L F1"}


class Plotter:
    """Applies the theme and renders the enabled figures."""

    def __init__(self, cfg: VizConfig, out_dir: Path) -> None:
        self.cfg = cfg
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._apply_theme()

    # -- theme / palette helpers --------------------------------------------
    def _apply_theme(self) -> None:
        t = self.cfg.theme
        sns.set_theme(style=t.style, context=t.context, font=t.font_family)
        plt.rcParams["figure.dpi"] = t.dpi
        plt.rcParams["savefig.bbox"] = "tight"
        plt.rcParams["axes.titleweight"] = "bold"

    def _figsize(self, w_scale: float = 1.0, h_scale: float = 1.0) -> tuple[float, float]:
        return (self.cfg.theme.fig_width * w_scale, self.cfg.theme.fig_height * h_scale)

    def _approach_order(self, present: list[str]) -> list[str]:
        order = [a for a in self.cfg.approach_order if a in present]
        order += [a for a in present if a not in order]
        return order

    def _approach_palette(self, approaches: list[str]) -> dict[str, str]:
        hues = self.cfg.palette.hues(len(approaches))
        return {
            a: self.cfg.palette.by_format.get(a, hues[i]) for i, a in enumerate(approaches)
        }

    def _annotate(self, ax: plt.Axes) -> None:
        if not self.cfg.theme.annotate_bars:
            return
        fmt = self.cfg.theme.value_format
        for container in ax.containers:
            labels = [fmt.format(v) if v else "" for v in container.datavalues]
            ax.bar_label(container, labels=labels, fontsize=8, padding=2)

    def _finish(self, fig: plt.Figure, name: str) -> None:
        if self.cfg.theme.despine:
            sns.despine(fig=fig)
        fig.tight_layout()
        for ext in self.cfg.theme.formats:
            fig.savefig(
                self.out_dir / f"{name}.{ext}",
                dpi=self.cfg.theme.dpi,
                transparent=self.cfg.theme.transparent,
            )
        plt.close(fig)

    # -- figures ------------------------------------------------------------
    def accuracy_overview(self, df: pd.DataFrame) -> None:
        """BERTScore and ROUGE-L by approach, grouped by model, per dataset."""
        for dataset, sub in df.groupby("dataset", sort=False):
            present = list(sub["approach"].unique())
            order = self._approach_order(present)
            fig, axes = plt.subplots(1, 2, figsize=self._figsize(1.25, 1.0), sharex=True)
            for ax, metric in zip(axes, ("bertscore", "rouge_l")):
                sns.barplot(
                    data=sub,
                    x="approach",
                    y=metric,
                    hue="model",
                    order=order,
                    errorbar=("ci", 95),
                    palette=self.cfg.palette.hues(sub["model"].nunique()),
                    ax=ax,
                )
                ax.set_title(METRIC_LABELS[metric])
                ax.set_xlabel("")
                ax.set_ylabel(METRIC_LABELS[metric])
                ax.set_ylim(0, 1)
                ax.tick_params(axis="x", rotation=30)
                self._annotate(ax)
                if ax is axes[0]:
                    ax.legend(title="model", fontsize=8, title_fontsize=9)
                else:
                    ax.get_legend().remove()
            fig.suptitle(f"Retrieval accuracy by approach — {dataset}", fontweight="bold")
            self._finish(fig, f"accuracy_overview_{dataset}")

    def token_consumption(self, df: pd.DataFrame) -> None:
        """Mean input vs output tokens by approach (averaged over models)."""
        pal = self.cfg.palette.semantic
        long = df.melt(
            id_vars=["dataset", "approach"],
            value_vars=["input_tokens", "output_tokens"],
            var_name="token_type",
            value_name="tokens",
        )
        long["token_type"] = long["token_type"].map(
            {"input_tokens": "input", "output_tokens": "output"}
        )
        for dataset, sub in long.groupby("dataset", sort=False):
            order = self._approach_order(list(sub["approach"].unique()))
            fig, ax = plt.subplots(figsize=self._figsize())
            sns.barplot(
                data=sub,
                x="approach",
                y="tokens",
                hue="token_type",
                order=order,
                hue_order=["input", "output"],
                palette=[pal.get("input", "#4C72B0"), pal.get("output", "#DD8452")],
                errorbar=None,
                ax=ax,
            )
            ax.set_title(f"Token consumption by approach — {dataset}")
            ax.set_xlabel("")
            ax.set_ylabel("mean tokens")
            ax.tick_params(axis="x", rotation=30)
            ax.legend(title="", fontsize=9)
            self._annotate(ax)
            self._finish(fig, f"token_consumption_{dataset}")

    def latency(self, df: pd.DataFrame) -> None:
        """TTFT and total generation time by approach, per dataset."""
        for dataset, sub in df.groupby("dataset", sort=False):
            order = self._approach_order(list(sub["approach"].unique()))
            pal = self._approach_palette(order)
            fig, axes = plt.subplots(1, 2, figsize=self._figsize(1.25, 1.0))
            for ax, col, title in (
                (axes[0], "ttft_s", "Time to first token (s)"),
                (axes[1], "total_generation_s", "Total generation time (s)"),
            ):
                sns.barplot(
                    data=sub, x="approach", y=col, order=order,
                    hue="approach", hue_order=order, palette=pal, legend=False,
                    errorbar=("ci", 95), ax=ax,
                )
                ax.set_title(title)
                ax.set_xlabel("")
                ax.set_ylabel("seconds")
                ax.tick_params(axis="x", rotation=30)
                self._annotate(ax)
            fig.suptitle(f"Latency by approach — {dataset}", fontweight="bold")
            self._finish(fig, f"latency_{dataset}")

    def sql_diagnostics(self, df: pd.DataFrame) -> None:
        """Text-to-SQL only: retry distribution + success/match/failure rates."""
        sub = df[df["pipeline"] == "sql"]
        if sub.empty:
            return
        for dataset, dsub in sub.groupby("dataset", sort=False):
            fig, axes = plt.subplots(1, 2, figsize=self._figsize(1.25, 1.0))

            # Left: retry-count distribution per model.
            sns.countplot(
                data=dsub, x="retry_count", hue="model",
                palette=self.cfg.palette.hues(dsub["model"].nunique()), ax=axes[0],
            )
            axes[0].set_title("Self-correction retries")
            axes[0].set_xlabel("retries to success (cap reached = failure)")
            axes[0].set_ylabel("questions")
            axes[0].legend(title="model", fontsize=8)

            # Right: execution accuracy / result-set match / failure rate.
            rates = (
                dsub.groupby("model")[
                    ["execution_accuracy", "result_set_match", "definitive_failure"]
                ]
                .mean()
                .reset_index()
                .melt(id_vars="model", var_name="metric", value_name="rate")
            )
            rates["metric"] = rates["metric"].map(
                {
                    "execution_accuracy": "exec. accuracy",
                    "result_set_match": "result match",
                    "definitive_failure": "failure",
                }
            )
            sns.barplot(
                data=rates, x="metric", y="rate", hue="model",
                palette=self.cfg.palette.hues(dsub["model"].nunique()), ax=axes[1],
            )
            axes[1].set_title("Query outcomes")
            axes[1].set_xlabel("")
            axes[1].set_ylabel("rate")
            axes[1].set_ylim(0, 1)
            self._annotate(axes[1])
            axes[1].legend(title="model", fontsize=8)

            fig.suptitle(f"Text-to-SQL diagnostics — {dataset}", fontweight="bold")
            self._finish(fig, f"sql_diagnostics_{dataset}")

    def parameter_scale(self, df: pd.DataFrame) -> None:
        """Primary accuracy metric vs model parameter count, line per approach."""
        params = self.cfg.model_params_b
        if not params:
            return
        metric = self.cfg.primary_metric
        d = df.copy()
        d["params_b"] = d["model"].map(params)
        d = d.dropna(subset=["params_b"])
        if d.empty:
            return
        for dataset, sub in d.groupby("dataset", sort=False):
            order = self._approach_order(list(sub["approach"].unique()))
            agg = (
                sub.groupby(["approach", "params_b"])[metric].mean().reset_index()
            )
            pal = self._approach_palette(order)
            fig, ax = plt.subplots(figsize=self._figsize())
            sns.lineplot(
                data=agg, x="params_b", y=metric, hue="approach",
                hue_order=order, palette=pal, marker="o", markersize=9,
                linewidth=2.5, ax=ax,
            )
            ax.set_title(f"Accuracy vs. model scale — {dataset}")
            ax.set_xlabel("parameters (billions)")
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_ylim(0, 1)
            ax.legend(title="approach", fontsize=8, ncol=2)
            self._finish(fig, f"parameter_scale_{dataset}")

    def tradeoff_scatter(self, df: pd.DataFrame) -> None:
        """Cost/quality frontier: primary metric vs mean output tokens per approach."""
        metric = self.cfg.primary_metric
        for dataset, sub in df.groupby("dataset", sort=False):
            agg = (
                sub.groupby(["approach", "model"])
                .agg(tokens=("output_tokens", "mean"), score=(metric, "mean"))
                .reset_index()
            )
            order = self._approach_order(list(agg["approach"].unique()))
            pal = self._approach_palette(order)
            fig, ax = plt.subplots(figsize=self._figsize())
            sns.scatterplot(
                data=agg, x="tokens", y="score", hue="approach", style="model",
                hue_order=order, palette=pal, s=180, ax=ax,
            )
            ax.set_title(f"Quality vs. output cost — {dataset}")
            ax.set_xlabel("mean output tokens")
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7, ncol=2)
            self._finish(fig, f"tradeoff_scatter_{dataset}")

    def format_heatmap(self, df: pd.DataFrame) -> None:
        """Model x approach accuracy heatmap, per dataset."""
        metric = self.cfg.primary_metric
        seq = self.cfg.palette.semantic.get("sequential", "rocket")
        for dataset, sub in df.groupby("dataset", sort=False):
            order = self._approach_order(list(sub["approach"].unique()))
            pivot = (
                sub.groupby(["model", "approach"])[metric]
                .mean()
                .reset_index()
                .pivot(index="model", columns="approach", values=metric)
                .reindex(columns=order)
            )
            fig, ax = plt.subplots(
                figsize=self._figsize(1.0, 0.55 + 0.35 * len(pivot))
            )
            sns.heatmap(
                pivot, annot=True, fmt=".2f", cmap=seq, vmin=0, vmax=1,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": METRIC_LABELS[metric]}, ax=ax,
            )
            ax.set_title(f"{METRIC_LABELS[metric]} by model and approach — {dataset}")
            ax.set_xlabel("approach")
            ax.set_ylabel("model")
            self._finish(fig, f"format_heatmap_{dataset}")


def add_approach_column(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the unified ``approach`` dimension (format for direct, 'sql' for sql)."""
    df = df.copy()
    df["approach"] = df.apply(
        lambda r: "sql" if r["pipeline"] == "sql" else r.get("format"), axis=1
    )
    return df


def render_all(df: pd.DataFrame, cfg: VizConfig, out_dir: Path) -> list[str]:
    """Render every enabled figure; return the list of figure base-names drawn."""
    df = add_approach_column(df)
    plotter = Plotter(cfg, out_dir)
    drawn: list[str] = []

    registry = {
        "accuracy_overview": plotter.accuracy_overview,
        "token_consumption": plotter.token_consumption,
        "latency": plotter.latency,
        "sql_diagnostics": plotter.sql_diagnostics,
        "parameter_scale": plotter.parameter_scale,
        "tradeoff_scatter": plotter.tradeoff_scatter,
        "format_heatmap": plotter.format_heatmap,
    }
    for name, fn in registry.items():
        if cfg.enabled(name):
            fn(df)
            drawn.append(name)
    return drawn
