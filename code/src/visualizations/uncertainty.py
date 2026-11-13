"""Pointwise database-cluster bootstrap intervals for means and paired effects.

Models are fixed configurations, not independent replications. Each draw retains every
question and configuration from a sampled database; means retain their run weighting.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from visualizations.latex_numbers import num, percent

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261006


@lru_cache(maxsize=64)
def database_weights(count: int) -> np.ndarray:
    """Database multiplicities, shared by all configurations in a comparison."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.multinomial(count, np.full(count, 1 / count), size=BOOTSTRAP_DRAWS)


def bounds(draws: np.ndarray) -> tuple[float, float]:
    if not np.isfinite(draws).all():
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def mean_intervals(
    data: pd.DataFrame,
    groups: list[str],
    metrics: dict[str, str],
    *,
    comparison: str | None = None,
    reference: str | None = None,
    effect: str = "difference",
    pair_keys: tuple[str, ...] = (),
    min_databases: int = 2,
) -> pd.DataFrame:
    """Return mean CIs and paired difference/ratio CIs, keyed like a summary.

    With no explicit reference, use the smallest model in each supplied family group.
    Groups below min_databases retain point effects but have NaN interval endpoints.
    The default preserves the existing procedure; restricted context-fit analyses use
    three to omit intervals based on only two databases. This is an omission rule,
    not a guarantee that larger cluster counts satisfy the bootstrap assumptions.
    Pair identities must match exactly; incomplete comparisons fail rather than silently
    changing cohorts. Ratios are ratios of means, never averages of individual ratios.
    """
    rows = []
    for key, group in data.groupby(groups, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(groups, key, strict=True))
        databases = sorted(group["db_id"].unique())
        weights = database_weights(len(databases))
        categories = list(group[comparison].unique()) if comparison else [None]
        baseline = reference
        if comparison and baseline is None:
            baseline = group.sort_values(["model_size_b", comparison])[comparison].iloc[0]
        if comparison and baseline not in categories:
            raise ValueError(f"Missing comparison reference {baseline!r} in {identity}")
        estimates, samples = {}, {}
        identities = None
        for category in categories:
            selected = group[group[comparison].eq(category)] if comparison else group
            if comparison:
                if not pair_keys:
                    raise ValueError("Paired effects require explicit pair keys")
                current = pd.MultiIndex.from_frame(selected[list(pair_keys)])
                if current.has_duplicates:
                    raise ValueError("Duplicate identities in paired comparison")
                if identities is not None and set(current) != identities:
                    raise ValueError("Paired comparison has unequal observation identities")
                identities = set(current)
            estimates[category], samples[category] = {}, {}
            for name, source in metrics.items():
                numeric = pd.to_numeric(selected[source], errors="raise")
                if not np.isfinite(numeric).all():
                    raise ValueError(f"Non-finite bootstrap metric: {source}")
                aggregate = selected.assign(_value=numeric).groupby("db_id")["_value"].agg(
                    ["sum", "count"]
                ).reindex(databases, fill_value=0)
                estimates[category][name] = float(numeric.mean())
                samples[category][name] = (
                    weights @ aggregate["sum"].to_numpy()
                    / (weights @ aggregate["count"].to_numpy())
                )
        for category in categories:
            row = {**identity}
            if comparison:
                row[comparison] = category
                row["effect_reference"] = baseline
            for name in metrics:
                low, high = bounds(samples[category][name])
                if len(databases) < min_databases:
                    low = high = float("nan")
                row[f"{name}_low"], row[f"{name}_high"] = low, high
                if comparison:
                    value, ref = estimates[category][name], estimates[baseline][name]
                    if effect == "ratio":
                        with np.errstate(divide="ignore", invalid="ignore"):
                            draws = samples[category][name] / samples[baseline][name]
                        estimate = value / ref if ref else float("nan")
                    elif effect == "difference":
                        draws = samples[category][name] - samples[baseline][name]
                        estimate = value - ref
                    else:
                        raise ValueError(f"Unknown effect: {effect}")
                    low, high = bounds(draws)
                    if len(databases) < min_databases:
                        low = high = float("nan")
                    row[f"{name}_effect"] = estimate
                    row[f"{name}_effect_low"] = low
                    row[f"{name}_effect_high"] = high
            rows.append(row)
    return pd.DataFrame(rows)


def interval_marks(axis, positions, rows, metric, *, horizontal=False, color="#333333"):
    """Draw CI endpoints directly, including intervals not containing the estimate."""
    if f"{metric}_low" not in rows:
        return
    low = rows[f"{metric}_low"].to_numpy(dtype=float)
    high = rows[f"{metric}_high"].to_numpy(dtype=float)
    valid = np.isfinite(low) & np.isfinite(high)
    if not valid.any():
        return
    positions = np.asarray(positions)[valid]
    low, high = low[valid], high[valid]
    if horizontal:
        axis.hlines(positions, low, high, color=color, linewidth=1.2, zorder=5)
        axis.plot(low, positions, "|", color=color, markersize=5)
        axis.plot(high, positions, "|", color=color, markersize=5)
    else:
        axis.vlines(positions, low, high, color=color, linewidth=1.2, zorder=5)
        axis.plot(positions, low, "_", color=color, markersize=5)
        axis.plot(positions, high, "_", color=color, markersize=5)


def table_estimate(
    row, metric: str, decimals=2, *, scale=1.0, suffix="", show_interval=True,
) -> str:
    """Two-line LaTeX cell, keeping intervals below their estimates to limit width."""
    if hasattr(row, "_asdict"):
        row = row._asdict()
    formatter = percent if suffix == r"\%" else num
    value = formatter(row[metric] * scale, decimals)
    if suffix and suffix != r"\%":
        value += suffix
    if not show_interval or f"{metric}_low" not in row:
        return value
    scaled = {f"{metric}_low": row[f"{metric}_low"] * scale,
              f"{metric}_high": row[f"{metric}_high"] * scale}
    if not np.isfinite([scaled[f"{metric}_low"], scaled[f"{metric}_high"]]).all():
        return (r"\shortstack[r]{" + value + r"\\{\scriptsize [not estimable]}}")
    low = formatter(scaled[f"{metric}_low"], decimals)
    high = formatter(scaled[f"{metric}_high"], decimals)
    return (r"\shortstack[r]{" + value + r"\\{\scriptsize "
            + f"[{low}; {high}]" + "}}")
