"""EDA step 2: pick-up rate as a function of customer-local hour.

Measures the pick-up curve over local hour of day and how that curve
varies by timezone and by age cohort, with Wilson 95% intervals and
chi-square tests so the differences are quantified rather than
eyeballed.

Run with::

    uv run python scripts/eda_pick_curve.py

Writes plots and a stats dump under ``reports/pick_curve/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from challenge.feature import add_local_time

DATA = ROOT / "data" / "historic.csv"
OUT = ROOT / "reports" / "pick_curve"

#: Minimum attempts in an (group, hour) bin before we read its rate.
MIN_BIN_N = 200
#: Minimum attempts in a group before we plot the group at all.
MIN_GROUP_N = 3000

#: Categorical slots 1-5 of the validated colorblind-safe palette.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
#: Distinct marker per series, so identity is never color-alone.
SERIES_MARKERS = ("o", "s", "^", "D", "v")
GREY = "#8a8a86"
LIGHT_GREY = "#f0efec"
INK = "#0b0b0b"

AGE_EDGES = [18, 30, 40, 50, 60, 200]
AGE_LABELS = ["18-30", "30-40", "40-50", "50-60", "60+"]

_LINES: list[str] = []


def emit(text: str = "") -> None:
    """Print a line and keep it for the stats dump.

    Args:
        text: Line to record.
    """
    print(text)
    _LINES.append(text)


def wilson_ci(
    k: np.ndarray, n: np.ndarray, z: float = 1.959963985
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion.

    Args:
        k: Successes per bin.
        n: Trials per bin.
        z: Normal quantile; default is the 95% two-sided value.

    Returns:
        Tuple of lower and upper bounds. Empty bins give ``nan``.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        denom = 1.0 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo = np.where(n > 0, centre - half, np.nan)
    hi = np.where(n > 0, centre + half, np.nan)
    return lo, hi


def hour_curve(df: pd.DataFrame, support: range) -> pd.DataFrame:
    """Pick-up rate per local hour with Wilson 95% intervals.

    Args:
        df: Frame with ``local_hour`` and boolean ``picked_up``.
        support: Hours to bin over.

    Returns:
        Frame indexed by hour with ``n``, ``k``, ``rate``, ``lo``,
        ``hi`` (rates and bounds as proportions).
    """
    grouped = df.groupby("local_hour")["picked_up"]
    n = grouped.size().reindex(support, fill_value=0)
    k = grouped.sum().reindex(support, fill_value=0)
    lo, hi = wilson_ci(k.to_numpy(), n.to_numpy())
    out = pd.DataFrame(
        {"n": n, "k": k, "rate": k / n.replace(0, np.nan), "lo": lo, "hi": hi}
    )
    out.index.name = "local_hour"
    return out


def peak_trough(curve: pd.DataFrame) -> dict[str, float]:
    """Peak hour, trough hour and spread over well-populated bins.

    Args:
        curve: Output of :func:`hour_curve`.

    Returns:
        Dict with peak/trough hours, their rates in percent and the
        peak-to-trough spread in percentage points.
    """
    solid = curve[curve["n"] >= MIN_BIN_N].dropna(subset=["rate"])
    if solid.empty:
        return {}
    peak = int(solid["rate"].idxmax())
    trough = int(solid["rate"].idxmin())
    return {
        "peak_hour": peak,
        "peak_pct": 100 * solid.loc[peak, "rate"],
        "trough_hour": trough,
        "trough_pct": 100 * solid.loc[trough, "rate"],
        "spread_pp": 100
        * (solid.loc[peak, "rate"] - solid.loc[trough, "rate"]),
    }


def chi2_hour_independence(df: pd.DataFrame) -> dict[str, float]:
    """Chi-square test of independence of hour and ``picked_up``.

    Args:
        df: Frame with ``local_hour`` and boolean ``picked_up``.

    Returns:
        Dict with the statistic, degrees of freedom, p-value and
        Cramer's V.
    """
    table = pd.crosstab(df["local_hour"], df["picked_up"])
    table = table[(table.sum(axis=1) >= MIN_BIN_N)]
    chi2, p, dof, _ = stats.chi2_contingency(table.to_numpy())
    n = float(table.to_numpy().sum())
    v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    return {"chi2": chi2, "dof": dof, "p": p, "cramers_v": v}


def deviation_from_pooled(
    curve: pd.DataFrame, pooled: pd.DataFrame
) -> dict[str, float]:
    """Test a group curve against the pooled curve, hour by hour.

    Under the null that the group follows the pooled hourly rates,
    the per-hour standardised residuals are approximately standard
    normal and their sum of squares is chi-square with one degree of
    freedom per hour. The test is mildly conservative because the
    group also contributes to the pooled rates.

    Args:
        curve: Group curve from :func:`hour_curve`.
        pooled: Pooled curve from :func:`hour_curve`.

    Returns:
        Dict with the chi-square statistic and p-value, the mean
        absolute deviation from pooled in percentage points, the
        deviation expected from sampling noise alone, their ratio and
        the largest single-hour deviation.
    """
    use = curve[curve["n"] >= MIN_BIN_N].index
    n = curve.loc[use, "n"].to_numpy(dtype=float)
    k = curve.loc[use, "k"].to_numpy(dtype=float)
    p0 = pooled.loc[use, "rate"].to_numpy(dtype=float)
    var = n * p0 * (1 - p0)
    z = (k - n * p0) / np.sqrt(var)
    chi2 = float(np.sum(z**2))
    dof = len(use)
    dev_pp = 100 * (k / n - p0)
    se_pp = 100 * np.sqrt(p0 * (1 - p0) / n)
    expected = float(np.mean(se_pp) * np.sqrt(2 / np.pi))
    observed = float(np.mean(np.abs(dev_pp)))
    worst = int(use[int(np.argmax(np.abs(dev_pp)))])
    return {
        "chi2": chi2,
        "dof": dof,
        "p": float(stats.chi2.sf(chi2, dof)),
        "mad_pp": observed,
        "mad_pp_noise": expected,
        "mad_ratio": observed / expected,
        "max_dev_pp": float(dev_pp[np.argmax(np.abs(dev_pp))]),
        "max_dev_hour": worst,
    }


def style_hour_axis(ax: plt.Axes, support: range) -> None:
    """Apply the shared local-hour x-axis styling.

    Args:
        ax: Axis to style.
        support: Hours that carry data.
    """
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 2))
    ax.axvspan(-0.5, min(support) - 0.5, color=LIGHT_GREY, zorder=0)
    ax.axvspan(max(support) + 0.5, 23.5, color=LIGHT_GREY, zorder=0)
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_overall(curve: pd.DataFrame, support: range) -> Path:
    """Plot the pooled pick-up curve with a CI band and bin counts.

    Args:
        curve: Pooled curve from :func:`hour_curve`.
        support: Hours that carry data.

    Returns:
        Path of the written PNG.
    """
    fig, (ax, ax_n) = plt.subplots(
        2,
        1,
        figsize=(9.5, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )
    x = curve.index.to_numpy()
    ax.fill_between(
        x,
        100 * curve["lo"],
        100 * curve["hi"],
        color=SERIES_COLORS[0],
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(x, 100 * curve["rate"], color=SERIES_COLORS[0], linewidth=2)
    solid = curve["n"] >= MIN_BIN_N
    ax.plot(
        x[solid],
        100 * curve.loc[solid, "rate"],
        marker="o",
        markersize=5,
        linestyle="none",
        color=SERIES_COLORS[0],
        markeredgecolor="#fcfcfb",
        markeredgewidth=1.2,
    )
    ax.set_ylabel("pick-up rate (%)")
    ax.set_title(
        "Pick-up rate by customer-local hour (95% Wilson intervals)",
        loc="left",
        color=INK,
    )
    style_hour_axis(ax, support)
    ax.text(
        3.5,
        ax.get_ylim()[1] * 0.92,
        "no attempts",
        color=GREY,
        ha="center",
        fontsize=9,
    )

    ax_n.bar(x, curve["n"], color=GREY, width=0.72)
    ax_n.set_ylabel("attempts")
    ax_n.set_xlabel("local hour of day")
    style_hour_axis(ax_n, support)
    fig.text(
        0.01,
        0.005,
        f"n = {int(curve['n'].sum()):,} attempts; bins under "
        f"{MIN_BIN_N} attempts are not read.",
        color=GREY,
        fontsize=9,
    )
    path = OUT / "pickup_by_local_hour.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return path


def plot_split(
    curves: dict[str, pd.DataFrame],
    pooled: pd.DataFrame,
    support: range,
    title: str,
    legend_title: str,
    filename: str,
) -> Path:
    """Plot group curves overlaid plus their deviation from pooled.

    Args:
        curves: Group label -> curve from :func:`hour_curve`.
        pooled: Pooled curve, drawn as a reference.
        support: Hours that carry data.
        title: Figure title.
        legend_title: Legend header.
        filename: Output file name inside ``reports/pick_curve``.

    Returns:
        Path of the written PNG.
    """
    fig, (ax, ax_d) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    x = pooled.index.to_numpy()
    ax.plot(
        x,
        100 * pooled["rate"],
        color=GREY,
        linewidth=1.6,
        linestyle="--",
        label="all customers",
        zorder=1,
    )
    for i, (label, curve) in enumerate(curves.items()):
        colour = SERIES_COLORS[i % len(SERIES_COLORS)]
        marker = SERIES_MARKERS[i % len(SERIES_MARKERS)]
        solid = curve["n"] >= MIN_BIN_N
        ax.fill_between(
            x[solid],
            100 * curve.loc[solid, "lo"],
            100 * curve.loc[solid, "hi"],
            color=colour,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            x[solid],
            100 * curve.loc[solid, "rate"],
            color=colour,
            linewidth=2,
            marker=marker,
            markersize=4.5,
            label=label,
        )
        dev = 100 * (curve.loc[solid, "rate"] - pooled.loc[solid, "rate"])
        se = 100 * np.sqrt(
            pooled.loc[solid, "rate"]
            * (1 - pooled.loc[solid, "rate"])
            / curve.loc[solid, "n"]
        )
        ax_d.errorbar(
            x[solid],
            dev,
            yerr=1.96 * se,
            color=colour,
            linewidth=1.8,
            elinewidth=0.9,
            capsize=2,
            marker=marker,
            markersize=4.5,
            label=label,
        )
    ax.set_ylabel("pick-up rate (%)")
    ax.set_xlabel("local hour of day")
    ax.set_title(title, loc="left", color=INK)
    style_hour_axis(ax, support)
    ax.legend(title=legend_title, frameon=False, fontsize=9, loc="upper left")

    ax_d.axhline(0, color=GREY, linewidth=1.2, linestyle="--")
    ax_d.set_ylabel("deviation from pooled curve (pp)")
    ax_d.set_xlabel("local hour of day")
    ax_d.set_title(
        "Deviation from the pooled curve (95% intervals)",
        loc="left",
        color=INK,
    )
    style_hour_axis(ax_d, support)
    ax_d.legend(title=legend_title, frameon=False, fontsize=9, loc="best")
    path = OUT / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return path


def curve_table(curve: pd.DataFrame) -> str:
    """Render a curve as a fixed-width text table.

    Args:
        curve: Curve from :func:`hour_curve`.

    Returns:
        Multi-line string with hour, n, rate and Wilson bounds.
    """
    rows = ["  hour       n   rate%    lo%    hi%"]
    for hour, row in curve.iterrows():
        if row["n"] == 0:
            continue
        rows.append(
            f"  {hour:>4}  {int(row['n']):>6}  "
            f"{100 * row['rate']:>6.1f} {100 * row['lo']:>6.1f} "
            f"{100 * row['hi']:>6.1f}"
        )
    return "\n".join(rows)


def report_split(
    df: pd.DataFrame,
    key: str,
    pooled: pd.DataFrame,
    support: range,
    heading: str,
) -> dict[str, pd.DataFrame]:
    """Print and return per-group curves for one split.

    Args:
        df: Frame with ``local_hour``, ``picked_up`` and ``key``.
        key: Grouping column.
        pooled: Pooled curve for comparison.
        support: Hours to bin over.
        heading: Section heading for the stats dump.

    Returns:
        Group label -> curve, only for groups above ``MIN_GROUP_N``.
    """
    emit()
    emit(heading)
    emit("=" * len(heading))
    sizes = df.groupby(key, observed=True).size().sort_values(ascending=False)
    kept: dict[str, pd.DataFrame] = {}
    emit("group sizes:")
    for label, size in sizes.items():
        verdict = "kept" if size >= MIN_GROUP_N else "DROPPED (too thin)"
        emit(f"  {label!s:<22} {int(size):>7} attempts  {verdict}")
    emit(f"(group kept when it has >= {MIN_GROUP_N} attempts)")

    if isinstance(df[key].dtype, pd.CategoricalDtype):
        order = [c for c in df[key].cat.categories if c in sizes.index]
    else:
        order = list(sizes.index)

    emit()
    emit(
        f"{'group':<22} {'peak':>5} {'peak%':>7} {'trough':>7} "
        f"{'trough%':>8} {'spread pp':>10} {'chi2 hour':>11} "
        f"{'p(hour)':>10} {'vs pooled chi2':>15} {'dof':>4} {'p':>10} "
        f"{'MAD pp':>8} {'noise pp':>9} {'ratio':>7}"
    )
    for label in order:
        if sizes.get(label, 0) < MIN_GROUP_N:
            continue
        sub = df[df[key] == label]
        curve = hour_curve(sub, support)
        kept[str(label)] = curve
        pt = peak_trough(curve)
        ind = chi2_hour_independence(sub)
        dev = deviation_from_pooled(curve, pooled)
        emit(
            f"{label!s:<22} {pt['peak_hour']:>5} {pt['peak_pct']:>7.1f} "
            f"{pt['trough_hour']:>7} {pt['trough_pct']:>8.1f} "
            f"{pt['spread_pp']:>10.1f} {ind['chi2']:>11.0f} "
            f"{ind['p']:>10.2e} {dev['chi2']:>15.0f} {dev['dof']:>4} "
            f"{dev['p']:>10.2e} "
            f"{dev['mad_pp']:>8.2f} {dev['mad_pp_noise']:>9.2f} "
            f"{dev['mad_ratio']:>7.1f}"
        )
    emit()
    for label, curve in kept.items():
        emit(f"{label} curve:")
        emit(curve_table(curve))
    return kept


def age_edge_check(df: pd.DataFrame, lo: int = 55, hi: int = 66) -> None:
    """Locate the age at which the curve flips, year by year.

    Compares each single-year age group's morning (local 9-12) and
    evening (local 18-20) pick-up rate, which is enough to show where
    the shape changes without committing to cohort edges.

    Args:
        df: Frame with ``age``, ``local_hour`` and ``picked_up``.
        lo: First age to print.
        hi: One past the last age to print.
    """
    emit()
    emit("AGE COHORT EDGE: single-year rates around the flip")
    emit("=" * 50)
    emit("  age       n  morning% (9-12)  evening% (18-20)")
    for age in range(lo, hi):
        sub = df[df["age"] == age]
        if len(sub) < MIN_BIN_N:
            continue
        am = sub.loc[sub["local_hour"].between(9, 12), "picked_up"].mean()
        pm = sub.loc[sub["local_hour"].between(18, 20), "picked_up"].mean()
        emit(
            f"  {age:>3}  {len(sub):>6}  {100 * am:>15.1f}  "
            f"{100 * pm:>16.1f}"
        )


def side_checks(df: pd.DataFrame, support: range) -> None:
    """Check volume confounding and weekday vs weekend shape.

    Args:
        df: Frame with local-time columns and ``picked_up``.
        support: Hours to bin over.
    """
    emit()
    emit("SIDE CHECK 1: is the hour effect confounded by attempt volume?")
    emit("=" * 62)
    pooled = hour_curve(df, support)
    solid = pooled[pooled["n"] >= MIN_BIN_N]
    # Edge hours 8 and 20 are only partly inside the calling window,
    # so compare volume and rate on the fully covered core hours too.
    core = solid[solid["n"] >= 0.5 * solid["n"].median()]
    for name, sub in (("all hours", solid), ("core hours", core)):
        r = float(np.corrcoef(sub["n"], sub["rate"])[0, 1])
        emit(
            f"{name:<10} hours {int(sub.index.min())}-"
            f"{int(sub.index.max())}: attempts per hour "
            f"{int(sub['n'].min()):,} to {int(sub['n'].max()):,} "
            f"({100 * (sub['n'].max() / sub['n'].min() - 1):.0f}% "
            f"range), rate spread "
            f"{100 * (sub['rate'].max() - sub['rate'].min()):.1f} pp, "
            f"Pearson r(attempts, rate) = {r:.2f}"
        )

    cell = (
        df.assign(date=df["local_datetime"].dt.date)
        .groupby(["local_hour", "date"])["picked_up"]
        .agg(["size", "sum"])
        .reset_index()
    )
    diffs = []
    for hour, grp in cell.groupby("local_hour"):
        if grp["size"].sum() < MIN_BIN_N:
            continue
        med = grp["size"].median()
        hi = grp[grp["size"] > med]
        lo = grp[grp["size"] <= med]
        if hi["size"].sum() == 0 or lo["size"].sum() == 0:
            continue
        diffs.append(
            100
            * (
                hi["sum"].sum() / hi["size"].sum()
                - lo["sum"].sum() / lo["size"].sum()
            )
        )
    emit(
        "within-hour busy vs quiet days: mean |rate difference| = "
        f"{np.mean(np.abs(diffs)):.2f} pp over {len(diffs)} hours"
    )

    emit()
    emit("SIDE CHECK 2: weekday vs weekend curve shape")
    emit("=" * 44)
    week = df.assign(
        daytype=np.where(df["local_dow"] >= 5, "weekend", "weekday")
    )
    for label in ("weekday", "weekend"):
        sub = week[week["daytype"] == label]
        curve = hour_curve(sub, support)
        pt = peak_trough(curve)
        dev = deviation_from_pooled(curve, pooled)
        rate = 100 * sub["picked_up"].mean()
        emit(
            f"{label:<8} n={len(sub):>7}  rate={rate:.1f}%"
            f"  peak={pt['peak_hour']} ({pt['peak_pct']:.1f}%)"
            f"  trough={pt['trough_hour']} ({pt['trough_pct']:.1f}%)"
            f"  spread={pt['spread_pp']:.1f} pp"
            f"  vs pooled chi2={dev['chi2']:.0f} p={dev['p']:.2f}"
            f"  MAD={dev['mad_pp']:.2f} pp (noise {dev['mad_pp_noise']:.2f})"
        )


def main() -> None:
    """Run the pick-up-curve EDA and write plots and stats."""
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df = add_local_time(df)
    df["picked_up"] = df["picked_up"].astype(bool)
    support = range(
        int(df["local_hour"].min()), int(df["local_hour"].max()) + 1
    )

    emit("PICK-UP RATE BY CUSTOMER-LOCAL HOUR")
    emit("=" * 35)
    emit(f"attempts: {len(df):,}   overall pick-up rate: "
         f"{100 * df['picked_up'].mean():.2f}%")
    emit(
        f"local hours observed: {support.start}-{support.stop - 1} "
        "(no attempts outside this window)"
    )
    emit(f"bin threshold: read a bin only when n >= {MIN_BIN_N}")

    pooled = hour_curve(df, support)
    emit()
    emit("pooled curve:")
    emit(curve_table(pooled))
    pt = peak_trough(pooled)
    ind = chi2_hour_independence(df)
    emit(
        f"peak hour {pt['peak_hour']} ({pt['peak_pct']:.1f}%), "
        f"trough hour {pt['trough_hour']} ({pt['trough_pct']:.1f}%), "
        f"spread {pt['spread_pp']:.1f} pp"
    )
    emit(
        f"chi-square(hour x picked_up): chi2={ind['chi2']:.0f} "
        f"dof={ind['dof']} p={ind['p']:.2e} Cramer's V={ind['cramers_v']:.3f}"
    )
    p1 = plot_overall(pooled, support)

    tz_curves = report_split(
        df, "timezone", pooled, support, "SPLIT BY TIMEZONE (local hour)"
    )
    p2 = plot_split(
        tz_curves,
        pooled,
        support,
        "Pick-up rate by local hour, per timezone",
        "timezone",
        "pickup_by_timezone.png",
    )

    df["age_cohort"] = pd.cut(
        df["age"],
        bins=AGE_EDGES,
        labels=AGE_LABELS,
        right=False,
        include_lowest=True,
    )
    age_curves = report_split(
        df, "age_cohort", pooled, support, "SPLIT BY AGE COHORT (local hour)"
    )
    p3 = plot_split(
        age_curves,
        pooled,
        support,
        "Pick-up rate by local hour, per age cohort",
        "age cohort",
        "pickup_by_age_cohort.png",
    )
    age_edge_check(df)

    side_checks(df, support)

    emit()
    emit(f"wrote {p1.name}, {p2.name}, {p3.name}")
    (OUT / "stats.txt").write_text("\n".join(_LINES) + "\n")


if __name__ == "__main__":
    main()
