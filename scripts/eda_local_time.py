"""EDA step 1: UTC vs customer-local call-attempt time.

Run with::

    uv run python scripts/eda_local_time.py

Writes plots and a stats dump under ``reports/local_time_feature/``.
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

from challenge.feature import (
    MULTI_ZONE_STATES,
    NO_DST_STATES,
    add_local_time,
    utc_offset_hours,
)

DATA = ROOT / "data" / "historic.csv"
OUT = ROOT / "reports" / "local_time_feature"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GREY = "#8a8a86"


def hour_share(hours: pd.Series, support: range) -> np.ndarray:
    """Return the normalised hour distribution over ``support``.

    Args:
        hours: Integer hour values.
        support: Hours to bin over.

    Returns:
        Array of shares summing to 1 (zeros if ``hours`` is empty).
    """
    counts = hours.value_counts().reindex(support, fill_value=0)
    total = counts.sum()
    if total == 0:
        return np.zeros(len(support), dtype=float)
    return (counts / total).to_numpy(dtype=float)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two discrete distributions."""
    return float(0.5 * np.abs(p - q).sum())


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits."""
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def chi_square_uniform(hours: pd.Series, support: range) -> dict[str, float]:
    """Chi-square goodness of fit of hour counts against uniform."""
    counts = hours.value_counts().reindex(support, fill_value=0)
    obs = counts.to_numpy(dtype=float)
    exp = np.full(len(support), obs.sum() / len(support))
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    dof = len(support) - 1
    return {
        "chi2": chi2,
        "dof": float(dof),
        "p_value": float(stats.chi2.sf(chi2, dof)),
        "cramers_v": float(np.sqrt(chi2 / (obs.sum() * dof))),
    }


def plot_utc_vs_local(df: pd.DataFrame) -> None:
    """Bar chart of attempt counts by UTC hour and by local hour."""
    hours = range(24)
    utc = df["utc_hour"].value_counts().reindex(hours, fill_value=0)
    loc = df["local_hour"].value_counts().reindex(hours, fill_value=0)
    x = np.arange(24)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - 0.21, utc, width=0.42, color=GREY, label="UTC hour")
    ax.bar(x + 0.21, loc, width=0.42, color=BLUE, label="Customer-local hour")
    ax.set_xticks(x)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Call attempts")
    ax.set_title(
        "Attempts by UTC hour vs customer-local hour "
        "(local time collapses onto an 08:00-20:00 window)"
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e2", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "utc_vs_local_hour.png", dpi=150)
    plt.close(fig)


def plot_by_timezone(df: pd.DataFrame, support: range) -> None:
    """Small multiples of local-hour share per timezone, states overlaid."""
    zones = sorted(df["timezone"].unique())
    pooled = hour_share(df["local_hour"], support)
    ncol = 4
    nrow = int(np.ceil(len(zones) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.3 * ncol, 2.6 * nrow), sharey=True
    )
    axes = np.atleast_1d(axes).ravel()
    xs = list(support)
    for ax, zone in zip(axes, zones):
        sub = df[df["timezone"] == zone]
        for _, grp in sub.groupby("state"):
            if len(grp) < 200:
                continue
            ax.plot(
                xs,
                hour_share(grp["local_hour"], support),
                color=AQUA,
                lw=0.8,
                alpha=0.45,
            )
        ax.plot(xs, pooled, color=GREY, lw=2, ls="--", label="pooled")
        ax.plot(
            xs,
            hour_share(sub["local_hour"], support),
            color=ORANGE,
            lw=2,
            label="timezone",
        )
        ax.set_title(f"{zone}\nn={len(sub):,}", fontsize=9)
        ax.set_xticks([8, 11, 14, 17, 20])
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[len(zones) :]:
        ax.set_visible(False)
    axes[0].set_ylabel("Share of attempts")
    handles = [
        plt.Line2D([], [], color=ORANGE, lw=2, label="Timezone"),
        plt.Line2D([], [], color=AQUA, lw=1, label="State (n>=200)"),
        plt.Line2D([], [], color=GREY, lw=2, ls="--", label="Pooled"),
    ]
    fig.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        ncol=3,
        bbox_to_anchor=(0.99, 0.01),
    )
    fig.suptitle("Local-hour distribution per timezone (states as thin lines)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(OUT / "local_hour_by_timezone.png", dpi=150)
    plt.close(fig)


def plot_pickup_rate(df: pd.DataFrame, support: range) -> None:
    """Pick-up rate by local hour with Wilson-style error bars."""
    g = df.groupby("local_hour")["picked_up"].agg(["mean", "count"])
    g = g.reindex(support).dropna()
    se = np.sqrt(g["mean"] * (1 - g["mean"]) / g["count"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.errorbar(
        g.index,
        g["mean"],
        yerr=1.96 * se,
        color=BLUE,
        lw=2,
        marker="o",
        ms=6,
        capsize=3,
        ecolor=GREY,
    )
    ax.axhline(
        df["picked_up"].mean(),
        color=ORANGE,
        ls="--",
        lw=1.5,
        label=f"overall {df['picked_up'].mean():.3f}",
    )
    ax.set_xticks(list(support))
    ax.set_xlabel("Customer-local hour")
    ax.set_ylabel("Pick-up rate")
    ax.set_title("Pick-up rate by customer-local hour (95% CI)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e2", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "pickup_rate_by_local_hour.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the local-time EDA and write plots plus a stats dump."""
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA)
    df = add_local_time(raw)
    df["utc_hour"] = pd.to_datetime(
        df["attempted_at_utc"], utc=True, format="mixed"
    ).dt.hour
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say(
        f"rows={len(df):,}  users={df.user_id.nunique():,}  "
        f"states={df.state.nunique()}"
    )
    say(f"states present: {' '.join(sorted(df.state.unique()))}")
    say(
        f"multi-zone states in data: "
        f"{' '.join(sorted(MULTI_ZONE_STATES & set(df.state)))}"
    )
    say(
        f"no-DST states in data: "
        f"{' '.join(sorted(NO_DST_STATES & set(df.state)))}"
    )
    say()

    # 1. UTC vs local support.
    say("== UTC hour vs local hour ==")
    say(f"UTC hours used:   {sorted(df.utc_hour.unique())}")
    say(f"Local hours used: {sorted(df.local_hour.unique())}")
    utc_counts = df.utc_hour.value_counts().reindex(range(24), fill_value=0)
    loc_counts = df.local_hour.value_counts().reindex(range(24), fill_value=0)
    say(
        pd.DataFrame(
            {
                "utc_n": utc_counts,
                "local_n": loc_counts,
                "utc_share": utc_counts / len(df),
                "local_share": loc_counts / len(df),
            }
        ).to_string(float_format=lambda v: f"{v:.4f}")
    )
    say()

    # 2. Uniformity tests.
    say("== Chi-square goodness of fit vs uniform ==")
    for label, support in [
        ("UTC hour, support 0-23", range(24)),
        ("Local hour, support 0-23", range(24)),
        ("Local hour, support 8-20 (observed)", range(8, 21)),
        ("Local hour, support 9-19 (interior)", range(9, 20)),
    ]:
        col = "utc_hour" if label.startswith("UTC") else "local_hour"
        sub = df[df[col].isin(support)]
        res = chi_square_uniform(sub[col], support)
        say(
            f"{label:38s} chi2={res['chi2']:12.1f} "
            f"dof={int(res['dof']):3d} p={res['p_value']:.3e} "
            f"V={res['cramers_v']:.4f}"
        )
    say()

    # 3. Agreement across timezones and states.
    support = range(8, 21)
    pooled = hour_share(df.local_hour, support)
    say("== Local-hour share by timezone (support 8-20) ==")
    tz_tab = pd.crosstab(
        df.timezone, df.local_hour, normalize="index"
    ).reindex(columns=list(support), fill_value=0.0)
    tz_tab.insert(0, "n", df.timezone.value_counts())
    say(tz_tab.to_string(float_format=lambda v: f"{v:.3f}"))
    say()

    rows = []
    for tz, grp in df.groupby("timezone"):
        p = hour_share(grp.local_hour, support)
        rows.append(
            {
                "key": tz,
                "level": "timezone",
                "n": len(grp),
                "tvd": total_variation(p, pooled),
                "jsd_bits": jensen_shannon(p, pooled),
            }
        )
    for st, grp in df.groupby("state"):
        p = hour_share(grp.local_hour, support)
        rows.append(
            {
                "key": st,
                "level": "state",
                "n": len(grp),
                "tvd": total_variation(p, pooled),
                "jsd_bits": jensen_shannon(p, pooled),
            }
        )
    div = pd.DataFrame(rows)
    tz_div = div[div.level == "timezone"].sort_values("tvd", ascending=False)
    st_div = div[div.level == "state"].sort_values("tvd", ascending=False)
    say("Divergence from pooled local-hour distribution, timezones:")
    say(
        tz_div[["key", "n", "tvd", "jsd_bits"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    say()
    say("States, top 8 by TVD (largest deviation from pooled):")
    say(
        st_div.head(8)[["key", "n", "tvd", "jsd_bits"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    say(
        f"state TVD: median={st_div.tvd.median():.4f} "
        f"p90={st_div.tvd.quantile(0.9):.4f} max={st_div.tvd.max():.4f}"
    )
    big = st_div[st_div.n >= 1000]
    say(
        f"states with n>=1000: max TVD={big.tvd.max():.4f} "
        f"({big.iloc[0].key}, n={int(big.iloc[0].n)})"
    )
    say()

    # Chi-square of independence: local hour vs timezone.
    tab = pd.crosstab(df.timezone, df.local_hour)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    v = np.sqrt(chi2 / (tab.to_numpy().sum() * (min(tab.shape) - 1)))
    say(
        f"independence local_hour x timezone: chi2={chi2:.1f} dof={dof} "
        f"p={p:.3e} Cramer's V={v:.4f}"
    )
    tab_s = pd.crosstab(df.state, df.local_hour)
    chi2s, ps, dofs, _ = stats.chi2_contingency(tab_s)
    vs = np.sqrt(chi2s / (tab_s.to_numpy().sum() * (min(tab_s.shape) - 1)))
    say(
        f"independence local_hour x state:    chi2={chi2s:.1f} dof={dofs} "
        f"p={ps:.3e} Cramer's V={vs:.4f}"
    )
    say()

    # 4. Pick-up rate preview.
    say("== Pick-up rate by local hour vs attempt share ==")
    pk = df.groupby("local_hour").agg(
        n=("picked_up", "size"), rate=("picked_up", "mean")
    )
    pk["attempt_share"] = pk.n / pk.n.sum()
    pk["rate_index"] = pk.rate / df.picked_up.mean()
    say(pk.to_string(float_format=lambda v: f"{v:.4f}"))
    corr = np.corrcoef(pk.attempt_share, pk.rate)[0, 1]
    say(f"overall pick-up rate = {df.picked_up.mean():.4f}")
    say(f"corr(attempt_share, pick-up rate) over local hours = {corr:.3f}")
    pk_utc = df.groupby("utc_hour")["picked_up"].mean()
    say(
        f"pick-up rate spread: local {pk.rate.min():.3f}-{pk.rate.max():.3f}, "
        f"UTC {pk_utc.min():.3f}-{pk_utc.max():.3f}"
    )
    say()

    # 5. DST sanity check.
    say("== DST sanity check ==")
    summer = pd.Timestamp("2025-07-15T18:00:00Z")
    winter = pd.Timestamp("2026-01-15T18:00:00Z")
    for st in ("OH", "AZ", "HI", "CA"):
        say(
            f"{st}: summer offset={utc_offset_hours(st, summer):+.1f}h  "
            f"winter offset={utc_offset_hours(st, winter):+.1f}h"
        )
    say(pd.crosstab(df.timezone, df.is_dst).to_string())
    say(
        df.groupby("timezone")
        .utc_offset_hours.agg(["min", "max", "nunique"])
        .to_string()
    )
    say()

    plot_utc_vs_local(df)
    plot_by_timezone(df, support)
    plot_pickup_rate(df, support)
    say(f"plots written to {OUT}")
    (OUT / "stats.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
