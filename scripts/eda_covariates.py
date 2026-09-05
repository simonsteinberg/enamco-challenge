"""EDA step 3: screen the remaining covariates against ``picked_up``.

Two questions per covariate:

1. Marginal association with the label (rate per level, Wilson CI,
   chi-square, Cramer's V).
2. The decisive one for this task: does it change the *shape* of the
   pick-up-versus-local-hour curve, or only its height? The deliverable
   is a per-user ranking of hours, so a covariate that moves everybody
   up or down by a constant factor is useless.

Run with::

    uv run python scripts/eda_covariates.py

Writes plots and a stats dump under ``reports/covariate_screen/``.
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
OUT = ROOT / "reports" / "covariate_screen"

# Categorical slots 1-7 of the validated reference palette.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
GREY = "#8a8a86"
INK = "#0b0b0b"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET]

#: Minimum attempts for a level to count as "adequate n".
MIN_LEVEL_N = 300
#: Minimum attempts in an (hour, level) cell before we read its rate.
MIN_CELL_N = 150

_lines: list[str] = []


def say(text: str = "") -> None:
    """Print a line and buffer it for the stats dump.

    Args:
        text: Line to emit.
    """
    print(text)
    _lines.append(text)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        k: Number of successes.
        n: Number of trials.
        z: Normal quantile, 1.96 for 95%.

    Returns:
        Lower and upper bound of the interval.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def cramers_v(table: np.ndarray) -> tuple[float, float, float, int]:
    """Chi-square test plus Cramer's V for a contingency table.

    Args:
        table: Counts, levels by outcome.

    Returns:
        Tuple of ``(V, bias_corrected_V, p_value, dof)``. The bias
        correction is Bergsma's, which matters here because
        ``campaign_id`` and ``state`` have many levels and raw V is
        inflated by level count alone.
    """
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.sum()
    r, c = table.shape
    v = np.sqrt(chi2 / (n * (min(r, c) - 1)))
    phi2 = max(0.0, chi2 / n - (r - 1) * (c - 1) / (n - 1))
    rr = r - (r - 1) ** 2 / (n - 1)
    cc = c - (c - 1) ** 2 / (n - 1)
    denom = min(rr, cc) - 1
    v_corr = np.sqrt(phi2 / denom) if denom > 0 else 0.0
    return float(v), float(v_corr), float(p), int(dof)


def marginal_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Pick-up rate per level of a categorical covariate.

    Args:
        df: Attempt-level frame containing ``picked_up``.
        col: Column holding the levels.

    Returns:
        Frame indexed by level with ``n``, ``k``, ``rate``, ``lo``,
        ``hi``.
    """
    g = df.groupby(col, observed=True)["picked_up"]
    out = pd.DataFrame({"n": g.size(), "k": g.sum().astype(int)})
    out["rate"] = out["k"] / out["n"]
    ci = [wilson(int(k), int(n)) for k, n in zip(out["k"], out["n"])]
    out["lo"] = [c[0] for c in ci]
    out["hi"] = [c[1] for c in ci]
    return out


def report_categorical(df: pd.DataFrame, col: str, show: int = 8) -> dict:
    """Print the marginal screen for one categorical covariate.

    Args:
        df: Attempt-level frame.
        col: Column to screen.
        show: How many levels to print in full.

    Returns:
        Summary dict with ``v``, ``v_corr``, ``p``, ``spread_pp``.
    """
    tab = marginal_table(df, col)
    ct = np.column_stack([tab["n"] - tab["k"], tab["k"]])
    v, v_corr, p, dof = cramers_v(ct)
    ok = tab[tab["n"] >= MIN_LEVEL_N]
    spread = (ok["rate"].max() - ok["rate"].min()) * 100 if len(ok) else 0.0
    say(f"--- {col}: {len(tab)} levels, chi2 p={p:.3g} (dof={dof})")
    say(f"    Cramer's V={v:.4f}  bias-corrected V={v_corr:.4f}")
    say(
        f"    spread over levels with n>={MIN_LEVEL_N}: "
        f"{spread:.2f} pp ({len(ok)} levels)"
    )
    view = tab.sort_values("rate", ascending=False)
    head = view.head(show) if len(view) > show else view
    for lvl, row in head.iterrows():
        say(
            f"      {str(lvl)[:22]:<22} n={int(row['n']):>6} "
            f"rate={row['rate']:.4f} "
            f"[{row['lo']:.4f}, {row['hi']:.4f}]"
        )
    if len(view) > show:
        say(f"      ... {len(view) - show} more levels omitted")
    return {"v": v, "v_corr": v_corr, "p": p, "spread_pp": spread}


def report_continuous(df: pd.DataFrame, col: str, bins: int = 5) -> dict:
    """Print the marginal screen for one continuous covariate.

    Args:
        df: Attempt-level frame.
        col: Numeric column to screen.
        bins: Number of quantile bins for the rate table.

    Returns:
        Summary dict with ``spearman``, ``pointbiserial``,
        ``spread_pp``.
    """
    x = df[col].astype(float)
    y = df["picked_up"].astype(int)
    rho, rho_p = stats.spearmanr(x, y)
    rpb, rpb_p = stats.pointbiserialr(y, x)
    q = pd.qcut(x, bins, duplicates="drop")
    tab = marginal_table(df.assign(_q=q), "_q")
    spread = (tab["rate"].max() - tab["rate"].min()) * 100
    say(f"--- {col}: Spearman={rho:.4f} (p={rho_p:.3g}), ")
    say(f"    point-biserial={rpb:.4f} (p={rpb_p:.3g})")
    say(f"    spread over {len(tab)} quantile bins: {spread:.2f} pp")
    for lvl, row in tab.iterrows():
        say(
            f"      {str(lvl)[:22]:<22} n={int(row['n']):>6} "
            f"rate={row['rate']:.4f} "
            f"[{row['lo']:.4f}, {row['hi']:.4f}]"
        )
    return {"spearman": float(rho), "rpb": float(rpb), "spread_pp": spread}


def _loglik(k: np.ndarray, n: np.ndarray, mu: np.ndarray) -> float:
    """Binomial log-likelihood on aggregated cells.

    The binomial coefficient is dropped, so this equals the
    Bernoulli log-likelihood of the underlying rows exactly.

    Args:
        k: Successes per cell.
        n: Trials per cell.
        mu: Fitted probability per cell.

    Returns:
        Log-likelihood.
    """
    mu = np.clip(mu, 1e-12, 1 - 1e-12)
    return float(np.sum(k * np.log(mu) + (n - k) * np.log1p(-mu)))


def fit_binomial(
    x: np.ndarray, n: np.ndarray, k: np.ndarray, iters: int = 60
) -> float:
    """Fit a binomial GLM by IRLS and return its log-likelihood.

    Aggregated-cell IRLS with a least-squares solve, so rank-deficient
    designs degrade gracefully instead of raising.

    Args:
        x: Design matrix, cells by parameters.
        n: Trials per cell.
        k: Successes per cell.
        iters: Maximum IRLS iterations.

    Returns:
        Maximised log-likelihood.
    """
    b = np.zeros(x.shape[1])
    ll = -np.inf
    for _ in range(iters):
        eta = x @ b
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = np.clip(n * mu * (1 - mu), 1e-9, None)
        z = eta + (k - n * mu) / w
        xtw = x.T * w
        b_new = np.linalg.lstsq(xtw @ x, xtw @ z, rcond=None)[0]
        step = float(np.max(np.abs(b_new - b)))
        b = b_new
        ll_new = _loglik(k, n, 1.0 / (1.0 + np.exp(-np.clip(x @ b, -30, 30))))
        if step < 1e-9 or abs(ll_new - ll) < 1e-9:
            ll = ll_new
            break
        ll = ll_new
    return ll


def interaction_test(df: pd.DataFrame, col: str) -> dict:
    """Test whether a covariate bends the hour curve or only shifts it.

    Compares three nested binomial models on (local_hour, level) cells:
    hour only, hour + covariate main effect, and the saturated
    hour-by-covariate model. Reports likelihood-ratio tests and AIC.

    Args:
        df: Attempt-level frame with ``local_hour`` and ``picked_up``.
        col: Covariate column, already binned to a few levels.

    Returns:
        Summary dict used by the verdict table and the plots.
    """
    cells = (
        df.groupby(["local_hour", col], observed=True)["picked_up"]
        .agg(n="size", k="sum")
        .reset_index()
    )
    cells = cells[cells["n"] > 0].copy()
    n = cells["n"].to_numpy(float)
    k = cells["k"].to_numpy(float)

    hour_d = pd.get_dummies(cells["local_hour"], prefix="h").to_numpy(float)
    lvl_d = pd.get_dummies(cells[col], prefix="l", drop_first=True)
    lvl_d = lvl_d.to_numpy(float)

    n_hours = hour_d.shape[1]
    n_lvls = lvl_d.shape[1] + 1

    hour_rate = cells.groupby("local_hour")["k"].sum() / cells.groupby(
        "local_hour"
    )["n"].sum()
    mu_a = cells["local_hour"].map(hour_rate).to_numpy(float)
    ll_a = _loglik(k, n, mu_a)

    ll_b = fit_binomial(np.column_stack([hour_d, lvl_d]), n, k)
    df_b = n_hours + n_lvls - 1

    ll_c, df_c = _loglik(k, n, k / n), len(cells)

    lr_main = 2 * (ll_b - ll_a)
    lr_int = 2 * (ll_c - ll_b)
    dof_int = max(df_c - df_b, 1)
    p_int = float(stats.chi2.sf(max(lr_int, 0.0), dof_int))
    aic_b, aic_c = -2 * ll_b + 2 * df_b, -2 * ll_c + 2 * df_c

    # Curve-shape diagnostics on levels with enough support.
    wide_n = cells.pivot(index="local_hour", columns=col, values="n")
    wide_k = cells.pivot(index="local_hour", columns=col, values="k")
    rate = wide_k / wide_n
    keep = wide_n.sum(axis=0) >= MIN_LEVEL_N
    rate, wide_n = rate.loc[:, keep], wide_n.loc[:, keep]
    lvl_mean = wide_k.loc[:, keep].sum(axis=0) / wide_n.sum(axis=0)
    norm = rate.div(lvl_mean, axis=1)
    pooled = wide_k.sum(axis=1) / wide_n.sum(axis=1)
    overall = wide_k.loc[:, keep].sum().sum() / wide_n.sum().sum()
    pooled_norm = pooled / overall
    solid = wide_n >= MIN_CELL_N
    dev = (norm.sub(pooled_norm, axis=0)).abs().where(solid)

    best, top3 = {}, {}
    for lvl in rate.columns:
        r = rate[lvl].where(solid[lvl]).dropna()
        if r.empty:
            continue
        best[lvl] = int(r.idxmax())
        top3[lvl] = tuple(int(h) for h in r.nlargest(3).index)
    top3_same = len(set(top3.values())) <= 1
    best_same = len(set(best.values())) <= 1

    return {
        "col": col,
        "n_levels": n_lvls,
        "lr_main": lr_main,
        "lr_int": lr_int,
        "dof_int": dof_int,
        "p_int": p_int,
        "aic_b": aic_b,
        "aic_c": aic_c,
        "aic_gain_int": aic_b - aic_c,
        "max_dev": float(np.nanmax(dev.to_numpy())) if dev.size else np.nan,
        "mean_dev": float(np.nanmean(dev.to_numpy())) if dev.size else np.nan,
        "best_hour": best,
        "top3": top3,
        "best_same": best_same,
        "top3_same": top3_same,
        "norm": norm,
        "pooled_norm": pooled_norm,
        "solid": solid,
    }


def report_interaction(res: dict) -> None:
    """Print the hour-interaction verdict for one covariate.

    Args:
        res: Output of :func:`interaction_test`.
    """
    say(f"--- {res['col']}: hour-shape test ({res['n_levels']} levels)")
    which = "interaction wins" if res["aic_gain_int"] > 0 else "additive wins"
    say(
        f"    LR main effect={res['lr_main']:.1f} | "
        f"LR interaction={res['lr_int']:.1f} on {res['dof_int']} dof, "
        f"p={res['p_int']:.3g}"
    )
    say(
        f"    AIC hour+main={res['aic_b']:.0f}  "
        f"AIC hour*cov={res['aic_c']:.0f}  "
        f"gain={res['aic_gain_int']:+.0f} ({which})"
    )
    say(
        f"    normalised-curve deviation: max={res['max_dev']:.3f} "
        f"mean={res['mean_dev']:.3f}"
    )
    for lvl, hr in res["best_hour"].items():
        say(
            f"      {str(lvl)[:22]:<22} best hour={hr:>2}  "
            f"top3={res['top3'][lvl]}"
        )
    say(
        f"    same best hour across levels: {res['best_same']} | "
        f"same top-3 set: {res['top3_same']}"
    )


#: Covariates whose apparent hour-interaction disappears once the
#: named confounder is conditioned on (see section 3b).
CONFOUNDED = {"product": "age"}


def verdict(res: dict, v_corr: float) -> tuple[str, str]:
    """Turn the numbers into a shape label and a keep/drop call.

    Args:
        res: Output of :func:`interaction_test`.
        v_corr: Bias-corrected Cramer's V from the marginal screen.

    Returns:
        Tuple of ``(shape_label, recommendation)``.
    """
    if res["col"] in CONFOUNDED:
        by = CONFOUNDED[res["col"]]
        return f"bends, but confounded with {by}", f"DROP (use {by})"
    bends = res["aic_gain_int"] > 0 and res["max_dev"] >= 0.10
    if bends and not res["best_same"]:
        return "changes the ranking", "KEEP"
    if bends:
        return "changes the curve mildly", "MAYBE"
    if v_corr >= 0.03:
        return "shifts level only", "DROP (for ranking)"
    return "no signal", "DROP"


def plot_effect_sizes(rows: list[dict], path: Path) -> None:
    """Bar chart comparing all covariates on marginal and shape effects.

    Args:
        rows: One dict per covariate with the summary numbers.
        path: PNG output path.
    """
    rows = sorted(rows, key=lambda r: r["max_dev"])
    labels = [r["name"] for r in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)

    ax = axes[0]
    ax.barh(y, [r["v_corr"] for r in rows], color=GREY, height=0.62)
    ax.set_title(
        "Marginal association\n(bias-corrected Cramer's V)", fontsize=11
    )
    ax.set_xlabel("V")

    ax = axes[1]
    colors = [
        BLUE if r["max_dev"] >= 0.10 and r["aic_gain_int"] > 0 else GREY
        for r in rows
    ]
    ax.barh(y, [r["max_dev"] for r in rows], color=colors, height=0.62)
    ax.axvline(0.10, color=ORANGE, lw=2, ls="--")
    ax.text(
        0.105, len(rows) - 0.6, "shape threshold", color=ORANGE, fontsize=9
    )
    ax.set_title(
        "Does it bend the hour curve?\nmax |rate/group-mean - pooled|",
        fontsize=11,
    )
    ax.set_xlabel(
        "normalised-curve deviation "
        "(blue = AIC also favours the interaction)"
    )

    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.grid(axis="x", color="#e6e6e2", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    fig.suptitle(
        "Covariate screen: level effects vs hour-ranking effects",
        fontsize=13,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hour_curves(results: list[dict], path: Path) -> None:
    """Small multiples of the normalised hour curve per covariate.

    Args:
        results: Outputs of :func:`interaction_test`, one per panel.
        path: PNG output path.
    """
    ncol = 4
    nrow = int(np.ceil(len(results) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.0 * ncol, 3.1 * nrow), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, res in zip(axes, results):
        norm, solid = res["norm"], res["solid"]
        many = norm.shape[1] > 7
        for i, lvl in enumerate(norm.columns):
            s = norm[lvl].where(solid[lvl])
            if many:
                ax.plot(s.index, s.to_numpy(), color=GREY, lw=0.7, alpha=0.45)
            else:
                ax.plot(
                    s.index,
                    s.to_numpy(),
                    color=SERIES[i % len(SERIES)],
                    lw=2,
                    marker="o",
                    ms=4,
                    label=str(lvl),
                )
        ax.plot(
            res["pooled_norm"].index,
            res["pooled_norm"].to_numpy(),
            color=INK,
            lw=2,
            ls="--",
            label="pooled",
        )
        ax.axhline(1.0, color="#e6e6e2", lw=1)
        bends = res["max_dev"] >= 0.10 and res["aic_gain_int"] > 0
        flag = "bends" if bends else "flat"
        ax.set_title(
            f"{res['col']}  ({flag}, max dev {res['max_dev']:.2f})",
            fontsize=10,
        )
        ax.grid(color="#f0f0ec", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    for ax in axes[len(results) :]:
        ax.set_visible(False)
    for ax in axes[-ncol:]:
        ax.set_xlabel("customer-local hour")
    for i in range(0, len(axes), ncol):
        axes[i].set_ylabel("rate / group mean")
    fig.suptitle(
        "Hour curve shape by covariate level (height normalised out)",
        fontsize=13,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_campaign_confounding(df: pd.DataFrame, path: Path) -> None:
    """Check whether campaigns are tied to specific hours or months.

    Args:
        df: Attempt-level frame with ``local_hour`` and ``month``.
        path: PNG output path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, col, xlabel in (
        (axes[0], "local_hour", "customer-local hour"),
        (axes[1], "month_idx", "month index of attempt"),
    ):
        share = pd.crosstab(df["campaign_id"], df[col], normalize="index")
        for _, row in share.iterrows():
            ax.plot(
                share.columns, row.to_numpy(), color=GREY, lw=0.7, alpha=0.4
            )
        pooled = df[col].value_counts(normalize=True).sort_index()
        ax.plot(
            pooled.index,
            pooled.to_numpy(),
            color=BLUE,
            lw=2.5,
            label="pooled",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("share of that campaign's attempts")
        ax.set_ylim(bottom=0)
        ax.grid(color="#f0f0ec", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=9, frameon=False)
    fig.suptitle(
        "Confounding check: each grey line is one of 50 campaigns",
        fontsize=13,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_frame() -> pd.DataFrame:
    """Load the historic attempts and derive the screened features.

    Returns:
        Attempt-level frame with local-time and derived columns.
    """
    df = pd.read_csv(DATA)
    df = add_local_time(df)
    att = pd.to_datetime(df["attempted_at_utc"], utc=True, format="mixed")
    signup = pd.to_datetime(df["signup_date"]).dt.tz_localize("UTC")
    csignup = pd.to_datetime(df["contract_signup_date"]).dt.tz_localize("UTC")
    df["tenure_days"] = (att - signup).dt.total_seconds() / 86400.0
    df["contract_age_days"] = (att - csignup).dt.total_seconds() / 86400.0
    df["signup_cohort"] = signup.dt.to_period("Q").astype(str)
    df["month_idx"] = (
        att.dt.year * 12 + att.dt.month - (att.dt.year.min() * 12 + 1)
    )
    df["month_idx"] = df["month_idx"] - df["month_idx"].min()
    df["age_cohort"] = pd.cut(
        df["age"],
        [0, 34, 44, 54, 200],
        labels=["<=34", "35-44", "45-54", "55+"],
    )
    df["tenure_q"] = pd.qcut(
        df["tenure_days"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    df["state_top"] = df["state"].where(
        df["state"].isin(df["state"].value_counts().head(8).index), "other"
    )
    n_contracts = df.groupby("user_id")["contract_id"].transform("nunique")
    df["n_contracts_per_user"] = n_contracts
    df["is_first_contract"] = df["contract_id"] == df.groupby("user_id")[
        "contract_id"
    ].transform("min")
    return df


def main() -> None:
    """Run the covariate screen and write plots plus the stats dump."""
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_frame()
    n, base = len(df), df["picked_up"].mean()
    say(f"rows={n}  users={df['user_id'].nunique()}  base rate={base:.4f}")
    say(
        f"attempts per user: median="
        f"{df.groupby('user_id').size().median():.0f}, "
        f"max={df.groupby('user_id').size().max()}"
    )
    say(
        "CAVEAT: rows repeat per user, so every test below treats "
        "correlated rows as independent. p-values are optimistic; "
        "read the effect sizes, not the stars."
    )
    say()

    say("=== 0. Identifier / duplicate-column checks ===")
    same_id = float((df["contract_id"] == df["user_id"]).mean())
    same_date = float(
        (df["contract_signup_date"] == df["signup_date"]).mean()
    )
    say(f"contract_id == user_id in {same_id:.1%} of rows")
    say(f"contract_signup_date == signup_date in {same_date:.1%} of rows")
    say(
        f"distinct contracts per user: "
        f"{sorted(df['n_contracts_per_user'].unique())}"
    )
    say(
        "is_first_contract always true: "
        f"{bool(df['is_first_contract'].all())}"
    )
    say(
        "contract_age_days identical to tenure_days: "
        f"{bool(np.allclose(df['contract_age_days'], df['tenure_days']))}"
    )
    say()

    say("=== 1. Marginal association with picked_up ===")
    marg: dict[str, dict] = {}
    cat_cols = [
        "has_email",
        "device_rented",
        "product",
        "campaign_id",
        "state",
    ]
    for col in cat_cols:
        marg[col] = report_categorical(df, col)
        say()
    for col in ["signup_cohort", "age_cohort", "tenure_q"]:
        marg[col] = report_categorical(df, col)
        say()
    cont: dict[str, dict] = {}
    for col in ["age", "tenure_days", "contract_age_days"]:
        cont[col] = report_continuous(df, col)
        say()
    say("--- local_hour (reference: the feature we already know matters)")
    marg["local_hour"] = report_categorical(df, "local_hour", show=24)
    say()

    say("=== 2. Confounding checks ===")
    hour_share = pd.crosstab(
        df["campaign_id"], df["local_hour"], normalize="index"
    )
    pooled_hour = df["local_hour"].value_counts(normalize=True).sort_index()
    dev = (hour_share - pooled_hour).abs().to_numpy().max()
    say(
        f"campaign x hour attempt share: max |campaign - pooled| = {dev:.4f} "
        f"(pooled share per hour ~ {pooled_hour.mean():.4f})"
    )
    chi_h = stats.chi2_contingency(
        pd.crosstab(df["campaign_id"], df["local_hour"])
    )
    say(f"campaign x hour attempt-count chi2 p={chi_h.pvalue:.3g}")
    chi_m = stats.chi2_contingency(
        pd.crosstab(df["campaign_id"], df["month_idx"])
    )
    say(f"campaign x month attempt-count chi2 p={chi_m.pvalue:.3g}")
    months = pd.crosstab(df["campaign_id"], df["month_idx"])
    say(
        "months each campaign is active in: "
        f"min={int((months > 0).sum(axis=1).min())} "
        f"max={int((months > 0).sum(axis=1).max())}"
    )
    by_month = df.groupby("month_idx")["picked_up"].agg(["size", "mean"])
    say("pick-up rate by calendar month index:")
    for m, row in by_month.iterrows():
        say(
            f"      m{int(m):<3} n={int(row['size']):>6} "
            f"rate={row['mean']:.4f}"
        )
    say(
        "tenure vs calendar time correlation (Spearman): "
        f"{stats.spearmanr(df['tenure_days'], df['month_idx']).statistic:.3f}"
    )
    say()

    say("=== 3. Decisive test: does the covariate bend the hour curve? ===")
    inter_cols = [
        "has_email",
        "device_rented",
        "product",
        "campaign_id",
        "timezone",
        "age_cohort",
        "tenure_q",
        "signup_cohort",
    ]
    results = [interaction_test(df, c) for c in inter_cols]
    for res in results:
        report_interaction(res)
        say()

    say("=== 3b. Is the product hour-effect just age in disguise? ===")
    ct = pd.crosstab(df["product"], df["age_cohort"], normalize="index")
    say("share of each product's rows by age cohort:")
    for prod, row in ct.iterrows():
        cells = "  ".join(f"{c}={v:.3f}" for c, v in row.items())
        say(f"      {prod:<10} {cells}")
    mean_age = df.groupby("product")["age"].mean().round(1).to_dict()
    say(f"mean age by product: {mean_age}")
    for cohort in df["age_cohort"].cat.categories:
        sub = df[df["age_cohort"] == cohort]
        r = interaction_test(sub, "product")
        say(
            f"      product within age {cohort:<6} n={len(sub):>6} "
            f"AIC gain={r['aic_gain_int']:+.0f} max_dev={r['max_dev']:.3f} "
            f"best hours={r['best_hour']}"
        )
    for prod in sorted(df["product"].unique()):
        sub = df[df["product"] == prod]
        r = interaction_test(sub, "age_cohort")
        say(
            f"      age within product {prod:<9} n={len(sub):>6} "
            f"AIC gain={r['aic_gain_int']:+.0f} max_dev={r['max_dev']:.3f} "
            f"best hours={r['best_hour']}"
        )
    say()

    say("=== 3c. Where exactly does the age flip happen? ===")
    fine = pd.cut(df["age"], [0, 30, 35, 40, 45, 50, 55, 60, 65, 200])
    tab = df.pivot_table(
        index=fine,
        columns="local_hour",
        values="picked_up",
        aggfunc="mean",
        observed=True,
    )
    sizes = df.groupby(fine, observed=True).size()
    for band, row in tab.iterrows():
        r = row.dropna()
        say(
            f"      age {band!s:<11} n={sizes[band]:>6} "
            f"best={int(r.idxmax()):>2} "
            f"top3={[int(h) for h in r.nlargest(3).index]} "
            f"rate@10={row.get(10, float('nan')):.3f} "
            f"rate@19={row.get(19, float('nan')):.3f}"
        )
    say()

    say("=== 4. Verdict table ===")
    marg_for = {
        "has_email": "has_email",
        "device_rented": "device_rented",
        "product": "product",
        "campaign_id": "campaign_id",
        "timezone": "state",
        "age_cohort": "age_cohort",
        "tenure_q": "tenure_q",
        "signup_cohort": "signup_cohort",
    }
    say(
        f"{'covariate':<16}{'V_corr':>9}{'spread_pp':>11}"
        f"{'max_dev':>9}{'AIC gain':>10}  shape / call"
    )
    rows = []
    for res in results:
        m = marg[marg_for[res["col"]]]
        shape, call = verdict(res, m["v_corr"])
        say(
            f"{res['col']:<16}{m['v_corr']:>9.4f}{m['spread_pp']:>11.2f}"
            f"{res['max_dev']:>9.3f}{res['aic_gain_int']:>10.0f}  "
            f"{shape} / {call}"
        )
        rows.append(
            {
                "name": res["col"],
                "v_corr": m["v_corr"],
                "max_dev": res["max_dev"],
                "aic_gain_int": res["aic_gain_int"],
            }
        )
    say()

    plot_effect_sizes(rows, OUT / "effect_sizes.png")
    plot_hour_curves(results, OUT / "hour_curve_by_group.png")
    plot_campaign_confounding(df, OUT / "campaign_confounding.png")
    (OUT / "stats.txt").write_text("\n".join(_lines) + "\n")
    print(f"\nwrote plots and stats.txt to {OUT}")


if __name__ == "__main__":
    main()
