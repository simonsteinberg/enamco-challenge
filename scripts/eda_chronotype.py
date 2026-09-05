"""EDA step 4: is there an individual-level chronotype?

Asks whether users differ from each other in *when* they pick up,
over and above the population-by-age-group local-hour curve. With a
median of 17 attempts per user, a per-user morning rate and evening
rate rest on ~8 trials each (SE ~0.17), so naive per-user statistics
are noise-dominated and the known ``age >= 60`` curve inversion
manufactures a spurious negative correlation. Everything here is run
inside age strata and against the population curve.

Tests, in order of weight:

1. Split-half reliability of a per-user morning-vs-evening preference
   index (primary).
2. Parametric bootstrap: observed between-user spread of that index
   against the spread expected if every user followed the population
   curve exactly.
3. Held-out log-loss, population curve vs population curve plus a
   shrunken per-user block offset.
4. The naive per-user morning-rate vs evening-rate scatter, for
   contrast only.
5. Sanity check that attempts-per-user is not confounded with
   pick-up rate.

Run with::

    uv run python scripts/eda_chronotype.py

Writes plots and a stats dump under ``reports/chronotype/``.
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
OUT = ROOT / "reports" / "chronotype"

SEED = 20250905
#: Local hours kept; all attempts fall inside this window.
HOUR_LO, HOUR_HI = 8, 20
#: Morning/midday block is local hour <= AM_MAX, evening block above.
AM_MAX = 13
#: Minimum attempts per (user, block) cell for the full-data index.
MIN_CELL = 3
#: Minimum attempts per (user, half, block) cell for split-half.
MIN_CELL_HALF = 3
#: Parametric bootstrap replicates.
N_SIM = 500
#: Split-half replicates under the null (kept smaller; same cost).
N_SIM_SPLIT = 200
#: Laplace weight smoothing the per-(group, hour) rate to the group.
SMOOTH = 20.0
#: Share of attempts held out for the log-loss comparison.
TEST_FRAC = 0.3
#: Ridge weights for the shrunken per-user block offset.
SHRINK_K = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)

STRATA = ("age<60", "age>=60")

#: Categorical slots of the validated colorblind-safe palette.
BLUE, ORANGE = "#2a78d6", "#eb6834"
GREY = "#8a8a86"
LIGHT_GREY = "#f0efec"
INK = "#0b0b0b"

_LINES: list[str] = []


def emit(text: str = "") -> None:
    """Print a line and keep it for the stats dump.

    Args:
        text: Line to record.
    """
    print(text)
    _LINES.append(text)


def load() -> pd.DataFrame:
    """Load attempts with local time, age stratum and block labels.

    Returns:
        Frame of attempts restricted to local hours 8-20, with
        ``stratum`` (age band), ``is_am`` (morning/midday block),
        ``y`` (picked up as float) and integer user codes.
    """
    raw = pd.read_csv(DATA)
    df = add_local_time(raw)
    df = df[df["local_hour"].between(HOUR_LO, HOUR_HI)].copy()
    df["stratum"] = np.where(df["age"] >= 60, "age>=60", "age<60")
    df["age_decade"] = (df["age"] // 10 * 10).astype(int)
    df["is_am"] = df["local_hour"] <= AM_MAX
    df["y"] = df["picked_up"].astype(float)
    df["ucode"] = pd.factorize(df["user_id"])[0]
    return df.reset_index(drop=True)


def curve(
    df: pd.DataFrame,
    mask: np.ndarray | None = None,
    keys: tuple[str, ...] = ("stratum",),
) -> pd.Series:
    """Estimate the population pick-up curve by group and hour.

    Rates are smoothed toward the stratum mean so thin bins cannot
    dominate; the smoothing is negligible for the bins we actually
    use.

    Args:
        df: Attempt frame from :func:`load`.
        mask: Optional boolean row mask (e.g. training rows only).
        keys: Grouping columns the curve is estimated within.

    Returns:
        Series indexed by ``(*keys, local_hour)`` giving the smoothed
        pick-up probability.
    """
    sub = df if mask is None else df[mask]
    cols = [*keys, "local_hour"]
    grp = sub.groupby(cols)["y"].agg(["sum", "count"])
    base = sub.groupby(list(keys))["y"].mean()
    if len(keys) == 1:
        prior = grp.index.get_level_values(keys[0]).map(base)
    else:
        prior = base.reindex(grp.index.droplevel("local_hour")).to_numpy()
    return (grp["sum"] + SMOOTH * prior) / (grp["count"] + SMOOTH)


def attach_p(
    df: pd.DataFrame,
    cur: pd.Series,
    keys: tuple[str, ...] = ("stratum",),
) -> np.ndarray:
    """Look up each attempt's population probability.

    Args:
        df: Attempt frame.
        cur: Curve from :func:`curve`.
        keys: The same grouping columns used to build ``cur``.

    Returns:
        Array of population probabilities aligned to ``df`` rows.
    """
    key = pd.MultiIndex.from_arrays(
        [df[c] for c in (*keys, "local_hour")]
    )
    return cur.reindex(key).to_numpy()


def cell_sums(
    code: np.ndarray, n_users: int, is_am: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sum a per-attempt quantity into per-user AM and PM totals.

    Args:
        code: Per-attempt user code (0..n_users-1).
        n_users: Number of distinct users.
        is_am: Boolean morning-block flag per attempt.
        w: Per-attempt weight to sum.

    Returns:
        Tuple ``(am_totals, pm_totals)``.
    """
    am = np.bincount(code, weights=w * is_am, minlength=n_users)
    pm = np.bincount(code, weights=w * ~is_am, minlength=n_users)
    return am, pm


def index_from(
    code: np.ndarray,
    n_users: int,
    is_am: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the per-user morning-minus-evening residual index.

    The index is ``mean(y - p)`` over the user's morning attempts
    minus ``mean(y - p)`` over their evening attempts, so the
    population age-group curve is divided out and only the individual
    deviation remains.

    Args:
        code: Per-attempt user code.
        n_users: Number of distinct users.
        is_am: Boolean morning-block flag per attempt.
        y: Binary outcome per attempt.
        p: Population probability per attempt.

    Returns:
        Tuple ``(index, n_am, n_pm)``, each of length ``n_users``.
        The index is NaN where a block has no attempts.
    """
    ones = np.ones_like(y)
    n_am, n_pm = cell_sums(code, n_users, is_am, ones)
    r_am, r_pm = cell_sums(code, n_users, is_am, y - p)
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = np.where(n_am > 0, r_am / n_am, np.nan) - np.where(
            n_pm > 0, r_pm / n_pm, np.nan
        )
    return idx, n_am, n_pm


def fisher_ci(r: float, n: int) -> tuple[float, float]:
    """Fisher z 95% confidence interval for a Pearson correlation.

    Args:
        r: Observed correlation.
        n: Number of paired observations.

    Returns:
        Tuple ``(low, high)``.
    """
    if n < 4 or abs(r) >= 1:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    return (float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)))


def spearman_brown(r: float) -> float:
    """Extend a split-half correlation to full test length.

    Args:
        r: Split-half correlation.

    Returns:
        Reliability of the full-length index.
    """
    return 2.0 * r / (1.0 + r) if r > -1 else float("nan")


def half_labels(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Assign each attempt to half A or B, balanced within each user.

    Args:
        df: Attempt frame.
        rng: Seeded random generator.

    Returns:
        Boolean array, ``True`` for half A.
    """
    order = rng.permutation(len(df))
    ranks = np.empty(len(df), dtype=np.int64)
    code = df["ucode"].to_numpy()[order]
    seen: dict[int, int] = {}
    for i, u in enumerate(code):
        k = seen.get(u, 0)
        ranks[i] = k
        seen[u] = k + 1
    out = np.empty(len(df), dtype=bool)
    out[order] = ranks % 2 == 0
    return out


def time_halves(df: pd.DataFrame) -> np.ndarray:
    """Split each user's attempts chronologically into early and late.

    Args:
        df: Attempt frame.

    Returns:
        Boolean array, ``True`` for the user's earlier attempts.
    """
    order = df.groupby("user_id")["local_datetime"].rank(method="first")
    n = df["user_id"].map(df.groupby("user_id")["y"].size())
    return (order <= n / 2).to_numpy()


def split_half(
    df: pd.DataFrame,
    p: np.ndarray,
    rng: np.random.Generator,
    is_a: np.ndarray | None = None,
    do_null: bool = True,
):
    """Run the split-half reliability test within each age stratum.

    Args:
        df: Attempt frame.
        p: Population probability per attempt.
        rng: Seeded random generator.
        is_a: Optional half assignment; random balanced halves if
            omitted.
        do_null: Whether to simulate the null distribution of ``r``.

    Returns:
        Dict mapping stratum to a result dict with the paired indices,
        the correlation, its CI and the null correlation distribution.
    """
    if is_a is None:
        is_a = half_labels(df, rng)
    res = {}
    for st in STRATA:
        m = (df["stratum"] == st).to_numpy()
        code, uniq = pd.factorize(df["ucode"].to_numpy()[m])
        nu = len(uniq)
        is_am = df["is_am"].to_numpy()[m]
        y = df["y"].to_numpy()[m]
        pp = p[m]
        a = is_a[m]

        def one(
            yv: np.ndarray,
            sel: np.ndarray,
            code: np.ndarray = code,
            nu: int = nu,
            is_am: np.ndarray = is_am,
            pp: np.ndarray = pp,
        ):
            return index_from(code[sel], nu, is_am[sel], yv[sel], pp[sel])

        ia, na_am, na_pm = one(y, a)
        ib, nb_am, nb_pm = one(y, ~a)
        keep = (
            (na_am >= MIN_CELL_HALF)
            & (na_pm >= MIN_CELL_HALF)
            & (nb_am >= MIN_CELL_HALF)
            & (nb_pm >= MIN_CELL_HALF)
        )
        xa, xb = ia[keep], ib[keep]
        r = float(np.corrcoef(xa, xb)[0, 1]) if keep.sum() > 3 else np.nan
        lo, hi = fisher_ci(r, int(keep.sum()))

        null_r = np.empty(N_SIM_SPLIT if do_null else 0)
        for s in range(len(null_r)):
            ysim = (rng.random(len(pp)) < pp).astype(float)
            sa = one(ysim, a)[0][keep]
            sb = one(ysim, ~a)[0][keep]
            null_r[s] = np.corrcoef(sa, sb)[0, 1]

        res[st] = {
            "xa": xa,
            "xb": xb,
            "n": int(keep.sum()),
            "n_users": nu,
            "r": r,
            "ci": (lo, hi),
            "sb": spearman_brown(r),
            "null_r": null_r,
        }
    return res


def variance_null(df: pd.DataFrame, p: np.ndarray, rng: np.random.Generator):
    """Compare observed index spread against the population-curve null.

    Simulates each attempt's outcome at its real hour under the
    population probability for the user's age group, recomputes the
    index and compares the between-user variance.

    Args:
        df: Attempt frame.
        p: Population probability per attempt.
        rng: Seeded random generator.

    Returns:
        Dict mapping stratum to observed/null variance and the pooled
        simulated index values for plotting.
    """
    res = {}
    for st in STRATA:
        m = (df["stratum"] == st).to_numpy()
        code, uniq = pd.factorize(df["ucode"].to_numpy()[m])
        nu = len(uniq)
        is_am = df["is_am"].to_numpy()[m]
        y = df["y"].to_numpy()[m]
        pp = p[m]

        idx, n_am, n_pm = index_from(code, nu, is_am, y, pp)
        keep = (n_am >= MIN_CELL) & (n_pm >= MIN_CELL)
        obs = idx[keep]
        obs_var = float(np.var(obs, ddof=1))

        sim_var = np.empty(N_SIM)
        pool = []
        for s in range(N_SIM):
            ysim = (rng.random(len(pp)) < pp).astype(float)
            si = index_from(code, nu, is_am, ysim, pp)[0][keep]
            sim_var[s] = np.var(si, ddof=1)
            if s < 20:
                pool.append(si)
        p_val = float((np.sum(sim_var >= obs_var) + 1) / (N_SIM + 1))
        null_var = float(sim_var.mean())
        excess = obs_var - null_var
        res[st] = {
            "obs": obs,
            "n": int(keep.sum()),
            "obs_var": obs_var,
            "null_var": null_var,
            "null_sd_of_var": float(sim_var.std(ddof=1)),
            "excess": excess,
            "true_sd": float(np.sqrt(max(excess, 0.0))),
            "p": p_val,
            "sim": np.concatenate(pool),
        }
    return res


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    """Mean binary log-loss.

    Args:
        y: Binary outcomes.
        p: Predicted probabilities.

    Returns:
        Mean negative log-likelihood per attempt.
    """
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def heldout(df: pd.DataFrame, rng: np.random.Generator) -> list[tuple]:
    """Compare held-out log-loss with and without a per-user offset.

    The offset is one ridge-shrunk Newton step in log-odds per
    ``(user, block)`` cell, fitted on the training attempts only, on
    top of a population curve also fitted on training attempts only.

    Args:
        df: Attempt frame.
        rng: Seeded random generator.

    Returns:
        List of ``(label, test_log_loss)`` rows, baseline first.
    """
    test = rng.random(len(df)) < TEST_FRAC
    cur = curve(df, ~test)
    p_all = attach_p(df, cur)
    y = df["y"].to_numpy()
    code = df["ucode"].to_numpy()
    nu = int(code.max()) + 1
    is_am = df["is_am"].to_numpy()

    base = logloss(y[test], p_all[test])
    rows = [("population curve only", base)]

    tr = ~test
    w = p_all * (1 - p_all)
    r_am, r_pm = cell_sums(code[tr], nu, is_am[tr], (y - p_all)[tr])
    w_am, w_pm = cell_sums(code[tr], nu, is_am[tr], w[tr])

    logit = np.log(p_all / (1 - p_all))
    for k in SHRINK_K:
        d_am, d_pm = r_am / (w_am + k), r_pm / (w_pm + k)
        delta = np.where(is_am, d_am[code], d_pm[code])
        p_new = 1.0 / (1.0 + np.exp(-(logit + np.clip(delta, -3, 3))))
        label = f"+ per-user block offset, k={k:g}"
        rows.append((label, logloss(y[test], p_new[test])))
    return rows


def naive_rates(df: pd.DataFrame) -> dict:
    """Per-user raw morning and evening pick-up rates, by stratum.

    Args:
        df: Attempt frame.

    Returns:
        Dict mapping stratum (plus ``pooled``) to rates and their
        correlation.
    """
    g = df.groupby(["user_id", "stratum", "is_am"])["y"].agg(["sum", "count"])
    g = g.reset_index()
    wide = g.pivot_table(
        index=["user_id", "stratum"],
        columns="is_am",
        values=["sum", "count"],
    )
    wide = wide.dropna()
    am = wide[("sum", True)] / wide[("count", True)]
    pm = wide[("sum", False)] / wide[("count", False)]
    ok = (wide[("count", True)] >= MIN_CELL) & (
        wide[("count", False)] >= MIN_CELL
    )
    am, pm = am[ok], pm[ok]
    st = am.index.get_level_values("stratum")
    out = {}
    for key in (*STRATA, "pooled"):
        m = np.ones(len(am), dtype=bool) if key == "pooled" else (st == key)
        a, b = am[m].to_numpy(), pm[m].to_numpy()
        r = float(np.corrcoef(a, b)[0, 1])
        out[key] = {"am": a, "pm": b, "r": r, "n": int(m.sum())}
    return out


def confound_check(df: pd.DataFrame) -> None:
    """Report whether attempts-per-user tracks pick-up rate.

    Args:
        df: Attempt frame.
    """
    emit()
    emit("5. SANITY CHECK: attempts per user vs pick-up rate")
    per = df.groupby(["user_id", "stratum"])["y"].agg(["mean", "count"])
    per = per.reset_index()
    for st in STRATA:
        sub = per[per["stratum"] == st]
        rho, pv = stats.spearmanr(sub["count"], sub["mean"])
        emit(
            f"  {st}: n_users={len(sub)}  "
            f"spearman(n_attempts, pick rate)={rho:+.3f} (p={pv:.3g})"
        )
        q = pd.qcut(sub["count"], 5, duplicates="drop")
        tab = sub.groupby(q, observed=True)["mean"].agg(["mean", "size"])
        for lab, row in tab.iterrows():
            emit(
                f"      attempts {lab!s:>16}: rate={row['mean']:.3f} "
                f"(n_users={int(row['size'])})"
            )


def style(ax: plt.Axes) -> None:
    """Apply the shared minimal axis style.

    Args:
        ax: Axes to style.
    """
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.grid(True, color=LIGHT_GREY, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)


def plot_split_half(res: dict) -> Path:
    """Plot half-A vs half-B index scatter per stratum.

    Args:
        res: Output of :func:`split_half`.

    Returns:
        Path to the written PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    for ax, (st, r) in zip(axes, res.items()):
        style(ax)
        ax.scatter(r["xa"], r["xb"], s=10, alpha=0.25, color=BLUE, lw=0)
        if r["n"] > 3:
            b, a = np.polyfit(r["xa"], r["xb"], 1)
            xs = np.linspace(r["xa"].min(), r["xa"].max(), 20)
            ax.plot(xs, a + b * xs, color=ORANGE, lw=2)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.axvline(0, color=GREY, lw=0.8)
        lo, hi = r["ci"]
        ax.set_title(
            f"{st}: r = {r['r']:+.3f} [{lo:+.3f}, {hi:+.3f}], "
            f"n = {r['n']}",
            fontsize=10,
            color=INK,
        )
        ax.set_xlabel("half A index (AM minus PM residual)")
    axes[0].set_ylabel("half B index")
    fig.suptitle(
        "Split-half reliability of the per-user morning/evening index",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    path = OUT / "split_half_scatter.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def plot_null(res: dict) -> Path:
    """Plot observed vs null-simulated index distributions.

    Args:
        res: Output of :func:`variance_null`.

    Returns:
        Path to the written PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, (st, r) in zip(axes, res.items()):
        style(ax)
        bins = np.linspace(-1, 1, 61)
        ax.hist(
            r["sim"],
            bins=bins,
            density=True,
            color=GREY,
            alpha=0.55,
            label="null: everyone follows the age-group curve",
        )
        ax.hist(
            r["obs"],
            bins=bins,
            density=True,
            histtype="step",
            lw=2,
            color=ORANGE,
            label="observed",
        )
        ax.set_title(
            f"{st}: SD obs {np.sqrt(r['obs_var']):.3f} vs null "
            f"{np.sqrt(r['null_var']):.3f} (p={r['p']:.3f})",
            fontsize=10,
            color=INK,
        )
        ax.set_xlabel("per-user index (AM minus PM residual)")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Between-user spread vs the spread sampling noise alone predicts",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    path = OUT / "index_observed_vs_null.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def plot_naive(res: dict) -> Path:
    """Plot the naive per-user morning-rate vs evening-rate scatter.

    Args:
        res: Output of :func:`naive_rates`.

    Returns:
        Path to the written PNG.
    """
    rng = np.random.default_rng(SEED)
    keys = ("pooled", *STRATA)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True, sharey=True)
    for ax, key in zip(axes, keys):
        r = res[key]
        style(ax)
        j = 0.02
        ax.scatter(
            r["am"] + rng.normal(0, j, len(r["am"])),
            r["pm"] + rng.normal(0, j, len(r["pm"])),
            s=9,
            alpha=0.18,
            color=BLUE,
            lw=0,
        )
        b, a = np.polyfit(r["am"], r["pm"], 1)
        xs = np.linspace(0, 1, 20)
        ax.plot(xs, a + b * xs, color=ORANGE, lw=2)
        ax.set_title(
            f"{key}: r = {r['r']:+.3f}, n = {r['n']}",
            fontsize=10,
            color=INK,
        )
        ax.set_xlabel("morning/midday pick-up rate")
    axes[0].set_ylabel("afternoon/evening pick-up rate")
    fig.suptitle(
        "Naive view (noise-dominated): per-user raw rates, pooled vs "
        "age-stratified",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    path = OUT / "naive_rate_scatter.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    """Run the chronotype analysis and write plots plus a stats dump."""
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = load()
    cur = curve(df)
    p = attach_p(df, cur)

    emit("CHRONOTYPE ANALYSIS")
    emit(
        f"attempts={len(df)}  users={df['user_id'].nunique()}  "
        f"blocks: AM=local {HOUR_LO}-{AM_MAX}, PM=local "
        f"{AM_MAX + 1}-{HOUR_HI}  seed={SEED}"
    )
    for st in STRATA:
        sub = df[df["stratum"] == st]
        emit(
            f"  {st}: attempts={len(sub)} users={sub['user_id'].nunique()} "
            f"rate={sub['y'].mean():.3f} "
            f"(AM {sub.loc[sub['is_am'], 'y'].mean():.3f} / "
            f"PM {sub.loc[~sub['is_am'], 'y'].mean():.3f})"
        )

    emit()
    emit("1. SPLIT-HALF RELIABILITY (primary test)")
    sh = split_half(df, p, rng)
    for st, r in sh.items():
        lo, hi = r["ci"]
        nr = r["null_r"]
        emit(
            f"  {st}: users with >={MIN_CELL_HALF} attempts in all four "
            f"(half, block) cells: {r['n']} of {r['n_users']}"
        )
        emit(
            f"      r(A, B) = {r['r']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
            f"Spearman-Brown = {r['sb']:+.4f}"
        )
        emit(
            f"      null r over {N_SIM_SPLIT} sims: mean {nr.mean():+.4f}, "
            f"sd {nr.std(ddof=1):.4f}, "
            f"P(null r >= observed) = "
            f"{(np.sum(nr >= r['r']) + 1) / (N_SIM_SPLIT + 1):.3f}"
        )

    emit()
    emit("1b. ROBUSTNESS OF THE SPLIT-HALF RESULT")
    keys = ("age_decade", "state")
    cur2 = curve(df, keys=keys)
    p2 = attach_p(df, cur2, keys=keys)
    sh2 = split_half(df, p2, rng, do_null=False)
    for st, r in sh2.items():
        lo, hi = r["ci"]
        emit(
            f"  finer baseline (age decade x state x hour), {st}: "
            f"r = {r['r']:+.4f} [{lo:+.4f}, {hi:+.4f}], n = {r['n']}"
        )
    sh3 = split_half(df, p, rng, is_a=time_halves(df), do_null=False)
    for st, r in sh3.items():
        lo, hi = r["ci"]
        emit(
            f"  chronological halves (early vs late), {st}: "
            f"r = {r['r']:+.4f} [{lo:+.4f}, {hi:+.4f}], n = {r['n']}"
        )

    emit()
    emit("2. VARIANCE DECOMPOSITION vs PARAMETRIC BOOTSTRAP NULL")
    vn = variance_null(df, p, rng)
    for st, r in vn.items():
        emit(
            f"  {st}: users with >={MIN_CELL} attempts per block: {r['n']}"
        )
        emit(
            f"      observed var = {r['obs_var']:.5f} "
            f"(sd {np.sqrt(r['obs_var']):.4f})"
        )
        emit(
            f"      null var     = {r['null_var']:.5f} "
            f"(sd {np.sqrt(r['null_var']):.4f}, sd of null var "
            f"{r['null_sd_of_var']:.5f})"
        )
        emit(
            f"      excess var   = {r['excess']:+.5f} -> implied true "
            f"between-user sd = {r['true_sd']:.4f}   p = {r['p']:.4f}"
        )

    emit()
    emit("3. HELD-OUT LOG-LOSS (30% of attempts, seed fixed)")
    rows = heldout(df, rng)
    base = rows[0][1]
    for label, ll in rows:
        emit(f"  {label:<34} log-loss = {ll:.5f}  delta = {ll - base:+.5f}")

    emit()
    emit("4. NAIVE PER-USER RATE CORRELATION (noise-dominated, for contrast)")
    nv = naive_rates(df)
    for key in ("pooled", *STRATA):
        r = nv[key]
        lo, hi = fisher_ci(r["r"], r["n"])
        emit(
            f"  {key:<9}: r(AM rate, PM rate) = {r['r']:+.4f} "
            f"95% CI [{lo:+.4f}, {hi:+.4f}]  n = {r['n']}"
        )

    confound_check(df)

    p1 = plot_split_half(sh)
    p2 = plot_null(vn)
    p3 = plot_naive(nv)
    emit()
    emit(f"wrote {p1.name}, {p2.name}, {p3.name}")
    (OUT / "stats.txt").write_text("\n".join(_LINES) + "\n")


if __name__ == "__main__":
    main()
