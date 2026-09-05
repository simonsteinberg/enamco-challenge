"""Sweep the per-user shrinkage strength against LightGBM residuals.

The production model stacks a LightGBM population curve and a
per-user AM/PM correction whose strength is set by
``challenge.model.DEFAULT_SHRINKAGE`` (``k``). This script picks
``k`` by held-out log loss.

Why the split is within-user and chronological
----------------------------------------------
The correction for a user is fitted from that user's own attempts,
so a plain random row split leaks: the same attempt would set the
offset and then be scored by it. We therefore split *each user's*
attempts into a fit part and a score part. The primary split is
chronological (each user's earliest 70% of attempts fit, latest 30%
score), which matches deployment: we always predict forward from a
user's history. A random within-user split is reported as a
robustness check, since the chronological one also absorbs any drift
over the campaign window.

The population model is fitted on the fit rows only, its residuals
on the fit rows drive the offsets, and every metric is computed on
the untouched score rows.

Run with::

    uv run python scripts/sweep_shrinkage.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from challenge.feature import add_local_time
from challenge.model import (
    CANDIDATE_LOCAL_HOURS,
    FEATURES,
    PM_START_HOUR,
    user_corrections,
)
from challenge.train import fit_population_model

HISTORIC = "data/historic.csv"
SEED = 42
SCORE_FRACTION = 0.3
EPS = 1e-6

SHRINKAGE_GRID = (
    0.0,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    8.0,
    12.0,
    20.0,
    35.0,
    50.0,
    100.0,
    200.0,
    float("inf"),
)


def split_within_user(
    frame: pd.DataFrame, mode: str, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split every user's attempts into fit and score parts.

    Users with a single attempt contribute it to the fit side only,
    because they cannot be scored without leaking.

    Args:
        frame: Historic attempts with ``user_id`` and
            ``attempted_at_utc``.
        mode: ``"chronological"`` keeps each user's earliest attempts
            for fitting; ``"random"`` shuffles within the user first.
        seed: Seed for the random mode.

    Returns:
        The fit frame and the score frame.
    """
    work = frame.copy()
    if mode == "chronological":
        order = work.sort_values(
            ["user_id", "attempted_at_utc"], kind="mergesort"
        )
    elif mode == "random":
        order = work.sample(frac=1.0, random_state=seed).sort_values(
            "user_id", kind="mergesort"
        )
    else:
        raise ValueError(f"unknown split mode: {mode}")

    rank = order.groupby("user_id").cumcount()
    size = order.groupby("user_id")["user_id"].transform("size")
    n_score = np.minimum(
        np.maximum(np.round(size * SCORE_FRACTION), 1), size - 1
    ).clip(lower=0)
    is_score = rank >= (size - n_score)
    return order[~is_score].copy(), order[is_score].copy()


def offsets_for_rows(
    frame: pd.DataFrame, corrections: pd.DataFrame
) -> np.ndarray:
    """Look up each row's AM or PM offset.

    Args:
        frame: Rows with ``user_id`` and ``local_hour``.
        corrections: Output of
            :func:`challenge.model.user_corrections`.

    Returns:
        The additive probability offset for each row; zero for users
        absent from ``corrections``.
    """
    joined = frame[["user_id", "local_hour"]].join(
        corrections, on="user_id"
    )
    joined = joined.fillna({"am_offset": 0.0, "pm_offset": 0.0})
    return np.where(
        joined["local_hour"].to_numpy() < PM_START_HOUR,
        joined["am_offset"].to_numpy(),
        joined["pm_offset"].to_numpy(),
    )


def top_block_by_model(
    ages: np.ndarray,
    am_offset: np.ndarray,
    pm_offset: np.ndarray,
    booster: object,
) -> np.ndarray:
    """Return the block of each user's top-ranked local hour.

    Mirrors :meth:`challenge.model.BestTimeModel.score_local_hours`:
    the offset is added to the population probability, clipped to
    ``[0, 1]``, and ties break towards the earlier hour.

    Args:
        ages: One age per user.
        am_offset: Per-user AM offset.
        pm_offset: Per-user PM offset.
        booster: Fitted population classifier.

    Returns:
        ``"am"`` or ``"pm"`` for each user.
    """
    hours = np.array(CANDIDATE_LOCAL_HOURS, dtype="int16")
    grid = pd.DataFrame(
        {
            "local_hour": pd.Series(
                np.tile(hours, len(ages)), dtype="int16"
            ),
            "age": pd.Series(
                np.repeat(ages, len(hours)), dtype="float64"
            ),
        }
    )
    base = booster.predict_proba(grid[list(FEATURES)])[:, 1]
    base = base.reshape(len(ages), len(hours))
    is_am = hours < PM_START_HOUR
    offset = np.where(is_am[None, :], am_offset[:, None], pm_offset[:, None])
    proba = np.clip(base + offset, 0.0, 1.0)
    best = proba.argmax(axis=1)  # argmax takes the earliest tie
    return np.where(is_am[best], "am", "pm")


def block_truth(score: pd.DataFrame) -> pd.DataFrame:
    """Per-user held-out AM and PM pick-up rates.

    Args:
        score: Held-out attempts with ``user_id``, ``local_hour`` and
            ``picked_up``.

    Returns:
        Frame indexed by ``user_id`` with ``am_rate``, ``pm_rate``
        and ``better_block``, restricted to users with at least one
        held-out attempt in each block and a strict difference
        between the two rates.
    """
    work = score[["user_id", "local_hour", "picked_up"]].copy()
    work["block"] = np.where(
        work["local_hour"] < PM_START_HOUR, "am_rate", "pm_rate"
    )
    rates = (
        work.groupby(["user_id", "block"])["picked_up"]
        .mean()
        .unstack("block")
    )
    for col in ("am_rate", "pm_rate"):
        if col not in rates.columns:
            rates[col] = np.nan
    rates = rates.dropna()
    rates = rates[rates["am_rate"] != rates["pm_rate"]]
    rates["better_block"] = np.where(
        rates["am_rate"] > rates["pm_rate"], "am", "pm"
    )
    return rates


def sweep(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Score every shrinkage value on a within-user held-out split.

    Args:
        frame: Full historic frame with local-time columns.
        mode: Split mode passed to :func:`split_within_user`.

    Returns:
        One row per shrinkage value with ``log_loss``, ``auc``,
        ``block_accuracy`` and ``se_vs_base``, the paired standard
        error of the log-loss difference against the no-correction
        baseline.
    """
    fit, score = split_within_user(frame, mode)
    booster = fit_population_model(fit)
    fit_base = booster.predict_proba(fit[list(FEATURES)])[:, 1]
    score_base = booster.predict_proba(score[list(FEATURES)])[:, 1]
    truth = score["picked_up"].astype(int).to_numpy()

    rates = block_truth(score)
    eval_users = rates.index.to_numpy()
    ages = (
        fit.groupby("user_id")["age"].last().reindex(eval_users).to_numpy()
    )
    better = rates["better_block"].to_numpy()

    rows = []
    losses: dict[float, np.ndarray] = {}
    for k in SHRINKAGE_GRID:
        corr = user_corrections(fit, fit_base, k)
        proba = np.clip(score_base + offsets_for_rows(score, corr), 0.0, 1.0)
        user_corr = corr.reindex(eval_users).fillna(0.0)
        picked = top_block_by_model(
            ages,
            user_corr["am_offset"].to_numpy(),
            user_corr["pm_offset"].to_numpy(),
            booster,
        )
        safe = np.clip(proba, EPS, 1.0 - EPS)
        losses[k] = -(
            truth * np.log(safe) + (1 - truth) * np.log(1.0 - safe)
        )
        rows.append(
            {
                "k": k,
                "log_loss": float(losses[k].mean()),
                "auc": roc_auc_score(truth, proba),
                "block_accuracy": float((picked == better).mean()),
            }
        )
    out = pd.DataFrame(rows)
    base_loss = losses[float("inf")]
    out["se_vs_base"] = [
        float(np.std(losses[k] - base_loss, ddof=1) / np.sqrt(len(truth)))
        for k in out["k"]
    ]
    out.attrs["n_fit"] = len(fit)
    out.attrs["n_score"] = len(score)
    out.attrs["n_block_users"] = len(eval_users)
    return out


def report(table: pd.DataFrame, mode: str) -> float:
    """Print one sweep table and return its best shrinkage value.

    Args:
        table: Output of :func:`sweep`.
        mode: Split mode, for the header.

    Returns:
        The shrinkage value with the lowest held-out log loss.
    """
    baseline = table.loc[table["k"] == float("inf")].iloc[0]
    best = table.loc[table["log_loss"].idxmin()]
    print(f"\n=== within-user {mode} split (seed {SEED}) ===")
    print(
        f"fit rows {table.attrs['n_fit']}, score rows "
        f"{table.attrs['n_score']}, users in block diagnostic "
        f"{table.attrs['n_block_users']}"
    )
    print(
        f"{'k':>8} {'log_loss':>10} {'vs base':>9} {'se':>8} "
        f"{'auc':>8} {'block_acc':>10}"
    )
    for _, row in table.iterrows():
        label = "inf" if np.isinf(row["k"]) else f"{row['k']:g}"
        delta = row["log_loss"] - baseline["log_loss"]
        note = "  <- no correction (baseline)" if np.isinf(row["k"]) else ""
        print(
            f"{label:>8} {row['log_loss']:>10.5f} {delta:>+9.5f} "
            f"{row['se_vs_base']:>8.5f} {row['auc']:>8.5f} "
            f"{row['block_accuracy']:>10.4f}{note}"
        )
    gain = baseline["log_loss"] - best["log_loss"]
    print(
        f"argmin k = {best['k']:g}: log loss {best['log_loss']:.5f}, "
        f"{gain:.5f} ({100 * gain / baseline['log_loss']:.2f}%) better "
        f"than no correction"
    )
    return float(best["k"])


def main() -> None:
    """Run the sweep on both split modes and print the tables."""
    frame = add_local_time(pd.read_csv(HISTORIC))
    print(
        f"[sweep] {len(frame)} attempts, "
        f"{frame['user_id'].nunique()} users"
    )
    print(__doc__.split("Run with")[0].strip())
    for mode in ("chronological", "random"):
        report(sweep(frame, mode), mode)


if __name__ == "__main__":
    main()
