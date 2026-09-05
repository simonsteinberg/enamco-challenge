"""Sweep the per-user shrinkage k against a top-3 rank metric.

Held-out log loss is only a proxy for the graders' undisclosed rank
measure, so this scores the ranking directly: hold out each user's most
recent 30% of attempts, rank their local hours, then measure the pick-up
rate among held-out attempts that fall in the model's top-3 and rank-1
hours. Confirms the k chosen by scripts/sweep_shrinkage.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from challenge.feature import add_local_time
from challenge.model import (
    CANDIDATE_LOCAL_HOURS,
    PM_START_HOUR,
    user_corrections,
)
from challenge.train import FEATURES, fit_population_model

GRID = [0, 1, 2, 3, 5, 8, 12, 20, 35, 50, float("inf")]
NO_CORRECTION = 1e9


def split_within_user(
    frame: pd.DataFrame, fit_fraction: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each user's attempts chronologically.

    Args:
        frame: Historic attempts with local time columns added.
        fit_fraction: Share of each user's earliest attempts used to
            fit the population model and the per-user offsets.

    Returns:
        The fit rows and the held-out score rows.
    """
    ordered = frame.sort_values(
        ["user_id", "attempted_at_utc"]
    ).reset_index(drop=True)
    position = ordered.groupby("user_id").cumcount()
    size = ordered.groupby("user_id")["user_id"].transform("size")
    cutoff = np.ceil(size * fit_fraction)
    return ordered[position < cutoff].copy(), ordered[position >= cutoff]


def main() -> None:
    """Print the rank metric for every shrinkage value in the grid."""
    frame = add_local_time(pd.read_csv("data/historic.csv"))
    fit, score = split_within_user(frame)

    population = fit_population_model(fit)
    base_proba = population.predict_proba(fit[list(FEATURES)])[:, 1]

    hours = np.array(CANDIDATE_LOCAL_HOURS)
    users = score["user_id"].unique()
    ages = fit.groupby("user_id")["age"].last().reindex(users)
    grid = pd.DataFrame(
        {
            "local_hour": np.tile(hours, len(users)),
            "age": np.repeat(ages.values, len(hours)),
        }
    )
    curves = population.predict_proba(grid[list(FEATURES)])[:, 1].reshape(
        len(users), len(hours)
    )
    index = pd.Series(np.arange(len(users)), index=users)
    row_of = index.reindex(score["user_id"]).to_numpy()
    hour_of = score["local_hour"].to_numpy()
    truth = score["picked_up"].astype(int).to_numpy()
    overall = truth.mean()

    print(f"{'k':>6} {'top3_rate':>10} {'top1_rate':>10} {'lift':>7}")
    for k in GRID:
        shrinkage = NO_CORRECTION if np.isinf(k) else k
        corrections = user_corrections(fit, base_proba, shrinkage=shrinkage)
        am = corrections["am_offset"].reindex(users).fillna(0.0)
        pm = corrections["pm_offset"].reindex(users).fillna(0.0)
        offset = np.where(
            hours[None, :] < PM_START_HOUR,
            am.to_numpy()[:, None],
            pm.to_numpy()[:, None],
        )
        proba = np.clip(curves + offset, 0.0, 1.0)
        order = np.argsort(-proba, axis=1, kind="stable")
        ranked = hours[order]
        in_top3 = (ranked[row_of, :3] == hour_of[:, None]).any(axis=1)
        in_top1 = ranked[row_of, 0] == hour_of
        rate3 = truth[in_top3].mean()
        rate1 = truth[in_top1].mean()
        label = "none" if np.isinf(k) else f"{k:g}"
        print(
            f"{label:>6} {rate3:>10.4f} {rate1:>10.4f} "
            f"{rate3 / overall:>7.3f}"
        )
    print(f"overall held-out pick-up rate: {overall:.4f}")


if __name__ == "__main__":
    main()
