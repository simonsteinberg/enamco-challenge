"""Shared model logic for the best-time-to-call predictor.

The model has two stacked parts, in the order the EDA justified them:

1. A population model ``P(picked_up | local_hour, age)`` fitted with
   LightGBM. This carries the dominant signal: the bimodal hour curve
   and its inversion for customers aged 60 and over.
2. A per-user AM/PM correction, shrunk towards zero. The chronotype
   analysis found a real but second-order individual timing
   preference (split-half r = 0.30 under 60), worth roughly 1.5% of
   held-out log loss.

Predictions are ranked over customer-local hours and only then mapped
back to UTC, because the timezone analysis showed the hour curves of
all timezones collapse onto each other in local time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from challenge.feature import state_to_timezone

#: Local hours that ever appear in the historic data. Outside this
#: window we have zero evidence, so we never rank those hours.
CANDIDATE_LOCAL_HOURS = tuple(range(8, 21))

#: Local hour at which the afternoon/evening block starts.
PM_START_HOUR = 14

#: Ridge/James-Stein shrinkage strength for the per-user correction.
#: Tuned by ``scripts/sweep_shrinkage.py`` on held-out log loss, on a
#: within-user split so a user's offset is never fitted on the rows it
#: is scored against. The earlier value of 5.0 came from a standalone
#: chronotype baseline and is too weak against the LightGBM residuals:
#: it scores worse than applying no correction at all. The log-loss
#: curve is flat over k = 12..50, so 20 is taken as the conservative
#: middle of that plateau.
DEFAULT_SHRINKAGE = 20.0

FEATURES = ("local_hour", "age")


@dataclass
class UserProfile:
    """Per-user attributes needed at prediction time.

    Attributes:
        age: Customer age in years.
        state: US state abbreviation, used only to pick a timezone.
        am_offset: Additive probability correction for local hours
            before :data:`PM_START_HOUR`.
        pm_offset: Additive probability correction from
            :data:`PM_START_HOUR` onwards.
    """

    age: float
    state: str
    am_offset: float = 0.0
    pm_offset: float = 0.0


@dataclass
class BestTimeModel:
    """Population model plus per-user corrections.

    Attributes:
        booster: Fitted LightGBM classifier.
        profiles: Per-user attributes keyed by ``user_id``.
        fallback_age: Age used for users with no history.
        fallback_state: State used for users with no history.
    """

    booster: object
    profiles: dict[int, UserProfile] = field(default_factory=dict)
    fallback_age: float = 40.0
    fallback_state: str = "NY"

    def population_proba(
        self, ages: np.ndarray, hours: np.ndarray
    ) -> np.ndarray:
        """Predict pick-up probability for age/hour pairs.

        Args:
            ages: Customer ages.
            hours: Customer-local hours of day.

        Returns:
            Probability of pick-up for each pair.
        """
        frame = pd.DataFrame(
            {
                "local_hour": pd.Series(hours, dtype="int16"),
                "age": pd.Series(ages, dtype="float64"),
            }
        )
        return self.booster.predict_proba(frame)[:, 1]

    def profile(self, user_id: int) -> UserProfile:
        """Return a user's profile, or a population fallback.

        Args:
            user_id: Identifier from the eval targets.

        Returns:
            The stored profile, or a neutral fallback for an unseen
            user (no correction, population-median age).
        """
        return self.profiles.get(
            int(user_id),
            UserProfile(age=self.fallback_age, state=self.fallback_state),
        )

    def score_local_hours(self, user_id: int) -> pd.DataFrame:
        """Score every candidate local hour for one user.

        Args:
            user_id: Identifier from the eval targets.

        Returns:
            Frame with ``local_hour`` and ``proba``, sorted best
            first. Ties break towards the earlier hour so the output
            is deterministic.
        """
        prof = self.profile(user_id)
        hours = np.array(CANDIDATE_LOCAL_HOURS, dtype="int16")
        base = self.population_proba(
            np.full(hours.shape, prof.age, dtype="float64"), hours
        )
        offset = np.where(
            hours < PM_START_HOUR, prof.am_offset, prof.pm_offset
        )
        proba = np.clip(base + offset, 0.0, 1.0)
        out = pd.DataFrame({"local_hour": hours, "proba": proba})
        return out.sort_values(
            ["proba", "local_hour"], ascending=[False, True]
        ).reset_index(drop=True)

    def top_utc_hours(
        self, user_id: int, date: str, k: int = 3
    ) -> list[int]:
        """Return the top-k UTC hours to call a user on a date.

        Args:
            user_id: Identifier from the eval targets.
            date: Calendar date ``YYYY-MM-DD`` in customer-local time.
            k: How many hours to return.

        Returns:
            ``k`` distinct UTC hours, most-confident first.
        """
        prof = self.profile(user_id)
        ranked = self.score_local_hours(user_id)
        seen: list[int] = []
        for local_hour in ranked["local_hour"]:
            utc_hour = local_to_utc_hour(prof.state, date, int(local_hour))
            if utc_hour not in seen:
                seen.append(utc_hour)
            if len(seen) == k:
                break
        return seen


def local_to_utc_hour(state: str, date: str, local_hour: int) -> int:
    """Convert a local hour on a date to the matching UTC hour.

    The offset depends on the date, so daylight saving time is applied
    for exactly the day in question.

    Args:
        state: US state abbreviation, mapped to an IANA timezone.
        date: Calendar date ``YYYY-MM-DD`` in customer-local time.
        local_hour: Hour of day 0-23 in customer-local time.

    Returns:
        The corresponding UTC hour, 0-23.
    """
    tz = ZoneInfo(state_to_timezone(state))
    naive = pd.Timestamp(f"{date} {int(local_hour):02d}:00:00")
    local = naive.tz_localize(tz, nonexistent="shift_forward", ambiguous=True)
    return int(local.tz_convert("UTC").hour)


def user_corrections(
    frame: pd.DataFrame,
    base_proba: np.ndarray,
    shrinkage: float = DEFAULT_SHRINKAGE,
) -> pd.Series:
    """Fit shrunk per-user AM/PM corrections on model residuals.

    Each user gets the mean residual of their attempts in each block,
    pulled towards zero by ``shrinkage`` pseudo-observations. A user
    with few attempts therefore lands close to the population curve,
    which is what stops the thin per-user data from overfitting.

    Args:
        frame: Historic attempts with ``user_id``, ``local_hour`` and
            ``picked_up``.
        base_proba: Population probability for each row of ``frame``.
        shrinkage: Pseudo-count pulling each offset towards zero.

    Returns:
        Frame indexed by ``user_id`` with ``am_offset`` and
        ``pm_offset`` columns.
    """
    work = frame[["user_id", "local_hour", "picked_up"]].copy()
    work["residual"] = work["picked_up"].astype("float64") - base_proba
    work["block"] = np.where(
        work["local_hour"] < PM_START_HOUR, "am_offset", "pm_offset"
    )
    grouped = work.groupby(["user_id", "block"])["residual"].agg(
        ["sum", "count"]
    )
    grouped["offset"] = grouped["sum"] / (grouped["count"] + shrinkage)
    wide = grouped["offset"].unstack("block")
    for col in ("am_offset", "pm_offset"):
        if col not in wide.columns:
            wide[col] = 0.0
    return wide[["am_offset", "pm_offset"]].fillna(0.0)
