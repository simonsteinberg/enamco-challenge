"""Contract tests for the model and the prediction pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from challenge.model import (
    CANDIDATE_LOCAL_HOURS,
    PM_START_HOUR,
    BestTimeModel,
    UserProfile,
    local_to_utc_hour,
    user_corrections,
)


class ConstantBooster:
    """Stub booster returning a fixed curve peaking at 19:00."""

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a two-column probability array."""
        hours = frame["local_hour"].to_numpy()
        p = np.where(hours == 19, 0.9, 0.1)
        return np.column_stack([1.0 - p, p])


@pytest.fixture
def model() -> BestTimeModel:
    return BestTimeModel(
        booster=ConstantBooster(),
        profiles={
            1: UserProfile(age=30.0, state="NY"),
            2: UserProfile(age=70.0, state="CA"),
        },
        fallback_age=40.0,
        fallback_state="NY",
    )


def test_local_to_utc_applies_dst():
    # New York is UTC-4 in summer and UTC-5 in winter.
    assert local_to_utc_hour("NY", "2025-07-15", 10) == 14
    assert local_to_utc_hour("NY", "2025-01-15", 10) == 15


def test_local_to_utc_no_dst_state_is_stable():
    # Arizona does not observe DST, so the offset never moves.
    assert local_to_utc_hour("AZ", "2025-07-15", 10) == 17
    assert local_to_utc_hour("AZ", "2025-01-15", 10) == 17


def test_local_to_utc_wraps_past_midnight():
    hour = local_to_utc_hour("CA", "2025-07-15", 20)
    assert 0 <= hour <= 23
    assert hour == 3


def test_top_hours_are_three_and_distinct(model):
    hours = model.top_utc_hours(1, "2025-07-15")
    assert len(hours) == 3
    assert len(set(hours)) == 3
    assert all(0 <= h <= 23 for h in hours)


def test_best_local_hour_maps_to_expected_utc(model):
    # Peak local hour is 19:00; NY in July is UTC-4, so 23:00 UTC.
    assert model.top_utc_hours(1, "2025-07-15")[0] == 23
    # Same local peak for the CA user, but UTC-7, so 02:00 UTC.
    assert model.top_utc_hours(2, "2025-07-15")[0] == 2


def test_only_candidate_hours_are_ranked(model):
    ranked = model.score_local_hours(1)
    assert set(ranked["local_hour"]) == set(CANDIDATE_LOCAL_HOURS)


def test_unseen_user_falls_back_without_crashing(model):
    hours = model.top_utc_hours(999, "2025-07-15")
    assert len(hours) == 3
    assert len(set(hours)) == 3


def test_am_offset_can_flip_the_ranking(model):
    model.profiles[3] = UserProfile(
        age=30.0, state="NY", am_offset=0.85, pm_offset=0.0
    )
    best_local = int(model.score_local_hours(3).loc[0, "local_hour"])
    assert best_local < PM_START_HOUR


def test_scoring_is_deterministic(model):
    first = model.top_utc_hours(1, "2025-07-15")
    second = model.top_utc_hours(1, "2025-07-15")
    assert first == second


def test_user_corrections_shrink_towards_zero():
    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "local_hour": [9, 9, 9],
            "picked_up": [True, True, True],
        }
    )
    base = np.array([0.5, 0.5, 0.5])
    out = user_corrections(frame, base, shrinkage=5.0)
    # User 1 has two positive residuals, user 2 only one, so user 1
    # moves further from zero. Neither reaches the raw residual 0.5.
    assert out.loc[1, "am_offset"] > out.loc[2, "am_offset"] > 0
    assert out.loc[1, "am_offset"] < 0.5
    assert (out["pm_offset"] == 0.0).all()


def test_user_corrections_stronger_shrinkage_is_smaller():
    frame = pd.DataFrame(
        {
            "user_id": [1, 1],
            "local_hour": [9, 9],
            "picked_up": [True, True],
        }
    )
    base = np.array([0.5, 0.5])
    weak = user_corrections(frame, base, shrinkage=1.0)
    strong = user_corrections(frame, base, shrinkage=100.0)
    assert weak.loc[1, "am_offset"] > strong.loc[1, "am_offset"] > 0
