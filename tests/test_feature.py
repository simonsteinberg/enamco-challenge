"""Tests for customer-local time features."""

from __future__ import annotations

import pandas as pd
import pytest

from challenge.feature import (
    NO_DST_STATES,
    STATE_TIMEZONES,
    add_local_time,
    state_to_timezone,
    utc_offset_hours,
)

SUMMER = "2025-07-15T18:00:00Z"
WINTER = "2026-01-15T18:00:00Z"


def test_mapping_covers_50_states_and_dc():
    assert len(STATE_TIMEZONES) == 51
    assert "DC" in STATE_TIMEZONES


def test_state_to_timezone_normalises_input():
    assert state_to_timezone(" oh ") == "America/New_York"
    assert state_to_timezone("CA") == "America/Los_Angeles"
    assert state_to_timezone("ZZ") == "America/New_York"
    assert state_to_timezone(None) == "America/New_York"


@pytest.mark.parametrize(
    ("state", "tz"),
    [("AZ", "America/Phoenix"), ("HI", "Pacific/Honolulu")],
)
def test_no_dst_states(state, tz):
    assert state in NO_DST_STATES
    assert STATE_TIMEZONES[state] == tz
    assert utc_offset_hours(state, SUMMER) == utc_offset_hours(state, WINTER)


def test_dst_shifts_offset_for_eastern_and_pacific():
    assert utc_offset_hours("OH", SUMMER) == -4.0
    assert utc_offset_hours("OH", WINTER) == -5.0
    assert utc_offset_hours("CA", SUMMER) == -7.0
    assert utc_offset_hours("CA", WINTER) == -8.0


def test_add_local_time_columns_and_values():
    df = pd.DataFrame(
        {
            "state": ["OH", "OH", "AZ", "AZ", "HI"],
            "attempted_at_utc": [SUMMER, WINTER, SUMMER, WINTER, SUMMER],
        }
    )
    out = add_local_time(df)
    assert list(out.local_hour) == [14, 13, 11, 11, 8]
    assert list(out.is_dst) == [True, False, False, False, False]
    assert list(out.utc_offset_hours) == [-4.0, -5.0, -7.0, -7.0, -10.0]
    assert out.local_datetime.dt.tz is None
    assert out.local_dow.between(0, 6).all()
    # Original frame is untouched.
    assert "local_hour" not in df.columns


def test_add_local_time_is_deterministic_and_order_preserving():
    df = pd.DataFrame(
        {
            "state": ["CA", "OH", "CA", "AZ"],
            "attempted_at_utc": [SUMMER] * 4,
        }
    )
    a = add_local_time(df)
    b = add_local_time(df)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.state) == ["CA", "OH", "CA", "AZ"]
    assert list(a.local_hour) == [11, 14, 11, 11]


def test_accepts_tz_aware_and_naive_inputs():
    naive = pd.DataFrame(
        {
            "state": ["OH"],
            "attempted_at_utc": [pd.Timestamp("2025-07-15 18:00")],
        }
    )
    aware = pd.DataFrame(
        {"state": ["OH"], "attempted_at_utc": [pd.Timestamp(SUMMER)]}
    )
    assert add_local_time(naive).local_hour[0] == 14
    assert add_local_time(aware).local_hour[0] == 14


def test_custom_column_names():
    df = pd.DataFrame({"st": ["OH"], "ts": [SUMMER]})
    out = add_local_time(df, state_col="st", utc_col="ts")
    assert out.local_hour[0] == 14
