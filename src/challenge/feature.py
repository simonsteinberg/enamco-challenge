"""Customer-local time features.

Converts UTC call attempts to the customer's local wall-clock time
using the stdlib :mod:`zoneinfo` database, so US daylight saving
transitions are applied automatically per date.

Caveats on the state -> timezone mapping:

* Several states span more than one IANA zone. We assign each state
  the zone covering the majority of its population, which is an
  approximation for users living in the minority zone. Affected
  states: AK, FL, ID, IN, KS, KY, MI, ND, NE, OR, SD, TN, TX.
* Arizona (except the Navajo Nation) and Hawaii do not observe
  daylight saving time. ``America/Phoenix`` and ``Pacific/Honolulu``
  encode that, so their UTC offset is constant all year.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

#: States assigned a majority-population zone despite spanning more
#: than one IANA timezone.
MULTI_ZONE_STATES: frozenset[str] = frozenset(
    {
        "AK",
        "FL",
        "ID",
        "IN",
        "KS",
        "KY",
        "MI",
        "ND",
        "NE",
        "OR",
        "SD",
        "TN",
        "TX",
    }
)

#: States that do not observe US daylight saving time.
NO_DST_STATES: frozenset[str] = frozenset({"AZ", "HI"})

#: US state (plus DC) abbreviation -> IANA timezone.
STATE_TIMEZONES: dict[str, str] = {
    # Eastern
    "CT": "America/New_York",
    "DC": "America/New_York",
    "DE": "America/New_York",
    "FL": "America/New_York",  # panhandle is Central
    "GA": "America/New_York",
    "IN": "America/New_York",  # NW/SW corners are Central
    "KY": "America/New_York",  # western KY is Central
    "MA": "America/New_York",
    "MD": "America/New_York",
    "ME": "America/New_York",
    "MI": "America/New_York",  # western UP is Central
    "NC": "America/New_York",
    "NH": "America/New_York",
    "NJ": "America/New_York",
    "NY": "America/New_York",
    "OH": "America/New_York",
    "PA": "America/New_York",
    "RI": "America/New_York",
    "SC": "America/New_York",
    "VA": "America/New_York",
    "VT": "America/New_York",
    "WV": "America/New_York",
    # Central
    "AL": "America/Chicago",
    "AR": "America/Chicago",
    "IA": "America/Chicago",
    "IL": "America/Chicago",
    "KS": "America/Chicago",  # far west is Mountain
    "LA": "America/Chicago",
    "MN": "America/Chicago",
    "MO": "America/Chicago",
    "MS": "America/Chicago",
    "ND": "America/Chicago",  # southwest is Mountain
    "NE": "America/Chicago",  # panhandle is Mountain
    "OK": "America/Chicago",
    "SD": "America/Chicago",  # west river is Mountain
    "TN": "America/Chicago",  # east TN is Eastern
    "TX": "America/Chicago",  # El Paso area is Mountain
    "WI": "America/Chicago",
    # Mountain
    "CO": "America/Denver",
    "ID": "America/Denver",  # northern panhandle is Pacific
    "MT": "America/Denver",
    "NM": "America/Denver",
    "UT": "America/Denver",
    "WY": "America/Denver",
    # Mountain, no DST
    "AZ": "America/Phoenix",
    # Pacific
    "CA": "America/Los_Angeles",
    "NV": "America/Los_Angeles",
    "OR": "America/Los_Angeles",  # far eastern OR is Mountain
    "WA": "America/Los_Angeles",
    # Alaska / Hawaii
    "AK": "America/Anchorage",  # far Aleutians are HST
    "HI": "Pacific/Honolulu",
}

#: Fallback zone for unknown state codes.
DEFAULT_TIMEZONE = "America/New_York"

LOCAL_TIME_COLUMNS = (
    "timezone",
    "local_datetime",
    "local_hour",
    "local_dow",
    "is_dst",
    "utc_offset_hours",
)


def state_to_timezone(state: str) -> str:
    """Map a US state abbreviation to an IANA timezone name.

    Args:
        state: Two-letter state or district abbreviation, e.g. ``OH``.
            Case and surrounding whitespace are ignored.

    Returns:
        IANA timezone name, e.g. ``America/New_York``. Unknown codes
        fall back to :data:`DEFAULT_TIMEZONE`.
    """
    if not isinstance(state, str):
        return DEFAULT_TIMEZONE
    return STATE_TIMEZONES.get(state.strip().upper(), DEFAULT_TIMEZONE)


def utc_offset_hours(state: str, when: pd.Timestamp) -> float:
    """Return the UTC offset in hours for a state at a UTC instant.

    Args:
        state: Two-letter state abbreviation.
        when: Timestamp of the call attempt. Naive values are treated
            as UTC.

    Returns:
        Signed offset in hours, e.g. ``-4.0`` for Eastern in summer.
    """
    ts = pd.Timestamp(when)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    local = ts.tz_convert(ZoneInfo(state_to_timezone(state)))
    return local.utcoffset().total_seconds() / 3600.0


def add_local_time(
    df: pd.DataFrame,
    state_col: str = "state",
    utc_col: str = "attempted_at_utc",
) -> pd.DataFrame:
    """Add customer-local time columns to a call-attempt frame.

    Groups rows by timezone and uses :meth:`~pandas.Series.dt.tz_convert`
    once per group, which is deterministic and far faster than a
    row-wise apply. DST is resolved per timestamp by ``zoneinfo``.

    Args:
        df: Frame with a state column and a UTC timestamp column.
        state_col: Name of the state abbreviation column.
        utc_col: Name of the UTC timestamp column. Strings are parsed;
            naive timestamps are assumed to be UTC.

    Returns:
        A copy of ``df`` with added columns: ``timezone``,
        ``local_datetime`` (tz-naive local wall clock),
        ``local_hour`` (0-23), ``local_dow`` (0=Monday),
        ``is_dst`` (bool), ``utc_offset_hours`` (float).
    """
    out = df.copy()
    utc = pd.to_datetime(out[utc_col], utc=True, format="mixed")
    out["timezone"] = out[state_col].map(state_to_timezone)

    local = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    offset = pd.Series(float("nan"), index=out.index, dtype="float64")
    dst = pd.Series(False, index=out.index, dtype="bool")

    for tz_name, idx in out.groupby("timezone", sort=True).groups.items():
        conv = utc.loc[idx].dt.tz_convert(ZoneInfo(str(tz_name)))
        off = conv.map(lambda t: t.utcoffset().total_seconds() / 3600.0)
        offset.loc[idx] = off.astype("float64")
        dst.loc[idx] = conv.map(lambda t: t.dst().total_seconds() != 0).astype(
            "bool"
        )
        local.loc[idx] = conv.dt.tz_localize(None)

    out["local_datetime"] = local
    out["local_hour"] = local.dt.hour.astype("int16")
    out["local_dow"] = local.dt.dayofweek.astype("int8")
    out["is_dst"] = dst
    out["utc_offset_hours"] = offset
    return out
