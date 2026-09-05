"""Predictions output format validation.

This is the single source of truth for the predictions.csv shape:

    user_id,date,hour_rank_1,hour_rank_2,hour_rank_3
    123,2025-04-15,18,17,19
    124,2025-04-15,9,10,8

* ``user_id``                  - integer
* ``date``                     - ISO date, ``YYYY-MM-DD``
* ``hour_rank_1/2/3``          - integers in [0, 23], no duplicates within
                                  a single (user, date)

Use the ``validate_predictions`` function programmatically, or run this
file as a script: ``python -m challenge.format predictions.csv``.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS: list[str] = [
    "user_id",
    "date",
    "hour_rank_1",
    "hour_rank_2",
    "hour_rank_3",
]
HOUR_COLUMNS: list[str] = ["hour_rank_1", "hour_rank_2", "hour_rank_3"]
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class FormatError(ValueError):
    """Raised when predictions.csv does not match the expected format."""


@dataclass
class ValidationResult:
    n_rows: int
    df: pd.DataFrame  # the loaded, type-coerced predictions, ready to score


def _check(condition: bool, msg: str) -> None:
    if not condition:
        raise FormatError(msg)


def validate_predictions(path: Path | str) -> ValidationResult:
    """Load and validate a predictions CSV.

    Returns the loaded dataframe (with ``user_id`` and hour columns coerced
    to ``int``) so that callers don't re-parse it. Raises ``FormatError``
    with a clear, actionable message on any violation.
    """
    p = Path(path)
    _check(p.exists(), f"predictions file not found: {p}")

    try:
        df = pd.read_csv(p)
    except Exception as exc:
        raise FormatError(f"could not parse {p} as CSV: {exc}") from exc

    # Header check.
    actual = list(df.columns)
    _check(
        actual == REQUIRED_COLUMNS,
        f"header mismatch.\n  expected: {REQUIRED_COLUMNS}\n  found:    {actual}",
    )

    _check(len(df) > 0, "predictions file is empty (no rows after the header).")

    # No NaN anywhere.
    null_counts = df.isnull().sum()
    nulls = null_counts[null_counts > 0].to_dict()
    _check(not nulls, f"null values present in columns: {nulls}")

    # user_id must be integer-like.
    try:
        df["user_id"] = df["user_id"].astype(int)
    except (ValueError, TypeError) as exc:
        raise FormatError(f"user_id is not all-integer: {exc}") from exc

    # date must look like YYYY-MM-DD and parse cleanly.
    bad_dates = df.loc[~df["date"].astype(str).str.match(DATE_RE), "date"]
    if not bad_dates.empty:
        raise FormatError(
            f"{len(bad_dates)} date value(s) not in YYYY-MM-DD format. "
            f"first bad: {bad_dates.iloc[0]!r}"
        )
    try:
        pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
    except Exception as exc:
        raise FormatError(f"date column did not parse as ISO dates: {exc}") from exc

    # Hour columns must be ints in [0, 23].
    for col in HOUR_COLUMNS:
        try:
            df[col] = df[col].astype(int)
        except (ValueError, TypeError) as exc:
            raise FormatError(f"{col} is not all-integer: {exc}") from exc
        out_of_range = df.loc[~df[col].between(0, 23), col]
        if not out_of_range.empty:
            raise FormatError(
                f"{col} contains {len(out_of_range)} value(s) outside [0, 23]. "
                f"first bad: {out_of_range.iloc[0]!r}"
            )

    # Within a row, the 3 hour ranks must be distinct -- otherwise rank order
    # is meaningless.
    dup_mask = (
        (df["hour_rank_1"] == df["hour_rank_2"])
        | (df["hour_rank_1"] == df["hour_rank_3"])
        | (df["hour_rank_2"] == df["hour_rank_3"])
    )
    _check(
        not dup_mask.any(),
        f"{int(dup_mask.sum())} row(s) have duplicate hours among hour_rank_1/2/3. "
        f"each (user, date) prediction must rank 3 distinct hours.",
    )

    # No duplicate (user_id, date) keys.
    dup_keys = df.duplicated(subset=["user_id", "date"], keep=False)
    _check(
        not dup_keys.any(),
        f"{int(dup_keys.sum())} duplicate (user_id, date) row(s). "
        f"each (user, date) must appear at most once.",
    )

    return ValidationResult(n_rows=len(df), df=df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate that a predictions CSV has the right format.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="predictions.csv",
        help="path to the predictions file (default: predictions.csv)",
    )
    args = parser.parse_args()

    try:
        result = validate_predictions(args.path)
    except FormatError as exc:
        print(f"[validate] FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[validate] OK -- {result.n_rows} rows, format looks good.")


if __name__ == "__main__":
    main()
