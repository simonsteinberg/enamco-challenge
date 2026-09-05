"""Prediction entrypoint.

This is a stub. It reads the eval targets and emits predictions.csv with the
constant ranking ``[10, 14, 18]`` (a naive business-hours guess). Replace the
``rank_hours`` function with your real model.

Output format (UTC integer hours, 0-23):

    user_id,date,hour_rank_1,hour_rank_2,hour_rank_3
    123,2025-04-15,18,17,19
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def rank_hours(user_id: int, date: str) -> tuple[int, int, int]:
    """Return the top-3 UTC hours to call this user on this date.

    Stub: returns (10, 14, 18) for everyone. Replace with your model.
    """
    return (10, 14, 18)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict best call hours.")
    parser.add_argument("--historic", default="data/historic.csv")
    parser.add_argument("--eval-targets", default="data/eval_targets.csv")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    eval_path = Path(args.eval_targets)
    if not eval_path.exists():
        raise SystemExit(f"[predict] eval targets not found: {eval_path}")

    with eval_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with Path(args.output).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["user_id", "date", "hour_rank_1", "hour_rank_2", "hour_rank_3"]
        )
        for row in rows:
            user_id = int(row["user_id"])
            date = row["date"]
            h1, h2, h3 = rank_hours(user_id, date)
            writer.writerow([user_id, date, h1, h2, h3])

    print(f"[predict] stub: wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
