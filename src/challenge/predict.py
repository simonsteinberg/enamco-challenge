"""Prediction entrypoint.

Loads the trained model, ranks customer-local hours per user, then
converts the top 3 back to UTC hours for the requested date so that
daylight saving time is applied for exactly that day.

Output format (UTC integer hours, 0-23)::

    user_id,date,hour_rank_1,hour_rank_2,hour_rank_3
    123,2025-04-15,18,17,19
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib

from challenge.model import BestTimeModel
from challenge.train import MODEL_PATH


def load_model(path: str | Path = MODEL_PATH) -> BestTimeModel:
    """Load the trained model artifact.

    Args:
        path: Path to the joblib artifact written by ``train.py``.

    Returns:
        The deserialized model.

    Raises:
        SystemExit: If the artifact is missing, i.e. train did not run.
    """
    model_path = Path(path)
    if not model_path.exists():
        raise SystemExit(
            f"[predict] model not found: {model_path}. Run `make train`."
        )
    return joblib.load(model_path)


def main() -> None:
    """Write top-3 UTC call hours for every eval target."""
    parser = argparse.ArgumentParser(
        description="Predict best call hours."
    )
    parser.add_argument("--historic", default="data/historic.csv")
    parser.add_argument("--eval-targets", default="data/eval_targets.csv")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()

    eval_path = Path(args.eval_targets)
    if not eval_path.exists():
        raise SystemExit(f"[predict] eval targets not found: {eval_path}")

    model = load_model(args.model)

    with eval_path.open() as f:
        rows = list(csv.DictReader(f))

    unseen = 0
    with Path(args.output).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["user_id", "date", "hour_rank_1", "hour_rank_2", "hour_rank_3"]
        )
        for row in rows:
            user_id = int(row["user_id"])
            if user_id not in model.profiles:
                unseen += 1
            hours = model.top_utc_hours(user_id, row["date"])
            writer.writerow([user_id, row["date"], *hours])

    print(
        f"[predict] wrote {len(rows)} predictions to {args.output} "
        f"({unseen} users had no history)"
    )


if __name__ == "__main__":
    main()
