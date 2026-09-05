"""Training entrypoint for the best-time-to-call model.

Fits the population model ``P(picked_up | local_hour, age)`` with
LightGBM, then fits shrunk per-user AM/PM corrections on its
residuals, and writes both to ``artifacts/model.joblib``.

The user profiles (age, state, corrections) are stored in the same
artifact so ``predict.py`` does not depend on re-reading the historic
file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from challenge.feature import add_local_time
from challenge.model import (
    DEFAULT_SHRINKAGE,
    FEATURES,
    BestTimeModel,
    UserProfile,
    user_corrections,
)

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
SEED = 42


def load_training_frame(historic: str | Path) -> pd.DataFrame:
    """Load historic attempts and add customer-local time columns.

    Args:
        historic: Path to the historic attempts CSV.

    Returns:
        The frame with ``local_hour`` and friends added.
    """
    raw = pd.read_csv(historic)
    return add_local_time(raw)


def fit_population_model(frame: pd.DataFrame) -> LGBMClassifier:
    """Fit the LightGBM population model.

    ``local_hour`` is passed as a categorical feature so the model can
    express the bimodal curve and its inversion for older customers
    without being forced into a monotone shape.

    Args:
        frame: Training frame with ``local_hour``, ``age`` and
            ``picked_up``.

    Returns:
        The fitted classifier.
    """
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        random_state=SEED,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    features = frame[list(FEATURES)]
    model.fit(
        features,
        frame["picked_up"].astype(int),
        categorical_feature=["local_hour"],
    )
    return model


def build_profiles(
    frame: pd.DataFrame, corrections: pd.DataFrame
) -> dict[int, UserProfile]:
    """Assemble per-user profiles from history and corrections.

    Args:
        frame: Historic attempts with ``user_id``, ``age``, ``state``.
        corrections: Output of
            :func:`challenge.model.user_corrections`.

    Returns:
        Mapping of ``user_id`` to :class:`UserProfile`.
    """
    latest = frame.sort_values("attempted_at_utc").groupby("user_id").last()
    joined = latest.join(corrections, how="left").fillna(
        {"am_offset": 0.0, "pm_offset": 0.0}
    )
    return {
        int(user_id): UserProfile(
            age=float(row["age"]),
            state=str(row["state"]),
            am_offset=float(row["am_offset"]),
            pm_offset=float(row["pm_offset"]),
        )
        for user_id, row in joined.iterrows()
    }


def holdout_report(frame: pd.DataFrame) -> None:
    """Print held-out log loss for the population model.

    Splits by user so a user's own rows cannot appear on both sides,
    which would flatter the per-user correction.

    Args:
        frame: The full training frame.
    """
    users = frame["user_id"].unique()
    train_users, test_users = train_test_split(
        users, test_size=0.3, random_state=SEED
    )
    train = frame[frame["user_id"].isin(set(train_users))]
    test = frame[frame["user_id"].isin(set(test_users))]
    model = fit_population_model(train)
    proba = model.predict_proba(test[list(FEATURES)])[:, 1]
    baseline = np.full(len(test), train["picked_up"].astype(int).mean())
    truth = test["picked_up"].astype(int)
    print(
        f"[train] holdout log loss: population "
        f"{log_loss(truth, proba):.5f} vs constant-rate "
        f"{log_loss(truth, baseline):.5f}"
    )


def main() -> None:
    """Train the model and write it to ``artifacts/``."""
    parser = argparse.ArgumentParser(
        description="Train the best-time-to-call model."
    )
    parser.add_argument("--historic", default="data/historic.csv")
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=DEFAULT_SHRINKAGE,
        help="Pseudo-counts pulling per-user offsets to zero.",
    )
    args = parser.parse_args()

    frame = load_training_frame(args.historic)
    print(f"[train] {len(frame)} attempts, {frame['user_id'].nunique()} users")

    holdout_report(frame)

    booster = fit_population_model(frame)
    base_proba = booster.predict_proba(frame[list(FEATURES)])[:, 1]
    corrections = user_corrections(frame, base_proba, args.shrinkage)
    profiles = build_profiles(frame, corrections)

    model = BestTimeModel(
        booster=booster,
        profiles=profiles,
        fallback_age=float(frame["age"].median()),
        fallback_state="NY",
    )
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[train] wrote {MODEL_PATH} with {len(profiles)} user profiles")


if __name__ == "__main__":
    main()
