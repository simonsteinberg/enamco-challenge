"""Training entrypoint.

This is a stub. Replace it with whatever training pipeline your model needs.
The only contract is:

  * `make train` calls `python -m challenge.train`.
  * Anything you write to `artifacts/` is available to `predict.py`.
  * `make all` must succeed end-to-end on a fresh checkout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the best-time-to-call model.")
    parser.add_argument(
        "--historic",
        default="data/historic.csv",
        help="Path to historic call attempts CSV.",
    )
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    # Stub: write a sentinel so predict.py can verify train ran. Replace this
    # with a real serialized model.
    sentinel = ARTIFACTS_DIR / "trained.txt"
    sentinel.write_text(f"trained on {args.historic}\n")
    print(f"[train] stub: wrote sentinel to {sentinel}")


if __name__ == "__main__":
    main()
