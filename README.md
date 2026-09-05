# Best Time To Call — ML Coding Challenge

Build a model that predicts the best time of day to call a customer about an
outstanding telco balance.

**Time:** ~60 minutes soft cap, 90 minutes hard cap.
**Runtime:** CPU-only Python with [`uv`](https://docs.astral.sh/uv/). No GPUs,
no large neural networks.
**LLM use:** Allowed and encouraged at build time (Claude Code, Cursor,
ChatGPT, etc.). Your trained artifact must be a deterministic, classical ML
model. An Anthropic API key is provided as a courtesy in the email with this
challenge.

## What you're building

Given:

- Historic call-attempt data for the last 360 days (`data/historic.csv`).
- A list of `(user_id, date)` evaluation targets (`data/eval_targets.csv`).

Produce, for each `(user_id, date)`, a ranked list of the **top-3 UTC hours**
(integers 0–23) at which we should call this user to maximize the chance they
pick up.

A "successful contact" is an outbound call answered by the user themselves —
not voicemail, not a busy signal, not a hang-up before connecting. The label
in the historic data is binary: `picked_up = 1` if successful, else `0`.

## Getting started

### Prerequisites

- Python 3.12 (`uv` will manage this for you).
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed.

### One-liner sanity check

```bash
make all
```

This runs `setup → train → predict → validate` end-to-end. The shipped
`train.py` and `predict.py` are stubs — `predict.py` ranks `[10, 14, 18]`
for everyone — so `make all` produces a well-formed but trivial
`predictions.csv`. Replace the stubs with your real model.

### `make` targets

| Target          | What it does                                                                          |
|-----------------|---------------------------------------------------------------------------------------|
| `make setup`    | `uv sync` — installs deps, creates `.venv/`.                                          |
| `make train`    | Runs `python -m challenge.train`. Write your trained model to `artifacts/`.           |
| `make predict`  | Runs `python -m challenge.predict`. Reads eval targets, writes `predictions.csv`.     |
| `make validate` | Runs `python -m challenge.format`. Confirms `predictions.csv` has the right shape.    |
| `make all`      | All of the above, in order.                                                           |

`make validate` only checks the *format* of your predictions file (column
names, integer hours in 0–23, no nulls, no duplicate `(user, date)` rows).
It does NOT tell you how good your model is — that's our job.

## Data

### `data/historic.csv` — one row per historic call attempt

| Column                  | Type      | Notes                                      |
|-------------------------|-----------|--------------------------------------------|
| `user_id`               | int       |                                            |
| `age`                   | int       |                                            |
| `state`                 | string    | US state abbreviation                      |
| `has_email`             | bool      |                                            |
| `signup_date`           | date      | First contract signup                      |
| `contract_id`           | int       |                                            |
| `contract_signup_date`  | date      |                                            |
| `product`               | enum      | `mobile` / `internet` / `landline`         |
| `device_rented`         | bool      |                                            |
| `attempted_at_utc`      | timestamp | UTC                                        |
| `picked_up`             | bool      | The target                                 |
| `campaign_id`           | int       | Categorical                                |

Notes to myself:
- mixture of numerical and nominal values.
- what is device_rented? usually I would ask a domain expert

### `data/eval_targets.csv`

One row per `(user_id, date)` to predict for. Drawn from the most recent
slice of the 360-day window. Some users you'll have seen in `historic.csv`,
some you won't.

## Prediction format

Top-3 only, hours as **UTC integers 0–23**:

```csv
user_id,date,hour_rank_1,hour_rank_2,hour_rank_3
123,2025-04-15,18,17,19
124,2025-04-15,9,10,8
```

`hour_rank_1` is your most-confident hour. The three hours within a single
row must be distinct. `make validate` checks this for you.

## Scoring

We score your `predictions.csv` on our end against held-back ground truth.
The metric rewards good rank quality on your top-3 predictions — getting the
single best hour as `hour_rank_1` matters more than getting it as
`hour_rank_3`. Choosing how to define and optimize for "best" is part of
what we're evaluating: feel free to use whatever loss/metric you think
fits the problem when training and validating internally.

## Reproducibility contract (important)

When we grade, we **drop in our held-back `eval_targets.csv`** and run
`make all` from a clean checkout. Your model must train and produce
predictions in that single command. A static `predictions.csv` checked into
your repo is **not** a valid submission.

Practically:

- Set seeds wherever you sample.
- Add any new dependencies you need to `pyproject.toml` (we ship with only
  `pandas`; pick whatever fits your approach).
- **Run `uv sync` and commit the resulting `uv.lock` with your submission.**
  We don't ship a lockfile, but we need yours to reproduce your environment
  exactly when grading.
- `make all` must finish on CPU in a reasonable time (a few minutes).

## Deliverables

1. The complete repo zipped up and attached to the email or as link:
- Your `train.py`, `predict.py`, helper modules, notebooks/plots if any
- A trained model in `artifacts/` (or trained on the fly during `make all`).
- Updated `pyproject.toml` plus a committed `uv.lock` so we can reproduce your environment exactly. (Run `uv sync` once before zipping.)
- `REPORT.md` describing your approach, EDA findings, and trade-offs.
2. A link to your screen recording (Google Drive, Loom, or YouTube unlisted)

## What we're looking for

- Clear problem framing
- Clear and transparent communication with regard to your overall thinking, approach and considerations. The solution should be easily understood by your teammates.
- Clean, reproducible code. Your repo should run end-to-end via `make all`.
- Efficient LLM use for real world tasks

We don't expect a state-of-the-art model. We expect deliberate choices, process, and documentation.
