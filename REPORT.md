# Report

## TL;DR

Age 60 splits our customers into two opposite calling strategies. Under
60 the best hour is 19:00 in the customer's own time; from 60 onwards
the pattern inverts, so they answer in the late morning and are hardest
to reach exactly when everyone else picks up. Calling a 60+ customer at
the population-best hour costs about 29 percentage points of pick-up
probability, which is wasted agent time at scale.

The other thing that matters is the clock. Timestamps are UTC and the US
spans six timezones, so we convert to the customer's local wall clock,
model there, then convert the chosen hours back to UTC for that date and
its daylight saving status. Every other column was screened and dropped.
Held-out log loss is 0.60013 against 0.63374 for a constant-rate
baseline.

## Approach

In business terms: we estimate, for each customer, how likely they are
to answer at each hour of their own day, then hand the collections team
the three best hours expressed in UTC so the dialler can use them
directly.

Technically, the pipeline is:

1. **Local time.** `challenge.feature.add_local_time` maps `state` to an
   IANA timezone and converts `attempted_at_utc` with stdlib `zoneinfo`,
   so daylight saving is applied per date. Adds `local_hour`,
   `local_dow`, `is_dst`, `utc_offset_hours`.
2. **Population model.** LightGBM classifier predicting
   `P(picked_up | local_hour, age)`, with `local_hour` passed as a
   categorical feature so the model can represent a bimodal curve and
   an inversion rather than a monotone trend. Two features only.
3. **Per-user correction.** A shrunk AM/PM residual offset per user
   (AM = local 08 to 13, PM = local 14 to 20), fitted on the population
   model's residuals: `offset = sum(residual) / (count + k)`. The
   shrinkage constant `k` is `DEFAULT_SHRINKAGE = 20.0` in
   `src/challenge/model.py`, tuned on held-out log loss by
   `scripts/sweep_shrinkage.py`. The held-out curve is flat over
   roughly k = 12 to 50, so the precise value is not critical and 20
   sits in the middle of that plateau.
4. **Ranking.** Score local hours 08 to 20 only, because the historic
   file contains zero attempts outside that window and we have no
   evidence there.
5. **Back to UTC.** Convert the top 3 local hours to UTC using the
   user's state and the DST status of that exact date, deduplicating so
   the three output hours are distinct.
6. **Fallback.** A user with no history gets the population curve for
   the median age with a zero offset. This affects 2 of the 1,774
   distinct eval users; the other 1,772 already appear in
   `data/historic.csv`.

The model recovers the age split on its own, without being told about
it. Asked for the best local hours at age 59 it returns 19, 20, 18; at
age 60 it returns 15, 10, 11.

## Train / validation split

Final artefact: trained on all 173,579 historic attempts from 9,719
users.

For measurement, `train.holdout_report` splits **by user**, 70/30, seed
42. Splitting by row would let a user's own attempts sit on both sides,
which would flatter the per-user correction and tell us nothing about
new customers.

| model | held-out log loss |
|---|---|
| constant rate (overall pick-up 0.331) | 0.63374 |
| LightGBM on `local_hour` + `age` | 0.60013 |

The per-user correction needs a different split again: because a
user's offset is fitted from that user's own attempts, a row split
would let the same attempt set the offset and then be scored by it.
`scripts/sweep_shrinkage.py` therefore splits **within** each user,
chronologically (earliest 70% of a user's attempts fit the offset,
latest 30% score it), which is how the model is actually used: we
always predict forward from history.

| per-user offset | held-out log loss |
|---|---|
| none (population model only) | 0.59822 |
| k = 5 (the chronotype study's value) | 0.61515 |
| k = 12 | 0.59556 |
| **k = 20 (shipped)** | **0.59223** |
| k = 35 (grid minimum) | 0.59209 |
| k = 50 | 0.59244 |

Worth stating plainly, because it corrected an earlier decision: the
chronotype study found k = 5 optimal against its own coarse population
curve, but against the LightGBM residuals k = 5 is too weak and scores
*worse* than applying no correction at all. Re-tuning moved the optimum
to the flat 12 to 50 region. The shipped k = 20 is about 1.0% better
than no correction on the chronological split, and 1.5% better on the
random within-user split (0.58846 vs 0.59724), which is the same order
of gain the chronotype study predicted.

Log loss is our proxy metric. The graders use an undisclosed rank
quality measure on the top-3 hours, and a well-calibrated probability
per hour is the most defensible way to produce a ranking without knowing
their exact scoring rule.

To check that the proxy does not mislead us, `scripts/sweep_rank_metric.py`
scores the ranking directly on the same within-user chronological split:
it measures the pick-up rate among held-out attempts that fall in the
model's top-3 and rank-1 hours. It agrees with log loss, k = 20 is best
on both.

| k | top-3 rate | rank-1 rate | lift |
|---|---|---|---|
| 5 | 0.4924 | 0.5174 | 1.502 |
| 12 | 0.4966 | 0.5222 | 1.514 |
| **20 (shipped)** | **0.4985** | **0.5246** | **1.520** |
| 35 | 0.4943 | 0.5232 | 1.507 |
| none | 0.4910 | 0.5229 | 1.497 |

Held-out base rate is 0.3279, so the three hours we recommend pick up at
about 1.52 times the rate of an average hour.

## EDA findings

Four investigations, run in order, each with a report and raw numbers
under `reports/`. The sequence matters: each step changed the question
the next one asked.

### 1. Convert to local time first (`reports/local_time_feature`)

A raw UTC hour mixes a New Yorker's breakfast with a Californian's dawn,
so it is not a meaningful feature. After conversion, attempts fall in
local hours 08 to 20 in every timezone, and inside the interior 09 to 19
they are close to uniform: every hour holds 8.1% to 9.2% of attempts
against 9.1% for perfect uniformity. Chi-square against uniform gives
Cramer's V = 0.014 over hours 09 to 19 in local time, versus 0.188 for
UTC hour over the full day.

Meaning: the dialler was already working in local business hours. There
is no hidden scheduling bias by state or timezone that we need to
correct for. Per-timezone deviation from the pooled distribution (total
variation distance) tracks 1 / sqrt(n) almost exactly, from 0.004 for
New York (n = 81,086) to 0.085 for Alaska (n = 353), which is what
sampling noise looks like.

DST handling was verified directly: New York and Los Angeles show two
distinct UTC offsets across the year, Phoenix and Honolulu show one.

### 2. The hour curve is bimodal (`reports/pick_curve`)

Overall pick-up rate is 0.331. By local hour it swings by a factor of 5.

| local hour | 08 | 10 | 12 | 15 | 17 | 19 | 20 |
|---|---|---|---|---|---|---|---|
| all customers, % | 10.2 | 29.3 | 39.2 | 22.8 | 35.3 | 49.5 | 47.3 |
| age 60+, % | 24.8 | 51.1 | 34.1 | 49.0 | 30.9 | 22.3 | 21.1 |

There is a late-morning hump peaking at 12:00, a mid-afternoon trough at
15:00, and the best window at 18:00 to 20:00. Not one hump, so no linear
hour term and no single "best time to call".

### 3. Local hour is causal, UTC hour is a trap

Plotted per timezone in local time, the curves collapse onto each other.
Every timezone with enough data peaks at local 19:00 and troughs at
local 08:00 to 09:00, four of five sit at or below their own sampling
noise level, and none deviates by more than about 2 pp in any hour.
Since these zones are up to 3 hours apart in UTC, a shared curve in
local time is direct evidence that local hour drives pick-up. UTC hour
only looks predictive because it is a proxy for timezone, and a model
fitted on it would be learning geography, not behaviour.

### 4. Age 60 is a step, not a gradient

Scanning age year by year, the flip lands exactly at 60.

| | morning rate | evening rate |
|---|---|---|
| age 59 | 27.0% | 46.9% |
| age 60 | 42.7% | 21.7% |

The 60+ group peaks at 10:00 (51.1%) and again at 15:00 (49.0%), and
falls to its worst hours of the day at 19:00 and 20:00 (22.3% and
21.1%), exactly where everyone else peaks. Its deviation from the pooled
curve reaches +26.2 pp at 15:00 and -27.2 pp at 19:00 against 0.87 pp of
sampling noise. Under 60 the cohorts differ in amplitude, not in phase:
they all peak in the evening.

Business consequence: age is not a nuisance variable here, it changes
which hour to dial. Sending a 60+ customer the population-optimal 19:00
slot instead of their own 10:00 costs roughly 29 pp of pick-up
probability.

### 5. Everything else was screened out (`reports/covariate_screen`)

Each covariate was judged twice: on its marginal association with
pick-up, and on whether it bends the pick-up-versus-hour curve after the
level effect is divided out. Only the second one can change a ranking.

| covariate | verdict |
|---|---|
| `age` | KEEP, the only ranking-relevant modifier |
| `state` | keep only as the timezone source, not as a feature |
| `product` | drop, it is age in disguise |
| `campaign_id` | drop, uninformative identifier |
| `signup_date`, tenure | drop, level effect at most |
| `has_email`, `device_rented` | drop, no effect |
| `contract_id`, `contract_signup_date` | drop, duplicate columns |

Two data-quality findings worth flagging to whoever owns the export:
`contract_id` equals `user_id` in 100% of rows, and
`contract_signup_date` equals `signup_date` in 100% of rows. Every user
has exactly one contract. Those columns carry nothing.

`product` looked predictive until we conditioned on age: landline users
average 55.9 years against 41.5 for mobile, and within each age band the
product interaction disappears (negative AIC gain) while the age
interaction survives inside every product. `campaign_id` has a
bias-corrected Cramer's V of 0.000 against pick-up; its per-campaign
rates run 0.315 to 0.346, which is exactly what 50 draws of about 3,470
coin flips at p = 0.33 look like. All 50 campaigns are active in all 12
months and campaign is independent of dialled hour (p = 0.85), so it is
not confounding the hour curve either.

### 6. Chronotype: the naive answer is wrong (`reports/chronotype`)

The plain-language question: beyond age, does each customer have their
own preferred time of day, and can we use it?

The naive test compares each user's morning pick-up rate to their
evening rate. It says no: correlation +0.001 across 7,555 users, flat
zero. That answer is an artefact of two things. First, per-user data is
thin (about 18 attempts per user, so roughly 8 per block), which makes
raw per-user rates almost pure noise. Second, population-level
differences in *how likely someone is to answer at all* dominate the raw
rates and swamp the timing signal we actually want.

The fix is to measure each user's preference as a residual from their
own age group's hour curve, then check whether that residual is stable
by splitting each user's attempts into two random halves and correlating
the two estimates. Split-half reliability:

| stratum | split-half r | simulated null r |
|---|---|---|
| age < 60 | +0.298 | +0.004 (sd 0.017) |
| age 60+ | +0.154 | -0.001 (sd 0.038) |

The null column is the same statistic on attempts simulated from the
population curve at the users' real hours, so it rules out the index
construction or the filter creating the correlation by itself. The
effect survives a finer baseline (age decade x state x hour: +0.271 /
+0.155) and a chronological split of earlier versus later attempts
(+0.283 / +0.138). The chronological version is the one that matters
operationally: the preference persists over time, so it is usable for
forward prediction.

Size, not just existence: a variance decomposition against a parametric
bootstrap says roughly 70% of the visible per-user spread is sampling
noise and about 30% of the variance is real. A one-standard-deviation
customer is tilted about 15 points of morning-minus-evening residual.
Real, but small next to the 39 point swing of the population hour curve,
which is why it enters the model as a heavily shrunk offset and not as
a per-user curve.

Day of week showed no signal at any point: 0.329 to 0.333 across all
seven days, with the same hour shape on weekdays and weekends.

## Trade-offs

What we deliberately did not do, and why.

- **One timezone per state.** 13 states span two zones (`AK FL ID IN KS
  KY MI ND NE OR SD TN TX`); we assign the majority-population zone, so
  a user in the Florida panhandle or in El Paso gets a 1 hour error.
  State is the finest geography in the file. A ZIP-to-timezone table
  would remove this. Note the error direction: it blurs the timezone
  collapse rather than creating it, so it does not threaten the main
  conclusion.
- **No calendar, seasonality or day-of-week features.** Day of week
  measured flat (0.329 to 0.333) and monthly rates are flat at 0.325 to
  0.342 with no trend. Adding them would spend parameters on noise.
- **Only an AM/PM tilt per user, not a per-user hour profile.** With
  about 18 attempts per user there is enough data for roughly one
  number, not a 13-hour shape. The 08/13 versus 14/20 cut is arbitrary,
  so a customer whose real preference is 08:00-versus-12:00 is invisible
  to us. That makes our chronotype estimate a lower bound on the
  individual structure that exists.
- **Hours outside local 08 to 20 are unrankable.** We have zero attempts
  there, so we never rank them. This is a limitation of the historic
  dialling policy, not of the model. The 60+ curve is still rising into
  08:00, so their true optimum may sit earlier than we can see. An
  experiment dialling a small random sample at 07:00 would settle it.
- **We optimised log loss, the graders use something else.** Their rank
  quality metric is undisclosed. Calibrated probabilities per hour are
  our proxy, and they induce the ranking directly.
- **P-values here are optimistic.** 173,579 rows come from 9,719 users,
  so attempts within a user are correlated and every test treating rows
  as independent understates the uncertainty. At this n almost anything
  reaches p < 0.05, which is why every keep/drop decision was made on
  effect size (Cramer's V, AIC gain, deviation versus sampling noise)
  rather than on significance.

With more time: a proper per-user random effect or hierarchical model
instead of the two-block offset; a ZIP-level timezone map; an
exploration budget on unobserved hours; and calibration checked directly
against a simulated top-3 rank metric rather than log loss alone.

## LLM workflow notes

The EDA was run as four background agents in parallel: local time
feature, pick-up curve, covariate screen, chronotype. Each wrote its own
report and raw numeric dump under `reports/`, and each has a
reproducible script under `scripts/`. That parallelism is the only
reason four separate investigations fit inside the time budget.

The briefs mattered more than the models. Each one named the statistical
trap up front: control for age before looking for individual chronotype;
use split-half reliability rather than a raw per-user correlation;
prefer effect sizes over p-values because rows repeat per user; check
whether campaign and dialled hour are confounded. Without those
instructions a naive agent run would have returned the wrong answer on
the chronotype question, because the obvious analysis (morning rate vs
evening rate) reports r = +0.001 and looks conclusive.

One concrete correction. A test asserting that a per-user offset can
flip the hour ranking failed on the first attempt. The offset used in
the test was too small to beat the population evening peak, which is the
shrinkage doing exactly what it should. The right fix was the test, not
the model, so the test now uses a large offset to prove the mechanism
works, and separate tests assert that shrinkage keeps real offsets
small.

## How to reproduce

`make all` runs setup, train, predict and validate end to end. `uv run
pytest` runs the 20 tests. The four EDA scripts under `scripts/`
regenerate the reports, for example
`uv run python scripts/eda_pick_curve.py`.
