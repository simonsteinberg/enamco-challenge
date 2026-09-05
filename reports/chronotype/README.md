# Individual chronotype: does per-user timing preference exist?

**Verdict: yes, a real and time-stable per-user chronotype exists, and it is
big enough to use, but only with heavy shrinkage toward the age-group curve.**
Split-half reliability is r = +0.30 (age<60) and +0.15 (age>=60) against a
simulated null of r = 0.00, and adding a shrunken per-user morning/evening
offset cuts held-out log-loss by 0.0088 (0.6011 -> 0.5923, about 1.5%).

Script: `scripts/eda_chronotype.py` (`uv run python scripts/eda_chronotype.py`).
Full numeric dump: `stats.txt`. Seed 20250905 throughout.

## Setup

Blocks are AM = local hours 8-13 and PM = local hours 14-20. The per-user
index is `mean(y - p)` over AM attempts minus `mean(y - p)` over PM attempts,
where `p` is the population pick-up rate for that user's age stratum and local
hour. Subtracting the age-group curve does two jobs: it removes the population
shape, and because a user's overall pick-propensity enters both blocks
equally, it cancels out of the difference. What is left is timing preference
only. Every analysis runs separately in `age < 60` and `age >= 60`.

## 1. Split-half reliability (primary test)

Each user's attempts are randomly split into two balanced halves; the index is
computed in each half and correlated across users. Users need at least 3
attempts in all four (half, block) cells.

| stratum | users kept | r (half A, half B) | 95% CI | Spearman-Brown | null r (200 sims) |
|---|---|---|---|---|---|
| age<60 | 3,896 of 8,297 | **+0.298** | [+0.269, +0.326] | +0.459 | +0.004 (sd 0.017) |
| age>=60 | 680 of 1,422 | **+0.154** | [+0.079, +0.226] | +0.266 | -0.001 (sd 0.038) |

The null column is the same statistic recomputed on attempts simulated from the
population curve at the users' real hours. It sits at zero, so the observed
correlation is not an artefact of the index construction or the filter.
`P(null r >= observed) = 0.005` in both strata (the floor for 200 sims).

Robustness (`stats.txt`, section 1b):

* Finer baseline curve (age decade x state x local hour): r = +0.271 / +0.155.
  So this is not leftover age-shape or geography leaking into a user-constant
  covariate.
* Chronological halves (a user's earlier attempts vs their later attempts):
  r = +0.283 / +0.138. The trait persists over time, which is what makes it
  usable for forward prediction rather than just an in-sample curiosity.

See `split_half_scatter.png`.

## 2. Variance decomposition vs parametric bootstrap null

Simulate every attempt at its real hour under the population probability for
the user's age group, recompute the index, compare spreads. Users with >= 3
attempts per block.

| stratum | users | observed var (sd) | null var (sd) | excess var | implied true between-user sd | p |
|---|---|---|---|---|---|---|
| age<60 | 6,446 | 0.0709 (0.266) | 0.0493 (0.222) | +0.0216 | 0.147 | 0.002 |
| age>=60 | 1,109 | 0.0716 (0.268) | 0.0560 (0.237) | +0.0157 | 0.125 | 0.002 |

Roughly 70% of the observed between-user spread is pure sampling noise, and
about 30% of the variance is real. A one-SD user is shifted about 15 points of
AM-minus-PM residual, i.e. roughly +/- 7 points on each block. That is real but
small next to the 39-point swing of the population hour curve (10% at 08:00 to
50% at 19:00). Chronotype refines the hour ranking; it does not overturn it.

`index_observed_vs_null.png` is the money plot: the observed distribution has
visibly fatter tails than the null in both strata.

## 3. Held-out log-loss

30% of attempts held out, population curve refit on the training rows only.
The per-user term is one ridge-shrunk Newton step in log-odds per (user, block)
cell, `delta = sum(y - p) / (sum(p(1-p)) + k)`, fitted on training rows.

| model | test log-loss | delta |
|---|---|---|
| population age-group hour curve only | 0.60110 | n/a |
| + per-user offset, k = 1 | 0.59788 | -0.0032 |
| + per-user offset, k = 2 | 0.59258 | -0.0085 |
| **+ per-user offset, k = 5** | **0.59230** | **-0.0088** |
| + per-user offset, k = 10 | 0.59470 | -0.0064 |
| + per-user offset, k = 20 | 0.59715 | -0.0040 |
| + per-user offset, k = 100 | 0.60015 | -0.0010 |

The curve is single-peaked in `k` with an optimum near 5, and the near-unshrunk
version (k = 1) gives back two thirds of the gain. Shrinkage is doing real
work, not just regularisation theatre.

## 4. The naive view, for contrast

Raw per-user AM pick rate vs PM pick rate, no control at all:

| slice | r | 95% CI | n |
|---|---|---|---|
| pooled | +0.001 | [-0.022, +0.023] | 7,555 |
| age<60 | +0.005 | [-0.020, +0.029] | 6,446 |
| age>=60 | -0.025 | [-0.084, +0.034] | 1,109 |

All three are indistinguishable from zero (`naive_rate_scatter.png`). This is
the trap: with ~8 trials per block the rates are almost pure noise, and the
correlation the naive analysis would report has nothing to do with the
question. A reviewer looking only at this table would conclude "no chronotype"
and be wrong. The signal only appears once the age-group curve is divided out
and the estimate is split-half validated.

The expected spurious negative from the `age >= 60` inversion does not show up
in the pooled raw-rate correlation, because level differences dominate the raw
rates rather than shape differences. It would still bite any model that
estimated a per-user timing term without an age control, which is why the index
is defined as a residual from the age-group curve.

## 5. Sanity check: contact volume is not confounded with pick-up rate

Spearman(attempts per user, user pick-up rate) = +0.082 (p = 6e-14) for
age<60 and +0.024 (p = 0.36) for age>=60. The age<60 effect is significant but
tiny: quintile mean rates run 0.304, 0.321, 0.319, 0.327, 0.324, and the whole
gap is the bottom quintile (users with <= 7 attempts). There is no
"hard-to-reach users get called more" pattern to leak into a per-user term.
The index is a within-user difference anyway, so a level confound cancels.

## Caveats

* Per-user data is thin. The split-half filter keeps only 3,896 / 8,297 and
  680 / 1,422 users; users with a single attempt contribute nothing here. The
  reliability figures describe users with enough data, and the effective
  reliability for a 5-attempt user is far lower.
* Reliability is not size. Spearman-Brown of 0.46 sounds impressive but the
  underlying true between-user SD (0.147) is modest.
* The AM/PM cut at local 13/14 is one arbitrary contrast. A user whose real
  preference is 08:00-vs-12:00 within the morning is invisible to this index,
  so this is a lower bound on individual timing structure.
* Multiple testing: several tests were run, but the primary result is far
  outside its simulated null (r = 0.30 vs null sd 0.017, about 18 sigma) and
  survives two independent robustness variants, so this is not a fishing
  artefact.
* The parametric bootstrap assumes independence of attempts within a user
  given the hour. Correlated bursts (same campaign, same week) would inflate
  the observed spread. The chronological-halves result argues against that
  being the whole story, since bursty within-period correlation would not
  survive an early-vs-late split.

## Implications for modelling

1. **`predict.py` should carry a per-user timing term.** 99.9% of eval targets
   are users seen in `historic.csv`, with a median of 18 prior attempts, so
   the term applies to essentially the whole eval set rather than a fringe.
2. **It must be shrunk toward the age-group curve.** Unshrunk per-user rates
   are worse than useless; a ridge weight around `k = 5` on the log-odds
   Newton step was the held-out optimum. Equivalently, trust a user's own
   history only once it is worth roughly 5 effective trials.
3. **Keep it as an offset on top of the `age >= 60` x local-hour curve**, not
   as a free per-user hour profile. There is only enough data per user for
   about one number (an AM-vs-PM tilt), not a 13-hour shape.
4. For users with no history, fall back to the age-group curve with a zero
   offset, which the shrinkage formula already does automatically.
5. Expected payoff is real but second-order: about 1.5% of log-loss, against
   the local-hour curve itself which is the dominant effect. Build the curve
   first, add the per-user tilt second.
