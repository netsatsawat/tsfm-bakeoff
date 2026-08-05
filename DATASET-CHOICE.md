# Where the deep-dive datasets came from

The full bake-off runs all ten datasets across all feasible horizons; that matrix is
committed in `results/cross_env_scores.json`. The deeper analyses in the articles and
in `RESEARCH-NOTES.md` (multivariate joint-vs-single, calibration autopsies, the
model-selection argument) concentrate on two datasets: **`bangkok_pm25_1h` and
`boom_telemetry_5t`**. This file records why those two, because they were chosen from
the measured results rather than by intuition.

## Why concentrate at all

Running five foundation-model families across ten datasets answers a question that is
already settled on strongly seasonal series (temperature, electricity load):
foundation models win comfortably there, and the remaining variation is small. The
open question is narrower and more useful: **when is a foundation model the wrong
purchase, and when is it clearly the right one?** Two datasets bracket that decision.

## The two, and what each is for

Measured as the best foundation model against the best non-foundation baseline
(classical or naive) on the same origins:

| dataset | seasonal strength | best FM | best baseline | FM edge |
|---|---|---|---|---|
| `bangkok_pm25_1h` | 0.37 | 0.647 | **0.589** (naive_last) | **-9.9%** |
| `boom_telemetry_5t` | 0.23 | **0.357** (Chronos-2) | 0.502 (AutoTheta) | **+29.0%** |

**`bangkok_pm25_1h`: the only dataset where foundation models lose.** Bangkok air
quality has an obvious daily cycle and looks like a forecasting problem, but seasonal
strength is only 0.37, spectral entropy is 0.54, and the distribution is heavy-tailed.
Surges dominate the error, surges are not predictable from history, and persistence is
the best available answer between them. Every model that smooths toward a seasonal
profile pays for it. This is the case that punishes buying on a leaderboard.

**`boom_telemetry_5t`: the largest foundation-model win in the study.** Observability
telemetry as released by Datadog: 91 to 100 variates per entity, overlapping daily and
weekly periodicity, and surges reaching 245x IQR. The benchmarked panel is a
six-variate sample of one group (`ds-139-5T`), and its +29% edge is measured on that
sample, not on the full 2,807-series corpus. It is the archetype general-purpose
forecasters are documented to struggle with, and it is where the classical side
collapses outright: MSTL scores 1.925, worse than the seasonal naive, with 0.0%
coverage of its nominal 80% interval. STL assumes a stable seasonal shape plus smooth
trend plus small residual; a 245x IQR surge violates that badly enough that the
decomposition stops meaning anything.

## Why this pair rather than a harder single dataset

A single hard dataset supports only "foundation models are/aren't worth it". These two
bracket the decision, which is the more useful claim: the same models lose to a
one-line baseline on one series and beat everything by 29% on another, and the
difference is a property of the **data**, not the models. That is the argument for
measuring your own series before you shop, and it cannot be made from one dataset.

They also share a useful property: **neither is strongly seasonal** (0.37 and 0.23).
So the contrast is not simply "seasonal good, noisy bad". Both sit in the awkward
middle where the decision is genuinely hard, and where a leaderboard is least
informative.

## Ten-dataset context

The full ten-dataset bake-off lives in `results/bakeoff_full2*` and the cross-family
matrix in `results/cross_env_scores.json`; the structure ladder in
`RESEARCH-NOTES.md` is what these two were selected from. The deep dives do not
replace the matrix; they go deeper on the two cases where the model-selection decision
is live.
