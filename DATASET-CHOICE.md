# Why two datasets, and why these two

The cross-environment study is deliberately narrow: **`bangkok_pm25_1h` and
`boom_telemetry_5t`**, nothing else.

## Why not all ten

Running four foundation-model families across ten datasets means four separate virtualenvs
× ten datasets × rolling origins, and most of that work answers a question already settled:
on strongly seasonal series (temperature, electricity load) foundation models win
comfortably and the interesting variation is small. Adding Toto and Moirai there would
move numbers a few percent and change no decision.

The open question is narrower and more useful: **when is a foundation model the wrong
purchase, and when is it clearly the right one?** Two datasets answer that, and they were
chosen from the measured results rather than by intuition.

## The two, and what each is for

Measured as the best foundation model against the best non-foundation baseline
(classical or naive) on the same origins:

| dataset | seasonal strength | best FM | best baseline | FM edge |
|---|---|---|---|---|
| `bangkok_pm25_1h` | 0.37 | 0.647 | **0.589** (naive_last) | **−9.9%** |
| `boom_telemetry_5t` | 0.23 | **0.357** (Chronos-2) | 0.502 (AutoTheta) | **+29.0%** |

**`bangkok_pm25_1h` — the only dataset where foundation models lose.** Bangkok air quality
has an obvious daily cycle and looks like a forecasting problem, but seasonal strength is
only 0.37, spectral entropy is 0.54, and the distribution is heavy-tailed. Surges dominate
the error, surges are not predictable from history, and persistence is the best available
answer between them. Every model that smooths toward a seasonal profile pays for it. This
is the case that punishes buying on a leaderboard.

**`boom_telemetry_5t` — the largest foundation-model win in the study.** High-dimensional
multivariate observability telemetry: 91–100 variates per entity, overlapping daily and
weekly periodicity, and surges reaching 245×IQR. It is the archetype general-purpose
forecasters are documented to struggle with, and it is where the classical side collapses
outright — MSTL scores 1.925, worse than the seasonal naive, with 0.0% coverage of its
nominal 80% interval. STL assumes a stable seasonal shape plus smooth trend plus small
residual; a 245×IQR surge violates that badly enough that the decomposition stops meaning
anything.

## Why this pair rather than a harder single dataset

A single hard dataset supports only "foundation models are/aren't worth it". These two
bracket the decision, which is the more useful claim: the same three models lose to a
one-line baseline on one series and beat everything by 29% on another, and the difference
is a property of the **data**, not the models. That is the argument for measuring your own
series before you shop, and it cannot be made from one dataset.

They also share a useful property: **neither is strongly seasonal** (0.37 and 0.23). So the
contrast is not simply "seasonal good, noisy bad" — both sit in the awkward middle where
the decision is genuinely hard, and where a leaderboard is least informative.

## Ten-dataset context

The full ten-dataset bake-off remains in `results/bakeoff_full2*`, and the structure ladder
in `RESEARCH-NOTES.md` is what these two were selected from. This narrower study does not
replace it; it goes deeper on the two cases where the model-selection decision is live.
