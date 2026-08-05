# Research notes: TSFM bake-off lab journal

This file is my working journal for the study, kept because the dead ends and
corrections are half the value. Everything in it is measured on this machine, not
quoted. Where a table's generating script ships in the repo it is named beside the
table; the two structure-characterization scripts were scratch files that did not
ship, so their numbers stand as measured records rather than regenerable tables. The
results sections cite `results/bakeoff_full2*` and `results/cross_env_scores.json`,
both committed.

One correction pass happened after an external review, and I would rather flag it
than hide it: the model I ran from Datadog is **Toto-Open-Base-1.0**, not the newer
Toto-2.0 family that tops GIFT-Eval. Earlier drafts of these notes and the result key
`toto_2p0` named it wrong; the key stays (it is baked into the committed artifacts)
and the prose here now says what actually ran. Section 5's daily-data sentence was
also corrected against the artifact it cites.

## 1. The model field: what can actually run side by side

Measured with `pip install --dry-run` before touching the environment.

| model | package | verdict |
|---|---|---|
| **TimesFM 2.5** (Google, 200M) | `timesfm[torch]==2.0.2` | installs clean, runs, ~300 ms/forecast on MPS |
| **Chronos-2 / Chronos-Bolt** (Amazon) | `chronos-forecasting==2.3.1` | installs clean (adds only `einops`); exposes `Chronos2Pipeline`, `ChronosBoltPipeline`, and native **multivariate** support |
| **AutoETS / AutoTheta / MSTL** | `statsforecast==2.1.1` | installs clean |
| **Histogram GBDT** | `scikit-learn` 1.7.2 | already present, `HistGradientBoostingRegressor` |
| LightGBM | `lightgbm==4.7.0` | **installs but cannot load**: `libomp.dylib` missing on this Mac. Substituted sklearn's histogram GBDT, same algorithm class, no OpenMP dependency. Uninstalled to keep the env honest. |
| **Toto** (Datadog, Toto-Open-Base-1.0) | `toto-ts==0.2.0` | **REFUSED to coexist**. Dependency closure downgrades `torch 2.11.0 -> 2.7.0`, `numpy 2.2.6 -> 1.26.4`, `pandas 2.3.3 -> 2.2.3`, and hard-pins `jupyter==1.1.1`. Would break the working TimesFM install. (The newer Toto-2.0 family leads GIFT-Eval; running it is queued work.) |
| **Moirai-2** (Salesforce) | `uni2ts==2.0.0` | **REFUSED to coexist**. Downgrades `torch -> 2.4.1`, `numpy -> 1.26.4`, `pandas -> 2.1.4`, pulls jax + lightning + tensorboard. |

**Finding worth publishing on its own:** you cannot benchmark the leading TSFMs in one
Python environment. Toto and Moirai each demand an incompatible numeric stack. A fair
bake-off needs one venv per model family and a forecasts-on-disk contract between
them. That is a real engineering cost nobody mentions in the papers.

## 2. The datasets: ten series, structure measured not assumed

`python3 datasets.py` fetches; `python3 prepare.py` audits. The ladder below came from
a scratch characterization script (not shipped); the audit numbers per series are
reproducible via `prepare.py`. Every source is key-free. The 2026 windows postdate
every cutoff on the TimesFM 2.5 model card (GiftEvalPretrain, Wikimedia Nov 2023,
Trends end-2022), so "zero-shot" is literally true for them.

| series | domain | freq | seasonal | trend | spec. entropy | acf1 | label |
|---|---|---|---|---|---|---|---|
| `bangkok_temp_1h` | weather | 1h | 0.86 | 0.60 | 0.331 | 0.93 | strong seasonal |
| `london_temp_1h` | weather | 1h | 0.86 | 0.96 | 0.359 | 0.99 | strong seasonal |
| `uk_demand_30min` | energy | 30min | 0.85 | 0.88 | 0.345 | 0.99 | strong seasonal |
| `uk_demand_daily` | energy | 1d | 0.53 | 0.92 | 0.428 | 0.91 | moderate seasonal |
| `bangkok_pm25_1h` | air quality | 1h | 0.37 | 0.77 | 0.535 | 0.97 | moderate seasonal |
| `white_noise_synth` | synthetic | 1h | 0.22 | 0.05 | **0.942** | 0.01 | none (noise-like) |
| `btc_returns_1h` | finance | 1h | 0.12 | 0.02 | **0.941** | -0.01 | none (noise-like) |
| `quake_magnitude_seq` | geophysics | event order | 0.11 | 0.01 | **0.945** | 0.07 | none (noise-like) |
| `btc_usd_1h` | finance | 1h | 0.02 | 0.99 | 0.157 | 1.00 | trend/drift only |
| `boom_telemetry_5t` | observability | 5min | see below | 0.48 avg | 0.41 avg | 0.99 med | mixture (see section 4) |

Seasonal and trend strength are the STL variance-decomposition measures; spectral
entropy is the normalized periodogram entropy, where a value near 1 means power spread
evenly across all frequencies, in other words unforecastable.

A BOOM bookkeeping note that bit me once already: the 24-series characterization
sample averaged seasonal strength 0.44, while the six-series panel actually
benchmarked in `bakeoff_full2` averages 0.225 (recorded in that run's JSON). Same
population, different sample. Tables in section 5 use the benchmarked panel's number.

### The no-trend, no-seasonality group (the requested hard case)

Three series, two of them real measurements:

- **`btc_returns_1h`**: hourly log returns of BTC-USD. Seasonal 0.12, trend 0.02,
  entropy 0.941, lag-1 autocorrelation -0.01. ADF p = 0.0000 (stationary), KPSS
  p = 0.10 (does not reject stationarity). The textbook real stationary non-seasonal
  series.
- **`quake_magnitude_seq`**: magnitudes of 2,657 consecutive global earthquakes at or
  above M4.5 from the USGS catalog, indexed by **event order** rather than clock time
  so no calendar artifact can sneak in. Gutenberg-Richter draws are physically
  memoryless: seasonal 0.11, trend 0.01, entropy 0.945, acf1 0.07.
- **`white_noise_synth`**: Gaussian noise, seed 7. The control that proves the harness
  cannot manufacture skill. If any model beats the mean here, the harness is leaking.

**Calibration detail that matters:** white noise measures seasonal strength **0.22**,
not 0.00, because STL always finds a little seasonality in noise. So 0.22 is roughly
the null floor of that statistic, and the two real series at 0.11 to 0.12 sit *below*
the noise floor. Do not read a seasonal strength of 0.2 as "weakly seasonal".

`btc_usd_1h` (the price, not the return) is kept as a separate case: seasonal 0.02 but
trend 0.99, ADF p = 0.67, KPSS p = 0.01, a genuine random walk. Random walk and white
noise are different failure modes and the naive baseline that wins differs between
them (last value vs context mean), which is exactly why both are in the set.

## 3. The pre-call contract: `prepare.py`

Eleven checks, each added because its absence causes a silent wrong answer. Full
rationale in the module docstring. The ones that changed a decision here:

1. **Regular grid.** TSFMs index by position, so one missing hour shifts every later
   point in "model time". `uk_demand_30min` has genuine irregular steps at the two
   clock-change days (46- and 50-period days), which is a real irregular grid, not
   dirty data.
2. **Gap policy.** Interpolate runs of at most 3 steps, *refuse* longer ones. Filling
   a multi-day outage with a straight line teaches a trend that never happened.
3. **NaN policy is per-model.** TimesFM needs a clean float array; Chronos tolerates
   NaN natively. Same prepared series, different admissibility.
4. **Scale ownership.** TimesFM (`normalize_inputs`) and Chronos both scale
   internally; a GBDT does not. Pre-scaling the first two is double work; not scaling
   the third is a bug. Getting this backwards makes you blame the model.
5. **Metric validity.** MAPE needs a ratio scale with a meaningful zero. It is
   **arithmetically invalid** for `bangkok_temp_1h`, `london_temp_1h` (0 degC is a
   convention), `btc_returns_1h`, `white_noise_synth` (zero-mean) and
   `boom_telemetry_5t` (ships normalized, crosses zero). Half the corpus. MASE is the
   only metric that spans all ten series, which is why it is the primary metric.

## 4. BOOM: the high-dimensional, overlapping-seasonality, sudden-surge archetype

That description is **observability telemetry**. The public benchmark of record is
**BOOM** (Datadog): 350M observations, 2,807 real multivariate series from their own
production telemetry, Apache-2.0, [arXiv 2505.14766](https://arxiv.org/abs/2505.14766),
NeurIPS 2025.

Layout: 2,807 separate ~1 MB arrow files, so a sample costs nothing against the
2.81 GB total. Each file is one entity: a matrix of **91 to 100 variates x 16,384
timesteps** at 10-second, 1-minute or 5-minute resolution.

Measured on a 24-series / 3-group / 5-minute characterization sample (scratch script,
not shipped; numbers stand as measured), the three claimed properties hold:

| claim | measurement |
|---|---|
| high-dimensional multivariate | 91 to 100 variates per entity as released; cross-variate mean abs correlation **0.16 / 0.20 / 0.43** by group |
| overlapping seasonal patterns | **9 of 24** series carry autocorrelation > 0.3 at *both* the daily (288) and weekly (2016) lag simultaneously |
| sudden anomalous surges | **4 of 24** series have excursions beyond 6x IQR; the worst reaches **245x IQR** |

It is a **mixture, not a type**: 9 strong-seasonal, 6 moderate, 5 trend-only, 4 weak.
One aggregate number over telemetry hides everything that matters, which is the same
lesson as the weekday/weekend/holiday split on the energy series.

Two honest caveats to carry into any BOOM result:

- **Not post-cutoff by date** (2024 data). It is still defensible for TimesFM because
  it is Datadog-internal telemetry first published May 2025, after GiftEvalPretrain
  was built; the argument is provenance, not chronology. Chronos-2 and Moirai-2 were
  released after BOOM's publication, so no such argument is available for them.
- **Home turf for Toto**, which trains on Datadog telemetry. Any Toto-vs-others result
  on BOOM must say so.

### Two data-quality traps hit here, both fixed

- **Degenerate variates.** Filtering on `std == 0` over the full 16,384 steps was not
  enough: a variate can be constant only within the trimmed tail actually used. Now
  rejects <20 distinct values, zero IQR or zero MAD. This dropped one whole group
  (`ds-825-5T`), the same group that showed mean abs correlation 0.97, meaning its
  variates were near-duplicates of one signal. A "multivariate win" measured there
  would have been an artifact.
- **Exploding surge ratio.** A MAD-denominated surge metric returned values up to
  **1.4e10** on near-constant variates: a division artifact, not a surge. Replaced
  with an IQR-based count plus a floored ratio.

## 5. Results: cite `results/bakeoff_full2*` only

11 models x 10 datasets x 16 rolling origins. MASE, lower is better; columns ordered
by measured seasonal strength.

| model | bkk_temp .86 | ldn_temp .86 | uk_30min .85 | uk_daily .53 | pm25 .37 | boom .23 | noise .22 | btc_ret .12 | quake .11 | btc_px .02 |
|---|---|---|---|---|---|---|---|---|---|---|
| timesfm_2p5 | 0.918 | **0.742** | **0.570** | **0.705** | 0.675 | 0.438 | 0.725 | 0.681 | 0.780 | 0.581 |
| chronos_2 | 0.907 | 0.824 | 0.573 | 0.782 | 0.648 | **0.357** | **0.721** | **0.678** | **0.776** | **0.565** |
| chronos_bolt | **0.900** | 0.817 | 0.783 | 0.774 | 0.662 | 0.418 | 0.729 | 0.678 | 0.778 | 0.582 |
| autotheta | 1.140 | 0.888 | 1.038 | 0.826 | 0.636 | 0.502 | 0.766 | 0.708 | 0.864 | 0.587 |
| autoets | 1.916 | 1.770 | 2.663 | 0.799 | 0.707 | 0.666 | 0.725 | 0.680 | 0.835 | 0.615 |
| mstl | 1.130 | 1.115 | 0.719 | 0.821 | 0.714 | 1.925 | 0.748 | 0.742 | 0.874 | 0.647 |
| gbdt_calendar | 1.772 | 1.334 | 1.299 | 1.118 | 0.880 | 0.939 | 0.776 | 0.730 | 0.894 | 1.086 |
| naive_last | 2.521 | 1.770 | 2.326 | 1.312 | **0.589** | 0.524 | 0.987 | 0.906 | 0.970 | 0.585 |
| naive_mean | 2.317 | 2.986 | 2.751 | 2.280 | 1.654 | 2.039 | 0.726 | 0.679 | 0.834 | 6.766 |
| seasonal_naive | 1.317 | 1.062 | 0.954 | 1.148 | 0.810 | 1.018 | 0.984 | 1.050 | 1.091 | 0.920 |
| seasonal_profile | 1.208 | 1.348 | 1.245 | 1.119 | 0.830 | 1.122 | 0.788 | 0.785 | 0.853 | 1.350 |

Family means across everything: chronos 0.594, timesfm 0.600, statistical 0.983,
ml 1.035, naive 1.334. (The GBDT number carries the feature-phase handicap disclosed
in `models.py`; read it as a lower bound on that family.)

**The seven findings that survived checking:**

1. **Seasonal strength predicts the winner better than model choice.** Foundation
   models dominate the left of the table and are irrelevant to the right.
2. **The margin depends entirely on the classical baseline you pick.** Against AutoETS
   on half-hourly load, TimesFM looks 4.7x better (0.570 vs 2.663). Against **MSTL**,
   the right tool for a 48-period season, the gap is **21%** (0.570 vs 0.719). On the
   daily dataset the story is different but not reversed: TimesFM wins the cell
   (0.705) and both Chronos variants beat all three classicals, yet the whole
   classical-to-foundation field compresses into a 0.774 to 0.826 band, a spread that
   would not survive 16 origins of noise. Adding MSTL was the single most
   consequential change to this study.
3. **No hallucinated structure.** On the three no-structure series the foundation
   models land within 1% of the context mean (noise 0.721 vs 0.726; returns 0.678 vs
   0.679) and 7% better on quakes. Models that *assume* structure are punished;
   seasonal naive is above 1.0 on all three.
4. **On a random walk it is a tie.** btc_px: chronos_2 0.565, timesfm 0.581,
   naive_last 0.585, a 3.5% spread at 16 origins. And naive_mean 6.766 confirms the
   harness punishes the wrong naive.
5. **On spiky weakly-seasonal data the lag wins.** pm25 goes to naive_last (0.589)
   ahead of every foundation model (0.648 to 0.675). Surges dominate the error and
   persistence is the best available answer between them.
6. **Telemetry is the biggest foundation-model win, and the classical wipeout.** BOOM
   (six-series panel, one group): chronos_2 0.357 vs naive_last 0.524, 32% better.
   **MSTL scores 1.925 and its 80% interval covered 0.0%**: STL's
   stable-seasonal-plus-smooth-trend assumption cannot survive 245x IQR surges.
7. **The operational case beats the accuracy case.** Mean ms per forecast:
   chronos_bolt 14.5, chronos_2 37.8, autotheta 146.6, timesfm 211.2, mstl 625.2,
   autoets 1,010.9, gbdt 1,137.7. The foundation models are the cheapest non-trivial
   option because they do a forward pass instead of refitting. Chronos-Bolt is **70x
   faster than AutoETS**.

### Multivariate: joint helps only in proportion to correlation

Chronos-2 on the same BOOM groups, joint vs one-at-a-time (TimesFM 2.5 cannot do
this):

| group | variates | mean abs corr | MASE joint | MASE single | joint better |
|---|---|---|---|---|---|
| ds-139-5T | 8 | 0.164 | 0.368 | **0.338** | 39.6% |
| ds-1558-5T | 8 | 0.198 | 0.472 | **0.469** | 47.9% |
| ds-2-5T | 8 | 0.430 | **0.153** | 0.160 | 52.1% |

At correlation 0.16 joint modeling is 9% **worse**; at 0.43 it is 4% better.
Cross-variate attention spends capacity on other series, so "multivariate" is a bet
that your variates are related, checkable with `np.corrcoef` before spending anything.
Effect sizes are small and this is 48 comparisons per group: directionally clear,
quantitatively provisional.

### Calibration: nobody is trustworthy out of the box

Coverage of the nominal 80% interval: TimesFM ranged 74 to 89% (most consistent),
Chronos-Bolt hit 54.4% on half-hourly load (over-confident), and the classical models
ranged from 62% (under-covered, on the strongly seasonal series) to 93% (over-wide),
with MSTL at 0.0% on telemetry. Conformalize on a holdout before any decision depends
on an interval.

## 6. Two harness bugs that produced confident wrong numbers

Both were mine, both are fixed, and both are the reason `bakeoff_full2` supersedes
`bakeoff_full`:

1. **A flat 2048-point context on the 553-observation daily series left exactly ONE
   usable origin.** That dataset silently contributed 10 forecasts instead of 200 and
   its whole row was one lucky day (TimesFM appeared at 0.584; the honest figure is
   0.705). Fixed by capping the context at half the usable span; the corrected
   16-origin configuration yields 176 forecasts on that dataset.
2. **Panel seasonal strength was taken from `series[0]`**, which labeled BOOM 0.009
   when the sampled mean is 0.23, putting a mixture in the wrong place on the
   structure axis. Now averaged across the series actually benchmarked.

A third and fourth surfaced in review, after the committed runs: the GBDT
feature-phase bug and the family-dependent context caps, both fixed in code and
disclosed in the README's threats-to-validity section. The committed artifacts predate
those fixes.

## 7. Capability probe: covariates, scenarios, horizons, backcast (2026-07-29)

Run against the installed `timesfm==2.0.2` / TimesFM 2.5 to answer the business
questions directly rather than from documentation. All measured.

### Arbitrary horizons: yes, and one production footgun

`forecast(horizon=h)` returns exactly `h` steps for h = 1, 7, 10, 33, 64: point shape
`(n, h)` and quantiles `(n, h, 10)`. So "forecast the next 10 hours" is `horizon=10`;
no padding, no rounding to a patch multiple.

**Asking for a horizon ABOVE the compiled `max_horizon` is silently allowed.**
Compiled at `max_horizon=64`, `horizon=128` returned a `(1, 128)` array with no error
or warning. Nothing tells you that you are outside the bounds the model was configured
for. Validate the horizon in your own service layer.

Error growth with horizon on a clean synthetic sine was mild (MAE 0.643 at h=1 to
0.895 at h=64). **Do not quote that**: it is one synthetic series; the ten-dataset
study used a fixed horizon per dataset, so there is no real-data horizon-scaling curve
here.

### Confidence intervals: yes, but uncalibrated

`use_continuous_quantile_head=True` gives `(n, h, 10)`: mean plus deciles 0.1 to 0.9
from the optional 30M quantile head. Coverage as measured across the corpus is in
section 5's calibration note. **Usable, not trustworthy; conformalize on a holdout
before any decision depends on the interval.**

### Covariates and what-if scenarios: yes, after two packaging fixes

`model.forecast_with_covariates()` exists and takes `dynamic_numerical_covariates`,
`dynamic_categorical_covariates`, `static_numerical_covariates`,
`static_categorical_covariates`, plus `xreg_mode`, `ridge` and normalization options.
Two prerequisites that each cost a debugging cycle:

1. **`return_backcast=True` is mandatory in `ForecastConfig`.** Otherwise it raises
   *"For XReg, `return_backcast` must be set to True"*. The regression needs the
   in-context fit to subtract from.
2. **`pip install timesfm[xreg]` pins `jax==0.2.22`, which has no working jaxlib on
   Python 3.10 / Apple Silicon**: it installs and then fails at import with *"jax
   requires jaxlib to be installed"*. Overriding with `jax>=0.4.30 jaxlib>=0.4.30`
   (resolved to 0.6.2) made it work. If you plan to use covariates, budget for that.

**Scenario forecasting demonstrably works.** One origin on UK demand, varying only the
*future* embedded-solar covariate:

| scenario | mean delta vs as-recorded | max abs delta |
|---|---|---|
| solar zeroed (heavy overcast) | **+1,892 MW** | 4,179 MW |
| solar x 0.5 | +946 MW | 2,089 MW |
| solar x 1.5 | -946 MW | 2,089 MW |

The signs are physically correct (less embedded PV means more grid demand) and the
response is monotonic. That is a working what-if lever.

A one-origin spot-check suggested covariates made accuracy *worse* (MAE 1,061 to
1,153 MW). **That was wrong, and section 7a settles it.** One origin is not a
measurement, the same mistake as the 8-origin smoke test that said TimesFM lost to
the weekly lag. `xreg_mode` does matter: `"xreg + timesfm"` (1,153) beat
`"timesfm + xreg"` (1,360) on that origin, and `"xreg + timesfm"` is used throughout
section 7a.

## 7a. Covariate evaluation: 129 daily origins, `covariate_test.py`

The proper run: same model, same origins, four variants. `plain` = no covariates,
`xreg_energy` = embedded solar + wind, `xreg_calendar` = day-of-week + bank-holiday
flags, `xreg_both` = all four. Horizon 48, context 512, ridge 1.0, `xreg + timesfm`.

One framing caveat before the numbers, disclosed in `covariate_test.py` itself: the
energy covariates feed the model the **actual realized** solar and wind values over
the forecast window. Real deployments must forecast their covariates, so the gains
below are a ceiling measured under perfect covariate foresight, not an achievable
production number.

**Overall MASE (129 origins):**

| variant | MASE | MAPE % | vs plain |
|---|---|---|---|
| `xreg_energy` | **0.573** | 5.04 | **-24.1%** |
| `xreg_both` | 0.657 | 5.85 | -12.9% |
| `plain` | 0.755 | 6.72 | baseline |
| `xreg_calendar` | 0.784 | 7.06 | **+3.8% worse** |

Win rate against plain, per origin: `xreg_energy` **80.6%**, `xreg_both` 67.4%,
`xreg_calendar` 41.9%. Cost: 75 to 83 ms per forecast, about 11% more latency.

**By day type, which is where it gets interesting:**

| variant | holiday (n=4) | weekday (n=88) | weekend (n=37) |
|---|---|---|---|
| `plain` | 1.031 | 0.690 | 0.881 |
| `xreg_energy` | 0.960 | **0.534** | **0.624** |
| `xreg_calendar` | **0.748** | 0.711 | 0.959 |
| `xreg_both` | 0.833 | 0.583 | 0.816 |

Four conclusions, in decreasing order of confidence:

1. **Physical drivers are the real win.** Solar + wind cut MASE 24% overall, 23% on
   weekdays and 29% on weekends, and win on 4 origins in 5 (under the foresight
   ceiling above). Embedded generation is a genuine exogenous driver of *net* grid
   demand, and the model cannot infer it from the target series alone.
2. **A calendar flag is a net LOSS overall.** `xreg_calendar` is worse than plain
   (0.784 vs 0.755), degrading weekdays (0.690 to 0.711) and weekends (0.881 to
   0.959). The naive instinct, "just add day-of-week and holidays", makes 361 days a
   year worse.
3. **The calendar flag does fix the holiday hole.** Plain TimesFM on bank holidays
   scores **1.031, worse than the naive it is scaled against**; the calendar variant
   scores **0.748, a 27.5% improvement**. The diagnosis ("it has no calendar") is
   correct and the fix works, but **n = 4 holidays**, so this is directional, not
   settled.
4. **Blending dilutes.** `xreg_both` beats plain everywhere, yet loses to
   `xreg_energy` on weekdays and weekends and to `xreg_calendar` on holidays.
   Averaging two covariate sets with opposite strengths gets you neither.

**The design that follows from this:** use physical/exogenous drivers as standing
covariates, and treat holidays as a *separate regime*, a switch, a holiday-specific
model, or a post-hoc adjustment, rather than mixing a holiday dummy into the everyday
model.

### Backcast: an anomaly-detection primitive nobody advertises

`return_backcast=True` changes the output to `(1, 504)` for a 512-step context and
24-step horizon: 480 reconstructed context points (everything except the first patch)
plus the 24 forecast points. `|actual - backcast|` over history is a ready-made
surprise score, which makes the same model serve anomaly detection without a second
system.

### Fine-tuning: documented, not verified here

The TimesFM repo added a LoRA example via HuggingFace Transformers + PEFT on
2026-04-09, and the model class exposes `save_pretrained`, `push_to_hub` and
`load_checkpoint`. Everything in this study is zero-shot; I have not run the
fine-tuning path, so treat it as available-and-documented rather than measured.

## 8. Still open

1. **Rerun Toto with the Toto-2.0 weights.** The checkpoint benchmarked here is
   Toto-Open-Base-1.0; the 2.0 family that tops GIFT-Eval is the natural next
   opponent, and the rerun would also pick up the post-review harness fixes.
2. **Fine-tuning.** TimesFM ships a LoRA example (Apr 2026); everything here is
   zero-shot. The single largest unexplored lever, since zero-shot plus physical
   covariates is already 24% ahead of zero-shot alone (under covariate foresight).
3. **Covariate forecasts instead of actuals**, to turn the section 7a ceiling into an
   achievable number.
4. **Holidays at n=4.** The -27.5% calendar effect wants a full year (8 UK bank
   holidays) before it carries weight.
5. **A holiday-regime design** rather than a holiday dummy; section 7a finding 4
   suggests a switch or separate model, and that is untested.
6. **Origins are 16 per series** in the main bake-off: enough to rank, not enough to
   separate models within ~5%. The covariate test uses 129.
7. **Conformal calibration.** No model tested delivered trustworthy 80% intervals.
