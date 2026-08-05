# Time-series foundation model bake-off

Thirteen models across four incompatible Python environments, scored under one set of
rules. No API keys, no GPU required.

The question is not "do foundation models work" — that is settled here. It is **which one,
at what horizon**, and the answer moves.

Companion code for the writing at [satsawat.ai](https://satsawat.ai).

---

## Contents

- [Headline results](#headline-results)
- [Caveats — read before quoting](#read-these-caveats-before-quoting-the-numbers)
- [Experimental protocol](#experimental-protocol)
- [The datasets](#the-datasets)
- [Metric definitions](#metric-definitions-enforced-in-code)
- [Why four environments](#why-four-environments)
- [The forecast contract](#the-forecast-contract)
- [Quick start](#quick-start)
- [Layout](#layout)
- [Environment notes that cost real time](#environment-notes-that-cost-real-time)
- [Adding a model](#adding-a-model)
- [Threats to validity](#threats-to-validity)
- [Superseded runs](#superseded-runs)
- [Licence](#licence)

---

## Headline results

> **Terminology.** One **cell** = one dataset paired with one horizon: all 13 models forecasting
> the same series, from the same origins, over the same distance, lowest MASE takes it. The
> article calls these *contests* — same thing, plainer word for a general reader. The code and
> this README say `cell` because that is what the variables are called.

38 dataset × horizon cells: 10 datasets, horizons of 1, 7, 14 and 30 days, 13 models across
4 environments, 16 rolling origins each. Lowest MASE per cell:

| family | cells won |
|---|---|
| **Foundation** | **30** |
| Naive | 5 |
| Classical | 2 |
| ML (GBDT) | 1 |

Per model:

| model | cells won |
|---|---|
| Chronos-2 | **13** |
| TimesFM 2.5 | 6 |
| Moirai-2 | 6 |
| Chronos-Bolt | 5 |
| **Toto-2.0** | **0** |

Three things worth sitting with:

**AutoETS won nothing.** Neither did the seasonal naive. If you are benchmarking a foundation
model against AutoETS to decide whether to adopt one, that question is already answered.

**The current GIFT-Eval leader won nothing either.** Toto-2.0 ran in all 38 cells with a
mean rank of 9.8 out of 13 (best 4, worst 13). That is not evidence Toto is bad — it is
evidence that a ranking computed on one benchmark at one horizon does not predict performance
on your data at your horizon. Note also that BOOM is Datadog's own benchmark and Toto trains
on Datadog telemetry, so part of this study is home turf for it.

**The horizon changes the winner on the same data.** Bangkok PM2.5:

| horizon | best foundation | best naive | winner |
|---|---|---|---|
| 1-day | 0.647 | **0.589** | naive_last |
| 7-day | **0.833** | 0.964 | Moirai-2 |
| 14-day | **0.962** | 1.098 | Moirai-2 |
| 30-day | **1.060** | 1.100 | Moirai-2 |

A single-horizon benchmark measures one point on that curve and reports it as a property of
the model. The crossover for this series sits somewhere between day 1 and day 7; this study
does not localise it further.

### Read these caveats before quoting the numbers

- 16 origins per cell. Enough to rank, not enough to separate models within a few percent —
  treat any gap under ~5% as a tie.
- Long horizons eat the series. On the 129-day datasets the 30-day cells run few origins and
  should be read as directional.
- BOOM carries 14 days per series, so its 14- and 30-day cells are impossible and are skipped
  rather than fudged. Its cells use one sampled series, which is not the panel — the 24-series
  aggregate in `results/bakeoff_full2*` behaves differently.
- **The denominators are not identical.** TimesFM is absent from two cells —
  `boom_telemetry_5t@ds-139-5T_v003#h2016` and `uk_demand_30min#h1440` — from runner failures
  left visible rather than papered over. Its 6 wins therefore come from 36 cells while
  Chronos-2's 13 come from 38. Toto ran the full 38 and won none.
- Every model ran zero-shot. No fine-tuning, which would move all of these.

---

## Experimental protocol

### Rolling origins

Origins are evenly spaced across the usable tail of each series, oldest first
([`benchmark.py:build_origins`](benchmark.py)):

```python
first, last = context, n - horizon
origins = np.linspace(first, last - 1, count)     # count = 16 by default
```

The lower bound is the context length, so every origin has a full window behind it. The upper
bound is `n - horizon`, so every origin has a complete horizon of actuals ahead of it. No
origin is ever scored against a partially observed future, and no origin is padded.

If the series cannot supply the requested number of origins the run **fails loudly** rather
than silently reducing `count`. This exists because of a real bug: a flat 2,048-point context
on a 553-observation daily series left exactly one usable origin, and that dataset contributed
10 forecasts instead of 176 while still producing a plausible-looking row. `MIN_CONTEXT`
carries the per-dataset overrides that came out of that (`uk_demand_daily: 120`).

### Contexts

Default context is 2,048 steps, clamped upward per dataset by `MIN_CONTEXT`. Foundation models
receive `use_ctx` directly. The same window is used for the MASE denominator, so the scale
factor and the model input are always drawn from identical data.

### Horizon conversion

Horizons are specified in **days** and converted per dataset frequency at runtime:

```python
horizon_steps = round(horizon_days * 1440 / STEP_MIN[key])
```

So "7-day" means the same business decision on 30-minute settlement data (336 steps) as on
hourly air quality (168 steps). The native `HORIZONS` table is the day-ahead default:

| dataset | native horizon | in steps |
|---|---|---|
| `uk_demand_30min` | 1 day | 48 |
| `uk_demand_daily` | 7 days | 7 |
| hourly series | 1 day | 24 |
| `boom_telemetry_5t` | 4 hours | 48 |

BOOM is deliberately different: 5-minute resolution over 14 days per series makes a 4-hour
horizon the meaningful observability decision, and makes 14- and 30-day cells impossible.

### Fairness constraints

Every model in a cell sees identical origins, identical context windows and an identical
horizon. This is enforced structurally rather than by convention — see
[the forecast contract](#the-forecast-contract).

---

## The datasets

Nine of the ten are windowed to 2026, which postdates every cutoff on the TimesFM 2.5 model
card. That is what makes "zero-shot" literally true rather than hopeful, and it is why ETTh1,
Electricity, Traffic and Monash are deliberately absent — those have been public long enough
that skill and recall cannot be separated.

| key | domain | freq | season | 2nd season | ratio scale? |
|---|---|---|---|---|---|
| `uk_demand_30min` | energy | 30min | 48 | 336 | yes |
| `uk_demand_daily` | energy | daily | 7 | — | yes |
| `bangkok_temp_1h` | weather | hourly | 24 | 168 | **no** (interval) |
| `london_temp_1h` | weather | hourly | 24 | 168 | **no** (interval) |
| `bangkok_pm25_1h` | air quality | hourly | 24 | 168 | yes |
| `btc_usd_1h` | finance | hourly | 24 | — | yes |
| `btc_returns_1h` | finance | hourly | 24 | — | **no** (crosses 0) |
| `quake_magnitude_seq` | geophysics | event order | 24 | — | yes |
| `white_noise_synth` | synthetic | hourly | 24 | — | **no** (crosses 0) |
| `boom_telemetry_5t` | telemetry | 5min | 288 | — | **no** (crosses 0) |

`uk_demand_30min` and `uk_demand_daily` are the same signal at two frequencies, included
specifically to test whether the frequency changes the verdict.

### The control group

Three series have **no trend and no seasonality by construction**:

- `btc_returns_1h` — hourly log returns. The textbook stationary, zero-mean, non-seasonal
  real series.
- `quake_magnitude_seq` — 2,657 successive global earthquake magnitudes (M≥4.5) indexed by
  event order, not time. Gutenberg–Richter draws are physically memoryless.
- `white_noise_synth` — Gaussian noise, seed 7, n=3096.

A forecasting benchmark without these is untrustworthy. If a model shows a large advantage
here, then the harness is leaking, the metric is broken, or the baseline is a straw man. In
this run the foundation models land within 1% of `naive_mean` on white noise, which is the
theoretically optimal answer — the control behaving exactly as designed.

### BOOM's leakage status is different, and worth stating precisely

BOOM (Datadog, [arXiv 2505.14766](https://arxiv.org/abs/2505.14766), NeurIPS 2025, Apache-2.0)
is 2024-dated, so it is **not** post-cutoff by date. It is defensible for TimesFM 2.5 on the
model card's declared corpus, and it is explicitly home turf for Toto. Read its cells with
that asymmetry in mind rather than as a clean zero-shot comparison.

The cross-environment study narrows to two datasets — the rationale is in
[`DATASET-CHOICE.md`](DATASET-CHOICE.md).

---

## Metric definitions, enforced in code

### MASE (primary)

The only metric valid for all ten series. For a forecast $\hat{y}$ over horizon $h$ at origin
$t$, with context $C_t$ and seasonal period $m$:

```
MASE = mean(|y - ŷ|) / denom(C_t, m)

denom(C, m) = mean(|C[m:] - C[:-m]|)      if len(C) > m
              mean(|diff(C)|)             otherwise
```

The denominator is the in-sample MAE of the **seasonal naive on that origin's own context**,
so 1.0 always means "no better than the naive it is scaled against". It is computed once, in
the truth file, and every model is divided by exactly the same number. A model cannot be
scored under a denominator it computed itself.

Note the fallback: when the context is shorter than one seasonal period the denominator
degrades to first differences, which is the non-seasonal naive. Both branches guard against a
zero denominator by falling back to 1.0.

### MAPE (restricted)

Computed **only** where the series is a ratio scale with a meaningful zero. That excludes five
of the ten datasets: both temperature series are an interval scale where zero is a convention
of the unit, and BTC returns, white noise and BOOM all cross zero.

The `ratio_scale` flag is written into the truth file at prepare time, not decided at report
time. Leading with MAPE would quietly change which datasets counted.

### WQL (probabilistic)

Weighted quantile loss over the nine deciles (`QLEVELS = 0.1 … 0.9`):

```
WQL = Σ_k 2·Σ_t [ q_k·(y-f) if y ≥ f else (1-q_k)·(f-y) ] / Σ|y|
```

Computed for classical models too, via their prediction intervals, so the probabilistic
comparison is not silently restricted to neural models.

### Coverage

Empirical coverage of the nominal 80% band. This is a **calibration check, not an accuracy
score** — do not rank on it. No model in this study delivered trustworthy 80% intervals across
the board, which is the finding, not a bug.

### The guard that matters

`score.py` **refuses to compare models scored on different origins.** A family that quietly
covered 12 origins while another did 16 would otherwise produce a comparison that looks fine
and is meaningless. The run aborts rather than reporting it.

---

## Why four environments

Python resolves one version of each package per environment, and these models disagree:

| environment | torch | numpy | why isolated |
|---|---|---|---|
| core | 2.13.0 | 2.2.6 | TimesFM + Chronos ×2 + classical + ML + naive |
| toto | 2.7.0 | 1.26.4 | `toto-ts==0.2.0` pins numpy back |
| moirai | 2.4.1 | 1.26.4 | `uni2ts==2.0.0` pins torch and numpy back |

NumPy 2.0 changed the C ABI, so extensions compiled against 2.x will not load against 1.26.
This is **binary incompatibility, not a version-label disagreement** — no resolver flag fixes
it, and no amount of `--force-reinstall` will either. The symptom is an import-time
`ValueError: numpy.dtype size changed`, which reads like a corrupt install and is not one.

Model checkpoints, for reproduction:

| model | checkpoint | context |
|---|---|---|
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | 2048 |
| Chronos-2 | `amazon/chronos-2` | 2048 |
| Chronos-Bolt | `amazon/chronos-bolt-small` | 2048 |

---

## The forecast contract

Each family runs in its own virtualenv and writes forecasts to disk;
[`score.py`](score.py) reads those and computes every metric while importing **no model code
at all**. Metric definitions therefore live in exactly one place.

[`forecast_contract.py`](forecast_contract.py) is stdlib-only by design — it is the one module
that must import cleanly in every environment regardless of what numpy or torch is pinned
there. It writes atomically and refuses to produce empty files.

The truth file, written by `run_core.py`, carries:

```jsonc
{
  "actuals":  { "<origin>": [...] },   // the horizon of true values
  "contexts": { "<origin>": [...] },   // the EXACT window each origin used
  "denom":    { "<origin>": 1.234 },   // per-origin MASE scale factor
  "horizon": 24, "season": 24, "context": 2048,
  "origins": [...],
  "ratio_scale": true                  // decides MAPE admissibility
}
```

**`contexts` is the load-bearing field.** Satellite runners read their model inputs from
there rather than reloading the series themselves. That is architectural, not stylistic: an
earlier version had them reload from cache, and because the BOOM cache carries no `series_id`
column, `run_moirai.py` silently forecast all 24 BOOM series concatenated into a single
sequence — producing entirely plausible numbers for the wrong data.

Neither of the two harness bugs in this project was caught by a test. Both were caught by a
number that looked slightly too good.

---

## Quick start

```bash
bash envs/setup.sh                                   # three virtualenvs, ~3.2 GB
.venv-core/bin/python datasets.py                    # fetch + cache, no API keys

.venv-core/bin/python   runners/run_core.py          # TimesFM, Chronos x2, classical, naive
.venv-toto/bin/python   runners/run_toto.py          # Toto-2.0
.venv-moirai/bin/python runners/run_moirai.py        # Moirai-2

python3 score.py                                     # stdlib only — no virtualenv needed
```

Run `run_core.py` **first** — it writes the truth file the satellite runners depend on.

Useful flags on the runners:

```bash
--origins 16          # rolling origins per cell (default 16)
--context 2048        # context steps, clamped up by MIN_CONTEXT
--horizon-days 7      # overrides the native horizon; converted per frequency
--keys bangkok_pm25_1h btc_returns_1h
```

`run_matrix.sh` drives the full 38-cell sweep across all four horizons.

New here? Start with **`timesfm_quickstart.ipynb`** — a hands-on comparison of TimesFM 2.5,
Chronos-2 and the classical stack on one series, at two horizons, already executed so you can
read it without running anything. It ends by re-running the same comparison at 1 day and
7 days, which is the cheapest way to see why a single-horizon benchmark misleads.

---

## Layout

```
datasets.py            ten series and their fetchers; caches to data/
prepare.py             eleven pre-call checks + structure diagnostics (STL, entropy, ADF/KPSS)
models.py              one adapter per family behind a single batch() call
benchmark.py           origins, horizons, MASE denominator — the shared definitions
covariate_test.py      TimesFM + XReg: do exogenous columns earn their keep?
forecast_contract.py   stdlib-only handoff between environments
score.py               metrics; imports no model
runners/               one per environment
envs/                  per-family requirements + setup.sh
results/               scored outputs — the numbers this README cites
```

---

## Environment notes that cost real time

None of these are in any model card:

- **Toto + Apple Silicon.** Toto samples a Gamma mixture and `aten::_standard_gamma` has no
  MPS kernel in torch 2.7. Needs `PYTORCH_ENABLE_MPS_FALLBACK=1`, set **before** torch imports.
- **Toto + Python 3.12.** `toto-ts` pulls lightning 2.3.3, which calls `pkg_resources` at
  import. setuptools 81 removed it and 3.12 does not bundle it, so a fresh environment dies
  on `import lightning`. Pinned `setuptools<81`.
- **Moirai + MPS.** gluonts' batchify builds float64 tensors for its own generated fields and
  MPS has no float64. Casting the input is not enough — the runner defaults to CPU.
- **LightGBM on macOS.** The wheel installs but cannot load without `libomp.dylib`;
  scikit-learn's `HistGradientBoostingRegressor` stands in.

Budget for environment isolation before you budget for GPU time. It is the part that will
actually stall you.

---

## Adding a model

1. Write an adapter in `models.py` exposing `batch(series, horizon) -> Forecast`, where
   `Forecast.quantiles` is `(h, 9)` aligned to `QLEVELS` or `None`.
2. If it installs alongside core, add it to `default_registry()`. If it does not, give it its
   own `envs/requirements-<name>.txt` and a runner under `runners/`.
3. A satellite runner reads `actuals`/`contexts`/`denom` from the truth file and **must not**
   reload the raw series. That rule is why the BOOM bug cannot recur.
4. Write via `forecast_contract.write_forecasts`. Do not hand-roll the JSON.

`score.py` needs no change — it discovers whatever is on disk and enforces the origin guard.

---

## Threats to validity

Stated plainly, because the numbers above are only worth what these are worth:

**Sample size.** 16 origins per cell supports ranking, not fine separation. Gaps under ~5%
are noise. The 30-day cells on 129-day datasets run few origins and are directional only.

**Single-series BOOM cells.** The headline BOOM figures sample one series. The 24-series panel
aggregate behaves differently and lives in `results/bakeoff_full2*`.

**Zero-shot only.** No fine-tuning anywhere. Fine-tuning would move every number here, quite
possibly reordering the table.

**No engineered global model.** The GBDT here uses calendar features only. The real incumbent
in retail and utilities is a global gradient-boosted model with years of history and
promotional covariates, and it is not represented. This is the most likely way the "foundation
models won" conclusion is overstated.

**Leakage is controlled by date, not proven.** Windowing to 2026 postdates the declared
cutoffs. It cannot rule out a corpus that was not fully declared.

**Unequal denominators.** TimesFM ran 36 of 38 cells; every other model ran all 38. Its win
count is not directly comparable to Chronos-2's without that caveat.

---

## Superseded runs

`results/` keeps the intermediate runs. Cite **`bakeoff_full2*`** — the only run with MSTL
included and both harness bugs fixed. `bakeoff_full*` (no `2`) predates the origin-count fix
and treats a panel's seasonal strength as its first series; `*_smoke` and `*_mstlsmoke` are
6–8 origin sanity runs. They are kept so the corrections stay auditable. Do not cite them.

`_archive_v1_timesfm_only/` is the original single-model study, kept deliberately: it is a
worked example of how easily a foundation-model gain gets overstated when the field is one
model wide.

---

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Data and model weights carry their own terms — see
[`NOTICE.md`](NOTICE.md). Note
**Moirai-2's weights are CC-BY-NC-4.0 (non-commercial)**; TimesFM, Chronos and Toto are
Apache-2.0.
