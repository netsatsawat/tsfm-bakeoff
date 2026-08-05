# Time-series foundation model bake-off

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white) ![13 models](https://img.shields.io/badge/models-13-eb6834) ![38 cells](https://img.shields.io/badge/dataset%C3%97horizon%20cells-38-2a78d6) ![No API keys](https://img.shields.io/badge/API%20keys-none-1baf7a)

Thirteen models, three isolated Python environments plus a dependency-free scorer, one
set of rules. No API keys, no GPU required.

The question is not "do foundation models work". On seasonal data, they do, and this
study settles it again. The question is **which one, at what horizon**, and the answer
moves more than any leaderboard suggests.

Companion code for the writing at [satsawat.ai](https://satsawat.ai).

## 🧭 Why this repository exists

A time-series foundation model (TSFM) is a pretrained model you call the way you call an
LLM: hand it the recent history of any series, get a forecast back, no fitting step. That
pitch is genuinely attractive, because the fitting step is where classical forecasting
projects go to die. But the pitch arrives with leaderboards, and I wanted to know what
survives contact with data the models have never seen, against baselines configured in
good faith.

Building that comparison turned out to be its own finding: the leading TSFMs cannot
share one Python environment (details below), which I suspect quietly discourages
exactly this kind of head-to-head. The engineering to make thirteen models comparable,
one venv per family with forecasts handed off on disk, is most of what this repository
is.

---

## Contents

- [Headline results](#headline-results)
- [Caveats: read before quoting](#read-these-caveats-before-quoting-the-numbers)
- [Experimental protocol](#experimental-protocol)
- [The datasets](#the-datasets)
- [Metric definitions](#metric-definitions-enforced-in-code)
- [Why three environments](#why-three-environments)
- [The forecast contract](#the-forecast-contract)
- [Quick start](#quick-start)
- [Layout](#layout)
- [Environment notes that cost real time](#environment-notes-that-cost-real-time)
- [Adding a model](#adding-a-model)
- [Threats to validity](#threats-to-validity)
- [Artifact provenance and superseded runs](#artifact-provenance-and-superseded-runs)
- [License](#license)

---

## 🏁 Headline results

> **Terminology.** One **cell** = one dataset paired with one horizon: all 13 models
> forecasting the same series, from the same origins, over the same distance, lowest
> MASE takes it. The articles call these *contests*, a plainer word for the same thing.
> The code and this README say `cell` because that is what the variables are called.

38 dataset x horizon cells: 10 datasets, horizons of 1, 7, 14 and 30 days, 13 models,
16 rolling origins each. Lowest MASE per cell:

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
| **Toto (Toto-Open-Base-1.0)** | **0** |

Three things worth sitting with:

**AutoETS won nothing.** Neither did the seasonal naive. If you are benchmarking a
foundation model against AutoETS to decide whether to adopt one, that question is
already answered.

**The GIFT-Eval leader's open predecessor won nothing either.** A naming correction
first, because an earlier draft of this study got it wrong: the checkpoint benchmarked
here is **Toto-Open-Base-1.0**, Datadog's openly released model, not the newer Toto-2.0
family that currently tops the GIFT-Eval leaderboard. (The result key `toto_2p0` in the
artifacts is a slip that came from the `toto-ts` 0.2.0 *package* version; it is kept so
the committed results stay internally consistent.) That checkpoint ran all 38 cells
with a mean rank of 9.8 out of 13 (best 4, worst 13) and won none, despite BOOM being
Datadog's own benchmark and Toto training on Datadog telemetry, so part of this study
is home turf for it. The honest conclusions: leaderboard pedigree of a model family
does not transfer automatically to its earlier open checkpoint, and rank on one
benchmark does not predict performance on your data at your horizon. Rerunning with
the Toto-2.0 weights is queued as future work.

**The horizon changes the winner on the same data.** Bangkok PM2.5:

| horizon | best foundation | best naive | winner |
|---|---|---|---|
| 1-day | 0.647 | **0.589** | naive_last |
| 7-day | **0.833** | 0.964 | Moirai-2 |
| 14-day | **0.962** | 1.098 | Moirai-2 |
| 30-day | **1.060** | 1.100 | Moirai-2 |

A single-horizon benchmark measures one point on that curve and reports it as a
property of the model. The crossover for this series sits somewhere between day 1 and
day 7; this study does not localize it further.

### Read these caveats before quoting the numbers

- 16 origins per cell. Enough to rank, not enough to separate models within a few
  percent. Treat any gap under ~5% as a tie.
- Long horizons crowd the short series. On the 129-day datasets the 30-day cells still
  run 16 origins, but the origins bunch into a short usable tail and their horizons
  overlap heavily, so those cells lean on partially shared information.
- BOOM carries 14 days per series, so its 14- and 30-day cells are impossible and are
  skipped rather than fudged. Its cross-environment cells use one sampled series, which
  is not the panel; the six-series panel aggregate (one BOOM group, `ds-139-5T`) in
  `results/bakeoff_full2*` behaves differently.
- **The denominators are not identical.** TimesFM is absent from two cells,
  `boom_telemetry_5t@ds-139-5T_v003#h2016` and `uk_demand_30min#h1440`, from runner
  failures left visible rather than papered over. Its 6 wins therefore come from 36
  cells while Chronos-2's 13 come from 38. Toto's h1 crash on the daily series was
  fixed and that cell re-run; `results/matrix.log` predates the re-run.
- Every model ran zero-shot. No fine-tuning, which would move all of these.

---

## 🔬 Experimental protocol

### Rolling origins

Origins are evenly spaced across the usable tail of each series, oldest first
([`benchmark.py:build_origins`](benchmark.py)):

```python
first, last = context, n - horizon
origins = np.linspace(first, last - 1, count)     # count = 16 by default
```

The lower bound is the context length, so every origin has a full window behind it. The
upper bound is `n - horizon`, so every origin has a complete horizon of actuals ahead
of it. No origin is ever scored against a partially observed future, and no origin is
padded.

When a series cannot supply the requested number of origins, the harness degrades the
count and now prints a warning saying so (in the runs committed here it degraded
silently; the aborts only fire at zero origins). This area carries a scar: a flat
2,048-point context on a 553-observation daily series once left exactly one usable
origin, and that dataset contributed 10 forecasts instead of 200 while still producing
a plausible-looking row. `MIN_CONTEXT` carries the per-dataset overrides that came out
of that (`uk_demand_daily: 120`), and the context cap at half the usable span is the
structural fix.

### Contexts

Default context is 2,048 steps, clamped per dataset so the context never eats the
series. One honesty note that matters for reading the committed artifacts: in the runs
bundled here, each neural family applied its own context cap (Chronos-2 clamps batches
to the shortest task; TimesFM and Chronos-Bolt slice their own windows), so context
windows were family-dependent even though origins and horizons were identical. The
harness now slices every task to the same `use_ctx` window before any adapter sees it.
The MASE denominator has always used that same window, so the scale factor and the
model input are drawn from the same data.

### Horizon conversion

Horizons are specified in **days** and converted per dataset frequency at runtime:

```python
horizon_steps = round(horizon_days * 1440 / STEP_MIN[key])
```

So "7-day" means the same business decision on 30-minute settlement data (336 steps) as
on hourly air quality (168 steps). The native `HORIZONS` table is the day-ahead
default:

| dataset | native horizon | in steps |
|---|---|---|
| `uk_demand_30min` | 1 day | 48 |
| `uk_demand_daily` | 7 days | 7 |
| hourly series | 1 day | 24 |
| `boom_telemetry_5t` | 4 hours | 48 |

BOOM is deliberately different: 5-minute resolution over 14 days per series makes a
4-hour horizon the meaningful observability decision, and makes 14- and 30-day cells
impossible.

### Fairness constraints

Every model in a cell sees identical origins and an identical horizon, enforced
structurally through [the forecast contract](#the-forecast-contract); `score.py` aborts
on mismatched origins. Context equality is enforced by the task slice described above
(and was family-dependent in the bundled runs; see
[Artifact provenance](#artifact-provenance-and-superseded-runs)).

---

## 📦 The datasets

Nine of the ten are windowed to 2026, which postdates every cutoff on the TimesFM 2.5
model card. That is what makes "zero-shot" literally true rather than hopeful, and it
is why ETTh1, Electricity, Traffic and Monash are deliberately absent: those have been
public long enough that skill and recall cannot be separated.

| key | domain | freq | season | 2nd season | ratio scale? |
|---|---|---|---|---|---|
| `uk_demand_30min` | energy | 30min | 48 | 336 | yes |
| `uk_demand_daily` | energy | daily | 7 | none | yes |
| `bangkok_temp_1h` | weather | hourly | 24 | 168 | **no** (interval) |
| `london_temp_1h` | weather | hourly | 24 | 168 | **no** (interval) |
| `bangkok_pm25_1h` | air quality | hourly | 24 | 168 | yes |
| `btc_usd_1h` | finance | hourly | 24 | none | yes |
| `btc_returns_1h` | finance | hourly | 24 | none | **no** (crosses 0) |
| `quake_magnitude_seq` | geophysics | event order | 24 | none | yes |
| `white_noise_synth` | synthetic | hourly | 24 | none | **no** (crosses 0) |
| `boom_telemetry_5t` | telemetry | 5min | 288 | none | **no** (crosses 0) |

`uk_demand_30min` and `uk_demand_daily` are the same signal at two frequencies,
included specifically to test whether the frequency changes the verdict.

### The control group

Three series have **no trend and no seasonality by construction**:

- `btc_returns_1h`: hourly log returns. The textbook stationary, zero-mean,
  non-seasonal real series.
- `quake_magnitude_seq`: 2,657 successive global earthquake magnitudes (M >= 4.5)
  indexed by event order, not time. Gutenberg-Richter draws are physically memoryless.
- `white_noise_synth`: Gaussian noise, seed 7, n=3096.

A forecasting benchmark without these is untrustworthy. If a model shows a large
advantage here, then the harness is leaking, the metric is broken, or the baseline is a
straw man. In the ten-dataset run, TimesFM and both Chronos variants land within 1% of
`naive_mean` on white noise, which is the theoretically optimal answer; in the
cross-environment cells Toto and Moirai sit 1 to 3% above it. The control behaves as
designed, with the sampling noise you would expect.

### BOOM's leakage status is different, and worth stating precisely

BOOM (Datadog, [arXiv 2505.14766](https://arxiv.org/abs/2505.14766), NeurIPS 2025,
Apache-2.0) is 2024-dated, so it is **not** post-cutoff by date. The defense differs
per model, and for two of them there is none: for TimesFM 2.5 the argument is
provenance (Datadog-internal telemetry first published May 2025, after
GiftEvalPretrain was built); for Toto, BOOM is explicitly home turf. Chronos-2 and
Moirai-2 were both released after BOOM's publication, so its presence in their
training corpora cannot be ruled out from release dates alone. Read the BOOM cells
with those asymmetries in mind rather than as a clean zero-shot comparison.

The deeper cross-environment analysis concentrates on two datasets; the rationale is in
[`DATASET-CHOICE.md`](DATASET-CHOICE.md), and the full ten-dataset matrix lives in
`results/cross_env_scores.json`.

---

## 📏 Metric definitions, enforced in code

### MASE (primary)

The only metric valid for all ten series. For a forecast over horizon h at origin t,
with context C and seasonal period m:

```
MASE = mean(|y - yhat|) / denom(C, m)

denom(C, m) = mean(|C[m:] - C[:-m]|)      if len(C) > m
              mean(|diff(C)|)             otherwise
```

The denominator is the in-sample MAE of the **seasonal naive on that origin's own
context**, so 1.0 always means "no better than the naive it is scaled against". It is
computed once, in the truth file, and every model is divided by exactly the same
number. A model cannot be scored under a denominator it computed itself.

Note the fallback: when the context is shorter than one seasonal period the denominator
degrades to first differences, which is the non-seasonal naive. Both branches guard
against a zero denominator by falling back to 1.0.

### MAPE (restricted)

Computed **only** where the series is a ratio scale with a meaningful zero. That
excludes five of the ten datasets: both temperature series are an interval scale where
zero is a convention of the unit, and BTC returns, white noise and BOOM all cross zero.
(Earthquake magnitudes are technically a log scale; MAPE is kept there because every
value sits far from zero, and MASE remains the metric that carries weight.)

The `ratio_scale` flag is written into the truth file at prepare time, not decided at
report time. Leading with MAPE would quietly change which datasets counted.

### WQL (probabilistic)

Weighted quantile loss over the nine deciles (`QLEVELS = 0.1 ... 0.9`): twice the
pinball loss, averaged over the nine levels, normalized by the sum of |actuals| per
origin. Computed for classical models too, via their prediction intervals, so the
probabilistic comparison is not silently restricted to neural models.

One provenance note: `score.py` historically computed a plain per-point pinball mean
without the normalization, and the `wql` column in the committed
`results/cross_env_scores.json` uses that older formula (comparable within a dataset,
not across datasets). The code now matches the definition above; the `wql` columns in
`results/bakeoff_full2*` already did.

### Coverage

Empirical coverage of the nominal 80% band. This is a **calibration check, not an
accuracy score**; do not rank on it. No model in this study delivered trustworthy 80%
intervals across the board, which is the finding, not a bug.

### The guard that matters

`score.py` **refuses to compare models scored on different origins.** A family that
quietly covered 12 origins while another did 16 would otherwise produce a comparison
that looks fine and is meaningless. The run aborts rather than reporting it.

---

## 🧩 Why three environments

Python resolves one version of each package per environment, and these models disagree:

| environment | torch | numpy | why isolated |
|---|---|---|---|
| core | 2.13.0 | 2.2.6 | TimesFM + Chronos x2 + classical + ML + naive |
| toto | 2.7.0 | 1.26.4 | `toto-ts==0.2.0` pins numpy back |
| moirai | 2.4.1 | 1.26.4 | `uni2ts==2.0.0` pins torch and numpy back |

NumPy 2.0 changed the C ABI, so extensions compiled against 2.x will not load against
1.26. This is **binary incompatibility, not a version-label disagreement**; no resolver
flag fixes it, and no amount of `--force-reinstall` will either. The symptom is an
import-time `ValueError: numpy.dtype size changed`, which reads like a corrupt install
and is not one.

The scorer is the deliberate fourth participant: `score.py` runs on a bare system
Python with no third-party imports at all, so the metric definitions cannot drift
toward any family's environment.

Model checkpoints, for reproduction:

| model | checkpoint | context |
|---|---|---|
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | 2048 |
| Chronos-2 | `amazon/chronos-2` | 2048 |
| Chronos-Bolt | `amazon/chronos-bolt-small` | 2048 |
| Toto | `Datadog/Toto-Open-Base-1.0` | 2048 |
| Moirai-2 | `Salesforce/moirai-2.0-R-small` | 2048 |

The committed core-family artifacts were produced on Python 3.10; `envs/setup.sh`
defaults to 3.12 (both work, and the setup script now fails fast if neither uv nor a
recent-enough python3 is available).

---

## 🤝 The forecast contract

Each family runs in its own virtualenv and writes forecasts to disk;
[`score.py`](score.py) reads those and computes every metric while importing **no model
code at all**. Metric definitions therefore live in exactly one place.

[`forecast_contract.py`](forecast_contract.py) is stdlib-only by design. It is the one
module that must import cleanly in every environment regardless of what numpy or torch
is pinned there. It writes atomically and refuses to produce empty files.

The truth file, written by `run_core.py`, carries:

```jsonc
{
  "actuals":  { "<origin>": [...] },   // the horizon of true values
  "contexts": { "<origin>": [...] },   // the EXACT window each origin used
  "scale":    { "<origin>": 1.234 },   // per-origin MASE denominator
  "horizon": 24, "season": 24, "context": 2048,
  "origins": [...],
  "ratio_scale": true                  // decides MAPE admissibility
}
```

**`contexts` is the load-bearing field.** Satellite runners read their model inputs
from there rather than reloading the series themselves. That is architectural, not
stylistic: an earlier version had them reload from cache, and because the BOOM cache
carries no `series_id` column, `run_moirai.py` silently forecast all sampled BOOM
series concatenated into a single sequence, producing entirely plausible numbers for
the wrong data.

Neither of the two harness bugs in this project was caught by a test. Both were caught
by a number that looked slightly too good.

---

## 🚀 Quick start

```bash
bash envs/setup.sh                                   # three virtualenvs; budget ~10 GB
.venv-core/bin/python datasets.py                    # fetch + cache, no API keys

.venv-core/bin/python   runners/run_core.py          # TimesFM, Chronos x2, classical, naive
.venv-toto/bin/python   runners/run_toto.py          # Toto (Toto-Open-Base-1.0)
.venv-moirai/bin/python runners/run_moirai.py        # Moirai-2

python3 score.py                                     # stdlib only; no virtualenv needed
```

Run `run_core.py` **first**: it writes the truth file the satellite runners depend on.

Useful flags on the runners:

```bash
--origins 16          # rolling origins per cell (default 16)
--context 2048        # context steps, clamped per dataset
--horizon-days 7      # overrides the native horizon; converted per frequency
--keys bangkok_pm25_1h btc_returns_1h
```

`run_matrix.sh` drives the full 38-cell sweep across all four horizons.

New here? Start with **`timesfm_quickstart.ipynb`**: a hands-on comparison of TimesFM
2.5, Chronos-2 and the classical stack on one series, already executed so you can read
it without running anything. It ends by re-running the same comparison at 1 day and
7 days, which is the cheapest way to see why a single-horizon benchmark misleads.

One reproducibility caveat for refetchers: the NESO, Open-Meteo, USGS and Binance
sources are live APIs and the BOOM download is not pinned to a dataset revision, so a
fresh `datasets.py` run reproduces the study's series only as those sources allow. The
committed results were produced from the cache as fetched for this study.

---

## 🗂️ Layout

```
datasets.py            ten series and their fetchers; caches to data/
prepare.py             eleven pre-call checks + structure diagnostics (STL, entropy, ADF/KPSS)
models.py              one adapter per family behind a single batch() call
benchmark.py           origins, horizons, MASE denominator; the shared definitions
covariate_test.py      TimesFM + XReg: do exogenous columns earn their keep?
forecast_contract.py   stdlib-only handoff between environments
score.py               metrics; imports no model
runners/               one per environment
envs/                  per-family requirements + setup.sh
results/               scored outputs; the numbers this README cites
```

---

## 🛠️ Environment notes that cost real time

None of these are in any model card:

- **Toto + Apple Silicon.** Toto samples a Gamma mixture and `aten::_standard_gamma`
  has no MPS kernel in torch 2.7. Needs `PYTORCH_ENABLE_MPS_FALLBACK=1`, set **before**
  torch imports.
- **Toto + Python 3.12.** `toto-ts` pulls lightning 2.3.3, which calls `pkg_resources`
  at import. setuptools 81 removed it and 3.12 does not bundle it, so a fresh
  environment dies on `import lightning`. Pinned `setuptools<81`.
- **Moirai + MPS.** gluonts' batchify builds float64 tensors for its own generated
  fields and MPS has no float64. Casting the input is not enough; the runner defaults
  to CPU.
- **LightGBM on macOS.** The wheel installs but cannot load without `libomp.dylib`;
  scikit-learn's `HistGradientBoostingRegressor` stands in.

Budget for environment isolation before you budget for GPU time. It is the part that
will actually stall you.

---

## ➕ Adding a model

1. Write an adapter in `models.py` exposing `batch(series, horizon) -> Forecast`, where
   `Forecast.quantiles` is `(h, 9)` aligned to `QLEVELS` or `None`.
2. If it installs alongside core, add it to `default_registry()`. If it does not, give
   it its own `envs/requirements-<name>.txt` and a runner under `runners/`.
3. A satellite runner reads `actuals`/`contexts`/`scale` from the truth file and
   **must not** reload the raw series. That rule is why the BOOM bug cannot recur.
4. Write via `forecast_contract.write_forecasts`. Do not hand-roll the JSON.

`score.py` needs no change; it discovers whatever is on disk and enforces the origin
guard.

---

## ⚖️ Threats to validity

Stated plainly, because the numbers above are only worth what these are worth:

**Sample size.** 16 origins per cell supports ranking, not fine separation. Gaps under
~5% are noise. The 30-day cells on 129-day datasets crowd their origins into a short
tail with heavily overlapping horizons.

**Point forecasts are not the same functional across models.** Chronos-2 and Moirai-2
are scored on their median, TimesFM on its point head, Chronos-Bolt on its mean output,
Toto on its sample mean. Absolute-error metrics favor medians, so the mixture puts a
thumb on the scale that this study measures but does not remove.

**Family-dependent context caps in the bundled runs.** Origins and horizons were
identical, but each neural family applied its own context cap (see
[Contexts](#contexts)). The harness now slices every task to one window; the committed
artifacts predate that.

**The GBDT comparator ran handicapped.** A phase bug in its seasonal-position features
(fixed in `models.py`, disclosed there) misaligned training and forecast rows in the
bundled runs, so the GBDT rows are best read as lower bounds and the
foundation-vs-GBDT margins as upper bounds.

**Single-series BOOM cells.** The headline BOOM figures sample one series. The
six-series panel aggregate (one group, `ds-139-5T`) behaves differently and lives in
`results/bakeoff_full2*`.

**Zero-shot only.** No fine-tuning anywhere. Fine-tuning would move every number here,
quite possibly reordering the table.

**No engineered global model.** The GBDT here uses calendar features only. The real
incumbent in retail and utilities is a global gradient-boosted model with years of
history and promotional covariates, and it is not represented. This is the most likely
way the "foundation models won" conclusion is overstated.

**Leakage is controlled by date, not proven.** Windowing to 2026 postdates the declared
cutoffs. It cannot rule out a corpus that was not fully declared, and the BOOM
asymmetries above apply.

**Unequal denominators.** TimesFM ran 36 of 38 cells; every other model ran all 38. Its
win count is not directly comparable to Chronos-2's without that caveat.

---

## 🗃️ Artifact provenance and superseded runs

`results/` keeps the intermediate runs. Cite **`bakeoff_full2*`**: the only run with
MSTL included and both harness bugs fixed. `bakeoff_full*` (no `2`) predates the
origin-count fix and treats a panel's seasonal strength as its first series; `*_smoke`
and `*_mstlsmoke` are 6 to 8 origin sanity runs. They are kept so the corrections stay
auditable. Do not cite them.

The committed artifacts also predate four code fixes made after review, each disclosed
where it lives: the GBDT feature-phase fix and the shared context slice (both above),
the `score.py` WQL normalization, and the Toto sampling seed. Each fix changes future
runs, not the committed numbers; a full rerun under the fixed harness, ideally with the
Toto-2.0 weights, is the natural next run.

---

## 📄 License

Code: MIT ([`LICENSE`](LICENSE)). Data and model weights carry their own terms; see
[`NOTICE.md`](NOTICE.md). Note **Moirai-2's weights are CC-BY-NC-4.0 (non-commercial)**;
TimesFM, Chronos and Toto are Apache-2.0.

---

Written by [Satsawat Natakarnkitkul](https://satsawat.ai), a data and AI practitioner
in ASEAN. Companion repositories:
[agent-failure-lab](https://github.com/netsatsawat/agent-failure-lab),
[markov_and_hidden_markov_model](https://github.com/netsatsawat/markov_and_hidden_markov_model),
[fft-seasonality](https://github.com/netsatsawat/fft-seasonality). Newsletter:
[AI in Practice](https://satsawat.ai/#newsletter)
