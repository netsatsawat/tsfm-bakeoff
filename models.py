#!/usr/bin/env python3
"""One adapter per model family, so the bake-off is actually apples-to-apples.

Every model receives the same prepared context, the same origins, the same horizon, and
is scored by the same code. The differences that remain are the models.

Each family's real API is different in ways that matter, and each difference is a place a
naive harness silently cheats:

  TimesFM 2.5   compile(max_context, max_horizon) BEFORE forecasting; returns
                (point, quantiles) with quantiles[..., 0] = mean and 1..9 = deciles.
                Scales internally (normalize_inputs). Univariate only; covariates via
                XReg, which is not used here.
  Chronos-2     input MUST be 3-D (n_series, n_variates, history). Returns a list of
                (n_variates, 21, horizon), its own 21-level quantile grid, from which
                the deciles are indices 2,4,...,18. Handles NaN natively. Genuinely
                multivariate: pass n_variates > 1 and it shares information across them.
  Chronos-Bolt  older, simpler: predict_quantiles(list_of_1d, h, levels) -> (q, mean).
  statsforecast .forecast(y=..., h=..., level=[...]) on a raw array. Asking for
                level=[20,40,60,80] yields the eight symmetric decile bounds, so the
                classical models can be scored on WQL too rather than being quietly
                excluded from the probabilistic comparison.
  GBDT          owns NOTHING. You build the features, you scale, you handle the calendar.
                That is the point of including it: it is the incumbent that already knows
                what a bank holiday is.

Naive models are not filler. On a random walk the optimal forecast is the last value; on
zero-mean noise it is the context mean. Both are in the field so the no-structure datasets
have a correct answer to be measured against.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

QLEVELS = np.round(np.arange(1, 10) / 10.0, 2)      # 0.1 ... 0.9
SF_LEVELS = [20, 40, 60, 80]                        # -> the 8 symmetric decile bounds


@dataclass
class Task:
    """One forecasting job: a context, and the calendar the horizon lands on."""
    y: np.ndarray
    ctx_index: pd.DatetimeIndex
    fut_index: pd.DatetimeIndex
    horizon: int


@dataclass
class Forecast:
    point: np.ndarray                    # (h,)
    quantiles: np.ndarray | None = None  # (h, 9) aligned to QLEVELS


class BaseModel:
    name = "base"
    family = "naive"
    needs_positive_flag = False

    def batch(self, tasks: Sequence[Task], spec) -> list[Forecast]:
        raise NotImplementedError


# --------------------------------------------------------------------- naive
class NaiveLast(BaseModel):
    """Repeat the last observation. The optimal forecast for a random walk."""
    name, family = "naive_last", "naive"

    def batch(self, tasks, spec):
        return [Forecast(np.repeat(t.y[-1], t.horizon)) for t in tasks]


class NaiveMean(BaseModel):
    """Mean of the context. The optimal forecast for zero-mean stationary noise."""
    name, family = "naive_mean", "naive"

    def batch(self, tasks, spec):
        return [Forecast(np.repeat(t.y.mean(), t.horizon)) for t in tasks]


class SeasonalNaive(BaseModel):
    """Last season's values. The bar any seasonal claim must clear."""
    name, family = "seasonal_naive", "naive"

    def batch(self, tasks, spec):
        out = []
        for t in tasks:
            m = spec.season
            if len(t.y) < m:
                out.append(Forecast(np.repeat(t.y[-1], t.horizon)))
                continue
            reps = int(np.ceil(t.horizon / m))
            out.append(Forecast(np.tile(t.y[-m:], reps)[:t.horizon]))
        return out


class SeasonalProfile(BaseModel):
    """Median of the last four seasonal cycles: a cheap, robust climatology."""
    name, family = "seasonal_profile", "naive"

    def batch(self, tasks, spec):
        m, out = spec.season, []
        for t in tasks:
            cycles = [t.y[-k * m:len(t.y) - (k - 1) * m] for k in range(1, 5)
                      if len(t.y) >= k * m]
            if not cycles:
                out.append(Forecast(np.repeat(t.y[-1], t.horizon)))
                continue
            prof = np.median(np.vstack(cycles), axis=0)
            reps = int(np.ceil(t.horizon / m))
            out.append(Forecast(np.tile(prof, reps)[:t.horizon]))
        return out


# --------------------------------------------------------------- statistical
class StatsForecastModel(BaseModel):
    family = "statistical"

    def __init__(self, kind: str, max_context: int = 1500):
        self.kind = kind
        self.name = kind.lower()
        self.max_context = max_context

    def _build(self, season: int, season_long: int | None):
        from statsforecast.models import MSTL, AutoETS, AutoTheta
        if self.kind == "MSTL":
            # MSTL is the honest classical comparator for high-frequency data. AutoETS
            # alone is a straw man here: statsforecast's ETS is not built for a seasonal
            # period of 48, which is exactly why half-hourly load forecasting moved to
            # MSTL/TBATS. Give the classical side its real tool.
            periods = [season] + ([season_long] if season_long else [])
            return MSTL(season_length=periods, trend_forecaster=AutoETS(model="ZZN"))
        cls = {"AutoETS": AutoETS, "AutoTheta": AutoTheta}[self.kind]
        return cls(season_length=season)

    def batch(self, tasks, spec):
        mdl = self._build(spec.season, getattr(spec, "season_long", None))
        out: list[Forecast] = []
        for t in tasks:
            y = np.ascontiguousarray(t.y[-self.max_context:], dtype=np.float64)
            try:
                res = mdl.forecast(y=y, h=t.horizon, level=SF_LEVELS)
                point = np.asarray(res["mean"], dtype=float)
                cols = []
                for lv in sorted(SF_LEVELS, reverse=True):        # 80,60,40,20
                    cols.append(np.asarray(res[f"lo-{lv}"], dtype=float))
                cols.append(point)                                # the 0.5 slot
                for lv in sorted(SF_LEVELS):                      # 20,40,60,80
                    cols.append(np.asarray(res[f"hi-{lv}"], dtype=float))
                q = np.column_stack(cols)                         # (h, 9) ascending
                q = np.sort(q, axis=1)                            # enforce monotonicity
                out.append(Forecast(point, q))
            except Exception:  # noqa: BLE001 - a failed fit is a real outcome
                out.append(Forecast(np.repeat(y[-1], t.horizon)))
        return out


# ------------------------------------------------------------------------ ML
class GbdtCalendar(BaseModel):
    """Histogram gradient boosting on calendar + seasonal-lag features.

    The enterprise incumbent, and the model that HAS a calendar. Features are all known
    at forecast time for horizons up to one season, so no recursion and no leakage:
    position in season, day of week, weekend flag, harmonics, and the value one and two
    seasons back. Refit per origin on the context only.

    sklearn's HistGradientBoostingRegressor rather than LightGBM: same algorithm class,
    and LightGBM's wheel could not load on this machine (missing libomp.dylib).
    """
    name, family = "gbdt_calendar", "ml"

    def __init__(self, max_context: int = 4000):
        self.max_context = max_context

    @staticmethod
    def _features(idx: pd.DatetimeIndex, season: int, pos: np.ndarray,
                  lag1: np.ndarray, lag2: np.ndarray) -> np.ndarray:
        # `pos` is the ABSOLUTE position in the series modulo the season. An earlier
        # version derived it from np.arange(len(idx)), which is phase-shifted between
        # training rows and the forecast horizon whenever the origin is not a multiple
        # of the season; the committed bake-off artifacts predate this fix, so the GBDT
        # rows there are handicapped and read as lower bounds.
        ang = 2 * np.pi * pos / season
        dow = idx.dayofweek.to_numpy()
        return np.column_stack([
            pos, dow, (dow >= 5).astype(float),
            np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang),
            idx.hour.to_numpy(), lag1, lag2,
        ])

    def batch(self, tasks, spec):
        from sklearn.ensemble import HistGradientBoostingRegressor
        m, out = spec.season, []
        for t in tasks:
            y = t.y[-self.max_context:]
            idx = t.ctx_index[-len(y):]
            if len(y) < 3 * m + 1:
                out.append(Forecast(np.repeat(y[-1], t.horizon)))
                continue
            # training rows start at 2 seasons in, so both lags exist
            tr = np.arange(2 * m, len(y))
            Xtr = self._features(idx[tr], m, tr % m, y[tr - m], y[tr - 2 * m])
            ytr = y[tr]
            fut_lag1 = y[len(y) - m: len(y) - m + t.horizon]
            fut_lag2 = y[len(y) - 2 * m: len(y) - 2 * m + t.horizon]
            if len(fut_lag1) < t.horizon or len(fut_lag2) < t.horizon:
                out.append(Forecast(np.repeat(y[-1], t.horizon)))
                continue
            fut_pos = (len(y) + np.arange(t.horizon)) % m
            Xte = self._features(t.fut_index, m, fut_pos, fut_lag1, fut_lag2)
            mdl = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.08, max_depth=6,
                early_stopping=False, random_state=0)
            mdl.fit(Xtr, ytr)
            out.append(Forecast(np.asarray(mdl.predict(Xte), dtype=float)))
        return out


# ----------------------------------------------------------------- TimesFM 2.5
class TimesFM25(BaseModel):
    name, family = "timesfm_2p5", "timesfm"
    needs_positive_flag = True

    def __init__(self, context: int = 2048,
                 checkpoint: str = "google/timesfm-2.5-200m-pytorch"):
        self.context = context
        self.checkpoint = checkpoint
        self.name = f"timesfm_2p5_ctx{context}"
        self._model = None
        self._compiled_for: tuple[int, int, bool] | None = None

    def _ensure(self, horizon: int, positive: bool):
        import timesfm
        import torch
        if self._model is None:
            torch.set_float32_matmul_precision("high")
            self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.checkpoint)
        key = (self.context, horizon, positive)
        if self._compiled_for != key:
            self._model.compile(timesfm.ForecastConfig(
                max_context=self.context,
                max_horizon=horizon,
                normalize_inputs=True,             # TimesFM owns scaling
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=positive,        # only claim it when it is true
                fix_quantile_crossing=True,
            ))
            self._compiled_for = key
        return self._model

    def batch(self, tasks, spec):
        horizon = tasks[0].horizon
        model = self._ensure(horizon, bool(spec.positive))
        inputs = [np.ascontiguousarray(t.y[-self.context:], dtype=np.float32)
                  for t in tasks]
        point, quant = model.forecast(horizon=horizon, inputs=inputs)
        point, quant = np.asarray(point), np.asarray(quant)
        out = []
        for i in range(len(tasks)):
            # quant[..., 0] is the mean; 1..9 are the deciles
            q = quant[i, :horizon, 1:10] if quant.ndim == 3 else None
            out.append(Forecast(point[i, :horizon], q))
        return out


# ------------------------------------------------------------------- Chronos
class Chronos2(BaseModel):
    """Amazon Chronos-2. 3-D input only; 21 native quantile levels."""
    name, family = "chronos_2", "chronos"
    DECILE_IDX = [2, 4, 6, 8, 10, 12, 14, 16, 18]   # 0.1 .. 0.9 of its 21-level grid

    def __init__(self, context: int = 2048, checkpoint: str = "amazon/chronos-2"):
        self.context = context
        self.checkpoint = checkpoint
        self.name = f"chronos_2_ctx{context}"
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            from chronos import BaseChronosPipeline
            self._pipe = BaseChronosPipeline.from_pretrained(self.checkpoint)
        return self._pipe

    def batch(self, tasks, spec):
        import torch
        pipe = self._ensure()
        horizon = tasks[0].horizon
        L = min(self.context, min(len(t.y) for t in tasks))
        arr = np.stack([t.y[-L:] for t in tasks]).astype("float32")   # (B, L)
        x = torch.tensor(arr)[:, None, :]                             # (B, 1, L)
        preds = pipe.predict(x, prediction_length=horizon)
        out = []
        for p in preds:
            a = np.asarray(p)              # (n_variates=1, 21, horizon)
            q = a[0][self.DECILE_IDX, :].T                            # (h, 9)
            out.append(Forecast(q[:, 4].copy(), q))    # median as point (see README threats)
        return out

    def batch_multivariate(self, group: np.ndarray, horizon: int) -> np.ndarray:
        """group: (n_variates, L) -> (n_variates, horizon) median forecasts.

        The capability TimesFM 2.5 does not have, and the reason BOOM is in the corpus.
        """
        import torch
        pipe = self._ensure()
        x = torch.tensor(group.astype("float32"))[None]               # (1, V, L)
        pred = np.asarray(pipe.predict(x, prediction_length=horizon)[0])  # (V, 21, h)
        return pred[:, 10, :]                                          # the 0.5 level


class ChronosBolt(BaseModel):
    name, family = "chronos_bolt", "chronos"

    def __init__(self, context: int = 2048,
                 checkpoint: str = "amazon/chronos-bolt-small"):
        self.context = context
        self.checkpoint = checkpoint
        self.name = f"chronos_bolt_{checkpoint.rsplit('-', 1)[-1]}"
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            from chronos import BaseChronosPipeline
            self._pipe = BaseChronosPipeline.from_pretrained(self.checkpoint)
        return self._pipe

    def batch(self, tasks, spec):
        import torch
        pipe = self._ensure()
        horizon = tasks[0].horizon
        inputs = [torch.tensor(t.y[-self.context:].astype("float32")) for t in tasks]
        q, mean = pipe.predict_quantiles(inputs=inputs, prediction_length=horizon,
                                         quantile_levels=list(QLEVELS))
        q, mean = np.asarray(q), np.asarray(mean)
        return [Forecast(mean[i, :horizon], q[i, :horizon, :])
                for i in range(len(tasks))]


# ------------------------------------------------------------------ registry
def default_registry(fm_context: int = 2048) -> list[BaseModel]:
    return [
        NaiveLast(), NaiveMean(), SeasonalNaive(), SeasonalProfile(),
        StatsForecastModel("AutoETS"), StatsForecastModel("AutoTheta"),
        StatsForecastModel("MSTL"),
        GbdtCalendar(),
        TimesFM25(context=fm_context),
        Chronos2(context=fm_context),
        ChronosBolt(context=fm_context),
    ]


# Model families that cannot share this environment. Recorded here rather than in prose
# so the harness can report them as skipped-with-reason instead of pretending the field
# is complete. Measured with `pip install --dry-run` on 2026-07-29.
UNAVAILABLE = {
    "toto_2p0": ("Datadog Toto, checkpoint Toto-Open-Base-1.0 (the newer Toto-2.0 "
                 "family tops GIFT-Eval; the key toto_2p0 is a slip from the toto-ts "
                 "0.2.0 package version). `toto-ts==0.2.0` downgrades torch 2.11->2.7, "
                 "numpy 2.2.6->1.26.4, pandas 2.3.3->2.2.3 and hard-pins "
                 "jupyter==1.1.1. Needs its own venv."),
    "moirai_2": ("Salesforce Moirai-2. `uni2ts==2.0.0` downgrades torch->2.4.1, "
                 "numpy->1.26.4, pandas->2.1.4 and pulls jax+lightning. Needs its own "
                 "venv."),
}
