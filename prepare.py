#!/usr/bin/env python3
"""The pre-call contract: what must be true of your data BEFORE a TSFM sees it.

A time-series foundation model takes a bare array of floats. No timestamps, no schema, no
complaint. That is exactly why it is easy to hand it something quietly wrong and get a
plausible-looking forecast back. Everything below is a check I have had to add after being
burned by its absence.

THE ELEVEN CHECKS
  1  regular grid      Observations must be evenly spaced. The model indexes by POSITION,
                       so a missing hour silently shifts every later point earlier in
                       "model time" and corrupts the learned periodicity.
  2  duplicates        Two rows for one timestamp double-count a period.
  3  DST / clock change A 23- or 25-hour day is a real irregular grid, not a data error.
                       Half-hourly UK settlement data has 46- and 50-period days.
  4  gaps              Short gaps interpolate defensibly. Long ones do not: filling a
                       three-day outage with a straight line teaches the model a trend
                       that never happened. Refuse instead.
  5  NaN policy        Per-model. TimesFM wants a clean float array; Chronos tolerates
                       NaN natively. Passing NaN to a model that cannot take it yields
                       NaN forecasts, or worse, silently zero-filled ones.
  6  positivity        If the quantity cannot go below zero, say so. TimesFM has
                       infer_is_positive for exactly this; without it, quantile bands
                       cross into negative demand.
  7  intermittency     Fraction of exact zeros. Above a few percent, point forecasts and
                       MAPE both stop meaning anything, and you want a different model
                       class entirely.
  8  scale ownership   TimesFM (normalize_inputs) and Chronos both scale internally.
                       Pre-scaling those is double work that can hurt. A GBDT or a linear
                       model owns nothing and needs you to do it. Get this backwards and
                       you will blame the model.
  9  context adequacy  The context window must cover at least two, preferably three, full
                       seasonal cycles, and must not exceed the compile-time max_context.
 10  metric validity   MAPE requires a ratio scale with a meaningful zero. Temperature in
                       Celsius is an interval scale: 0 degC is a convention, so MAPE there
                       is arithmetic nonsense. This check is why the benchmark reports
                       MASE everywhere and MAPE only where it is defined.
 11  level shifts      Report step changes and extreme outliers. Do NOT auto-fix: whether
                       a shift is a meter change or a real regime change is a judgement
                       the data cannot make for you.

Run it standalone to audit every dataset:
    python3 prepare.py
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

import datasets as ds

# Model families and what they expect from you. Keep this table honest; it is the reason
# the harness can share one prepared series across very different models.
FAMILY_REQUIREMENTS = {
    "timesfm":  {"nan_ok": False, "scales_itself": True,  "wants_positivity_flag": True},
    "chronos":  {"nan_ok": True,  "scales_itself": True,  "wants_positivity_flag": False},
    "statistical": {"nan_ok": False, "scales_itself": True, "wants_positivity_flag": False},
    "ml":       {"nan_ok": False, "scales_itself": False, "wants_positivity_flag": False},
    "naive":    {"nan_ok": False, "scales_itself": True,  "wants_positivity_flag": False},
}

MAX_INTERPOLATE_STEPS = 3     # beyond this, a gap is a hole in the record, not a blip
INTERMITTENCY_LIMIT = 0.02    # >2% exact zeros: point metrics stop being meaningful


def structure_features(v: np.ndarray, spec: ds.Spec) -> dict[str, Any]:
    """Quantify how much structure a series actually has.

    A forecasting bake-off is only interpretable if you know what you handed each model.
    These are the standard measures (Hyndman/Wang feature set plus stationarity tests),
    and they let the benchmark answer the question that matters: does a foundation model's
    advantage track the amount of structure present, or does it appear out of nowhere?

      trend_strength     1 - Var(remainder) / Var(trend + remainder), from an STL fit.
                         0 = no trend at all, ->1 = trend dominates.
      seasonal_strength  1 - Var(remainder) / Var(seasonal + remainder). Same scale.
                         Below ~0.3 there is effectively nothing periodic to exploit.
      spectral_entropy   Shannon entropy of the normalized periodogram, in [0, 1].
                         ->1 means the power is spread evenly across all frequencies,
                         i.e. white noise, i.e. inherently unforecastable.
      acf1               Lag-1 autocorrelation. ~0 means no short memory either.
      acf_season         Autocorrelation at the seasonal lag.
      adf_p / kpss_p     Stationarity. ADF p<0.05 rejects a unit root (stationary);
                         KPSS p<0.05 rejects stationarity. A random walk shows
                         ADF p high AND KPSS p low; white noise the reverse.
    """
    out: dict[str, Any] = {}
    n = len(v)
    try:
        from statsmodels.tsa.seasonal import STL
        period = spec.season
        if n >= 2 * period + 1 and period >= 2:
            stl = STL(v, period=period, robust=True).fit()
            rem_var = float(np.var(stl.resid))
            tr = float(np.var(stl.trend + stl.resid))
            se = float(np.var(stl.seasonal + stl.resid))
            out["trend_strength"] = round(max(0.0, 1 - rem_var / tr), 3) if tr > 0 else 0.0
            out["seasonal_strength"] = (round(max(0.0, 1 - rem_var / se), 3)
                                        if se > 0 else 0.0)
    except Exception as e:  # noqa: BLE001
        out["stl_error"] = f"{type(e).__name__}"

    # spectral entropy of the periodogram, normalized to [0, 1]
    x = v - v.mean()
    if x.std() > 0:
        power = np.abs(np.fft.rfft(x)) ** 2
        power = power[1:]                     # drop DC
        p = power / power.sum()
        p = p[p > 0]
        out["spectral_entropy"] = round(
            float(-(p * np.log(p)).sum() / np.log(len(p))), 3)

    def acf(k: int) -> float:
        if n <= k + 2 or x.std() == 0:
            return float("nan")
        a, b = x[:-k], x[k:]
        denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        return float((a * b).sum() / denom) if denom else float("nan")

    out["acf1"] = round(acf(1), 3)
    out["acf_season"] = round(acf(spec.season), 3)
    if spec.season_long and n > spec.season_long + 2:
        out["acf_season_long"] = round(acf(spec.season_long), 3)
        # "Overlapping seasonality" is a measurable claim, not a vibe: both periods must
        # carry real autocorrelation at the same time.
        out["overlapping_seasonality"] = bool(
            out["acf_season"] > 0.3 and out["acf_season_long"] > 0.3)

    # Surge / anomaly character. Deliberately IQR-based and expressed as a COUNT plus a
    # floored ratio: an unfloored MAD denominator returns values up to 1.4e10 on
    # near-constant telemetry variates, which is a division artifact, not a surge.
    dev = np.abs(v - np.median(v))
    q1, q3 = np.percentile(v, [25, 75])
    iqr = float(q3 - q1)
    span = float(np.percentile(v, 99.9) - np.percentile(v, 0.1))
    scale = max(iqr, 0.01 * span) if span > 0 else 0.0
    if scale > 0:
        out["surge_count_6iqr"] = int((dev > 6 * iqr).sum()) if iqr > 0 else 0
        out["surge_frac_6iqr"] = round(float((dev > 6 * iqr).mean()), 5) if iqr > 0 else 0.0
        out["surge_max_over_iqr"] = round(float(dev.max() / scale), 1)
        out["tail_ratio_p999_p50"] = round(
            float(np.percentile(dev, 99.9) / scale), 1)
    out["iqr"] = round(iqr, 6)

    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        sample = v[-4000:]
        out["adf_p"] = round(float(adfuller(sample, autolag="AIC")[1]), 4)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["kpss_p"] = round(float(kpss(sample, nlags="auto")[1]), 4)
    except Exception as e:  # noqa: BLE001
        out["stationarity_error"] = f"{type(e).__name__}"

    # A single label so the results table can be read at a glance.
    ss = out.get("seasonal_strength", 0.0)
    ts = out.get("trend_strength", 0.0)
    ent = out.get("spectral_entropy", 1.0)
    if ss < 0.25 and ts < 0.35 and ent > 0.85:
        out["structure_label"] = "none (noise-like)"
    elif ss >= 0.6:
        out["structure_label"] = "strong seasonal"
    elif ss >= 0.25:
        out["structure_label"] = "moderate seasonal"
    elif ts >= 0.35:
        out["structure_label"] = "trend/drift only"
    else:
        out["structure_label"] = "weak"
    return out


@dataclass
class Prepared:
    key: str
    spec: ds.Spec
    frame: pd.DataFrame                  # timestamp, value on a regular grid
    audit: dict[str, Any] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)

    @property
    def values(self) -> np.ndarray:
        return self.frame["value"].to_numpy(dtype=np.float64)

    @property
    def mape_valid(self) -> bool:
        """Check 10: ratio scale, meaningful zero, and no near-zero denominators."""
        return bool(self.audit["ratio_scale"] and self.audit["near_zero_frac"] == 0.0)

    def requirements(self, family: str) -> dict:
        return FAMILY_REQUIREMENTS[family]

    def assert_ready_for(self, family: str) -> None:
        req = self.requirements(family)
        if self.blockers:
            raise ValueError(f"{self.key}: unresolved blockers {self.blockers}")
        if not req["nan_ok"] and not np.isfinite(self.values).all():
            raise ValueError(f"{self.key}: non-finite values but family {family!r} "
                             f"cannot accept NaN")


def series_ids(key: str) -> list[str | None]:
    """Every forecastable series in a dataset.

    Most datasets here are a single series and return [None]. A panel dataset (BOOM)
    returns one id per group/variate pair, because a high-dimensional multivariate
    dataset is many forecasting tasks that happen to share an entity.
    """
    raw, _ = ds.load(key)
    if "group" not in raw.columns:
        return [None]
    pairs = raw[["group", "variate"]].drop_duplicates()
    return [f"{g}/{v}" for g, v in pairs.itertuples(index=False)]


def prepare(key: str, refresh: bool = False, series_id: str | None = None) -> Prepared:
    raw, spec = ds.load(key, refresh=refresh)
    audit: dict[str, Any] = {"rows_in": len(raw)}
    blockers: list[str] = []

    if series_id is not None and "group" in raw.columns:
        group, variate = series_id.split("/", 1)
        raw = raw[(raw["group"] == group) & (raw["variate"] == variate)]
        audit["series_id"] = series_id
    elif "group" in raw.columns:
        raise ValueError(f"{key} is a panel; pass series_id (see series_ids('{key}'))")

    df = raw[["timestamp", "value"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 2. duplicates
    dupes = int(df.duplicated("timestamp").sum())
    audit["duplicate_timestamps"] = dupes
    if dupes:
        df = df.groupby("timestamp", as_index=False)["value"].mean()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1 + 3. regular grid, and irregular steps that are really clock changes
    steps = df["timestamp"].diff().dropna()
    expected = pd.tseries.frequencies.to_offset(spec.freq)
    modal = steps.mode().iloc[0] if len(steps) else pd.Timedelta(0)
    audit["modal_step"] = str(modal)
    audit["irregular_steps"] = int((steps != modal).sum())
    audit["step_histogram"] = {str(k): int(v) for k, v in
                              steps.value_counts().head(4).items()}

    full = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq=expected)
    reindexed = (df.set_index("timestamp")["value"].reindex(full))
    audit["grid_len"] = len(full)
    audit["missing_on_grid"] = int(reindexed.isna().sum())

    # 4. gaps: interpolate short runs, refuse long ones
    isna = reindexed.isna().to_numpy()
    runs: list[int] = []
    run = 0
    for flag in isna:
        if flag:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    audit["gap_runs"] = len(runs)
    audit["longest_gap_steps"] = max(runs) if runs else 0
    if runs and max(runs) > MAX_INTERPOLATE_STEPS:
        blockers.append(
            f"gap of {max(runs)} steps exceeds the {MAX_INTERPOLATE_STEPS}-step "
            f"interpolation limit; fill it deliberately or shorten the window")
    series = reindexed.interpolate(limit=MAX_INTERPOLATE_STEPS, limit_area="inside")
    audit["interpolated"] = int(reindexed.isna().sum() - series.isna().sum())
    series = series.dropna()

    v = series.to_numpy(dtype=np.float64)

    # 6 + 7. positivity and intermittency
    audit["min"], audit["max"] = float(v.min()), float(v.max())
    audit["declared_positive"] = spec.positive
    audit["observed_nonpositive"] = int((v <= 0).sum())
    audit["zero_frac"] = float((v == 0).mean())
    audit["intermittent"] = bool(audit["zero_frac"] > INTERMITTENCY_LIMIT)
    scale = float(np.median(np.abs(v))) or 1.0
    audit["near_zero_frac"] = float((np.abs(v) < 0.01 * scale).mean())

    # 10. metric validity
    audit["ratio_scale"] = bool(spec.positive)

    # 9. context adequacy (reported; the harness enforces against its own context)
    audit["season"] = spec.season
    audit["cycles_available"] = round(len(v) / spec.season, 1)
    if spec.season_long:
        audit["long_cycles_available"] = round(len(v) / spec.season_long, 1)

    # 11. level shifts and outliers, reported not repaired
    d = np.diff(v)
    mad = float(np.median(np.abs(d - np.median(d)))) or 1e-9
    robust_sigma = 1.4826 * mad
    audit["step_outliers_8sigma"] = int((np.abs(d) > 8 * robust_sigma).sum())
    roll = pd.Series(v).rolling(spec.season, min_periods=spec.season).mean()
    audit["level_shift_ratio"] = (round(float(roll.max() / roll.min()), 2)
                                  if roll.notna().any() and roll.min() > 0 else None)

    audit["rows_out"] = int(len(v))
    audit.update(structure_features(v, spec))
    frame = pd.DataFrame({"timestamp": series.index.to_numpy(), "value": v})
    return Prepared(key, spec, frame, audit, blockers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*", default=list(ds.SPECS))
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    print(f"{'series':18}{'n':>7}{'irreg':>7}{'miss':>6}{'interp':>7}{'maxgap':>7}"
          f"{'zero%':>7}{'cycles':>8}{'MAPE?':>7}{'8sig':>6}  blockers")
    rows = []
    for key in args.keys:
        p = prepare(key, refresh=args.refresh)
        a = p.audit
        print(f"{key:18}{a['rows_out']:>7,}{a['irregular_steps']:>7}"
              f"{a['missing_on_grid']:>6}{a['interpolated']:>7}"
              f"{a['longest_gap_steps']:>7}{a['zero_frac'] * 100:>6.1f}%"
              f"{a['cycles_available']:>8}{'yes' if p.mape_valid else 'NO':>7}"
              f"{a['step_outliers_8sigma']:>6}  "
              f"{'; '.join(p.blockers) if p.blockers else '-'}")
        rows.append((key, a))

    print("\nwhy some series cannot use MAPE (check 10):")
    for key, a in rows:
        if not a["ratio_scale"]:
            print(f"  {key}: interval scale (zero is a convention, not an absence) "
                  f"-> MASE only")
        elif a["near_zero_frac"] > 0:
            print(f"  {key}: {a['near_zero_frac'] * 100:.1f}% of values are within 1% "
                  f"of the median scale -> MAPE unstable")

    print("\nirregular steps found (check 1 and 3):")
    for key, a in rows:
        if a["irregular_steps"]:
            print(f"  {key}: {a['irregular_steps']} irregular, modal step "
                  f"{a['modal_step']}, histogram {a['step_histogram']}")

    print("\nwho owns scaling (check 8):")
    for fam, req in FAMILY_REQUIREMENTS.items():
        owner = "the model" if req["scales_itself"] else "YOU, before the call"
        print(f"  {fam:12} scaling: {owner:24} NaN tolerated: {req['nan_ok']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
