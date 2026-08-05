#!/usr/bin/env python3
"""Core-family runner: TimesFM 2.5, Chronos-2, Chronos-Bolt, classical, ML, naive.

    .venv-core/bin/python runners/run_core.py

Writes forecasts/<dataset>/<model>.csv for every core model, and — uniquely among the
runners — forecasts/_truth/<dataset>.json.

WHY THE TRUTH FILE IS WRITTEN HERE AND ONLY HERE
MASE needs a per-origin denominator: the in-sample MAE of the seasonal naive on that
origin's own context. It depends on the series and its seasonal period, which score.py
deliberately never loads. Computing it once and storing it per origin guarantees that every
model — in every environment, possibly run days apart on different machines — is divided by
exactly the same number. If each runner derived its own, the families would quietly not be
comparable and nothing would flag it.

CONSISTENCY WITH THE PUBLISHED TEN-DATASET RUN
Origins, contexts, horizons and the MASE denominator are imported from benchmark.py rather
than reimplemented, so a number produced here means the same thing as the same number in
results/bakeoff_full2*. Reimplementing them is how two studies of the same data end up
disagreeing for reasons nobody can find later.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))

import datasets as ds                                          # noqa: E402
import models as M                                             # noqa: E402
import prepare as P                                            # noqa: E402
from benchmark import HORIZONS, MIN_CONTEXT, build_origins, mase_denominator  # noqa: E402
from forecast_contract import write_forecasts                  # noqa: E402

FORECASTS = CODE / "forecasts"
DEFAULT_DATASETS = ["bangkok_pm25_1h", "boom_telemetry_5t"]


def plan(key: str, origins: int, context: int, series_id=None, horizon=None):
    """The exact (series, origins, context) the other runners must reproduce.

    Returned so run_toto.py / run_moirai.py can call it too — they need identical origins
    or score.py refuses to compare them.
    """
    spec = ds.SPECS[key]
    horizon = horizon or HORIZONS[key]
    ctx = max(context, MIN_CONTEXT.get(key, 0))
    prep = P.prepare(key, series_id=series_id)
    if prep.blockers:
        raise SystemExit(f"{key}: unresolved blockers {prep.blockers}")
    y = prep.values
    idx = pd.DatetimeIndex(prep.frame["timestamp"])
    use_ctx = min(ctx, max(MIN_CONTEXT.get(key, 0), (len(y) - horizon) // 2))
    os_ = build_origins(len(y), horizon, use_ctx, origins)
    if not os_:
        raise SystemExit(f"{key}: too short for {origins} origins ({len(y)} obs)")
    return prep, spec, y, idx, horizon, use_ctx, os_


def rows_from(fcs, os_, horizon):
    for o, fc in zip(os_, fcs):
        q = fc.quantiles
        for s in range(horizon):
            r = {"origin": o, "step": s, "point": float(fc.point[s])}
            if q is not None:
                for i, d in enumerate(range(10, 100, 10)):
                    r[f"q{d}"] = float(q[s][i])
            yield r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--origins", type=int, default=16)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--series-cap", type=int, default=1,
                    help="panel datasets (BOOM) hold many series; how many to run")
    # Horizon as a first-class axis. Given in DAYS and converted per dataset frequency, so
    # "7d" means the same business decision on 30-minute demand and hourly air quality.
    # Every model within a run faces the identical horizon; the scorer enforces it.
    ap.add_argument("--horizon-days", type=float, default=None,
                    help="forecast span in days; omitted = the per-dataset default in benchmark.py")
    a = ap.parse_args()

    # Panel datasets carry many series under one key. Each is forecast independently and
    # stored under "<key>@<series_id>" so score.py needs no concept of panels — it just
    # sees more datasets, each with a matching truth file.
    jobs = []
    for key in a.datasets:
        sids = P.series_ids(key)
        jobs += [(key, sid) for sid in (sids[:a.series_cap] if sids != [None] else [None])]

    # minutes per step, used to turn --horizon-days into steps
    STEP_MIN = {"uk_demand_30min": 30, "uk_demand_daily": 1440, "bangkok_temp_1h": 60,
                "london_temp_1h": 60, "bangkok_pm25_1h": 60, "btc_usd_1h": 60,
                "btc_returns_1h": 60, "white_noise_synth": 60, "boom_telemetry_5t": 5,
                "quake_magnitude_seq": 60}

    for key, sid in jobs:
        horizon_override = None
        if a.horizon_days is not None:
            horizon_override = int(round(a.horizon_days * 1440 / STEP_MIN[key]))
        # BOOM series ids look like "ds-139-5T/v003" — the slash would be read as a
        # directory separator and the truth file write fails. Sanitise for filesystem use;
        # the unsanitised id stays in the metadata.
        label = key if sid is None else f"{key}@{str(sid).replace('/', '_')}"
        if horizon_override:
            label = f"{label}#h{horizon_override}"
        prep, spec, y, idx, horizon, use_ctx, os_ = plan(
            key, a.origins, a.context, series_id=sid, horizon=horizon_override)
        print(f"\n{label}: n={len(y)} horizon={horizon} season={spec.season} "
              f"context={use_ctx} origins={len(os_)}")

        truth = {
            "actuals": {str(o): y[o:o + horizon].tolist() for o in os_},
            # The exact context window each origin used. Satellite runners read this
            # instead of reloading the series themselves — reconstructing a panel member
            # from the cache is how run_moirai.py silently forecast all 24 BOOM series
            # concatenated together and still looked plausible.
            "contexts": {str(o): y[max(0, o - use_ctx):o].tolist() for o in os_},
            "scale": {str(o): mase_denominator(y[max(0, o - use_ctx):o], spec.season)
                      for o in os_},
            # MAPE validity is a property of the series, decided once by prepare.py's
            # audit — not something each runner should re-judge.
            "ratio_scale": bool(prep.mape_valid),
            "horizon": horizon, "season": spec.season, "context": use_ctx, "origins": os_,
        }
        tdir = FORECASTS / "_truth"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / f"{label}.json").write_text(json.dumps(truth, indent=1) + "\n")
        print(f"  truth written ({len(os_)} origins, mape_valid={truth['ratio_scale']})")

        tasks = [M.Task(y=y[:o], ctx_index=idx[:o], fut_index=idx[o:o + horizon],
                        horizon=horizon) for o in os_]

        for mdl in M.default_registry(fm_context=use_ctx):
            try:
                prep.assert_ready_for(mdl.family)
            except ValueError as e:
                print(f"  {mdl.name:26} SKIP  {e}")
                continue
            t0 = time.perf_counter()
            try:
                fcs = mdl.batch(tasks, spec)
            except Exception as e:      # one model must not take the whole run down
                print(f"  {mdl.name:26} FAIL  {type(e).__name__}: {e}")
                continue
            per = (time.perf_counter() - t0) / len(tasks)
            write_forecasts(FORECASTS, label, mdl.name, rows_from(fcs, os_, horizon), {
                "context": use_ctx, "horizon": horizon, "n_origins": len(os_),
                "seconds_per_forecast": round(per, 6), "env": "core",
                "series_id": sid, "horizon_days": a.horizon_days,
                # Chronos-Bolt's architecture fixes prediction_length at 64. Past that the
                # pipeline rolls autoregressively, so it is no longer doing the single
                # forward pass its speed advantage comes from. Flag it, or a long-horizon
                # Bolt row reads as native performance.
                "autoregressive_rollouts": (
                    -(-horizon // 64) if mdl.name.startswith("chronos_bolt") and horizon > 64 else None),
            })
            print(f"  {mdl.name:26} ok  {per*1000:8.1f} ms/forecast")

    print(f"\nNext:  .venv-toto/bin/python runners/run_toto.py --datasets {' '.join(a.datasets)}")
    print(f"       python3 score.py --datasets {' '.join(a.datasets)}")


if __name__ == "__main__":
    main()
