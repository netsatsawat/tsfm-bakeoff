#!/usr/bin/env python3
"""Rolling-origin bake-off: every model, every series, identical treatment.

    python3 benchmark.py                          # everything
    python3 benchmark.py --datasets uk_demand_30min --origins 8   # smoke run
    python3 benchmark.py --multivariate           # the Chronos-2 joint-vs-single test

METRIC CHOICE, STATED ONCE
  MASE is primary and is the only metric reported for every series. It scales each
  origin's error by the in-sample MAE of the seasonal naive on that origin's own context,
  so a 40 GW demand series and a zero-mean return series are comparable, and 1.0 always
  means "no better than the naive it is scaled against".

  MAPE appears only where it is arithmetically valid: a ratio scale with a meaningful
  zero and no near-zero denominators. That excludes five of the ten series (both
  temperatures, BTC returns, white noise and BOOM), which is exactly why leading with
  MAPE would have quietly changed which datasets counted.

  WQL covers the nine deciles for every model that emits them, including the classical
  ones via prediction intervals, so the probabilistic comparison is not silently
  restricted to the neural models.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import datasets as ds
import models as M
import prepare as P

OUT = Path(__file__).resolve().parent / "results"
# Horizon = one seasonal cycle, except where that is impractically long. Fixed per
# dataset so every model faces the identical task.
HORIZONS = {
    "uk_demand_30min": 48, "uk_demand_daily": 7, "bangkok_temp_1h": 24,
    "london_temp_1h": 24, "bangkok_pm25_1h": 24, "btc_usd_1h": 24,
    "btc_returns_1h": 24, "quake_magnitude_seq": 24, "white_noise_synth": 24,
    "boom_telemetry_5t": 48,          # 4 hours at 5-minute resolution
}
MIN_CONTEXT = {"uk_demand_daily": 120}


def mase_denominator(context: np.ndarray, season: int) -> float:
    if len(context) <= season:
        return float(np.mean(np.abs(np.diff(context)))) or 1.0
    return float(np.mean(np.abs(context[season:] - context[:-season]))) or 1.0


def wql(y: np.ndarray, q: np.ndarray) -> float:
    total = 0.0
    for k, lv in enumerate(M.QLEVELS):
        f = q[:, k]
        total += 2.0 * np.sum(np.where(y >= f, lv * (y - f), (1 - lv) * (f - y)))
    denom = np.sum(np.abs(y))
    return float(total / (len(M.QLEVELS) * denom)) if denom > 0 else float("nan")


def build_origins(n: int, horizon: int, context: int, count: int) -> list[int]:
    """Evenly spaced origins across the usable tail, oldest first."""
    first, last = context, n - horizon
    if last <= first:
        return []
    if count >= last - first:
        return list(range(first, last))
    return [int(round(x)) for x in np.linspace(first, last - 1, count)]


def run_dataset(key: str, registry, origins: int, context: int,
                series_cap: int) -> list[dict]:
    spec = ds.SPECS[key]
    horizon = HORIZONS[key]
    ctx = max(context, MIN_CONTEXT.get(key, 0))
    sids = P.series_ids(key)[:series_cap]
    rows: list[dict] = []

    for sid in sids:
        prep = P.prepare(key, series_id=sid)
        if prep.blockers:
            print(f"  SKIP {key}/{sid}: {prep.blockers}")
            continue
        y = prep.values
        idx = pd.DatetimeIndex(prep.frame["timestamp"])
        # Never let the context eat the series. A flat 2048-point context on
        # uk_demand_daily (553 obs) left exactly ONE usable origin, so that dataset
        # silently contributed 10 forecasts instead of 200 and its row in the results
        # table was one lucky day. Leave at least half the usable span for origins.
        use_ctx = min(ctx, max(MIN_CONTEXT.get(key, 0), (len(y) - horizon) // 2))
        os_ = build_origins(len(y), horizon, use_ctx, origins)
        if not os_:
            print(f"  SKIP {key}/{sid}: too short ({len(y)} obs)")
            continue
        if len(os_) < origins:
            print(f"  WARN {key}/{sid}: only {len(os_)} usable origins "
                  f"(requested {origins}); results are thinner here")

        # Every task carries EXACTLY the use_ctx window, so no adapter can quietly see
        # more or less history than another. The committed bake-off artifacts predate
        # this slice: in those runs each neural family applied its own context cap.
        tasks = [M.Task(y=y[max(0, o - use_ctx):o],
                        ctx_index=idx[max(0, o - use_ctx):o],
                        fut_index=idx[o:o + horizon],
                        horizon=horizon) for o in os_]
        targets = [y[o:o + horizon] for o in os_]
        denoms = [mase_denominator(y[max(0, o - use_ctx):o], spec.season) for o in os_]

        for mdl in registry:
            prep.assert_ready_for(mdl.family)
            t0 = time.perf_counter()
            try:
                fcs = mdl.batch(tasks, spec)
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {key}/{sid} {mdl.name}: {type(e).__name__}: "
                      f"{str(e)[:120]}")
                traceback.print_exc(limit=1)
                continue
            elapsed = (time.perf_counter() - t0) / len(tasks)

            for i, fc in enumerate(fcs):
                yt, f = targets[i], np.asarray(fc.point, dtype=float)
                if f.shape != yt.shape or not np.isfinite(f).all():
                    continue
                row = {
                    "dataset": key, "series": sid or "-", "model": mdl.name,
                    "family": mdl.family, "origin": os_[i],
                    "date": str(idx[os_[i]].date()),
                    "mase": float(np.mean(np.abs(yt - f)) / denoms[i]),
                    "seconds": elapsed,
                }
                if prep.mape_valid:
                    row["mape"] = float(np.mean(np.abs((yt - f) / yt)) * 100)
                if fc.quantiles is not None and fc.quantiles.shape == (len(yt), 9):
                    row["wql"] = wql(yt, np.asarray(fc.quantiles, dtype=float))
                    lo, hi = fc.quantiles[:, 0], fc.quantiles[:, 8]
                    row["cov80"] = float(np.mean((yt >= lo) & (yt <= hi)) * 100)
                rows.append(row)
    return rows


def multivariate_test(context: int, origins: int) -> list[dict]:
    """Chronos-2 joint vs one-at-a-time on the same BOOM groups.

    TimesFM 2.5 cannot do this at all, which is the point: if joint modelling pays on
    high-dimensional telemetry, that is a capability gap, not a tuning difference.
    """
    key = "boom_telemetry_5t"
    spec, horizon = ds.SPECS[key], HORIZONS[key]
    raw, _ = ds.load(key)
    mdl = M.Chronos2(context=context)
    rows: list[dict] = []

    for group, g in raw.groupby("group"):
        wide = g.pivot_table(index="timestamp", columns="variate", values="value")
        wide = wide.dropna()
        mat = wide.to_numpy().T                       # (variates, T)
        V, T = mat.shape
        os_ = build_origins(T, horizon, min(context, T - horizon - 1), origins)
        corr = np.corrcoef(mat)
        off = corr[~np.eye(V, dtype=bool)]
        mean_abs_corr = float(np.mean(np.abs(off[np.isfinite(off)])))

        for o in os_:
            ctx = mat[:, max(0, o - context):o]
            tgt = mat[:, o:o + horizon]
            joint = mdl.batch_multivariate(ctx, horizon)
            single = np.vstack([
                mdl.batch([M.Task(y=ctx[v], ctx_index=wide.index[:ctx.shape[1]],
                                  fut_index=wide.index[o:o + horizon],
                                  horizon=horizon)], spec)[0].point
                for v in range(V)])
            for v in range(V):
                den = mase_denominator(ctx[v], spec.season)
                rows.append({
                    "group": group, "variate": wide.columns[v], "origin": int(o),
                    "n_variates": V, "mean_abs_corr": round(mean_abs_corr, 3),
                    "mase_joint": float(np.mean(np.abs(tgt[v] - joint[v])) / den),
                    "mase_single": float(np.mean(np.abs(tgt[v] - single[v])) / den),
                })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(HORIZONS))
    ap.add_argument("--origins", type=int, default=24)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--series-cap", type=int, default=8,
                    help="max series per panel dataset (BOOM)")
    ap.add_argument("--multivariate", action="store_true")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.multivariate:
        rows = multivariate_test(args.context, args.origins)
        mv = pd.DataFrame(rows)
        mv.to_csv(OUT / "multivariate_boom.csv", index=False)
        agg = (mv.groupby(["group", "n_variates", "mean_abs_corr"])
               [["mase_joint", "mase_single"]].mean().round(3))
        agg["joint_better_pct"] = (mv.groupby("group")
                                   .apply(lambda d: (d.mase_joint < d.mase_single).mean()
                                          * 100, include_groups=False).round(1).values)
        print("\n=== Chronos-2: joint multivariate vs one series at a time (BOOM) ===")
        print(agg.to_string())
        (OUT / "multivariate_boom.json").write_text(
            json.dumps(json.loads(agg.reset_index().to_json(orient="records")), indent=2))
        return 0

    registry = M.default_registry(fm_context=args.context)
    print(f"models   : {[m.name for m in registry]}")
    print(f"datasets : {args.datasets}")
    print(f"origins  : {args.origins} per series, context {args.context}\n")
    for name, why in M.UNAVAILABLE.items():
        print(f"  not run: {name} -- {why}")
    print()

    all_rows: list[dict] = []
    for key in args.datasets:
        print(f"--- {key}")
        t0 = time.perf_counter()
        rows = run_dataset(key, registry, args.origins, args.context, args.series_cap)
        all_rows.extend(rows)
        print(f"    {len(rows)} scored forecasts in {time.perf_counter() - t0:.0f}s")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / f"bakeoff_{args.tag}_per_origin.csv", index=False)

    # per dataset x model
    agg = (df.groupby(["dataset", "model"])
           .agg(mase=("mase", "mean"), mase_med=("mase", "median"),
                mape=("mape", "mean"), wql=("wql", "mean"),
                cov80=("cov80", "mean"), sec=("seconds", "mean"), n=("mase", "size"))
           .round(4).reset_index())
    agg.to_csv(OUT / f"bakeoff_{args.tag}_by_dataset.csv", index=False)

    pivot = agg.pivot(index="model", columns="dataset", values="mase")
    # Seasonal strength per dataset. For a PANEL this must be the mean across the series
    # actually benchmarked: taking series[0] labelled BOOM 0.009 when the sampled mean
    # is ~0.44, which would have put a mixture in the wrong place on the structure axis.
    struct = {}
    for key in args.datasets:
        vals = [P.prepare(key, series_id=s).audit.get("seasonal_strength")
                for s in P.series_ids(key)[:args.series_cap]]
        vals = [v for v in vals if v is not None]
        struct[key] = round(float(np.mean(vals)), 3) if vals else None
    order = sorted(args.datasets, key=lambda k: -(struct.get(k) or 0))
    pivot = pivot[[c for c in order if c in pivot.columns]]

    print("\n=== MASE by dataset (columns ordered by seasonal strength, high -> low) ===")
    print("seasonal strength: "
          + "  ".join(f"{k}={struct.get(k)}" for k in pivot.columns))
    print(pivot.round(3).to_string())

    summary = {
        "generated": pd.Timestamp.utcnow().isoformat(),
        "platform": f"{platform.platform()} python {platform.python_version()}",
        "context": args.context, "origins": args.origins,
        "horizons": {k: HORIZONS[k] for k in args.datasets},
        "seasonal_strength": struct,
        "models": [m.name for m in registry],
        "unavailable": M.UNAVAILABLE,
        "mase_by_dataset": json.loads(pivot.round(4).to_json()),
        "by_dataset_model": json.loads(agg.to_json(orient="records")),
    }
    (OUT / f"bakeoff_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}/bakeoff_{args.tag}*.csv/.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
