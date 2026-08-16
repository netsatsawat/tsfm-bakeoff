#!/usr/bin/env python3
"""Does XReg earn its keep, and does a holiday flag fix the calendar blind spot?

Two claims in the article are asserted rather than measured:

  1. "TimesFM has no calendar, so it loses on bank holidays."  -> measured, but only as
     plain TimesFM. If handing it a holiday flag fixes that, the fix is cheap and the
     claim needs qualifying.
  2. "Covariates are how your domain knowledge goes back in."  -> plausible, and my
     one-origin spot-check suggested covariates made accuracy WORSE. One origin proves
     nothing either way, so this settles it properly.

Four variants of the same model on the same origins:

    plain           forecast(), no covariates
    xreg_energy     + embedded solar and wind (physical drivers of net grid demand)
    xreg_calendar   + day-of-week and bank-holiday flags
    xreg_both       all four

Rolling daily origins over the 2026 window, horizon 48 (one day), broken out by day type
so the four bank holidays are visible instead of averaged away.

    python3 covariate_test.py --origins 20        # quick
    python3 covariate_test.py                     # all daily origins in the window

WHY IT NEEDS A SEPARATE SCRIPT: benchmark.py is deliberately dataset-agnostic and its
Task carries only a target series. Threading dataset-specific covariates through it would
put domain knowledge in the harness. This experiment is one dataset.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import prepare as P

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
KEY = "uk_demand_30min"
HORIZON = 48
CONTEXT = 512
SEASON = 48
RIDGE = 1.0

# UK bank holidays, England & Wales, inside the test window.
# Good Friday 3 Apr, Easter Monday 6 Apr, Early May 4 May, Spring 25 May.
HOLIDAYS = {"2026-04-03", "2026-04-06", "2026-05-04", "2026-05-25",
            "2026-01-01", "2025-12-25", "2025-12-26", "2025-01-01",
            "2025-04-18", "2025-04-21", "2025-05-05", "2025-05-26",
            "2025-08-25", "2025-12-29"}


def load_aligned() -> tuple[np.ndarray, pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Prepared target plus covariates aligned onto the same grid.

    The target goes through prepare.py so this shares the benchmark's cleaning (regular
    grid, dedup, short-gap interpolation). Covariates are merged onto those timestamps and
    any point created by interpolation gets its covariate filled the same way, so the two
    can never drift apart by a row.
    """
    prep = P.prepare(KEY)
    frame = prep.frame.copy()
    raw, _ = __import__("datasets").load(KEY)
    cols = ["embedded_solar_generation", "embedded_wind_generation"]
    cols = [c for c in cols if c in raw.columns]
    merged = frame.merge(raw[["timestamp"] + cols], on="timestamp", how="left")
    for c in cols:
        merged[c] = merged[c].interpolate(limit=6, limit_direction="both")
    idx = pd.DatetimeIndex(merged["timestamp"])
    covs = {c: merged[c].to_numpy(float) for c in cols}
    covs["dow"] = idx.dayofweek.to_numpy()
    covs["is_holiday"] = np.isin(idx.strftime("%Y-%m-%d"),
                                 list(HOLIDAYS)).astype(int)
    print(f"aligned: {len(merged):,} rows, covariates {list(covs)}")
    print(f"  non-finite after fill: "
          f"{ {c: int((~np.isfinite(v)).sum()) for c, v in covs.items()} }")
    return merged["value"].to_numpy(float), idx, covs


def build_model():
    import timesfm
    torch.set_float32_matmul_precision("high")
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(
        max_context=CONTEXT, max_horizon=HORIZON, normalize_inputs=True,
        use_continuous_quantile_head=True, force_flip_invariance=True,
        infer_is_positive=True, fix_quantile_crossing=True,
        # Mandatory for XReg. Without it forecast_with_covariates() raises.
        return_backcast=True))
    return m


VARIANTS = {
    "plain": ([], []),
    "xreg_energy": (["embedded_solar_generation", "embedded_wind_generation"], []),
    "xreg_calendar": ([], ["dow", "is_holiday"]),
    "xreg_both": (["embedded_solar_generation", "embedded_wind_generation"],
                  ["dow", "is_holiday"]),
}


def run_variant(model, name, y, covs, origins, batch=16):
    num_keys, cat_keys = VARIANTS[name]
    preds: list[np.ndarray] = []
    t0 = time.perf_counter()
    for b in range(0, len(origins), batch):
        chunk = origins[b:b + batch]
        inputs = [y[o - CONTEXT:o].tolist() for o in chunk]
        if not num_keys and not cat_keys:
            p, _ = model.forecast(horizon=HORIZON,
                                  inputs=[np.asarray(v) for v in inputs])
            preds.extend(np.asarray(p)[i][-HORIZON:] for i in range(len(chunk)))
            continue
        # NOTE: the horizon slice [o:o+HORIZON] feeds the model the ACTUAL realized
        # covariate values over the forecast window. For solar and wind that is
        # perfect weather foresight, so the measured gains are a CEILING on what a
        # real deployment (which must forecast its covariates) can collect.
        dyn_num = {k: [covs[k][o - CONTEXT:o + HORIZON].tolist() for o in chunk]
                   for k in num_keys}
        dyn_cat = {k: [covs[k][o - CONTEXT:o + HORIZON].astype(int).tolist()
                       for o in chunk] for k in cat_keys}
        out = model.forecast_with_covariates(
            inputs=inputs,
            dynamic_numerical_covariates=dyn_num or None,
            dynamic_categorical_covariates=dyn_cat or None,
            xreg_mode="xreg + timesfm",
            ridge=RIDGE,
            normalize_xreg_target_per_input=True,
        )
        seq = out[0] if isinstance(out, tuple) else out
        for arr in seq:
            preds.append(np.asarray(arr).reshape(-1)[-HORIZON:])
    return preds, (time.perf_counter() - t0) / len(origins)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=int, default=0, help="0 = every daily origin")
    ap.add_argument("--start", default="2026-03-01")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    y, idx, covs = load_aligned()

    # daily origins at midnight, needing full context and a full horizon
    days = pd.date_range(args.start, idx.max().normalize(), freq="D")
    origins = []
    for d in days:
        pos = idx.searchsorted(d)
        if pos < CONTEXT or pos + HORIZON > len(y) or idx[pos] != d:
            continue
        origins.append(int(pos))
    if args.origins:
        origins = origins[:args.origins]
    print(f"origins: {len(origins)} daily, horizon {HORIZON}, context {CONTEXT}\n")

    model = build_model()
    targets = [y[o:o + HORIZON] for o in origins]
    denoms = [float(np.mean(np.abs(y[max(0, o - CONTEXT):o][SEASON:]
                                   - y[max(0, o - CONTEXT):o][:-SEASON])))
              for o in origins]

    rows = []
    for name in VARIANTS:
        preds, sec = run_variant(model, name, y, covs, origins)
        print(f"  {name:14} {sec * 1000:6.0f} ms/forecast")
        for i, o in enumerate(origins):
            f, act = preds[i], targets[i]
            if f.shape != act.shape or not np.isfinite(f).all():
                continue
            date = idx[o].strftime("%Y-%m-%d")
            rows.append({
                "variant": name, "origin": o, "date": date,
                "daytype": ("holiday" if date in HOLIDAYS
                            else "weekend" if idx[o].dayofweek >= 5 else "weekday"),
                "mase": float(np.mean(np.abs(act - f)) / denoms[i]),
                "mape": float(np.mean(np.abs((act - f) / act)) * 100),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "covariate_test_per_origin.csv", index=False)

    print("\n=== overall (mean over all origins) ===")
    overall = df.groupby("variant")[["mase", "mape"]].mean().round(4)
    overall["n"] = df.groupby("variant").size()
    print(overall.to_string())

    print("\n=== by day type: MASE (n in brackets) ===")
    piv = df.pivot_table(index="variant", columns="daytype", values="mase",
                         aggfunc="mean").round(3)
    counts = df[df.variant == "plain"].groupby("daytype").size().to_dict()
    piv.columns = [f"{c} (n={counts.get(c, 0)})" for c in piv.columns]
    print(piv.to_string())

    print("\n=== does the calendar flag fix the holiday gap? ===")
    for dt in ("weekday", "weekend", "holiday"):
        sub = df[df.daytype == dt]
        if sub.empty:
            continue
        base = sub[sub.variant == "plain"]["mase"].mean()
        for v in ("xreg_calendar", "xreg_both", "xreg_energy"):
            got = sub[sub.variant == v]["mase"].mean()
            delta = (got / base - 1) * 100
            verdict = "better" if delta < 0 else "worse"
            print(f"  {dt:8} {v:14} {got:.3f} vs plain {base:.3f}  "
                  f"({delta:+.1f}% {verdict})")

    print("\n=== win rate vs plain, per origin ===")
    wide = df.pivot_table(index="origin", columns="variant", values="mase")
    for v in ("xreg_energy", "xreg_calendar", "xreg_both"):
        if v in wide:
            print(f"  {v:14} beats plain on "
                  f"{(wide[v] < wide['plain']).mean() * 100:.1f}% of origins")

    (OUT / "covariate_test.json").write_text(json.dumps({
        "generated": pd.Timestamp.utcnow().isoformat(),
        "n_origins": len(origins), "horizon": HORIZON, "context": CONTEXT,
        "ridge": RIDGE, "xreg_mode": "xreg + timesfm",
        "overall": json.loads(overall.to_json(orient="index")),
        "by_daytype_mase": json.loads(piv.to_json()),
        "daytype_counts": counts,
    }, indent=2))
    print(f"\nwrote {OUT}/covariate_test*.csv/.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
