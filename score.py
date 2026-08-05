#!/usr/bin/env python3
"""Score every model that has written forecasts, under identical rules.

This file imports NO model and NO framework: standard library only. That is the whole
design: metric definitions live in exactly one place, so a model cannot be advantaged by
being scored inside its own process under its own conventions. It also means scoring needs
no virtualenv and runs anywhere, including on a machine where none of the models install.

Metric rules, enforced here rather than per-family:

  MASE   primary, and the only metric valid on every series. Scaled by the in-sample MAE of
         the seasonal naive on that origin's own context, so 1.0 always means "no better
         than the naive it is scaled against".
  MAPE   computed ONLY where the series is a ratio scale with a meaningful zero. It is
         arithmetically meaningless on temperature in degC and on anything crossing zero.
  WQL    weighted quantile loss over the nine deciles (2x pinball, normalized by
         sum(|actual|)), for every model that emits a distribution.
  cov80  share of actuals inside the 10-90 band. Should be 80. A calibration check, not
         an accuracy score.

Usage:
  python3 score.py                      # every dataset with forecasts on disk
  python3 score.py --datasets bangkok_pm25_1h
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from forecast_contract import read_forecasts, available_models  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
FORECASTS = ROOT / "forecasts"
TRUTH = ROOT / "forecasts" / "_truth"


def _load_truth(dataset):
    """Actuals + per-origin naive scale, written once by the core runner.

    The scale factor is computed there because it depends on the series and its seasonal
    period, which the scorer deliberately does not load. It is stored per origin so every
    model is divided by exactly the same denominator.
    """
    p = TRUTH / f"{dataset}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no truth file for {dataset} ({p}). Run runners/run_core.py first; it writes "
            "the actuals and the per-origin MASE scale that every model is scored against."
        )
    return json.loads(p.read_text())


def _mase(err_abs, scale):
    return (sum(err_abs) / len(err_abs)) / scale if scale > 0 else float("nan")


def _wql(actual, q, levels):
    """Weighted quantile loss, matching benchmark.py's definition exactly:
    2x the pinball loss, averaged over levels, normalized by sum(|actual|).

    The committed results/cross_env_scores.json predates this normalization; its
    wql column is a plain per-point pinball mean, comparable within one dataset
    but not across datasets. Rows written by this version are cross-comparable.
    """
    tot = 0.0
    for a, qs in zip(actual, q):
        for lv, qv in zip(levels, qs):
            d = a - qv
            tot += 2.0 * ((lv * d) if d >= 0 else ((lv - 1) * d))
    denom = sum(abs(a) for a in actual)
    return tot / (len(levels) * denom) if denom > 0 else float("nan")


def score_dataset(dataset, models=None):
    truth = _load_truth(dataset)
    actuals = {int(k): v for k, v in truth["actuals"].items()}
    scales = {int(k): v for k, v in truth["scale"].items()}
    ratio_scale = truth.get("ratio_scale", False)
    levels = [i / 10 for i in range(1, 10)]

    found = models or available_models(FORECASTS, dataset)
    rows, origin_sets = [], {}

    for model in found:
        by, meta = read_forecasts(FORECASTS, dataset, model)
        origin_sets[model] = set(by)
        mases, mapes, wqls, covs = [], [], [], []
        for o, slot in sorted(by.items()):
            if o not in actuals:
                continue
            a, p = actuals[o], slot["point"]
            if len(a) != len(p):
                raise ValueError(f"{model}/{dataset} origin {o}: horizon {len(p)} != truth {len(a)}")
            mases.append(_mase([abs(x - y) for x, y in zip(a, p)], scales[o]))
            if ratio_scale and all(abs(x) > 1e-9 for x in a):
                mapes.append(100 * sum(abs((x - y) / x) for x, y in zip(a, p)) / len(a))
            if slot["q"]:
                wqls.append(_wql(a, slot["q"], levels))
                inside = sum(1 for x, qs in zip(a, slot["q"]) if qs[0] <= x <= qs[8])
                covs.append(100 * inside / len(a))
        rows.append({
            "dataset": dataset, "model": model, "n_origins": len(mases),
            "mase": round(statistics.fmean(mases), 4) if mases else None,
            "mase_med": round(statistics.median(mases), 4) if mases else None,
            "mape": round(statistics.fmean(mapes), 4) if mapes else None,
            "wql": round(statistics.fmean(wqls), 4) if wqls else None,
            "cov80": round(statistics.fmean(covs), 2) if covs else None,
            "sec": meta.get("seconds_per_forecast"),
            "env": meta.get("env"),
        })

    # A model evaluated on different origins from its competitors is not comparable, and
    # the difference is invisible in the aggregate. Fail loudly instead.
    if len(origin_sets) > 1:
        ref_model, ref = next(iter(origin_sets.items()))
        for m, s in origin_sets.items():
            if s != ref:
                raise SystemExit(
                    f"ABORT {dataset}: '{m}' scored on {len(s)} origins but '{ref_model}' on "
                    f"{len(ref)}. Origins must be identical across models or the comparison "
                    f"is meaningless. Re-run the odd one out with the same --origins."
                )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "cross_env_scores.json"))
    a = ap.parse_args()

    datasets = a.datasets or sorted(p.name for p in FORECASTS.iterdir()
                                    if p.is_dir() and not p.name.startswith("_"))
    if not datasets:
        raise SystemExit(f"no forecasts found under {FORECASTS}. Run the runners first.")

    all_rows = []
    for d in datasets:
        rows = score_dataset(d)
        all_rows += rows
        print(f"\n{d}   (ratio-scale MAPE: {'yes' if _load_truth(d).get('ratio_scale') else 'no'})")
        print(f"  {'model':24}{'MASE':>8}{'WQL':>9}{'cov80':>8}{'n':>5}   env")
        for r in sorted(rows, key=lambda x: (x["mase"] is None, x["mase"])):
            print(f"  {r['model']:24}{r['mase']!s:>8}{r['wql']!s:>9}{r['cov80']!s:>8}"
                  f"{r['n_origins']:>5}   {r['env'] or '-'}")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_rows, indent=1) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
