"""The handoff between model environments and the scorer.

WHY THIS EXISTS
Toto and Moirai cannot share a Python environment with TimesFM. `toto-ts==0.2.0` forces
torch 2.11->2.7, numpy 2.2.6->1.26.4 and pandas 2.3.3->2.2.3; `uni2ts==2.0.0` forces
torch->2.4.1, numpy->1.26.4, pandas->2.1.4. The numpy 2.x -> 1.26 move is the fatal one:
NumPy 2.0 changed the C ABI, so extensions compiled against 2.x will not load against 1.26.
One environment cannot satisfy all of them.

So each model family runs in its own virtualenv and writes forecasts to disk. `score.py`
reads those files and computes every metric. The scorer imports no model and no framework,
which means it is also the only place metric definitions live; no family can be scored
under subtly different rules from another.

DELIBERATELY DEPENDENCY-FREE
Only the standard library. This module is imported by every venv, and those venvs
disagree about numpy and pandas versions by construction. Using either one here would
recreate the problem this file exists to solve.

FORMAT
  forecasts/<dataset>/<model>.csv     one row per (origin, step)
      origin  integer index of the forecast origin
      step    0-based position within the horizon
      point   the point forecast
      q10..q90  the nine deciles (blank if the model emits no distribution)
  forecasts/<dataset>/<model>.meta.json
      model, dataset, context, horizon, n_origins, seconds_per_forecast, env, versions

Origins must match across models or the comparison is meaningless; `score.py` asserts it.
"""
from __future__ import annotations

import csv
import json
import os
import pathlib

DECILES = [f"q{d}" for d in range(10, 100, 10)]
FIELDS = ["origin", "step", "point"] + DECILES


def forecast_dir(root: str | os.PathLike, dataset: str) -> pathlib.Path:
    p = pathlib.Path(root) / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_forecasts(root, dataset, model, rows, meta):
    """rows: iterable of dicts with keys FIELDS (deciles optional -> written blank).

    Writing is all-or-nothing via a temp file: a half-written forecast CSV that still
    parses is worse than no file, because the scorer would silently score a truncated run.
    """
    d = forecast_dir(root, dataset)
    final = d / f"{model}.csv"
    tmp = d / f".{model}.csv.partial"
    n = 0
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
            n += 1
    if n == 0:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{model}/{dataset}: refusing to write an empty forecast file")
    os.replace(tmp, final)

    meta = dict(meta, model=model, dataset=dataset, rows=n)
    (d / f"{model}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return final


def read_forecasts(root, dataset, model):
    """-> (by_origin: {origin: {"point": [...], "q": [[9 deciles] per step] | None}}, meta)"""
    d = pathlib.Path(root) / dataset
    path = d / f"{model}.csv"
    if not path.exists():
        raise FileNotFoundError(f"no forecasts for {model} on {dataset}: {path}")
    by: dict[int, dict] = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            o = int(r["origin"])
            slot = by.setdefault(o, {"point": [], "q": []})
            slot["point"].append(float(r["point"]))
            qs = [r[k] for k in DECILES]
            slot["q"].append([float(v) for v in qs] if all(v != "" for v in qs) else None)
    for o, slot in by.items():
        if any(q is None for q in slot["q"]):
            slot["q"] = None          # partial quantiles are not usable; treat as point-only
    meta_path = d / f"{model}.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return by, meta


def available_models(root, dataset):
    d = pathlib.Path(root) / dataset
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.csv"))
