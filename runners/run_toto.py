#!/usr/bin/env python3
"""Toto (Datadog) runner, in its own virtualenv.

The checkpoint is Toto-Open-Base-1.0, the openly released model. Datadog's newer
Toto-2.0 family (which tops GIFT-Eval) was not runnable here; rerunning with it is
queued future work. The result key "toto_2p0" is a naming slip kept for artifact
compatibility: it came from the toto-ts 0.2.0 PACKAGE version, not a model generation.

    .venv-toto/bin/python runners/run_toto.py

Toto cannot share an environment with TimesFM: `toto-ts==0.2.0` pins numpy back to 1.26.4,
and NumPy 2.0 changed the C ABI, so extensions compiled against 2.x will not load. This
runner therefore imports NO core-family model code. It reads the plan (which origins, which
context) from the truth file that run_core.py already wrote, so both families forecast
exactly the same tasks.

Run run_core.py FIRST. Without its truth file this runner cannot know the origins, and
score.py would refuse to compare mismatched ones anyway.

CAVEAT WORTH KEEPING IN VIEW
BOOM is Datadog's own benchmark and Toto trains on Datadog telemetry, so `boom_telemetry_5t`
is home turf for this model. A Toto win there is weaker evidence than the same margin on
`bangkok_pm25_1h`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import os

# Toto samples from a Gamma mixture, and `aten::_standard_gamma` has no MPS kernel as of
# torch 2.7: the forecast call dies with NotImplementedError on Apple Silicon. The CPU
# fallback is the documented workaround and only affects that one operator. Must be set
# BEFORE torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))

# Only the stdlib-only contract is shared. Importing datasets.py/prepare.py here would drag
# in the core numeric stack this environment deliberately does not have.
from forecast_contract import write_forecasts  # noqa: E402

FORECASTS = CODE / "forecasts"
DEFAULT_DATASETS = ["bangkok_pm25_1h", "boom_telemetry_5t"]
CHECKPOINT = "Datadog/Toto-Open-Base-1.0"
DECILES = np.arange(1, 10) / 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--samples", type=int, default=256, help="Toto is sample-based; deciles are empirical")
    # Toto asserts num_samples %% samples_per_batch == 0, and the default batch is 10,
    # so the obvious 256 fails outright. Pick a batch that always divides.
    ap.add_argument("--sample-batch", type=int, default=64)
    a = ap.parse_args()

    from toto.model.toto import Toto
    from toto.inference.forecaster import TotoForecaster
    from toto.data.util.dataset import MaskedTimeseries

    # Toto's forecasts are Monte Carlo samples; seed so a rerun reproduces them.
    # The committed cross-environment artifacts predate this seed.
    torch.manual_seed(0)
    np.random.seed(0)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Toto.from_pretrained(CHECKPOINT).to(device).eval()
    forecaster = TotoForecaster(model.model)
    print(f"Toto loaded on {device}: {CHECKPOINT}")

    for key in a.datasets:
        tpath = FORECASTS / "_truth" / f"{key}.json"
        if not tpath.exists():
            raise SystemExit(f"no truth for {key}. Run .venv-core/bin/python runners/run_core.py first.")
        truth = json.loads(tpath.read_text())
        origins, horizon, context = truth["origins"], truth["horizon"], truth["context"]
        contexts = {int(k): v for k, v in truth["contexts"].items()}
        print(f"\n{key}: {len(origins)} origins, horizon {horizon}, context {context}")

        rows, t0 = [], time.perf_counter()
        for o in origins:
            ctx = np.asarray(contexts[o], dtype="float32")
            series = torch.tensor(ctx, device=device)[None, None, :]        # (batch, variate, time)
            mt = MaskedTimeseries(
                series=series,
                padding_mask=torch.ones_like(series, dtype=torch.bool),
                id_mask=torch.zeros_like(series),
                timestamp_seconds=torch.zeros_like(series),
                time_interval_seconds=torch.full((1, 1), 1, device=device),
            )
            with torch.inference_mode():
                fc = forecaster.forecast(mt, prediction_length=horizon, num_samples=a.samples,
                                         samples_per_batch=a.sample_batch)
            samples = np.asarray(fc.samples.squeeze().cpu())
            # squeeze() destroys the horizon axis when horizon == 1 (a daily series at a
            # 1-day horizon), leaving a bare (samples,) vector and an AxisError downstream.
            # Restore the axis explicitly rather than trusting the shape.
            if samples.ndim == 1:
                samples = samples.reshape(1, -1) if horizon == 1 else samples.reshape(-1, 1)
            elif samples.shape[0] != horizon:
                samples = samples.T
            # Sample mean as the point forecast. Other adapters emit medians, which
            # MAE favors; the mix is disclosed in README "Threats to validity".
            point = samples.mean(axis=1)
            qs = np.quantile(samples, DECILES, axis=1).T                    # (horizon, 9)
            for s in range(horizon):
                r = {"origin": o, "step": s, "point": float(point[s])}
                for i, d in enumerate(range(10, 100, 10)):
                    r[f"q{d}"] = float(qs[s][i])
                rows.append(r)
        per = (time.perf_counter() - t0) / len(origins)

        write_forecasts(FORECASTS, key, "toto_2p0", rows, {
            "context": context, "horizon": horizon, "n_origins": len(origins),
            "seconds_per_forecast": round(per, 6), "env": "toto",
            "checkpoint": CHECKPOINT, "num_samples": a.samples, "seed": 0,
            "torch": torch.__version__, "numpy": np.__version__,
        })
        print(f"  toto_2p0 ok  {per*1000:.1f} ms/forecast")

    print("\nNext:  python3 score.py --datasets " + " ".join(a.datasets))


if __name__ == "__main__":
    main()
