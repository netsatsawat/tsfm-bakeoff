#!/usr/bin/env python3
"""Moirai-2 (Salesforce) runner, in its own virtualenv.

    .venv-moirai/bin/python runners/run_moirai.py

Isolated because `uni2ts==2.0.0` pins torch 2.4.1 / numpy 1.26.4 and pulls jax, lightning
and tensorboard. Like the Toto runner, this imports no core-family model code and takes its
origins from the truth file run_core.py wrote, so every family forecasts identical tasks.

LICENSE WARNING
Moirai-2 weights are released CC-BY-NC-4.0, NON-COMMERCIAL. Results are fine for research
and publication; do not ship a commercial product built on them. TimesFM, Chronos and Toto
are all Apache-2.0 by contrast.

Moirai is gluonts-native: it builds a PyTorchPredictor and consumes a gluonts dataset,
rather than exposing a plain array call like the others. The shim below is the smallest
thing that turns one context window into a gluonts entry.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))

from forecast_contract import write_forecasts  # noqa: E402

FORECASTS = CODE / "forecasts"
DEFAULT_DATASETS = ["bangkok_pm25_1h"]
CHECKPOINT = "Salesforce/moirai-2.0-R-small"
DECILES = np.arange(1, 10) / 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--samples", type=int, default=100)
    # CPU by default. gluonts' batchify builds float64 tensors for its own generated
    # fields (not just the target), and MPS has no float64 kernel, so device="mps" dies
    # with "Cannot convert a MPS Tensor to float64" no matter how the input is cast.
    # moirai-2.0-R-small is small enough that CPU is a fine trade for actually working.
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    a = ap.parse_args()

    from gluonts.dataset.pandas import PandasDataset
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    device = a.device
    module = Moirai2Module.from_pretrained(CHECKPOINT)
    print(f"Moirai-2 loaded ({CHECKPOINT}), device {device}")

    for dataset in a.datasets:
        tpath = FORECASTS / "_truth" / f"{dataset}.json"
        if not tpath.exists():
            raise SystemExit(f"no truth for {dataset}. Run run_core.py first.")
        truth = json.loads(tpath.read_text())
        origins, horizon, context = truth["origins"], truth["horizon"], truth["context"]
        contexts = {int(k): v for k, v in truth["contexts"].items()}
        print(f"\n{dataset}: {len(origins)} origins, horizon {horizon}, context {context}")

        model = Moirai2Forecast(
            module=module, prediction_length=horizon, context_length=context,
            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=1, device=device)

        rows, t0 = [], time.perf_counter()
        for o in origins:
            ctx = np.asarray(contexts[o])
            # gluonts wants a timestamped frame; the index is arbitrary here because Moirai
            # infers periodicity from the context, but it must be regular.
            # float32 is mandatory on Apple Silicon: gluonts' batchify builds tensors from
            # the frame's dtype, and MPS has no float64 kernel; a float64 column dies with
            # "Cannot convert a MPS Tensor to float64 dtype".
            frame = pd.DataFrame(
                {"target": ctx.astype("float32")},
                index=pd.date_range("2024-01-01", periods=len(ctx), freq="h"),
            )
            gds = PandasDataset(frame, target="target")
            fc = next(iter(predictor.predict(gds)))
            # Moirai-2 returns a gluonts QuantileForecast (predicted quantiles directly),
            # not the SampleForecast that Toto and older gluonts models emit. Handle both
            # so this runner does not break on a checkpoint that changes representation.
            if hasattr(fc, "samples"):
                samples = np.asarray(fc.samples)              # (num_samples, horizon)
                point = samples.mean(axis=0)
                qs = np.quantile(samples, DECILES, axis=0).T  # (horizon, 9)
            else:
                qs = np.stack([np.asarray(fc.quantile(float(d))) for d in DECILES], axis=1)
                point = np.asarray(fc.mean) if hasattr(fc, "mean") else qs[:, 4]
            for s in range(horizon):
                r = {"origin": o, "step": s, "point": float(point[s])}
                for i, d in enumerate(range(10, 100, 10)):
                    r[f"q{d}"] = float(qs[s][i])
                rows.append(r)
        per = (time.perf_counter() - t0) / len(origins)

        write_forecasts(FORECASTS, dataset, "moirai_2", rows, {
            "context": context, "horizon": horizon, "n_origins": len(origins),
            "seconds_per_forecast": round(per, 6), "env": "moirai",
            "checkpoint": CHECKPOINT, "license": "CC-BY-NC-4.0 (non-commercial)",
            "torch": torch.__version__, "numpy": np.__version__,
        })
        print(f"  moirai_2 ok  {per*1000:.1f} ms/forecast")

    print("\nNext:  python3 score.py --datasets " + " ".join(a.datasets))


if __name__ == "__main__":
    main()
