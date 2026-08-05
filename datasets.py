#!/usr/bin/env python3
"""Six diverse, no-API-key, post-cutoff series. Fetch once, cache to CSV.

WHY SIX AND WHY THESE
---------------------
A single dataset tells you almost nothing about a forecasting model. These are chosen to
span the axis that actually decides whether a foundation model helps: how much structure
the series has.

    uk_demand_30min   energy      30min   double seasonality (daily + weekly), very high
                                          structure. The best case for a TSFM.
    uk_demand_daily   energy      daily   same signal aggregated, weekly seasonality only.
                                          Tests whether frequency changes the verdict.
    bangkok_temp_1h   weather     hourly  near-sinusoidal tropical daily cycle, smooth,
                                          low variance. Easiest series here.
    london_temp_1h    weather     hourly  same variable, mid-latitude: weather fronts add
                                          multi-day drift a daily cycle cannot explain.
    bangkok_pm25_1h   air qual.   hourly  spiky, heavy-tailed, weak seasonality. The kind
                                          of series that embarrasses point forecasts.
    btc_usd_1h        finance     hourly  near-random-walk, no seasonality. Included as a
                                          CONTROL: if a model "wins" here by a wide
                                          margin, suspect the evaluation, not the model.

All windows are 2026 (plus 2025 context for the 30-minute series), which postdates every
cutoff declared on the TimesFM 2.5 model card and every published GIFT-Eval/LOTSA build.
Sources: NESO Open Data Licence; Open-Meteo (CC-BY 4.0, no key); Binance public REST.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / "data"
UA = {"User-Agent": "forecast-bench/1.0"}
WINDOW = ("2026-03-01", "2026-07-07")


@dataclass(frozen=True)
class Spec:
    key: str
    domain: str
    freq: str                # pandas offset alias
    season: int              # primary seasonal period, in steps
    season_long: int | None  # secondary period (e.g. weekly), if any
    unit: str
    positive: bool
    note: str
    extra: dict = field(default_factory=dict)


SPECS: dict[str, Spec] = {
    "uk_demand_30min": Spec("uk_demand_30min", "energy", "30min", 48, 336, "MW", True,
                            "UK national demand, half-hourly settlement periods"),
    "uk_demand_daily": Spec("uk_demand_daily", "energy", "D", 7, None, "MWh/day", True,
                            "UK national demand aggregated to daily total"),
    "bangkok_temp_1h": Spec("bangkok_temp_1h", "weather", "h", 24, 168, "degC", False,
                            "Bangkok 2m air temperature",
                            {"lat": 13.7563, "lon": 100.5018, "var": "temperature_2m"}),
    "london_temp_1h": Spec("london_temp_1h", "weather", "h", 24, 168, "degC", False,
                           "London 2m air temperature",
                           {"lat": 51.5072, "lon": -0.1276, "var": "temperature_2m"}),
    "bangkok_pm25_1h": Spec("bangkok_pm25_1h", "air_quality", "h", 24, 168,
                            "ug/m3", True, "Bangkok PM2.5",
                            {"lat": 13.7563, "lon": 100.5018, "var": "pm2_5"}),
    "btc_usd_1h": Spec("btc_usd_1h", "finance", "h", 24, None, "USD", True,
                       "BTC-USD hourly close (random walk: drift, no seasonality)",
                       {"symbol": "BTCUSDT"}),
    # ---- the no-trend, no-seasonality group -------------------------------------
    # A forecasting benchmark without these is untrustworthy. If a model shows a large
    # advantage on a series with no exploitable structure, the harness is leaking, the
    # metric is broken, or the baseline is a straw man. These exist to catch that.
    "btc_returns_1h": Spec("btc_returns_1h", "finance", "h", 24, None, "log-return",
                           False,
                           "BTC hourly log returns: the textbook stationary, "
                           "zero-mean, non-seasonal real series",
                           {"derive_from": "btc_usd_1h"}),
    "quake_magnitude_seq": Spec("quake_magnitude_seq", "geophysics", "h", 24, None,
                                "Mw", True,
                                "Successive global earthquake magnitudes, indexed in "
                                "event order: Gutenberg-Richter draws, physically "
                                "memoryless",
                                {"minmagnitude": 4.5}),
    "white_noise_synth": Spec("white_noise_synth", "synthetic", "h", 24, None, "unit",
                              False,
                              "Gaussian white noise, seed 7: the control that proves "
                              "the harness cannot manufacture skill",
                              {"seed": 7, "n": 3096}),
    # ---- high-dimensional multivariate telemetry ---------------------------------
    # The archetype general-purpose TSFMs are documented to struggle on: overlapping
    # seasonalities, sudden anomalous surges, heavy tails, and dozens of correlated
    # variates per entity. BOOM (Datadog, arXiv 2505.14766, NeurIPS 2025, Apache-2.0)
    # is the public benchmark for it: 350M observations, 2,807 real multivariate series
    # from Datadog's own production telemetry. Each released file is a matrix of
    # (variates x 16,384 steps) at 10-second, 1-minute or 5-minute resolution.
    #
    # Leakage status is different from the 2026 series and worth stating precisely: BOOM
    # data is 2024-dated, so it is NOT post-cutoff by date. It is defensible for TimesFM
    # anyway because it is Datadog-internal telemetry first published in May 2025, after
    # GiftEvalPretrain was built, so it cannot be in the declared corpus. Note the
    # converse caveat: Toto trains on Datadog telemetry, so BOOM is home turf for Toto
    # and away turf for everyone else.
    "boom_telemetry_5t": Spec("boom_telemetry_5t", "observability", "5min", 288, 2016,
                              "normalised", False,
                              "BOOM production telemetry, 5-minute, sampled variates "
                              "from multivariate groups",
                              {"repo": "Datadog/BOOM",
                               "groups": ["ds-2-5T", "ds-139-5T", "ds-825-5T",
                                          "ds-1558-5T"],
                               "variates_per_group": 8,
                               "steps": 4032}),
}


def _get(url: str, retries: int = 3):
    last: Exception | None = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"GET failed: {url}") from last


# ------------------------------------------------------------------ fetchers
def fetch_neso(years: tuple[str, ...] = ("2025", "2026")) -> pd.DataFrame:
    ckan = "https://api.neso.energy/api/3/action"
    pkg = _get(f"{ckan}/package_show?id=historic-demand-data")
    by_year = {}
    for res in pkg["result"]["resources"]:
        for y in years:
            if y in res["name"]:
                by_year[y] = res["id"]
    rows: list[dict] = []
    for y in years:
        offset = 0
        while True:
            q = urllib.parse.urlencode({"resource_id": by_year[y], "limit": 5000,
                                        "offset": offset})
            doc = _get(f"{ckan}/datastore_search?{q}")
            batch = doc["result"]["records"]
            if not batch:
                break
            rows.extend(batch)
            offset += len(batch)
            if len(batch) < 5000:
                break
    df = pd.DataFrame(rows)
    df["SETTLEMENT_DATE"] = pd.to_datetime(df["SETTLEMENT_DATE"], errors="coerce")
    for c in ("SETTLEMENT_PERIOD", "ND", "EMBEDDED_SOLAR_GENERATION",
              "EMBEDDED_WIND_GENERATION"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"])
    df = df[(df.SETTLEMENT_PERIOD.between(1, 50)) & (df.ND > 0)]
    df = df.sort_values(["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"])
    df = df.drop_duplicates(["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"])
    ts = (df.SETTLEMENT_DATE
          + pd.to_timedelta((df.SETTLEMENT_PERIOD - 1) * 30, unit="m"))
    out = pd.DataFrame({"timestamp": ts.values, "value": df.ND.values})
    for c in ("EMBEDDED_SOLAR_GENERATION", "EMBEDDED_WIND_GENERATION"):
        if c in df:
            out[c.lower()] = df[c].values
    return out.reset_index(drop=True)


def fetch_open_meteo(spec: Spec) -> pd.DataFrame:
    host = ("air-quality-api" if spec.domain == "air_quality" else "archive-api")
    path = "air-quality" if spec.domain == "air_quality" else "archive"
    q = urllib.parse.urlencode({
        "latitude": spec.extra["lat"], "longitude": spec.extra["lon"],
        "start_date": WINDOW[0], "end_date": WINDOW[1],
        "hourly": spec.extra["var"], "timezone": "UTC",
    })
    doc = _get(f"https://{host}.open-meteo.com/v1/{path}?{q}")
    h = doc["hourly"]
    return pd.DataFrame({"timestamp": pd.to_datetime(h["time"]),
                         "value": h[spec.extra["var"]]})


def fetch_quakes(spec: Spec) -> pd.DataFrame:
    """Global earthquake magnitudes in event order (USGS FDSN, no key).

    Why this is the cleanest real no-structure series available: the magnitude of the
    next earthquake is drawn from the Gutenberg-Richter distribution and is, to the
    limits of current geophysics, independent of the magnitude of the last one. There is
    no diurnal cycle in tectonics and no trend over a few months. The series is indexed
    by EVENT ORDER, not clock time, which is deliberate -- it removes any residual
    time-of-day artefact from the catalogue itself.
    """
    rows: list[dict] = []
    start = pd.Timestamp(WINDOW[0])
    end = pd.Timestamp(WINDOW[1]) + pd.Timedelta(days=1)
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=30), end)
        q = urllib.parse.urlencode({
            "format": "geojson",
            "starttime": cursor.strftime("%Y-%m-%d"),
            "endtime": chunk_end.strftime("%Y-%m-%d"),
            "minmagnitude": spec.extra["minmagnitude"],
            "orderby": "time-asc",
            "limit": 20000,
        })
        doc = _get(f"https://earthquake.usgs.gov/fdsnws/event/1/query?{q}")
        for f in doc.get("features", []):
            p = f.get("properties", {})
            if p.get("mag") is None or p.get("time") is None:
                continue
            rows.append({"event_time": int(p["time"]), "value": float(p["mag"])})
        cursor = chunk_end
    df = pd.DataFrame(rows).drop_duplicates("event_time").sort_values("event_time")
    # Index by event order on a synthetic hourly grid: position is the only axis that
    # matters for a memoryless sequence, and every model here indexes by position.
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=len(df), freq="h"),
        "value": df["value"].to_numpy(),
        "event_time_ms": df["event_time"].to_numpy(),
    })


def fetch_boom(spec: Spec) -> pd.DataFrame:
    """Sample BOOM groups into a long-format panel: group, variate, timestamp, value.

    Downloads only the chosen groups (~1 MB each) rather than the full 2.81 GB repo.
    Keeping the group column matters: it is what lets a multivariate model forecast all
    variates of one entity jointly, which is the whole question BOOM exists to ask.
    """
    import pyarrow as pa
    from huggingface_hub import hf_hub_download

    steps = spec.extra["steps"]
    keep = spec.extra["variates_per_group"]
    frames: list[pd.DataFrame] = []
    for group in spec.extra["groups"]:
        path = hf_hub_download(spec.extra["repo"],
                              f"{group}/data-00000-of-00001.arrow",
                              repo_type="dataset")
        with pa.memory_map(path, "rb") as src:
            try:
                tbl = pa.ipc.open_file(src).read_all()
            except pa.ArrowInvalid:
                src.seek(0)
                tbl = pa.ipc.open_stream(src).read_all()
        row = tbl.to_pylist()[0]
        mat = np.asarray(row["target"], dtype=float)          # (variates, length)
        start = pd.Timestamp(row["start"])
        freq = str(row["freq"]).replace("T", "min").replace("S", "s")
        # Prefer the LAST `steps` columns: the tail is the part a forecaster would use,
        # and it keeps every group the same length.
        mat = mat[:, -steps:]
        idx = pd.date_range(start=start, periods=mat.shape[1], freq=freq)
        # Drop variates that are constant or mostly missing; they are not forecasting
        # tasks, they are configuration. Then take the first `keep` survivors.
        chosen = 0
        for vi in range(mat.shape[0]):
            v = mat[vi]
            finite = np.isfinite(v)
            # Reject non-tasks. A constant or near-constant variate is configuration, not
            # a forecasting problem, and it poisons every scale-relative metric: MASE
            # divides by the naive error and surge ratios divide by the MAD, both of which
            # go to zero. Checking std on the FULL matrix was not enough -- a variate can
            # be constant only within the trimmed tail, which is what we actually use.
            if finite.mean() < 0.99:
                continue
            vv = v[finite]
            if vv.size < 100 or np.unique(np.round(vv, 12)).size < 20:
                continue
            q1, q3 = np.percentile(vv, [25, 75])
            if (q3 - q1) <= 0 or np.median(np.abs(vv - np.median(vv))) <= 0:
                continue
            frames.append(pd.DataFrame({
                "group": group, "variate": f"v{vi:03d}",
                "timestamp": idx, "value": v,
            }))
            chosen += 1
            if chosen >= keep:
                break
    return pd.concat(frames, ignore_index=True)


def fetch_white_noise(spec: Spec) -> pd.DataFrame:
    rng = np.random.default_rng(spec.extra["seed"])
    n = spec.extra["n"]
    return pd.DataFrame({
        "timestamp": pd.date_range(WINDOW[0], periods=n, freq="h"),
        "value": rng.standard_normal(n),
    })


def fetch_binance(spec: Spec) -> pd.DataFrame:
    start = int(pd.Timestamp(WINDOW[0], tz="UTC").timestamp() * 1000)
    end = int((pd.Timestamp(WINDOW[1], tz="UTC")
               + pd.Timedelta(days=1)).timestamp() * 1000)
    rows: list[list] = []
    cursor = start
    while cursor < end:
        q = urllib.parse.urlencode({"symbol": spec.extra["symbol"], "interval": "1h",
                                    "startTime": cursor, "limit": 1000})
        batch = _get(f"https://api.binance.com/api/v3/klines?{q}")
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 3_600_000
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows).iloc[:, [0, 4]]
    df.columns = ["open_ms", "close"]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(df.open_ms, unit="ms"),
        "value": pd.to_numeric(df.close),
    })


# ------------------------------------------------------------------ public API
def load(key: str, refresh: bool = False) -> tuple[pd.DataFrame, Spec]:
    spec = SPECS[key]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["timestamp"]), spec

    if key == "uk_demand_30min":
        df = fetch_neso()
    elif key == "uk_demand_daily":
        base, _ = load("uk_demand_30min", refresh=refresh)
        # MW at 30-minute resolution -> MWh per half hour is value/2; daily total MWh.
        daily = (base.set_index("timestamp")["value"].div(2.0)
                 .resample("D").sum(min_count=40))
        df = daily.dropna().rename("value").reset_index()
    elif key == "btc_usd_1h":
        df = fetch_binance(spec)
    elif key == "btc_returns_1h":
        base, _ = load(spec.extra["derive_from"], refresh=refresh)
        px = base["value"].to_numpy(dtype=float)
        r = np.diff(np.log(px))
        df = pd.DataFrame({"timestamp": base["timestamp"].to_numpy()[1:], "value": r})
    elif key == "quake_magnitude_seq":
        df = fetch_quakes(spec)
    elif key == "white_noise_synth":
        df = fetch_white_noise(spec)
    elif key == "boom_telemetry_5t":
        df = fetch_boom(spec)
    else:
        df = fetch_open_meteo(spec)

    sort_cols = [c for c in ("group", "variate") if c in df.columns] + ["timestamp"]
    df = df.dropna(subset=["value"]).sort_values(sort_cols).reset_index(drop=True)
    df.to_csv(path, index=False)
    return df, spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--keys", nargs="*", default=list(SPECS))
    args = ap.parse_args()
    print(f"{'series':18} {'domain':12} {'freq':6} {'n':>7}  span")
    for key in args.keys:
        df, spec = load(key, refresh=args.refresh)
        print(f"{key:18} {spec.domain:12} {spec.freq:6} {len(df):7,}  "
              f"{df.timestamp.min()} -> {df.timestamp.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
