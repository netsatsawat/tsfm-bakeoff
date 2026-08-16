#!/usr/bin/env python3
"""No number in the README without a runnable path behind it.

One assertion, wired into CI: every headline number in README.md is
recomputed here from results/cross_env_scores.json, the committed scoring
artifact, and the README must literally contain the recomputed string.

Recomputed from the artifact, never restated:
  - the protocol counts (cells, datasets, models, origins, environments)
  - cells won per family and per model, after collapsing context-length
    variants (`chronos_2_ctx968` -> `chronos_2`) and keeping the lowest
    MASE per (cell, model)
  - Toto's cell coverage and rank distribution
  - TimesFM's missing cells, which make its denominator smaller
  - the Bangkok PM2.5 best-foundation vs best-naive table

Horizon days come from the same conversion the runner uses: STEP_MIN is
read out of runners/run_core.py rather than duplicated here.

No network, no model, no benchmark: it reads committed JSON and text.
Stdlib only, matching the scorer.
"""

import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCORES = REPO / "results" / "cross_env_scores.json"

# Taxonomy, not measurement: which key belongs to which README table row.
# Every win count below is recomputed. `covers exactly the models in the
# artifact` guards these lists against a model being added and silently
# dropping out of the family totals.
FAMILIES = [
    ("**Foundation**", ["chronos_2", "chronos_bolt_small", "timesfm_2p5",
                        "moirai_2", "toto_2p0"]),
    ("Naive", ["naive_last", "naive_mean", "seasonal_naive",
               "seasonal_profile"]),
    ("Classical", ["autoets", "autotheta", "mstl"]),
    ("ML (GBDT)", ["gbdt_calendar"]),
]
BOLD_FAMILY = {"**Foundation**"}

MODEL_ROWS = [
    ("Chronos-2", "chronos_2"),
    ("TimesFM 2.5", "timesfm_2p5"),
    ("Moirai-2", "moirai_2"),
    ("Chronos-Bolt", "chronos_bolt_small"),
    ("**Toto (Toto-Open-Base-1.0)**", "toto_2p0"),
]
BOLD_MODEL = {"Chronos-2", "**Toto (Toto-Open-Base-1.0)**"}

# How the README writes each key where it names a winner.
NAME = {"chronos_2": "Chronos-2", "chronos_bolt_small": "Chronos-Bolt",
        "timesfm_2p5": "TimesFM 2.5", "moirai_2": "Moirai-2",
        "toto_2p0": "Toto", "naive_last": "naive_last",
        "naive_mean": "naive_mean", "seasonal_naive": "seasonal_naive",
        "seasonal_profile": "seasonal_profile", "autoets": "autoets",
        "autotheta": "autotheta", "mstl": "mstl",
        "gbdt_calendar": "gbdt_calendar"}

problems = []


def check(name, condition, detail=""):
    print(f"  {'ok ' if condition else 'FAIL'} {name}" +
          (f" ({detail})" if detail and not condition else ""))
    if not condition:
        problems.append(name)


def base_model(model):
    """chronos_2_ctx968 and chronos_2_ctx2048 are one model at two contexts."""
    return re.sub(r"_ctx\d+$", "", model)


def dataset_key(cell):
    """`boom_telemetry_5t@ds-139-5T_v003#h2016` -> `boom_telemetry_5t`."""
    return cell.split("#h")[0].split("@")[0]


def step_minutes():
    """The runner's minutes-per-step table, parsed out of the runner."""
    src = (REPO / "runners" / "run_core.py").read_text(encoding="utf-8")
    return ast.literal_eval(re.search(r"STEP_MIN = (\{.*?\})", src,
                                      re.S).group(1))


def horizon_days(cell, step_min):
    """Invert the runner's `round(days * 1440 / STEP_MIN[key])`."""
    return round(int(cell.split("#h")[1]) * step_min[dataset_key(cell)] / 1440)


def main() -> int:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    rows = json.loads(SCORES.read_text(encoding="utf-8"))
    step_min = step_minutes()

    # Lowest MASE per (cell, model) across context variants and environments.
    best = {}
    for r in rows:
        key = (r["dataset"], base_model(r["model"]))
        if key not in best or r["mase"] < best[key]:
            best[key] = r["mase"]

    cells = sorted({c for c, _ in best})
    models = sorted({m for _, m in best})
    datasets = sorted({dataset_key(c) for c in cells})
    origins = sorted({r["n_origins"] for r in rows})
    envs = sorted({r["env"] for r in rows})
    days = sorted({horizon_days(c, step_min) for c in cells})

    def cell_scores(cell):
        return {m: v for (c, m), v in best.items() if c == cell}

    wins = {m: 0 for m in models}
    for cell in cells:
        scored = cell_scores(cell)
        wins[min(scored, key=scored.get)] += 1

    n_cells, n_models, n_datasets = len(cells), len(models), len(datasets)

    print(f"protocol, recomputed from {SCORES.relative_to(REPO)}:")
    check("family map covers exactly the models in the artifact",
          sorted(k for _, ks in FAMILIES for k in ks) == models,
          f"artifact has {models}")
    check("every cell was scored at one origin count", len(origins) == 1,
          str(origins))
    n_origins = origins[0]
    phrase = (", ".join(str(d) for d in days[:-1]) + f" and {days[-1]} days")
    check(f"README headline states {n_cells} cells, {n_datasets} datasets, "
          f"horizons of {phrase}, {n_models} models",
          f"{n_cells} dataset x horizon cells: {n_datasets} datasets, "
          f"horizons of {phrase}, {n_models} models," in readme)
    check(f"README states {n_origins} rolling origins each",
          f"{n_origins} rolling origins each" in readme)
    check(f"README repeats {n_origins} origins per cell in the caveats",
          f"{n_origins} origins per cell" in readme)
    check(f"terminology note says all {n_models} models",
          f"all {n_models} models" in readme)
    check(f"models badge reads {n_models}", f"badge/models-{n_models}-" in
          readme)
    check(f"cells badge reads {n_cells}", f"cells-{n_cells}-" in readme)
    check(f"{len(envs)} environments in the artifact match envs/*.txt",
          envs == sorted(p.stem for p in (REPO / "envs").glob("*.txt")),
          f"artifact {envs}")
    check("README calls that three isolated Python environments",
          len(envs) == 3 and "three isolated Python environments" in readme)
    check(f"all {n_datasets} dataset keys appear in the dataset table",
          all(f"`{d}`" in readme for d in datasets),
          str([d for d in datasets if f"`{d}`" not in readme]))

    print("cells won per family:")
    for label, keys in FAMILIES:
        won = sum(wins[k] for k in keys)
        mark = "**" if label in BOLD_FAMILY else ""
        check(f"README family table: {label} {won}",
              f"| {label} | {mark}{won}{mark} |" in readme)
    # Sum the FAMILY-mapped wins, not wins.values(). The latter is incremented
    # once per cell and so equals n_cells by construction. It stayed green even
    # when a whole cell was deleted from the artifact. What can actually break is
    # the family map failing to cover a winner: a new model wins cells, belongs to
    # no family, and the family table silently under-counts.
    mapped = {k for _label, keys in FAMILIES for k in keys}
    orphans = sorted(m for m in wins if m not in mapped)
    check("every model that won a cell belongs to a family",
          not orphans, f"unmapped winner(s): {orphans}")
    family_total = sum(wins[k] for _label, keys in FAMILIES for k in keys
                       if k in wins)
    check(f"family wins sum to the {n_cells} cells",
          family_total == n_cells, f"{family_total} mapped vs {n_cells} cells")
    overlap = [k for k in mapped
               if sum(k in keys for _label, keys in FAMILIES) > 1]
    check("no model is counted in two families", not overlap, str(overlap))

    print("cells won per model:")
    for label, key in MODEL_ROWS:
        mark = "**" if label in BOLD_MODEL else ""
        check(f"README model table: {label} {wins[key]}",
              f"| {label} | {mark}{wins[key]}{mark} |" in readme)
    check("AutoETS won nothing, and the README says so",
          wins["autoets"] == 0 and "**AutoETS won nothing.**" in readme,
          f"autoets won {wins['autoets']}")
    check("the seasonal naive won nothing, and the README says so",
          wins["seasonal_naive"] == 0 and
          "Neither did the seasonal naive." in readme,
          f"seasonal_naive won {wins['seasonal_naive']}")

    print("Toto's coverage and rank spread:")
    toto_cells = [c for c in cells if (c, "toto_2p0") in best]
    ranks = []
    for cell in toto_cells:
        scored = cell_scores(cell)
        ranks.append(sorted(scored, key=scored.get).index("toto_2p0") + 1)
    mean_rank = sum(ranks) / len(ranks)
    check(f"README says Toto ran all {len(toto_cells)} cells",
          len(toto_cells) == n_cells and
          f"ran all {n_cells} cells" in readme, f"{len(toto_cells)} cells")
    check(f"README quotes mean rank {mean_rank:.1f} of {n_models} "
          f"(best {min(ranks)}, worst {max(ranks)}), won none",
          wins["toto_2p0"] == 0 and
          f"mean rank of {mean_rank:.1f} out of {n_models} "
          f"(best {min(ranks)}, worst {max(ranks)}) and won none" in readme)

    print("TimesFM's smaller denominator:")
    missing = [c for c in cells if (c, "timesfm_2p5") not in best]
    tf_cells = n_cells - len(missing)
    check(f"README names the {len(missing)} cells TimesFM is absent from",
          all(f"`{c}`" in readme for c in missing),
          str([c for c in missing if f"`{c}`" not in readme]))
    check(f"README says TimesFM's {wins['timesfm_2p5']} wins come from "
          f"{tf_cells} cells",
          f"Its {wins['timesfm_2p5']} wins therefore come from {tf_cells}"
          in readme)
    check(f"README says Chronos-2's {wins['chronos_2']} come from {n_cells}",
          f"Chronos-2's {wins['chronos_2']} come from {n_cells}." in readme)
    others = {m: sum(1 for c in cells if (c, m) in best)
              for m in models if m != "timesfm_2p5"}
    check(f"README threat: TimesFM ran {tf_cells} of {n_cells}, every other "
          f"model all {n_cells}",
          all(v == n_cells for v in others.values()) and
          f"TimesFM ran {tf_cells} of {n_cells} cells; every other model ran "
          f"all {n_cells}." in readme,
          str({m: v for m, v in others.items() if v != n_cells}))

    print("Bangkok PM2.5, best foundation vs best naive:")
    foundation = dict(FAMILIES)["**Foundation**"]
    naive = dict(FAMILIES)["Naive"]
    pm25 = sorted((c for c in cells if dataset_key(c) == "bangkok_pm25_1h"),
                  key=lambda c: horizon_days(c, step_min))
    for cell in pm25:
        scored = cell_scores(cell)
        f_best = min(scored[m] for m in foundation if m in scored)
        n_best = min(scored[m] for m in naive if m in scored)
        f_mark = "**" if f_best < n_best else ""
        n_mark = "" if f_best < n_best else "**"
        winner = NAME[min(scored, key=scored.get)]
        row = (f"| {horizon_days(cell, step_min)}-day | "
               f"{f_mark}{f_best:.3f}{f_mark} | {n_mark}{n_best:.3f}{n_mark} "
               f"| {winner} |")
        check(f"README horizon table: {row.strip('| ')}", row in readme)

    if problems:
        print(f"\n{len(problems)} README claim(s) drifted: {problems}")
        return 1
    print("\nevery quoted README number matches its artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
