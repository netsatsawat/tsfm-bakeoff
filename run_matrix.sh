#!/usr/bin/env bash
# Full matrix: every dataset x every feasible horizon x every model x every environment.
#
# Combinations that cannot work are SKIPPED, not failed:
#   - a horizon longer than half the series leaves no room for context + origins
#   - BOOM carries 14 days per series, so 14d and 30d are impossible there
# The runner refuses those itself; this driver logs the skip and continues, because one
# infeasible cell must not take down a multi-hour run.
#
# Horizons are given in DAYS and converted per dataset frequency, so "7d" is the same
# business decision on 30-minute demand as on hourly air quality. Within any single cell
# every model faces the identical horizon — score.py enforces that.
#
# Usage:  bash run_matrix.sh [origins]

set -uo pipefail          # deliberately NOT -e: a failing cell must not abort the matrix
cd "$(dirname "${BASH_SOURCE[0]}")"

ORIGINS="${1:-16}"
LOG="results/matrix.log"
mkdir -p results
: > "$LOG"

DATASETS=(uk_demand_30min uk_demand_daily bangkok_temp_1h london_temp_1h bangkok_pm25_1h
          btc_usd_1h btc_returns_1h white_noise_synth quake_magnitude_seq boom_telemetry_5t)
HORIZONS=(1 7 14 30)

say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

say "matrix start: ${#DATASETS[@]} datasets x ${#HORIZONS[@]} horizons, $ORIGINS origins"

for ds in "${DATASETS[@]}"; do
  for h in "${HORIZONS[@]}"; do
    # BOOM holds 14 days per series; a 14- or 30-day horizon cannot leave room for context.
    if [ "$ds" = "boom_telemetry_5t" ] && [ "$h" -ge 14 ]; then
      say "SKIP  $ds @ ${h}d — only 14 days of data per series"; continue
    fi
    say "RUN   $ds @ ${h}d"
    if .venv-core/bin/python runners/run_core.py \
         --datasets "$ds" --origins "$ORIGINS" --horizon-days "$h" \
         --series-cap 1 >> "$LOG" 2>&1; then
      say "  core ok"
    else
      say "  core SKIPPED/FAILED (see $LOG) — continuing"
      continue      # no truth file, so the satellites have nothing to read
    fi

    # Label the core runner produced, so the satellites target exactly the same cell.
    label=$(ls -t forecasts | grep -E "^${ds}(@|#)" | head -1)
    [ -z "$label" ] && { say "  no label found, skipping satellites"; continue; }

    .venv-toto/bin/python   runners/run_toto.py   --datasets "$label" >> "$LOG" 2>&1 \
      && say "  toto ok" || say "  toto failed"
    .venv-moirai/bin/python runners/run_moirai.py --datasets "$label" >> "$LOG" 2>&1 \
      && say "  moirai ok" || say "  moirai failed"
  done
done

say "matrix complete — scoring"
python3 score.py --out results/cross_env_scores.json >> "$LOG" 2>&1 \
  && say "scored -> results/cross_env_scores.json" || say "scoring failed, see $LOG"
say "done"
