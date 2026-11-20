#!/usr/bin/env bash
# Study driver for the California weather-year campaign: runs several scenario
# overlays, each through the generic run_slurm.sh (one Slurm job per rule).
# For a single ordinary scenario use run_slurm.sh directly.
#
#   bash run_slurm_weather_years.sh ca2040_wy2019_z4.yaml ca2040_wy2019_county.yaml
#   bash run_slurm_weather_years.sh $(cd config/weather_years && ls ca2040_wy*_z4.yaml)
#
# Each overlay becomes its own run_slurm.sh invocation. Overlays run
# CONCURRENCY-at-a-time; within each, up to JOBS rule-jobs are in flight.
#
# Config layering (left to right, later wins):
#   repo_data/config/config.{slurm,common,plotting,api,sector,default}.yaml   auto-loaded by the Snakefile
#   repo_data/config/config.california.yaml                                   passed here
#   config/weather_years/<overlay>.yaml                                       passed here
#
# The build target comes from config/weather_years/manifest.tsv and is the
# add_extra_components output, so ll/opts/sector never enter the DAG.
#
# Site settings are Slurm's SBATCH_* variables (see slurm_submit.sh). For
# backwards compatibility PARTITION and EMAIL are still accepted, and the
# partition defaults to serc because this campaign runs on Sherlock. Like
# run_slurm.sh, this script must run inside a job or an sh_dev shell, never on
# the login node.
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SBATCH_PARTITION="${SBATCH_PARTITION:-${PARTITION:-serc}}"
[ -z "${EMAIL:-}" ] || export SBATCH_MAIL_USER="${SBATCH_MAIL_USER:-$EMAIL}"
CONCURRENCY="${CONCURRENCY:-4}"    # concurrent snakemake drivers
# 0 while benchmarking: a retry re-runs with attempt*mem and would pollute
# MaxRSS data. run_slurm.sh reads JOBS and RESTART_TIMES itself.
export RESTART_TIMES="${RESTART_TIMES:-0}"

CA_CONFIG=repo_data/config/config.california.yaml
OVERLAY_DIR=config/weather_years
MANIFEST="$OVERLAY_DIR/manifest.tsv"

[ $# -ge 1 ] || { echo "usage: $0 <overlay.yaml> [overlay.yaml ...]" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "missing $MANIFEST -- run $OVERLAY_DIR/generate_overlays.sh" >&2; exit 2; }

mkdir -p logs/drivers

run_one() {
  local overlay="$1" row run_name target
  row=$(awk -F'\t' -v o="$overlay" '$1==o{print; exit}' "$MANIFEST")
  if [ -z "$row" ]; then
    echo "[SKIP] $overlay not in $MANIFEST" >&2
    return 1
  fi
  run_name=$(printf '%s' "$row" | cut -f2)
  target=$(printf '%s' "$row" | cut -f3)

  echo "[START] $run_name -> $target"
  # NB: the target MUST precede --configfile. `--configfile` is nargs='+' and
  # would otherwise swallow the target as a third config file.
  # shellcheck disable=SC2086
  bash run_slurm.sh \
    "$target" \
    --configfile "$CA_CONFIG" "$OVERLAY_DIR/$overlay" \
    --rerun-triggers $RERUN_TRIGGERS \
    > "logs/drivers/${run_name}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "[DONE ] $run_name"
  else
    echo "[FAIL ] $run_name (rc=$rc) -- see logs/drivers/${run_name}.log" >&2
  fi
  return $rc
}

# Scheduling.
#
# Snakemake locks a run's input/output FILE SETS, not the working directory, so
# two runs collide only when both want to CREATE the same file. Almost all
# outputs live under resources/<run.name>/, but two tiers do not:
#
#   global   resources/powerplants/powerplants.csv, data/cpuc/..., data/caiso/...,
#            data/nrel/...        -- shared by every overlay
#   per-year data/godeeep/historical/*_{year}_*.nc, cutouts/*_{year}.nc
#            -- shared by the z4 and county overlays of the SAME weather year
#
# Hence: bootstrap one overlay alone so the global tier exists, then serialise
# within a weather year and parallelise across years.
BOOTSTRAP="${BOOTSTRAP:-1}"

# Only rebuild on mtime. Without this a code or config edit mid-campaign pulls
# the shared retrieve rules back into every DAG at once and they all collide.
RERUN_TRIGGERS="${RERUN_TRIGGERS:-mtime}"

year_of() { printf '%s' "$1" | sed -n 's/.*_wy\([0-9]\{4\}\)_.*/\1/p'; }

declare -A GROUP
YEARS=()
for overlay in "$@"; do
  y=$(year_of "$overlay")
  [ -n "$y" ] || y="ungrouped_$overlay"
  [ -n "${GROUP[$y]:-}" ] || YEARS+=("$y")
  GROUP[$y]="${GROUP[$y]:-} $overlay"
done

fails=0

if [ "$BOOTSTRAP" = "1" ] && [ $# -gt 1 ]; then
  echo "=== bootstrap: building globally-shared artifacts via $1 ==="
  run_one "$1" || fails=$((fails + 1))
fi

run_group() {
  local rc=0 o
  for o in "$@"; do run_one "$o" || rc=1; done
  return $rc
}

for y in "${YEARS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do wait -n; done
  # shellcheck disable=SC2086
  run_group ${GROUP[$y]} &
done
wait

for overlay in "$@"; do
  row=$(awk -F'\t' -v o="$overlay" '$1==o{print; exit}' "$MANIFEST")
  [ -n "$row" ] || continue
  t=$(printf '%s' "$row" | cut -f3)
  [ -f "$t" ] || { echo "[MISSING] $t"; fails=$((fails + 1)); }
done

echo "=== ${#@} overlays attempted, ${fails} missing target(s) ==="
exit $(( fails > 0 ? 1 : 0 ))
