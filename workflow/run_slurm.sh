#!/usr/bin/env bash
# Run one or more PyPSA-USA scenario overlays, fanning each rule out to Slurm.
#
#   bash run_slurm.sh ca2040_wy2019_z4.yaml ca2040_wy2019_county.yaml
#   bash run_slurm.sh $(cd config/weather_years && ls ca2040_wy*_z4.yaml)
#
# Each overlay becomes its own `snakemake --cluster` invocation. Overlays run
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
# Sherlock notes: no --account (there are no accounts); this script itself must
# run inside a job or an sh_dev shell, never on the login node.
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PARTITION="${PARTITION:-serc}"
EMAIL="${EMAIL:-ctehran@stanford.edu}"
JOBS="${JOBS:-20}"                 # concurrent Slurm rule-jobs per overlay
CONCURRENCY="${CONCURRENCY:-4}"    # concurrent snakemake drivers
RESTART_TIMES="${RESTART_TIMES:-0}"  # 0 while benchmarking: a retry re-runs with
                                     # attempt*mem and would pollute MaxRSS data
# Use the venv binary directly rather than `uv run`, which would re-resolve the
# environment on every invocation. See CLAUDE.md for the uv setup.
SNAKEMAKE="${SNAKEMAKE:-$(cd .. && pwd)/.venv/bin/snakemake}"

CA_CONFIG=repo_data/config/config.california.yaml
OVERLAY_DIR=config/weather_years
MANIFEST="$OVERLAY_DIR/manifest.tsv"

[ $# -ge 1 ] || { echo "usage: $0 <overlay.yaml> [overlay.yaml ...]" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "missing $MANIFEST -- run $OVERLAY_DIR/generate_overlays.sh" >&2; exit 2; }

mkdir -p logs/slurm logs/drivers

export PARTITION EMAIL
# slurm_submit.sh rounds the float mem_mb some rules produce; see that file.
SBATCH_CMD="bash slurm_submit.sh {rule} {threads} {resources.mem_mb} {resources.walltime}"

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
  # shellcheck disable=SC2086
  # NB: the target MUST precede --configfile. `--configfile` is nargs='+' and
  # would otherwise swallow the target as a third config file.
  $SNAKEMAKE \
    "$target" \
    --cluster "$SBATCH_CMD" \
    --configfile "$CA_CONFIG" "$OVERLAY_DIR/$overlay" \
    --jobs "$JOBS" \
    --latency-wait 60 \
    --rerun-incomplete \
    --restart-times "$RESTART_TIMES" \
    --default-resources "mem_mb=8000" "walltime='02:00:00'" \
    --printshellcmds \
    > "logs/drivers/${run_name}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "[DONE ] $run_name"
  else
    echo "[FAIL ] $run_name (rc=$rc) -- see logs/drivers/${run_name}.log" >&2
  fi
  return $rc
}

fails=0
for overlay in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do wait -n; done
  run_one "$overlay" &
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
