#!/bin/bash
# One press for a staged equivalence benchmark.
#
#   tests/equivalence/submit_benchmark.sh              # smoke, then full run chained afterok
#   tests/equivalence/submit_benchmark.sh --smoke-only # just the smoke
#   tests/equivalence/submit_benchmark.sh --full-only  # skip the smoke (cache and driver already proven)
#   tests/equivalence/submit_benchmark.sh --dry-run    # print the sbatch lines, submit nothing
#
# Why this exists: on 2026-09-14 the harness was verified for a whole day and
# nobody submitted a job, because the launch needed eight env knobs and a
# decision. This script makes it one decidable action, and the Slurm
# dependency means one press yields both runs without racing the shared cache
# (the driver forbids two concurrent runs against one EQ_CACHE).
#
# Stage 1, the smoke: a small interconnect built on both sides and stopped at
# the assembled stage. Proves the driver, the environment, the config gate and
# the cache in a couple of hours before the 20 h reservation is spent.
# Stage 2, the full run: the driver's own defaults (whole USA), released by
# Slurm only if the smoke exits 0.
#
# Every knob is an env var with a default. Nothing here is a path to edit.
# Runs on a login node: it calls sbatch, git and sha256sum, nothing heavier.
#
# Output: one `key=value` line per fact on stdout, so a caller (the brain's
# /benchmark skill) can write a run card without parsing prose.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVER="tests/equivalence/run_equivalence.sbatch"
[ -f "$REPO/$DRIVER" ] || { echo "ERROR: $REPO/$DRIVER not found" >&2; exit 1; }
cd "$REPO"

do_smoke=1; do_full=1; dry=0
for a in "$@"; do
    case "$a" in
        --smoke-only) do_full=0 ;;
        --full-only)  do_smoke=0 ;;
        --dry-run)    dry=1 ;;
        -h|--help)    sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument $a" >&2; exit 2 ;;
    esac
done

# --- smoke stage knobs (small, stops before the solve) ----------------------
SMOKE_INTERCONNECT="${EQ_SMOKE_INTERCONNECT:-western}"
SMOKE_SIMPL="${EQ_SMOKE_SIMPL:-20}"
SMOKE_CLUSTERS="${EQ_SMOKE_CLUSTERS:-4}"
SMOKE_PRONG="${EQ_SMOKE_PRONG:-2}"
SMOKE_UNTIL="${EQ_SMOKE_UNTIL:-assembled}"
# Sized from the western smoke runs of 2026-09-14..18: 2-20 min wall, peak
# RSS 15 GB once (43580538), 2 GB typically. Sherlock bills 1/CPU + 0.25/GB.
SMOKE_TIME="${EQ_SMOKE_TIME:-01:00:00}"
SMOKE_CPUS="${EQ_SMOKE_CPUS:-4}"
SMOKE_MEM="${EQ_SMOKE_MEM:-24GB}"

# --- full stage knobs: the driver's own defaults unless overridden ----------
FULL_INTERCONNECT="${EQ_INTERCONNECT:-usa}"
FULL_TIME="${EQ_FULL_TIME:-}"          # empty = the driver's #SBATCH --time
FULL_MEM="${EQ_FULL_MEM:-}"
FULL_CPUS="${EQ_FULL_CPUS:-}"

PARTITION="${EQ_PARTITION:-serc}"
STAMP="$(date +%Y%m%d-%H%M)"

# --- provenance, taken BEFORE submission so the card names what will run ----
develop_sha="$(git rev-parse HEAD)"
baseline_ref="${EQ_BASELINE_REF:-master-benchmark}"
baseline_sha="$(git rev-parse "$baseline_ref" 2>/dev/null || echo unknown)"
cfg_name="config.equivalence.yaml"
[ "$FULL_INTERCONNECT" = "western" ] || cfg_name="config.equivalence-${FULL_INTERCONNECT}.yaml"
cfg_path="workflow/repo_data/config/$cfg_name"
if [ -f "$cfg_path" ]; then cfg_sha="$(sha256sum "$cfg_path" | cut -c1-12)"; else cfg_sha="missing"; fi
# hash12 over resolved config + code (experiments/runs/README.md): same config
# on different code is a different run.
hash12="$(printf '%s\n%s\n%s\n%s\n%s\n' "$develop_sha" "$baseline_sha" "$cfg_sha" \
    "$FULL_INTERCONNECT" "${EQ_OPTS:-3h}" | sha256sum | cut -c1-12)"
run_name="eq-${FULL_INTERCONNECT}"
[ "$do_full" = 1 ] || run_name="eq-smoke-${SMOKE_INTERCONNECT}"

echo "repo=$REPO"
echo "develop_sha=$develop_sha"
echo "baseline_ref=$baseline_ref"
echo "baseline_sha=$baseline_sha"
echo "config=$cfg_path"
echo "config_sha12=$cfg_sha"
echo "run_id=${run_name}-${hash12}"
echo "submitted_at=$(date -u +%FT%TZ)"
echo "dirty_paths=$(git status --porcelain | grep -c . || true)"

_submit() {  # prints the job id, or a fake one on --dry-run
    if [ "$dry" = 1 ]; then
        echo "DRY: $*" >&2
        echo "dry-$RANDOM"
    else
        sbatch --parsable "$@"
    fi
}

smoke_job=""
if [ "$do_smoke" = 1 ]; then
    smoke_job="$(EQ_INTERCONNECT="$SMOKE_INTERCONNECT" EQ_SIMPL="$SMOKE_SIMPL" \
        EQ_CLUSTERS="$SMOKE_CLUSTERS" EQ_PRONG="$SMOKE_PRONG" EQ_UNTIL="$SMOKE_UNTIL" \
        EQ_RUN_ID="smoke-${SMOKE_INTERCONNECT}-${STAMP}" \
        EQ_EXTRA_ARGS="${EQ_SMOKE_EXTRA_ARGS:---verdict-exit report}" \
        _submit -J "eq-smoke-${SMOKE_INTERCONNECT}" -p "$PARTITION" \
            --time="$SMOKE_TIME" --cpus-per-task="$SMOKE_CPUS" --mem="$SMOKE_MEM" \
            --mail-type=END,FAIL "$DRIVER")"
    echo "smoke_job=$smoke_job"
    echo "smoke_run_dir=workflow/results/equivalence/smoke-${SMOKE_INTERCONNECT}-${STAMP}"
    echo "smoke_slurm_log=$REPO/slurm-eq-${smoke_job}.out"
fi

full_job=""
if [ "$do_full" = 1 ]; then
    extra=()
    [ -n "$smoke_job" ] && extra+=(--dependency="afterok:${smoke_job}")
    [ -n "$FULL_TIME" ] && extra+=(--time="$FULL_TIME")
    [ -n "$FULL_MEM" ]  && extra+=(--mem="$FULL_MEM")
    [ -n "$FULL_CPUS" ] && extra+=(--cpus-per-task="$FULL_CPUS")
    full_job="$(EQ_INTERCONNECT="$FULL_INTERCONNECT" \
        EQ_RUN_ID="${FULL_INTERCONNECT}-p${EQ_PRONG:-2}-${EQ_OPTS:-3h}-${STAMP}" \
        _submit -J "eq-${FULL_INTERCONNECT}" -p "$PARTITION" --mail-type=END,FAIL \
            ${extra[@]+"${extra[@]}"} "$DRIVER")"
    echo "full_job=$full_job"
    echo "full_run_dir=workflow/results/equivalence/${FULL_INTERCONNECT}-p${EQ_PRONG:-2}-${EQ_OPTS:-3h}-${STAMP}"
    echo "full_slurm_log=$REPO/slurm-eq-${full_job}.out"
    [ -n "$smoke_job" ] && echo "dependency=afterok:${smoke_job}"
fi

echo "watch=squeue --me -o '%.10i %.18j %.2t %.10M %R'"
