#!/bin/bash
# =============================================================================
# PyPSA-USA — regenerate the reporting artifacts for a finished run
# =============================================================================
# CH2's run_figures.sbatch pattern: one memory-bound job, parameterised only by
# environment variables, and tolerant of a single failing group -- `|| status=1`
# rather than `set -e` aborting, so one bad plot does not lose the other twenty.
#
# Sizing rationale: the equivalence figures load one solved USA network per
# side (O(300 MB) on disk, several GB resident once expanded) plus the two
# regions GeoJSONs for the choropleths. That is memory-bound, not CPU-bound,
# and each group runs in its own process. 64 GB / 2 h is ~2x the largest group
# observed; raise with `sbatch --mem=`.
#
#   mkdir -p logs/controller
#   sbatch workflow/run_report.sbatch
#   PYPSA_REPORT_GROUPS="benchmarks stats" sbatch workflow/run_report.sbatch
#   EQ_RUN_ID=usa-p2-3h-20260914-1200 sbatch workflow/run_report.sbatch
#
# Chained after a set of controllers by workflow/run_slurm_all.sh:
#   PYPSA_REPORT=1 bash workflow/run_slurm_all.sh
#
# Groups (PYPSA_REPORT_GROUPS, space separated, default all three):
#   benchmarks   workflow/report_benchmarks.py  -> per-rule mem/time requests
#   stats        workflow/collect_solve_statistics.sh -> one tidy CSV
#   equivalence  tests.equivalence.run --side none -> comparison tables + PNGs
#                for an EXISTING run directory; builds nothing.
#
#SBATCH --job-name=pypsa-report
#SBATCH --partition=serc
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --open-mode=append
#SBATCH -o slurm-report-%x-%j.out
#SBATCH -e slurm-report-%x-%j.err

set -uo pipefail     # deliberately NOT -e: see the `|| status=1` pattern below

# --- Where am I? -------------------------------------------------------------
# sbatch spools this file to node-local storage, so ${BASH_SOURCE[0]} is
# useless inside a job. Use the submission directory; no repository path is
# written down here.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/workflow/run_report.sbatch" ]; then
    REPO="${PYPSA_REPO:-$SLURM_SUBMIT_DIR}"
else
    REPO="${PYPSA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi
if [ ! -f "$REPO/workflow/Snakefile" ]; then
    echo "ERROR: \$REPO=$REPO is not a pypsa-usa checkout." >&2
    exit 1
fi

LOG_DIR="$REPO/logs/controller"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/report-${SLURM_JOB_ID:-local-$$}.log") 2>&1
echo "[report] repo : $REPO"
echo "[report] job  : ${SLURM_JOB_ID:-<no slurm>} on $(hostname)"

# --- Environment -------------------------------------------------------------
ml load devel "${PYPSA_UV_MODULE:-uv/0.10.8}" 2>/dev/null || true
ml load system git/2.45.1
ml load gcc/12.4.0
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH" >&2; exit 1; }

PYPSA_CACHE_ROOT="${PYPSA_CACHE_ROOT:-${SCRATCH:?SCRATCH is unset}/pypsa-usa/cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PYPSA_CACHE_ROOT/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PYPSA_CACHE_ROOT/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PYPSA_CACHE_ROOT/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
export MPLBACKEND=Agg
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

cd "$REPO"

GROUPS="${PYPSA_REPORT_GROUPS:-benchmarks stats equivalence}"
status=0

for group in $GROUPS; do
    echo "=== group: $group ==="
    case "$group" in
        benchmarks)
            uv run python workflow/report_benchmarks.py workflow/benchmarks \
                | tee "$LOG_DIR/report_benchmarks-${SLURM_JOB_ID:-local}.txt" || status=1
            ;;
        stats)
            bash workflow/collect_solve_statistics.sh \
                "${PYPSA_STATS_CSV:-$REPO/workflow/results/solve_statistics.csv}" || status=1
            ;;
        equivalence)
            # --side none: compare, tabulate and plot an EXISTING run directory;
            # no snakemake, no rebuild. EQ_RUN_ID selects which run; without it
            # tests.equivalence.paths mints today's default id.
            for prong in ${EQ_PRONGS:-1 2}; do
                echo "--- prong $prong ---"
                uv run python -m tests.equivalence.run \
                    --prong "$prong" --side none ${EQ_REPORT_ARGS:-} || status=1
            done
            ;;
        *)
            echo "unknown report group '$group' (want: benchmarks stats equivalence)" >&2
            status=1
            ;;
    esac
done

echo "[report] finished with status $status"
exit "$status"
