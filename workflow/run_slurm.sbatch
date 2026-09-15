#!/bin/bash
# =============================================================================
# PyPSA-USA Slurm controller — one job, one config, one Snakemake scheduler
# =============================================================================
# The scheduler runs HERE, inside a compute job, and submits one Slurm job per
# rule through workflow/snakemake_profiles/sherlock. It never runs on a login
# node. This is CH2's ch2_controller.sbatch with the hard-coded repository
# path, the hard-coded conda interpreter and the missing cancel-my-children
# trap taken out.
#
#   mkdir -p logs/controller                      # once, from the repo root
#   sbatch workflow/run_slurm.sbatch <configfile> [snakemake targets/args...]
#
# Everything about the job itself is parameterised on the sbatch command line,
# which overrides the #SBATCH defaults below:
#
#   sbatch --time=12:00:00 --mem=16G -c 4 workflow/run_slurm.sbatch <config>
#
# One-node ("monolithic") mode — no per-rule submission, Snakemake runs every
# rule inside THIS job. Use it for equivalence runs and for DAGs under ~40
# homogeneous jobs; see docs/hpc.md for the decision rule.
#
#   sbatch -c 16 --mem=160G --time=24:00:00 \
#       workflow/run_slurm.sbatch --local <configfile> [targets...]
#
# Env knobs (all optional, all with tracked defaults):
#   PYPSA_SLURM_PARTITION  serc          partition for the CHILD jobs
#   PYPSA_SLURM_LOGDIR     $SCRATCH/...  per-rule Slurm stdout/stderr
#   PYPSA_SBATCH_EXTRA     ""            extra sbatch flags for child jobs
#   PYPSA_NOLOCK           0             see "--nolock" below
#   PYPSA_CACHE_ROOT       $SCRATCH/...  uv / pip / matplotlib caches
#   GRB_LICENSE_FILE       site default  Gurobi licence
#   PYPSA_SNAKEMAKE        "uv run snakemake"
#   PYPSA_RESULTS_ARCHIVE  ""            rsync results/ here when set
#
#SBATCH --job-name=pypsa-usa
#SBATCH --partition=serc
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@180
#
# Slurm's own capture goes to a RELATIVE path, i.e. into the directory sbatch
# was run from, which by construction exists and is writable. sbatch does NOT
# create -o/-e directories and a missing one fails the job at launch, before a
# single line of this script runs -- so the tracked default cannot be
# logs/controller/. The script tees its whole output into logs/controller/
# itself (mkdir -p'd below); that is the copy that stays with the run.
#SBATCH -o slurm-ctrl-%x-%j.out
#SBATCH -e slurm-ctrl-%x-%j.err

set -euo pipefail

# --- 0. Where am I? ----------------------------------------------------------
# sbatch spools this script to node-local storage, so ${BASH_SOURCE[0]} is
# useless inside a job; $SLURM_SUBMIT_DIR is where sbatch was run. Outside a
# job (bash workflow/run_slurm.sbatch ...) fall back to the script's own
# location. Either way no repository path is written down in this file --
# that is the bug in the CH2 controller (an absolute `cd` into its own
# checkout) and in the retired run_usa.sbatch, whose hard-coded REPO= path had
# stopped existing.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/workflow/run_slurm.sbatch" ]; then
    REPO="${PYPSA_REPO:-$SLURM_SUBMIT_DIR}"
else
    REPO="${PYPSA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi
if [ ! -f "$REPO/workflow/Snakefile" ]; then
    echo "ERROR: \$REPO=$REPO is not a pypsa-usa checkout (no workflow/Snakefile)." >&2
    echo "       Submit from the repository root, or set PYPSA_REPO." >&2
    exit 1
fi

# --- 1. Arguments ------------------------------------------------------------
EXECUTOR=cluster
if [ "${1:-}" = "--local" ] || [ "${1:-}" = "--monolithic" ]; then
    EXECUTOR=local
    shift
fi
CONFIGFILE="${1:-}"
if [ -z "$CONFIGFILE" ]; then
    echo "usage: sbatch workflow/run_slurm.sbatch [--local] <configfile> [targets/args...]" >&2
    exit 2
fi
shift
# Accept both `repo_data/config/x.yaml` (relative to workflow/) and a path
# relative to the repository root or an absolute one.
case "$CONFIGFILE" in
    /*) ;;
    *) [ -f "$REPO/workflow/$CONFIGFILE" ] || CONFIGFILE="$REPO/$CONFIGFILE" ;;
esac

TAG="${SLURM_JOB_ID:-local-$$}"
LOG_DIR="$REPO/logs/controller"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/ctrl-$TAG.log") 2>&1

echo "[ctrl] repo      : $REPO"
echo "[ctrl] config    : $CONFIGFILE"
echo "[ctrl] executor  : $EXECUTOR"
echo "[ctrl] job       : ${SLURM_JOB_ID:-<no slurm>} on $(hostname)"
echo "[ctrl] targets   : ${*:-<config default>}"

# --- 2. Never leave orphans --------------------------------------------------
# CH2's controllers were killed mid-solve and left queued children plus
# half-written networks behind; resolve_lowres_rem_gep.sbatch exists only to
# repair that damage. Two mechanisms, both scoped to THIS controller:
#   * the work runs under `setsid`, so its whole process group can be signalled
#     (snakemake's rule processes are not our direct children);
#   * every child Slurm job is named smk-$PYPSA_JOB_TAG-<rule> by the profile,
#     so scancel can match this controller's children and nobody else's.
export PYPSA_JOB_TAG="${PYPSA_JOB_TAG:-$TAG}"

cleanup() {
    local rc=$?
    trap - EXIT TERM INT
    echo "[ctrl] cleanup (exit $rc)"
    if [ -n "${WORK_PGID:-}" ]; then
        kill -TERM -- "-$WORK_PGID" 2>/dev/null || true
        sleep 10
        kill -KILL -- "-$WORK_PGID" 2>/dev/null || true
    fi
    if [ "$EXECUTOR" = cluster ] && command -v squeue >/dev/null 2>&1; then
        squeue --me -h -o '%i %j' \
            | awk -v p="smk-$PYPSA_JOB_TAG-" -v self="${SLURM_JOB_ID:-0}" \
                  '$1 != self && index($2, p) == 1 {print $1}' \
            | xargs -r scancel || true
    fi
    exit "$rc"
}
trap cleanup EXIT TERM INT

# --- 3. Environment ----------------------------------------------------------
# Module order matters: `ml load devel uv` pulls gcc/14.2.0 back in, so gcc 12
# must be loaded AFTER uv. GCC 14 hard-errors on incompatible-pointer-types,
# which breaks the datrie 0.8.2 source build (a snakemake dependency).
ml load devel "${PYPSA_UV_MODULE:-uv/0.10.8}" 2>/dev/null || true
ml load system git/2.45.1
ml load gcc/12.4.0
export CFLAGS="${CFLAGS:--Wno-error=incompatible-pointer-types}"
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH (ml load devel uv)" >&2; exit 1; }
echo "[ctrl] git : $(git --version)"
echo "[ctrl] uv  : $(uv --version)"

# Caches off the 15 GB NFS-backed $HOME.
PYPSA_CACHE_ROOT="${PYPSA_CACHE_ROOT:-${SCRATCH:?SCRATCH is unset}/pypsa-usa/cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PYPSA_CACHE_ROOT/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PYPSA_CACHE_ROOT/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PYPSA_CACHE_ROOT/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
export MPLBACKEND=Agg
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# CH2 left this as a COMMENT at the top of run_slurm.sh and relied on the
# ambient environment; export it, and fail loudly rather than four hours into
# the DAG at the first solve.
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/share/software/user/restricted/gurobi/11.0.2/licenses/gurobi.lic}"
if [ ! -r "$GRB_LICENSE_FILE" ]; then
    echo "WARNING: no readable Gurobi licence at $GRB_LICENSE_FILE;" >&2
    echo "         solve_network will fail unless the solver is HiGHS." >&2
fi

# sbatch defaults to --export=ALL, and the profile passes it explicitly, so
# everything exported here reaches every child job.
export PYPSA_SLURM_PARTITION="${PYPSA_SLURM_PARTITION:-serc}"
export PYPSA_SBATCH_EXTRA="${PYPSA_SBATCH_EXTRA:-}"

cd "$REPO/workflow"

# --- 4. Per-rule Slurm logs --------------------------------------------------
# sbatch does not create -o/-e directories, and a missing one kills the child
# at launch with no diagnostic. Create one per rule up front. On $SCRATCH, not
# on Oak and not in $HOME (org policy); 90-day purge is fine for job capture.
export PYPSA_SLURM_LOGDIR="${PYPSA_SLURM_LOGDIR:-$SCRATCH/pypsa-usa/slurm-logs/$PYPSA_JOB_TAG}"
if [ "$EXECUTOR" = cluster ]; then
    mkdir -p "$PYPSA_SLURM_LOGDIR"
    grep -hoE '^rule[[:space:]]+[A-Za-z0-9_]+' Snakefile rules/*.smk \
        | awk '{print $2}' | sort -u \
        | while read -r r; do mkdir -p "$PYPSA_SLURM_LOGDIR/$r"; done
    echo "[ctrl] child logs: $PYPSA_SLURM_LOGDIR/<rule>/<rule>-<jobid>.{out,err}"
fi

# --- 5. Stale locks ----------------------------------------------------------
# Left behind by a predecessor controller that Slurm killed.
if [ -d .snakemake/locks ] && [ -n "$(ls -A .snakemake/locks 2>/dev/null)" ]; then
    echo "[ctrl] clearing stale locks"
    ${PYPSA_SNAKEMAKE:-uv run snakemake} --unlock --configfile "$CONFIGFILE" || true
fi

# --- 6. Assemble the command -------------------------------------------------
# `uv run snakemake` rather than an absolute interpreter path: CH2's reason for
# the absolute path was that --cluster bakes the SUBMITTING interpreter into
# every generated jobscript. It does -- snakemake uses sys.executable -- but
# under uv that is $REPO/.venv/bin/python, which is already a stable absolute
# path, and --export=ALL carries the module environment to the child. Override
# with PYPSA_SNAKEMAKE if you want to pin it explicitly.
SMK=(${PYPSA_SNAKEMAKE:-uv run snakemake})

ARGS=(--configfile "$CONFIGFILE")
if [ "$EXECUTOR" = cluster ]; then
    ARGS+=(--profile snakemake_profiles/sherlock)
else
    # One node, no submission. -j from the cores this job actually holds.
    ARGS+=(-j "${SLURM_CPUS_PER_TASK:-4}"
           --scheduler greedy
           --rerun-triggers mtime
           --rerun-incomplete
           --keep-going
           --printshellcmds
           --latency-wait 30
           --default-resources mem_mb=8000 'walltime="02:00:00"')
fi

# --nolock: CH2's rule, stated exactly. It is safe ONLY when every concurrent
# controller writes to a DIFFERENT run.name, because that is what keeps two
# DAGs from touching the same output files -- the lock is the only other thing
# preventing it. run_slurm_all.sh checks that invariant across its config array
# and sets PYPSA_NOLOCK=1 itself. Never set it by hand to get past a lock left
# by a crashed job; section 5 above already removes those.
if [ "${PYPSA_NOLOCK:-0}" = "1" ]; then
    echo "[ctrl] --nolock: caller asserts every concurrent config has a distinct run.name"
    ARGS+=(--nolock)
fi

echo "[ctrl] ${SMK[*]} ${ARGS[*]} $*"

# --- 7. Run ------------------------------------------------------------------
# setsid: own process group, so cleanup() can take snakemake's grandchildren
# with it. `&` + `wait` rather than a foreground call, because bash only runs a
# trap between commands and a foreground child would swallow the pre-timeout
# TERM until it returned on its own.
status=0
setsid "${SMK[@]}" "${ARGS[@]}" "$@" &
WORK_PID=$!
WORK_PGID="$WORK_PID"
wait "$WORK_PID" || status=$?
WORK_PGID=""

echo "[ctrl] snakemake exited with status $status"

# --- 8. Persist --------------------------------------------------------------
# $SCRATCH purges after 90 days of inactivity; results belong on Oak.
if [ -n "${PYPSA_RESULTS_ARCHIVE:-}" ] && [ "$status" = 0 ]; then
    echo "[ctrl] rsync results/ -> $PYPSA_RESULTS_ARCHIVE"
    mkdir -p "$PYPSA_RESULTS_ARCHIVE"
    rsync -a results/ "$PYPSA_RESULTS_ARCHIVE/" || true
fi

exit "$status"
