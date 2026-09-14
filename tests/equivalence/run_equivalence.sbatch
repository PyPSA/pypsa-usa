#!/bin/bash
# One parameterised Sherlock driver for a Tier C equivalence run.
#
#   sbatch tests/equivalence/run_equivalence.sbatch
#   EQ_UNTIL=assembled EQ_PRONG=2 sbatch tests/equivalence/run_equivalence.sbatch
#
# Replaces run_usa.sbatch and run_usa_cf.sbatch, which differed only in env
# knobs and both hard-coded a repository path that no longer exists.
#
# Mode: MONOLITHIC — one job builds both sides, one after the other, with
# Snakemake running locally inside it (no --cluster). That is deliberate for an
# equivalence run: per-rule submission would put the two arms in different
# environments and add scheduler nondeterminism to the thing being measured,
# and both sides share one data/ cache, so two concurrent DAGs would race on
# the same retrieve targets. Never run two of these at once against one cache.
#
# Everything is an env knob with a default; nothing below is a path you have to
# edit. The repository root is resolved from where you submitted from.
#
#SBATCH -J eq
#SBATCH -p serc
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160GB
#SBATCH --signal=B:TERM@180
#SBATCH -o /scratch/groups/iazevedo/kamran/eq-usa/%x-%j.out
#SBATCH -e /scratch/groups/iazevedo/kamran/eq-usa/%x-%j.err

set -euo pipefail

# --- 0. Where am I? ----------------------------------------------------------
# sbatch spools this script to node-local storage, so $BASH_SOURCE is useless
# inside a job; $SLURM_SUBMIT_DIR is where `sbatch` was run. Outside a job
# (bash tests/equivalence/run_equivalence.sbatch) fall back to the script's own
# location. Either way, no repository path is written down here.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/tests/equivalence/run_equivalence.sbatch" ]; then
    REPO="${EQ_REPO:-$SLURM_SUBMIT_DIR}"
else
    REPO="${EQ_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
fi
if [ ! -f "$REPO/tests/equivalence/run.py" ]; then
    echo "ERROR: \$REPO=$REPO is not a pypsa-usa checkout (no tests/equivalence/run.py)." >&2
    echo "       Submit from the repository root, or set EQ_REPO." >&2
    exit 1
fi

# --- 1. Run parameters -------------------------------------------------------
export EQ_INTERCONNECT="${EQ_INTERCONNECT:-usa}"
export EQ_SIMPL="${EQ_SIMPL:-300}"
export EQ_CLUSTERS="${EQ_CLUSTERS:-134}"
export EQ_OPTS="${EQ_OPTS:-3h}"
export EQ_UNTIL="${EQ_UNTIL:-}"
export EQ_BASELINE_REF="${EQ_BASELINE_REF:-master-benchmark}"
EQ_PRONG="${EQ_PRONG:-2}"
EQ_JOBS="${EQ_JOBS:-8}"
EQ_TIMEOUT="${EQ_TIMEOUT:-43200}"
EQ_EXTRA_ARGS="${EQ_EXTRA_ARGS:-}"

# One run directory per run (plan D5). Minted here and exported so bash and
# python resolve the SAME directory; tests/equivalence/paths.run_id() honours
# EQ_RUN_ID over its own default.
export EQ_RUN_ID="${EQ_RUN_ID:-${EQ_INTERCONNECT}-p${EQ_PRONG}-${EQ_OPTS}-$(date +%Y%m%d-%H%M)}"
RUN_DIR="$REPO/workflow/results/equivalence/$EQ_RUN_ID"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

# From here on everything this driver prints is also kept with the run, so the
# run directory is self-contained even after the $SCRATCH slurm logs age out.
exec > >(tee -a "$LOG_DIR/driver-${SLURM_JOB_ID:-local}.log") 2>&1
TEE_PID=$!
echo "[driver] repo    : $REPO"
echo "[driver] run id  : $EQ_RUN_ID"
echo "[driver] run dir : $RUN_DIR"
echo "[driver] job     : ${SLURM_JOB_ID:-<no slurm>} on $(hostname)"

# --- 2. Never leave orphans --------------------------------------------------
# CH2's controllers were killed mid-solve and left half-written networks plus
# queued children behind; one of its sbatch scripts exists only to repair that
# damage. Kill our own process tree, and scancel any Slurm job this run tagged,
# on any exit path including the pre-timeout TERM.
cleanup() {
    local rc=$?
    trap - EXIT TERM INT
    echo "[driver] cleanup (exit $rc): stopping children"
    # Everything we started except the tee that is writing this line.
    for pid in $(pgrep -P $$ 2>/dev/null || true); do
        [ "$pid" = "${TEE_PID:-}" ] && continue
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 5
    for pid in $(pgrep -P $$ 2>/dev/null || true); do
        [ "$pid" = "${TEE_PID:-}" ] && continue
        kill -KILL "$pid" 2>/dev/null || true
    done
    if command -v squeue >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
        squeue --me -h -o '%i %j' \
            | awk -v tag="$EQ_RUN_ID" -v self="$SLURM_JOB_ID" '$1 != self && index($2, tag) {print $1}' \
            | xargs -r scancel || true
    fi
    exit "$rc"
}
trap cleanup EXIT TERM INT

# --- 3. Environment ----------------------------------------------------------
# Caches off $HOME (15 GB, NFS-backed): uv, pip, matplotlib and everything else
# that defaults there.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$GROUP_SCRATCH/kamran/cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$GROUP_SCRATCH/kamran/cache/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$GROUP_SCRATCH/kamran/cache/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
export MPLBACKEND=Agg
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/share/software/user/restricted/gurobi/11.0.2/licenses/gurobi.lic}"

# GCC 14 (cluster default) hard-errors on incompatible-pointer-types, which
# breaks the datrie 0.8.2 source build (a snakemake dep). Use gcc 12 and keep
# the warning non-fatal as a fallback.
ml load gcc/12.4.0
export CFLAGS="${CFLAGS:--Wno-error=incompatible-pointer-types}"

# System git (1.8, CentOS 7) predates `git worktree`, which baseline
# provisioning needs.
ml load system git/2.45.1

# uv is a module here, not a $HOME install.
ml load devel "${EQ_UV_MODULE:-uv/0.10.8}" 2>/dev/null || true
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH (ml load devel uv)" >&2; exit 1; }

echo "[driver] git : $(git --version)"
echo "[driver] uv  : $(uv --version)"

# --- 4. Shared data + cutouts ------------------------------------------------
# 13 GB of inputs live once, on $GROUP_SCRATCH, and BOTH sides read them through
# symlinks: workflow/{data,cutouts} in the develop checkout point at the cache,
# and the baseline worktree's copies point at the develop checkout's
# (build.provision_baseline_worktree). Never copied per side.
CACHE="${EQ_CACHE:-$GROUP_SCRATCH/kamran/pypsa-usa-eq-cache}"
mkdir -p "$CACHE"

# Optional top-up from another checkout's data tree. UNSET BY DEFAULT and it
# should stay that way: the footprint-dependent directories below are
# provenance-verified in the cache, and a sync from a regional campaign once
# imported Texas-only nrel_exclusion artifacts into a USA run (job 41566193
# built ERCOT profiles on BOTH sides). They are excluded even when it is set.
if [ -n "${EQ_STAGE_SRC:-}" ]; then
    echo "[driver] staging $EQ_STAGE_SRC -> $CACHE (footprint-dependent dirs excluded)"
    rsync -a --exclude 'nrel_exclusion/' --exclude 'godeeep/' --exclude 'zenodo/' \
        "$EQ_STAGE_SRC/data/" "$CACHE/data/"
    rsync -a "$EQ_STAGE_SRC/cutouts/" "$CACHE/cutouts/"
fi

# Fail fast if the verified inputs are missing or are the wrong footprint
# (Texas caps are ~68K; national is 400K+).
for f in caps_onwind_reference.nc caps_solar_reference.nc \
         avail_onwind_reference.nc avail_solar_reference.nc; do
    p="$CACHE/data/nrel_exclusion/derived/$f"
    if [ ! -s "$p" ] || [ "$(stat -c%s "$p")" -lt 400000 ]; then
        echo "ERROR: $p missing or suspiciously small (regional footprint?)" >&2
        exit 1
    fi
done
for f in godeeep/historical/solar_gen_cf_2012_compressed.nc \
         godeeep/historical/wind_gen_cf_2012_125m_compressed.nc; do
    if [ ! -s "$CACHE/data/$f" ]; then
        echo "ERROR: $CACHE/data/$f missing" >&2
        exit 1
    fi
done

for name in data cutouts; do
    tgt="$REPO/workflow/$name"
    if [ -e "$tgt" ] && [ ! -L "$tgt" ]; then
        echo "ERROR: $tgt exists and is not a symlink; refusing to replace" >&2
        exit 1
    fi
    ln -sfn "$CACHE/$name" "$tgt"
done

# --- 5. Clear stale snakemake locks ------------------------------------------
# Left behind by a killed predecessor job; best-effort on both sides.
cd "$REPO"
for wf in "$REPO/workflow" "$REPO/.worktrees/master-benchmark/workflow"; do
    if [ -d "$wf/.snakemake/locks" ] && [ -n "$(ls -A "$wf/.snakemake/locks" 2>/dev/null)" ]; then
        echo "[driver] clearing stale locks in $wf"
        (cd "$wf" && uv run snakemake --unlock) || true
    fi
done

# --- 6. The run --------------------------------------------------------------
# run.py resolves the context, gates the config, writes run_meta.json, then
# builds develop and master SEQUENTIALLY and compares.
echo "[driver] prong=$EQ_PRONG interconnect=$EQ_INTERCONNECT simpl=$EQ_SIMPL clusters=$EQ_CLUSTERS opts=$EQ_OPTS until=${EQ_UNTIL:-<full>}"
# `set -e` would abort before $? could be read, so capture it explicitly: the
# cleanup trap and the artifact listing below must run on a failed run too.
status=0
# shellcheck disable=SC2086
uv run python -m tests.equivalence.run \
    --prong "$EQ_PRONG" \
    --side both \
    --jobs "$EQ_JOBS" \
    --timeout "$EQ_TIMEOUT" \
    $EQ_EXTRA_ARGS || status=$?

echo "[driver] finished with status $status; artifacts in $RUN_DIR"
ls -1 "$RUN_DIR" || true
exit "$status"
