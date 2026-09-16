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
# Mail on end or failure so completion reaches a human without polling. Slurm
# uses the submitting account's address when --mail-user is unset.
#SBATCH --mail-type=END,FAIL
#
# Slurm's own capture goes to a RELATIVE path, i.e. into the directory you
# submitted from, which by construction exists and is writable. An absolute
# site path here is a per-user hard-code that silently fails the job for anyone
# else. The driver additionally tees everything into <run_dir>/logs/, which is
# the copy that stays with the run.
#SBATCH -o slurm-eq-%j.out
#SBATCH -e slurm-eq-%j.err

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
# 3h is the emissions-UNCONSTRAINED twin and the default on both sides
# (paths.OPTS must agree, or bash and python mint different run ids). The
# national 1 Mt CO2 cap, EQ_OPTS=REM-3h, is OPT-IN: it looked infeasible at
# USA scale on 2026-09-01 and a failed solve aborts the harness, so the run
# that secures the comparison goes first.
export EQ_OPTS="${EQ_OPTS:-3h}"
export EQ_UNTIL="${EQ_UNTIL:-}"
export EQ_BASELINE_REF="${EQ_BASELINE_REF:-master-benchmark}"
EQ_PRONG="${EQ_PRONG:-2}"
EQ_JOBS="${EQ_JOBS:-8}"
# Per-SIDE snakemake cap. Two sides build sequentially inside one job, so a
# flat default can exceed the job's own --time and get the whole thing killed
# mid-rule instead of letting side one finish cleanly. Derive it from the time
# this job actually has left, minus a margin for the compare/table/figure tail,
# split across the two sides. Falls back to half of --time when Slurm cannot
# tell us.
_remaining_seconds() {
    local end now
    end="$(scontrol show job "${SLURM_JOB_ID:-}" 2>/dev/null \
        | tr ' ' '\n' | sed -n 's/^EndTime=//p' | head -1)" || true
    [ -n "$end" ] || return 1
    end="$(date -d "$end" +%s 2>/dev/null)" || return 1
    now="$(date +%s)"
    echo $(( end - now ))
}
if [ -z "${EQ_TIMEOUT:-}" ]; then
    if remaining="$(_remaining_seconds)" && [ "${remaining:-0}" -gt 3600 ]; then
        # 900 s reserved for compare + tables + figures after the second build.
        EQ_TIMEOUT=$(( (remaining - 900) / 2 ))
        echo "[driver] EQ_TIMEOUT derived from the job's remaining time: ${EQ_TIMEOUT}s per side"
    else
        # --time is 20:00:00 = 72000 s; same arithmetic as the derived branch.
        EQ_TIMEOUT=$(( (72000 - 900) / 2 ))
        echo "[driver] EQ_TIMEOUT could not be derived; using ${EQ_TIMEOUT}s per side"
    fi
fi
EQ_EXTRA_ARGS="${EQ_EXTRA_ARGS:-}"

# --- 2. Environment ----------------------------------------------------------
# Before anything that needs uv or git: the run id below is slugged by calling
# into the harness itself, so the interpreter has to exist first.
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

# --- 3. The run directory ----------------------------------------------------
# One run directory per run (plan D5). Slugged through the SAME rule python
# uses, by calling it, so the directory bash creates and the directory python
# writes into cannot diverge. A raw EQ_RUN_ID of "usa/p2 cap" would otherwise
# have bash mkdir a nested path while python wrote to "usa-p2-cap".
cd "$REPO"
_raw_run_id="${EQ_RUN_ID:-${EQ_INTERCONNECT}-p${EQ_PRONG}-${EQ_OPTS}-$(date +%Y%m%d-%H%M)}"
EQ_RUN_ID="$(uv run python -m tests.equivalence.paths --slug "$_raw_run_id")"
export EQ_RUN_ID
RUN_DIR="$REPO/workflow/results/equivalence/$EQ_RUN_ID"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

# From here on everything this driver prints is also kept with the run, so the
# run directory is self-contained even after the Slurm logs age out.
exec > >(tee -a "$LOG_DIR/driver-${SLURM_JOB_ID:-local}.log") 2>&1
TEE_PID=$!
echo "[driver] repo    : $REPO"
echo "[driver] run id  : $EQ_RUN_ID"
echo "[driver] run dir : $RUN_DIR"
echo "[driver] job     : ${SLURM_JOB_ID:-<no slurm>} on $(hostname)"

# --- 4. Never leave orphans --------------------------------------------------
# CH2's controllers were killed mid-solve and left half-written networks plus
# queued children behind; one of its sbatch scripts exists only to repair that
# damage. Kill our own process tree, and scancel any Slurm job this run tagged,
# on any exit path including the pre-timeout TERM.
cleanup() {
    local rc=$?
    trap - EXIT TERM INT
    echo "[driver] cleanup (exit $rc): stopping the run's process group"
    # The work runs under `setsid`, in its OWN process group, for exactly this
    # moment: snakemake forks rule processes that are not our direct children,
    # so signalling only $$'s children leaves the solver running until Slurm
    # kills the node. Signal the whole group by its negative pgid. The tee that
    # is writing this line is NOT in that group, so the log survives the kill.
    if [ -n "${WORK_PGID:-}" ]; then
        kill -TERM -- "-$WORK_PGID" 2>/dev/null || true
        sleep 10
        kill -KILL -- "-$WORK_PGID" 2>/dev/null || true
    fi
    if command -v squeue >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
        squeue --me -h -o '%i %j' \
            | awk -v tag="$EQ_RUN_ID" -v self="$SLURM_JOB_ID" '$1 != self && index($2, tag) {print $1}' \
            | xargs -r scancel || true
    fi
    exit "$rc"
}
trap cleanup EXIT TERM INT

# --- 5. Shared data + cutouts ------------------------------------------------
# 13 GB of inputs live once, on $GROUP_SCRATCH, and BOTH sides read them through
# symlinks: workflow/{data,cutouts} in the develop checkout point at the cache,
# and the baseline worktree's copies point at the develop checkout's
# (build.provision_baseline_worktree). Never copied per side.
CACHE="${EQ_CACHE:-$GROUP_SCRATCH/kamran/pypsa-usa-eq-cache}"
mkdir -p "$CACHE"

# Optional top-up from another checkout's data tree, covering BOTH data/ and
# cutouts/. UNSET BY DEFAULT and it should stay that way for a warm cache: the
# footprint-dependent directories are provenance-verified there, and a sync from
# a regional campaign once imported Texas-only nrel_exclusion artifacts into a
# USA run (job 41566193 built ERCOT profiles on BOTH sides). Those three
# directories are excluded even when EQ_STAGE_SRC is set; on a cold cache the
# workflow's own retrieve rules fetch them from the declared Zenodo records.
if [ -n "${EQ_STAGE_SRC:-}" ]; then
    echo "[driver] staging $EQ_STAGE_SRC -> $CACHE (footprint-dependent dirs excluded)"
    rsync -a --exclude 'nrel_exclusion/' --exclude 'godeeep/' --exclude 'zenodo/' \
        "$EQ_STAGE_SRC/data/" "$CACHE/data/"
    rsync -a "$EQ_STAGE_SRC/cutouts/" "$CACHE/cutouts/"
fi

# Provenance check on the footprint-dependent inputs. ABSENT is fine — a cold
# cache is a first run, and the workflow's own retrieve rules fetch these from
# the declared Zenodo records. PRESENT BUT TOO SMALL is fatal: that is a
# regional footprint sitting where a national one should be (Texas caps are
# ~68K, national is 400K+), which builds ERCOT profiles on BOTH sides and looks
# like a clean comparison. See docs/equivalence.md, "starting from an empty
# cache".
cold_cache=0
for f in caps_onwind_reference.nc caps_solar_reference.nc \
         avail_onwind_reference.nc avail_solar_reference.nc; do
    p="$CACHE/data/nrel_exclusion/derived/$f"
    if [ ! -e "$p" ]; then
        cold_cache=1
    elif [ "$(stat -c%s "$p")" -lt 400000 ]; then
        echo "ERROR: $p is $(stat -c%s "$p") bytes, far below the national size." >&2
        echo "       A regional footprint in the national cache builds the wrong" >&2
        echo "       profiles on BOTH sides. Remove it and let the retrieve rules" >&2
        echo "       refetch, or restage a verified copy." >&2
        exit 1
    fi
done
for f in godeeep/historical/solar_gen_cf_2012_compressed.nc \
         godeeep/historical/wind_gen_cf_2012_125m_compressed.nc; do
    [ -s "$CACHE/data/$f" ] || cold_cache=1
done
if [ "$cold_cache" = 1 ]; then
    echo "[driver] cold cache at $CACHE: some footprint-dependent inputs are absent."
    echo "[driver] the retrieve rules will fetch them; the FIRST run is much longer."
fi

for name in data cutouts; do
    tgt="$REPO/workflow/$name"
    if [ -e "$tgt" ] && [ ! -L "$tgt" ]; then
        echo "ERROR: $tgt exists and is not a symlink; refusing to replace" >&2
        exit 1
    fi
    ln -sfn "$CACHE/$name" "$tgt"
done

# --- 6. Clear stale snakemake locks ------------------------------------------
# Left behind by a killed predecessor job; best-effort on both sides.
if [ "$EQ_INTERCONNECT" = western ]; then _cfg_name=config.equivalence.yaml; else _cfg_name="config.equivalence-${EQ_INTERCONNECT}.yaml"; fi
for wf in "$REPO/workflow" "$REPO/.worktrees/master-benchmark/workflow"; do
    if [ -d "$wf/.snakemake/locks" ] && [ -n "$(ls -A "$wf/.snakemake/locks" 2>/dev/null)" ]; then
        echo "[driver] clearing stale locks in $wf"
        # --unlock still parses the Snakefile, and master's needs the harness config
        # (KeyError 'enable' in retrieve.smk without it, job 43616413): pass the
        # side's configfile, and fail loudly if the unlock itself fails.
        if [ "$wf" = "$REPO/workflow" ]; then _cfg="repo_data/config/$_cfg_name"; else _cfg="config/$_cfg_name"; fi
        (cd "$wf" && uv run snakemake --unlock --configfile "$_cfg") || { echo "ERROR: snakemake --unlock failed in $wf" >&2; exit 1; }
    fi
done

# --- 7. The run --------------------------------------------------------------
# run.py resolves the context, gates the config, writes run_meta.json, then
# builds develop and master SEQUENTIALLY and compares.
echo "[driver] prong=$EQ_PRONG interconnect=$EQ_INTERCONNECT simpl=$EQ_SIMPL clusters=$EQ_CLUSTERS opts=$EQ_OPTS until=${EQ_UNTIL:-<full>} jobs=$EQ_JOBS timeout=${EQ_TIMEOUT}s/side"
# `set -e` would abort before $? could be read, so capture it explicitly: the
# cleanup trap and the artifact listing below must run on a failed run too.
status=0
# setsid puts the whole run in its own process group so the cleanup trap can
# take snakemake's grandchildren down with it. `&` + `wait` rather than a plain
# foreground call, because bash only runs a trap between commands: a foreground
# child would swallow the pre-timeout TERM until it returned on its own.
# shellcheck disable=SC2086
setsid uv run python -m tests.equivalence.run \
    --prong "$EQ_PRONG" \
    --side both \
    --jobs "$EQ_JOBS" \
    --timeout "$EQ_TIMEOUT" \
    $EQ_EXTRA_ARGS &
WORK_PID=$!
WORK_PGID="$WORK_PID"   # setsid makes the child its own group leader
wait "$WORK_PID" || status=$?
WORK_PGID=""

echo "[driver] finished with status $status; artifacts in $RUN_DIR"
ls -1 "$RUN_DIR" || true
exit "$status"
