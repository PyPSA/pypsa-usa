#!/bin/bash
# USA-scale Tier C equivalence run (prong 2: simpl=300 -> 134 ReEDS zones,
# full chain through solve_network) + per-rule benchmark capture for the
# resource-request re-fit (upstream issue #808).
#
# Submit from the repo root:  sbatch tests/equivalence/run_usa.sbatch
#
# What it does:
#   1. Stages the shared data/ and cutouts/ caches from the CH2 checkout on
#      Oak to $GROUP_SCRATCH (jobs must not do heavy I/O against Oak), then
#      points workflow/data and workflow/cutouts at the staged copies.
#   2. Runs the harness twice, candidate + anchor each time:
#        run A  opts REM-3h : national 1,000,000 t CO2 cap
#                             (regional_Co2_limits_equivalence_usa.csv)
#        run B  opts 3h     : emissions-unconstrained twin
#      Run B reuses every artifact up to prepare_network; only the prepared
#      and solved networks rebuild. Manifests + HTML report are archived per
#      run under workflow/results/equivalence/usa-{rem1mt,noco2}/.
#
#SBATCH -J eq-usa-p2
#SBATCH -p normal
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160GB
#SBATCH -o /scratch/groups/iazevedo/kamran/eq-usa/%x-%j.out
#SBATCH -e /scratch/groups/iazevedo/kamran/eq-usa/%x-%j.err

set -euo pipefail

REPO=/oak/stanford/groups/iazevedo/kamran/refactor/pypsa-usa
CH2_WF=/oak/stanford/groups/iazevedo/kamran/CH2/pypsa-usa/workflow
CACHE="$GROUP_SCRATCH/kamran/pypsa-usa-cache"

# Keep uv/pip/matplotlib caches off $HOME (15 GB NFS quota).
export UV_CACHE_DIR="$GROUP_SCRATCH/kamran/cache/uv"
export PIP_CACHE_DIR="$GROUP_SCRATCH/kamran/cache/pip"
export XDG_CACHE_HOME="$GROUP_SCRATCH/kamran/cache/xdg"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"

export GRB_LICENSE_FILE=/share/software/user/restricted/gurobi/11.0.2/licenses/gurobi.lic

# GCC 14 (cluster default) hard-errors on incompatible-pointer-types, which
# breaks the datrie 0.8.2 source build (snakemake dep). Use gcc 12 and keep
# the warning non-fatal as a fallback.
ml load gcc/12.4.0
export CFLAGS="-Wno-error=incompatible-pointer-types"

# --- 1. Stage data + cutouts Oak -> scratch (idempotent resume via rsync) ---
mkdir -p "$CACHE"
rsync -a "$CH2_WF/data/" "$CACHE/data/"
rsync -a "$CH2_WF/cutouts/" "$CACHE/cutouts/"

for name in data cutouts; do
    tgt="$REPO/workflow/$name"
    if [ -e "$tgt" ] && [ ! -L "$tgt" ]; then
        echo "ERROR: $tgt exists and is not a symlink; refusing to replace" >&2
        exit 1
    fi
    ln -sfn "$CACHE/$name" "$tgt"
done

# --- 2. Harness, both opts variants -----------------------------------------
cd "$REPO"
export EQ_INTERCONNECT=usa
export EQ_SIMPL=300
export EQ_CLUSTERS=134a

archive() { # archive <label>: keep this run's manifests + report
    dst="$REPO/workflow/results/equivalence/usa-$1"
    mkdir -p "$dst"
    cp -p "$REPO"/workflow/results/equivalence/manifest_*_usa.json "$dst"/ 2>/dev/null || true
    cp -p "$REPO"/workflow/results/equivalence/findings_*.json "$dst"/ 2>/dev/null || true
    cp -p "$REPO"/workflow/results/equivalence/*.html "$dst"/ 2>/dev/null || true
}

# Unconstrained run FIRST: the REM 1 Mt cap looks infeasible at USA scale
# (barrier verdict + stalled disambiguation simplex, 2026-09-01), and a failed
# solve aborts the harness — ordering this way secures the anchor build,
# comparison, and benchmark data before the REM question is settled.
EQ_OPTS=3h uv run python -m tests.equivalence.run \
    --prong 2 --side both --jobs 8 --timeout 43200
archive noco2

EQ_OPTS=REM-3h uv run python -m tests.equivalence.run \
    --prong 2 --side both --jobs 8 --timeout 21600
archive rem1mt
