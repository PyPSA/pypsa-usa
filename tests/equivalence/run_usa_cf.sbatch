#!/bin/bash
# USA-scale Tier C equivalence run, CF-comparison variant: prong 2
# (simpl=300), EQ_UNTIL=assembled — builds both sides through
# build_renewable_profiles / demand / add_electricity (candidate) and
# simplify_network (anchor) and compares the demand + profile artifacts.
# No prepare/solve: this grounds the master-vs-develop renewable-CF
# construction comparison in data without paying for the full-year solve.
#
# First run of the historical-2012 + full-year-snapshots configuration
# (2026-09-01): GODEEEP CFs come from the Zenodo-published historical 2012
# records (pre-staged into the scratch cache), DL-17 leap-day patch active.
#
# Submit from the repo root:  sbatch tests/equivalence/run_usa_cf.sbatch
#
#SBATCH -J eq-usa-cf
#SBATCH -p normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160GB
#SBATCH -o /scratch/groups/iazevedo/kamran/eq-usa/%x-%j.out
#SBATCH -e /scratch/groups/iazevedo/kamran/eq-usa/%x-%j.err

set -euo pipefail

REPO=/oak/stanford/groups/iazevedo/kamran/refactor/pypsa-usa
CH2_WF=/oak/stanford/groups/iazevedo/kamran/CH2/pypsa-usa/workflow
# Campaign-scoped cache (2026-09-01). The old shared pypsa-usa-cache staged
# CH2's data/ wholesale, which silently imported ANOTHER campaign's
# Texas-only nrel_exclusion/derived artifacts into a USA run (job 41566193
# built ERCOT profiles on both sides). This cache holds only provenance-
# verified inputs for footprint-dependent data:
#   - nrel_exclusion/derived: national artifacts, md5-verified vs Zenodo
#     record 20316475 (17,890 substations, 292/300 s300 clusters)
#   - godeeep+zenodo/historical: 2012 CFs, md5-verified vs records
#     20127513 / 20127520
# The rsync below excludes those dirs so a CH2 sync can never overwrite them.
CACHE="$GROUP_SCRATCH/kamran/pypsa-usa-eq-cache"

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

# System git (1.8, CentOS 7) predates `git worktree`, which anchor
# provisioning needs.
ml load system git/2.45.1

# --- 1. Stage data + cutouts Oak -> scratch (idempotent resume via rsync) ---
# Footprint-dependent dirs are EXCLUDED: they are pre-verified in the cache
# (see header) and must never be overwritten by whatever CH2 happens to hold.
mkdir -p "$CACHE"
rsync -a --exclude 'nrel_exclusion/' --exclude 'godeeep/' --exclude 'zenodo/' \
    "$CH2_WF/data/" "$CACHE/data/"
rsync -a "$CH2_WF/cutouts/" "$CACHE/cutouts/"

# Fail fast if the verified inputs are missing or the wrong footprint
# (Texas caps ≈ 68K; national ≈ 1.4M+).
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

# --- 2. Harness, assembled stage only ---------------------------------------
cd "$REPO"

# Clear stale snakemake locks left by a killed predecessor job (best-effort).
for wf in "$REPO/workflow" "$REPO/.worktrees/anchor-e7f8bd70/workflow"; do
    if [ -d "$wf/.snakemake/locks" ] && [ -n "$(ls -A "$wf/.snakemake/locks" 2>/dev/null)" ]; then
        (cd "$wf" && uv run snakemake --unlock) || true
    fi
done
export EQ_INTERCONNECT=usa
export EQ_SIMPL=300
export EQ_CLUSTERS=134a
export EQ_UNTIL=assembled
export EQ_OPTS=3h   # unused before prepare_network; kept for target naming

uv run python -m tests.equivalence.run \
    --prong 2 --side both --jobs 8 --timeout 82800

dst="$REPO/workflow/results/equivalence/usa-cf-assembled"
mkdir -p "$dst"
cp -p "$REPO"/workflow/results/equivalence/manifest_*_usa.json "$dst"/ 2>/dev/null || true
cp -p "$REPO"/workflow/results/equivalence/findings_*.json "$dst"/ 2>/dev/null || true
cp -p "$REPO"/workflow/results/equivalence/*.html "$dst"/ 2>/dev/null || true
