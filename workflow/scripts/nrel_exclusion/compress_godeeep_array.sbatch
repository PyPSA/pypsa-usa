#!/bin/bash
#SBATCH -A iazevedo
#SBATCH -p serc
#SBATCH --job-name=godeeep-compress
#SBATCH --array=1-129%16
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH -o logs/run_slurm/godeeep-compress-%A_%a.out
#SBATCH -e logs/run_slurm/godeeep-compress-%A_%a.err

# Sweep the GODEEEP capacity-factor compression: one array task per tasks.tsv row.
#
# The default array range (1-129) matches the full registry: 3 techs
# (solar, wind_125m, wind_100m) x 43 years (1980-2022). Regenerate the task list
# first and use the range it prints:
#
#   cd workflow
#   mkdir -p logs/run_slurm
#   python scripts/nrel_exclusion/make_compress_tasks.py \
#       --output /oak/stanford/groups/iazevedo/GoDEEEP_Capacity_Factors_compressed/tasks.tsv
#   sbatch scripts/nrel_exclusion/compress_godeeep_array.sbatch
#
# Tasks are idempotent (sha256 sidecar skip), so re-submitting the whole array
# after a partial sweep only rebuilds what is missing or corrupt. Extra flags are
# passed through to compress_godeeep_task.sh:
#
#   sbatch scripts/nrel_exclusion/compress_godeeep_array.sbatch --dest-root /path/to/registry
#
# Submit from workflow/ (log paths are relative to $SLURM_SUBMIT_DIR) or set
# WORKFLOW_DIR. Not part of the Snakemake DAG — the registry is built once, out of
# band, then consumed read-only.

set -euo pipefail
umask 0002

WORKFLOW_DIR="${WORKFLOW_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$WORKFLOW_DIR"

CONDA_SH="${CONDA_SH:-/home/groups/iazevedo/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-pypsa-usa}"
set +u  # conda's activation hooks trip over `set -u`
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

# Keep BLAS/HDF5 from oversubscribing the shared node.
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

bash scripts/nrel_exclusion/compress_godeeep_task.sh \
    "${SLURM_ARRAY_TASK_ID:?run under sbatch --array, or pass an index to compress_godeeep_task.sh directly}" \
    "$@"
