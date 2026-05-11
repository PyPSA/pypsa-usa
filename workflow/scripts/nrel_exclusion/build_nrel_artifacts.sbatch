#!/bin/bash
#SBATCH -A iazevedo
#SBATCH -p serc
#SBATCH --job-name=nrel-artifacts
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -o logs/run_slurm/nrel-artifacts-%j.out
#SBATCH -e logs/run_slurm/nrel-artifacts-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=asiaz@stanford.edu

set -euo pipefail
cd /home/groups/iazevedo/asia/pypsa-usa/workflow

# Pass CLI args through (e.g. sbatch ...sbatch limited), and env overrides
# (TECHS, APPLY_CEC, APPLY_BOEM, etc) via --export.
bash scripts/nrel_exclusion/build_nrel_artifacts.sh "$@"
