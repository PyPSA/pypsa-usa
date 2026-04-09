#!/bin/bash
#SBATCH -A iazevedo
#SBATCH -p service
#SBATCH --job-name=smk-driver
#SBATCH --time=01-23:59:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH -o logs/run_slurm/smk-driver-%j.out
#SBATCH -e logs/run_slurm/smk-driver-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hdaniels@stanford.edu

set -euo pipefail
cd /oak/stanford/groups/iazevedo/henry/pypsa-usa/workflow

export GRB_LICENSE_FILE=/share/software/user/restricted/gurobi/11.0.2/licenses/gurobi.lic

/home/groups/iazevedo/henry/miniforge3/bin/mamba run -n pypsa-usa bash run_slurm.sh
