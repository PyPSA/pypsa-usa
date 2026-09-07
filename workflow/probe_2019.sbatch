#!/bin/bash
# Probe run: weather year 2019, both resolutions, purely to collect per-rule
# resource data before launching the remaining 45 overlays.
#
#   sbatch probe_2019.sbatch
#
# This is the DRIVER job. It holds two `snakemake --cluster` processes, which
# submit each rule as its own Slurm job. Its own CPU/memory needs are tiny; its
# walltime must cover the whole build span including queue wait for rule jobs.
#SBATCH -p serc
#SBATCH -J ca2040_probe
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -o logs/drivers/probe-%j.out
#SBATCH -e logs/drivers/probe-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ctehran@stanford.edu

set -uo pipefail
cd /oak/stanford/groups/iazevedo/kamran/CH3/pypsa-usa/workflow

export PARTITION=serc
export EMAIL=ctehran@stanford.edu
export JOBS=20          # concurrent Slurm rule-jobs per overlay
export CONCURRENCY=2    # both overlays at once
export RESTART_TIMES=0  # a retry would re-run with attempt*mem and skew MaxRSS

echo "=== probe start $(date -Is) on $(hostname) ==="
bash run_slurm.sh ca2040_wy2019_z4.yaml ca2040_wy2019_county.yaml
rc=$?
echo "=== probe end $(date -Is) rc=$rc ==="
exit $rc
