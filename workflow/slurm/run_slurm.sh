#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -p serc
#SBATCH -N 1
#SBATCH --output log/snake-%j.out

# go to a particular directory
cd $GROUP_HOME/kamran/pypsa-usa/workflow 

# make things fail on errors
set -o nounset
set -o errexit
set -x

### run your commands here!
snakemake --cluster "sbatch -t 0:30:00 -N 1 -c 14 --mem=30gb " --jobs 8 --latency-wait 10
