#!/bin/bash 
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --output log/snake-%j.out

# activate conda in general
# source /home/ntpierce/.bashrc # if you have the conda init setting

# activate a specific conda environment, if you so choose
conda activate pypsa-usa

# go to a particular directory
cd $GROUP_HOME/kamran/pypsa-usa/workflow 

# make things fail on errors
set -o nounset
set -o errexit
set -x

### run your commands here!
snakemake --cluster "sbatch -t 0:30:00 -N 1 -c 14 --mem=30gb " --jobs 5 --latency-wait 60
