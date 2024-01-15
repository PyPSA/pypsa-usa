# #!/bin/bash
# #SBATCH -t 01:00:00
# #SBATCH -p serc
# #SBATCH -N 1
# #SBATCH -cpus-per-task=1
# #SBATCH --output log/snake-%j.out

# # go to a particular directory
# cd $GROUP_HOME/kamran/pypsa-usa/workflow 

# # make things fail on errors
# set -o nounset
# set -o errexit
# set -x

### run your commands here!
snakemake --cluster "sbatch -A {cluster.account} -p {cluster.partition} -t {cluster.time} -o {cluster.output} -e {cluster.error}" --cluster-config config/config.cluster.yaml --jobs 8 --latency-wait 10
