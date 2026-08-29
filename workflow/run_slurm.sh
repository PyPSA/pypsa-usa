# SLURM specifications live in config/config.slurm.yaml & the individual rules
# GRB_LICENSE_FILE=/share/software/user/restricted/gurobi/11.0.2/licenses/gurobi.lic⁠
snakemake --cluster "sbatch -A {cluster.account} --mail-type ALL --mail-user {cluster.email}  -p {cluster.partition} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb} --time {resources.walltime}" --cluster-config config/config.slurm.yaml --jobs 20 --latency-wait 60 --rerun-incomplete --configfile config/CH1/config.tamu.single_horizon.bau.yaml
