# SLURM specifications made in default.cluster.yaml & the individual rules
# GRB_LICENSE_FILE=/share/software/user/restricted/gurobi/11.0.2/licenses/gurobi.lic⁠
snakemake --cluster "sbatch -A {cluster.account} --mail-type ALL --mail-user {cluster.email} -p {cluster.partition} -t {cluster.walltime} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb}" --cluster-config config/config.cluster.yaml --jobs 10 --latency-wait 60 --configfile config/config.default.yaml --rerun-incomplete
