# SLURM specifications made in default.cluster.yaml & the individual rules
snakemake --cluster "sbatch -A {cluster.account} -p {cluster.partition} -t {cluster.time} -o {cluster.output} -e {cluster.error} --mem {resources.mem_mb}" --cluster-config config/config.cluster.yaml --jobs 9 --latency-wait 10
