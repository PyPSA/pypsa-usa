#Bash Script for running snakemake with cluster
#Load Modules and Python Packages
ml python/3.9.0
ml gurobi/10.0.1_py39

#Run Script
# snakemake -j6 --cluster config/config.cluster.yaml

# snakemake -j6 --cluster "sbatch --time={resources.time_min} --mem={resources.mem_mb} --partition=serc --account=iazevedo -c {resources.cpus} -o logs/slurm/{rule}_{wildcards} -e logs/slurm/{rule}_{wildcards}"

# snakemake -j6 --cluster "sbatch --partition=serc --account=iazevedo -c 1 -o logs/slurm/{rule}_{wildcards} -e logs/slurm/{rule}_{wildcards}"

snakemake -j6 --cluster "sbatch --time={resources.time_min} --partition=serc --account=iazevedo -o logs/slurm/{rule}_{wildcards} -e logs/slurm/{rule}_{wildcards}" --latency-wait 60