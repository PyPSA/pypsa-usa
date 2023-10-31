#Bash Script for running snakemake with cluster
#Load Modules and Python Packages
ml python/3.9.0
ml gurobi/10.0.1_py39

#Run Script
snakemake -j6 --cluster config/config.cluster.yaml
