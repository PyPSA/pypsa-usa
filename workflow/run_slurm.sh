# SLURM specifications made in default.cluster.yaml & the individual rules
# export GRB_LICENSE_FILE=/share/software/user/restricted/gurobi/11.0.2/licenses/gurobi.lic⁠
# snakemake --cluster "sbatch -A {cluster.account} --mail-type ALL --mail-user {cluster.email}  -p {cluster.partition} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb} --time {cluster.walltime}" --cluster-config config/260128_config/config.cluster.yaml --jobs 20 --latency-wait 60 --rerun-incomplete --configfile config/260128_config/config.egs_conus_z134_nz2050_perfect.yaml


# snakemake resources/egs_conus_z134_nz2050_perfect/usa/profile_solar.nc \
#   --cluster "sbatch -A {cluster.account} --mail-type ALL --mail-user {cluster.email} -p {cluster.partition} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb} --time {cluster.walltime}" \
#   --cluster-config config/260128_config/config.cluster.yaml \
#   --jobs 1 --latency-wait 60 --rerun-incomplete \
#   --configfile config/260128_config/config.egs_conus_z134_nz2050_perfect.yaml \
#   --set-resources build_renewable_profiles:mem_mb=40000

# 134 node network, egs_conus_z134_nz2050_perfect
# snakemake --cluster "sbatch -A {cluster.account} --mail-type ALL --mail-user {cluster.email} -p {cluster.partition} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb} --time {cluster.walltime}" \
#     --cluster-config config/260128_config/config.cluster.yaml \
#     --jobs 40 --latency-wait 60 --rerun-incomplete \
#     --configfile config/260128_config/config.egs_conus_z134_nz2050_perfect.yaml \
#     --set-resources build_renewable_profiles:mem_mb=40000 \
#     --set-resources prepare_network:mem_mb=12000 \
#     --set-resources solve_network:mem_mb=500000 \
#     --set-resources plot_statistics:mem_mb=10000

# snakemake --cluster "sbatch -A {cluster.account} --mail-type ALL --mail-user {cluster.email} -p {cluster.partition} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb} --time {cluster.walltime}" \
#     --cluster-config config/260128_config/config.cluster.yaml \
#     --jobs 40 --latency-wait 60 --rerun-incomplete \
#     --configfile config/260128_config/config.egs_conus_z15_nz2050_perfect.yaml \
#     --set-resources build_renewable_profiles:mem_mb=40000 \
#     --set-resources prepare_network:mem_mb=12000 \
#     --set-resources solve_network:mem_mb=500000

# 134 node network, no_egs_conus_z134_nz2050_perfect
# snakemake --cluster "sbatch -A {cluster.account} --mail-type ALL --mail-user {cluster.email} -p {cluster.partition} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb} --time {cluster.walltime}" \
#    --cluster-config config/260128_config/config.cluster.yaml \
#    --jobs 40 --latency-wait 60 --rerun-incomplete \
#    --configfile config/260128_config/config.no_egs_conus_z134_nz2050_perfect.yaml \
#    --set-resources build_renewable_profiles:mem_mb=40000,prepare_network:mem_mb=12000 \
#    --set-resources solve_network:mem_mb=550000

# 134 node network, egs_conus_z134_bau_perfect
# snakemake --cluster "sbatch -A {cluster.account} --mail-type ALL --mail-user {cluster.email} -p {cluster.partition} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb} --time {cluster.walltime}" \
#     --cluster-config config/260128_config/config.cluster.yaml \
#     --jobs 40 --latency-wait 60 --rerun-incomplete \
#     --configfile config/260128_config/config.egs_conus_z134_bau_perfect.yaml \
#     --set-resources build_renewable_profiles:mem_mb=40000,prepare_network:mem_mb=12000 \
#     --set-resources solve_network:mem_mb=500000 \
#     --set-resources plot_statistics:mem_mb=10000

# 134 node network, no_egs_conus_z134_bau_perfect
# snakemake --cluster "sbatch -A {cluster.account} --mail-type ALL --mail-user {cluster.email} -p {cluster.partition} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb} --time {cluster.walltime}" \
#     --cluster-config config/260128_config/config.cluster.yaml \
#     --jobs 40 --latency-wait 60 --rerun-incomplete \
#     --configfile config/260128_config/config.no_egs_conus_z134_bau_perfect.yaml \
#     --set-resources build_renewable_profiles:mem_mb=40000,prepare_network:mem_mb=12000 \
#     --set-resources solve_network:mem_mb=500000 \
#     --set-resources plot_statistics:mem_mb=10000

# 134 node network, egs_conus_z134_nz2035_perfect
# snakemake --cluster "sbatch -A {cluster.account} --mail-type ALL --mail-user {cluster.email} -p {cluster.partition} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb} --time {cluster.walltime}" \
#     --cluster-config config/260128_config/config.cluster.yaml \
#     --jobs 40 --latency-wait 60 --rerun-incomplete \
#     --configfile config/260128_config/config.egs_conus_z134_nz2035_perfect.yaml \
#     --set-resources build_renewable_profiles:mem_mb=40000,prepare_network:mem_mb=12000 \
#     --set-resources solve_network:mem_mb=500000 \
#     --set-resources plot_statistics:mem_mb=10000

# # MISO, county
# snakemake --cluster "sbatch -A {cluster.account} --mail-type ALL --mail-user {cluster.email} -p {cluster.partition} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb} --time {cluster.walltime}" \
#     --cluster-config config/260128_config/config.cluster.yaml \
#     --jobs 40 --latency-wait 60 --rerun-incomplete \
#     --configfile config/260128_config/config.egs_miso_county_nz2050_perfect.yaml \
#     --set-resources build_renewable_profiles:mem_mb=40000 \
#     --set-resources add_extra_components:mem_mb=80000 \
#     --set-resources solve_network:mem_mb=900000

## WECC, county
# snakemake --cluster "sbatch -A {cluster.account} --mail-type ALL --mail-user {cluster.email} -p {cluster.partition} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb} --time {cluster.walltime}" \
#     --cluster-config config/260128_config/config.cluster.yaml \
#     --jobs 40 --latency-wait 60 --rerun-incomplete \
#     --configfile config/260128_config/config.egs_wecc_county_nz2050_perfect.yaml \
#     --set-resources build_renewable_profiles:mem_mb=40000 \
#     --set-resources add_extra_components:mem_mb=80000 \
#     --set-resources solve_network:mem_mb=600000

# ## ERCOT
# snakemake --cluster "sbatch -A {cluster.account} --mail-type ALL --mail-user {cluster.email} -p {cluster.partition} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb} --time {cluster.walltime}" \
#      --cluster-config config/260128_config/config.cluster.yaml \
#      --jobs 40 --latency-wait 60 --rerun-incomplete \
#      --configfile config/260128_config/config.egs_ercot_county_nz2050_perfect.yaml \
#      --set-resources solve_network:mem_mb=400000
    # --set-resources plot_statistics:mem_mb=10000

#Test EGS Western 33 node network
# snakemake --cluster "sbatch -A {cluster.account} --mail-type ALL --mail-user {cluster.email} -p {cluster.partition} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb} --time {cluster.walltime}" \
#     --cluster-config config/config.cluster.yaml \
#     --jobs 40 --latency-wait 60 --rerun-incomplete \
#     --configfile config/config.egs_western.yaml \
#     --set-resources build_renewable_profiles:mem_mb=40000 

# 33 node network, egs_western_z33_nz2035_perfect
snakemake --cluster "sbatch -A {cluster.account} --mail-type ALL --mail-user {cluster.email} -p {cluster.partition} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb} --time {cluster.walltime}" \
    --cluster-config config/260128_config/config.cluster.yaml \
    --jobs 40 --latency-wait 60 --rerun-incomplete \
    --configfile config/260128_config/config.egs_western_z33_nz2035_perfect.yaml \
    --set-resources build_renewable_profiles:mem_mb=40000,prepare_network:mem_mb=12000 \
    --set-resources solve_network:mem_mb=300000 


# snakemake --cluster "sbatch -A {cluster.account} --mail-type ALL --mail-user {cluster.email} -p {cluster.partition} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb} --time {cluster.walltime}" \
#     --cluster-config config/260128_config/config.cluster.yaml \
#     --jobs 40 --latency-wait 60 --rerun-incomplete \
#     --configfile config/260128_config/config.egs_western_z33_atlite_nz2035_perfect.yaml --unlock