(tutorial)=
# Tutorial

```{note}
**If you have not done so, please follow the [installation instructions](https://pypsa-usa.readthedocs.io/en/latest/about-install.html) [github issues](https://github.com/PyPSA/pypsa-usa/issues)**
```

## Set Configuration

To start, you'll want to set the proper network configuration for your studies purpose. The default configuration in `config/config.default.yaml` using the `western` interconnect and 30 nodes is a good place to start!

You can find more information on each configuration setting on the [configurations page](https://pypsa-usa.readthedocs.io/en/latest/config-configuration.html).


## Run workflow

To run the workflow, `cd` into the `workflow` directory and run the `snakemake` from your terminal with your selection of config file:

```bash
snakemake -j1 --configfile config/config.default.yaml
```

where 1 indicates the number of cores used.

## Running on HPC Cluster

If you are running the workflow on an High-Performance Compute (HPC) cluster, you will first need to update the configuration settings in `config.cluster.yaml`. Update the account, partition, email, and chdir fields to match the information of your institutions cluster.

Next, identify the name of the configuration file you would like to run by editing the `run_slurm.sh` script. The default value is the `--configfile config/config.default.yaml`.

To run, open a terminal within a login node of your cluster and run the script included in the `workflow` directory:

```bash
bash run_slurm.sh
```

We have included settings in the Snakemake workflow to dynamically request reasources from an HPC cluster based on the size of the pypsa-usa model you decide to run. To modify these resource selections checkout the `memory` and `threads` fields in individual snakemake rules.

## Examine Results

Result plots and images are automatically built in the `workflow/results` folder. To further analyze the results of a solved network, you can use pypsa to analyze the `elec_s_{clusters}_ec_l{l}_{opts}.nc` file in the `results/{interconnect}/networks/` folder. (Tutorial juyper notebook is on the way!)

(troubleshooting)=
## Troubleshooting:

To force the execution of a portion of the workflow up to a given rule, cd to the `workflow` directory and run:

```bash
snakemake -j4 -R build_shapes  --until build_base_network
```
where `build_shapes` is forced to run, and `build_base_network` is the last rule you would like to run.
