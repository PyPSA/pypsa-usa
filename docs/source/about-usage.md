(usage)=
# Usage

```{note}
**If you have not done so, please follow the [installation instructions](https://pypsa-usa.readthedocs.io/en/latest/about-install.html) [github issues](https://github.com/PyPSA/pypsa-usa/issues)**
```

## Set Configuration

To start, you'll want to set the proper network configuration for your studies purpose. The default configuration in `config/config.default.yaml` using the `western` interconnect and 30 nodes is a good place to start!

You can find more information on each configuration setting on the [configurations page](https://pypsa-usa.readthedocs.io/en/latest/config-configuration.html).


## Run workflow

To run the workflow, `cd` into the `workflow` directory and run the `snakemake` from your terminal with your selection of config file:

UV:
```console
uv run snakemake -j1 --configfile config/config.default.yaml --scheduler-ilp-solver GUROBI_CMD
```

mamba:
```console
mamba activate pypsa-usa
snakemake -j1 --configfile config/config.default.yaml
```

### Generate Data Model

To generate the data model only, specify the rule `data_model` in the `snakemake` call. The `data_model` rule generates the network file that is passed into the `solve_network` rule. This network will **not** include any additional policy constraints and only includes input data (ie. the network is not solved). The network is available in the `resources/networks/` folder.

UV:
```console
uv run snakemake data_model -j1 --configfile config/config.default.yaml --scheduler-ilp-solver GUROBI_CMD
```

mamba:
```console
mamba activate pypsa-usa
snakemake data_model -j1 --configfile config/config.default.yaml
```


## Running on HPC Cluster

If you are running the workflow on an High-Performance Compute (HPC) cluster, you will first need to update the configuration settings in `config.cluster.yaml`. Update the account, partition, email, and chdir fields to match the information of your institutions cluster.

Next, identify the name of the configuration file you would like to run by editing the `run_slurm.sh` script. The default value is the `--configfile config/config.default.yaml`.

To run, open a terminal within a login node of your cluster and run the script included in the `workflow` directory:

```console
bash run_slurm.sh
```

We have included settings in the Snakemake workflow to dynamically request resources from an HPC cluster based on the size of the pypsa-usa model you decide to run. To modify these resource selections checkout the `memory` and `threads` fields in individual snakemake rules.

## Examine Results

### After the run: where outputs land

All outputs are written to `workflow/results/{run name}/{interconnect}/`, where `{run name}`
is the `run: name:` field of your configuration file (`Default` in
`config/config.default.yaml`). Inside you will find three folders: `networks/` holds the
solved network files, `figures/` holds automatically generated maps and plots along with
summary CSVs (capacities, generation, statistics) in its `statistics/` subfolders, and
`configs/` holds a snapshot of the configuration used for the run. A good first stop is the
`figures/` folder to sanity-check the system maps and statistics before diving into the
network file itself.

### Analyzing a solved network

To further analyze the results of a solved network, you can use pypsa to open the
`elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc` file in the
`results/{run name}/{interconnect}/networks/` folder. With the default configuration this is
`results/Default/western/networks/elec_s75_c33_ec_lv1.0_REM-3h_E.nc`.

The filename encodes the scenario. Each component is described briefly below; see the
[wildcards page](https://pypsa-usa.readthedocs.io/en/latest/config-wildcards.html) for full
details:

| Component | Example | Meaning |
|-----------|---------|---------|
| `s{simpl}` | `s75` | Number of buses after pre-clustering simplification |
| `c{clusters}` | `c33` | Final number of clustered buses (zones) |
| `ec` | `ec` | Fixed marker: extra components (e.g. storage) have been added |
| `l{ll}` | `lv1.0` | Transmission expansion limit (`v`olume or `c`ost, factor or `opt`) |
| `{opts}` | `REM-3h` | Dash-separated options (here: regional emissions limit, 3-hourly resolution) |
| `{sector}` | `E` | Sectors included (`E` = electricity only, `E-G` adds natural gas) |

(troubleshooting)=
## Troubleshooting:

To force the execution of a portion of the workflow up to a given rule, cd to the `workflow` directory and run:

```console
uv run snakemake -j4 -R build_shapes --until build_base_network --configfile config/config.default.yaml
```
where `build_shapes` is forced to run, and `build_base_network` is the last rule you would like to run.

```{note}
Every `snakemake` invocation must include `--configfile` (the Snakefile does not set a
default configuration file). Omitting it fails with `KeyError: 'scenario'`.
```
