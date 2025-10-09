(usage)=
# Usage

```{note}
**If you have not done so, please follow the [installation instructions](https://pypsa-usa.readthedocs.io/en/latest/about-install.html) [github issues](https://github.com/PyPSA/pypsa-usa/issues)**
```

PyPSA-USA can be used in two ways: through the Python API (recommended for PyPI installations) or through direct Snakemake calls (for development installations).

## Option 1: Python API (Recommended for PyPI Installation)

The Python API provides a clean interface for running PyPSA-USA workflows programmatically:

```python
from pypsa_usa.api import run_workflow, set_default_workspace
from pathlib import Path

# Set your default workspace (only needed once)
set_default_workspace("/path/to/my/project/workspace")

# Run a complete workflow
success = run_workflow(
    config="config.default.yaml",  # Uses bundled default config
    targets=["all"],
    cores=4
)

# Run specific targets
success = run_workflow(
    config="my_custom_config.yaml",
    targets=["data_model", "solve_network"],
    cores=8
)

# Override default workspace for specific runs
success = run_workflow(
    user_workspace="/home/user/special_project",
    config="config.default.yaml",
    targets=["all"],
    cores=4,
    dryrun=True
)
```

### API Configuration

When using the Python API, you can provide configuration in several ways:

1. **Use bundled default config** (no config file needed):
   ```python
   run_workflow(targets=["all"], cores=4)  # Uses default workspace
   ```

2. **Provide custom config file path**:
   ```python
   run_workflow(config="path/to/my_config.yaml", targets=["all"], cores=4)
   ```

3. **Pass config as dictionary**:
   ```python
   config_dict = {
       "scenario": {"interconnect": "western", "clusters": 30},
       "run": {"name": "my_run"}
   }
   run_workflow(config=config_dict, targets=["all"], cores=4)
   ```

### Workspace Management

PyPSA-USA provides convenient workspace management to avoid specifying the same path repeatedly:

```python
from pypsa_usa.api import set_default_workspace, get_default_workspace

# Set your default workspace (only needed once)
set_default_workspace("/path/to/my/project/workspace")

# Check your current default workspace
current_workspace = get_default_workspace()
print(f"Default workspace: {current_workspace}")

# Now you can run workflows without specifying workspace
success = run_workflow(targets=["all"], cores=4)

# Override default workspace for specific runs
success = run_workflow(
    user_workspace="/home/user/special_project",
    targets=["all"],
    cores=4
)
```

**Configuration Storage**: Your default workspace is stored in `~/.config/pypsa-usa/config.json` following the XDG Base Directory Specification.

## Option 2: Direct Snakemake Usage (Development Installation)

For development installations, you can use Snakemake directly:

### Set Configuration

To start, you'll want to set the proper network configuration for your studies purpose. The default configuration in `config/config.default.yaml` using the `western` interconnect and 30 nodes is a good place to start!

You can find more information on each configuration setting on the [configurations page](https://pypsa-usa.readthedocs.io/en/latest/config-configuration.html).

### Run workflow

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

To generate the data model only, specify the rule `data_model` in the `snakemake` call. The `data_model` rule generates the network file that is passed into the `solve_network` rule. This network will **not** include any additional policy constraints and only includes input data (ie. the network is not solved). The network is available in the `resources/` folder.

UV:
```console
uv run snakemake data_model -j1 --configfile config/config.default.yaml --scheduler-ilp-solver GUROBI_CMD
```

mamba:
```console
mamba activate pypsa-usa
snakemake data_model -j1 --configfile config/config.default.yaml
```

## File Organization

### User Workspace

PyPSA-USA creates a `user_workspace/` directory in your current working directory to store:

- **Configuration files**: Your custom config files
- **Runtime data**: Downloaded data from `retrieve_` rules
- **Resources**: Built network files and intermediate results
- **Results**: Final outputs, plots, and analysis results
- **Logs**: Execution logs and benchmarks

```
user_workspace/
├── config/           # Your custom configuration files
├── data/            # Downloaded data (e.g., EFS load profiles)
├── resources/       # Built network files and intermediate data
├── results/         # Final outputs, plots, and analysis
└── logs/           # Execution logs and benchmarks
```

### Bundled Data

PyPSA-USA includes essential data files bundled with the package:

- **Geospatial data**: Shape files, transmission lines, bus regions
- **Cost data**: Technology costs, fuel prices, emissions factors
- **Policy constraints**: RPS targets, transmission constraints
- **Default configurations**: Ready-to-use config templates

These files are automatically accessible through the package and don't need to be downloaded separately.


## Running on HPC Cluster

If you are running the workflow on an High-Performance Compute (HPC) cluster, you will first need to update the configuration settings in `config.cluster.yaml`. Update the account, partition, email, and chdir fields to match the information of your institutions cluster.

Next, identify the name of the configuration file you would like to run by editing the `run_slurm.sh` script. The default value is the `--configfile config/config.default.yaml`.

To run, open a terminal within a login node of your cluster and run the script included in the `workflow` directory:

```console
bash run_slurm.sh
```

We have included settings in the Snakemake workflow to dynamically request reasources from an HPC cluster based on the size of the pypsa-usa model you decide to run. To modify these resource selections checkout the `memory` and `threads` fields in individual snakemake rules.

## Examine Results

Result plots and images are automatically built in the `workflow/results` folder. To further analyze the results of a solved network, you can use pypsa to analyze the `elec_s_{clusters}_ec_l{l}_{opts}.nc` file in the `results/{interconnect}/networks/` folder. (Tutorial juyper notebook is on the way!)

(troubleshooting)=
## Troubleshooting:

To force the execution of a portion of the workflow up to a given rule, cd to the `workflow` directory and run:

```console
snakemake -j4 -R build_shapes  --until build_base_network
```
where `build_shapes` is forced to run, and `build_base_network` is the last rule you would like to run.
