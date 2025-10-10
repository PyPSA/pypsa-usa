# API Reference

PyPSA-USA provides a Python API for programmatic workflow execution. This is the recommended way to use PyPSA-USA when installed via PyPI.

## Core API Functions

### `run_workflow`

Execute PyPSA-USA workflows programmatically using the Snakemake Python API.

```python
from pypsa_usa.api import run_workflow

success = run_workflow(
    user_workspace=None,
    config=None,
    targets=None,
    cores=1,
    dryrun=False,
    forceall=False,
    forcetargets=False,
    **snakemake_kwargs
)
```

#### Parameters

- **`user_workspace`** (`str` or `Path`): Path to directory where all intermediate files, results, and logs will be stored. If `None`, uses the default workspace from user configuration. This directory will be created if it doesn't exist. Should be an absolute path for clarity.
- **`config`** (`str`, `dict`, or `None`): Configuration for the workflow
  - `str`: Path to a YAML configuration file
  - `dict`: Configuration dictionary
  - `None`: Use bundled default configuration
- **`targets`** (`list` or `None`): List of Snakemake targets to run
  - Default: `["all"]`
  - Examples: `["data_model"]`, `["solve_network"]`, `["all"]`
- **`cores`** (`int`): Number of CPU cores to use for parallel execution
  - Default: `1`
- **`dryrun`** (`bool`): If `True`, perform a dry run without executing commands
  - Default: `False`
- **`forceall`** (`bool`): Force execution of all rules, even if outputs exist
  - Default: `False`
- **`forcetargets`** (`bool`): Force execution of target rules
  - Default: `False`
- **`**snakemake_kwargs`**: Additional keyword arguments passed to `snakemake.snakemake()`

#### Returns

- **`bool`**: `True` if workflow execution was successful, `False` otherwise

#### Examples

```python
from pathlib import Path
from pypsa_usa.api import set_default_workspace

# Set your default workspace (only needed once)
set_default_workspace("/path/to/my/project/workspace")

# Basic usage with default configuration
success = run_workflow(cores=4)

# Run specific targets
success = run_workflow(targets=["data_model", "solve_network"], cores=8)

# Use custom configuration file
success = run_workflow(config="my_config.yaml", targets=["all"], cores=4)

# Pass configuration as dictionary
config = {
    "scenario": {"interconnect": "western", "clusters": 50, "simpl": 100},
    "run": {"name": "my_analysis"},
}
success = run_workflow(config=config, cores=4)

# Override default workspace for specific runs
success = run_workflow(
    user_workspace="/home/user/special_project", targets=["all"], cores=4, dryrun=True
)

# Force re-execution of all rules
success = run_workflow(targets=["all"], cores=4, forceall=True)
```

### `set_default_workspace`

Set the user's default workspace directory.

```python
set_default_workspace(workspace)
```

#### Parameters

- **`workspace`** (`str` or `Path`): Path to set as default workspace

#### Returns

- **`Path`**: The resolved workspace path

#### Example

```python
from pypsa_usa.api import set_default_workspace

# Set default workspace
workspace = set_default_workspace("/home/user/pypsa_projects")
print(f"Default workspace set to: {workspace}")
```

### `get_default_workspace`

Get the user's current default workspace.

```python
get_default_workspace()
```

#### Returns

- **`Path` or `None`**: Path to default workspace if set, `None` otherwise

#### Example

```python
from pypsa_usa.api import get_default_workspace

# Get current default workspace
workspace = get_default_workspace()
if workspace:
    print(f"Current default workspace: {workspace}")
else:
    print("No default workspace set")
```

## Configuration Options

### Default Configuration

When no configuration is provided, PyPSA-USA uses a bundled default configuration with:

- **Interconnect**: `western`
- **Clusters**: `30`
- **Simplification**: `75`
- **Sector**: `E` (electricity only)
- **Options**: `REM-3h` (renewable energy mix with 3-hour resolution)

### Custom Configuration

You can provide custom configuration in several ways:

#### 1. Configuration File

Create a YAML file with your settings:

```yaml
# my_config.yaml
scenario:
  interconnect: "western"
  clusters: 50
  simpl: 100
  ll: "v1.0"
  opts: "REM-3h"
  sector: "E"

run:
  name: "my_analysis"
  validation: true

costs:
  year: 2020
  version: "v0.3.0"
```

Then use it:

```python
success = run_workflow(config="my_config.yaml", cores=4)
```

#### 2. Configuration Dictionary

Pass configuration directly as a Python dictionary:

```python
config = {
    "scenario": {"interconnect": "western", "clusters": 50, "simpl": 100},
    "run": {"name": "my_analysis"},
}
success = run_workflow(config=config, cores=4)
```

## Common Workflow Targets

### `all`
Complete workflow from data retrieval to final results and plots.

### `data_model`
Generate the network data model without solving. Creates network files in `user_workspace/resources/`.

### `solve_network`
Solve the optimization problem for the network. Requires `data_model` to be completed first.

### `retrieve_*`
Download external data (e.g., `retrieve_nrel_efs_data` for load profiles).

### `build_*`
Build specific components (e.g., `build_shapes`, `build_cost_data`).

## Error Handling

The API provides basic error handling and logging:

```python
from pypsa_usa.api import run_workflow

try:
    success = run_workflow(config="my_config.yaml", targets=["all"], cores=4)
    if success:
        print("Workflow completed successfully!")
    else:
        print("Workflow failed. Check logs in user_workspace/logs/")
except Exception as e:
    print(f"Error running workflow: {e}")
```

## Advanced Usage

### Custom Snakemake Arguments

You can pass additional arguments to the underlying Snakemake execution:

```python
success = run_workflow(
    targets=["all"],
    cores=4,
    # Additional Snakemake arguments
    keepgoing=True,  # Continue on errors
    latency_wait=60,  # Wait for missing files
    scheduler="greedy",  # Use greedy scheduler
)
```

### Working Directory Management

By default, PyPSA-USA creates a `user_workspace/` directory in your current working directory. You can specify a custom working directory:

```python
success = run_workflow(targets=["all"], cores=4, workdir="/path/to/my/workspace")
```

## Integration with Other Tools

### Jupyter Notebooks

```python
# In a Jupyter notebook
from pypsa_usa.api import run_workflow
import matplotlib.pyplot as plt

# Run workflow
success = run_workflow(cores=4)

if success:
    # Load and analyze results
    import pypsa

    n = pypsa.Network(
        "user_workspace/results/western/networks/elec_s75_c30_ec_lv1.0_REM-3h_E.nc"
    )

    # Create custom plots
    n.plot()
    plt.title("PyPSA-USA Network")
    plt.show()
```

### Scripts and Automation

```python
# automation_script.py
from pypsa_usa.api import run_workflow
import os


def run_scenario_analysis(interconnect, clusters):
    config = {
        "scenario": {"interconnect": interconnect, "clusters": clusters},
        "run": {"name": f"{interconnect}_{clusters}clusters"},
    }

    success = run_workflow(config=config, targets=["all"], cores=8)

    return success


# Run multiple scenarios
scenarios = [("western", 30), ("western", 50), ("eastern", 30)]

for interconnect, clusters in scenarios:
    print(f"Running {interconnect} with {clusters} clusters...")
    success = run_scenario_analysis(interconnect, clusters)
    print(f"Completed: {success}")
```
