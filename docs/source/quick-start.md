# Quick Start Guide

Get up and running with PyPSA-USA in minutes!

## Installation

### Option 1: PyPI Installation (Recommended)

```bash
pip install pypsa-usa
```

### Option 2: Development Installation

```bash
git clone https://github.com/PyPSA/pypsa-usa.git
cd pypsa-usa
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Your First Workflow

### Using the Python API

```python
from pypsa_usa.api import run_workflow, set_default_workspace
from pathlib import Path

# Set your default workspace (only needed once)
set_default_workspace("/path/to/my/project/workspace")

# Run a simple workflow with default settings
success = run_workflow(cores=4)

if success:
    print("Workflow completed successfully!")
    print("Results are in your workspace results/ directory")
else:
    print("Workflow failed. Check logs/ for details.")
```

### Using Snakemake Directly (Development Installation)

```bash
# Navigate to workflow directory
cd workflow

# Run with default configuration
uv run snakemake -j4 --configfile config/config.default.yaml
```

## What Happens Next?

1. **Data Download**: PyPSA-USA downloads necessary data (load profiles, cost data, etc.)
2. **Network Building**: Creates a power system network for the western interconnect
3. **Optimization**: Solves the capacity expansion and dispatch problem
4. **Results**: Generates plots and analysis in `user_workspace/results/`

## Explore Your Results

After the workflow completes, you'll find:

- **Plots**: Capacity maps, production charts, emissions analysis
- **Network Files**: PyPSA network objects for further analysis
- **Data**: All intermediate and final results

```python
# Load and explore your results
import pypsa

# Load the solved network
n = pypsa.Network(
    "user_workspace/results/western/networks/elec_s75_c30_ec_lv1.0_REM-3h_E.nc"
)

# View network summary
print(n)

# Plot the network
n.plot()
```

## Next Steps

- **Customize Configuration**: Modify settings in `config/config.default.yaml`
- **Different Scenarios**: Try different interconnects, cluster numbers, or time periods
- **Advanced Usage**: See the [API Reference](api-reference.md) for more options
- **Documentation**: Explore the full [documentation](about-usage.md) for detailed guidance

## Need Help?

- Check the [troubleshooting section](about-usage.md#troubleshooting)
- Review the [configuration guide](config-configuration.md)
- Contact us at ktehranchi@stanford.edu

## Example: Custom Configuration

```python
from pypsa_usa.api import run_workflow

# Custom configuration
config = {
    "scenario": {
        "interconnect": "western",
        "clusters": 50,  # More detailed network
        "simpl": 100,  # Higher simplification
        "ll": "v1.0",
        "opts": "REM-3h",
        "sector": "E",
    },
    "run": {"name": "my_first_run"},
}

# Run with custom settings
success = run_workflow(config=config, targets=["all"], cores=8)
```

This will create a more detailed model with 50 clusters and save results in `user_workspace/results/my_first_run/`.
