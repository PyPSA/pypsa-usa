[![DOI](https://zenodo.org/badge/500892486.svg)](https://zenodo.org/doi/10.5281/zenodo.10815964)
[![Documentation Status](https://readthedocs.org/projects/pypsa-usa/badge/?version=latest)](https://pypsa-usa.readthedocs.io/en/latest/?badge=latest)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# PyPSA-USA: An Open-Source Energy System Optimization Model for the United States

PyPSA-USA is an open-source power systems model of the bulk transmission systems in the United States. This workflow draws from the work of [pypsa-eur](https://pypsa-eur.readthedocs.io/en/latest/index.html) to build a highly configurable power systems model that can be used for capacity expansion modeling and production cost simulation.

## Quick Start

### Option 1: PyPI Installation (Recommended)

Install PyPSA-USA directly from PyPI:

```bash
pip install pypsa-usa
```

Then use the Python API:

```python
from pypsa_usa.api import run_workflow, set_default_workspace
from pathlib import Path

# Set your default workspace (only needed once)
set_default_workspace("/path/to/my/project/workspace")

# Now you can run workflows without specifying workspace
success = run_workflow(config="config.default.yaml", targets=["all"], cores=4)
```

### Option 2: Development Installation

For development or advanced usage, clone the repository:

```bash
git clone https://github.com/PyPSA/pypsa-usa.git
cd pypsa-usa
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Documentation

See our [readthedocs](https://pypsa-usa.readthedocs.io/en/latest/) for complete installation and usage instructions.

![pypsa-usa Base Network](https://github.com/PyPSA/pypsa-usa/blob/0fe788397238b14f07857a9748aa86c7781bfa27/docs/source/_static/PyPSA-USA_network.png)

# Contributing

We welcome your contributions to this project. Please see the [contributions](https://pypsa-usa.readthedocs.io/en/latest/contributing.html) guide in our readthedocs page for more information. Please do not hesitate to reachout to ktehranchi@stanford.edu with specific questions, requests, or feature ideas.

# License

The project is licensed under MIT License.

# Citation

See the CITATION.cff and github citation button on the right to generate your citation.
