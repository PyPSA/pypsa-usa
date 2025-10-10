# Installation

PyPSA-USA can be installed in two ways: as a PyPI package for easy use, or from source for development and advanced usage.

## Option 1: PyPI Installation (Recommended)

The easiest way to install PyPSA-USA is through PyPI:

```console
pip install pypsa-usa
```

### Quick Start with PyPI Installation

```python
from pypsa_usa.api import run_workflow

# Run a simple workflow with default settings
success = run_workflow(cores=4)

# Or with custom configuration
success = run_workflow(config="my_config.yaml", targets=["all"], cores=4)
```

### System Requirements

- Python 3.8 or higher
- Solver (HiGHS, Gurobi, CPLEX, etc.)
- EIA API key

## Option 2: Development Installation

For development, customization, or advanced usage, install from source:

### Step 1. Clone GitHub Repository

Users can clone the repository using HTTPS, SSH, or GitHub CLI. See [GitHub docs](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) for information on the different cloning methods. If you run into issues, follow GitHub troubleshooting suggestions [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/troubleshooting-cloning-errors#https-cloning-errors).

#### Using HTTPS

```console
git clone https://github.com/PyPSA/pypsa-usa.git
```

#### Using SSH-Key

If it your first time cloning a **repository through ssh**, you will need to set up your git with an ssh-key by following these [directions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

```console
git clone git@github.com:PyPSA/pypsa-usa.git
```

### Step 2. Initialize Configuration files

From the command line, run the script `init_pypsa_usa.sh` to copy configuration file
templates into the `workflow/config` folder.

```console
bash init_pypsa_usa.sh
```

### Step 3: Set-up Environment (mamba or UV)

PyPSA-USA can be managed though either [`UV`](https://github.com/astral-sh/uv) or [`mamba`](https://github.com/mamba-org/mamba). Users only need to install one, not both!

```{seealso}
If you are planning to develop `PyPSA-USA`, please see our [contribution guidelines](./contributing.md#code-contributions) for installing additional dependencies.
```

### Step 3a: `uv` installation

[`UV`](https://docs.astral.sh/uv/) is a new python package managment tool from [`Astral`](https://astral.sh/), the creators of [`ruff`](https://github.com/astral-sh/ruff). It replaces `mamba`, `conda`, and `pip` commands for one package and virtual environment managment tool. Instructions for installing `UV` can be found [here](https://docs.astral.sh/uv/getting-started/installation/).

Once `UV` is installed, you can activate the environemnt with:

```console
uv venv
source .venv/bin/activate
```

```{warning}
If you are migrating from `mamba`/`conda`, you may need to install system level dependencies that conda has previously handeled. These include, `HDF5` and `GDAL>=3.1` libraries.
```

### Step 3b: `mamba` Installation

Alternatively, PyPSA-USA can manage project dependencies through `conda`/`mamba`. You can download and install `mamba` following the [instructions](https://mamba.readthedocs.io/en/latest/mamba-installation.html). Follow links for mambaforge installation. There are two ways to install `mamba`, the first (recommended) method will start with a fresh install, meaning if you have previously installed `conda` environments, you will need to recreate these `conda` envs. If you already have `conda` installed and do not wish to install `mamba`, you can follow the same set of instructions replacing any `mamba` with `conda`

Once `mamba` is installed, use the environment file within the git repository to create the PyPSA-USA conda environment. This step can take ~10-20 minutes. After creating the mamba environment, you will need to activate it before running the snakemake workflow.

```console
mamba env create -f workflow/envs/environment.yaml
mamba activate pypsa-usa
```

You also have the option to use `miniconda`. Download [`miniconda`](https://docs.conda.io/en/latest/miniconda.html) following their [instructions](https://docs.conda.io/en/latest/miniconda.html).

## Step 4: Install a Solver

PyPSA-USA uses an external solver to solve the optimization problem formulated in the workflow. After you install your solver and confirm it is accessible within your conda environment, update your solving configuration to match your solver of choice.
You can download and install several free options here:

- [HiGHS](https://highs.dev/)
- [Cbc](https://projects.coin-or.org/Cbc#DownloadandInstall)
- [GLPK](https://www.gnu.org/software/glpk/)
- [Ipopt](https://coin-or.github.io/Ipopt/INSTALL.html)

and the non-free, commercial software (for some of which free academic licenses are available)

- [Gurobi](https://www.gurobi.com/documentation/quickstart.html)
- [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio)

## Step 5: Get an EIA API Key

The PyPSA-USA workflow leverages the EIA API in several steps. The default configuration activates dynamic fuel-cost prices, which requires EIA API key. You can quickly get your key by completing this [form](https://www.eia.gov/opendata/register.php).

The API key will be emailed to you, and you can copy the key into the `config.api.yaml` file.
