# Installation

## Step 1: Clone Github Repository 

Clone this repository and all submodules 

```bash 
$ git clone --recurse-submodules https://github.com/PyPSA/pypsa-usa.git
```

```{note}
If the repository was cloned without the `--recurse-submodules` flag, run the 
commands `git submodule init` and `git submodule update `
```

```bash
$ git submodule init
$ git submodule update 
```

## Step 2: Create Conda Environment 

PyPSA-USA uses conda/mamba to manage project dependencies. You can download and install mamba following the [instructions](https://mamba.readthedocs.io/en/latest/mamba-installation.html). Follow links for mambaforge installation. There are two ways to install mamba, the first (recommended) method will start with a fresh install, meaning if you have previously installed conda environments, you will need to recreate these conda envs. If you already have conda installed and do not wish to install mamba, you can follow the same set of instructions replacing any `mamba` with `conda`

Once mamba is installed, use the environment file within your git repository to activate the `pypsa-usa` conda environment. This step can take ~10-20 minutes. After creating the mamba environment, you will only need to activate it before running the snakemake workflow.

```bash 
$ mamba env create -f workflow/envs/environment.yaml
$ mamba activate pypsa-usa
```

You also have the option to use miniconda. Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) following their [instructions](https://docs.conda.io/en/latest/miniconda.html).


## Step 3: Install a Solver

PyPSA-USA uses an external solver to solve the optimization problem formulated in the workflow. After you install your solver and confirm it is accessible within your conda environment, update your solving configuration to match your solver of choice.
You can download and install several free options here:

- [HiGHS](https://highs.dev/)
- [Cbc](https://projects.coin-or.org/Cbc#DownloadandInstall)
- [GLPK](https://www.gnu.org/software/glpk/)
- [Ipopt](https://coin-or.github.io/Ipopt/INSTALL.html)

and the non-free, commercial software (for some of which free academic licenses are available)

- [Gurobi](https://www.gurobi.com/documentation/quickstart.html)
- [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio)