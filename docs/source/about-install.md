# Installation

## Step 1. Clone GitHub Repository

Users can clone the repository using HTTPS, SSH, or GitHub CLI. Ensure you retrieve the submodules in the repository when cloning, using the `--recurse-submodules` flag. See [GitHub docs](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) for information on the different cloning methods. If you run into issues, follow GitHub troubleshooting suggestions [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/troubleshooting-cloning-errors#https-cloning-errors).

```{note}
If the repository is cloned without the `--recurse-submodules` flag, run the following commands.

    $ git submodule init
    $ git submodule update
```

### Using SSH-Key

If it your first time cloning a **repository through ssh**, you will need to set up your git with an ssh-key by following these [directions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

```bash
$ git clone --recurse-submodules git@github.com:PyPSA/pypsa-usa.git
```

### Using HTTPS

```bash
$ git clone --recurse-submodules https://github.com/PyPSA/pypsa-usa.git
```

## Step 2. Initialize Configuration files

From the command line, run the script `init_pypsa_usa.sh` to copy configuration file
templates into the `workflow/config` folder.

```bash
$ bash init_pypsa_usa.sh
```

## Step 3: Create Conda Environment

PyPSA-USA uses conda/mamba to manage project dependencies. You can download and install mamba following the [instructions](https://mamba.readthedocs.io/en/latest/mamba-installation.html). Follow links for mambaforge installation. There are two ways to install mamba, the first (recommended) method will start with a fresh install, meaning if you have previously installed conda environments, you will need to recreate these conda envs. If you already have conda installed and do not wish to install mamba, you can follow the same set of instructions replacing any `mamba` with `conda`

Once mamba is installed, use the environment file within your git repository to activate the `pypsa-usa` conda environment. This step can take ~10-20 minutes. After creating the mamba environment, you will only need to activate it before running the snakemake workflow.

```bash
$ mamba env create -f workflow/envs/environment.yaml
$ mamba activate pypsa-usa
```

You also have the option to use miniconda. Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) following their [instructions](https://docs.conda.io/en/latest/miniconda.html).


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
