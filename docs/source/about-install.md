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

PyPSA-USA uses mamba to manage project dependencies. You can download and install mamba following the [instructions](https://mamba.readthedocs.io/en/latest/mamba-installation.html). Follow links for mambaforge installation. There are two ways to install mamba, the first (recommended) method will start with a fresh install, meaning if you have previously installed conda environments, you will need to recreate these conda envs.

Once mamba is installed, use the environment file within your git repository to activate the `pypsa-usa` conda environment. This step can take ~10-20 minutes. After creating the mamba environment, you will only need to activate it before running the snakemake workflow.

```bash 
$ conda env create -f workflow/envs/environment.yaml
$ conda activate pypsa-usa
```

You also have the option to use miniconda. Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) following their [instructions](https://docs.conda.io/en/latest/miniconda.html). You can create the environment using the same commannds as above with `mamba` replaced with `conda`.

