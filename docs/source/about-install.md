# Installation

## Clone Repository 

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

## Conda Environment 

PyPSA-USA uses Anaconda to manage project dependencies. Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 
following their 
[instructions](https://docs.conda.io/en/latest/miniconda.html). 

Once conda is installed, download the provided environemnt file and activate the `pypsa-usa` conda environment. 

```bash 
$ conda env create -f environment.yaml
$ conda activate pypsa-usa
```

