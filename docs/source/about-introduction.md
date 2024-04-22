(introduction)=
# Introduction

```{warning}
**This model is under active development. If you would like to chat about using the model please don't hesitate to reach out to ktehranchi@stanford.edu and trevor_barnes@sfu.ca for support**
```

PyPSA-USA is an open-source energy system dataset of the United States energy system with continental US coverage.

PyPSA-USA provides you with a toolkit to customize the **data** of energy system model with ease. Through configuration file you can control the spatial, temporal, and operational resolution of your energy system model with access to cleaned and prepared historical and forecasted data. This means, you can build a model of **ERCOT, WECC, or the Eastern interconnection**, where the transmission network is clustered to **N# of user defined nodes**, which can respect the boundaries of **balancing areas, states, or REeDs GIS Shapes**, using **historical EIA-930 demand data years 2018-2023** OR **NREL EFS forcasted demand [2030, 2040, 2050]**, with **historical daily/monthly fuel prices from ISOs/EIA [choice of year]**, AND imported capital cost projections from the **NREL Annual Technology Baseline**.

You can create data model- and export to use in your own homebrewed optimization model via csv tables, or xarray netCDF model.

Beyond creating a data model, PyPSA-USA also provides an interface for running capacity expansion planning and operational simulation models with DC power flow with the Python for Power System Analysis package. You can run expansion planning exercises which integrate regional and national policy constraints like RPS standards, emissions standards, PRMs, and more.

PyPSA-USA builds on and leverages the work of [PyPSA-EUR](https://pypsa-eur.readthedocs.io/en/latest/index.html) developed by TU Berlin. PyPSA-USA is actively developed by the [INES Research Group](https://ines.stanford.edu) at Stanford University and the [ΔE+ Research Group](https://www.sfu.ca/see/research/delta-e.html) at Simon Fraser University.

(workflow)=
## Workflow

![pypsa-usa workflow](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/repo_data/dag.jpg?raw=true)

(folder-structure)=
## Folder Structure

The project is organized based on the folder structure below. The workflow folder contains all data and scripts neccesary to run the pypsa-usa model. After the first time you run the snakemake file, your directory will be built and populated with the associated data. Because most of the files are too large to store on github, we pull data from various sources into the `data` folder. The `repo_data` folder contains smaller files suitable for github. The resources folder contains intermediate files built by snakemake rules through the workflow. You'll see sub-folders created for each interconnection you run the model with.

The envs folder contains the conda env yaml files neccesary to build your mamba/conda environment. The scripts folder contains the individual python scripts that are referenced in the Snakefile rules.

```bash
├── .gitignore
├── README.md
├── LICENSE.md
├── docs
├── report
├── workflow
│   ├── envs
|   │   └── environment.yaml
│   ├── logs
|   │   └── example.log
│   ├── scripts
|   │   ├── script1.py
|   │   └── script2.R
│   ├── config
|   │   ├── config.yaml
|   │   └── config.example.yaml
│   ├── resources
|   │   ├── folder1
|   │   └── intermediate_data_example.csv
│   ├── repo_data
|   │   ├── example.tiff
|   │   └── example2.csv
│   ├── data
|   │   └── breakthrough_network
|   │   └── WECC_ADS
|   │   └── otherfolders
│   ├── results
|   │   ├── example_network.nc
|   │   └── example_data.csv
|   └── Snakefile
```
