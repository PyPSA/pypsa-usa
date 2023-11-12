(introduction)=
# Introduction

PyPSA-USA is an open-source bulk power system planning model for the United-States. PyPSA-USA is built on the Python for Power System Analysis (pypsa) platform, and leverages much of the work of [PyPSA-EUR](https://pypsa-eur.readthedocs.io/en/latest/index.html) throughout the workflow. For some introduction to the pypsa modeling workflow- please see the [pypsa-eur](https://youtu.be/ty47YU1_eeQ?si=Cz90jWcN1xk1Eq4i) introductory video.

(workflow)=
## Workflow 

Add general description about [snakemake](https://snakemake.readthedocs.io/en/stable/index.html) 

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
