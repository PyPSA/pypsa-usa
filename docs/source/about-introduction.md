(introduction)=
# Introduction

PyPSA-USA is an open-source tool that enables you to model and simulate the United States energy system with flexibility.

PyPSA-USA offers a versatile toolkit that allows you to customize the **data** for your energy system model with ease. Through a simple configuration file, you can control the spatial, temporal, and operational resolution of your model. Access a range of cleaned and prepared historical and forecasted data to build a model tailored to your needs.

Whether you’re focusing on **ERCOT, WECC, or the Eastern Interconnection**, PyPSA-USA gives you the flexibility to:
- Choose between multiple transmission networks.
- Cluster the nodal network a user-defined number of nodes, respecting county lines, balancing areas, states, NERC region boundaries.
- Utilize **historical EIA-930 demand data** (2018-2023) or **NREL EFS forecasted demand** (2030, 2040, 2050).
- Incorporate **historical daily/monthly fuel prices** from ISOs/EIA for your chosen year.
- Import cost projections from the **NREL Annual Technology Baseline** and **Annual Energy Outlook**.

You can create and export data models for use in your own optimization models via CSV tables or xarray netCDF formats.

PyPSA-USA also provides an interface for running capacity expansion planning and operational simulation models with the Python for Power System Analysis (pypsa) package. You can run expansion planning exercises which integrate regional and national policy constraints like RPS standards, emissions standards, PRMs, and more.

PyPSA-USA builds on and leverages the work of [PyPSA-EUR](https://pypsa-eur.readthedocs.io/en/latest/index.html) developed by TU Berlin. It is actively developed by the [INES Research Group](https://ines.stanford.edu) at Stanford University and the [ΔE+ Research Group](https://www.sfu.ca/see/research/delta-e.html) at Simon Fraser University. We welcome contributions and collaborations from the community- please don't hesitate to reach out!

(workflow)=
## Workflow

The diagram below illustrates the workflow of PyPSA-USA, highlighting how the data flows through the model scripts.

![pypsa-usa workflow](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/repo_data/dag.jpg?raw=true)

(folder-structure)=
## Folder Structure

PyPSA-USA is organized to facilitate easy navigation and efficient execution. Below is the folder structure of the project. Each folder serves a specific purpose, from environment setup to data processing and storage. After running the Snakemake file for the first time, your directory will be built and populated with the necessary data files.

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


```{warning}
**This model is under active development. If you need assistance or would like to discuss using the model, please reach out to ktehranchi@stanford.edu and trevor_barnes@sfu.ca.**
```
