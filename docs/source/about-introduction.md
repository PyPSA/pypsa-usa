# Introduction

PyPSA-USA is a model for the US bulk power system built on the pypsa platform. For some introduction to the pypsa modeling workflow- please see the [pypsa-eur](https://youtu.be/ty47YU1_eeQ?si=Cz90jWcN1xk1Eq4i) introductory video.

## Workflow 

Add general description about [snakemake](https://snakemake.readthedocs.io/en/stable/index.html) 

![pypsa-usa workflow](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/repo_data/dag.jpg?raw=true)

## Folder Structure 

The project follows the [recommended folder structure](https://snakemake.readthedocs.io/en/stable/snakefiles/deployment.html) from Snakemake 

```bash
├── .gitignore
├── README.md
├── LICENSE.md
├── workflow
│   ├── rules
|   │   ├── module1.smk
|   │   └── module2.smk
│   ├── envs
|   │   ├── tool1.yaml
|   │   └── tool2.yaml
│   ├── scripts
|   │   ├── script1.py
|   │   └── script2.R
│   ├── notebooks
|   │   ├── notebook1.py.ipynb
|   │   └── notebook2.r.ipynb
│   ├── report
|   │   ├── plot1.rst
|   │   └── plot2.rst
|   └── Snakefile
├── config
│   ├── config.yaml
│   └── some-sheet.tsv
├── results
└── resources
```

## System Requirements 

TBD