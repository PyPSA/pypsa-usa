# PyPSA-USA

This workflow optimizes the power system model provided by [BreakthroughEnergy](https://www.breakthroughenergy.org/). Note that the functionalities of the workflow are still limited as the it is in initial stage.

# Installation

For installation clone this repository and create the dedicated `conda` environment

```
conda env create -f environment.yaml
conda activate pypsa-usa
```

download submodules after cloning:

`git submodule update --init`

update submodule with:

`git submodule update --remote`


# Workflow

## Configuration

## Execution 
To execute the workflow, go into the `workflow` directory and execute `snakemake` from your terminal, i.e.

```bash
snakemake -j6
```

where 6 indicates the number of used cores, you may change it to your preferred number. This will run the first rule defined in the `Snakefile`.


To force the execution of a portion of the workflow up to a given rule, cd to the `workflow` directory and run:

```bash
snakemake -j4 -R build_shapes  --until build_base_network
```
where `build_shapes` is forced to run, and `build_base_network`  is the last rule you would like to run.

## Examine Results


<!-- # Scope -->

# License

The project is licensed under MIT License.
