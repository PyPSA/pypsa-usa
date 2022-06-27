# PyPSA-USA powered by BreakthroughEnergy

This workflow optimizes the power system model provided by [BreakthroughEnergy](https://www.breakthroughenergy.org/) via the [PowersimData package](https://github.com/Breakthrough-Energy/PowerSimData). Note that the functionalities of the workflow are still limited as the it is in initial stage. The project is funded by the BreakthroughEnergy Initiative.

# Installation

For installation clone this repository and create the dedicated `conda` environment

```
conda env create -f environment.yaml
conda activate pypsa-usa
```

download submodules after cloning:

```
git submodule update --init
```

update submodule with:

```
git submodule update --remote

```

# Workflow

For executing the workflow go into the `workflow` directory and execute `snakemake` from your terminal, i.e.

```bash
snakemake -j6
```

where 6 indicates the number of used cores, you may change it to your preferred number. This will run the first rule defined in the `Snakefile`.

<!-- # Scope -->

# License

The project is licensed under MIT License.
