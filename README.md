# PyPSA-USA

PyPSA-USA is an open-source power systems model of the bulk transmission systems in the United States. This workflow draws from the work of [pypsa-eur](https://pypsa-eur.readthedocs.io/en/latest/index.html) and [pypsa-meets-earth](https://pypsa-earth.readthedocs.io/en/latest/how_to_contribute.html) to build a highly configurable power systems model that can be used for capacity expansion modeling, production cost simulation, and power flow analysis. This model is currently under development, and is only stable under certain configurations detailed below.

The model draws data from:

- The [BreakthroughEnergy](https://www.breakthroughenergy.org/) transmission network model. This model is a X bus network, with X lines above 69 kV, X substations, and x lines.
- Powerplant Data can be drawn from three options: the Breakthrough Network, the WECC Anchor Data Set Production Cost Model, or the [PUDL](https://catalystcoop-pudl.readthedocs.io/en/latest/index.html#) dataset. (only breakthrough functioning currently in master branch)
- Historical load data from the EIA via the [GridEmissions](https://github.com/jdechalendar/gridemissions/) tool.
- Forecasted load data from the [WECC ADS PCM](https://www.wecc.org/ReliabilityModeling/Pages/AnchorDataSet.aspx).
- Renewable time series based on ERA5 and SARAH, assembled using the atlite tool. (under development)
- Geographical potentials for wind and solar generators based on land use (CORINE) and excluding nature reserves (Natura2000) are computed with the atlite library.(under development)

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

![pypsa-usa workflow](https://github.com/PyPSA/pypsa-breakthroughenergy-usa/blob/dev_atlite_integration/workflow/repo_data/dag.jpg?raw=true)

## Configuration

**This workflow has only been thoroughly tested for the `western` interconnection wildcard.**

## Execution 
To execute the workflow, go into the `workflow` directory and execute `snakemake` from your terminal. 

```bash
snakemake -j6
```

where 6 indicates the number of used cores, you may change it to your preferred number. This will run the workflow defined in the `Snakefile`.

For your initial execution you must set the configuration for retrieving data to true as specified in `config.defualt.yaml`. After the first run, you should set these to false to avoid redownloading the files.

```
enable:
  retrieve_databundle: true
  retrieve_cutout: true 
  retrieve_natura_raster: true 
```

### Troubleshooting:

To force the execution of a portion of the workflow up to a given rule, cd to the `workflow` directory and run:

```bash
snakemake -j4 -R build_shapes  --until build_base_network
```
where `build_shapes` is forced to run, and `build_base_network`  is the last rule you would like to run.

## Examine Results

# Contact
You can contact ctehran@stanford.edu for immediate questions on the usage of the tool.


<!-- # Scope -->

# License

The project is licensed under MIT License.
