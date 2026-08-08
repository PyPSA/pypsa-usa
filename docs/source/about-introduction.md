(introduction)=
# Introduction

PyPSA-USA is an open-source model of the United States bulk energy system for
**capacity-expansion planning**, **production-cost simulation**, and **power-flow
analysis**. It combines a curated data pipeline for the US grid — transmission networks,
generators, demand, renewable resources, costs, and policies — with the optimization
framework of [PyPSA](https://pypsa.readthedocs.io/en/latest/), so you can go from raw
public data to a solved planning or operations model through a single configuration file.

Through that configuration file you control the spatial, temporal, and operational
resolution of your model: which interconnection to study, how many network zones to
resolve, which weather and demand years to use, which technologies may be expanded, and
which policy constraints apply. You can also stop the pipeline before the optimization
stage and export the assembled data model (netCDF/CSV) into your own tools.

PyPSA-USA builds on and leverages the work of
[PyPSA-EUR](https://pypsa-eur.readthedocs.io/en/latest/index.html) developed by TU
Berlin. PyPSA-USA is actively developed by the
[INES Research Group](https://ines.stanford.edu) at
[Stanford University](https://www.stanford.edu/) and the
[ΔE+ Research Group](https://www.sfu.ca/see/research/delta-e.html) at
[Simon Fraser University](https://www.sfu.ca/).

```{note}
This model is under active development. If you need assistance or would like to discuss
using the model, please reach out to **ktehranchi@stanford.edu** and
**trevor_barnes@sfu.ca**
```

## What the model is

A PyPSA-USA model is a [PyPSA network](https://pypsa.readthedocs.io/en/latest/user-guide/components.html):
buses (network zones) connected by transmission lines and links, populated with
generators, storage, and loads, and optimized over a year of hourly (or coarser)
snapshots. Depending on configuration, the optimizer either dispatches an existing
fleet (production-cost mode) or co-optimizes dispatch with new investment in
generation, storage, and transmission (capacity expansion), subject to national and
state policy constraints such as Renewable Portfolio Standards, emissions limits, and
planning-reserve margins.

The {doc}`model-components` page explains how PyPSA components are used in PyPSA-USA
and how the spatial and temporal resolution is chosen; {doc}`model-constraints`
documents the custom constraints PyPSA-USA adds on top of PyPSA's core formulation.

## Power Sector

Whether you're focusing on **ERCOT, WECC, or the Eastern Interconnection**, PyPSA-USA
gives you the flexibility to:

- Choose between multiple transmission networks (ReEDS zonal topologies or the
  synthetic TAMU/Breakthrough Energy nodal network).
- Cluster the network to a user-defined number of zones, respecting county lines,
  balancing areas, states, or REeDS zone boundaries.
- Use **historical EIA-930 demand** (2018-2023), **NREL EFS electrification
  scenarios**, or **EER forecasted demand** (2030, 2040, 2050).
- Build renewable capacity-factor profiles from the **GODEEEP** dataset (default) or
  compute them with **Atlite** from ERA5 weather data.
- Incorporate **historical daily/monthly fuel prices** from ISOs/EIA for your chosen
  year.
- Import cost projections from the **NREL Annual Technology Baseline** and **Annual
  Energy Outlook**.

You can create and export data models for use in your own optimization models via CSV
tables or xarray netCDF formats, or run expansion-planning studies that integrate
regional and national policy constraints like RPS/CES standards, emissions limits,
technology capacity targets, and planning-reserve margins.

## Sector Coupling

Sector-coupling studies build on the power sector to represent other parts of the
energy system. PyPSA-USA allows you to:

- Model end-use technology options by sector (residential, commercial, transportation,
  industrial)
- Include multiple end-use demands (heating, cooling, etc.)
- Study the natural gas sector at a state level
- Evaluate different electrification policies across sectors

Additionally, data from the NREL EFS, EIA Annual Energy Outlook, EIA consumption
surveys, among other sources, are automatically pulled into PyPSA-USA to limit or
enforce production and capacity requirements across sectors.

(workflow)=
## Workflow

PyPSA-USA is orchestrated by [Snakemake](https://snakemake.readthedocs.io/): every
model artifact is produced by a rule, and rules chain into a directed acyclic graph
(DAG) from raw data retrieval through network assembly to the solved model. The diagram
below shows the rule graph for a power-sector run; {doc}`model-workflow` walks through
what each stage does and which files it produces.

:::{figure} _static/dag.svg
:width: 100%
:alt: Rule graph of the PyPSA-USA power-sector workflow

Power-sector rule graph. Data flows top-to-bottom: shapes and the base network are
built first, the network is aggregated and clustered (`aggregate_to_substations`,
`cluster_simpl`), then demand, renewable profiles, generators, and costs are attached
at cluster resolution before the final clustering, constraint preparation, and solve.
:::

(folder-structure)=
## Folder Structure

The repository separates checked-in code and seed data from generated artifacts. All
paths below are relative to the repository root; `resources/` and `results/` are
created under `workflow/` on first run.

```console
├── README.md
├── LICENSE.md
├── init_pypsa_usa.sh            # one-time setup: copies default configs into place
├── pyproject.toml               # python dependencies (uv); pins pypsa/atlite/linopy
├── docs                         # this documentation (sphinx + myst)
├── tests                        # static + integration test suites
└── workflow
    ├── Snakefile                # entry point: wildcards, paths, top-level rules
    ├── rules                    # snakemake rule definitions (*.smk)
    ├── scripts                  # python scripts executed by the rules
    ├── config                   # your run configuration (config.default.yaml, ...)
    ├── repo_data                # small checked-in seed data (shapes, costs, dag)
    ├── envs                     # conda environment specification
    ├── data                     # downloaded raw data bundles
    ├── cutouts                  # atlite weather cutouts (optional, large)
    ├── resources                # generated intermediates (networks/, profiles/, ...)
    │   ├── networks             #   elec_base_network.nc ... elec_s{simpl}_c{clusters}.nc
    │   ├── busmaps              #   bus -> cluster mappings for each aggregation stage
    │   ├── profiles             #   renewable capacity-factor profiles
    │   ├── geospatial           #   region and zone shapes (geojson)
    │   ├── costs                #   technology cost tables per investment year
    │   └── ...                  #   demand/, prices/, powerplants/, ...
    ├── benchmarks               # per-rule runtime/memory measurements
    ├── logs                     # per-rule logs
    └── results                  # solved networks, statistics, and figures per run
```
