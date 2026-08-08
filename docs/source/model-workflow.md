(model-workflow)=
# Workflow

PyPSA-USA is orchestrated by [Snakemake](https://snakemake.readthedocs.io/): every
artifact — a shape file, a clustered network, a solved model — is produced by a *rule*,
and rules chain into a directed acyclic graph (DAG) that Snakemake resolves from
whatever target you request back to raw data. This page walks through that graph
stage by stage: what each rule does, what it produces, and why the pipeline is ordered
the way it is. For the wildcards that parameterize the file names
(`{interconnect}`, `{simpl}`, `{clusters}`, `{ll}`, `{opts}`, `{sector}`), see
{doc}`config-wildcards`.

:::{figure} _static/dag.svg
:width: 100%
:alt: Rule graph of the PyPSA-USA power-sector workflow

Rule graph for a power-sector run of the default (Western Interconnection)
configuration. Each node is a Snakemake rule; arrows point from producer to consumer.
:::

## The shape of the pipeline: cluster early

The load-bearing architectural fact about PyPSA-USA is that **spatial aggregation
happens before the data-heavy build stages**. The base network is reduced from
thousands of nodal buses to the `{simpl}` cluster resolution first, and *then*
renewable profiles, demand, generators, and costs are built directly at that
resolution. Building per-bus data at ~50-200 cluster buses instead of ~3,000 nodal
buses cuts the memory and runtime of the heavy rules several-fold, and it means every
downstream artifact is keyed to a stable set of cluster buses.

The practical consequences:

- `{simpl}` (set in `scenario: simpl:`) is the resolution at which *data is built*;
  `{clusters}` is the final transmission resolution the model is solved at.
- Most intermediate files carry `_s{simpl}` in their name; anything keyed to buses
  downstream of `cluster_simpl` refers to cluster buses, not substations.
- The bus-to-cluster mappings for every aggregation stage are saved as busmaps
  (`resources/busmaps/`), so precomputed substation- or node-keyed datasets can be
  remapped into the clustered space (this is exactly what `aggregate_egs` does for
  geothermal supply curves).

## Stage by stage

### 1. Shapes and base network

| Rule | Script | Key outputs |
|------|--------|-------------|
| `build_shapes` | `build_shapes.py` | `resources/geospatial/` state, county, balancing-authority, REeDS-zone, and offshore shapes |
| `build_base_network` | `build_base_network.py` | `resources/networks/elec_base_network.nc`, `resources/busmaps/bus2sub.csv`, `sub.csv` |
| `build_bus_regions` | `build_bus_regions.py` | `resources/geospatial/regions_onshore.geojson`, `regions_offshore.geojson` |

`build_shapes` assembles the geographic backbone: interconnection boundaries, states,
counties, EIA balancing authorities, REeDS zones, and offshore wind areas.
`build_base_network` reads the chosen transmission dataset (the synthetic
TAMU/Breakthrough Energy nodal network, or a ReEDS zonal topology) and produces the
first PyPSA network: buses, lines, links, and transformers, each bus annotated with
its geographic memberships (state, county, balancing area, REeDS zone — see
{doc}`model-network-schema`). `build_bus_regions` computes the Voronoi-style onshore
and offshore service regions around each bus that later steps use to allocate
resources and demand.

### 2. Aggregation and pre-clustering

| Rule | Script | Key outputs |
|------|--------|-------------|
| `aggregate_to_substations` | `aggregate_to_substations.py` | `resources/networks/elec_b.nc`, `resources/busmaps/busmap_b.csv` |
| `cluster_simpl` | `cluster_simpl.py` | `resources/networks/elec_s{simpl}.nc`, `resources/geospatial/regions_*_s{simpl}.geojson`, `resources/busmaps/busmap_s{simpl}.csv` |

`aggregate_to_substations` is a pure topology reduction: nodal buses collapse onto
their substations, preserving electrical connectivity. `cluster_simpl` then runs
kmeans (or modularity) clustering to the `{simpl}` resolution, honoring the
`model_topology: topological_boundaries` setting so clusters never straddle county,
state, REeDS-zone, or balancing-area lines. Its outputs — the clustered network, the
matching cluster regions, and the busmap — define the spatial frame everything below
is built in. (In configuration, this stage's options live under
`clustering: simplify_network:`, a historical name kept for compatibility.)

### 3. Data at cluster resolution

| Rule | Script | Key outputs |
|------|--------|-------------|
| `build_renewable_profiles` | `build_renewable_profiles.py` | `resources/profiles/profile_{tech}_s{simpl}.nc` |
| `aggregate_egs` (if EGS enabled) | `aggregate_egs.py` | `resources/profiles/specs_EGS_s{simpl}.nc`, `profile_EGS_s{simpl}.nc` |
| `build_electrical_demand` | `build_demand.py` | `resources/demand/` per-cluster demand |
| `add_demand` | `add_demand.py` | `resources/networks/elec_s{simpl}_dem.nc` |
| `build_powerplants` | `build_powerplants.py` | `resources/powerplants/powerplants.csv` |
| `build_fuel_prices` | `build_fuel_prices.py` | `resources/prices/` fuel-price tables |
| `build_cost_data` | `build_cost_data.py` | `resources/costs/costs_{year}.csv` |
| `add_electricity` | `add_electricity.py` | `resources/networks/elec_s{simpl}_l_pp.pkl` |

Renewable capacity-factor profiles come either from the pre-computed **GODEEEP**
dataset (default) or are computed with **Atlite** from ERA5 weather cutouts; either
way they are produced per cluster region (see {doc}`data-generators`). Demand is
disaggregated from balancing-authority or state level to cluster buses using
population- and load-based allocation factors and attached by `add_demand`
({doc}`data-demand`). `add_electricity` then attaches everything to the network:
existing thermal and hydro fleets from PUDL/EIA data, renewable profiles, fuel
prices, and technology costs. Its output is a pickled network because it carries
non-netCDF metadata used downstream.

### 4. Final clustering and scenario preparation

| Rule | Script | Key outputs |
|------|--------|-------------|
| `cluster_network` | `cluster_network.py` | `resources/networks/elec_s{simpl}_c{clusters}.nc`, `resources/busmaps/busmap_s{simpl}_{clusters}.csv` |
| `add_extra_components` | `add_extra_components.py` | `..._ec.nc` |
| `prepare_network` | `prepare_network.py` | `..._ec_l{ll}_{opts}.nc` |
| `add_sectors` | `build_sector.py` (sector runs) | `..._ec_l{ll}_{opts}_{sector}.nc` |

`cluster_network` reduces the network to its final `{clusters}` resolution (kmeans or
modularity), with optional suffixes controlling how existing transmission is treated
(see {doc}`config-spatial`). `add_extra_components` adds extendable storage
(battery durations, pumped hydro) and other investment options.
`prepare_network` applies the `{ll}` transmission-limit scenario and the `{opts}`
tokens — temporal resolution (`3h`, segmentation), emissions limits or prices, and
per-carrier cost/potential adjustments ({doc}`model-constraints`). Power-only runs
pass through `add_sectors` unchanged (`{sector}` = `E`); sector-coupled runs attach
the natural-gas, buildings, transport, and industry representations here
({doc}`data-sectors`).

### 5. Solve and postprocess

| Rule | Script | Key outputs |
|------|--------|-------------|
| `solve_network` | `solve_network.py` | `results/{run}/{interconnect}/networks/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc` |
| `plot_network_maps`, `plot_statistics`, ... | `plot_*.py` | `results/{run}/{interconnect}/figures/` |
| `export_statistics` | `export_statistics.py` | `results/{run}/{interconnect}/` summary CSVs |

`solve_network` builds the optimization problem with [Linopy](https://linopy.readthedocs.io/),
registers PyPSA-USA's custom constraints (policy standards, reserve margins, capacity
targets — see {doc}`model-constraints`), and hands it to the configured solver. The
solved network, per-carrier statistics, and standard figures land under `results/`,
organized by run name and interconnect.

:::{figure} _static/generated/example_outputs.png
:width: 100%
:alt: Optimal capacity by carrier and an example dispatch week from a solved network

Example outputs from a solved California test system: optimized capacity by carrier
(left) and a week of dispatch against load (right). Regenerate with
`snakemake docs_figures`.
:::

## Data retrieval

Upstream of everything sit the `retrieve_*` rules, which download and cache raw data
on first use:

| Rule | What it fetches |
|------|-----------------|
| `retrieve_zenodo_databundles` | Core data bundle (transmission networks, shapes, seed data) |
| `retrieve_pudl` | PUDL generator, plant, and fuel data (EIA 860/923 derived) |
| `retrieve_caiso_data` | CAISO fuel-price data |
| `retrieve_nrel_efs_data` | NREL Electrification Futures Study demand scenarios |
| `retrieve_eer_demand_data` | EER forecasted demand profiles (when `demand: profile: eer`) |
| `retrieve_gridemissions_data` | Historical grid emissions data |
| `retrieve_nrel_exclusion_artifact` | NREL reV-based land-access exclusion layers |
| `retrieve_res_eulp` / `retrieve_com_eulp` | NREL End-Use Load Profiles (sector runs) |
| `retrieve_sector_databundle` | Sector-coupling data bundle (sector runs) |
| `retrieve_seismic_risk_mask` | Seismic-risk exclusion mask (EGS runs) |

Historical EIA demand additionally requires a free
[EIA API key](https://www.eia.gov/opendata/) (see {doc}`about-install`).

## Running parts of the pipeline

Two targets are worth knowing beyond `all`:

- `data_model` builds everything up to the assembled, unsolved network — the full
  data pipeline with no solver required.
- `dag` regenerates the rule-graph image on this page
  (`workflow/repo_data/dag.jpg`).

Any intermediate file can also be requested directly, and `--until <rule>` /
`-R <rule>` stop early or force re-execution — see {doc}`about-usage` for worked
commands.
