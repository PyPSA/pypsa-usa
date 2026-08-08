(model-components)=
# Networks & Components

A PyPSA-USA model is a [PyPSA](https://pypsa.readthedocs.io/en/latest/) network. PyPSA
defines the component classes — buses, generators, lines, links, storage — and their
optimization semantics; PyPSA-USA populates them with US data and conventions. This
page explains how each component is used here and how the model's spatial and temporal
resolution is put together. The authoritative reference for component attributes and
equations is the
[PyPSA components documentation](https://pypsa.readthedocs.io/en/latest/user-guide/components.html);
the custom columns PyPSA-USA adds on top are cataloged in {doc}`model-network-schema`.

## How PyPSA components are used

| PyPSA component | Use in PyPSA-USA |
|-----------------|------------------|
| `Bus` | A network zone (a clustered group of substations). Every bus carries geographic memberships — state, county, balancing authority, REeDS zone, NERC region — that policy constraints and reporting aggregate over. |
| `Carrier` | Technology/fuel labels (`solar`, `onwind`, `OCGT`, `coal`, `battery` durations, ...) plus emissions intensities used by CO2 accounting. |
| `Generator` | The existing thermal, hydro, and renewable fleet (from PUDL/EIA data) and candidate expansion capacity (`extendable_carriers`). Variable renewables carry per-snapshot capacity-factor profiles (`p_max_pu`); fuel-burning units carry marginal costs built from fuel prices and heat rates. |
| `Line` | AC transmission between zones, with impedance and thermal ratings aggregated through clustering. Expansion is controlled by the `{ll}` wildcard. |
| `Link` | HVDC ties in the power model; in sector-coupled runs, also every conversion process (heat pumps, electrolysis, EV charging, gas furnaces, ...). |
| `StorageUnit` | Power-sector storage with fixed energy/power ratio: battery storage at 2-10 hour durations and pumped hydro at 8-12 hours. |
| `Store` | Sector-coupled energy carriers (natural gas storage, CO2 accounting, fuel inventories) where energy and power are sized independently. |
| `Load` | Zonal electricity demand (and sector demands in sector-coupled runs), attached per bus and snapshot. |
| `GlobalConstraint` | System-wide limits such as emissions caps produced by the `{opts}` tokens and policy constraints ({doc}`model-constraints`). |

## Spatial structure

PyPSA-USA models one of four footprints, chosen by the `{interconnect}` wildcard:
`western`, `eastern`, `texas`, or `usa`. Within that footprint the model resolves
space at two configurable levels:

1. **`{simpl}` — the data resolution.** The nodal transmission network (~3,000-80,000
   buses depending on interconnect) is aggregated to substations and then clustered to
   `{simpl}` zones early in the pipeline. Renewable profiles, demand, and the
   generator fleet are all built at this resolution ({doc}`model-workflow`).
2. **`{clusters}` — the transmission resolution.** The final clustering step reduces
   the network to `{clusters}` zones, which is what the optimization sees. Suffixes
   (`m`, `a`, `c`) control how existing transmission capacity is carried into the
   clustered network ({doc}`config-spatial`).

:::{figure} _static/generated/network_aggregation.png
:width: 100%
:alt: The same network at nodal, simpl, and clusters resolution

The two-stage spatial aggregation on a California test system: the nodal base network
(left) is clustered to 20 `{simpl}` zones (center), at which resolution demand,
renewable profiles, and generators are built, and finally to 4 `{clusters}` zones
(right) for the optimization. Regenerate with `snakemake docs_figures`.
:::

Clustering respects administrative boundaries: with
`model_topology: topological_boundaries` set to `county`, `reeds_zone`, `state`, or
`balancing_area`, no cluster crosses a boundary of that type. This is what lets
state-level policies (RPS, emissions caps) and zonal interface limits stay well-defined
on the clustered network: every bus belongs unambiguously to a state, REeDS zone, and
balancing authority, recorded as bus attributes ({doc}`model-network-schema`).

## Temporal structure

- **Snapshots.** A model year is an hourly (8,760-snapshot) series, configured under
  `snapshots:`. Weather- and demand-data years are aligned by configuration
  (`renewable_weather_years`, demand profile selection — {doc}`data-demand`).
- **Temporal resolution.** Hourly snapshots can be coarsened — averaged to n-hourly
  (`3h`) or clustered into representative segments (`nSEG`) — either via
  `clustering: temporal:` in configuration or per-run with `{opts}` tokens
  ({doc}`config-wildcards`).
- **Planning horizons and foresight.** `scenario: planning_horizons:` selects the
  investment year(s). A single horizon gives a static expansion plan; multiple
  horizons run either with perfect foresight (one optimization over all years) or
  myopically (sequential years, each seeing only itself), set by `foresight:`
  ({doc}`config-configuration`).

## Investment and dispatch

In capacity-expansion mode, carriers listed under `electricity: extendable_carriers`
may build new capacity (`p_nom_opt` ≥ existing), with annualized capital costs from
the NREL ATB ({doc}`data-costs`) and operating costs from fuel prices and heat rates.
Everything else is dispatch-only. In production-cost mode no investment is allowed and
the existing fleet is dispatched against demand. The optimization itself — objective,
nodal balance, flow physics — is standard PyPSA; PyPSA-USA's additions are the custom
constraints documented in {doc}`model-constraints`.
