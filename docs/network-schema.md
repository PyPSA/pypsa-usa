# PyPSA-USA Network Schema

Custom columns added to PyPSA components by PyPSA-USA scripts. PyPSA
built-in attributes are documented at
https://pypsa.readthedocs.io/en/latest/user-guide/components.html and
are not repeated here.

## Conventions

- **Aggregation strategy** is what to register in `bus_strategies` /
  `generator_strategies` when clustering. The default `consense` fails
  if values disagree within a cluster — use it only for columns
  guaranteed identical across any cluster.
- **NaN policy** records whether NaN is allowed at this stage and what
  it means semantically.
- A column appears here only after it has been observed in a
  `[schema ...]` log line during pipeline execution.

## Bus

| Column | dtype | Added by | Consumed by | Aggregation | NaN policy | Description |
|--------|-------|----------|-------------|-------------|------------|-------------|
| LAF_state | float | build_base_network.py | aggregate_to_substations.py, cluster_network.py, add_electricity.py, build_demand.py | sum | NaN = bus has no Pd / offshore (filled with 0 in build_demand) | Load Allocation Factor within a state; `Pd / sum(Pd) over full_state`. Not yet in `aggregate_to_substations.bus_strategies` or `cluster_network.bus_strategies` — falls back to `consense` and crashes; fix tracked separately. |
| Pd | float | build_base_network.py | aggregate_to_substations.py, cluster_network.py, build_demand.py | sum | NaN possible for offshore buses; filled with 0 in build_demand | Active power demand at the bus (MW) from the Breakthrough Energy source, used to disaggregate zonal demand to buses. |
| balancing_area | str | build_base_network.py (via ba_shape map) | aggregate_to_substations.py, build_powerplants.py, build_fuel_prices.py, build_demand.py, add_electricity.py, plot_validation_production.py | consense (safe — equal within substation cluster) | NaN if bus falls outside BA shapes; "Offshore" assigned in offshore-bus block | EIA Balancing Authority name covering the bus. |
| county | str | build_base_network.py (via county_shape GEOID map) | aggregate_to_substations.py, add_extra_components.py, cluster_network.py, build_bus_regions.py, build_demand.py | consense (safe) | NaN possible if shape lookup misses | County GEOID for the bus; dropped after substation aggregation when `topological_boundaries` is `reeds_zone` or `state`. |
| interconnect | str | build_base_network.py (sourced from input buses table) | aggregate_to_substations.py, cluster_network.py, build_natural_gas.py, build_heat.py, build_electricity_sector.py, add_extra_components.py, add_sectors.py, build_fuel_prices.py, plot_*.py | consense (safe within an interconnect run) | None expected | One of `western, eastern, texas, usa, Offshore` — drives interconnect-level filtering and lookups. |
| nerc_reg | str | build_base_network.py (`assign_reeds_memberships` via REeDS membership table) | aggregate_to_substations.py, cluster_network.py, plot_statistics.py, plot_validation_production.py | consense (safe — county-aligned via groupby mode) | NaN if county/reeds_zone lookup misses | NERC region (e.g. WECC, MRO) from REeDS memberships. |
| rec_trading_zone | str | build_base_network.py (`reeds_state.map(REC_TRADING_ZONE_MAPPER)`) | opts/policy.py | consense (safe — derived from reeds_state) | Falls back to `reeds_state` value if no REC zone defined | Renewable Energy Credit trading zone; used by RPS portfolio constraints. |
| reeds_ba | str | build_base_network.py (via reeds_shape map) | aggregate_to_substations.py, cluster_network.py, add_extra_components.py, build_bus_regions.py, build_electricity_sector.py | consense (safe — county-aligned via groupby mode) | NaN if shape lookup misses | REeDS balancing-area code (e.g. p10). |
| reeds_state | str | build_base_network.py (`reeds_zone.map(reeds_memberships.st)`) | aggregate_to_substations.py, cluster_network.py, add_extra_components.py, add_sectors.py, summary_sector.py, opts/policy.py | consense (safe) | NaN if reeds_zone is NaN | Two-letter state code from REeDS memberships. |
| reeds_zone | str | build_base_network.py (via reeds_shape map) | aggregate_to_substations.py, cluster_network.py, add_extra_components.py, build_bus_regions.py, build_electricity_sector.py, add_sectors.py | consense (safe — county-aligned via groupby mode) | NaN if shape lookup misses | REeDS zone code (e.g. p33). |
| state | str | build_base_network.py (via state_shape map) | aggregate_to_substations.py, add_extra_components.py, build_demand.py | consense (safe within substation) | NaN if shape lookup misses; `"Offshore"` for offshore buses | US state name covering the bus. |
| sub_id | int | build_base_network.py (`buses.sub_id.astype(int)`) | aggregate_to_substations.py, build_bus_regions.py, aggregate_egs.py, build_base_network.py (offshore matching) | consense (safe within a substation cluster — id is the cluster) | None expected | Breakthrough-Energy substation id; offshore buses get synthetic ids starting at 50000. |
| substation_off | bool | build_base_network.py (constant `False`; offshore buses set to `True`) | aggregate_to_substations.py (then dropped) | consense | Never NaN | Flag marking offshore-substation buses for downstream filtering. |
| trans_grp | str | build_base_network.py (`reeds_zone.map(reeds_memberships.transgrp)`) | aggregate_to_substations.py, cluster_network.py, add_extra_components.py | consense (safe — county-aligned via groupby mode) | NaN if reeds_zone lookup misses | REeDS transmission group code. |
| trans_reg | str | build_base_network.py (`reeds_zone.map(reeds_memberships.transreg)`) | aggregate_to_substations.py, add_extra_components.py | consense (safe — county-aligned via groupby mode) | NaN if reeds_zone lookup misses | REeDS transmission region code. |

## Generator

(no custom columns observed yet — extend as later-stage smoke tests are added)

## Line

| Column | dtype | Added by | Consumed by | Aggregation | NaN policy | Description |
|--------|-------|----------|-------------|-------------|------------|-------------|
| interconnect | str | build_base_network.py (`add_branches_from_file`) | cluster_network.py (drops/regenerates), plot_*.py | consense (safe within interconnect run) | None expected | Interconnect tag inherited from the source line table; dropped during `cluster_network` and rebuilt from bus assignments. |
| underwater_fraction | float | build_base_network.py (`add_branches_from_file`, constant `0.0`) | add_electricity.py (DC capex), cluster_network.py | consense (constant 0.0 for AC lines) | Never NaN (initialized 0) | Fraction of line length that runs underwater; non-zero values come from offshore wind / DC link logic in renewable profile and link builders. |

## Link

| Column | dtype | Added by | Consumed by | Aggregation | NaN policy | Description |
|--------|-------|----------|-------------|-------------|------------|-------------|
| underwater_fraction | float | build_base_network.py (`add_dclines_from_file`, constant `0.0`); reset to `0` for new links in aggregate_to_substations.py | add_electricity.py (HVDC overhead vs submarine capex blend) | consense | Never NaN (initialized 0) | Fraction of DC link length underwater; used to split HVDC capex between overhead and submarine costs. |

## Load

(no custom columns observed yet — extend as later-stage smoke tests are added)

## StorageUnit

(no custom columns observed yet — extend as later-stage smoke tests are added)

## Transformer

| Column | dtype | Added by | Consumed by | Aggregation | NaN policy | Description |
|--------|-------|----------|-------------|-------------|------------|-------------|
| interconnect | str | build_base_network.py (`add_branches_from_file`) | (read alongside lines during clustering) | consense (safe within interconnect run) | None expected | Interconnect tag inherited from the source branch table for transformer-class rows. |
| underwater_fraction | float | build_base_network.py (`add_branches_from_file`, constant `0.0`) | n/a (carried through but not consumed for transformers) | consense (constant 0.0) | Never NaN (initialized 0) | Inherited from the unified branch-loader; meaningless for transformers but present for schema consistency with Line. |
