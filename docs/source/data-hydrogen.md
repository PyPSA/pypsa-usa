(data-hydrogen)=
# Hydrogen Sector
## Overview

The hydrogen sector is enabled by `H` in the `sector` wildcard, either on its own (`E-H`) or with
natural gas and the end-use sectors (`E-G-H`). Like natural gas, hydrogen is balanced at state
level: every state gets a hydrogen bus carrying its exogenous, non-power hydrogen demand. The
hydrogen used by the power sector is endogenous.

Hydrogen can be produced by the technologies listed in `sector: hydrogen: production`:

- electrolysis (`h2_electrolysis`) at every AC bus,
- steam methane reforming without (`h2_smr`) or with carbon capture (`h2_smr_ccs`), at every AC
  bus, drawing gas from the state gas bus and electricity from the AC bus.

It can be stored underground in each state (`storage`) and piped between states already
connected by AC transmission or natural gas pipelines (`pipelines`). New
storage is a salt cavern where the state overlaps the salt sites of ReEDS, else a hard rock cavern
where it overlaps the hard rock sites, else an underground pipe. New storage is separate from
the existing natural gas storage, which holds hydrogen only when retrofitted (below). Every store keeps 30% of its capacity as cushion gas
([EIA AEO2026 HMM Assumptions](https://www.eia.gov/outlooks/aeo/assumptions/pdf/HMM_Assumptions.pdf),
Table 6), with separately sized injection and withdrawal. The hydrogen turbines (`OCHT`, `CCHT`) are generators burning a drop-in fuel when
the hydrogen sector is off, extendable when listed in `electricity: extendable_carriers` (`OCHT`
by default); with it, they draw from the state hydrogen bus. Existing OCGT and CCGT can also be
retrofitted to them (`electricity: retrofit`, see {ref}`data-costs`). Hydrogen capacity is new,
with one vintage per investment period, or retrofitted from natural gas infrastructure (below).

The energy used to compress hydrogen into storage and along pipelines is counted as a loss of
hydrogen. As for natural gas, pipelines lose a share of the hydrogen they carry per distance, to
compression and leakage. The CO2 emitted by hydrogen production is tracked on a state `h2-co2`
bus. With CO2 storage (`co2: storage`), the CO2 captured goes to the storage of the bus; without
it, the captured CO2 is not tracked further. With natural gas (`E-G-H`), the methane leaking from
the gas the reformers draw is tracked like that of the other gas users.

## Retrofitted natural gas infrastructure

With natural gas (`E-G-H`), `sector: hydrogen: retrofit` lets the model retrofit natural gas
storage and pipelines to hydrogen. The natural gas infrastructure is sized once for all investment
periods, so it is retrofitted in the first one.

- `storage`: salt dome, depleted field and aquifer reservoirs. A retrofitted share of a reservoir
  holds 0.2551 times its natural gas energy as hydrogen ([Lackey et al., 2023](https://doi.org/10.1029/2022GL101420)), with the same cushion
  share, and its wells withdraw 0.8 (salt domes, assuming turbulent flow through the wells) or 0.4
  (depleted fields and aquifers, [Amid et al., 2016](https://doi.org/10.1016/j.ijhydene.2016.02.036))
  times their natural gas rate: the hydrogen withdrawal takes the share of the gas withdrawal that the hydrogen energy
  takes of the gas energy. New compressors inject the hydrogen through these wells, at most at
  the hydrogen withdrawal rate; the gas left injects at most at its own withdrawal rate. With the
  gas storage aggregated by state
  (`sector: natural_gas: storage_aggregation: state`), a state's storage takes the mean
  conversion factors and costs of its formations, weighted by their working gas (energy, store)
  or their withdrawal (rates, injection and withdrawal).
- `pipelines`: each natural gas pipeline between states can carry 0.6 times its natural gas
  capacity as hydrogen
  ([European Hydrogen Backbone, 2020](https://gasforclimate2050.eu/wp-content/uploads/2020/07/2020_European-Hydrogen-Backbone_Report.pdf)),
  in either direction, losing as much along the way as a new hydrogen
  pipeline.

What is retrofitted is taken from the natural gas capacity: the gas capacity left plus the
hydrogen capacity over its conversion factor is at most the existing gas capacity. The gas
storage or pipelines that can be retrofitted therefore always contract
(`sector: natural_gas: infra_contraction`), and may also retire.

## Sector models without natural gas

Every `sector` wildcard but `E` builds a sector coupled model. In `E-H`, the electricity demand
stays a single load per bus, as in the electricity only model. The power plants are connected to
state level fuel buses (as with `G`) to track their emissions. Without the natural gas network,
gas, coal and oil are bought, by the power plants and steam methane reforming alike, at the price
the electricity only model charges the power plants: the AEO2026 electric power price of each
investment period, the same in every state, or, with `conventional: dynamic_fuel_price`, the
capacity weighted fuel cost of the state's plants. CO2 limits and portfolio standards then
follow the sector formulation.

## Data Sources

```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,22,33
   :file: datatables/hydrogen_sources.csv
```

The cost and performance data, in 2023 USD (converted to `costs: dollar_year`) and on a higher heating value basis, are listed with
their sources and conversions in `workflow/repo_data/costs/hydrogen_costs.csv`.
