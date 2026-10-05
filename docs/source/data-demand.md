(data-demand)=
# Electricity Demand

PyPSA-USA offers access to both exogenously defined historical and future forecasted electrical demand data.

## Historical Demand

Historical demand data is imported from the EIA930 via the [GridEmissions](https://github.com/jdechalendar/gridemissions) tool, covering the years 2018-2023. This data is defined at the balancing area region level.

## Forecasted Demand

Forecasted demand is sourced from the NREL Electrification Futures Study (EFS), providing hourly demand forecasts for the years 2030, 2040, and 2050. The EFS data includes forecasts for varying levels and speeds of electrification across sectorally specified residential, commercial, and industrial end-uses. The non-sector coupled setting in pypsa-usa aggregates these demands to one load per node.

The EFS also provides electrification cases, with reference, medium, and high electrification cases, with slow, moderate, and rapid speeds. These scenarios can be controlled via the configuration `demand: scenario: efs_case: / efs_speed:`.

A third source, selected with `profile: eer`, provides state-level hourly demand scenarios from
the EER dataset (retrieved automatically from Zenodo). Three scenario files are available via
the `demand: scenario: eer_file:` setting: `demand_EER2025_100by2050`,
`demand_EER2025_Baseline_AEO2023`, and `demand_EER2025_IRAlow`. EER profiles are defined for
the model years 2021, 2025, 2030, 2035, 2040, 2045, and 2050, and each profile is indexed to a
historical weather year, so `renewable_weather_years` must contain exactly one year from
2007-2013 or 2016-2023.

(servm-demand)=
### CPUC SERVM (California)

`profile: servm` uses the hourly load forecast the California Public Utilities Commission
publishes for its 2026 Integrated Resource Planning cycle, produced with the SERVM
production-cost model. It is a **California-only** dataset — use it with a footprint scoped
to California (`model_topology: include: reeds_state: ['CA']`); the maintained entry point is
`workflow/repo_data/config/config.california.yaml`. The [California model](california-model.md)
page is the full reference for that configuration — every dataset that goes into it, the
weather-year options, and the caveats.

The workflow retrieves one CSV per forecast year from

```
https://files.cpuc.ca.gov/energy/modeling/2026_servm_updates/HourlyLoad_CA_Regions_V2025E_2224_Mon_{year}.csv
```

(~118 MB each, via `retrieve_cpuc_servm_load`). **Nine forecast years are published: 2026,
2028, 2030, 2032, 2035, 2037, 2040, 2042, and 2045.** Unlike EFS, SERVM demand is *not*
interpolated or AEO-scaled between published years, so `scenario: planning_horizons` must be
drawn from that set — any other year raises in `ReadServm`.

#### Regions

SERVM reports six California load regions. Each maps onto the balancing areas PyPSA-USA
carries on its buses (`workflow/repo_data/CPUC/servm_region_map.csv`):

| SERVM region | PyPSA-USA balancing area(s) | Notes |
| --- | --- | --- |
| `PGE` | `CISO-PGAE` | PG&E CAISO footprint |
| `SCE` | `CISO-SCE` | Includes Valley Electric Association's California load, per the CPUC data dictionary |
| `SDGE` | `CISO-SDGE` | San Diego Gas & Electric |
| `IID` | `IID` | Imperial Irrigation District |
| `LADWP` | `LDWP` | Los Angeles Department of Water and Power |
| `NCNC` | `BANC` + `TIDC` | Northern California non-CAISO: Balancing Authority of Northern California and Turlock Irrigation District |

Four balancing areas carry a deliberately **empty** region in the mapping file, so their
(small) load shares are dropped with a log message, while any *unknown* balancing area
introduced by upstream relabeling still hard-fails: `CISO-VEA` (Valley Electric Association, a
Nevada footprint) and the California slivers served by `PACW` (Siskiyou/Del Norte/Modoc, whose
load lives in CPUC's non-CA PACW region file), `WALC` (WAPA Desert Southwest, Colorado River)
and `NEVP` (NV Energy, Tahoe/CalNeva) — none of which the CPUC California-region files cover.

Because the SERVM regions are balancing areas rather than states, the demand is disaggregated
with a purpose-built weights table (`build_servm_load_weights`) instead of the generic
state/BA path: a cluster bus can straddle two SERVM regions (Los Angeles County holds both
`LDWP` and `CISO-SCE` buses), so a bus receives the sum of its share of every region it
overlaps. The underlying per-bus weights are still the `bus_allocation` weights described
below.

The same weights table also places California's *out-of-state contracted supply*: with
`electricity: remote_contracted_resources: enable: true`, each resource the CPUC ledger
attributes to a SERVM region is attached at that region's max-LAF bus. See
[Out-of-State Contracted Resources](remote_contracted_resources) under Generators & Storage
Units.

#### Components

Each file publishes several load components per region (`Load`, `BTMPV`, `EV`, `DATA_CEN`, ...)
alongside `Net Load`. **Only `Net Load` is dispatched against by the model** — it is what
remains after behind-the-meter PV and other embedded resources. Every published component is
nonetheless carried through on the `subsector` index level and written to the zonal artifact
`resources/<run>/demand/{interconnect}/power_zonal_components_s{simpl}.parquet`, so the
component split stays available for reporting. Components that exist for only some regions
(`EV` and friends are published for PGE/SCE/SDGE only) align to `NaN` there.

#### Weather years

Each forecast-year file stacks 25 weather years (2000-2024) of a full hourly year.
`electricity: demand: scenario: servm_weather_years:` selects which one to use. It takes a
list with **exactly one** entry; multiple entries are reserved for stochastic scenarios
(phase 3) and currently raise `NotImplementedError`, because the demand output path is not
weather-year specific.

**Set `servm_weather_years` equal to the top-level `renewable_weather_years`.** Drawing load
and wind/solar profiles from different weather years decorrelates them and will understate
both the peak-net-load and the flexibility need. A mismatch is permitted but logs a warning.

Not every SERVM weather year has a matching renewable profile: which years the GODEEEP dataset
can supply, and at what cost in land screening, is tabulated under
[Historical weather-year availability](godeeep_weather_years) and summarised for California on
the [California model](california-model.md) page.

#### Timezone and calendar caveats

SERVM strips are in **fixed Pacific Standard Time (UTC−8) with no daylight-saving
transition**. This is not stated in the source files; it was verified empirically from the
behind-the-meter PV solar-noon centroid, which sits at hour 12.52 in December and 12.68 in
July — a DST-observing series would move by a full hour between the two. The strips are
rolled forward 8 hours to UTC before being attached.

Two calendar misalignments are accepted, and are immaterial for an hourly
capacity-expansion model, but matter if you compare hour-for-hour against another source:

1. **Monday-start synthetic calendar.** SERVM lays each year's 8760 hours on a synthetic
   calendar that starts on a Monday, so weekday-versus-weekend hours do not line up with the
   real weekdays of the planning horizon.
2. **Leap weather years.** For a leap *weather* year the SERVM strip contains February 29 and
   omits December 31, while PyPSA-USA's snapshots do the opposite (`get_snapshots` drops
   February 29 from leap planning horizons). Every hour after February therefore lands one
   calendar day earlier than it sat in the source file.

The strip is mapped **positionally** onto the network's own per-period snapshots rather than
onto a synthesised `date_range` — the latter would run a day short of December 31 for the
leap planning horizons (2028, 2032, 2040). As a consequence each planning horizon must carry
exactly 8760 snapshots; a truncated snapshot window cannot be used with `profile: servm`.

## Demand Disaggregation

All of the demand sources above arrive at a coarser resolution than the network: EIA930
historical demand is reported per balancing area, while EFS and EER forecasts are reported per
state. `add_demand` distributes each zone's hourly profile across the network buses within that
zone using load-allocation factors. Every bus carries a demand weight (`load_weight`) assigned
in `build_base_network`, and `electricity: demand: bus_allocation:` selects how that weight is
computed:

- **`population`** (the default) weights each bus by 2020 Decennial Census county population.
  A county's population is split evenly across the substations in that county, and each
  substation's share is then split evenly across its buses (`build_bus_population.py`), so a
  multi-bus substation does not absorb a multiple of a single-bus substation's weight. Buses
  with no county assignment (offshore, unmapped) get weight 0.
- **`breakthrough`** reproduces the legacy behaviour by using the nominal-demand column (`Pd`)
  inherited from the 2016-vintage Breakthrough Energy base network.

Either way, a bus's allocation factor is its share of the summed weight in its state or
balancing area (state-level factors, `LAF_state`, are precomputed in `build_base_network` so
that states split across interconnects are handled consistently). These weights are summed as
the network is aggregated to substations and clustered to `{simpl}` resolution, so demand is
allocated at the same resolution the rest of the model is built at: `add_demand` reads
`elec_s{simpl}.nc` and writes `elec_s{simpl}_dem.nc` before generators are attached in
`add_electricity`. The county-level 2020 Decennial Census population data shown below is the
same source that underpins the population layers used elsewhere in the workflow, including the
urban/rural splits used in sector-coupling studies.

:::{figure} _static/pop_layout/population.png
:width: 90%
Resident population by county from the 2020 Decennial Census (DEC Demographic and Housing
Characteristics). Demand within each state or balancing area is allocated to buses in
proportion to the population they serve.
:::

:::{figure} _static/pop_layout/urban.png
:width: 90%
Urban share of housing units by county from the 2020 Decennial Census. Urban/rural population
splits drive the disaggregation of building demands in sector-coupling studies.
:::

## Usage

The user determines whether to use historical demand years via a combination of the planning
horizons setting, and the electricity demand setting. If conducting historical simulations, the
user must select a planning horizon in the past (2018-2023) and set `profile: eia`.

If conducting forward-looking planning cases the user must set a future planning horizon —
2030, 2040, or 2050 with `profile: efs`; any of 2021, 2025, 2030, 2035, 2040, 2045, and
2050 with `profile: eer`; or any of 2026, 2028, 2030, 2032, 2035, 2037, 2040, 2042, and 2045
with `profile: servm` (California only).

For planning horizons between the EFS data years, PyPSA-USA implements a scaling factor that
interpolates between future years or scales historical demand using forecasts from the Annual
Energy Outlook (AEO).

```
scenario:
  planning_horizons: [] # Historical (2018-2023) or future year(s)

electricity:
  demand:
    profile: efs # efs, eia, eer, servm
    scenario:
      efs_case: reference # reference, medium, high
      efs_speed: moderate # slow, moderate, rapid
      eer_file: demand_EER2025_100by2050.h5 # used when profile: eer
      servm_weather_years: [2019] # used when profile: servm; exactly one year, 2000-2024
      aeo: reference
```

### Data
```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,22,22,22
   :file: datatables/demand.csv
```
