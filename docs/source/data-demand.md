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

## Demand Disaggregation

All of the demand sources above arrive at a coarser resolution than the network: EIA930
historical demand is reported per balancing area, while EFS and EER forecasts are reported per
state. `add_demand` distributes each zone's hourly profile across the network buses within that
zone using population-based load-allocation factors. Every bus carries a demand weight (`Pd`)
inherited from the Breakthrough Energy base network, which reflects the population served by
each substation; a bus's allocation factor is its share of the summed weight in its state or
balancing area (state-level factors are precomputed in `build_base_network` so that states
split across interconnects are handled consistently). These weights are summed as the network
is aggregated to substations and clustered to `{simpl}` resolution, so demand is allocated at
the same resolution the rest of the model is built at: `add_demand` reads `elec_s{simpl}.nc`
and writes `elec_s{simpl}_dem.nc` before generators are attached in `add_electricity`. The
county-level 2020 Decennial Census population data shown below underpins the population layers
used for population-based allocation across the workflow, including the urban/rural splits used
in sector-coupling studies.

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
2030, 2040, or 2050 with `profile: efs`, or any of 2021, 2025, 2030, 2035, 2040, 2045, and
2050 with `profile: eer`.

For planning horizons between the EFS data years, PyPSA-USA implements a scaling factor that
interpolates between future years or scales historical demand using forecasts from the Annual
Energy Outlook (AEO).

```
scenario:
  planning_horizons: [] # Historical (2018-2023) or future year(s)

electricity:
  demand:
    profile: efs # efs, eia, eer
    scenario:
      efs_case: reference # reference, medium, high
      efs_speed: moderate # slow, moderate, rapid
      eer_file: demand_EER2025_100by2050.h5 # used when profile: eer
      aeo: reference
```

### Data
```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,22,22,22
   :file: datatables/demand.csv
```
