(data-demand)=
# Electricity Demand

PyPSA-USA offers access to both exogenously defined historical and future forecasted electrical demand data.

## Historical Demand

Historical demand data is imported from the EIA930 via the [GridEmissions](https://github.com/jdechalendar/gridemissions) tool, covering the years 2018-2023. This data is defined at the balancing area region level.

## Forecasted Demand

Forecasted demand is sourced from the NREL Electrification Futures Study (EFS), providing hourly demand forecasts for the years 2030, 2040, and 2050. The EFS data includes forecasts for varying levels and speeds of electrification across sectorally specified residential, commercial, and industrial end-uses. The non-sector coupled setting in pypsa-usa aggregates these demands to one load per node.

The EFS also provides electrification cases, with reference, medium, and high electrification cases, with slow, moderate, and rapid speeds. These scenarios can be controlled via the configuration `demand: scenario: efs_case: / efs_speed:`.

## Demand Disaggregation

Electrical load is disaggregated based on population, folling the implementation in the nodal network dataset. See the paper on the [nodal network](./data-transmission.md#tamu-synthetic-nodal-network) for more information on specifics of load disaggregation.

## Usage

The user determines weather to use historical demand years via a combination of the planning horizons setting, and the electricity demand setting. If conducting historical simulations, the user must select a planning horizon in the past (2018-2023), and set `profile: eia`.

If conducting forward-looking planning cases the user must set future planning_horizon year (2025- 2050) and set `profile: efs`.

For the years between 2030, 2040, and 2050, PyPSA-USA implements a scaling factor that interpolates between future years or scales historical demand using forecasts from the Annual Energy Outlook (AEO).

```
scenario:
  planning_horizons: [] # Historical or Future Year(s)

electricity:
  demand:
    profile: efs # efs, eia
    scenario:
      efs_case: reference # reference, medium, high
      efs_speed: moderate # slow, moderate, rapid
      aeo: reference
```

### Data
```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,22,22,22
   :file: datatables/demand.csv
```
