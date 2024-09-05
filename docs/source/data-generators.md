(data-generators)=
# Generators

PyPSA-USA utilizes the [Public Utility Data Liberation (PUDL)](https://catalystcoop-pudl.readthedocs.io/en/latest/index.html) project database as the core source for generator and storage device data. The PUDL database aggregates and cleans data from various agencies, including the Energy Information Agency (EIA), Federal Energy Regulatory Commission (FERC), and the National Renewable Energy Laboratory (NREL). This integration supports reproducibility and ensures continuity as new reports are released.

## Generator Data Integration

PyPSA-USA integrates unit-level generator data from PUDL, which includes:

- **Heat Rates**
- **Plant Fuel Costs**
- **Seasonal Derating**
- **Power and Energy Capacities**
- **Fuel type and historical Costs**

## Thermal Unit Commitment and Ramping Constraints

To model thermal unit commitment and ramping constraints, data from the WECC Anchor Data Set (ADS) is incorporated. This dataset is used by transmission and system planners across the WECC region and includes:

- **Start-up and Shut-down Costs**
- **Minimum Up and Down Time**
- **Ramping Limits**

For plants outside the WECC, and for internal plants missing data, PyPSA-USA imputes values using capacity-weighted averages by technology type.

## Renewable Resource Constraints

Renewable resources like solar and wind are constrained by technical capacity limits based on land-use and resource characteristics. These limits are calculated using various land-use layers that progressively reduce the land available for resource development.

- **Solar and Wind Capacity Limits**: Determined by multiple land-use layers.
- **Geothermal and Pumped Hydro Storage (PHS)**: These resources require more complex modeling due to subsurface and surface characteristics. Regional supply curves for these resources, including capital costs and technical capacity, are incorporated from specialized datasets.
    - **PHS**: Uses data from the NREL Closed-Loop PHS dataset.
    - **Geothermal Resources**: Availability data is sourced from FGEM, with further details to be provided in a forthcoming paper.

## Fuel Costs

In production cost-minimizing optimization models, a generator’s marginal cost to produce electricity is a primary driver of dispatch decisions and electricity prices. However, generator fuel prices and efficiencies are not uniformly available across the United States, and generators often enter into bilateral contracts that are not directly correlated with wholesale fuel prices. To address these challenges, PyPSA-USA integrates fuel prices and unit-level fuel costs across varying spatial scopes and temporal scales.

- **Fuel Price Integration**:
    - Fuel prices are collected and overlaid to select the highest resolution available, defaulting to coarser data if necessary.
    - Single-point unit-level generator fuel efficiencies are sourced from a CEMS-based dataset (D. Suri et. al.) (citation inbound).
    - Monthly unit-level fuel prices and additional plant efficiencies are collected via PUDL EIA-923.

- **Data Imputation**:
    - Missing data is imputed using capacity-weighted averages calculated by NERC region and unit technology type.
    - Wholesale daily natural gas prices for fuel regions across the WECC are imputed using CAISO OASIS data.
    - Monthly fuel prices for coal and natural gas, spatially resolved by state, are supplemented by data from the EIA.
    - For technologies like biomass and nuclear, where fuel prices are not available from other sources, projected fuel costs from the NREL ATB are used.

- **Future Fuel Costs**:
    - Forecasted annual fuel prices are imported from the EIA's Annual Energy Outlook (AEO).

# Data
```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,22,22,22
   :file: datatables/generators.csv
```
