(data-generators)=
# Generators & Storage Units

PyPSA-USA utilizes the [Public Utility Data Liberation (PUDL)](https://catalystcoop-pudl.readthedocs.io/en/latest/index.html) project database as the core source for generator and storage device data. The PUDL database aggregates and cleans data from various agencies, including the Energy Information Agency (EIA), Federal Energy Regulatory Commission (FERC), and the National Renewable Energy Laboratory (NREL). This integration supports reproducibility and ensures continuity as new reports are released. The PUDL data is supplemented with data from the WECC Anchor Data Set (ADS) as well as the EIA API.

## Modeling Generators and Energy Storage

PyPSA-USA provides unit-level generator data on unit level Heat Rates, plant fuel costs seasonal derating, power and energy capacities, fuel types, and more. While generator data is input at a EIA unit-level, the model clusters generators by their technology type (named `carrier`) to reduce the computational cost of optimization models. PyPSA-USA generators are clustered to Combined-Cycle Gas Turbines (CCGT), Open-Cycle Gas Turbines (OCGT), coal, CCGTs with Carbon Capture and Storage (CCS), coal with CCS, oil, Hydrogen Combustion Turbines, Nuclear (large-scale AP1000), Small Modular Reactor Nuclear, biomass, traditional geothermal, waste, hydro, utility-scale solar, onshore wind, fixed-bottom offshore wind, floating offshore wind, and Enhanced Geothermal Systems (EGS). Storage Units include Li-ion battery energy storage systems (2-10 hour storage capacity) and Pumped-Hydro Storage (8-12 hour storage capacity). Users have control over the clustering settings using the configuration settings described in the [configuration section](./config-configuration.md)

## Fuel Costs and Heat-rates

In production cost-minimizing optimization models, a generator’s marginal cost to produce electricity is a primary driver of dispatch decisions and electricity prices. However, generator fuel prices and efficiencies are not uniformly available across the United States, and generators often enter into bilateral contracts that are not directly correlated with wholesale fuel prices. To address these challenges, PyPSA-USA provides a few options for the source of generator fuel prices, Generator heat-rates are also assimilated from multiple data-sources by selecting the highest-quality available data-source for a given generation unit before falling-back to coarser data.

- **Fuel Price Integration**:
    - Fuel prices are collected and overlaid to select the highest resolution available, defaulting to coarser data if necessary.
    - Single-point unit-level generator fuel efficiencies are sourced from a CEMS-based dataset [D. Suri et. al.](https://arxiv.org/pdf/2408.05209).
    - Monthly unit-level fuel prices and additional plant efficiencies are collected via PUDL EIA-923.

- **Data Imputation**:
    - Missing data is imputed using capacity-weighted averages calculated by NERC region and unit technology type.
    - Wholesale daily natural gas prices for fuel regions across the WECC are imputed using CAISO OASIS data.
    - Monthly fuel prices for coal and natural gas, spatially resolved by state, are supplemented by data from the EIA.
    - For technologies like biomass and nuclear, where fuel prices are not available from other sources, projected fuel costs from the NREL ATB are used.

- **Future Fuel Costs**:
    - Forecasted annual fuel prices are imported from the EIA's Annual Energy Outlook (AEO).

## Renewable Resources

**Solar & Wind Profiles**

PyPSA-USA leverages the Atlite tool to provide access to decades of weather data with varying spatial resolutions. Atlite is used to estimate hourly renewable resource availability across the United States, typically at a spatial resolution of 30 km² cells. Within PyPSA-USA, users can configure:

- **Weather Year**
- **Turbine Type**
- **Solar Array Type**
- **Land-Use Parameters**
- **Availability Simulation Parameters**

The hourly renewable capacity factors calculated by Atlite are weighted based on land-use availability factors. This ensures that areas unsuitable for specific technology types do not disproportionately affect the renewable resource capacity assigned to each node. These weighted capacity factors are aggregated into 41,564 distinct zones across the United States. These zones are then clustered using one of the clustering algorithms developed for PyPSA-Eur.

**Enhanced Geothermal (EGS) and Pumped Hydro Storage (PHS)**: These resources require more complex modeling due to subsurface and surface characteristics. Regional supply curves for these resources, including capital costs and technical capacity, are incorporated from specialized datasets.
    - **PHS**: Uses data from the [NREL Closed-Loop PHS dataset](https://www2.nrel.gov/gis/psh-supply-curves).
    - **EGS**: Availability data is sourced from [FGEM](https://fgem.readthedocs.io/en/latest/), with further details to be provided in a forthcoming paper.


# Data
```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,22,22,22
   :file: datatables/generators.csv
```
Renewables Data:
```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,22,22,22
   :file: datatables/renewables.csv
```
