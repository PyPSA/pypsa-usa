(data-costs)=
# Costs
## Costs and Candidate Resources

In investment optimization models, new energy devices can be built based on expected future costs, operational characteristics, technical capacity potential, and resource availability constraints. In PyPSA-USA, candidate resource expansion is guided by forecasted costs for technologies from the NREL Annual Technology Baseline (ATB).

### Implemented Candidate Resources

PyPSA-USA includes a variety of candidate resources, each with specific parameters:

- **Coal Plants**: With and without Carbon Capture Storage (CCS) at 95% and 99% capture rates.
- **Natural Gas**: Combustion Turbines and Combined Cycle plants, with and without 95% CCS.
- **Nuclear Reactors**: Small Modular Reactors (SMR) and AP1000.
- **Renewable Energy**: Utility-scale onshore wind, fixed-bottom and floating offshore wind, utility-scale solar.
- **Energy Storage**: 2-10 hour Battery Energy Storage Systems (BESS).
- **Pumped Hydro Storage (PHS)**: A method of storing energy by moving water between reservoirs at different elevations.

### Cost Parameters

The model uses forecasted data from the NREL ATB for:

- **Capital Expenditure (CapEx)**
- **Operations and Maintenance (O&M) Costs**
- **Capital Recovery Periods**
- **Fuel Efficiencies**
- **Weighted Average Cost of Capital (WACC)**

To reflect regional differences, capital costs are adjusted using EIA state-level CapEx multipliers.

### Fossil Generator Costs

In production cost-minimizing optimization models, a generator’s marginal cost to produce electricity is a primary driver of dispatch decisions and electricity prices. However, generator fuel prices and efficiencies are not uniformly available across the United States, and generators often enter into bilateral contracts that are not directly correlated with wholesale fuel prices. To address these challenges, PyPSA-USA integrates fuel prices and unit-level fuel costs across varying spatial scopes and temporal scales.

- **Fuel Price Integration**: 
    - Fuel prices are collected and overlaid to select the highest resolution available, defaulting to coarser data if necessary.
    - Single-point unit-level generator fuel efficiencies are sourced from a CEMS-based dataset [cite dhruv].
    - Monthly unit-level fuel prices and additional plant efficiencies are collected via PUDL. These prices are derived from EIA-923 reported unit-level monthly fuel delivery costs and plant efficiencies.
    - Data is filtered by Z-score and interquartile range to remove outliers, which might arise due to discrepancies between monthly fuel deliveries and fuel usage.

- **Data Imputation**:
    - Missing data is imputed using capacity-weighted averages calculated by NERC region and unit technology type.
    - Wholesale daily natural gas prices for fuel regions across the WECC are imputed using CAISO OASIS data.
    - Monthly fuel prices for coal and natural gas, spatially resolved by state, are supplemented by data from the EIA.
    - For technologies like biomass and nuclear, where fuel prices are not available from other sources, projected fuel costs from the NREL ATB are used.

- **Future Forecasting**:
    - Forecasted annual fuel prices are imported from the EIA's Annual Energy Outlook (AEO) and can be used to scale the higher-resolution historical daily and monthly fuel prices.

The impact of the selection of fuel price data is evident through the back-casting validation results presented in Section X. In clustered networks, which are often used in Capacity Expansion Models (CEM), the relative fuel costs of a small number of generators can significantly impact simulated emissions and the binding effects of emissions and production-based policy constraints. We continue to integrate the highest quality available fuel prices and explore these impacts in the results section. A summary of fuel cost data sources is available in below.

### Renewable Resource Constraints

Renewable resources like solar and wind are constrained by technical capacity limits based on land-use and resource characteristics. These limits are calculated using various land-use layers that progressively reduce the land available for resource development.

- **Solar and Wind Capacity Limits**: Determined by multiple land-use layers.
- **Geothermal and Pumped Hydro Storage (PHS)**: These resources require more complex modeling due to subsurface and surface characteristics. Regional supply curves for these resources, including capital costs and technical capacity, are incorporated from specialized datasets.
    - **PHS**: Uses data from the NREL Closed-Loop PHS dataset.
    - **Geothermal Resources**: Availability data is sourced from FGEM, with further details to be provided in a forthcoming paper.

For more details on the land-use layers and how they constrain resource development, see X section.
