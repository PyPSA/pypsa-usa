(data-costs)=
# Costs
## Costs and Candidate Resources

 In PyPSA-USA, candidate resource forecasted capital and operating costs are defined by the NREL Annual Technology Baseline (ATB) accessed through the PUDL project.

### Implemented Candidate Resources

PyPSA-USA includes a variety of candidate resources, each with specific parameters:

- **Coal Plants**: With and without Carbon Capture Storage (CCS) at 95% and 99% capture rates.
- **Natural Gas**: Combustion Turbines and Combined Cycle plants, with and without 95% CCS.
- **Nuclear Reactors**: Small and Large Nuclear Reactors
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

### Renewable Resource Constraints

Renewable resources like solar and wind are constrained by technical capacity limits based on land-use and resource characteristics. These limits are calculated using various land-use layers that progressively reduce the land available for resource development.

- **Solar and Wind Capacity Limits**: Determined by multiple land-use layers.
- **Geothermal and Pumped Hydro Storage (PHS)**: These resources require more complex modeling due to subsurface and surface characteristics. Regional supply curves for these resources, including capital costs and technical capacity, are incorporated from specialized datasets.
    - **PHS**: Uses data from the NREL Closed-Loop PHS dataset.
    - **Geothermal Resources**: Availability data is sourced from FGEM, with further details to be provided in a forthcoming paper.
