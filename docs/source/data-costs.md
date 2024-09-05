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

To reflect regional differences, capital costs are adjusted using [EIA state-level CapEx multipliers](https://www.eia.gov/analysis/studies/powerplants/capitalcost/pdf/capital_cost_AEO2020.pdf).

## Fuel Costs

PyPSA-USA integrates fuel costs that varry across spatial scopes and temporal scales. For more information, see [here](./data-generators.md#fuel-costs)

## Sector Costs

Running sector studies will use the same power system costs as electrical only studies. Costs specific to each sector can be found in the [service sector](./data-services.md), [transportation sector](./data-transportation.md), and [industrial sector](./data-industrial.md) pages accordingly.
