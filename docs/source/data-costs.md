(data-costs)=
# Costs
## Costs and Candidate Resources

 In PyPSA-USA, candidate resource forecasted capital and operating costs are defined by the NREL Annual Technology Baseline (ATB) accessed through the PUDL project. The model currently uses the 2024 ATB which provides data for expected costs across the years 2025 - 2050. Users are able to configure which ATB model case and scenario to reference:

 ```yaml
   atb:
    model_case: "Market" # Market, R&D
    scenario: "Moderate" # Advanced, Conservative, Moderate
```

To reflect regional differences, capital costs are adjusted using [EIA state-level CapEx multipliers](https://www.eia.gov/analysis/studies/powerplants/capitalcost/pdf/capital_cost_AEO2020.pdf).

### Candidate Resources

- **Coal Plants**: With and without Carbon Capture Storage (CCS) at 95% and 99% capture rates.
- **Natural Gas**: Combustion Turbines and Combined Cycle plants, with and without 95% CCS.
- **Hydrogen Combustion Turbines**: Hydrogen Combusion Turbines are implemented under the assumption of market-available hydrogen drop-in fuel. Following the default assumptions in the [ReEDS Hydrogen implementation](https://nrel.github.io/ReEDS-2.0/model_documentation.html#drop-in-renewable-fuel). This implementation does not account for the energy or costs required to produce or transport the fuel. Future work will implement a more detailed production, transport, and storage model of hydrogen.
- **Nuclear Reactors**: Large Nuclear Reactors (AP1000) and Small Modular Reactors
- **Renewable Energy**: Utility-scale onshore wind, fixed-bottom and floating offshore wind, utility-scale solar.
- **Battery Energy Storage**: 2-10 hour Battery Energy Storage Systems (BESS).
- **Pumped Hydro Storage (PHS)**: Supply curves for 8-12 hour PHS are integrated from the [NREL Closed-Loop PHS dataset](https://www2.nrel.gov/gis/psh-supply-curves).
- **Enhanced Geothermal Systems (EGS**): Methods for implementation will be released in a forthcoming paper.

## Fuel Costs

PyPSA-USA integrates fuel costs that varry across spatial scopes and temporal scales. For more information, see [here](./data-generators.md#fuel-costs-and-heat-rates)

## Sector Costs

Running sector studies will use the same power system costs as electrical only studies. Costs specific to each sector can be found in the [service sector](./data-services.md), [transportation sector](./data-transportation.md), and [industrial sector](./data-industrial.md) pages accordingly.
