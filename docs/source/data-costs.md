(data-costs)=
# Costs
## Costs and Candidate Resources

 In PyPSA-USA, candidate resource forecasted capital and operating costs are defined by the NREL Annual Technology Baseline (ATB) accessed through the PUDL project. The model currently uses the 2024 ATB which provides data for expected costs across the years 2025 - 2050. The full ATB scenario grid is exported by `build_cost_data` for every planning horizon, and the configuration below selects which slice of that grid each carrier ultimately uses.

### Selecting an ATB scenario and model case

ATB exposes two orthogonal axes that bound the cost trajectory of every technology:

- **`scenario`** — `Moderate` (default), `Advanced`, or `Conservative`. Reflects how aggressively technology costs decline over time. `Advanced` is most optimistic; `Conservative` is most pessimistic.
- **`model_case`** — `Market` (default) or `R&D`. `Market` bakes current policy incentives (PTC/ITC) into the financing assumptions; `R&D` strips those out and isolates pure technology learning.

The global default applies to every carrier unless an override is set:

```yaml
costs:
  atb:
    model_case: "Market"   # Market, R&D
    scenario: "Moderate"   # Moderate, Advanced, Conservative
```

#### Per-carrier overrides

To stress-test specific carriers under a different scenario without touching the global default, add an `overrides` block keyed by the carrier's pypsa-name. Either field can be set independently — anything left out falls back to the global default:

```yaml
costs:
  atb:
    model_case: "Market"
    scenario: "Moderate"
    overrides:
      solar:
        scenario: "Advanced"            # cheaper solar
      onwind:
        scenario: "Advanced"
        model_case: "R&D"               # no-PTC view of onshore wind
      nuclear:
        scenario: "Conservative"        # higher-cost nuclear
```

Resolution order, highest priority first:
1. Per-carrier override under `costs.atb.overrides[<pypsa-name>]`
2. Global `costs.atb.scenario` / `costs.atb.model_case`
3. Hardcoded fallback (`Moderate` / `Market`)

Carrier names must match the keys in `ATB_TECH_MAPPER` (`workflow/scripts/constants.py`) — e.g. `solar`, `onwind`, `offwind`, `offwind_floating`, `nuclear`, `SMR`, `CCGT`, `CCGT-95CCS`, `coal`, `coal-95CCS`, `geothermal`, `biomass`, `4hr_battery_storage`, etc.

#### How selection happens

`build_cost_data` writes one row per `(pypsa-name, parameter, atb_scenario, atb_model_case)` into `resources/costs/costs_{year}.csv`, so every ATB combination is available without rerunning the rule. Scenario-independent rows (EGS supply curves, transmission costs, emissions factors) carry `NA` in the scenario columns and apply universally. The `load_costs` helper in `_helpers.py` reads the config and selects the right row per carrier before pivoting, so changing `costs.atb` only requires re-running the downstream rules — not `build_cost_data`.

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
