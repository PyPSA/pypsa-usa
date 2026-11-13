(data-costs)=
# Costs
## Costs and Candidate Resources

 In PyPSA-USA, candidate resource forecasted capital and operating costs are defined by the NREL Annual Technology Baseline (ATB) accessed through the PUDL project. The model currently uses the 2025 ATB; costs are taken for the years 2024 - 2050 (planning horizons earlier than 2024 use the 2024 costs, later ones the 2050 costs). The full ATB scenario grid is exported by `build_cost_data` for every planning horizon, and the configuration below selects which slice of that grid each carrier ultimately uses.

### Dollar year

Every cost in the model is in real US dollars of one base year, `costs: dollar_year` (2023 by default). The 2025 ATB is published in 2023 dollars and passes through unconverted; inputs in other dollar years, such as the AEO2026 fuel prices (2025 dollars) and the EIA building equipment costs (2022 dollars), are converted by `scripts/money.py`. A foreign-currency value is exchanged at the annual-average rate of its own year and then inflated in US dollars with the BLS CPI-U annual average. The CPI-U series comes from the [`cpi`](https://github.com/palewire/cpi) package and the exchange rates are annual averages of the ECB daily reference rates from the [`currencyconverter`](https://github.com/alexprengere/currencyconverter) package; both are pinned in the environment, so no table in the repository has to be kept current. Nominal time series are deflated by the year of each observation, so every investment period is priced in real base-year dollars. The config, including the dollar year, is stored on the solved network in `n.meta`.

### Selecting an ATB scenario and model case

ATB exposes two orthogonal axes that bound the cost trajectory of every technology:

- **`scenario`** — `Moderate` (default), `Advanced`, or `Conservative`. Reflects how aggressively technology costs decline over time. `Advanced` is most optimistic; `Conservative` is most pessimistic.
- **`model_case`** — `R&D + TC` (default), `R&D`, `Exp + TC`, or `Exp`. `R&D` uses ATB's research-and-development cost drivers; `Exp` uses its expanded cost drivers. `+ TC` adds the tax credits enacted as of July 2025 to the financing assumptions; without it, no tax credits are assumed.

ATB publishes the `Exp` cases only for utility PV, land-based wind, and utility-scale batteries. For every other carrier, `build_cost_data` fills an `Exp` slice from the matching `R&D` case (`Exp` → `R&D`, `Exp + TC` → `R&D + TC`), so selecting `Exp` never leaves a carrier without costs.

```{note}
The 2024 ATB's `Market` financial case no longer exists, and `R&D` now names a cost-driver case rather than a financial case. A config that still sets `model_case: "Market"` fails schema validation, and one that names it in `overrides` stops in `load_costs` with the list of available cases. `R&D + TC` is the closest equivalent of the old `Market` default, since both carry tax credits in the financing.
```

The global default applies to every carrier unless an override is set:

```yaml
costs:
  atb:
    model_case: "R&D + TC"   # R&D + TC, R&D, Exp + TC, Exp
    scenario: "Moderate"     # Moderate, Advanced, Conservative
```

#### Per-carrier overrides

To stress-test specific carriers under a different scenario without touching the global default, add an `overrides` block keyed by the carrier's pypsa-name. Either field can be set independently — anything left out falls back to the global default:

```yaml
costs:
  atb:
    model_case: "R&D + TC"
    scenario: "Moderate"
    overrides:
      solar:
        scenario: "Advanced"            # cheaper solar
      onwind:
        scenario: "Advanced"
        model_case: "R&D"               # no-tax-credit view of onshore wind
      nuclear:
        scenario: "Conservative"        # higher-cost nuclear
```

Resolution order, highest priority first:
1. Per-carrier override under `costs.atb.overrides[<pypsa-name>]`
2. Global `costs.atb.scenario` / `costs.atb.model_case`
3. Hardcoded fallback (`Moderate` / `R&D + TC`)

Carrier names must match the keys in `ATB_TECH_MAPPER` (`workflow/scripts/constants.py`) — e.g. `solar`, `onwind`, `offwind`, `offwind_floating`, `nuclear`, `SMR`, `CCGT`, `CCGT-95CCS`, `coal`, `coal-95CCS`, `geothermal`, `biomass`, `4hr_battery_storage`, etc.

#### How selection happens

`build_cost_data` writes one row per `(pypsa-name, parameter, atb_scenario, atb_model_case)` into `resources/costs/costs_{year}.csv`, so every ATB combination is available without rerunning the rule. Scenario-independent rows (EGS supply curves, transmission costs, emissions factors) carry `NA` in the scenario columns and apply universally. The `load_costs` helper in `_helpers.py` reads the config and selects the right row per carrier before pivoting, so changing `costs.atb` only requires re-running the downstream rules — not `build_cost_data`.

To reflect regional differences, capital costs are adjusted using [EIA state-level CapEx multipliers](https://www.eia.gov/analysis/studies/powerplants/capitalcost/pdf/capital_cost_AEO2020.pdf).

### Candidate Resources

- **Coal Plants**: With and without Carbon Capture Storage (CCS) at 95% and 99% capture rates.
- **Natural Gas**: Combustion Turbines and Combined Cycle plants, with and without 95% CCS.
- **Hydrogen Combustion Turbines**: Hydrogen Combusion Turbines are implemented under the assumption of market-available hydrogen drop-in fuel. Following the default assumptions in the [ReEDS Hydrogen implementation](https://nrel.github.io/ReEDS-2.0/model_documentation.html#drop-in-renewable-fuel). This implementation does not account for the energy or costs required to produce or transport the fuel. Future work will implement a more detailed production, transport, and storage model of hydrogen.
- **Nuclear Reactors**: Large Nuclear Reactors (AP1000) and Small Modular Reactors
- **Biomass with 95% Carbon Capture (`biomass-95CCS`)**: Capital cost, fixed and variable O&M and heat rate of Case 8 of the [Sargent & Lundy report for the EIA](https://www.eia.gov/analysis/studies/powerplants/capitalcost/pdf/capital_cost_AEO2025.pdf), a 50 MW bubbling fluidized bed burning wood chips with amine capture (2023 USD, converted to `costs: dollar_year`). The capital cost holds owner's costs, land and a one-mile grid connection, but no CO2 transport or storage. The WACC, biomass fuel cost, 30-year cost recovery period and 45-year lifetime are those of the dedicated biomass plant, and construction financing adds the share of the capital cost that it has for that plant. Each MMBtu of biomass burnt removes 0.089 t CO2 (0.303 t/MWh): 95% of the report's 206 lb CO2/MMBtu (40 CFR 98 Subpart C, Table C-1). In sector models the plants draw biomass from a state biomass bus and their removals count towards the power sector, whose emissions may be net negative. Not extendable by default.
- **Retrofits of Existing Gas Plants** (`electricity: retrofit`): as the upgrades of [ReEDS](https://github.com/ReEDS-Model/ReEDS) (`reeds/core/setup/b_inputs.gms`), existing OCGT can be retrofitted to OCHT, existing CCGT to CCHT or CCGT-90CCS, in every investment period they are still active in. A hydrogen retrofit costs 33% (OCHT) or 28% (CCHT) of the ATB overnight capital cost of a new OCGT or CCGT (`inputs/scalars.csv`, `cost_upgrade_gasct2h2ct` and `cost_upgrade_gascc2h2cc`); a carbon capture retrofit costs the ReEDS default `atb22_1p2_moderate` cost of `inputs/upgrades/upgrade_costs_ccs_gas.csv` (2004 USD, converted to `costs: dollar_year`, stored in `workflow/repo_data/costs/reeds_gas_ccs_retrofit_costs.csv`). Carbon capture retrofits capture 90% of the CO2, the ReEDS rate for retrofits (`GSw_CCS_Rate_Upgrade_mod`), with the heat rate and O&M of the ATB CCGT-95CCS plant (the ReEDS `Gas-CC-CCS_mod`). The retrofit runs for the configured `lifetime`; its cost is annualized with the ATB WACC of the technology retrofitted to, over that lifetime or the ATB cost recovery period, whichever is shorter. The retrofit keeps the plant's efficiency times the ATB efficiency of the technology retrofitted to over that of the plant's, takes the ATB fixed and variable O&M of the technology retrofitted to, and burns the plant's fuel (carbon capture) or hydrogen (the drop-in fuel above, or the state hydrogen in sector models with `H`). The plant and its retrofits share the plant's fuel input, so carbon capture lowers the plant's net output; the plants that can be retrofitted retire economically whatever `electricity: retirement`. Only existing plants are retrofitted, not new ones. Retrofits count towards clean energy standards as their technology does (OCHT, CCHT and CCGT-90CCS are eligible).
- **Renewable Energy**: Utility-scale onshore wind, fixed-bottom and floating offshore wind, utility-scale solar.
- **Battery Energy Storage**: 2-10 hour Battery Energy Storage Systems (BESS).
- **Pumped Hydro Storage (PHS)**: Supply curves for 8-12 hour PHS are integrated from the [NREL Closed-Loop PHS dataset](https://www2.nrel.gov/gis/psh-supply-curves).
- **Enhanced Geothermal Systems (EGS**): Methods for implementation will be released in a forthcoming paper.

## Fuel Costs

PyPSA-USA integrates fuel costs that varry across spatial scopes and temporal scales. For more information, see [here](./data-generators.md#fuel-costs-and-heat-rates)

## Sector Costs

Running sector studies will use the same power system costs as electrical only studies. Costs specific to each sector can be found in the [service sector](./data-services.md), [transportation sector](./data-transportation.md), and [industrial sector](./data-industrial.md) pages accordingly.
