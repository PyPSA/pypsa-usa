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

### Regional capital costs from ReEDS

With `costs: regional_multipliers: enable` (on by default), new generators and storage take regional capital costs from NREL ReEDS instead:

- **County overnight-capex multipliers**: `reg_cap_cost_diff_default.csv` gives each county a relative capex difference per ReEDS technology group, applied as `1 + x` to the overnight capex only; FOM, grid connection and construction financing are unchanged. `column_to_tech` maps the groups to carriers, and each bus takes the value of the county it lies in. A generator that clustering merges keeps the capacity-weighted mean of its members' multipliers, in every investment period; storage and carriers first added to the clustered network take the county of the clustered bus. Carriers not mapped keep the EIA state multipliers, and with `column_to_tech: null` no carrier takes a county multiplier.
- **Wind and solar interconnection cost** (`renewable_interconnection`, off by default): each ReEDS supply-curve site carries its spur-line, point-of-interconnection and network-reinforcement cost (the radial export cable offshore). `build_renewable_profiles` places the sites in the bus regions and takes their capacity-weighted mean per bus, in `costs: dollar_year`. It replaces the ATB's uniform grid-connection cost; buses without a site keep the ATB cost, and these carriers skip the EIA state multiplier. Where clustering merges generators with and without a site, only the share of the capacity that has one swaps the ATB grid connection for the ReEDS cost. The supply curves and site maps come from Zenodo (`rule retrieve_reeds_vre_file`) in the variant `renewable_land_access` selects, and fixed-bottom and floating offshore sites are split at the `renewable: offwind` depth limit with GEBCO. The mean covers every site in a bus region, uneconomic ones included, while builds go to the cheapest sites first; with large bus regions (coarse `simpl`) it therefore overstates the cost, which is why the default is `false`; turn it on for a fine `simpl`. Supply-curve bins that resolve this are left to [#849](https://github.com/PyPSA/pypsa-usa/issues/849).

Both are annualized at the carrier's capital charge rate, and each investment period prices its new builds with that year's costs. The ReEDS tables and their release are listed in `workflow/repo_data/costs/README.md`.

### Regional transmission costs from ReEDS

With `lines: regional_costs` (on by default), AC lines and HVDC links are priced from two ReEDS tables instead of the national `HVAC overhead` / `HVDC overhead` cost per km:

- **Route cost**: `transmission_distance_cost_500kVac_county.csv` gives the 500 kV AC cost of each county pair over the great-circle distance between the county centroids, so regional cost, terrain and route detour are in the price per great-circle km. A line takes the cost of its endpoint counties' pair; other lines, mostly within one county, take the mean of their two counties' typical cost (the median of each county's 8 shortest pairs), with state and national medians as fallbacks.
- **Voltage class** (`lines: voltage_factor`, on by default): the cost is scaled by `(kV / 500)^b`, fitted through 500 kV to the per-region voltage ratios of `rev_transmission_basecost.csv`. HVDC links take the DC/AC ratio of the same table (0.41).

Both tables are in 2004 USD and are converted to `costs: dollar_year`. The unit cost is set in `aggregate_to_substations`, while lines still have their own voltage and county, and is carried through clustering as a capacity-weighted mean. `capital_cost` is the annuity times that unit cost times the great-circle length of the clustered corridor; `lines: length_factor` is not applied, since the ReEDS prices include the detour. The submarine share of HVDC links and the converter stations keep their national costs.

### Annualized capital cost and tax credits

Capex (`capex_per_kw`, which holds construction financing) is annualized at `capital_charge_rate`, the ATB fixed charge rate: the capital recovery factor of the ATB WACC over the cost recovery period times ATB's project finance factor, which holds taxes and depreciation. Carriers ATB publishes no fixed charge rate for (gas, coal and oil plants, hydrogen_ct, transmission) are annualized at the capital recovery factor alone. The regional overnight multipliers and the ReEDS interconnection costs are annualized at the same rate.

By default the tax credits enter through a `+ TC` model case, with the credit of each commercial operation year, so ATB's phase-out carries over:

- **ITC** (utility PV, offshore wind, geothermal, hydropower, nuclear and SMR): lowers the fixed charge rate, and with it the capital cost. In the 2025 ATB wind and solar keep their credit for builds through 2029; geothermal, hydropower and nuclear keep it through 2037 and lose it by 2041.
- **PTC** (land-based wind): ATB deducts it from the LCOE only. `build_cost_data` recovers it as `ptc_per_mwh` from `LCOE = (FCR × capex + FOM) / (8760 × CF) + VOM + fuel − PTC`, a credit levelized over the cost recovery period, and `add_extra_components` takes it off the marginal cost of each extendable generator built in that year, in every period it operates. The existing fleet gets none.
- **Batteries**: ATB publishes no financing for them. They take the WACC and project finance factor of geothermal, which claims the same 48E ITC with the same technology-neutral phase-out, over their own 20-year cost recovery period.

Biomass gets no credit, as in the ATB. Cases without `+ TC` grant no credit.

#### Setting the tax credits

`costs: itc_modifier` and `costs: ptc_modifier` override the credits of the selected case per carrier, with one value for every build year or a mapping per build year (investment period):

```yaml
costs:
  itc_modifier:
    solar: 0.3                    # every build year
    offwind: {2030: 0.3, 2040: 0}  # 2035 is unset and keeps the ATB credit
  ptc_modifier:
    onwind: {2030: 27.5, 2035: 0}
```

A carrier or year that is not set, or set to `null`, keeps the credit of its ATB case; `0` sets no credit. The overrides are applied by `load_costs` (`scripts/tax_credits.py`), so every rule reads the same costs:

- **ITC** is the effective share of the capital charge the credit pays back. `build_cost_data` stores the ITC each ATB row holds as `itc_effective`: a `+ TC` case also changes the WACC of every technology, so the credit is read off ATB's project finance factor (fixed charge rate over capital recovery factor) against the case without tax credits, `1 − factor(+ TC) / factor`. A carrier with an ITC set is annualized at `capital_charge_rate × (1 − ITC) / (1 − itc_effective)`, which keeps the WACC of its case and replaces only the credit; FOM is unaffected, and `load_costs` logs both values. The effective ITC holds the depreciation basis reduction and the tax gross-up of the credit, so it differs from the statutory rate.
- **PTC** is the first-year credit in USD/MWh of `costs: dollar_year`, paid for 10 years and levelized over the cost recovery period at the carrier's WACC, the basis of the ATB-derived `ptc_per_mwh`.
- A project claims one of the two. A positive value of one drops the ATB value of the other for that carrier and year, unless both are set.

Carriers whose costs are not in the cost table (pumped hydro takes its capital cost from its supply curve) cannot be set here; `load_costs` warns about them.

### Candidate Resources

- **Coal Plants**: With and without Carbon Capture Storage (CCS) at 95% and 99% capture rates.
- **Natural Gas**: Combustion Turbines and Combined Cycle plants, with and without 95% CCS.
- **Hydrogen Combustion Turbines**: Hydrogen Combusion Turbines are implemented under the assumption of market-available hydrogen drop-in fuel. Following the default assumptions in the [ReEDS Hydrogen implementation](https://nrel.github.io/ReEDS-2.0/model_documentation.html#drop-in-renewable-fuel). This implementation does not account for the energy or costs required to produce or transport the fuel. Future work will implement a more detailed production, transport, and storage model of hydrogen.
- **Nuclear Reactors**: Large Nuclear Reactors (AP1000) and Small Modular Reactors
- **Biomass with 95% Carbon Capture (`biomass-95CCS`)**: Capital cost, fixed and variable O&M and heat rate of Case 8 of the [Sargent & Lundy report for the EIA](https://www.eia.gov/analysis/studies/powerplants/capitalcost/pdf/capital_cost_AEO2025.pdf), a 50 MW bubbling fluidized bed burning wood chips with amine capture (2023 USD, converted to `costs: dollar_year`). The capital cost holds owner's costs, land and a one-mile grid connection, but no CO2 transport or storage. The WACC, biomass fuel cost, 30-year cost recovery period and 45-year lifetime are those of the dedicated biomass plant, and construction financing adds the share of the capital cost that it has for that plant. Each MMBtu of biomass burnt removes 0.089 t CO2 (0.303 t/MWh): 95% of the report's 206 lb CO2/MMBtu (40 CFR 98 Subpart C, Table C-1). In sector models the plants draw biomass from a state biomass bus and their removals count towards the power sector, whose emissions may be net negative. Not extendable by default.
- **Renewable Energy**: Utility-scale onshore wind, fixed-bottom and floating offshore wind, utility-scale solar.
- **Battery Energy Storage**: 2-10 hour Battery Energy Storage Systems (BESS).
- **Pumped Hydro Storage (PHS)**: Supply curves for 8-12 hour PHS are integrated from the [NREL Closed-Loop PHS dataset](https://www2.nrel.gov/gis/psh-supply-curves).
- **Enhanced Geothermal Systems (EGS**): Methods for implementation will be released in a forthcoming paper.

## Fuel Costs

PyPSA-USA integrates fuel costs that varry across spatial scopes and temporal scales. For more information, see [here](./data-generators.md#fuel-costs-and-heat-rates)

## Sector Costs

Running sector studies will use the same power system costs as electrical only studies. Costs specific to each sector can be found in the [service sector](./data-services.md), [transportation sector](./data-transportation.md), and [industrial sector](./data-industrial.md) pages accordingly.
