"""Combines all time independent cost data sources into a standard format."""

import logging

import constants as const
import duckdb
import numpy as np
import pandas as pd
from _helpers import calculate_annuity
from build_sector_costs import (
    EfsIceTransportationData,
    EfsTechnologyData,
    EiaBuildingData,
)
from eia import ProjectedFuelCost
from money import deflator, dollar_year_from_config, parse_unit_year

logger = logging.getLogger(__name__)

# Impute emissions factor data
# https://www.eia.gov/environment/emissions/co2_vol_mass.php
# Units: [tCO2/MWh_thermal]
EMISSIONS_DATA = [
    {"pypsa-name": "coal", "parameter": "co2_emissions", "value": 0.3453},
    {"pypsa-name": "oil", "parameter": "co2_emissions", "value": 0.34851},
    {"pypsa-name": "geothermal", "parameter": "co2_emissions", "value": 0.04029},
    {"pypsa-name": "waste", "parameter": "co2_emissions", "value": 0.1016},
    {"pypsa-name": "gas", "parameter": "co2_emissions", "value": 0.18058},
    {"pypsa-name": "CCGT", "parameter": "co2_emissions", "value": 0.18058},
    {"pypsa-name": "OCGT", "parameter": "co2_emissions", "value": 0.18058},
    {
        "pypsa-name": "geothermal",
        "parameter": "heat_rate_mmbtu_per_mwh",
        "value": 8.881,
    },  # AEO 2023
]

LIFETIME_DATA = [
    {"pypsa-name": "coal", "parameter": "lifetime", "value": 70},
    {"pypsa-name": "oil", "parameter": "lifetime", "value": 55},  # using gas CT
    # Confirm with Jabs / NREL. 30 is way too small
    {"pypsa-name": "geothermal", "parameter": "lifetime", "value": 70},
    {"pypsa-name": "waste", "parameter": "lifetime", "value": 55},  # using gas CT
    {"pypsa-name": "CCGT", "parameter": "lifetime", "value": 55},
    {"pypsa-name": "OCGT", "parameter": "lifetime", "value": 55},
    {"pypsa-name": "CCGT-95CCS", "parameter": "lifetime", "value": 55},
    {"pypsa-name": "CCGT-97CCS", "parameter": "lifetime", "value": 55},
    {"pypsa-name": "coal-95CCS", "parameter": "lifetime", "value": 70},
    {"pypsa-name": "coal-99CCS", "parameter": "lifetime", "value": 70},
    {"pypsa-name": "SMR", "parameter": "lifetime", "value": 40},
    {"pypsa-name": "nuclear", "parameter": "lifetime", "value": 60},
    {"pypsa-name": "biomass", "parameter": "lifetime", "value": 30},
    {"pypsa-name": "offwind_floating", "parameter": "lifetime", "value": 30},
    {"pypsa-name": "offwind", "parameter": "lifetime", "value": 30},
    {"pypsa-name": "onwind", "parameter": "lifetime", "value": 30},
    {"pypsa-name": "solar", "parameter": "lifetime", "value": 30},
    {
        "pypsa-name": "2hr_battery_storage",
        "parameter": "lifetime",
        "value": 20,
    },  # inquired with NREL on why they have CRP of 20 but lifetime of 15
    {"pypsa-name": "4hr_battery_storage", "parameter": "lifetime", "value": 20},
    {"pypsa-name": "6hr_battery_storage", "parameter": "lifetime", "value": 20},
    {"pypsa-name": "8hr_battery_storage", "parameter": "lifetime", "value": 20},
    {"pypsa-name": "10hr_battery_storage", "parameter": "lifetime", "value": 20},
]  # https://github.com/NREL/ReEDS-2.0/blob/e65ed5ed4ffff973071839481309f77d12d802cd/inputs/plant_characteristics/maxage.csv#L4

# NREL ATB release read from PUDL (first carried by PUDL v2026.9.0). The 2025 ATB
# replaced the Market / R&D financial cases with cost-driver cases crossed with a
# tax-credit flag: R&D, R&D + TC, Exp (expanded cost drivers) and Exp + TC.
ATB_REPORT_YEAR = 2025

# Exp cases are only published for some technologies (utility PV, land-based wind,
# utility-scale batteries); every other carrier takes the R&D cost-driver case with
# the same tax-credit treatment.
ATB_MODEL_CASE_FALLBACK = {"Exp": "R&D", "Exp + TC": "R&D + TC"}


def create_duckdb_instance():
    """Set up DuckDB to read parquet files directly."""
    duckdb.connect(database=":memory:", read_only=False)
    # Install httpfs extension to access remote files if needed
    duckdb.query("INSTALL httpfs;")


def load_pudl_atb_data(parquet_path: str):
    """Loads ATB data directly from parquet files."""
    create_duckdb_instance()

    query = f"""
    WITH finance_cte AS (
        SELECT
            wacc_real,
            technology_description,
            model_case_nrelatb,
            scenario_atb,
            projection_year,
            cost_recovery_period_years,
            report_year
        FROM read_parquet('{parquet_path}/core_nrelatb__yearly_projected_financial_cases_by_scenario.parquet')
    )
    SELECT *
    FROM read_parquet('{parquet_path}/core_nrelatb__yearly_projected_cost_performance.parquet') atb
    LEFT JOIN finance_cte AS finance
        ON atb.technology_description = finance.technology_description
            AND atb.model_case_nrelatb = finance.model_case_nrelatb
            AND atb.scenario_atb = finance.scenario_atb
            AND atb.projection_year = finance.projection_year
            AND atb.cost_recovery_period_years = finance.cost_recovery_period_years
            AND atb.report_year = finance.report_year
    WHERE atb.report_year = {ATB_REPORT_YEAR}
    """
    return duckdb.query(query).to_df()


def match_technology(row, tech_dict):
    for key, value in tech_dict.items():
        # Match technology and techdetail
        if row["technology_description"] == value.get("technology") and row[
            "technology_description_detail_1"
        ] == value.get("techdetail"):
            return key
        # Match technology and techdetail2
        elif row["technology_description"] == value.get("technology") and row[
            "technology_description_detail_2"
        ] == value.get("techdetail2"):
            return key

    return None


# Temperature rise (K) of the water in a storage water heater, from the DOE test
# procedures behind the EIA efficiencies (EIA, Updated Buildings Sector Appliance
# and Equipment Costs and Efficiencies, 2023; page numbers as printed)
# https://www.eia.gov/analysis/studies/buildings/equipcosts/pdf/full.pdf
# - residential units are rated by UEF (EIA pp. 56, 59, 62 for gas, oil, electric),
#   tested 58 -> 125 F (10 CFR 430 Subpart B, Appendix E, sections 2.3-2.4)
#   https://www.ecfr.gov/current/title-10/chapter-II/subchapter-D/part-430/subpart-B/appendix-Appendix%20E%20to%20Subpart%20B%20of%20Part%20430
# - commercial units are rated by thermal efficiency (EIA pp. 148, 151, 156 for gas,
#   electric, oil), tested 70 -> 140 F (10 CFR 431 Subpart G, Appendix A, sections
#   3.1.2 and 5.1.2)
#   https://www.ecfr.gov/current/title-10/chapter-II/subchapter-D/part-431/subpart-G/appendix-Appendix%20A%20to%20Subpart%20G%20of%20Part%20431
WATER_HEATER_DELTA_T = {"Residential": 67 * 5 / 9, "Commercial": 70 * 5 / 9}
KJ_PER_GALLON_KELVIN = 3.785 * 4.186  # 3.785 kg of water per gallon, cp of 4.186 kJ/kg/K


def size_water_heaters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prices storage water heaters per MW of input and adds the energy to power ratio of their tank.

    A storage water heater costs a fixed amount per unit (USD/unit), for a unit of the
    given tank volume (gal) and input capacity (kW). The energy to power ratio is the
    heat the tank holds across its test temperature rise, in MWh per MW of input.
    """
    sizes = df[df.parameter.isin(["tank volume", "input capacity"])]
    sizes = sizes.pivot(index="technology", columns="parameter", values="value")

    per_unit = df.unit == "USD/unit"
    techs = df.technology[per_unit]
    assert techs.isin(sizes.index).all(), "No tank volume and input capacity for a USD/unit cost"
    df.loc[per_unit, "value"] /= techs.map(sizes["input capacity"]).to_numpy() / 1e3
    df.loc[per_unit, "unit"] = "USD/MW"

    delta_t = sizes.index.str.split().str[0].map(WATER_HEATER_DELTA_T).to_numpy()
    tank_mwh = sizes["tank volume"] * KJ_PER_GALLON_KELVIN * delta_t / 3.6e6
    ratio = tank_mwh / (sizes["input capacity"] / 1e3)

    rows = df[df.parameter == "tank volume"].copy()
    rows["parameter"] = "energy to power ratio"
    rows["value"] = rows.technology.map(ratio).to_numpy()
    rows["unit"] = "MWh/MW"
    rows["further_description"] = "Heat the tank holds across its test temperature rise per MW of input capacity"
    return pd.concat([df, rows])


def get_sector_costs(
    efs_tech_costs: str,
    efs_icev_costs: str,
    eia_tech_costs,
    year: int,
    dollar_year: int,
    additional_costs_csv: str | None = None,
) -> pd.DataFrame:
    """Gets end-use tech costs for sector coupling studies, in US dollars of ``dollar_year``."""

    def correct_units(df: pd.DataFrame) -> pd.DataFrame:
        df = size_water_heaters(df)

        df.unit = df.unit.str.replace("$/", "USD/")

        df.loc[df.unit.str.contains("/kW"), "value"] *= 1e3
        df.unit = df.unit.str.replace("/kW", "/MW")

        # 1 euro = 1.11 USD
        # taked on Sept. 2, 2024

        df.loc[df.unit.str.contains("EUR/"), "value"] *= 1.11
        df.unit = df.unit.str.replace("EUR/", "USD/")

        return df

    def get_investment_year_data(df: pd.DataFrame, year: int) -> pd.DataFrame:
        return df[df.year == year].drop(columns="year")

    def calculate_capex(df: pd.DataFrame, discount_rate: float) -> pd.DataFrame:
        """Calcualtes capex based on annuity payments."""
        capex = df.copy().set_index(["technology", "parameter"])
        capex = capex.value.unstack().fillna(0)

        # n years should be
        # n.snapshot_weightings.loc[n.investment_periods[x]].objective.sum() / const.HOURS_PER_YEAR

        capex["capital_cost"] = (
            (
                calculate_annuity(
                    capex["lifetime"],
                    discount_rate,
                )
                + capex["FOM"] / 100
            )
            * capex["investment"]
            * 1
        )

        capex = capex["capital_cost"].dropna().to_frame(name="value")

        investment = df[df.parameter == "investment"].set_index("technology")

        assert len(capex) == len(investment)

        final = capex.reindex_like(investment)
        final["parameter"] = "capital_cost"
        final["unit"] = investment.unit
        final["source"] = investment.source
        final["further_description"] = investment.further_description

        return final

    bev = EfsTechnologyData(efs_tech_costs).get_data("Transportation")
    ice = EfsIceTransportationData(efs_icev_costs).get_data()
    builidng = EiaBuildingData(eia_tech_costs, dollar_year).get_data()

    df = pd.concat([bev, ice, builidng])
    df = get_investment_year_data(df, year)
    df = df.rename(columns={"further description": "further_description"})

    if additional_costs_csv:
        additional = pd.read_csv(additional_costs_csv)
        assert all(x in df.columns for x in additional.columns)
        df = pd.concat([df, additional]).reset_index(drop=True)

    df = correct_units(df)

    discount_rate = 0.05
    capex = calculate_capex(df, discount_rate)
    capex = capex.reset_index()[df.columns]

    final = pd.concat([df, capex])
    final["value"] = final.value.round(4)

    final = final.rename(columns={"technology": "pypsa-name"})

    return final


# Transmission cost assumptions are independent of ATB scenario / model case.
# TEPCC 2023; WACC & Lifetime from https://emp.lbl.gov/publications/improving-estimates-transmission
# Subsea: Purvins et al. (2018): https://doi.org/10.1016/j.jclepro.2018.03.095
TRANSMISSION_DATA = [
    {"pypsa-name": "HVAC overhead", "parameter": "capex_per_mw_km", "value": 1541},
    {
        "pypsa-name": "HVAC overhead",
        "parameter": "cost_recovery_period_years",
        "value": 60,
    },
    {"pypsa-name": "HVAC overhead", "parameter": "wacc_real", "value": 0.044},
    {"pypsa-name": "HVDC overhead", "parameter": "capex_per_mw_km", "value": 1026.53},
    {
        "pypsa-name": "HVDC overhead",
        "parameter": "cost_recovery_period_years",
        "value": 60,
    },
    {"pypsa-name": "HVDC overhead", "parameter": "wacc_real", "value": 0.044},
    {"pypsa-name": "HVDC submarine", "parameter": "capex_per_mw_km", "value": 504.141},
    {
        "pypsa-name": "HVDC submarine",
        "parameter": "cost_recovery_period_years",
        "value": 60,
    },
    {"pypsa-name": "HVDC submarine", "parameter": "wacc_real", "value": 0.044},
    {
        "pypsa-name": "HVDC inverter pair",
        "parameter": "capex_per_kw",
        "value": 173.730,
    },
    {
        "pypsa-name": "HVDC inverter pair",
        "parameter": "cost_recovery_period_years",
        "value": 60,
    },
    {"pypsa-name": "HVDC inverter pair", "parameter": "wacc_real", "value": 0.044},
]


# Technologies paying the ATB fuel cost of their scenario rather than an AEO price
ATB_FUEL_CARRIERS = ["nuclear", "SMR", "biomass"]

# Biomass power with carbon capture: repo_data/costs/reeds_beccs_reference.csv holds
# the "beccs_mod" rows of ReEDS inputs/plant_characteristics/beccs_reference.csv, in 2010 USD
USD_2010_2_USD_2023 = 304.703 / 218.076  # CPI-U annual mean (FRED CPIAUCSL), 2023 / 2010

ATB_VALUE_COLS = [
    "cost_recovery_period_years",
    "capacity_factor",
    "capex_per_kw",
    "capex_overnight_per_kw",
    "capex_overnight_additional_per_kw",
    "capex_grid_connection_per_kw",
    "capex_construction_finance_factor",
    "fuel_cost_per_mwh",
    "heat_rate_mmbtu_per_mwh",
    "heat_rate_penalty",
    "levelized_cost_of_energy_per_mwh",
    "net_output_penalty",
    "opex_fixed_per_kw",
    "opex_variable_per_mwh",
    "wacc_real",
]


def build_aeo_fuel_costs(api: str, tech_year: int, aeo_scenario: str, dollar_year: int) -> pd.DataFrame:
    """
    Loads AEO fuel cost data, applies tech mappings, returns long-format frame.

    The fuel costs are the AEO2026 electric power prices of the tech year (see
    ``eia.ProjectedFuelCost``) in $/MMBtu of the dollar year named in their
    unit, converted to $/MWh_th of the model's ``dollar_year``. Years outside
    the AEO years take the price of the nearest year.
    """
    rows = []
    for name, fuel in {"natural_gas": "gas", "coal": "coal", "distillate_fuel_oil": "oil"}.items():
        df = ProjectedFuelCost(fuel, "power", api, aeo_scenario).get_data()
        units = df.units.unique().tolist()
        if len(units) != 1:
            raise ValueError(f"AEO {fuel} prices mix units: {units}")
        source_year, unit = parse_unit_year(units[0])
        if unit != "$/MMBtu":
            raise ValueError(f"Expected AEO {fuel} prices in $/MMBtu, got {units[0]}")
        year = min(df.index, key=lambda x: abs(x - tech_year))
        if year != tech_year:
            logger.warning(f"No AEO2026 {fuel} price for {tech_year}; using {year}")
        rows.append(
            {
                "pypsa-name": name,
                "parameter": "fuel_cost_real_per_mwhth",
                "value": df.value.at[year] * deflator(source_year, dollar_year) * 3.412,
            },
        )
    aeo = pd.DataFrame(rows)

    tech_fuel_map = {
        "CCGT": "natural_gas",
        "OCGT": "natural_gas",
        "CCGT-95CCS": "natural_gas",
        "CCGT-97CCS": "natural_gas",
        "coal-95CCS": "coal",
        "coal-99CCS": "coal",
        "oil": "distillate_fuel_oil",
    }
    tech_fuels = pd.DataFrame(
        [
            {
                "pypsa-name": new_name,
                "parameter": "fuel_cost_real_per_mwhth",
                "value": aeo.loc[aeo["pypsa-name"] == source_name, "value"].values[0],
            }
            for new_name, source_name in tech_fuel_map.items()
        ],
    )
    return pd.concat([aeo, tech_fuels], ignore_index=True)


def complete_atb_slice(pudl_atb_year: pd.DataFrame, atb_scenario: str, atb_model_case: str) -> pd.DataFrame:
    """Rows of one (atb_scenario, model_case) slice, completed from its fallback case.

    Every carrier ATB covers in ``atb_scenario`` must end up in the slice: one left
    out would reach ``load_costs`` without capex and be zero-filled there. Returns an
    empty frame when ATB does not publish the case for this scenario at all.
    """
    in_scenario = pudl_atb_year[pudl_atb_year.scenario_atb == atb_scenario]
    sub = in_scenario[in_scenario.model_case_nrelatb == atb_model_case]
    if sub.empty:
        return sub

    fallback = ATB_MODEL_CASE_FALLBACK.get(atb_model_case)
    if fallback is not None:
        fill = in_scenario[
            (in_scenario.model_case_nrelatb == fallback) & ~in_scenario["pypsa-name"].isin(sub["pypsa-name"])
        ]
        if not fill.empty:
            logger.info(
                f"ATB {atb_scenario} / {atb_model_case} is not published for "
                f"{sorted(fill['pypsa-name'].unique())}; using {fallback}",
            )
            sub = pd.concat([sub, fill], ignore_index=True)

    missing = set(in_scenario["pypsa-name"]) - set(sub["pypsa-name"])
    if missing:
        raise ValueError(f"ATB {atb_scenario} / {atb_model_case} has no data for {sorted(missing)}")
    return sub


def get_beccs_reference(beccs_reference: str, year: int) -> pd.Series:
    """
    ReEDS BECCS costs and heat rate of ``year``, in 2023 USD.

    Overnight capital cost (USD/kW), fixed O&M (USD/kW-year), variable O&M
    (USD/MWh) and heat rate (MMBtu/MWh) of the ReEDS ``beccs_mod`` plant,
    linearly interpolated between the ReEDS years and held constant outside them.
    """
    df = pd.read_csv(beccs_reference, index_col="t").sort_index()
    data = df[["capcost", "fom", "vom", "heatrate"]].apply(lambda x: float(np.interp(year, x.index, x)))
    data[["capcost", "fom", "vom"]] *= USD_2010_2_USD_2023
    return data


def build_beccs(pivot_atb: pd.DataFrame, beccs: pd.Series) -> pd.DataFrame:
    """
    Biomass power with carbon capture (biomass-CCS) from dedicated biomass power (ATB Biopower).

    It takes the capital cost, O&M and heat rate of the ReEDS BECCS plant
    (``get_beccs_reference``) and the rest of the biomass plant: its fuel cost
    per MWh of biomass, grid connection, financing, cost recovery period and
    lifetime. The ReEDS capital cost is an overnight cost; the construction
    financing cost of biomass (``capex_construction_finance_factor``, USD/kW in
    the ATB) is added to it as it is, like the grid connection.
    Every MWh of biomass burnt removes ``constants.BECCS_CO2_REMOVAL`` of CO2.
    """
    df = pivot_atb[pivot_atb["pypsa-name"] == "biomass"].copy()
    if df.empty:
        return df
    df["pypsa-name"] = "biomass-CCS"
    grid = df["capex_grid_connection_per_kw"].fillna(0)
    df["capex_overnight_per_kw"] = beccs.capcost
    df["capex_per_kw"] = df["capex_overnight_per_kw"] + grid + df["capex_construction_finance_factor"]
    df["opex_fixed_per_kw"] = beccs.fom
    df["opex_variable_per_mwh"] = beccs.vom
    df["heat_rate_mmbtu_per_mwh"] = beccs.heatrate
    df["co2_emissions"] = -const.BECCS_CO2_REMOVAL
    # ATB figures of the biomass plant that do not hold for it
    df[["fuel_cost_per_mwh", "capacity_factor", "levelized_cost_of_energy_per_mwh"]] = np.nan
    return df


def process_atb_scenario(
    pudl_atb_sub: pd.DataFrame,
    aeo: pd.DataFrame,
    beccs: pd.Series | None = None,
) -> pd.DataFrame:
    """Run the full cost pipeline for one (atb_scenario, model_case) slice.

    Concatenates static data (emissions, transmission, lifetime, AEO fuels), takes the
    ATB fuel costs of nuclear, SMR and biomass, derives biomass-CCS from ``beccs``
    (``get_beccs_reference``) if given, pivots,
    derives hydrogen_ct, marginal cost, annualized capex etc., and returns a long-form
    DataFrame with columns [pypsa-name, parameter, value] for this scenario+case.
    """
    pudl_atb = pudl_atb_sub.melt(
        id_vars="pypsa-name",
        value_vars=ATB_VALUE_COLS,
        var_name="parameter",
        value_name="value",
    )

    pudl_atb = pd.concat(
        [
            pudl_atb,
            pd.DataFrame(EMISSIONS_DATA),
            pd.DataFrame(TRANSMISSION_DATA),
            pd.DataFrame(LIFETIME_DATA),
        ],
        ignore_index=True,
    )
    pudl_atb = pudl_atb.drop_duplicates(
        subset=["pypsa-name", "parameter"],
        keep="last",
    )

    pudl_atb = pd.concat([pudl_atb, aeo], ignore_index=True)

    pivot_atb = pudl_atb.pivot(
        index="pypsa-name",
        columns="parameter",
        values="value",
    ).reset_index()

    # ATB fuel costs are per MWh of electricity
    atb_fuel = pivot_atb["pypsa-name"].isin(ATB_FUEL_CARRIERS)
    pivot_atb.loc[atb_fuel, "fuel_cost_real_per_mwhth"] = (
        pivot_atb.loc[atb_fuel, "fuel_cost_per_mwh"] * 3.412 / pivot_atb.loc[atb_fuel, "heat_rate_mmbtu_per_mwh"]
    )

    if beccs is not None:
        pivot_atb = pd.concat([pivot_atb, build_beccs(pivot_atb, beccs)], ignore_index=True)

    # Create Hydrogen Combustion Turbine from OCGT using assumptions per ReEDS
    # https://nrel.github.io/ReEDS-2.0/model_documentation.html#hydrogen
    hydrogen_ct = pivot_atb[pivot_atb["pypsa-name"] == "OCGT"].copy()
    hydrogen_ct["pypsa-name"] = "hydrogen_ct"
    hydrogen_ct["capex_overnight_per_kw"] *= 1.03
    hydrogen_ct["capex_per_kw"] = (
        (hydrogen_ct["capex_overnight_per_kw"] + hydrogen_ct["capex_grid_connection_per_kw"])
        * hydrogen_ct["capex_construction_finance_factor"]
        / 100
    )
    hydrogen_ct["fuel_cost_real_per_mwhth"] = 20 * 3.412  # 20 USD/MMBtu * 3.412 MMBtu/MWh_th
    hydrogen_ct["co2_emissions"] = 0
    pivot_atb = pd.concat([pivot_atb, hydrogen_ct], ignore_index=True)

    pivot_atb["efficiency"] = 3.412 / pivot_atb["heat_rate_mmbtu_per_mwh"]
    pivot_atb["fuel_cost"] = pivot_atb["fuel_cost_real_per_mwhth"] / pivot_atb["efficiency"]
    pivot_atb["marginal_cost"] = pivot_atb["opex_variable_per_mwh"] + pivot_atb["fuel_cost"]

    # Impute storage WACC from Utility Scale Solar. TODO: Revisit this assumption
    for x in [2, 4, 6, 8, 10]:
        pivot_atb.loc[
            pivot_atb["pypsa-name"] == f"{x}hr_battery_storage",
            "wacc_real",
        ] = pivot_atb.loc[
            pivot_atb["pypsa-name"] == "solar",
            "wacc_real",
        ].values[0]
        pivot_atb.loc[
            pivot_atb["pypsa-name"] == f"{x}hr_battery_storage",
            "efficiency",
        ] = 0.85  # 2023 ATB

    pivot_atb["annualized_capex_per_mw"] = (
        calculate_annuity(
            pivot_atb["cost_recovery_period_years"],
            pivot_atb["wacc_real"],
        )
        * pivot_atb["capex_per_kw"]
        * 1
    ) * 1e3

    pivot_atb["annualized_capex_per_mw_km"] = (
        calculate_annuity(
            pivot_atb["cost_recovery_period_years"],
            pivot_atb["wacc_real"],
        )
        * pivot_atb["capex_per_mw_km"]
        * 1
    )

    # Grid interconnection costs per MW-KM
    # Land-based: 1 mile of spur line; offshore: 30 km of subsea cable
    pivot_atb["capex_grid_connection_per_kw_km"] = pivot_atb["capex_grid_connection_per_kw"] / 1.609
    pivot_atb.loc[
        pivot_atb["pypsa-name"].str.contains("offshore"),
        "capex_grid_connection_per_kw_km",
    ] = pivot_atb["capex_grid_connection_per_kw"] / 30

    pivot_atb["annualized_connection_capex_per_mw_km"] = (
        calculate_annuity(
            pivot_atb["cost_recovery_period_years"],
            pivot_atb["wacc_real"],
        )
        * pivot_atb["capex_grid_connection_per_kw_km"]
        * 1
    )

    pivot_atb["annualized_capex_fom"] = pivot_atb["annualized_capex_per_mw"] + (pivot_atb["opex_fixed_per_kw"] * 1e3)
    result = pivot_atb.melt(
        id_vars=["pypsa-name"],
        value_vars=pivot_atb.columns.difference(["pypsa-name"]),
        var_name="parameter",
        value_name="value",
    )
    result = result.reset_index(drop=True)
    result["value"] = result["value"].round(3)
    return result


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_cost_data", year=2030)
        rootpath = ".."
    else:
        rootpath = "."

    aeo_params = snakemake.params.aeo or {}
    dollar_year = dollar_year_from_config(snakemake.config)

    tech_year = snakemake.wildcards.year
    if int(tech_year) < 2024:
        logger.warning(
            "Minimum cost year supported is 2024, using 2024 expansion costs.",
        )
    years = range(2024, 2051)
    tech_year = min(years, key=lambda x: abs(x - int(tech_year)))

    # Path to parquet files
    parquet_path = snakemake.params.pudl_path

    # Import PUDLs ATB data
    pudl_atb = load_pudl_atb_data(parquet_path)
    pudl_atb["pypsa-name"] = pudl_atb.apply(
        match_technology,
        axis=1,
        tech_dict=const.ATB_TECH_MAPPER,
    )
    pudl_atb = pudl_atb[pudl_atb["pypsa-name"].notnull()]

    # Group by pypsa-name and filter for correct cost recovery period
    pudl_atb = (
        pudl_atb.groupby("pypsa-name")[pudl_atb.columns]
        .apply(
            lambda x: x[x["cost_recovery_period_years"] == const.ATB_TECH_MAPPER[x.name].get("crp", 30)],
        )
        .reset_index(drop=True)
    )

    # Filter to the requested projection year (with 2030 fallback for offwind_floating)
    pudl_atb_filt = pudl_atb[pudl_atb.projection_year == tech_year]
    if tech_year < 2030:
        logger.warning(
            "Using 2030 ATB data for offwind_floating; earlier data not available.",
        )
        pudl_atb_offwind_floating = pudl_atb[
            (pudl_atb["pypsa-name"] == "offwind_floating") & (pudl_atb.projection_year == 2030)
        ]
        pudl_atb_year = pd.concat(
            [pudl_atb_filt, pudl_atb_offwind_floating],
            ignore_index=True,
        )
    else:
        pudl_atb_year = pudl_atb_filt

    # Build AEO fuel cost data once — independent of ATB scenario / case
    aeo = build_aeo_fuel_costs(
        snakemake.params.eia_api,
        tech_year,
        aeo_params.get("scenario", "reference"),
        dollar_year,
    )

    # BECCS costs of ReEDS, independent of ATB scenario / case
    beccs = get_beccs_reference(snakemake.input.beccs_reference, tech_year)

    # Iterate over every (atb_scenario, model_case) so that the final CSV holds
    # the entire ATB scenario grid; downstream selects the correct slice via
    # `costs.atb.scenario`/`model_case` (with optional per-carrier overrides).
    scenarios = pudl_atb_year.scenario_atb.dropna().unique()
    cases = pudl_atb_year.model_case_nrelatb.dropna().unique()
    logger.info(
        f"Building costs for {len(scenarios)} scenarios x {len(cases)} model cases "
        f"= {len(scenarios) * len(cases)} ATB combinations",
    )

    scenario_results = []
    for atb_scenario in scenarios:
        for atb_model_case in cases:
            sub = complete_atb_slice(pudl_atb_year, atb_scenario, atb_model_case)
            if sub.empty:
                continue
            result = process_atb_scenario(sub, aeo, beccs)
            result["atb_scenario"] = atb_scenario
            result["atb_model_case"] = atb_model_case
            scenario_results.append(result)

    pudl_atb = pd.concat(scenario_results, ignore_index=True)

    # EGS costs do not vary with ATB scenario; tag as NA so the loader applies them universally
    egs_costs = pd.read_csv(snakemake.input.egs_costs)
    egs_costs = egs_costs.query("investment_horizon == @tech_year").drop(
        columns="investment_horizon",
    )
    egs_costs["atb_scenario"] = pd.NA
    egs_costs["atb_model_case"] = pd.NA
    pudl_atb = pd.concat([pudl_atb, egs_costs], ignore_index=True)

    pudl_atb.to_csv(snakemake.output.tech_costs, index=False)

    # sector costs
    sector_costs = get_sector_costs(
        snakemake.input.efs_tech_costs,
        snakemake.input.efs_icev_costs,
        snakemake.input.eia_tech_costs,
        tech_year,
        dollar_year,
        snakemake.input.additional_costs,
    )
    sector_costs.to_csv(snakemake.output.sector_costs, index=False)
