"""
Combines all time independent cost data sources into a standard format.
"""

import logging
from typing import Dict, List, Union

import constants as const
import pandas as pd
from _helpers import mock_snakemake

logger = logging.getLogger(__name__)

# https://atb.nrel.gov/electricity/2023/equations_&_variables
ATB_CMP_MAPPER = {
    "CAPEX": "CAPEX",
    "CF": "CF",  # Capacity Factor
    "Fixed O&M": "FOM",
    "Fuel": "F",  # Fuel costs, converted to $/MWh, using heat rates
    "Heat Rate": "HR",
    # "CRF":"CRF", # capital recovery factor
    "WACC Real": "WACCR",
    "WACC Nominal": "WACCN",
    # "GCC": "GCC", # Grid Connection Costs
    "OCC": "OCC",  # Overnight Capital Costs
    "Variable O&M": "VOM",
    "Heat Rate": "HR",
    # "Heat Rate Penalty":"HRP"
}


def build_core_metric_key(
    core_metric_parameter: str,
    technology: str,
    core_metric_case: str = "Market",
    scenario_code: str = "Moderate",
    year: int = 2030,
    crpyears: int = None,
    tech_name: str = None,
    tech_alias: str = None,
    tech_detail: str = None,
) -> str:
    """
    Builds core_metric_key to interface with NREL ATB.

    Note
    ----
    Will not work with Debt Fraction
    """
    logger.info(f"building core metric key for {core_metric_parameter}, {technology}")
    # Core Metric Parameter (metric to extract)
    try:
        cmp = ATB_CMP_MAPPER[core_metric_parameter]
    except KeyError as ex:
        logger.warning(f"Financial parameter of {core_metric_parameter} not available")
        return ""

    # Market or R&D
    if core_metric_case != "Market":
        cmc = "R"
    else:
        cmc = "M"

    # Scenario
    if scenario_code == "Advanced":
        scenario = "A"
    elif scenario_code == "Conservative":
        scenario = "C"
    else:
        scenario = "M"  # Moderate

    # Year
    year_short = int(int(year) % 100)

    # Cost Recovery Period
    if not crpyears:
        crp = const.ATB_TECH_MAPPER[technology]["crp"]
    else:
        crp = crpyears

    # technology name
    if not tech_name:
        name = const.ATB_TECH_MAPPER[technology]["name"]
    else:
        name = tech_name

    # technology alias
    if not tech_alias:
        alias = const.ATB_TECH_MAPPER[technology]["alias"]
    else:
        alias = tech_alias

    # technology detail
    if not tech_detail:
        detail = const.ATB_TECH_MAPPER[technology]["detail"]
    else:
        detail = tech_detail

    if cmp in ("WACCR", "WACCR"):  # different formatting for WACC
        return f"{cmc}{crp}{cmp}{name}{scenario}{year_short}"
    else:
        return f"{cmc}{crp}{cmp}{name}{alias}{detail}{scenario}{year_short}"


def find_core_metric_key(
    atb: pd.DataFrame,
    technology: str,
    core_metric_parameter: str,
    year: int = 2030,
) -> str:
    """
    Finds the core_metric_key from NREL ATB given the display_name, crp, and .
    """
    tech = const.ATB_TECH_MAPPER[technology]
    scenario = tech.get("scenario", "Moderate")
    core_metric_case = tech.get("core_metric_case", "Market")

    if core_metric_parameter != "WACC Real":
        criteria = (
            (atb.display_name == tech["display_name"])
            & (atb.core_metric_parameter == core_metric_parameter)
            & (atb.core_metric_variable == int(year))
            & (atb.core_metric_case == core_metric_case)
            & (atb.scenario == scenario)
            & (atb.crpyears.astype(int) == tech["crp"])
        )
        filtered_atb = atb.loc[criteria]
    else:
        tech_name = tech.get("technology", tech["display_name"].split(" - ")[0])
        criteria = (
            (atb.technology == tech_name)
            & (atb.core_metric_parameter == core_metric_parameter)
            & (atb.core_metric_variable == int(year))
            & (atb.core_metric_case == core_metric_case)
            & (atb.scenario == scenario)
            & (atb.crpyears.astype(int) == tech["crp"])
        )
        filtered_atb = atb.loc[criteria]
    if filtered_atb.shape[0] != 1:
        raise KeyError(
            f"No default core_metric_key found for {technology} {core_metric_parameter}",
        )
    return filtered_atb.iloc[0].name


def get_atb_data(atb: pd.DataFrame, techs: str | list[str], **kwargs) -> pd.DataFrame:
    """
    Gets ATB data for specific financial parameters.

    Args:
        atb: pd.DataFrame,
            raw ATB dataframe
        techs: Union[str,List[str]]
            technologies to extract data for. If more than one technology is
            provided, all data is appended together
        kwargs:
            values to override defaullts in ATB

    Returns:
        dataframe of atb data for provided techs
    """
    data = []

    if not isinstance(techs, list):
        techs = [techs]

    for technology in techs:
        missing = []

        # get fixed operating cost
        core_metric_parameter = "Fixed O&M"
        try:
            core_metric_key = find_core_metric_key(
                atb,
                technology,
                core_metric_parameter,
                **kwargs,
            )
            data.append(
                [
                    technology,
                    "FOM",
                    atb.loc[core_metric_key]["value"],
                    atb.loc[core_metric_key]["units"],
                    "NREL ATB",
                    core_metric_key,
                ],
            )
        except KeyError:
            missing.append(f"{core_metric_parameter}")

        # get variable operating cost
        core_metric_parameter = "Variable O&M"
        try:
            core_metric_key = find_core_metric_key(
                atb,
                technology,
                core_metric_parameter,
                **kwargs,
            )
            data.append(
                [
                    technology,
                    "VOM",
                    atb.loc[core_metric_key]["value"],
                    f"{atb.loc[core_metric_key]['units']}_e",
                    "NREL ATB",
                    core_metric_key,
                ],
            )
        except KeyError:
            missing.append(f"{core_metric_parameter}")

        # get lifetime - lifetime is the user defined crp
        data.append(
            [
                technology,
                "lifetime",
                const.ATB_TECH_MAPPER[technology]["crp"],
                "years",
                "NREL ATB",
                "User Defined CRP",
            ],
        )

        # get capital cost
        core_metric_parameter = "CAPEX"
        try:
            try:
                core_metric_key = find_core_metric_key(
                    atb,
                    technology,
                    core_metric_parameter,
                    **kwargs,
                )
            except KeyError:
                core_metric_key = find_core_metric_key(
                    atb,
                    technology,
                    "OCC",
                    **kwargs,
                )
                logger.warning(
                    f"Using OCC for {technology} investment- no ATB CAPEX found.",
                )
            data.append(
                [
                    technology,
                    "investment",
                    atb.loc[core_metric_key]["value"],
                    f"{atb.loc[core_metric_key]['units']}_e",
                    "NREL ATB",
                    core_metric_key,
                ],
            )
        except KeyError:
            missing.append(f"{core_metric_parameter}")

        # get efficiency
        core_metric_parameter = "Heat Rate"
        try:
            core_metric_key = find_core_metric_key(
                atb,
                technology,
                core_metric_parameter,
                **kwargs,
            )
            data.append(
                [
                    technology,
                    "efficiency",
                    3.412 / atb.loc[core_metric_key]["value"],
                    "MWH_th/MWH_elec"  # atb.loc[core_metric_key]["units"],
                    "NREL ATB",
                    core_metric_key,
                ],
            )
        except KeyError:
            missing.append(f"{core_metric_parameter}")

        # get discount rate
        core_metric_parameter = "WACC Real"
        try:
            core_metric_key = find_core_metric_key(
                atb,
                technology,
                core_metric_parameter,
                **kwargs,
            )
            data.append(
                [
                    technology,
                    "discount rate",
                    atb.loc[core_metric_key]["value"],
                    "per unit",
                    "NREL ATB",
                    core_metric_key,
                ],
            )
        except KeyError:
            missing.append(f"{core_metric_parameter}")

        if len(missing) > 0:
            logger.warning(f"Missing ATB data for {technology}: {missing}")

    df = pd.DataFrame(
        data,
        columns=[
            "technology",
            "parameter",
            "value",
            "unit",
            "source",
            "further description",
        ],
    )
    df["value"] = df["value"].round(3)

    return df


def correct_units(
    df: pd.DataFrame,
    eur_conversion: dict[str, float] = None,
) -> pd.DataFrame:
    """
    Alligns units to be the same as PyPSA.

    Arguments
    ---------
    df: pd.DataFrame,
    eur_conversion: Dict[str, float]
        If wanting to convert from eur to another unit, provide the new unit
        and conversion rate as a dictionary (ie. {"USD": 1.05})
    """

    # kW -> MW
    df.loc[df.unit.str.contains("/kW"), "value"] *= 1e3
    df.unit = df.unit.str.replace("/kW", "/MW")

    # MMBtu/MWh -> per unit efficiency
    df.loc[df.unit == "MMBtu/MWh", "value"] = (
        3.412 / df.loc[df.unit == "MMBtu/MWh", "value"]
    )
    df.unit = df.unit.str.replace("MMBtu/MWh", "per unit")

    # Eur -> USD
    if eur_conversion:
        convert_to = list(eur_conversion.keys())[0]  # ie. USD
        df.loc[df.unit.str.contains("EUR/"), "value"] *= eur_conversion[convert_to]
        df.unit = df.unit.str.replace("EUR/", f"{convert_to}/")

    # $ -> USD (for consistancy)
    df.unit = df.unit.str.replace("$/", "USD/")

    return df


def correct_fixed_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Changes fixed cost from ATB Data from $/kW-year to %/year.

    Note
    ----
    Input data should follow pypsa costs datastructure
    """

    df_fom = df[(df.parameter == "FOM") & (~df.unit.str.startswith("%/"))]

    # this method of slicing a df is quite inefficienct :(
    for tech in techs:
        fom = df.loc[(df.technology == tech) & (df.parameter == "FOM"), "value"]
        capex = df.loc[
            (df.technology == tech) & (df.parameter == "investment"),
            "value",
        ]

        assert fom.shape == capex.shape  # should each only have one row

        df.loc[(df.technology == tech) & (df.parameter == "FOM"), "value"] = (
            fom.iloc[-1] / capex.iloc[-1] * 100
        )
        df.loc[(df.technology == tech) & (df.parameter == "FOM"), "unit"] = "%/year"

    return df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_cost_data", year=2030)
        rootpath = ".."
    else:
        rootpath = "."

    year = snakemake.wildcards.year

    eur = pd.read_csv(snakemake.input.pypsa_technology_data)
    eur = correct_units(eur, {"USD": const.EUR_2_USD})

    # Pull all "default" from ATB
    atb = pd.read_parquet(snakemake.input.nrel_atb).set_index("core_metric_key")
    techs = list(const.ATB_TECH_MAPPER.keys())
    atb_extracted = get_atb_data(atb, techs, year=year)
    atb_extracted = correct_fixed_cost(atb_extracted)

    # merge dataframes
    costs = pd.concat([eur, atb_extracted])
    costs = costs.drop_duplicates(subset=["technology", "parameter"], keep="last")

    # align merged data
    costs = costs.reset_index(drop=True)
    costs["value"] = costs["value"].round(3)

    costs.to_csv(snakemake.output.tech_costs, index=False)
