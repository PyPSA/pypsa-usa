"""Combines all time independent cost data sources into a standard format"""

from _helpers import mock_snakemake
from typing import List, Dict, Union
import logging
import pandas as pd
import constants as const

logger = logging.getLogger(__name__)

# https://atb.nrel.gov/electricity/2023/equations_&_variables
ATB_CMP_MAPPER = {
    "CAPEX":"CAPEX",
    "CF": "CF", # Capacity Factor
    "Fixed O&M":"FOM",
    "Fuel":"F", # Fuel costs, converted to $/MWh, using heat rates
    "Heat Rate":"HR",
    # "CRF":"CRF", # capital recovery factor
    "WACC Real":"WACCR",
    "WACC Nominal":"WACCN",
    # "GCC": "GCC", # Grid Connection Costs
    "OCC": "OCC", # Overnight Capital Costs
    "Variable O&M": "VOM",
    "Heat Rate":"HR",
    # "Heat Rate Penalty":"HRP"
}

def build_core_metric_key(
    core_metric_parameter: str,
    technology: str,
    core_metric_case: str = "Market",
    scenario_code:str = "Moderate",
    year: int = 2030,
    crpyears: int = None,
    tech_name: str = None,
    tech_alias: str = None, 
    tech_detail: str = None, 
) -> str:
    """Builds core_metric_key to interface with NREL ATB
    
    Note
    ----
    Will not work with Debt Fraction 
    """
    
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
        scenario = "M" # Moderate

    # Year
    year_short = int(int(year) % 100)

    # Cost Recovery Period 
    if not crpyears:
        crp = const.ATB_TECH_MAPPER[technology]["crp"]["default"]
    else:
        if not crpyears in const.ATB_TECH_MAPPER[technology]["crp"]["options"]:
            logger.warning(f"Invalid crp selection of {tech_name}")
            crp = const.ATB_TECH_MAPPER[technology]["crp"]["default"]
        else:
            crp = crpyears
        
    # technology name
    if not tech_name:
        name = const.ATB_TECH_MAPPER[technology]["name"]["default"]
    else:
        if not tech_name in const.ATB_TECH_MAPPER[technology]["name"]["options"]:
            logger.warning(f"Invalid technology name of {tech_name}")
            name = const.ATB_TECH_MAPPER[technology]["name"]["default"]
        else:
            name = tech_name

    # technology alias
    if not tech_alias:
        alias = const.ATB_TECH_MAPPER[technology]["alias"]["default"]
    else:
        if not tech_alias in const.ATB_TECH_MAPPER[technology]["alias"]["options"]:
            logger.warning(f"Invalid technology alias of {tech_alias}")
            alias = const.ATB_TECH_MAPPER[technology]["alias"]["default"]
        else:
            alias = tech_alias

    # technology detail
    if not tech_detail:
        detail = const.ATB_TECH_MAPPER[technology]["detail"]["default"]
    else:
        if not tech_detail in const.ATB_TECH_MAPPER[technology]["alias"]["options"]:
            logger.warning(f"Invalid technology alias of {tech_alias}")
            detail = const.ATB_TECH_MAPPER[technology]["detail"]["default"]
        else:
            detail = tech_detail
    
    if cmp in ("WACCR", "WACCR"): # different formatting for WACC
        return f"{cmc}{crp}{cmp}{name}{scenario}{year_short}"
    else:
        return f"{cmc}{crp}{cmp}{name}{alias}{detail}{scenario}{year_short}"

def get_atb_data(atb: pd.DataFrame, techs: Union[str,List[str]], **kwargs) -> pd.DataFrame:
    """Gets ATB data for specific financial parameters 
    
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
        
        # get fixed operating cost 
        core_metric_parameter = "Fixed O&M"
        core_metric_key = build_core_metric_key(core_metric_parameter, technology, **kwargs)
        try:
            data.append([
                technology,
                "FOM",
                atb.loc[core_metric_key]["value"],
                atb.loc[core_metric_key]["units"],
                "NREL ATB",
                core_metric_key
            ])
        except KeyError:
            logger.info(f"No ATB fixed costs for {technology}")
        
        # get variable operating cost 
        core_metric_parameter = "Variable O&M"
        core_metric_key = build_core_metric_key(core_metric_parameter, technology, **kwargs)
        try:
            data.append([
                technology,
                "VOM",
                atb.loc[core_metric_key]["value"],
                f"{atb.loc[core_metric_key]['units']}_e",
                "NREL ATB",
                core_metric_key
            ])
        except KeyError:
            logger.info(f"No ATB variable costs for {technology}")
        
        # get lifetime - lifetime is the default crp
        data.append([
            technology,
            "lifetime",
            const.ATB_TECH_MAPPER[technology]["crp"]["default"],
            "years",
            "NREL ATB",
            core_metric_key
        ])
        
        # get capital cost 
        core_metric_parameter = "CAPEX" 
        core_metric_key = build_core_metric_key(core_metric_parameter, technology, **kwargs)
        try:
            data.append([
                technology,
                "investment",
                atb.loc[core_metric_key]["value"],
                f"{atb.loc[core_metric_key]['units']}_e",
                "NREL ATB",
                core_metric_key
            ])
        except KeyError:
            logger.info(f"No ATB capital costs for {technology}")
        
        # get efficiency 
        core_metric_parameter = "Heat Rate" 
        core_metric_key = build_core_metric_key(core_metric_parameter, technology, **kwargs)
        try:
            data.append([
                technology,
                "efficiency",
                atb.loc[core_metric_key]["value"],
                atb.loc[core_metric_key]["units"],
                "NREL ATB",
                core_metric_key
            ])
        except KeyError:
            logger.info(f"No ATB heat rate for {technology}")
        
        # get discount rate 
        core_metric_parameter = "WACC Real" 
        core_metric_key = build_core_metric_key(core_metric_parameter, technology, **kwargs)
        try:
            data.append([
                technology,
                "discount rate",
                atb.loc[core_metric_key]["value"],
                "per unit",
                "NREL ATB",
                core_metric_key
            ])
        except KeyError:
            logger.info(f"No ATB WACC for {technology}")
        
    df = pd.DataFrame(data, columns=[
        "technology",
        "parameter",
        "value",
        "unit",
        "source",
        "further description"
    ])
    df["value"] = df["value"].round(3)
    
    return df

def correct_units(df: pd.DataFrame, eur_conversion: Dict[str, float] = None) -> pd.DataFrame:
    """Alligns units to be the same as PyPSA
    
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
    df.loc[df.unit == "MMBtu/MWh", "value"] = 3.412 / df.loc[df.unit == "MMBtu/MWh", "value"]
    df.unit = df.unit.str.replace("MMBtu/MWh", "per unit")
    
    # Eur -> USD 
    if eur_conversion:
        convert_to = list(eur_conversion.keys())[0] # ie. USD 
        df.loc[df.unit.str.contains("EUR/"), "value"] *= eur_conversion[convert_to]
        df.unit = df.unit.str.replace("EUR/", f"{convert_to}/")
    
    # $ -> USD (for consistancy)
    df.unit = df.unit.str.replace("$/", "USD/")
    
    return df
    
def correct_fixed_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Changes fixed cost from $/W to %/year
    
    Note
    ----
    Input data should follow pypsa costs datastructure 
    """
    
    # get technologies to fix 
    # "Gasnetz", "gas storage" are expressed as only a percentage 
    # Values are very different between PyPSA and ATB causing merge issues for:
    #   - 'battery storage'
    #   - 'home battery storage'
    techs_to_skip = ["Gasnetz", "gas storage", "battery storage", "home battery storage"]
    df_fom = df[(df.parameter == "FOM") & (~df.unit.str.startswith("%/"))]
    techs = [x for x in df_fom.technology.unique() if x not in techs_to_skip]
    
    # this method of slicing a df is quite inefficienct :( 
    for tech in techs:
        fom = df.loc[(df.technology == tech) & (df.parameter == "FOM"), "value"]
        capex = df.loc[(df.technology == tech) & (df.parameter == "investment"), "value"]
        
        assert fom.shape == capex.shape # should each only have one row 
        
        df.loc[(df.technology == tech) & (df.parameter == "FOM"), "value"] = fom.iloc[-1] / capex.iloc[-1] * 100
        df.loc[(df.technology == tech) & (df.parameter == "FOM"), "unit"] = "%/year"
        
    return df

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("build_cost_data", year=2030)
        rootpath = ".."
    else:
        rootpath = "."
        
    eur = pd.read_csv(snakemake.input.pypsa_technology_data)
    atb = pd.read_parquet(snakemake.input.nrel_atb).set_index("core_metric_key")
    
    year = snakemake.wildcards.year
    
    # get technologies to replace by the ATB
    techs = [x for x in eur.technology.unique() if x in const.ATB_TECH_MAPPER]
    atb_extracted = get_atb_data(atb, techs, year=year)
    
    # merge dataframes 
    costs = pd.concat([eur, atb_extracted])
    costs = costs.drop_duplicates(subset = ["technology", "parameter"], keep="last")
    
    # align merged data 
    costs = correct_units(costs, {"USD": const.EUR_2_USD})
    costs = correct_fixed_cost(costs)
    costs = costs.reset_index(drop=True)
    costs["value"] = costs["value"].round(3)
    
    costs.to_csv(snakemake.output.tech_costs, index=False)