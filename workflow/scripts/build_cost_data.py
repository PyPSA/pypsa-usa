"""Combines all time independent cost data sources into a standard format"""

from _helpers import mock_snakemake
from typing import List, Dict, Union
import logging
import pandas as pd

logger = logging.getLogger(__name__)

EUR_2_USD = 1.07 # taken on 12-12-2023

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

"""
pypsa-name:{
    "technology":"ATB-Name",
    "name":"ATB-tech-abreviation",
    "alias":"ATB-tech-alias.",
    "detail":"ATB-tech-detail",
    "crp":"ATB-crp",
    }
"""
ATB_TECH_MAPPER = {
    "biomass":{
        "technology":"Biopower",
        "name":{"default":"B","options":["B"]},
        "alias":{"default":"B","options":["B"]},
        "detail":{"default":"D","options":["D"]},
        "crp":{"default":45,"options":[20,30,45]},
    },
    "coal":{
        "technology":"Coal_FE",
        "name":{"default":"CFE","options":["CFE"]},
        "alias":{"default":"C","options":["C"]},
        "detail":{"default":"95CCS","options":["95CCS","99CCS","IGCC"]},
        "crp":{"default":30,"options":[20,30,75]},
    },
    "coal_retro":{
        "technology":"Coal_Retrofits",
        "name":{"default":"CR","options":["CR"]},
        "alias":{"default":"C","options":["C"]},
        "detail":{"default":"90CCS","options":["90CCS","95CCS"]},
        "crp":{"default":30,"options":[20,30,75]},
    },
    "geothermal":{
        "technology":"Geothermal",
        "name":{"default":"G","options":["G"]},
        "alias":{"default":"G","options":["G"]},
        "detail":{"default":"HF","options":["DEGSB","DEGSF","HB","HF","NFEGSB","NFEGSF"]},
        "crp":{"default":30,"options":[20,30]},
    },
    "hydro":{
        "technology":"Hydropower",
        "name":{"default":"H","options":["H"]},
        "alias":{"default":"H","options":["H"]},
        "detail":{"default":"NPD1","options":["NPD1","NPD2","NPD3","NPD4","NPD5","NPD6","NPD7","NPD8","NSD1","NSD2","NSD3","NSD4"]},
        "crp":{"default":100,"options":[20,30,100]},
    },
    "CCGT":{ # natural gas
        "technology":"NaturalGas_FE",
        "name":{"default":"NGFE","options":["NGFE"]},
        "alias":{"default":"NG","options":["NG"]},
        "detail":{"default":"CCFF","options":["CCFF","CCFF95CCS","CCFF97CCS","CTFF","CCHF","CCHF95CCS","CCHF97CCS","FC","FC98CCS"]},
        "crp":{"default":30,"options":[20,30,55]},
    },
    "OCGT":{ # natural gas
        "technology":"NaturalGas_FE",
        "name":{"default":"NGFE","options":["NGFE"]},
        "alias":{"default":"NG","options":["NG"]},
        "detail":{"default":"CTFF","options":["CCFF","CCFF95CCS","CCFF97CCS","CTFF","CCHF","CCHF95CCS","CCHF97CCS","FC","FC98CCS"]},
        "crp":{"default":30,"options":[20,30,55]},
    },
    "natural_gas_retrofit":{
        "technology":"NaturalGas_Retrofits",
        "name":{"default":"NGR","options":["NGR"]},
        "alias":{"default":"NG","options":["NG"]},
        "detail":{"default":"CCFC90CCS","options":["CCFC90CCS","CCFC95CCS","CCFF97CCS","CCHC90CCS","CCHC95CCS"]},
        "crp":{"default":30,"options":[20,30,55]},
    },
    "nuclear":{
        "technology":"Nuclear", 
        "name":{"default":"N","options":["N"]},
        "alias":{"default":"N","options":["N"]},
        "detail":{"default":"AP1000","options":["AP1000"]}, # large scale nuclear 
        "crp":{"default":60,"options":[20,30,60]},
    },
    "smr":{
        "technology":"Nuclear", 
        "name":{"default":"N","options":["N"]},
        "alias":{"default":"N","options":["N"]},
        "detail":{"default":"SMR","options":["SMR"]}, # small modular reactor
        "crp":{"default":60,"options":[20,30,60]},
    },
    "solar-rooftop commercial":{
        "technology":"CommPV",
        "name":{"default":"CPV","options":["CPV"]},
        "alias":{"default":"CPV","options":["CPV"]},
        "detail":{"default":"C5","options":["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]},
        "crp":{"default":20,"options":[20,30]},
    },
    "solar-concentrated":{
        "technology":"CSP",
        "name":{"default":"CSP","options":["CSP"]},
        "alias":{"default":"CSP","options":["CSP"]},
        "detail":{"default":"C2","options":["C2","C3","C8"]},
        "crp":{"default":20,"options":[20,30]},
    },
    "solar-rooftop":{
        "technology":"ResPV",
        "name":{"default":"RPV","options":["RPV"]},
        "alias":{"default":"RPV","options":["RPV"]},
        "detail":{"default":"C5","options":["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]},
        "crp":{"default":20,"options":[20,30]},
    },
    "solar-utility":{
        "technology":"UtilityPV",
        "name":{"default":"UPV","options":["UPV"]},
        "alias":{"default":"UPV","options":["UPV"]},
        "detail":{"default":"C5","options":["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]},
        "crp":{"default":20,"options":[20,30]},
    },
    "solar-utility-plus-battery":{
        "technology":"Utility-Scale PV-Plus-Battery",
        "name":{"default":"USPVPB","options":["USPVPB"]},
        "alias":{"default":"PVS","options":["PVS"]},
        "detail":{"default":"C5","options":["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]},
        "crp":{"default":20,"options":[20,30]},
    },
    "commercial battery storage":{
        "technology":"Commercial Battery Storage",
        "name":{"default":"CBS","options":["CBS"]},
        "alias":{"default":"CBS","options":["CBS"]},
        "detail":{"default":"4H","options":["1H","2H","4H","6H","8H"]},
        "crp":{"default":20,"options":[20,30]},
    },
    "home battery storage":{
        "technology":"Residential Battery Storage",
        "name":{"default":"RBS","options":["RBS"]},
        "alias":{"default":"RBS","options":["RBS"]},
        "detail":{"default":"5W125W","options":["5W125W","5W20W"]},
        "crp":{"default":20,"options":[20,30]},
    },
    "battery storage":{
        "technology":"Utility-Scale Battery Storage",
        "name":{"default":"USBS","options":["USBS"]},
        "alias":{"default":"USBS","options":["USBS"]},
        "detail":{"default":"8H","options":["2H","4H","6H","8H","10H"]},
        "crp":{"default":20,"options":[20,30]},
    },
    "wind-distributed":{
        "technology":"DistributedWind",
        "name":{"default":"DW","options":["DW"]},
        "alias":{"default":"MDW","options":["CDW","LDW","MDW","RDW"]},
        "detail":{"default":"C7","options":["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]},
        "crp":{"default":30,"options":[20,30]},
    },
    "onwind":{
        "technology":"LandbasedWind",
        "name":{"default":"LW","options":["LW"]},
        "alias":{"default":"LBW","options":["LBW"]},
        "detail":{"default":"C4","options":["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]},
        "crp":{"default":30,"options":[20,30]},
    },
    "offwind":{
        "technology":"OffShoreWind",
        "name":{"default":"OSW","options":["OSW"]},
        "alias":{"default":"OW","options":["OW"]},
        "detail":{"default":"C3","options":["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14"]},
        "crp":{"default":30,"options":[20,30]},
    },
    "Pumped-Storage-Hydro-bicharger":{
        "technology":"Pumped Storage Hydropower",
        "name":{"default":"PSH","options":["PSH"]},
        "alias":{"default":"PSH","options":["PSH"]},
        "detail":{"default":"NC1","options":["NC1","NC2","NC3","NC4","NC5","NC6","NC7","NC8","NC1","NC2","NC3","NC4"]},
        "crp":{"default":100,"options":[20,30,100]},
    },
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
    year_short = int(year % 100)

    # Cost Recovery Period 
    if not crpyears:
        crp = ATB_TECH_MAPPER[technology]["crp"]["default"]
    else:
        if not crpyears in ATB_TECH_MAPPER[technology]["crp"]["options"]:
            logger.warning(f"Invalid crp selection of {tech_name}")
            crp = ATB_TECH_MAPPER[technology]["crp"]["default"]
        else:
            crp = crpyears
        
    # technology name
    if not tech_name:
        name = ATB_TECH_MAPPER[technology]["name"]["default"]
    else:
        if not tech_name in ATB_TECH_MAPPER[technology]["name"]["options"]:
            logger.warning(f"Invalid technology name of {tech_name}")
            name = ATB_TECH_MAPPER[technology]["name"]["default"]
        else:
            name = tech_name

    # technology alias
    if not tech_alias:
        alias = ATB_TECH_MAPPER[technology]["alias"]["default"]
    else:
        if not tech_alias in ATB_TECH_MAPPER[technology]["alias"]["options"]:
            logger.warning(f"Invalid technology alias of {tech_alias}")
            alias = ATB_TECH_MAPPER[technology]["alias"]["default"]
        else:
            alias = tech_alias

    # technology detail
    if not tech_detail:
        detail = ATB_TECH_MAPPER[technology]["detail"]["default"]
    else:
        if not tech_detail in ATB_TECH_MAPPER[technology]["alias"]["options"]:
            logger.warning(f"Invalid technology alias of {tech_alias}")
            detail = ATB_TECH_MAPPER[technology]["detail"]["default"]
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
                round(atb.loc[core_metric_key]["value"],2),
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
                round(atb.loc[core_metric_key]["value"],2),
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
            ATB_TECH_MAPPER[technology]["crp"]["default"],
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
                round(atb.loc[core_metric_key]["value"],2),
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
                round(atb.loc[core_metric_key]["value"],2),
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
                round(atb.loc[core_metric_key]["value"],2),
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
    
    return df

def correct_units(df: pd.DataFrame) -> pd.DataFrame:
    """Alligns units to be the same as PyPSA
    
    Note
    ----
    Input data should follow pypsa costs datastructure 
    """
    
    # kW -> MW
    df.loc[df.unit.str.contains("/kW"), "value"] *= 1e3
    df.unit = df.unit.str.replace("/kW", "/MW")
    
    # MMBtu/MWh -> per unit efficiency 
    df.loc[df.unit == "MMBtu/MWh", "value"] = 3.412 / df.loc[df.unit == "MMBtu/MWh", "value"]
    df.unit = df.unit.str.replace("MMBtu/MWh", "per unit")
    
    # Eur -> USD 
    df.loc[df.unit.str.contains("EUR/"), "value"] *= EUR_2_USD
    df.unit = df.unit.str.replace("EUR/", "USD/")
    
    # $ -> USD (for consistancy)
    df.unit = df.unit.str.replace("$/", "USD/")
    
    return df
    
def correct_fixed_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Changes fixed cost from $/MW to %/year
    
    Note
    ----
    Input data should follow pypsa costs datastructure 
    """
    
    # get technologies to fix 
    # "Gasnetz", "gas storage" are expressed as only a percentage 
    df_fom = df[(df.parameter == "FOM") & (~df.unit.str.startswith("%/"))]
    techs = [x for x in df_fom.technology.unique() if x not in ["Gasnetz", "gas storage"]]
    
    # this method of slicing a df is quite inefficienct :( 
    for tech in techs:
        fom = df.loc[(df.technology == tech) & (df.parameter == "FOM"), "value"].reset_index(drop=True)
        capex = df.loc[(df.technology == tech) & (df.parameter == "investment"), "value"].reset_index(drop=True)
        
        assert fom.shape == capex.shape # should each only have one row 
        
        df.loc[(df.technology == tech) & (df.parameter == "FOM"), "value"] = fom / capex * 100
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
    techs = [x for x in eur.technology.unique() if x in ATB_TECH_MAPPER]
    atb_extracted = get_atb_data(atb, techs, year=year)
    
    # merge dataframes 
    costs = pd.concat([eur, atb_extracted])
    costs = costs.drop_duplicates(subset = ["technology", "parameter"], keep="last")
    
    # align merged data 
    costs = correct_units(costs)
    costs = correct_fixed_cost(costs)
    costs = costs.reset_index(drop=True)
    
    costs.to_csv(snakemake.output.tech_costs, index=False)