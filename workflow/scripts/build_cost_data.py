"""Combines all time independent cost data sources into a standard format"""

from _helpers import mock_snakemake
from typing import List, Dict, Union
import logging

logger = logging.getLogger(__name__)

# https://atb.nrel.gov/electricity/2023/equations_&_variables
ATB_CMP_MAPPER = {
    "CAPEX":"CAPEX",
    "CF": "CF", # Capacity Factor
    "Fixed O&M":"FOM",
    "Fuel":"F", # Fuel costs, converted to $/MWh, using heat rates
    "Heat Rate":"HR",
    "CRF":"CRF", # capital recovery factor
    "WACC Real":"WACCR",
    "WACC Nominal":"WACCN",
    "GCC": "GCC", # Grid Connection Costs
    "OCC": "OCC", # Overnight Capital Costs
    "Variable O&M": "VOM",
    "Heat Rate":"HR",
    "Heat Rate Penalty":"HRP"
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
    "biopower":{
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
    "natural_gas":{
        "technology":"NaturalGas_FE",
        "name":{"default":"NGFE","options":["NGFE"]},
        "alias":{"default":"NG","options":["NG"]},
        "detail":{"default":"CCFF","options":["CCFF","CCFF95CCS","CCFF97CCS","CTFF","CCHF","CCHF95CCS","CCHF97CCS","FC","FC98CCS"]},
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
        "technology":"NaturalGas_Retrofits",
        "name":{"default":"N","options":["N"]},
        "alias":{"default":"N","options":["N"]},
        "detail":{"default":"AP1000","options":["AP1000","SMR"]},
        "crp":{"default":60,"options":[20,30,60]},
    },
    "solar_commercial":{
        "technology":"CommPV",
        "name":{"default":"CPV","options":["CPV"]},
        "alias":{"default":"CPV","options":["CPV"]},
        "detail":{"default":"C5","options":["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]},
        "crp":{"default":20,"options":[20,30]},
    },
    "solar_concentrated":{
        "technology":"CSP",
        "name":{"default":"CSP","options":["CSP"]},
        "alias":{"default":"CSP","options":["CSP"]},
        "detail":{"default":"C2","options":["C2","C3","C8"]},
        "crp":{"default":20,"options":[20,30]},
    },
    "solar_residential":{
        "technology":"ResPV",
        "name":{"default":"RPV","options":["RPV"]},
        "alias":{"default":"RPV","options":["RPV"]},
        "detail":{"default":"C5","options":["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]},
        "crp":{"default":20,"options":[20,30]},
    },
    "solar_utility":{
        "technology":"UtilityPV",
        "name":{"default":"UPV","options":["UPV"]},
        "alias":{"default":"UPV","options":["UPV"]},
        "detail":{"default":"C5","options":["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]},
        "crp":{"default":20,"options":[20,30]},
    },
    "solar_utility_plus_battery":{
        "technology":"Utility-Scale PV-Plus-Battery",
        "name":{"default":"USPVPB","options":["USPVPB"]},
        "alias":{"default":"PVS","options":["PVS"]},
        "detail":{"default":"C5","options":["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]},
        "crp":{"default":20,"options":[20,30]},
    },
    "storage_battery_commercial":{
        "technology":"Commercial Battery Storage",
        "name":{"default":"CBS","options":["CBS"]},
        "alias":{"default":"CBS","options":["CBS"]},
        "detail":{"default":"4H","options":["1H","2H","4H","6H","8H"]},
        "crp":{"default":20,"options":[20,30]},
    },
    "storage_battery_residential":{
        "technology":"Residential Battery Storage",
        "name":{"default":"RBS","options":["RBS"]},
        "alias":{"default":"RBS","options":["RBS"]},
        "detail":{"default":"5W125W","options":["5W125W","5W20W"]},
        "crp":{"default":20,"options":[20,30]},
    },
    "storage_battery_utility":{
        "technology":"Utility-Scale Battery Storage",
        "name":{"default":"USBS","options":["USBS"]},
        "alias":{"default":"USBS","options":["USBS"]},
        "detail":{"default":"8H","options":["2H","4H","6H","8H","10H"]},
        "crp":{"default":20,"options":[20,30]},
    },
    "wind_distributed":{
        "technology":"DistributedWind",
        "name":{"default":"DW","options":["DW"]},
        "alias":{"default":"MDW","options":["CDW","LDW","MDW","RDW"]},
        "detail":{"default":"C7","options":["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]},
        "crp":{"default":30,"options":[20,30]},
    },
    "wind_onshore":{
        "technology":"LandbasedWind",
        "name":{"default":"LW","options":["LW"]},
        "alias":{"default":"LBW","options":["LBW"]},
        "detail":{"default":"C4","options":["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]},
        "crp":{"default":30,"options":[20,30]},
    },
    "wind_offshore":{
        "technology":"OffShoreWind",
        "name":{"default":"OSW","options":["OSW"]},
        "alias":{"default":"OW","options":["OW"]},
        "detail":{"default":"C3","options":["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14"]},
        "crp":{"default":30,"options":[20,30]},
    },
    "storage_pumped_hydro":{
        "technology":"Pumped Storage Hydropower",
        "name":{"default":"H","options":["H"]},
        "alias":{"default":"H","options":["H"]},
        "detail":{"default":"NPD1","options":["NPD1","NPD2","NPD3","NPD4","NPD5","NPD6","NPD7","NPD8","NSD1","NSD2","NSD3","NSD4"]},
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
    
    if technology not in ATB_TECH_MAPPER.keys():
        raise KeyError(f"Invalid technology of {technology}")
    
    # Core Metric Parameter (metric to extract)
    try: 
        cmp = ATB_CMP_MAPPER[core_metric_parameter]
    except KeyError as ex:
        # logger.error(f"Financial parameter of {core_metric_parameter} not available")
        raise KeyError(ex)

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
    
    
    return f"{cmc}{crp}{cmp}{name}{alias}{detail}{scenario}{year_short}"

if __name__ == "__main__":
    pass