"""Module for holding global constant values"""

###########################################
# Constants for GIS Coordinate Reference Systems
###########################################
MEASUREMENT_CRS = "EPSG:5070"
GPS_CRS = "EPSG:4326"

###################
# General Constants
###################

# convert euros to USD
EUR_2_USD = 1.07 # taken on 12-12-2023

# energy content of natural gas 
# (1 MCF) * (1.036 MMBTU / 1 MCF) * (0.293 MWh / 1 MMBTU)
# https://www.eia.gov/tools/faqs/faq.php?id=45&t=8
NG_MCF_2_MWH = 0.3035

# Extract only the continental united states 
STATES_TO_REMOVE = [
    "Hawaii", 
    "Alaska", 
    "Commonwealth of the Northern Mariana Islands", 
    "United States Virgin Islands", 
    "Guam", 
    "Puerto Rico", 
    "American Samoa"
]

################################
# Constants for ADS WECC mapping 
################################

# maps ADS tech name to PyPSA name 
ADS_TECH_MAPPER= {
    'Solar_Tracking':'solar-utility',
    'NG_Industrial':'OCGT', 
    'NG_Areo':'OCGT',
    'Wind_Onshore':'onwind',
    'Geo_Binary':'geothermal', 
    'Solar_CSP6':'central solar thermal', 
    'NG_Single shaft':'OCGT', 
    'Solar_CSP0':'central solar thermal',
    'Geo_Double flash':'geothermal',
    'Solar_Photovoltaic':'solar-utility',
    'Natural Gas_Steam Turbine':'CCGT',
    'Subbituminous Coal_Steam Turbine':'coal',
    'Water_Hydraulic Turbine':'hydro', 
    'Natural Gas_Gas Turbine':'OCGT',
    'Wind_Wind Turbine':'onwind', 
    'MWH_Battery Storage':'battery storage', 
    'Nuclear_Nuclear':'nuclear',
    'Solar_NonTracking':'solar-utility', 
    'Landfill Gas_Internal Combustion':'other',
    'Electricity_Battery Storage':'battery storage', 
    'Solar_Non-Tracking':'solar-utility', 
    'DFO_ICE':'oil',
    'OBG_GT-NG':'other', 
    'DFO_ST':'oil', 
    'WDS_ST':'biomass',
    'Solar_Fixed':'solar-utility',
    'NG_Aero':'OCGT',
    'Biomass Waste_Internal Combustion':'biomass', 
    'OBG_ICE':'other',
    'LFG_ICE':'waste',
    'NG_GT-NG':'OCGT',
    'Wind_WT':'onwind',
    'Natural Gas_Combined Cycle':'CCGT',
    'Uranium_Nuclear':'nuclear',
    'Electricity_Non-Tracking':'battery storage',
}

# maps ADS sub-type tech name to PyPSA name 
ADS_SUB_TYPE_TECH_MAPPER = {
    'SolarPV-Tracking':'solar-utility',
    'CT-NatGas-Industrial':'OCGT',
    'CT-NatGas-Aero':'OCGT',
    'Hydro':'hydro',
    'Bio-ICE':'biomass',
    'PS-Hydro':'hydro',
    'SolarPV-NonTracking':'solar-utility',
    'WT-Onshore':'onwind',
    'Bio-ST':'CCGT',
    'ST-WasteHeat':'CCGT',
    'Geo-BinaryCycle':'geothermal',
    'ST-NatGas':'CCGT',
    'SolarThermal-CSP6':'central solar thermal',
    'CCWhole-NatGas-SingleShaft':'CCGT',
    'ICE-NatGas':'OCGT',
    'HydroRPS':'hydro',
    'ST-Nuclear':'nuclear',
    'Bio-CT':'biomass',
    'ST-Other':'CCGT',
    'CT-OilDistillate':'oil',
    'ST-Coal':'coal',
    'CCWhole-NatGas-Aero':'CCGT',
    'Bio-CC':'biomass',
    'CCWhole-NatGas-Industrial':'CCGT',
    'SolarThermal-CSP0':'central solar thermal',
    'PS-HydroRPS':'hydro',
    'Battery Storage':'battery storage',
    'Geo-DoubleFlash':'geothermal',
    'ICE-OilDistillate':'oil',
    'HYDRO':'hydro',
    'CT-Aero':'OCGT',
    'DR':'demand response',
    'MotorLoad':'other',
    'DG-BTM':'solar-rooftop',
    'CT-AB-Cogen':'OCGT',
    'CCPart-Steam':'CCGT',
    'CC-AB-Cogen':'CCGT',
    'UnknownPwrFloMdl':'other',
    'hydro':'hydro',
    'DC-Intertie':'other',
    'VAR-Device':'other'
}

# maps ADS carrier names to PyPSA name 
ADS_CARRIER_NAME = {
    'Solar':'solar',
    'NG':'gas', 
    'Water':'hydro', 
    'Bio':'biomass', 
    'Wind':'wind', 
    'WH':'waste',
    'Geo':'geothermal',
    'Uranium':'nuclear',
    'Petroleum Coke':'oil',
    'Coal':'coal',
    'NatGas':'gas',
    'Oil':'oil',
    'Electricity':'battery',
    'Natural Gas':'gas',
    'Subbituminous Coal':'coal',
    'Combined Cycle':'gas',
    'MWH':'battery',
    'Nuclear':'nuclear',
    'Landfill Gas':'waste',
    'DFO':'oil',
    'OBG':'other',
    'WDS':'biomass',
    'Biomass Waste':'waste',
    'LFG':'waste'
}

# maps ADS fuel name to PyPSA name 
ADS_FUEL_MAPPER = {
    'Solar':'Solar',
    'NG':'Gas',
    'Water':'Hydro',
    'Bio':'Biomass',
    'Wind':'Wind',
    'WH':'Waste', ##TODO: #33 add waste into cost data
    'Geo': 'Geothermal',
    'Uranium':'Nuclear',
    'Petroleum Coke':'Oil',
    'Coal':'Coal',
    'NatGas':'Gas',
    'Oil':'Oil',
    'Electricity':'Battery',
    'Natural Gas':'Gas',
    'Subbituminous Coal':'Coal',
    'Combined Cycle':'Gas_CC',
    'MWH':'Battery',
    'Nuclear':'Nuclear',
    'Landfill Gas':'Waste',
    'DFO':'Oil',
    'OBG':'Other',
    'WDS':'Biomass',
    'Biomass Waste':'Biomass',
    'LFG':'Waste',
}

###########################
# Constants for EIA mapping 
###########################

# maps EIA tech_type name to PyPSA name
# {tech_type: pypsa carrier name}
EIA_CARRIER_MAPPER = {
        'Nuclear':'nuclear',
        'Coal':'coal', 
        'Gas_SC':'OCGT', 
        'Gas_CC':'CCGT', 
        'Oil':'oil', 
        'Geothermal':'geothermal',
        'Biomass':'biomass', 
        'Other':'other', 
        'Waste':'waste',
        'Hydro':'hydro',
        'Battery':'battery',
        'Solar':'solar',
        'Wind':'onwind',
}

EIA_PRIME_MOVER_MAPPER = {
    'BA': 'Energy Storage, Battery',
    'CE': 'Energy Storage, Compressed Air',
    'CP': 'Energy Storage, Concentrated Solar Power',
    'FW': 'Energy Storage, Flywheel',
    'PS': 'Energy Storage, Reversible Hydraulic Turbine (Pumped Storage)',
    'ES': 'Energy Storage, Other (specify in SCHEDULE 7)',
    'ST': 'Steam Turbine, including nuclear, geothermal and solar steam (does not include combined cycle)',
    'GT': 'Combustion (Gas) Turbine (does not include the combustion turbine part of a combined cycle; see code CT, below)',
    'IC': 'Internal Combustion Engine (diesel, piston, reciprocating)',
    'CA': 'Combined Cycle Steam Part',
    'CT': 'Combined Cycle Combustion Turbine Part',
    'CS': 'Combined Cycle Single Shaft (combustion turbine and steam turbine share a single generator)',
    'HY': 'Hydrokinetic, Axial Flow Turbine',
    'HB': 'Hydrokinetic, Wave Buoy',
    'HK': 'Hydrokinetic, Other (specify in SCHEDULE 7)',
    'BT': 'Hydroelectric Turbine (includes turbines associated with delivery of water by pipeline)',
    'PV': 'Photovoltaic',
    'WT': 'Wind Turbine, Onshore',
    'WS': 'Wind Turbine, Offshore',
    'FC': 'Fuel Cell',
    'OT': 'Other (specify in SCHEDULE 7)'
}

NERC_REGION_MAPPER = {
    'WECC':'western',
    'TRE':'texas',
    'SERC':'eastern',
    'RFC':'eastern',
    'NPCC':'eastern',
    'MRO':'eastern',
}

STATES_INTERCONNECT_MAPPER = {
    "AL":"Eastern",
    "AK":None,
    "AZ":"Western",
    "AR":"Eastern",
    "AS":None,
    "CA":"Western",
    "CO":"Western",
    "CT":"Eastern",
    "DE":"Eastern",
    "DC":"Eastern",
    "FL":"Eastern",
    "GA":"Eastern",
    "GU":None,
    "HI":None,
    "ID":"Western",
    "IL":"Eastern",
    "IN":"Eastern",
    "IA":"Eastern",
    "KS":"Eastern",
    "KY":"Eastern",
    "LA":"Eastern",
    "ME":"Eastern",
    "MD":"Eastern",
    "MA":"Eastern",
    "MI":"Eastern",
    "MN":"Eastern",
    "MS":"Eastern",
    "MO":"Eastern",
    "MT":"Western",
    "NE":"Eastern",
    "NV":"Western",
    "NH":"Eastern",
    "NJ":"Eastern",
    "NM":"Western",
    "NY":"Eastern",
    "NC":"Eastern",
    "ND":"Eastern",
    "MP":None,
    "OH":"Eastern",
    "OK":"Eastern",
    "OR":"Western",
    "PA":"Eastern",
    "PR":None,
    "RI":"Eastern",
    "SC":"Eastern",
    "SD":"Eastern",
    "TN":"Eastern",
    "TX":"Texas",
    "TT":None,
    "UT":"Western",
    "VT":"Eastern",
    "VA":"Eastern",
    "VI":"Eastern",
    "WA":"Western",
    "WV":"Eastern",
    "WI":"Eastern",
    "WY":"Western",
}
################################
# Constants for Breakthrough mapping 
################################

BREAKTHROUGH_TECH_MAPPER = {
    'wind_offshore': 'offwind',
    'wind': 'onwind',
}
################################
# Constants for NREL ATB mapping 
################################

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
    "hydro":{ # dammed hydro 
        "technology":"Hydropower",
        "name":{"default":"H","options":["H"]},
        "alias":{"default":"H","options":["H"]},
        "detail":{"default":"NPD1","options":["NPD1","NPD2","NPD3","NPD4","NPD5","NPD6","NPD7","NPD8"]},
        "crp":{"default":100,"options":[20,30,100]},
    },
    "ror":{ # run of river  
        "technology":"Hydropower",
        "name":{"default":"H","options":["H"]},
        "alias":{"default":"H","options":["H"]},
        "detail":{"default":"NSD1","options":["NSD1","NSD2","NSD3","NSD4"]},
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
    "nuclear":{ # large scale nuclear 
        "technology":"Nuclear", 
        "name":{"default":"N","options":["N"]},
        "alias":{"default":"N","options":["N"]},
        "detail":{"default":"AP1000","options":["AP1000"]}, 
        "crp":{"default":60,"options":[20,30,60]},
    },
    "SMR":{ # small modular reactor
        "technology":"Nuclear", 
        "name":{"default":"N","options":["N"]},
        "alias":{"default":"N","options":["N"]},
        "detail":{"default":"SMR","options":["SMR"]}, 
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
    "central solar thermal":{
        "technology":"CSP",
        "name":{"default":"CSP","options":["CSP"]},
        "alias":{"default":"CSP","options":["CSP"]},
        "detail":{"default":"C2","options":["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]},
        "crp":{"default":30,"options":[20,30]},
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
        "detail":{"default":"C4T1","options":["C1T1","C2T1","C3T1","C4T1","C5T1","C6T1","C7T1","C8T2","C9T3","C10T4"]},
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

###########################################
# Constants for NREL Locational Multipliers
###########################################

# {pypsa-name: csv-name} 
CAPEX_LOCATIONAL_MULTIPLIER = {
    "nuclear":"nuclear-1117mw", 
    # "oil", 
    "CCGT":"natural-gas-430mw-90ccs", 
    "OCGT":"natural-gas-430mw-90ccs", 
    "coal":"coal-ultra-supercritical-90ccs", 
    "geothermal":"geothermal-50mw", 
    "solar":"spv-150mw", 
    "onwind":"onshore-wind-200mw",
    "hydro":"hydro-100mw"
}