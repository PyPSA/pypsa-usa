"""
Module for holding global constant values.
"""

###########################################
# Constants for GIS Coordinate Reference Systems
###########################################


MEASUREMENT_CRS = "EPSG:5070"
GPS_CRS = "EPSG:4326"

###################
# General Constants
###################

# convert euros to USD
EUR_2_USD = 1.07  # taken on 12-12-2023

# energy content of natural gas
# Assumes national averages for the conversion
# https://www.eia.gov/naturalgas/monthly/pdf/table_25.pdf
# (1036 BTU / CF) * (0.293 Wh / 1 BTU) * (1 MWh / 1,000,000 Wh) * (1,000,000 CF / 1 MMCF) = 303.5 MWh / MMCF

NG_MWH_2_MMCF = 303.5  # MWh / MMCF
NG_MCF_2_MWH = 0.3035  # TODO get rid of this and just use single constant

# $/MMBtu * (1 MMBtu / 0.293 MWh) = $/MWh_thermal
NG_Dol_MMBTU_2_MWH = 3.4129

LBS_TON = 2000  # lbs/ short ton
COAL_BTU_LB = 9396  # BTU/lb - EIA US AVERAGE TODO: differentiate between coal types
MMBTU_MWHthemal = 3.4129  # MMBTU to MWh_thermal
COAL_dol_ton_2_MWHthermal = (
    LBS_TON**-1 * COAL_BTU_LB * 1000**-1 * MMBTU_MWHthemal
)  # $/ton * ton/BTU * BTU/MWh_thermal

################################
# Constants for ADS WECC mapping
################################

# maps ADS tech name to PyPSA name
ADS_TECH_MAPPER = {
    "Solar_Tracking": "solar-utility",
    "NG_Industrial": "OCGT",
    "NG_Areo": "OCGT",
    "Wind_Onshore": "onwind",
    "Geo_Binary": "geothermal",
    "Solar_CSP6": "central solar thermal",
    "NG_Single shaft": "OCGT",
    "Solar_CSP0": "central solar thermal",
    "Geo_Double flash": "geothermal",
    "Solar_Photovoltaic": "solar-utility",
    "Natural Gas_Steam Turbine": "CCGT",
    "Subbituminous Coal_Steam Turbine": "coal",
    "Water_Hydraulic Turbine": "hydro",
    "Natural Gas_Gas Turbine": "OCGT",
    "Wind_Wind Turbine": "onwind",
    "MWH_Battery Storage": "battery storage",
    "Nuclear_Nuclear": "nuclear",
    "Solar_NonTracking": "solar-utility",
    "Landfill Gas_Internal Combustion": "other",
    "Electricity_Battery Storage": "battery storage",
    "Solar_Non-Tracking": "solar-utility",
    "DFO_ICE": "oil",
    "OBG_GT-NG": "other",
    "DFO_ST": "oil",
    "WDS_ST": "biomass",
    "Solar_Fixed": "solar-utility",
    "NG_Aero": "OCGT",
    "Biomass Waste_Internal Combustion": "biomass",
    "OBG_ICE": "other",
    "LFG_ICE": "waste",
    "NG_GT-NG": "OCGT",
    "Wind_WT": "onwind",
    "Natural Gas_Combined Cycle": "CCGT",
    "Uranium_Nuclear": "nuclear",
    "Electricity_Non-Tracking": "battery storage",
}

# maps ADS sub-type tech name to PyPSA name
ADS_SUB_TYPE_TECH_MAPPER = {
    "SolarPV-Tracking": "solar-utility",
    "CT-NatGas-Industrial": "OCGT",
    "CT-NatGas-Aero": "OCGT",
    "Hydro": "hydro",
    "Bio-ICE": "biomass",
    "PS-Hydro": "hydro",
    "SolarPV-NonTracking": "solar-utility",
    "WT-Onshore": "onwind",
    "Bio-ST": "CCGT",
    "ST-WasteHeat": "CCGT",
    "Geo-BinaryCycle": "geothermal",
    "ST-NatGas": "CCGT",
    "SolarThermal-CSP6": "central solar thermal",
    "CCWhole-NatGas-SingleShaft": "CCGT",
    "ICE-NatGas": "OCGT",
    "HydroRPS": "hydro",
    "ST-Nuclear": "nuclear",
    "Bio-CT": "biomass",
    "ST-Other": "CCGT",
    "CT-OilDistillate": "oil",
    "ST-Coal": "coal",
    "CCWhole-NatGas-Aero": "CCGT",
    "Bio-CC": "biomass",
    "CCWhole-NatGas-Industrial": "CCGT",
    "SolarThermal-CSP0": "central solar thermal",
    "PS-HydroRPS": "hydro",
    "Battery Storage": "battery storage",
    "Geo-DoubleFlash": "geothermal",
    "ICE-OilDistillate": "oil",
    "HYDRO": "hydro",
    "CT-Aero": "OCGT",
    "DR": "demand response",
    "MotorLoad": "other",
    "DG-BTM": "solar-rooftop",
    "CT-AB-Cogen": "OCGT",
    "CCPart-Steam": "CCGT",
    "CC-AB-Cogen": "CCGT",
    "UnknownPwrFloMdl": "other",
    "hydro": "hydro",
    "DC-Intertie": "other",
    "VAR-Device": "other",
}

# maps ADS carrier names to PyPSA name
ADS_CARRIER_NAME = {
    "Solar": "solar",
    "NG": "gas",
    "Water": "hydro",
    "Bio": "biomass",
    "Wind": "wind",
    "WH": "waste",
    "Geo": "geothermal",
    "Uranium": "nuclear",
    "Petroleum Coke": "oil",
    "Coal": "coal",
    "NatGas": "gas",
    "Oil": "oil",
    "Electricity": "battery",
    "Natural Gas": "gas",
    "Subbituminous Coal": "coal",
    "Combined Cycle": "gas",
    "MWH": "battery",
    "Nuclear": "nuclear",
    "Landfill Gas": "waste",
    "DFO": "oil",
    "OBG": "other",
    "WDS": "biomass",
    "Biomass Waste": "waste",
    "LFG": "waste",
}

# maps ADS fuel name to PyPSA name
ADS_FUEL_MAPPER = {
    "Solar": "Solar",
    "NG": "Gas",
    "Water": "Hydro",
    "Bio": "Biomass",
    "Wind": "Wind",
    "WH": "Waste",  ##TODO: #33 add waste into cost data
    "Geo": "Geothermal",
    "Uranium": "Nuclear",
    "Petroleum Coke": "Oil",
    "Coal": "Coal",
    "NatGas": "Gas",
    "Oil": "Oil",
    "Electricity": "Battery",
    "Natural Gas": "Gas",
    "Subbituminous Coal": "Coal",
    "Combined Cycle": "Gas_CC",
    "MWH": "Battery",
    "Nuclear": "Nuclear",
    "Landfill Gas": "Waste",
    "DFO": "Oil",
    "OBG": "Other",
    "WDS": "Biomass",
    "Biomass Waste": "Biomass",
    "LFG": "Waste",
}

###########################
# Constants for EIA mapping
###########################

# renaming moved to pre-processing


###############################
# Constants for Region Mappings
###############################

# Extract only the continental united states
STATES_TO_REMOVE = [
    "Hawaii",
    "Alaska",
    "Commonwealth of the Northern Mariana Islands",
    "United States Virgin Islands",
    "Guam",
    "Puerto Rico",
    "American Samoa",
]

NERC_REGION_MAPPER = {
    "WECC": "western",
    "TRE": "texas",
    "SERC": "eastern",
    "RFC": "eastern",
    "NPCC": "eastern",
    "MRO": "eastern",
}

EIA_930_REGION_MAPPER = {
    "CAL": "western",
    "CAR": "eastern",
    "CENT": "eastern",
    "FLA": "eastern",
    "MIDA": "eastern",
    "MIDW": "eastern",
    "NW": "western",
    "NE": "eastern",
    "NY": "eastern",
    "SE": "eastern",
    "SW": "western",
    "TEN": "eastern",
    "TEX": "texas",
}

EIA_BA_2_REGION = {
    "AEC": "SE",
    "AECI": "MIDW",
    "AVA": "NW",
    "AVRN": "NW",
    "AZPS": "SW",
    "BANC": "CAL",
    "BPAT": "NW",
    "CHPD": "NW",
    "CISO": "CAL",
    "CPLE": "CAR",
    "CPLW": "CAR",
    "DEAA": "SW",
    "DOPD": "NW",
    "DUK": "CAR",
    "EEI": "MIDW",
    "EPE": "SW",
    "ERCO": "TEX",
    "FMPP": "FLA",
    "FPC": "FLA",
    "FPL": "FLA",
    "GCPD": "NW",
    "GRID": "NW",
    "GRIF": "SW",
    "GVL": "FLA",
    "GWA": "NW",
    "HGMA": "SW",
    "HST": "FLA",
    "IID": "CAL",
    "IPCO": "NW",
    "ISNE": "NE",
    "JEA": "FLA",
    "LDWP": "CAL",
    "LGEE": "MIDW",
    "MISO": "MIDW",
    "NEVP": "NW",
    "NSB": "FLA",
    "NWMT": "NW",
    "NYIS": "NY",
    "PACE": "NW",
    "PACW": "NW",
    "PGE": "NW",
    "PJM": "MIDA",
    "PNM": "SW",
    "PSCO": "NW",
    "PSEI": "NW",
    "SC": "CAR",
    "SCEG": "CAR",
    "SCL": "NW",
    "SEC": "FLA",
    "SEPA": "SE",
    "SOCO": "SE",
    "SPA": "CENT",
    "SRP": "SW",
    "SWPP": "CENT",
    "TAL": "FLA",
    "TEC": "FLA",
    "TEPC": "SW",
    "TIDC": "CAL",
    "TPWR": "NW",
    "TVA": "TEN",
    "WACM": "NW",
    "WALC": "SW",
    "WAUW": "NW",
    "WWA": "NW",
    "YAD": "CAR",
}

STATES_INTERCONNECT_MAPPER = {
    "AL": "eastern",
    "AK": None,
    "AZ": "western",
    "AR": "eastern",
    "AS": None,
    "CA": "western",
    "CO": "western",
    "CT": "eastern",
    "DE": "eastern",
    "DC": "eastern",
    "FL": "eastern",
    "GA": "eastern",
    "GU": None,
    "HI": None,
    "ID": "western",
    "IL": "eastern",
    "IN": "eastern",
    "IA": "eastern",
    "KS": "eastern",
    "KY": "eastern",
    "LA": "eastern",
    "ME": "eastern",
    "MD": "eastern",
    "MA": "eastern",
    "MI": "eastern",
    "MN": "eastern",
    "MS": "eastern",
    "MO": "eastern",
    "MT": "western",
    "NE": "eastern",
    "NV": "western",
    "NH": "eastern",
    "NJ": "eastern",
    "NM": "western",
    "NY": "eastern",
    "NC": "eastern",
    "ND": "eastern",
    "MP": None,
    "OH": "eastern",
    "OK": "eastern",
    "OR": "western",
    "PA": "eastern",
    "PR": None,
    "RI": "eastern",
    "SC": "eastern",
    "SD": "eastern",
    "TN": "eastern",
    "TX": "texas",
    "TT": None,
    "UT": "western",
    "VT": "eastern",
    "VA": "eastern",
    "VI": "eastern",
    "WA": "western",
    "WV": "eastern",
    "WI": "eastern",
    "WY": "western",
    "AB": "canada",
    "BC": "canada",
    "MB": "canada",
    "NB": "canada",
    "NL": "canada",
    "NT": "canada",
    "NS": "canada",
    "NU": "canada",
    "ON": "canada",
    "PE": "canada",
    "QC": "canada",
    "SK": "canada",
    "YT": "canada",
    "MX": "mexico",
}

STATE_2_CODE = {
    # United States
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "American Samoa": "AS",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Guam": "GU",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Northern Mariana Islands": "MP",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Puerto Rico": "PR",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Trust Territories": "TT",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Virgin Islands": "VI",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    # Canada
    "Alberta": "AB",
    "British Columbia": "BC",
    "Manitoba": "MB",
    "New Brunswick": "NB",
    "Newfoundland and Labrador": "NL",
    "Northwest Territories": "NT",
    "Nova Scotia": "NS",
    "Nunavut": "NU",
    "Ontario": "ON",
    "Prince Edward Island": "PE",
    "Quebec": "QC",
    "Saskatchewan": "SK",
    "Yukon": "YT",
    # Mexico
    "Mexico": "MX",
}

# Simplified dictionary to map states to their primary time zones.
# Note: This does not account for states with multiple time zones or specific exceptions.
STATE_2_TIMEZONE = {
    "AL": "US/Central",
    "AK": "US/Alaska",
    "AZ": "US/Mountain",
    "AR": "US/Central",
    "CA": "US/Pacific",
    "CO": "US/Mountain",
    "CT": "US/Eastern",
    "DE": "US/Eastern",
    "FL": "US/Eastern",
    "GA": "US/Eastern",
    "HI": "Pacific/Honolulu",
    "ID": "US/Mountain",
    "IL": "US/Central",
    "IN": "US/Eastern",
    "IA": "US/Central",
    "KS": "US/Central",
    "KY": "US/Eastern",
    "LA": "US/Central",
    "ME": "US/Eastern",
    "MD": "US/Eastern",
    "MA": "US/Eastern",
    "MI": "US/Eastern",
    "MN": "US/Central",
    "MS": "US/Central",
    "MO": "US/Central",
    "MT": "US/Mountain",
    "NE": "US/Central",
    "NV": "US/Pacific",
    "NH": "US/Eastern",
    "NJ": "US/Eastern",
    "NM": "US/Mountain",
    "NY": "US/Eastern",
    "NC": "US/Eastern",
    "ND": "US/Central",
    "OH": "US/Eastern",
    "OK": "US/Central",
    "OR": "US/Pacific",
    "PA": "US/Eastern",
    "RI": "US/Eastern",
    "SC": "US/Eastern",
    "SD": "US/Central",
    "TN": "US/Central",
    "TX": "US/Central",
    "UT": "US/Mountain",
    "VT": "US/Eastern",
    "VA": "US/Eastern",
    "WA": "US/Pacific",
    "WV": "US/Eastern",
    "WI": "US/Central",
    "WY": "US/Mountain",
}

################################
# Constants for Breakthrough mapping
################################

BREAKTHROUGH_TECH_MAPPER = {
    "wind_offshore": "offwind",
    "wind": "onwind",
}
################################
# Constants for NREL ATB mapping
################################
"""
If you want to use the default ATB technology, the minimum must be defined: pypsa-name, technology, crp.
If a default classificiation is not defined by ATB, then the remaining fields will be used to extract data from the ATB.
pypsa-name:{
    "display_name": "Utility-Scale Battery Storage - 4Hr",
    "crp":  ATB Capital Recovery Period value,
    "technology": ATB Technology Name- In most cases technology name is embedded in the display name preceeding ' - ', for example "Biopower - Dedicated" has a technology name of "Biopower". In the cases of Coal and Natural gas this is not true, so the technology name must be defined. If adding new technologies- keep an eye on the logger.warnings and check the ATB directly.
    "scenario": ATB Technology Cost Scenario- "Moderate" used if not defined,
    "core_metric_case": ATB Core Metric Case- "Market" used if not defined,
    }
"""
ATB_TECH_MAPPER = {
    "biomass": {
        "display_name": "Biopower - Dedicated",
        "crp": 45,
    },
    "coal": {
        "display_name": "Coal-new",
        "technology": "Coal_FE",
        "crp": 30,
    },
    "coal_95CCS": {
        "display_name": "Coal-95%-CCS",
        "technology": "Coal_FE",
        "crp": 30,
    },
    "coal_99CCS": {
        "display_name": "Coal-99%-CCS",
        "technology": "Coal_FE",
        "crp": 30,
    },
    "geothermal": {
        "display_name": "Geothermal - Hydro / Flash",
        "crp": 30,
    },
    "hydro": {  # dammed hydro
        "display_name": "Hydropower - NPD 1",
        "crp": 100,
    },
    "ror": {  # run of river
        "display_name": "Hydropower - NSD 1",
        "crp": 100,
    },
    "CCGT": {  # natural gas
        "display_name": "NG Combined Cycle (F-Frame)",
        "technology": "NaturalGas_FE",
        "crp": 30,
    },
    "OCGT": {  # natural gas
        "display_name": "NG Combustion Turbine (F-Frame)",
        "technology": "NaturalGas_FE",
        "crp": 30,
    },
    "CCGT_95CCS": {  # natural gas
        "display_name": "NG Combined Cycle (F-Frame) 95% CCS",
        "technology": "NaturalGas_FE",
        "crp": 30,
    },
    "nuclear": {  # large scale nuclear
        "display_name": "Nuclear - AP1000",
        "crp": 60,
    },
    "SMR": {  # small modular reactor
        "display_name": "Nuclear - Small Modular Reactor",
        "crp": 60,
    },
    "solar-rooftop commercial": {
        "display_name": "Commercial PV - Class 5",
        "crp": 20,
    },
    "solar-rooftop": {
        "display_name": "Residential PV - Class 5",
        "crp": 20,
    },
    "central solar thermal": {
        "display_name": "CSP - Class 2",
        "crp": 30,
    },
    "home battery storage": {
        "display_name": "Residential Battery Storage - 5 kW - 12.5 kWh",
        "crp": 20,
    },
    "onwind": {
        "display_name": "Land-Based Wind - Class 4 - Technology 1",
        "technology": "LandbasedWind",
        "crp": 30,
    },
    "offwind": {
        "display_name": "Offshore Wind - Class 5",
        "technology": "OffShoreWind",
        "crp": 30,
    },
    "offwind_floating": {
        "display_name": "Offshore Wind - Class 13",
        "technology": "OffShoreWind",
        "crp": 30,
    },
    "Pumped-Storage-Hydro-bicharger": {
        "display_name": "Pumped Storage Hydropower - National Class 1",
        "crp": 100,
    },
    "solar": {
        "display_name": "Utility PV - Class 5",
        "crp": 20,
    },
    "2hr_battery_storage": {
        "display_name": "Utility-Scale Battery Storage - 2Hr",
        "crp": 20,
    },
    "4hr_battery_storage": {
        "display_name": "Utility-Scale Battery Storage - 4Hr",
        "crp": 20,
    },
    "6hr_battery_storage": {
        "display_name": "Utility-Scale Battery Storage - 6Hr",
        "crp": 20,
    },
    "8hr_battery_storage": {
        "display_name": "Utility-Scale Battery Storage - 8Hr",
        "crp": 20,
    },
    "10hr_battery_storage": {
        "display_name": "Utility-Scale Battery Storage - 10Hr",
        "crp": 20,
    },
}

###########################################
# Constants for NREL Locational Multipliers
###########################################

# {pypsa-name: csv-name}
CAPEX_LOCATIONAL_MULTIPLIER = {
    "nuclear": "nuclear-1117mw",
    # "oil",
    "CCGT": "natural-gas-430mw-90ccs",
    "OCGT": "natural-gas-430mw-90ccs",
    "coal": "coal-ultra-supercritical-90ccs",
    "geothermal": "geothermal-50mw",
    "solar": "spv-150mw",
    "onwind": "onshore-wind-200mw",
    "hydro": "hydro-100mw",
}
