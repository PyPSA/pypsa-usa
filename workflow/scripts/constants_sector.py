"""
Constants specific for sector coupling.
"""

from enum import Enum

###################
# General Constants
###################


class SecNames(Enum):
    RESIDENTIAL = "res"
    COMMERCIAL = "com"
    INDUSTRY = "ind"
    TRANSPORT = "trn"
    POWER = "pwr"


class SecCarriers(Enum):
    ELECTRICITY = "elec"
    HEATING = "heat"
    COOLING = "cool"
    LPG = "lpg"
    SPACE_HEATING = "space-heat"
    WATER_HEATING = "water-heat"


##############################
# Constants for Transportation
##############################


class Transport(Enum):
    ROAD = "veh"
    AIR = "air"
    BOAT = "boat"
    RAIL = "rail"


class RoadTransport(Enum):
    LIGHT = "lgt"
    MEDIUM = "med"
    HEAVY = "hvy"
    BUS = "bus"


class RoadTransportUnits(Enum):
    LIGHT = "kVMT"
    MEDIUM = "kVMT"
    HEAVY = "kVMT"
    BUS = "kVMT"


class AirTransport(Enum):
    PASSENGER = "psg"


class AirTransportUnits(Enum):
    PASSENGER = "kSeatMiles"


class RailTransport(Enum):
    PASSENGER = "psg"
    SHIPPING = "ship"


class RailTransportUnits(Enum):
    PASSENGER = "kPassengerMiles"
    SHIPPING = "kTonMiles"


class BoatTransport(Enum):
    SHIPPING = "ship"


class BoatTransportUnits(Enum):
    SHIPPING = "kTonMiles"


"""
These numbers are giving odd results :(
"""
# class TransportMwhConversion(Enum):
#     """Converts demand units into MWh approximates.

#     Used in Sankey diagrams

#     https://en.wikipedia.org/wiki/Energy_efficiency_in_transport
#     """
#     VEH_LGT_EV = 0.25 # (25 kWh / 100 miles) (10 100miles / 1 kmiles) (1 MWh / 1000kwh)
#     VEH_MED_EV = 0.40 # (400 Wh / 1 miles) (1000 miles / 1 kmiles) (1 kWh / 1000wh) (1 MWh / 1000kwh)
#     VEH_HVY_EV = 1.7 # (1.7 kWh / 1 miles) (1000 miles / 1 kmiles) (1 MWh / 1000kwh)
#     VEH_BUS_EV = 1.7 # (1.7 kWh / 1 miles) (1000 miles / 1 kmiles) (1 MWh / 1000kwh)
#     VEH_LGT_LPG = 0.915 # 1 / [(40 mpg) (1 kMiles / 1000 miles) (gal / 0.0366 MWh)]
#     VEH_MED_LPG = 1.464 # 1 / [(25 mpg) (1 kMiles / 1000 miles) (gal / 0.0366 MWh)]
#     VEH_HVY_LPG = 4.575 # 1 / [(8 mpg) (1 kMiles / 1000 miles) (gal / 0.0366 MWh)]
#     VEH_BUS_LPG = 4.575 # 1 / [(8 mpg) (1 kMiles / 1000 miles) (gal / 0.0366 MWh)]
#     AIR_PSG = 0.402 # 1 / [(91 psg-miles / gal) (1 k psg-miles / 1000 psg-miles) (gal / 0.0366 MWh)]
#     BOAT_SHIP = 0.033 # (74 kJ / ton-km) (1km / 0.621miles) (1000 miles / 1 kmiles) (kWh / 3600sec) (1 MWh / 1000 kWh)
#     RAIL_PSG =  0.078 # 1 / [(468 psg-miles / gal) (1 k psg-miles / 1000 psg-miles) (gal / 0.0366 MWh)]
#     RAIL_SHIP = 0.077 # kTon-Miles 1 / [(473 miles-ton / gallon) (1 kMiles / 1000 miles) (gal / 0.0366 MWh)]


class TransportEfficiency(Enum):
    """Approximate MWh/MWh efficiencies"""

    LPG = 0.20  # LLNL estimates 21 combined
    ELEC = 0.75


# bus conversion:
# - (pg 4) https://www.apta.com/wp-content/uploads/APTA-2022-Public-Transportation-Fact-Book.pdf
# - (141.5 / 999.5) = 0.14157
VMT_UNIT_CONVERSION = {
    "light_duty": 1,  # VMT
    "med_duty": 1,  # VMT
    "heavy_duty": 1,  # VMT
    "bus": 0.14157,  # PMT -> VMT
}

########################
# Constants for Industry
########################

# https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt
FIPS_2_STATE = {
    "01": "ALABAMA",
    "02": "ALASKA",
    "04": "ARIZONA",
    "05": "ARKANSAS",
    "06": "CALIFORNIA",
    "08": "COLORADO",
    "09": "CONNECTICUT",
    "10": "DELAWARE",
    "11": "DISTRICT OF COLUMBIA",
    "12": "FLORIDA",
    "13": "GEORGIA",
    "15": "HAWAII",
    "16": "IDAHO",
    "17": "ILLINOIS",
    "18": "INDIANA",
    "19": "IOWA",
    "20": "KANSAS",
    "21": "KENTUCKY",
    "22": "LOUISIANA",
    "23": "MAINE",
    "24": "MARYLAND",
    "25": "MASSACHUSETTS",
    "26": "MICHIGAN",
    "27": "MINNESOTA",
    "28": "MISSISSIPPI",
    "29": "MISSOURI",
    "30": "MONTANA",
    "31": "NEBRASKA",
    "32": "NEVADA",
    "33": "NEW HAMPSHIRE",
    "34": "NEW JERSEY",
    "35": "NEW MEXICO",
    "36": "NEW YORK",
    "37": "NORTH CAROLINA",
    "38": "NORTH DAKOTA",
    "39": "OHIO",
    "40": "OKLAHOMA",
    "41": "OREGON",
    "42": "PENNSYLVANIA",
    "44": "RHODE ISLAND",
    "45": "SOUTH CAROLINA",
    "46": "SOUTH DAKOTA",
    "47": "TENNESSEE",
    "48": "TEXAS",
    "49": "UTAH",
    "50": "VERMONT",
    "51": "VIRGINIA",
    "53": "WASHINGTON",
    "54": "WEST VIRGINIA",
    "55": "WISCONSIN",
    "56": "WYOMING",
}

# only grouped to level 2
# https://www23.statcan.gc.ca/imdb/p3VD.pl?Function=getVD&TVD=1181553
# https://github.com/NREL/Industry-Energy-Tool/tree/master/data_foundation/
NAICS = {
    11: "Agriculture, forestry, fishing and hunting Agriculture, forestry, fishing and hunting",
    21: "Mining, quarrying, and oil and gas extraction Mining, quarrying, and oil and gas extraction",
    22: "Utilities",
    23: "Construction",
    31: "Manufacturing",
    32: "Manufacturing",
    33: "Manufacturing",
    41: "Wholesale trade",
    44: "Retail trade",
    45: "Retail trade",
    48: "Transportation and warehousing",
    49: "Transportation and warehousing",
    51: "Information and cultural industries",
    52: "Finance and insurance",
    53: "Real estate and rental and leasing",
    54: "Professional, scientific and technical services",
    55: "Management of companies and enterprises",
    56: "Administrative and support, waste management and remediation services",
    61: "Educational services",
    62: "Health care and social assistance",
    71: "Arts, entertainment and recreation",
    72: "Accommodation and food services",
    81: "Other services (except public administration)",
    91: "Public administration",
}
