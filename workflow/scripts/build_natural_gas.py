"""Module for adding the gas sector"""

import pypsa 
import pandas as pd
import geopandas as gpd
from _helpers import configure_logging, mock_snakemake
import constants

###
# HELPERS
###

def get_state_center_points(shapefile: str) -> gpd.GeoDataFrame:
    """Gets centerpoints of states using county shapefile"""
    gdf = gpd.read_file(shapefile)
    gdf = (
        gdf
        .dissolve("STATE_NAME")
        .rename(columns={"STUSPS":"STATE", "geometry":"shape"})
        )
    gdf["geometry"] = gdf["shape"].map(lambda x: x.centroid)
    gdf = gdf[["STATE", "geometry"]]
    return gdf[~gdf.index.isin(constants.STATES_TO_REMOVE)]
    
###
# READING FUNCTIONS
###

def read_gas_import_export_locations(geojson: str) -> gpd.GeoDataFrame:
    """Reads in import/export data"""

    gdf = gpd.read_file("./Natural_Gas_Import_Export.geojson")
    gdf = gdf[gdf.STATUS == "IN SERVICE"]
    return gdf[["STATE","COUNTY","IMPVOL", "EXPVOL", "geometry"]].copy()

def read_gas_import_export_data(xlsx: str) -> pd.DataFrame:
    """Reads in import/export data
    
    Returns a dataframe with units in mmcf where first header is states 
    and second header is country import/export to/from
    """
    df = pd.read_excel("./NG_MOVE_POE1_A_EPG0_IRP_MMCF_A.xls", sheet_name="Data 1", skiprows=2, index_col=0)
    
    regex_pattern = r"U.S. Natural Gas Pipeline *"
    df_filtered = df.filter(regex=regex_pattern, axis=1)

    df = (
        df
        .drop(columns=df_filtered.columns)
        .rename(columns={x:x.replace("Million Cubic Feet", "MMcf") for x in df.columns})
        .fillna(0)
    )

    states = [x.split(",")[1][1:3] for x in df.columns]
    country = [x.split(" ")[-2] for x in df.columns]

    assert len(states) == len(country)
    assert all([x in ("Canada", "Mexico") for x in country])

    df.columns = pd.MultiIndex.from_tuples(zip(states, country))
    df.index = pd.to_datetime(df.index)
    
    return df

def read_eia_191(csv: str) -> pd.DataFrame:
    """Reads in EIA form 191 for natural gas storage"""
    df = pd.read_csv(csv, index_col=0)
    df = df.rename(columns={x:x.replace("<BR>", " ") for x in df.columns})
    df.columns = df.columns.str.strip()
    df = df[df.Status == "Active"]
    df = df[[
        "Report State", 
        "County Name", 
        "Working Gas Capacity(Mcf)", 
        "Total Field Capacity(Mcf)", 
        "Maximum Daily Delivery(Mcf)"
    ]]
    df = df.rename(columns={
        "Report State":"STATE", 
        "County Name":"COUNTY", 
        "Working Gas Capacity(Mcf)":"MIN_CAPACITY_MMCF", 
        "Total Field Capacity(Mcf)":"MAX_CAPACITY_MMCF", 
        "Maximum Daily Delivery(Mcf)":"MAX_DAILY_DELIEVERY_MMCF", 
    })
    df[["MIN_CAPACITY_MMCF", "MAX_CAPACITY_MMCF", "MAX_DAILY_DELIEVERY_MMCF"]] = (
        df[["MIN_CAPACITY_MMCF", "MAX_CAPACITY_MMCF", "MAX_DAILY_DELIEVERY_MMCF"]] * 0.001
    )

    return df.groupby(["STATE", "COUNTY"]).sum()

def read_eia_757(csv: str) -> pd.DataFrame:
    """Reads in EIA form 757 for natural gas processing"""
    df = pd.read_csv(csv).fillna(0)
    df = df.rename(columns={x:x.replace("<BR>", " ") for x in df.columns})
    df.columns = df.columns.str.strip()
    df = df[[
        "Report State", 
        "County Name", 
        "Plant Capacity", 
        "Plant Flow", 
        "BTU Content"
    ]]
    df["Report State"] = df["Report State"].str.capitalize()
    df = df.rename(columns={
        "Report State":"STATE", 
        "County Name":"COUNTY", 
        "Plant Capacity":"CAPACITY_MMCF", 
        "Plant Flow":"FLOW_MMCF", 
        "BTU Content":"BTU_CONTENT", 
    })

    return df.groupby(["STATE", "COUNTY"]).sum()

def read_gas_pipline(xlsx: str, year: int = 2022) -> pd.DataFrame:
    """Reads in state to state gas flow capacity """
    df = pd.read_excel(xlsx, sheet_name="Pipeline State2State Capacity", skiprows=1, index_col=0)
    df.columns = df.columns.str.strip()
    df = df[df.index == int(year)]
    df = df.rename(columns={
        "State From":"STATE_FROM",
        "County From":"COUNTRY_FROM",
        "State To":"STATE_TO",
        "County To":"COUNTRY_TO",
        "Capacity (mmcfd)":"CAPACITY_MMCFD"
    })
    df = df[["STATE_FROM","STATE_TO","CAPACITY_MMCFD"]]
    return df.groupby(["STATE_TO", "STATE_FROM"]).sum()

###
# BUS CREATION 
###

def build_state_gas_buses(n: pypsa.Network, states: gpd.GeoDataFrame) -> None:
    
    states["interconnect"] = states.STATE.map(constants.STATES_INTERCONNECT_MAPPER)
    
    n.madd(
        "Bus", 
        names=states.STATE,
        suffix=" gas",
        x=states.geometry.x,
        y=states.geometry.y,
        carrier="gas",
        unit="MMCF",
        interconnect=states.interconnect,
        country=states.STATE ,
        location=states.index # full state name
    )
    
###
# PRODUCERS
###

def build_gas_producers(n: pypsa.Network, producers: pd.DataFrame) -> None:
    """Applies EIA form 757 (gas production facilities)"""
    
    df = producers.reset_index().drop(columns=["COUNTY"]).groupby("STATE").sum()
    df["bus"] = df.index + " gas"
    
    n.madd(
        "Generator", 
        names=df.index,
        suffix=" gas production",
        bus=df.bus,
        carrier="gas",
        p_nom_extendable=False,
        marginal_costs=0.35, # to update (https://www.eia.gov/analysis/studies/drilling/pdf/upstream.pdf)
        p_nom=df.CAPACITY_MMCF
    )

###
# STORAGE
###

def build_storage_facilities(n: pypsa.Network, storage: pd.DataFrame, **kwargs) -> None:
    
    df = storage.reset_index().drop(columns=["COUNTY"]).groupby("STATE").sum()
    df["bus"] = df.index + " gas"
    
    n.madd(
        "Bus",
        names=df.index,
        suffix=" gas storage",
        carrier="gas storage",
        unit="MMCF",
    )
    
    cyclic_storage = kwargs.get("cyclic_storage", False)
    n.madd(
        "Store",
        names=df.index,
        suffix=" gas storage",
        bus=df.bus + " gas storage",
        carrier="gas storage",
        e_nom_extendable=False,
        e_nom=df.MAX_CAPACITY_MMCF,
        e_cyclic=cyclic_storage,
        e_min_pu=df.MAX_CAPACITY_MMCF / df.MAX_CAPACITY_MMCF,
        marginal_cost=0 # to update
    )
    
    # must do two links, rather than a bidirectional one, to constrain 
    # daily discharge limits 
    n.madd(
        "Link",
        names=df.index,
        suffix=" charge gas storage",
        carrier="gas storage",
        bus0=df.index + " gas",
        bus1=df.index + " gas storage",
        p_min_pu=0,
        p_max_pu=1,
        p_nom_extendable=False,
        marginal_cost=0
    )
    
    n.madd(
        "Link",
        names=df.index,
        suffix=" discharge gas storage",
        carrier="gas storage",
        bus0=df.index + " gas storage",
        bus1=df.index + " gas",
        p_min_pu=0,
        p_max_pu=1,
        p_nom_extendable=False,
        marginal_cost=0
    )

###
# PIPELINES
###



### 
# MAIN FUNCTION TO EXECUTE
###

def build_natural_gas(
    n: pypsa.Network,
    counties: str = "../data/counties/cb_2020_us_county_500k.shp",
    eia_757: str = "../data/natural-gas/EIA-757.csv",
    eia_191: str = "../data/natural-gas/EIA-191.csv",
) -> pypsa.Network:

    # create natural gas buses 
    # states = get_state_center_points(snakemake.input.counties)
    # build_state_gas_buses(states)
    
    # create production facilities (generators)
    # production_facilities = read_eia_757(eia_757)
    # build_gas_producers(n, production_facilities)
    
    # create storage facilities (storage units)
    storage_facilities = read_eia_191(eia_191)
    build_storage_facilities(n, storage_facilities)


if __name__ == "__main__":

    n = pypsa.Network("../resources/texas/elec_s_40_ec_lv1.25_Co2L1.25.nc")
    build_natural_gas(n=n)

    # read_gas_import_export_locations("./Natural_Gas_Import_Export.geojson")
    # read_gas_import_export_data("./NG_MOVE_POE1_A_EPG0_IRP_MMCF_A.xls")
    # read_gas_import_export_data("./NG_MOVE_POE1_A_EPG0_ENP_MMCF_A.xls")
    # read_eia_191("./EIA-191.csv")
    # read_eia_757("./EIA-757.csv")
    # read_gas_pipline("./EIA-StatetoStateCapacity_Jan2023.xlsx")