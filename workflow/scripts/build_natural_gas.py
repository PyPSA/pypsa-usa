"""Module for adding the gas sector"""

import pypsa 
import pandas as pd
import geopandas as gpd
import constants
import logging
logger = logging.getLogger(__name__)
from typing import List, Dict
from math import pi
import numpy as np

###
# HELPERS
###

def get_state_boundaries(shapefile: str) -> pd.DataFrame:
    """Gets admin boundaries of state"""
    gdf = gpd.read_file(shapefile)
    return (
        gdf
        .dissolve("STATE_NAME")
        .rename(columns={"STUSPS":"STATE"})
        .reset_index()
        [["STATE_NAME", "STATE","geometry"]]
    )

def get_state_center_points(shapefile: str) -> pd.DataFrame:
    """Gets centerpoints of states using county shapefile"""
    gdf = get_state_boundaries(shapefile).rename(columns={"geometry":"shape"})
    gdf["geometry"] = gdf["shape"].map(lambda x: x.centroid)
    gdf[["x","y"]] = gdf["geometry"].apply(lambda x: pd.Series({"x": x.x, "y": x.y}))
    gdf = gdf[["STATE", "x", "y"]]
    return pd.DataFrame(gdf[~gdf.index.isin(constants.STATES_TO_REMOVE)])
    
def filter_on_interconnect(
    df: pd.DataFrame, 
    interconnect: str, 
    states_2_interconnect: Dict[str,str]
) -> pd.DataFrame:
    """Name of states must be in column called 'STATE' """
    
    if interconnect == "usa":
        return df
    else:
        df["interconnect"] = df.STATE.map(states_2_interconnect)
        assert not df.interconnect.isna().any() 
        df = df[df.interconnect == interconnect]
        if df.empty:
            logger.warning(f"Empty natural gas data for interconnect {interconnect}")
        return df.drop(columns="interconnect")
    
###
# READING FUNCTIONS
###

def read_gas_import_export_locations(geojson: str) -> gpd.GeoDataFrame:
    """Reads in import/export data"""

    gdf = gpd.read_file(geojson)
    gdf = gdf[gdf.STATUS == "IN SERVICE"]
    return gdf[["STATE","COUNTY","IMPVOL", "EXPVOL", "geometry"]].copy()

def read_gas_import_export_data(xlsx: str) -> pd.DataFrame:
    """Reads in import/export data
    
    Returns a dataframe with units in mmcf where first header is states 
    and second header is country import/export to/from
    """
    df = pd.read_excel(xlsx, sheet_name="Data 1", skiprows=2, index_col=0)
    
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

def read_eia_191(csv: str, **kwargs) -> pd.DataFrame:
    """Reads in EIA form 191 for natural gas storage"""
    
    df = pd.read_csv(csv, index_col=0)
    df = df.rename(columns={x:x.replace("<BR>", " ") for x in df.columns})
    df.columns = df.columns.str.strip()
    
    regions_to_remove = kwargs.get("regions_to_remove", None)
    if regions_to_remove:
        df = df[~df.Region.isin(regions_to_remove)]
    
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
    df["STATE"] = df["STATE"].str.upper()

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
    df["STATE"] = df["STATE"].str.upper()

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

def read_pipeline_linepack(geojson: str, states: gpd.GeoDataFrame) -> pd.DataFrame:
    """Reads in linepack energy limits
    
    https://atlas.eia.gov/apps/3652f0f1860d45beb0fed27dc8a6fc8d/explore
    """
    
    gdf = gpd.read_file(geojson)
    
    length_in_state = gpd.sjoin(gdf.to_crs("4269"), states, how="right", predicate="within").reset_index()
    length_in_state = (
        length_in_state[["STATE_NAME", "STATE", "TYPEPIPE", "Shape_Leng", "Shape__Length"]]
        .rename(columns={"Shape_Leng":"LENGTH_DEG", "Shape__Length":"LENGTH_M"})
        .groupby(by=["STATE_NAME", "STATE", "TYPEPIPE"]).sum().reset_index()
    )
    
    # https://publications.anl.gov/anlpubs/2008/02/61034.pdf
    intrastate_radius = 12 * 0.0254 # inches in meters (24in dia)
    interstate_radius = 18 * 0.0254 # inches meters (36in dia)

    volumne_in_state = length_in_state.copy()
    volumne_in_state["RADIUS"] = volumne_in_state.TYPEPIPE.map(lambda x: interstate_radius if x == "Interstate" else intrastate_radius)
    volumne_in_state["VOLUME_M3"] = volumne_in_state.LENGTH_M * pi * volumne_in_state.RADIUS ** 2 
    volumne_in_state = volumne_in_state[["STATE_NAME", "STATE", "VOLUME_M3"]]
    volumne_in_state = volumne_in_state.groupby(by=["STATE_NAME", "STATE"]).sum()
    
    # https://publications.anl.gov/anlpubs/2008/02/61034.pdf
    max_pressure = 8000 # kPa
    min_pressure = 4000 # kPa

    energy_in_state = volumne_in_state.copy()
    energy_in_state["MAX_ENERGY_kJ"] = energy_in_state.VOLUME_M3 * max_pressure
    energy_in_state["MIN_ENERGY_kJ"] = energy_in_state.VOLUME_M3 * min_pressure
    energy_in_state["NOMINAL_ENERGY_kJ"] = (energy_in_state.MAX_ENERGY_kJ + energy_in_state.MIN_ENERGY_kJ) / 2
    
    # https://apps.cer-rec.gc.ca/Conversion/conversion-tables.aspx#s1ss1
    # 1 GJ to 947.8171 CF
    # TODO: replace with heating value 
    kj_2_mmcf = 1e-6 * 947.8171 * 1e-6 # kj -> GJ -> cf -> mmcf
    
    final = energy_in_state.copy()
    final["MAX_ENERGY_MMCF"] = final.MAX_ENERGY_kJ * kj_2_mmcf
    final["MIN_ENERGY_MMCF"] = final.MIN_ENERGY_kJ * kj_2_mmcf
    final["NOMINAL_ENERGY_MMCF"] = final.NOMINAL_ENERGY_kJ * kj_2_mmcf

    return final[["STATE_NAME", "STATE", "MAX_ENERGY_MMCF", "MIN_ENERGY_MMCF", "NOMINAL_ENERGY_MMCF"]]

###
# BUSES 
###

def build_state_gas_buses(n: pypsa.Network, states: pd.DataFrame) -> None:
    
    # TODO: reformat states so names is on the index to remove the "to_list()"
    
    n.madd(
        "Bus", 
        names=states.STATE,
        suffix=" gas",
        x=states.x.to_list(),
        y=states.y.to_list(),
        carrier="gas",
        unit="MMCF",
        interconnect=states.interconnect.to_list(),
        country=states.STATE.to_list(), # for consistency 
        STATE=states.STATE.to_list(),
        STATE_NAME=states.index
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
    df["bus"] = df.index
    
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
        bus=df.index + " gas storage",
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
# IMPORTS/EXPORTS
###

def filter_imports_exports_on_interconnect(
    df: pd.DataFrame, 
    interconnect: str, 
    states_2_interconnect: Dict[str,str]
) -> pd.DataFrame:
    
    if interconnect != "usa":
        df = df[[x for x in df.columns if states_2_interconnect[x] in interconnect]]
        if df.empty:
            logger.warning(f"Empty natural gas import/export dataframe for interconnect {interconnect}")
    return df
            

def build_import_export_facilities(n: pypsa.Network, df: pd.DataFrame, direction: str) -> None:
    
    assert direction in ("import", "export")
    
    for country in ("Canada", "Mexico"):
        
        data = df.xs(country,level=1,axis=1).T.groupby(level=0).sum().T
        
        if data.empty: # ie. Texas does not connect to Canada
            continue
        
        n.madd(
            "Store",
            names=data.columns,
            suffix=f" gas {direction}",
            bus=data.columns + f" gas {direction}",
            carrier=f"gas {direction}",
            e_nom_extendable=True,
            capital_cost=0,
            e_nom=0,
            e_cyclic=False,
            marginal_cost=0,
        )
        
        
###
# PIPELINES
###

def assign_pipeline_interconnects(df: pd.DataFrame, states_2_interconnect: Dict[str,str]):
    
    df["INTERCONNECT_TO"] = df.STATE_TO.map(states_2_interconnect)
    df["INTERCONNECT_FROM"] = df.STATE_FROM.map(states_2_interconnect)
    
    assert not df.isna().any().any()    
    
    return df

def get_domestic_pipelines(df: pd.DataFrame, interconnect: str) -> pd.DataFrame:
    """Gets all pipelines fully within the interconnect"""
    
    if interconnect != "usa":
        df = df[
            (df.INTERCONNECT_TO == interconnect) & (df.INTERCONNECT_FROM == interconnect)
        ]
        if df.empty:
            logger.error(f"Empty natural gas domestic pipelines for interconnect {interconnect}")
    else:
        df = df[
            ~(df[["INTERCONNECT_TO", "INTERCONNECT_FROM"]].isin(["canada", "mexico"])).all(axis=1)
        ]
    return df
    
def get_domestic_pipeline_connections(df: pd.DataFrame, interconnect: str) -> pd.DataFrame:
    """Gets all pipelines within the usa that connect to the interconnect"""
    
    if interconnect == "usa":
        # no domestic connections 
        return pd.DataFrame(columns=df.columns)
    else:
        # get rid of international connections
        df = df[
            ~((df.INTERCONNECT_TO.isin(["canada", "mexico"])) | (df.INTERCONNECT_FROM.isin(["canada", "mexico"])))
        ]
        # get rid of pipelines within the interconnect 
        return df[
            (df["INTERCONNECT_TO"].eq(interconnect) | df["INTERCONNECT_FROM"].eq(interconnect)) & 
            ~(df["INTERCONNECT_TO"].eq(interconnect) & df["INTERCONNECT_FROM"].eq(interconnect)) 
        ]

def get_international_pipeline_connections(df: pd.DataFrame, interconnect: str) -> pd.DataFrame:
    """Gets all international pipeline connections"""
    df = df[
            (df.INTERCONNECT_TO.isin(["canada", "mexico"])) | 
            (df.INTERCONNECT_FROM.isin(["canada", "mexico"]))
        ]
    if interconnect == "usa":
        return df
    else:
        return df[
            (df.INTERCONNECT_TO == interconnect) | 
            (df.INTERCONNECT_FROM == interconnect)
        ]

def build_pipelines(n: pypsa.Network, df: pd.DataFrame) -> None:
    """Builds links between states"""
    
    df["NAME"] = df.STATE_FROM + " " + df.STATE_TO
    
    n.madd(
        "Link",
        names=df.NAME,
        suffix=" pipeline",
        carrier="gas pipeline",
        unit="MMCF",
        bus0=df.STATE_FROM + " gas",
        bus1=df.STATE_TO + " gas",
        p_nom=round(df.CAPACITY_MMCFD / 24), # get a hourly flow rate 
        p_min_pu=0,
        p_max_pu=1,
        p_nom_extendable=False,
        marginal_cost=0,
    )

def build_import_export_pipelines(n: pypsa.Network, df: pd.DataFrame, interconnect: str) -> None:
    """Builds import and export buses for pipelines to connect to.
    
    Dataframe must have a 'STATE_TO', 'STATE_FROM', 'INTERCONNECT_TO', and
    'INTERCONNECT_FROM' columns
    
    The function does the following
    - exisitng domestic buses are retained 
    - new import export buses are created based on region
        - "WA BC gas export"
        - "WA BC gas import"
    - new one way links are added with capacity limits 
        - "WA BC gas export" 
        - "WA BC gas import" 
    - stores are added WITHOUT energy limits 
        - "WA BC gas export" 
        - "WA BC gas import" 
    """
    
    if interconnect != "usa":
        to_from = df[df.INTERCONNECT_TO==interconnect].copy()
        from_to = df[df.INTERCONNECT_FROM==interconnect].copy()
    else:
        to_from = df[~df.INTERCONNECT_TO.isin(["canada", "mexico"])].copy()
        from_to = df[~df.INTERCONNECT_FROM.isin(["canada", "mexico"])].copy()
        
    to_from["NAME"] = to_from.STATE_FROM + " " + to_from.STATE_TO
    from_to["NAME"] = from_to.STATE_TO + " " + from_to.STATE_FROM
    
    to_from = to_from.set_index("NAME")
    from_to = from_to.set_index("NAME")
    
    n.madd(
        "Bus",
        names=to_from.index,
        suffix=" gas export",
        carrier="gas export",
        unit="MMCF",
    )
    
    n.madd(
        "Bus",
        names=from_to.index,
        suffix=" gas import",
        carrier="gas import",
        unit="MMCF",
    )
    
    n.madd(
        "Link",
        names=to_from.index,
        suffix=" gas export",
        carrier="gas export",
        unit="MMCF",
        bus0=to_from.STATE_FROM + " gas",
        bus1=to_from.index + " gas export",
        p_nom=round(to_from.CAPACITY_MMCFD / 24), # get a hourly flow rate 
        p_min_pu=0,
        p_max_pu=1,
        p_nom_extendable=False,
        marginal_cost=0,
    )
    
    n.madd(
        "Link",
        names=from_to.index,
        suffix=" gas import",
        carrier="gas import",
        unit="MMCF",
        bus0=from_to.index + " gas import",
        bus1=from_to.STATE_FROM + " gas",
        p_nom=round(from_to.CAPACITY_MMCFD / 24), # get a hourly flow rate 
        p_min_pu=0,
        p_max_pu=1,
        p_nom_extendable=False,
        marginal_cost=0,
    )
    
    n.madd(
        "Store",
        names=to_from.index,
        suffix=" gas export",
        unit="MMCF",
        bus=to_from.index + " gas export",
        carrier="gas export",
        e_nom_extendable=True,
        capital_cost=0,
        e_nom=0,
        e_cyclic=False,
        marginal_cost=0,
    )
    
    n.madd(
        "Store",
        names=from_to.index,
        unit="MMCF",
        suffix=" gas import",
        bus=from_to.index + " gas import",
        carrier="gas import",
        e_nom_extendable=True,
        capital_cost=0,
        e_nom=0,
        e_cyclic=False,
        marginal_cost=0,
    )

def build_linepack(n: pypsa.Network, df: pd.DataFrame) -> None:
    """Builds storage units to represent linepack"""
    
    df = df.set_index("STATE")
    
    n.madd(
        "StorageUnit",
        names=df.index,
        unit="MMCF",
        suffix=" linepack",
        bus=df.index + " gas",
        carrier="gas pipeline",
        p_nom=0,
        p_nom_extendable=False,
        p_nom_min=0,
        p_nom_max=np.inf,
        marginal_cost=0,
        capital_cost=0,
        state_of_charge_initial=df.NOMINAL_ENERGY_MMCF,
        state_of_charge_initial_per_period=False,
        cyclic_state_of_charge=True,
        cyclic_state_of_charge_per_period=False,
        max_hours=1,
        efficiency_store=1,
        efficiency_dispatch=1,
        standing_loss=0
    )
    
### 
# MAIN FUNCTION TO EXECUTE
###

def build_natural_gas(
    n: pypsa.Network,
    interconnect: str = "texas",
    counties: str = "../data/counties/cb_2020_us_county_500k.shp",
    eia_757: str = "../data/natural-gas/EIA-757.csv",
    eia_191: str = "../data/natural-gas/EIA-191.csv",
    imports: str = "../data/natural-gas/NG_MOVE_POE1_A_EPG0_IRP_MMCF_M.xls",
    exports: str = "../data/natural-gas/NG_MOVE_POE1_A_EPG0_ENP_MMCF_M.xls",
    pipelines: str = "../data/natural-gas/EIA-StatetoStateCapacity_Jan2023.xlsx",
    linepack: str = "../data/natural-gas/Natural_Gas_Pipelines.geojson",
) -> pypsa.Network:

    ###
    # CREATE GAS CARRIER
    ###
    
    n.add("Carrier","gas")

    ###
    # CREATE STATE LEVEL BUSES
    ###

    centroids = get_state_center_points(counties)
    centroids = filter_on_interconnect(centroids, interconnect, constants.STATES_INTERCONNECT_MAPPER)
    centroids["interconnect"] = centroids.STATE.map(constants.STATES_INTERCONNECT_MAPPER)
    build_state_gas_buses(n, centroids)
    
    ###
    # CREATE PRODUCTION FACILITIES
    ###
    
    production_facilities = read_eia_757(eia_757).reset_index()
    production_facilities = filter_on_interconnect(production_facilities, interconnect, constants.STATES_INTERCONNECT_MAPPER)
    build_gas_producers(n, production_facilities)
    
    ###
    # CREATE STORAGE FACILITIES
    ###
    
    storage_facilities = read_eia_191(eia_191, regions_to_remove=constants.STATES_TO_REMOVE).reset_index()
    storage_facilities = filter_on_interconnect(storage_facilities, interconnect, constants.STATES_INTERCONNECT_MAPPER)
    build_storage_facilities(n, storage_facilities)
    
    ###
    # CREATE PIPELINES
    ###
    
    pipelines = read_gas_pipline(pipelines).reset_index()
    
    pipelines = pipelines[~(
        (pipelines.STATE_TO == "Gulf of Mexico") | (pipelines.STATE_FROM == "Gulf of Mexico"))]
    pipelines.STATE_TO = pipelines.STATE_TO.map(constants.STATE_2_CODE)
    pipelines.STATE_FROM = pipelines.STATE_FROM.map(constants.STATE_2_CODE)
    
    pipelines = assign_pipeline_interconnects(pipelines, constants.STATES_INTERCONNECT_MAPPER)
    
    domestic_piplines = get_domestic_pipelines(pipelines, interconnect)
    domestic_pipeline_connections = get_domestic_pipeline_connections(pipelines, interconnect)
    international_pipeline_connections = get_international_pipeline_connections(pipelines, interconnect)
    
    if domestic_piplines.empty:
        logger.warning(f"No domestic gas pipelines to add for {interconnect}")
    else:
        build_pipelines(n, domestic_piplines)
        
    build_import_export_pipelines(n, domestic_pipeline_connections, interconnect)
    build_import_export_pipelines(n, international_pipeline_connections, interconnect)

    ###
    # CREATE INTERNATIONAL IMPORT EXPORT ENERGY LIMITS 
    ###
    
    # imports = read_gas_import_export_data(imports)
    # imports = filter_imports_exports_on_interconnect(imports, interconnect, constants.STATES_INTERCONNECT_MAPPER)
    
    # exports = read_gas_import_export_data(exports)
    # exports = filter_imports_exports_on_interconnect(exports, interconnect, constants.STATES_INTERCONNECT_MAPPER)
    
    # build_import_export_facilities(n, imports, "import")
    # build_import_export_facilities(n, exports, "export")

    ###
    # CREATE DOMESTIC IMPORTS EXPORTS
    ###
    
    

    ###
    # CREATE LIENPACK
    ###
    
    states = get_state_boundaries(counties)
    pipeline_linepack = read_pipeline_linepack(linepack, states)
    pipeline_linepack = filter_on_interconnect(pipeline_linepack, interconnect, constants.STATES_INTERCONNECT_MAPPER)
    build_linepack(n, pipeline_linepack)
    

if __name__ == "__main__":

    n = pypsa.Network("../resources/texas/elec_s_40_ec_lv1.25_Co2L1.25.nc")
    build_natural_gas(n=n)