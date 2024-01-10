"""Module for adding the gas sector"""

import pypsa 
import pandas as pd
import geopandas as gpd
import constants
import logging
logger = logging.getLogger(__name__)
from typing import List, Dict

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
        df = df[df.interconnect.isin(interconnect)]
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
# BUSES 
###

def build_state_gas_buses(n: pypsa.Network, states: gpd.GeoDataFrame, interconnect: str = "usa") -> None:
    
    states["interconnect"] = states.STATE.map(constants.STATES_INTERCONNECT_MAPPER)
    
    if interconnect != "usa":
        states = states[states.interconnect.isin(interconnect)]
        if states.empty:
            logger.warning(f"Empty natural gas buses dataframe for interconnect {interconnect}")
            return
    
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

def build_gas_producers(n: pypsa.Network, producers: pd.DataFrame, interconnect: str = "usa") -> None:
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
        to_from = df[df.INTERCONENCT_TO==interconnect]
        from_to = df[df.INTERCONENCT_FROM==interconnect]
    else:
        to_from = df[~df.INTERCONENCT_TO.isin(["canada", "mexico"])]
        from_to = df[~df.INTERCONENCT_FROM.isin(["canada", "mexico"])]
        
    to_from["NAME"] = to_from.STATE_FROM + " " + to_from.STATE_TO
    from_to["NAME"] = from_to.STATE_TO + " " + from_to.STATE_FROM
    
    n.madd(
        "Bus",
        names=to_from.NAME,
        suffix=" gas export",
        carrier="gas export",
        unit="MMCF",
    )
    
    n.madd(
        "Bus",
        names=from_to.NAME,
        suffix=" gas import",
        carrier="gas import",
        unit="MMCF",
    )
    
    n.madd(
        "Link",
        names=to_from.NAME,
        suffix=" gas export",
        carrier="gas export",
        unit="MMCF",
        bus0=to_from.STATE_FROM + " gas",
        bus1=to_from.NAME + " gas export",
        p_nom=round(to_from.CAPACITY_MMCFD / 24), # get a hourly flow rate 
        p_min_pu=0,
        p_max_pu=1,
        p_nom_extendable=False,
        marginal_cost=0,
    )
    
    n.madd(
        "Link",
        names=from_to.NAME,
        suffix=" gas import",
        carrier="gas import",
        unit="MMCF",
        bus0=from_to.NAME + " gas import",
        bus1=from_to.STATE_FROM + " gas",
        p_nom=round(from_to.CAPACITY_MMCFD / 24), # get a hourly flow rate 
        p_min_pu=0,
        p_max_pu=1,
        p_nom_extendable=False,
        marginal_cost=0,
    )
    
    n.madd(
        "Store",
        names=to_from.NAME,
        suffix=" gas export",
        unit="MMCF",
        bus=to_from.NAME + " gas export",
        carrier="gas export",
        e_nom_extendable=True,
        capital_cost=0,
        e_nom=0,
        e_cyclic=False,
        marginal_cost=0,
    )
    
    n.madd(
        "Store",
        names=from_to.NAME,
        unit="MMCF",
        suffix=" gas import",
        bus=from_to.NAME + " gas import",
        carrier="gas import",
        e_nom_extendable=True,
        capital_cost=0,
        e_nom=0,
        e_cyclic=False,
        marginal_cost=0,
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
    pipelines: str = "../data/natural-gas/EIA-StatetoStateCapacity_Jan2023.xlsx"
) -> pypsa.Network:

    # create natural gas buses 
    # states = get_state_center_points(snakemake.input.counties)
    # build_state_gas_buses(states)
    
    ###
    # CREATE PRODUCTION FACILITIES
    ###
    
    # production_facilities = read_eia_757(eia_757)
    # production_facilities = filter_on_interconnect(production_facilities, interconnect, constants.STATES_INTERCONNECT_MAPPER)
    # build_gas_producers(n, production_facilities)
    
    ###
    # CREATE STORAGE FACILITIES
    ###
    
    # storage_facilities = read_eia_191(eia_191)
    # storage_facilities = filter_on_interconnect(storage_facilities, interconnect, constants.STATES_INTERCONNECT_MAPPER)
    # build_storage_facilities(n, storage_facilities)
    
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

if __name__ == "__main__":

    n = pypsa.Network("../resources/texas/elec_s_40_ec_lv1.25_Co2L1.25.nc")
    build_natural_gas(n=n)

    # read_gas_import_export_locations("./Natural_Gas_Import_Export.geojson")
    # read_gas_import_export_data("./NG_MOVE_POE1_A_EPG0_IRP_MMCF_A.xls")
    # read_gas_import_export_data("./NG_MOVE_POE1_A_EPG0_ENP_MMCF_A.xls")
    # read_eia_191("./EIA-191.csv")
    # read_eia_757("./EIA-757.csv")
    # read_gas_pipline("./EIA-StatetoStateCapacity_Jan2023.xlsx")