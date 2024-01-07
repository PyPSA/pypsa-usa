"""Module for adding the gas sector"""

import pypsa 
import pandas as pd
import geopandas as gpd

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

if __name__ == "__main__":
    read_gas_import_export_locations("./Natural_Gas_Import_Export.geojson")
    read_gas_import_export_data("./NG_MOVE_POE1_A_EPG0_IRP_MMCF_A.xls")
    read_gas_import_export_data("./NG_MOVE_POE1_A_EPG0_ENP_MMCF_A.xls")
    read_eia_191("./EIA-191.csv")
    read_eia_757("./EIA-757.csv")
    read_gas_pipline("./EIA-StatetoStateCapacity_Jan2023.xlsx")