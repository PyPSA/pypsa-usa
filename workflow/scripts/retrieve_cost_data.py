"""Retrieves cost data"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from _helpers import (
    progress_retrieve, 
    mock_snakemake
)
from typing import List, Dict, Union
import requests

logger = logging.getLogger(__name__)

def get_eia_data_api(url:str, api: str = None, facets: str = None) -> pd.DataFrame:
    """Gets EIA data using the API"""
    if not api:
        logger.warning("No API key provided")
        return pd.DataFrame()
    
    data = request_eia_api_data(url, api, facets)
    df = convert_eia_api_to_dataframe(data)
    return df
    
def request_eia_api_data(url:str, api: str, facets: str = None) -> Dict[str,Union[Dict,str]]:
    """Retrieves data from EIA API
    
    Args:
        url:str, 
            in the form of "https://api.eia.gov/v2/"
        api: str, 
            in the form of "xxxx"
        facets: str = None
            in the form of "frequency=monthly&data[0]=value&..." 
    """
    
    if facets:
        request = f"{url}?api_key={api}&{facets}"
    else:
        request = f"{url}?api_key={api}"
        
    response = requests.get(request)

    if response.status_code == 200:
        data = response.json()  # Assumes the response is in JSON format
    else:
        logger.warning(f"EIA Request failed with status code: {response.status_code}")
        data = {}
        
    return data

def convert_eia_api_to_dataframe(data: Dict[str,Union[Dict,str]]) -> pd.DataFrame:
    """Converts data called from EIA API to dataframe"""
    df = pd.DataFrame.from_dict(data["response"]["data"])
    df["period"] = pd.to_datetime(df["period"])
    df = df.set_index("period")
    return df

def format_eia_api_data(df: pd.DataFrame) -> pd.DataFrame:
    """Adds default values and 'state' column to data"""

    # add state 
    df["state"] = df["series-description"].map(lambda x: x.split("Natural Gas")[0].strip())
    df["value"] = df["value"].fillna(np.nan)
    usa_average = df[df.state == "U.S."].to_dict()["value"]

    df = df.reset_index()
    df["use_average"] = ~pd.notna(df["value"]) # not sure why this cant be built into the lambda function 
    df["value"] = df.apply(lambda x: x["value"] if not x["use_average"] else usa_average[x["period"]], axis=1)
    df = df.drop(columns=["use_average"])
    return df

def get_eia_data_xlsx(url: str) -> pd.DataFrame:
    """Gets EIA data via direct downlaod of a xlsx"""
    return pd.read_excel(url, sheet_name="Data 1", skiprows=2, index_col=0)

def format_eia_data_xlsx(df: pd.DataFrame) -> pd.DataFrame:
    """Formats costs data from direct download"""
    
    # not all states have data, so backfill using USA average in these cases
    fill_values = {x:df["U.S. Natural Gas Electric Power Price (Dollars per Thousand Cubic Feet)"] for x in df.columns}
    df = df.fillna(fill_values).reset_index()
    
    # unpivot data 
    df = df.melt(id_vars="Date", var_name="series-description")
    
    # adjust units and format to match structre returned from API
    df["units"] = df["series-description"].map(lambda x: x.split("(")[1].split(")")[0].strip())
    df["units"] = df["units"].map(lambda x: "$/MCF" if x == "Dollars per Thousand Cubic Feet" else x)
    df["state"] = df["series-description"].map(lambda x: x.split("Natural Gas")[0].strip())
    df["period"] = pd.to_datetime(df["Date"])
    df = df.set_index("period").drop(columns=["Date"])
    
    try:
        assert len(df["units"].unique() == 1)
    except AssertionError:
        logger.warning(f"Units inconsistent for EIA cost data")
    
    return df    
        
if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("retrieve_cost_data", year=2030)
        rootpath = ".."
    else:
        rootpath = "."

    # get nrel atb power generation data
    atb_year = 2023
    parquet = f"https://oedi-data-lake.s3.amazonaws.com/ATB/electricity/parquet/{atb_year}/ATBe.parquet"
    save_atb = snakemake.output.nrel_atb
    
    if not Path(save_atb).exists():
        logger.info(f"Downloading ATB costs from '{parquet}'")
        progress_retrieve(parquet, save_atb)
        
    # get nrel atb transportation data 
    xlsx = "https://atb-archive.nrel.gov/transportation/2020/files/2020_ATB_Data_VehFuels_Download.xlsx"
    save_atb_transport = snakemake.output.nrel_atb_transport
    
    if not Path(save_atb_transport).exists():
        logger.info(f"Downloading ATB transport costs from '{xlsx}'")
        progress_retrieve(xlsx, save_atb_transport)
        
    # get eia monthly fuel cost data 
    eia_api_key = snakemake.params.eia_api_key
    ng_electric_power_price = snakemake.output.ng_electric_power_price
    ng_industrial_price = snakemake.output.ng_industrial_price
    ng_commercial_price = snakemake.output.ng_commercial_price
    ng_residential_price = snakemake.output.ng_residential_price
    
    if eia_api_key:
        logger.info("Extracting EIA data via API")
        # electric power producer natural gas price
        if not Path(ng_electric_power_price).exists():
            url = "https://api.eia.gov/v2/natural-gas/pri/sum/data/"
            facets = "frequency=monthly&data[0]=value&facets[process][]=PEU&start=2022-01&end=2023-01&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
            logger.info(f"Downloading EIA electric power producer natural gas costs from '{url}'")
            df = get_eia_data_api(url, eia_api_key, facets)
            df = format_eia_api_data(df)
            df.to_csv(ng_electric_power_price)
        
        # industrial customer natural gas price
        if not Path(ng_industrial_price).exists():
            url = "https://api.eia.gov/v2/natural-gas/pri/sum/data/"
            facets = "frequency=monthly&data[0]=value&facets[process][]=PIN&start=2022-01&end=2023-01&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
            logger.info(f"Downloading EIA industrial natural gas costs from '{url}'")
            df = get_eia_data_api(url, eia_api_key, facets)
            df = format_eia_api_data(df)
            df.to_csv(ng_industrial_price)
            
        # commercial customer natural gas price
        if not Path(ng_commercial_price).exists():
            url = "https://api.eia.gov/v2/natural-gas/pri/sum/data/"
            facets = "frequency=monthly&data[0]=value&facets[process][]=PCS&start=2022-01&end=2023-01&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
            logger.info(f"Downloading EIA commercial natural gas costs from '{url}'")
            df = get_eia_data_api(url, eia_api_key, facets)
            df = format_eia_api_data(df)
            df.to_csv(ng_commercial_price)
            
        # residential customer natural gas price
        if not Path(ng_residential_price).exists():
            url = "https://api.eia.gov/v2/natural-gas/pri/sum/data/"
            facets = "frequency=monthly&data[0]=value&facets[process][]=PRS&start=2022-01&end=2023-01&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
            logger.info(f"Downloading EIA residential natural gas costs from '{url}'")
            df = get_eia_data_api(url, eia_api_key, facets)
            df = format_eia_api_data(df)
            df.to_csv(ng_residential_price)
    else:
        logger.info("Extracting EIA data via direct download")
        # electric power producer natural gas price
        if not Path(ng_electric_power_price).exists():
            url = "https://www.eia.gov/dnav/ng/xls/NG_PRI_SUM_A_EPG0_PEU_DMCF_M.xls"
            logger.info(f"Downloading EIA electric power producer natural gas costs from '{url}'")
            df = get_eia_data_xlsx(url)
            df = format_eia_data_xlsx(df)
            df = df[df.index.year == 2022]
            df.to_csv(ng_electric_power_price)
        
        # industrial customer natural gas price
        if not Path(ng_industrial_price).exists():
            url = "https://www.eia.gov/dnav/ng/xls/NG_PRI_SUM_A_EPG0_PIN_DMCF_M.xls"
            logger.info(f"Downloading EIA industrial natural gas costs from '{url}'")
            df = get_eia_data_xlsx(url)
            df = format_eia_data_xlsx(df)
            df = df[df.index.year == 2022]
            df.to_csv(ng_industrial_price)
        
        # commercial customer natural gas price
        if not Path(ng_commercial_price).exists():
            url = "https://www.eia.gov/dnav/ng/xls/NG_PRI_SUM_A_EPG0_PEU_DMCF_M.xls"
            logger.info(f"Downloading EIA commercial natural gas costs from '{url}'")
            df = get_eia_data_xlsx(url)
            df = format_eia_data_xlsx(df)
            df = df[df.index.year == 2022]
            df.to_csv(ng_commercial_price)
        
        # residential customer natural gas price
        if not Path(ng_residential_price).exists():
            url = "https://www.eia.gov/dnav/ng/xls/NG_PRI_SUM_A_EPG0_PEU_DMCF_M.xls"
            logger.info(f"Downloading EIA residential natural gas costs from '{url}'")
            df = get_eia_data_xlsx(url)
            df = format_eia_data_xlsx(df)
            df = df[df.index.year == 2022]
            df.to_csv(ng_residential_price)
            
    # get european template data 
    version = snakemake.params.pypsa_costs_version
    tech_year = snakemake.wildcards.year
    csv = f"https://raw.githubusercontent.com/PyPSA/technology-data/{version}/outputs/costs_{tech_year}.csv"
    save_tech_data = snakemake.output.pypsa_technology_data
    if not Path(save_tech_data).exists():
        logger.info(f"Downloading PyPSA-Eur costs from '{csv}'")
        progress_retrieve(csv, save_tech_data)