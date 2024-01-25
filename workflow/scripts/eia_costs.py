"""Module to extract EIA Cost Data

Available Fuels include: 
    - "gas"
    
Available Industries include: 
    - "power"
    - "residential" 
    - "commercial" 
    - "industry" 

Examples: 
>>> costs = EiaCosts(2020)
>>> costs.get_fuel_cost("gas", "power")

                                           series-description  value  units   state  
period                                                                               
2020-01-15  U.S. Natural Gas Electric Power Price (Dollars...   2.74  $/MCF    U.S.  
2020-02-15  U.S. Natural Gas Electric Power Price (Dollars...   2.50  $/MCF    U.S.  
...                                                       ...    ...    ...     ...  
2020-11-15  Wyoming Natural Gas Price Sold to Electric Pow...   3.09  $/MCF Wyoming  
2020-12-15  Wyoming Natural Gas Price Sold to Electric Pow...   3.30  $/MCF Wyoming


>>> # note, U.S.A price is still reported! 
>>> costs = EiaCostsApi(2020, "xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
>>> costs.get_fuel_cost("gas", "residential")

                                           series-description  value  units   state  
period                                                                               
2020-01-01  Alabama Price of Natural Gas Delivered to Resi...  14.62  $/MCF Alabama  
2020-02-01  Alabama Price of Natural Gas Delivered to Resi...  14.18  $/MCF Alabama  
...                                                       ...    ...    ...     ...  
2020-11-01  Wyoming Price of Natural Gas Delivered to Resi...   8.54  $/MCF Wyoming  
2020-12-01  Wyoming Price of Natural Gas Delivered to Resi...   8.00  $/MCF Wyoming

"""

from abc import ABC, abstractmethod
from typing import Union, Dict

import pandas as pd
import numpy as np
import requests

import logging

from pandas.core.api import DataFrame as DataFrame
logger = logging.getLogger(__name__)

class Strategy(ABC):
    """
    Arguments:
    """
    
    def __init__(self, year: int, api_key: str = None):
        self.api_key = api_key
        self._year = self._set_year(year)
    
    @staticmethod
    def _set_year(year: int) -> int:
        if year < 2005:
            logger.info(f"year must be > 2004. Recieved {year}. Setting to 2005")
            return 2005
        elif year > 2023:
            logger.info(f"year must be < 2024. Recieved {year}. Setting to 2023")
            return 2023
        else:
            return year
    
    @property
    def year(self):
        return self._year
        
    @year.setter
    def year(self, new_year: int) -> int:
        self._year = self._set_year(new_year)

    @abstractmethod
    def _get_url(self, fuel: str, industry: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def _retrieve_data(self, url: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def _format_data(self, fuel: str, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def get_fuel_cost(self, fuel: str, industry: str = "power") -> pd.DataFrame:
        
        # check for industries 
        industries = ("residential", "commercial", "industry", "power")
        if industry not in industries:
            logger.error(f"Industry must be in {industries}; recieved {industry}")
            return pd.DataFrame()
        
        # check for fuels
        fuels = ("gas", "coal")
        if fuel not in fuels:
            logger.error(f"Industry must be in {fuels}; recieved {fuel}")
            return pd.DataFrame()
        
        logger.info(f"Retrieving {industry} {fuel} data for {self.year}...")
        url = self._get_url(fuel, industry)
        df = self._retrieve_data(url)
        return self._format_data(fuel, df)
        

class EiaCosts(Strategy):
    """
    Retrieves data via direct downloads 
    """
    
    def __init__(self, year):
        super().__init__(year=year)
        
    def _get_url(self, fuel: str, industry: str) -> str:
        if fuel == "gas":
            if industry == "power":
                code = "PEU"
            elif industry == "residential":
                code = "PRS"
            elif industry == "commercial":
                code = "PCS"
            elif industry == "industry":
                code = "PIN"
            else:
                raise NotImplementedError
            return f"https://www.eia.gov/dnav/ng/xls/NG_PRI_SUM_A_EPG0_{code}_DMCF_M.xls"
        else:
            raise NotImplementedError
            
            
    def _retrieve_data(self, url: str) -> DataFrame:
        return pd.read_excel(
            url, 
            sheet_name="Data 1", 
            skiprows=2, 
            index_col=0
        )
        
    def _format_data(self, fuel: str, df: pd.DataFrame) -> DataFrame:
        if fuel == "gas":
            df = self._format_natural_gas(df)
            return df[df.index.year == self.year].copy() 
    
    @staticmethod
    def _format_natural_gas(df: pd.DataFrame) -> pd.DataFrame:
        
        # not all states have data, so backfill using USA average in these cases
        fill_column = [x for x in df.columns if x.startswith(
            (f"U.S. Natural Gas", "United States Natural Gas", "U.S. Price of Natural Gas")
        )]
        if len(fill_column) != 1:
            logger.warning(f"No fill column selected")
        else:
            fill_values = {x:df[fill_column[0]] for x in df.columns}
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
        
class EiaCostsApi(Strategy):
    """
    Retrieves data via API access
    """
    
    def __init__(self, year, api_key):
        super().__init__(year=year, api_key=api_key)
        
    def _get_url(self, fuel: str, industry: str) -> str:
        if fuel == "gas":
            if industry == "power":
                code = "PEU"
            elif industry == "residential":
                code = "PRS"
            elif industry == "commercial":
                code = "PCS"
            elif industry == "industry":
                code = "PIN"
            else:
                raise NotImplementedError
            
            base = "https://api.eia.gov/v2/natural-gas/pri/sum/data/"
            facets = f"frequency=monthly&data[0]=value&facets[process][]={code}&start={self.year}-01&end={self.year+1}-01&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
            
            return f"{base}?api_key={self.api_key}&{facets}"
        
        else:
            raise NotImplementedError
            
    def _retrieve_data(self, url: str) -> DataFrame:
        
        data = self._request_eia_data(url)
        if not data:
            raise ValueError
        
        df = pd.DataFrame.from_dict(data["response"]["data"])
        df["period"] = pd.to_datetime(df["period"])
        return df.set_index("period").copy()
        
    def _format_data(self, fuel: str, df: pd.DataFrame) -> pd.DataFrame:
        if fuel == "gas":
            return self._format_natural_gas(df)

    def _format_natural_gas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds default values and 'state' column to data"""

        # add state 
        # split happens twice to account for inconsistent naming 
        df["state"] = df["series-description"].map(lambda x: x.split("Natural Gas")[0].strip())
        df["state"] = df["state"].map(lambda x: x.split("Price of")[0].strip())
        df["value"] = df["value"].fillna(np.nan)
        
        df["state"] = df.state.map(lambda x: "U.S." if x == "United States" else x)
        usa_average = df[df.state == "U.S."].to_dict()["value"]

        df = df.reset_index()
        df["use_average"] = ~pd.notna(df["value"]) # not sure why this cant be built into the lambda function 
        df["value"] = df.apply(lambda x: x["value"] if not x["use_average"] else usa_average[x["period"]], axis=1)
        df = df.drop(columns=["use_average"]).copy()
        
        return (
            df
            [["series-description", "value", "units", "state", "period"]]
            .sort_values(["state", "period"])
            .set_index("period")
        )
        
    @staticmethod
    def _request_eia_data(url:str) -> Dict[str,Union[Dict,str]]:
        """Retrieves data from EIA API
        Args:
            url:str, 
                in the form of "https://api.eia.gov/v2/" with api and facets
        """
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()  # Assumes the response is in JSON format
        else:
            logger.error(f"EIA Request failed with status code: {response.status_code}")
            data = {}
            
        return data
    
    
    