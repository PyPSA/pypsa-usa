"""
Extracts EIA data 

Examples: 
>>> costs = FuelCosts("gas", "power", 2020, "xxxxxxxxxxxxxxxx")
>>> costs.get_data()

                                           series-description  value  units   state  
period                                                                               
2020-01-15  U.S. Natural Gas Electric Power Price (Dollars...   2.74  $/MCF    U.S.  
2020-02-15  U.S. Natural Gas Electric Power Price (Dollars...   2.50  $/MCF    U.S.  
...                                                       ...    ...    ...     ...  
2020-11-15  Wyoming Natural Gas Price Sold to Electric Pow...   3.09  $/MCF Wyoming  
2020-12-15  Wyoming Natural Gas Price Sold to Electric Pow...   3.30  $/MCF Wyoming


>>> costs = FuelCosts("gas", "residential", 2020, "xxxxxxxxxxxxxxxx")
>>> costs.get_data()

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
import math 
import constants
import yaml

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

import logging

logger = logging.getLogger(__name__)

API_BASE = "https://api.eia.gov/v2/"

STATE_CODES = constants.STATE_2_CODE

# exceptions
class InputException(Exception):
    """Class for exceptions"""
    
    def __init__(self, propery, valid_options, recived_option) -> None:
        self.message = f" {propery} must be in {valid_options}; recieved {recived_option}"
        
    def __str__(self):
        return self.message

# creator 
class EiaData(ABC):
    """Creator class to extract EIA data"""
    
    @abstractmethod
    def data_creator(self): # type DataExtractor
        """Gets the data"""
        pass
    
    def get_data(self) -> pd.DataFrame:
        product = self.data_creator()
        data = product.retrieve_data()
        return product.format_data(data)

# concrete creator 
class FuelCosts(EiaData):
    
    def __init__(self, fuel: str, industry: str, year: int, api: str) -> None:
        self.fuel = fuel
        self.industry = industry # (power|residential|commercial|industrial|imports|exports)
        self.year = year
        self.api = api
    
    def data_creator(self) -> pd.DataFrame:
        if self.fuel == "gas":
            return GasCosts(self.industry, self.year, self.api)
        elif self.fuel == "coal":
            return CoalCosts(self.industry, self.year, self.api)
        else:
            raise InputException(
                propery="Fuel Costs", 
                valid_options=["gas", "coal"], 
                recived_option=self.fuel
            )

# concrete creator 
class Trade(EiaData):
    
    def __init__(self, fuel: str, direction: str, year: int, api: str) -> None:
        self.fuel = fuel
        self.direction = direction # (imports|exports)
        self.year = year
        self.api = api

    def data_creator(self) -> pd.DataFrame:
        if self.fuel == "gas":
            return GasTrade(self.direction, self.year, self.api)
        else:
            raise InputException(
                propery="Energy Trade", 
                valid_options=["gas"], 
                recived_option=self.fuel
            )

# concrete creator 
class Production(EiaData):
    
    def __init__(self, fuel: str, production: str, year: int, api: str) -> None:
        self.fuel = fuel
        self.production = production # (marketed|gross)
        self.year = year
        self.api = api
    
    def data_creator(self) -> pd.DataFrame:
        if self.fuel == "gas":
            return GasProduction(self.production, self.year, self.api)
        else:
            raise InputException(
                property="Production",
                valid_options=["gas"],
                recieved_option=self.fuel
            )

# concrete creator 
class Demand(EiaData):
    """Not yet implemented"""
    
    def __init__(self, fuel: str, year: int, api: str) -> None:
        self.fuel = fuel
        self.year = year
        self.api = api
    
    def data_creator(self) -> pd.DataFrame:
        if self.fuel == "electricity":
            raise NotImplementedError()
        else:
            raise InputException(
                propery="Demand", 
                valid_options=["electricity"], 
                recived_option=self.fuel
            )

class Storage(EiaData):
    
    def __init__(self, fuel: str, storage: str, year: int, api: str) -> None:
        self.fuel = fuel
        self.storage = storage # (base|working|total|withdraw)
        self.year = year
        self.api = api

    def data_creator(self) -> pd.DataFrame:
        if self.fuel == "gas":
            return GasStorage(self.storage, self.year, self.api)
        else:
            raise InputException(
                propery="Storage", 
                valid_options=["gas"], 
                recived_option=self.fuel
            )

# product
class DataExtractor(ABC):
    """extracts and formats data"""
    
    def __init__(self, year: int, api_key: str = None):
        self.api_key = api_key
        self.year = self._set_year(year)
    
    @abstractmethod
    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Formats retrieved data from EIA
        
                    series-description    value  units   state  
        period                                                   
        2020-01-15  description of data   2.74  $/MCF    U.S.  
        ...
        """
        pass
    
    @abstractmethod
    def build_url(self) -> str:
        """Builds API url"""
        pass
    
    def retrieve_data(self) -> pd.DataFrame:
        url = self.build_url()
        data = self._request_eia_data(url)
        return pd.DataFrame.from_dict(data["response"]["data"])
    
    @staticmethod
    def _set_year(year: int) -> int:
        if year < 2009:
            logger.info(f"year must be > 2008. Recieved {year}. Setting to 2009")
            return 2009
        elif year > 2022:
            logger.info(f"year must be < 2023. Recieved {year}. Setting to 2022")
            return 2022
        else:
            return year
    
    @staticmethod
    def _request_eia_data(url:str) -> Dict[str,Union[Dict,str]]:
        """Retrieves data from EIA API
        
        url in the form of "https://api.eia.gov/v2/" followed by api key and facets
        """
        
        # sometimes running into HTTPSConnectionPool error. adding in retries helped
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        response = session.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()  # Assumes the response is in JSON format
        else:
            logger.error(f"EIA Request failed with status code: {response.status_code}")
            raise requests.ConnectionError(f"Status code {response.status_code}")

    @staticmethod
    def _format_period(dates: pd.Series) -> pd.Series:
        """Parses dates into a standard monthly format"""
        try: # try to convert to YYYY-MM-DD format
            return pd.to_datetime(dates, format="%Y-%m-%d")
        except ValueError:
            try: # try to convert to YYYY-MM format
                return pd.to_datetime(dates + "-01", format="%Y-%m-%d")
            except ValueError:
                return pd.NaT

# concrete product 
class GasCosts(DataExtractor):
    
    industry_codes = {
        "power":"PEU",
        "residential":"PRS",
        "commercial":"PCS",
        "industrial":"PIN",
        "imports":"PRP",
        "exports":"PNP",
    }
    
    def __init__(self, industry: str, year: int, api_key: str) -> None:
        self.industry = industry
        if industry not in self.industry_codes.keys():
            raise InputException(
                propery="Gas Costs", 
                valid_options=list(self.industry_codes), 
                recived_option=industry
            )
        super().__init__(year, api_key)
    
    def build_url(self) -> str:
        base_url = "natural-gas/pri/sum/data/"
        facets = f"frequency=monthly&data[0]=value&facets[process][]={self.industry_codes[self.industry]}&start={self.year}-01&end={self.year+1}-01&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"
    
    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Formats natural gas cost data"""
        
        # format dates
        df["period"] = self._format_period(df.period)
        df = df.set_index("period").copy()

        # split happens twice to account for inconsistent naming 
        # Sometimes "U.S. Natural Gas price"
        # Sometimes "Price of U.S. Natural Gas"
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
        
# concrete product 
class CoalCosts(DataExtractor):
    
    industry_codes = {
        "power":"PEU",
    }
    
    def __init__(self, industry: str, year: int, api_key: str) -> None:
        self.industry = industry
        if industry != "power":
            raise InputException(
                propery="Coal Costs", 
                valid_options=list(self.industry_codes), 
                recived_option=industry
            )
        super().__init__(year, api_key)
            
    def build_url(self) -> str:
        base_url = "coal/shipments/by-mine-by-plant/data/"
        facets = f"frequency=quarterly&data[0]=price&start={self.year}-Q1&end={self.year+1}-Q1&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"
            
    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df[
            (df.coalType.isin(("All", "all"))) & 
            ~(df.price == "w") & # withheld value. Will be assigned usa average 
            ~(df.price.isna()) 
        ].copy()
        
        # sometimes prices come in the format of xx.xx.xx, so drop everything after the second "."
        df["price"] = df.price.map(lambda x: float(x) if len(x.split(".")) < 2 else float(".".join(x.split(".")[:2])))
        
        # get data at a per quarter level 
        df[["year", "quarter"]] = df.period.str.split("-", expand=True)
        df = (
            df
            [["plantStateDescription", "price", "price-units", "year", "quarter"]]
            .rename(columns={"plantStateDescription":"state", "price-units":"unit"})
            .groupby(by=["state", "unit", "year", "quarter"]).mean()
            .reset_index()
        )
        df = df[df.year.astype(int) == self.year].copy() # api is bringing in an extra quarter 
        
        # Expand data to be at a per month level 
        dfs = []
        dates = pd.date_range(start=f"{self.year}-01-01", end="{self.year}-12-01", freq="MS")
        for date in dates:
            quarter = math.floor((date.month - 1) / 3) + 1 # months 1-3 are Q1, months 4-6 are Q2 ... 
            df_month = df[df.quarter == f"Q{quarter}"].copy()
            df_month["period"] = date
            dfs.append(df_month)
        states = pd.concat(dfs).drop(columns=["year", "quarter"])

        # add a usa average datapoint for missing values 
        usa_average = []
        for date in dates:
            usa_average.append([
                "U.S.",
                "average dollars per ton",
                states[states.period == date].price.mean(),
                date
            ])
            usa = pd.DataFrame(usa_average, columns=states.columns)

        final = pd.concat([states, usa]).reset_index(drop=True)
        final["series-description"] = final.state.map(lambda x: f"{x} Coal Electric Power Price")

        return final.set_index("period")

class ElectricityDemand(DataExtractor):
    """Extracts demand by balancing authority 
    
    TODO: Develop method to extract data 5000 entries at a time. 
    We can probably use the offset option in the EIA API to just call it 
    however many times we need
    https://www.eia.gov/opendata/documentation.php
    """
    
    def __init__(self, year: int, api_key: str) -> None:
        super().__init__(year, api_key)
        
    def build_url(self) -> str:
        base_url = "electricity/rto/region-data/data/"
        facets = f"frequency=hourly&data[0]=value&facets[type][]=D&start={self.year}-01-01T00&end={self.year}-12-31T00&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"
    
    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

class GasTrade(DataExtractor):
    """Gets imports/exports by point of entry"""
    
    direction_codes = {
        "imports":"IRP",
        "exports":"ENP"
    }

    def __init__(self, direction: str, year: int, api_key: str) -> None:
        self.direction = direction
        if self.direction not in list(self.direction_codes):
            raise InputException(
                propery="Natural Gas Imports and Exports", 
                valid_options=list(self.direction_codes), 
                recived_option=direction
            )
        super().__init__(year, api_key)
            
    def build_url(self) -> str:
        poe = "poe1" if self.direction == "imports" else "poe2"
        base_url = f"natural-gas/move/{poe}/data/"
        facets = f"frequency=monthly&data[0]=value&facets[process][]={self.direction_codes[self.direction]}&start={self.year}-01&end={self.year+1}-01&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"
    
    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df["period"] = self._format_period(df.period).copy()
        df["state"] = df["series-description"].map(self.extract_state)
        
        return (
            df
            [["series-description", "value", "units", "state", "period"]]
            .sort_values(["state", "period"])
            .set_index("period")
        )
    
    @staticmethod
    def extract_state(description: str) -> str:
        """Extracts state from series descripion
        
        Input will be in one of the following forms
        - "Massena, NY Natural Gas Pipeline Imports From Canada"
        - "U.S. Natural Gas Pipeline Imports From Mexico"
        """
        try: # state level 
            return description.split(",")[1].split(" ")[1] 
        except IndexError: # country level 
            return description.split(" Natural Gas Pipeline")[0]

class GasStorage(DataExtractor):
    """Underground storage facilites for natural gas"""
    
    # https://www.eia.gov/naturalgas/storage/basics/
    storage_codes = {
        "base":"SAB", 
        "working":"SAO",
        "total":"SAT",
        "withdraw":"SAW"
    }
    
    def __init__(self, storage: str, year: int, api_key: str) -> None:
        self.storage = storage
        if self.storage not in list(self.storage_codes):
            raise InputException(
                propery="Natural Gas Underground Storage", 
                valid_options=list(self.storage_codes), 
                recived_option=storage
            )
        super().__init__(year, api_key)

    def build_url(self) -> str:
        base_url = "natural-gas/stor/sum/data/"
        facets = f"frequency=monthly&data[0]=value&facets[process][]={self.storage_codes[self.storage]}&start={self.year}-01&end={self.year+1}-01&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df[~(df["area-name"] == "NA")].copy()
        df["period"] = self._format_period(df.period)
        df["state"] = (
            df["series-description"]
            .map(self.extract_state)
            .map(self.map_state_names)
        )
        
        return (
            df
            [["series-description", "value", "units", "state", "period"]]
            .sort_values(["state", "period"])
            .set_index("period")
        )

    @staticmethod
    def extract_state(description: str) -> str:
        """Extracts state from series descripion"""
        return description.split(" Natural ")[0]
    
    @staticmethod
    def map_state_names(state: str) -> str:
        """Maps state name to code"""
        return "U.S." if state == "U.S. Total" else STATE_CODES[state]

class GasProduction(DataExtractor):
    """Dry natural gas production"""

    production_codes = {
        "market":"VGM", 
        "gross":"FGW", # gross withdrawls
    }
    
    def __init__(self, production: str, year: int, api_key: str) -> None:
        self.production = production
        if self.production not in list(self.production_codes):
            raise InputException(
                propery="Natural Gas Production", 
                valid_options=list(self.production_codes), 
                recived_option=production
            )
        super().__init__(year, api_key)

    def build_url(self) -> str:
        base_url = "natural-gas/prod/sum/data/"
        facets = f"frequency=monthly&data[0]=value&facets[process][]={self.production_codes[self.production]}&start={self.year}-01&end={self.year+1}-01&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df[~(df["area-name"] == "NA")].copy()
        df["period"] = self._format_period(df.period)
        df["state"] = (
            df["series-description"]
            .map(self.extract_state)
            .map(self.map_state_names)
        )
        
        return (
            df
            [["series-description", "value", "units", "state", "period"]]
            .sort_values(["state", "period"])
            .set_index("period")
        )
        
    @staticmethod
    def extract_state(description: str) -> str:
        """Extracts state from series descripion
        
        Input will be in one of the following forms
        - "Maryland Natural Gas Marketed Production (MMcf)"
        - "Idaho Marketed Production of Natural Gas (MMcf)"
        """
        return description.split(" Natural Gas")[0].split(" Marketed")[0]
    
    @staticmethod
    def map_state_names(state: str) -> str:
        """Maps state name to code"""
        return "U.S." if state == "U.S." else STATE_CODES[state]

if __name__ == "__main__":
    with open("./../config/config.api.yaml", "r") as file:
        yaml_data = yaml.safe_load(file)
    api = yaml_data["api"]["eia"]
    print(Production("gas", "market", 2020, api).get_data())