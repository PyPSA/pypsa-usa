"""
Extracts EIA data.

Public Classes include:
- FuelCosts(fuel, industry, year, api)
- Trade(fuel, direction, year, api)
- Production(fuel, production, year, api)
- EnergyDemand(sector, year, api, scenario)
- Storage(fuel, storage, year, api)
- Emissions(sector, year, api, fuel)

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

import logging
import math
from abc import ABC, abstractmethod
from typing import Optional

import constants
import numpy as np
import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = logging.getLogger(__name__)

API_BASE = "https://api.eia.gov/v2/"

STATE_CODES = constants.STATE_2_CODE

# https://www.eia.gov/outlooks/aeo/assumptions/case_descriptions.php
AEO_SCENARIOS = {
    "reference": "ref2023",  # reference
    "aeo2022": "aeo2022ref",  # AEO2022 Reference case
    "no_ira": "noIRA",  # No inflation reduction act
    "low_ira": "lowupIRA",  # Low Uptake of Inflation Reduction Act
    "high_ira": "highupIRA",  # High Uptake of Inflation Reduction Act
    "high_growth": "highmacro",  # High Economic Growth
    "low_growth": "lowmacro",  # Low Economic Growth
    "high_oil_price": "highprice",  # High Oil Price
    "low_oil_price": "lowprice",  # Low Oil Price
    "high_oil_gas_supply": "highogs",  # High Oil and Gas Supply
    "low_oil_gas_supply": "lowogs",  # Low Oil and Gas Supply
    "high_ztc": "highZTC",  # High Zero-Carbon Technology Cost
    "low_ztc": "lowZTC",  # Low Zero-Carbon Technology Cost
    "high_growth_high_ztc": "highmachighZTC",  # High Economic Growth-High Zero-Carbon Technology Cost
    "high_growth_low_ztc": "highmaclowZTC",  # High Economic Growth-Low Zero-Carbon Technology Cost
    "low_growth_high_ztc": "lowmachighZTC",  # Low Economic Growth-High Zero-Carbon Technology Cost
    "low_growth_low_ztc": "lowmaclowZTC",  # Low Economic Growth-Low Zero-Carbon Technology Cost
    "fast_build_high_lng": "lng_hp_fast",  # Fast Builds Plus High LNG Price
    "high_lng": "lng_hp",  # High LNG Price
    "low_lng": "lng_lp",  # Low LNG Price
}


# exceptions
class InputException(Exception):
    """
    Class for exceptions.
    """

    def __init__(self, propery, valid_options, recived_option) -> None:
        self.message = (
            f" {propery} must be in {valid_options}; recieved {recived_option}"
        )

    def __str__(self):
        return self.message


# creator
class EiaData(ABC):
    """
    Creator class to extract EIA data.
    """

    @abstractmethod
    def data_creator(self):  # type DataExtractor
        """
        Gets the data.
        """
        pass

    def get_data(self, pivot: bool = False) -> pd.DataFrame:
        product = self.data_creator()
        df = product.retrieve_data()
        df = product.format_data(df)
        if pivot:
            df = product._pivot_data(df)
        return df

    def get_api_call(self) -> pd.DataFrame:
        product = self.data_creator()
        return product.build_url()

    def get_raw_data(self) -> pd.DataFrame:
        product = self.data_creator()
        return product.retrieve_data()


# concrete creator
class FuelCosts(EiaData):

    def __init__(
        self,
        fuel: str,
        year: int,
        api: str,
        industry: Optional[str] = None,
        grade: Optional[str] = None,
    ) -> None:
        self.fuel = fuel
        self.year = year
        self.api = api
        self.industry = (
            industry  # (power|residential|commercial|industrial|imports|exports)
        )
        self.grade = grade  # (total|regular|premium|midgrade|diesel)

    def data_creator(self) -> pd.DataFrame:
        if self.fuel == "gas":
            assert self.industry
            return GasCosts(self.industry, self.year, self.api)
        elif self.fuel == "coal":
            assert self.industry
            return CoalCosts(self.industry, self.year, self.api)
        elif self.fuel == "lpg":
            assert self.grade
            return LpgCosts(self.grade, self.year, self.api)
        elif self.fuel == "heating_oil":
            return HeatingFuelCosts("fuel_oil", self.year, self.api)
        elif self.fuel == "propane":
            return HeatingFuelCosts("propane", self.year, self.api)
        else:
            raise InputException(
                propery="Fuel Costs",
                valid_options=["gas", "coal", "lpg", "heating_oil", "propane"],
                recived_option=self.fuel,
            )


# concrete creator
class Trade(EiaData):

    def __init__(self, fuel: str, direction: str, year: int, api: str) -> None:
        self.fuel = fuel
        self.direction = direction  # (imports|exports)
        self.year = year
        self.api = api

    def data_creator(self) -> pd.DataFrame:
        if self.fuel == "gas":
            return GasTrade(self.direction, self.year, self.api)
        else:
            raise InputException(
                propery="Energy Trade",
                valid_options=["gas"],
                recived_option=self.fuel,
            )


# concrete creator
class Production(EiaData):

    def __init__(self, fuel: str, production: str, year: int, api: str) -> None:
        self.fuel = fuel  # (gas)
        self.production = production  # (marketed|gross)
        self.year = year
        self.api = api

    def data_creator(self) -> pd.DataFrame:
        if self.fuel == "gas":
            return GasProduction(self.production, self.year, self.api)
        else:
            raise InputException(
                property="Production",
                valid_options=["gas"],
                recieved_option=self.fuel,
            )


# concrete creator
class EnergyDemand(EiaData):
    """
    Energy demand at a annual national level.

    If historical year is provided, monthly energy consumption for that
    year is provided. If a future year is provided, annual projections
    from 2023 up to that year are provided based on the scenario given
    """

    def __init__(
        self,
        sector: str,
        year: int,
        api: str,
        scenario: Optional[str] = None,
    ) -> None:
        self.sector = sector  # (residential, commercial, transport, industry)
        self.year = year
        self.api = api
        self.scenario = scenario  # only for AEO scenario

    def data_creator(self) -> pd.DataFrame:
        if self.year < 2024:
            if self.scenario:
                logger.warning("Can not apply AEO scenario to historical demand")
            return HistoricalSectorEnergyDemand(self.sector, self.year, self.api)
        elif self.year >= 2024:
            aeo = "reference" if not self.scenario else self.scenario
            return ProjectedSectorEnergyDemand(self.sector, self.year, aeo, self.api)
        else:
            raise InputException(
                propery="EnergyDemand",
                valid_options="year",
                recived_option=self.year,
            )


class TransportationDemand(EiaData):
    """
    Transportation demand in VMT (or similar).

    If historical year is provided, monthly energy consumption for that
    year is provided. If a future year is provided, annual projections
    from 2023 up to that year are provided based on the scenario given
    """

    def __init__(
        self,
        vehicle: str,
        year: int,
        api: str,
        units: str = "travel",  # travel | btu
        scenario: Optional[str] = None,
    ) -> None:
        self.vehicle = vehicle
        self.year = year
        self.api = api
        self.units = units
        self.scenario = scenario

    def data_creator(self) -> pd.DataFrame:
        if self.units == "travel":
            if self.year < 2024:
                return HistoricalTransportTravelDemand(
                    self.vehicle,
                    self.year,
                    self.api,
                )
            elif self.year >= 2024:
                aeo = "reference" if not self.scenario else self.scenario
                return ProjectedTransportTravelDemand(
                    self.vehicle,
                    self.year,
                    aeo,
                    self.api,
                )
            else:
                raise InputException(
                    propery="TransportationTravelDemand",
                    valid_options=range(2017, 2051),
                    recived_option=self.year,
                )
        elif self.units == "btu":
            if self.year < 2024:
                return HistoricalTransportBtuDemand(self.vehicle, self.year, self.api)
            elif self.year >= 2024:
                aeo = "reference" if not self.scenario else self.scenario
                return ProjectedTransportBtuDemand(
                    self.vehicle,
                    self.year,
                    aeo,
                    self.api,
                )
            else:
                raise InputException(
                    propery="TransportationBtuDemand",
                    valid_options=range(2017, 2051),
                    recived_option=self.year,
                )
        else:
            raise InputException(
                propery="TransportationDemand",
                valid_options=("travel", "btu"),
                recived_option=self.units,
            )


class TransportationFuelUse(EiaData):
    """
    Transportation energy use by fuel type in TBTU.

    If historical year is provided, energy use for that year is
    provided. If a future year is provided, annual projections from 2023
    up to that year are provided based on the scenario given
    """

    def __init__(
        self,
        vehicle: str,
        year: int,
        api: str,
        scenario: Optional[str] = None,
    ) -> None:
        self.vehicle = vehicle
        self.year = year
        self.api = api
        self.scenario = scenario
        self.aeo = "reference" if not self.scenario else self.scenario

    def data_creator(self) -> pd.DataFrame:
        if self.year > 2016:
            return HistoricalProjectedTransportFuelUse(
                self.vehicle,
                self.year,
                self.aeo,
                self.api,
            )
        else:
            raise InputException(
                propery="TransportationFuelUse",
                valid_options=range(2017, 2051),
                recived_option=self.year,
            )


# concrete creator
class Storage(EiaData):

    def __init__(self, fuel: str, storage: str, year: int, api: str) -> None:
        self.fuel = fuel
        self.storage = storage  # (base|working|total|withdraw)
        self.year = year
        self.api = api

    def data_creator(self) -> pd.DataFrame:
        if self.fuel == "gas":
            return GasStorage(self.storage, self.year, self.api)
        else:
            raise InputException(
                propery="Storage",
                valid_options=["gas"],
                recived_option=self.fuel,
            )


# concrete creator
class Emissions(EiaData):

    def __init__(self, sector: str, year: int, api: str, fuel: str = None) -> None:
        self.sector = sector  # (power|residential|commercial|industry|transport|total)
        self.year = year  # 1970 - 2021
        self.api = api
        self.fuel = "all" if not fuel else fuel  # (coal|oil|gas|all)

    def data_creator(self):
        return StateEmissions(self.sector, self.fuel, self.year, self.api)


class Seds(EiaData):
    """
    State Energy Demand System.
    """

    def __init__(self, metric: str, sector: str, year: int, api: str) -> None:
        self.metric = metric  # (consumption)
        self.sector = sector  # (residential|commercial|industry|transport|total)
        self.year = year  # 1970 - 2022
        self.api = api

    def data_creator(self):
        if self.metric == "consumption":
            return SedsConsumption(self.sector, self.year, self.api)
        else:
            raise InputException(
                propery="SEDS",
                valid_options=["consumption"],
                recived_option=self.metric,
            )


class ElectricPowerData(EiaData):
    def __init__(self, year: int, api_key: str) -> None:
        self.year = year
        self.api_key = api_key

    def data_creator(self):
        return ElectricPowerOperationalData(self.year, self.api_key)


# product
class DataExtractor(ABC):
    """
    Extracts and formats data.
    """

    def __init__(self, year: int, api_key: str = None):
        self.api_key = api_key
        # self.year = self._set_year(year)
        self.year = year

    @abstractmethod
    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formats retrieved data from EIA.

                    series-description    value  units   state
        period
        2020-01-15  description of data   2.74  $/MCF    U.S.
        ...
        """
        pass

    @abstractmethod
    def build_url(self) -> str:
        """
        Builds API url.
        """
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
    def _request_eia_data(url: str) -> dict[str, dict | str]:
        """
        Retrieves data from EIA API.

        url in the form of "https://api.eia.gov/v2/" followed by api key and facets
        """

        # sometimes running into HTTPSConnectionPool error. adding in retries helped
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504],
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))

        response = session.get(url, timeout=30)
        if response.status_code == 200:
            return response.json()  # Assumes the response is in JSON format
        else:
            logger.error(f"EIA Request failed with status code: {response.status_code}")
            raise requests.ConnectionError(f"Status code {response.status_code}")

    @staticmethod
    def _format_period(dates: pd.Series) -> pd.Series:
        """
        Parses dates into a standard monthly format.
        """
        try:  # try to convert to YYYY-MM-DD format
            return pd.to_datetime(dates, format="%Y-%m-%d")
        except ValueError:
            try:  # try to convert to YYYY-MM format
                return pd.to_datetime(dates + "-01", format="%Y-%m-%d")
            except ValueError:
                return pd.NaT

    @staticmethod
    def _pivot_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivots data on period and state.
        """
        df = df.reset_index()
        try:
            return df.pivot(
                index="period",
                columns="state",
                values="value",
            )
        # Im not actually sure why sometimes we are hitting this :(
        # ValueError: Index contains duplicate entries, cannot reshape
        except ValueError:
            logger.info("Reshaping using pivot_table and aggfunc='mean'")
            return df.pivot_table(
                index="period",
                columns="state",
                values="value",
                aggfunc="mean",
            )

    @staticmethod
    def _assign_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        return df.astype(
            {"series-description": str, "value": float, "units": str, "state": str},
        )


# concrete product
class GasCosts(DataExtractor):

    industry_codes = {
        "power": "PEU",
        "residential": "PRS",
        "commercial": "PCS",
        "industrial": "PIN",
        "imports": "PRP",
        "exports": "PNP",
    }

    def __init__(self, industry: str, year: int, api_key: str) -> None:
        self.industry = industry
        if industry not in self.industry_codes.keys():
            raise InputException(
                propery="Gas Costs",
                valid_options=list(self.industry_codes),
                recived_option=industry,
            )
        super().__init__(year, api_key)

    def build_url(self) -> str:
        base_url = "natural-gas/pri/sum/data/"
        facets = f"frequency=monthly&data[0]=value&facets[process][]={self.industry_codes[self.industry]}&start={self.year}-01&end={self.year+1}-01&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formats natural gas cost data.
        """

        # format dates
        df["period"] = self._format_period(df.period)
        df = df.set_index("period").copy()

        # split happens twice to account for inconsistent naming
        # Sometimes "U.S. Natural Gas price"
        # Sometimes "Price of U.S. Natural Gas"
        df["state"] = df["series-description"].map(
            lambda x: x.split("Natural Gas")[0].strip(),
        )
        df["state"] = df["state"].map(lambda x: x.split("Price of")[0].strip())
        df["value"] = df["value"].fillna(np.nan)

        df["state"] = df.state.map(lambda x: "U.S." if x == "United States" else x)
        usa_average = df[df.state == "U.S."].to_dict()["value"]

        df = df.reset_index()
        df["use_average"] = ~pd.notna(
            df["value"],
        )  # not sure why this cant be built into the lambda function
        df["value"] = df.apply(
            lambda x: x["value"] if not x["use_average"] else usa_average[x["period"]],
            axis=1,
        )
        df = df.drop(columns=["use_average"]).copy()

        df = (
            df[["series-description", "value", "units", "state", "period"]]
            .sort_values(["state", "period"])
            .set_index("period")
        )

        return self._assign_dtypes(df)


# concrete product
class CoalCosts(DataExtractor):

    industry_codes = {
        "power": "PEU",
    }

    def __init__(self, industry: str, year: int, api_key: str) -> None:
        self.industry = industry
        if industry != "power":
            raise InputException(
                propery="Coal Costs",
                valid_options=list(self.industry_codes),
                recived_option=industry,
            )
        super().__init__(year, api_key)

    def build_url(self) -> str:
        base_url = "coal/shipments/receipts/data/"
        facets = f"frequency=quarterly&data[0]=price&facets[coalRankId][]=TOT&start={self.year}-Q1&end={self.year+1}-Q1&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df[
            (df.coalRankId.isin(["TOT"]))
            & ~(df.price == "w")
            & ~(df.price.isna())  # withheld value. Will be assigned usa average
        ].copy()

        # sometimes prices come in the format of xx.xx.xx, so drop everything after the second "."
        df["price"] = df.price.map(
            lambda x: (
                float(x) if len(x.split(".")) < 2 else float(".".join(x.split(".")[:2]))
            ),
        )

        # get data at a per quarter level
        df[["year", "quarter"]] = df.period.str.split("-", expand=True)
        df = (
            df[["plantStateDescription", "price", "price-units", "year", "quarter"]]
            .rename(
                columns={
                    "plantStateDescription": "state",
                    "price-units": "units",
                    "price": "value",
                },
            )
            .groupby(by=["state", "units", "year", "quarter"])
            .mean()
            .reset_index()
        )
        df = df[
            df.year.astype(int) == self.year
        ].copy()  # api is bringing in an extra quarter

        # Expand data to be at a per month level
        dfs = []
        dates = pd.date_range(
            start=f"{self.year}-01-01",
            end=f"{self.year}-12-01",
            freq="MS",
        )
        for date in dates:
            quarter = (
                math.floor((date.month - 1) / 3) + 1
            )  # months 1-3 are Q1, months 4-6 are Q2 ...
            df_month = df[df.quarter == f"Q{quarter}"].copy()
            df_month["period"] = date
            dfs.append(df_month)
        states = pd.concat(dfs).drop(columns=["year", "quarter"])

        # add a usa average datapoint for missing values
        usa_average = []
        for date in dates:
            usa_average.append(
                [
                    "U.S.",
                    "average dollars per ton",
                    states[states.period == date].value.mean(),
                    date,
                ],
            )
            usa = pd.DataFrame(usa_average, columns=states.columns)

        final = pd.concat([states, usa]).reset_index(drop=True)
        final["series-description"] = final.state.map(
            lambda x: f"{x} Coal Electric Power Price",
        )

        final = final.set_index("period")

        return self._assign_dtypes(final)


class LpgCosts(DataExtractor):
    """
    This is motor gasoline!

    Not heating fuel!
    """

    grade_codes = {
        "total": "EPM0",
        "regular": "EPMR",
        "premium": "EPMP",
        "midgrade": "EPMM",
        "diesel": "EPD2D",
    }

    # https://en.wikipedia.org/wiki/Petroleum_Administration_for_Defense_Districts
    padd_2_state = {
        "PADD 1A": ["CT", "ME", "MA", "NH", "RI", "VT"],
        "PADD 1B": ["DE", "DC", "MD", "NJ", "NY", "PA"],
        "PADD 1C": ["FL", "GA", "NC", "SC", "VA", "WV"],
        "PADD 2": [
            "IL",
            "IN",
            "IA",
            "KS",
            "KY",
            "MI",
            "MN",
            "MO",
            "NE",
            "ND",
            "SD",
            "OH",
            "OK",
            "TN",
            "WI",
        ],
        "PADD 3": ["AL", "AR", "LA", "MS", "NM", "TX"],
        "PADD 4": ["CO", "ID", "MT", "UT", "WY"],
        "PADD 5": ["AL", "AZ", "CA", "HI", "NV", "OR", "WA"],
    }

    def __init__(self, grade: str, year: int, api_key: str) -> None:
        self.grade = grade
        if not grade in self.grade_codes:
            raise InputException(
                propery="Lpg Costs",
                valid_options=list(self.grade_codes),
                recived_option=grade,
            )
        super().__init__(year, api_key)

    def build_url(self) -> str:
        base_url = "petroleum/pri/gnd/data/"
        facets = f"frequency=weekly&data[0]=value&facets[product][]={self.grade_codes[self.grade]}&facets[duoarea][]=R1X&facets[duoarea][]=R1Y&facets[duoarea][]=R1Z&facets[duoarea][]=R20&facets[duoarea][]=R30&facets[duoarea][]=R40&facets[duoarea][]=R50&start={self.year}-01-01&end={self.year}-12-31&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:

        data = df[
            ["period", "area-name", "series-description", "value", "units"]
        ].copy()

        data["state"] = data["area-name"].map(self.padd_2_state)
        data = data.explode("state")

        data["units"] = data.units.str.replace("GAL", "gal")

        final = data.set_index("period").drop(columns="area-name")

        return self._assign_dtypes(final)


class HeatingFuelCosts(DataExtractor):
    """
    Note, only returns data from October to March!
    """

    heating_fuel_codes = {"fuel_oil": "No 2 Fuel Oil", "propane": "Propane"}

    def __init__(self, heating_fuel: str, year: int, api_key: str) -> None:
        self.heating_fuel = heating_fuel
        if not heating_fuel in self.heating_fuel_codes:
            raise InputException(
                propery="Heating Fuel Costs",
                valid_options=list(self.heating_fuel_codes),
                recived_option=heating_fuel,
            )
        super().__init__(year, api_key)

    def build_url(self) -> str:
        base_url = "petroleum/pri/wfr/data/"
        facets = f"frequency=weekly&data[0]=value&facets[process][]=PRS&start={self.year}-01-01&end={self.year}-12-31&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:

        data = df[
            (
                df["product-name"].str.startswith(
                    self.heating_fuel_codes[self.heating_fuel],
                )
            )
            & ~(df.duoarea.str.startswith("R"))
        ].copy()

        data = data[["period", "duoarea", "series-description", "value", "units"]]

        data["state"] = data.duoarea.str[1:]
        data["state"] = data.state.map(lambda x: "USA" if x == "US" else x)

        data["units"] = data.units.str.replace("GAL", "gal")

        final = data[~(data.value.isna())].copy()

        final = final.set_index("period").drop(columns="duoarea")

        return self._assign_dtypes(final)


# class HistoricalMonthlySectorEnergyDemand(DataExtractor):
#     """
#     Extracts historical energy demand at a monthly and national level.

#     Note, this is end use energy consumed (does not include losses)
#     - https://www.eia.gov/totalenergy/data/flow-graphs/electricity.php
#     - https://www.eia.gov/outlooks/aeo/pdf/AEO2023_Release_Presentation.pdf (pg 17)
#     """

#     sector_codes = {
#         "residential": "TNR",
#         "commercial": "TNC",
#         "industry": "TNI",
#         "transport": "TNA",
#         "all": "TNT",  # total energy consumed by all end-use sectors
#     }

#     def __init__(self, sector: str, year: int, api: str) -> None:
#         self.sector = sector
#         if sector not in self.sector_codes.keys():
#             raise InputException(
#                 propery="Historical Energy Demand",
#                 valid_options=list(self.sector_codes),
#                 recived_option=sector,
#             )
#         super().__init__(year, api)

#     def build_url(self) -> str:
#         base_url = "total-energy/data/"
#         facets = f"frequency=monthly&data[0]=value&facets[msn][]={self.sector_codes[self.sector]}CBUS&start={self.year}-01&end={self.year}-12&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
#         return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

#     def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
#         df.index = pd.to_datetime(df.period)
#         df = df.rename(
#             columns={"seriesDescription": "series-description", "unit": "units"},
#         )
#         df["state"] = "U.S."
#         df = df[["series-description", "value", "units", "state"]].sort_index()
#         return self._assign_dtypes(df)


class HistoricalSectorEnergyDemand(DataExtractor):
    """
    Extracts historical energy demand at a yearly national level.

    Note, this is end use energy consumed (does not include losses)
    - https://www.eia.gov/totalenergy/data/flow-graphs/electricity.php
    - https://www.eia.gov/outlooks/aeo/pdf/AEO2023_Release_Presentation.pdf (pg 17)
    """

    sector_codes = {
        "residential": "TNR",
        "commercial": "TNC",
        "industry": "TNI",
        "transport": "TNA",
        "all": "TNT",  # total energy consumed by all end-use sectors
    }

    def __init__(self, sector: str, year: int, api: str) -> None:
        self.sector = sector
        if sector not in self.sector_codes.keys():
            raise InputException(
                propery="Historical Energy Demand",
                valid_options=list(self.sector_codes),
                recived_option=sector,
            )
        super().__init__(year, api)

    def build_url(self) -> str:
        base_url = "total-energy/data/"
        facets = f"frequency=annual&data[0]=value&facets[msn][]={self.sector_codes[self.sector]}CBUS&start={self.year}&end=2023&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.period)
        df.index = df.index.year
        df = df.rename(
            columns={"seriesDescription": "series-description", "unit": "units"},
        )
        df["state"] = "U.S."
        df = df[["series-description", "value", "units", "state"]].sort_index()
        assert len(df.units.unique()) == 1
        assert df.units.unique()[0] == "Trillion Btu"
        df["value"] = df.value.astype(float)
        df["value"] = df.value.div(1000).round(6)
        df["units"] = "quads"
        return self._assign_dtypes(df)


class ProjectedSectorEnergyDemand(DataExtractor):
    """
    Extracts projected energy demand at a national level from AEO 2023.
    """

    # https://www.eia.gov/outlooks/aeo/assumptions/case_descriptions.php
    scenario_codes = AEO_SCENARIOS

    # note, these are all "total energy use by end use - total gross end use consumption"
    # https://www.eia.gov/totalenergy/data/flow-graphs/electricity.php
    sector_codes = {
        "residential": "cnsm_enu_resd_NA_dele_NA_NA_qbtu",
        "commercial": "cnsm_enu_comm_NA_dele_NA_NA_qbtu",
        "industry": "cnsm_enu_idal_NA_dele_NA_NA_qbtu",
        "transport": "cnsm_enu_trn_NA_dele_NA_NA_qbtu",
    }

    def __init__(self, sector: str, year: int, scenario: str, api: str):
        super().__init__(year, api)
        self.scenario = scenario
        self.sector = sector
        if scenario not in self.scenario_codes.keys():
            raise InputException(
                propery="Projected Energy Demand Scenario",
                valid_options=list(self.scenario_codes),
                recived_option=scenario,
            )
        if sector not in self.sector_codes.keys():
            raise InputException(
                propery="Projected Energy Demand Sector",
                valid_options=list(self.sector_codes),
                recived_option=sector,
            )

    def build_url(self) -> str:
        base_url = "aeo/2023/data/"
        facets = f"frequency=annual&data[0]=value&facets[scenario][]={self.scenario_codes[self.scenario]}&facets[seriesId][]={self.sector_codes[self.sector]}&start=2024&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.period)
        df.index = df.index.year
        df = df.rename(columns={"seriesName": "series-description", "unit": "units"})
        df["state"] = "U.S."
        df = df[["series-description", "value", "units", "state"]].sort_index()
        return self._assign_dtypes(df)


class HistoricalTransportTravelDemand(DataExtractor):
    """
    Gets Transport demand in units of travel.
    """

    # https://www.eia.gov/outlooks/aeo/assumptions/case_descriptions.php
    scenario_codes = AEO_SCENARIOS

    # units will be different umong these!
    vehicle_codes = {
        "light_duty": "kei_trv_trn_NA_ldv_NA_NA_blnvehmls",
        "med_duty": "kei_trv_trn_NA_cml_NA_NA_blnvehmls",
        "heavy_duty": "kei_trv_trn_NA_fght_NA_NA_blnvehmls",
        "bus": "_trv_trn_NA_bst_NA_NA_bpm",
        "rail_passenger": "_trv_trn_NA_rlp_NA_NA_bpm",
        "boat_shipping": "kei_trv_trn_NA_dmt_NA_NA_blntnmls",
        "rail_shipping": "kei_trv_trn_NA_rail_NA_NA_blntnmls",
        "air": "kei_trv_trn_NA_air_NA_NA_blnseatmls",
    }

    def __init__(self, vehicle: str, year: int, api: str) -> None:
        self.vehicle = vehicle
        if vehicle not in self.vehicle_codes.keys():
            raise InputException(
                propery="Historical Transport Travel Demand",
                valid_options=list(self.vehicle_codes),
                recived_option=vehicle,
            )
        year = self.check_available_data_year(year)
        super().__init__(year, api)

    def check_available_data_year(self, year: int) -> int:
        if self.vehicle in ("bus", "rail_passenger"):
            if year < 2018:
                logger.error(
                    f"{self.vehicle} data not available for {year}. Returning data for year 2018.",
                )
                return 2018
        return year

    def build_url(self) -> str:
        if self.year >= 2022:
            aeo = 2023
        elif self.year >= 2015:
            aeo = self.year + 1
        else:
            raise NotImplementedError

        base_url = f"aeo/{aeo}/data/"
        scenario = f"ref{aeo}"

        facets = f"frequency=annual&data[0]=value&facets[scenario][]={scenario}&facets[seriesId][]={self.vehicle_codes[self.vehicle]}&start={self.year}&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.period)
        df.index = df.index.year
        df = df.rename(
            columns={"seriesName": "series-description", "unit": "units"},
        )
        df["state"] = "U.S."
        df["series-description"] = df["series-description"].map(
            lambda x: x.split("Transportation : Travel Indicators : ")[1],
        )
        df = df[["series-description", "value", "units", "state"]].sort_index()
        return self._assign_dtypes(df)


class ProjectedTransportTravelDemand(DataExtractor):
    """
    Gets Transport demand in units of travel.
    """

    # https://www.eia.gov/outlooks/aeo/assumptions/case_descriptions.php
    scenario_codes = AEO_SCENARIOS

    # units will be different umong these!
    vehicle_codes = {
        "light_duty": "kei_trv_trn_NA_ldv_NA_NA_blnvehmls",
        "med_duty": "kei_trv_trn_NA_cml_NA_NA_blnvehmls",
        "heavy_duty": "kei_trv_trn_NA_fght_NA_NA_blnvehmls",
        "bus": "_trv_trn_NA_bst_NA_NA_bpm",
        "rail_passenger": "_trv_trn_NA_rlp_NA_NA_bpm",
        "boat_shipping": "kei_trv_trn_NA_dmt_NA_NA_blntnmls",
        "rail_shipping": "kei_trv_trn_NA_rail_NA_NA_blntnmls",
        "air": "kei_trv_trn_NA_air_NA_NA_blnseatmls",
    }

    def __init__(self, vehicle: str, year: int, scenario: str, api: str) -> None:
        self.vehicle = vehicle
        self.scenario = scenario
        if scenario not in self.scenario_codes.keys():
            raise InputException(
                propery="Projected Transport Travel Demand Scenario",
                valid_options=list(self.scenario_codes),
                recived_option=scenario,
            )
        if vehicle not in self.vehicle_codes.keys():
            raise InputException(
                propery="Projected Transport Travel Demand",
                valid_options=list(self.vehicle_codes),
                recived_option=vehicle,
            )
        super().__init__(year, api)

    def build_url(self) -> str:
        base_url = "aeo/2023/data/"
        facets = f"frequency=annual&data[0]=value&facets[scenario][]={self.scenario_codes[self.scenario]}&facets[seriesId][]={self.vehicle_codes[self.vehicle]}&start=2024&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.period)
        df.index = df.index.year
        df = df.rename(
            columns={"seriesName": "series-description", "unit": "units"},
        )
        df["state"] = "U.S."
        df["series-description"] = df["series-description"].map(
            lambda x: x.split("Transportation : Travel Indicators : ")[1],
        )
        df = df[["series-description", "value", "units", "state"]].sort_index()
        return self._assign_dtypes(df)


class HistoricalTransportBtuDemand(DataExtractor):
    """
    Gets Transport demand in units of btu.
    """

    # units will be different umong these!
    vehicle_codes = {
        "light_duty": "cnsm_NA_trn_ldv_use_NA_NA_qbtu",
        "med_duty": "cnsm_NA_trn_cml_use_NA_NA_qbtu",
        "heavy_duty": "cnsm_NA_trn_fght_use_NA_NA_qbtu",
        "bus": "cnsm_NA_trn_bst_use_NA_NA_qbtu",
        "rail_passenger": "cnsm_NA_trn_rlp_use_NA_NA_qbtu",
        "boat_shipping": "cnsm_NA_trn_shdt_use_NA_NA_qbtu",
        "rail_shipping": "cnsm_NA_trn_rlf_use_NA_NA_qbtu",
        "air": "cnsm_NA_trn_air_use_NA_NA_qbtu",
        "boat_international": "cnsm_NA_trn_shint_use_NA_NA_qbtu",
        "boat_recreational": "cnsm_NA_trn_rbt_use_NA_NA_qbtu",
        "military": "cnsm_NA_trn_milu_use_NA_NA_qbtu",
        "lubricants": "cnsm_NA_trn_lbc_use_NA_NA_qbtu",
        "pipeline": "cnsm_NA_trn_pft_use_NA_NA_qbtu",
    }

    def __init__(self, vehicle: str, year: int, api: str) -> None:
        self.vehicle = vehicle
        if vehicle not in self.vehicle_codes.keys():
            raise InputException(
                propery="Historical BTU Transport Demand",
                valid_options=list(self.vehicle_codes),
                recived_option=vehicle,
            )
        year = self.check_available_data_year(year)
        super().__init__(year, api)

    def check_available_data_year(self, year: int) -> int:
        if self.vehicle in ("bus", "rail_passenger"):
            if year < 2018:
                logger.error(
                    f"{self.vehicle} data not available for {year}. Returning data for year 2018.",
                )
                return 2018
        return year

    def build_url(self) -> str:
        if self.year >= 2022:
            aeo = 2023
        elif self.year >= 2015:
            aeo = self.year + 1
        else:
            raise NotImplementedError

        base_url = f"aeo/{aeo}/data/"
        scenario = f"ref{aeo}"

        facets = f"frequency=annual&data[0]=value&facets[scenario][]={scenario}&facets[seriesId][]={self.vehicle_codes[self.vehicle]}&start={self.year}&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.period)
        df.index = df.index.year
        df = df.rename(
            columns={"seriesName": "series-description", "unit": "units"},
        )
        df["state"] = "U.S."
        df["series-description"] = df["series-description"].map(
            lambda x: x.split("Transportation : Energy Use by Mode : ")[1],
        )
        df = df[["series-description", "value", "units", "state"]].sort_index()
        return self._assign_dtypes(df)


class ProjectedTransportBtuDemand(DataExtractor):
    """
    Gets Transport demand in units of quads.
    """

    # https://www.eia.gov/outlooks/aeo/assumptions/case_descriptions.php
    scenario_codes = AEO_SCENARIOS

    # units will be different umong these!
    vehicle_codes = {
        "light_duty": "cnsm_NA_trn_ldv_use_NA_NA_qbtu",
        "med_duty": "cnsm_NA_trn_cml_use_NA_NA_qbtu",
        "heavy_duty": "cnsm_NA_trn_fght_use_NA_NA_qbtu",
        "bus": "cnsm_NA_trn_bst_use_NA_NA_qbtu",
        "rail_passenger": "cnsm_NA_trn_rlp_use_NA_NA_qbtu",
        "boat_shipping": "cnsm_NA_trn_shdt_use_NA_NA_qbtu",
        "rail_shipping": "cnsm_NA_trn_rlf_use_NA_NA_qbtu",
        "air": "cnsm_NA_trn_air_use_NA_NA_qbtu",
        "boat_international": "cnsm_NA_trn_shint_use_NA_NA_qbtu",
        "boat_recreational": "cnsm_NA_trn_rbt_use_NA_NA_qbtu",
        "military": "cnsm_NA_trn_milu_use_NA_NA_qbtu",
        "lubricants": "cnsm_NA_trn_lbc_use_NA_NA_qbtu",
        "pipeline": "cnsm_NA_trn_pft_use_NA_NA_qbtu",
    }

    def __init__(self, vehicle: str, year: int, scenario: str, api: str) -> None:
        self.vehicle = vehicle
        self.scenario = scenario
        if scenario not in self.scenario_codes.keys():
            raise InputException(
                propery="Projected Transport BTU Demand Scenario",
                valid_options=list(self.scenario_codes),
                recived_option=scenario,
            )
        if vehicle not in self.vehicle_codes.keys():
            raise InputException(
                propery="Projected Transport BTU Demand",
                valid_options=list(self.vehicle_codes),
                recived_option=vehicle,
            )
        super().__init__(year, api)

    def build_url(self) -> str:
        base_url = "aeo/2023/data/"
        facets = f"frequency=annual&data[0]=value&facets[scenario][]={self.scenario_codes[self.scenario]}&facets[seriesId][]={self.vehicle_codes[self.vehicle]}&start=2024&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.period)
        df.index = df.index.year
        df = df.rename(
            columns={"seriesName": "series-description", "unit": "units"},
        )
        df["state"] = "U.S."
        df["series-description"] = df["series-description"].map(
            lambda x: x.split("Transportation : Energy Use by Mode : ")[1],
        )
        df = df[["series-description", "value", "units", "state"]].sort_index()
        return self._assign_dtypes(df)


class HistoricalProjectedTransportFuelUse(DataExtractor):
    """
    Gets Transport Energy Use by fuel.
    """

    # https://www.eia.gov/outlooks/aeo/assumptions/case_descriptions.php
    scenario_codes = AEO_SCENARIOS

    # units will be different umong these!
    vehicle_codes = {
        "light_duty": [
            f"&facets[seriesId][]=cnsm_NA_trn_ldty_{x}_NA_NA_trlbtu"
            for x in ("NA", "dfo", "elc", "eth", "hdg", "mgs", "ng", "prop")
        ],
        "med_duty": [
            f"&facets[seriesId][]=cnsm_NA_trn_cltr_{x}_NA_NA_trlbtu"
            for x in ("NA", "dfo", "elc", "e85", "hdg", "mgs", "ng", "prop")
        ],
        "heavy_duty": [
            f"&facets[seriesId][]=cnsm_NA_trn_frtt_{x}_NA_NA_trlbtu"
            for x in ("NA", "dfo", "elc", "e85", "hdg", "mgs", "ng", "prop")
        ],
        "bus": [  # transit bus
            f"&facets[seriesId][]=cnsm_NA_trn_tbus_{x}_NA_NA_trlbtu"
            for x in ("NA", "dfo", "elc", "e85", "hdg", "mgs", "ng", "prop")
        ],
        "rail_passenger": [  # commuter rail
            f"&facets[seriesId][]=cnsm_NA_trn_crail_{x}_NA_NA_trlbtu"
            for x in ("NA", "dsl", "lng", "cng", "elc")
        ],
        "boat_shipping": [
            f"&facets[seriesId][]=cnsm_NA_trn_dmt_{x}_NA_NA_trlbtu"
            for x in ("NA", "cng", "dfo", "lng", "rfo")
        ],
        "rail_shipping": [
            f"&facets[seriesId][]=cnsm_NA_trn_frail_{x}_NA_NA_trlbtu"
            for x in ("NA", "cng", "dfo", "lng", "rfo")
        ],
        "air": [
            f"&facets[seriesId][]=cnsm_NA_trn_air_{x}_NA_NA_trlbtu"
            for x in ("NA", "avga", "dac", "gav", "dft", "iac", "jfl")
        ],
    }

    def __init__(self, vehicle: str, year: int, scenario: str, api: str) -> None:
        self.vehicle = vehicle
        self.scenario = scenario
        if vehicle not in self.vehicle_codes.keys():
            raise InputException(
                propery="Transport Energy Use by Fuel",
                valid_options=list(self.vehicle_codes),
                recived_option=vehicle,
            )
        if scenario not in self.scenario_codes.keys():
            raise InputException(
                propery="Transport Energy Use by Fuel",
                valid_options=list(self.scenario_codes),
                recived_option=scenario,
            )
        super().__init__(year, api)

    def build_url(self) -> str:
        if self.year >= 2022:
            aeo = 2023
        elif self.year >= 2015:
            aeo = self.year + 1
        else:
            raise NotImplementedError

        base_url = f"aeo/{aeo}/data/"
        scenario = f"ref{aeo}"

        if self.year >= 2024:
            facets = f"frequency=annual&data[0]=value&facets[scenario][]={self.scenario_codes[self.scenario]}&facets[seriesId][]={''.join(self.vehicle_codes[self.vehicle])}&start=2024&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        elif self.year >= 2022:  # switch years in api call
            facets = f"frequency=annual&data[0]=value&facets[scenario][]={self.scenario_codes[self.scenario]}&facets[seriesId][]={''.join(self.vehicle_codes[self.vehicle])}&start={self.year}&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        else:
            facets = f"frequency=annual&data[0]=value&facets[scenario][]={scenario}&facets[seriesId][]={''.join(self.vehicle_codes[self.vehicle])}&start={self.year}&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"

        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.period)
        df.index = df.index.year
        df = df.rename(
            columns={"seriesName": "series-description", "unit": "units"},
        )
        df["state"] = "U.S."
        df["series-description"] = (
            df["series-description"]
            .map(
                lambda x: x.split("Transportation Energy Use : ")[1],
            )
            .map(lambda x: x.split(" : ")[-1])
        )  # strip out vehicle type
        df = df[["series-description", "value", "units", "state"]].sort_index()
        return self._assign_dtypes(df)


class GasTrade(DataExtractor):
    """
    Gets imports/exports by point of entry.
    """

    direction_codes = {
        "imports": "IRP",
        "exports": "ENP",
    }

    def __init__(self, direction: str, year: int, api_key: str) -> None:
        self.direction = direction
        if self.direction not in list(self.direction_codes):
            raise InputException(
                propery="Natural Gas Imports and Exports",
                valid_options=list(self.direction_codes),
                recived_option=direction,
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

        df = (
            df[["series-description", "value", "units", "state", "period"]]
            .sort_values(["state", "period"])
            .set_index("period")
        )

        return self._assign_dtypes(df)

    @staticmethod
    def extract_state(description: str) -> str:
        """
        Extracts state from series descripion.

        Input will be in one of the following forms
        - "Massena, NY Natural Gas Pipeline Imports From Canada"
        - "U.S. Natural Gas Pipeline Imports From Mexico"
        """
        try:  # state level
            return description.split(",")[1].split(" ")[1]
        except IndexError:  # country level
            return description.split(" Natural Gas Pipeline")[0]


class GasStorage(DataExtractor):
    """
    Underground storage facilites for natural gas.
    """

    # https://www.eia.gov/naturalgas/storage/basics/
    storage_codes = {
        "base": "SAB",
        "working": "SAO",
        "total": "SAT",
        "withdraw": "SAW",
    }

    def __init__(self, storage: str, year: int, api_key: str) -> None:
        self.storage = storage
        if self.storage not in list(self.storage_codes):
            raise InputException(
                propery="Natural Gas Underground Storage",
                valid_options=list(self.storage_codes),
                recived_option=storage,
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
            df["series-description"].map(self.extract_state).map(self.map_state_names)
        )

        df = (
            df[["series-description", "value", "units", "state", "period"]]
            .sort_values(["state", "period"])
            .set_index("period")
        )

        return self._assign_dtypes(df)

    @staticmethod
    def extract_state(description: str) -> str:
        """
        Extracts state from series descripion.
        """
        return description.split(" Natural ")[0]

    @staticmethod
    def map_state_names(state: str) -> str:
        """
        Maps state name to code.
        """
        return "U.S." if state == "U.S. Total" else STATE_CODES[state]


class GasProduction(DataExtractor):
    """
    Dry natural gas production.
    """

    production_codes = {
        "market": "VGM",
        "gross": "FGW",  # gross withdrawls
    }

    def __init__(self, production: str, year: int, api_key: str) -> None:
        self.production = production
        if self.production not in list(self.production_codes):
            raise InputException(
                propery="Natural Gas Production",
                valid_options=list(self.production_codes),
                recived_option=production,
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
            df["series-description"].map(self.extract_state).map(self.map_state_names)
        )

        df = (
            df[["series-description", "value", "units", "state", "period"]]
            .sort_values(["state", "period"])
            .set_index("period")
        )

        return self._assign_dtypes(df)

    @staticmethod
    def extract_state(description: str) -> str:
        """
        Extracts state from series descripion.

        Input will be in one of the following forms
        - "Maryland Natural Gas Marketed Production (MMcf)"
        - "Idaho Marketed Production of Natural Gas (MMcf)"
        """
        return description.split(" Natural Gas")[0].split(" Marketed")[0]

    @staticmethod
    def map_state_names(state: str) -> str:
        """
        Maps state name to code.
        """
        return "U.S." if state == "U.S." else STATE_CODES[state]


class StateEmissions(DataExtractor):
    """
    State Level CO2 Emissions.
    """

    sector_codes = {
        "commercial": "CC",
        "power": "EC",
        "industrial": "IC",
        "residential": "RC",
        "transport": "TC",
        "total": "TT",
    }

    fuel_codes = {
        "coal": "CO",
        "gas": "NG",
        "oil": "PE",
        "all": "TO",  # coal + gas + oil = all emissions
    }

    def __init__(self, sector: str, fuel: str, year: int, api_key: str) -> None:
        self.sector = sector
        self.fuel = fuel
        if self.sector not in list(self.sector_codes):
            raise InputException(
                propery="State Level Emissions",
                valid_options=list(self.sector_codes),
                recived_option=sector,
            )
        if self.fuel not in list(self.fuel_codes):
            raise InputException(
                propery="State Level Emissions",
                valid_options=list(self.fuel_codes),
                recived_option=fuel,
            )
        super().__init__(year, api_key)
        if self.year > 2021:
            logger.warning(f"Emissions data only available until {2021}")
            self.year = 2021

    def build_url(self) -> str:
        base_url = "co2-emissions/co2-emissions-aggregates/data/"
        facets = f"frequency=annual&data[0]=value&facets[sectorId][]={self.sector_codes[self.sector]}&facets[fuelId][]={self.fuel_codes[self.fuel]}&start={self.year}&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df[~(df["state-name"] == "NA")].copy()
        df = df.rename(
            columns={
                "value-units": "units",
                "state-name": "state",
                "sector-name": "series-description",
            },
        )
        df["series-description"] = df["series-description"].str.cat(
            df["fuel-name"],
            sep=" - ",
        )

        df = (
            df[["series-description", "value", "units", "state", "period"]]
            .sort_values(["state", "period"])
            .set_index("period")
        )

        return self._assign_dtypes(df)


class SedsConsumption(DataExtractor):
    """
    State Level End-Use Consumption.
    """

    sector_codes = {
        "commercial": "TNCCB",
        "industrial": "TNICB",
        "residential": "TNRCB",
        "transport": "TNACB",
        "total": "TNTCB",
    }

    def __init__(self, sector: str, year: int, api_key: str) -> None:
        self.sector = sector
        if self.sector not in list(self.sector_codes):
            raise InputException(
                propery="State Level Consumption",
                valid_options=list(self.sector_codes),
                recived_option=sector,
            )
        super().__init__(year, api_key)
        if self.year > 2022:
            logger.warning(f"SEDS data only available until {2022}")
            self.year = 2022

    def build_url(self) -> str:
        base_url = "seds/data/"
        facets = f"frequency=annual&data[0]=value&facets[seriesId][]={self.sector_codes[self.sector]}&start={self.year - 1}&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.rename(
            columns={
                "unit": "units",
                "stateId": "state",
                "seriesDescription": "series-description",
            },
        )

        df = (
            df[["series-description", "value", "units", "state", "period"]]
            .sort_values(["state", "period"])
            .set_index("period")
        )

        return self._assign_dtypes(df)


class ElectricPowerOperationalData(DataExtractor):
    """
    Electric Power Operational Data.
    """

    sector_codes = {
        "all_sectors": [98],
    }

    def __init__(self, year: int, api_key: str) -> None:
        super().__init__(year, api_key)
        if self.year > 2021:
            logger.warning(
                f"Electric power operational data only available until {2021}",
            )
            self.year = 2021

    def build_url(self) -> str:
        base_url = "electricity/electric-power-operational-data/data/"
        facets = f"frequency=annual&data[0]=generation&facets[sectorid][]={self.sector_codes['all_sectors'][0]}&start={self.year-1}&end={self.year+1}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df.period.astype(int) == self.year].copy()
        df = df.rename(
            columns={
                "generation-units": "units",
                "stateDescription": "stateName",
                "fuelTypeDescription": "series-description",
                "location": "state",
                "generation": "value",
            },
        )
        df = df[
            ["state", "stateName", "fueltypeid", "series-description", "value", "units"]
        ].sort_values(["state"])

        return self._assign_dtypes(df)


if __name__ == "__main__":
    with open("./../config/config.api.yaml") as file:
        yaml_data = yaml.safe_load(file)
    api = yaml_data["api"]["eia"]
    # print(FuelCosts("coal", 2020, api, industry="power").get_data(pivot=True))
    print(FuelCosts("heating_oil", 2020, api).get_data(pivot=False))
    # print(Emissions("transport", 2019, api).get_data(pivot=True))
    # print(Storage("gas", "total", 2019, api).get_data(pivot=True))
    # print(EnergyDemand("residential", 2030, api).get_data(pivot=False))
    # print(
    #     TransportationFuelUse("light_duty", 2023, api).get_data(
    #         pivot=False,
    #     ),
    # )
    # print(EnergyDemand("residential", 2015, api).get_data(pivot=False))
    # print(Seds("consumption", "residential", 2022, api).get_data(pivot=False))
