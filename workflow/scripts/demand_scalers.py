"""Demand Scalers for electricity and sector studies."""

import logging
import sqlite3
from abc import ABC, abstractmethod

import pandas as pd
from eia import EnergyDemand, TransportationDemand

logger = logging.getLogger(__name__)


class DemandScaler(ABC):
    """Allow the scaling of input data bases on different energy projections."""

    def __init__(self):
        self.projection = self.get_projections()

    @abstractmethod
    def get_projections(self) -> pd.DataFrame:
        """Get implementation specific energy projections."""
        pass

    def get_growth(self, start_year: int, end_year: int, sector: str) -> float:
        """Returns decimal change between two years."""
        min_year = self.projection.index.min()
        max_year = self.projection.index.max()

        if start_year < min_year:
            logger.warning(f"Setting base demand scaling year to {min_year}")
            start_year = min_year
        if end_year > max_year:
            logger.warning(f"Setting final demand scaling year to {max_year}")
            end_year = max_year

        start = self.projection.at[start_year, sector]
        end = self.projection.at[end_year, sector]

        return end / start

    def scale(
        self,
        df: pd.DataFrame,
        start_year: int,
        end_year: int,
        sector: str,
    ) -> pd.DataFrame:
        """Scales data."""
        growth = self.get_growth(start_year, end_year, sector)
        new = df.mul(growth)
        return self.reindex(new, end_year)

    @staticmethod
    def reindex(df: pd.DataFrame, year: int) -> pd.DataFrame:
        """
        Reindex a dataframe for a different planning horizon.

        Input dataframe will be...

        |                     | BusName_1 | BusName_2 | ... | BusName_n |
        |---------------------|-----------|-----------|-----|-----------|
        | 2019-01-01 00:00:00 |    aaa    |    ddd    |     |    ggg    |
        | 2019-01-01 01:00:00 |    bbb    |    eee    |     |    hhh    |
        | ...                 |    ...    |    ...    |     |    ...    |
        | 2019-02-28 23:00:00 |    ccc    |    fff    |     |    iii    |

        Output dataframe will be...

        |                     | BusName_1 | BusName_2 | ... | BusName_n |
        |---------------------|-----------|-----------|-----|-----------|
        | 2030-01-01 00:00:00 |    aaa    |    ddd    |     |    ggg    |
        | 2030-01-01 01:00:00 |    bbb    |    eee    |     |    hhh    |
        | ...                 |    ...    |    ...    |     |    ...    |
        | 2030-02-28 23:00:00 |    ccc    |    fff    |     |    iii    |
        """
        new = df.copy()
        new.index = new.index.map(lambda x: x.replace(year=year))
        return new


class AeoElectricityScaler(DemandScaler):
    """Scales against EIA Annual Energy Outlook electricity projections."""

    def __init__(self, pudl: str, scenario: str = "reference"):
        self.pudl = pudl
        self.scenario = scenario
        self.region = "united_states"
        super().__init__()

    def get_projections(self) -> pd.DataFrame:
        """
        Get sector yearly END-USE ELECTRICITY growth rates from AEO.

        |      | power | units |
        |----- |-------|-------|
        | 2021 |  ###  |  ###  |
        | 2022 |  ###  |  ###  |
        | 2023 |  ###  |  ###  |
        | ...  |       |       |
        | 2049 |  ###  |  ###  |
        | 2050 |  ###  |  ###  |
        """
        con = sqlite3.connect(self.pudl)
        df = pd.read_sql_query(
            f"""
        SELECT
        projection_year,
        technology_description_eiaaeo,
        gross_generation_mwh
        FROM
        core_eiaaeo__yearly_projected_generation_in_electric_sector_by_technology
        WHERE
        electricity_market_module_region_eiaaeo = "{self.region}" AND
        model_case_eiaaeo = "{self.scenario}"
        """,
            con,
        )

        df = (
            df.drop(columns=["technology_description_eiaaeo"])
            .rename(
                columns={"projection_year": "year", "gross_generation_mwh": "power"},
            )
            .groupby("year")
            .sum()
        )
        df["units"] = "mwh"
        return df


class EfsElectricityScalar(DemandScaler):
    """Scales against NREL Electrification Futures Study electricity projections."""

    def __init__(self, filepath: str):
        self.efs = filepath
        self.region = "united_states"
        super().__init__()

    def read(self) -> pd.DataFrame:
        """Read in raw EFS data."""
        df = pd.read_csv(self.efs, engine="pyarrow")
        return (
            df.drop(
                columns=[
                    "Electrification",
                    "TechnologyAdvancement",
                    "LocalHourID",
                    "Sector",
                    "Subsector",
                ],
            )
            .groupby(["Year", "State"])
            .sum()
        )

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Function interpolates between provided demand data years."""
        efs_years = df.index
        new_years = range(min(efs_years), max(efs_years) + 1)
        df = df.reindex(new_years)
        return df.interpolate()

    def get_projections(self) -> pd.DataFrame:
        """
        Get sector yearly END-USE ELECTRICITY growth rates from EFS. Linear
        interpolates missing values.

        |      | power | units |
        |----- |-------|-------|
        | 2018 |  ###  |  ###  |
        | 2019 |  ###  |  ###  |
        | 2020 |  ###  |  ###  |
        | ...  |       |       |
        | 2049 |  ###  |  ###  |
        | 2050 |  ###  |  ###  |
        """
        df = self.read().reset_index()
        if self.region == "united_states":
            df = df.drop(columns="State").groupby("Year").sum()
        else:
            raise NotImplementedError
        df = self.interpolate(df)
        df = df.rename(columns={"LoadMW": "power"})
        df["units"] = "MWh"
        return df


class AeoScaler(DemandScaler):
    """Scales according to AEO data previously extracted."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        super().__init__()

    def get_projections(self) -> pd.DataFrame:
        """Get yearly END-USE growth rates at a national level."""
        return pd.read_csv(self.filepath, index_col=0)


"""
Deprecated in favour of processing data beforehand to avoid connection errors
keep until next release for testing purposes
"""


class AeoEnergyScalerApi(DemandScaler):
    """Scales against EIA Annual Energy Outlook energy projections."""

    def __init__(self, api: str, scenario: str = "reference"):
        self.api = api
        self.scenario = scenario
        self.region = "united_states"
        super().__init__()

    def get_sector_data(self, years: list[int], sector: str) -> pd.DataFrame:
        """Function to piece togehter historical and projected values."""
        start_year = min(years)
        end_year = max(years)

        data = []

        if start_year < 2024:
            data.append(
                EnergyDemand(sector=sector, year=start_year, api=self.api).get_data(),
            )
        if end_year >= 2024:
            data.append(
                EnergyDemand(sector=sector, year=end_year, api=self.api).get_data(),
            )
        return pd.concat(data)

    def get_projections(self) -> pd.DataFrame:
        """
        Get sector yearly END-USE ENERGY growth rates from AEO at a NATIONAL
        level.

        |      | residential | commercial  | industrial  | transport  | units |
        |----- |-------------|-------------|-------------|------------|-------|
        | 2018 |     ###     |     ###     |     ###     |     ###    |  ###  |
        | 2019 |     ###     |     ###     |     ###     |     ###    |  ###  |
        | 2020 |     ###     |     ###     |     ###     |     ###    |  ###  |
        | ...  |             |             |             |            |       |
        | 2049 |     ###     |     ###     |     ###     |     ###    |  ###  |
        | 2050 |     ###     |     ###     |     ###     |     ###    |  ###  |
        """
        years = range(2017, 2051)

        # sectors = ("residential", "commercial", "industry", "transport")
        sectors = ("residential", "commercial", "industry")

        df = pd.DataFrame(
            index=years,
        )

        for sector in sectors:
            sector_data = self.get_sector_data(years, sector).sort_index()
            df[sector] = sector_data.value

        df["units"] = "quads"
        return df


class AeoVmtScalerApi(DemandScaler):
    """Scales against EIA Annual Energy Outlook vehicle mile traveled projections."""

    def __init__(self, api: str, scenario: str = "reference"):
        self.api = api
        self.scenario = scenario
        self.region = "united_states"
        super().__init__()

    def get_historical_value(self, year: int, sector: str) -> float:
        """Returns single year value at a time."""
        return TransportationDemand(vehicle=sector, year=year, api=self.api).get_data(pivot=True).values[0][0]

    def get_future_values(
        self,
        year: int,
        sector: str,
    ) -> pd.DataFrame:
        """Returns all values from 2024 onwards."""
        return TransportationDemand(
            vehicle=sector,
            year=year,
            api=self.api,
            scenario=self.scenario,
        ).get_data()

    def get_projections(self) -> pd.DataFrame:
        """
        Get sector yearly END-USE ENERGY growth rates from AEO at a NATIONAL
        level.

        |      | light_duty | med_duty  | heavy_duty  | bus  | units |
        |----- |------------|-----------|-------------|------|-------|
        | 2018 |     ###    |    ###    |     ###     | ###  |  ###  |
        | 2019 |     ###    |    ###    |     ###     | ###  |  ###  |
        | 2020 |     ###    |    ###    |     ###     | ###  |  ###  |
        | ...  |            |           |             |      |       |
        | 2049 |     ###    |    ###    |     ###     | ###  |  ###  |
        | 2050 |     ###    |    ###    |     ###     | ###  |  ###  |
        """
        years = range(2017, 2051)

        vehicles = ("light_duty", "med_duty", "heavy_duty", "bus")

        df = pd.DataFrame(
            columns=["light_duty", "med_duty", "heavy_duty", "bus"],
            index=years,
        )

        for year in sorted(years):
            if year < 2024:
                for vehicle in vehicles:
                    df.at[year, vehicle] = self.get_historical_value(
                        year,
                        vehicle,
                    )

        for vehicle in vehicles:
            aeo = self.get_future_values(max(years), vehicle)
            for year in years:
                if year < 2024:
                    continue
                df.at[year, vehicle] = aeo.at[year, "value"]

        df["units"] = "thousand VMT"
        return df
