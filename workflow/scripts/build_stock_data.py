"""
Builds End-Use initial stock data.
"""

# to supress warning in water heat xlsx
# UserWarning: Print area cannot be set to Defined name: data!$A:$J
import warnings

warnings.simplefilter("ignore")

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from constants import STATE_2_CODE, STATES_CENSUS_DIVISION_MAPPER
from eia import TransportationFuelUse

logger = logging.getLogger(__name__)

CODE_2_STATE = {v: k for k, v in STATE_2_CODE.items()}
"""
Hardcoded build years based on building year constructed starting from 2000
https://www.eia.gov/consumption/commercial/data/2018/bc/pdf/b6.pdf
"""
CECS_BUILD_YEARS = {
    # year_built: percent of stock
    2018: 9.1,
    2009: 15.6,
    2000: 50.0,  # assumed from replacement from 2000 data
}
"""
Hardcoded appliance build years from residenital energy consumption survey
https://www.eia.gov/consumption/residential/data/2020/hc/pdf/HC%206.1.pdf
"""
RECS_BUILD_YEARS = {
    # year_built: percent of stock
    2020: 10.8,
    2018: 13.8,
    2015: 21.7,
    2010: 18.0,
    2005: 11.9,
    2000: 19.1,
}


class Recs:
    """
    Processes state level residential energy consumption survey stock data.

    Provide folder to any/all of these files from. Do not change file name!
    https://www.eia.gov/consumption/residential/data/2020/index.php?view=state#hc
    - Highlights for space heating in U.S. homes by state, 2020
    - Highlights for air conditioning in U.S. homes by state, 2020
    - Highlights for space heating fuel in U.S. homes by state, 2020
    - Highlights for water heating in U.S. homes by state, 2020
    """

    file_mapper = {
        "aircon_stock": "State Air Conditioning",
        "space_heat_stock": "State Space Heating",
        # "water_heat": "State Water Heating",
        "space_heat_fuel": "State Space Heating Fuels",
        "water_heat_fuel": "State Water Heating Fuels",
    }

    column_mapper = {
        "aircon_stock": {
            "Unnamed: 1": "total_stock",
            "Uses air-conditioning equipment": "ac_stock",
            "Unnamed: 3": "ac_percent",
            "Uses central air-conditioning unitb": "ac_central_stock",
            "Unnamed: 5": "ac_central_percent",
            "Uses individual air-conditioning unitc": "ac_individual_stock",
            "Unnamed: 7": "ac_individual_percent",
            "Unnamed: 8": "dehumidifier_stock",
            "Unnamed: 9": "dehumidifier_percent",
            "Unnamed: 10": "ceiling_fan_stock",
            "Unnamed: 11": "ceiling_fan_percent",
        },
        "space_heat_stock": {
            "Totala: 1": "total_stock",
            "Furnace": "furnace_stock",
            "Unnamed: 3": "furnace_percent",
            "Central heat pump": "central_hp_stock",
            "Unnamed: 5": "central_hp_percent",
            "Steam or hot water boiler": "boiler_stock",
            "Unnamed: 7": "boiler_percent",
            "Unnamed: 8": "secondary_heater_stock",
            "Unnamed: 9": "secondary_heater_percent",
            "Unnamed: 10": "humidifier_stock",
            "Unnamed: 11": "humidifier_percent",
        },
        "space_heat_fuel": {
            "Unnamed: 1": "total_stock",
            "Electricity": "electricity_stock",
            "Unnamed: 3": "electricity_percent",
            "Natural gas": "gas_stock",
            "Unnamed: 5": "gas_percent",
            "Propane": "propane_stock",
            "Unnamed: 7": "propane_percent",
            "Fuel oil or kerosene": "lpg_stock",
            "Unnamed: 9": "lpg_percent",
        },
        "water_heat_fuel": {
            "Unnamed: 1": "total_stock",
            "Electricity": "electricity_stock",
            "Unnamed: 3": "electricity_percent",
            "Natural gas": "gas_stock",
            "Unnamed: 5": "gas_percent",
            "Propane": "propane_stock",
            "Unnamed: 7": "propane_percent",
            "Fuel oil or kerosene": "lpg_stock",
            "Unnamed: 9": "lpg_percent",
        },
    }

    def __init__(self, root_dir: str) -> None:
        self.dir = root_dir

    @staticmethod
    def _valid_name(s: str):
        assert s in [
            "aircon_stock",
            "space_heat_stock",
            "water_heat_fuel",
            "space_heat_fuel",
        ]

    def _read(self, stock: str) -> pd.DataFrame:
        """
        Reads in the data.
        """
        f = Path(self.dir, f"{self.file_mapper[stock]}.xlsx")

        df = (
            pd.read_excel(
                f,
                sheet_name="data",
                skiprows=4,
                index_col=0,
                skipfooter=2,
            )
            .rename(columns=self.column_mapper[stock], index={"All homes": "USA"})
            .dropna()
            .astype(str)  # avoids downcasting object dtype error
            .replace("Q", np.nan)  # > 50% RSE or n < 10
            .replace("N", np.nan)  # No households
            .astype(float)
        )
        df.index = df.index.map(lambda x: x.strip())
        return df

    def _get_data(self, stock: str, fillna: bool = False) -> pd.DataFrame:
        """
        Formats data.
        """
        self._valid_name(stock)
        df = self._read(stock)
        if fillna:
            return self._fill_missing(df)
        else:
            return df

    def get_percentage(self, stock: str) -> pd.DataFrame:
        """
        Gets percentage of stock per state.
        """
        df = self._get_data(stock, fillna=True)
        return df[[x for x in df.columns if x.endswith("percent")]]

    def get_absolute(self, stock: str) -> pd.DataFrame:
        """
        Gets raw stock values per state.
        """
        df = self._get_data(stock, fillna=False)
        return df[[x for x in df.columns if x.endswith("stock")]]

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values with USA average.
        """
        columns = df.columns
        for col in columns:
            df[col] = df[col].fillna(df.at["USA", col])
        return df


class Cecs:
    """
    Processes census level commercial energy consumption survey stock data.

    Provide path to excel files of 'c7.xlsx', 'c8.xlsx', 'c9.xlsx' from:
    - https://www.eia.gov/consumption/commercial/data/2018/index.php?view=consumption#major
    """

    # Percentages from Page 7 from Table C7 at
    # https://www.eia.gov/consumption/commercial/data/2018/index.php?view=consumption#major
    usa_avg_space_heating = {
        "Electricity": 0.313,  # 1856 / 5918
        "Natural gas": 0.396,  # 2344 / 5918
        "Fuel oil": 0.038,  # 222 / 5918
        "District heat": 0.013,  # 78 / 5918
        "Propane": 0.057,  # 338 / 5918
    }
    usa_avg_water_heating = {
        "Electricity": 0.471,  # 2785 / 5918
        "Natural gas": 0.319,  # 1885 / 5918
        "Fuel oil": 0.012,  # 72 / 5918
        "District heat": 0.006,  # 38 / 5918
        "Propane": 0.025,  # 145 / 5918
    }
    usa_avg_cooling = {
        "Electricity": 0.775,  # 4584 / 5918
        "Natural gas": 0.0,  # 4 / 5918
        "District chilled": 0.01,  # 55 / 5918
    }

    census_name_map = {
        "NewEngland": "new_england",
        "Middle Atlantic": "mid_atlantic",
        "EastNorthCentral": "east_north_central",
        "WestNorthCentral": "west_north_central",
        "SouthAtlantic": "south_atlantic",
        "EastSouthCentral": "east_south_central",
        "WestSouthcentral": "west_south_central",
        "Mountain": "mountain",
        "Pacific": "pacific",
    }

    state_2_census_division = STATES_CENSUS_DIVISION_MAPPER
    code_2_state = CODE_2_STATE

    def __init__(self, root_dir: Path | str) -> None:
        self.dir = root_dir

    @staticmethod
    def _valid_name(s: str):
        assert s in ["aircon_fuel", "space_heat_fuel", "water_heat_fuel"]

    def _read(self, fuel: str) -> pd.DataFrame:
        """
        Reads in the data.
        """

        skip_rows = self._get_skip_rows(fuel)

        dfs = []

        for f_name in ("c7", "c8", "c9"):

            f = Path(self.dir, f"{f_name}.xlsx")

            df = (
                pd.read_excel(
                    f,
                    sheet_name="data",
                    skiprows=skip_rows,
                    index_col=0,
                    skipfooter=2,
                    usecols="A:D",
                )
                .dropna()
                .astype(str)  # avoids downcasting object dtype error
                .replace("Q", np.nan)  # > 50% RSE or n < 10
                .replace("N", np.nan)  # No households
                .astype(float)
            )
            df = df.rename(columns={x: x.replace("\n", "") for x in df.columns}).rename(
                columns=self.census_name_map,
            )
            dfs.append(df)

        return pd.concat(dfs, axis=1, join="outer")

    @staticmethod
    def _get_skip_rows(fuel: str) -> list[int]:
        """
        Gets rows to skip when reading in data files.
        """

        keep_rows = [3, 4]

        match fuel:
            case "space_heat_fuel":  # primary energy source
                keep_rows.extend([163, 164, 165, 166, 167])  # excludes 'other'
            case "water_heat_fuel":  # primary and secondary energy source
                keep_rows.extend([174, 175, 176, 177, 178])
            case "aircon_fuel":  # primary and secondary energy source
                keep_rows.extend([170, 171, 172])
            case _:
                raise NotImplementedError

        # 307 is number of lines in the excel file
        return [x for x in range(0, 307) if x not in keep_rows]

    def _get_data(
        self,
        fuel: str,
        as_percent: bool = True,
        fillna: bool = False,
        by_state: bool = True,
    ) -> pd.DataFrame:
        """
        Formats data.
        """
        self._valid_name(fuel)
        data = self._read(fuel)

        if as_percent:
            df = data.T
            df = df.div(df["All buildings"], axis=0)
            df = df.drop(columns=["All buildings"]).T
        else:
            df = data

        if by_state:
            df = self._expand_by_state(df).T
        else:
            df = df.T

        if fillna:
            return self._fill_missing(df, fuel)
        else:
            return df

    def _expand_by_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps census division to state.
        """

        states = pd.DataFrame(index=df.index)
        for state, census_division in self.state_2_census_division.items():
            try:
                states[self.code_2_state[state]] = df[census_division]
            except KeyError:
                logger.info(f"No Census Division for state {state}")
        return states

    def _fill_missing(self, df: pd.DataFrame, fuel: str) -> pd.DataFrame:
        """
        Fills missing values with USA average.
        """
        match fuel:
            case "space_heat_fuel":
                fill_values = self.usa_avg_space_heating
            case "water_heat_fuel":
                fill_values = self.usa_avg_water_heating
            case "aircon_fuel":
                fill_values = self.usa_avg_cooling
            case _:
                raise NotImplementedError

        for col in df.columns:
            df[col] = df[col].fillna(fill_values[col])

        return df

    """
    The two following functions are included for consistencey with the Recs class
    """

    def get_percentage(self, fuel: str, by_state: bool = True) -> pd.DataFrame:
        """
        Gets percentage of stock at a national level.
        """
        return (
            self._get_data(fuel, as_percent=True, by_state=by_state, fillna=True)
            .mul(100)
            .round(2)
        )

    def get_absolute(self, fuel: str, by_state: bool = True) -> pd.DataFrame:
        """
        Gets raw stock values at national level.
        """
        return self._get_data(fuel, as_percent=False, by_state=by_state, fillna=False)


###
# Public methods
###


def get_residential_stock(root_dir: str, load: str) -> pd.DataFrame:
    """
    Gets residential stock values as a percetange.

    Pass folder of data from the residential energy consumption survey
    """

    recs = Recs(root_dir)

    match load:
        case "space_heating":
            df = recs.get_percentage("space_heat_fuel")
        case "water_heating":
            df = recs.get_percentage("water_heat_fuel")
        case "cooling":
            df = recs.get_percentage("aircon")
        case _:
            raise NotImplementedError

    fuels = ["electricity", "gas", "lpg"]

    df = df.rename(columns={x: x.split("_percent")[0] for x in df.columns})
    df = df[[x for x in fuels if x in df.columns]]
    return df


def get_commercial_stock(root_dir: Path | str, fuel: str) -> pd.DataFrame:
    """
    Gets commercial fuel values as a percetange.
    """

    cecs = Cecs(root_dir)

    match fuel:
        case "space_heating":
            df = cecs.get_percentage("space_heat_fuel")
        case "water_heating":
            df = cecs.get_percentage("water_heat_fuel")
        case "cooling":
            df = cecs.get_percentage("aircon_fuel")
        case _:
            raise NotImplementedError

    fuels = {"Electricity": "electricity", "Natural gas": "gas", "Fuel oil": "lpg"}

    df = df[[x for x in fuels if x in df.columns]]
    return df.rename(columns=fuels)


def get_transport_stock(api: str, year: int) -> pd.DataFrame:
    """
    Gets exisiting transport stock by fuel consumption.

    Note: 'gas' in the return dataframe is natural gas! 'lpg' is motor
    gasoline and disel
    """

    def _get_data(api: str, year: int) -> pd.DataFrame:

        dfs = []

        for vehicle in ("light_duty", "med_duty", "heavy_duty", "bus"):
            df = TransportationFuelUse(vehicle, year, api).get_data(
                pivot=False,
            )  # note: cannot pivot this due to a bug
            df = df[df.index == year].reset_index()
            df["vehicle"] = vehicle

            # needed since transit bus is a subset of bus category
            if vehicle == "bus":
                df["series-description"] = df["series-description"].str.replace(
                    "Transit Bus",
                    "Total",
                )

            dfs.append(
                df.pivot(
                    index="series-description",
                    columns="vehicle",
                    values="value",
                ),
            )

        return pd.concat(dfs, axis=1).fillna(0)

    def get_percentage(api: str, year: int) -> pd.DataFrame:
        """
        Gets percentage of stock at a national level.
        """
        df = get_absolute(api, year)

        for col in df.columns:
            df[col] = df[col].div(df.at["Total", col])

        df = df.drop(index=["Total"])

        return df.mul(100).round(2)

    def get_absolute(api: str, year: int) -> pd.DataFrame:
        """
        Gets raw stock values at national level.
        """
        return _get_data(api, year).round(2)

    df = get_percentage(api, year).T
    df = (
        df.rename(
            columns={
                "Distillate Fuel Oil": "lpg",
                "Electricity": "electricity",
                "Ethanol": "ethanol",
                "Hydrogen": "hydrogen",
                "Motor Gasoline": "lpg",
                "Natural Gas": "gas",
                "Propane": "propane",
                "E85": "e85",
            },
        )
        .T.groupby(level=0)
        .sum()
    )

    return df.loc[["electricity", "lpg", "gas"]]


if __name__ == "__main__":
    # print(get_residential_stock("./../../testing", "space_heating"))
    # print(get_commercial_stock("./../../testing", "space_heating"))

    with open("./../config/config.api.yaml") as file:
        yaml_data = yaml.safe_load(file)
    api = yaml_data["api"]["eia"]

    print(get_transport_stock(api, 2024))
