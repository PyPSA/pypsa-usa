"""
Builds End-Use initial stock data.
"""

# to supress warning in water heat xlsx
# UserWarning: Print area cannot be set to Defined name: data!$A:$J
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
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
Hardcoded build years based on building year constructed starting from 2000
https://www.eia.gov/consumption/commercial/data/2018/bc/pdf/b6.pdf
"""
CECS_STOCK = {
    # name: percent of stock
    "phu": 9.1,  # packaged heating unit
    "gas-furnace": 15.6,
    "elec-furnace": 50.0,
    "boilers": 50.0,
    "heat_pump": 50.0,
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


class RecsStock:
    """
    Processes state level residential energy consumption survey stock data.

    - https://www.eia.gov/consumption/residential/data/2020/index.php?view=state#hc
    """

    file_mapper = {
        "aircon": "State Air Conditioning",
        "space_heat": "State Space Heating",
        "water_heat": "State Water Heating",
    }

    column_mapper = {
        "aircon": {
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
        "space_heat": {
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
        "water_heat": {
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
        assert s in ["aircon", "space_heat", "water_heat"]

    def _read(self, stock: str) -> pd.DataFrame:
        """
        Reads in the data.
        """
        f = Path(self.dir, f"{self.file_mapper[stock]}.xlsx")
        return (
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

    def _get_data(self, stock: str) -> pd.DataFrame:
        """
        Formats data.
        """
        self._valid_name(stock)
        return self._read(stock)

    def get_percentage(self, stock: str) -> pd.DataFrame:
        """
        Gets percentage of stock per state.
        """
        df = self._get_data(stock)
        return df[[x for x in df.columns if x.endswith("percent")]]

    def get_absolute(self, stock: str) -> pd.DataFrame:
        """
        Gets raw stock values per state.
        """
        df = self._get_data(stock)
        return df[[x for x in df.columns if x.endswith("stock")]]


if __name__ == "__main__":
    recs = RecsStock("./../../testing")
    print(recs.get_percentage("space_heat"))
