"""
Builds costs data from EFS studies.

https://www.nrel.gov/docs/fy18osti/70485.pdf
https://data.nrel.gov/submissions/78
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
import xarray as xr
from constants import MMBTU_MWHthemal


class EfsTechnologyData:
    """
    Extracts End Use Technology data from EFS.

    Public methods for "Transportation" and "Buildings" sector include:
    - get_capex()
    - get_lifetime()
    - get_efficiency()
    - get_fixed_costs()

    Buildings sector will return both commercial and residential data
    """

    def __init__(self, file_path: str, efs_case: str = "Moderate Advancement") -> None:
        self.file_path = file_path
        self.data = self._read_efs()
        self._check_efs_case(efs_case)
        self.efs_case = efs_case

    @property
    def efs_case(self):
        return self._efs_case

    @efs_case.setter
    def efs_case(self, case):
        self._check_efs_case(case)
        self._efs_case = case

    @staticmethod
    def _check_efs_case(case: str) -> None:
        assert case in ("Slow Advancement", "Rapid Advancement", "Moderate Advancement")

    def _read_efs(self) -> pd.DataFrame:
        return pd.read_excel(self.file_path, sheet_name="EFS Data")

    def initialize(self, sector: str):  # returns EfsSectorData
        if sector == "Transportation":
            return EfsTransportationData(self.data, self.efs_case)
        elif sector == "Buildings":
            return EfsBuildingData(self.data, self.efs_case)
        else:
            raise NotImplementedError

    def get_capex(self, sector: str) -> pd.DataFrame:
        return self.initialize(sector).get_capex()

    def get_lifetime(self, sector: str) -> pd.DataFrame:
        return self.initialize(sector).get_lifetime()

    def get_efficiency(self, sector: str) -> pd.DataFrame:
        return self.initialize(sector).get_efficiency()

    def get_fixed_costs(self, sector: str) -> pd.DataFrame:
        return self.initialize(sector).get_fixed_costs()


class EfsSectorData(ABC):

    def __init__(self, data: pd.DataFrame, efs_case: str) -> None:
        self.data = data
        self.efs_case = efs_case

    @abstractmethod
    def get_capex(self):
        pass

    @abstractmethod
    def get_efficiency(self):
        pass

    @abstractmethod
    def get_fixed_costs(self):
        pass

    @abstractmethod
    def get_lifetime(self):
        pass

    @staticmethod
    def _format_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(
            columns=[
                "Sector",
                "Subsector",
                "Technology",
                "Census Division",
                "EFS Case",
            ],
        ).rename(
            columns={
                "Metric": "parameter",
                "Units": "unit",
                "Value": "value",
                "Year": "year",
            },
        )

    def _format_data_structure(
        self,
        df: pd.DataFrame,
        source: Optional[str] = "",
        description: Optional[str] = "",
    ) -> pd.DataFrame:
        data = df.copy()
        data["technology"] = data.Subsector + " " + data.Technology
        data["source"] = source
        data["further description"] = description
        return self._format_columns(data)[
            [
                "technology",
                "parameter",
                "value",
                "unit",
                "source",
                "further description",
                "year",
            ]
        ]

    def expand_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs lienar interpolations to fill in values for all years.

        Note: This is quite an enefficient function. :(
        """
        data = df.copy()
        years = range(data.year.min(), data.year.max())
        ds = df.set_index(
            [
                "technology",
                "parameter",
                "unit",
                "source",
                "further description",
                "year",
            ],
        ).to_xarray()
        return ds.interp(year=years).to_dataframe().reset_index()


class EfsTransportationData(EfsSectorData):

    # Assumptions from https://www.nrel.gov/docs/fy18osti/70485.pdf
    wh_per_gallon = 33700  # footnote 24

    # Assumptions from https://atb.nrel.gov/transportation/2022/definitions
    lifetime_miles = {
        "light_duty_cars": 178000,
        "light_duty_trucks": 198000,
        "medium_duty_trucks": 198000,
        "heavy_duty_trucks": 538000,  # class 8
        "buses": 335000,  # class 4
    }

    lifetime = 15  # years

    fixed_cost = 0.06  # $/mile for BEV

    def __init__(self, data: pd.DataFrame, efs_case: str) -> None:
        super().__init__(data, efs_case)

    def get_capex(self):
        df = self.data.copy()
        df = df[
            (df.Sector == "Transportation")
            & (df["EFS Case"] == self.efs_case)
            & (df.Metric == "Capital Cost")
        ].copy()
        source = "NREL EFS at https://data.nrel.gov/submissions/78"
        description = "Supplemented with NREL ATB"
        df["Metric"] = "investment"
        df = self._format_data_structure(df, source=source, description=description)
        df = self._correct_capex_units(df)
        return self.expand_data(df)

    def get_lifetime(self):
        df = self.data.copy()
        df = df[
            (df.Sector == "Transportation")
            & (df["EFS Case"] == self.efs_case)
            & (df.Metric == "Capital Cost")
        ].copy()
        df["Metric"] = "lifetime"
        df["Value"] = self.lifetime
        df["Units"] = "years"
        df = self._format_data_structure(df, source="NREL ATB", description="")
        return self.expand_data(df)

    def get_efficiency(self):
        """
        Only pulls main efficiency.
        """
        df = self.data.copy()
        df = df[
            (df.Sector == "Transportation")
            & (df["EFS Case"] == self.efs_case)
            & (df.Metric == "Main Efficiency")
        ].copy()
        source = "NREL EFS at https://data.nrel.gov/submissions/78"
        df["Metric"] = "efficiency"
        df = self._format_data_structure(df, source=source, description="")
        df = self._correct_efficiency_units(df)
        return self.expand_data(df)

    def get_fixed_costs(self):
        df = self.get_capex()
        df["parameter"] = "FOM"
        return self._correct_fom_units(df)

    def assign_vehicle_miles(self, name: str) -> float:
        """
        Must match class level VMT assumptions.
        """
        if name.startswith("Light Duty Cars"):
            return self.lifetime_miles["light_duty_cars"]
        elif name.startswith("Light Duty Trucks"):
            return self.lifetime_miles["light_duty_trucks"]
        elif name.startswith("Medium Duty Trucks"):
            return self.lifetime_miles["medium_duty_trucks"]
        elif name.startswith("Heavy Duty Trucks"):
            return self.lifetime_miles["heavy_duty_trucks"]
        elif name.startswith("Bus"):
            return self.lifetime_miles["buses"]
        else:
            print(f"Not assigning miles for {name}")
            return 0

    def _correct_capex_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Coverts per unit into per VMT.
        """
        corrected = df.copy()
        corrected["miles"] = corrected.technology.map(self.assign_vehicle_miles)
        corrected["value"] = corrected.value.div(corrected.miles)
        corrected["unit"] = "$/mile"
        return corrected.drop(columns=["miles"])

    def _correct_efficiency_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Coverts MPGe into per Miles/MWh.
        """
        corrected = df.copy()
        corrected["value"] = corrected.value.mul(1 / self.wh_per_gallon).mul(1e6)
        corrected["unit"] = "miles/MWh"
        return corrected

    def _correct_fom_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts $/mile costs to %/year.
        """
        corrected = df.copy()
        corrected["value"] = (1 / corrected.value).mul(self.fixed_cost).mul(100)
        corrected["unit"] = "%/year"
        return corrected


class EfsBuildingData(EfsSectorData):

    mmbtu_2_mwh = MMBTU_MWHthemal

    # Assumptions from https://atb.nrel.gov/transportation/2022/definitions

    # table a3
    lifetimes = {  # units in years
        "ashp": 15,
        "furnace": 15,
        "elec_resistance": 20,
        "hpwh": 13,  # heat pump water heater
        "ngwh": 13,  # natural gas water heater
        "ewh": 13,  # electrical water heater
    }

    # table a3
    # using commercal since it gives at a per-unit level
    fixed_cost = {  # units in $/kBTU/yr
        "ashp": 1.47,
        "furnace": 1.03,
        "elec_resistance": 0.01,
        "hpwh": 2.29,
        "ngwh": 0.55,
        "ewh": 0.88,
    }

    def __init__(self, data: pd.DataFrame, efs_case: str) -> None:
        super().__init__(data, efs_case)

    def get_capex(self):
        df = self.data.copy()
        df = df[
            (df.Sector == "Buildings")
            & (df["EFS Case"] == self.efs_case)
            & (df.Metric == "Installed Cost")
            & (df.Units.str.startswith("2016$/kBtu/hr"))
        ].copy()
        source = "NREL EFS at https://data.nrel.gov/submissions/78"
        description = ""
        df["Metric"] = "investment"
        df = self._format_data_structure(df, source=source, description=description)
        df = self._correct_capex_units(df)
        return self.expand_data(df)

    def get_lifetime(self):
        df = self.get_capex()
        df["tech_type"] = df.technology.map(self.assign_tech_types)
        df["value"] = df.tech_type.map(self.lifetimes)
        df["unit"] = "years"
        df["parameter"] = "lifetime"
        return df.drop(columns=["tech_type"])

    def get_efficiency(self):
        df = self.data.copy()
        df = df[
            (df.Sector == "Buildings")
            & (df["EFS Case"] == self.efs_case)
            & (df.Metric == "Efficiency")
        ].copy()
        source = "NREL EFS at https://data.nrel.gov/submissions/78"
        df["Metric"] = "efficiency"
        df["Units"] = "per unit"
        df = self._format_data_structure(df, source=source, description="")
        return self.expand_data(df)

    def get_fixed_costs(self):
        df = self.get_capex()
        df["parameter"] = "FOM"
        return self._correct_fom_units(df)

    def assign_tech_types(self, name: str) -> float:
        """
        Must match class level VMT assumptions.
        """
        if name.endswith("Air Source Heat Pump"):
            return "ashp"
        elif name.endswith("Heat Pump Water Heater"):
            return "hpwh"
        else:
            print(f"Not assigning tech type for {name}")
            return 0

    def _correct_capex_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Coverts $/kBtu to $/MWh.
        """
        corrected = df.copy()
        corrected["value"] = corrected.value.mul(1000).mul(self.mmbtu_2_mwh)
        corrected["unit"] = "$/Mwh"
        return corrected

    def _correct_fom_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts $/kBtu costs to %/year.
        """
        corrected = df.copy()
        corrected["tech_type"] = corrected.technology.map(self.assign_tech_types)
        corrected["fom"] = corrected.tech_type.map(self.fixed_cost)  # $ /kBTU
        corrected["fom"] = corrected.fom.mul(1000).mul(self.mmbtu_2_mwh)  # $ /MWh
        corrected["value"] = corrected.fom.div(corrected.value).mul(100)
        corrected["unit"] = "%/year"
        return corrected.drop(columns=["tech_type", "fom"])


if __name__ == "__main__":
    file_path = "../data/costs/EFS_Technology_Data.xlsx"
    efs = EfsTechnologyData(file_path)
    print(efs.get_capex("Buildings"))
    print(efs.get_fixed_costs("Buildings"))
    print(efs.get_lifetime("Buildings"))
    print(efs.get_efficiency("Buildings"))
