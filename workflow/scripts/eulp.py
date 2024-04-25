"""
Holds data processing class for NREL End Use Load Profiles.

 See the <https://data.openei.org/submissions/4520>`_ (End Use Load Profile) data description for further information on the data

See `retrieve_eulp` rule for the data extraction
"""

import logging
from typing import Optional

import pandas as pd


class Eulp:
    _elec_group = [
        # res and com
        "out.electricity.lighting_exterior.energy_consumption.kwh",  #
        "out.electricity.lighting_interior.energy_consumption.kwh",  #
        "out.electricity.water_systems.energy_consumption.kwh",
        # res only
        "out.electricity.bath_fan.energy_consumption.kwh",
        "out.electricity.ceiling_fan.energy_consumption.kwh",  #
        "out.electricity.clothes_dryer.energy_consumption.kwh",  #
        "out.electricity.clothes_washer.energy_consumption.kwh",  #
        "out.electricity.range_oven.energy_consumption.kwh",
        "out.electricity.dishwasher.energy_consumption.kwh",  #
        "out.electricity.ext_holiday_light.energy_consumption.kwh",
        "out.electricity.extra_refrigerator.energy_consumption.kwh",
        "out.electricity.freezer.energy_consumption.kwh",  #
        "out.electricity.lighting_garage.energy_consumption.kwh",  #
        "out.electricity.mech_vent.energy_consumption.kwh",  #
        "out.electricity.permanent_spa_pump.energy_consumption.kwh",  #
        "out.electricity.hot_tub_pump.energy_consumption.kwh",
        "out.electricity.house_fan.energy_consumption.kwh",
        "out.electricity.plug_loads.energy_consumption.kwh",  #
        "out.electricity.pool_pump.energy_consumption.kwh",  #
        "out.electricity.pv.energy_consumption.kwh",  #
        "out.electricity.range_fan.energy_consumption.kwh",
        "out.electricity.recirc_pump.energy_consumption.kwh",
        "out.electricity.refrigerator.energy_consumption.kwh",  #
        "out.electricity.vehicle.energy_consumption.kwh",
        "out.electricity.well_pump.energy_consumption.kwh",  #
        # com only
        "out.electricity.fans.energy_consumption.kwh",
        "out.electricity.interior_equipment.energy_consumption.kwh",
        "out.electricity.pumps.energy_consumption.kwh",
    ]

    _heat_group = [
        # res and com
        "out.electricity.heating.energy_consumption.kwh",  #
        "out.natural_gas.heating.energy_consumption.kwh",
        "out.natural_gas.water_systems.energy_consumption.kwh",
        "out.natural_gas.hot_water.energy_consumption.kwh",  #
        # res only
        "out.electricity.heating_fans_pumps.energy_consumption.kwh",  #
        "out.electricity.heating_hp_bkup.energy_consumption.kwh",  #
        "out.electricity.heating_hp_bkup_fa.energy_consumption.kwh",  #
        "out.electricity.hot_water.energy_consumption.kwh",  #
        "out.electricity.permanent_spa_heat.energy_consumption.kwh",  #
        "out.electricity.pool_heater.energy_consumption.kwh",  #
        "out.fuel_oil.heating.energy_consumption.kwh",  #
        "out.fuel_oil.heating_hp_bkup.energy_consumption.kwh",  #
        "out.fuel_oil.hot_water.energy_consumption.kwh",  #
        "out.fuel_oil.total.energy_consumption.kwh",  #
        "out.electricity.heating_supplement.energy_consumption.kwh",
        "out.natural_gas.heating_hp_bkup.energy_consumption.kwh",  #
        "out.electricity.fans_heating.energy_consumption.kwh",
        "out.natural_gas.clothes_dryer.energy_consumption.kwh",  #
        "out.natural_gas.cooking_range.energy_consumption.kwh",
        "out.natural_gas.range_oven.energy_consumption.kwh",  #
        "out.natural_gas.fireplace.energy_consumption.kwh",  #
        "out.natural_gas.grill.energy_consumption.kwh",  #
        "out.natural_gas.hot_tub_heater.energy_consumption.kwh",
        "out.natural_gas.lighting.energy_consumption.kwh",  #
        "out.natural_gas.permanent_spa_heat.energy_consumption.kwh",  #
        "out.natural_gas.pool_heater.energy_consumption.kwh",  #
        "out.natural_gas.permanent_spa_heat.energy_consumption.kwh",  #
        "out.propane.clothes_dryer.energy_consumption.kwh",  #
        "out.propane.cooking_range.energy_consumption.kwh",
        "out.propane.heating.energy_consumption.kwh",  #
        "out.propane.heating_hp_bkup.energy_consumption.kwh",  #
        "out.propane.hot_water.energy_consumption.kwh",  #
        "out.propane.range_oven.energy_consumption.kwh",  #
        "out.propane.water_systems.energy_consumption.kwh",
        "out.wood.heating.energy_consumption.kwh",
        "out.electricity.hot_tub_heater.energy_consumption.kwh",
        # com only
        "out.other_fuel.heating.energy_consumption.kwh",
    ]

    _cool_group = [
        # res and com
        "out.electricity.cooling.energy_consumption.kwh",  #
        # res only
        "out.electricity.fans_cooling.energy_consumption.kwh",
        "out.electricity.cooling_fans_pumps.energy_consumption.kwh",  #
        # com only
        "out.district_cooling.cooling.energy_consumption.kwh",
        "out.electricity.heat_rejection.energy_consumption.kwh",
        "out.electricity.refrigeration.energy_consumption.kwh",
    ]

    def __init__(
        self,
        filepath: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> None:
        if filepath:
            df = self.read_data(filepath)
            df = self.aggregate_data(df)
            self.data = self.resample_data(df)
        elif isinstance(df, pd.DataFrame):
            self.data = df
            assert (self.data.columns == ["electricity", "heating", "cooling"]).all()
        else:
            raise TypeError(
                f"missing 1 required positional argument: 'filepath' or 'df'",
            )

    def __add__(self, other):
        if isinstance(other, Eulp):
            return Eulp(df=self.data.add(other.data))
        else:
            raise TypeError()

    def __radd__(self, other):
        return self.__add__(other)

    def __str__(self):
        return "Properties are 'data', 'electric', 'heating', 'cooling'"

    def __repr__(self):
        return f"\n{self.data.head(3)}\n\n from {self.data.index[0]} to {self.data.index[-1]}"

    @property
    def electric(self):
        return self.data["electricity"]

    @property
    def heating(self):
        return self.data["heating"]

    @property
    def cooling(self):
        return self.data["heating"]

    @classmethod
    def read_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, engine="pyarrow", index_col="timestamp")
        df.index = pd.to_datetime(df.index)
        return df

    @classmethod
    def resample_data(self, df: pd.DataFrame, resample: str = "1h") -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.resample(resample).sum()

    def aggregate_data(self, df: pd.DataFrame) -> pd.DataFrame:

        def aggregate_sector(df: pd.DataFrame, columns: list[str]) -> pd.Series:
            sector_columns = [x for x in columns if x in df.columns.to_list()]
            return df[sector_columns].sum(axis=1)

        dfs = []
        sectors = {
            "electricity": self._elec_group,
            "heating": self._heat_group,
            "cooling": self._cool_group,
        }
        for sector, sector_cols in sectors.items():
            sector_load = aggregate_sector(df, sector_cols)
            sector_load.name = sector
            dfs.append(sector_load)

        return pd.concat(dfs, axis=1).mul(1e-3)  # kwh -> MWh

    def plot(
        self,
        sectors: Optional[list[str] | str] = ["electricity", "heating", "cooling"],
    ):

        if isinstance(sectors, str):
            sectors = [sectors]

        df = self.data[sectors]

        return df.plot(xlabel="", ylabel="MWh")
