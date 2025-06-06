"""
Holds data processing class for NREL End Use Load Profiles.

 See the <https://data.openei.org/submissions/4520>`_ (End Use Load Profile) data description for further information on the data

See `retrieve_eulp` rule for the data extraction
"""

from typing import ClassVar

import pandas as pd
from matplotlib.axes import Axes


class Eulp:
    """
    End use by sector.

    The groups are:
    - electric -> end use electricity
    - cooling -> end use space cooling
    - heat -> end use space and water cooling
    - space_heat -> end use space heating
    - water_heat -> end use water heating
    """

    _elec_group: ClassVar[list[str]] = [
        # residential
        "out.electricity.ceiling_fan.energy_consumption.kwh",
        "out.electricity.clothes_dryer.energy_consumption.kwh",
        "out.electricity.clothes_washer.energy_consumption.kwh",
        "out.electricity.dishwasher.energy_consumption.kwh",
        "out.electricity.freezer.energy_consumption.kwh",
        "out.electricity.lighting_exterior.energy_consumption.kwh",
        "out.electricity.lighting_garage.energy_consumption.kwh",
        "out.electricity.lighting_interior.energy_consumption.kwh",
        "out.electricity.mech_vent.energy_consumption.kwh",
        "out.electricity.permanent_spa_heat.energy_consumption.kwh",
        "out.electricity.permanent_spa_pump.energy_consumption.kwh",
        "out.electricity.plug_loads.energy_consumption.kwh",
        "out.electricity.pool_pump.energy_consumption.kwh",
        "out.electricity.pv.energy_consumption.kwh",
        "out.electricity.range_oven.energy_consumption.kwh",
        "out.electricity.refrigerator.energy_consumption.kwh",
        "out.electricity.well_pump.energy_consumption.kwh",
        "out.natural_gas.lighting.energy_consumption.kwh",
        # commercial
        "out.electricity.exterior_lighting.energy_consumption.kwh",
        "out.electricity.fans.energy_consumption.kwh",
        "out.electricity.heat_recovery.energy_consumption.kwh",
        "out.electricity.heat_rejection.energy_consumption.kwh",
        "out.electricity.interior_equipment.energy_consumption.kwh",
        "out.electricity.interior_lighting.energy_consumption.kwh",
        "out.electricity.pumps.energy_consumption.kwh",
        "out.electricity.refrigeration.energy_consumption.kwh",
        "out.electricity.water_systems.energy_consumption.kwh",
    ]

    _heat_group: ClassVar[list[str]] = [
        # residential
        "out.electricity.heating.energy_consumption.kwh",
        "out.electricity.heating_fans_pumps.energy_consumption.kwh",
        "out.electricity.heating_hp_bkup.energy_consumption.kwh",
        "out.electricity.heating_hp_bkup_fa.energy_consumption.kwh",
        "out.electricity.hot_water.energy_consumption.kwh",
        "out.electricity.pool_heater.energy_consumption.kwh",
        "out.natural_gas.heating.energy_consumption.kwh",
        "out.natural_gas.heating_hp_bkup.energy_consumption.kwh",
        "out.natural_gas.clothes_dryer.energy_consumption.kwh",
        "out.natural_gas.fireplace.energy_consumption.kwh",
        "out.natural_gas.grill.energy_consumption.kwh",
        "out.natural_gas.hot_water.energy_consumption.kwh",
        "out.natural_gas.range_oven.energy_consumption.kwh",
        "out.propane.clothes_dryer.energy_consumption.kwh",
        "out.propane.heating.energy_consumption.kwh",
        "out.propane.heating_hp_bkup.energy_consumption.kwh",
        "out.propane.hot_water.energy_consumption.kwh",
        "out.propane.range_oven.energy_consumption.kwh",
        "out.fuel_oil.heating.energy_consumption.kwh",
        "out.fuel_oil.heating_hp_bkup.energy_consumption.kwh",
        "out.fuel_oil.hot_water.energy_consumption.kwh",
        "out.natural_gas.permanent_spa_heat.energy_consumption.kwh",
        "out.natural_gas.pool_heater.energy_consumption.kwh",
        # commercial
        "out.district_heating.heating.energy_consumption.kwh",
        "out.district_heating.water_systems.energy_consumption.kwh",
        "out.natural_gas.heating.energy_consumption.kwh",
        "out.natural_gas.interior_equipment.energy_consumption.kwh",
        "out.natural_gas.water_systems.energy_consumption.kwh",
        "out.other_fuel.heating.energy_consumption.kwh",
        "out.other_fuel.water_systems.energy_consumption.kwh",
        "out.electricity.heating.energy_consumption.kwh",
    ]

    _space_heat_group: ClassVar[list[str]] = [
        # residential
        "out.electricity.heating.energy_consumption.kwh",
        "out.electricity.heating_fans_pumps.energy_consumption.kwh",
        "out.electricity.heating_hp_bkup.energy_consumption.kwh",
        "out.electricity.heating_hp_bkup_fa.energy_consumption.kwh",
        "out.natural_gas.heating.energy_consumption.kwh",
        "out.natural_gas.heating_hp_bkup.energy_consumption.kwh",
        "out.natural_gas.clothes_dryer.energy_consumption.kwh",
        "out.natural_gas.fireplace.energy_consumption.kwh",
        "out.natural_gas.grill.energy_consumption.kwh",
        "out.natural_gas.range_oven.energy_consumption.kwh",
        "out.propane.clothes_dryer.energy_consumption.kwh",
        "out.propane.heating.energy_consumption.kwh",
        "out.propane.heating_hp_bkup.energy_consumption.kwh",
        "out.propane.range_oven.energy_consumption.kwh",
        "out.fuel_oil.heating.energy_consumption.kwh",
        "out.fuel_oil.heating_hp_bkup.energy_consumption.kwh",
        # commercial
        "out.district_heating.heating.energy_consumption.kwh",
        "out.district_heating.water_systems.energy_consumption.kwh",
        "out.natural_gas.heating.energy_consumption.kwh",
        "out.natural_gas.interior_equipment.energy_consumption.kwh",
        "out.natural_gas.water_systems.energy_consumption.kwh",
        "out.other_fuel.heating.energy_consumption.kwh",
        "out.other_fuel.water_systems.energy_consumption.kwh",
        "out.electricity.heating.energy_consumption.kwh",
    ]

    _water_heat_group: ClassVar[list[str]] = [
        # residential
        "out.electricity.hot_water.energy_consumption.kwh",
        "out.electricity.pool_heater.energy_consumption.kwh",
        "out.natural_gas.hot_water.energy_consumption.kwh",
        "out.propane.hot_water.energy_consumption.kwh",
        "out.fuel_oil.hot_water.energy_consumption.kwh",
        "out.natural_gas.permanent_spa_heat.energy_consumption.kwh",
        "out.natural_gas.pool_heater.energy_consumption.kwh",
        # commercial
        "out.district_heating.water_systems.energy_consumption.kwh",
        "out.natural_gas.water_systems.energy_consumption.kwh",
        "out.other_fuel.water_systems.energy_consumption.kwh",
    ]

    _cool_group: ClassVar[list[str]] = [
        # residential
        "out.electricity.cooling.energy_consumption.kwh",
        "out.electricity.cooling_fans_pumps.energy_consumption.kwh",
        # commercial
        "out.electricity.cooling.energy_consumption.kwh,",
        "out.district_heating.cooling.energy_consumption.kwh",
        "out.other_fuel.cooling.energy_consumption.kwh",
        "out.district_cooling.cooling.energy_consumption.kwh",
    ]

    def __init__(
        self,
        filepath: str | None = None,
        df: pd.DataFrame | None = None,
    ) -> None:
        if filepath:
            df = self._read_data(filepath)
            df = self._aggregate_data(df)
            self.data = self._resample_data(df)
        elif isinstance(df, pd.DataFrame):
            self.data = df
            assert all(
                x
                in [
                    "electricity",
                    "heating",
                    "cooling",
                    "water_heating",
                    "space_heating",
                ]
                for x in self.data.columns
            )
        else:
            raise TypeError(
                "missing 1 required positional argument: 'filepath' or 'df'",
            )

    def __add__(self, other):
        if isinstance(other, Eulp):
            return Eulp(df=self.data.add(other.data))
        else:
            raise TypeError()

    def __radd__(self, other):
        return self.__add__(other)

    def __str__(self):
        return f"Properties are {self.data.columns}"

    def __repr__(self):
        return f"\n{self.data.head(3)}\n\n from {self.data.index[0]} to {self.data.index[-1]}"

    @property
    def electric(self):  # noqa: D102
        return self.data["electricity"]

    @property
    def heating(self):  # noqa: D102
        return self.data["heating"]

    @property
    def space_heating(self):  # noqa: D102
        return self.data["space_heating"]

    @property
    def water_heating(self):  # noqa: D102
        return self.data["water_heating"]

    @property
    def cooling(self):  # noqa: D102
        return self.data["cooling"]

    @staticmethod
    def _read_data(filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, engine="pyarrow", index_col="timestamp")
        df.index = pd.to_datetime(df.index)
        return df

    @staticmethod
    def _resample_data(df: pd.DataFrame) -> pd.DataFrame:
        """Locked to resampling at 1hr."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.map(lambda x: x.replace(year=2018))
        resampled = df.resample("1h").sum()
        assert len(resampled == 8760), "Length of resampled != 8760 :("
        return resampled.sort_index()

    def _aggregate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        def aggregate_sector(df: pd.DataFrame, columns: list[str]) -> pd.Series:
            sector_columns = [x for x in columns if x in df.columns.to_list()]
            return df[sector_columns].sum(axis=1)

        dfs = []
        sectors = {
            "electricity": self._elec_group,
            "heating": self._heat_group,
            "space_heating": self._space_heat_group,
            "water_heating": self._water_heat_group,
            "cooling": self._cool_group,
        }
        for sector, sector_cols in sectors.items():
            sector_load = aggregate_sector(df, sector_cols)
            sector_load.name = sector
            dfs.append(sector_load)

        return pd.concat(dfs, axis=1).mul(1e-3)  # kwh -> MWh

    def plot(
        self,
        sectors: list[str] | str | None = [
            "electricity",
            "heating",
            "cooling",
            "space_heating",
            "water_heating",
        ],
        resample: str | None = None,
    ) -> Axes:
        """Plot load profiles."""
        if isinstance(sectors, str):
            sectors = [sectors]

        df = self.data[sectors]

        if resample:
            df = df.resample(resample).sum()

        return df.plot(xlabel="", ylabel="MW")

    def to_csv(self, path_or_buf: str, **kwargs):
        """Save load data as a csv."""
        self.data.to_csv(path_or_buf=path_or_buf, **kwargs)


class EulpTotals:
    """End use by fuel."""

    _elec_group: ClassVar[list[str]] = ["out.electricity.total.energy_consumption.kwh"]

    _ng_group: ClassVar[list[str]] = ["out.natural_gas.total.energy_consumption.kwh"]

    _oil_group: ClassVar[list[str]] = ["out.fuel_oil.total.energy_consumption.kwh"]

    _propane_group: ClassVar[list[str]] = ["out.propane.total.energy_consumption.kwh"]

    def __init__(
        self,
        filepath: str | None = None,
        df: pd.DataFrame | None = None,
    ) -> None:
        if filepath:
            df = self._read_data(filepath)
            df = self._aggregate_data(df)
            self.data = self._resample_data(df)
        elif isinstance(df, pd.DataFrame):
            self.data = df
            assert (self.data.columns == ["electricity", "gas", "oil", "propane"]).all()
        else:
            raise TypeError(
                "missing 1 required positional argument: 'filepath' or 'df'",
            )

    def __add__(self, other):
        if isinstance(other, EulpTotals):
            return EulpTotals(df=self.data.add(other.data))
        else:
            raise TypeError()

    def __radd__(self, other):
        return self.__add__(other)

    def __str__(self):
        return "Properties are 'data', 'electric', 'gas', 'oil', 'propane'"

    def __repr__(self):
        return f"\n{self.data.head(3)}\n\n from {self.data.index[0]} to {self.data.index[-1]}"

    @property
    def electric(self):  # noqa: D102
        return self.data["electricity"]

    @property
    def gas(self):  # noqa: D102
        return self.data["gas"]

    @property
    def oil(self):  # noqa: D102
        return self.data["oil"]

    @property
    def propane(self):  # noqa: D102
        return self.data["propane"]

    @staticmethod
    def _read_data(filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, engine="pyarrow", index_col="timestamp")
        df.index = pd.to_datetime(df.index)
        return df

    @staticmethod
    def _resample_data(df: pd.DataFrame) -> pd.DataFrame:
        """Locked to resampling at 1hr."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.map(lambda x: x.replace(year=2018))
        resampled = df.resample("1h").sum()
        assert len(resampled == 8760), "Length of resampled != 8760 :("
        return resampled.sort_index()

    def _aggregate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        def aggregate_sector(df: pd.DataFrame, columns: list[str]) -> pd.Series:
            sector_columns = [x for x in columns if x in df.columns.to_list()]
            return df[sector_columns].sum(axis=1)

        dfs = []
        sectors = {
            "electricity": self._elec_group,
            "gas": self._ng_group,
            "oil": self._oil_group,
            "propane": self._propane_group,
        }
        for sector, sector_cols in sectors.items():
            sector_load = aggregate_sector(df, sector_cols)
            sector_load.name = sector
            dfs.append(sector_load)

        return pd.concat(dfs, axis=1).mul(1e-3)  # kwh -> MWh

    def plot(
        self,
        sectors: list[str] | str | None = ["electricity", "gas", "oil", "propane"],
    ) -> Axes:
        """Plot load profiles."""
        if isinstance(sectors, str):
            sectors = [sectors]

        df = self.data[sectors]

        return df.plot(xlabel="", ylabel="MWh")

    def to_csv(self, path_or_buf: str, **kwargs):
        """Save load data as a csv."""
        self.data.to_csv(path_or_buf=path_or_buf, **kwargs)


if __name__ == "__main__":
    print(Eulp("./../data/eulp/res/TX/mobile_home.csv"))
