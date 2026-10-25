"""
Holds data processing class for NREL End Use Load Profiles.

 See the <https://data.openei.org/submissions/4520>`_ (End Use Load Profile) data description for further information on the data

See `retrieve_eulp` rule for the data extraction and `build_eulp_loads` for the state aggregation
"""

from typing import ClassVar

import pandas as pd
from matplotlib.axes import Axes


class Eulp:
    """
    End use loads of one ResStock or ComStock building type, hourly in MWh.

    The groups are:
    - electricity -> end use electricity, other than space or water heating and cooling
    - heating -> useful space and water heating
    - space_heating -> useful space heating
    - water_heating -> useful water heating
    - cooling -> useful space cooling
    - gas_other -> natural gas, other than space and water heating

    Heating and cooling are useful energy because the service sector supplies
    them through furnaces, heat pumps and air conditioners at their efficiency.
    ResStock reports the delivered loads directly. ComStock reports only final
    energy per fuel, which is converted with state-level useful-energy factors
    (see ``build_eulp_loads.py``). Gas used for cooking, drying, fireplaces and
    pools is final energy drawn straight from the gas network; propane and
    fuel-oil uses other than heating are not modelled.
    """

    KBTU_TO_KWH: ClassVar[float] = 0.29307107

    _res_elec_group: ClassVar[list[str]] = [
        "out.electricity.ceiling_fan.energy_consumption.kwh",
        "out.electricity.clothes_dryer.energy_consumption.kwh",
        "out.electricity.clothes_washer.energy_consumption.kwh",
        "out.electricity.cooling_fans_pumps.energy_consumption.kwh",
        "out.electricity.dishwasher.energy_consumption.kwh",
        "out.electricity.freezer.energy_consumption.kwh",
        "out.electricity.heating_fans_pumps.energy_consumption.kwh",
        "out.electricity.lighting_exterior.energy_consumption.kwh",
        "out.electricity.lighting_garage.energy_consumption.kwh",
        "out.electricity.lighting_interior.energy_consumption.kwh",
        "out.electricity.mech_vent.energy_consumption.kwh",
        "out.electricity.permanent_spa_heat.energy_consumption.kwh",
        "out.electricity.permanent_spa_pump.energy_consumption.kwh",
        "out.electricity.plug_loads.energy_consumption.kwh",
        "out.electricity.pool_heater.energy_consumption.kwh",
        "out.electricity.pool_pump.energy_consumption.kwh",
        "out.electricity.pv.energy_consumption.kwh",
        "out.electricity.range_oven.energy_consumption.kwh",
        "out.electricity.refrigerator.energy_consumption.kwh",
        "out.electricity.well_pump.energy_consumption.kwh",
    ]

    _res_gas_other_group: ClassVar[list[str]] = [
        "out.natural_gas.clothes_dryer.energy_consumption.kwh",
        "out.natural_gas.fireplace.energy_consumption.kwh",
        "out.natural_gas.grill.energy_consumption.kwh",
        "out.natural_gas.lighting.energy_consumption.kwh",
        "out.natural_gas.permanent_spa_heat.energy_consumption.kwh",
        "out.natural_gas.pool_heater.energy_consumption.kwh",
        "out.natural_gas.range_oven.energy_consumption.kwh",
    ]

    # delivered loads, in kBtu
    _res_useful_groups: ClassVar[dict[str, list[str]]] = {
        "space_heating": ["out.load.heating.energy_delivered.kbtu"],
        "water_heating": ["out.load.hot_water.energy_delivered.kbtu"],
        "cooling": ["out.load.cooling.energy_delivered.kbtu"],
    }

    _com_elec_group: ClassVar[list[str]] = [
        "out.electricity.exterior_lighting.energy_consumption.kwh",
        "out.electricity.fans.energy_consumption.kwh",
        "out.electricity.heat_recovery.energy_consumption.kwh",
        "out.electricity.heat_rejection.energy_consumption.kwh",
        "out.electricity.interior_equipment.energy_consumption.kwh",
        "out.electricity.interior_lighting.energy_consumption.kwh",
        "out.electricity.pumps.energy_consumption.kwh",
        "out.electricity.refrigeration.energy_consumption.kwh",
    ]

    _com_gas_other_group: ClassVar[list[str]] = [
        "out.natural_gas.interior_equipment.energy_consumption.kwh",
    ]

    # final energy, weighted by the useful-energy factors
    _com_useful_groups: ClassVar[dict[str, list[str]]] = {
        "space_heating": [
            "out.district_heating.heating.energy_consumption.kwh",
            "out.electricity.heating.energy_consumption.kwh",
            "out.natural_gas.heating.energy_consumption.kwh",
            "out.other_fuel.heating.energy_consumption.kwh",
        ],
        "water_heating": [
            "out.district_heating.water_systems.energy_consumption.kwh",
            "out.electricity.water_systems.energy_consumption.kwh",
            "out.natural_gas.water_systems.energy_consumption.kwh",
            "out.other_fuel.water_systems.energy_consumption.kwh",
        ],
        "cooling": [
            "out.district_cooling.cooling.energy_consumption.kwh",
            "out.district_heating.cooling.energy_consumption.kwh",
            "out.electricity.cooling.energy_consumption.kwh",
            "out.other_fuel.cooling.energy_consumption.kwh",
        ],
    }

    def __init__(
        self,
        filepath: str | None = None,
        df: pd.DataFrame | None = None,
        stock: str = "res",
        factors: pd.Series | None = None,
    ) -> None:
        if filepath:
            assert stock in ("res", "com"), "stock must be 'res' or 'com'"
            if stock == "com":
                needed = {c for cols in self._com_useful_groups.values() for c in cols}
                missing = needed.difference(factors.index if factors is not None else [])
                assert not missing, f"ComStock useful-energy factors missing for {sorted(missing)}"
            self.stock = stock
            self.factors = factors
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
                    "gas_other",
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
        assert len(resampled) == 8760, "Length of resampled != 8760 :("
        return resampled.sort_index()

    def _aggregate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        def aggregate_sector(df: pd.DataFrame, columns: list[str], weights: pd.Series | None = None) -> pd.Series:
            sector_columns = [x for x in columns if x in df.columns.to_list()]
            if weights is None:
                return df[sector_columns].sum(axis=1)
            return df[sector_columns].mul(weights[sector_columns]).sum(axis=1)

        if self.stock == "res":
            # delivered loads carry small negative values (simulation noise)
            useful = {
                end_use: aggregate_sector(df, cols).clip(lower=0).mul(self.KBTU_TO_KWH)
                for end_use, cols in self._res_useful_groups.items()
            }
            electricity = aggregate_sector(df, self._res_elec_group)
            gas_other = aggregate_sector(df, self._res_gas_other_group)
        else:
            useful = {
                end_use: aggregate_sector(df, cols, self.factors) for end_use, cols in self._com_useful_groups.items()
            }
            electricity = aggregate_sector(df, self._com_elec_group)
            gas_other = aggregate_sector(df, self._com_gas_other_group)

        data = pd.DataFrame(
            {
                "electricity": electricity,
                "heating": useful["space_heating"] + useful["water_heating"],
                "space_heating": useful["space_heating"],
                "water_heating": useful["water_heating"],
                "cooling": useful["cooling"],
                "gas_other": gas_other,
            },
        )
        return data.mul(1e-3)  # kwh -> MWh

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


if __name__ == "__main__":
    print(Eulp("./../data/eulp/res/TX/mobile_home.csv", stock="res"))
