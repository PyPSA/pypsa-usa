"""
Builds the demand data for the PyPSA network.

**Relevant Settings**

.. code:: yaml

    snapshots:
        start:
        end:
        inclusive:

    scenario:
    interconnect:
    planning_horizons:


**Inputs**

    - base_network:
    - eia: (GridEmissions data file)
    - efs: (NREL EFS Load Forecasts)

**Outputs**

    - demand: Path to the demand CSV file.
"""

# snakemake is not liking this futures import. Removing type hints in context class
# from __future__ import annotations

import calendar
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import constants as const
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import configure_logging
from eia import EnergyDemand

logger = logging.getLogger(__name__)

STATE_2_CODE = const.STATE_2_CODE
CODE_2_STATE = {value: key for key, value in STATE_2_CODE.items()}
STATE_TIMEZONE = const.STATE_2_TIMEZONE

FIPS_2_STATE = const.FIPS_2_STATE
NAICS = const.NAICS
TBTU_2_MWH = const.TBTU_2_MWH


class Context:
    """
    The Context defines the interface of interest to clients.
    """

    def __init__(self, read_strategy, write_strategy) -> None:
        """
        (read_strategy: ReadStrategy, write_strategy: WriteStrategy)
        """
        self._read_strategy = read_strategy
        self._write_strategy = write_strategy

    @property
    def read_strategy(self):  # returns ReadStrategy:
        """
        The Context maintains a reference to the Strategy objects.
        """
        return self._read_strategy

    @read_strategy.setter
    def strategy(self, strategy) -> None:  # arg is ReadStrategy
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """
        self._read_strategy = strategy

    @property
    def write_strategy(self):  # returns WriteStrategy:
        """
        The Context maintains a reference to the Strategy objects.
        """
        return self._write_strategy

    @write_strategy.setter
    def strategy(self, strategy) -> None:  # arg is WriteStrategy
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """
        self._write_strategy = strategy

    def _read(self) -> pd.DataFrame:
        """
        Delegate reading to the strategy.
        """
        return self._read_strategy.read_demand()

    def _write(self, demand: pd.DataFrame, zone: str, **kwargs) -> pd.DataFrame:
        """
        Delegate writing to the strategy.
        """
        return self._write_strategy.dissagregate_demand(demand, zone, **kwargs)

    def prepare_demand(self, **kwargs) -> pd.DataFrame:
        """
        Read in and dissagregate demand.
        """
        demand = self._read()
        return self._write(demand, self._read_strategy.zone, **kwargs)

    def prepare_multiple_demands(
        self,
        sectors: str | list[str],
        fuels: str | list[str],
        **kwargs,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Returns demand by end-use energy carrier and sector.
        """
        if isinstance(sectors, str):
            sectors = [sectors]

        if isinstance(fuels, str):
            fuels = [fuels]

        demand = self._read()

        data = {}
        for sector in sectors:
            data[sector] = {}
            for fuel in fuels:
                data[sector][fuel] = self._write(
                    demand,
                    self._read_strategy.zone,
                    sector=sector,
                    fuel=fuel,
                    **kwargs,
                )

        return data


###
# READ STRATEGIES
###


class ReadStrategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.
    """

    def __init__(self, filepath: Optional[str | list[str]] = None) -> None:
        self.filepath = filepath

    @property
    def units():
        return "MW"

    @abstractmethod
    def _read_data(self, **kwargs) -> Any:
        """
        Reads raw data into any arbitraty data structure.
        """
        pass

    def read_demand(self) -> pd.DataFrame:
        """
        Public interface to extract data.
        """

        data = self._read_data()
        df = self._format_data(data)
        self._check_index(df)
        return df

    @abstractmethod
    def _format_data(self, data: Any) -> pd.DataFrame:
        """
        Formats raw data into following datastructure.

        This datastructure MUST be indexed with the following INDEX labels:
        - snapshot (use self._format_snapshot_index() to format this)
        - sector (must be in "all", "industry", "residential", "commercial", "transport")
        - subsector (any value)
        - end use fuel (must be in "all", "electricity", "heat", "cool", "lpg")

        This datastructure MUST be indexed with the following COLUMN labels:
        - Per geography type (ie. dont mix state and ba headers)

        |                     |        |           |               | geo_name_1 | geo_name_2 | ... | geo_name_n |
        | snapshot            | sector | subsector | fuel          |            |            |     |            |
        |---------------------|--------|-----------|---------------|------------|------------|-----|------------|
        | 2019-01-01 00:00:00 | all    | all       | electricity   |    ###     |    ###     |     |    ###     |
        | 2019-01-01 01:00:00 | all    | all       | electricity   |    ###     |    ###     |     |    ###     |
        | 2019-01-01 02:00:00 | all    | all       | electricity   |    ###     |    ###     |     |    ###     |
        | ...                 | ...    | ...       | ...           |            |            |     |    ###     |
        | 2019-12-31 23:00:00 | all    | all       | electricity   |    ###     |    ###     |     |    ###     |
        """
        pass

    def _check_index(self, df: pd.DataFrame) -> None:
        """
        Enforces dimension labels.
        """
        assert all(
            x in ["snapshot", "sector", "subsector", "fuel"] for x in df.index.names
        )

        assert all(
            x in ["all", "industry", "residential", "commercial", "transport"]
            for x in df.index.get_level_values("sector").unique()
        )

        assert all(
            x in ["all", "electricity", "heat", "cool", "lpg"]
            for x in df.index.get_level_values("fuel").unique()
        )

    @staticmethod
    def _format_snapshot_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Makes index into datetime.
        """
        if df.index.nlevels > 1:
            if "snapshot" not in df.index.names:
                logger.warning("Can not format snapshot index level")
                return df
            else:
                df.index = df.index.set_levels(
                    pd.to_datetime(df.index.get_level_values("snapshot")),
                    level="snapshot",
                )
                return df
        else:
            df.index = pd.to_datetime(df.index)
            df.index.name = "snapshot"
            return df


class ReadEia(ReadStrategy):
    """
    Reads data from GridEmissions.
    """

    def __init__(self, filepath: str | None = None) -> None:
        super().__init__(filepath)
        self._zone = "ba"

    @property
    def zone(self):
        return self._zone

    def _read_data(self) -> pd.DataFrame:
        """
        Reads raw data.
        """

        if not self.filepath:
            logger.error("Must provide filepath for EIA data")
            sys.exit()

        logger.info("Building Load Data using EIA demand")
        return pd.read_csv(self.filepath, engine="pyarrow", index_col="timestamp")

    def _format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Formats raw data.
        """
        df = data.copy()
        df = self._correct_balancing_areas(df)
        df = self._format_snapshot_index(df)
        df["fuel"] = "electricity"
        df["sector"] = "all"
        df["subsector"] = "all"
        df = df.set_index([df.index, "sector", "subsector", "fuel"])
        return df.fillna(0)

    @staticmethod
    def _correct_balancing_areas(df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine EIA Demand Data to Match GIS Shapes.
        """
        df["Arizona"] = df.pop("SRP") + df.pop("AZPS")
        df["Carolina"] = (
            df.pop("CPLE")
            + df.pop("CPLW")
            + df.pop("DUK")
            + df.pop("SC")
            + df.pop("SCEG")
            + df.pop("YAD")
        )
        df["Florida"] = (
            df.pop("FPC")
            + df.pop("FPL")
            + df.pop("GVL")
            + df.pop("JEA")
            + df.pop("NSB")
            + df.pop("SEC")
            + df.pop("TAL")
            + df.pop("TEC")
            + df.pop("HST")
            + df.pop("FMPP")
        )
        return df


class ReadEfs(ReadStrategy):
    """
    Reads in electrifications future study demand.
    """

    def __init__(self, filepath: str | None = None) -> None:
        super().__init__(filepath)
        self._zone = "state"

    @property
    def zone(self):
        return self._zone

    def _read_data(self) -> pd.DataFrame:

        if not self.filepath:
            logger.error("Must provide filepath for EFS data")
            sys.exit()

        logger.info("Building Load Data using EFS demand")
        return pd.read_csv(self.filepath, engine="pyarrow").round(3)

    def _format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Formats raw data.
        """

        df = data.copy()
        df = self._build_snapshots(df)
        df = self._format_snapshot_index(df).reset_index()
        df = df.rename(columns={"Sector": "sector", "Subsector": "subsector"})
        df["sector"] = df.sector.map(
            {
                "Commercial": "commercial",
                "Residential": "residential",
                "Industrial": "industry",
                "Transportation": "transport",
            },
        )
        df["fuel"] = "electricity"
        df["LoadMW"] = df.LoadMW.astype(float)
        df["State"] = df.State.map(CODE_2_STATE)
        df = pd.pivot_table(
            df,
            values="LoadMW",
            index=["snapshot", "sector", "subsector", "fuel"],
            columns=["State"],
            aggfunc="sum",
        )
        return df

    def _build_snapshots(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds snapshots based on UTC time.
        """

        df = self._apply_timezones(df)
        df = self._build_datetime(df)
        return df.set_index("time").sort_index()

    @staticmethod
    def _apply_timezones(df: pd.DataFrame) -> pd.DataFrame:
        """
        Changes local time to relative time from UTC.
        """

        def apply_timezone_shift(timezone: str) -> int:
            """
            All shifts realitive to UTC time.
            """
            if timezone == "US/Pacific":
                return 8
            elif timezone == "US/Mountain":
                return 7
            elif timezone == "US/Central":
                return 6
            elif timezone == "US/Eastern":
                return 5
            elif timezone == "US/Alaska":
                return 9
            elif timezone == "Pacific/Honolulu":
                return 11
            else:
                raise KeyError(f"Timezone {timezone} not mapped :(")

        # mapper of {state:0} where value is offset from UTC
        utc_shift = {
            state: apply_timezone_shift(STATE_TIMEZONE[state])
            for state in STATE_TIMEZONE
        }

        df["utc_shift"] = df.State.map(utc_shift)
        df["UtcHourID"] = df.LocalHourID + df.utc_shift
        df["UtcHourID"] = df.UtcHourID.map(lambda x: x if x < 8761 else x - 8760)
        df = df.drop(columns=["utc_shift"])
        return df

    @staticmethod
    def _build_datetime(df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds snapshot from EFS data.
        """
        # minus 1 cause indexing starts at 1
        df["hoy"] = pd.to_timedelta(df.UtcHourID - 1, unit="h")
        # assign everything 2018 (non-leap year) then correct just the year
        # EFS does not do leap-years, so we must do this
        df["time"] = pd.to_datetime(2018, format="%Y") + df.hoy
        time = pd.to_datetime(
            {
                "year": df.Year,
                "month": df.time.dt.month,
                "day": df.time.dt.day,
                "hour": df.time.dt.hour,
            },
        )
        df["time"] = time
        return df.drop(columns=["Year", "UtcHourID", "hoy"])

    def get_growth_rate(self):
        """
        Public method to get yearly energy totals.

        Yearly values are linearlly interpolated between EFS planning years

        Returns:

        |      | State 1 | State 2 | ... | State n |
        |----- |---------|---------|-----|---------|
        | 2018 |  ###    |   ###   |     |   ###   |
        | 2019 |  ###    |   ###   |     |   ###   |
        | 2020 |  ###    |   ###   |     |   ###   |
        | 2021 |  ###    |   ###   |     |   ###   |
        | 2022 |  ###    |   ###   |     |   ###   |
        | ...  |         |         |     |   ###   |
        | 2049 |  ###    |   ###   |     |   ###   |
        | 2050 |  ###    |   ###   |     |   ###   |
        """

        # extract efs provided data
        efs_years = self._read_data()[["Year", "State", "LoadMW"]]
        efs_years = efs_years.groupby(["Year", "State"]).sum().reset_index()
        efs_years = efs_years.pivot(index="Year", columns="State", values="LoadMW")
        efs_years.index = pd.to_datetime(efs_years.index, format="%Y")

        # interpolate in between years
        new_index = pd.date_range(
            str(efs_years.index.min()),
            str(efs_years.index.max()),
            freq="YS",
        )
        all_years = efs_years.reindex(efs_years.index.union(new_index)).interpolate(
            method="linear",
        )
        all_years.index = all_years.index.year

        return all_years


class ReadEulp(ReadStrategy):
    """
    Reads in End Use Load Profile data.
    """

    def __init__(self, filepath: str | list[str], stock: str) -> None:
        super().__init__(filepath)
        assert stock in ("res", "com")
        self._stock = stock
        self._zone = "state"

    @property
    def zone(self):
        return self._zone

    @property
    def stock(self):
        if self._stock == "res":
            return "residential"
        elif self._stock == "com":
            return "commercial"
        else:
            raise NotImplementedError

    def _read_data(self) -> dict[str, pd.DataFrame]:
        files = [self.filepath] if isinstance(self.filepath, str) else self.filepath
        data = {}
        for f in files:
            state = self._extract_state(f)
            data[state] = pd.read_csv(f, index_col="timestamp", parse_dates=True)
        return data

    def _format_data(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = self._collapse_data(data)
        df["fuel"] = df.fuel.map(
            {"electricity": "electricity", "cooling": "cool", "heating": "heat"},
        )
        assert not df.fuel.isna().any()
        df["sector"] = self.stock
        df["subsector"] = "all"
        df = df.pivot_table(
            index=["snapshot", "sector", "subsector", "fuel"],
            values="value",
            columns="state",
            aggfunc="sum",
        )
        df = df.rename(columns=CODE_2_STATE)
        assert len(df.index.get_level_values("snapshot").unique()) == 8760
        assert not df.empty
        return df

    @staticmethod
    def _extract_state(filepath: str) -> str:
        return Path(filepath).stem

    @staticmethod
    def _collapse_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        dfs = []
        for state, state_df in data.items():
            df = (
                state_df.melt(
                    var_name="fuel",
                    value_name="value",
                    ignore_index=False,
                )
                .reset_index()
                .rename(columns={"timestamp": "snapshot"})
            )
            df["state"] = state
            dfs.append(df)
        return pd.concat(dfs)


class ReadCliu(ReadStrategy):
    """
    Reads in county level industry data.

    Data is taken from:
    - County level industrual use to get annual energy breakdown by fuel
    - EPRI load profile to get hourly load profiles
    - MECS is used to scale county level data

    Sources:
    - https://data.nrel.gov/submissions/97
    - https://www.eia.gov/consumption/manufacturing/data/2014/#r3 (Table 3.2)
    - https://loadshape.epri.com/enduse
    - https://github.com/NREL/Industry-Energy-Tool/
    """

    # these are super rough and can be imporoved
    # EPRI gives by old NERC regions, these are the mappings I used
    # ECAR -> RFC and some MRO
    # ERCOT -> ERCOT
    # MAAC -> SERC and NPCC
    # MAIN -> MRO
    # MAPP -> SPP and some MRO
    # WSCC -> WECC (but left seperate here)
    EPRI_NERC_2_STATE = {
        "ECAR": ["IL", "IN", "KY", "MI", "OH", "WI", "WV"],
        "ERCOT": ["TX"],
        "MAAC": ["MD"],
        "MAIN": ["IA", "KS", "MN", "NE", "ND", "SD"],
        "MAPP": ["AR", "DE", "MO", "MT", "NM", "OK"],
        "NYPCC_NY": ["NJ", "NY"],
        "NPCC_NE": ["CT", "ME", "MA", "NH", "PA", "RI", "VT"],
        "SERC_STV": ["AL", "GA", "LA", "MS", "NC", "SC", "TN", "VA"],
        "SERC_FL": ["FL"],
        "SPP": [],
        "WSCC_NWP": ["ID", "OR", "WA"],
        "WSCC_RA": ["AZ", "CO", "UT", "WY"],
        "WSCC_CNV": ["CA", "NV"],
    }

    # https://www.epri.com/research/products/000000003002018167
    EPRI_SEASON_2_MONTH = {
        "Peak": [5, 6, 7, 8, 9],  # may -> sept
        "OffPeak": [1, 2, 3, 4, 10, 11, 12],  # oct -> april
    }

    EPRI_ENDUSE = {
        "HVAC": "electricity",
        "Lighting": "electricity",
        "MachineDrives": "electricity",
        "ProcessHeating": "heat",
    }

    YEAR = 2014  # default year for CLIU

    def __init__(
        self,
        filepath: str | list[str],
        epri_filepath: Optional[str] = None,
        mecs_filepath: Optional[str] = None,
        fips_filepath: Optional[str] = None,
    ) -> None:
        super().__init__(filepath)
        self._epri_filepath = epri_filepath
        self._mecs_filepath = mecs_filepath
        self._fips_filepath = fips_filepath
        self._zone = "state"

    @property
    def zone(self):
        return self._zone

    def _read_data(self) -> Any:
        """
        Unzipped 'County_industry_energy_use.gz' csv file.
        """
        df = pd.read_csv(
            self.filepath,
            dtype={
                "fips_matching": int,
                "naics": int,
                "Coal": float,
                "Coke_and_breeze": float,
                "Diesel": float,
                "LPG_NGL": float,
                "MECS_NAICS": float,
                "MECS_Region": str,
                "Natural_gas": float,
                "Net_electricity": float,
                "Other": float,
                "Residual_fuel_oil": float,
                "Total": float,
                "fipscty": int,
                "fipstate": int,
                "subsector": int,
            },
        )
        df["fipstate"] = df.fipstate.map(lambda x: FIPS_2_STATE[f"{x:02d}"].title())
        return df

    def _format_data(self, county_demand: Any, scale_mecs: bool = True) -> pd.DataFrame:
        annual_demand = self._group_by_naics(county_demand, group_by_subsector=False)
        if scale_mecs:
            assert self._mecs_filepath, "Must provide MECS data"
            assert self._fips_filepath, "Must provide FIPS data"
            mecs = self._read_mecs_data()
            fips = self._read_fips_data()
            annual_demand = self._apply_mecs(annual_demand, mecs, fips)
        annual_demand = self._group_by_fuels(annual_demand) * TBTU_2_MWH
        profiles = self.get_demand_profiles(add_lpg=True)
        return self._apply_profiles(annual_demand, profiles)

    def get_demand_profiles(self, add_lpg: bool = True):
        """
        Public method to extract EPRI data.
        """
        if not self._epri_filepath:
            logger.warning("No EPRI profiles provided")
            return
        df = self._read_epri_data(profile_only=False)
        df = self._format_epri_data(df, abrv=False)
        df = self._add_epri_snapshots(df)
        if add_lpg:
            df = self._add_lpg_epri(df)
        return self._norm_epri_data(df)

    def _apply_profiles(
        self,
        annual_demand: pd.DataFrame,
        profiles: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Applies profile data to annual demand data.

        This is quite a heavy operation, as it can be up to 8760hrs,
        3000+ counties, 50+ subsectors and 4 fuels.
        """

        def profile_2_xarray(df: pd.DataFrame) -> xr.Dataset:
            """
            Converts EPRI profiles to dataset.
            """
            assert all(x in ("snapshot", "state") for x in df.index.names)
            assert all(x in ("electricity", "cool", "heat", "lpg") for x in df.columns)
            return xr.Dataset.from_dataframe(df)

        def annual_demand_2_xarray(df: pd.DataFrame) -> xr.Dataset:
            """
            Converts CLIU profiles to dataset.
            """
            assert all(
                x in ("state", "sector", "subsector", "end_use", "county")
                for x in df.index.names
            )
            assert all(x in ("electricity", "cool", "heat", "lpg") for x in df.columns)
            return xr.Dataset.from_dataframe(df)

        def remove_counties(df: pd.DataFrame) -> pd.DataFrame:
            """
            Removes counties to reduce data.
            """
            indices = df.index.names
            df.index = df.index.droplevel("county")
            return df.reset_index().groupby([x for x in indices if x != "county"]).sum()

        def check_variables(ds: xr.Dataset) -> xr.Dataset:
            """
            Confirms all data variables are present.
            """

            assert "electricity" in ds.data_vars

            for fuel in ("heat", "cool", "lpg"):
                if fuel not in ds.data_vars:
                    var = {fuel: ds["electricity"] * 0}
                    ds = ds.assign(var)

            return ds

        annual_demand = remove_counties(annual_demand)  # todo: retain county data
        annual_demand = annual_demand_2_xarray(annual_demand)
        profiles = profile_2_xarray(profiles)

        annual_demand = check_variables(annual_demand)
        profiles = check_variables(profiles)

        demand = profiles * annual_demand
        dims = [x for x in demand.sizes]

        # todo: add county arregation
        if "county" in dims:
            indices = dims.remove("state")
            columns = "county"
        else:
            indices = dims
            columns = "state"

        demand = (
            demand.to_dataframe()
            .reset_index()
            .melt(id_vars=indices, var_name="fuel")
            .pivot(index=["snapshot", "sector", "subsector", "fuel"], columns=columns)
        )
        demand.columns = demand.columns.droplevel(level=None)

        return demand

    @staticmethod
    def _group_by_naics(
        data: pd.DataFrame,
        group_by_subsector: bool = True,
    ) -> pd.DataFrame:
        """
        Groups by naics level 2 values (subsector) in TBtu.

        If group_by_subsector, all subsectors are collapsed into each
        other
        """
        df = data.rename(columns={"fips_matching": "county", "fipstate": "state"})
        df = df.drop(
            columns=["naics", "MECS_NAICS", "MECS_Region", "fipscty"],
        )

        if group_by_subsector:
            df["subsector"] = df.subsector.map(NAICS)
        else:
            df["subsector"] = "all"

        df["sector"] = "industry"

        return df.groupby(["county", "state", "sector", "subsector"]).sum()

    @staticmethod
    def _group_by_fuels(data: pd.DataFrame) -> pd.DataFrame:
        """
        Groups in electricity, heat, cool, and lpg.
        """
        df = data.copy()
        df["electricity"] = df["Net_electricity"] + df["Other"].div(3)
        df["heat"] = (
            df["Coal"] + df["Coke_and_breeze"] + df["Natural_gas"] + df["Other"].div(3)
        )
        df["cool"] = 0
        df["lpg"] = df["LPG_NGL"] + df["Residual_fuel_oil"] + df["Other"].div(3)
        return df[["electricity", "heat", "cool", "lpg"]]

    def _read_epri_data(self, profile_only=True) -> pd.DataFrame:
        """
        Reads EPRI shapes.

        If profile only, daily normalized data is provided, without
        magnitudes
        """
        dtypes = {f"HR{x}": float for x in range(1, 25)}
        dtypes.update(
            {
                "Region": str,
                "Season": str,
                "Day Type": str,
                "Sector": str,
                "End Use": str,
                "Scale Multiplier": float,
            },
        )
        df = pd.read_csv(self._epri_filepath, dtype=dtypes)

        if not profile_only:
            cols_2_scale = [f"HR{x}" for x in range(1, 25)]
            df[cols_2_scale] = df[cols_2_scale].mul(df["Scale Multiplier"], axis=0)

        return df.drop(columns="Scale Multiplier")

    def _format_epri_data(self, data: pd.DataFrame, abrv: bool = True) -> pd.DataFrame:
        """
        Formats EPRI into following dataframe.

        abrv: bool = True
            Keep abreviated state name, else replace with full name

        |       |       |      | electricity | heat    |
        | state | month | hour |             |         |
        |-------|-------|------|-------------|---------|
        | AL    |   1   |   0  | 0.70        |  0.80   |
        | AL    |   1   |   1  | 0.75        |  0.82   |
        | ...   |       |      |             |         |
        | WY    |   12  |  23  | 0.80        |  0.85   |
        """
        df = data.copy()
        df["end_use"] = df["End Use"].map(self.EPRI_ENDUSE)
        df["state"] = df.Region.map(self.EPRI_NERC_2_STATE)
        df["month"] = df.Season.map(self.EPRI_SEASON_2_MONTH)
        df = (
            df.explode("state")
            .explode("month")
            .drop(
                columns=[
                    "Day Type",
                    "Sector",
                    "End Use",
                    # "Scale Multiplier",
                    "Region",
                    "Season",
                ],
            )
            .groupby(["state", "month", "end_use"])
            .mean()
        )
        df = (
            df.rename(
                columns={x: int(x.replace("HR", "")) - 1 for x in df.columns},
            )  # start hour is zero
            .reset_index()
            .melt(id_vars=["state", "month", "end_use"], var_name="hour")
            .pivot(index=["state", "month", "hour"], columns=["end_use"])
        )
        df.columns = df.columns.droplevel(0)  # drops name "value"
        df.columns.name = ""
        if abrv:
            return df
        else:
            return df.rename(index=CODE_2_STATE, level="state")

    def _add_epri_snapshots(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Sets up epri data with snapshots.

        |       |                     | electricity | heat    |
        | state | snapshot            |             |         |
        |-------|---------------------|-------------|---------|
        | AL    | 2014-01-01 00:00:00 | 0.70        |  0.80   |
        | AL    | 2014-01-01 01:00:00 | 0.75        |  0.82   |
        | ...   |                     |             |         |
        | WY    | 2014-12-31 23:00:00 | 0.80        |  0.85   |
        """

        def get_days_in_month(year: int, month: int) -> list[int]:
            return [x for x in range(1, calendar.monthrange(year, month)[1] + 1)]

        df = data.copy().reset_index()
        df["day"] = 1
        df["year"] = self.YEAR
        df["day"] = df.month.map(lambda x: get_days_in_month(self.YEAR, x))
        df = df.explode("day")
        df["snapshot"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        return df.set_index(["state", "snapshot"]).drop(
            columns=["year", "month", "day", "hour"],
        )

    @staticmethod
    def _norm_epri_data(data: pd.DataFrame) -> pd.DataFrame:
        normed = []
        for state in data.index.get_level_values("state").unique():
            df = data.xs(state, level="state", drop_level=False)
            normed.append(df.apply(lambda x: x / x.sum()))
        return pd.concat(normed)

    @staticmethod
    def _add_lpg_epri(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds an LPG column of data as an average.
        """
        df["lpg"] = df.mean(axis=1)
        return df

    def _read_mecs_data(self) -> pd.DataFrame:
        """
        Reads MECS data; units of TBTU.

        Adapted from:
        https://github.com/NREL/Industry-Energy-Tool/blob/master/data_foundation/data_comparison/compare_mecs.py
        """

        cols_renamed = {
            "Code(a)": "NAICS",
            "Electricity(b)": "Net_electricity",
            "Fuel Oil": "Residual_fuel_oil",
            "Fuel Oil(c)": "Diesel",
            "Gas(d)": "Natural_gas",
            "natural gasoline)(e)": "LPG_NGL",
            "and Breeze": "Coke_and_breeze",
            "Other(f)": "Other",
        }

        mecs = (
            pd.read_excel(self._mecs_filepath, sheet_name="table 3.2", header=10)
            .rename(columns=cols_renamed)
            .dropna(axis=0, how="all")
        )
        # >7 filters out all naics codes
        mecs["Region"] = mecs[mecs.Total.apply(lambda x: len(str(x)) > 7)].Total
        mecs["Region"] = mecs.Region.ffill()
        mecs = mecs.dropna(axis=0).drop("Subsector and Industry", axis=1)
        mecs["NAICS"] = mecs.NAICS.astype(int)
        mecs = (
            mecs.set_index(["Region", "NAICS"], drop=True)
            .replace({"*": "0", "Q": "0", "W": "0"})
            .astype(float)
        )
        assert not (mecs == np.NaN).any().any()
        return mecs

    def _apply_mecs(
        self,
        cliu: pd.DataFrame,
        mecs: pd.DataFrame,
        fips: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Scales CLIU data by MECS data.
        """

        def assign_mecs_regions(cliu: pd.DataFrame, fips: pd.DataFrame) -> pd.DataFrame:
            """
            Adds a mecs column to the cliu df with associated fips region.
            """
            df = cliu.copy()
            df["Region"] = df.index.get_level_values("state").map(
                fips.set_index("state")["mecs"].to_dict(),
            )
            return df

        def get_cliu_totals(cliu: pd.DataFrame, fips: pd.DataFrame) -> pd.DataFrame:
            """
            Gets regional energy totals for the year per fuel from CLIU.

            |           | Coal | Coke_and_breeze | Diesel | LPG_NGL | Natural_gas | Net_electricity | Other | Residual_fuel_oil | Total |
            |-----------|------|-----------------|--------|---------|-------------|-----------------|-------|-------------------|-------|
            | Midwest   | 462  | 23              | 377    | 97      | 2249        | 1188            | 806   | 15                | 5220  |
            | Northeast | 72   | 5               | 162    | 20      | 609         | 308             | 325   | 32                | 1536  |
            | South     | 406  | 7               | 511    | 55      | 4376        | 1502            | 2823  | 66                | 9749  |
            | West      | 167  | 1               | 321    | 26      | 1567        | 582             | 619   | 33                | 3320  |
            """
            df = cliu.copy()
            df = assign_mecs_regions(df, fips)
            return df.reset_index(drop=True).groupby("Region").sum()

        def get_mecs_totals(mecs: pd.DataFrame) -> pd.DataFrame:
            """
            Gets regional energy totals for the year per fuel from MECS.
            """
            df = mecs.copy()
            df = df.reset_index().drop(columns=["NAICS"])
            df["Region"] = df.Region.map(lambda x: x.split(" ")[0])
            df = df[df.Region != "Total"].copy()
            return df.groupby("Region").sum()

        def get_scaling_factors(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
            """
            Gets factor to multiply df1 by to become df2.
            """
            assert all([x in df2.columns for x in df1.columns])
            assert all([x in df2.index for x in df1.index.unique()])
            return df2.div(df1)

        def scale_industrial_demand(
            cliu: pd.DataFrame,
            factor: pd.DataFrame,
        ) -> pd.DataFrame:
            fuels = cliu.columns
            cliu = assign_mecs_regions(cliu, fips)
            for fuel in fuels:
                cliu["scale"] = cliu.Region.map(factor[fuel])
                cliu[fuel] *= cliu.scale
            return cliu.drop(columns=["Region", "scale"])

        cliu_totals = get_cliu_totals(cliu, fips)
        mecs_total = get_mecs_totals(mecs)
        scaling_factors = get_scaling_factors(cliu_totals, mecs_total)
        return scale_industrial_demand(cliu, scaling_factors)

    def _read_fips_data(self) -> pd.DataFrame:
        """
        Reads FIPS data.
        """

        return (
            pd.read_csv(self._fips_filepath)
            .drop(columns=["FIPS_County", "FIPS State"])
            .rename(
                columns={
                    "State": "state",
                    "County_Name": "county",
                    "COUNTY_FIPS": "county_code",
                    "MECS_Region": "mecs",
                },
            )
        )


###
# WRITE STRATEGIES
###


class WriteStrategy(ABC):
    """
    Disaggregates demand based on a specified method.
    """

    def __init__(self, n: pypsa.Network) -> None:
        self.n = n

    @abstractmethod
    def _get_load_allocation_factor(
        self,
        df: Optional[pd.Series] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Load allocation set on population density.

        df: pd.Series
            Load zone mapping from self._get_load_dissagregation_zones(...)

        returns pd.Series
            Format is a bus index, with the laf for the value
        """
        pass

    def dissagregate_demand(
        self,
        df: pd.DataFrame,
        zone: str,
        sector: str | list[str] | None = None,
        subsector: str | list[str] | None = None,
        fuel: str | list[str] | None = None,
        sns: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """
        Public load dissagregation method.

        df: pd.DataFrame
            Demand dataframe
        zone: str
            Zones of demand ('ba', 'state', 'reeds')
        sector: Optional[str | List[str]] = None,
            Sectors to group
        subsector: Optional[str | List[str]] = None,
            Subsectors to group
        fuel: Optional[str | List[str]] = None,
            End use fules to group
        sns: Optional[pd.DatetimeIndex] = None
            Filter data over this period. If not provided, use network snapshots

        Data is returned in the format of:

        |                     | BusName_1 | BusName_2 | ... | BusName_n |
        |---------------------|-----------|-----------|-----|-----------|
        | 2019-01-01 00:00:00 |    ###    |    ###    |     |    ###    |
        | 2019-01-01 01:00:00 |    ###    |    ###    |     |    ###    |
        | 2019-01-01 02:00:00 |    ###    |    ###    |     |    ###    |
        | ...                 |           |           |     |    ###    |
        | 2019-12-31 23:00:00 |    ###    |    ###    |     |    ###    |
        """

        # 'state' is states based on power regions
        # 'full_state' is actual geographic boundaries
        assert zone in ("ba", "state", "reeds")
        self._check_datastructure(df)

        # get zone area demand for specific sector and fuel
        demand = self._filter_demand(df, sector, subsector, fuel, sns)
        demand = self._group_demand(demand)
        if demand.empty:
            demand = self._make_empty_demand(columns=df.columns)

        # assign buses to dissagregation zone
        dissagregation_zones = self._get_load_dissagregation_zones(zone)

        # get implementation specific dissgregation factors
        laf = self._get_load_allocation_factor(df=dissagregation_zones, zone=zone)

        # disaggregate load to buses
        zone_data = dissagregation_zones.to_frame(name="zone").join(
            laf.to_frame(name="laf"),
        )
        return self._disaggregate_demand_to_buses(demand, zone_data)

    def _get_load_dissagregation_zones(self, zone: str) -> pd.Series:
        """
        Map each bus to the load dissagregation zone (states, ba, ...)
        """
        if zone == "ba":
            return self._get_balanceing_area_zones()
        elif zone == "state":
            return self._get_state_zones()
        elif zone == "reeds":
            return self._get_reeds_zones()
        else:
            raise NotImplementedError

    @staticmethod
    def _check_datastructure(df: pd.DataFrame) -> None:
        """
        Confirms formatting of input datastructure.
        """
        assert all(
            x in ["snapshot", "sector", "subsector", "fuel"] for x in df.index.names
        )
        assert not df.empty

    def _filter_on_snapshots(
        self,
        df: pd.DataFrame,
        sns: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Filters demand on network snapshots.
        """
        filtered = df[df.index.get_level_values("snapshot").isin(sns)].copy()
        filtered = filtered[~filtered.index.duplicated(keep="last")]  # issue-272
        assert len(filtered.index.get_level_values("snapshot").unique()) == len(
            sns.unique(),
        ), "Filtered snapshots do not match expected snapshots"
        return filtered

    @staticmethod
    def _filter_on_use(df: pd.DataFrame, level: str, values: list[str]):
        return df[df.index.get_level_values(level).isin(values)].copy()

    @staticmethod
    def _group_demand(df: pd.DataFrame, agg_strategy: str = "sum") -> pd.DataFrame:
        grouped = df.droplevel(level=["sector", "subsector", "fuel"])
        grouped = grouped.groupby(level="snapshot")
        if agg_strategy == "sum":
            return grouped.sum()
        elif agg_strategy == "mean":
            return grouped.mean()
        else:
            raise NotImplementedError

    def _filter_demand(
        self,
        df: pd.DataFrame,
        sectors: str | list[str] | None = None,
        subsectors: str | list[str] | None = None,
        fuels: str | list[str] | None = None,
        sns: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """
        Filters on snapshots, sector, and fuel.
        """

        n = self.n

        if isinstance(sns, pd.DatetimeIndex):
            assert len(sns) == len(n.snapshots)
            filtered = self._filter_on_snapshots(df, sns)
            df = filtered.reset_index()
            df = df.groupby(["snapshot", "sector", "subsector", "fuel"]).sum()
            assert filtered.shape == df.shape  # no data should have changed
        else:  # profile and planning year are the same
            snapshots = n.snapshots
            df = self._filter_on_snapshots(df, snapshots)

        if sectors:
            if isinstance(sectors, str):
                sectors = [sectors]
            df = self._filter_on_use(df, "sector", sectors)
        if subsectors:
            if isinstance(subsectors, str):
                subsectors = [subsectors]
            df = self._filter_on_use(df, "subsector", subsectors)
        if fuels:
            if isinstance(fuels, str):
                fuels = [fuels]
            df = self._filter_on_use(df, "fuel", fuels)

        return df

    def _disaggregate_demand_to_buses(
        self,
        demand: pd.DataFrame,
        laf: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Zone power demand is disaggregated to buses proportional to laf.
        """

        all_load = []

        for load_zone in laf.zone.unique():
            load = laf[laf.zone == load_zone]
            load_per_bus = pd.DataFrame(
                data=([demand[load_zone]] * len(load.index)),
                index=load.index,
            )
            dissag_load = load_per_bus.mul(laf.laf, axis=0).dropna()
            assert dissag_load.shape == load_per_bus.shape  # ensure no data is lost
            dissag_load = dissag_load.T  # set snapshot as index
            dissag_load = dissag_load.loc[:, (dissag_load != 0).any(axis=0)]
            all_load.append(dissag_load)

        load = pd.concat(all_load, axis=1)
        assert not load.isna().any().any()  # no data should be added during concat
        return load

    def _get_balanceing_area_zones(self) -> pd.Series:
        n = self.n
        zones = n.buses.balancing_area.replace(
            {
                "^CISO.*": "CISO",
                "^ERCO.*": "ERCO",
                "^MISO.*": "MISO",
                "^SPP.*": "SPP",
                "^PJM.*": "PJM",
                "^NYISO.*": "NYIS",
                "^ISONE.*": "ISNE",
            },
            regex=True,
        )
        zones = zones.replace({"": "missing"})
        if "missing" in zones:
            logger.warning("Missing BA Assignment for load dissagregation")
        return zones

    def _get_state_zones(self) -> pd.Series:
        n = self.n
        return n.buses.state

    def _get_reeds_zones(self) -> pd.Series:
        n = self.n
        return n.buses.reeds_zone

    def _make_empty_demand(self, columns: list[str]) -> pd.DataFrame:
        """
        Make a demand dataframe with zeros.
        """
        n = self.n
        return (
            pd.DataFrame(columns=columns, index=n.snapshots.get_level_values(1))
            .infer_objects()
            .fillna(0)
        )


class WritePopulation(WriteStrategy):
    """
    Based on Population Density from Breakthrough Energy.
    """

    def __init__(self, n: pypsa.Network) -> None:
        super().__init__(n)

    def _get_load_allocation_factor(
        self,
        df: pd.Series,
        zone: str,
        **kwargs,
    ) -> pd.Series:
        """
        Pulls weighting from 'build_base_network'.
        """
        logger.info("Setting load allocation factors based on BE population density")
        n = self.n
        if zone == "state":
            return n.buses.LAF_state.fillna(0)
        else:
            n.buses.Pd = n.buses.Pd.fillna(0)
            bus_load = n.buses.Pd.to_frame(name="Pd").join(df.to_frame(name="zone"))
            zone_loads = bus_load.groupby("zone")["Pd"].transform("sum")
            return bus_load.Pd / zone_loads


class WriteIndustrial(WriteStrategy):
    """
    Based on county level energy use from 2014.

    https://data.nrel.gov/submissions/97
    """

    def __init__(self, n: pypsa.Network, filepath: str) -> None:
        super().__init__(n)
        self.data = self._read_data(filepath)

    def _get_load_allocation_factor(self, zone: str, **kwargs) -> pd.Series:
        logger.info("Setting load allocation factors based on industrial demand")
        if zone == "state":
            return self._dissagregate_on_state()
        elif zone == "reeds":
            return self._dissagregate_on_reeds()
        elif zone == "ba":
            # return self._dissagregate_on_ba()
            return self._dissagregate_on_state()
        else:
            raise NotImplementedError

    def _dissagregate_on_state(self) -> pd.Series:
        laf_per_county = self.data.copy()
        totals = {}
        for state in laf_per_county.state.unique():
            df = self.data[self.data.state == state]
            totals[state] = df.demand_TBtu.sum().round(6)

        laf_per_county["laf"] = laf_per_county.apply(
            lambda x: x.demand_TBtu / totals[x.state],
            axis=1,
        )
        laf_per_county = laf_per_county.set_index("county")

        # need to account for multiple buses being in a single county
        dfs = []
        load_buses = self.get_load_buses_per_county()
        for county, buses in load_buses.items():
            dfs.append(self.get_laf_per_bus(laf_per_county, county, buses))

        laf_load_buses = pd.concat(dfs)

        laf_all_buses = pd.Series(index=self.n.buses.index).fillna(0)
        laf_all_buses[laf_load_buses.index] = laf_load_buses

        return laf_all_buses

    def _dissagregate_on_ba(self) -> pd.Series:
        raise NotImplementedError

    def _dissagregate_on_reeds(self) -> pd.Series:
        raise NotImplementedError

    @staticmethod
    def _read_data(filepath: str) -> pd.DataFrame:
        """
        Unzipped 'County_industry_energy_use.gz' csv file.
        """
        df = pd.read_csv(
            filepath,
            dtype={
                "fips_matching": int,
                "naics": int,
                "Coal": float,
                "Coke_and_breeze": float,
                "Diesel": float,
                "LPG_NGL": float,
                "MECS_NAICS": float,
                "MECS_Region": str,
                "Natural_gas": float,
                "Net_electricity": float,
                "Other": float,
                "Residual_fuel_oil": float,
                "Total": float,
                "fipscty": int,
                "fipstate": int,
                "subsector": int,
            },
        )
        df = (
            df[["fips_matching", "fipstate", "Total"]]
            .rename(
                columns={
                    "fips_matching": "county",
                    "fipstate": "state",
                    "Total": "demand_TBtu",
                },
            )
            .groupby(["county", "state"])
            .sum()
            .reset_index()
        )
        df["state"] = df.state.map(lambda x: FIPS_2_STATE[f"{x:02d}"].title())
        return df

    def get_load_buses_per_county(self) -> dict[str, list[str]]:
        """
        Gets a list of load buses, indexed by county.

        Note, load buses follow BE mapping of Pd
        """
        n = self.n
        buses_per_county = n.buses[["Pd", "county"]].fillna(0)
        buses_per_county = buses_per_county[buses_per_county.Pd != 0]

        mapper = {}

        for county in buses_per_county.county.unique():
            df = buses_per_county[buses_per_county.county == county]
            mapper[county] = df.index.to_list()

        return mapper

    @staticmethod
    def get_laf_per_bus(
        df: pd.DataFrame,
        county: str | int,
        buses: list[str],
    ) -> pd.Series:
        """
        Evenly distributes laf to buses within a county.
        """

        county_laf = df.at[int(county), "laf"]
        num_buses = len(buses)
        bus_laf = county_laf / num_buses

        return pd.Series(index=buses).fillna(bus_laf).round(6)


###
# helpers
###


def reindex_demand(df: pd.DataFrame, year: int) -> pd.DataFrame:
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


def expand_demands(
    demands: dict[str, dict[str : pd.DataFrame]],
    planning_horizons: list[int],
    scale_method: str,
) -> dict[str, dict[str : pd.DataFrame]]:
    """
    Expands demand to match planning horizons.
    """

    expanded = {}

    for sector, fuels in demands.items():
        expanded[sector] = {}
        for fuel, demand in fuels.items():
            dfs = [expand_demand(demand, x) for x in planning_horizons]
            expanded[sector][fuel] = pd.concat(dfs)
    return expanded


def expand_demand(
    demand: pd.DataFrame,
    planning_horizon: int,
) -> pd.DataFrame:
    df = demand[demand.index.year == planning_horizon]

    if not df.empty:
        return df

    profile_year = demand.index[0].year
    df = demand[demand.index.year == profile_year]
    df = reindex_demand(df, planning_horizon)
    # scale demand here
    return df


def scale_demand(method: str):
    """
    Scales demand.
    """
    if method == "efs":
        growth_rate = ReadEfs(snakemake.input.efs).get_growth_rate()
        logger.warning("No scale appied for efs data")
    elif method == "aeo":
        growth_rate = get_aeo_growth_rate(eia_api, [profile_year, planning_horizons])
        logger.warning("No scale appied for aeo data")
    else:
        raise NotImplementedError


def get_aeo_growth_rate(
    api: str,
    years: list[str],
    aeo_scenario: str = "reference",
) -> pd.DataFrame:
    """
    Get sector yearly END-USE ENERGY growth rates from AEO at a NATIONAL level.

    |      | residential | commercial  | industrial  | transport  | units |
    |----- |-------------|-------------|-------------|------------|-------|
    | 2018 |     ###     |     ###     |     ###     |     ###    |  ###  |
    | 2019 |     ###     |     ###     |     ###     |     ###    |  ###  |
    | 2020 |     ###     |     ###     |     ###     |     ###    |  ###  |
    | ...  |             |             |             |            |       |
    | 2049 |     ###     |     ###     |     ###     |     ###    |  ###  |
    | 2050 |     ###     |     ###     |     ###     |     ###    |  ###  |
    """

    def get_historical_value(api: str, year: int, sector: str) -> float:
        """
        Returns single year value at a time.
        """
        energy = EnergyDemand(sector=sector, year=year, api=api).get_data()
        return energy.value.div(1000).sum()  # trillion btu -> quads

    def get_future_values(
        api: str,
        year: int,
        sector: str,
        scenario: str,
    ) -> pd.DataFrame:
        """
        Returns all values from 2024 onwards.
        """
        energy = EnergyDemand(
            sector=sector,
            year=year,
            api=api,
            scenario=scenario,
        ).get_data()
        return energy

    logger.info("Getting AEO growth rate")

    assert min(years) > 2017
    assert max(years) < 2051

    sectors = ("residential", "commercial", "industry", "transport")

    df = pd.DataFrame(
        columns=["residential", "commercial", "industry", "transport"],
        index=years,
    )

    for year in sorted(years):
        if year < 2024:
            for sector in sectors:
                df.at[year, sector] = get_historical_value(api, year, sector)

    for sector in sectors:
        aeo = get_future_values(api, max(years), sector, aeo_scenario)
        for year in years:
            if year < 2024:
                continue
            df.at[year, sector] = aeo.at[year, "value"]

    df["units"] = "quads"
    return df


###
# main entry point
###

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        # snakemake = mock_snakemake(
        #     "build_electrical_demand",
        #     interconnect="texas",
        #     end_use="power",
        # )
        snakemake = mock_snakemake(
            "build_sector_demand",
            interconnect="texas",
            end_use="transport",
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)

    # extract user demand configuration parameters

    demand_params = snakemake.params.demand_params
    end_use = snakemake.wildcards.end_use
    eia_api = snakemake.params.eia_api

    if end_use == "power":  # electricity only study
        demand_profile = demand_params["profile"]
        scale_method = demand_params["scale"]
        demand_disaggregation = demand_params["disaggregation"]
    else:
        demand_profile = demand_params["profile"][end_use]
        scale_method = demand_params["scale"][end_use]
        demand_disaggregation = demand_params["disaggregation"][end_use]

    if scale_method == "aeo":
        assert eia_api, "Must provide EIA API key to scale demand by AEO"

    planning_horizons = n.investment_periods.to_list()
    profile_year = snakemake.params.profile_year

    # set reading and writitng strategies

    demand_files = snakemake.input.demand_files

    sns = n.snapshots

    if demand_profile == "efs":
        assert all(
            year in (2018, 2020, 2024, 2030, 2040, 2050) for year in planning_horizons
        )
        reader = ReadEfs(demand_files)
        sns = n.snapshots.get_level_values(1)

    elif demand_profile == "eia":
        assert profile_year in range(2018, 2023, 1)
        reader = ReadEia(demand_files)
        sns = n.snapshots.get_level_values(1).map(
            lambda x: x.replace(year=profile_year),
        )

    elif demand_profile == "eulp":
        stock = {"residential": "res", "commercial": "com"}
        reader = ReadEulp(demand_files, stock[end_use])  # 'res' or 'com'
        sns = n.snapshots.get_level_values(1).map(
            lambda x: x.replace(year=2018),
        )
    elif demand_profile == "cliu":
        cliu_file = demand_files[0]
        epri_file = demand_files[1]
        mecs_file = demand_files[2]
        fips_file = demand_files[3]
        sns = n.snapshots.get_level_values(1).map(
            lambda x: x.replace(year=2014),
        )
        reader = ReadCliu(
            cliu_file,
            epri_filepath=epri_file,
            mecs_filepath=mecs_file,
            fips_filepath=fips_file,
        )

    else:
        raise NotImplementedError

    if demand_disaggregation == "pop":
        writer = WritePopulation(n)
    elif demand_disaggregation == "cliu":
        cliu_file = snakemake.input.county_industrial_energy
        writer = WriteIndustrial(n, cliu_file)
    else:
        raise NotImplementedError

    demand_converter = Context(reader, writer)

    # extract demand based on strategies
    # this is raw demand, not scaled or garunteed to align to snapshots

    if end_use == "power":  # only one demand for electricity only studies
        demand = demand_converter.prepare_demand(sns=sns)  # pd.DataFrame
        demands = {"power": {"electricity": demand}}
    else:
        fuels = ("electricity", "heat", "cool")
        sector = end_use  # residential, commercial, industry, transport
        demands = demand_converter.prepare_multiple_demands(
            sector,
            fuels,
            sns=sns,
        )  # dict[str, dict[str, pd.DataFrame]]

    # assign demand to planning years and scale
    demands = expand_demands(
        demands,
        planning_horizons,
        scale_method,
    )  # dict[str, dict[str, pd.DataFrame]]

    # electricity sector study
    if end_use == "power":
        demands[end_use]["electricity"].round(4).to_csv(
            snakemake.output.elec_demand,
            index=True,
        )
    # sector coupling demand
    else:
        demands[end_use]["electricity"].round(4).to_csv(
            snakemake.output.elec_demand,
            index=True,
        )
        demands[end_use]["heat"].round(4).to_csv(
            snakemake.output.heat_demand,
            index=True,
        )
        demands[end_use]["cool"].round(4).to_csv(
            snakemake.output.cool_demand,
            index=True,
        )
