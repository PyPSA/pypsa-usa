"""
Builds the demand data for the PyPSA network.

**Relevant Settings**

.. code:: yaml

    network_configuration:

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

import logging
import sys
from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import constants as const
import pandas as pd
import pypsa
import xarray as xr
from _helpers import configure_logging
from eia import EnergyDemand

logger = logging.getLogger(__name__)

STATE_2_CODE = const.STATE_2_CODE
CODE_2_STATE = {value: key for key, value in STATE_2_CODE.items()}
STATE_TIMEZONE = const.STATE_2_TIMEZONE


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


###
# READ STRATEGIES
###


class ReadStrategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.
    """

    def __init__(self, filepath: str | None = None) -> None:
        self.filepath = filepath

    @property
    def units():
        return "MW"

    @abstractmethod
    def _read_data(self, **kwargs) -> pd.DataFrame:
        """
        Reads raw data into any arbitraty data structure.
        """
        pass

    def read_demand(self) -> pd.DataFrame:
        """
        Public interface to extract data.
        """

        df = self._read_data()
        df = self._format_data(df)
        self._check_index(df)
        return df

    @abstractmethod
    def _format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formats raw data into following datastructure.

        This datastructure MUST be indexed with the following INDEX labels:
        - snapshot (use self._format_snapshot_index() to format this)
        - sector (must be in "all", "industry", "residential", "commercial", "transport")
        - subsector (any value)
        - end use fuel (must be in "all", "electricity", "heat", "cool", "gas")

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
            x in ["all", "electricity", "heat", "cool", "gas"]
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

    def _format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formats raw data.
        """
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
        return pd.read_csv(self.filepath, engine="pyarrow")

    def _format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formats raw data.
        """

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
        df["time"] = pd.to_datetime(df.Year, format="%Y") + df.hoy
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

        # assign buses to dissagregation zone
        dissagregation_zones = self._get_load_dissagregation_zones(zone)

        # get implementation specific dissgregation factors
        laf = self._get_load_allocation_factor(df=dissagregation_zones, zone=zone)

        # checks that sum of all LAFs is equal to the number of zones.
        # assert abs(laf.sum() - len(dissagregation_zones.unique())) <= 0.0001

        # disaggregate load to buses
        zone_data = dissagregation_zones.to_frame(name="zone").join(
            laf.to_frame(name="laf"),
        )
        return self._disaggregate_demand_to_buses(demand, zone_data)

    def _get_load_dissagregation_zones(self, zone: str) -> pd.Series:
        """
        Map each bus to the load dissagregation zone (ie.

        states, ba, ...)
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
        assert len(filtered.index.get_level_values("snapshot").unique()) == len(sns)
        return filtered

    @staticmethod
    def _filter_on_use(df: pd.DataFrame, level: str, values: list[str]):
        return df[df.index.get_level_values(level) in values].copy()

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

        if isinstance(sns, pd.DatetimeIndex):  # not sure if leap years will break this
            snapshots = sns
            assert len(snapshots) == len(n.snapshots)
            filtered = self._filter_on_snapshots(df, snapshots)
            df = filtered.reset_index()
            df["snapshot"] = df.snapshot.map(
                lambda x: x.replace(year=n.snapshots[0].year),
            )
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


class WritePopulation(WriteStrategy):
    """
    Based on Population Density from Breakthrough Energy.
    """

    def __init__(self, n: pypsa.Network) -> None:
        super().__init__(n)

    def _get_load_allocation_factor(self, df: pd.Series, zone: str) -> pd.Series:
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


###
# helpers
###


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

        snakemake = mock_snakemake("build_demand", interconnect="western")
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.base_network)

    # build snapshots

    snapshot_config = snakemake.params["snapshots"]
    n.set_snapshots(
        pd.date_range(
            freq="h",
            start=pd.to_datetime(snapshot_config["start"]),
            end=pd.to_datetime(snapshot_config["end"]),
            inclusive=snapshot_config["inclusive"],
        ),
    )

    # extract user demand configuration parameters

    demand_params = snakemake.params.demand_params
    demand_profile = demand_params.get("profile", "eia")
    demand_scale = demand_params.get("scale", 1)
    demand_disaggregation = demand_params.get("disaggregation", "pop")
    eia_api = snakemake.params.eia_api

    if demand_scale == "aeo":
        assert eia_api, "Must provide EIA API key to scale demand by AEO"

    profile_year = pd.to_datetime(snapshot_config["start"]).year
    planning_horizons = snakemake.params.planning_horizons
    assert len(planning_horizons) == 1  # remove for myopic/rolling horizon
    planning_year = planning_horizons[0]

    # set reading and writitng strategies

    if demand_profile == "efs":
        assert planning_year in (2018, 2020, 2024, 2030, 2040, 2050)
        reader = ReadEfs(snakemake.input.efs)
        sns = n.snapshots.map(lambda x: x.replace(year=planning_year))
        profile_year = planning_year  # do not scale EFS data

    elif demand_profile == "eia":
        assert profile_year in range(2018, 2023, 1)
        reader = ReadEia(snakemake.input.eia[0])
        sns = n.snapshots

    else:
        raise NotImplementedError

    if demand_disaggregation == "pop":
        writer = WritePopulation(n)

    else:
        raise NotImplementedError

    demand_converter = Context(reader, writer)

    # extract demand based on strategies

    # optional arguments of 'fuel', 'sector', 'subsector', 'sns'
    demand = demand_converter.prepare_demand(sns=sns)

    # scale demand based on planning year and user input

    if profile_year == planning_year:
        pass
    elif isinstance(demand_scale, (int, float)):
        demand *= demand_scale
    elif demand_scale == "efs":
        growth_rate = ReadEfs(snakemake.input.efs).get_growth_rate()
        logger.warning("No scale appied for efs data")
    elif demand_scale == "aeo":
        growth_rate = get_aeo_growth_rate(eia_api, [profile_year, planning_year])
        logger.warning("No scale appied for aeo data")

    demand.round(4).to_csv(snakemake.output.demand, index=True)
