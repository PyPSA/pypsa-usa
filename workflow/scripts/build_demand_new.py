# PyPSA USA Authors
"""
Builds the demand data for the PyPSA network.

Call a demand source, or multiple via...

    context = Context(AdsDemand(demand_path))
    context.prepare_demand(demand_path)
    # logic to apply ADS demand

    context.strategy = EiaDemand()
    context.prepare_demand(demand_path)
    # logic to apply other demand from eia


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
    - ads_renewables:
    - ads_2032:
    - eia: (GridEmissions data file)
    - efs: (NREL EFS Load Forecasts)

**Outputs**

    - demand: Path to the demand CSV file.
"""

from __future__ import annotations

import logging
from itertools import product
from pathlib import Path

from typing import List, Optional

import constants as const
import pandas as pd
import xarray as xr
import pypsa
from _helpers import configure_logging

import sys

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

STATE_2_CODE = const.STATE_2_CODE
CODE_2_STATE = {value: key for key, value in STATE_2_CODE.items()}
STATE_TIMEZONE = const.STATE_2_TIMEZONE


class Context:
    """
    The Context defines the interface of interest to clients.
    """

    def __init__(
        self, read_strategy: ReadStrategy, write_strategy: WriteStrategy
    ) -> None:
        self._read_strategy = read_strategy
        self._write_strategy = write_strategy

    @property
    def read_strategy(self) -> ReadStrategy:
        """
        The Context maintains a reference to the Strategy objects.
        """
        return self._read_strategy

    @read_strategy.setter
    def strategy(self, strategy: ReadStrategy) -> None:
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """
        self._read_strategy = strategy

    @property
    def write_strategy(self) -> WriteStrategy:
        """
        The Context maintains a reference to the Strategy objects.
        """
        return self._write_strategy

    @write_strategy.setter
    def strategy(self, strategy: WriteStrategy) -> None:
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """
        self._write_strategy = strategy

    def _read(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Delegate reading to the strategy.
        """
        return self._read_strategy.prepare_demand(filepath, **kwargs)

    def _write(self, demand: pd.DataFrame, n: pypsa.Network) -> pd.DataFrame:
        """
        Delegate writing to the strategy.
        """
        return self._write_strategy.retrieve_demand(demand, n)

    def prepare_demand(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Arguments
            fuel: str = None,
            sector: str = None,
            year: int = None
        """
        return self._read(filepath, *kwargs)

    def retrieve_demand(self, filepath: str, n: pypsa.Network, **kwargs) -> None:
        """
        Reads demand to apply to a network.
        """
        demand = self._read(filepath, *kwargs)
        return self._write(demand, n)


###
# READ STRATEGIES
###


class ReadStrategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.
    """

    def __init__(self, filepath: Optional[str] = None) -> None:
        self.filepath = filepath
        self.demand = self._get_demand()

    @property
    def units():
        return "MW"

    @abstractmethod
    def _read_data(self, **kwargs) -> pd.DataFrame:
        """
        Reads raw data into any arbitraty data structure
        """
        pass

    @abstractmethod
    def _format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formats raw data into following datastructure.

        This datastructure MUST be indexed with the following INDEX labels:
        - snapshot (use self._format_snapshot_index() to format this)
        - sector (must be in "all", "industry", "residential", "commercial", "transport")
        - subsector (any value)
        - end use fuel (must be in "all", "elec", "heat", "cool", "gas")

        This datastructure MUST be indexed with the following COLUMN labels:
        - Per geography type (ie. dont mix state and ba headers)

        |                     |        |           |        | geo_name_1 | geo_name_2 | ... | geo_name_n |
        | snapshot            | sector | subsector | fuel   |            |            |     |            |
        |---------------------|--------|-----------|--------|------------|------------|-----|------------|
        | 2019-01-01 00:00:00 | all    | all       | elec   |    ###     |    ###     |     |    ###     |
        | 2019-01-01 01:00:00 | all    | all       | elec   |    ###     |    ###     |     |    ###     |
        | 2019-01-01 02:00:00 | all    | all       | elec   |    ###     |    ###     |     |    ###     |
        | ...                 | ...    | ...       | ...    |            |            |     |    ###     |
        | 2019-12-31 23:00:00 | all    | all       | elec   |    ###     |    ###     |     |    ###     |

        """
        pass

    def _get_demand(self) -> pd.DataFrame:
        """
        Getter for returning all demand
        """
        df = self._read_data()
        df = self._format_data(df)
        self._check_index(df)
        return df

    def _check_index(self, df: pd.DataFrame) -> None:
        """
        Enforces dimension labels
        """
        assert all(
            x in ["snapshot", "sector", "subsector", "fuel"] for x in df.index.names
        )

        assert all(
            x in ["all", "industry", "residential", "commercial", "transport"]
            for x in df.index.get_level_values("sector").unique()
        )

        assert all(
            x in ["all", "elec", "heat", "cool", "gas"]
            for x in df.index.get_level_values("fuel").unique()
        )

    @staticmethod
    def _format_snapshot_index(df: pd.DataFrame) -> pd.DataFrame:
        """Makes index into datetime"""
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

    @staticmethod
    def _filter_pandas(
        df: pd.DataFrame, index: str, value: List[str] | List[int]
    ) -> pd.DataFrame:
        return df[df.index.get_level_values(index).isin(value)].copy()

    def prepare_demand(
        self,
        fuel: str | List[str] = None,
        sector: str | List[str] = None,
        year: int = None,
    ) -> pd.DataFrame:
        """
        Public interface to extract data
        """

        demand = self.demand

        if fuel:
            if isinstance(fuel, str):
                fuel = [fuel]
            demand = self._filter_pandas(demand, "fuel", fuel)
        if sector:
            if isinstance(sector, str):
                sector = [sector]
            demand = self._filter_pandas(demand, "sector", sector)
        if year:
            demand = self._filter_pandas(demand, "year", [year])

        return demand


class ReadEia(ReadStrategy):
    """Reads data from GridEmissions"""

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
        Combine EIA Demand Data to Match GIS Shapes
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
    Reads in electrifications future study demand
    """

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
            }
        )
        df["fuel"] = "elec"
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
        """Builds snapshots based on UTC time"""

        df = self._apply_timezones(df)
        df = self._build_datetime(df)
        return df.set_index("time").sort_index()

    @staticmethod
    def _apply_timezones(df: pd.DataFrame) -> pd.DataFrame:
        """Changes local time to relative time from UTC"""

        def apply_timezone_shift(timezone: str) -> int:
            """All shifts realitive to UTC time"""
            if timezone == "US/Pacific":
                return -8
            elif timezone == "US/Mountain":
                return -7
            elif timezone == "US/Central":
                return -6
            elif timezone == "US/Eastern":
                return -5
            elif timezone == "US/Alaska":
                return -9
            elif timezone == "Pacific/Honolulu":
                return -11
            else:
                raise KeyError(f"Timezone {timezone} not mapped :(")

        # mapper of {state:0} where value is offset from UTC
        utc_shift = {
            state: apply_timezone_shift(STATE_TIMEZONE[state])
            for state in STATE_TIMEZONE
        }

        df["utc_shift"] = df.State.map(utc_shift)
        df["UtcHourID"] = df.LocalHourID + df.utc_shift
        df["UtcHourID"] = df.UtcHourID.map(lambda x: x if x > 0 else x + 8760)
        df = df.drop(columns=["utc_shift"])
        return df

    @staticmethod
    def _build_datetime(df: pd.DataFrame) -> pd.DataFrame:
        """Builds snapshot from EFS data"""
        # minus 1 cause indexing starts at 1
        df["hoy"] = pd.to_timedelta(df.UtcHourID - 1, unit="h")
        df["time"] = pd.to_datetime(df.Year, format="%Y") + df.hoy
        return df.drop(columns=["Year", "UtcHourID", "hoy"])


###
# WRITE STRATEGIES
###


class WriteStrategy(ABC):
    """
    Disaggregates demand based on a specified method
    """

    def __init__(self, demand: pd.DataFrame, zone: str, n: pypsa.Network) -> None:
        self.demand = demand
        self._check_datastructure()
        assert zone in ("ba", "state", "reeds")
        self.zone = zone
        self.n = n

    @abstractmethod
    def _get_load_allocation_factor(self) -> pd.Series:
        """Sets load allocation factor per bus"""
        pass

    def dissagregate_demand(
        self,
        sectors: Optional[str | List[str]] = None,
        subsectors: Optional[str | List[str]] = None,
        fuels: Optional[str | List[str]] = None,
    ) -> pd.DataFrame:
        """
        Public load dissagregation method

        Data is returned in the format of:

        |                     | BusName_1 | BusName_2 | ... | BusName_n |
        |---------------------|-----------|-----------|-----|-----------|
        | 2019-01-01 00:00:00 |    ###    |    ###    |     |    ###    |
        | 2019-01-01 01:00:00 |    ###    |    ###    |     |    ###    |
        | 2019-01-01 02:00:00 |    ###    |    ###    |     |    ###    |
        | ...                 |           |           |     |    ###    |
        | 2019-12-31 23:00:00 |    ###    |    ###    |     |    ###    |
        """

        # get zone area demand for specific sector and fuel
        demand = self._filter_demand(sectors, subsectors, fuels)
        demand = self._group_demand(demand)

        # assign buses to dissagregation zone
        dissagregation_zones = self._get_load_dissagregation_zones()

        # get implementation specific dissgregation factors
        laf = self._get_load_allocation_factor(dissagregation_zones)

        # disaggregate load to buses
        return self._disaggregate_demand_to_buses(demand)

    def _get_load_dissagregation_zones(self) -> pd.Series:
        """Map each bus to the load dissagregation zone (ie. states, ba, ...)"""
        if self.zone == "ba":
            return self._get_balanceing_area_zones()
        elif self.zone == "state":
            return self._get_state_zones()
        elif self.zone == "reeds":
            return self._get_state_zones()
        else:
            raise NotImplementedError

    def _check_datastructure(self) -> None:
        """Confirms formatting of input datastructure"""
        assert all(
            x in ["snapshot", "sector", "subsector", "fuel"]
            for x in self.demand.index.names
        )
        assert not self.demand.empty

    def _filter_on_snapshots(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters demand on network snapshots.
        """
        snapshots = n.snapshots
        return df[df.index.get_level_values("snapshot") == snapshots].copy()

    @staticmethod
    def _filter_on_use(df: pd.DataFrame, level: str, values: List[str]):
        return df[df.index.get_level_values(level) in values].copy()

    @staticmethod
    def _group_demand(df: pd.DataFrame, agg_strategy: str = "sum") -> pd.DataFrame:
        grouped = df.index.droplevel(level=["sector", "subsector", "fuel"])
        grouped = grouped.groupby(level="snapshot")
        if agg_strategy == "sum":
            return grouped.sum()
        elif agg_strategy == "mean":
            return grouped.mean()
        else:
            raise NotImplementedError

    def _filter_demand(
        self,
        sectors: Optional[str | List[str]] = None,
        subsectors: Optional[str | List[str]] = None,
        fuels: Optional[str | List[str]] = None,
    ) -> pd.DataFrame:
        """Filters on snapshots, sector, and fuel"""

        df = self.demand
        df = self._filter_on_snapshots(df)
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

    def _disaggregate_demand_to_buses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zone power demand is disaggregated to buses proportional to Pd.
        """
        n = self.n
        demand_aligned = df.reindex(
            columns=n.buses["load_zone"].unique(),
            fill_value=0,
        )
        bus_demand = pd.DataFrame()
        for load_zone in n.buses["load_zone"].unique():
            LAF = n.buses.loc[n.buses["load_zone"] == load_zone, "LAF"]
            zone_bus_demand = (
                demand_aligned[load_zone].values.reshape(-1, 1) * LAF.values.T
            )
            bus_demand = pd.concat(
                [bus_demand, pd.DataFrame(zone_bus_demand, columns=LAF.index)],
                axis=1,
            )
        bus_demand.index = n.snapshots
        n.buses.drop(columns=["LAF"], inplace=True)
        return bus_demand.fillna(0)

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
    """Based on Population Density from Breakthrough Energy"""

    def __init__(self, demand: pd.DataFrame, n: pypsa.Network) -> None:
        super().__init__(demand, n)

    def _get_load_allocation_factor(self, df: pd.Series) -> pd.Series:
        """Load allocation set on population density"""

        n = self.n
        n.buses.Pd = n.buses.Pd.fillna(0)
        group_sums = n.buses.groupby("load_zone")["Pd"].transform("sum")
        n.buses["LAF"] = n.buses["Pd"] / group_sums
        return n.buses.LAF


###
# helpers
###


def attach_demand(n: pypsa.Network, demand_per_bus: pd.DataFrame, carrier: str = "AC"):
    """
    Add demand to network from specified configuration setting.

    Returns network with demand added.
    """
    n.madd(
        "Load",
        demand_per_bus.columns,
        bus=demand_per_bus.columns,
        p_set=demand_per_bus,
        carrier="AC",
    )


###
# main entry point
###

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_demand", interconnect="western")
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.base_network)

    snapshot_config = snakemake.params["snapshots"]
    n.set_snapshots(
        pd.date_range(
            freq="h",
            start=pd.to_datetime(snapshot_config["start"]),
            end=pd.to_datetime(snapshot_config["end"]),
            inclusive=snapshot_config["inclusive"],
        ),
    )

    demand_path = snakemake.input.efs
    configuration = snakemake.config["network_configuration"]

    ReadEfs(filepath=demand_path)
    print("done")

    # if configuration == "eia":
    #     demand_converter = Context(ReadEia(), WriteEia())
    # else:
    #     demand_converter = Context(ReadEia(), WriteEia())

    # # optional arguments of 'fuel', 'sector', 'year'
    # demand = demand_converter.retrieve_demand(demand_path, n)

    # attach_demand(n, demand)
