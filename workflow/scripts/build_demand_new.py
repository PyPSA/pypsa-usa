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
from _helpers import local_to_utc

import sys

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


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
        - Hourly snapshots
        - Sector
        - End use fuel

        This datastructure MUST be indexed with the following COLUMN labels:
        - Per geography type (ie. dont mix state and ba headers)

        |                     |        |        | geo_name_1 | geo_name_2 | ... | geo_name_n |
        | snapshot            | sector | fuel   |            |            |     |            |
        |---------------------|--------|--------|------------|------------|-----|------------|
        | 2019-01-01 00:00:00 | all    | elec   |    ###     |    ###     |     |    ###     |
        | 2019-01-01 01:00:00 | all    | elec   |    ###     |    ###     |     |    ###     |
        | 2019-01-01 02:00:00 | all    | elec   |    ###     |    ###     |     |    ###     |
        | ...                 | ...    | ...    |            |            |     |    ###     |
        | 2019-12-31 23:00:00 | all    | elec   |    ###     |    ###     |     |    ###     |

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
        assert all(x in ["snapshot", "sector", "fuel"] for x in df.index.names)

    @staticmethod
    def _format_snapshot_index(df: pd.DataFrame, assign_name=True) -> pd.DataFrame:
        """Makes index into datetime"""
        if df.index.nlevels > 1:
            logger.warning("Can not format snapshot index level")
            return df
        df.index = pd.to_datetime(df.index)
        if not assign_name:
            return df
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
        df = df.set_index([df.index, "sector", "fuel"])
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

    def _read_data(self, filepath: str) -> pd.DataFrame:

        if not self.filepath:
            logger.error("Must provide filepath for EFS data")
            sys.exit()

        logger.info("Building Load Data using EFS demand")
        return pd.read_csv(filepath, engine="pyarrow")

    def _format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formats raw data.
        """
        df = self.correct_data(df)
        df = df.melt(id_vars="timestamp")
        df.timestamp = pd.to_datetime(df.timestamp)
        df["YEAR"] = df.timestamp.dt.year
        df["FUEL"] = "electricity"
        df["SECTOR"] = "all"
        return df.rename(
            columns={
                "timestamp": "HOUR",
                "value": "VALUE",
                "variable": "REGION",
            },
        )


###
# WRITE STRATEGIES
###


class WriteStrategy(ABC):
    """
    Disaggregates demand based on a specified method
    """

    def __init__(self, n: pypsa.Network) -> None:
        self.n = n

    def dissagregate_demand(self, demand: pd.DataFrame) -> pd.DataFrame:
        """
        Dissagregates demand based on user defined strategy

        Data is returned in the format of:

        |                     | BusName_1 | BusName_2 | ... | BusName_n |
        |---------------------|-----------|-----------|-----|-----------|
        | 2019-01-01 00:00:00 |    ###    |    ###    |     |    ###    |
        | 2019-01-01 01:00:00 |    ###    |    ###    |     |    ###    |
        | 2019-01-01 02:00:00 |    ###    |    ###    |     |    ###    |
        | ...                 |           |           |     |    ###    |
        | 2019-12-31 23:00:00 |    ###    |    ###    |     |    ###    |

        """
        demand = self.filter_on_snapshots(demand)
        demand = self.pivot_data(demand)
        # self.update_load_dissagregation_names(n)
        demand = self.get_demand_buses(demand)
        # self.set_load_allocation_factor(n)
        return self.disaggregate_demand_to_buses(demand)

    @staticmethod
    def pivot_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivots data for easier processing.
        """
        df = df[["HOUR", "REGION", "VALUE"]].pivot(index="HOUR", columns="REGION")
        return df.loc[:, ("VALUE")]

    @abstractmethod
    def update_load_dissagregation_names(self):
        """
        Corrects load dissagreagation names.
        """
        pass

    @abstractmethod
    def get_demand_buses(self, demand: pd.DataFrame):
        """
        Applies load aggregation facto to network.
        """
        pass

    def set_load_allocation_factor(self):
        """
        Defines Load allocation factor for each bus according to load_dissag
        for balancing areas.
        """
        n = self.n
        self.n.buses.Pd = n.buses.Pd.fillna(0)
        group_sums = n.buses.groupby("load_dissag")["Pd"].transform("sum")
        n.buses["LAF"] = n.buses["Pd"] / group_sums

    def filter_on_snapshots(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters demand on network snapshots.
        """
        n = self.n
        df = df.set_index("HOUR")
        df = df.loc[n.snapshots.intersection(df.index)]
        return df.reset_index(names="HOUR").drop_duplicates(
            subset=["HOUR", "REGION", "YEAR", "FUEL", "SECTOR"], keep="first"
        )

    def disaggregate_demand_to_buses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zone power demand is disaggregated to buses proportional to Pd.
        """
        n = self.n
        demand_aligned = df.reindex(
            columns=n.buses["load_dissag"].unique(),
            fill_value=0,
        )
        bus_demand = pd.DataFrame()
        for load_dissag in n.buses["load_dissag"].unique():
            LAF = n.buses.loc[n.buses["load_dissag"] == load_dissag, "LAF"]
            zone_bus_demand = (
                demand_aligned[load_dissag].values.reshape(-1, 1) * LAF.values.T
            )
            bus_demand = pd.concat(
                [bus_demand, pd.DataFrame(zone_bus_demand, columns=LAF.index)],
                axis=1,
            )
        bus_demand.index = n.snapshots
        n.buses.drop(columns=["LAF"], inplace=True)
        return bus_demand.fillna(0)


class WriteEia(WriteStrategy):
    """
    Write EIA demand data.
    """

    def update_load_dissagregation_names(self, n: pypsa.Network):
        n.buses["load_dissag"] = n.buses.balancing_area.replace(
            {"^CISO.*": "CISO", "^ERCO.*": "ERCO"},
            regex=True,
        )
        n.buses["load_dissag"] = n.buses.load_dissag.replace({"": "missing_ba"})

    def get_demand_buses(self, demand: pd.DataFrame, n: pypsa.Network):
        intersection = set(demand.columns).intersection(n.buses.load_dissag.unique())
        return demand[list(intersection)]


###
# helpers
###


def attach_demand(n: pypsa.Network, demand_per_bus: pd.DataFrame):
    """
    Add demand to network from specified configuration setting.

    Returns network with demand added.
    """
    demand_per_bus.index = pd.to_datetime(demand_per_bus.index)
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

    demand_path = snakemake.input.eia
    configuration = snakemake.config["network_configuration"]

    ReadEia(filepath=demand_path)
    print("done")

    # if configuration == "eia":
    #     demand_converter = Context(ReadEia(), WriteEia())
    # else:
    #     demand_converter = Context(ReadEia(), WriteEia())

    # # optional arguments of 'fuel', 'sector', 'year'
    # demand = demand_converter.retrieve_demand(demand_path, n)

    # attach_demand(n, demand)
