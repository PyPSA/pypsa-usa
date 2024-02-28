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

from typing import List

import constants as const
import pandas as pd
import xarray as xr
import pypsa
from _helpers import configure_logging
from _helpers import local_to_utc

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

    # def __init__(self, filepath: str) -> None:
    #     self.filepath = filepath
    #     self.demand = self._get_demand()
    #     self._check_index()

    @abstractmethod
    def _read_data(self, filepath: str) -> pd.DataFrame:
        """
        Reads raw data.
        """
        pass

    @abstractmethod
    def _format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Formats raw data.
        """
        return data

    def _get_demand(self, filepath: str) -> pd.DataFrame:
        """
        Gets raw data.
        """
        df = self._read_data(filepath)
        return self._format_data(df)

    def _check_index(self) -> None:
        """
        Add asserts on index labels.
        """
        assert all(
            x in ["YEAR", "HOUR", "REGION", "SECTOR", "FUEL", "VALUE"]
            for x in self.demand.columns
        )

    @staticmethod
    def _filter_pandas(
        df: pd.DataFrame, index: str, value: list[str] | list[int]
    ) -> pd.DataFrame:
        return df[df[index].isin(value)].copy()

    def prepare_demand(
        self,
        filepath: str,
        fuel: str | list[str] = None,
        sector: str | list[str] = None,
        year: int = None,
    ) -> pd.DataFrame:

        demand = self._get_demand(filepath)

        if fuel:
            if isinstance(fuel, str):
                fuel = [fuel]
            self._filter_pandas(demand, "fuel", fuel)
        if sector:
            if isinstance(sector, str):
                sector = [sector]
            self._filter_pandas(demand, "sector", sector)
        if year:
            self._filter_pandas(demand, "year", [year])

        return demand


class ReadEia(ReadStrategy):

    def _read_data(self, filepath: str) -> pd.DataFrame:
        """
        Reads raw data.
        """
        logger.info("Building Load Data using EFS demand")
        return pd.read_csv(filepath, engine="pyarrow")
        # df = pd.read_csv(filepath, engine="pyarrow", index_col="timestamp").dropna(axis=1)
        # return xr.Dataset.from_dataframe(df)

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

    @staticmethod
    def correct_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Corrects balancing authority naming.
        """
        df["Arizona"] = df.pop("SRP") + df.pop("AZPS")
        return df


###
# WRITE STRATEGIES
###


class WriteStrategy(ABC):
    """
    Retrieves demand based on a specified network.
    """

    def retrieve_demand(self, demand: pd.DataFrame, n: pypsa.Network) -> pd.DataFrame:
        """
        Writes demand.
        """
        demand = self.filter_on_snapshots(demand, n)
        demand = self.pivot_data(demand)
        self.update_load_dissagregation_names(n)
        demand = self.get_demand_buses(demand, n)
        self.set_load_allocation_factor(n)
        return self.disaggregate_demand_to_buses(demand, n)

    @staticmethod
    def pivot_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivots data for easier processing.
        """
        df = df[["HOUR", "REGION", "VALUE"]].pivot(index="HOUR", columns="REGION")
        return df.loc[:, ("VALUE")]

    @abstractmethod
    def update_load_dissagregation_names(self, n: pypsa.Network):
        """
        Corrects load dissagreagation names.
        """
        pass

    @abstractmethod
    def get_demand_buses(self, demand: pd.DataFrame, n: pypsa.Network):
        """
        Applies load aggregation facto to network.
        """
        pass

    def set_load_allocation_factor(self, n: pypsa.Network):
        """
        Defines Load allocation factor for each bus according to load_dissag
        for balancing areas.
        """
        n.buses.Pd = n.buses.Pd.fillna(0)
        group_sums = n.buses.groupby("load_dissag")["Pd"].transform("sum")
        n.buses["LAF"] = n.buses["Pd"] / group_sums

    def filter_on_snapshots(self, df: pd.DataFrame, n: pypsa.Network) -> pd.DataFrame:
        """
        Filters demand on network snapshots.
        """
        df = df.set_index("HOUR")
        df = df.loc[n.snapshots.intersection(df.index)]
        return df.reset_index(names="HOUR").drop_duplicates(
            subset=["HOUR", "REGION", "YEAR", "FUEL", "SECTOR"], keep="first"
        )

    def disaggregate_demand_to_buses(
        self, df: pd.DataFrame, n: pypsa.Network
    ) -> pd.DataFrame:
        """
        Zone power demand is disaggregated to buses proportional to Pd.
        """
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

    if configuration == "eia":
        demand_converter = Context(ReadEia(), WriteEia())
    else:
        demand_converter = Context(ReadEia(), WriteEia())

    # optional arguments of 'fuel', 'sector', 'year'
    demand = demand_converter.retrieve_demand(demand_path, n)

    attach_demand(n, demand)
