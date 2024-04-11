"""
Module for adding the gas sector.

**Description**

This module will add a state level copperplate natural gas network to the model.
Specifically, it will do the following

- Adds state level natural gas buses
- Converts exisitng OCGT and CCGT generators to links
- Creates capacity constrained pipelines between state gas buses (links)
- Creates capacity constraind natural gas processing facilites (generators)
- Creates capacity and energy constrainted underground gas storage facilities
- Creates energy constrained linepack storage (storage units)
- Creates capacity constrained pipelines to states neighbouring the interconnect
- Creates capacity and energy constrained import/exports to international connections
- Adds import/export historical natural gas prices


**Relevant Settings**

.. code:: yaml

    sector:
      natural_gas:
        allow_imports_exports: true # only true implemented
        cyclic_storage: false

**Inputs**

- n: pypsa.Network:
    - Network to add the natural gas network to. Note, the electrical network represntation should be done by this point.

- year: int,
    - Year to extract natural gas data for. Must be between ``2009`` and ``2022``

- api: str,
    - EIA API key. Get from https://www.eia.gov/opendata/register.php

- interconnect: str = "western",
    - Name of interconnect. Must be in ("eastern", "western", "texas", "usa")

- county_path: str
    - ``data/counties/cb_2020_us_county_500k.shp``: County shapes in the USA

- pipelines_path: str
    - ``EIA-StatetoStateCapacity_Jan2023.xlsx`` : State to state pipeline capacity from EIA

- pipeline_shape_path: str:
    - ``pipelines.geojson`` at a National level

**Outputs**

- `pypsa.Network`
"""

import logging

import constants
import geopandas as gpd
import pandas as pd
import pypsa
from pypsa.components import Network

logger = logging.getLogger(__name__)
from abc import ABC, abstractmethod
from math import pi
from typing import Dict, List, Union

import eia
import numpy as np
import yaml

###
# Constants
###

# for converting everthing into MWh_th
MWH_2_MMCF = constants.NG_MWH_2_MMCF
MMCF_2_MWH = 1 / MWH_2_MMCF
KJ_2_MWH = (1 / 1000) * (1 / 3600)

###
# Geolocation of Assets class
###


class StateGeometry:
    """
    Holds state boundry data.
    """

    def __init__(self, shapefile: str) -> None:
        """
        Counties shapefile.
        """
        self._counties = gpd.read_file(shapefile)
        self._state_center_points = None
        self._states = None

    @property
    def counties(self) -> gpd.GeoDataFrame:
        """
        Spatially resolved counties.
        """
        return self._counties

    @property
    def states(self) -> gpd.GeoDataFrame:
        """
        Spatially resolved states.
        """
        if self._states:
            return self._states
        else:
            self._states = self._get_state_boundaries()
            return self._states

    @property
    def state_center_points(self) -> gpd.GeoDataFrame:
        """
        Center points of Sates.
        """
        if self._state_center_points:
            return self._state_center_points
        else:
            if not self._states:
                self._states = self._get_state_boundaries()
            self._state_center_points = self._get_state_center_points()
            return self._state_center_points

    def _get_state_boundaries(self) -> gpd.GeoDataFrame:
        """
        Gets admin boundaries of state.
        """
        return (
            self._counties.dissolve("STATE_NAME")
            .rename(columns={"STUSPS": "STATE"})
            .reset_index()[["STATE_NAME", "STATE", "geometry"]]
        )

    def _get_state_center_points(self) -> gpd.GeoDataFrame:
        """
        Gets centerpoints of states using county shapefile.
        """
        gdf = self._states.copy().rename(columns={"geometry": "shape"})
        gdf["geometry"] = gdf["shape"].map(lambda x: x.centroid)
        gdf[["x", "y"]] = gdf["geometry"].apply(
            lambda x: pd.Series({"x": x.x, "y": x.y}),
        )
        return gdf[["STATE", "x", "y"]]


###
# MAIN DATA INTERFACE
###


class GasData(ABC):
    """
    Main class to interface with data.
    """

    state_2_interconnect = constants.STATES_INTERCONNECT_MAPPER
    state_2_name = {v: k for k, v in constants.STATE_2_CODE.items()}
    name_2_state = constants.STATE_2_CODE
    states_2_remove = [
        x for x, y in constants.STATES_INTERCONNECT_MAPPER.items() if not y
    ]

    def __init__(self, year: int, interconnect: str) -> None:
        self.year = year
        if interconnect.lower() not in ("western", "eastern", "texas", "usa"):
            logger.debug(f"Invalid interconnect of {interconnect}. Setting to 'usa'")
            self.interconnect = "usa"  # no filtering of data
        else:
            self.interconnect = interconnect.lower()
        self._data = self._get_data()

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @abstractmethod
    def read_data(self) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        pass

    @abstractmethod
    def format_data(self, data: Union[pd.DataFrame, gpd.GeoDataFrame]) -> pd.DataFrame:
        pass

    def _get_data(self) -> pd.DataFrame:
        data = self.read_data()
        return self.format_data(data)

    @abstractmethod
    def build_infrastructure(self, n: pypsa.Network) -> None:
        pass

    def filter_on_interconnect(
        self,
        df: pd.DataFrame,
        additional_removals: list[str] = None,
    ) -> pd.DataFrame:
        """
        Name of states must be in column called 'STATE'.
        """

        states_2_remove = self.states_2_remove
        if additional_removals:
            states_2_remove += additional_removals

        if "STATE" not in df.columns:
            logger.debug("Natual gas data notfiltered due to incorrect data formatting")
            return df

        df = df[~df.STATE.isin(states_2_remove)].copy()

        if self.interconnect == "usa":
            return df
        else:
            df["interconnect"] = df.STATE.map(self.state_2_interconnect)
            assert not df.interconnect.isna().any()
            df = df[df.interconnect == self.interconnect]
            if df.empty:
                logger.warning(
                    f"Empty natural gas data for interconnect {self.interconnect}",
                )
            return df.drop(columns="interconnect")


class GasBuses(GasData):
    """
    Creator for natural gas buses.

    Argumets:
        County shapefile of United States
    """

    def __init__(self, interconnect: str, counties: str) -> None:
        self.states = StateGeometry(counties)
        super().__init__(
            year=2020,
            interconnect=interconnect,
        )  # year locked for location mapping

    def read_data(self) -> gpd.GeoDataFrame:
        return pd.DataFrame(self.states.state_center_points)

    def format_data(self, data: gpd.GeoDataFrame) -> pd.DataFrame:
        data = pd.DataFrame(data)
        data["name"] = data.STATE.map(self.state_2_name)
        return self.filter_on_interconnect(data)

    def build_infrastructure(self, n: Network) -> None:

        states = self.data.copy().set_index("STATE")

        n.madd(
            "Bus",
            names=states.index,
            suffix=" gas",
            x=states.x,
            y=states.y,
            carrier="gas",
            unit="MWh_th",
            interconnect=self.interconnect,
            country=states.index,  # for consistency
            STATE=states.index,
            STATE_NAME=states.name,
        )


class GasStorage(GasData):
    """
    Creator for underground storage.
    """

    def __init__(self, year: int, interconnect: str, api: str) -> None:
        self.api = api
        super().__init__(year, interconnect)

    def read_data(self):
        base = eia.Storage("gas", "base", self.year, self.api).get_data()
        base["storage_type"] = "base_capacity"
        total = eia.Storage("gas", "total", self.year, self.api).get_data()
        total["storage_type"] = "total_capacity"
        working = eia.Storage("gas", "working", self.year, self.api).get_data()
        working["storage_type"] = "working_capacity"

        final = pd.concat([base, total, working])
        final["value"] = pd.to_numeric(final.value)
        return final

    def format_data(self, data: pd.DataFrame):
        df = data.copy()
        df["value"] = df.value * MWH_2_MMCF
        df = (
            df.reset_index()
            .drop(columns=["period", "series-description", "units"])  # units in MWh_th
            .groupby(["state", "storage_type"])
            .mean()  # get average yearly capacity
            .reset_index()
            .rename(columns={"value": "capacity"})
            .pivot(columns="storage_type", index="state")
        )
        df.columns = df.columns.droplevel(0)
        df.columns.name = ""
        df = df.reset_index()
        df = df.rename(
            columns={
                "base_capacity": "MIN_CAPACITY_MWH",
                "total_capacity": "MAX_CAPACITY_MWH",
                "working_capacity": "WORKING_CAPACITY_MWH",
                "state": "STATE",
            },
        )
        return self.filter_on_interconnect(df, ["U.S."])

    def build_infrastructure(self, n: pypsa.Network, **kwargs):

        df = self.data.copy()
        df.index = df.STATE
        df["state_name"] = df.index.map(self.state_2_name)

        if "gas storage" not in n.carriers.index:
            n.add("Carrier", "gas storage", color="#d35050", nice_name="Gas Storage")

        n.madd(
            "Bus",
            names=df.index,
            suffix=" gas storage",
            carrier="gas storage",
            unit="MWh_th",
        )

        cyclic_storage = kwargs.get("cyclic_storage", True)
        n.madd(
            "Store",
            names=df.index,
            suffix=" gas storage",
            bus=df.index + " gas storage",
            carrier="gas storage",
            e_nom_extendable=False,
            e_nom=df.MAX_CAPACITY_MWH,
            e_cyclic=cyclic_storage,
            e_min_pu=df.MIN_CAPACITY_MWH / df.MAX_CAPACITY_MWH,
            e_initial=df.MAX_CAPACITY_MWH - df.MIN_CAPACITY_MWH,  # same as working
            marginal_cost=0,  # to update
        )

        # must do two links, rather than a bidirectional one, to constrain charge limits
        # Right now, chanrge limits are set at being able to drain the reservoir
        # over one full month
        n.madd(
            "Link",
            names=df.index,
            suffix=" charge gas storage",
            carrier="gas storage",
            bus0=df.index + " gas",
            bus1=df.index + " gas storage",
            p_nom=(df.MAX_CAPACITY_MWH - df.MIN_CAPACITY_MWH) / (30 * 24),
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=False,
            marginal_cost=0,
        )

        n.madd(
            "Link",
            names=df.index,
            suffix=" discharge gas storage",
            carrier="gas storage",
            bus0=df.index + " gas storage",
            bus1=df.index + " gas",
            p_nom=(df.MAX_CAPACITY_MWH - df.MIN_CAPACITY_MWH) / (30 * 24),
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=False,
            marginal_cost=0,
        )


class GasProcessing(GasData):
    """
    Creator for processing capacity.
    """

    def __init__(self, year: int, interconnect: str, api: str) -> None:
        self.api = api
        super().__init__(year=year, interconnect=interconnect)

    def read_data(self) -> pd.DataFrame:
        return eia.Production("gas", "market", self.year, self.api).get_data()

    def format_data(self, data: pd.DataFrame):
        df = data.copy()

        df["value"] = (
            df.value.astype(float) * MWH_2_MMCF / 30 / 24
        )  # get monthly average hourly capacity (based on 30 days / month)
        df = (
            df.reset_index()
            .drop(columns=["period", "series-description", "units"])  # units in MW_th
            .groupby(["state"])
            .max()  # get average yearly capacity
            .reset_index()
            .rename(columns={"state": "STATE", "value": "p_nom"})
        )
        return self.filter_on_interconnect(df, ["U.S."])

    def build_infrastructure(self, n: pypsa.Network, **kwargs):

        df = self.data.copy()
        df = df.set_index("STATE")
        df["bus"] = df.index + " gas"

        capacity_mult = kwargs.get("capacity_multiplier", 1)
        p_nom_extendable = False if capacity_mult == 1 else True
        p_nom_mult = 1 if capacity_mult >= 1 else capacity_mult
        p_nom_max_mult = capacity_mult

        n.madd(
            "Generator",
            names=df.index,
            suffix=" gas production",
            bus=df.bus,
            carrier="gas",
            p_nom_extendable=p_nom_extendable,
            capital_cost=0.01,  # to update
            marginal_costs=0.35,  # https://www.eia.gov/analysis/studies/drilling/pdf/upstream.pdf
            p_nom=df.p_nom * p_nom_mult,
            p_nom_min=0,
            p_nom_max=df.p_nom * p_nom_max_mult,
        )


class _GasPipelineCapacity(GasData):

    def __init__(self, year: int, interconnect: str, xlsx: str) -> None:
        self.xlsx = xlsx
        super().__init__(year, interconnect)

    def read_data(self) -> pd.DataFrame:
        return pd.read_excel(
            self.xlsx,
            sheet_name="Pipeline State2State Capacity",
            skiprows=1,
            index_col=0,
        )

    def format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df.columns = df.columns.str.strip()
        df = df[df.index == int(self.year)]
        df["Capacity (mmcfd)"] = (
            df["Capacity (mmcfd)"] * MWH_2_MMCF / 24
        )  # divide by 24 to get hourly
        df = df.rename(
            columns={
                "State From": "STATE_NAME_FROM",
                "County From": "COUNTRY_FROM",
                "State To": "STATE_NAME_TO",
                "County To": "COUNTRY_TO",
                "Capacity (mmcfd)": "CAPACITY_MW",
            },
        )
        df = (
            df.astype(
                {
                    "STATE_NAME_FROM": "str",
                    "COUNTRY_FROM": "str",
                    "STATE_NAME_TO": "str",
                    "COUNTRY_TO": "str",
                    "CAPACITY_MW": "float",
                },
            )[["STATE_NAME_FROM", "STATE_NAME_TO", "CAPACITY_MW"]]
            .groupby(["STATE_NAME_TO", "STATE_NAME_FROM"])
            .sum()
            .reset_index()
        )

        df = df[
            ~(
                (
                    df.STATE_NAME_TO.isin(
                        ["Gulf of Mexico", "Gulf of Mexico - Deepwater"],
                    )
                )
                | (
                    df.STATE_NAME_FROM.isin(
                        ["Gulf of Mexico", "Gulf of Mexico - Deepwater"],
                    )
                )
            )
        ]

        df = self.assign_pipeline_interconnects(df)
        return self.extract_pipelines(df)

    @abstractmethod
    def build_infrastructure(self, n: pypsa.Network) -> None:
        pass

    @abstractmethod
    def extract_pipelines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts pipelines for that region.

        Used in Format data
        """
        pass

    def assign_pipeline_interconnects(self, df: pd.DataFrame):
        """
        Adds interconnect labels to the pipelines.
        """

        df["STATE_TO"] = df.STATE_NAME_TO.map(self.name_2_state)
        df["STATE_FROM"] = df.STATE_NAME_FROM.map(self.name_2_state)

        df["INTERCONNECT_TO"] = df.STATE_TO.map(self.state_2_interconnect)
        df["INTERCONNECT_FROM"] = df.STATE_FROM.map(self.state_2_interconnect)

        assert not df.isna().any().any()

        return df


class InterconnectGasPipelineCapacity(_GasPipelineCapacity):
    """
    Pipeline capacity within the interconnect.
    """

    def __init__(self, year: int, interconnect: str, xlsx: str) -> None:
        super().__init__(year, interconnect, xlsx)

    def extract_pipelines(self, data: pd.DataFrame) -> pd.DataFrame:

        df = data.copy()
        # for some reason drop duplicates is not wokring here and I cant figure out why :(
        # df = df.drop_duplicates(subset=["STATE_TO", "STATE_FROM"], keep=False).copy()
        df = df[~df.apply(lambda x: x.STATE_TO == x.STATE_FROM, axis=1)].copy()

        if self.interconnect != "usa":
            df = df[
                (df.INTERCONNECT_TO == self.interconnect)
                & (df.INTERCONNECT_FROM == self.interconnect)
            ]
            if df.empty:
                logger.error(
                    f"Empty natural gas domestic pipelines for interconnect {self.interconnect}",
                )
        else:
            df = df[
                ~(
                    df[["INTERCONNECT_TO", "INTERCONNECT_FROM"]].isin(
                        ["canada", "mexico"],
                    )
                ).all(axis=1)
            ]
        return df.reset_index(drop=True)

    def build_infrastructure(self, n: pypsa.Network) -> None:

        df = self.data.copy()

        if "gas pipeline" not in n.carriers.index:
            n.add("Carrier", "gas pipeline", color="#d35050", nice_name="Gas Pipeline")

        df.index = df.STATE_FROM + " " + df.STATE_TO

        n.madd(
            "Link",
            names=df.index,
            suffix=" pipeline",
            carrier="gas pipeline",
            unit="MW",
            bus0=df.STATE_FROM + " gas",
            bus1=df.STATE_TO + " gas",
            p_nom=df.CAPACITY_MW,
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=False,
        )


class TradeGasPipelineCapacity(_GasPipelineCapacity):
    """
    Pipeline capcity connecting to the interconnect.
    """

    def __init__(
        self,
        year: int,
        interconnect: str,
        xlsx: str,
        api: str,
        domestic: bool = True,
    ) -> None:
        self.domestic = domestic
        self.api = api
        super().__init__(year, interconnect, xlsx)

    def extract_pipelines(self, data: pd.DataFrame) -> pd.DataFrame:

        df = data.copy()
        if self.domestic:
            return self._get_domestic_pipeline_connections(df)
        else:
            return self._get_international_pipeline_connections(df)

    def _get_domestic_pipeline_connections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gets all pipelines within the usa that connect to the interconnect.
        """

        if self.interconnect == "usa":
            # no domestic connections
            return pd.DataFrame(columns=df.columns)
        else:
            # get rid of international connections
            df = df[
                ~(
                    (df.INTERCONNECT_TO.isin(["canada", "mexico"]))
                    | (df.INTERCONNECT_FROM.isin(["canada", "mexico"]))
                )
            ]
            # get rid of pipelines within the interconnect
            return df[
                (
                    df["INTERCONNECT_TO"].eq(self.interconnect)
                    | df["INTERCONNECT_FROM"].eq(self.interconnect)
                )
                & ~(
                    df["INTERCONNECT_TO"].eq(self.interconnect)
                    & df["INTERCONNECT_FROM"].eq(self.interconnect)
                )
            ]

    def _get_international_pipeline_connections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gets all international pipeline connections.
        """
        df = df[
            (df.INTERCONNECT_TO.isin(["canada", "mexico"]))
            | (df.INTERCONNECT_FROM.isin(["canada", "mexico"]))
        ]
        if self.interconnect == "usa":
            return df
        else:
            return df[
                (df.INTERCONNECT_TO == self.interconnect)
                | (df.INTERCONNECT_FROM == self.interconnect)
            ]

    def _get_international_costs(
        self,
        direction: str,
        interpoloation_method: str = "zero",
    ) -> pd.DataFrame:
        """
        Gets timeseries of international costs in $/MWh.

        interpolation_method can be one of:
        - linear, zero
        """

        assert direction in ("imports", "exports")

        # fuel costs/profits at a national level
        costs = eia.FuelCosts("gas", direction, self.year, self.api).get_data()

        # fuel costs come in MCF, so first convert to MMCF
        costs = costs[["value"]].astype("float")
        costs = costs / 1000 * MWH_2_MMCF

        return costs.resample("H").asfreq().interpolate(method=interpoloation_method)

    def build_infrastructure(self, n: pypsa.Network) -> None:
        """
        Builds import and export bus+link+store to connect to.

        Dataframe must have a 'STATE_TO', 'STATE_FROM', 'INTERCONNECT_TO', and
        'INTERCONNECT_FROM' columns

        The function does the following
        - exisitng domestic buses are retained
        - new import export buses are created based on region
            - "WA BC gas export"
            - "WA BC gas import"
        - new one way links are added with capacity limits
            - "WA BC gas export"
            - "WA BC gas import"
        - stores are added WITHOUT energy limits
            - "WA BC gas export"
            - "WA BC gas import"
        """

        df = self.data.copy()

        if self.interconnect != "usa":
            to_from = df[df.INTERCONNECT_TO == self.interconnect].copy()  # exports
            from_to = df[df.INTERCONNECT_FROM == self.interconnect].copy()  # imports
        else:
            to_from = df[~df.INTERCONNECT_TO.isin(["canada", "mexico"])].copy()
            from_to = df[~df.INTERCONNECT_FROM.isin(["canada", "mexico"])].copy()

        to_from["NAME"] = to_from.STATE_TO + " " + to_from.STATE_FROM
        from_to["NAME"] = from_to.STATE_FROM + " " + from_to.STATE_TO

        to_from = to_from.set_index("NAME")
        from_to = from_to.set_index("NAME")

        if "gas export" not in n.carriers.index:
            n.add("Carrier", "gas export", color="#d35050", nice_name="Gas Export")

        n.madd(
            "Bus",
            names=to_from.index,
            suffix=" gas export",
            carrier="gas export",
            unit="",
            country=to_from.STATE_TO,
            interconnect=self.interconnect,
        )

        if "gas import" not in n.carriers.index:
            n.add("Carrier", "gas import", color="#d35050", nice_name="Gas Import")

        n.madd(
            "Bus",
            names=from_to.index,
            suffix=" gas import",
            carrier="gas import",
            unit="",
            country=from_to.STATE_FROM,
            interconnect=self.interconnect,
        )

        if not self.domestic:
            export_costs = self._get_international_costs("exports")
            export_costs = export_costs[
                (export_costs.index >= n.snapshots[0])
                & (export_costs.index <= n.snapshots[-1])
            ].copy()
            for link in to_from.index:
                export_costs[link] = export_costs["value"]
            export_costs = export_costs.drop(columns=["value"])
        else:
            export_costs = 0

        n.madd(
            "Link",
            names=to_from.index,
            suffix=" gas export",
            carrier="gas export",
            unit="MW",
            bus0=to_from.STATE_TO + " gas",
            bus1=to_from.index + " gas export",
            p_nom=to_from.CAPACITY_MW,
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=False,
            efficiency=1,  # must be 1 for proper cost accounting
            marginal_cost=export_costs * (-1),  # note the negative value!
        )

        if not self.domestic:
            import_costs = self._get_international_costs("imports")
            import_costs = import_costs[
                (import_costs.index >= n.snapshots[0])
                & (import_costs.index <= n.snapshots[-1])
            ].copy()
            for link in from_to.index:
                import_costs[link] = import_costs["value"]
            import_costs = import_costs.drop(columns=["value"])
        else:
            import_costs = 0

        n.madd(
            "Link",
            names=from_to.index,
            suffix=" gas import",
            carrier="gas import",
            unit="MW",
            bus0=from_to.index + " gas import",
            bus1=from_to.STATE_FROM + " gas",
            p_nom=from_to.CAPACITY_MW,
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=False,
            efficiency=1,  # must be 1 for proper cost accounting
            marginal_cost=import_costs,
        )

        n.madd(
            "Store",
            names=to_from.index,
            suffix=" gas export",
            unit="MWh_th",
            bus=to_from.index + " gas export",
            carrier="gas export",
            e_nom_extendable=True,
            capital_cost=0,
            e_nom=0,
            e_cyclic=False,
            e_cyclic_per_period=False,
            marginal_cost=0,
        )

        n.madd(
            "Store",
            names=from_to.index,
            unit="MWh_th",
            suffix=" gas import",
            bus=from_to.index + " gas import",
            carrier="gas import",
            e_nom_extendable=True,
            capital_cost=0,
            e_nom=0,
            e_cyclic=False,
            e_cyclic_per_period=False,
            marginal_cost=0,
        )


class TradeGasPipelineEnergy(GasData):
    """
    Creator of gas energy limits.
    """

    def __init__(self, year: int, interconnect: str, direction: str, api: str) -> None:
        self.api = api
        self.dir = direction
        super().__init__(year, interconnect)

    def read_data(self) -> pd.DataFrame:
        assert self.dir in ("imports", "exports")
        return eia.Trade("gas", self.dir, self.year, self.api).get_data()

    def format_data(self, n: pypsa.Network) -> None:
        """
        Adds international import/export limits.
        """
        df = self.data.copy()
        df["value"] = df.value.astype("float")
        return (
            df.drop(columns=["series-description", "units"])
            .reset_index()
            .groupby(["period", "state"])
            .sum()
            .reset_index()
        )

    def build_infrastructure(self, n: Network) -> None:
        pass


class PipelineLinepack(GasData):
    """
    Creator for linepack infrastructure.
    """

    def __init__(
        self,
        year: int,
        interconnect: str,
        counties: str,
        pipelines: str,
    ) -> None:
        self.counties = StateGeometry(counties)
        self.states = self.counties.states
        self.pipeline_geojson = pipelines
        super().__init__(year, interconnect)

    def read_data(self) -> gpd.GeoDataFrame:
        """https://atlas.eia.gov/apps/3652f0f1860d45beb0fed27dc8a6fc8d/explore"""
        return gpd.read_file(self.pipeline_geojson)

    def format_data(self, data: gpd.GeoDataFrame) -> pd.DataFrame:
        gdf = data.copy()
        states = self.states.copy()

        length_in_state = gpd.sjoin(
            gdf.to_crs("4269"),
            states,
            how="right",
            predicate="within",
        ).reset_index()
        length_in_state = (
            length_in_state[
                ["STATE_NAME", "STATE", "TYPEPIPE", "Shape_Leng", "Shape__Length"]
            ]
            .rename(columns={"Shape_Leng": "LENGTH_DEG", "Shape__Length": "LENGTH_M"})
            .groupby(by=["STATE_NAME", "STATE", "TYPEPIPE"])
            .sum()
            .reset_index()
        )

        # https://publications.anl.gov/anlpubs/2008/02/61034.pdf
        intrastate_radius = 12 * 0.0254  # inches in meters (24in dia)
        interstate_radius = 18 * 0.0254  # inches meters (36in dia)

        volumne_in_state = length_in_state.copy()
        volumne_in_state["RADIUS"] = volumne_in_state.TYPEPIPE.map(
            lambda x: interstate_radius if x == "Interstate" else intrastate_radius,
        )
        volumne_in_state["VOLUME_M3"] = (
            volumne_in_state.LENGTH_M * pi * volumne_in_state.RADIUS**2
        )
        volumne_in_state = volumne_in_state[["STATE_NAME", "STATE", "VOLUME_M3"]]
        volumne_in_state = volumne_in_state.groupby(by=["STATE_NAME", "STATE"]).sum()

        # https://publications.anl.gov/anlpubs/2008/02/61034.pdf
        max_pressure = 8000  # kPa
        min_pressure = 4000  # kPa

        energy_in_state = volumne_in_state.copy()
        energy_in_state["MAX_ENERGY_kJ"] = energy_in_state.VOLUME_M3 * max_pressure
        energy_in_state["MIN_ENERGY_kJ"] = energy_in_state.VOLUME_M3 * min_pressure
        energy_in_state["NOMINAL_ENERGY_kJ"] = (
            energy_in_state.MAX_ENERGY_kJ + energy_in_state.MIN_ENERGY_kJ
        ) / 2

        final = energy_in_state.copy()
        final["MAX_ENERGY_MWh"] = final.MAX_ENERGY_kJ * KJ_2_MWH
        final["MIN_ENERGY_MWh"] = final.MIN_ENERGY_kJ * KJ_2_MWH
        final["NOMINAL_ENERGY_MWh"] = final.NOMINAL_ENERGY_kJ * KJ_2_MWH

        final = final[
            ["MAX_ENERGY_MWh", "MIN_ENERGY_MWh", "NOMINAL_ENERGY_MWh"]
        ].reset_index()
        return self.filter_on_interconnect(final)

    def build_infrastructure(self, n: pypsa.Network, **kwargs) -> None:

        df = self.data.copy()
        df = df.set_index("STATE")

        if "gas pipeline" not in n.carriers.index:
            n.add("Carrier", "gas pipeline", color="#d35050", nice_name="Gas Pipeline")

        cyclic_storage = kwargs.get("cyclic_storage", True)

        n.madd(
            "Store",
            names=df.index,
            unit="MWh_th",
            suffix=" linepack",
            bus=df.index + " gas",
            carrier="gas pipeline",
            e_nom=df.MAX_ENERGY_MWh,
            e_nom_extendable=False,
            e_nom_min=0,
            e_nom_max=np.inf,
            e_min_pu=df.MIN_ENERGY_MWh / df.MAX_ENERGY_MWh,
            e_max_pu=1,
            e_initial=df.NOMINAL_ENERGY_MWh,
            e_initial_per_period=False,
            e_cyclic=cyclic_storage,
            e_cyclic_per_period=True,
            p_set=0,
            marginal_cost=0,
            capital_cost=1,
            standing_loss=0,
            lifetime=np.inf,
        )


class ImportExportLimits(GasData):
    """
    Adds constraints for import export limits.
    """

    def __init__(self, year: int, interconnect: str, api: str) -> None:
        self.api = api
        super().__init__(year, interconnect)

    def read_data(self) -> pd.DataFrame:
        imports = eia.Trade("gas", "imports", self.year, self.api).get_data()
        exports = eia.Trade("gas", "exports", self.year, self.api).get_data()
        return pd.concat([imports, exports])

    def format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = (
            data.reset_index()
            .drop(columns=["series-description"])
            .groupby(["period", "units", "state"])
            .sum()
            .reset_index()
            .rename(columns={"state": "STATE"})
            .copy()
        )
        # may need to add ["U.S."] to states to remove here
        return self.filter_on_interconnect(df)

    def build_infrastructure(self, n: pypsa.Network) -> None:
        pass


def convert_generators_2_links(n: pypsa.Network, carrier: str):
    """
    Replace Generators with cross sector links.

    Links bus1 are the bus the generator is attached to. Links bus0 are state
    level followed by the suffix (ie. "WA gas" if " gas" is the bus0_suffix)

    n: pypsa.Network,
    carrier: str,
        carrier of the generator to convert to a link
    bus0_suffix: str,
        suffix to attach link to
    """

    plants = n.generators[n.generators.carrier == carrier].copy()
    plants["STATE"] = plants.bus.map(n.buses.STATE)

    pnl = {}

    # copy over pnl parameters
    for c in n.iterate_components(["Generator"]):
        for param, df in c.pnl.items():
            # skip result vars
            if param not in (
                "p_min_pu",
                "p_max_pu",
                "p_set",
                "q_set",
                "marginal_cost",
                "marginal_cost_quadratic",
                "efficiency",
                "stand_by_cost",
            ):
                continue
            cols = [p for p in plants.index if p in df.columns]
            if cols:
                pnl[param] = df[cols]

    n.madd(
        "Link",
        names=plants.index,
        bus0=plants.STATE + " gas",
        bus1=plants.bus,
        carrier=plants.carrier,
        p_nom_min=plants.p_nom_min / plants.efficiency,
        p_nom=plants.p_nom / plants.efficiency,  # links rated on input capacity
        p_nom_max=plants.p_nom_max / plants.efficiency,
        p_nom_extendable=plants.p_nom_extendable,
        ramp_limit_up=plants.ramp_limit_up,
        ramp_limit_down=plants.ramp_limit_down,
        efficiency=plants.efficiency,
        marginal_cost=plants.marginal_cost
        * plants.efficiency,  # fuel costs rated at delievered
        capital_cost=plants.capital_cost
        * plants.efficiency,  # links rated on input capacity
        lifetime=plants.lifetime,
    )

    for param, df in pnl.items():
        n.links_t[param] = n.links_t[param].join(df, how="inner")

    # remove generators
    n.mremove("Generator", plants.index)


###
# MAIN FUNCTION TO EXECUTE
###


def build_natural_gas(
    n: pypsa.Network,
    year: int,
    api: str,
    interconnect: str = "western",
    county_path: str = "../data/counties/cb_2020_us_county_500k.shp",
    pipelines_path: str = "../data/natural_gas/EIA-StatetoStateCapacity_Jan2023.xlsx",
    pipeline_shape_path: str = "../data/natural_gas/pipelines.geojson",
    **kwargs,
) -> None:

    cyclic_storage = kwargs.get("cyclic_storage", True)

    # add gas carrier

    n.add("Carrier", "gas", color="#d35050", nice_name="Natural Gas")

    # add state level gas buses

    buses = GasBuses(interconnect, county_path)
    buses.build_infrastructure(n)

    # add state level natural gas processing facilities

    production = GasProcessing(year, interconnect, api)
    production.build_infrastructure(n, capacity_multiplier=1)

    # add state level gas storage facilities

    storage = GasStorage(year, interconnect, api)
    storage.build_infrastructure(n, cyclic_storage=cyclic_storage)

    # add interconnect pipelines

    pipelines = InterconnectGasPipelineCapacity(year, interconnect, pipelines_path)
    pipelines.build_infrastructure(n)

    # add pipelines for imports/exports
    # TODO: have trade pipelines share data to only instantiate once

    pipelines_domestic = TradeGasPipelineCapacity(
        year,
        interconnect,
        pipelines_path,
        api,
        domestic=True,
    )
    pipelines_domestic.build_infrastructure(n)
    pipelines_international = TradeGasPipelineCapacity(
        year,
        interconnect,
        pipelines_path,
        api,
        domestic=False,
    )
    pipelines_international.build_infrastructure(n)

    # import_energy_constraints = TradeGasPipelineEnergy(year, interconnect, "imports", api)
    # import_energy_constraints.build_infrastructure(n)
    # export_energyconstraints = TradeGasPipelineEnergy(year, interconnect, "exports", api)
    # export_energyconstraints.build_infrastructure(n)

    # add pipeline linepack

    linepack = PipelineLinepack(year, interconnect, county_path, pipeline_shape_path)
    linepack.build_infrastructure(n, cyclic_storage=cyclic_storage)

    # convert existing generators to cross-sector links
    for carrier in ("CCGT", "OCGT"):
        convert_generators_2_links(n, carrier)


if __name__ == "__main__":

    n = pypsa.Network("../resources/western/elec_s_40_ec_lv1.25_Co2L1.25.nc")
    year = 2019
    with open("./../config/config.api.yaml") as file:
        yaml_data = yaml.safe_load(file)
    api = yaml_data["api"]["eia"]
    build_natural_gas(n=n, year=year, api=api)
