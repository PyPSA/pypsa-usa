"""
Module for adding the gas sector.

This module will add a state level copperplate natural gas network to the model.
Specifically, it will do the following

- Creates capacity constrained pipelines between state gas buses (links)
- Creates capacity constraind natural gas processing facilites (generators)
- Creates capacity and energy constrainted underground gas storage facilities
- Creates energy constrained linepack storage (storage units)
- Creates capacity constrained pipelines to states neighbouring the interconnect
- Creates capacity and energy constrained import/exports to international connections
- Adds import/export historical natural gas prices
"""

import logging

import geopandas as gpd
import pandas as pd
import pypsa
from constants import NG_MWH_2_MMCF, STATE_2_CODE, STATES_INTERCONNECT_MAPPER
from pypsa.components import Network

logger = logging.getLogger(__name__)
from abc import ABC, abstractmethod
from math import pi
from typing import Any, Optional

import eia
import numpy as np
import yaml

###
# Constants
###

# for converting everthing into MWh_th
MWH_2_MMCF = NG_MWH_2_MMCF
KJ_2_MWH = (1 / 1000) * (1 / 3600)

CODE_2_STATE = {v: k for k, v in STATE_2_CODE.items()}

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

    state_2_interconnect = STATES_INTERCONNECT_MAPPER
    state_2_name = CODE_2_STATE
    name_2_state = STATE_2_CODE
    states_2_remove = [x for x, y in STATES_INTERCONNECT_MAPPER.items() if not y]

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
    def read_data(self) -> pd.DataFrame | gpd.GeoDataFrame:
        pass

    @abstractmethod
    def format_data(self, data: pd.DataFrame | gpd.GeoDataFrame) -> pd.DataFrame:
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
            logger.debug(
                "Natual gas data not filtered due to incorrect data formatting",
            )
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

    @abstractmethod
    def filter_on_sate(
        self,
        n: pypsa.Network,
        df: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Called before adding infrastructure to check if only modelling a subset
        of interconnect.
        """
        pass


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

    def filter_on_sate(
        self,
        n: pypsa.Network,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        states_in_model = n.buses[
            ~n.buses.carrier.isin(
                ["gas storage", "gas trade", "gas pipeline"],
            )
        ].reeds_state.unique()

        if "STATE" not in df.columns:
            logger.debug(
                "Natual gas data not filtered due to incorrect data formatting",
            )
            return df

        df = df[df.STATE.isin(states_in_model)].copy()

        return df

    def build_infrastructure(self, n: Network) -> None:

        df = self.filter_on_sate(n, self.data)

        states = df.set_index("STATE")

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
        # REVIST THIS
        # start storage a 2/3 full
        df["e_initial"] = df.MIN_CAPACITY_MWH + (df.MAX_CAPACITY_MWH - df.MIN_CAPACITY_MWH).div(1.5)

        return self.filter_on_interconnect(df, ["U.S."])

    def filter_on_sate(
        self,
        n: pypsa.Network,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        states_in_model = n.buses[
            ~n.buses.carrier.isin(
                ["gas storage", "gas trade", "gas pipeline"],
            )
        ].reeds_state.unique()

        if "STATE" not in df.columns:
            logger.debug(
                "Natual gas data not filtered due to incorrect data formatting",
            )
            return df

        df = df[df.STATE.isin(states_in_model)].copy()

        return df

    def build_infrastructure(self, n: pypsa.Network, **kwargs):

        df = self.filter_on_sate(n, self.data)
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
            interconnect=self.interconnect,
            country=df.index,
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
            # e_initial=df.MAX_CAPACITY_MWH - df.MIN_CAPACITY_MWH,
            e_initial=df.e_initial,
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

    def filter_on_sate(
        self,
        n: pypsa.Network,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        states_in_model = n.buses[
            ~n.buses.carrier.isin(
                ["gas storage", "gas trade", "gas pipeline"],
            )
        ].reeds_state.unique()

        if "STATE" not in df.columns:
            logger.debug(
                "Natual gas data not filtered due to incorrect data formatting",
            )
            return df

        df = df[df.STATE.isin(states_in_model)].copy()

        return df

    def build_infrastructure(self, n: pypsa.Network, **kwargs):

        df = self.filter_on_sate(n, self.data)
        df = df.set_index("STATE")
        df["bus"] = df.index + " gas"

        capacity_mult = kwargs.get("capacity_multiplier", 1)
        p_nom_extendable = False if capacity_mult == 1 else True
        p_nom_mult = 1 if capacity_mult >= 1 else capacity_mult
        p_nom_max_mult = capacity_mult

        if "gas production" not in n.carriers.index:
            n.add(
                "Carrier",
                "gas production",
                color="#d35050",
                nice_name="Gas Production",
            )

        n.madd(
            "Bus",
            names=df.index,
            suffix=" gas production",
            carrier="gas production",
            unit="MWh_th",
            country=df.index,
            interconnect=self.interconnect,
            STATE=df.index,
        )

        # marginal cost
        # https://www.aer.ca/providing-information/data-and-reports/statistical-reports/st98/natural-gas/supply-costs
        # Table S5.6 with Variable Operating Cost average of ~$63 CAD/ 1000 m3
        # (63 CAD/ 1000 m3) (1 m3 / 35.5 CF) (1,000,000 CF / MMCF) (1 MMCF / 303.5 MWH) (1 USD / 0.75 CAD)
        # ~7.5 $/MWh

        n.madd(
            "Link",
            names=df.index,
            suffix=" gas production",
            carrier="gas production",
            unit="MW",
            bus0=df.index + " gas production",
            bus1=df.index + " gas",
            efficiency=1,
            p_nom_extendable=p_nom_extendable,
            capital_cost=0.01,  # to update
            marginal_cost=7.5,
            p_nom=(df.p_nom * p_nom_mult).round(2),
            p_nom_min=0,
            p_nom_max=df.p_nom * p_nom_max_mult,
        )

        n.madd(
            "Store",
            names=df.index,
            unit="MWh",
            suffix=" gas production",
            bus=df.index + " gas production",
            carrier="gas production",
            capital_cost=0,
            marginal_cost=0,
            e_cyclic=False,
            e_cyclic_per_period=False,
            e_nom=0,
            e_nom_extendable=True,
            e_nom_min=0,
            e_nom_max=np.inf,
            e_min_pu=-1,
            e_max_pu=0,
        )


class _GasPipelineCapacity(GasData):

    def __init__(
        self,
        year: int,
        interconnect: str,
        xlsx: str,
        api: Optional[str] = None,
    ) -> None:
        self.xlsx = xlsx
        self.api = api
        super().__init__(year, interconnect)

    def read_data(self) -> pd.DataFrame:
        return pd.read_excel(
            self.xlsx,
            sheet_name="Pipeline State2State Capacity",
            skiprows=1,
            index_col=0,
        )

    def get_states_in_model(self, n: pypsa.Network) -> list[str]:
        return n.buses[
            ~n.buses.carrier.isin(
                ["gas storage", "gas trade", "gas pipeline"],
            )
        ].reeds_state.unique()

    def filter_on_sate(
        self,
        n: pypsa.Network,
        df: pd.DataFrame,
        in_spatial_scope: bool,
    ) -> pd.DataFrame:

        states_in_model = self.get_states_in_model(n)

        if ("STATE_TO" and "STATE_FROM") not in df.columns:
            logger.debug(
                "Natual gas data not filtered due to incorrect data formatting",
            )
            return df

        if in_spatial_scope:
            df = df[(df.STATE_TO.isin(states_in_model)) & (df.STATE_FROM.isin(states_in_model))].copy()
        else:
            df = df[(df.STATE_TO.isin(states_in_model)) | (df.STATE_FROM.isin(states_in_model))].copy()

        return df[~(df.STATE_TO == df.STATE_FROM)].copy()

    def format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df.columns = df.columns.str.strip()
        df = df[df.index == int(self.year)]
        df["Capacity (mmcfd)"] = df["Capacity (mmcfd)"] * MWH_2_MMCF / 24  # divide by 24 to get hourly
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

        if self.api:
            trade = self._get_capacity_based_on_trade_flows()
            df = self._merge_capacity_trade_data(df, trade)

        # slight buffer for when building constraints
        df["CAPACITY_MW"] = np.ceil(df["CAPACITY_MW"].mul(1.02))

        return self.extract_pipelines(df)

    def _get_capacity_based_on_trade_flows(self) -> pd.DataFrame:
        """Check that trade flows do not exceed design capacity

        See Issue #487
        https://github.com/PyPSA/pypsa-usa/issues/487
        """
        df = pd.concat(
            [
                eia.Trade("gas", False, "exports", self.year, self.api).get_data(),
                eia.Trade("gas", True, "exports", self.year, self.api).get_data(),
            ],
        )
        df["STATE_FROM"] = df.state.map(lambda x: x.split("-")[0])
        df["STATE_TO"] = df.state.map(lambda x: x.split("-")[1])
        df["CAPACITY_MW"] = df.value.mul(MWH_2_MMCF).div(365).div(24)  # MMCF/year -> MW
        df = df.reset_index(drop=True).drop(
            columns=["series-description", "value", "units", "state"],
        )
        df["STATE_NAME_TO"] = df.STATE_TO.map(self.state_2_name)
        df["STATE_NAME_FROM"] = df.STATE_FROM.map(self.state_2_name)
        df["INTERCONNECT_TO"] = df.STATE_TO.map(self.state_2_interconnect)
        df["INTERCONNECT_FROM"] = df.STATE_FROM.map(self.state_2_interconnect)

        return df

    def _merge_capacity_trade_data(
        self,
        capacity: pd.DataFrame,
        trade: pd.DataFrame,
    ) -> pd.DataFrame:

        df = pd.concat([capacity, trade])
        df = df.sort_values(by="CAPACITY_MW", ascending=False)
        df = df.drop_duplicates(
            subset=[
                "STATE_NAME_TO",
                "STATE_NAME_FROM",
                "STATE_TO",
                "STATE_FROM",
                "INTERCONNECT_TO",
                "INTERCONNECT_FROM",
            ],
            keep="first",
        )

        return df.sort_values("STATE_NAME_TO")

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

    def __init__(
        self,
        year: int,
        interconnect: str,
        xlsx: str,
        api: Optional[str] = None,
    ) -> None:
        super().__init__(year, interconnect, xlsx, api)

    def extract_pipelines(self, data: pd.DataFrame) -> pd.DataFrame:

        df = data.copy()
        # for some reason drop duplicates is not wokring here and I cant figure out why :(
        # df = df.drop_duplicates(subset=["STATE_TO", "STATE_FROM"], keep=False).copy()
        df = df[~df.apply(lambda x: x.STATE_TO == x.STATE_FROM, axis=1)].copy()

        if self.interconnect != "usa":
            df = df[(df.INTERCONNECT_TO == self.interconnect) & (df.INTERCONNECT_FROM == self.interconnect)]
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

        df = self.filter_on_sate(n, self.data, in_spatial_scope=True)

        if df.empty:
            # happens for single state models
            logger.info("No gas pipelines added within interconnect")
            return

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
        super().__init__(year, interconnect, xlsx, api)

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

        # get rid of international connections
        df = df[~((df.INTERCONNECT_TO.isin(["canada", "mexico"])) | (df.INTERCONNECT_FROM.isin(["canada", "mexico"])))]

        if self.interconnect == "usa":
            return df
        else:
            # get rid of pipelines that exist only in other interconnects
            return df[df["INTERCONNECT_TO"].eq(self.interconnect) | df["INTERCONNECT_FROM"].eq(self.interconnect)]

    def _get_international_pipeline_connections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gets all international pipeline connections.
        """
        df = df[(df.INTERCONNECT_TO.isin(["canada", "mexico"])) | (df.INTERCONNECT_FROM.isin(["canada", "mexico"]))]
        if self.interconnect == "usa":
            return df
        else:
            return df[(df.INTERCONNECT_TO == self.interconnect) | (df.INTERCONNECT_FROM == self.interconnect)]

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
        costs = eia.FuelCosts("gas", self.year, self.api, industry=direction).get_data()

        # fuel costs come in MCF, so first convert to MMCF
        costs = costs[["value"]].astype("float")
        costs = costs / 1000 * MWH_2_MMCF

        return costs.resample("1h").asfreq().interpolate(method=interpoloation_method)

    def _expand_costs(self, n: pypsa.Network, costs: pd.DataFrame) -> pd.DataFrame:
        """
        Expands import/export costs over snapshots and investment periods.
        """
        expanded_costs = []
        for invesetment_period in n.investment_periods:
            # reindex to match any tsa
            cost = costs.copy()
            cost.index = cost.index.map(lambda x: x.replace(year=invesetment_period))
            cost = cost.reindex(n.snapshots.get_level_values(1), method="nearest")
            # set investment periods
            # https://stackoverflow.com/a/56278736/14961492
            old_idx = cost.index.to_frame()
            old_idx.insert(0, "period", invesetment_period)
            cost.index = pd.MultiIndex.from_frame(old_idx)
            expanded_costs.append(cost)
        return pd.concat(expanded_costs)

    def _add_zero_capacity_connections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Will add a zero capacity link if a connection is missing due to no
        capacity.

        For example, the input data frame of...

        |   | STATE_NAME_TO    | STATE_NAME_FROM  | CAPACITY_MW | STATE_TO | STATE_FROM | INTERCONNECT_TO | INTERCONNECT_FROM |
        |---|------------------|------------------|-------------|----------|------------|-----------------|-------------------|
        | 0 | British Columbia | Washington       | 100         | BC       | WA         | canada          | western           |
        | 1 | Idaho            | British Columbia | 50          | ID       | BC         | western         | canada            |
        | 2 | Washington       | British Columbia | 120         | WA       | BC         | western         | canada            |

        Will get converted to...

        |   | STATE_NAME_TO    | STATE_NAME_FROM  | CAPACITY_MW | STATE_TO | STATE_FROM | INTERCONNECT_TO | INTERCONNECT_FROM |
        |---|------------------|------------------|-------------|----------|------------|-----------------|-------------------|
        | 0 | British Columbia | Washington       | 100         | BC       | WA         | canada          | western           |
        | 1 | Idaho            | British Columbia | 50          | ID       | BC         | western         | canada            |
        | 2 | Washington       | British Columbia | 120         | WA       | BC         | western         | canada            |
        | 3 | British Columbia | Idaho            | 0           | BC       | ID         | canada          | western           |
        """

        @staticmethod
        def missing_connections(df: pd.DataFrame) -> list[tuple[str, str]]:
            connections = set(
                map(tuple, df[["STATE_NAME_TO", "STATE_NAME_FROM"]].values),
            )
            missing_connections = []

            for conn in connections:
                reverse_conn = (conn[1], conn[0])
                if reverse_conn not in connections:
                    missing_connections.append(reverse_conn)

            return missing_connections

        connections = missing_connections(df)

        if not connections:
            return df

        state_2_code = df.set_index("STATE_NAME_TO")["STATE_TO"].to_dict()
        state_2_code.update(df.set_index("STATE_NAME_FROM")["STATE_FROM"].to_dict())

        state_2_interconnect = df.set_index("STATE_NAME_TO")["INTERCONNECT_TO"].to_dict()
        state_2_interconnect.update(
            df.set_index("STATE_NAME_FROM")["INTERCONNECT_FROM"].to_dict(),
        )

        zero_capacity = []
        for connection in connections:
            zero_capacity.append(
                [
                    connection[0],
                    connection[1],
                    0,
                    state_2_code[connection[0]],
                    state_2_code[connection[1]],
                    state_2_interconnect[connection[0]],
                    state_2_interconnect[connection[1]],
                ],
            )

        zero_df = pd.DataFrame(zero_capacity, columns=df.columns)

        return pd.concat([df, zero_df])

    def _get_marginal_costs(
        self,
        n: pypsa.Network,
        connections: pd.DataFrame,
        imports: bool,
    ) -> pd.DataFrame:
        """
        Gets time varrying import/export costs.
        """

        df = connections.copy()

        states_in_model = self.get_states_in_model(n)

        if imports:
            costs = self._get_international_costs("imports")
            df = df[df.STATE_TO.isin(states_in_model)]
        else:
            # multiple by -1 cause exporting makes money
            costs = self._get_international_costs("exports").mul(-1)
            df = df[df.STATE_FROM.isin(states_in_model)]

        for link in df.index:
            costs[link] = costs["value"]

        return costs.drop(columns=["value"])

    def _assign_country(self, n: pypsa.Network, template: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns country column.

        Country is always in model spatial scope.
        """

        df = template.copy()
        states_in_model = self.get_states_in_model(n)

        df["COUNTRY"] = np.where(
            df.STATE_FROM.isin(states_in_model),
            df.STATE_FROM,
            df.STATE_TO,
        )

        return df

    def _assign_link_buses(
        self,
        n: pypsa.Network,
        template: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Assigns bus names for links.
        """

        def assign_bus0_name(row) -> str:
            if row["STATE_FROM"] in states_in_model:
                return f"{row['STATE_FROM']} gas"
            else:
                return f"{row['STATE_FROM']} {row['STATE_TO']} gas trade"

        def assign_bus1_name(row) -> str:
            if row["STATE_TO"] in states_in_model:
                return f"{row['STATE_TO']} gas"
            else:
                return f"{row['STATE_FROM']} {row['STATE_TO']} gas trade"

        df = template.copy()
        states_in_model = self.get_states_in_model(n)

        df["bus0"] = df.apply(assign_bus0_name, axis=1)
        df["bus1"] = df.apply(assign_bus1_name, axis=1)

        return df

    def _assign_stores(self, template: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns if the associated store should be sink or source.

        If bus0 is a trading bus, energy will flow into the model (ie.
        imports) If bus0 is a state gas bus, energy will flow out of the
        model (ie. exports)
        """

        df = template.copy()

        df["store"] = df.bus0.map(
            lambda x: "import" if x.endswith(" trade") else "export",
        )

        return df

    def build_infrastructure(self, n: pypsa.Network) -> None:
        """
        Builds import and export bus+link+store to connect to.

        Dataframe must have a 'STATE_TO', 'STATE_FROM', 'INTERCONNECT_TO', and
        'INTERCONNECT_FROM' columns

        The function does the following
        - exisitng domestic buses are retained
        - new import export buses are created based on region
            - "WA BC gas trade"
            - "BC WA gas trade"
        - new one way links are added with capacity limits
            - "WA BC gas trade"
            - "BC WA gas trade"
        - stores are added WITHOUT energy limits
            - "WA BC gas trade"
            - "BC WA gas trade"
        """

        df = self.filter_on_sate(n, self.data, in_spatial_scope=False)

        df = self._add_zero_capacity_connections(df)

        template = df.copy()
        template["NAME"] = template.STATE_FROM + " " + template.STATE_TO
        template = template.set_index("NAME")

        template = self._assign_country(n, template)
        template = self._assign_link_buses(n, template)
        template = self._assign_stores(template)

        # remove any conections within geographic scope
        if self.domestic:
            template = template[
                ~(template.STATE_TO.isin(n.buses.reeds_state) & template.STATE_FROM.isin(n.buses.reeds_state))
            ]

        store_imports = template[template.store == "import"].copy()
        store_exports = template[template.store == "export"].copy()

        if not self.domestic:
            import_costs = self._get_marginal_costs(n, template, True)
            export_costs = self._get_marginal_costs(n, template, False)
            marginal_cost = pd.concat([import_costs, export_costs], axis=1)
            marginal_cost = self._expand_costs(n, marginal_cost)
        else:
            marginal_cost = 0

        if "gas trade" not in n.carriers.index:
            n.add("Carrier", "gas trade", color="#d35050", nice_name="Gas Trade")

        n.madd(
            "Bus",
            names=template.index,
            suffix=" gas trade",
            carrier="gas trade",
            unit="MWh",
            country=template.COUNTRY,
            interconnect=self.interconnect,
        )

        n.madd(
            "Link",
            names=template.index,
            suffix=" gas trade",
            carrier="gas trade",
            unit="MW",
            bus0=template.bus0,
            bus1=template.bus1,
            p_nom=template.CAPACITY_MW,
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=False,
            efficiency=1,  # must be 1 for proper cost accounting
            marginal_cost=marginal_cost,
        )

        n.madd(
            "Store",
            names=store_exports.index,
            suffix=" gas trade",
            unit="MWh",
            bus=store_exports.bus1,
            carrier="gas trade",
            capital_cost=0,
            marginal_cost=0,
            e_cyclic=False,
            e_cyclic_per_period=False,
            e_nom=0,
            e_nom_extendable=True,
            e_nom_min=0,
            e_nom_max=np.inf,
            e_min_pu=0,
            e_max_pu=1,
        )

        n.madd(
            "Store",
            names=store_imports.index,
            unit="MWh",
            suffix=" gas trade",
            bus=store_imports.bus0,
            carrier="gas trade",
            capital_cost=0,
            marginal_cost=0,
            e_cyclic=False,
            e_cyclic_per_period=False,
            e_nom=0,
            e_nom_extendable=True,
            e_nom_min=0,
            e_nom_max=np.inf,
            e_min_pu=-1,  # minus 1 for energy addition!
            e_max_pu=0,
        )


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

    def filter_on_sate(
        self,
        n: pypsa.Network,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        states_in_model = n.buses[
            ~n.buses.carrier.isin(
                ["gas storage", "gas trade", "gas pipeline"],
            )
        ].reeds_state.unique()

        if "STATE" not in df.columns:
            logger.debug(
                "Natual gas data not filtered due to incorrect data formatting",
            )
            return df

        df = df[df.STATE.isin(states_in_model)].copy()

        return df

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
            length_in_state[["STATE_NAME", "STATE", "TYPEPIPE", "Shape_Leng", "Shape__Length"]]
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
        volumne_in_state["VOLUME_M3"] = volumne_in_state.LENGTH_M * pi * volumne_in_state.RADIUS**2
        volumne_in_state = volumne_in_state[["STATE_NAME", "STATE", "VOLUME_M3"]]
        volumne_in_state = volumne_in_state.groupby(by=["STATE_NAME", "STATE"]).sum()

        # https://publications.anl.gov/anlpubs/2008/02/61034.pdf
        max_pressure = 8000  # kPa
        min_pressure = 4000  # kPa

        energy_in_state = volumne_in_state.copy()
        energy_in_state["MAX_ENERGY_kJ"] = energy_in_state.VOLUME_M3 * max_pressure
        energy_in_state["MIN_ENERGY_kJ"] = energy_in_state.VOLUME_M3 * min_pressure
        energy_in_state["NOMINAL_ENERGY_kJ"] = (energy_in_state.MAX_ENERGY_kJ + energy_in_state.MIN_ENERGY_kJ) / 2

        final = energy_in_state.copy()
        final["MAX_ENERGY_MWh"] = final.MAX_ENERGY_kJ * KJ_2_MWH
        final["MIN_ENERGY_MWh"] = final.MIN_ENERGY_kJ * KJ_2_MWH
        final["NOMINAL_ENERGY_MWh"] = final.NOMINAL_ENERGY_kJ * KJ_2_MWH

        final = final[["MAX_ENERGY_MWh", "MIN_ENERGY_MWh", "NOMINAL_ENERGY_MWh"]].reset_index()
        return self.filter_on_interconnect(final)

    def build_infrastructure(self, n: pypsa.Network, **kwargs) -> None:

        df = self.filter_on_sate(n, self.data)
        df = df.set_index("STATE")

        if "gas pipeline" not in n.carriers.index:
            n.add("Carrier", "gas pipeline", color="#d35050", nice_name="Gas Pipeline")

        cyclic_storage = kwargs.get("cyclic_storage", True)
        standing_loss = kwargs.get("standing_loss", 0)

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
            standing_loss=standing_loss,
            lifetime=np.inf,
        )


def _remove_marginal_costs(n: pypsa.Network):
    """
    Removes marginal costs of CCGT and OCGT plants.
    """

    links = n.links[n.links.carrier.str.contains("CCGT") | n.links.carrier.str.contains("OCGT")].index

    n.links.loc[links, "marginal_cost"] = 0


###
# MAIN FUNCTION TO EXECUTE
###


def build_natural_gas(
    n: pypsa.Network,
    year: int,
    api: str,
    interconnect: str = "western",
    county_path: str = "../data/counties/cb_2020_us_county_500k.shp",
    pipelines_path: str = "../data/natural_gas/EIA-StatetoStateCapacity_Feb2024.xlsx",
    pipeline_shape_path: str = "../data/natural_gas/pipelines.geojson",
    options: Optional[dict[str, Any]] = None,
    **kwargs,
) -> None:

    if not options:
        options = {}

    cyclic_storage = options.get("cyclic_storage", True)
    standing_loss = options.get("standing_loss", 0)

    # add state level natural gas processing facilities

    production = GasProcessing(year, interconnect, api)
    production.build_infrastructure(n, capacity_multiplier=1)

    # add state level gas storage facilities

    storage = GasStorage(year, interconnect, api)
    storage.build_infrastructure(n, cyclic_storage=cyclic_storage)

    # add interconnect pipelines

    pipelines = InterconnectGasPipelineCapacity(year, interconnect, pipelines_path, api)
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

    # add pipeline linepack

    linepack = PipelineLinepack(year, interconnect, county_path, pipeline_shape_path)
    linepack.build_infrastructure(
        n,
        cyclic_storage=cyclic_storage,
        standing_loss=standing_loss,
    )

    _remove_marginal_costs(n)


if __name__ == "__main__":
    n = pypsa.Network("../resources/Washington/western/elec_s10_c4m_ec_lv1.0_3h.nc")
    year = 2018
    with open("./../config/config.api.yaml") as file:
        yaml_data = yaml.safe_load(file)
    api = yaml_data["api"]["eia"]

    pipelines = InterconnectGasPipelineCapacity(
        year,
        "western",
        "../data/natural_gas/EIA-StatetoStateCapacity_Feb2024.xlsx",
        api,
    )
    # pipelines.build_infrastructure(n)

    build_natural_gas(n=n, year=year, api=api)
