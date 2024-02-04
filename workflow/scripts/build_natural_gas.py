"""Module for adding the gas sector"""

import pypsa 
import pandas as pd
import geopandas as gpd
from pypsa.components import Network
import constants
import logging
logger = logging.getLogger(__name__)
from typing import List, Dict, Union
from math import pi
import numpy as np
import yaml

import eia 

from abc import ABC, abstractmethod


###
# Geolocation of Assets class 
###

class StateGeometry:
    """Holds state boundry data"""
    
    def __init__(self, shapefile: str) -> None:
        """Counties shapefile"""
        self._counties = gpd.read_file(shapefile)
        self._state_center_points = None
        self._states = None
        
    @property
    def counties(self) -> gpd.GeoDataFrame:
        """Spatially resolved counties"""
        return self._counties
    
    @property
    def states(self) -> gpd.GeoDataFrame:
        """Spatially resolved states"""
        if self._states:
            return self._states
        else:
            self._states = self._get_state_boundaries(self)
            return self._states
    
    @property
    def state_center_points(self) -> gpd.GeoDataFrame:
        """Center points of Sates"""
        if self._state_center_points:
            return self._state_center_points
        else:
            if not self._states:
                self._states = self._get_state_boundaries()
            self._state_center_points = self._get_state_center_points()
            return self._state_center_points
                
        
    def _get_state_boundaries(self) -> gpd.GeoDataFrame:
        """Gets admin boundaries of state"""
        return (
            self._counties
            .dissolve("STATE_NAME")
            .rename(columns={"STUSPS":"STATE"})
            .reset_index()
            [["STATE_NAME", "STATE","geometry"]]
        )

    def _get_state_center_points(self) -> gpd.GeoDataFrame:
        """Gets centerpoints of states using county shapefile"""
        gdf = self._states.copy().rename(columns={"geometry":"shape"})
        gdf["geometry"] = gdf["shape"].map(lambda x: x.centroid)
        gdf[["x","y"]] = gdf["geometry"].apply(lambda x: pd.Series({"x": x.x, "y": x.y}))
        return gdf[["STATE", "x", "y"]]


###
# MAIN DATA INTERFACE
###

class GasData(ABC):
    """Main class to interface with data"""
    
    state_2_interconnect = constants.STATES_INTERCONNECT_MAPPER
    state_2_name = {v: k for k, v in constants.STATE_2_CODE.items()}
    states_2_remove = [x for x, y in constants.STATES_INTERCONNECT_MAPPER.items() if not y]
    
    def __init__(self, year: int, interconnect: str) -> None:
        self.year = year
        if interconnect.lower() not in ("western", "eastern", "texas", "usa"):
            logger.debug(f"Invalid interconnect of {interconnect}. Setting to 'usa'")
            self.interconnect = "usa" # no filtering of data
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
    
    def filter_on_interconnect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Name of states must be in column called 'STATE'"""
        
        if "STATE" not in df.columns:
            logger.debug("Natual gas data not filtered due to incorrect data formatting")
            return df
        
        df = df[~df.STATE.isin(self.states_2_remove)].copy()
        
        if self.interconnect == "usa":
            return df
        else:
            df["interconnect"] = df.STATE.map(self.state_2_interconnect)
            assert not df.interconnect.isna().any()
            df = df[df.interconnect == self.interconnect]
            if df.empty:
                logger.warning(f"Empty natural gas data for interconnect {self.interconnect}")
            return df.drop(columns="interconnect")

class GasBuses(GasData):
    """Creator for natural gas buses
    
    Argumets:
        County shapefile of United States
    """
    
    def __init__(self, interconnect: str, counties: str) -> None:
        self.states = StateGeometry(counties)
        super().__init__(year=2020, interconnect=interconnect) # year locked for location mapping
        
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
            unit="MMCF",
            interconnect=self.interconnect,
            country=states.index, # for consistency 
            STATE=states.index,
            STATE_NAME=states.name
        )

class GasStorage(GasData):
    
    def __init__(self, year: int, interconnect: str, api: str) -> None:
        self.api = api
        super().__init__(year, interconnect)
        
    
    def read_data(self):
        base = eia.Storage("gas", "base", self.year, self.api)
        total = eia.Storage("gas", "total", self.year, self.api)
        return pd.concat([base, total])
    
    def format_data(self, data: pd.DataFrame):
        return self.filter_on_interconnect(data)
    
    def build_infrastructure(self, n: pypsa.Network, **kwargs):
        
        df = (
            self.data
            .copy()
            .reset_index()
            .drop(columns=["COUNTY"])
            .groupby("STATE")
            .sum()
        )
        df["bus"] = df.index
        
        n.madd(
            "Bus",
            names=df.index,
            suffix=" gas storage",
            carrier="gas storage",
            unit="MMCF",
        )
        
        cyclic_storage = kwargs.get("cyclic_storage", False)
        n.madd(
            "Store",
            names=df.index,
            suffix=" gas storage",
            bus=df.index + " gas storage",
            carrier="gas storage",
            e_nom_extendable=False,
            e_nom=df.MAX_CAPACITY_MMCF,
            e_cyclic=cyclic_storage,
            e_min_pu=df.MAX_CAPACITY_MMCF / df.MAX_CAPACITY_MMCF,
            marginal_cost=0 # to update
        )
        
        # must do two links, rather than a bidirectional one, to constrain 
        # daily discharge limits 
        n.madd(
            "Link",
            names=df.index,
            suffix=" charge gas storage",
            carrier="gas storage",
            bus0=df.index + " gas",
            bus1=df.index + " gas storage",
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=False,
            marginal_cost=0
        )
        
        n.madd(
            "Link",
            names=df.index,
            suffix=" discharge gas storage",
            carrier="gas storage",
            bus0=df.index + " gas storage",
            bus1=df.index + " gas",
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=False,
            marginal_cost=0
        )

class GasProcessing(GasData):
    
    def __init__(self, interconnect: str, eia_757: str) -> None:
        self.eia_757 = eia_757
        super().__init__(year=2017, interconnect=interconnect) # only 2017 is available
        
    
    def read_data(self) -> pd.DataFrame:
        return pd.read_csv(self.eia_757).fillna(0)
    
    def format_data(self, data: pd.DataFrame):
        df = data.copy()
        df = df.rename(columns={x:x.replace("<BR>", " ") for x in df.columns})
        df.columns = df.columns.str.strip()
        df = df[[
            "Report State", 
            "County Name", 
            "Plant Capacity", 
            "Plant Flow", 
            "BTU Content"
        ]]
        df["Report State"] = df["Report State"].str.capitalize()
        df = df.rename(columns={
            "Report State":"STATE", 
            "County Name":"COUNTY", 
            "Plant Capacity":"CAPACITY_MMCF", 
            "Plant Flow":"FLOW_MMCF", 
            "BTU Content":"BTU_CONTENT", 
        })
        df["STATE"] = df["STATE"].str.upper()

        df = (
            df
            .drop(columns=["COUNTY"])
            .groupby("STATE")
            .sum()
            .reset_index()
        )
        
        return self.filter_on_interconnect(df)
    
    def build_infrastructure(self, n: pypsa.Network, **kwargs):

        df = self.data.copy()
        df = df.set_index("STATE")
        df["bus"] = df.index + " gas"
        
        n.madd(
            "Generator", 
            names=df.index,
            suffix=" gas production",
            bus=df.bus,
            carrier="gas",
            p_nom_extendable=False,
            marginal_costs=0.35, # https://www.eia.gov/analysis/studies/drilling/pdf/upstream.pdf
            p_nom=df.CAPACITY_MMCF
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
            index_col=0
        )
    
    def format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df.columns = df.columns.str.strip()
        df = df[df.index == int(self.year)]
        df = df.rename(columns={
            "State From":"STATE_FROM",
            "County From":"COUNTRY_FROM",
            "State To":"STATE_TO",
            "County To":"COUNTRY_TO",
            "Capacity (mmcfd)":"CAPACITY_MMCFD"
        })
        df = (
            df
            .astype(
                {
                    "STATE_FROM":"str",
                    "COUNTRY_FROM":"str",
                    "STATE_TO":"str",
                    "COUNTRY_TO":"str",
                    "CAPACITY_MMCFD":"float"
                }
            )
            [["STATE_FROM","STATE_TO","CAPACITY_MMCFD"]]
            .groupby(["STATE_TO", "STATE_FROM"])
            .sum()
            .reset_index()
            .map(self.assign_pipeline_interconnects)
        )
        return self.extract_pipelines(df)

    @abstractmethod
    def build_infrastructure(self, n: pypsa.Network) -> None:
        pass

    @abstractmethod
    def extract_pipelines(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts pipelines for that region. Used in Format data"""
        pass

    def assign_pipeline_interconnects(self, df: pd.DataFrame):
        """Adds interconnect labels to the pipelines"""
        
        df["INTERCONNECT_TO"] = df.STATE_TO.map(self.state_2_interconnect)
        df["INTERCONNECT_FROM"] = df.STATE_FROM.map(self.state_2_interconnect)
        
        assert not df.isna().any().any()    
        
        return df

class InterconnectGasPipelineCapacity(_GasPipelineCapacity):
    """Pipeline capacity within the interconnect"""

    def __init__(self, year: int, interconnect: str, xlsx: str) -> None:
        super().__init__(year, interconnect, xlsx)

    def extract_pipelines(self, data: pd.DataFrame) -> pd.DataFrame:
        
        df = data.copy()
        # for some reason drop duplicates is not wokring here and I cant figure out why :(
        # df = df.drop_duplicates(subset=["STATE_TO", "STATE_FROM"], keep=False).copy()
        df = df[~df.apply(lambda x: x.STATE_TO == x.STATE_FROM, axis=1)].copy()
        
        if self.interconnect != "usa":
            df = df[
                (df.INTERCONNECT_TO == self.interconnect) & (df.INTERCONNECT_FROM == self.interconnect)
            ]
            if df.empty:
                logger.error(f"Empty natural gas domestic pipelines for interconnect {self.interconnect}")
        else:
            df = df[
                ~(df[["INTERCONNECT_TO", "INTERCONNECT_FROM"]].isin(["canada", "mexico"])).all(axis=1)
            ]
        return df.reset_index(drop=True)


    def build_infrastructure(self, n: pypsa.Network) -> None:
       
        df = self.data.copy()
        
        df.index = df.STATE_FROM + " " + df.STATE_TO
        
        n.madd(
            "Link",
            names=df.index,
            suffix=" pipeline",
            carrier="gas pipeline",
            unit="MMCF",
            bus0=df.STATE_FROM + " gas",
            bus1=df.STATE_TO + " gas",
            p_nom=round(df.CAPACITY_MMCFD / 24), # get a hourly flow rate 
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=False,
        )

class TradeGasPipelineCapacity(_GasPipelineCapacity):
    """Pipeline capcity connecting to the interconnect"""
    
    def __init__(self, year: int, interconnect: str, xlsx: str, domestic: bool = True) -> None:
        self.domestic = domestic
        super().__init__(year, interconnect, xlsx)
        
        
    def extract_pipelines(self, data: pd.DataFrame) -> pd.DataFrame:
        
        df = data.copy()
        if self.domestic:
            return self._get_domestic_pipeline_connections(df)
        else: 
            return self._get_international_pipeline_connections(df)

    def _get_domestic_pipeline_connections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gets all pipelines within the usa that connect to the interconnect"""
        
        if self.interconnect == "usa":
            # no domestic connections 
            return pd.DataFrame(columns=df.columns)
        else:
            # get rid of international connections
            df = df[
                ~((df.INTERCONNECT_TO.isin(["canada", "mexico"])) | (df.INTERCONNECT_FROM.isin(["canada", "mexico"])))
            ]
            # get rid of pipelines within the interconnect 
            return df[
                (df["INTERCONNECT_TO"].eq(self.interconnect) | df["INTERCONNECT_FROM"].eq(self.interconnect)) & 
                ~(df["INTERCONNECT_TO"].eq(self.interconnect) & df["INTERCONNECT_FROM"].eq(self.interconnect)) 
            ]

    def _get_international_pipeline_connections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gets all international pipeline connections"""
        df = df[
                (df.INTERCONNECT_TO.isin(["canada", "mexico"])) | 
                (df.INTERCONNECT_FROM.isin(["canada", "mexico"]))
            ]
        if self.interconnect == "usa":
            return df
        else:
            return df[
                (df.INTERCONNECT_TO == self.interconnect) | 
                (df.INTERCONNECT_FROM == self.interconnect)
            ]
            
    def build_infrastructure(self, n: pypsa.Network) -> None:
        """Builds import and export bus+link+store to connect to
        
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
            to_from = df[df.INTERCONNECT_TO==self.interconnect].copy() # exports
            from_to = df[df.INTERCONNECT_FROM==self.interconnect].copy() # imports
        else:
            to_from = df[~df.INTERCONNECT_TO.isin(["canada", "mexico"])].copy()
            from_to = df[~df.INTERCONNECT_FROM.isin(["canada", "mexico"])].copy()
            
        to_from["NAME"] = to_from.STATE_FROM + " " + to_from.STATE_TO
        from_to["NAME"] = from_to.STATE_TO + " " + from_to.STATE_FROM
        
        to_from = to_from.set_index("NAME")
        from_to = from_to.set_index("NAME")
        
        n.madd(
            "Bus",
            names=to_from.index,
            suffix=" gas export",
            carrier="gas export",
            unit="MMCF",
            country=to_from.STATE_TO,
            interconnect=self.interconnect,
        )
        
        n.madd(
            "Bus",
            names=from_to.index,
            suffix=" gas import",
            carrier="gas import",
            unit="MMCF",
            country=from_to.STATE_FROM,
            interconnect=self.interconnect,
        )
        
        n.madd(
            "Link",
            names=to_from.index,
            suffix=" gas export",
            carrier="gas export",
            unit="MMCF",
            bus0=to_from.STATE_TO + " gas",
            bus1=to_from.index + " gas export",
            p_nom=round(to_from.CAPACITY_MMCFD / 24), # get a hourly flow rate 
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=False,
            marginal_cost=0,
        )
        
        n.madd(
            "Link",
            names=from_to.index,
            suffix=" gas import",
            carrier="gas import",
            unit="MMCF",
            bus0=from_to.index + " gas import",
            bus1=from_to.STATE_FROM + " gas",
            p_nom=round(from_to.CAPACITY_MMCFD / 24), # get a hourly flow rate 
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=False,
            marginal_cost=0,
        )
        
        n.madd(
            "Store",
            names=to_from.index,
            suffix=" gas export",
            unit="MMCF",
            bus=to_from.index + " gas export",
            carrier="gas export",
            e_nom_extendable=True,
            capital_cost=0,
            e_nom=0,
            e_cyclic=False,
            marginal_cost=0,
        )
        
        n.madd(
            "Store",
            names=from_to.index,
            unit="MMCF",
            suffix=" gas import",
            bus=from_to.index + " gas import",
            carrier="gas import",
            e_nom_extendable=True,
            capital_cost=0,
            e_nom=0,
            e_cyclic=False,
            marginal_cost=0,
        )

class PipelineLinepack(GasData):
    
    def __init__(self, year: int, interconnect: str, counties: str, pipelines: str) -> None:
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
        
        length_in_state = gpd.sjoin(gdf.to_crs("4269"), states, how="right", predicate="within").reset_index()
        length_in_state = (
            length_in_state[["STATE_NAME", "STATE", "TYPEPIPE", "Shape_Leng", "Shape__Length"]]
            .rename(columns={"Shape_Leng":"LENGTH_DEG", "Shape__Length":"LENGTH_M"})
            .groupby(by=["STATE_NAME", "STATE", "TYPEPIPE"]).sum().reset_index()
        )
        
        # https://publications.anl.gov/anlpubs/2008/02/61034.pdf
        intrastate_radius = 12 * 0.0254 # inches in meters (24in dia)
        interstate_radius = 18 * 0.0254 # inches meters (36in dia)

        volumne_in_state = length_in_state.copy()
        volumne_in_state["RADIUS"] = volumne_in_state.TYPEPIPE.map(lambda x: interstate_radius if x == "Interstate" else intrastate_radius)
        volumne_in_state["VOLUME_M3"] = volumne_in_state.LENGTH_M * pi * volumne_in_state.RADIUS ** 2 
        volumne_in_state = volumne_in_state[["STATE_NAME", "STATE", "VOLUME_M3"]]
        volumne_in_state = volumne_in_state.groupby(by=["STATE_NAME", "STATE"]).sum()
        
        # https://publications.anl.gov/anlpubs/2008/02/61034.pdf
        max_pressure = 8000 # kPa
        min_pressure = 4000 # kPa

        energy_in_state = volumne_in_state.copy()
        energy_in_state["MAX_ENERGY_kJ"] = energy_in_state.VOLUME_M3 * max_pressure
        energy_in_state["MIN_ENERGY_kJ"] = energy_in_state.VOLUME_M3 * min_pressure
        energy_in_state["NOMINAL_ENERGY_kJ"] = (energy_in_state.MAX_ENERGY_kJ + energy_in_state.MIN_ENERGY_kJ) / 2
        
        # https://apps.cer-rec.gc.ca/Conversion/conversion-tables.aspx#s1ss1
        # 1 GJ to 947.8171 CF
        # TODO: replace with heating value 
        kj_2_mmcf = 1e-6 * 947.8171 * 1e-6 # kj -> GJ -> cf -> mmcf
        
        final = energy_in_state.copy()
        final["MAX_ENERGY_MMCF"] = final.MAX_ENERGY_kJ * kj_2_mmcf
        final["MIN_ENERGY_MMCF"] = final.MIN_ENERGY_kJ * kj_2_mmcf
        final["NOMINAL_ENERGY_MMCF"] = final.NOMINAL_ENERGY_kJ * kj_2_mmcf

        final = final[["MAX_ENERGY_MMCF", "MIN_ENERGY_MMCF", "NOMINAL_ENERGY_MMCF"]].reset_index()
        return self.filter_on_interconnect(final)
        
    def build_infrastructure(self, n: pypsa.Network) -> None:
        
        df = self.data.copy()
        df = df.set_index("STATE")
        
        n.madd(
            "StorageUnit",
            names=df.index,
            unit="MMCF",
            suffix=" linepack",
            bus=df.index + " gas",
            carrier="gas pipeline",
            p_nom=0,
            p_nom_extendable=False,
            p_nom_min=0,
            p_nom_max=np.inf,
            marginal_cost=0,
            capital_cost=0,
            state_of_charge_initial=df.NOMINAL_ENERGY_MMCF,
            state_of_charge_initial_per_period=False,
            cyclic_state_of_charge=True,
            cyclic_state_of_charge_per_period=False,
            max_hours=1,
            efficiency_store=1,
            efficiency_dispatch=1,
            standing_loss=0
        )

class ImportExportLimits(GasData):
    """Adds constraints for import export limits"""
    
    def __init__(self, year: int, interconnect: str, api: str) -> None:
        self.api = api
        super().__init__(year, interconnect)
        
    def read_data(self) -> pd.DataFrame:
        imports = eia.Trade("gas", "imports", self.year, self.api).get_data()
        exports = eia.Trade("gas", "exports", self.year, self.api).get_data()
        return pd.concat([imports, exports])
    
    def format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = (
            data
            .reset_index()
            .drop(columns=["series-description"])
            .groupby(["period", "units", "state"])
            .sum()
            .reset_index()
            .rename(columns={"state":"STATE"})
            .copy()
        )
        # may need to add ["U.S."] to states to remove here 
        return self.filter_on_interconnect(df)
    
    def build_infrastructure(self, n: pypsa.Network) -> None:
        pass
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
    eia_757_path: str = "../data/natural_gas/eia_757.csv",
) -> None:

    # add gas carrier
    
    n.add("Carrier","gas")

    # add state level gas buses

    buses = GasBuses(interconnect, county_path)
    buses.build_infrastructure(n)
    
    # add state level natural gas processing facilities 
    
    production = GasProcessing(interconnect, eia_757_path)
    production.build_infrastructure(n)
    
    # add state level gas storage facilities 
    
    storage = GasStorage(year, interconnect, api)
    storage.build_infrastructure(n)
    
    # add interconnect pipelines 
    
    pipelines = InterconnectGasPipelineCapacity(year, interconnect, pipelines_path)
    pipelines.build_infrastructure(n)
    
    # add pipelines for imports/exports 
    
    # TODO: have trade pipelines share data to only instantiate once
    pipelines_domestic = TradeGasPipelineCapacity(year, interconnect, pipelines_path, domestic=True)
    pipelines_international = TradeGasPipelineCapacity(year, interconnect, pipelines_path, domestic=False)
    
    pipelines_domestic.build_infrastructure(n)
    pipelines_international.build_infrastructure(n)
    
    # add pipeline linepack 
    
    linepack = PipelineLinepack(year, interconnect, county_path, pipeline_shape_path)
    linepack.build_infrastructure(n)

if __name__ == "__main__":

    n = pypsa.Network("../resources/western/elec_s_30_ec_lv1.25_Co2L1.25.nc")
    year = 2020
    with open("./../config/config.api.yaml", "r") as file:
        yaml_data = yaml.safe_load(file)
    api = yaml_data["api"]["eia"]
    build_natural_gas(n=n, year=year, api=api)