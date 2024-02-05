"""Generic module to add a new energy network

Creates new sector ontop of existing one. Note: Currently, can only be built ontop of electricity sector

Marginal costs are handeled as follows:
- Links are the VOM of just the technology
- Replacement generators contain time varrying fuel costs
"""

import pypsa
import pandas as pd
import geopandas as gpd
from typing import List, Union, Dict
import logging
logger = logging.getLogger(__name__)
from _helpers import configure_logging
from add_electricity import load_costs
from build_natural_gas import build_natural_gas
from shapely.geometry import Point
import constants
import sys

def assign_bus_2_state(n: pypsa.Network, shp: str,  states_2_include: List[str] = None, state_2_state_name: Dict[str, str] = None) -> None:
    """Adds a state column to the network buses dataframe
    
    The shapefile must be the counties shapefile
    """
    
    buses = n.buses[["x", "y"]].copy()
    buses["geometry"] = buses.apply(lambda x: Point(x.x, x.y), axis=1)
    buses = gpd.GeoDataFrame(buses, crs="EPSG:4269")
    
    states = gpd.read_file(shp).dissolve("STUSPS")["geometry"]
    states = gpd.GeoDataFrame(states)
    if states_2_include:
        states = states[states.index.isin(states_2_include)]
    
    # project to avoid CRS warning from geopandas 
    buses_projected = buses.to_crs("EPSG:3857")
    states_projected = states.to_crs("EPSG:3857")
    gdf = gpd.sjoin_nearest(buses_projected, states_projected, how="left")
    
    n.buses["STATE"] = n.buses.index.map(gdf.index_right)
    
    if state_2_state_name:
        n.buses["STATE_NAME"] = n.buses.STATE.map(state_2_state_name)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_sectors",
            interconnect="western",
            # simpl="",
            clusters="40",
            ll="v1.25",
            opts="Co2L1.25",
            sector="E-G",
        )
    configure_logging(snakemake)
    
    n = pypsa.Network(snakemake.input.network)
    
    sectors = snakemake.wildcards.sector.split("-")
    
    # exit if only electricity network
    if all(s == "E" for s in sectors):
        n.export_to_netcdf(snakemake.output.network)
        sys.exit() 
    
    # map states to each clustered bus
    
    if snakemake.wildcards.interconnect == "usa":
        states_2_map = [x for x,y in constants.STATES_INTERCONNECT_MAPPER.items() if y in ("western", "eastern", "texas")]
    else:
        states_2_map = [x for x,y in constants.STATES_INTERCONNECT_MAPPER.items() if y == snakemake.wildcards.interconnect]
        
    code_2_state = {v: k for k, v in constants.STATE_2_CODE.items()}
    assign_bus_2_state(n, snakemake.input.counties, states_2_map, code_2_state)
    
    if "G" in sectors:
        build_natural_gas(
            n=n,
            year=pd.to_datetime(snakemake.parmas.snapshots["start"]).year,
            api=snakemake.params.api["eia"],
            interconnect=snakemake.params.interconnect,
            county=snakemake.input.county_path,
            pipelines_path=snakemake.input.pipeline_capacity,
            pipeline_shape_path=snakemake.input.pipeline_shape,
            eia_757_path=snakemake.input.eia_757
        )
        
    n.export_to_netcdf(snakemake.output.network)