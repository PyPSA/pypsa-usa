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

def convert_generators_2_links(n: pypsa.Network, carrier: str, bus0_suffix: str):
    """Replace Generators with cross sector links. 
    
    Links bus1 are the bus the generator is attached to. Links bus0 are state 
    level followed by the suffix (ie. "WA gas" if " gas" is the bus0_suffix)
    
    n: pypsa.Network, 
    carrier: str,
        carrier of the generator to convert to a link
    bus0_suffix: str,
        suffix to attach link to 
    """
    
    plants = n.generators[n.generators.carrier==carrier].copy()
    plants["STATE"] = plants.bus.map(n.buses.STATE)
    
    n.madd(
        "Link",
        names=plants.index,
        bus0=plants.STATE + bus0_suffix,
        bus1=plants.bus,
        carrier=plants.carrier,
        p_nom_min=plants.p_nom_min,
        p_nom=plants.p_nom,
        p_nom_max=plants.p_nom_max,
        p_nom_extendable=plants.p_nom_extendable,
        ramp_limit_up=plants.ramp_limit_up,
        ramp_limit_down=plants.ramp_limit_down,
        efficiency=plants.efficiency,
        marginal_cost=plants.marginal_cost,
        capital_cost=plants.capital_cost,
        lifetime=plants.lifetime,
    )
    
    # copy time varrying parameters 
    # for gen in plants.index: 
    #     n.
    
    # remove generators 
    n.mremove("Generator", plants.index)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_sectors",
            interconnect="texas",
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
            interconnect=snakemake.wildcards.interconnect,
            counties=snakemake.input.counties,
            eia_757=snakemake.input.eia_757,
            eia_191=snakemake.input.eia_191,
            pipelines=snakemake.input.pipelines,
        )
        
        # convert existing generators to links 
        for carrier in ("CCGT", "OCGT"):
            convert_generators_2_links(n, carrier, " gas")
        
    n.export_to_netcdf(snakemake.output.network)