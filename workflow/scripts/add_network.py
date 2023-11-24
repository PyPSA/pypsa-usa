"""Generic module to add a new energy network

This module will do two things: 
1. Add new buses to all exisitng buses at the same location as existing buses 
based on carrier types 
2. If generators of the new bus carrier are identified, the generator is replaced with 
a one directional link that takes on the generator parameters, and a new generator at the 
new bus 
"""

import pypsa
import pandas as pd
from typing import List, Dict
import logging
logger = logging.getLogger(__name__)
from _helpers import configure_logging


def add_carrier(n: pypsa.Network, carrier: str, costs: pd.DataFrame = pd.DataFrame()):
    """Adds new carrier into the network with emission factor"""
    if not carrier in n.carriers.index:
        try:
            n.add("Carrier", carrier, co2_emissions=costs.at[carrier, "CO2 intensity"])
        except KeyError:
            logger.debug(f"{carrier} does not have an emission factor ")
            n.add("Carrier", carrier, co2_emissions=0)

def add_buses(n: pypsa.Network, carrier: str, carrier_follow: str = "AC"):
    """Creates buses with new carriers at same geographic location
    
    carrier: str
        New carrier type 
    carrier_follow: str
        Carrier of buses to follow
    """
    buses = n.buses[n.buses.carrier == carrier_follow]
    if buses.empty:
        logger.debug(f"Can not create new buses for type {carrier}")
    else:
        n.madd(
            "Bus", 
            names=buses.index,
            suffix=f" {carrier}",
            x=buses.x,
            y=buses.y,
            interconnect=buses.interconnect,
            country=buses.country,
            carrier=carrier
        )

def add_links(n: pypsa.Network, carrier: str, add_generators: bool = True):
    """Replace Generators with cross sector links
    
    carrier: List[str]
    add_generators: bool = True
        Replace exisiting generators with new mining generators at each node 
    """
    
    plants = n.generators[n.generators.carrier.isin(carrier)]
    n.madd(
        "Link",
        names=plants.index,
        bus0=f"{plants.bus} {carrier}", # incorrect with anything except base network
        bus1=plants.bus,
        carrier=carrier,
        p_nom_min=plants.p_nom_min,
        p_nom=plants.p_nom,
        p_nom=plants.p_nom_max,
        p_nom_extendable=plants.p_nom_extendable,
        ramp_limit_up=plants.ramp_limit_up,
        ramp_limit_down=plants.ramp_limit_down,
        efficiency=plants.efficiency,
        marginal_cost=plants.marginal_cost,
        capital_cost=plants.capital_cost,
        lifetime=plants.lifetime,
    )
    n.mremove("Generator", plants.index)
    
    if add_generators:
        n.madd("Generator", names=f"{plants.index} {carrier}", bus=f"{plants.bus} {carrier}")

def add_network(n: pypsa.Network, carrier: Dict[str,List[str]], costs: pd.DataFrame = pd.DataFrame(), **kwargs):
    """Creates new network ontop of existing one
    
    n: pypsa.Network, 
    carrier: str
        Carrier to represent in the network
    technologies: List[str] = None
        Exisitng technologies to map to new carrier network . If 
    costs: pd.DataFrame
        If not provided, emissions are not created for new carrier
    """
    
    techs
    
    add_carrier(n, carrier, costs)
    add_buses(n, carrier)
    add_links(n, carrier, **kwargs)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_network",
            interconnect="western",
            # simpl="",
            clusters="30",
            sector_opts="G",
        )
    configure_logging(snakemake)

    opts = snakemake.wildcards.opts.split("-")
    
    if "G" in opts:
        add_network(n, )