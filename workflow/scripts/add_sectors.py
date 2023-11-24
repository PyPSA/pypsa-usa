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
from typing import List, Union
import logging
logger = logging.getLogger(__name__)
from _helpers import configure_logging
from add_electricity import load_costs

def add_carrier(n: pypsa.Network, carrier: str, costs: pd.DataFrame = pd.DataFrame()):
    """Adds new carrier into the network with emission factor"""
    if not carrier in n.carriers.index:
        try:
            n.add("Carrier", carrier, co2_emissions=costs.at[carrier, "co2_emissions"])
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
            zone_id=buses.zone_id,
            carrier=carrier
        )

def add_links(n: pypsa.Network, new_carrier: str, old_carrier: List[str], add_generators: bool = True):
    """Replace Generators with cross sector links
    
    carrier: List[str]
    add_generators: bool = True
        Replace exisiting generators with new mining generators at each node 
    """
    
    plants = n.generators[n.generators.carrier.isin(old_carrier)]
    n.madd(
        "Link",
        names=plants.index,
        bus0=plants.bus + f" {new_carrier}", # incorrect with anything except electrical network
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
    n.mremove("Generator", plants.index)
    
    if add_generators:
        n.madd("Generator", names=plants.index + f" {new_carrier}", bus= plants.bus + f" {new_carrier}")
    n.madd("Generator", names=plants.index + f" {new_carrier}", bus= plants.bus + f" {new_carrier}")

def add_sector(n: pypsa.Network, new_carrier: str, old_carriers: Union[str, List[str]] = None, costs: pd.DataFrame = pd.DataFrame(), **kwargs):
    """Creates new sector ontop of existing one
    
    Note: Currently, can only be built ontop of electricity sector
    
    n: pypsa.Network, 
    new_carrier: str
        Carrier to represent in the network
    old_carrier: List[str] = None
        Exisitng technologies to map to new carrier network . If none are provided, then 
        only buses are created, without links or generators
    costs: pd.DataFrame
        If not provided, emissions are not created for new carrier
    """
    
    add_carrier(n, new_carrier, costs)
    add_buses(n, new_carrier)
    if not isinstance(old_carriers, list):
        old_carriers = [old_carriers]
    add_links(n, new_carrier, old_carriers, **kwargs)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_sectors",
            interconnect="western",
            # simpl="",
            clusters="30",
            ll="v1.25",
            opts="Co2L0.75",
            sectors="G",
        )
    configure_logging(snakemake)
    
    n = pypsa.Network(snakemake.input.network)

    params = snakemake.params
    
    Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    costs = load_costs(
        snakemake.input.tech_costs,
        params.costs,
        params.electricity["max_hours"],
        Nyears,
    )

    sectors = snakemake.wildcards.sectors.split("-")
    
    if "G" in sectors:
        new_carrier = "gas"
        old_carriers = ["CCGT", "OCGT"]
        add_sector(n, new_carrier, old_carriers, costs, add_generators=False)
        
    n.export_to_netcdf(snakemake.output.network)