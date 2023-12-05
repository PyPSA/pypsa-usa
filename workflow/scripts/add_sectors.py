"""Generic module to add a new energy network

Creates new sector ontop of existing one. Note: Currently, can only be built ontop of electricity sector

Marginal costs are handeled as follows:
- Links are the VOM of just the technology
- Replacement generators contain time varrying fuel costs
"""

import pypsa
import pandas as pd
from typing import List, Union
import logging
logger = logging.getLogger(__name__)
from _helpers import configure_logging
from add_electricity import load_costs

def add_carrier(n: pypsa.Network, carrier: str, costs: pd.DataFrame = pd.DataFrame(), **kwargs):
    """Adds new carrier into the network with emission factor"""
    if not carrier in n.carriers.index:
        attrs = {}
        if kwargs.get("tech_colors"):
            attrs["color"] = kwargs["tech_colors"].get(carrier)
        if kwargs.get("nice_names"):
            attrs["nice_names"] = kwargs["nice_names"].get(carrier)
        if not costs.empty:
            try:
                attrs["co2_emissions"] = costs.at[carrier, "co2_emissions"]
            except KeyError:
                logger.debug(f"{carrier} does not have an emission factor ")
        n.add("Carrier", carrier, **attrs)
            

def add_buses(n: pypsa.Network, new_carrier: str, old_carrier: str):
    """Creates buses with new carriers at same geographic location"""
    buses = n.generators[n.generators.carrier == old_carrier].bus.drop_duplicates()
    if buses.empty:
        logger.debug(f"Can not create new buses for type {new_carrier}")
        return
    buses = pd.DataFrame(buses, columns=["bus"])
    buses["new_bus"] = buses.bus + f" {new_carrier}"
    buses = buses[~(buses["new_bus"].isin(n.buses.index))]
    if buses.empty:
        logger.info(f"No new buses to create for {new_carrier}")
    else:
        b = n.buses[n.buses.index.isin(buses.bus)]
        n.madd(
            "Bus", 
            names=b.index,
            suffix=f" {new_carrier}",
            x=b.x,
            y=b.y,
            interconnect=b.interconnect,
            country=b.country,
            zone_id=b.zone_id,
            carrier=new_carrier
        )

def add_buses_from_ac(n: pypsa.Network, new_carrier: str):
    """Adds buses at all AC locations"""
    buses = n.buses[n.buses.carrier == "AC"]
    n.madd(
        "Bus", 
        names=buses.bus,
        suffix=f" {new_carrier}",
        x=buses.x,
        y=buses.y,
        interconnect=buses.interconnect,
        country=buses.country,
        zone_id=buses.zone_id,
        carrier=new_carrier
    )


def convert_generators_2_links(n: pypsa.Network, new_carrier: str, old_carrier: str, costs: pd.DataFrame = pd.DataFrame()):
    """Replace Generators with cross sector links
    
    n: pypsa.Network, 
    new_carrier: str,
        New carrier of the network (ie. Gas)
    old_carriers: str,
        Old carrier (ie. "CCGT")
    costs: pd.DataFrame = pd.DataFrame(),
    """
    
    plants = n.generators[n.generators.carrier==old_carrier]
    marginal_costs = get_static_marginal_costs(n, old_carrier, costs)
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
        marginal_cost=marginal_costs,
        capital_cost=plants.capital_cost,
        lifetime=plants.lifetime,
    )

def add_generators(n: pypsa.Network, new_carrier: str, old_carrier: str, costs: pd.DataFrame = None):
    """Attaches generators to mine raw fuel"""
    
    plants = n.generators[n.generators.carrier == old_carrier].drop_duplicates(subset="bus")
    
    # check if mining plants have already been added
    plants["new_name"] = plants.bus.map(lambda x: f"{x} {new_carrier}")
    existing_plants = n.generators.bus.drop_duplicates().to_list()
    plants = plants.loc[~plants.new_name.isin(existing_plants)]
    if plants.empty:
        return
    
    fuel_cost = get_static_marginal_costs(n, new_carrier, costs)
    assert fuel_cost >= 0
    
    # bus names are done differently cause its not working if you just do 
    # > bus = plants.bus + f" {new_carrier}"
    n.madd(
        "Generator", 
        names=plants.bus,
        suffix=f" {new_carrier}",
        bus=[f"{x} {new_carrier}" for x in plants.bus],
        carrier=new_carrier,
        p_nom_extendable=True,
        marginal_costs=fuel_cost
    )

def remove_generators(n: pypsa.Network, old_carrier: str):
    """Remove generators from old buses"""
    plants = n.generators[n.generators.carrier == old_carrier]
    n.mremove("Generator", plants.index)
    n.generators_t["marginal_cost"] = n.generators_t["marginal_cost"][[x for x in n.generators_t["marginal_cost"].columns if x not in plants.index]]

def get_static_marginal_costs(n: pypsa.Network, carrier: str, costs: pd.DataFrame = None):
    """Gets static VOM costs
    
    Tries to read in raw VOM from costs sheet, else takes average 
    VOM from existing network. Else, returns zero 
    """
    if isinstance(costs, pd.DataFrame):
        try:
            return costs.at[old_carrier, "VOM"]
        except KeyError:
            logger.warning(f"Can not locate marginal costs for {carrier}")
    plants = n.generators[n.generators.carrier==carrier]
    if plants.empty:
        return 0
    else:
        return plants.marginal_cost.mean()

def get_variable_marginal_cost(n: pypsa.Network, carrier: str):
    """Gets exising time dependent marginal costs"""
    gens = [f"{x} {carrier}" for x in n.buses.index] # ie. AVA0 0 CCGT
    marginal_cost = n.generators_t.marginal_cost[[x for x in gens if x in n.generators_t.marginal_cost.columns]]
    if marginal_cost.empty:
        logger.info(f"No generators with time varrying marginal cost for {old_carrier}")
        return pd.DataFrame(index=n.snapshots)
    else:
        return marginal_cost

def copy_variable_marginal_cost(n: pypsa.Network, new_carrier: str, old_carrier: str, costs: pd.DataFrame):
    """Copys over marginal cost values"""
    
    varaible_marginal_cost = get_variable_marginal_cost(n, old_carrier) # ie. CCGT VOM + Gas Costs
    if varaible_marginal_cost.empty:
        logger.info(f"No generators with time varrying marginal cost for {old_carrier}")
        return
    static_marginal_cost = get_static_marginal_costs(n, old_carrier, costs) # ie. CCGT VOM
    
    # extract out only fuel costs 
    varaible_marginal_cost = varaible_marginal_cost - static_marginal_cost #ie. gas costs  
    
    # assign proper names 
    plants = n.generators[n.generators.carrier == old_carrier] # ie. CCGT generators 
    name_mapper = {f"{bus} {old_carrier}":f"{bus} {new_carrier}" for bus in plants.bus.unique()} # ie. {AVA0 0 CCGT: AVA0 0 gas}
    varaible_marginal_cost = varaible_marginal_cost.rename(columns=name_mapper)
    
    # copy over values 
    varaible_marginal_cost_apply = varaible_marginal_cost[[x for x in varaible_marginal_cost.columns if x not in n.generators_t.marginal_cost.columns]]
    n.generators_t.marginal_cost = n.generators_t.marginal_cost.join(varaible_marginal_cost_apply)

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
    
    nice_names = params.plotting.get("nice_names", None)
    tech_colors = params.plotting.get("tech_colors", None)

    sectors = snakemake.wildcards.sectors.split("-")
    
    if "G" in sectors:
        new_carrier = "gas"
        add_carrier(n, new_carrier, costs, nice_names=nice_names, tech_colors=tech_colors)
        for old_carrier in ("CCGT", "OCGT"):
            add_buses(n, new_carrier, old_carrier)
            add_generators(n, new_carrier, old_carrier, costs)
            copy_variable_marginal_cost(n, new_carrier, old_carrier, costs) # done seperate in case a gen only has static value
            convert_generators_2_links(n, new_carrier, old_carrier, costs)
            remove_generators(n, old_carrier) # must come last 
        
    n.export_to_netcdf(snakemake.output.network)