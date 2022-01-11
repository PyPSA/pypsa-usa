# Copyright 2021-2022 Martha Frysztacki (KIT)

import pypsa
import pandas as pd

import logging

def add_buses_from_file(n, fn_buses):

    buses = pd.read_csv(fn_buses, index_col=0)
    logger.info(f"Adding {len(buses)} buses to the network.")

    n.madd("Bus", buses.index,
           Pd = buses.Pd,
           type = buses.type,
           v_nom = buses.baseKV,
           zone_id = buses.zone_id)

    return n


def add_branches_from_file(n, fn_branches):

    branches = pd.read_csv(fn_branches, index_col=0)

    for tech in ["Line", "Transformer"]:
        tech_branches = branches.query("branch_device_type == @tech")
        tech_branches.from_bus_id = tech_branches.from_bus_id.astype(str)
        logger.info(f"Adding {len(tech_branches)} branches as {tech}s to the network.")

        n.madd(tech,
               tech_branches.index,
               bus0 = tech_branches.from_bus_id,
               bus1 = tech_branches.to_bus_id,
               r = tech_branches.r,
               x = tech_branches.r,
               s_nom = tech_branches.rateA,
               v_nom = tech_branches.from_bus_id.map(n.buses.v_nom),
               interconnect = tech_branches.interconnect)

    n.lines.length = 1000 #FIX THIS->haversine??

    return n

def add_dclines_from_file(n, fn_dclines):

    dclines = pd.read_csv(fn_dclines, index_col=0)

    logger.info(f"Adding {len(dclines)} dc-lines as Links to the network.")

    n.madd("Link",
           dclines.index,
           bus0 = dclines.from_bus_id,
           bus1 = dclines.to_bus_id,
           p_nom = dclines.Pt)

    n.links.length = 1000 #FIX THIS->haversine??

    return n


def add_conventional_plants_from_file(n, fn_plants, renewable_techs):

    plants = pd.read_csv(fn_plants, index_col=0)
    plants = plants.query("type not in @renewable_techs")

    logger.info(f"Adding {len(plants)} conventional generators to the network.")

    n.madd("Generator", plants.index,
           bus=plants.bus_id,
           p_nom=plants.Pmax,
           marginal_cost=plants.GenFuelCost,
    )

    return n

def add_renewable_plants_from_file(n, fn_plants, renewable_techs):

    plants = pd.read_csv(fn_plants, index_col=0)

    for tech in renewable_techs:
        tech_plants = plants.query("type == @tech")
        tech_plants.index = tech_plants.index.astype(str)

        logger.info(f"Adding {len(tech_plants)} {tech} generators to the network.")

        p = pd.read_csv(snakemake.input[tech], index_col=0)
        p.index = n.snapshots
        p_max_pu = p.multiply(1./tech_plants.Pmax)

        n.madd("Generator", tech_plants.index,
               bus = tech_plants.bus_id,
               p_nom = tech_plants.Pmax,
               marginal_cost = tech_plants.GenFuelCost,
               p_max_pu = p_max_pu
        )

    return n

def add_demand_from_file(n, fn_demand):

    """
    Zone power demand is disaggregated to buses proportional to Pd,
    where Pd is the real power demand (MW).

    """

    demand = pd.read_csv(fn_demand, index_col=0)
    demand.index = n.snapshots
    demand.columns = demand.columns.astype(int)

    demand_per_bus_pu = (n.buses.set_index('zone_id').Pd
                         /n.buses.groupby('zone_id').sum().Pd)

    demand_per_bus = demand_per_bus_pu.multiply(demand)
    demand_per_bus.columns = n.buses.index

    n.madd("Load", demand_per_bus.columns,
           bus = demand_per_bus.columns,
           p_set = demand_per_bus
    )

    return n


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    #create network
    n = pypsa.Network()
    n.set_snapshots(
        pd.date_range(freq='h', start="2016-01-01", end="2017-01-01", closed='left')
    )

    #add buses, transformers, lines and links
    n = add_buses_from_file(n, snakemake.input['buses'])
    n = add_branches_from_file(n, snakemake.input['lines'])
    n = add_dclines_from_file(n, snakemake.input['links'])

    #add generators
    renewable_techs = snakemake.config['renewable_techs']
    n = add_conventional_plants_from_file(n, snakemake.input['plants'], renewable_techs)
    n = add_renewable_plants_from_file(n, snakemake.input['plants'], renewable_techs)

    #add load
    n = add_demand_from_file(n, snakemake.input['demand'])

    #export network
    n.export_to_netcdf(snakemake.output[0])
