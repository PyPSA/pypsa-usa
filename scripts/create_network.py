# Copyright 2021-2022 Martha Frysztacki (KIT)

import pypsa
import pandas as pd

import sys
sys.path.append(snakemake.config['subworkflow'] + "scripts/")
from add_electricity import load_costs, _add_missing_carriers_from_costs

import logging

idx = pd.IndexSlice

def add_buses_from_file(n, fn_buses):

    buses = pd.read_csv(fn_buses, index_col=0)
    logger.info(f"Adding {len(buses)} buses to the network.")

    n.madd("Bus", buses.index,
           Pd = buses.Pd,
           #type = buses.type, # do we need this?
           v_nom = buses.baseKV,
           zone_id = buses.zone_id)

    return n


def add_branches_from_file(n, fn_branches):

    branches = pd.read_csv(fn_branches, dtype={'from_bus_id': str}, index_col=0)

    for tech in ["Line", "Transformer"]:
        tech_branches = branches.query("branch_device_type == @tech")
        logger.info(f"Adding {len(tech_branches)} branches as {tech}s to the network.")

        n.madd(tech,
               tech_branches.index,
               bus0 = tech_branches.from_bus_id,
               bus1 = tech_branches.to_bus_id,
               r = tech_branches.r*(n.buses.loc[tech_branches.from_bus_id]['v_nom'].values**2)/100,
               x = tech_branches.x*(n.buses.loc[tech_branches.from_bus_id]['v_nom'].values**2)/100,
               b = tech_branches.b/(n.buses.loc[tech_branches.from_bus_id]['v_nom'].values**2),
               s_nom = tech_branches.rateA,
               v_nom = tech_branches.from_bus_id.map(n.buses.v_nom),
               interconnect = tech_branches.interconnect,
               type='Rail'
        )
    return n


def add_custom_line_type(n):

    n.line_types.loc['Rail'] = pd.Series(
        [60, 0.0683, 0.335, 15, 1.01],
        index=['f_nom','r_per_length','x_per_length','c_per_length','i_nom']
    )


def add_dclines_from_file(n, fn_dclines):

    dclines = pd.read_csv(fn_dclines, index_col=0)

    logger.info(f"Adding {len(dclines)} dc-lines as Links to the network.")

    n.madd("Link",
           dclines.index,
           bus0 = dclines.from_bus_id,
           bus1 = dclines.to_bus_id,
           p_nom = dclines.Pt,
           carrier = "DC",
           underwater_fraction = 0.,
    )

    return n


def add_conventional_plants_from_file(n, fn_plants, conventional_techs, costs):

    _add_missing_carriers_from_costs(n, costs, conventional_techs)

    plants = pd.read_csv(fn_plants, index_col=0)
    plants.replace(['dfo','ng'],['oil','OCGT'],inplace=True)

    for tech in conventional_techs:
        tech_plants = plants.query("type == @tech")
        tech_plants.index = tech_plants.index.astype(str)

        logger.info(f"Adding {len(tech_plants)} {tech} generators to the network.")

        n.madd("Generator", tech_plants.index,
           bus=tech_plants.bus_id.astype(str),
           p_nom=tech_plants.Pmax,
           marginal_cost=tech_plants.GenIOB*tech_plants.GenFuelCost/1.14, #divide or multiply the currency to make it the same as marginal cost
           p_nom_extendable=False,
           carrier = tech_plants.type,
           weight = 1.,
           efficiency = costs.at[tech, 'efficiency']
        )

    return n


def add_renewable_plants_from_file(n, fn_plants, renewable_techs, extendable_techs, costs):

    _add_missing_carriers_from_costs(n, costs, renewable_techs)

    plants = pd.read_csv(fn_plants, index_col=0)
    plants.replace(['wind','wind_offshore'], ['onwind','offwind'], inplace=True)

    for tech in renewable_techs:
        tech_plants = plants.query("type == @tech")
        tech_plants.index = tech_plants.index.astype(str)

        logger.info(f"Adding {len(tech_plants)} {tech} generators to the network.")

        if tech in ["onwind", "offwind"]:
            p = pd.read_csv(snakemake.input["wind"], index_col=0)
        else:
            p = pd.read_csv(snakemake.input[tech], index_col=0)

        p.index = n.snapshots

        if (tech_plants.Pmax==0).any():
            #p_nom is the maximum of {Pmax, dispatch}
            p_nom = (pd.concat([p.max(axis=0), tech_plants['Pmax']], axis=1)
                     .max(axis=1))
            p_max_pu = (p[p_nom.index] / p_nom).fillna(0) #some values remain 0
        else:
            p_nom = tech_plants.Pmax
            p_max_pu = p[tech_plants.index] / p_nom

        if tech in extendable_techs:
            p_nom_extendable = True
        else:
            p_nom_extendable = False

        n.madd("Generator", tech_plants.index,
               bus=tech_plants.bus_id,
               p_nom_min=p_nom,
               p_nom=p_nom,
               marginal_cost=tech_plants.GenIOB*tech_plants.GenFuelCost/1.14,
               capital_cost=costs.at[tech, 'capital_cost'],
               p_max_pu=p_max_pu,
               p_nom_extendable=p_nom_extendable,
               carrier=tech,
               weight=1.,
               efficiency = costs.at[tech, 'efficiency']
        )

    # hack to remove generators without capacity (required for SEG to work)
    # shouldn't exist, in fact...
    p_max_pu_norm = n.generators_t.p_max_pu.max()
    remove_g = p_max_pu_norm[p_max_pu_norm == 0.].index
    logger.info(f"removing {len(remove_g)} {tech} generators {remove_g} with no renewable potential.")
    n.mremove("Generator", remove_g)

    return n

def add_demand_from_file(n, fn_demand):

    """
    Zone power demand is disaggregated to buses proportional to Pd,
    where Pd is the real power demand (MW).
    """

    demand = pd.read_csv(fn_demand, index_col=0)
    demand.index = n.snapshots
    #zone_id is int, therefore demand.columns should be int first
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

    #attach load costs
    Nyears = n.snapshot_weightings.generators.sum() / 8784.
    costs = load_costs(
        snakemake.input.tech_costs, snakemake.config['costs'],
        snakemake.config['electricity'], Nyears
    )

    #add buses, transformers, lines and links
    n = add_buses_from_file(n, snakemake.input['buses'])
    n = add_branches_from_file(n, snakemake.input['lines'])
    n = add_dclines_from_file(n, snakemake.input['links'])
    add_custom_line_type(n)

    #add generators
    renewable_techs = snakemake.config['renewable_techs']
    conventional_techs = snakemake.config['conventional_techs']
    n = add_conventional_plants_from_file(n, snakemake.input['plants'], conventional_techs, costs)
    n = add_renewable_plants_from_file(
        n, snakemake.input['plants'],  renewable_techs, snakemake.config["extendable_techs"], costs
    )

    #add load
    n = add_demand_from_file(n, snakemake.input['demand'])

    #export network
    n.export_to_netcdf(snakemake.output[0])
