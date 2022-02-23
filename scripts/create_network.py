# Copyright 2021-2022 Martha Frysztacki (KIT)

import pypsa
import pandas as pd

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

def annuity(m, r):
    if isinstance(r, pd.Series):
        return pd.Series(1 / m, index=r.index).where(r == 0, r / (1. - 1. / (1. + r) ** m))
    elif r > 0:
        return r / (1. - 1. / (1. + r) ** m)
    else:
        return 1 / m

def add_branches_from_file(n, fn_branches):

    branches = pd.read_csv(fn_branches, dtype={'from_bus_id': str}, index_col=0)

    for tech in ["Line", "Transformer"]:
        tech_branches = branches.query("branch_device_type == @tech")
        logger.info(f"Adding {len(tech_branches)} branches as {tech}s to the network.")

        n.madd(tech,
               tech_branches.index,
               bus0 = tech_branches.from_bus_id,
               bus1 = tech_branches.to_bus_id,
               r = tech_branches.r*(n.buses.loc[tech_branches.from_bus_id]['v_nom'].values**2),
               x = tech_branches.x*(n.buses.loc[tech_branches.from_bus_id]['v_nom'].values**2),
               b = tech_branches.b/(n.buses.loc[tech_branches.from_bus_id]['v_nom'].values**2),
               s_nom = tech_branches.rateA,
               v_nom = tech_branches.from_bus_id.map(n.buses.v_nom),
               interconnect = tech_branches.interconnect)
    return n

def add_dclines_from_file(n, fn_dclines):

    dclines = pd.read_csv(fn_dclines, index_col=0)

    logger.info(f"Adding {len(dclines)} dc-lines as Links to the network.")

    n.madd("Link",
           dclines.index,
           bus0 = dclines.from_bus_id,
           bus1 = dclines.to_bus_id,
           p_nom = dclines.Pt)

    return n


def add_conventional_plants_from_file(n, fn_plants, conventional_techs, costs):

    plants = pd.read_csv(fn_plants, index_col=0)
    plants.replace(['dfo','ng'],['oil','gas'],inplace=True)

    for tech in conventional_techs:
        tech_plants = plants.query("type == @tech")
        tech_plants.index = tech_plants.index.astype(str)

        logger.info(f"Adding {len(tech_plants)} {tech} generators to the network.")


        n.madd("Generator", tech_plants.index,
           bus=tech_plants.bus_id.astype(str),
           p_nom=tech_plants.Pmax,
           marginal_cost=costs.at[tech, 'marginal_cost']*1.14,
           p_nom_extendable=False,
           carrier = tech_plants.type,
           weight = 1.
        )

    return n

def load_costs(Nyears=1., tech_costs=None, config=None, elec_config=None):
    if tech_costs is None:
        tech_costs = snakemake.input.tech_costs

    if config is None:
        config = snakemake.config['costs']

    # set all asset costs and other parameters
    costs = pd.read_csv(tech_costs, index_col=list(range(3))).sort_index()

    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"),"value"] *= 1e3
    costs.loc[costs.unit.str.contains("USD"),"value"] *= config['USD2013_to_EUR2013']

    costs = (costs.loc[idx[:,config['year'],:], "value"]
             .unstack(level=2).groupby("technology").sum(min_count=1))

    costs = costs.fillna({"CO2 intensity" : 0,
                          "FOM" : 0,
                          "VOM" : 0,
                          "discount rate" : config['discountrate'],
                          "efficiency" : 1,
                          "fuel" : 0,
                          "investment" : 0,
                          "lifetime" : 25})

    costs["capital_cost"] = ((annuity(costs["lifetime"], costs["discount rate"]) +
                             costs["FOM"]/100.) *
                             costs["investment"] * Nyears['stores'])

    costs.at['OCGT', 'fuel'] = costs.at['gas', 'fuel']
    costs.at['CCGT', 'fuel'] = costs.at['gas', 'fuel']

    costs['marginal_cost'] = costs['VOM'] + costs['fuel'] / costs['efficiency']

    costs = costs.rename(columns={"CO2 intensity": "co2_emissions"})

    costs.at['OCGT', 'co2_emissions'] = costs.at['gas', 'co2_emissions']
    costs.at['CCGT', 'co2_emissions'] = costs.at['gas', 'co2_emissions']

    costs.at['solar', 'capital_cost'] = 0.5*(costs.at['solar-rooftop', 'capital_cost'] +
                                             costs.at['solar-utility', 'capital_cost'])

    def costs_for_storage(store, link1, link2=None, max_hours=1.):
        capital_cost = link1['capital_cost'] + max_hours * store['capital_cost']
        if link2 is not None:
            capital_cost += link2['capital_cost']
        return pd.Series(dict(capital_cost=capital_cost,
                              marginal_cost=0.,
                              co2_emissions=0.))

    if elec_config is None:
        elec_config = snakemake.config['electricity']
    max_hours = elec_config['max_hours']
    costs.loc["battery"] = \
        costs_for_storage(costs.loc["battery storage"], costs.loc["battery inverter"],
                          max_hours=max_hours['battery'])
    costs.loc["H2"] = \
        costs_for_storage(costs.loc["hydrogen storage"], costs.loc["fuel cell"],
                          costs.loc["electrolysis"], max_hours=max_hours['H2'])

    for attr in ('marginal_cost', 'capital_cost'):
        overwrites = config.get(attr)
        if overwrites is not None:
            overwrites = pd.Series(overwrites)
            costs.loc[overwrites.index, attr] = overwrites
    costs.rename(index={'onwind':'wind','offwind':'wind_offshore'},inplace=True)
    return costs

def add_renewable_plants_from_file(n, fn_plants, renewable_techs, costs):

    plants = pd.read_csv(fn_plants, index_col=0)

    for tech in renewable_techs:
        tech_plants = plants.query("type == @tech")
        tech_plants.index = tech_plants.index.astype(str)

        logger.info(f"Adding {len(tech_plants)} {tech} generators to the network.")

        if tech=="wind_offshore":
            p = pd.read_csv(snakemake.input["wind"], index_col=0)
        else:
            p = pd.read_csv(snakemake.input[tech], index_col=0)

        p.index = n.snapshots
        p_max_pu = p.multiply(1./tech_plants.Pmax)

        n.madd("Generator", tech_plants.index,
               bus = tech_plants.bus_id,
               p_nom_min = tech_plants.Pmax, #I forget what Tom said last time, but if we want to make it extendable for renewable units, this p should be min. Otherwise, the capacity will be cut to minimise the objective function.
               marginal_cost = costs.at[tech, 'marginal_cost']*1.14,
               capital_cost = costs.at[tech, 'capital_cost']*1.14, #divide or multiply the currency to make it the same as marginal cost
               p_max_pu = p_max_pu,
               p_nom_extendable = True,
               carrier = tech,
               weight = 1.
        )

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
    Nyears = n.snapshot_weightings.sum() / 8784.
    costs = load_costs(Nyears)

    #add buses, transformers, lines and links
    n = add_buses_from_file(n, snakemake.input['buses'])

    n = add_branches_from_file(n, snakemake.input['lines'])
    n = add_dclines_from_file(n, snakemake.input['links'])

    #add generators
    renewable_techs = snakemake.config['renewable_techs']
    conventional_techs = snakemake.config['conventional_techs']
    n = add_conventional_plants_from_file(n, snakemake.input['plants'], conventional_techs, costs)
    n = add_renewable_plants_from_file(n, snakemake.input['plants'], renewable_techs, costs)

    #add load
    n = add_demand_from_file(n, snakemake.input['demand'])

    #export network
    n.export_to_netcdf(snakemake.output[0])
