'''
Building Multi-Period Network from Breakthrough 

'''

import pypsa, os, sys, numpy as np, pandas as pd, logging
from itertools import product
import geopandas as gpd
import networkx as nx
from _helpers import configure_logging
from shapely.geometry import LineString, Point
import pdb
import logging
from add_electricity import load_costs, _add_missing_carriers_from_costs

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def add_buses_from_file(n, fn_buses, interconnect="Western"):

    buses = pd.read_csv(fn_buses, index_col=0)
    if interconnect != "usa":
        buses = pd.read_csv(fn_buses, index_col=0).query(
            "interconnect == @interconnect"
        )

    logger.info(f"Adding {len(buses)} buses to the network.")

    n.madd(
        "Bus",
        buses.index,
        Pd=buses.Pd, # used to decompose zone demand to bus demand
        # type = buses.type # do we need this? 
        v_nom=buses.baseKV,
        zone_id=buses.zone_id,
    )

    return n


def add_branches_from_file(n, fn_branches):

    branches = pd.read_csv(
        fn_branches, dtype={"from_bus_id": str, "to_bus_id": str}, index_col=0
    ).query("from_bus_id in @n.buses.index and to_bus_id in @n.buses.index")

    for tech in ["Line", "Transformer"]:
        tech_branches = branches.query("branch_device_type == @tech")
        logger.info(f"Adding {len(tech_branches)} branches as {tech}s to the network.")

        # Why are the resistance, reactance, and susceptance scaled by Vnom^2?
        n.madd( 
            tech,
            tech_branches.index,
            bus0=tech_branches.from_bus_id,
            bus1=tech_branches.to_bus_id,
            r=tech_branches.r
            * (n.buses.loc[tech_branches.from_bus_id]["v_nom"].values ** 2)
            / 100,
            x=tech_branches.x
            * (n.buses.loc[tech_branches.from_bus_id]["v_nom"].values ** 2)
            / 100,
            b=tech_branches.b
            / (n.buses.loc[tech_branches.from_bus_id]["v_nom"].values ** 2),
            s_nom=tech_branches.rateA,
            v_nom=tech_branches.from_bus_id.map(n.buses.v_nom),
            interconnect=tech_branches.interconnect,
            type="Rail",
        )

    return n


def add_custom_line_type(n):

    n.line_types.loc["Rail"] = pd.Series(
        [60, 0.0683, 0.335, 15, 1.01],
        index=["f_nom", "r_per_length", "x_per_length", "c_per_length", "i_nom"],
    )


def add_dclines_from_file(n, fn_dclines):

    dclines = pd.read_csv(
        fn_dclines, dtype={"from_bus_id": str, "to_bus_id": str}, index_col=0
    ).query("from_bus_id in @n.buses.index and to_bus_id in @n.buses.index")

    logger.info(f"Adding {len(dclines)} dc-lines as Links to the network.")

    n.madd(
        "Link",
        dclines.index,
        bus0=dclines.from_bus_id,
        bus1=dclines.to_bus_id,
        p_nom=dclines.Pt,
        carrier="DC",
        underwater_fraction=0.0, #DC line in bay is underwater, but does network have this line?
    )

    return n


def add_conventional_plants_from_file(
    n, fn_plants, conventional_carriers, extendable_carriers, costs):

    _add_missing_carriers_from_costs(n, costs, conventional_carriers)

    plants = pd.read_csv(fn_plants, dtype={"bus_id": str}, index_col=0).query(
        "bus_id in @n.buses.index"
    )
    plants.replace(["dfo"], ["oil"], inplace=True)

    for tech in conventional_carriers:
        tech_plants = plants.query("type == @tech")
        tech_plants.index = tech_plants.index.astype(str)

        logger.info(f"Adding {len(tech_plants)} {tech} generators to the network.")

        if tech in extendable_carriers:
            p_nom_extendable = True
        else:
            p_nom_extendable = False

        n.madd(
            "Generator",
            tech_plants.index,
            bus=tech_plants.bus_id.astype(str),
            p_nom=tech_plants.Pmax,
            p_nom_extendable=p_nom_extendable,
            marginal_cost=tech_plants.GenIOB
            * tech_plants.GenFuelCost
            / 1.14,  # divide or multiply the currency to make it the same as marginal cost
            carrier=tech_plants.type,
            weight=1.0,
            efficiency=costs.at[tech, "efficiency"],
        )

    return n


def add_renewable_plants_from_file(
    n, fn_plants, renewable_carriers, extendable_carriers, costs):

    _add_missing_carriers_from_costs(n, costs, renewable_carriers)

    plants = pd.read_csv(fn_plants, dtype={"bus_id": str}, index_col=0).query(
        "bus_id in @n.buses.index"
    )
    plants.replace(["wind_offshore"], ["offwind"], inplace=True)

    for tech in renewable_carriers:
        tech_plants = plants.query("type == @tech")
        tech_plants.index = tech_plants.index.astype(str)

        logger.info(f"Adding {len(tech_plants)} {tech} generators to the network.")

        if tech in ["wind", "offwind"]: #is this going to double add wind?
            p = pd.read_csv(snakemake.input["wind"], index_col=0)
        else:
            p = pd.read_csv(snakemake.input[tech], index_col=0)
        intersection = set(p.columns).intersection(tech_plants.index)
        p = p[list(intersection)]

        p.index = n.snapshots
        p.columns = p.columns.astype(str)

        if (tech_plants.Pmax == 0).any():
            # p_nom is the maximum of {Pmax, dispatch}
            p_nom = pd.concat([p.max(axis=0), tech_plants["Pmax"]], axis=1).max(axis=1)
            p_max_pu = (p[p_nom.index] / p_nom).fillna(0)  # some values remain 0
        else:
            p_nom = tech_plants.Pmax
            p_max_pu = p[tech_plants.index] / p_nom

        if tech in extendable_carriers:
            p_nom_extendable = True
        else:
            p_nom_extendable = False

        n.madd(
            "Generator",
            tech_plants.index,
            bus=tech_plants.bus_id,
            p_nom_min=p_nom,
            p_nom=p_nom,
            marginal_cost=tech_plants.GenIOB * tech_plants.GenFuelCost / 1.14,
            capital_cost=costs.at[tech, "capital_cost"],
            p_max_pu=p_max_pu,
            p_nom_extendable=p_nom_extendable,
            carrier=tech,
            weight=1.0,
            efficiency=costs.at[tech, "efficiency"],
        )

    # hack to remove generators without capacity (required for SEG to work)
    # shouldn't exist, in fact...
    p_max_pu_norm = n.generators_t.p_max_pu.max()
    remove_g = p_max_pu_norm[p_max_pu_norm == 0.0].index
    logger.info(
        f"removing {len(remove_g)} {tech} generators {remove_g} with no renewable potential."
    )
    n.mremove("Generator", remove_g)

    return n


def add_demand_from_file(n, fn_demand):

    """
    Zone power demand is disaggregated to buses proportional to Pd,
    where Pd is the real power demand (MW).
    """

    demand = pd.read_csv(fn_demand, index_col=0)
    # zone_id is int, therefore demand.columns should be int first
    demand.columns = demand.columns.astype(int)
    demand.index = n.snapshots

    intersection = set(demand.columns).intersection(n.buses.zone_id.unique())
    demand = demand[list(intersection)]

    demand_per_bus_pu = (
        n.buses.set_index("zone_id").Pd / n.buses.groupby("zone_id").sum().Pd
    )

    demand_per_bus = demand_per_bus_pu.multiply(demand)

    demand_per_bus.columns = n.buses.index

    n.madd(
        "Load", demand_per_bus.columns, bus=demand_per_bus.columns, p_set=demand_per_bus
    )

    return n


def add_multi_period_demand(n, investment_periods, load_forecasts):

    # loads_prev = n.loads_t.p_set.iloc[:8760,:]
    # gen_prev = n.generators_t.p_max_pu.iloc[:8760,:] 
    loads_df = pd.DataFrame()
    # generators_df = pd.DataFrame()
    for year in investment_periods:
        demand_index = n.snapshots[n.snapshots.get_level_values(0) == year]
        # load = loads_prev * 1.05
        load.index = demand_index
        loads_df = loads_df.append(load)
        # gen = gen_prev * 1.05
        # gen.index = demand_index
        # generators_df = generators_df.append(gen)
        # load_prev = load
        # gen_prev = gen

    # adding load and renewable gen timeseries data back to the network
    n.madd("Load", loads_df.columns, bus=loads_df.columns, p_set=loads_df )
    n.generators_t.p_max_pu = generators_df
    n.generators_t.p_max_pu 
    n.export_to_netcdf(multi_period_path)




if __name__ == "__main__":

    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("create_multi_period_network", **{'interconnect':'western'})
    configure_logging(snakemake)
    
    # create network
    n = pypsa.Network()

    # create investment period snapshots
    investment_periods = snakemake.config["planning_horizons"]
    snapshots = pd.DatetimeIndex([])
    for year in investment_periods:
        period = pd.date_range(
            start="{}-01-01 00:00".format(year),
            freq="1H",
            periods=8760 / float(1),
        )
        snapshots = snapshots.append(period)

    # convert snapshots to multiindex and assign to network
    n.snapshots = pd.MultiIndex.from_arrays([snapshots.year, snapshots])
    n.investment_periods = investment_periods
    n.snapshots

    n.investment_period_weightings["years"] = list(np.diff(investment_periods)) + [5]
    r = snakemake.config["investment_period_discount_rate"]
    T = 0 # Starting time... change if first period is not current year
    for period, nyears in n.investment_period_weightings.years.items():
        discounts = [(1 / (1 + r) ** t) for t in range(T, T + nyears)]
        n.investment_period_weightings.at[period, "objective"] = sum(discounts)
        T += nyears
    n.investment_period_weightings


    # add investment year load data
    add_multi_period_demand(n, investment_periods, snakemake.input.load_forecasts)


    ##############
    # n.set_snapshots(
    #     pd.date_range(freq="h", start="2016-01-01", end="2017-01-01", closed="left")
    # )

    # attach load costs
    Nyears = n.snapshot_weightings.generators.sum() / 8784.0
    costs = load_costs(
        snakemake.input.tech_costs,
        snakemake.config["costs"],
        snakemake.config["electricity"],
        Nyears,
    )

    # should renaming technologies move to config.yaml?
    costs = costs.rename(index={"onwind": "wind", "OCGT": "ng"})

    interconnect = snakemake.wildcards.interconnect
    # interconnect in raw data given with an uppercase first letter
    if interconnect != "usa":
        interconnect = interconnect[0].upper() + interconnect[1:]

    # add buses, transformers, lines and links
    n = add_buses_from_file(n, snakemake.input["buses"], interconnect=interconnect)
    n = add_branches_from_file(n, snakemake.input["lines"])
    n = add_dclines_from_file(n, snakemake.input["links"])
    add_custom_line_type(n)

    # add renewable generators
    renewable_carriers = list(
        set(snakemake.config["allowed_carriers"]).intersection(
            set(["wind", "solar", "offwind", "hydro"])
        )
    )
    n = add_renewable_plants_from_file(
        n,
        snakemake.input["plants"],
        renewable_carriers,
        snakemake.config["extendable_carriers"],
        costs,
    )

    # add conventional generators
    conventional_carriers = list(
        set(snakemake.config["allowed_carriers"]).intersection(
            set(["coal", "ng", "nuclear", "oil", "geothermal"])
        )
    )
    n = add_conventional_plants_from_file(
        n,
        snakemake.input["plants"],
        conventional_carriers,
        snakemake.config["extendable_carriers"],
        costs,
    )

    # add load
    n = add_demand_from_file(n, snakemake.input["demand"])

    # export bus2sub interconnect data
    logger.info(f"exporting bus2sub and sub data for {interconnect}")
    if interconnect == "usa": #if usa interconnect do not filter bc all sub are in usa
        bus2sub = (
            pd.read_csv(snakemake.input.bus2sub)
            .set_index("bus_id")
        )
        bus2sub.to_csv(snakemake.output.bus2sub)
    else:
        bus2sub = (
            pd.read_csv(snakemake.input.bus2sub)
            .query("interconnect == @interconnect")
            .set_index("bus_id")
        )
        bus2sub.to_csv(snakemake.output.bus2sub)

    # export sub interconnect data
    if interconnect == "usa": #if usa interconnect do not filter bc all sub are in usa
        sub = (
            pd.read_csv(snakemake.input.sub)
            .set_index("sub_id")
        )
        sub.to_csv(snakemake.output.sub)
    else:
        sub = (
            pd.read_csv(snakemake.input.sub)
            .query("interconnect == @interconnect")
            .set_index("sub_id")
        )
        sub.to_csv(snakemake.output.sub)

    # export network
    n.export_to_netcdf(snakemake.output.network)



############################################################################################################




n = pypsa.Network(clustered_path)
years = [2020, 2025, 2030, 2035, 2040, 2045]
freq = "1" #hourly data

snapshots = pd.DatetimeIndex([])
for year in years:
    period = pd.date_range(
        start="{}-01-01 00:00".format(year),
        freq="{}H".format(freq),
        periods=8760 / float(freq),
    )
    snapshots = snapshots.append(period)

# convert to multiindex and assign to network
n.snapshots = pd.MultiIndex.from_arrays([snapshots.year, snapshots])
n.investment_periods = years
n.snapshots

n.investment_period_weightings["years"] = list(np.diff(years)) + [5]
r = 0.01
T = 0
for period, nyears in n.investment_period_weightings.years.items():
    discounts = [(1 / (1 + r) ** t) for t in range(T, T + nyears)]
    n.investment_period_weightings.at[period, "objective"] = sum(discounts)
    T += nyears
n.investment_period_weightings

loads_prev = network.loads_t.p_set.iloc[:8760,:]
gen_prev = network.generators_t.p_max_pu.iloc[:8760,:] 
loads_df = pd.DataFrame()
generators_df = pd.DataFrame()
for year in years:
    demand_index = n.snapshots[n.snapshots.get_level_values(0) == year]
    load = loads_prev * 1.05
    load.index = demand_index
    loads_df = loads_df.append(load)
    gen = gen_prev * 1.05
    gen.index = demand_index
    generators_df = generators_df.append(gen)
    load_prev = load
    gen_prev = gen

# loads_df
n.madd("Load", loads_df.columns, bus=loads_df.columns, p_set=loads_df )

n.generators_t.p_max_pu = generators_df
n.generators_t.p_max_pu 


n.export_to_netcdf(multi_period_path)
