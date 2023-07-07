# Copyright 2021-2022 Martha Frysztacki (KIT)
# Modified by Kamran Tehranchi (Stanford University)

import pypsa, pandas as pd, logging, geopandas as gpd
from geopandas.tools import sjoin


idx = pd.IndexSlice

def add_buses_from_file(n, buses, interconnect):

    if interconnect != "usa":
        buses = buses.query(
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
        balancing_area= buses.balancing_area,
    )

    return n

def add_branches_from_file(n, fn_branches):

    branches = pd.read_csv(
        fn_branches, dtype={"from_bus_id": str, "to_bus_id": str}, index_col=0
    ).query("from_bus_id in @n.buses.index and to_bus_id in @n.buses.index")

    for tech in ["Line", "Transformer"]:
        tech_branches = branches.query("branch_device_type == @tech")
        logger.info(f"Adding {len(tech_branches)} branches as {tech}s to the network.")

        # S_base = 100 MVA
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

def assign_bus_ba(PATH_BUS, PATH_BA_SHP, PATH_OFFSHORE_SHP, bus_locs):
    bus_df = pd.read_csv(PATH_BUS, index_col=0)
    bus_locs["geometry"] = gpd.points_from_xy(bus_locs["lon"], bus_locs["lat"])
    bus_df_locs = pd.merge(bus_df, bus_locs['geometry'], left_index=True, right_index=True, how='left') #merging bus data w/ geometry data

    ba_region_shapes = gpd.read_file(PATH_BA_SHP)
    offshore_shapes = gpd.read_file(PATH_OFFSHORE_SHP)
    combined_shapes = gpd.GeoDataFrame(pd.concat([ba_region_shapes, offshore_shapes],ignore_index=True))
    
    ba_points = sjoin(gpd.GeoDataFrame(bus_df_locs["geometry"],crs= 4326), combined_shapes, how='left',predicate='within')
    ba_points = ba_points.rename(columns={'name':'balancing_area'})
    bus_df_final = pd.merge(bus_df, ba_points['balancing_area'], left_index=True, right_index=True,how='left')
    # import pdb; pdb.set_trace()
    #for identifying duplicants-- below
    # df = bus_df_final.reset_index().groupby(['bus_id']).size().reset_index(name='count').sort_values('count')
    # df_issues = df[df['count']>1]
    # bus_df_final.loc[df_issues.bus_id]
    return bus_df_final

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    # create network
    n = pypsa.Network()

    interconnect = snakemake.wildcards.interconnect
    # interconnect in raw data given with an uppercase first letter
    if interconnect != "usa":
        interconnect = interconnect[0].upper() + interconnect[1:]

    #assign locations and balancing authorities to buses
    bus2sub = pd.read_csv(snakemake.input.bus2sub).set_index("bus_id")
    sub = pd.read_csv(snakemake.input.sub).set_index("sub_id")
    buslocs = pd.merge(bus2sub, sub, left_on="sub_id", right_index=True)

    bus_df = assign_bus_ba(snakemake.input["buses"], snakemake.input["onshore_shapes"],snakemake.input["offshore_shapes"],buslocs)

    # add buses, transformers, lines and links
    n = add_buses_from_file(n, bus_df, interconnect=interconnect)
    n = add_branches_from_file(n, snakemake.input["lines"])
    n = add_dclines_from_file(n, snakemake.input["links"])
    add_custom_line_type(n)

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
    # import pdb; pdb.set_trace()

    # export network
    n.export_to_netcdf(snakemake.output.network)
