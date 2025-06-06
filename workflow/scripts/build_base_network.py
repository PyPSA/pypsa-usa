"""Builds base pypsa network with lines, buses, transformers."""

# BY PyPSA-USA Authors
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging
from build_shapes import load_na_shapes
from shapely.geometry import Polygon
from sklearn.neighbors import BallTree


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth
    (specified in decimal degrees).

    All args must be of equal length.
    source: https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6378.137 * c
    return km


def add_buses_from_file(
    n: pypsa.Network,
    buses: gpd.GeoDataFrame,
    interconnect: str,
) -> pypsa.Network:
    if interconnect != "usa":
        buses = buses.query(
            "interconnect == @interconnect",
        )

    logger.info(f"Adding {len(buses)} buses to the network.")

    n.madd(
        "Bus",
        buses.index,
        Pd=buses.Pd,  # used to decompose zone demand to bus demand
        v_nom=buses.baseKV,
        balancing_area=buses.balancing_area,
        state=buses.state,
        country=buses.country,
        county=buses.county,
        reeds_zone=buses.reeds_zone,
        reeds_ba=buses.reeds_ba,
        interconnect=buses.interconnect,
        x=buses.lon,
        y=buses.lat,
        sub_id=buses.sub_id.astype(int),
        substation_off=False,
        poi=False,
        LAF_state=buses.LAF_state,
    )

    n.buses.loc[n.buses.sub_id.astype(int) >= 41012, "substation_off"] = True  # mark offshore buses
    return n


def add_branches_from_file(n: pypsa.Network, fn_branches: str) -> pypsa.Network:
    branches = pd.read_csv(
        fn_branches,
        dtype={"from_bus_id": str, "to_bus_id": str},
        index_col=0,
    ).query("from_bus_id in @n.buses.index and to_bus_id in @n.buses.index")
    branches.loc[branches.rateA == 0, "rateA"] = 0.01

    for tech in ["Line", "Transformer"]:
        tech_branches = branches.query("branch_device_type == @tech")
        logger.info(f"Adding {len(tech_branches)} branches as {tech}s to the network.")

        # S_base = 100 MVA
        n.madd(
            tech,
            tech_branches.index,
            bus0=tech_branches.from_bus_id,
            bus1=tech_branches.to_bus_id,
            r=tech_branches.r * (n.buses.loc[tech_branches.from_bus_id]["v_nom"].values ** 2) / 100,
            x=tech_branches.x * (n.buses.loc[tech_branches.from_bus_id]["v_nom"].values ** 2) / 100,
            b=tech_branches.b / (n.buses.loc[tech_branches.from_bus_id]["v_nom"].values ** 2),
            s_nom=tech_branches.rateA,
            v_nom=tech_branches.from_bus_id.map(n.buses.v_nom),
            interconnect=tech_branches.interconnect,
            type="temp",  # temporarily then over ridden by assign_line_types
            carrier="AC",
            underwater_fraction=0.0,
        )
    return n


def add_custom_line_type(n: pypsa.Network):
    n.line_types.loc["temp"] = pd.Series(
        [60, 0.0683, 0.335, 15, 1.01],
        index=["f_nom", "r_per_length", "x_per_length", "c_per_length", "i_nom"],
    )


def assign_line_types(n: pypsa.Network):
    n.lines.type = n.lines.v_nom.map(snakemake.config["lines"]["types"])


def add_dclines_from_file(n: pypsa.Network, fn_dclines: str) -> pypsa.Network:
    dclines = pd.read_csv(
        fn_dclines,
        dtype={"from_bus_id": str, "to_bus_id": str},
        index_col=0,
    ).query("from_bus_id in @n.buses.index and to_bus_id in @n.buses.index")

    logger.info(f"Adding {len(dclines)} dc-lines as Links to the network.")

    n.madd(
        "Link",
        dclines.index,
        bus0=dclines.from_bus_id,
        bus1=dclines.to_bus_id,
        p_nom=dclines.Pt,
        carrier="DC",
        underwater_fraction=0.0,  # DC line in bay is underwater, but does network have this line?
    )

    return n


def assign_sub_id(buses: pd.DataFrame, bus_locs: pd.DataFrame) -> pd.DataFrame:
    """Adds sub id to dataframe as a new column."""
    buses["sub_id"] = bus_locs.sub_id
    return buses


def assign_bus_location(buses: pd.DataFrame, buslocs: pd.DataFrame) -> gpd.GeoDataFrame:
    """Attaches coordinates and sub ids to each bus."""
    gdf_bus = pd.merge(
        buses,
        buslocs[["lat", "lon"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    gdf_bus["geometry"] = gpd.points_from_xy(gdf_bus["lon"], gdf_bus["lat"])
    return gpd.GeoDataFrame(gdf_bus, crs=4326)


def map_bus_to_region(
    buses: gpd.GeoDataFrame,
    shape: gpd.GeoDataFrame,
    names: str,
) -> gpd.GeoDataFrame:
    """
    Maps a bus to a geographic region.

    Args:
        buses: gpd.GeoDataFrame,
        shape: gpd.GeoDataFrame,
        name: str
            column name in shape to merge
    """
    names.append("geometry")
    shape_filtered = shape[names]
    return gpd.sjoin(buses, shape_filtered, how="left").drop(columns=["index_right"])


def assign_line_length(n: pypsa.Network):
    """Assigns line length to each line in the network using Haversine distance."""
    bus_df = n.buses[["x", "y"]]
    bus0 = bus_df.loc[n.lines.bus0].values
    bus1 = bus_df.loc[n.lines.bus1].values
    distances = haversine_np(bus0[:, 0], bus0[:, 1], bus1[:, 0], bus1[:, 1])
    n.lines["length"] = distances


def create_grid(polygon, cell_size):
    """
    Creates a grid of square cells over a given polygon and returns the centers
    of the cells.

    :param polygon: A Shapely Polygon object.
    :param cell_size: The length of each side of the square cells.
    :return: List of (latitude, longitude) tuples for the center of each
        cell.
    """
    minx, miny, maxx, maxy = polygon.bounds
    grid_cells = []

    # Create the grid cells
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            grid_cells.append(
                Polygon(
                    [
                        (x, y),
                        (x + cell_size, y),
                        (x + cell_size, y + cell_size),
                        (x, y + cell_size),
                    ],
                ),
            )
            y += cell_size
        x += cell_size

    # Convert to a GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells)

    # Filter to only those cells that intersect the polygon
    intersecting_cells = grid_gdf[grid_gdf.intersects(polygon)]

    # Find the center of each cell
    centers = intersecting_cells.geometry.centroid

    # Remove centroids that are outside of the polygon
    centers = [center for center in centers if center.within(polygon)]

    # Return the coordinates of the centers
    return [(center.y, center.x) for center in centers]


def build_offshore_buses(
    offshore_shapes: gpd.GeoDataFrame,
    offshore_spacing: int,
) -> pd.DataFrame:
    """
    Build dataframe of offshore buses by creating evenly spaced grid cells inside of the offshore shapes.

    Args:
        offshore_shapes: gpd.GeoDataFrame
            Offshore shapes to create grid cells inside of.
        offshore_spacing: int
            Spacing between grid cells in meters.
    """
    offshore_buses = pd.DataFrame()
    offshore_shapes = offshore_shapes.to_crs("EPSG:5070")
    for shape in offshore_shapes.geometry:
        cell_centers = create_grid(shape, offshore_spacing)
        cell_centers = pd.DataFrame(cell_centers, columns=["lat", "lon"])
        offshore_buses = pd.concat([offshore_buses, cell_centers], ignore_index=True)
    # reproject back to EPSG:4326
    offshore_buses = gpd.GeoDataFrame(
        offshore_buses,
        geometry=gpd.points_from_xy(offshore_buses.lon, offshore_buses.lat),
        crs="EPSG:5070",
    )
    offshore_buses = offshore_buses.to_crs("EPSG:4326")
    offshore_buses.lat = offshore_buses.geometry.y
    offshore_buses.lon = offshore_buses.geometry.x
    offshore_buses["sub_id"] = np.arange(50000, 50000 + len(offshore_buses))  # replace with onshore sub_ids
    offshore_buses.index = np.arange(3008161, 3008161 + len(offshore_buses))
    return offshore_buses


def add_offshore_buses(n: pypsa.Network, offshore_buses: pd.DataFrame) -> pypsa.Network:
    """Add offshore buses to network and assigns it a POI."""
    n.madd(
        "Bus",
        offshore_buses.index,
        Pd=0,
        v_nom=230,
        balancing_area="Offshore",
        state="Offshore",
        country="US",
        interconnect="Offshore",
        x=offshore_buses.lon,
        y=offshore_buses.lat,
        substation_off=True,
        poi_sub=False,
        poi_bus=False,
    )

    # assigns offshore buses to an onshore POI as sub_id
    poi_buses = n.buses.loc[n.buses.poi_sub]  # identify the buses at the POI
    eligible_poi_buses = poi_buses.loc[poi_buses.groupby("sub_id")["v_nom"].idxmax()]
    offshore_buses_matched = match_missing_buses(eligible_poi_buses, n.buses.loc[n.buses.substation_off])

    # Reassign offshore buses region attributes from their POI bus assignments
    region_attrs = ["sub_id", "balancing_area", "state", "country", "interconnect", "reeds_zone", "reeds_ba", "county"]
    for attr in region_attrs:
        n.buses.loc[offshore_buses_matched.index, attr] = n.buses.loc[
            offshore_buses_matched.bus_assignment,
            attr,
        ].values
    return n


def assign_texas_poi(n: pypsa.Network) -> pypsa.Network:
    tx_pois_sub = [40893, 40894, 40729, 40908, 40854, 40760, 40776]
    tx_pois_bus = [3007405, 3007407, 3007096, 3007429, 3007321, 3007171, 3007205]
    n.buses.loc[n.buses.sub_id.isin(tx_pois_sub), "poi_sub"] = True
    n.buses.loc[n.buses.sub_id.isin(tx_pois_bus), "poi_bus"] = True
    return n


def identify_osw_poi(n: pypsa.Network) -> pypsa.Network:
    """Identify offshore wind points of interconnections in the base network."""
    offshore_lines = n.lines.loc[n.lines.bus0.isin(n.buses.loc[n.buses.substation_off].index)]
    poi_bus_ids = offshore_lines.bus1.unique()
    poi_sub_ids = n.buses.loc[poi_bus_ids, "sub_id"].unique()
    n.buses.loc[n.buses.index.isin(poi_bus_ids), "poi_bus"] = True
    n.buses.loc[n.buses.sub_id.isin(poi_sub_ids), "poi_sub"] = True
    n.buses.poi_bus = n.buses.poi_bus.astype(bool).fillna(False)
    n.buses.poi_sub = n.buses.poi_sub.astype(bool).fillna(False)
    return n


def match_missing_buses(buses_to_match_to, missing_buses):
    """Match buses missing region assignment to their nearest bus."""
    missing_buses = missing_buses.copy()
    missing_buses["bus_assignment"] = None

    buses_to_match_to["geometry"] = gpd.points_from_xy(
        buses_to_match_to["x"],
        buses_to_match_to["y"],
    )

    # from: https://stackoverflow.com/questions/58893719/find-nearest-point-in-other-dataframe-with-a-lot-of-data
    # Create a BallTree
    tree = BallTree(buses_to_match_to[["x", "y"]].values, leaf_size=2)
    # Query the BallTree on each feature from 'appart' to find the distance
    # to the nearest 'pharma' and its id
    missing_buses["distance_nearest"], missing_buses["id_nearest"] = tree.query(
        missing_buses[["x", "y"]].values,  # The input array for the query
        k=1,  # The number of nearest neighbors
    )
    missing_buses["bus_assignment"] = buses_to_match_to.reset_index().iloc[missing_buses.id_nearest].Bus.values
    missing_buses = missing_buses.drop(columns=["id_nearest"])
    return missing_buses


def remove_breakthrough_offshore(n: pypsa.Network) -> pypsa.Network:
    """
    Remove Offshore buses, Branches, Transformers, and Generators from the
    original BE network.
    """
    # rm any lines/buses associated with offshore substation buses
    n.mremove(
        "Line",
        n.lines.loc[n.lines.bus0.isin(n.buses.loc[n.buses.substation_off].index)].index,
    )
    n.mremove("Bus", n.buses.loc[n.buses.substation_off].index)
    return n


def assign_missing_state_regions(gdf_bus: gpd.GeoDataFrame):
    """
    Assign buses missing state and countries to their nearest neighbor bus
    value.
    """
    buses = gdf_bus.copy()
    buses = buses.reset_index().rename(columns={"bus_id": "Bus", "lon": "x", "lat": "y"}).set_index("Bus")

    missing = buses.loc[buses.full_state.isna()]
    if missing.empty:
        return gdf_bus
    buses = buses.loc[~buses.full_state.isna()]
    buses = buses.loc[~buses.full_state.isin(["Offshore"])]
    missing = match_missing_buses(buses, missing)

    # check if error western / texas. can make this a function
    missing = missing.reset_index().drop_duplicates("Bus").set_index("Bus")
    buses = buses.reset_index().drop_duplicates("Bus").set_index("Bus")

    missing.full_state = buses.loc[missing.bus_assignment.values].full_state.values

    buses = buses.reset_index().rename(columns={"Bus": "bus_id", "x": "lon", "y": "lat"}).set_index("bus_id")
    missing = missing.reset_index().rename(columns={"Bus": "bus_id", "x": "lon", "y": "lat"}).set_index("bus_id")

    # reassigning values to original dataframe
    gdf_bus.loc[missing.index, "full_state"] = missing.full_state
    return gdf_bus


def assign_missing_regions(n: pypsa.Network):
    """Assign buses missing state and countries to their nearest neighbor bus value."""
    # Define the region attributes to check and assign
    region_attrs = ["balancing_area", "state", "country", "reeds_zone", "reeds_ba", "county", "interconnect"]

    # Identify buses with any missing region attributes
    buses = n.buses.copy()
    missing = buses.loc[buses[region_attrs].isna().any(axis=1)]

    if missing.empty:
        return

    # Get reference buses with complete data
    reference_buses = buses.loc[~buses[region_attrs].isna().any(axis=1)]
    reference_buses = reference_buses.loc[~reference_buses.state.isin(["Offshore"])]

    # Match missing buses to their nearest neighbors
    missing = match_missing_buses(reference_buses, missing)

    # Assign region attributes from matched buses to missing buses
    for attr in region_attrs:
        if attr in n.buses.columns:  # Only assign if column exists
            n.buses.loc[missing.index, attr] = reference_buses.loc[missing.bus_assignment, attr].values


def assign_reeds_memberships(n: pypsa.Network, fn_reeds_memberships: str):
    """Assigns REeDS zone and balancing area memberships to buses."""
    reeds_memberships = pd.read_csv(fn_reeds_memberships, index_col=0)
    n.buses["nerc_reg"] = n.buses.reeds_zone.map(reeds_memberships.nercr)
    n.buses["trans_reg"] = n.buses.reeds_zone.map(reeds_memberships.transreg)
    n.buses["trans_grp"] = n.buses.reeds_zone.map(reeds_memberships.transgrp)
    n.buses["reeds_state"] = n.buses.reeds_zone.map(reeds_memberships.st)

    # Groupby county, and assign the most common reeds_zone, reeds_ba, reeds_state, nerc
    # This is a fix for the few counties that are split between unaligned county GIS and Reeds Zone Shapes.
    n.buses["reeds_zone"] = n.buses.groupby("county")["reeds_zone"].transform(
        lambda x: x.mode()[0],
    )
    n.buses["reeds_ba"] = n.buses.groupby("county")["reeds_ba"].transform(
        lambda x: x.mode()[0],
    )
    n.buses["reeds_state"] = n.buses.groupby("county")["reeds_state"].transform(
        lambda x: x.mode()[0],
    )
    n.buses["nerc_reg"] = n.buses.groupby("county")["nerc_reg"].transform(
        lambda x: x.mode()[0],
    )
    n.buses["trans_reg"] = n.buses.groupby("county")["trans_reg"].transform(
        lambda x: x.mode()[0],
    )
    n.buses["trans_grp"] = n.buses.groupby("county")["trans_grp"].transform(
        lambda x: x.mode()[0],
    )


def modify_breakthrough_substations(buslocs: pd.DataFrame):
    sub_fixes = {
        35017: {"lon": -123.0922, "lat": 48.5372},
        35033: {"lon": -122.78053, "lat": 48.65694},
        37584: {"lon": -117.10501, "lat": 32.54935},
        36116: {"lon": -122.4555, "lat": 37.8780},
        36145: {"lon": -122.3121, "lat": 37.8211},
        39718: {"lon": -106.49655, "lat": 31.76924},
        39731: {"lon": -106.3232, "lat": 31.7093},
        35116: {"lon": -122.462, "lat": 48.982},
        37707: {"lon": -115.4550, "lat": 32.6866},
        35976: {"lon": -120.8735, "lat": 39.4691},
        39211: {"lon": -104.6295, "lat": 39.3372},
        38886: {"lon": -106.5569, "lat": 31.8004},
        39574: {"lon": -115.0836, "lat": 42.8692},
        39547: {"lon": -113.4500, "lat": 42.6942},
        38928: {"lon": -108.0648, "lat": 39.0692},
        39570: {"lon": -114.3526, "lat": 42.6286},
        39571: {"lon": -114.0353, "lat": 42.5435},
    }
    for i in sub_fixes.keys():
        buslocs.loc[buslocs.sub_id == i, "lon"] = sub_fixes[i]["lon"]
        buslocs.loc[buslocs.sub_id == i, "lat"] = sub_fixes[i]["lat"]
    return buslocs


def main(snakemake):
    # create network
    n = pypsa.Network()
    n.name = "PyPSA-USA"

    model_topology = snakemake.params.model_topology
    interconnect = snakemake.wildcards.interconnect
    # interconnect in raw data given with an uppercase first letter
    if interconnect != "usa":
        interconnect = interconnect[0].upper() + interconnect[1:]

    # assign locations and balancing authorities to buses
    bus2sub = pd.read_csv(snakemake.input.bus2sub).set_index("bus_id")
    sub = pd.read_csv(snakemake.input.sub).set_index("sub_id")
    buslocs = pd.merge(bus2sub, sub, left_on="sub_id", right_index=True)
    buslocs = modify_breakthrough_substations(buslocs)

    # merge bus data with geometry data
    df_bus = pd.read_csv(snakemake.input["buses"], index_col=0)
    df_bus = assign_sub_id(df_bus, buslocs)
    gdf_bus = assign_bus_location(df_bus, buslocs)

    # test dropping duplicate bus ids earlier
    gdf_bus = gdf_bus.reset_index().drop_duplicates(subset="bus_id", keep="first").set_index("bus_id")

    # balancing authority shape
    ba_region_shapes = gpd.read_file(snakemake.input["onshore_shapes"])
    offshore_shapes = gpd.read_file(snakemake.input["offshore_shapes"])
    ba_shape = gpd.GeoDataFrame(
        pd.concat([ba_region_shapes, offshore_shapes], ignore_index=True),
    )
    ba_shape = ba_shape.rename(columns={"name": "balancing_area"})
    # Load country, state, county, and REeDs shapes
    state_shape = gpd.read_file(snakemake.input["state_shapes"])
    state_shape = state_shape.rename(columns={"name": "state"})
    na_shape = load_na_shapes(countries=["US"]).rename(columns={"name": "full_state"})
    reeds_shape = gpd.read_file(snakemake.input["reeds_shapes"]).rename(
        columns={"name": "reeds_zone"},
    )
    county_shape = gpd.read_file(snakemake.input["county_shapes"]).rename(
        columns={"GEOID": "county"},
    )

    # assign ba, state, and country to each bus
    gdf_bus = map_bus_to_region(gdf_bus, na_shape, ["full_state"])  # for laf
    gdf_bus = map_bus_to_region(gdf_bus, ba_shape, ["balancing_area"])
    gdf_bus = map_bus_to_region(gdf_bus, state_shape, ["state"])
    gdf_bus = map_bus_to_region(gdf_bus, state_shape, ["country"])
    gdf_bus = map_bus_to_region(gdf_bus, reeds_shape, ["reeds_zone", "reeds_ba"])
    gdf_bus = map_bus_to_region(gdf_bus, county_shape, ["county"])

    # assign load allocation factors to buses for state level dissagregation
    gdf_bus = assign_missing_state_regions(gdf_bus)

    # if dissagregating based with breakthrough energy on states, the LAF must
    # be calcualted here to capture splitting of states from the interconnect
    group_sums = gdf_bus.groupby("full_state")["Pd"].transform("sum")
    gdf_bus["LAF_state"] = gdf_bus["Pd"] / group_sums
    gdf_bus = gdf_bus.drop(columns=["full_state"])

    # Removing few duplicated shapes where GIS shapes were overlapping. TODO: Fix GIS shapes
    gdf_bus = gdf_bus.reset_index().drop_duplicates(subset="bus_id", keep="first").set_index("bus_id")

    # add buses, transformers, lines and links
    n = add_buses_from_file(n, gdf_bus, interconnect=interconnect)
    n = add_branches_from_file(n, snakemake.input["lines"])
    n = add_dclines_from_file(n, snakemake.input["links"])

    # identify offshore points of interconnection, and remove unncess components from BE network
    n = identify_osw_poi(n)
    if interconnect == "Texas" or interconnect == "usa":
        n = assign_texas_poi(n)
    n = remove_breakthrough_offshore(n)
    assign_missing_regions(n)

    # build offshore network configuration
    offshore_buses = build_offshore_buses(
        offshore_shapes,
        snakemake.params.build_offshore_network["bus_spacing"],
    )
    n = add_offshore_buses(n, offshore_buses)

    # Assign Lines Types and Missing Region Memberships
    add_custom_line_type(n)
    assign_line_types(n)
    assign_line_length(n)
    assign_missing_regions(n)
    assign_reeds_memberships(n, snakemake.input.reeds_memberships)

    p_max_pu = 1
    n.links["p_max_pu"] = p_max_pu
    n.links["p_min_pu"] = -p_max_pu

    # Filter Network to Only Specified Regions
    if model_topology is not None:
        for region_type in model_topology:
            rm_buses = n.buses.loc[~(n.buses[f"{region_type}"].isin(model_topology[region_type]))]
            rm_lines = n.lines.loc[(n.lines.bus0.isin(rm_buses.index)) | (n.lines.bus1.isin(rm_buses.index))]
            rm_transformers = n.transformers.loc[
                (n.transformers.bus0.isin(rm_buses.index)) | (n.transformers.bus1.isin(rm_buses.index))
            ]
            rm_links = n.links.loc[(n.links.bus0.isin(rm_buses.index)) | (n.links.bus1.isin(rm_buses.index))]
            n.mremove("Line", rm_lines.index.tolist())
            n.mremove("Transformer", rm_transformers.index.tolist())
            n.mremove("Link", rm_links.index.tolist())
            n.mremove("Bus", rm_buses.index.tolist())
            logger.info(
                f"Filtered network to {model_topology[region_type]}. Removed {len(rm_buses)} buses, {len(n.buses)} remaining.",
            )
            assert len(n.buses) > 0, (
                "No buses remaining in network. Check your model_topology: inclusion:, you may be filtering the wrong zones for the selected interconnect"
            )

    col_list = ["poi_bus", "poi_sub", "poi"]
    n.buses = n.buses.drop(columns=[col for col in col_list if col in n.buses])

    if (
        len(
            n.buses.loc[n.buses.balancing_area.isna() | n.buses.state.isna() | n.buses.country.isna()],
        )
        > 0
    ):
        logger.info(
            f"Network is missing BA/State/Country information for {len(n.buses.loc[n.buses.balancing_area.isna() | n.buses.state.isna() | n.buses.country.isna()])} buses.",
        )

    # export bus2sub interconnect data
    bus2sub = n.buses[
        [
            "sub_id",
            "interconnect",
            "balancing_area",
            "x",
            "y",
            "state",
            "country",
            "county",
        ]
    ]

    bus2sub.to_csv(snakemake.output.bus2sub)
    subs = (
        n.buses[["sub_id", "x", "y", "interconnect"]]
        .set_index("sub_id")
        .drop_duplicates()
        .rename(columns={"x": "lon", "y": "lat"})
    )
    subs.to_csv(snakemake.output.sub)

    # Export GIS Mapping Files
    n.buses.to_csv(snakemake.output.bus_gis)
    lines_gis = n.lines.copy()
    lines_gis["bus0"] = lines_gis.bus0.astype(str)
    lines_gis["bus1"] = lines_gis.bus1.astype(str)
    lines_gis["lat1"] = n.buses.loc[lines_gis.bus0].y.values
    lines_gis["lon1"] = n.buses.loc[lines_gis.bus0].x.values
    lines_gis["lat2"] = n.buses.loc[lines_gis.bus1].y.values
    lines_gis["lon2"] = n.buses.loc[lines_gis.bus1].x.values
    lines_gis["WKT_geometry"] = (
        "LINESTRING ("
        + lines_gis.lon1.astype(str).values
        + " "
        + lines_gis.lat1.astype(str).values
        + ", "
        + lines_gis.lon2.astype(str).values
        + " "
        + lines_gis.lat2.astype(str).values
        + ")"
    )
    lines_gis.to_csv(snakemake.output.lines_gis)

    # export network
    n.export_to_netcdf(snakemake.output.network)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_base_network", interconnect="texas")
    configure_logging(snakemake)
    main(snakemake)
