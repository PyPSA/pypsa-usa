# BY PyPSA-USA Authors
"""The HIFLD base grid: line reduction, site clustering, seams, re-keying and POIs."""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import pytest
import xarray as xr
from shapely.geometry import LineString, MultiLineString, box

sys.path.insert(0, str(Path(__file__).parent.parent))

from build_base_network import TEXAS_POI_SUB_IDS, assign_poi_from_file, merge_footprint_islands
from build_hifld_grid import (
    assign_interconnects,
    be_offshore_poi_sites,
    class_of,
    cluster_sites,
    haversine_km,
    largest_component_per_interconnect,
    reduce_lines,
    rekey_by_coordinate,
)
from build_renewable_profiles import rekey_caps_to_grid_substations

pytestmark = pytest.mark.fast

# 0.001 degree of latitude is 111 m
DEG_PER_M = 1 / 111_195


def test_reduce_lines_fills_voltage_drops_records_and_measures_routes():
    raw = gpd.GeoDataFrame(
        {
            "ID": ["1", "2", "3", "4", "5", "6"],
            "VOLTAGE": [345.0, -999999.0, -999999.0, 138.0, 500.0, 230.0],
            "VOLT_CLASS": ["345", "100-161", "NOT AVAILABLE", "100-161", "DC", "220-287"],
            "STATUS": ["IN SERVICE", "NOT AVAILABLE", "IN SERVICE", "UNDER CONSTRUCTION", "IN SERVICE", "INACTIVE"],
            "TYPE": ["AC; OVERHEAD", "OVERHEAD", "OVERHEAD", "AC; OVERHEAD", "DC; OVERHEAD", "AC; OVERHEAD"],
        },
        geometry=[
            LineString([(-100, 40), (-100, 40.1), (-99.9, 40.1)]),
            MultiLineString([[(-90, 30), (-90, 30.05)], [(-90, 30.05), (-90.05, 30.05)]]),
            LineString([(0, 0), (0, 1)]),
            LineString([(0, 0), (0, 1)]),
            LineString([(-120, 45), (-118, 34)]),
            LineString([(0, 0), (0, 1)]),
        ],
        crs=4326,
    )
    out = reduce_lines(raw)

    assert out.id.tolist() == [1, 2, 5]  # no voltage at all, under construction, inactive: dropped
    assert out.kv.tolist() == [345.0, 138.0, 500.0]  # missing voltage takes its class
    assert out.dc.tolist() == [0, 0, 1]
    # endpoints: first point of the first part, last point of the last part
    assert out.loc[1, ["lon1", "lat1", "lon2", "lat2"]].tolist() == [-90, 30, -90.05, 30.05]
    expected = haversine_km(-100, 40, -100, 40.1) + haversine_km(-100, 40.1, -99.9, 40.1)
    assert out.loc[0, "route_km"] == pytest.approx(expected, abs=1e-3)


def test_class_of_bins_nominal_voltages():
    assert class_of([69, 115, 161, 230, 287, 345, 500, 765]).tolist() == [69, 138, 138, 230, 230, 345, 500, 765]


def test_cluster_sites_merges_within_radius_transitively():
    # A-B 150 m, B-C 150 m, C-D 300 m: A, B, C form one site (A-C is 300 m), D its own
    lat = 40 + np.array([0, 150, 300, 600]) * DEG_PER_M
    site = cluster_sites(np.full(4, -100.0), lat, radius_km=0.2)
    assert site[0] == site[1] == site[2] != site[3]


def test_assign_interconnects_cuts_the_seam_and_smooths_a_stray_seed():
    # a chain of five sites; BE says the west two are Western, the east three Eastern,
    # but site 1's nearest BE substation is an Eastern one far away
    sites = pd.DataFrame({"lon": [-105.0, -104.9, -104.8, -104.7, -104.6], "lat": 40.0})
    be = pd.DataFrame(
        {
            "lon": [-105.0, -104.8, -104.7, -104.6, -104.9],
            "lat": [40.0, 40.0, 40.0, 40.0, 41.0],
            "interconnect": ["Western", "Eastern", "Eastern", "Eastern", "Eastern"],
        },
    )
    bus0 = np.array([0, 1, 2, 3])
    bus1 = np.array([1, 2, 3, 4])
    label, crosses = assign_interconnects(sites, bus0, bus1, be)
    assert crosses.sum() == 1  # exactly one line bridges the two systems
    assert set(label) == {"Western", "Eastern"}


def test_largest_component_per_interconnect_drops_fragments():
    interconnect = np.array(["Eastern"] * 5)
    keep = largest_component_per_interconnect(5, np.array([0, 1, 3]), np.array([1, 2, 4]), interconnect)
    assert keep.tolist() == [True, True, True, False, False]


def test_rekey_by_coordinate_stays_on_its_own_side_of_the_seam():
    buses = pd.DataFrame(
        {"lat": [40.0, 40.0], "lon": [-100.0, -100.3], "interconnect": ["Eastern", "Western"]},
        index=[10, 20],
    )
    new, km = rekey_by_coordinate([40.0, 40.0], [-100.01, -100.01], buses, 75.0, restrict_to=["Eastern", "Western"])
    assert new.tolist() == [10, 20]
    assert km[1] > 20
    far, _ = rekey_by_coordinate([45.0], [-90.0], buses, 75.0)
    assert far.tolist() == [-1]


def test_be_offshore_poi_sites_takes_landfalls_and_texas_sites():
    be_sub = pd.DataFrame(
        {
            "sub_id": [100, 200, 41012, *TEXAS_POI_SUB_IDS],
            "name": ["LANDFALL", "INLAND", "OFFSHORE", *[f"TX{i}" for i in range(7)]],
            "lat": 30.0,
            "lon": -90.0,
            "interconnect": ["Eastern", "Eastern", "Eastern", *["Texas"] * 7],
        },
    )
    be_bus2sub = pd.DataFrame({"bus_id": [1, 2, 3], "sub_id": [100, 200, 41012]})
    be_branch = pd.DataFrame(
        {"from_bus_id": [3, 1], "to_bus_id": [1, 2], "branch_device_type": ["Line", "Line"]},
    )
    sites = be_offshore_poi_sites(be_bus2sub, be_sub, be_branch)
    assert sites.sub_id.tolist() == [100, *TEXAS_POI_SUB_IDS]
    assert sites.source.tolist() == ["be_offshore_line"] + ["texas_hardcoded"] * 7
    assert sites.loc[0, "name"] == "LANDFALL"


def test_assign_poi_from_file_marks_buses_and_substations(tmp_path):
    n = pypsa.Network()
    n.add("Bus", ["1", "2", "3"], sub_id=[10, 10, 20])
    pd.DataFrame({"sub_id": [10, 99], "bus_id": [1, 999]}).to_csv(tmp_path / "poi.csv", index=False)
    assign_poi_from_file(n, tmp_path / "poi.csv")
    assert n.buses.poi_bus.tolist() == [True, False, False]
    assert n.buses.poi_sub.tolist() == [True, True, False]


def test_rekey_caps_to_grid_substations_moves_entries_into_the_region_holding_them(tmp_path):
    pd.DataFrame(
        {
            "sub_id": [1, 2, 3, 4],
            "lat": [40.0, 40.2, 45.0, 40.52],
            "lon": [-100.0, -99.5, -90.0, -100.03],
            "interconnect": "Eastern",
        },
    ).to_csv(tmp_path / "be_sub.csv", index=False)
    gpd.GeoDataFrame(
        {"name": ["100001.0", "100002.0"]},
        geometry=[box(-101, 39, -99.8, 40.5), box(-99.8, 39, -99, 40.5)],
        crs=4326,
    ).to_file(tmp_path / "regions.geojson")
    caps = xr.Dataset(
        {
            "p_nom_max": ("bus", [10.0, 30.0, 5.0, 7.0]),
            "weight": ("bus", [10.0, 30.0, 5.0, 7.0]),
            "avg_cf": ("bus", [0.2, 0.4, 0.3, 0.5]),
        },
        coords={"bus": ["1.0", "2.0", "3.0", "4.0"]},
    )
    out = rekey_caps_to_grid_substations(caps, tmp_path / "be_sub.csv", tmp_path / "regions.geojson")
    # BE 1 lies in region 100001, BE 2 in 100002, BE 4 just north of 100001 (inside
    # the coastline allowance); BE 3 is outside every region and drops
    caps_by_sub = out["p_nom_max"].to_series()
    assert caps_by_sub.to_dict() == {"100001": 17.0, "100002": 30.0}
    assert float(out["avg_cf"].sel(bus="100001")) == pytest.approx((0.2 * 10 + 0.5 * 7) / 17)


def test_merge_footprint_islands_moves_islands_onto_the_nearest_main_substation():
    n = pypsa.Network()
    # main network: a-b-c; island d-e near c; f alone near a; g offshore at d's substation
    n.add(
        "Bus",
        list("abcdefg"),
        x=[0.0, 1.0, 2.0, 2.3, 2.6, -0.2, 3.0],
        y=0.0,
        sub_id=[1, 2, 3, 4, 5, 6, 4],  # g is offshore, wired to island substation 4
        interconnect="Western",
        county=["A", "B", "C", "D", "E", "F", "G"],
        load_weight=[100.0, 100.0, 100.0, 0.5, 0.5, 0.5, 0.0],
        substation_off=[False] * 6 + [True],
    )
    n.add("Line", ["ab", "bc", "de"], bus0=["a", "b", "d"], bus1=["b", "c", "e"], x=0.1, s_nom=100)
    merge_footprint_islands(n)
    assert n.buses.sub_id.tolist() == [1, 2, 3, 3, 3, 1, 3]
    assert n.buses.loc[["d", "e"], "x"].tolist() == [2.0, 2.0]
    assert n.buses.loc["g", "x"] == 3.0  # offshore bus keeps its position
    assert n.buses.loc["f", "county"] == "A"
    assert n.buses.load_weight.sum() == 301.5  # nothing is dropped


def _two_area_network(island_x, island_load):
    """Main network a-b-c on the equator, and a two-bus island d-e starting at ``island_x``."""
    n = pypsa.Network()
    n.add(
        "Bus",
        list("abcde"),
        x=[0.0, 1.0, 2.0, island_x, island_x + 0.5],
        y=0.0,
        sub_id=[1, 2, 3, 4, 5],
        interconnect="Western",
        state=["WA"] * 3 + ["AZ"] * 2,
        load_weight=[100.0, 100.0, 100.0, island_load, island_load],
        substation_off=False,
    )
    n.add("Line", ["ab", "bc", "de"], bus0=["a", "b", "d"], bus1=["b", "c", "e"], x=0.1, s_nom=100)
    return n


@pytest.mark.parametrize(
    ("island_x", "island_load"),
    [
        (2.3, 20.0),  # 33 km away but 12% of load: a real area, not a boundary fragment
        (12.0, 0.5),  # 0.3% of load but over 1,000 km away: a second area of the footprint
    ],
)
def test_merge_footprint_islands_keeps_a_large_or_distant_island(island_x, island_load):
    n = _two_area_network(island_x, island_load)
    before = n.buses.loc[["d", "e"], ["x", "y", "sub_id", "state"]].copy()
    merge_footprint_islands(n)
    pd.testing.assert_frame_equal(n.buses.loc[["d", "e"], ["x", "y", "sub_id", "state"]], before)
