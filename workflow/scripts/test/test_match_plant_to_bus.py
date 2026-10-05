"""Unit tests for the plant -> bus match in ``add_electricity``.

``match_plant_to_bus`` places every existing power plant on a network bus. Its
first pass is meant to keep a plant inside its own ReEDS zone, but it keys
``plants["country"]`` (county codes like ``p06029``) against
``buses["reeds_zone"]`` (zone codes like ``p10``) — two vocabularies that never
intersect — so the pass is inert and every plant is placed by the unconstrained
nearest-point match.

That match is harmless on master, which runs it on the NODAL network (one bus
per substation): the nearest point is ~1 km away and 100 % of existing MW stays
in the plant's own zone. After the simplify-early refactor develop runs the same
function against the ``{simpl}`` cluster CENTROIDS, where the nearest point is
tens of km away and can sit in a neighbouring zone (western: 33 plants /
1,320 MW misplaced, e.g. Helms PHS 1,053 MW p9 -> p10).

The fix matches against the substations of ``bus2sub.csv`` and then maps the
matched substation through ``busmap_s{simpl}.csv``. These tests pin:
  * the zone crossing the old path produces and the new path does not,
  * the invariant that makes it right: the assigned cluster carries the zone of
    the plant's NEAREST SUBSTATION, never of the nearest centroid,
  * that ``bus_locations=None`` still reproduces the old assignment exactly, so
    the change is opt-in at the call site.

Geometry (all at latitude 0, so degrees of longitude are the only axis):

    zone A substations  0.0   0.1          -> cluster "A1 0" centroid 0.05
    zone B substations              1.0          3.0   -> cluster "B1 0" centroid 2.0
    plant                       0.9

The plant is 0.1 deg from a zone-B substation but 0.85 deg from the zone-A
centroid and 1.1 deg from the zone-B centroid: nearest substation is B, nearest
centroid is A.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from add_electricity import match_plant_to_bus

pytestmark = pytest.mark.fast

#: Cluster centroids, i.e. what ``n.buses`` looks like at ``add_electricity``
#: in the simplify-early DAG.
CLUSTERS = {
    "A1 0": {"x": 0.05, "y": 0.0, "reeds_zone": "p_a"},
    "B1 0": {"x": 2.0, "y": 0.0, "reeds_zone": "p_b"},
}

#: ``bus2sub.csv``: one row per RAW base-network bus, several per substation.
#: ``sub_id`` arrives as a float in the real file ("35827.0"), so it is written
#: that way here. ``reeds_zone`` is test-side ground truth, not read by the code.
BUS2SUB_ROWS = [
    # Bus,     sub_id,  x,    y,   reeds_zone
    ("2020000", "10.0", 0.0, 0.0, "p_a"),
    ("2020001", "10.0", 0.0, 0.0, "p_a"),  # second raw bus on the same substation
    ("2020002", "11.0", 0.1, 0.0, "p_a"),
    ("2020003", "20.0", 1.0, 0.0, "p_b"),
    ("2020004", "21.0", 3.0, 0.0, "p_b"),
]

SUB_TO_CLUSTER = {"10": "A1 0", "11": "A1 0", "20": "B1 0", "21": "B1 0"}


@pytest.fixture
def network():
    n = pypsa.Network()
    n.add("Bus", list(CLUSTERS), x=[b["x"] for b in CLUSTERS.values()], y=[b["y"] for b in CLUSTERS.values()])
    n.buses["reeds_zone"] = [b["reeds_zone"] for b in CLUSTERS.values()]
    return n


@pytest.fixture
def bus_locations():
    return pd.DataFrame(BUS2SUB_ROWS, columns=["Bus", "sub_id", "x", "y", "reeds_zone"])


@pytest.fixture
def busmap():
    return pd.DataFrame(
        {"sub_id": list(SUB_TO_CLUSTER), "cluster_bus": list(SUB_TO_CLUSTER.values())},
        dtype=str,
    )


def _plants(lonlat, **extra):
    """A minimal plants frame: what ``match_plant_to_bus`` actually reads."""
    lon, lat = zip(*lonlat)
    df = pd.DataFrame(
        {
            "longitude": list(lon),
            "latitude": list(lat),
            "p_nom": [100.0] * len(lon),
            # county code, exactly as filter_plants_by_region leaves it: it
            # never matches a ReEDS zone, which is what makes pass 1 inert.
            "country": ["p06029"] * len(lon),
            **extra,
        },
    )
    df.index = [f"plant_{i}" for i in range(len(df))]
    return df


# --- (a) the zone crossing ---------------------------------------------------


def test_centroid_match_crosses_a_zone_boundary_and_substation_match_does_not(network, bus_locations, busmap):
    """The plant at 0.9 belongs to zone B; only the substation match puts it there."""
    plants = _plants([(0.9, 0.0)])

    old = match_plant_to_bus(network, plants)
    assert old.loc["plant_0", "bus_assignment"] == "A1 0"
    assert network.buses.loc["A1 0", "reeds_zone"] == "p_a"

    new = match_plant_to_bus(network, plants, bus_locations=bus_locations, busmap=busmap)
    assert new.loc["plant_0", "bus_assignment"] == "B1 0"
    assert network.buses.loc["B1 0", "reeds_zone"] == "p_b"


def test_distance_nearest_is_measured_to_the_matched_substation(network, bus_locations, busmap):
    """``distance_nearest`` keeps its meaning: the distance to the matched point."""
    plants = _plants([(0.9, 0.0)])

    old = match_plant_to_bus(network, plants)
    new = match_plant_to_bus(network, plants, bus_locations=bus_locations, busmap=busmap)

    # 0.9 -> centroid 0.05 is 0.85 deg; 0.9 -> substation at 1.0 is 0.1 deg.
    assert old.loc["plant_0", "distance_nearest"] == pytest.approx(0.85)
    assert new.loc["plant_0", "distance_nearest"] == pytest.approx(0.1)


def test_the_metric_stays_euclidean_in_raw_degrees(network, bus_locations, busmap):
    """Equivalence with master depends on master's own metric, not haversine.

    At latitude 60 a degree of longitude is half a degree of latitude on the
    ground; under haversine the plant below would match the substation to its
    north, under raw-degree Euclidean the one to its east. Master uses the
    latter, so the fix must too.
    """
    subs = pd.DataFrame(
        {
            "Bus": ["b0", "b1"],
            "sub_id": ["10", "20"],
            "x": [0.0, 0.3],  # 0.3 deg east  -> ~16.7 km at lat 60
            "y": [60.4, 60.0],  # 0.4 deg north -> ~44.5 km
            "reeds_zone": ["p_a", "p_b"],
        },
    )
    plants = _plants([(0.0, 60.0)])
    out = match_plant_to_bus(network, plants, bus_locations=subs, busmap=busmap)
    assert out.loc["plant_0", "bus_assignment"] == "B1 0"  # the eastern substation
    assert out.loc["plant_0", "distance_nearest"] == pytest.approx(0.3)


# --- (b) the invariant -------------------------------------------------------


@pytest.mark.parametrize("lon", [-0.5, 0.0, 0.04, 0.06, 0.3, 0.55, 0.9, 1.4, 2.6, 4.0])
def test_assigned_cluster_carries_the_zone_of_the_nearest_substation(network, bus_locations, busmap, lon):
    """For every plant, zone(assigned cluster) == zone(nearest substation)."""
    plants = _plants([(lon, 0.0)])
    out = match_plant_to_bus(network, plants, bus_locations=bus_locations, busmap=busmap)

    nearest = bus_locations.iloc[int(np.argmin(np.hypot(bus_locations["x"] - lon, bus_locations["y"] - 0.0)))]
    assigned_zone = network.buses.loc[out.loc["plant_0", "bus_assignment"], "reeds_zone"]
    assert assigned_zone == nearest["reeds_zone"]


def test_a_plant_whose_substation_is_not_in_the_busmap_is_dropped_and_logged(
    network,
    bus_locations,
    busmap,
    caplog,
):
    orphan = pd.concat(
        [bus_locations, pd.DataFrame([("2020005", "99.0", 0.9, 0.0, "p_b")], columns=bus_locations.columns)],
        ignore_index=True,
    )
    plants = _plants([(0.9, 0.0), (3.0, 0.0)])

    with caplog.at_level(logging.WARNING):
        out = match_plant_to_bus(network, plants, bus_locations=orphan, busmap=busmap)

    assert list(out.index) == ["plant_1"]
    assert "Dropped 1 of 2 plants (100.0 MW)" in caplog.text


def test_the_inert_zone_pass_is_announced(network, bus_locations, busmap, caplog):
    """A dead constraint must not read as a live one."""
    with caplog.at_level(logging.WARNING):
        match_plant_to_bus(network, _plants([(0.9, 0.0)]), bus_locations=bus_locations, busmap=busmap)
    assert "Zone-constrained plant match is inert" in caplog.text

    caplog.clear()
    in_zone = _plants([(0.9, 0.0)])
    in_zone["country"] = "p_a"
    with caplog.at_level(logging.WARNING):
        match_plant_to_bus(network, in_zone, bus_locations=bus_locations, busmap=busmap)
    assert "Zone-constrained plant match is inert" not in caplog.text


# --- (c) the regression guard ------------------------------------------------


@pytest.mark.parametrize("lon", [-0.5, 0.0, 0.04, 0.06, 0.3, 0.55, 0.9, 1.4, 2.6, 4.0])
def test_without_the_extras_the_legacy_assignment_is_unchanged(network, lon):
    """``bus_locations=None`` is the old nearest-centroid match, unchanged."""
    plants = _plants([(lon, 0.0)])
    out = match_plant_to_bus(network, plants)

    centroids = network.buses[["x", "y"]]
    expected = centroids.index[int(np.argmin(np.hypot(centroids["x"] - lon, centroids["y"] - 0.0)))]
    assert out.loc["plant_0", "bus_assignment"] == expected


def test_one_extra_alone_does_not_switch_paths(network, bus_locations, busmap):
    """Both arguments are required; either alone keeps the legacy behaviour."""
    plants = _plants([(0.9, 0.0)])
    assert match_plant_to_bus(network, plants, bus_locations=bus_locations).loc["plant_0", "bus_assignment"] == "A1 0"
    assert match_plant_to_bus(network, plants, busmap=busmap).loc["plant_0", "bus_assignment"] == "A1 0"


def test_duplicate_bus2sub_rows_do_not_change_the_match(network, bus_locations, busmap):
    """bus2sub is passed un-deduplicated because that is master's bus set.

    Extra raw buses on a substation are extra copies of the same point, so they
    can only ever be chosen as the nearest point together.
    """
    plants = _plants([(0.02, 0.0), (0.9, 0.0), (3.4, 0.0)])
    deduped = bus_locations.drop_duplicates(subset="sub_id")
    a = match_plant_to_bus(network, plants, bus_locations=bus_locations, busmap=busmap)
    b = match_plant_to_bus(network, plants, bus_locations=deduped, busmap=busmap)
    pd.testing.assert_series_equal(a["bus_assignment"], b["bus_assignment"])
    pd.testing.assert_series_equal(a["distance_nearest"], b["distance_nearest"])


def test_float_and_integer_substation_ids_are_the_same_key(network, bus_locations, busmap):
    """A float sub_id from bus2sub and an integer one from busmap_s must join.

    bus2sub writes ``35827.0``, busmap_s writes ``35827``.
    """
    ints = bus_locations.assign(sub_id=bus_locations["sub_id"].str.replace(".0", "", regex=False))
    out = match_plant_to_bus(network, _plants([(0.9, 0.0)]), bus_locations=ints, busmap=busmap)
    assert out.loc["plant_0", "bus_assignment"] == "B1 0"
    assert not out["bus_assignment"].isnull().any()
