"""Unit tests for build_servm_load_weights helpers."""

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_servm_load_weights import (
    build_load_weights,
    compose_busmaps,
    load_servm_region_map,
)


@pytest.fixture
def region_map():
    """Mirror of repo_data/CPUC/servm_region_map.csv, CISO-VEA excluded."""
    return pd.Series(
        {
            "CISO-PGAE": "PGE",
            "CISO-SCE": "SCE",
            "CISO-SDGE": "SDGE",
            "IID": "IID",
            "LDWP": "LADWP",
            "BANC": "NCNC",
            "TIDC": "NCNC",
            "CISO-VEA": None,
        },
        name="servm_region",
    ).rename_axis("balancing_area")


def make_buses(records):
    """Base-network buses from (bus, balancing_area, load_weight) triples."""
    df = pd.DataFrame(records, columns=["bus", "balancing_area", "load_weight"])
    return df.set_index("bus")


def test_load_region_map_keeps_excluded_ba_as_null(tmp_path):
    path = tmp_path / "servm_region_map.csv"
    path.write_text(
        "balancing_area,servm_region,note\nCISO-PGAE,PGE,\nCISO-VEA,,Nevada footprint\n",
    )
    mapping = load_servm_region_map(str(path))
    assert mapping["CISO-PGAE"] == "PGE"
    # excluded, not unknown: the row survives so the BA stays distinguishable
    assert "CISO-VEA" in mapping.index
    assert pd.isna(mapping["CISO-VEA"])


def test_compose_busmaps_chains_base_to_cluster():
    busmap_b = pd.Series({"b1": "10", "b2": "10", "b3": "20"}, name="sub_id")
    busmap_s = pd.Series({10: "c1", 20: "c2"}, name="cluster_bus")

    composed = compose_busmaps(busmap_b, busmap_s)

    assert composed.to_dict() == {"b1": "c1", "b2": "c1", "b3": "c2"}


def test_laf_sums_to_one_per_region(region_map):
    buses = make_buses(
        [
            ("b1", "CISO-PGAE", 30.0),
            ("b2", "CISO-PGAE", 10.0),
            ("b3", "CISO-SCE", 60.0),
        ],
    )
    busmap = pd.Series({"b1": "c1", "b2": "c2", "b3": "c3"})

    weights = build_load_weights(buses, region_map, busmap)

    assert set(weights.columns) == {"bus", "servm_region", "laf"}
    per_region = weights.groupby("servm_region").laf.sum()
    assert per_region.round(12).eq(1.0).all()
    pge = weights[weights.servm_region == "PGE"].set_index("bus").laf
    assert pge["c1"] == pytest.approx(0.75)
    assert pge["c2"] == pytest.approx(0.25)


def test_straddling_cluster_splits_across_regions(region_map):
    """A cluster spanning LADWP and SCE gets one row per region, each at laf 1."""
    buses = make_buses(
        [
            ("b1", "LDWP", 30.0),
            ("b2", "CISO-SCE", 70.0),
        ],
    )
    busmap = pd.Series({"b1": "c1", "b2": "c1"})

    weights = build_load_weights(buses, region_map, busmap)

    assert len(weights) == 2
    assert set(weights.bus) == {"c1"}
    assert weights.set_index("servm_region").laf.to_dict() == {"LADWP": 1.0, "SCE": 1.0}


def test_unmapped_ba_with_load_raises(region_map):
    buses = make_buses(
        [
            ("b1", "CISO-PGAE", 30.0),
            ("b2", "CISO-PGE", 20.0),  # hypothetical upstream relabeling
        ],
    )
    busmap = pd.Series({"b1": "c1", "b2": "c2"})

    with pytest.raises(ValueError, match="CISO-PGE"):
        build_load_weights(buses, region_map, busmap)


def test_blank_ba_with_load_raises(region_map):
    buses = make_buses(
        [
            ("b1", "CISO-PGAE", 30.0),
            ("b2", None, 20.0),
        ],
    )
    busmap = pd.Series({"b1": "c1", "b2": "c2"})

    with pytest.raises(ValueError, match="b2"):
        build_load_weights(buses, region_map, busmap)


def test_blank_ba_without_load_is_dropped(region_map):
    """Offshore buses carry no balancing area but also no weight."""
    buses = make_buses(
        [
            ("b1", "CISO-PGAE", 30.0),
            ("b2", "Offshore", 0.0),
            ("b3", None, 0.0),
        ],
    )
    busmap = pd.Series({"b1": "c1", "b2": "c1", "b3": "c1"})

    weights = build_load_weights(buses, region_map, busmap)

    assert weights.laf.tolist() == [1.0]


def test_vea_dropped_without_error(region_map):
    buses = make_buses(
        [
            ("b1", "CISO-PGAE", 30.0),
            ("b2", "CISO-PGAE", 10.0),
            ("b3", "CISO-VEA", 60.0),
        ],
    )
    busmap = pd.Series({"b1": "c1", "b2": "c2", "b3": "c3"})

    weights = build_load_weights(buses, region_map, busmap)

    assert "c3" not in set(weights.bus)
    # the excluded weight leaves PGE's internal shares untouched
    assert weights.set_index("bus").laf.to_dict() == {"c1": pytest.approx(0.75), "c2": pytest.approx(0.25)}


def test_missing_region_warns_not_raises(region_map, caplog):
    buses = make_buses([("b1", "CISO-PGAE", 30.0)])
    busmap = pd.Series({"b1": "c1"})

    with caplog.at_level("WARNING"):
        weights = build_load_weights(buses, region_map, busmap)

    assert weights.servm_region.tolist() == ["PGE"]
    assert "IID" in caplog.text


def test_bus_missing_from_busmap_raises(region_map):
    buses = make_buses(
        [
            ("b1", "CISO-PGAE", 30.0),
            ("b2", "CISO-SCE", 20.0),
        ],
    )
    busmap = pd.Series({"b1": "c1"})

    with pytest.raises(ValueError, match="b2"):
        build_load_weights(buses, region_map, busmap)


def test_empty_result_raises(region_map):
    buses = make_buses([("b1", "CISO-VEA", 30.0)])
    busmap = pd.Series({"b1": "c1"})

    with pytest.raises(ValueError, match="No bus carries load weight"):
        build_load_weights(buses, region_map, busmap)
