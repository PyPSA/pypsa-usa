"""Unit tests for the balancing-area (``profile: eia``) demand allocation."""

import logging
import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from aggregate_to_substations import aggregate_to_substations
from build_demand import WriteBalancingArea, get_demand_params, load_base_buses

pytestmark = pytest.mark.fast

#: base bus: (sub_id, balancing_area, load_weight).
BASE_BUSES = {
    "b1": ("1", "CISO-PGAE", 30.0),
    "b2": ("2", "BANC", 10.0),
    "b3": ("3", "CISO-SCE", 70.0),
    "b4": ("4", "Offshore", 0.0),
}

#: substation -> cluster bus. Cluster c1 straddles CISO and BANC.
CLUSTERS = {"1": "c1", "2": "c1", "3": "c2", "4": "c2"}


def make_base_network():
    n = pypsa.Network()
    n.add("Bus", list(BASE_BUSES), v_nom=230.0, x=0.0, y=0.0)
    n.buses["sub_id"] = [v[0] for v in BASE_BUSES.values()]
    n.buses["balancing_area"] = [v[1] for v in BASE_BUSES.values()]
    n.buses["load_weight"] = [v[2] for v in BASE_BUSES.values()]
    for col, value in {
        "interconnect": "Western",
        "state": "CA",
        "country": "US",
        "county": "c",
        "reeds_zone": "p9",
        "reeds_ba": "CISO",
        "reeds_state": "CA",
        "Pd": 1.0,
        "LAF_state": 0.25,
    }.items():
        n.buses[col] = value
    n.add("Line", "l1", bus0="b1", bus1="b3", x=0.1, r=0.01, s_nom=100.0, length=1.0)
    return n


@pytest.fixture
def aggregated(tmp_path):
    """The clustered network, and the base buses read back from disk."""
    base = make_base_network()
    path = tmp_path / "elec_base_network.nc"
    base.export_to_netcdf(path)
    busmap = base.buses.sub_id.rename("sub_id")
    substations, _ = aggregate_to_substations(base, busmap, "reeds_zone")
    assert "balancing_area" not in substations.buses  # the regression of #967
    n = pypsa.Network()
    n.add("Bus", sorted(set(CLUSTERS.values())), interconnect="Western")
    busmap.to_csv(tmp_path / "busmap_b.csv")
    pd.Series(CLUSTERS, name="cluster_bus").to_csv(tmp_path / "busmap_s.csv")
    base_buses = load_base_buses(str(path), tmp_path / "busmap_b.csv", tmp_path / "busmap_s.csv")
    return n, base_buses


def make_demand(**columns):
    df = pd.DataFrame(columns, index=pd.date_range("2019-01-01", periods=2, freq="h", name="snapshot"))
    df["sector"], df["subsector"], df["fuel"] = "all", "all", "electricity"
    return df.set_index(["sector", "subsector", "fuel"], append=True)


def test_demand_conserved_after_balancing_area_is_aggregated_away(aggregated):
    n, base_buses = aggregated

    result = WriteBalancingArea(n, base_buses).dissagregate_demand(
        make_demand(CISO=[100.0, 200.0], BANC=[8.0, 9.0]),
        "ba",
    )

    # the straddling cluster takes 30% of CISO and all of BANC
    assert result["c1"].tolist() == pytest.approx([38.0, 69.0])
    assert result["c2"].tolist() == pytest.approx([70.0, 140.0])


def test_unmapped_western_areas_are_dropped_with_a_warning(aggregated, caplog):
    n, base_buses = aggregated
    with caplog.at_level(logging.WARNING):
        result = WriteBalancingArea(n, base_buses).dissagregate_demand(
            make_demand(CISO=[100.0, 100.0], BANC=[8.0, 8.0], SCL=[50.0, 50.0], TPWR=[5.0, 5.0]),
            "ba",
        )

    assert result.sum(axis=1).tolist() == pytest.approx([108.0, 108.0])
    assert "Seattle City Light, 50.0 MW" in caplog.text
    assert "Tacoma Power, 5.0 MW" in caplog.text


def test_unmapped_western_areas_do_not_warn_outside_the_west(aggregated, caplog):
    n, base_buses = aggregated
    n.buses["interconnect"] = "Eastern"
    with caplog.at_level(logging.WARNING):
        WriteBalancingArea(n, base_buses).dissagregate_demand(
            make_demand(CISO=[1.0, 1.0], BANC=[1.0, 1.0], SCL=[50.0, 50.0]),
            "ba",
        )
    assert "SCL" not in caplog.text


def test_weighted_bus_outside_the_network_raises(aggregated):
    n, base_buses = aggregated
    base_buses.loc["b1", "bus"] = None
    with pytest.raises(ValueError, match="do not map"):
        WriteBalancingArea(n, base_buses)


def test_eia_profile_selects_balancing_area_writer():
    assert get_demand_params("power", {"profile": "eia"})[:2] == ("eia", "ba")
