"""Unit tests for the county-level industrial load allocation in ``build_demand``.

``build_industry_demand`` attaches demand to the ``{simpl}`` cluster buses, which
no longer carry ``county``. The CLIU county loads are matched to the
base-network buses and summed onto the cluster buses through the busmaps.
"""

import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_demand import WriteIndustrial, load_base_buses

pytestmark = pytest.mark.fast


@pytest.fixture
def cliu_file(tmp_path):
    """CLIU rows (one per county and NAICS code) for two CA and two TX counties."""
    path = tmp_path / "cliu.csv"
    pd.DataFrame(
        {
            "fips_matching": [6001, 6001, 6003, 48001, 48003],
            "fipstate": [6, 6, 6, 48, 48],
            "naics": [311, 312, 311, 311, 311],
            "Total": [20.0, 10.0, 10.0, 20.0, 20.0],
        },
    ).to_csv(path, index=False)
    return path


def test_industrial_laf_is_summed_onto_cluster_buses(base_network, cliu_file):
    """Cluster buses carry no county; the factors come from their base buses."""
    assert "county" not in base_network.buses.columns
    base_buses = pd.DataFrame(
        {
            "county": ["p06001", "p06003", "p48001", "p48001", "p48003", "p48003"],
            "load_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            "bus": ["z1", "z1", "z2", "z3", "z3", "z2"],
        },
        index=["b1", "b2", "b3", "b4", "b5", "b6"],
    )

    laf = WriteIndustrial(base_network, cliu_file, base_buses)._get_load_allocation_factor(zone="state")

    # CA: both counties on z1. TX: p48001 (half of TX) split over b3/b4, p48003 all on
    # b5 because b6 carries no load weight.
    assert laf.to_dict() == pytest.approx({"z1": 1.0, "z2": 0.25, "z3": 0.75})


def test_load_base_buses_composes_busmaps(tmp_path):
    """Base buses keep their county and are keyed to the cluster bus via both busmaps."""
    n = pypsa.Network()
    n.add("Bus", ["b1", "b2", "b3"], county=["p06001", "p06003", "p06003"], load_weight=[1.0, 2.0, 0.0])
    network = tmp_path / "elec_base_network.nc"
    n.export_to_netcdf(network)

    busmap_b = tmp_path / "busmap_b.csv"
    pd.Series({"b1": 100, "b2": 100, "b3": 101}, name="sub_id").rename_axis("Bus").to_csv(busmap_b)
    busmap_s = tmp_path / "busmap_s.csv"
    pd.Series({100: "z1", 101: "z2"}, name="busmap").rename_axis("sub_id").to_csv(busmap_s)

    buses = load_base_buses(network, busmap_b, busmap_s)

    assert buses.bus.to_dict() == {"b1": "z1", "b2": "z1", "b3": "z2"}
    assert buses.county.to_dict() == {"b1": "p06001", "b2": "p06003", "b3": "p06003"}
    assert buses.load_weight.tolist() == [1.0, 2.0, 0.0]
