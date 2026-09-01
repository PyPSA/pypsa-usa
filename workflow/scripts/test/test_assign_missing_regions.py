# BY PyPSA-USA Authors
"""``build_base_network.assign_missing_regions`` fills only missing attributes.

Regression for the Trans Bay Cable misassignment: a bus with a correctly
resolved ``county`` but a NaN ``reeds_zone`` (shape gap at the SF waterfront)
must keep its county — the whole-row copy previously moved SF's only HVDC
infeed terminal into San Mateo County.
"""

import sys
from pathlib import Path

import pandas as pd
import pypsa
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from build_base_network import assign_missing_regions

REGION_ATTRS = ["balancing_area", "state", "country", "reeds_zone", "reeds_ba", "county", "interconnect"]


def _network(rows):
    n = pypsa.Network()
    for name, x, y, attrs in rows:
        n.add("Bus", name, x=x, y=y)
        for k, v in attrs.items():
            n.buses.loc[name, k] = v
    for attr in REGION_ATTRS:
        if attr not in n.buses.columns:
            n.buses[attr] = pd.NA
    return n


COMPLETE_SM = {
    "balancing_area": "CISO-PGAE",
    "state": "California",
    "country": "p06081",
    "reeds_zone": "p9",
    "reeds_ba": "CISO",
    "county": "p06081",
    "interconnect": "Western",
}


def test_partial_missing_keeps_resolved_attributes():
    """A bus with only reeds_zone/reeds_ba missing keeps its own county."""
    potrero = {k: v for k, v in COMPLETE_SM.items() if k not in ("reeds_zone", "reeds_ba")}
    potrero["county"] = "p06075"
    potrero["country"] = "p06075"
    n = _network(
        [
            ("potrero", -122.377, 37.7418, potrero),
            ("san_mateo", -122.33, 37.55, COMPLETE_SM),
        ],
    )
    assign_missing_regions(n)
    assert n.buses.loc["potrero", "county"] == "p06075"  # NOT clobbered to p06081
    assert n.buses.loc["potrero", "reeds_zone"] == "p9"  # filled from neighbor
    assert n.buses.loc["potrero", "reeds_ba"] == "CISO"


def test_fully_missing_bus_still_filled_completely():
    n = _network(
        [
            ("orphan", -122.0, 37.6, {}),
            ("san_mateo", -122.33, 37.55, COMPLETE_SM),
        ],
    )
    assign_missing_regions(n)
    for attr in REGION_ATTRS:
        assert n.buses.loc["orphan", attr] == COMPLETE_SM[attr], attr


def test_complete_buses_untouched():
    n = _network([("san_mateo", -122.33, 37.55, COMPLETE_SM)])
    before = n.buses.copy()
    assign_missing_regions(n)
    pd.testing.assert_frame_equal(n.buses, before)


@pytest.mark.parametrize("missing_attr", ["county", "state", "balancing_area"])
def test_single_missing_attribute_filled_others_kept(missing_attr):
    attrs = dict(COMPLETE_SM)
    attrs["county"] = "p06075"
    attrs["country"] = "p06075"
    del attrs[missing_attr]
    n = _network(
        [
            ("partial", -122.377, 37.7418, attrs),
            ("san_mateo", -122.33, 37.55, COMPLETE_SM),
        ],
    )
    assign_missing_regions(n)
    assert n.buses.loc["partial", missing_attr] == COMPLETE_SM[missing_attr]
    for attr in REGION_ATTRS:
        if attr in attrs:
            assert n.buses.loc["partial", attr] == attrs[attr], attr
