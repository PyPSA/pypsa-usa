"""Unit tests for parsing the sector demand file names in ``add_demand``."""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from add_demand import parse_demand_file

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    ("demand_file", "simpl", "expected"),
    [
        ("resources/r/demand/western/residential_space-heating_s20.pkl", "20", ["residential", "space-heating"]),
        ("resources/r/demand/western/transport_light-duty_s.pkl", "", ["transport", "light-duty"]),
        ("resources/r/demand/western/industry_heating_sall.pkl", "all", ["industry", "heating"]),
    ],
)
def test_parse_demand_file_strips_simpl_tag(demand_file, simpl, expected):
    """Demand outputs carry ``_s{simpl}``; it is not part of the end use."""
    assert parse_demand_file(demand_file, simpl) == expected
