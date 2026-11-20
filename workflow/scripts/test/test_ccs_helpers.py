"""Carbon-capture rate and residual emissions read from a carrier name (``_helpers``)."""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import ccs_capture_rate, ccs_co2_emissions

pytestmark = pytest.mark.fast


def test_capture_rate_is_read_from_the_carrier_or_component_name():
    assert ccs_capture_rate("CCGT-95CCS") == 0.95
    assert ccs_capture_rate("coal-99CCS") == 0.99
    assert ccs_capture_rate("p1 CCGT-97CCS_2030") == 0.97
    assert ccs_capture_rate("CCGT") is None


def test_ccs_emissions_are_the_uncaptured_share():
    assert ccs_co2_emissions("CCGT-95CCS", 0.2) == pytest.approx(0.01)
    # bit for bit the expression the callers used before sharing it
    assert ccs_co2_emissions("coal-99CCS", 0.3453) == (1 - 99 / 100) * 0.3453
    with pytest.raises(ValueError, match="carbon capture"):
        ccs_co2_emissions("CCGT", 0.2)
