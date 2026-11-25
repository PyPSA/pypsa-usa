"""Unit tests for picking the carriers ``add_extra_components`` offers at every bus.

A renewable carrier with no resource site in the footprint (e.g. ``offwind_floating``
in an inland run) is skipped by ``attach_wind_and_solar``. It must not then be added at
every bus with ``p_max_pu = 1`` as if it were a new conventional technology (#974).
"""

import logging
import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from add_extra_components import get_new_carriers

pytestmark = pytest.mark.fast

SITE_CARRIERS = {"onwind", "solar", "offwind", "offwind_floating", "hydro", "EGS"}
EVERYWHERE = {"nuclear"}


def test_renewable_without_sites_is_not_offered(caplog):
    extendable = ["solar", "onwind", "offwind_floating", "CCGT", "nuclear"]
    present = {"solar", "onwind", "CCGT", "nuclear"}

    with caplog.at_level(logging.WARNING, logger="add_extra_components"):
        new = get_new_carriers(extendable, present, SITE_CARRIERS, EVERYWHERE)

    assert new == ["nuclear"]
    assert "offwind_floating" in caplog.text


def test_new_conventional_carrier_is_offered():
    extendable = ["solar", "CCGT-95CCS", "OCHT"]
    present = {"solar", "CCGT"}

    assert get_new_carriers(extendable, present, SITE_CARRIERS, set()) == ["CCGT-95CCS", "OCHT"]


def test_egs_without_sites_is_not_offered():
    assert get_new_carriers(["EGS", "CCGT"], {"CCGT"}, SITE_CARRIERS, set()) == []


def test_nuclear_is_offered_even_when_present():
    assert get_new_carriers(["nuclear"], {"nuclear"}, SITE_CARRIERS, EVERYWHERE) == ["nuclear"]


def test_no_warning_when_every_renewable_has_sites(caplog):
    with caplog.at_level(logging.WARNING, logger="add_extra_components"):
        new = get_new_carriers(["solar", "onwind"], {"solar", "onwind"}, SITE_CARRIERS, set())

    assert new == []
    assert caplog.text == ""
