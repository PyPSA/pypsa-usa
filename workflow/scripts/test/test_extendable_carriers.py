"""Parse-time check of ``electricity: extendable_carriers`` (``_helpers.check_extendable_carriers``)."""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import check_extendable_carriers

pytestmark = pytest.mark.fast

DEFAULT = {
    "Generator": [
        "solar",
        "onwind",
        "offwind_floating",
        "OCGT",
        "CCGT",
        "CCGT-95CCS",
        "coal",
        "nuclear",
        "hydrogen_ct",
    ],
    "StorageUnit": ["4hr_battery_storage", "8hr_battery_storage"],
    "Store": [],
    "Link": [],
}


def test_the_default_lists_raise_no_message():
    assert check_extendable_carriers(DEFAULT) == []
    assert check_extendable_carriers({**DEFAULT, "StorageUnit": ["10hr_PHS", "2hr_battery_storage"]}) == []


def test_a_generator_carrier_under_storage_units_is_reported():
    """A conventional carrier under StorageUnit is skipped by add_extra_components (#382)."""
    messages = check_extendable_carriers({**DEFAULT, "StorageUnit": ["4hr_battery_storage", "CCGT"]})
    assert len(messages) == 1
    assert "StorageUnit: ['CCGT'] are not storage carriers" in messages[0]
    assert "4hr_battery_storage" in messages[0]  # names what the list does take


def test_a_storage_carrier_under_generators_is_reported():
    messages = check_extendable_carriers({**DEFAULT, "Generator": ["solar", "8hr_PHS"]})
    assert len(messages) == 1
    assert "Generator: ['8hr_PHS'] are storage carriers" in messages[0]


def test_store_and_link_entries_are_reported_as_unused():
    messages = check_extendable_carriers({**DEFAULT, "Store": ["H2_tank"], "Link": ["HVDC"]})
    assert [m.split(":")[0] for m in messages] == [
        "electricity.extendable_carriers.Store",
        "electricity.extendable_carriers.Link",
    ]


def test_missing_lists_are_treated_as_empty():
    assert check_extendable_carriers({"Generator": ["solar"]}) == []
