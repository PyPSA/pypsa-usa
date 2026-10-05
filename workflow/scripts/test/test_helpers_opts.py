"""Parsing of the emission-cap tokens in the ``{opts}`` wildcard."""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import update_config_from_wildcards

pytestmark = pytest.mark.fast


def _wildcards(opts):
    return SimpleNamespace(opts=opts, get=lambda k, d=None: {"opts": opts}.get(k, d))


def _config(**electricity):
    return {
        "electricity": dict(electricity),
        "clustering": {"temporal": {}},
        "costs": {"emission_prices": {}},
    }


def test_co2l_factor_scales_co2base():
    cfg = update_config_from_wildcards(_config(co2base=1000.0), _wildcards("Co2L0.05-3h"), inplace=False)
    assert cfg["electricity"]["co2limit_enable"] is True
    assert cfg["electricity"]["co2limit"] == pytest.approx(50.0)


def test_co2l_factor_without_co2base_names_the_key():
    with pytest.raises(ValueError, match="co2base"):
        update_config_from_wildcards(_config(), _wildcards("Co2L0.05"), inplace=False)


def test_bare_co2l_only_enables_the_cap():
    cfg = update_config_from_wildcards(_config(co2limit=42.0), _wildcards("Co2L-3h"), inplace=False)
    assert cfg["electricity"]["co2limit_enable"] is True
    assert cfg["electricity"]["co2limit"] == 42.0  # the "2" in the name is not a factor


def test_ch4l_value_and_bare_token():
    cfg = update_config_from_wildcards(_config(), _wildcards("CH4L200"), inplace=False)
    assert cfg["electricity"]["gaslimit"] == pytest.approx(200e6)
    cfg = update_config_from_wildcards(_config(gaslimit=7.0), _wildcards("CH4L"), inplace=False)
    assert cfg["electricity"]["gaslimit_enable"] is True
    assert cfg["electricity"]["gaslimit"] == 7.0
