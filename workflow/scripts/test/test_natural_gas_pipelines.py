"""Unit tests for the pipelines between state gas buses in ``build_natural_gas.InterconnectGasPipelineCapacity``.

The class is built without its EIA-backed ``__init__``, so nothing here needs
the API or the sector databundle.
"""

import math
import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_natural_gas import InterconnectGasPipelineCapacity

pytestmark = pytest.mark.fast

# great circle distance of 10 degrees along the equator
DISTANCE_KM = 6371.0 * math.radians(10)


def make_pipelines():
    pipelines = InterconnectGasPipelineCapacity.__new__(InterconnectGasPipelineCapacity)
    pipelines.year = 2018
    pipelines.interconnect = "usa"
    pipelines.api = None
    pipelines._data = pd.DataFrame(
        {"STATE_FROM": ["CA", "TX"], "STATE_TO": ["TX", "CA"], "CAPACITY_MW": [100.0, 50.0]},
    )
    return pipelines


@pytest.fixture
def network(base_network):
    base_network.add("Bus", ["CA gas", "TX gas"], carrier="gas", x=[0.0, 10.0], y=[0.0, 0.0])
    return base_network


KWARGS = {"length_factor": 1.25, "loss_per_1000km": 0.0155, "fom_per_km": 1.15, "contraction": False}


def test_pipeline_losses_and_fixed_om_scale_with_route_length(network):
    make_pipelines().build_infrastructure(network, **KWARGS)

    links = network.links.loc[["CA TX pipeline", "TX CA pipeline"]]
    length = 1.25 * DISTANCE_KM
    assert links.bus0.tolist() == ["CA gas", "TX gas"]
    assert links.bus1.tolist() == ["TX gas", "CA gas"]
    assert links.length.tolist() == pytest.approx([length] * 2)
    assert links.efficiency.tolist() == pytest.approx([1 - 0.0155 * length / 1000] * 2)
    assert links.capital_cost.tolist() == pytest.approx([1.15 * length] * 2)
    assert links.p_nom.tolist() == [100.0, 50.0]
    assert not links.p_nom_extendable.any()


def test_pipelines_contract_down_to_zero(network):
    make_pipelines().build_infrastructure(network, **(KWARGS | {"contraction": True}))

    links = network.links.loc[["CA TX pipeline", "TX CA pipeline"]]
    assert links.p_nom_extendable.all()
    assert (links.p_nom_min == 0).all()
    assert links.p_nom_max.tolist() == [100.0, 50.0]


@pytest.mark.parametrize("missing", list(KWARGS))
def test_pipelines_need_every_parameter(network, missing):
    kwargs = KWARGS.copy()
    del kwargs[missing]

    with pytest.raises(KeyError, match=missing):
        make_pipelines().build_infrastructure(network, **kwargs)
