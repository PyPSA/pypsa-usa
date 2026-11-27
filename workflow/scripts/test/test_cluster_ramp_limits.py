"""Ramp limits of generators merged by ``cluster_network`` (#989)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cluster_network import clustering_for_n_clusters

#: merges the fixture's two eastern buses
EAST_BUSMAP = pd.Series({"z1": "z1", "z2": "X", "z3": "X"})
STRATEGIES = {
    "generators": {
        "ramp_limit_up": "capacity_weighted_average",
        "ramp_limit_down": "capacity_weighted_average",
    },
}


def _add_pair(n, carrier, limits, p_nom):
    names = [f"{carrier}2", f"{carrier}3"]
    n.add(
        "Generator",
        names,
        bus=["z2", "z3"],
        carrier=carrier,
        p_nom=list(p_nom),
        ramp_limit_up=list(limits),
        ramp_limit_down=list(limits),
    )


@pytest.mark.fast
def test_merged_ramp_limits(base_network):
    n = base_network.copy()
    _add_pair(n, "hydro", [np.nan, np.nan], [10.0, 30.0])  # renewables have no limit
    _add_pair(n, "coal", [0.4, 0.8], [100.0, 300.0])
    _add_pair(n, "biomass", [np.nan, 0.5], [100.0, 100.0])
    _add_pair(n, "geothermal", [0.2, 0.6], [0.0, 0.0])  # no capacity: plain mean
    clustered = clustering_for_n_clusters(
        n,
        2,
        custom_busmap=EAST_BUSMAP,
        aggregate_carriers=["hydro", "coal", "biomass", "geothermal"],
        aggregation_strategies=STRATEGIES,
    ).n
    gens = clustered.generators.set_index("carrier").loc[["hydro", "coal", "biomass", "geothermal"]]
    for attr in ("ramp_limit_up", "ramp_limit_down"):
        limits = gens[attr]
        # PyPSA's capacity_weighted_average gives 0 here, which freezes the output
        assert np.isnan(limits["hydro"])
        assert limits["coal"] == pytest.approx((0.4 * 100 + 0.8 * 300) / 400)
        # the unit without a limit can move its whole capacity
        assert limits["biomass"] == pytest.approx((1.0 * 100 + 0.5 * 100) / 200)
        assert limits["geothermal"] == pytest.approx(0.4)


@pytest.mark.fast
def test_other_ramp_strategies_are_kept(base_network):
    n = base_network.copy()
    _add_pair(n, "coal", [0.4, 0.8], [100.0, 300.0])
    clustered = clustering_for_n_clusters(
        n,
        2,
        custom_busmap=EAST_BUSMAP,
        aggregate_carriers=["coal"],
        aggregation_strategies={"generators": {"ramp_limit_up": "max", "ramp_limit_down": "min"}},
    ).n
    coal = clustered.generators.set_index("carrier").loc["coal"]
    assert coal["ramp_limit_up"] == pytest.approx(0.8)
    assert coal["ramp_limit_down"] == pytest.approx(0.4)
