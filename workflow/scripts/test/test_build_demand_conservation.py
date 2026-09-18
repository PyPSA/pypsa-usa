"""Demand disaggregation must conserve every demand key's energy (HF-28 / HF-29).

Ported from develop's ``wip/hf28-demand``; the baseline branch has to carry the
same fix or the whole-USA comparison measures the fix instead of the refactor.

``WriteStrategy.dissagregate_demand`` splits a zonal demand table across buses
with per-bus load allocation factors. The factors are *normalised* somewhere
else than they are *consumed*: ``build_base_network`` normalises ``LAF_state``
over ``full_state`` (which separates the District of Columbia), while the
consumption key on this branch is ``n.buses.state``. That column does not
reproduce ``full_state``, so ``sum(laf)`` inside a demand key was not 1 and the
allocated demand was not the demand table's column: measured on the whole-USA
benchmark 2026-09-15, +2.28 % against the EFS state totals here (Maryland
1.588, Virginia 1.409).

The fix renormalises the factors within the consumption key, whatever that key
is, and folds the District of Columbia's demand column into Maryland so it is
not silently discarded.
"""

import os
import sys

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_demand import WritePopulation, WriteStrategy

#: The four-bus toy network. ``laf`` is what ``build_base_network`` writes:
#: normalised over ``full_state``, so the DC bus carries a full unit of mass on
#: its own while the two Maryland buses carry 0.6 / 0.4 between them. Its
#: consumption key (``state``) files that bus under Maryland, so Maryland's
#: factors sum to 2.0 -- the defect, in miniature.
BUSES = {
    #        state        laf
    "md_a": ("Maryland", 0.6),
    "md_b": ("Maryland", 0.4),
    "dc": ("Maryland", 1.0),  # geographically DC, keyed to Maryland
    "va": ("Virginia", 1.0),
}


def make_network(buses=BUSES):
    """A network with the two columns the state disaggregation reads."""
    n = pypsa.Network()
    for bus in buses:
        n.add("Bus", bus)
    n.buses["state"] = [buses[b][0] for b in buses]
    n.buses["LAF_state"] = [buses[b][1] for b in buses]
    n.buses["Pd"] = [buses[b][1] for b in buses]
    return n


def make_demand(values):
    """Zonal demand on the reader's 4-level contract: one column per state key."""
    periods = len(next(iter(values.values())))
    snapshots = pd.date_range("2030-01-01", periods=periods, freq="h", name="snapshot")
    frame = pd.DataFrame(
        {key: np.asarray(series, dtype=float) for key, series in values.items()},
        index=snapshots,
    )
    frame["sector"] = "all"
    frame["subsector"] = "all"
    frame["fuel"] = "electricity"
    return frame.set_index(["sector", "subsector", "fuel"], append=True)


def allocated_by_key(result, n):
    """Sum the disaggregated load back up to the state key it came from."""
    keys = n.buses.state.reindex(result.columns)
    return result.T.groupby(keys).sum().T


def test_per_key_demand_is_conserved_when_laf_is_normalised_on_another_column():
    """Maryland's factors sum to 2.0; its buses must still receive exactly D[MD]."""
    n = make_network()
    demand = make_demand({"Maryland": [1000, 1100, 1200], "Virginia": [500, 400, 300]})

    result = WritePopulation(n).dissagregate_demand(demand, "state")

    per_key = allocated_by_key(result, n)
    expected = demand.droplevel(["sector", "subsector", "fuel"])
    for key in ("Maryland", "Virginia"):
        assert per_key[key].tolist() == pytest.approx(expected[key].tolist(), abs=1e-9)
    assert result.sum().sum() == pytest.approx(expected.to_numpy().sum(), abs=1e-9)


def test_the_unfixed_allocation_would_have_doubled_maryland():
    """Guard the premise: without renormalisation Maryland gets 2x its demand."""
    n = make_network()
    demand = make_demand({"Maryland": [1000.0], "Virginia": [500.0]})

    assert n.buses.LAF_state[n.buses.state == "Maryland"].sum() == pytest.approx(2.0)

    result = WritePopulation(n).dissagregate_demand(demand, "state")
    assert allocated_by_key(result, n).loc[:, "Maryland"].iloc[0] == pytest.approx(1000.0)
    assert result.loc[:, "dc"].iloc[0] == pytest.approx(500.0)
    assert result.loc[:, "md_a"].iloc[0] == pytest.approx(300.0)
    assert result.loc[:, "md_b"].iloc[0] == pytest.approx(200.0)


def test_zero_laf_mass_allocates_zero_and_is_logged(caplog):
    """A state whose buses carry no weight gets 0.0, not NaN, and a warning."""
    buses = dict(BUSES)
    buses["de"] = ("Delaware", 0.0)
    n = make_network(buses)
    demand = make_demand({"Maryland": [1000.0], "Virginia": [500.0], "Delaware": [100.0]})

    with caplog.at_level("WARNING", logger="build_demand"):
        result = WritePopulation(n).dissagregate_demand(demand, "state")

    assert not result.isna().any().any()
    assert float(result.get("de", pd.Series([0.0])).sum()) == pytest.approx(0.0)
    assert result.sum().sum() == pytest.approx(1500.0, abs=1e-9)

    warnings = "\n".join(record.getMessage() for record in caplog.records)
    assert "Delaware" in warnings
    assert "100.0" in warnings


def test_district_of_columbia_demand_is_folded_into_maryland_exactly_once():
    """DC has a demand column but no bus key; its MW must land in Maryland."""
    n = make_network()
    demand = make_demand(
        {
            "Maryland": [1000.0, 1000.0],
            "Virginia": [500.0, 500.0],
            "District of Columbia": [200.0, 100.0],
        },
    )

    result = WritePopulation(n).dissagregate_demand(demand, "state")

    per_key = allocated_by_key(result, n)
    assert per_key["Maryland"].tolist() == pytest.approx([1200.0, 1100.0])
    assert per_key["Virginia"].tolist() == pytest.approx([500.0, 500.0])
    assert result.sum().sum() == pytest.approx(3300.0, abs=1e-9)
    assert list(per_key.columns) == ["Maryland", "Virginia"]


def test_without_the_fold_the_dc_column_would_be_dropped():
    """Pin the size of the bug the fold closes: the whole DC column."""
    demand = make_demand({"Maryland": [1000.0], "District of Columbia": [200.0]})
    flat = demand.droplevel(["sector", "subsector", "fuel"])

    folded = WriteStrategy._fold_demand_keys(flat, "state")
    assert "District of Columbia" not in folded.columns
    assert folded["Maryland"].iloc[0] == pytest.approx(1200.0)

    untouched = WriteStrategy._fold_demand_keys(flat, "reeds")
    assert "District of Columbia" in untouched.columns


class _WriteFixedLaf(WriteStrategy):
    """A writer whose factors are normalised over states, consumed over BAs.

    This is ``WriteIndustrial`` in miniature: its ``_get_load_allocation_factor``
    builds state-normalised county shares and hands them to a ``ba`` (or
    ``reeds``) consumption key.
    """

    def __init__(self, n, laf):
        super().__init__(n)
        self._laf = laf

    def _get_load_allocation_factor(self, df=None, **kwargs):
        return self._laf


def test_renormalisation_is_applied_for_every_zone_type():
    """A 'ba' key whose factors were normalised over states still conserves."""
    n = make_network()
    n.buses["balancing_area"] = ["PJM", "PJM", "PJM", "PJM"]
    laf = pd.Series([0.6, 0.4, 1.0, 1.0], index=n.buses.index)
    demand = make_demand({"PJM": [3000.0, 1500.0]})

    result = _WriteFixedLaf(n, laf).dissagregate_demand(demand, "ba")

    assert result.sum(axis=1).tolist() == pytest.approx([3000.0, 1500.0])
    assert result.loc[:, "md_a"].tolist() == pytest.approx([600.0, 300.0])


def test_demand_for_an_absent_key_is_logged_with_its_mw(caplog):
    """The silent drop that hid the DC bug must now be loud."""
    n = make_network()
    demand = make_demand({"Maryland": [1000.0], "Virginia": [500.0], "Texas": [700.0]})

    with caplog.at_level("WARNING", logger="build_demand"):
        result = WritePopulation(n).dissagregate_demand(demand, "state")

    warnings = "\n".join(record.getMessage() for record in caplog.records)
    assert "Texas" in warnings
    assert "700.0" in warnings
    assert result.sum().sum() == pytest.approx(1500.0, abs=1e-9)


# --- (f) the fold must never override a key a bus actually carries -----------


def test_fold_is_skipped_when_a_bus_carries_the_dc_key():
    """A DC bus with a DC demand column keeps its own demand; nothing is folded."""
    buses = dict(BUSES)
    buses["dc"] = ("District of Columbia", 1.0)  # the future fix: DC keeps its own key
    n = make_network(buses)
    demand = make_demand({"Maryland": [100.0], "Virginia": [50.0], "District of Columbia": [50.0]})

    result = WritePopulation(n).dissagregate_demand(demand, "state")

    assert float(result["dc"].iloc[0]) == pytest.approx(50.0)
    assert float(result[["md_a", "md_b"]].iloc[0].sum()) == pytest.approx(100.0)
    assert result.sum().sum() == pytest.approx(200.0, abs=1e-9)


# --- (g) a share under LAF_DROP_THRESHOLD is zeroed and the rest rescaled ----


def test_sub_threshold_share_is_zeroed_and_the_key_still_conserves():
    """A bus with a vanishing factor gets exactly 0; its key's demand is intact."""
    buses = dict(BUSES)
    buses["md_tiny"] = ("Maryland", 1e-9)
    n = make_network(buses)
    demand = make_demand({"Maryland": [1000.0], "Virginia": [500.0]})

    result = WritePopulation(n).dissagregate_demand(demand, "state")

    assert "md_tiny" not in result.columns or float(result["md_tiny"].iloc[0]) == 0.0
    per_key = allocated_by_key(result, n)
    assert per_key["Maryland"].iloc[0] == pytest.approx(1000.0, abs=1e-9)
    assert result.sum().sum() == pytest.approx(1500.0, abs=1e-9)
