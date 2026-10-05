"""Demand disaggregation must conserve every demand key's energy (HF-28 / HF-29).

``WriteStrategy.dissagregate_demand`` splits a zonal demand table across buses
with per-bus load allocation factors. The factors are *normalised* somewhere
else than they are *consumed*: ``build_base_network`` normalises ``LAF_state``
over ``full_state`` (which separates the District of Columbia), while the
consumption key is ``n.buses.reeds_state`` on develop and ``n.buses.state`` on
master. Neither reproduces ``full_state``, so ``sum(laf)`` inside a demand key
was not 1 and the allocated demand was not the demand table's column: measured
on the whole-USA benchmark 2026-09-15, +2.28 % on master and +1.85 % on develop
against the EFS state totals (Maryland's factors summed to 2.008 on develop).

The fix renormalises the factors within the consumption key, whatever that key
is, and folds the District of Columbia's demand column into Maryland so it is
not silently discarded.

These tests pin:
  (a) per-key conservation when the normalisation key and the consumption key
      disagree -- exactly the Maryland/DC geometry of the bug,
  (b) a key with zero factor mass allocates zero (never NaN) and says so,
  (c) the DC fold: a demand column with no bus under the current key lands in
      Maryland once, and the national total is conserved,
  (d) the renormalisation is not state-specific -- a 'ba' key with factors
      normalised over states conserves too,
  (e) a demand column that no bus carries is logged with the MW it drops.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import get_multiindex_snapshots
from build_demand import WritePopulation, WriteStrategy

pytestmark = pytest.mark.fast

#: The four-bus toy network. ``laf`` is what ``build_base_network`` writes:
#: normalised over ``full_state``, so the DC bus carries a full unit of mass on
#: its own while the two Maryland buses carry 0.6 / 0.4 between them. Its
#: consumption key (``reeds_state``) files that bus under Maryland, so Maryland's
#: factors sum to 2.0 -- the defect, in miniature.
BUSES = {
    #        reeds_state  reeds_zone  laf   comment
    "md_a": ("MD", "p123", 0.6),
    "md_b": ("MD", "p123", 0.4),
    "dc": ("MD", "p123", 1.0),  # geographically DC, keyed to Maryland
    "va": ("VA", "p99", 1.0),
}

CODE_2_NAME = {"MD": "Maryland", "VA": "Virginia", "DE": "Delaware"}


def make_network(buses=BUSES):
    """A network with the two columns the state disaggregation reads."""
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        {"start": "2030-01-01 00:00", "end": "2030-01-01 03:00", "inclusive": "both"},
        [2030],
    )
    n.set_investment_periods(periods=[2030])
    for bus in buses:
        n.add("Bus", bus)
    n.buses["reeds_state"] = [buses[b][0] for b in buses]
    n.buses["reeds_zone"] = [buses[b][1] for b in buses]
    n.buses["LAF_state"] = [buses[b][2] for b in buses]
    n.buses["load_weight"] = [buses[b][2] for b in buses]
    return n


def make_demand(n, values):
    """Zonal demand on the reader's 4-level contract: one column per state key."""
    periods = len(next(iter(values.values())))
    snapshots = n.snapshots.get_level_values(1)[:periods]
    frame = pd.DataFrame(
        {key: np.asarray(series, dtype=float) for key, series in values.items()},
        index=snapshots,
    )
    frame.index.name = "snapshot"
    frame["sector"] = "all"
    frame["subsector"] = "all"
    frame["fuel"] = "electricity"
    return frame.set_index(["sector", "subsector", "fuel"], append=True)


def allocated_by_key(result, n):
    """Sum the disaggregated load back up to the state key it came from."""
    keys = n.buses.reeds_state.map(CODE_2_NAME)
    return result.T.groupby(keys.reindex(result.columns)).sum().T


# --- (a) per-key conservation when the two keys disagree ---------------------


def test_per_key_demand_is_conserved_when_laf_is_normalised_on_another_column():
    """Maryland's factors sum to 2.0; its buses must still receive exactly D[MD]."""
    n = make_network()
    demand = make_demand(
        n,
        {"Maryland": [1000, 1100, 1200, 1300], "Virginia": [500, 400, 300, 200]},
    )

    result = WritePopulation(n).dissagregate_demand(demand, "state")

    per_key = allocated_by_key(result, n)
    expected = demand.droplevel(["sector", "subsector", "fuel"])
    for key in ("Maryland", "Virginia"):
        pd.testing.assert_series_equal(
            per_key[key],
            expected[key],
            check_names=False,
            rtol=0,
            atol=1e-9,
        )

    # and the national total, which is what the +1.85 % error was measured on
    assert result.sum().sum() == pytest.approx(expected.to_numpy().sum(), abs=1e-9)


def test_the_unfixed_allocation_would_have_doubled_maryland():
    """Guard the premise: without renormalisation Maryland gets 2x its demand."""
    n = make_network()
    demand = make_demand(n, {"Maryland": [1000.0], "Virginia": [500.0]})
    raw = n.buses.LAF_state
    maryland = n.buses.index[n.buses.reeds_state == "MD"]

    assert raw.loc[maryland].sum() == pytest.approx(2.0)

    result = WritePopulation(n).dissagregate_demand(demand, "state")
    assert allocated_by_key(result, n).loc[:, "Maryland"].iloc[0] == pytest.approx(1000.0)
    # the shares keep their relative proportions, they are only rescaled
    assert result.loc[:, "dc"].iloc[0] == pytest.approx(500.0)
    assert result.loc[:, "md_a"].iloc[0] == pytest.approx(300.0)
    assert result.loc[:, "md_b"].iloc[0] == pytest.approx(200.0)


# --- (b) a key with zero factor mass ----------------------------------------


def test_zero_laf_mass_allocates_zero_and_is_logged(caplog):
    """A state whose buses carry no weight gets 0.0, not NaN, and a warning."""
    buses = dict(BUSES)
    buses["de"] = ("DE", "p122", 0.0)
    n = make_network(buses)
    demand = make_demand(
        n,
        {"Maryland": [1000.0], "Virginia": [500.0], "Delaware": [100.0]},
    )

    with caplog.at_level(logging.WARNING, logger="build_demand"):
        result = WritePopulation(n).dissagregate_demand(demand, "state")

    assert not result.isna().any().any()
    assert float(result.get("de", pd.Series([0.0])).sum()) == pytest.approx(0.0)
    # Delaware's demand is unplaceable and therefore absent from the total
    assert result.sum().sum() == pytest.approx(1500.0, abs=1e-9)

    warnings = "\n".join(record.getMessage() for record in caplog.records)
    assert "Delaware" in warnings
    assert "100.0" in warnings


# --- (c) the District of Columbia fold ---------------------------------------


def test_district_of_columbia_demand_is_folded_into_maryland_exactly_once():
    """DC has a demand column but no bus key; its MW must land in Maryland."""
    n = make_network()
    demand = make_demand(
        n,
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
    n = make_network()
    demand = make_demand(n, {"Maryland": [1000.0], "District of Columbia": [200.0]})
    folded = WriteStrategy._fold_demand_keys(
        demand.droplevel(["sector", "subsector", "fuel"]),
        "state",
    )

    assert "District of Columbia" not in folded.columns
    assert folded["Maryland"].iloc[0] == pytest.approx(1200.0)
    # a non-state key is left alone: the fold is a state-name rule
    untouched = WriteStrategy._fold_demand_keys(
        demand.droplevel(["sector", "subsector", "fuel"]),
        "reeds",
    )
    assert "District of Columbia" in untouched.columns


# --- (d) the renormalisation is key-agnostic ---------------------------------


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
    demand = make_demand(n, {"PJM": [3000.0, 1500.0]})

    result = _WriteFixedLaf(n, laf).dissagregate_demand(demand, "ba")

    assert result.sum(axis=1).tolist() == pytest.approx([3000.0, 1500.0])
    assert result.loc[:, "md_a"].tolist() == pytest.approx([600.0, 300.0])


# --- (e) a demand column no bus carries --------------------------------------


def test_demand_for_an_absent_key_is_logged_with_its_mw(caplog):
    """The silent drop that hid the DC bug must now be loud."""
    n = make_network()
    demand = make_demand(n, {"Maryland": [1000.0], "Virginia": [500.0], "Texas": [700.0]})

    with caplog.at_level(logging.WARNING, logger="build_demand"):
        result = WritePopulation(n).dissagregate_demand(demand, "state")

    warnings = "\n".join(record.getMessage() for record in caplog.records)
    assert "Texas" in warnings
    assert "700.0" in warnings
    assert result.sum().sum() == pytest.approx(1500.0, abs=1e-9)


# --- (f) the fold must never override a key a bus actually carries -----------


def test_fold_is_skipped_when_a_bus_carries_the_dc_key():
    """A DC bus with a DC demand column keeps its own demand; nothing is folded."""
    buses = dict(BUSES)
    buses["dc"] = ("DC", "p123", 1.0)  # the future fix: DC keeps its own key
    n = make_network(buses)
    demand = make_demand(
        n,
        {"Maryland": [100.0], "Virginia": [50.0], "District of Columbia": [50.0]},
    )

    result = WritePopulation(n).dissagregate_demand(demand, "state")

    assert float(result["dc"].iloc[0]) == pytest.approx(50.0)
    assert float(result[["md_a", "md_b"]].iloc[0].sum()) == pytest.approx(100.0)
    assert result.sum().sum() == pytest.approx(200.0, abs=1e-9)


# --- (g) a share under LAF_DROP_THRESHOLD is zeroed and the rest rescaled ----


def test_sub_threshold_share_is_zeroed_and_the_key_still_conserves():
    """A bus with a vanishing factor gets exactly 0; its key's demand is intact."""
    buses = dict(BUSES)
    buses["md_tiny"] = ("MD", "p123", 1e-9)
    n = make_network(buses)
    demand = make_demand(n, {"Maryland": [1000.0], "Virginia": [500.0]})

    result = WritePopulation(n).dissagregate_demand(demand, "state")

    assert "md_tiny" not in result.columns or float(result["md_tiny"].iloc[0]) == 0.0
    per_key = allocated_by_key(result, n)
    assert per_key["Maryland"].iloc[0] == pytest.approx(1000.0, abs=1e-9)
    assert result.sum().sum() == pytest.approx(1500.0, abs=1e-9)
