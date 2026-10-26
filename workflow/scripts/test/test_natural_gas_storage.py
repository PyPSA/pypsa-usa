"""Unit tests for underground gas storage built from EIA-191 in ``build_natural_gas.GasStorage``.

The raw frame mirrors the PUDL ``core_eia191__monthly_gas_storage`` table, and
``read_data`` is patched to return it, so nothing here touches the network.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_natural_gas import GasStorage
from constants import NG_STORAGE_CHARGE_DISCHARGE_RATIO
from summary_natural_gas import get_underground_storage

pytestmark = pytest.mark.fast

MWH_PER_MCF = 303.5 / 1000

COLUMNS = [
    "storage_field_id_eia",
    "report_date",
    "state",
    "field_type",
    "operational_status",
    "base_gas_mcf",
    "working_gas_capacity_mcf",
    "total_field_capacity_mcf",
    "maximum_daily_delivery_mcf",
]

RAW = pd.DataFrame(
    [
        # only the latest month of the report is used
        ["r1", "2025-11-01", "CA", "depleted field", "active", 9e6, 9e6, 9e6, 9e6],
        ["r1", "2026-05-01", "CA", "depleted field", "active", 1.0e6, 2.0e6, 3.0e6, 240e3],
        # missing total is base + working
        ["r2", "2026-05-01", "CA", "depleted field", "active", 0.5e6, 1.5e6, np.nan, 120e3],
        # missing working gas capacity is total - base
        ["r3", "2026-05-01", "CA", "aquifer", "active", 0.6e6, np.nan, 1.0e6, 48e3],
        ["r4", "2026-05-01", "CA", "aquifer", "inactive", 5e6, 5e6, 10e6, 100e3],
        # the cushion is total - working, not the reported (non-additive) base gas
        ["r5", "2026-05-01", "TX", "salt dome", "active", 0.5e6, 1.6e6, 2.0e6, 480e3],
        # an ID that stopped reporting (e.g. reported again under a new owner)
        ["r6", "2021-10-01", "CA", "depleted field", "active", 7e6, 7e6, 14e6, 700e3],
        ["r7", "2026-05-01", "AK", "depleted field", "active", 1e6, 1e6, 2e6, 10e3],
    ],
    columns=COLUMNS,
)


@pytest.fixture
def make_storage(monkeypatch):
    monkeypatch.setattr(GasStorage, "read_data", lambda self: RAW.copy())

    def make(aggregation="state", year=2018):
        return GasStorage(year, "usa", "unused", aggregation=aggregation)

    return make


def test_storage_sums_the_latest_active_reports_per_state(make_storage):
    data = make_storage().data.set_index("STATE")

    assert sorted(data.index) == ["CA", "TX"]  # AK is out of every interconnect
    ca, tx = data.loc["CA"], data.loc["TX"]
    assert ca.MAX_CAPACITY_MWH == pytest.approx((3.0e6 + 2.0e6 + 1.0e6) * MWH_PER_MCF)
    assert ca.MIN_CAPACITY_MWH == pytest.approx((1.0e6 + 0.5e6 + 0.6e6) * MWH_PER_MCF)
    assert ca.WORKING_CAPACITY_MWH == pytest.approx((2.0e6 + 1.5e6 + 0.4e6) * MWH_PER_MCF)
    assert ca.DISCHARGE_MW == pytest.approx((240e3 + 120e3 + 48e3) * MWH_PER_MCF / 24)
    assert ca.CHARGE_MW == pytest.approx(ca.DISCHARGE_MW * NG_STORAGE_CHARGE_DISCHARGE_RATIO)
    assert (tx.MAX_CAPACITY_MWH, tx.MIN_CAPACITY_MWH, tx.WORKING_CAPACITY_MWH) == pytest.approx(
        (2.0e6 * MWH_PER_MCF, 0.4e6 * MWH_PER_MCF, 1.6e6 * MWH_PER_MCF),
    )
    # the usable range of the store is the working gas capacity
    assert (data.MAX_CAPACITY_MWH - data.MIN_CAPACITY_MWH).tolist() == pytest.approx(data.WORKING_CAPACITY_MWH.tolist())


def test_storage_by_formation(make_storage):
    data = make_storage("formation").data.set_index("NAME")

    assert sorted(data.index) == ["CA aquifer", "CA depleted-field", "TX salt-dome"]
    assert data.at["CA aquifer", "MAX_CAPACITY_MWH"] == pytest.approx(1.0e6 * MWH_PER_MCF)
    assert data.at["CA depleted-field", "MAX_CAPACITY_MWH"] == pytest.approx(5.0e6 * MWH_PER_MCF)


def test_latest_report_does_not_depend_on_model_year(make_storage):
    pd.testing.assert_frame_equal(make_storage(year=2018).data, make_storage(year=2030).data)


def test_unknown_aggregation_raises(make_storage):
    with pytest.raises(ValueError, match="aggregation"):
        make_storage("county")


@pytest.mark.parametrize("aggregation", ["state", "formation"])
def test_storage_components(base_network, make_storage, aggregation):
    """One store with base gas as its floor, charged and discharged through the state gas bus."""
    n = base_network
    n.add("Bus", ["CA", "TX"], suffix=" gas", carrier="gas")
    storage = make_storage(aggregation)

    storage.build_infrastructure(n)

    for _, row in storage.data.iterrows():
        name = row.NAME
        store = n.stores.loc[f"{name} gas storage"]
        assert store.bus == f"{name} gas storage"
        assert store.e_nom == pytest.approx(row.MAX_CAPACITY_MWH)
        assert 0 < store.e_min_pu == pytest.approx(row.MIN_CAPACITY_MWH / row.MAX_CAPACITY_MWH)
        charge = n.links.loc[f"{name} charge gas storage"]
        discharge = n.links.loc[f"{name} discharge gas storage"]
        assert (charge.bus0, charge.bus1) == (f"{row.STATE} gas", f"{name} gas storage")
        assert (discharge.bus0, discharge.bus1) == (f"{name} gas storage", f"{row.STATE} gas")
        assert discharge.p_nom == pytest.approx(row.DISCHARGE_MW)
        assert charge.p_nom == pytest.approx(NG_STORAGE_CHARGE_DISCHARGE_RATIO * row.DISCHARGE_MW)
    assert n.buses.loc[storage.data.NAME + " gas storage", "country"].tolist() == storage.data.STATE.tolist()


def test_underground_storage_summary_sums_formations(base_network, make_storage):
    n = base_network
    n.add("Bus", ["CA", "TX"], suffix=" gas", carrier="gas")
    make_storage("formation").build_infrastructure(n)
    stores = n.stores.index[n.stores.carrier == "gas storage"]
    n.stores_t.e = pd.DataFrame(1.0, index=n.snapshots, columns=stores)

    data = get_underground_storage(n)

    assert sorted(data) == ["CA", "TX"]
    assert (data["CA"]["CA gas storage"] == 2.0).all()  # depleted field + aquifer
    assert (data["TX"]["TX gas storage"] == 1.0).all()
