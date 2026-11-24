"""
Test the gas storage constraints.

When a gas storage may retire or is retrofitted to hydrogen, its store, injection
and withdrawal are each optimised between zero and their existing capacity, and
``add_gas_storage_constraints`` ties them: the withdrawal keeps the
share of its existing capacity that the store keeps, and the injection at least
that share and at most the withdrawal:
discharge / discharge_0 == store / store_0, charge / charge_0 >= store / store_0
and charge <= discharge.
"""

import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from _helpers import get_multiindex_snapshots
from build_natural_gas import GasStorage
from opts.sector import add_gas_storage_constraints

pytestmark = pytest.mark.fast

CHARGE_0, DISCHARGE_0, STORE_0 = 20.0, 60.0, 5000.0
INJECTION_LOSS = 0.0103

STORE = "CA gas storage"
CHARGE = "CA charge gas storage"
DISCHARGE = "CA discharge gas storage"

ROWS = ("Link-gas_storage_withdrawal_share", "Link-gas_storage_injection_share", "Link-gas_storage_injection_limit")


def storage_network(retire: bool = True, hours: int = 2) -> pypsa.Network:
    """One California gas storage on a gas bus, over ``hours`` hourly snapshots."""
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": f"2030-01-01 {hours - 1:02d}:00", "inclusive": "both"},
        invest_periods=[2030],
    )
    n.set_investment_periods(periods=[2030])
    n.add("Bus", "p1", carrier="AC", reeds_state="CA")
    n.add("Bus", "CA gas", carrier="gas")

    storage = GasStorage.__new__(GasStorage)
    storage.interconnect = "usa"
    storage._data = pd.DataFrame(
        {
            "STATE": ["CA"],
            "NAME": ["CA"],
            "MAX_CAPACITY_MWH": [STORE_0],
            "MIN_CAPACITY_MWH": [2000.0],
            "CHARGE_MW": [CHARGE_0],
            "DISCHARGE_MW": [DISCHARGE_0],
            "e_initial": [4000.0],
        },
    )
    storage.build_infrastructure(
        n,
        cyclic_storage=True,
        injection_loss=INJECTION_LOSS,
        compressor_fom=1277.7,
        economic_retirement=retire,
    )
    return n


def solve(n: pypsa.Network, supply: float, load: list[float], availability: list[float] | None = None) -> None:
    """Meet ``load`` from a ``supply`` available as ``availability`` and the storage, with the constraints added."""
    p_max_pu = pd.Series(availability, index=n.snapshots) if availability else 1.0
    n.add(
        "Generator",
        "supply",
        bus="CA gas",
        p_nom=supply,
        p_max_pu=p_max_pu,
        marginal_cost=1.0,
        build_year=2030,
        lifetime=20,
    )
    n.add("Load", "demand", bus="CA gas", p_set=load)
    status, condition = n.optimize(
        multi_investment_periods=True,
        solver_name="highs",
        extra_functionality=lambda n, sns: add_gas_storage_constraints(n),
    )
    assert (status, condition) == ("ok", "optimal")


def shares(n: pypsa.Network) -> tuple[float, float, float]:
    """Optimised share of the existing capacity of the injection, withdrawal and store."""
    return (
        n.links.at[CHARGE, "p_nom_opt"] / CHARGE_0,
        n.links.at[DISCHARGE, "p_nom_opt"] / DISCHARGE_0,
        n.stores.at[STORE, "e_nom_opt"] / STORE_0,
    )


def terms(constraint, **sel):
    """{variable label: coefficient} of the single constraint row picked by ``sel``."""
    row_vars = constraint.vars.isel(**sel).values.ravel()
    row_coeffs = constraint.coeffs.isel(**sel).values.ravel()
    return {int(v): float(c) for v, c in zip(row_vars, row_coeffs) if int(v) != -1}


def var_label(model, name, component):
    return int(model.variables[name].labels.sel(name=component).item())


@pytest.mark.parametrize(
    ("row_name", "rate", "rate_0", "sign"),
    [
        ("Link-gas_storage_withdrawal_share", DISCHARGE, DISCHARGE_0, "="),
        ("Link-gas_storage_injection_share", CHARGE, CHARGE_0, ">="),
    ],
)
def test_share_rows_are_scaled_to_reciprocal_coefficients(row_name, rate, rate_0, sign):
    """Rate / sqrt(r) - sqrt(r) * store with r = rate_0 / store_0, i.e. rate - r * store rescaled."""
    n = storage_network()
    n.optimize.create_model(multi_investment_periods=True)
    add_gas_storage_constraints(n)
    model = n.model

    row = model.constraints[row_name]
    sqrt_r = (rate_0 / STORE_0) ** 0.5
    assert terms(row, name=0) == pytest.approx(
        {
            var_label(model, "Link-p_nom", rate): 1 / sqrt_r,
            var_label(model, "Store-e_nom", STORE): -sqrt_r,
        },
    )
    assert row.sign.item() == sign
    assert float(row.rhs.item()) == 0.0


def test_injection_limit_row():
    """Charge - discharge <= 0, in MW."""
    n = storage_network()
    n.optimize.create_model(multi_investment_periods=True)
    add_gas_storage_constraints(n)
    model = n.model

    row = model.constraints["Link-gas_storage_injection_limit"]
    assert terms(row, name=0) == pytest.approx(
        {var_label(model, "Link-p_nom", CHARGE): 1.0, var_label(model, "Link-p_nom", DISCHARGE): -1.0},
    )
    assert row.sign.item() == "<="
    assert float(row.rhs.item()) == 0.0


def test_injection_sized_to_its_own_need():
    """
    Solved behaviour: the injection need sets its share when it is the larger one.

    The 30 MW supply meets a 10 MW then 40 MW load, so 10 MW is withdrawn in the
    second hour and 10 / (1 - injection loss) MW injected in the first, half of the
    existing injection but a sixth of the existing withdrawal. The injection
    carries the fixed O&M, so it is sized to its need; the store and withdrawal
    keep one share, within that of the injection, and withdraw at least what is
    injected.
    """
    n = storage_network()
    solve(n, supply=30.0, load=[10.0, 40.0])

    charge, discharge, store = shares(n)
    assert charge == pytest.approx(10.0 / (1 - INJECTION_LOSS) / CHARGE_0, rel=1e-6)
    assert discharge == pytest.approx(store, rel=1e-6)
    assert discharge <= charge + 1e-9
    assert charge * CHARGE_0 <= discharge * DISCHARGE_0 + 1e-6


def test_withdrawal_need_lifts_the_injection():
    """
    Solved behaviour: a larger withdrawal share lifts the injection and the store to it.

    The 31 MW supply refills the storage at 4 MW over six hours of a 27 MW load
    and a 48 MW peak takes 17 MW out of it in the seventh. The injection needs only
    17 / (1 - injection loss) / 6 MW, but the withdrawal keeps 17 / 60 of its
    existing capacity, so the store and the injection keep the same share: 20 * 17 / 60 MW.
    """
    n = storage_network(hours=7)
    solve(n, supply=31.0, load=[27.0] * 6 + [48.0])

    charge, discharge, store = shares(n)
    assert discharge == pytest.approx(17.0 / DISCHARGE_0, abs=1e-6)
    assert charge == pytest.approx(discharge, abs=1e-6)
    assert store == pytest.approx(discharge, abs=1e-6)


def test_wells_cap_the_injection():
    """
    Solved behaviour: the injection need lifts the withdrawal, and the store, to it.

    The supply is only there in the first hour, so 15 MWh for 2.5 MW of load over
    the next six is injected then, at 15 / (1 - injection loss) MW. The injection
    runs through the wells, so the withdrawal keeps at least that rate, though
    2.5 MW would do. A small fixed cost on the withdrawal keeps it at that rate.
    """
    n = storage_network(hours=7)
    n.links.at[DISCHARGE, "capital_cost"] = 1.0
    solve(n, supply=40.0, load=[10.0] + [2.5] * 6, availability=[1.0] + [0.0] * 6)

    injection = 15.0 / (1 - INJECTION_LOSS)
    assert n.links.at[CHARGE, "p_nom_opt"] == pytest.approx(injection, rel=1e-6)
    assert n.links.at[DISCHARGE, "p_nom_opt"] == pytest.approx(injection, rel=1e-6)
    _, discharge, store = shares(n)
    assert store == pytest.approx(discharge, rel=1e-6)


def test_fixed_storage_is_not_constrained():
    """A storage that may not retire, and is not retrofitted, is fixed, so no rows are added."""
    n = storage_network(retire=False)
    # an extendable store of another carrier, so the model has an objective
    n.add("Store", "other", bus="CA gas", e_nom_extendable=True, capital_cost=1.0, build_year=2030, lifetime=20)
    n.optimize.create_model(multi_investment_periods=True)
    add_gas_storage_constraints(n)

    for row in ROWS:
        assert row not in n.model.constraints
