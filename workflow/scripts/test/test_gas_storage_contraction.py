"""
Test the gas storage contraction constraints.

With ``infra_contraction``, the store, injection and withdrawal of an underground
gas storage are each optimised between zero and their existing capacity, and
``add_gas_storage_contraction_constraints`` keeps the injection at least at the
share of its existing capacity that the other two keep:
charge / charge_0 >= discharge / discharge_0 and charge / charge_0 >= store / store_0.
"""

import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from _helpers import get_multiindex_snapshots
from build_natural_gas import GasStorage
from opts.sector import add_gas_storage_contraction_constraints

pytestmark = pytest.mark.fast

CHARGE_0, DISCHARGE_0, STORE_0 = 20.0, 60.0, 5000.0
INJECTION_LOSS = 0.0103

STORE = "CA gas storage"
CHARGE = "CA charge gas storage"
DISCHARGE = "CA discharge gas storage"


def storage_network(contraction: bool = True, hours: int = 2) -> pypsa.Network:
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
        contraction=contraction,
    )
    return n


def solve(n: pypsa.Network, supply: float, load: list[float]) -> None:
    """Meet ``load`` from a fixed ``supply`` and the storage, with the constraints added."""
    n.add("Generator", "supply", bus="CA gas", p_nom=supply, marginal_cost=1.0, build_year=2030, lifetime=20)
    n.add("Load", "demand", bus="CA gas", p_set=load)
    status, condition = n.optimize(
        multi_investment_periods=True,
        solver_name="highs",
        extra_functionality=lambda n, sns: add_gas_storage_contraction_constraints(n),
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
    ("row_name", "variable", "part", "part_0"),
    [
        ("Link-gas_storage_discharge_contraction", "Link-p_nom", DISCHARGE, DISCHARGE_0),
        ("Link-gas_storage_store_contraction", "Store-e_nom", STORE, STORE_0),
    ],
)
def test_rows_are_scaled_to_reciprocal_coefficients(row_name, variable, part, part_0):
    """Charge / sqrt(r) - sqrt(r) * part >= 0 with r = charge_0 / part_0, i.e. charge - r * part >= 0 rescaled."""
    n = storage_network()
    n.optimize.create_model(multi_investment_periods=True)
    add_gas_storage_contraction_constraints(n)
    model = n.model

    row = model.constraints[row_name]
    sqrt_r = (CHARGE_0 / part_0) ** 0.5
    assert terms(row, name=0) == pytest.approx(
        {
            var_label(model, "Link-p_nom", CHARGE): 1 / sqrt_r,
            var_label(model, variable, part): -sqrt_r,
        },
    )
    assert row.sign.item() == ">="
    assert float(row.rhs.item()) == 0.0


def test_injection_sized_to_its_own_need():
    """
    Solved behaviour: the injection need sets its share when it is the larger one.

    The 30 MW supply meets a 10 MW then 40 MW load, so 10 MW is withdrawn in the
    second hour and 10 / (1 - injection loss) MW injected in the first, half of the
    existing injection but a sixth of the existing withdrawal. The injection
    carries the fixed O&M, so it is sized to its need, and the withdrawal and
    store stay within its share.
    """
    n = storage_network()
    solve(n, supply=30.0, load=[10.0, 40.0])

    charge, discharge, store = shares(n)
    assert charge == pytest.approx(10.0 / (1 - INJECTION_LOSS) / CHARGE_0, rel=1e-6)
    assert discharge <= charge + 1e-9
    assert store <= charge + 1e-9


def test_withdrawal_need_lifts_the_injection():
    """
    Solved behaviour: a larger withdrawal share lifts the injection to it.

    The 31 MW supply refills the storage at 4 MW over six hours of a 27 MW load
    and a 48 MW peak takes 17 MW out of it in the seventh. The injection needs only
    17 / (1 - injection loss) / 6 MW, but the withdrawal keeps 17 / 60 of its
    existing capacity, so the injection keeps the same share: 20 * 17 / 60 MW.
    """
    n = storage_network(hours=7)
    solve(n, supply=31.0, load=[27.0] * 6 + [48.0])

    charge, discharge, store = shares(n)
    assert discharge == pytest.approx(17.0 / DISCHARGE_0, abs=1e-6)
    assert charge == pytest.approx(discharge, abs=1e-6)
    assert store <= charge + 1e-9


def test_fixed_storage_is_not_constrained():
    """Without contraction no storage part is extendable, so no rows are added."""
    n = storage_network(contraction=False)
    # an extendable store of another carrier, so the model has an objective
    n.add("Store", "other", bus="CA gas", e_nom_extendable=True, capital_cost=1.0, build_year=2030, lifetime=20)
    n.optimize.create_model(multi_investment_periods=True)
    add_gas_storage_contraction_constraints(n)

    assert "Link-gas_storage_discharge_contraction" not in n.model.constraints
    assert "Link-gas_storage_store_contraction" not in n.model.constraints
