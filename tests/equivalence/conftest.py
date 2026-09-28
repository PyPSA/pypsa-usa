"""Fixtures for the equivalence harness's fast tests.

Everything here is synthetic and in memory: 3-bus networks and 24-hour profile
datasets. No ``data/``, no ``resources/``, no network access, so
``pytest -m fast tests/equivalence`` runs in seconds.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

#: (carrier, bus, p_nom, p_nom_opt) for the three synthetic generators.
GENERATORS = (
    ("solar", 0, 100.0, 120.0),
    ("onwind", 1, 200.0, 210.0),
    ("CCGT", 2, 300.0, 300.0),
)
ZONES = ("p1", "p1", "p2")
N_SNAPSHOTS = 4


def make_network(
    bus_names: tuple[str, ...] = ("b0", "b1", "b2"),
    zones: tuple[str, ...] = ZONES,
    scale: float = 1.0,
    weighting: float = 1.0,
    n_snapshots: int = N_SNAPSHOTS,
    solved: bool = True,
):
    """A 3-bus pypsa Network with known capacities, dispatch, storage and load.

    ``scale`` multiplies every capacity and dispatch value, which is how the
    tests construct a known delta between the two sides.
    """
    import pypsa

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=n_snapshots, freq="h"))
    if weighting != 1.0:
        n.snapshot_weightings.loc[:, :] = weighting
    for i, b in enumerate(bus_names):
        n.add("Bus", b, x=float(i), y=float(-i), carrier="AC")
    n.buses["reeds_zone"] = list(zones)

    for carrier, bus_i, p_nom, p_nom_opt in GENERATORS:
        n.add(
            "Generator",
            f"{bus_names[bus_i]} {carrier}",
            bus=bus_names[bus_i],
            carrier=carrier,
            p_nom=p_nom * scale,
            p_nom_opt=p_nom_opt * scale,
        )
    n.add(
        "StorageUnit",
        f"{bus_names[1]} battery",
        bus=bus_names[1],
        carrier="battery",
        p_nom=50.0 * scale,
        p_nom_opt=60.0 * scale,
    )
    for i, b in enumerate(bus_names):
        n.add("Load", f"{b} load", bus=b, carrier="AC")

    sns = n.snapshots
    if solved:
        gen_p = pd.DataFrame(
            {
                name: np.full(len(sns), 10.0 * (i + 1) * scale)
                for i, name in enumerate(n.generators.index)
            },
            index=sns,
        )
        n.generators_t["p"] = gen_p
        # Alternating discharge/charge: only the positive part is energy out.
        store_p = pd.DataFrame(
            {n.storage_units.index[0]: np.tile([20.0, -20.0], len(sns) // 2 + 1)[: len(sns)] * scale},
            index=sns,
        )
        n.storage_units_t["p"] = store_p
    load_p = pd.DataFrame(
        {name: np.full(len(sns), 100.0 * (i + 1)) for i, name in enumerate(n.loads.index)},
        index=sns,
    )
    load_p.iloc[0] *= 2.0  # give 'peak' something to find
    n.loads_t["p_set"] = load_p
    return n


@pytest.fixture
def tiny_pair():
    """(master, develop) — two identical 3-bus networks."""
    return make_network(), make_network()


@pytest.fixture
def tiny_network():
    """One 3-bus network."""
    return make_network()


def make_objective(objective: float, constant: float):
    """A minimal stand-in carrying only what :func:`objective_row` reads.

    ``Network.objective`` is a computed property on pypsa 1.x, so a real
    network cannot be handed an arbitrary solved objective in a unit test. The
    two attributes below are the entire interface the metric uses.
    """

    class _Solved:
        def __init__(self, obj, const):
            self.objective = obj
            self.objective_constant = const

    return _Solved(objective, constant)


def make_profile(n_bus: int = 3, n_time: int = 24, seed: int = 0, bus_prefix: str = "b", cf_scale: float = 1.0):
    """A renewable-profile Dataset: ``profile(time, bus)`` and ``p_nom_max(bus)``."""
    rng = np.random.default_rng(seed)
    buses = [f"{bus_prefix}{i}" for i in range(n_bus)]
    time = pd.date_range("2030-01-01", periods=n_time, freq="h")
    profile = np.clip(rng.random((n_time, n_bus)) * cf_scale, 0.0, 1.0)
    return xr.Dataset(
        {
            "profile": (("time", "bus"), profile),
            "p_nom_max": (("bus",), np.linspace(1000.0, 3000.0, n_bus)),
        },
        coords={"time": time, "bus": buses},
    )


@pytest.fixture
def tiny_profiles():
    """(master, develop) profile Datasets over the same 3 buses and 24 hours."""
    return make_profile(seed=1), make_profile(seed=1)
