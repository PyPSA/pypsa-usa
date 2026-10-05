"""HF-26: does attaching the existing renewable fleet conserve its MW.

Tier A (``fast``): a 3-bus / 2-cluster toy network and four plants. No
``data/``, no ``resources/``, no snakemake.

``workflow/scripts/add_electricity.py::attach_renewable_capacities_to_atlite``
is byte-identical on ``master`` and ``develop``. Its last step is::

    mapped_values = generators_tech.sub_assignment.map(caps_per_bus).dropna()
    n.generators.loc[mapped_values.index, "p_nom"] = mapped_values

``caps_per_bus`` is the plants' MW summed per bus, and ``generators_tech`` is
the network's existing generators of that carrier — which at this stage come
from the renewable PROFILE file. So a plant can only keep its MW if its bus
already carries a profile-derived generator of the same carrier. ``.dropna()``
throws the rest away, and the function's own ``logger.info`` announces it:
"N GW of <tech> plants that are not in the network. See git issue #16".

What differs between the branches is the BUS SPACE the function runs on:

- master runs it on the NODAL network, one bus per substation, and its western
  profile files cover 544 of 1,972 onwind and 808 of 1,972 solar substations, so
  the plants at the other ~1,400 are dropped;
- develop runs it after simplify-early at s{simpl} CLUSTER granularity, where a
  cluster carries a profile generator if ANY of its substations does, so the
  plants master drops land on their cluster instead.

Both cases are expressible here, because "the bus space" is nothing more than
which buses exist and which of them the profile put a generator on. The two
tests below are the same four plants and the same function, run once per bus
space: 750 MW in, 250 MW attached nodally, 750 MW attached at cluster
granularity.

Deltas ledger DL-20; hot-fix registry HF-26.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.fast

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "workflow" / "scripts"


@pytest.fixture(scope="module")
def attach():
    """``attach_renewable_capacities_to_atlite`` from the workflow scripts.

    Imported the way ``ab.py`` imports ``plot_network_maps``: the workflow's
    scripts are flat modules that import each other by bare name, so
    ``workflow/scripts`` goes on ``sys.path`` and the module is imported by its
    own name. Importing the real thing is the whole point — a copy of the ten
    lines under test would keep passing after the original changed.
    """
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    add_electricity = pytest.importorskip("add_electricity")
    return add_electricity.attach_renewable_capacities_to_atlite


#: (plant id, bus in the NODAL space, cluster it belongs to, MW).
#: Two plants at the profiled substation, two at the unprofiled one.
PLANTS = (
    ("plant_a", "b0", "c0", 100.0),
    ("plant_b", "b0", "c0", 150.0),
    ("plant_c", "b1", "c0", 200.0),
    ("plant_d", "b1", "c0", 300.0),
)
PLANTS_TOTAL_MW = 750.0
#: What the profile file covers in each bus space. Nodally it misses ``b1``;
#: at cluster granularity ``c0`` is covered because ``b0`` is inside it.
PROFILED_NODAL = ("b0", "b2")
PROFILED_CLUSTER = ("c0", "c1")


def _network(buses: tuple[str, ...], profiled: tuple[str, ...], carrier: str = "onwind"):
    """A toy network with one profile-derived ``carrier`` generator per profiled bus."""
    import pypsa

    n = pypsa.Network()
    for b in buses:
        n.add("Bus", b, carrier="AC")
    for b in profiled:
        # p_nom 0: the profile-derived generator is extendable and carries no
        # existing capacity until the fleet is attached to it.
        n.add("Generator", f"{b} {carrier}", bus=b, carrier=carrier, p_nom=0.0, p_nom_extendable=True)
    # A generator of another carrier, at the bus the renewable profile misses:
    # it must not receive the onwind fleet.
    n.add("Generator", f"{buses[-1]} CCGT", bus=buses[-1], carrier="CCGT", p_nom=500.0)
    return n


def _plants(bus_field: str, carrier: str = "onwind") -> pd.DataFrame:
    """The four plants, addressed in one bus space or the other."""
    col = {"nodal": 1, "cluster": 2}[bus_field]
    return pd.DataFrame(
        {
            "bus_assignment": [p[col] for p in PLANTS],
            "carrier": [carrier] * len(PLANTS),
            "p_nom": [p[3] for p in PLANTS],
            "build_year": [2005, 2010, 2015, 2020],
            "prime_mover_code": ["WT"] * len(PLANTS),
        },
        index=[p[0] for p in PLANTS],
    )


def _attached_mw(n, carrier: str = "onwind") -> float:
    g = n.generators
    return float(g.loc[g["carrier"] == carrier, "p_nom"].sum())


def test_existing_renewable_capacity_is_conserved(attach):
    """The same fleet, the same function, two bus spaces: 250 MW vs 750 MW.

    The nodal call reproduces master's loss and the cluster-granularity call
    reproduces develop's conservation, which is HF-26 in nine lines.
    """
    nodal = _network(("b0", "b1", "b2"), PROFILED_NODAL)
    attach(nodal, _plants("nodal"), ["onwind"])
    nodal_mw = _attached_mw(nodal)

    clustered = _network(("c0", "c1"), PROFILED_CLUSTER)
    attach(clustered, _plants("cluster"), ["onwind"])
    cluster_mw = _attached_mw(clustered)

    # Master: only the plants at the profiled substation survive.
    assert nodal_mw == pytest.approx(250.0), nodal_mw
    assert nodal_mw < PLANTS_TOTAL_MW
    assert PLANTS_TOTAL_MW - nodal_mw == pytest.approx(500.0), "b1's two plants should be the loss"

    # Develop: every plant's cluster carries a profile generator, so nothing is lost.
    assert cluster_mw == pytest.approx(PLANTS_TOTAL_MW), cluster_mw

    # And the difference is exactly what the harness waives as HF-26.
    assert cluster_mw - nodal_mw == pytest.approx(500.0)


def test_the_fleet_lands_only_on_its_own_carrier(attach):
    """The CCGT at the unprofiled bus must not absorb the onwind fleet.

    Without this, "conserved" could be satisfied by dumping the MW on whatever
    generator happens to share the bus.
    """
    clustered = _network(("c0", "c1"), PROFILED_CLUSTER)
    attach(clustered, _plants("cluster"), ["onwind"])
    g = clustered.generators
    assert float(g.loc[g["carrier"] == "CCGT", "p_nom"].sum()) == pytest.approx(500.0)
    assert _attached_mw(clustered, "onwind") == pytest.approx(PLANTS_TOTAL_MW)


def test_p_nom_min_follows_p_nom_so_the_fleet_cannot_be_decommissioned(attach):
    """Attached existing capacity is a floor, not a starting point.

    ``p_nom_min`` is set alongside ``p_nom``; if the loss in the nodal case ever
    became a silent ``p_nom_min`` of 0 with a non-zero ``p_nom``, the solver
    could retire the fleet and the capacity metric would still look right.
    """
    clustered = _network(("c0", "c1"), PROFILED_CLUSTER)
    attach(clustered, _plants("cluster"), ["onwind"])
    g = clustered.generators
    onwind = g[g["carrier"] == "onwind"]
    assert float(onwind["p_nom_min"].sum()) == pytest.approx(PLANTS_TOTAL_MW)
    assert (onwind["p_nom_min"] == onwind["p_nom"]).all()


def test_pumped_storage_plants_are_excluded_before_the_map(attach):
    """``prime_mover_code == 'PS'`` is dropped by the function, on both branches.

    Guards the one filter that runs BEFORE the lossy ``.map(...).dropna()``, so a
    change there cannot be mistaken for HF-26.
    """
    clustered = _network(("c0", "c1"), PROFILED_CLUSTER)
    plants = _plants("cluster")
    plants.loc["plant_d", "prime_mover_code"] = "PS"
    attach(clustered, plants, ["onwind"])
    assert _attached_mw(clustered) == pytest.approx(PLANTS_TOTAL_MW - 300.0)
