"""Fast unit tests for the parts of ``tests.equivalence.compare`` that judge.

Synthetic netCDF profiles in ``tmp_path`` only: no ``resources/``, no build, no
run directory. The end-to-end pass lives in ``test_equivalence.py`` and is not
``fast``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests.equivalence import compare
from tests.equivalence.paths import ArtifactPair

from .conftest import make_profile

pytestmark = pytest.mark.fast


def _pair() -> ArtifactPair:
    return ArtifactPair(
        stage="profile_onwind",
        develop="develop.nc",
        master="master.nc",
        kind="profile",
    )


def _write(tmp_path, master: xr.Dataset, develop: xr.Dataset):
    mp, dp = tmp_path / "master.nc", tmp_path / "develop.nc"
    master.to_netcdf(mp)
    develop.to_netcdf(dp)
    return dp, mp


def _nodal_master(n_bus: int = 6, seed: int = 80, n_time: int = 12) -> xr.Dataset:
    """A master-style NODAL profile file with float-formatted bus labels."""
    ds = make_profile(n_bus=n_bus, n_time=n_time, seed=seed, bus_prefix="")
    return ds.assign_coords(bus=[f"{35827 + i}.0" for i in range(n_bus)])


def _busmap(n_bus: int = 6) -> pd.Series:
    """Substation id -> cluster; the first half is c0, the second c1."""
    return pd.Series(
        {str(35827 + i): ("c0" if i < n_bus // 2 else "c1") for i in range(n_bus)},
        dtype=object,
    )


def _cluster_findings(findings: list[dict]) -> list[dict]:
    return [f for f in findings if f["component"] == "cluster_set"]


def test_a_develop_only_cluster_is_a_finding_with_its_mw(tmp_path):
    """The one-sided cluster is named, with the capacity it carries.

    Silently pooled, ``p87 0`` (2,158 MW of solar on the western leg) read as a
    capacity-factor difference spread over every quantile row. It is a row-set
    difference and is reported as one.
    """
    from tests.equivalence import metrics

    master = _nodal_master()
    develop = metrics.aggregate_profile_to_clusters(master, _busmap())
    extra = make_profile(n_bus=1, n_time=12, seed=81).assign_coords(bus=["p87 0"])
    extra["p_nom_max"][:] = np.array([2158.0])
    develop = xr.concat([develop, extra], dim="bus")

    dp, mp = _write(tmp_path, master, develop)
    findings: list[dict] = []
    notes: list[dict] = []
    compare.compare_profiles(_pair(), dp, mp, findings, prong=2, busmap=_busmap(), notes=notes)

    cs = _cluster_findings(findings)
    assert len(cs) == 1, findings
    f = cs[0]
    assert f["stage"] == "profile_onwind"
    assert f["column"] == "<index>"
    assert f["kind"] == "row_set"
    assert f["detail"]["only_develop"] == {"p87 0": pytest.approx(2158.0)}
    assert f["detail"]["only_master"] == {}
    assert f["detail"]["only_develop_mw"] == pytest.approx(2158.0)
    assert f["detail"]["n_master"] == 2
    assert f["detail"]["n_develop"] == 3
    assert f["detail"]["n_common"] == 2

    assert notes == [
        {
            "kind": "profile_rollup",
            "stage": "profile_onwind",
            "rolled_up": True,
            "n_master": 2,
            "n_develop": 3,
            "n_common": 2,
            "only_master": {},
            "only_develop": {"p87 0": pytest.approx(2158.0)},
            "only_master_mw": 0.0,
            "only_develop_mw": pytest.approx(2158.0),
            "equal": False,
        },
    ]


def test_matching_cluster_sets_emit_no_cluster_finding(tmp_path):
    from tests.equivalence import metrics

    master = _nodal_master(seed=82)
    develop = metrics.aggregate_profile_to_clusters(master, _busmap())
    dp, mp = _write(tmp_path, master, develop)

    findings: list[dict] = []
    notes: list[dict] = []
    compare.compare_profiles(_pair(), dp, mp, findings, prong=2, busmap=_busmap(), notes=notes)

    assert _cluster_findings(findings) == []
    assert notes[0]["equal"] is True
    assert notes[0]["rolled_up"] is True
    # Same physics, same caps: the system aggregates agree too.
    assert findings == []


def test_the_cluster_set_finding_is_waived_by_the_shipped_hf24_waiver():
    """The shipped DL-18 / HF-24 waiver covers exactly this finding, on western p2."""
    finding = {
        "stage": "profile_solar",
        "component": "cluster_set",
        "column": "<index>",
        "kind": "row_set",
        "prong": 2,
        "interconnect": "western",
    }
    shipped = compare.load_waivers()
    assert compare.is_waived(finding, shipped) is True
    # ... and nowhere else: the usa leg has not been measured.
    assert compare.is_waived(dict(finding, interconnect="usa"), shipped) is False
    assert compare.is_waived(dict(finding, prong=1), shipped) is False


def test_prong_1_does_not_roll_up_or_compare_cluster_sets(tmp_path):
    master = _nodal_master(seed=83)
    develop = master.copy(deep=True)
    dp, mp = _write(tmp_path, master, develop)

    findings: list[dict] = []
    notes: list[dict] = []
    compare.compare_profiles(_pair(), dp, mp, findings, prong=1, busmap=_busmap(), notes=notes)

    assert _cluster_findings(findings) == []
    assert notes == []


def test_prong_2_without_a_busmap_records_that_nothing_was_rolled_up(tmp_path):
    master = _nodal_master(seed=84)
    develop = master.copy(deep=True)
    dp, mp = _write(tmp_path, master, develop)

    findings: list[dict] = []
    notes: list[dict] = []
    compare.compare_profiles(_pair(), dp, mp, findings, prong=2, busmap=None, notes=notes)

    assert notes == [{"kind": "profile_rollup", "stage": "profile_onwind", "rolled_up": False}]
