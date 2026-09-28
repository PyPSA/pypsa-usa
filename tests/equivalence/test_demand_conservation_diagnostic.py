"""Tier A tests for the demand-conservation diagnostic (HF-28).

The checker answers one question -- does the load attached to the buses add up
to the demand table it was split from? -- so its own arithmetic has to be right
before its verdict means anything. Everything here is a 3-bus network and a
6-row demand table in ``tmp_path``; no ``resources/``, no real artifacts.

The numbers it reproduces on the real thing are recorded in the module
docstring of ``diagnostics/demand_conservation.py`` (usa s300 develop,
2026-09-15: +1.85 %, Maryland ``sum(LAF_state)`` 2.008).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa
import pytest

from .diagnostics.demand_conservation import (
    allocated_by_key,
    bus_keys,
    conservation_frame,
    main,
    table_by_key,
)

pytestmark = pytest.mark.fast

SNAPSHOTS = pd.date_range("2030-01-01", periods=4, freq="h", name="snapshot")

#: bus -> (reeds_state, LAF_state, load MW). Maryland's factors sum to 2.0 and
#: its buses hold twice Maryland's demand: the defect, in miniature.
BUSES = {
    "md_a": ("MD", 1.0, 200.0),
    "md_b": ("MD", 1.0, 200.0),
    "va": ("VA", 1.0, 150.0),
}


def make_network(path):
    """A 3-bus network with loads and the columns the checker reads."""
    n = pypsa.Network()
    n.set_snapshots(SNAPSHOTS)
    for bus, (state, laf, load) in BUSES.items():
        n.add("Bus", bus)
        n.add("Load", f"{bus} load", bus=bus, p_set=pd.Series(load, index=SNAPSHOTS))
    n.buses["reeds_state"] = [BUSES[b][0] for b in BUSES]
    n.buses["LAF_state"] = [BUSES[b][1] for b in BUSES]
    n.export_to_netcdf(str(path))
    return n


def make_table(path, values, snapshots=SNAPSHOTS):
    """A resolved zonal demand table on the reader's 4-level contract."""
    frame = pd.DataFrame(
        {key: np.broadcast_to(np.asarray(value, dtype=float), (len(snapshots),)) for key, value in values.items()},
        index=snapshots,
    )
    frame.index.name = "snapshot"
    frame["sector"] = "all"
    frame["subsector"] = "all"
    frame["fuel"] = "electricity"
    # two components, so the sum over the non-snapshot levels is exercised
    half = frame.copy()
    for key in values:
        half[key] = frame[key] / 2.0
    other = half.copy()
    other["subsector"] = "other"
    stacked = pd.concat([half, other]).set_index(["sector", "subsector", "fuel"], append=True)
    stacked.to_parquet(path)
    return stacked


def test_bus_keys_prefers_state_and_falls_back_to_reeds_state(tmp_path):
    n = make_network(tmp_path / "n.nc")
    keys, column = bus_keys(n)
    assert column == "reeds_state"
    assert keys.tolist() == ["Maryland", "Maryland", "Virginia"]

    n.buses["state"] = ["Maryland", "Maryland", "Virginia"]
    assert bus_keys(n)[1] == "state"
    assert bus_keys(n, "reeds_state")[1] == "reeds_state"


def test_allocated_and_table_are_summed_the_same_way(tmp_path):
    n = make_network(tmp_path / "n.nc")
    keys, _ = bus_keys(n)

    allocated = allocated_by_key(n, keys)
    assert allocated["Maryland"] == pytest.approx(400.0)
    assert allocated["Virginia"] == pytest.approx(150.0)

    make_table(tmp_path / "d.parquet", {"Maryland": 200.0, "Virginia": 150.0})
    table = table_by_key(tmp_path / "d.parquet", SNAPSHOTS)
    assert table["Maryland"] == pytest.approx(200.0)


def test_table_is_restricted_to_the_network_snapshots(tmp_path):
    """A year-long table against a sampled network must not measure the sampling."""
    long_index = pd.date_range("2030-01-01", periods=48, freq="h", name="snapshot")
    # 200 MW over the four hours the network models, 1,000 MW for the rest of
    # the table: a mean over the whole table would report 933 MW.
    ramp = {"Maryland": [200.0] * 4 + [1000.0] * 44}
    make_table(tmp_path / "d.parquet", ramp, snapshots=long_index)

    table = table_by_key(tmp_path / "d.parquet", SNAPSHOTS)
    assert table["Maryland"] == pytest.approx(200.0)

    with pytest.raises(SystemExit):
        table_by_key(tmp_path / "d.parquet", pd.date_range("2045-01-01", periods=2, freq="h"))


def test_conservation_frame_reports_the_over_allocation(tmp_path):
    n = make_network(tmp_path / "n.nc")
    keys, _ = bus_keys(n)
    make_table(tmp_path / "d.parquet", {"Maryland": 200.0, "Virginia": 150.0, "Texas": 90.0})

    df = conservation_frame(
        allocated_by_key(n, keys),
        table_by_key(tmp_path / "d.parquet", SNAPSHOTS),
        pd.to_numeric(n.buses.LAF_state).groupby(keys).sum(),
    )

    assert df.loc["Maryland", "delta_mw"] == pytest.approx(200.0)
    assert df.loc["Maryland", "delta_pct"] == pytest.approx(100.0)
    assert df.loc["Maryland", "laf_sum"] == pytest.approx(2.0)
    assert df.loc["Virginia", "delta_mw"] == pytest.approx(0.0)
    # a demand key no bus carries is reported, not silently skipped
    assert not bool(df.loc["Texas", "has_bus"])
    assert df.loc["Texas", "table_mw"] == pytest.approx(90.0)


def test_main_writes_a_png_and_its_csv_twin(tmp_path, capsys):
    """Every figure has a CSV twin; the harness convention holds here too."""
    make_network(tmp_path / "n.nc")
    make_table(tmp_path / "d.parquet", {"Maryland": 200.0, "Virginia": 150.0})
    outdir = tmp_path / "out"

    assert main(
        [
            "--network",
            str(tmp_path / "n.nc"),
            "--demand-table",
            str(tmp_path / "d.parquet"),
            "--outdir",
            str(outdir),
        ],
    ) == 0

    assert (outdir / "demand_conservation.png").exists()
    written = pd.read_csv(outdir / "demand_conservation.csv", index_col="demand_key")
    assert written.loc["Maryland", "allocated_mw"] == pytest.approx(400.0)
    assert "+57.1429 %" in capsys.readouterr().out  # (550 - 350) / 350
