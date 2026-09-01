"""Tests for the SERVM demand write (disaggregation) strategy."""

import numpy as np
import pandas as pd
import pypsa
import pytest
from _helpers import get_multiindex_snapshots
from build_demand import WriteServm

REGIONS = ("PGE", "SCE")


def make_network(buses=("bus_a", "bus_b", "bus_c")):
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        {"start": "2028-01-01 00:00", "end": "2028-01-01 03:00", "inclusive": "both"},
        [2028],
    )
    n.set_investment_periods(periods=[2028])
    for bus in buses:
        n.add("Bus", bus)
    return n


def write_weights(path, rows):
    """rows: iterable of (bus, servm_region, laf)."""
    pd.DataFrame(rows, columns=["bus", "servm_region", "laf"]).to_csv(path, index=False)
    return str(path)


def make_demand(n, values):
    """Zonal demand frame on the reader's 4-level contract.

    ``values`` maps region -> list of hourly values for the Net Load component.
    A second (BTMPV) component is always added so the subsector filter is
    actually exercised.
    """
    snapshots = n.snapshots.get_level_values(1)
    frames = []
    for subsector, scale in (("Net Load", 1.0), ("BTMPV", 100.0)):
        frame = pd.DataFrame(
            {region: np.asarray(series, dtype=float) * scale for region, series in values.items()},
            index=snapshots,
        )
        frame.index.name = "snapshot"
        frame["sector"] = "all"
        frame["subsector"] = subsector
        frame["fuel"] = "electricity"
        frames.append(frame.set_index(["sector", "subsector", "fuel"], append=True))
    return pd.concat(frames).sort_index()


def test_writeservm_matrix_product_matches_manual(tmp_path):
    """Bus load is the region-share weighted sum of regional demand."""
    n = make_network()
    weights_file = write_weights(
        tmp_path / "weights.csv",
        [
            ("bus_a", "PGE", 0.75),
            ("bus_b", "PGE", 0.25),
            ("bus_c", "SCE", 1.0),
        ],
    )

    demand = make_demand(n, {"PGE": [100, 200, 300, 400], "SCE": [10, 20, 30, 40]})

    writer = WriteServm(n, weights_file)
    result = writer.dissagregate_demand(demand, "servm", subsector="Net Load")

    assert list(result.columns) == ["bus_a", "bus_b", "bus_c"]
    np.testing.assert_allclose(result["bus_a"], [75, 150, 225, 300])
    np.testing.assert_allclose(result["bus_b"], [25, 50, 75, 100])
    np.testing.assert_allclose(result["bus_c"], [10, 20, 30, 40])

    # the BTMPV component (100x) must not have leaked into the modeled load
    assert result.to_numpy().sum() == pytest.approx(
        demand.xs("Net Load", level="subsector").to_numpy().sum(),
    )


def test_straddling_bus_receives_sum_of_both_regions(tmp_path):
    """A cluster spanning two SERVM regions collects a share of each."""
    n = make_network(buses=("bus_a", "bus_b"))
    weights_file = write_weights(
        tmp_path / "weights.csv",
        [
            ("bus_a", "PGE", 0.6),
            ("bus_b", "PGE", 0.4),
            ("bus_a", "SCE", 0.1),  # bus_a straddles PGE and SCE
            ("bus_b", "SCE", 0.9),
        ],
    )

    demand = make_demand(n, {"PGE": [100, 100, 100, 100], "SCE": [50, 50, 50, 50]})

    writer = WriteServm(n, weights_file)
    result = writer.dissagregate_demand(demand, "servm", subsector="Net Load")

    np.testing.assert_allclose(result["bus_a"], [65.0] * 4)  # 0.6*100 + 0.1*50
    np.testing.assert_allclose(result["bus_b"], [85.0] * 4)  # 0.4*100 + 0.9*50
    # nothing is created or lost
    np.testing.assert_allclose(result.sum(axis=1), [150.0] * 4)


def test_weights_bus_not_in_network_raises(tmp_path):
    """Weights built against a different network must fail loudly."""
    n = make_network(buses=("bus_a",))
    weights_file = write_weights(
        tmp_path / "weights.csv",
        [("bus_a", "PGE", 0.5), ("bus_missing", "PGE", 0.5)],
    )

    with pytest.raises(ValueError, match="not in the network"):
        WriteServm(n, weights_file)


def test_region_without_weights_is_dropped_with_warning(tmp_path, caplog):
    """Demand for a region absent from the weights table cannot be allocated."""
    n = make_network(buses=("bus_a",))
    weights_file = write_weights(tmp_path / "weights.csv", [("bus_a", "PGE", 1.0)])

    demand = make_demand(n, {"PGE": [100] * 4, "SCE": [50] * 4})

    writer = WriteServm(n, weights_file)
    with caplog.at_level("WARNING", logger="build_demand"):
        result = writer.dissagregate_demand(demand, "servm", subsector="Net Load")

    np.testing.assert_allclose(result["bus_a"], [100.0] * 4)
    assert "No bus weights found for SERVM region(s)" in caplog.text


def test_wrong_zone_is_rejected(tmp_path):
    n = make_network(buses=("bus_a",))
    weights_file = write_weights(tmp_path / "weights.csv", [("bus_a", "PGE", 1.0)])
    demand = make_demand(n, {"PGE": [100] * 4})

    writer = WriteServm(n, weights_file)
    with pytest.raises(AssertionError):
        writer.dissagregate_demand(demand, "state", subsector="Net Load")
