"""Tests for log_network_schema in _helpers."""

import logging
import os
import sys

import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from _helpers import log_network_schema


@pytest.fixture
def small_network():
    """Tiny network with custom columns on Bus and Line."""
    n = pypsa.Network()
    n.add("Bus", "b0", v_nom=230.0)
    n.add("Bus", "b1", v_nom=230.0)
    n.add("Line", "l0", bus0="b0", bus1="b1", x=0.1, r=0.01, s_nom=100)
    n.buses["custom_col"] = [1.0, 2.0]
    return n


def test_entry_returns_snapshot_with_cols_and_rows(small_network):
    snapshot = log_network_schema(small_network, stage="entry")
    assert "Bus" in snapshot
    assert "Line" in snapshot
    assert "custom_col" in snapshot["Bus"]["cols"]
    assert snapshot["Bus"]["rows"] == 2
    assert snapshot["Line"]["rows"] == 1
    assert snapshot["Bus"]["cols"] == sorted(snapshot["Bus"]["cols"])


def test_entry_logs_one_line_per_nonempty_component(small_network, caplog):
    with caplog.at_level(logging.INFO, logger="_helpers"):
        log_network_schema(small_network, stage="entry")
    messages = [r.message for r in caplog.records if "[schema entry]" in r.message]
    assert any("Bus: 2 rows" in m for m in messages)
    assert any("Line: 1 rows" in m for m in messages)
    assert not any("Generator" in m for m in messages)


def test_exit_with_baseline_logs_column_diff(small_network, caplog):
    baseline = log_network_schema(small_network, stage="entry")
    small_network.buses["new_col"] = [9.0, 9.0]
    small_network.buses = small_network.buses.drop(columns=["custom_col"])
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="_helpers"):
        log_network_schema(small_network, stage="exit", baseline=baseline)
    diff_lines = [r.message for r in caplog.records if "[schema exit]" in r.message]
    bus_diff = next(m for m in diff_lines if "Bus" in m and "cols" in m)
    assert "+cols=['new_col']" in bus_diff
    assert "-cols=['custom_col']" in bus_diff


def test_exit_logs_row_count_change(small_network, caplog):
    baseline = log_network_schema(small_network, stage="entry")
    small_network.remove("Bus", "b1")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="_helpers"):
        log_network_schema(small_network, stage="exit", baseline=baseline)
    diff_lines = [r.message for r in caplog.records if "[schema exit]" in r.message]
    assert any("Bus: 2 -> 1 rows" in m for m in diff_lines)


def test_exit_quiet_for_unchanged_components(small_network, caplog):
    baseline = log_network_schema(small_network, stage="entry")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="_helpers"):
        log_network_schema(small_network, stage="exit", baseline=baseline)
    diff_lines = [r.message for r in caplog.records if "[schema exit]" in r.message]
    assert diff_lines == []
