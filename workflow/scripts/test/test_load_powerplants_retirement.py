# BY PyPSA-USA Authors
"""``add_electricity.load_powerplants`` retirement semantics.

Announced (EIA ``planned_generator_retirement_date``) retirements are honored
for existing/proposed units when ``honor_planned_retirements`` is on (the
default); with it off, every live unit is pinned to the far-future sentinel.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from add_electricity import load_powerplants

COLUMNS = [
    "generator_name",
    "p_nom",
    "build_year",
    "operational_status",
    "generator_retirement_date",
    "planned_generator_retirement_date",
    "current_planned_generator_operating_date",
    "nerc_region",
]


def _write_plants(tmp_path, rows):
    path = tmp_path / "powerplants.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


ROWS = [
    # OTC steamer with an announced retirement before the first period
    ["otc_steam", 1000.0, 1970, "existing", None, "2026-12-01", None, "WECC"],
    # announced retirement after the first period: stays
    ["late_retiree", 500.0, 1990, "existing", None, "2040-06-01", None, "WECC"],
    # no announcement: stays
    ["evergreen", 200.0, 2001, "existing", None, None, None, "WECC"],
    # actually retired already (status retired keeps its real date)
    ["already_retired", 300.0, 1980, "retired", "2020-01-01", None, None, "WECC"],
    # proposed unit whose build year comes from the planned operating date
    ["future_build", 100.0, None, "proposed", None, None, "2032-05-01", "WECC"],
]


def test_announced_retirement_before_first_period_is_dropped(tmp_path):
    plants = load_powerplants(_write_plants(tmp_path, ROWS), [2030])
    assert "otc_steam" not in plants.index
    assert "late_retiree" in plants.index
    assert "evergreen" in plants.index
    assert "already_retired" not in plants.index


def test_flag_off_restores_indefinite_lifetimes(tmp_path):
    plants = load_powerplants(
        _write_plants(tmp_path, ROWS),
        [2030],
        honor_planned_retirements=False,
    )
    assert "otc_steam" in plants.index
    assert plants.loc["otc_steam", "generator_retirement_date"] == pd.Timestamp("2100-01-01")


def test_announced_date_lands_on_the_retirement_column(tmp_path):
    plants = load_powerplants(_write_plants(tmp_path, ROWS), [2030])
    assert plants.loc["late_retiree", "generator_retirement_date"] == pd.Timestamp("2040-06-01")
    assert plants.loc["evergreen", "generator_retirement_date"] == pd.Timestamp("2100-01-01")


def test_boundary_year_semantics(tmp_path):
    """Retirement in the first-period year itself excludes the unit (`> horizon`)."""
    rows = [["boundary", 400.0, 1995, "existing", None, "2030-12-31", None, "WECC"]]
    plants = load_powerplants(_write_plants(tmp_path, rows), [2030])
    assert plants.empty

    plants = load_powerplants(_write_plants(tmp_path, rows), [2026])
    assert "boundary" in plants.index


def test_proposed_units_unaffected_by_flag(tmp_path):
    for flag in (True, False):
        plants = load_powerplants(
            _write_plants(tmp_path, ROWS),
            [2035],
            honor_planned_retirements=flag,
        )
        assert "future_build" in plants.index
        assert plants.loc["future_build", "build_year"] == pytest.approx(2032)
