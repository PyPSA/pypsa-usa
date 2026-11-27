"""Unit tests for the AEO electricity demand scaling. Nothing here touches the network."""

import os
import sys

import duckdb
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import build_demand

pytestmark = pytest.mark.fast

TABLE = "core_eiaaeo__yearly_projected_generation_in_electric_sector_by_technology.parquet"


def test_growth_uses_a_single_aeo_edition(tmp_path, monkeypatch):
    """Overlapping AEO editions must not be summed."""
    rows = [
        # report_year, projection_year, technology, generation
        (2023, 2022, "coal", 60.0),
        (2023, 2022, "nuclear", 40.0),
        (2023, 2030, "coal", 50.0),
        (2023, 2030, "nuclear", 60.0),
        (2025, 2030, "coal", 70.0),
        (2025, 2030, "nuclear", 80.0),
    ]
    df = pd.DataFrame(
        rows,
        columns=["report_year", "projection_year", "technology_description_eiaaeo", "gross_generation_mwh"],
    )
    df["electricity_market_module_region_eiaaeo"] = "united_states"
    df["model_case_eiaaeo"] = "reference"
    df.to_parquet(tmp_path / TABLE)

    query = duckdb.query
    monkeypatch.setattr(
        build_demand.duckdb,
        "query",
        lambda sql: None if sql.startswith("INSTALL") else query(sql),
    )

    scaler = build_demand.AeoElectricityScaler(str(tmp_path))

    assert scaler.get_growth(2022, 2030, "power") == pytest.approx(1.1)
