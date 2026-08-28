"""Unit tests for build_bus_population helpers."""

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_bus_population import assign_load_weight, load_county_population


@pytest.fixture
def county_population():
    return pd.Series(
        {"p06001": 1000.0, "p06037": 3000.0, "p06075": 500.0},
        name="!!Total",
    )


@pytest.fixture
def gdf_bus():
    """Buses spanning two counties; p06075 has no buses.

    p06001: sub 1 (2 buses) + sub 2 (1 bus); p06037: sub 3 (1 bus);
    bus b5 is offshore (no county).
    """
    return pd.DataFrame(
        {
            "sub_id": [1, 1, 2, 3, 4],
            "county": ["p06001", "p06001", "p06001", "p06037", None],
        },
        index=["b1", "b2", "b3", "b4", "b5"],
    )


def test_population_conserved_for_counties_with_buses(gdf_bus, county_population):
    weight = assign_load_weight(gdf_bus, county_population)
    # p06075 has no buses, so only the other two counties' population lands
    assert weight.sum() == pytest.approx(4000.0)


def test_even_split_across_subs_then_buses(gdf_bus, county_population):
    weight = assign_load_weight(gdf_bus, county_population)
    # p06001 (1000) splits evenly over subs 1 and 2 (500 each);
    # sub 1's share splits over its two buses
    assert weight["b1"] == pytest.approx(250.0)
    assert weight["b2"] == pytest.approx(250.0)
    assert weight["b3"] == pytest.approx(500.0)
    assert weight["b4"] == pytest.approx(3000.0)


def test_countyless_bus_gets_zero(gdf_bus, county_population):
    weight = assign_load_weight(gdf_bus, county_population)
    assert weight["b5"] == 0.0


def test_index_matches_input(gdf_bus, county_population):
    weight = assign_load_weight(gdf_bus, county_population)
    assert list(weight.index) == list(gdf_bus.index)


def test_county_missing_from_census_gets_zero(county_population):
    gdf = pd.DataFrame(
        {"sub_id": [1, 2], "county": ["p06001", "p99999"]},
        index=["b1", "b2"],
    )
    weight = assign_load_weight(gdf, county_population)
    assert weight["b1"] == pytest.approx(1000.0)
    assert weight["b2"] == 0.0


def test_load_county_population_parses_census_export(tmp_path):
    """Round-trip the data.census.gov P1 export format (two header rows)."""
    csv = tmp_path / "DECENNIALDHC2020.P1-Data.csv"
    csv.write_text(
        '"GEO_ID","NAME","P1_001N","P1_001NA",\n'
        '"Geography","Geographic Area Name"," !!Total","Annotation of  !!Total",\n'
        '"0500000US01001","Autauga County, Alabama","58805","null",\n'
        '"0500000US06037","Los Angeles County, California","10014009","null",\n',
    )
    pop = load_county_population(str(csv))
    assert pop["p01001"] == 58805.0
    assert pop["p06037"] == 10014009.0
    assert pop.index.name == "county"
