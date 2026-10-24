"""Unit tests for the linepack sizing in ``build_natural_gas.PipelineLinepack``.

Linepack is the gas held in the pipes, so the store must carry the fuel energy
of that gas, and a pipe crossing a state border must count towards both states.

The fixtures are two square states (UT, CO) and a handful of pipelines written
to ``tmp_path``, so nothing here needs the sector databundle.
"""

import os
import sys
from math import pi

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_natural_gas import PipelineLinepack

pytestmark = pytest.mark.fast

INTERSTATE_RADIUS = 18 * 0.0254  # m, as assumed in PipelineLinepack
INTRASTATE_RADIUS = 12 * 0.0254  # m


@pytest.fixture
def counties(tmp_path):
    """Two square 'counties', one per state, split at -110 deg longitude."""
    gdf = gpd.GeoDataFrame(
        {"STATE_NAME": ["Utah", "Colorado"], "STUSPS": ["UT", "CO"]},
        geometry=[box(-112, 39, -110, 41), box(-110, 39, -108, 41)],
        crs="EPSG:4269",
    )
    path = tmp_path / "counties.shp"
    gdf.to_file(path)
    return path


@pytest.fixture
def pipes():
    """One interstate pipe crossing the UT/CO border, one intrastate pipe in UT."""
    return gpd.GeoDataFrame(
        {
            "TYPEPIPE": ["Interstate", "Intrastate", "Interstate"],
            # deliberately wrong; length must come from the geometry
            "Shape_Leng": [1.0, 1.0, 1.0],
            "Shape__Length": [1.0, 1.0, 1.0],
        },
        geometry=[
            LineString([(-111, 40), (-109, 40)]),
            LineString([(-111.5, 39.5), (-111.5, 40.5)]),
            None,  # the EIA layer carries rows without geometry
        ],
        crs="EPSG:4326",
    )


@pytest.fixture
def linepack(tmp_path, counties, pipes):
    path = tmp_path / "pipelines.geojson"
    pipes.to_file(path, driver="GeoJSON")
    return PipelineLinepack(2020, "usa", str(counties), str(path))


def _projected_length(geom) -> float:
    return gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(5070).length.iloc[0]


def test_border_crossing_pipe_counts_in_both_states(linepack, pipes):
    data = linepack.data.set_index("STATE")

    # CO only has the half of the interstate pipe east of the border
    assert set(data.index) == {"UT", "CO"}

    interstate = pipes.geometry.iloc[0]
    intrastate = pipes.geometry.iloc[1]
    vol_interstate = _projected_length(interstate) * pi * INTERSTATE_RADIUS**2
    vol_intrastate = _projected_length(intrastate) * pi * INTRASTATE_RADIUS**2

    share_co = data.at["CO", "MAX_ENERGY_MWh"] / data.MAX_ENERGY_MWh.sum()
    expected_share_co = 0.5 * vol_interstate / (vol_interstate + vol_intrastate)
    assert share_co == pytest.approx(expected_share_co, rel=0.02)


def test_energy_is_fuel_content_of_the_pipe(linepack, pipes):
    data = linepack.data

    volume = _projected_length(pipes.geometry.iloc[0]) * pi * INTERSTATE_RADIUS**2
    volume += _projected_length(pipes.geometry.iloc[1]) * pi * INTRASTATE_RADIUS**2

    # ideal gas at 8 MPa and 60 F, times the heating value of methane (~890 kJ/mol)
    mol_per_m3 = 8000e3 / (8.314 * 288.71)
    expected_mwh_per_m3 = mol_per_m3 * 890 / 1e3 / 3600  # ~0.82 MWh_th per m3 of pipe

    assert data.MAX_ENERGY_MWh.sum() / volume == pytest.approx(expected_mwh_per_m3, rel=0.05)


def test_pressure_band_sets_min_and_nominal_levels(linepack):
    data = linepack.data

    assert (data.MIN_ENERGY_MWh / data.MAX_ENERGY_MWh).to_numpy() == pytest.approx(0.5)
    assert (data.NOMINAL_ENERGY_MWh / data.MAX_ENERGY_MWh).to_numpy() == pytest.approx(0.75)
