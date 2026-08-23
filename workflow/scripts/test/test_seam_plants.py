"""Unit tests for the seam-plant distance bound in filter_plants_by_region.

``filter_plants_by_region`` re-adds "must add" plants — those outside every ReEDS
shape of the run's interconnect whose ReEDS membership disagrees with their EIA
``interconnection`` column — so that imprecise ReEDS shapes never silently delete a
legitimate border plant. Since DL-11 the regions layers only tile the model footprint
in scoped runs, so that unconditional add-back let plants thousands of km away survive
the filter and then attach to the nearest in-footprint bus (match_plant_to_bus applies
no distance bound). These tests pin:
  * the default (gate-off) path keeps every must-add plant — legacy behavior, so
    unfiltered interconnect/usa runs are unchanged,
  * the gate-on path keeps in-footprint and near-seam plants but drops distant ones,
  * the drop is reported loudly with plant name, carrier, state, MW and distance.
"""

import logging
import os
import sys

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from add_electricity import SEAM_PLANT_MAX_KM, filter_plants_by_region

pytestmark = pytest.mark.fast

# Geometry (EPSG:4326, distances measured in EPSG:5070 as the code does):
# the footprint is the onshore box [-120,-119]x[36,37] plus an offshore box to its
# west. IN_FOOTPRINT sits inside it (0 km), SEAM sits ~49.7 km east of it (inside the
# 100 km bound), FAR sits ~496.7 km east (outside it).
PLANT_LONLAT = {
    "in_footprint": (-119.5, 36.5),
    "seam": (-118.44, 36.5),
    "far": (-113.4, 36.5),
}


@pytest.fixture
def regions_onshore():
    return gpd.GeoDataFrame(
        {"name": ["footprint"], "country": ["p1"]},
        geometry=[box(-120.0, 36.0, -119.0, 37.0)],
        crs="EPSG:4326",
    )


@pytest.fixture
def regions_offshore():
    return gpd.GeoDataFrame(
        {"name": ["footprint_offshore"], "country": ["p1"]},
        geometry=[box(-121.0, 36.0, -120.0, 37.0)],
        crs="EPSG:4326",
    )


@pytest.fixture
def reeds_shapes():
    """Interconnect ReEDS shapes, deliberately disjoint from every test plant.

    This is what pushes all three plants into the `plants_no_region` branch.
    """
    return gpd.GeoDataFrame(
        {"name": ["p9"], "reeds_ba": ["BA9"]},
        geometry=[box(-100.0, 40.0, -99.0, 41.0)],
        crs="EPSG:4326",
    )


@pytest.fixture
def all_reeds_shapes():
    """National ReEDS shapes covering every test plant (stored in EPSG:3857 upstream)."""
    return gpd.GeoDataFrame(
        {"rb": ["p50"], "BA_Code": ["BA50"]},
        geometry=[box(-125.0, 30.0, -95.0, 45.0)],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)


@pytest.fixture
def reeds_memberships():
    """Membership puts the national shape in a different interconnect than the plants.

    `interconnect` ("eastern") != the plants' EIA `interconnection` ("western"), which
    is exactly the condition that routes a plant into `plants_must_add`.
    """
    return pd.DataFrame({"ba": ["p50"], "interconnect": ["eastern"]})


@pytest.fixture
def plants():
    df = pd.DataFrame(
        {
            "generator_name": list(PLANT_LONLAT),
            "longitude": [lon for lon, _ in PLANT_LONLAT.values()],
            "latitude": [lat for _, lat in PLANT_LONLAT.values()],
            "p_nom": [100.0, 50.0, 195.0],
            "carrier": ["solar", "onwind", "solar"],
            "state": ["CA", "NV", "NM"],
            "interconnection": ["western"] * 3,
        },
    )
    return df.set_index("generator_name")


def _filter(plants, regions_onshore, regions_offshore, reeds_shapes, all_reeds_shapes, memberships, **kwargs):
    return filter_plants_by_region(
        plants,
        regions_onshore,
        regions_offshore,
        reeds_shapes,
        all_reeds_shapes,
        memberships,
        **kwargs,
    )


def test_gate_off_keeps_every_seam_plant(
    plants,
    regions_onshore,
    regions_offshore,
    reeds_shapes,
    all_reeds_shapes,
    reeds_memberships,
):
    """Legacy behavior: with no footprint scoping, all must-add plants are retained."""
    result = _filter(
        plants,
        regions_onshore,
        regions_offshore,
        reeds_shapes,
        all_reeds_shapes,
        reeds_memberships,
    )
    assert set(result.index) == set(PLANT_LONLAT)


def test_gate_off_is_the_default(
    plants,
    regions_onshore,
    regions_offshore,
    reeds_shapes,
    all_reeds_shapes,
    reeds_memberships,
):
    """Passing footprint_scoped=False explicitly matches omitting it entirely."""
    default = _filter(
        plants,
        regions_onshore,
        regions_offshore,
        reeds_shapes,
        all_reeds_shapes,
        reeds_memberships,
    )
    explicit = _filter(
        plants,
        regions_onshore,
        regions_offshore,
        reeds_shapes,
        all_reeds_shapes,
        reeds_memberships,
        footprint_scoped=False,
    )
    assert list(default.index) == list(explicit.index)


def test_gate_on_drops_only_distant_seam_plants(
    plants,
    regions_onshore,
    regions_offshore,
    reeds_shapes,
    all_reeds_shapes,
    reeds_memberships,
):
    """Footprint-scoped: in-footprint and near-seam plants stay, the far one goes."""
    result = _filter(
        plants,
        regions_onshore,
        regions_offshore,
        reeds_shapes,
        all_reeds_shapes,
        reeds_memberships,
        footprint_scoped=True,
    )
    assert set(result.index) == {"in_footprint", "seam"}
    assert "far" not in result.index


def test_gate_on_logs_dropped_plant_details(
    caplog,
    plants,
    regions_onshore,
    regions_offshore,
    reeds_shapes,
    all_reeds_shapes,
    reeds_memberships,
):
    """The drop is reported loudly: name, carrier, state, MW, distance and a summary."""
    with caplog.at_level(logging.WARNING, logger="add_electricity"):
        _filter(
            plants,
            regions_onshore,
            regions_offshore,
            reeds_shapes,
            all_reeds_shapes,
            reeds_memberships,
            footprint_scoped=True,
        )
    warnings = "\n".join(r.message for r in caplog.records if r.levelno >= logging.WARNING)
    assert "far" in warnings
    assert "solar" in warnings
    assert "NM" in warnings
    assert "195.0 MW" in warnings
    assert "497 km" in warnings  # ~496.7 km, rendered to whole km
    assert "dropped 1 of 3" in warnings
    # the near-seam plant is kept, so it must not appear in the drop log
    assert "'seam'" not in warnings


def test_seam_bound_constant_is_100km():
    """The bound is a module constant, not config plumbing (see the design note)."""
    assert SEAM_PLANT_MAX_KM == 100.0
