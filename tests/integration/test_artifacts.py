"""Tier B — load each produced artifact and assert shape.

Asserts per-stage: bus counts match wildcards, no NaN in load-bearing
columns, netCDF roundtrips, filename wildcard propagation. No numerical
regression here — that's Tier C (deferred).
"""

from __future__ import annotations

import dill
import pandas as pd
import pypsa
import pytest

pytestmark = pytest.mark.integration


# ----- stage 1: aggregate_to_substations -----------------------------------


class TestPostAggregateToSubstations:
    """Assertions on the elec_b.nc artifact (post-aggregate_to_substations)."""

    def test_file_exists(self, built):
        """elec_b.nc was produced by snakemake."""
        assert built.elec_b.exists(), f"missing {built.elec_b}"

    def test_bus_count_reasonable(self, built):
        """Substation count falls in the band expected for the test config."""
        n = pypsa.Network(str(built.elec_b))
        # CA-only Western yields O(50) substations after aggregation
        assert 10 < len(n.buses) < 200, (
            f"unexpected substation count {len(n.buses)} — config.test.yaml may have drifted"
        )

    def test_no_nan_in_coordinates(self, built):
        """Every bus has finite x/y coordinates."""
        n = pypsa.Network(str(built.elec_b))
        assert not n.buses[["x", "y"]].isna().any().any()


# ----- stage 2: cluster_simpl ----------------------------------------------


class TestPostClusterSimpl:
    """Assertions on elec_s{simpl}.nc and the busmap CSV (post-cluster_simpl)."""

    def test_file_exists(self, built):
        """elec_s.nc was produced by snakemake."""
        assert built.elec_s.exists(), f"missing {built.elec_s}"

    def test_bus_count_equals_simpl(self, built):
        """Cluster count matches the {simpl} wildcard exactly."""
        n = pypsa.Network(str(built.elec_s))
        assert len(n.buses) == int(built.simpl), f"cluster_simpl produced {len(n.buses)} buses, expected {built.simpl}"

    def test_busmap_exported(self, built):
        """cluster_simpl writes the busmap CSV consumed by aggregate_egs."""
        # PR #11 dependency: cluster_simpl now exports busmap_s{simpl}.csv
        # so aggregate_egs can remap substation-keyed supply curves.
        assert built.busmap_s.exists(), f"missing {built.busmap_s}"

    def test_busmap_covers_all_substations(self, built):
        """Every substation bus appears in the busmap index."""
        busmap = pd.read_csv(built.busmap_s, index_col=0)
        n_sub = pypsa.Network(str(built.elec_b))
        # Every substation bus must appear in the busmap index
        missing = set(n_sub.buses.index) - set(busmap.index.astype(str))
        assert not missing, f"busmap missing {len(missing)} substations"


# ----- stage 3: add_demand --------------------------------------------------


class TestPostAddDemand:
    """Assertions on elec_s{simpl}_dem.nc (post-add_demand)."""

    def test_file_exists(self, built):
        """elec_s_dem.nc was produced by snakemake."""
        assert built.elec_s_dem.exists(), f"missing {built.elec_s_dem}"

    def test_loads_attached(self, built):
        """At least one load is attached and p_set has no NaNs."""
        n = pypsa.Network(str(built.elec_s_dem))
        assert len(n.loads) > 0
        assert not n.loads_t.p_set.isna().any().any()


# ----- stage 4: add_electricity --------------------------------------------


class TestPostAddElectricity:
    """Assertions on elec_s{simpl}_l_pp.pkl (post-add_electricity)."""

    def test_file_exists(self, built):
        """elec_s_l_pp.pkl was produced by snakemake."""
        assert built.elec_s_l_pp.exists(), f"missing {built.elec_s_l_pp}"

    def test_pickle_loads(self, built):
        """The dill pickle deserialises into a pypsa.Network with generators."""
        with open(built.elec_s_l_pp, "rb") as f:
            n = dill.load(f)
        assert isinstance(n, pypsa.Network)
        assert len(n.generators) > 0

    def test_no_nan_in_load_bearing_columns(self, built):
        """generators/loads/buses load-bearing columns have no NaNs."""
        with open(built.elec_s_l_pp, "rb") as f:
            n = dill.load(f)
        assert not n.generators[["p_nom", "bus", "carrier"]].isna().any().any()
        assert not n.loads[["bus"]].isna().any().any()
        assert not n.buses[["x", "y", "carrier"]].isna().any().any()


# ----- stage 5: cluster_network --------------------------------------------


class TestPostClusterNetwork:
    """Assertions on elec_s{simpl}_c{clusters}.nc (post-cluster_network)."""

    def test_file_exists(self, built):
        """elec_s_c.nc was produced by snakemake."""
        assert built.elec_s_c.exists(), f"missing {built.elec_s_c}"

    def test_cluster_count_minimum(self, built):
        """Final cluster count meets the {clusters} = '4m' minimum."""
        n = pypsa.Network(str(built.elec_s_c))
        # 4m = at least 4 clusters (the 'm' suffix means minimum)
        assert len(n.buses) >= 4, f"cluster_network produced {len(n.buses)} buses, expected >=4"

    def test_no_nan_in_buses(self, built):
        """Buses in the final clustered network have no NaN in x/y/carrier."""
        n = pypsa.Network(str(built.elec_s_c))
        assert not n.buses[["x", "y", "carrier"]].isna().any().any()

    def test_netcdf_roundtrip(self, built, tmp_path):
        """The final network survives a netCDF write/read cycle unchanged."""
        n = pypsa.Network(str(built.elec_s_c))
        out = tmp_path / "roundtrip.nc"
        n.export_to_netcdf(str(out))
        n2 = pypsa.Network(str(out))
        # static frames must match exactly after write/read cycle
        pd.testing.assert_frame_equal(
            n.buses.sort_index(),
            n2.buses.sort_index(),
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            n.generators.sort_index(),
            n2.generators.sort_index(),
            check_dtype=False,
        )

    def test_filename_has_expected_wildcards(self, built):
        """The output filename embeds the {simpl} and {clusters} wildcards."""
        # Catches accidental drift in the cluster_network output pattern
        name = built.elec_s_c.name
        assert f"s{built.simpl}" in name
        assert f"c{built.clusters}" in name
