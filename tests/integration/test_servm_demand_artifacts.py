"""Tier B — assert shape and orientation of the CPUC SERVM demand artifacts.

Built from ``workflow/repo_data/config/config.test.california.yaml`` by the
``servm_built`` fixture (``snakemake --until add_demand``). These are structural
checks — conservation of the allocation weights, alignment of the per-bus demand
CSV with the network it was built against, and the PST->UTC roll. No numerical
regression against the CPUC reference case here; that is the (separate)
benchmark rule ``run.benchmark_cpuc`` reserves.
"""

from __future__ import annotations

import pandas as pd
import pypsa
import pytest

pytestmark = pytest.mark.integration

# The six CPUC SERVM California load regions (ReadServm.REGIONS).
SERVM_REGIONS = {"IID", "LADWP", "NCNC", "PGE", "SCE", "SDGE"}


class TestServmLoadWeights:
    """Assertions on servm_load_weights_s{simpl}.csv (build_servm_load_weights)."""

    def test_file_exists(self, servm_built):
        """The weights table was produced by snakemake."""
        assert servm_built.weights.exists(), f"missing {servm_built.weights}"

    def test_columns(self, servm_built):
        """The table carries the three columns WriteServm reads."""
        weights = pd.read_csv(servm_built.weights)
        assert {"bus", "servm_region", "laf"}.issubset(weights.columns)

    def test_laf_sums_to_one_per_region(self, servm_built):
        """Allocation factors conserve each region's demand exactly."""
        weights = pd.read_csv(servm_built.weights)
        totals = weights.groupby("servm_region")["laf"].sum()
        assert not totals.empty, "no SERVM regions in the weights table"
        pd.testing.assert_series_equal(
            totals,
            pd.Series(1.0, index=totals.index, name="laf"),
            check_exact=False,
            rtol=1e-6,
        )

    def test_regions_are_known(self, servm_built):
        """Only the six published SERVM regions appear (CISO-VEA is excluded)."""
        weights = pd.read_csv(servm_built.weights)
        unknown = set(weights.servm_region.unique()) - SERVM_REGIONS
        assert not unknown, f"unexpected SERVM region(s) {sorted(unknown)}"

    def test_buses_exist_in_network(self, servm_built):
        """Every weighted bus is a bus of the network the weights were built for."""
        weights = pd.read_csv(servm_built.weights)
        n = pypsa.Network(str(servm_built.elec_s))
        unknown = set(weights.bus.astype(str)) - set(n.buses.index.astype(str))
        assert not unknown, f"weights reference {len(unknown)} bus(es) not in elec_s{servm_built.simpl}.nc"


class TestServmDemand:
    """Assertions on power_electricity_s{simpl}.csv (build_electrical_demand)."""

    def test_file_exists(self, servm_built):
        """The per-bus demand CSV was produced by snakemake."""
        assert servm_built.demand.exists(), f"missing {servm_built.demand}"

    def test_columns_are_network_buses(self, servm_built):
        """Demand columns are buses of the network, and there is at least one."""
        demand = pd.read_csv(servm_built.demand, index_col=0)
        n = pypsa.Network(str(servm_built.elec_s))
        assert len(demand.columns) > 0
        unknown = set(demand.columns.astype(str)) - set(n.buses.index.astype(str))
        assert not unknown, f"demand columns are not network buses: {sorted(unknown)[:10]}"

    def test_row_count_matches_snapshots(self, servm_built):
        """One row per network snapshot.

        Compares against the demand-attached network: ``elec_s{simpl}.nc``
        still carries PyPSA's default ``['now']`` index — snapshots are only
        set when ``add_demand`` attaches the load.
        """
        demand = pd.read_csv(servm_built.demand, index_col=0)
        n = pypsa.Network(str(servm_built.elec_s_dem))
        assert len(demand) == len(n.snapshots), (
            f"demand has {len(demand)} rows, network has {len(n.snapshots)} snapshots"
        )

    def test_demand_is_positive(self, servm_built):
        """No NaNs, no negative load, and a non-trivial total."""
        demand = pd.read_csv(servm_built.demand, index_col=0)
        assert not demand.isna().any().any()
        assert (demand.to_numpy() >= 0).all(), "SERVM Net Load produced negative demand"
        assert demand.to_numpy().sum() > 0

    def test_attached_to_network(self, servm_built):
        """add_demand attached the SERVM load to elec_s{simpl}_dem.nc."""
        n = pypsa.Network(str(servm_built.elec_s_dem))
        assert len(n.loads) > 0
        assert not n.loads_t.p_set.isna().any().any()
        assert n.loads_t.p_set.to_numpy().sum() > 0

    def test_peak_hour_consistent_with_pst(self, servm_built):
        """The annual peak lands in the CA afternoon/evening once rolled to UTC.

        ``ReadServm`` rolls the fixed-PST strip forward by 8 hours, so the
        snapshot index is effectively UTC. A California system peak sits in the
        late afternoon local time; 15:00-20:00 PST maps to UTC hours 23-04. The
        band is deliberately wide — it is testing the direction and magnitude of
        the roll, not the exact peak hour. (Peak *date* is not asserted: SERVM
        lays its hours on a synthetic Monday-start calendar, and a leap weather
        year shifts every post-February hour by one calendar day.)
        """
        demand = pd.read_csv(servm_built.demand, index_col=0, parse_dates=True)
        peak_hour = demand.sum(axis=1).idxmax().hour
        assert peak_hour in {
            23,
            0,
            1,
            2,
            3,
            4,
        }, f"annual peak at UTC hour {peak_hour}; expected 23-04 for a PST-rolled California profile"


class TestServmZonalComponents:
    """Assertions on power_zonal_components_s{simpl}.parquet."""

    def test_file_exists(self, servm_built):
        """The component-resolved zonal artifact was produced."""
        assert servm_built.zonal_components.exists(), f"missing {servm_built.zonal_components}"

    def test_regions_are_columns(self, servm_built):
        """Columns are the SERVM regions; every region carries nonzero energy."""
        zonal = pd.read_parquet(servm_built.zonal_components)
        assert set(zonal.columns) <= SERVM_REGIONS, f"unexpected column(s) {sorted(set(zonal.columns) - SERVM_REGIONS)}"
        net_load = zonal.xs("Net Load", level="subsector")
        assert (net_load.sum() > 0).all(), (
            f"zero annual energy in region(s) {sorted(net_load.columns[net_load.sum() <= 0])}"
        )

    def test_net_load_component_present(self, servm_built):
        """``Net Load`` — the only component the model dispatches against — is kept."""
        zonal = pd.read_parquet(servm_built.zonal_components)
        assert "Net Load" in zonal.index.get_level_values("subsector")

    def test_components_beyond_net_load_are_kept(self, servm_built):
        """The zonal artifact stays component-resolved (BTMPV, EV, ... survive)."""
        zonal = pd.read_parquet(servm_built.zonal_components)
        components = set(zonal.index.get_level_values("subsector"))
        assert components - {"Net Load"}, "zonal artifact collapsed to Net Load only"
