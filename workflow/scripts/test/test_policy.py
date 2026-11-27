"""
Test the content.py policy constraints functionality.

This module contains tests for the policy constraints in PyPSA-USA,
including Technology Capacity Targets (TCT), Renewable Portfolio Standards (RPS),
and Regional CO2 Limits.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import get_multiindex_snapshots
from opts._helpers import get_region_buses
from prepare_network import average_every_nhours
from summary import get_node_emissions_timeseries

logger = logging.getLogger(__name__)

# Fixtures


@pytest.fixture
def policy_network(base_network):
    """
    Adapt base network for policy constraint testing (RPS, TCT, CO2 limits).

    Extends the base network with parameters needed for policy constraint testing.
    """
    n = base_network.copy()

    # For policy constraints, we want clearer isolation of regions
    # Add a nuclear generator (non-renewable but clean)
    n.add(
        "Generator",
        "nuclear1",
        bus="z3",
        p_nom=300,
        p_nom_extendable=True,
        carrier="nuclear",
        capital_cost=2500,
        marginal_cost=5,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
        p_nom_max=1500,
    )

    # Add a coal generator (high CO2 emissions)
    n.add(
        "Generator",
        "coal1",
        bus="z1",
        p_nom=500,
        p_nom_extendable=True,
        carrier="coal",
        capital_cost=2000,
        marginal_cost=30,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
        p_nom_max=1000,
    )

    # Add CO2 emissions to carriers
    n.carriers.loc["coal", "co2_emissions"] = 0.8  # tonnes CO2 per MWh
    n.carriers.loc["nuclear", "co2_emissions"] = 0.0
    n.carriers.loc["gas", "co2_emissions"] = 0.4  # tonnes CO2 per MWh

    # Add nice_name to carriers for emissions calculation
    n.carriers["nice_name"] = n.carriers.index

    return n


@pytest.fixture
def clustered_policy_network(policy_network):
    """Create a time-clustered version of the policy network."""
    return average_every_nhours(policy_network, "3h")


@pytest.fixture
def co2_config():
    """Create a config dictionary for regional CO2 limit constraints."""
    return {
        "electricity": {
            "regional_Co2_limits": os.path.join(os.path.dirname(__file__), "fixtures/regional_co2_limits.csv"),
        },
        "scenario": {
            "planning_horizons": ["2030"],
        },
    }


@pytest.fixture
def rps_config():
    """Create a config dictionary for RPS constraints."""
    # Create config dictionary
    config = {
        "electricity": {
            "portfolio_standards": os.path.join(os.path.dirname(__file__), "fixtures/portfolio_standards.csv"),
        },
    }

    # Create a mock snakemake object
    class MockSnakemake:
        def __init__(self):
            self.input = type(
                "obj",
                (object,),
                {
                    "rps_reeds": os.path.join(os.path.dirname(__file__), "fixtures/rps_reeds.csv"),
                    "ces_reeds": os.path.join(os.path.dirname(__file__), "fixtures/ces_reeds.csv"),
                },
            )
            # rule solve_network passes no planning_horizons param; horizons come from the network

    snakemake = MockSnakemake()

    return config, snakemake


@pytest.fixture
def tct_config():
    """Create a config dictionary for TCT constraints."""
    return {
        "electricity": {
            "technology_capacity_targets": os.path.join(
                os.path.dirname(__file__),
                "fixtures/technology_capacity_targets.csv",
            ),
        },
    }


def test_add_regional_co2limit(policy_network, co2_config):
    """Test that regional CO2 limits are correctly added to the network."""
    from opts.policy import add_regional_co2limit

    n = policy_network
    config = co2_config

    # Add regional CO2 limits
    def extra_functionality(n, _):
        add_regional_co2limit(n, config)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Check that constraints were added
    assert any("co2_limit" in c for c in n.model.constraints), "No CO2 limit constraints were added"

    # Get emissions data
    emissions = get_node_emissions_timeseries(n)
    # Check that emissions are within limits for each region
    # Get the regional CO2 limits from config file
    co2_limits = pd.read_csv(config["electricity"]["regional_Co2_limits"])
    epsilon = 1e-2  # Small numerical tolerance
    for _, row in co2_limits.iterrows():
        limit = row["limit"]
        region_list = [region.strip() for region in row.regions.split(",")]
        region_buses = get_region_buses(n, region_list)
        constraint_emissions = emissions.loc[:, region_buses.index].sum().sum()
        assert constraint_emissions <= limit + epsilon, f"Emissions in region {row.name} exceed limit of {limit}"


def test_add_regional_co2limit_clustered(clustered_policy_network, co2_config):
    """Test that regional CO2 limits are correctly added to a time-clustered network."""
    from opts.policy import add_regional_co2limit

    n = clustered_policy_network
    config = co2_config

    # Add regional CO2 limits
    def extra_functionality(n, _):
        add_regional_co2limit(n, config)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Check that constraints were added
    assert any("co2_limit" in c for c in n.model.constraints), "No CO2 limit constraints were added"

    # Get emissions data
    emissions = get_node_emissions_timeseries(n)

    # Get the regional CO2 limits from config file
    co2_limits = pd.read_csv(config["electricity"]["regional_Co2_limits"])
    epsilon = 1e-2  # Small numerical tolerance
    for _, row in co2_limits.iterrows():
        limit = row["limit"]
        region_list = [region.strip() for region in row.regions.split(",")]
        region_buses = get_region_buses(n, region_list)
        constraint_emissions = emissions.loc[:, region_buses.index].sum().sum()
        assert constraint_emissions <= limit + epsilon, f"Emissions in region {row.name} exceed limit of {limit}"


def test_regional_co2limit_caps_the_emissions_of_each_period(tmp_path):
    """
    Over two periods, each limit caps the emissions of its own period.

    The CC plant vents to an atmosphere store built as ``add_co2_storage`` builds
    it, which the limit reads at the end of its period. A binding 2040 limit must
    leave 2030 alone, rather than also counting what was vented in 2030.
    """
    from opts.policy import add_regional_co2limit

    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 01:00", "inclusive": "both"},
        invest_periods=[2030, 2040],
    )
    n.set_investment_periods(periods=[2030, 2040])
    for col in n.snapshot_weightings.columns:  # each period stands for one year
        n.snapshot_weightings[col] = 8760 / 2
    n.add(
        "Bus",
        "p1",
        carrier="AC",
        country="US",
        reeds_zone="p1",
        reeds_state="UT",
        interconnect="western",
        nerc_reg="WECC",
    )
    n.add("Load", "p1", bus="p1", p_set=10)
    n.add("Generator", "p1 clean", bus="p1", carrier="clean", p_nom=100, marginal_cost=100)
    # the limit only reads the atmosphere when an emitting generator is in the region
    n.add("Carrier", "coal", co2_emissions=0.3)
    n.add("Generator", "p1 coal", bus="p1", carrier="coal", p_nom=100, efficiency=0.4, marginal_cost=50)

    # CC plant venting 0.04 t/MWh to the atmosphere store, as in add_co2_storage
    n.add("Carrier", "co2")
    n.add("Bus", ["p1 atmosphere", "p1 CCGT-95CC_2030"], carrier=["co2", "CCGT-95CC"])
    n.add(
        "Store",
        "p1 atmosphere",
        bus="p1 atmosphere",
        e_nom_extendable=True,
        e_min_pu=-1,
        e_initial_per_period=True,
        carrier="co2",
    )
    n.add("Generator", "p1 CCGT-95CC_2030", bus="p1 CCGT-95CC_2030", carrier="CCGT-95CC", p_nom=100, marginal_cost=15)
    n.add(
        "Link",
        "p1 CCGT-95CC_2030",
        bus0="p1 CCGT-95CC_2030",
        bus1="p1",
        bus2="p1 atmosphere",
        efficiency2=0.04,
        p_nom=100,
        carrier="CCGT-95CC",
    )

    def vented(n):
        flows = n.links_t.p2["p1 CCGT-95CC_2030"].mul(-1).mul(n.snapshot_weightings.generators)
        return flows.groupby(level="period").sum()

    # unconstrained, the CC plant meets the load in both periods
    annual = 10 * 8760 * 0.04
    limits = tmp_path / "regional_co2_limits.csv"
    pd.DataFrame(
        {"name": ["p1"] * 2, "regions": ["p1"] * 2, "planning_horizon": [2030, 2040], "limit": [1e9, annual / 2]}
    ).to_csv(
        limits,
        index=False,
    )

    def extra_functionality(n, _):
        add_regional_co2limit(n, {"electricity": {"regional_Co2_limits": str(limits)}})

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    assert vented(n).tolist() == pytest.approx([annual, annual / 2])
    # the atmosphere starts every period empty
    atmosphere = n.stores_t.e["p1 atmosphere"].groupby(level="period").last()
    assert atmosphere.tolist() == pytest.approx(vented(n).tolist())


def test_add_rps_constraints(policy_network, rps_config):
    """Test that RPS constraints are correctly added to the network."""
    from opts.policy import add_RPS_constraints

    n = policy_network
    config, snakemake = rps_config

    # Add RPS constraints
    def extra_functionality(n, _):
        add_RPS_constraints(n, config, snakemake=snakemake)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Check that constraints were added
    assert any("rps_limit" in c for c in n.model.constraints), "No RPS limit constraints were added"

    # Get the portfolio standards from config file
    portfolio_standards = pd.read_csv(config["electricity"]["portfolio_standards"])

    # Check that renewable generation meets RPS requirements
    for _, row in portfolio_standards.iterrows():
        region_list = [region.strip() for region in row.region.split(",")]
        region_buses = get_region_buses(n, region_list)

        if region_buses.empty:
            continue

        carriers = [carrier.strip() for carrier in row.carrier.split(",")]
        region_gens = n.generators[n.generators.bus.isin(region_buses.index)]
        region_gens_eligible = region_gens[region_gens.carrier.isin(carriers)]

        if region_gens_eligible.empty:
            continue

        # Calculate total generation from eligible sources
        # energy over the standard's planning horizon, each snapshot weighted by its hours
        weights = n.snapshot_weightings.generators.loc[row.planning_horizon]
        eligible_generation = (
            n.generators_t.p.loc[row.planning_horizon, region_gens_eligible.index].mul(weights, axis=0).sum().sum()
        )

        # Calculate total demand in the region
        region_demand = (
            n.loads_t.p_set.loc[row.planning_horizon, n.loads.bus.isin(region_buses.index)]
            .mul(weights, axis=0)
            .sum()
            .sum()
        )
        logger.info(
            f"RPS Check: Region: {row.region}, Carriers: {carriers}, "
            f"Eligible Generation: {eligible_generation:.2f} MWh, "
            f"Total Demand: {region_demand:.2f} MWh, "
            f"Required %: {row.pct * 100:.1f}%, "
            f"Actual %: {(eligible_generation / region_demand) * 100:.1f}%",
        )
        # Check if RPS requirement is met with small epsilon for rounding errors
        epsilon = 1e-3
        assert eligible_generation >= (row.pct * region_demand) - epsilon, (
            f"RPS requirement of {row.pct * 100}% not met for region {row.region}"
        )


def test_rps_weights_snapshots_by_their_hours(policy_network, rps_config):
    """The RPS counts energy: generation and demand are weighted by the snapshot hours."""
    from opts.policy import add_RPS_constraints

    n = policy_network
    config, snakemake = rps_config
    weights = pd.Series(np.linspace(1.0, 7.0, len(n.snapshots)), index=n.snapshots)
    n.snapshot_weightings["generators"] = weights
    n.snapshot_weightings["objective"] = weights
    n.optimize.create_model(multi_investment_periods=True)
    add_RPS_constraints(n, config, snakemake=snakemake)

    names = [c for c in n.model.constraints if "rps_limit" in c]
    assert names
    for name in names:
        con = n.model.constraints[name]
        coeffs = con.coeffs.values[con.vars.values != -1]
        # each generator's output enters with the hours of its snapshot
        assert np.isin(np.round(coeffs, 9), np.round(weights.to_numpy(), 9)).all()
        assert len(np.unique(np.round(coeffs, 9))) > 1


def test_sector_rps_weights_snapshots_by_their_hours(policy_network, rps_config):
    """The sector RPS weights both the eligible and the total generation by the snapshot hours."""
    from opts.policy import add_RPS_constraints_sector

    n = policy_network
    config, snakemake = rps_config
    config = {
        "electricity": {
            **config["electricity"],
            "conventional_carriers": ["gas", "nuclear"],
            "renewable_carriers": ["solar", "onwind"],
            "extendable_carriers": {"Generator": []},
        },
    }
    weights = pd.Series(np.linspace(1.0, 7.0, len(n.snapshots)), index=n.snapshots)
    n.snapshot_weightings["generators"] = weights
    n.optimize.create_model(multi_investment_periods=True)
    add_RPS_constraints_sector(n, config, snakemake=snakemake)

    # model label of Generator-p -> (generator, snapshot weight)
    labels = n.model["Generator-p"].labels.to_pandas()
    stacked = labels.stack()
    owner = pd.Series(stacked.index.get_level_values("name"), index=stacked.to_numpy())
    hours = pd.Series(weights.reindex(stacked.index.droplevel("name")).to_numpy(), index=stacked.to_numpy())

    names = [c for c in n.model.constraints if c.endswith("_limit") and c.startswith("GlobalConstraint-")]
    assert names
    for name in names:
        con = n.model.constraints[name]
        valid = con.vars.values != -1
        terms = pd.DataFrame({"var": con.vars.values[valid], "coeff": con.coeffs.values[valid]})
        # a generator can enter twice (eligible and total generation): add its terms up
        coeff = terms[terms["var"].isin(owner.index)].groupby("var")["coeff"].sum()
        # each generator's coefficient is its share (1 - pct or -pct) times the snapshot hours
        per_hour = (coeff / coeff.index.map(hours)).groupby(coeff.index.map(owner))
        assert (per_hour.max() - per_hour.min()).abs().max() < 1e-9
        assert coeff.abs().round(9).nunique() > 1


def test_add_rps_constraints_clustered(clustered_policy_network, rps_config):
    """Test that RPS constraints are correctly added to a time-clustered network."""
    from opts.policy import add_RPS_constraints

    n = clustered_policy_network
    config, snakemake = rps_config

    # Add RPS constraints
    def extra_functionality(n, _):
        add_RPS_constraints(n, config, snakemake=snakemake)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Check that constraints were added
    assert any("rps_limit" in c for c in n.model.constraints), "No RPS limit constraints were added"

    # Get the portfolio standards from config file
    portfolio_standards = pd.read_csv(config["electricity"]["portfolio_standards"])

    # Check that renewable generation meets RPS requirements
    for _, row in portfolio_standards.iterrows():
        region_list = [region.strip() for region in row.region.split(",")]
        region_buses = get_region_buses(n, region_list)

        if region_buses.empty:
            continue

        carriers = [carrier.strip() for carrier in row.carrier.split(",")]
        region_gens = n.generators[n.generators.bus.isin(region_buses.index)]
        region_gens_eligible = region_gens[region_gens.carrier.isin(carriers)]

        if region_gens_eligible.empty:
            continue

        # Calculate total generation from eligible sources
        # energy over the standard's planning horizon, each snapshot weighted by its hours
        weights = n.snapshot_weightings.generators.loc[row.planning_horizon]
        eligible_generation = (
            n.generators_t.p.loc[row.planning_horizon, region_gens_eligible.index].mul(weights, axis=0).sum().sum()
        )

        # Calculate total demand in the region
        region_demand = (
            n.loads_t.p_set.loc[row.planning_horizon, n.loads.bus.isin(region_buses.index)]
            .mul(weights, axis=0)
            .sum()
            .sum()
        )
        logger.info(
            f"RPS Check (Clustered): Region: {row.region}, Carriers: {carriers}, "
            f"Eligible Generation: {eligible_generation:.2f} MWh, "
            f"Total Demand: {region_demand:.2f} MWh, "
            f"Required %: {row.pct * 100:.1f}%, "
            f"Actual %: {(eligible_generation / region_demand) * 100:.1f}%",
        )
        # Check if RPS requirement is met with small epsilon for rounding errors
        epsilon = 1e-3
        assert eligible_generation >= (row.pct * region_demand) - epsilon, (
            f"RPS requirement of {row.pct * 100}% not met for region {row.region} in clustered network"
        )


def test_add_technology_capacity_target_constraints(policy_network, tct_config):
    """Test that technology capacity target constraints are correctly added to the network."""
    from opts.policy import add_technology_capacity_target_constraints

    n = policy_network
    config = tct_config

    # Add TCT constraints
    def extra_functionality(n, _):
        add_technology_capacity_target_constraints(n, config)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Check that constraints were added
    assert any("min" in c for c in n.model.constraints), "No TCT minimum constraints were added"
    assert any("max" in c for c in n.model.constraints), "No TCT maximum constraints were added"

    # Get the technology capacity targets from config file
    tct_data = pd.read_csv(config["electricity"]["technology_capacity_targets"])

    # Check that capacity targets are met
    for _, target in tct_data.iterrows():
        region_list = [region.strip() for region in target.region.split(",")]
        region_buses = get_region_buses(n, region_list)

        if region_buses.empty:
            continue

        carriers = [carrier.strip() for carrier in target.carrier.split(",")]

        # Get total capacity (existing + new) for the target technology
        total_capacity = 0

        # Check generators
        gens = n.generators[(n.generators.bus.isin(region_buses.index)) & (n.generators.carrier.isin(carriers))]
        total_capacity += gens.p_nom_opt.sum()

        # Check storage units
        storage = n.storage_units[
            (n.storage_units.bus.isin(region_buses.index)) & (n.storage_units.carrier.isin(carriers))
        ]
        total_capacity += storage.p_nom_opt.sum()

        # Check links
        links = n.links[(n.links.bus0.isin(region_buses.index)) & (n.links.carrier.isin(carriers))]
        total_capacity += links.p_nom_opt.sum()

        # Get min and max targets, handling NaN values
        min_target = float(target["min"]) if not pd.isna(target["min"]) else None
        max_target = float(target["max"]) if not pd.isna(target["max"]) else None

        logger.info(
            f"TCT Check: Region: {target.region}, Carrier: {carriers}, "
            f"Total Capacity: {total_capacity:.2f} MW, "
            f"Min Target: {min_target}, "
            f"Max Target: {max_target}",
        )

        # Check minimum capacity if specified
        if min_target is not None:
            assert total_capacity >= min_target, (
                f"Minimum capacity target of {min_target} MW not met for {target['carrier']} in {target['region']}"
            )

        # Check maximum capacity if specified
        if max_target is not None:
            assert total_capacity <= max_target, (
                f"Maximum capacity target of {max_target} MW exceeded for {target['carrier']} in {target['region']}"
            )


def test_apply_forced_retirements_zeroes_nonextendable(policy_network, tmp_path):
    """apply_forced_retirements should zero non-extendable generators matching max=0 targets.

    Simulates the brownfield scenario where prepare_brownfield() has locked in
    previous-period capacity as non-extendable. Verifies that:
    - Non-extendable CCGT in the target region (CA/z1) is zeroed.
    - Non-extendable CCGT outside the target region (TX/z2) is untouched.
    - Extendable generators are not modified (the TCT constraint handles those).
    """
    from opts.policy import apply_forced_retirements

    n = policy_network.copy()

    # Add non-extendable CCGT generators simulating brownfield-locked capacity
    n.add(
        "Generator",
        "ccgt_ca",
        bus="z1",
        carrier="CCGT",
        p_nom=500.0,
        p_nom_min=0.0,
        p_nom_max=500.0,
        p_nom_extendable=False,
        marginal_cost=40,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
    )
    n.add(
        "Generator",
        "ccgt_tx",
        bus="z2",
        carrier="CCGT",
        p_nom=300.0,
        p_nom_min=0.0,
        p_nom_max=300.0,
        p_nom_extendable=False,
        marginal_cost=40,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
    )
    # Extendable CCGT — should NOT be touched by apply_forced_retirements
    n.add(
        "Generator",
        "ccgt_ca_ext",
        bus="z1",
        carrier="CCGT",
        p_nom=0.0,
        p_nom_min=0.0,
        p_nom_max=1000.0,
        p_nom_extendable=True,
        marginal_cost=40,
        capital_cost=600,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
    )

    # Write a TCT CSV forcing CCGT retirement in CA (reeds_state) by 2030
    tct_csv = tmp_path / "tct_forced.csv"
    tct_csv.write_text(
        "name,planning_horizon,region,carrier,min,max\nccgt_retire,2030,CA,CCGT,,0\n",
    )

    config = {"electricity": {"technology_capacity_targets": str(tct_csv)}}

    apply_forced_retirements(n, planning_horizon=2030, config=config)

    # CA non-extendable CCGT should be zeroed
    assert n.generators.loc["ccgt_ca", "p_nom"] == 0.0, "ccgt_ca p_nom should be 0"
    assert n.generators.loc["ccgt_ca", "p_nom_max"] == 0.0, "ccgt_ca p_nom_max should be 0"

    # TX non-extendable CCGT should be untouched
    assert n.generators.loc["ccgt_tx", "p_nom"] == 300.0, "ccgt_tx p_nom should be unchanged"

    # Extendable CCGT should be untouched (handled by TCT constraint, not this function)
    assert n.generators.loc["ccgt_ca_ext", "p_nom_max"] == 1000.0, "extendable ccgt should be unchanged"


def test_apply_forced_retirements_future_horizon_skipped(policy_network, tmp_path):
    """Forced retirement rows with planning_horizon > current should be skipped."""
    from opts.policy import apply_forced_retirements

    n = policy_network.copy()
    n.add(
        "Generator",
        "ccgt_ca",
        bus="z1",
        carrier="CCGT",
        p_nom=500.0,
        p_nom_min=0.0,
        p_nom_max=500.0,
        p_nom_extendable=False,
        marginal_cost=40,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
    )

    # Retirement target is 2050 — should not apply during the 2030 solve
    tct_csv = tmp_path / "tct_future.csv"
    tct_csv.write_text(
        "name,planning_horizon,region,carrier,min,max\nccgt_retire,2050,CA,CCGT,,0\n",
    )

    config = {"electricity": {"technology_capacity_targets": str(tct_csv)}}

    apply_forced_retirements(n, planning_horizon=2030, config=config)

    assert n.generators.loc["ccgt_ca", "p_nom"] == 500.0, "2050 target should not retire capacity in 2030"


def test_apply_forced_retirements_all_region(policy_network, tmp_path):
    """planning_horizon='all' with region='all' should zero matching gens in every region."""
    from opts.policy import apply_forced_retirements

    n = policy_network.copy()
    n.add(
        "Generator",
        "coal_ca",
        bus="z1",
        carrier="coal",
        p_nom=400.0,
        p_nom_min=0.0,
        p_nom_max=400.0,
        p_nom_extendable=False,
        marginal_cost=30,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
    )
    n.add(
        "Generator",
        "coal_tx",
        bus="z2",
        carrier="coal",
        p_nom=250.0,
        p_nom_min=0.0,
        p_nom_max=250.0,
        p_nom_extendable=False,
        marginal_cost=30,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
    )

    tct_csv = tmp_path / "tct_all.csv"
    tct_csv.write_text(
        "name,planning_horizon,region,carrier,min,max\ncoal_retire,all,all,coal,,0\n",
    )

    config = {"electricity": {"technology_capacity_targets": str(tct_csv)}}

    apply_forced_retirements(n, planning_horizon=2030, config=config)

    assert n.generators.loc["coal_ca", "p_nom"] == 0.0
    assert n.generators.loc["coal_tx", "p_nom"] == 0.0


def test_add_rps_constraints_without_params(policy_network, rps_config):
    """The solve rule passes no ``planning_horizons`` param; horizons must come from the network."""
    from opts.policy import add_RPS_constraints

    n = policy_network
    config, snakemake = rps_config
    assert not hasattr(snakemake, "params")

    n.optimize.create_model(multi_investment_periods=True)
    add_RPS_constraints(n, config, snakemake=snakemake)

    rps = [c for c in n.model.constraints if c.endswith("_rps_limit")]
    assert rps, "no RPS constraint was added"
    assert all("2030" in c for c in rps)


def test_tct_storage_mandate_counts_existing_and_all_durations(policy_network, tmp_path):
    """A storage mandate listing every battery carrier nets out existing batteries and lets any duration count."""
    from opts.policy import add_technology_capacity_target_constraints

    n = policy_network
    n.add("StorageUnit", "battery_existing", bus="z1", carrier="battery", p_nom=100, max_hours=2)
    for hours in (4, 8):
        n.add(
            "StorageUnit",
            f"battery_{hours}hr",
            bus="z1",
            carrier=f"{hours}hr_battery_storage",
            p_nom_extendable=True,
            max_hours=hours,
            capital_cost=100 * hours,
        )

    fn = tmp_path / "tct.csv"
    fn.write_text(
        "name,planning_horizon,region,carrier,min,max\n"
        'reeds_storage_mandate,2030,CA,"battery, 4hr_battery_storage, 8hr_battery_storage",150,\n',
    )
    n.optimize.create_model(multi_investment_periods=True)
    add_technology_capacity_target_constraints(n, {"electricity": {"technology_capacity_targets": str(fn)}})

    (name,) = [c for c in n.model.constraints if c.startswith("GlobalConstraint-") and c.endswith("_min")]
    con = n.model.constraints[name]
    # 150 MW mandate minus the 100 MW existing battery
    assert float(con.rhs) == 50
    # every extendable battery in CA enters the left-hand side, whatever its duration
    # (the base network already carries an extendable 'battery1' on z1)
    su = n.storage_units
    expected = su.index[su.p_nom_extendable & (su.bus == "z1") & su.carrier.str.contains("battery")]
    assert {"battery_4hr", "battery_8hr"} <= set(expected)
    lhs_vars = set(con.lhs.vars.values.ravel()) - {-1}
    p_nom = n.model["StorageUnit-p_nom"]
    assert lhs_vars == {int(p_nom.sel(name=name).labels) for name in expected}


def test_shipped_storage_mandates_cover_every_battery_carrier():
    """The shipped storage mandates count existing batteries and every new-build battery duration."""
    from constants import ATB_TECH_MAPPER

    tct = pd.read_csv(
        Path(__file__).parents[2] / "repo_data" / "config" / "policy_constraints" / "technology_capacity_targets.csv"
    )
    storage = tct[tct.name == "reeds_storage_mandate"]
    battery_carriers = {"battery"} | {c for c in ATB_TECH_MAPPER if c.endswith("hr_battery_storage")}
    for carriers in storage.carrier.unique():
        assert {c.strip() for c in carriers.split(",")} == battery_carriers


def test_process_reeds_ces_long_format(tmp_path):
    """The ReEDS CES file is long, headed '*t,st,Value'; zero fractions are dropped."""
    from opts.policy import CES_CARRIERS, _process_reeds_data

    fn = tmp_path / "ces_fraction.csv"
    fn.write_text("*t,st,Value\n2029,CA,0\n2030,CA,0.605\n2040,CA,0.917\n2040,CO,0.726\n")

    ces = _process_reeds_data(fn, CES_CARRIERS, value_col="pct")

    assert list(ces.columns) == ["region", "planning_horizon", "carrier", "pct"]
    assert ces.planning_horizon.dtype.kind == "i"
    assert len(ces) == 3
    assert ces.set_index(["region", "planning_horizon"]).pct.loc[("CA", 2030)] == 0.605
    assert set(ces.carrier) == {", ".join(CES_CARRIERS)}


def test_process_reeds_rps_ignores_carveouts(tmp_path):
    """The ReEDS RPS file is long; only rps_all is used, the solar/wind carve-outs are ignored."""
    from opts.policy import RPS_CARRIERS, _process_reeds_data

    fn = tmp_path / "rps_fraction.csv"
    fn.write_text("t,st,rps_all,rps_solar,rps_wind\n2030,DE,0.22,0.05,0\n2030,AZ,0,0,0\n")

    rps = _process_reeds_data(fn, RPS_CARRIERS, value_col="pct")

    assert rps.region.tolist() == ["DE"]
    assert rps.pct.tolist() == [0.22]


def test_process_reeds_rejects_wide_format(tmp_path):
    """The old wide layout (one column per year) is no longer read."""
    from opts.policy import CES_CARRIERS, _process_reeds_data

    fn = tmp_path / "ces_fraction.csv"
    fn.write_text("st,2030,2040\nCA,0.571,0.8\n")

    with pytest.raises(ValueError, match="long-format"):
        _process_reeds_data(fn, CES_CARRIERS, value_col="pct")


def test_shipped_reeds_portfolio_files_parse():
    """The ReEDS RPS and CES files shipped in repo_data parse and keep the post-2025 horizons."""
    from opts.policy import CES_CARRIERS, RPS_CARRIERS, _process_reeds_data

    reeds_dir = Path(__file__).parents[2] / "repo_data" / "config" / "policy_constraints" / "reeds"
    rps = _process_reeds_data(reeds_dir / "rps_fraction.csv", RPS_CARRIERS, value_col="pct")
    ces = _process_reeds_data(reeds_dir / "ces_fraction.csv", CES_CARRIERS, value_col="pct")

    for df in (rps, ces):
        assert df.planning_horizon.between(2010, 2050).all()
        assert df.pct.between(0, 1).all()
        assert 2030 in set(df.planning_horizon)
    assert len(ces.region.unique()) == 16


def test_add_rps_constraints_zone_without_eligible_gens_does_not_stop_others(policy_network, rps_config, tmp_path):
    """A zone with no eligible generators is skipped; the remaining zones still get their constraint."""
    from opts.policy import add_RPS_constraints

    n = policy_network
    config, snakemake = rps_config

    # give CA a carrier no generator has, so its group is empty; TX keeps its real carriers
    ps = pd.read_csv(config["electricity"]["portfolio_standards"])
    ps.loc[ps.region == "CA", "carrier"] = "geothermal"
    fn = tmp_path / "portfolio_standards.csv"
    ps.to_csv(fn, index=False)
    config["electricity"]["portfolio_standards"] = str(fn)

    n.optimize.create_model(multi_investment_periods=True)
    add_RPS_constraints(n, config, snakemake=snakemake)

    rps = [c for c in n.model.constraints if c.endswith("_rps_limit")]
    assert rps, "the non-empty zone lost its constraint"
    assert not any(c.startswith("GlobalConstraint-CA") for c in rps)


def test_portfolio_standard_carriers_are_model_carriers():
    """Portfolio standards match carriers by exact name, so a misspelt one silently gets no credit (#871)."""
    from constants import ATB_TECH_MAPPER
    from opts.policy import CES_CARRIERS, RPS_CARRIERS

    # ATB technologies, plus the carriers build_cost_data derives from the gas plants and the biomass plant
    model_carriers = set(ATB_TECH_MAPPER) | {"OCHT", "CCHT", "CCGT-90CCS", "biomass-95CCS"}
    assert set(RPS_CARRIERS) <= model_carriers
    assert set(CES_CARRIERS) <= model_carriers
    assert {"CCGT-95CCS", "CCGT-97CCS", "coal-95CCS", "coal-99CCS"} <= set(CES_CARRIERS)


def test_rps_and_ces_in_one_zone_and_year_get_one_constraint_each(policy_network, tmp_path):
    """A REC trading zone with both an RPS and a CES in a year gets two constraints, not a name clash."""
    from opts.policy import add_RPS_constraints

    (tmp_path / "rps.csv").write_text("t,st,rps_all,rps_solar,rps_wind\n2030,CA,0.5,0,0\n")
    (tmp_path / "ces.csv").write_text("*t,st,Value\n2030,CA,0.6\n")
    (tmp_path / "custom.csv").write_text("planning_horizon,region,carrier,pct\n")
    config = {"electricity": {"portfolio_standards": str(tmp_path / "custom.csv")}}
    inputs = {"rps_reeds": str(tmp_path / "rps.csv"), "ces_reeds": str(tmp_path / "ces.csv")}
    snakemake = type("MockSnakemake", (), {"input": type("Input", (), inputs)})

    n = policy_network
    n.optimize.create_model(multi_investment_periods=True)
    add_RPS_constraints(n, config, snakemake=snakemake)

    names = [c for c in n.model.constraints if c.startswith("GlobalConstraint-CA_2030")]
    assert sorted(names) == ["GlobalConstraint-CA_2030_ces_limit", "GlobalConstraint-CA_2030_rps_limit"]


def test_tct_existing_target_on_a_string_csv(policy_network, tmp_path):
    """A ``max: existing`` row, read by pandas 3 as a str row, gets a numeric max constraint."""
    from opts.policy import add_technology_capacity_target_constraints

    fn = tmp_path / "technology_capacity_targets.csv"
    fn.write_text("name,region,carrier,planning_horizon,min,max\nnuclear_no_build,west,nuclear,all,,existing\n")
    n = policy_network
    n.optimize.create_model(multi_investment_periods=True)

    add_technology_capacity_target_constraints(n, {"electricity": {"technology_capacity_targets": str(fn)}})

    assert any("max" in c for c in n.model.constraints)
