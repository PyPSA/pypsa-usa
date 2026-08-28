"""
Calculates summary files.

Adapted from PyPSA-Eur summary statistics reporting script
 - https://github.com/PyPSA/pypsa-eur/blob/master/scripts/make_summary.py
"""

import logging

import pandas as pd
import pypsa
from pypsa.statistics import StatisticsAccessor

logger = logging.getLogger(__name__)


def _iter_components(n: pypsa.Network, names) -> list:
    """Non-empty components of ``n`` among ``names`` (replaces deprecated iterate_components)."""
    return [n.components[name] for name in names if not n.components[name].static.empty]


###
# ENERGY SUPLPY
###


def get_primary_energy_use(n: pypsa.Network) -> pd.DataFrame:
    """Gets timeseries primary energy use by bus and carrier."""
    link_energy_use = (
        StatisticsAccessor(n)
        .withdrawal(
            components=["Link", "Store", "StorageUnit"],
            groupby_time=False,
            groupby=["bus", "carrier"],
        )
        .droplevel("component")
    )

    gen_dispatch = (
        StatisticsAccessor(n)
        .supply(
            groupby_time=False,
            components=["Generator"],
            groupby=["name", "bus", "carrier"],
        )
        .droplevel("component")
    )
    gen_eff = n.get_switchable_as_dense("Generator", "efficiency")

    gen_energy_use = gen_dispatch.T.mul(1 / gen_eff, axis=0, level="name").T.droplevel(
        "name",
    )

    return (
        pd.concat([gen_energy_use, link_energy_use])
        # .reset_index() commenting this out seems to fix issue in multi-horizon indexing
        .groupby(["bus", "carrier"])
        .sum()
    )


def get_energy_timeseries(n: pypsa.Network) -> pd.DataFrame:
    """Gets timeseries energy production."""

    def _get_energy_one_port(n: pypsa.Network, c: str) -> pd.DataFrame:
        return (
            c.dynamic.p.multiply(  # .multiply(n.snapshot_weightings.generators, axis=0)
                c.static.sign,
            )
            .T.groupby(c.static.carrier)
            .sum()
            .T
        )

    def _get_energy_multi_port(n: pypsa.Network, c: str) -> pd.DataFrame:
        c_energies = (
            pd.DataFrame(
                index=n.snapshots,
                columns=c.static.carrier.unique(),
            )
            .astype(float)
            .fillna(0)
        )
        for port in [col[3:] for col in c.static.columns if col[:3] == "bus"]:
            if port == "0":  # only track flow in one direction
                continue
            totals = c.dynamic["p" + port]  # .multiply(n.snapshot_weightings.generators,axis=0,)
            if totals.empty:
                continue
            # remove values where bus is missing (bug in nomopyomo)
            no_bus = c.static.index[c.static["bus" + port] == ""]
            totals.loc[no_bus] = float(
                n.components[c.name].defaults.loc["p" + port, "default"],
            )
            c_energies -= totals.T.groupby(c.static.carrier).sum().T
        return c_energies

    energy = []
    for c in _iter_components(n, n.one_port_components | n.branch_components):
        if c.name in ("Generator", "StorageUnit", "Store"):
            e = _get_energy_one_port(n, c)
        elif c.name in ("Link"):
            e = _get_energy_multi_port(n, c)
        else:
            continue
        energy.append(e)

    return pd.concat(energy, axis=1)


###
# ENERGY DEMAND
###


def get_demand_timeseries(n: pypsa.Network) -> pd.DataFrame:
    """Gets timeseries energy demand."""
    return pd.DataFrame(n.loads_t.p.sum(axis=1)).rename(columns={0: "Demand"})


###
# COSTS
###


def get_generator_marginal_costs(
    n: pypsa.Network,
    resample_period: str = "D",
) -> pd.DataFrame:
    """
    Gets generator marginal costs of Units with static MC and units with time
    varying MC.
    """
    df_mc = (
        n.get_switchable_as_dense("Generator", "marginal_cost")
        .loc[n.investment_periods[0]]
        .resample(resample_period)
        .mean()
    )
    df_long = pd.melt(
        df_mc.reset_index(),
        id_vars=["timestep"],
        var_name="Generator",
        value_name="Value",
    )
    df_long["Carrier"] = df_long["Generator"].map(n.generators.carrier)
    return df_long


def get_fuel_costs(n: pypsa.Network) -> pd.DataFrame:
    """
    Gets fuel costs per generator, bus, and carrier.

    Units are $/MWh
    """
    # approximates for 2030
    fixed_voms = {
        "coal": 8.18,
        "oil": 6.42,
        "CCGT": 1.84,
        "OCGT": 6.44,
        "nuclear": 2.47,
    }

    # will return generator level of (fuel_costs / efficiency)
    marginal_costs = n.get_switchable_as_dense("Generator", "marginal_cost").loc[n.investment_periods[0]].T
    marginal_costs = marginal_costs[marginal_costs.index.map(n.generators.carrier).isin(list(fixed_voms))]
    voms = pd.Series(
        index=marginal_costs.index,
        data=marginal_costs.index.map(n.generators.carrier).map(fixed_voms).astype(float).fillna(0),
    ).astype(float)
    marginal_costs = marginal_costs.subtract(voms, axis=0)

    # remove the efficiency cost
    eff = n.get_switchable_as_dense("Generator", "efficiency").loc[n.investment_periods[0]].T
    eff = eff[eff.index.map(n.generators.carrier).isin(list(fixed_voms))]
    fuel_costs = marginal_costs.mul(eff, axis=0)

    # add indices for bus and carrier
    fuel_costs = fuel_costs.reset_index()
    fuel_costs["bus"] = fuel_costs.Generator.map(n.generators.bus)
    fuel_costs["carrier"] = fuel_costs.Generator.map(n.generators.carrier)
    fuel_costs = fuel_costs.groupby(["carrier", "bus", "Generator"]).sum().T

    fuel_costs.index = pd.to_datetime(fuel_costs.index)
    return fuel_costs.T


###
# EMISSIONS
###


def get_node_carrier_emissions_timeseries(n: pypsa.Network) -> pd.DataFrame:
    """Gets timeseries emissions by bus and carrier."""
    energy = get_primary_energy_use(n)
    co2 = n.carriers[["nice_name", "co2_emissions"]].reset_index().set_index("nice_name")[["co2_emissions"]].squeeze()
    return energy.T.mul(n.snapshot_weightings.objective, axis=0).T.mul(co2, level="carrier", axis=0)


def get_node_emissions_timeseries(n: pypsa.Network) -> pd.DataFrame:
    """Gets timeseries emissions per node."""
    return (
        get_node_carrier_emissions_timeseries(n)
        .droplevel("carrier")
        # .reset_index() fix for multi-horizon
        .groupby("bus")
        .sum()
        .T
    )


def get_tech_emissions_timeseries(n: pypsa.Network) -> pd.DataFrame:
    """Gets timeseries emissions per technology."""
    return (
        get_node_carrier_emissions_timeseries(n)
        .droplevel("bus")
        # .reset_index()
        .groupby("carrier")
        .sum()
        .T
    )
