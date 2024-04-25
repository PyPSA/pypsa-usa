"""
Calcualtes summary files.

Adapted from PyPSA-Eur summary statistics reporting script
 - https://github.com/PyPSA/pypsa-eur/blob/master/scripts/make_summary.py
"""

import logging

import pandas as pd
import pypsa
from _helpers import configure_logging
from pypsa.statistics import StatisticsAccessor, get_bus_and_carrier

logger = logging.getLogger(__name__)


###
# ENERGY SUPLPY
###


def get_primary_energy_use(n: pypsa.Network) -> pd.DataFrame:
    """
    Gets timeseries primary energy use by bus and carrier.
    """

    link_energy_use = (
        StatisticsAccessor(n)
        .withdrawal(
            comps=["Link", "Store", "StorageUnit"],
            aggregate_time=False,
            groupby=get_bus_and_carrier,
        )
        .droplevel("component")
    )

    gen_dispatch = (
        StatisticsAccessor(n)
        .supply(
            aggregate_time=False,
            comps=["Generator"],
            groupby=pypsa.statistics.get_name_bus_and_carrier,
        )
        .droplevel("component")
    )
    gen_eff = n.get_switchable_as_dense("Generator", "efficiency")

    gen_energy_use = gen_dispatch.T.mul(1 / gen_eff, axis=0, level="name").T.droplevel(
        "name",
    )

    return (
        pd.concat([gen_energy_use, link_energy_use])
        .reset_index()
        .groupby(["bus", "carrier"])
        .sum()
    )


def get_energy_total(n: pypsa.Network):
    """
    Gets energy production totals.
    """

    def _get_energy_one_port(n: pypsa.Network, c: str) -> pd.DataFrame:
        return (
            c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0)
            .sum()
            .multiply(c.df.sign)
            .groupby(c.df.carrier)
            .sum()
        )

    def _get_energy_multi_port(n: pypsa.Network, c: str) -> pd.DataFrame:
        c_energies = pd.Series(0.0, c.df.carrier.unique())
        for port in [col[3:] for col in c.df.columns if col[:3] == "bus"]:
            totals = (
                c.pnl["p" + port]
                .multiply(n.snapshot_weightings.generators, axis=0)
                .sum()
            )
            # remove values where bus is missing (bug in nomopyomo)
            no_bus = c.df.index[c.df["bus" + port] == ""]
            totals.loc[no_bus] = float(
                n.component_attrs[c.name].loc["p" + port, "default"],
            )
            c_energies -= totals.groupby(c.df.carrier).sum()
        return c_energies

    energy = []
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        if c.name in ("Generator", "StorageUnit", "Store"):
            e = _get_energy_one_port(n, c)
        elif c.name in ("Link"):
            e = _get_energy_multi_port(n, c)
        else:
            continue
        energy.append(e)

    return pd.concat(energy, axis=1)


def get_energy_timeseries(n: pypsa.Network) -> pd.DataFrame:
    """
    Gets timeseries energy production.
    """

    def _get_energy_one_port(n: pypsa.Network, c: str) -> pd.DataFrame:
        return (
            c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0)
            .multiply(c.df.sign)
            .T.groupby(c.df.carrier)
            .sum()
            .T
        )

    def _get_energy_multi_port(n: pypsa.Network, c: str) -> pd.DataFrame:
        c_energies = pd.DataFrame(
            index=n.snapshots,
            columns=c.df.carrier.unique(),
        ).fillna(0)
        for port in [col[3:] for col in c.df.columns if col[:3] == "bus"]:
            if port == "0":  # only track flow in one direction
                continue
            totals = c.pnl["p" + port].multiply(
                n.snapshot_weightings.generators,
                axis=0,
            )
            # remove values where bus is missing (bug in nomopyomo)
            no_bus = c.df.index[c.df["bus" + port] == ""]
            totals.loc[no_bus] = float(
                n.component_attrs[c.name].loc["p" + port, "default"],
            )
            c_energies -= totals.T.groupby(c.df.carrier).sum().T
        return c_energies

    energy = []
    for c in n.iterate_components(n.one_port_components | n.branch_components):
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
    """
    Gets timeseries energy demand.
    """
    return pd.DataFrame(n.loads_t.p.sum(1)).rename(columns={0: "Demand"})


def get_demand_base(n: pypsa.Network) -> pd.DataFrame:
    """
    Gets Nodal Sum of Demand.
    """
    return pd.DataFrame(n.loads_t.p).rename(columns=n.loads.bus).sum(0)


###
# ENERGY CAPACITY
###


def get_capacity_base(n: pypsa.Network) -> pd.DataFrame:
    """
    Gets starting capacities.

    NOTE: Link capacities are grouped by both bus0 and bus1!!
    It is up to the user to filter this by bus on the returned dataframe
    """
    totals = []
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        if c.name in ("Generator", "StorageUnit"):
            totals.append((c.df.p_nom).groupby(by=[c.df.bus, c.df.carrier]).sum())
        elif c.name == "Link":
            totals.append(
                (c.df.p_nom)
                .groupby(by=[c.df.bus0, c.df.carrier])
                .sum()
                .rename_axis(index={"bus0": "bus"}),
            ),
            totals.append(
                (c.df.p_nom)
                .groupby(by=[c.df.bus1, c.df.carrier])
                .sum()
                .rename_axis(index={"bus1": "bus"}),
            )
    return pd.concat(totals)


def get_capacity_brownfield(
    n: pypsa.Network,
    retirement_method="economic",
) -> pd.DataFrame:
    """
    Gets optimal brownfield pnom capacity.

    NOTE: Link capacities are grouped by both bus0 and bus1!!
    It is up to the user to filter this by bus on the returned dataframe
    """

    def _technical_retirement(c: pypsa.components.Component) -> pd.DataFrame:
        if c.name == "Link":
            return pd.concat(
                [
                    (c.df.p_nom_opt)
                    .groupby(by=[c.df.bus0, c.df.carrier])
                    .sum()
                    .rename_axis(index={"bus0": "bus"}),
                    (c.df.p_nom_opt)
                    .groupby(by=[c.df.bus1, c.df.carrier])
                    .sum()
                    .rename_axis(index={"bus1": "bus"}),
                ],
            )
        else:
            return (c.df.p_nom_opt).groupby(by=[c.df.bus, c.df.carrier]).sum()

    def _economic_retirement(c: str) -> pd.DataFrame:
        if c.name == "Link":
            return pd.concat(
                [
                    (c.df.p_nom_opt)
                    .groupby(by=[c.df.bus0, c.df.carrier])
                    .sum()
                    .rename_axis(index={"bus0": "bus"}),
                    (c.df.p_nom_opt)
                    .groupby(by=[c.df.bus1, c.df.carrier])
                    .sum()
                    .rename_axis(index={"bus1": "bus"}),
                ],
            )
        else:
            return (c.df.p_nom_opt).groupby(by=[c.df.bus, c.df.carrier]).sum()

    totals = []
    if retirement_method == "technical":
        if c.name in ("Generator", "StorageUnit", "Link"):
            totals.append(_technical_retirement(c))
        return pd.concat(totals)
    elif retirement_method == "economic":
        for c in n.iterate_components(n.one_port_components | n.branch_components):
            if c.name in ("Generator", "StorageUnit", "Link"):
                totals.append(_economic_retirement(c))
        return pd.concat(totals)
    else:
        logger.error(
            f"Retirement method must be one of 'technical' or 'economic'. Recieved {retirement_method}.",
        )
        raise NotImplementedError


###
# COSTS
###


def get_capital_costs(n: pypsa.Network) -> pd.DataFrame:
    return n.statistics.capex() - n.statistics.installed_capex()


def get_generator_marginal_costs(
    n: pypsa.Network,
    resample_period: str = "d",
) -> pd.DataFrame:
    """
    Gets generator marginal costs of Units with static MC and units with time
    varying MC.
    """
    df_mc = (
        n.get_switchable_as_dense("Generator", "marginal_cost")
        .resample(resample_period)
        .mean()
    )
    df_long = pd.melt(
        df_mc.reset_index(),
        id_vars=["snapshot"],
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
    marginal_costs = n.get_switchable_as_dense("Generator", "marginal_cost").T
    marginal_costs = marginal_costs[
        marginal_costs.index.map(n.generators.carrier).isin(list(fixed_voms))
    ]
    voms = pd.Series(
        index=marginal_costs.index,
        data=marginal_costs.index.map(n.generators.carrier).map(fixed_voms).fillna(0),
    ).astype(float)
    marginal_costs = marginal_costs.subtract(voms, axis=0)

    # remove the efficiency cost
    eff = n.get_switchable_as_dense("Generator", "efficiency").T
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
    """
    Gets timeseries emissions by bus and carrier.
    """

    energy = get_primary_energy_use(n)
    co2 = (
        n.carriers[["nice_name", "co2_emissions"]]
        .reset_index()
        .set_index("nice_name")[["co2_emissions"]]
        .squeeze()
    )
    return energy.mul(co2, level="carrier", axis=0)


def get_node_emissions_timeseries(n: pypsa.Network) -> pd.DataFrame:
    """
    Gets timeseries emissions per node.
    """

    return (
        get_node_carrier_emissions_timeseries(n)
        .droplevel("carrier")
        .reset_index()
        .groupby("bus")
        .sum()
        .T
    )


def get_tech_emissions_timeseries(n: pypsa.Network) -> pd.DataFrame:
    """
    Gets timeseries emissions per technology.
    """

    return (
        get_node_carrier_emissions_timeseries(n)
        .droplevel("bus")
        .reset_index()
        .groupby("carrier")
        .sum()
        .T
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_figures",
            interconnect="texas",
            clusters=40,
            ll="v1.25",
            opts="Co2L1.25",
            sector="E",
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    # get_energy_total(n)
    # get_energy_timeseries(n)
