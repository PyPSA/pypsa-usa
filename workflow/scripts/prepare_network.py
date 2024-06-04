# SPDX-FileCopyrightText: : 2017-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8
"""
Prepare PyPSA network for solving according to :ref:`opts` and :ref:`ll`, such
as.

- adding an annual **limit** of carbon-dioxide emissions,
- adding an exogenous **price** per tonne emissions of carbon-dioxide (or other kinds),
- setting an **N-1 security margin** factor for transmission line capacities,
- specifying an expansion limit on the **cost** of transmission expansion,
- specifying an expansion limit on the **volume** of transmission expansion, and
- reducing the **temporal** resolution by averaging over multiple hours
  or segmenting time series into chunks of varying lengths using ``tsam``.

Relevant Settings
-----------------

.. code:: yaml

    costs:
        year:
        version:
        fill_values:
        emission_prices:
        marginal_cost:
        capital_cost:

    electricity:
        co2limit:
        max_hours:

.. seealso::
    Documentation of the configuration file ``config/config.yaml`` at
    :ref:`costs_cf`, :ref:`electricity_cf`

Inputs
------

- ``resources/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.
- ``networks/elec_s{simpl}_{clusters}.nc``: confer :ref:`cluster`

Outputs
-------

- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Complete PyPSA network that will be handed to the ``solve_network`` rule.

Description
-----------

.. tip::
    The rule :mod:`prepare_elec_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`prepare_network`.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
from _helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from add_electricity import load_costs, update_transmission_costs
from pypsa.descriptors import expand_series

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def get_social_discount(t, r=0.01):
    """
    Calculate for a given time t and social discount rate r [per unit] the
    social discount.
    """
    return 1 / (1 + r) ** t


def get_investment_weighting(time_weighting, r=0.01):
    """
    Define cost weighting.

    Returns cost weightings depending on the the time_weighting
    (pd.Series) and the social discountrate r
    """
    end = time_weighting.cumsum()
    start = time_weighting.cumsum().shift().fillna(0)
    return pd.concat([start, end], axis=1).apply(
        lambda x: sum(
            get_social_discount(t, r) for t in range(int(x.iloc[0]), int(x.iloc[1]))
        ),
        axis=1,
    )


def add_co2limit(n, co2limit, Nyears=1.0):
    n.add(
        "GlobalConstraint",
        "CO2Limit",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=co2limit * Nyears,
    )


def add_gaslimit(n, gaslimit, Nyears=1.0):
    sel = n.carriers.index.intersection(["OCGT", "CCGT", "CHP"])
    n.carriers.loc[sel, "gas_usage"] = 1.0

    n.add(
        "GlobalConstraint",
        "GasLimit",
        carrier_attribute="gas_usage",
        sense="<=",
        constant=gaslimit * Nyears,
    )


def add_emission_prices(n, emission_prices={"co2": 0.0}, exclude_co2=False):
    if exclude_co2:
        emission_prices.pop("co2")
    ep = (
        pd.Series(emission_prices).rename(lambda x: x + "_emissions")
        * n.carriers.filter(like="_emissions")
    ).sum(axis=1)
    gen_ep = n.generators.carrier.map(ep) / n.generators.efficiency
    n.generators["marginal_cost"] += gen_ep
    n.generators_t["marginal_cost"] += gen_ep[n.generators_t["marginal_cost"].columns]
    su_ep = n.storage_units.carrier.map(ep) / n.storage_units.efficiency_dispatch
    n.storage_units["marginal_cost"] += su_ep


def set_line_s_max_pu(n, s_max_pu=0.7):
    n.lines["s_max_pu"] = s_max_pu
    logger.info(f"N-1 security margin of lines set to {s_max_pu}")


def set_transmission_limit(n, ll_type, factor, costs, Nyears=1):
    links_dc_b = n.links.carrier == "DC" if not n.links.empty else pd.Series()

    _lines_s_nom = (
        np.sqrt(3)
        * n.lines.type.map(n.line_types.i_nom)
        * n.lines.num_parallel
        * n.lines.bus0.map(n.buses.v_nom)
    )
    lines_s_nom = n.lines.s_nom.where(n.lines.type == "", _lines_s_nom)

    col = "capital_cost" if ll_type == "c" else "length"
    ref = (
        lines_s_nom @ n.lines[col]
        + n.links.loc[links_dc_b, "p_nom"] @ n.links.loc[links_dc_b, col]
    )

    update_transmission_costs(n, costs)

    if factor == "opt" or float(factor) > 1.0:
        n.lines["s_nom_min"] = lines_s_nom
        n.lines["s_nom_extendable"] = True

        n.links.loc[links_dc_b, "p_nom_min"] = n.links.loc[links_dc_b, "p_nom"]
        n.links.loc[links_dc_b, "p_nom_extendable"] = True

    if factor != "opt":
        con_type = "expansion_cost" if ll_type == "c" else "volume_expansion"
        rhs = float(factor) * ref
        n.add(
            "GlobalConstraint",
            f"l{ll_type}_limit",
            type=f"transmission_{con_type}_limit",
            sense="<=",
            constant=rhs,
            carrier_attribute="AC, DC",
        )

    return n


def average_every_nhours(n, offset):
    logger.info(f"Resampling the network to {offset}")
    m = n.copy(with_time=False)

    snapshot_weightings = n.snapshot_weightings.resample(offset).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.pnl.items():
            if not df.empty:
                pnl[k] = df.resample(offset).mean()

    return m


def apply_time_segmentation(n, segments, solver_name="cbc"):
    try:
        import tsam.timeseriesaggregation as tsam
    except ImportError:
        raise ModuleNotFoundError(
            "Optional dependency 'tsam' not found." "Install via 'pip install tsam'",
        )

    # get all time-dependent data
    columns = pd.MultiIndex.from_tuples([], names=["component", "key", "asset"])
    raw = pd.DataFrame(index=n.snapshots, columns=columns)
    for c in n.iterate_components():
        for attr, pnl in c.pnl.items():
            # exclude e_min_pu which is used for SOC of EVs in the morning
            if not pnl.empty and attr != "e_min_pu":
                df = pnl.copy()
                df.columns = pd.MultiIndex.from_product([[c.name], [attr], df.columns])
                raw = pd.concat([raw, df], axis=1)
    raw = raw.dropna(axis=1)
    sn_weightings = {}

    for year in raw.index.levels[0]:
        logger.info(f"Find representative snapshots for {year}.")
        raw_t = raw.loc[year]
        # normalise all time-dependent data
        annual_max = raw_t.max().replace(0, 1)
        raw_t = raw_t.div(annual_max, level=0)
        # get representative segments
        agg = tsam.TimeSeriesAggregation(
            raw_t,
            hoursPerPeriod=len(raw_t),
            noTypicalPeriods=1,
            noSegments=int(segments),
            segmentation=True,
            solver=solver_name,
        )
        segmented = agg.createTypicalPeriods()

        weightings = segmented.index.get_level_values("Segment Duration")
        offsets = np.insert(np.cumsum(weightings[:-1]), 0, 0)
        timesteps = [raw_t.index[0] + pd.Timedelta(f"{offset}h") for offset in offsets]
        snapshots = pd.DatetimeIndex(timesteps)
        sn_weightings[year] = pd.Series(
            weightings,
            index=snapshots,
            name="weightings",
            dtype="float64",
        )

    sn_weightings = pd.concat(sn_weightings)
    n.set_snapshots(sn_weightings.index)
    n.snapshot_weightings = n.snapshot_weightings.mul(sn_weightings, axis=0)

    return n


def enforce_autarky(n, only_crossborder=False):
    if only_crossborder:
        lines_rm = n.lines.loc[
            n.lines.bus0.map(n.buses.country) != n.lines.bus1.map(n.buses.country)
        ].index
        links_rm = n.links.loc[
            n.links.bus0.map(n.buses.country) != n.links.bus1.map(n.buses.country)
        ].index
    else:
        lines_rm = n.lines.index
        links_rm = n.links.loc[n.links.carrier == "DC"].index
    n.mremove("Line", lines_rm)
    n.mremove("Link", links_rm)


def set_line_nom_max(
    n,
    s_nom_max_set=np.inf,
    p_nom_max_set=np.inf,
    s_nom_max_ext=np.inf,
    p_nom_max_ext=np.inf,
):
    if np.isfinite(s_nom_max_ext) and s_nom_max_ext > 0:
        logger.info(f"Limiting line extensions to {s_nom_max_ext} MW")
        n.lines["s_nom_max"] = n.lines["s_nom"] + s_nom_max_ext

    if np.isfinite(p_nom_max_ext) and p_nom_max_ext > 0:
        logger.info(f"Limiting link extensions to {p_nom_max_ext} MW")
        hvdc = n.links.index[n.links.carrier == "DC"]
        n.links.loc[hvdc, "p_nom_max"] = n.links.loc[hvdc, "p_nom"] + p_nom_max_ext

    n.lines["s_nom_max"] = n.lines.s_nom_max.clip(upper=s_nom_max_set)
    n.links["p_nom_max"] = n.links.p_nom_max.clip(upper=p_nom_max_set)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_network",
            simpl="",
            clusters="36",
            interconnect="western",
            ll="v1.0",
            opts="REM-1000SEG",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    n = pypsa.Network(snakemake.input[0])
    Nyears = n.snapshot_weightings.loc[n.investment_periods[0]].objective.sum() / 8760.0
    costs = load_costs(
        snakemake.input.tech_costs,
        snakemake.params.costs,
        snakemake.params.max_hours,
        Nyears,
    )

    # Set Investment Period Year Weightings
    # 'fillna(1)' needed if only one period
    inv_per_time_weight = (
        n.investment_periods.to_series().diff().shift(-1).ffill().fillna(1)
    )
    n.investment_period_weightings["years"] = inv_per_time_weight
    # set Investment Period Objective weightings
    social_discountrate = snakemake.params.costs["social_discount_rate"]
    objective_w = get_investment_weighting(
        n.investment_period_weightings["years"],
        social_discountrate,
    )
    n.investment_period_weightings["objective"] = objective_w

    set_line_s_max_pu(n, snakemake.params.lines["s_max_pu"])

    # temporal averaging
    time_resolution = snakemake.params.time_resolution
    is_string = isinstance(time_resolution, str)
    if is_string and time_resolution.lower().endswith("h"):
        n = average_every_nhours(n, time_resolution)

    # segments with package tsam

    if is_string and time_resolution.lower().endswith("seg"):
        solver_name = snakemake.config["solving"]["solver"]["name"]
        segments = int(time_resolution.lower().replace("seg", ""))
        n = apply_time_segmentation(n, segments, solver_name)

    if snakemake.params.co2limit_enable:
        add_co2limit(n, snakemake.params.co2limit, Nyears)

    if snakemake.params.gaslimit_enable:
        add_gaslimit(n, snakemake.params.gaslimit, Nyears)

    emission_prices = snakemake.params.costs["emission_prices"]
    if emission_prices["enable"]:
        add_emission_prices(
            n,
            dict(co2=snakemake.params.costs["emission_prices"]["co2"]),
        )

    ll_type, factor = snakemake.wildcards.ll[0], snakemake.wildcards.ll[1:]
    set_transmission_limit(n, ll_type, factor, costs, Nyears)

    set_line_nom_max(
        n,
        s_nom_max_set=snakemake.params.lines.get("s_nom_max", np.inf),
        p_nom_max_set=snakemake.params.links.get("p_nom_max", np.inf),
        s_nom_max_ext=snakemake.params.lines.get("max_extension", np.inf),
        p_nom_max_ext=snakemake.params.links.get("max_extension", np.inf),
    )

    if snakemake.params.autarky["enable"]:
        only_crossborder = snakemake.params.autarky["by_country"]
        enforce_autarky(n, only_crossborder=only_crossborder)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
