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
"""

import logging

import numpy as np
import pandas as pd
import pypsa
from _helpers import (
    configure_logging,
    is_transport_model,
    set_scenario_config,
    update_config_from_wildcards,
)

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
        lambda x: sum(get_social_discount(t, r) for t in range(int(x.iloc[0]), int(x.iloc[1]))),
        axis=1,
    )


def add_co2limit(n, co2limit, num_years=1.0):
    n.add(
        "GlobalConstraint",
        "CO2Limit",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=co2limit * num_years,
    )


def add_gaslimit(n, gaslimit, num_years=1.0):
    sel = n.carriers.index.intersection(["OCGT", "CCGT", "CHP"])
    n.carriers.loc[sel, "gas_usage"] = 1.0

    n.add(
        "GlobalConstraint",
        "GasLimit",
        carrier_attribute="gas_usage",
        sense="<=",
        constant=gaslimit * num_years,
    )


def add_emission_prices(n, emission_prices={"co2": 0.0}, exclude_co2=False):
    if exclude_co2:
        emission_prices.pop("co2")
    ep = (pd.Series(emission_prices).rename(lambda x: x + "_emissions") * n.carriers.filter(like="_emissions")).sum(
        axis=1,
    )
    gen_ep = n.generators.carrier.map(ep) / n.generators.efficiency
    n.generators["marginal_cost"] += gen_ep
    n.generators_t["marginal_cost"] += gen_ep[n.generators_t["marginal_cost"].columns]
    su_ep = n.storage_units.carrier.map(ep) / n.storage_units.efficiency_dispatch
    n.storage_units["marginal_cost"] += su_ep


def set_line_s_max_pu(n, transport_model, s_max_pu=0.7):
    if not transport_model:
        logger.info(f"N-1 security margin of lines set to {s_max_pu}")
        n.lines["s_max_pu"] = s_max_pu


def set_transmission_limit(n, ll_type, factor):
    """
    Set transmission limits according to ll wildcard.

    For transport models we track expandable AC links via their carrier
    initially, then re-name them to AC. We don't set expandability
    earlier in the model to avoid rebuilding the network multiple times
    from earlier stages when testing sensitivities to the transmission
    limits wildcards.
    """
    logger.info(f"Setting transmission limit for {ll_type} to {factor}")

    dc_links = n.links.carrier == "DC" if not n.links.empty else pd.Series()
    ac_links_exp = n.links.carrier == "AC_exp" if not n.links.empty else pd.Series()
    ac_links_existing = n.links.carrier == "AC" if not n.links.empty else pd.Series()

    n.links.loc[ac_links_exp, "carrier"] = "AC"  # rename AC_exp carrier to AC

    lines_s_nom = n.lines.s_nom
    col = "capital_cost" if ll_type == "c" else "length"
    ref = (
        lines_s_nom @ n.lines[col]
        + n.links.loc[dc_links, "p_nom"] @ n.links.loc[dc_links, col]
        + n.links.loc[ac_links_existing, "p_nom"] @ n.links.loc[ac_links_existing, col]
    )
    ref_dc = n.links.loc[dc_links, "p_nom"] @ n.links.loc[dc_links, col]

    if factor == "opt" or float(factor) > 1.0:
        # if opt allows expansion set respective lines/links to extendable
        # all links prior to this point have extendable set to false
        n.lines["s_nom_min"] = lines_s_nom
        n.lines["s_nom_extendable"] = True

        n.links.loc[dc_links, "p_nom_min"] = n.links.loc[dc_links, "p_nom"]
        n.links.loc[dc_links, "p_nom_extendable"] = True

        n.links.loc[ac_links_exp, "p_nom_min"] = n.links.loc[ac_links_exp, "p_nom"]
        n.links.loc[ac_links_exp, "p_nom_extendable"] = True
    if factor != "opt":
        con_type = "expansion_cost" if ll_type == "c" else "volume_expansion"
        if transport_model:
            # Transport models have links split to existing and non-existing
            # The global constraint applies to the p_nom_opt of extendable capacity
            # thus we must only include the 'new' transmission capacity as reference
            rhs = ((float(factor) - 1.0) * ref) + ref_dc
        else:
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

    def resample_multi_index(df, offset, func):
        sw = []
        for year in df.index.levels[0]:
            sw_ = df.loc[year].resample(offset).apply(func)
            sns = sw_.index
            sns = sns[~((sns.month == 2) & (sns.day == 29))]
            sw_ = sw_.loc[sns]
            sw.append(sw_)
        snapshot_weightings = pd.concat(sw)
        snapshot_weightings.index = pd.MultiIndex.from_arrays(
            [snapshot_weightings.index.year, snapshot_weightings.index],
            names=["period", "timestep"],
        )
        return snapshot_weightings

    snapshot_weightings = resample_multi_index(n.snapshot_weightings, offset, "sum")
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings
    m.investment_periods = n.investment_periods

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.pnl.items():
            if not df.empty:
                pnl[k] = resample_multi_index(df, offset, "mean")
    return m


def is_leap_year(year: int) -> bool:
    """Check if a given year is a leap year."""
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return True
    else:
        return False


def apply_time_segmentation(n, segments, solver_name="cbc"):
    try:
        import tsam.timeseriesaggregation as tsam
    except ImportError:
        raise ModuleNotFoundError(
            "Optional dependency 'tsam' not found.Install via 'pip install tsam'",
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

        # hack to get around that TSAM will add leap days in
        if is_leap_year(year):
            raw_t.index = raw_t.index.map(lambda x: x.replace(year=year + 1))

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

        if is_leap_year(year):
            snapshots = snapshots.map(lambda x: x.replace(year=year))

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
        lines_rm = n.lines.loc[n.lines.bus0.map(n.buses.country) != n.lines.bus1.map(n.buses.country)].index
        links_rm = n.links.loc[n.links.bus0.map(n.buses.country) != n.links.bus1.map(n.buses.country)].index
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
            # simpl="",
            clusters="100",
            interconnect="western",
            ll="v1.0",
            opts="500SEG",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)
    params = snakemake.params
    transport_model = is_transport_model(params.transmission_network)

    n = pypsa.Network(snakemake.input[0])
    num_years = n.snapshot_weightings.loc[n.investment_periods[0]].objective.sum() / 8760.0
    costs = pd.read_csv(snakemake.input.tech_costs)
    costs = costs.pivot(index="pypsa-name", columns="parameter", values="value")
    # Set Investment Period Year Weightings
    # 'fillna(1)' needed if only one period
    inv_per_time_weight = n.investment_periods.to_series().diff().shift(-1).ffill().fillna(1)
    n.investment_period_weightings["years"] = inv_per_time_weight
    # set Investment Period Objective weightings
    social_discountrate = params.costs["social_discount_rate"]
    objective_w = get_investment_weighting(
        n.investment_period_weightings["years"],
        social_discountrate,
    )
    n.investment_period_weightings["objective"] = objective_w

    set_line_s_max_pu(n, transport_model, params.lines["s_max_pu"])

    # temporal averaging
    time_resolution = params.time_resolution
    is_string = isinstance(time_resolution, str)
    if is_string and time_resolution.lower().endswith("h"):
        n = average_every_nhours(n, time_resolution)

    # segments with package tsam

    if is_string and time_resolution.lower().endswith("seg"):
        solver_name = snakemake.config["solving"]["solver"]["name"]
        segments = int(time_resolution.lower().replace("seg", ""))
        n = apply_time_segmentation(n, segments, solver_name)

    if params.co2limit_enable:
        add_co2limit(n, params.co2limit, num_years)

    if params.gaslimit_enable:
        add_gaslimit(n, params.gaslimit, num_years)

    emission_prices = params.costs["emission_prices"]
    if emission_prices["enable"]:
        add_emission_prices(
            n,
            dict(co2=params.costs["emission_prices"]["co2"]),
        )

    ll_type, factor = snakemake.wildcards.ll[0], snakemake.wildcards.ll[1:]
    set_transmission_limit(n, ll_type, factor)

    set_line_nom_max(
        n,
        s_nom_max_set=params.lines.get("s_nom_max", np.inf),
        p_nom_max_set=params.links.get("p_nom_max", np.inf),
        s_nom_max_ext=params.lines.get("max_extension", np.inf),
        p_nom_max_ext=params.links.get("max_extension", np.inf),
    )

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
