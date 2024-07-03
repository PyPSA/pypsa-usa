"""
Plots Sector Coupling Statistics.
"""

from math import ceil

import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import seaborn as sns

###
# HELPERS
###


def get_load_name_per_sector(sector: str) -> list[str]:
    match sector:
        case "res" | "com":
            return ["elec", "urban-heat", "rural-heat", "cool"]
        case "ind":
            return ["elec", "heat"]
        case "trn":
            vehicles = ("lgt", "med", "hvy", "bus")
            fuels = ("elec", "lpg")
            return [f"{f}-{v}" for v in vehicles for f in fuels]
        case _:
            raise NotImplementedError


###
# GETTERS
###


def get_load_per_sector_per_fuel(n: pypsa.Network, sector: str, fuel: str, period: int):
    """
    Time series load per bus per fuel per sector.
    """
    loads = n.loads[
        (n.loads.carrier.str.startswith(sector)) & (n.loads.carrier.str.endswith(fuel))
    ]
    return n.loads_t.p[loads.index].loc[period]


def get_hp_cop(n: pypsa.Network) -> pd.DataFrame:
    """
    Com and res hps have the same cop.
    """
    cops = n.links_t.efficiency
    ashp = cops[[x for x in cops.columns if x.endswith("res-urban-ashp")]]
    ashp = ashp.rename(
        columns={x: x.replace("res-urban-ashp", "ashp") for x in ashp.columns},
    )
    gshp = cops[[x for x in cops.columns if x.endswith("res-rural-gshp")]]
    gshp = gshp.rename(
        columns={x: x.replace("res-rural-gshp", "gshp") for x in gshp.columns},
    )
    return ashp.join(gshp)


def get_sector_production_timeseries(n: pypsa.Network, sector: str) -> pd.DataFrame:
    """
    Gets timeseries production to meet sectoral demand.
    """
    supply = n.statistics.supply("Link", nice_names=False, aggregate_time=False).T
    return supply[[x for x in supply.columns if x.startswith(sector)]]


def get_capacity_per_link_per_node(
    n: pypsa.Network,
    sector: str,
    include_elec: bool = False,
) -> pd.Series:
    if include_elec:
        df = n.links[n.links.carrier.str.startswith(sector)]
    else:
        df = n.links[
            (n.links.carrier.str.startswith(sector))
            & ~(n.links.carrier.str.endswith("elec-infra"))
        ]
    df = df[["carrier", "p_nom_opt"]]
    df["node"] = df.index.map(lambda x: x.split(f" {sector}-")[0])
    df["carrier"] = df.carrier.map(lambda x: x.split(f"{sector}-")[1])
    return df.reset_index(drop=True).groupby(["node", "carrier"]).sum().squeeze()


def get_total_capacity_per_node(
    n: pypsa.Network,
    sector: str,
    include_elec: bool = False,
) -> pd.Series:
    if include_elec:
        df = n.links[n.links.carrier.str.startswith(sector)]
    else:
        df = n.links[
            (n.links.carrier.str.startswith(sector))
            & ~(n.links.carrier.str.endswith("elec-infra"))
        ]
    df = df[["p_nom_opt"]]
    df["node"] = df.index.map(lambda x: x.split(f" {sector}-")[0])
    return df.reset_index(drop=True).groupby(["node"]).sum().squeeze()


def get_capacity_percentage_per_node(
    n: pypsa.Network,
    sector: str,
    include_elec: bool = False,
) -> pd.DataFrame:
    total = get_total_capacity_per_node(n, sector, include_elec)
    df = get_capacity_per_link_per_node(n, sector, include_elec).to_frame()
    df["total"] = df.index.get_level_values("node").map(total)
    df["percentage"] = (df.p_nom_opt / df.total).round(4) * 100
    return df


def get_sector_max_production_timeseries(n: pypsa.Network, sector: str) -> pd.DataFrame:
    """
    Max production timeseries at a carrier level.
    """
    eff = n.get_switchable_as_dense("Link", "efficiency")
    eff = eff[[x for x in eff.columns if f"{sector}-" in x]]
    cap = n.links[n.links.index.str.contains(f"{sector}-")].p_nom_opt
    return eff.mul(cap)


def get_load_factor_timeseries(
    n: pypsa.Network,
    sector: str,
    include_elec: bool = False,
) -> pd.DataFrame:
    max_prod = get_sector_max_production_timeseries(n, sector)
    act_prod = get_sector_production_timeseries(n, sector)

    max_prod = (
        max_prod.rename(columns={x: x.split(f"{sector}-")[1] for x in max_prod.columns})
        .T.groupby(level=0)
        .sum()
        .T
    )
    act_prod = (
        act_prod.rename(columns={x: x.split(f"{sector}-")[1] for x in act_prod.columns})
        .T.groupby(level=0)
        .sum()
        .T
    )

    lf = act_prod.div(max_prod).mul(100).round(3)

    if include_elec:
        return lf
    else:
        return lf[[x for x in lf.columns if not x.endswith("-infra")]]


###
# PLOTTERS
###


def plot_load_per_sector(
    n: pypsa.Network,
    sector: str,
    sharey: bool = True,
    log: bool = True,
) -> tuple:
    """
    Load per bus per sector per fuel.
    """

    fuels = get_load_name_per_sector(sector)
    investment_period = n.investment_periods[0]

    nrows = ceil(len(fuels) / 2)

    fig, axs = plt.subplots(
        ncols=2,
        nrows=nrows,
        figsize=(14, 5 * nrows),
        sharey=sharey,
    )

    ylabel = "VMT" if sector == "trn" else "MW"

    row = 0
    col = 0

    for i, fuel in enumerate(fuels):

        row = i // 2
        col = i % 2

        df = get_load_per_sector_per_fuel(n, sector, fuel, investment_period)
        avg = df.mean(axis=1)

        palette = sns.color_palette(["lightgray"], df.shape[1])

        if nrows > 1:

            sns.lineplot(
                df,
                color="lightgray",
                legend=False,
                palette=palette,
                ax=axs[row, col],
            )
            sns.lineplot(avg, ax=axs[row, col])

            axs[row, col].set_xlabel("")
            axs[row, col].set_ylabel(ylabel)
            axs[row, col].set_title(f"{fuel} load")

            if log:
                axs[row, col].set(yscale="log")

        else:

            sns.lineplot(
                df,
                color="lightgray",
                legend=False,
                palette=palette,
                ax=axs[i],
            )
            sns.lineplot(avg, ax=axs[i])

            axs[i].set_xlabel("")
            axs[i].set_ylabel(ylabel)
            axs[i].set_title(f"{fuel} load")

            if log:
                axs[i].set(yscale="log")

    fig.tight_layout()

    return fig, axs


def plot_hp_cop(n: pypsa.Network) -> tuple:
    """
    Plots gshp and ashp cops.
    """

    investment_period = n.investment_periods[0]

    cops = get_hp_cop(n).loc[investment_period]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharey=True)

    for i, hp in enumerate(["ashp", "gshp"]):

        df = cops[[x for x in cops if x.endswith(hp)]]
        avg = df.mean(axis=1)

        palette = sns.color_palette(["lightgray"], df.shape[1])

        sns.lineplot(df, color="lightgray", legend=False, palette=palette, ax=axs[i])
        sns.lineplot(avg, ax=axs[i])

        axs[i].set_xlabel("")
        axs[i].set_ylabel("COP")
        axs[i].set_title(f"{hp}")

    fig.tight_layout()

    return fig, axs


def plot_sector_production_timeseries(n: pypsa.Network, sharey: bool = True) -> tuple:
    """
    Plots timeseries production.
    """

    investment_period = n.investment_periods[0]

    sectors = ("res", "com", "ind", "trn")

    nrows = ceil(len(sectors) / 2)

    fig, axs = plt.subplots(
        ncols=2,
        nrows=nrows,
        figsize=(14, 5 * nrows),
        sharey=sharey,
    )

    row = 0
    col = 0

    for i, sector in enumerate(sectors):

        row = i // 2
        col = i % 2

        df = get_sector_production_timeseries(n, sector).loc[investment_period]
        df = df.rename(columns={x: x.split(f"{sector}-")[1] for x in df.columns})

        # get rid of vehicle specific VMT demand data
        if sector == "trn":
            df = df[[x for x in df.columns if len(x.split("-")) <= 1]]

        if nrows > 1:

            df.plot.area(ax=axs[row, col])
            axs[row, col].set_xlabel("")
            axs[row, col].set_ylabel("MW")
            axs[row, col].set_title(f"{sector}")

        else:

            df.plot.area(ax=axs[i])
            axs[i].set_xlabel("")
            axs[i].set_ylabel("MW")
            axs[i].set_title(f"{sector}")

    fig.tight_layout()

    return fig, axs


def plot_sector_production(n: pypsa.Network, sharey: bool = True) -> tuple:
    """
    Plots model period production.
    """

    investment_period = n.investment_periods[0]

    sectors = ("res", "com", "ind", "trn")

    nrows = ceil(len(sectors) / 2)

    fig, axs = plt.subplots(
        ncols=2,
        nrows=nrows,
        figsize=(14, 5 * nrows),
        sharey=sharey,
    )

    row = 0
    col = 0

    for i, sector in enumerate(sectors):

        row = i // 2
        col = i % 2

        df = get_sector_production_timeseries(n, sector).loc[investment_period]
        df = df.rename(columns={x: x.split(f"{sector}-")[1] for x in df.columns})

        # get rid of vehicle specific VMT demand data
        if sector == "trn":
            df = df[[x for x in df.columns if len(x.split("-")) <= 1]]

        df = df.sum(axis=0).to_frame(name="value")

        if nrows > 1:

            df.plot.bar(ax=axs[row, col])
            axs[row, col].set_xlabel("")
            axs[row, col].set_ylabel("MWh")
            axs[row, col].set_title(f"{sector}")
            axs[row, col].tick_params(axis="x", labelrotation=45)

        else:

            df.plot.bar(ax=axs[i])
            axs[i].set_xlabel("")
            axs[i].set_ylabel("MW")
            axs[i].set_title(f"{sector}")

    fig.tight_layout()

    return fig, axs


def plot_capacity_percentage_per_node(n: pypsa.Network, sharey: bool = True) -> tuple:
    """
    Plots capacity percentage per node.
    """

    sectors = ("res", "com", "ind")

    nrows = ceil(len(sectors) / 2)

    fig, axs = plt.subplots(
        ncols=2,
        nrows=nrows,
        figsize=(14, 5 * nrows),
        sharey=sharey,
    )

    row = 0
    col = 0

    for i, sector in enumerate(sectors):

        row = i // 2
        col = i % 2

        df = get_capacity_percentage_per_node(n, sector)
        df = df.reset_index()[["node", "carrier", "percentage"]]
        df = df.pivot(columns="carrier", index="node", values="percentage")

        if nrows > 1:

            df.plot(kind="bar", stacked=True, ax=axs[row, col])
            axs[row, col].set_xlabel("")
            axs[row, col].set_ylabel("Percentage (%)")
            axs[row, col].set_title(f"{sector} Capacity")

        else:

            df.plot(kind="bar", stacked=True, ax=axs[i])
            axs[i].set_xlabel("")
            axs[i].set_ylabel("Percentage (%)")
            axs[i].set_title(f"{sector} Capacity")

    fig.tight_layout()

    return fig, axs


def plot_sector_load_factor_timeseries(n: pypsa.Network, sharey: bool = True) -> tuple:
    """
    Plots timeseries of load factor resampled to days.
    """

    investment_period = n.investment_periods[0]

    sectors = ("res", "com", "ind")

    nrows = ceil(len(sectors) / 2)

    fig, axs = plt.subplots(
        ncols=2,
        nrows=nrows,
        figsize=(14, 5 * nrows),
        sharey=sharey,
    )

    row = 0
    col = 0

    for i, sector in enumerate(sectors):

        row = i // 2
        col = i % 2

        df = (
            get_load_factor_timeseries(n, sector)
            .loc[investment_period]
            .resample("d")
            .mean()
            .dropna()
        )

        if nrows > 1:

            df.plot(ax=axs[row, col])
            axs[row, col].set_xlabel("")
            axs[row, col].set_ylabel("Load Factor (%)")
            axs[row, col].set_title(f"{sector}")

        else:

            df.plot(ax=axs[i])
            axs[i].set_xlabel("")
            axs[i].set_ylabel("Load Factor (%)")
            axs[i].set_title(f"{sector}")

    fig.tight_layout()

    return fig, axs


def plot_sector_load_factor_boxplot(n: pypsa.Network, sharey: bool = True) -> tuple:
    """
    Plots boxplot of load factors.
    """

    investment_period = n.investment_periods[0]

    sectors = ("res", "com", "ind")

    nrows = ceil(len(sectors) / 2)

    fig, axs = plt.subplots(
        ncols=2,
        nrows=nrows,
        figsize=(14, 5 * nrows),
        sharey=sharey,
    )

    row = 0
    col = 0

    for i, sector in enumerate(sectors):

        row = i // 2
        col = i % 2

        df = get_load_factor_timeseries(n, sector).loc[investment_period]

        if nrows > 1:

            sns.boxplot(df, ax=axs[row, col])
            axs[row, col].set_xlabel("")
            axs[row, col].set_ylabel("Load Factor (%)")
            axs[row, col].set_title(f"{sector}")
            axs[row, col].tick_params(axis="x", labelrotation=45)

        else:

            sns.boxplot(df, ax=axs[i])
            axs[i].set_xlabel("")
            axs[i].set_ylabel("Load Factor (%)")
            axs[i].set_title(f"{sector}")
            axs[i].tick_params(axis="x", labelrotation=45)

    fig.tight_layout()

    return fig, axs
