"""
Plots flow of energy for sector coupling studies.

Used to compare results agasint Lawrence Berkly Energy Flow charts here:
https://flowcharts.llnl.gov/commodities/energy
"""

import pandas as pd
import plotly.graph_objects as go
import pypsa
from _helpers import configure_logging, mock_snakemake
from constants import TBTU_2_MWH
from pypsa.descriptors import get_switchable_as_dense
from pypsa.statistics import StatisticsAccessor

# These are node colors! Energy Services and Rejected Energy links do not
# follow node color assignment and are corrected in code
COLORS = {
    "Solar": "rgba(255,216,0,1)",
    "Nuclear": "rgba(205,0,0,1)",
    "Hydro": "rgba(0,0,255,1)",
    "Wind": "rgba(146,10,146,1)",
    "Geothermal": "rgba(146,90,10,1)",
    "Natural Gas": "rgba(68,172,245,1)",
    "Coal": "rgba(105,105,105,1)",
    "Biomass": "rgba(145,239,145,1)",
    "Petroleum": "rgba(0,96,0,1)",
    "Electricity Generation": "rgba(231,155,52,1)",
    "Residential": "rgba(255,188,200,1)",
    "Commercial": "rgba(255,188,200,1)",
    "Industrial": "rgba(255,188,200,1)",
    "Transportation": "rgba(255,188,200,1)",
    "Rejected Energy": "rgba(186,186,186,1)",
    "Energy Services": "rgba(97,97,97,1)",
}

SANKEY_CODE_MAPPER = {name: num for num, name in enumerate(COLORS)}

NAME_MAPPER = {
    "Solar": "Solar",
    "Reservoir & Dam": "Hydro",
    "Fixed Bottom Offshore Wind": "Wind",
    "Floating Offshore Wind": "Wind",
    "Onshore Wind": "Wind",
    "Biomass": "Biomass",
    "Combined-Cycle Gas": "Natural Gas",
    "Nuclear": "Nuclear",
    "Open-Cycle Gas": "Natural Gas",
    "gas": "Natural Gas",
    "Geothermal": "Geothermal",
    "Coal": "Coal",
    "coal": "Coal",
    "Oil": "Petroleum",
    "com": "Commercial",
    "res": "Residential",
    "trn": "Transportation",
    "ind": "Industrial",
}

# when accounting energy delievered/rejected to end-use sector, we just
# count energy in and energy out. We do not want to count energy lost for
# end-use cross sector links (like air conditioners or heat pumps)
END_USE_TECH_EXCLUSIONS = {"air-con", "heat-pump"}

###
# POWER GENERATION SECTOR
###


def _get_generator_primary_energy(n: pypsa.Network) -> pd.DataFrame:
    """
    Gets primary energy use from all generators as a positive number.
    """
    weightings = n.snapshot_weightings
    eff = get_switchable_as_dense(n, "Generator", "efficiency")
    p = n.generators_t.p.div(eff).mul(weightings.generators, axis=0)
    p = p.reset_index().drop(columns=["timestep"]).groupby(by="period").sum().T
    p["carrier"] = p.index.map(n.generators.carrier).map(n.carriers.nice_name)
    p["bus_carrier"] = p.index.map(n.generators.bus).map(n.buses.carrier)
    p["Component"] = "Generator"
    return (
        p.reset_index(drop=True).groupby(["Component", "carrier", "bus_carrier"]).sum()
    )


def get_AC_generator_primary(n: pypsa.Network, investment_period: int) -> pd.DataFrame:
    """
    Gets AC primary energy use.
    """
    df = _get_generator_primary_energy(n).droplevel("Component")[[investment_period]]
    df = df.reset_index()
    df = (
        df[df.bus_carrier == "AC"]
        .rename(columns={"carrier": "source", investment_period: "value"})
        .copy()
    )
    df["target"] = "Electricity Generation"
    return df[["source", "target", "value"]]


def _get_generator_losses(n: pypsa.Network, investment_period: int) -> pd.DataFrame:
    """
    Gets Rejected Energy Values for generators.
    """
    used = StatisticsAccessor(n).energy_balance("Generator")[[investment_period]]
    total = _get_generator_primary_energy(n).droplevel("Component")[[investment_period]]
    return total - used


def get_AC_generator_rejected(n: pypsa.Network, investment_period: int) -> pd.DataFrame:
    """
    Gets AC Rejected Energy Values for generators.
    """
    df = _get_generator_losses(n, investment_period)
    df = df.reset_index()
    df = (
        df[df.bus_carrier == "AC"]
        .rename(columns={investment_period: "value"})
        .drop(columns=["carrier", "bus_carrier"])
        .copy()
    )
    df["target"] = "Rejected Energy"
    df["source"] = "Electricity Generation"
    df = df.groupby(["target", "source"], as_index=False).sum()
    return df[["source", "target", "value"]]


def get_AC_link_primary(n: pypsa.Network, investment_period: int) -> pd.DataFrame:
    """
    Gets AC links primary energy usage.
    """
    df = StatisticsAccessor(n).energy_balance("Link")[[investment_period]]
    df = df.reset_index()
    df = (
        df[(df.bus_carrier == "AC") & (df[investment_period] >= 0)]
        .rename(columns={"carrier": "source", investment_period: "value"})
        .copy()
    )
    df["target"] = "Electricity Generation"
    return df[["source", "target", "value"]]


def get_AC_link_rejected(n: pypsa.Network, investment_period: int) -> pd.DataFrame:
    """
    Gets AC energy rejected from links.

    This is rejected energy from the power sector, not the end-use!
    """

    def ac_links(n: pypsa.Network, investment_period: int) -> list[str]:
        df = StatisticsAccessor(n).energy_balance("Link")[[investment_period]]
        df = df.reset_index()
        df_carriers = df[(df.bus_carrier == "AC") & (df[investment_period] >= 0)]
        return df_carriers.carrier.to_list()

    df = StatisticsAccessor(n).energy_balance("Link")[[investment_period]]
    df = df.reset_index()
    carriers = ac_links(n, investment_period)
    df = df[df.carrier.isin(carriers)]

    primary = (
        df[~(df.bus_carrier == "AC")].drop(columns=["bus_carrier"]).set_index("carrier")
    )
    used = df[df.bus_carrier == "AC"].drop(columns=["bus_carrier"]).set_index("carrier")
    rejected = primary.mul(-1) - used  # -1 because links remove energy from bus0

    rejected = rejected.reset_index().rename(columns={investment_period: "value"})
    rejected["target"] = "Rejected Energy"
    rejected["source"] = "Electricity Generation"
    rejected = rejected.groupby(["target", "source"], as_index=False).sum()
    return rejected[["source", "target", "value"]]


###
# END-USE ENERGY TRACKING
###


def get_end_use_delievered(n: pypsa.Network, investment_period: int) -> pd.DataFrame:
    """
    Gets delievered energy to end use sectors from end use sectors.
    """

    dfs = []

    for end_use in ("res", "com", "ind", "trn"):
        dfs.append(_get_end_use_delievered_per_sector(n, investment_period, end_use))

    return pd.concat(dfs)


def _get_end_use_delievered_per_sector(
    n: pypsa.Network,
    investment_period: int,
    sector: str,
) -> pd.DataFrame:
    """
    Gets energy delievered to an end use sector.

    This will track, for example, the amount of natural gas required to
    power the industrial sector. Or the amount of electricity needed for
    transportation sector. This is delievered energy (used + rejected =
    delievered)
    """

    def assign_source(s: str) -> str:
        if (s == "dist") or (s == "evs"):
            return "Electricity Generation"
        else:
            return s

    exclusion = END_USE_TECH_EXCLUSIONS

    df = StatisticsAccessor(n).energy_balance("Link")[[investment_period]]
    df = df.reset_index()
    df = df[df.bus_carrier.map(lambda x: True if f"{sector}-" in x else False)]
    df = df[~(df.carrier.isin(exclusion))]
    df["carrier"] = df.carrier.map(lambda x: x.split("-")[0])
    df["source"] = df.carrier.map(assign_source)
    df["target"] = sector
    df = df.rename(columns={investment_period: "value"})
    return df[["source", "target", "value"]]


def get_end_use_rejected(n: pypsa.Network, investment_period: int) -> pd.DataFrame:
    """
    Gets delievered energy to end use sectors from end use sectors.
    """

    dfs = []

    for end_use in ("res", "com", "ind", "trn"):
        dfs.append(_get_end_use_rejected_per_sector(n, investment_period, end_use))

    return pd.concat(dfs)


def _get_end_use_rejected_per_sector(
    n: pypsa.Network,
    investment_period: int,
    sector: str,
) -> pd.DataFrame:
    """
    Gets energy rejected from an end-use sector.
    """

    delievered = _get_end_use_delievered_per_sector(
        n,
        investment_period,
        sector,
    ).value.sum()
    used = (
        get_energy_services(n, investment_period)
        .set_index("source")
        .at[sector, "value"]
    )
    rejected = delievered - used
    assert rejected >= 0

    return pd.DataFrame(
        pd.DataFrame(
            data=[[sector, "Rejected Energy", rejected]],
            columns=["source", "target", "value"],
        ),
    )


def get_energy_services(n: pypsa.Network, investment_period: int) -> pd.DataFrame:
    """
    Gets used end-use to energy_servives.
    """
    df = StatisticsAccessor(n).energy_balance("Load")[[investment_period]]
    df = df.mul(-1)  # load is negative on the system
    df = df.reset_index()
    df["source"] = df.bus_carrier.map(lambda x: x.split("-")[0])
    df["target"] = "Energy Services"
    df = (
        df.drop(columns=["carrier", "bus_carrier"])
        .groupby(["source", "target"], as_index=False)
        .sum()
        .rename(columns={investment_period: "value"})
    )
    return df[["source", "target", "value"]]


###
# Chart Formatting
###


def map_sankey_name(name: str):
    try:
        return NAME_MAPPER[name]
    except KeyError:
        return name


def get_sankey_dataframe(n: pypsa.Network, investment_period: int) -> pd.DataFrame:
    dfs = [
        get_AC_generator_primary(n, investment_period),
        get_AC_generator_rejected(n, investment_period),
        get_AC_link_primary(n, investment_period),
        get_AC_link_rejected(n, investment_period),
        get_end_use_delievered(n, investment_period),
        get_end_use_rejected(n, investment_period),
        get_energy_services(n, investment_period),
    ]
    df = pd.concat(dfs).groupby(["source", "target"], as_index=False).sum()
    df["source"] = df.source.map(map_sankey_name)
    df["target"] = df.target.map(map_sankey_name)
    return df.groupby(["source", "target"], as_index=False).sum()[
        ["source", "target", "value"]
    ]


def format_sankey_data(data: pd.DataFrame) -> pd.DataFrame:

    def assign_link_color(row: pd.Series) -> str:
        if row.target == "Rejected Energy":
            return COLORS["Rejected Energy"]
        elif row.target == "Energy Services":
            return COLORS["Energy Services"]
        else:
            return COLORS[row.source]

    df = data.copy()
    df["value"] = df.value.mul(1 / TBTU_2_MWH).div(1000)  # MWH -> quads
    df["node_color"] = df.source.map(COLORS)
    df["link_color"] = df.apply(assign_link_color, axis=1)
    df["link_color"] = df.link_color.str.replace(",1)", ",0.5)")
    df["source"] = df.source.map(SANKEY_CODE_MAPPER)
    df["target"] = df.target.map(SANKEY_CODE_MAPPER)
    return df


###
# ENTRY POINT
###

if __name__ == "__main__":

    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_energy_sankey",
            interconnect="texas",
            clusters=20,
            ll="v1.0",
            opts="500SEG",
            sector="E-G",
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)

    output_file = snakemake.output

    for investment_period in n.investment_periods:

        df = get_sankey_dataframe(n, investment_period)
        df = format_sankey_data(df)

        fig = go.Figure(
            data=[
                go.Sankey(
                    valueformat=".0f",
                    valuesuffix="Quads",
                    node=dict(
                        pad=15,
                        thickness=15,
                        line=dict(color="black", width=0.5),
                        label=list(SANKEY_CODE_MAPPER.keys()),
                        color=[COLORS[x] for x in SANKEY_CODE_MAPPER.keys()],
                    ),
                    link=dict(
                        source=df.source.to_list(),
                        target=df.target.to_list(),
                        value=df.value.to_list(),
                        # label =  sankey_data.label.to_list(),
                        color=df.link_color.to_list(),
                    ),
                ),
            ],
        )

        fig.update_layout(
            title_text=f"USA Energy Consumption in {investment_period}: 1000 Quads",
            font_size=10,
        )
        fig.show()
