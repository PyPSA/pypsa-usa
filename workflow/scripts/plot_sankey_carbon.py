"""
Plots flow of carbon for sector coupling studies.

Used to compare results agasint Lawrence Berkly Energy Flow charts here:
https://flowcharts.llnl.gov/commodities/energy
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import pypsa
from _helpers import configure_logging, mock_snakemake
from constants import TBTU_2_MWH
from summary_sector import _get_links_in_state

# These are node colors! Energy Services and Rejected Energy links do not
# follow node color assignment and are corrected in code
COLORS = {
    "Natural Gas": "rgba(68,172,245,1)",
    "Coal": "rgba(105,105,105,1)",
    "Petroleum": "rgba(0,96,0,1)",
    "Electricity Generation": "rgba(231,155,52,1)",
    "Residential": "rgba(255,188,200,1)",
    "Commercial": "rgba(255,188,200,1)",
    "Industrial": "rgba(255,188,200,1)",
    "Transportation": "rgba(255,188,200,1)",
    "CO2e Emissions": "rgba(186,186,186,1)",
    # "Energy Services": "rgba(97,97,97,1)",
}

POSITIONS = {
    "Natural Gas": (0.01, 0.01),
    "Coal": (0.01, 0.50),
    "Petroleum": (0.01, 0.99),
    "Electricity Generation": (0.33, 0.01),
    "Residential": (0.66, 0.4),
    "Commercial": (0.66, 0.6),
    "Industrial": (0.66, 0.8),
    "Transportation": (0.66, 0.99),
    "CO2e Emissions": (0.99, 0.50),
}

SANKEY_CODE_MAPPER = {name: num for num, name in enumerate(COLORS)}

NAME_MAPPER = {
    "Solar": "Solar",
    "solar": "Solar",
    "Reservoir & Dam": "Hydro",
    "hydro": "Hydro",
    "Fixed Bottom Offshore Wind": "Wind",
    "Floating Offshore Wind": "Wind",
    "offwind_floating": "Wind",
    "Onshore Wind": "Wind",
    "onwind": "Wind",
    "Biomass": "Biomass",
    "biomass": "Biomass",
    "Combined-Cycle Gas": "Natural Gas",
    "CCGT": "Natural Gas",
    "Nuclear": "Nuclear",
    "nuclear": "Nuclear",
    "Open-Cycle Gas": "Natural Gas",
    "OCGT": "Natural Gas",
    "gas": "Natural Gas",
    "Geothermal": "Geothermal",
    "geothermal": "Geothermal",
    "Coal": "Coal",
    "coal": "Coal",
    "Oil": "Petroleum",
    "oil": "Petroleum",
    "com": "Commercial",
    "res": "Residential",
    "trn": "Transportation",
    "ind": "Industrial",
}

###
# Power Sector
###


def get_pwr_flows(n: pypsa.Network, investment_period: int, state: str) -> pd.DataFrame:

    if state:
        links_in_state = _get_links_in_state(n, state)
    else:
        links_in_state = n.links.index.to_list()

    links = n.links.loc[links_in_state].copy()

    links = links[links.bus2.str.endswith("co2") & ~(links.carrier.str.startswith(("trn-", "com-", "res-", "ind-")))]

    ccs_mapper = {x: x.split("-")[0] for x in links.carrier}

    weights = n.snapshot_weightings.objective

    links = (
        n.links_t["p2"][links.index]
        .mul(-1)
        .mul(weights, axis=0)
        .loc[investment_period]
        .rename(columns=n.links.carrier)
        .rename(columns=ccs_mapper)
        .T.groupby(level=0)
        .sum()
        .T.sum()
    )

    consumption = links.to_frame(name="value").reset_index(names="source")
    consumption["target"] = "Electricity Generation"

    supply = pd.DataFrame(
        [["Electricity Generation", "CO2e Emissions", links.sum()]],
        columns=["source", "target", "value"],
    )

    return pd.concat([consumption, supply])[["source", "target", "value"]]


def get_sector_flows(
    n: pypsa.Network,
    sector: str,
    investment_period: int,
    state: str,
) -> pd.DataFrame:

    weights = n.snapshot_weightings.objective

    if state:
        links_in_state = _get_links_in_state(n, state)
    else:
        links_in_state = n.links.index.to_list()

    links = n.links.loc[links_in_state].copy()

    links = links[links.bus2.str.endswith("co2") & links.carrier.str.startswith(f"{sector}-")]

    links = (
        n.links_t["p2"][links.index]
        .mul(-1)
        .mul(weights, axis=0)
        .rename(columns=n.links.bus0)
        .rename(columns=n.buses.carrier)
        .loc[investment_period]
        .T.groupby(level=0)
        .sum()
        .T.sum()
    )

    links_consumption = links.to_frame(name="value").reset_index(names="source")
    links_consumption["target"] = sector

    links_supply = pd.DataFrame(
        [[sector, "CO2e Emissions", links.sum()]],
        columns=["source", "target", "value"],
    )

    return pd.concat([links_consumption, links_supply])[["source", "target", "value"]]


def get_sankey_dataframe(
    n: pypsa.Network,
    investment_period: int,
    state: Optional[str] = None,
) -> pd.DataFrame:
    dfs = [
        get_pwr_flows(n, investment_period, state),
        get_sector_flows(n, "res", investment_period, state),
        get_sector_flows(n, "com", investment_period, state),
        get_sector_flows(n, "ind", investment_period, state),
        get_sector_flows(n, "trn", investment_period, state),
    ]
    df = pd.concat(dfs)
    return df.groupby(["source", "target"], as_index=False).sum()[["source", "target", "value"]]


def format_sankey_data(
    data: pd.DataFrame,
    color_mapper: dict[str, str],
    name_mapper: dict[str, str],
    sankey_codes: dict[str, int],
) -> pd.DataFrame:

    def map_sankey_name(name: str):
        try:
            return name_mapper[name]
        except KeyError:
            return name

    def assign_link_color(row: pd.Series) -> str:
        if row.source == "Electricity Generation":
            return color_mapper["Electricity Generation"]
        elif row.target == "CO2e Emissions":
            return color_mapper["CO2e Emissions"]
        else:
            return color_mapper[row.source]

    df = data.copy()

    # standarize name and group together
    df["source"] = df.source.map(map_sankey_name)
    df["target"] = df.target.map(map_sankey_name)
    df = df.groupby(["source", "target"], as_index=False).sum()

    # assign colors
    df["node_color"] = df.source.map(color_mapper)
    df["link_color"] = df.apply(assign_link_color, axis=1)
    df["link_color"] = df.link_color.str.replace(",1)", ",0.5)")

    # map names to node numbers
    df["source"] = df.source.map(sankey_codes)
    df["target"] = df.target.map(sankey_codes)

    # convert units
    df["value"] = df.value.mul(1 / TBTU_2_MWH)  # MWH -> TBTU
    return df


###
# ENTRY POINT
###

if __name__ == "__main__":

    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_sankey_energy",
            simpl="70",
            opts="3h",
            clusters="29m",
            ll="v1.0",
            sector_opts="",
            sector="E-G",
            planning_horizons="2018",
            interconnect="western",
        )
        rootpath = ".."
    else:
        rootpath = "."

    configure_logging(snakemake)

    results_dir = Path(rootpath, snakemake.params.root_dir)

    n = pypsa.Network(snakemake.input.network)

    output_file = snakemake.output

    states = n.buses.reeds_state.unique()
    states = [x for x in states if x]  # remove ""

    X = {node: POSITIONS[node][0] for node in SANKEY_CODE_MAPPER}
    Y = {node: POSITIONS[node][1] for node in SANKEY_CODE_MAPPER}

    assert len(n.investment_periods) == 1
    investment_period = n.investment_periods[0]

    # plot state level

    for state in states:

        df = get_sankey_dataframe(
            n=n,
            investment_period=investment_period,
            state=state,
        )
        df = format_sankey_data(df, COLORS, NAME_MAPPER, SANKEY_CODE_MAPPER)

        fig = go.Figure(
            data=[
                go.Sankey(
                    arrangement="snap",
                    valueformat=".0f",
                    valuesuffix="MT",
                    node=dict(
                        pad=10,
                        thickness=15,
                        line=dict(color="black", width=0.5),
                        label=list(SANKEY_CODE_MAPPER),
                        color=[COLORS[x] for x in SANKEY_CODE_MAPPER],
                        # x=[X[x] for x in SANKEY_CODE_MAPPER],
                        # y=[Y[x] for x in SANKEY_CODE_MAPPER],
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
            title_text=f"{state} Carbon Flow in {investment_period} (MT)",
            font_size=24,
            font_color="black",
            font_family="Arial",
        )

        fig_name_html = Path(results_dir, state, "sankey", "carbon.html")
        fig_name_png = Path(results_dir, state, "sankey", "carbon.png")
        if not fig_name_html.parent.exists():
            fig_name_html.parent.mkdir(parents=True)

        fig.write_html(str(fig_name_html))
        fig.write_image(str(fig_name_png))

    # plot system level

    df = get_sankey_dataframe(
        n=n,
        investment_period=investment_period,
    )
    df = format_sankey_data(df, COLORS, NAME_MAPPER, SANKEY_CODE_MAPPER)

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                valueformat=".0f",
                valuesuffix="MT",
                node=dict(
                    pad=10,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    label=list(SANKEY_CODE_MAPPER),
                    color=[COLORS[x] for x in SANKEY_CODE_MAPPER],
                    # x=[X[x] for x in SANKEY_CODE_MAPPER],
                    # y=[Y[x] for x in SANKEY_CODE_MAPPER],
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
        title_text=f"System Carbon Flow in {investment_period} (MT)",
        font_size=24,
        font_color="black",
        font_family="Arial",
    )

    fig_name_html = Path(results_dir, "system", "sankey", "carbon.html")
    fig_name_png = Path(results_dir, "system", "sankey", "carbon.png")
    if not fig_name_html.parent.exists():
        fig_name_html.parent.mkdir(parents=True)

    fig.write_html(str(fig_name_html))
    fig.write_image(str(fig_name_png))
