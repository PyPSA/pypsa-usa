"""
Plots flow of energy for sector coupling studies.

Used to compare results agasint Lawrence Berkly Energy Flow charts here:
https://flowcharts.llnl.gov/commodities/energy
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly
import plotly.graph_objects as go
import pypsa
from _helpers import configure_logging, mock_snakemake
from constants import ATB_TECH_MAPPER, TBTU_2_MWH
from constants_sector import TransportEfficiency
from pypsa.descriptors import get_switchable_as_dense
from summary_sector import _get_gens_in_state, _get_links_in_state

# These are node colors! Energy Services and Rejected Energy links do not
# follow node color assignment and are corrected in code
COLORS = {
    "Solar": "rgba(255,216,0,1)",
    "Nuclear": "rgba(205,0,0,1)",
    "Hydro": "rgba(0,0,255,1)",
    "Wind": "rgba(146,10,146,1)",
    "Geothermal": "rgba(146,90,10,1)",
    "Natural Gas": "rgba(68,172,245,1)",
    "Coal": "rgba(30,30,30,1)",
    "Biomass": "rgba(145,239,145,1)",
    "Petroleum": "rgba(0,96,0,1)",
    "Electricity Generation": "rgba(231,155,52,1)",
    "Residential": "rgba(255,188,200,1)",
    "Commercial": "rgba(255,188,200,1)",
    "Industrial": "rgba(255,188,200,1)",
    "Transportation": "rgba(255,188,200,1)",
    "Rejected Energy": "rgba(150,150,150,1)",
    "Energy Services": "rgba(250,210,250,1)",
}

# default (x, y) positions of nodes
# matches start postions of LLNL sankey
# 0 or 1 cause formatting issues.
POSITIONS = {
    "Solar": (0.01, 0.01),
    "Nuclear": (0.01, 0.1),
    "Hydro": (0.01, 0.2),
    "Wind": (0.01, 0.3),
    "Geothermal": (0.01, 0.4),
    "Natural Gas": (0.01, 0.5),
    "Coal": (0.01, 0.75),
    "Biomass": (0.01, 0.85),
    "Petroleum": (0.01, 0.99),
    "Electricity Generation": (0.5, 0.01),
    "Residential": (0.66, 0.4),
    "Commercial": (0.66, 0.6),
    "Industrial": (0.66, 0.8),
    "Transportation": (0.66, 0.99),
    "Rejected Energy": (0.99, 0.33),
    "Energy Services": (0.99, 0.66),
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
# POWER GENERATION SECTOR
###


def _get_consumption_generators(
    n: pypsa.Network,
    period: int,
    state: Optional[str] = None,
) -> pd.DataFrame:

    if state:
        generators = _get_gens_in_state(n, state)
    else:
        generators = n.generators.index.to_list()

    weights = n.snapshot_weightings.objective
    eff = get_switchable_as_dense(n, "Generator", "efficiency")

    df = n.generators_t["p"].div(eff).loc[period][generators].rename(columns=n.generators.carrier)
    df = df.T.groupby(level=0).sum().T
    df = df.mul(weights, axis=0).sum().to_frame(name="value").reset_index(names="source")

    return df


def _get_rejected_generators(
    n: pypsa.Network,
    period: int,
    state: Optional[str] = None,
) -> pd.DataFrame:

    if state:
        generators = _get_gens_in_state(n, state)
    else:
        generators = n.generators.index.to_list()

    weights = n.snapshot_weightings.objective
    eff = get_switchable_as_dense(n, "Generator", "efficiency")

    consumption = n.generators_t["p"].div(eff).loc[period][generators].rename(columns=n.generators.carrier)
    production = n.generators_t["p"].loc[period][generators].rename(columns=n.generators.carrier)

    df = consumption - production
    df = df.T.groupby(level=0).sum().T
    df = df.mul(weights, axis=0).sum().to_frame(name="value").reset_index(names="source")

    return df


def _get_consumption_links(
    n: pypsa.Network,
    period: int,
    state: Optional[str] = None,
) -> pd.DataFrame:
    if state:
        links = _get_links_in_state(n, state)
    else:
        links = n.links.index.to_list()

    weights = n.snapshot_weightings.objective
    df = n.links_t["p0"].loc[period][links].rename(columns=n.links.carrier)
    df = df.T.groupby(level=0).sum().T
    df = df.mul(weights, axis=0).sum().to_frame(name="value").reset_index(names="source")
    return df


def _get_rejected_links(
    n: pypsa.Network,
    period: int,
    state: Optional[str] = None,
) -> pd.DataFrame:

    if state:
        links = _get_links_in_state(n, state)
    else:
        links = n.links.index.to_list()

    weights = n.snapshot_weightings.objective
    df = n.links_t["p0"].loc[period].add(n.links_t["p1"].loc[period])
    df = df[links].rename(columns=n.links.carrier)
    df = df.T.groupby(level=0).sum().T
    df = df.mul(weights, axis=0).sum().to_frame(name="value").reset_index(names="source")
    return df


def _remove_ccs(s: str) -> str:
    if s.endswith("CCS"):
        return s.split("-")[0]
    else:
        return s


def get_electricity_consumption(
    n: pypsa.Network,
    carriers: list[str],
    period: int,
    state: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.concat(
        [
            _get_consumption_generators(n, period, state),
            _get_consumption_links(n, period, state),
        ],
    )
    df = df[df.source.isin(carriers)]
    df["source"] = df.source.map(_remove_ccs)
    df = df.groupby(["source"], as_index=False).sum()
    df["target"] = "Electricity Generation"
    return df[["source", "target", "value"]]


def get_electricity_rejected(
    n: pypsa.Network,
    carriers: list[str],
    period: int,
    state: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.concat(
        [
            _get_rejected_generators(n, period, state),
            _get_rejected_links(n, period, state),
        ],
    )
    df = df[df.source.isin(carriers)]
    df = df.drop(columns="source").sum().to_frame(name="value").reset_index(drop=True)
    df["source"] = "Electricity Generation"
    df["target"] = "Rejected Energy"
    return df[["source", "target", "value"]]


###
# SECTORS
###


def _get_sector_consumption(
    n: pypsa.Network,
    sector: str,
    fuel: str,
    period: int,
    state: Optional[str] = None,
) -> float:

    if fuel == "elec":
        fuel = "AC"

    if state:
        links_in_state = _get_links_in_state(n, state)
    else:
        links_in_state = n.links.index.to_list()

    weights = n.snapshot_weightings.objective

    buses = n.buses[n.buses.carrier == fuel].index.to_list()
    links = n.links.loc[links_in_state].copy()

    df = links[links.carrier.str.startswith(f"{sector}-") & links.bus0.isin(buses)]

    df = n.links_t["p0"][df.index].mul(weights, axis=0).loc[period]

    return df.sum().sum()


def _get_service_supply(
    n: pypsa.Network,
    sector: str,
    period: int,
    state: Optional[str] = None,
) -> float:

    if state:
        links_in_state = _get_links_in_state(n, state)
    else:
        links_in_state = n.links.index.to_list()

    buses = n.loads[n.loads.carrier.str.startswith(f"{sector}-")].index
    links = n.links[(n.links.index.isin(links_in_state)) & (n.links.bus1.isin(buses))]

    links = links[
        ~links.index.str.endswith("space-heat-charger")
        & ~links.index.str.endswith("space-heat-discharger")
        & ~links.index.str.endswith("space-cool-charger")
        & ~links.index.str.endswith("space-cool-discharger")
        & ~links.index.str.endswith("water-heat-charger")
    ]

    weights = n.snapshot_weightings.objective

    supplied = n.links_t["p1"].loc[period][links.index].mul(weights, axis=0).mul(-1)

    return supplied.sum().sum()


def _get_service_rejected(
    n: pypsa.Network,
    sector: str,
    period: int,
    state: Optional[str] = None,
) -> float:

    if state:
        links_in_state = _get_links_in_state(n, state)
    else:
        links_in_state = n.links.index.to_list()

    links = n.links[n.links.index.isin(links_in_state) & n.links.carrier.str.startswith(f"{sector}-")]

    eff = get_switchable_as_dense(n, "Link", "efficiency")
    supply = n.links_t["p1"].mul(-1)

    # rejected will be less than 0 for COP > 1
    rejected = supply.div(eff) - supply
    rejected = rejected.where(rejected >= 0, 0)

    weights = n.snapshot_weightings.objective

    rejected = rejected.loc[period][links.index].mul(weights, axis=0)

    return rejected.sum().sum()


def _get_industry_supply(
    n: pypsa.Network,
    period: int,
    state: Optional[str] = None,
) -> float:

    if state:
        links_in_state = _get_links_in_state(n, state)
    else:
        links_in_state = n.links.index.to_list()

    buses = n.loads[n.loads.carrier.str.startswith("ind-")].index
    links = n.links[(n.links.index.isin(links_in_state)) & (n.links.bus1.isin(buses))]

    weights = n.snapshot_weightings.objective

    supplied = n.links_t["p1"].loc[period][links.index].mul(weights, axis=0).mul(-1)

    return supplied.sum().sum()


def _get_industry_rejected(
    n: pypsa.Network,
    period: int,
    state: Optional[str] = None,
) -> float:

    if state:
        links_in_state = _get_links_in_state(n, state)
    else:
        links_in_state = n.links.index.to_list()

    links = n.links[n.links.index.isin(links_in_state) & n.links.carrier.str.startswith("ind-")]

    eff = get_switchable_as_dense(n, "Link", "efficiency")
    supply = n.links_t["p1"].mul(-1)

    # rejected will be less than 0 for COP > 1
    rejected = supply.div(eff) - supply
    rejected = rejected.where(rejected >= 0, 0)

    weights = n.snapshot_weightings.objective

    rejected = rejected.loc[period][links.index].mul(weights, axis=0)

    return rejected.sum().sum()


def _get_transport_supply(
    n: pypsa.Network,
    period: int,
    state: Optional[str] = None,
) -> float:

    if state:
        links_in_state = _get_links_in_state(n, state)
    else:
        links_in_state = n.links.index.to_list()

    # get load aggregation buses only
    buses = n.buses[n.buses.carrier.isin(["AC", "oil"])].index
    links = n.links[
        (n.links.index.isin(links_in_state)) & (n.links.bus0.isin(buses)) & (n.links.carrier.str.startswith("trn-"))
    ].index

    weights = n.snapshot_weightings.objective

    # p0 and p1 will give same value as efficiency is applied further down
    supply = n.links_t["p1"].loc[period][links].mul(weights, axis=0).mul(-1)

    # apply approximate efficiencies
    eff = supply.copy()
    for col in eff.columns:
        if "elec" in col:
            eff[col] = TransportEfficiency.ELEC.value
        elif "lpg" in col:
            eff[col] = TransportEfficiency.LPG.value
        else:
            raise ValueError

    return supply.mul(eff).sum().sum()


def _get_transport_rejected(
    n: pypsa.Network,
    period: int,
    state: Optional[str] = None,
) -> float:

    if state:
        links_in_state = _get_links_in_state(n, state)
    else:
        links_in_state = n.links.index.to_list()

    # get load aggregation buses only
    buses = n.buses[n.buses.carrier.isin(["AC", "oil"])].index
    links = n.links[
        (n.links.index.isin(links_in_state)) & (n.links.bus0.isin(buses)) & (n.links.carrier.str.startswith("trn-"))
    ].index

    weights = n.snapshot_weightings.objective

    # p0 and p1 will give same value as efficiency is applied further down
    supply = n.links_t["p1"].loc[period][links].mul(weights, axis=0).mul(-1)

    # apply approximate efficiencies
    eff = supply.copy()
    for col in eff.columns:
        if "elec" in col:
            eff[col] = TransportEfficiency.ELEC.value
        elif "lpg" in col:
            eff[col] = TransportEfficiency.LPG.value
        else:
            raise ValueError

    rejected = supply - supply.mul(eff)

    return rejected.sum().sum()


def get_energy_flow_res(
    n: pypsa.Network,
    period: int,
    state: Optional[str] = None,
) -> pd.DataFrame:

    elec_consumption = _get_sector_consumption(n, "res", "elec", period, state)
    lpg_consumption = _get_sector_consumption(n, "res", "oil", period, state)
    gas_consumption = _get_sector_consumption(n, "res", "gas", period, state)

    supply = _get_service_supply(n, "res", period, state)
    rejected = _get_service_rejected(n, "res", period, state)

    df = pd.DataFrame(
        [
            ["Electricity Generation", "Residential", elec_consumption],
            ["Natural Gas", "Residential", gas_consumption],
            ["Petroleum", "Residential", lpg_consumption],
            ["Residential", "Energy Services", supply],
            ["Residential", "Rejected Energy", rejected],
        ],
        columns=["source", "target", "value"],
    )

    return df


def get_energy_flow_com(
    n: pypsa.Network,
    period: int,
    state: Optional[str] = None,
) -> pd.DataFrame:

    elec_consumption = _get_sector_consumption(n, "com", "elec", period, state)
    lpg_consumption = _get_sector_consumption(n, "com", "oil", period, state)
    gas_consumption = _get_sector_consumption(n, "com", "gas", period, state)

    supply = _get_service_supply(n, "com", period, state)
    rejected = _get_service_rejected(n, "com", period, state)

    df = pd.DataFrame(
        [
            ["Electricity Generation", "Commercial", elec_consumption],
            ["Natural Gas", "Commercial", gas_consumption],
            ["Petroleum", "Commercial", lpg_consumption],
            ["Commercial", "Energy Services", supply],
            ["Commercial", "Rejected Energy", rejected],
        ],
        columns=["source", "target", "value"],
    )

    return df


def get_energy_flow_ind(
    n: pypsa.Network,
    period: int,
    state: Optional[str] = None,
) -> pd.DataFrame:

    elec_consumption = _get_sector_consumption(n, "ind", "elec", period, state)
    coal_consumption = _get_sector_consumption(n, "ind", "coal", period, state)
    gas_consumption = _get_sector_consumption(n, "ind", "gas", period, state)

    supply = _get_industry_supply(n, period, state)
    rejected = _get_industry_rejected(n, period, state)

    df = pd.DataFrame(
        [
            ["Electricity Generation", "Industrial", elec_consumption],
            ["Natural Gas", "Industrial", gas_consumption],
            ["Coal", "Industrial", coal_consumption],
            ["Industrial", "Energy Services", supply],
            ["Industrial", "Rejected Energy", rejected],
        ],
        columns=["source", "target", "value"],
    )

    return df


def get_energy_flow_trn(
    n: pypsa.Network,
    period: int,
    state: Optional[str] = None,
) -> pd.DataFrame:

    elec_consumption = _get_sector_consumption(n, "trn", "elec", period, state)
    oil_consumption = _get_sector_consumption(n, "trn", "oil", period, state)

    supply = _get_transport_supply(n, period, state)
    rejected = _get_transport_rejected(n, period, state)

    df = pd.DataFrame(
        [
            ["Electricity Generation", "Transportation", elec_consumption],
            ["Petroleum", "Transportation", oil_consumption],
            ["Transportation", "Energy Services", supply],
            ["Transportation", "Rejected Energy", rejected],
        ],
        columns=["source", "target", "value"],
    )

    return df


###
# Chart Formatting
###


def get_sankey_dataframe(
    n: pypsa.Network,
    investment_period: int,
    pwr_carriers: list[str],
    state: Optional[str] = None,
) -> pd.DataFrame:
    dfs = [
        get_electricity_consumption(n, pwr_carriers, investment_period, state),
        get_electricity_rejected(n, pwr_carriers, investment_period, state),
        get_energy_flow_res(n, investment_period, state),
        get_energy_flow_com(n, investment_period, state),
        get_energy_flow_ind(n, investment_period, state),
        get_energy_flow_trn(n, investment_period, state),
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
        if row.target == "Rejected Energy":
            return color_mapper["Rejected Energy"]
        elif row.target == "Energy Services":
            return color_mapper["Energy Services"]
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

    power_carriers = ATB_TECH_MAPPER.keys()

    X = {node: POSITIONS[node][0] for node in SANKEY_CODE_MAPPER}
    Y = {node: POSITIONS[node][1] for node in SANKEY_CODE_MAPPER}

    assert len(n.investment_periods) == 1
    investment_period = n.investment_periods[0]

    # plot state level

    for state in states:

        df = get_sankey_dataframe(
            n=n,
            pwr_carriers=power_carriers,
            investment_period=investment_period,
            state=state,
        )
        df = format_sankey_data(df, COLORS, NAME_MAPPER, SANKEY_CODE_MAPPER)

        fig = go.Figure(
            data=[
                go.Sankey(
                    arrangement="snap",
                    valueformat=".0f",
                    valuesuffix="TBTU",
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
            title_text=f"{state} Energy Flow in {investment_period} (TBTU)",
            font_size=24,
            font_color="black",
            font_family="Arial",
            # width=1500,
            # height=750,
        )

        fig_name_html = Path(results_dir, state, "sankey", "energy.html")
        fig_name_png = Path(results_dir, state, "sankey", "energy.png")
        if not fig_name_html.parent.exists():
            fig_name_html.parent.mkdir(parents=True)

        fig.write_html(str(fig_name_html))
        fig.write_image(str(fig_name_png))

    # plot system level

    df = get_sankey_dataframe(
        n=n,
        pwr_carriers=power_carriers,
        investment_period=investment_period,
    )
    df = format_sankey_data(df, COLORS, NAME_MAPPER, SANKEY_CODE_MAPPER)

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                valueformat=".0f",
                valuesuffix="TBTU",
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
        title_text=f"System Energy Flow in {investment_period} (TBTU)",
        font_size=24,
        font_color="black",
        font_family="Arial",
        # width=1500,
        # height=750,
    )

    fig_name_html = Path(results_dir, "system", "sankey", "energy.html")
    fig_name_png = Path(results_dir, "system", "sankey", "energy.png")
    if not fig_name_html.parent.exists():
        fig_name_html.parent.mkdir(parents=True)

    fig.write_html(str(fig_name_html))
    fig.write_image(str(fig_name_png))
