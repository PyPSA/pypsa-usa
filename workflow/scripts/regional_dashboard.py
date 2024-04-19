"""
Dash app for exploring regional disaggregated data.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pypsa
from dash import Dash, Input, Output, callback, dcc, html
from pypsa.statistics import StatisticsAccessor

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.propagate = False

import calendar
from pathlib import Path
from typing import Dict, List

from plot_statistics import get_color_palette
from summary import (
    get_demand_timeseries,
    get_energy_timeseries,
    get_node_carrier_emissions_timeseries,
    get_node_emissions_timeseries,
    get_tech_emissions_timeseries,
)

###
# IDS
###

# dropdowns
DROPDOWN_SELECT_STATE = "dropdown_select_state"
DROPDOWN_SELECT_NODE = "dropdown_select_node"
DROPDOWN_SELECT_PLOT_THEME = "dropdown_select_plot_theme"

# buttons
BUTTON_SELECT_ALL_STATES = "button_select_all_states"
BUTTON_SELECT_ALL_NODES = "button_select_all_nodes"

# slider
SLIDER_SELECT_TIME = "slider_select_time"
SLIDER_SELECT_LMP_DAY = "slider_select_lmp_day"

# radio buttons
RADIO_BUTTON_RESAMPLE = "radio_button_resample"

# graphics
GRAPHIC_MAP = "graphic_map"
GRAPHIC_DISPATCH = "graphic_dispatch"
GRAPHIC_LOAD = "graphic_load"
GRAPHIC_CF = "graphic_cf"
GRAPHIC_EMISSIONS_STATE = "graphic_emissions_state"
GRAPHIC_EMISSIONS_FUEL = "graphic_emissions_fuel"
GRAPHIC_EMISSIONS_ACCUMULATED_STATE = "graphic_emissions_accumulated_state"
GRAPHIC_EMISSIONS_ACCUMULATED_FUEL = "graphic_emissions_accumulated_fuel"
GRAPHIC_CAPACITY = "graphic_capacity"
GRAPHIC_GENERATION = "graphic_generation"
GRAPHIC_LMP_TIMESERIES = "graphic_lmp_timeseries"
GRAPHIC_LMP_MAP = "graphic_lmp_map"

# tabs
TABS_UPDATE = "tabs_update"
TABS_CONTENT = "tabs_content"
TAB_NODES = "tab_nodes"
TAB_DISPATCH = "tab_dispatch"
TAB_CF = "tab_cf"
TAB_EMISSIONS = "tab_emissions"
TAB_LMP = "tab_lmp"

###
# APP SETUP
###

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
logger.info("Reading configuration options")

# add logic to build network name
network_path = Path(
    "..",
    "results",
    "western",
    "networks",
    "elec_s_80_ec_lv1.0_RCo2L-SAFER-RPS_E.nc",
)
NETWORK = pypsa.Network(str(network_path))
TIMEFRAME = NETWORK.snapshots

# geolocated node data structures
ALL_STATES = NETWORK.buses.country.unique()

# dispatch data structure
DISPATCH = (
    StatisticsAccessor(NETWORK)
    .energy_balance(
        aggregate_time=False,
        aggregate_bus=False,
        comps=["Store", "StorageUnit", "Link", "Generator"],
    )
    .mul(1e-3)
)
DISPATCH = DISPATCH[~(DISPATCH.index.get_level_values("carrier") == "Dc")]
DISPATCH = DISPATCH.droplevel(["component", "bus_carrier"]).reset_index()
DISPATCH["state"] = DISPATCH.bus.map(NETWORK.buses.country)
DISPATCH = DISPATCH.drop(columns="bus")
DISPATCH = DISPATCH.groupby(["carrier", "state"]).sum()

# variable renewable data structure
var_renew_carriers = [
    x
    for x in DISPATCH.index.get_level_values("carrier").unique()
    if (x == "Solar" or x.endswith("Wind"))
]
VAR_RENEW = DISPATCH[
    DISPATCH.index.get_level_values("carrier").isin(var_renew_carriers)
]

# load data structure
LOAD = (
    StatisticsAccessor(NETWORK)
    .energy_balance(aggregate_time=False, aggregate_bus=False, comps=["Load"])
    .mul(1e-3)
    .mul(-1)
)
LOAD = LOAD.droplevel(["component", "carrier", "bus_carrier"]).reset_index()
LOAD["state"] = LOAD.bus.map(NETWORK.buses.country)
LOAD = LOAD.drop(columns="bus")
LOAD = LOAD.groupby(["state"]).sum()

# net load data structure
NET_LOAD = LOAD - VAR_RENEW.droplevel("carrier").groupby("state").sum()

# capacity factor data structure
CF = StatisticsAccessor(NETWORK).capacity_factor(
    aggregate_time=False,
    groupby=pypsa.statistics.get_bus_and_carrier,
    comps=["Generator"],
)
CF = (
    CF[
        CF.index.get_level_values("carrier").isin(
            [
                x
                for x in NETWORK.carriers.nice_name
                if ((x == "Solar") or (x.endswith("Wind"))) or (x == "Reservoir & Dam")
            ],
        )
    ]
    .droplevel("component")
    .reset_index()
)
CF["state"] = CF.bus.map(NETWORK.buses.country)
CF = CF.drop(columns="bus").groupby(["state", "carrier"]).mean()

# emissions data structure

TECH_EMISSIONS = get_node_carrier_emissions_timeseries(NETWORK).reset_index()
TECH_EMISSIONS["state"] = TECH_EMISSIONS.bus.map(NETWORK.buses.country)
TECH_EMISSIONS = (
    TECH_EMISSIONS.drop(columns="bus").groupby(["state", "carrier"]).sum().T
)
zeros = TECH_EMISSIONS.columns[(np.abs(TECH_EMISSIONS) < 1e-7).all()]
TECH_EMISSIONS = TECH_EMISSIONS.drop(columns=zeros).T.mul(1e-6)  # T -> MT

###
# INITIALIZATION
###

logger.info("Starting app")
app = Dash(external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "PyPSA-USA Dashboard"

###
# CALLBACK FUNCTIONS
###

# select states to include


def state_dropdown(states: list[str]) -> html.Div:
    return html.Div(
        children=[
            html.H4("States to Include"),
            dcc.Dropdown(
                id=DROPDOWN_SELECT_STATE,
                options=states,
                value=states,
                multi=True,
                persistence=True,
            ),
            html.Button(
                children=["Select All"],
                id=BUTTON_SELECT_ALL_STATES,
                n_clicks=0,
            ),
        ],
    )


@app.callback(
    Output(DROPDOWN_SELECT_STATE, "value"),
    Input(BUTTON_SELECT_ALL_STATES, "n_clicks"),
)
def select_all_countries(_: int) -> list[str]:
    return ALL_STATES


# plot map


def plot_map(n: pypsa.Network, states: list[str]) -> html.Div:

    nodes = n.buses[n.buses.country.isin(states)]

    usa_map = go.Figure()

    usa_map.add_trace(
        go.Scattermapbox(
            mode="markers",
            lon=nodes.x,
            lat=nodes.y,
            marker=dict(size=10, color="red"),
            text=nodes.index,
        ),
    )

    # Update layout to include map
    usa_map.update_layout(
        mapbox=dict(style="carto-positron", center=dict(lon=-95, lat=35), zoom=3),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return html.Div(children=[dcc.Graph(figure=usa_map)], id=GRAPHIC_MAP)


@app.callback(
    Output(GRAPHIC_MAP, "children"),
    Input(DROPDOWN_SELECT_STATE, "value"),
)
def plot_map_callback(
    states: list[str] = ALL_STATES,
) -> html.Div:
    return plot_map(NETWORK, states)


# plotting options


def select_resample() -> html.Div:
    return html.Div(
        children=[
            html.H4("Resample Period"),
            dcc.RadioItems(
                id=RADIO_BUTTON_RESAMPLE,
                options=["1h", "2h", "4h", "24h", "W"],
                value="1h",
                inline=True,
            ),
        ],
    )


def time_slider(snapshots: pd.date_range) -> html.Div:
    return html.Div(
        children=[
            html.H4("Weeks To Plot"),
            dcc.RangeSlider(
                id=SLIDER_SELECT_TIME,
                min=snapshots.min().week,
                max=snapshots.max().week,
                step=1,
                value=[snapshots.min().week, snapshots.max().week],
            ),
        ],
    )


# dispatch


def plot_dispatch(
    n: pypsa.Network,
    dispatch: pd.DataFrame,
    states: list[str],
    resample: str,
    timeframe: pd.date_range,
) -> html.Div:

    energy_mix = dispatch[dispatch.index.get_level_values("state").isin(states)]
    energy_mix = energy_mix.droplevel("state").groupby("carrier").sum().T
    energy_mix.index = pd.to_datetime(energy_mix.index)

    energy_mix = energy_mix.loc[timeframe]

    energy_mix = energy_mix.resample(resample).sum()

    color_palette = get_color_palette(n)

    fig = px.area(
        energy_mix,
        x=energy_mix.index,
        y=energy_mix.columns,
        color_discrete_map=color_palette,
    )

    title = "Dispatch [GW]"
    fig.update_layout(
        title=dict(text=title, font=dict(size=24)),
        xaxis_title="",
        yaxis_title="Power [GW]",
    )

    return html.Div(children=[dcc.Graph(figure=fig)], id=GRAPHIC_DISPATCH)


@app.callback(
    Output(GRAPHIC_DISPATCH, "children"),
    Input(DROPDOWN_SELECT_STATE, "value"),
    Input(RADIO_BUTTON_RESAMPLE, "value"),
    Input(SLIDER_SELECT_TIME, "value"),
)
def plot_dispatch_callback(
    states: list[str] = ALL_STATES,
    resample: list[str] = "1h",
    weeks: list[int] = [TIMEFRAME.min().week, TIMEFRAME.max().week],
) -> html.Div:

    # plus one because strftime indexs from 0
    timeframe = pd.Series(TIMEFRAME.strftime("%U").astype(int) + 1, index=TIMEFRAME)
    timeframe = timeframe[timeframe.isin(range(weeks[0], weeks[-1], 1))]

    return plot_dispatch(NETWORK, DISPATCH, states, resample, timeframe.index)


# load


def plot_load(
    load: pd.DataFrame,
    net_load: pd.DataFrame,
    states: list[str],
    resample: str,
    timeframe: pd.date_range,
) -> html.Div:

    state_load = load[load.index.isin(states)].sum()
    state_net_load = net_load[net_load.index.isin(states)].sum()

    data = pd.concat(
        [state_load, state_net_load],
        axis=1,
        keys=["Absolute Load", "Net Load"],
    )
    data.index = pd.to_datetime(data.index)

    data = data.loc[timeframe]

    data = data.resample(resample).sum()

    fig = px.line(data)

    title = "System Load [GW]"
    fig.update_layout(
        title=dict(text=title, font=dict(size=24)),
        xaxis_title="",
        yaxis_title="Power [GW]",
    )

    return html.Div(children=[dcc.Graph(figure=fig)], id=GRAPHIC_LOAD)


@app.callback(
    Output(GRAPHIC_LOAD, "children"),
    Input(DROPDOWN_SELECT_STATE, "value"),
    Input(RADIO_BUTTON_RESAMPLE, "value"),
    Input(SLIDER_SELECT_TIME, "value"),
)
def plot_load_callback(
    states: list[str] = ALL_STATES,
    resample: list[str] = "1h",
    weeks: list[int] = [TIMEFRAME.min().week, TIMEFRAME.max().week],
) -> html.Div:

    # plus one because strftime indexs from 0
    timeframe = pd.Series(TIMEFRAME.strftime("%U").astype(int) + 1, index=TIMEFRAME)
    timeframe = timeframe[timeframe.isin(range(weeks[0], weeks[-1], 1))]

    return plot_load(LOAD, NET_LOAD, states, resample, timeframe.index)


# capacity factor


def plot_cf(df: pd.DataFrame, carrier: str) -> go.Figure:

    fig = go.Figure(
        go.Heatmap(x=df.hour, y=df.month, z=df[carrier], colorscale="Viridis"),
    )

    fig.update_layout(
        title=f"{carrier}",
        yaxis={"tickangle": -15},
        xaxis={"title": "Hour"},
        xaxis_nticks=24,
        yaxis_nticks=len(df.month.unique()),
    )

    return fig


def plot_cfs(cf: pd.DataFrame, states: list[str]) -> html.Div:
    """
    Creates list of heatmaps dependent on number of carriers.
    """

    cfs = []

    cf_df = cf.copy()
    cf_df = (
        cf[cf.index.get_level_values("state").isin(states)]
        .droplevel("state")
        .reset_index()
        .groupby("carrier")
        .mean()
        .T
    )
    cf_df.index = pd.to_datetime(cf_df.index)

    cf_df["month"], cf_df["hour"] = cf_df.index.month, cf_df.index.hour
    cf_df["month"] = cf_df["month"].apply(lambda x: calendar.month_name[x])
    cf_df = cf_df.reset_index(drop=True)

    carriers = [x for x in cf_df.columns if not x in ["month", "hour"]]

    for carrier in carriers:
        cfs.append(plot_cf(cf_df, carrier))

    return html.Div(children=[dcc.Graph(figure=x) for x in cfs], id=GRAPHIC_CF)


@app.callback(
    Output(GRAPHIC_CF, "children"),
    Input(DROPDOWN_SELECT_STATE, "value"),
)
def plot_cf_callback(states: list[str] = ALL_STATES) -> html.Div:
    return plot_cfs(CF, states)


# emissions


def plot_emissions_state(
    n: pypsa.Network,
    states: list[str],
    resample: str,
    timeframe: pd.date_range,
) -> html.Div:

    # get data
    emissions = get_node_emissions_timeseries(n).mul(1e-6)  # T -> MT
    emissions = emissions.T.groupby(n.buses.country).sum().T
    emissions.index = pd.to_datetime(emissions.index)

    emissions = emissions[states]
    emissions = emissions.loc[timeframe]
    emissions = emissions.resample(resample).sum()

    # plot data
    fig = px.area(
        emissions,
        x=emissions.index,
        y=emissions.columns,
    )

    title = "State CO2 Emissions"
    fig.update_layout(
        title={"text": title},
        xaxis_title="",
        yaxis_title="Emissions [MT]",
    )

    return html.Div(children=[dcc.Graph(figure=fig)], id=GRAPHIC_EMISSIONS_STATE)


@app.callback(
    Output(GRAPHIC_EMISSIONS_STATE, "children"),
    Input(DROPDOWN_SELECT_STATE, "value"),
    Input(RADIO_BUTTON_RESAMPLE, "value"),
    Input(SLIDER_SELECT_TIME, "value"),
)
def plot_emissions_state_callback(
    states: list[str] = ALL_STATES,
    resample: str = "1h",
    weeks: list[int] = [TIMEFRAME.min().week, TIMEFRAME.max().week],
) -> html.Div:

    # plus one because strftime indexs from 0
    timeframe = pd.Series(TIMEFRAME.strftime("%U").astype(int) + 1, index=TIMEFRAME)
    timeframe = timeframe[timeframe.isin(range(weeks[0], weeks[-1], 1))]

    return plot_emissions_state(NETWORK, states, resample, timeframe.index)


def plot_emissions_fuel(
    n: pypsa.Network,
    df: pd.DataFrame,  # TECH_EMISSIONS
    states: list[str],
    resample: str,
    timeframe: pd.date_range,
) -> html.Div:

    emissions = (
        df[df.index.get_level_values("state").isin(states)]
        .droplevel("state")
        .reset_index()
        .groupby("carrier")
        .sum()
        .T
    )
    emissions.index = pd.to_datetime(emissions.index)

    emissions = emissions.loc[timeframe]
    emissions = emissions.resample(resample).sum()

    color_palette = get_color_palette(n)

    # plot data
    fig = px.area(
        emissions,
        x=emissions.index,
        y=emissions.columns,
        color_discrete_map=color_palette,
    )

    title = "CO2 Emissions by Technology"
    fig.update_layout(
        title={"text": title},
        xaxis_title="",
        yaxis_title="Emissions [MT]",
    )

    return html.Div(children=[dcc.Graph(figure=fig)], id=GRAPHIC_EMISSIONS_FUEL)


@app.callback(
    Output(GRAPHIC_EMISSIONS_FUEL, "children"),
    Input(DROPDOWN_SELECT_STATE, "value"),
    Input(RADIO_BUTTON_RESAMPLE, "value"),
    Input(SLIDER_SELECT_TIME, "value"),
)
def plot_emissions_fuel_callback(
    states: list[str] = ALL_STATES,
    resample: str = "1h",
    weeks: list[int] = [TIMEFRAME.min().week, TIMEFRAME.max().week],
) -> html.Div:

    # plus one because strftime indexs from 0
    timeframe = pd.Series(TIMEFRAME.strftime("%U").astype(int) + 1, index=TIMEFRAME)
    timeframe = timeframe[timeframe.isin(range(weeks[0], weeks[-1], 1))]

    return plot_emissions_fuel(
        NETWORK,
        TECH_EMISSIONS,
        states,
        resample,
        timeframe.index,
    )


def plot_accumulated_emissions_state(
    n: pypsa.Network,
    states: list[str],
    resample: str,
    timeframe: pd.date_range,
) -> html.Div:

    # get data
    emissions = get_node_emissions_timeseries(n).mul(1e-6)  # T -> MT
    emissions = emissions.T.groupby(n.buses.country).sum().T.cumsum()
    emissions.index = pd.to_datetime(emissions.index)

    emissions = emissions[states]
    emissions = emissions.loc[timeframe]
    emissions = emissions.resample(resample).sum()

    # plot data
    fig = px.area(
        emissions,
        x=emissions.index,
        y=emissions.columns,
    )

    title = "Accumulated State CO2 Emissions"
    fig.update_layout(
        title={"text": title},
        xaxis_title="",
        yaxis_title="Emissions [MT]",
    )

    return html.Div(
        children=[dcc.Graph(figure=fig)],
        id=GRAPHIC_EMISSIONS_ACCUMULATED_STATE,
    )


@app.callback(
    Output(GRAPHIC_EMISSIONS_ACCUMULATED_STATE, "children"),
    Input(DROPDOWN_SELECT_STATE, "value"),
    Input(RADIO_BUTTON_RESAMPLE, "value"),
    Input(SLIDER_SELECT_TIME, "value"),
)
def plot_accumulated_emissions_state_callback(
    states: list[str] = ALL_STATES,
    resample: str = "1h",
    weeks: list[int] = [TIMEFRAME.min().week, TIMEFRAME.max().week],
) -> html.Div:

    # plus one because strftime indexs from 0
    timeframe = pd.Series(TIMEFRAME.strftime("%U").astype(int) + 1, index=TIMEFRAME)
    timeframe = timeframe[timeframe.isin(range(weeks[0], weeks[-1], 1))]

    return plot_accumulated_emissions_state(NETWORK, states, resample, timeframe.index)


def plot_accumulated_emissions_fuel(
    n: pypsa.Network,
    df: pd.DataFrame,  # TECH_EMISSIONS
    states: list[str],
    timeframe: pd.date_range,
) -> html.Div:

    emissions = (
        df[df.index.get_level_values("state").isin(states)]
        .droplevel("state")
        .reset_index()
        .groupby("carrier")
        .sum()
        .T
    )
    emissions.index = pd.to_datetime(emissions.index)
    emissions = emissions.loc[timeframe]
    emissions = emissions.cumsum()

    color_palette = get_color_palette(n)

    # plot data
    fig = px.area(
        emissions,
        x=emissions.index,
        y=emissions.columns,
        color_discrete_map=color_palette,
    )

    title = "Accumulated CO2 Emissions by Technology"
    fig.update_layout(
        title={"text": title},
        xaxis_title="",
        yaxis_title="Emissions [MT]",
    )

    return html.Div(
        children=[dcc.Graph(figure=fig)],
        id=GRAPHIC_EMISSIONS_ACCUMULATED_FUEL,
    )


@app.callback(
    Output(GRAPHIC_EMISSIONS_ACCUMULATED_FUEL, "children"),
    Input(DROPDOWN_SELECT_STATE, "value"),
)
def plot_accumulated_emissions_fuel_callback(
    states: list[str] = ALL_STATES,
    weeks: list[int] = [TIMEFRAME.min().week, TIMEFRAME.max().week],
) -> html.Div:

    # plus one because strftime indexs from 0
    timeframe = pd.Series(TIMEFRAME.strftime("%U").astype(int) + 1, index=TIMEFRAME)
    timeframe = timeframe[timeframe.isin(range(weeks[0], weeks[-1], 1))]

    return plot_accumulated_emissions_fuel(
        NETWORK,
        TECH_EMISSIONS,
        states,
        timeframe.index,
    )


# locational marginal price


def plot_lmp_timeseries(
    n: pypsa.Network,
    states: list[str],
    resample: str,
    timeframe: pd.date_range,
) -> html.Div:

    lmp = n.buses_t.marginal_price.T
    lmp["state"] = lmp.index.map(n.buses.country)
    lmp = lmp[lmp.state.isin(states)].drop(columns="state").T
    lmp.index = pd.to_datetime(lmp.index)

    lmp = lmp.loc[timeframe]
    lmp = lmp.resample(resample).sum()

    # plot data
    fig = px.line(
        lmp,
        x=lmp.index,
        y=lmp.columns,
    )

    title = "Locational Marginal Price"
    fig.update_layout(
        title={"text": title},
        xaxis_title="",
        yaxis_title="$/MWh",
    )

    return html.Div(children=[dcc.Graph(figure=fig)], id=GRAPHIC_LMP_TIMESERIES)


@app.callback(
    Output(GRAPHIC_LMP_TIMESERIES, "children"),
    Input(DROPDOWN_SELECT_STATE, "value"),
    Input(RADIO_BUTTON_RESAMPLE, "value"),
    Input(SLIDER_SELECT_TIME, "value"),
)
def plot_lmp_timeseries_callback(
    states: list[str] = ALL_STATES,
    resample: str = "1h",
    weeks: list[int] = [TIMEFRAME.min().week, TIMEFRAME.max().week],
) -> html.Div:

    # plus one because strftime indexs from 0
    timeframe = pd.Series(TIMEFRAME.strftime("%U").astype(int) + 1, index=TIMEFRAME)
    timeframe = timeframe[timeframe.isin(range(weeks[0], weeks[-1], 1))]

    return plot_lmp_timeseries(NETWORK, states, resample, timeframe.index)


def plot_lmp_map(n: pypsa.Network, states: list[str]) -> html.Div:

    df = n.buses_t.marginal_price
    df = df.resample("24h").mean()
    df.index = df.index.map(lambda x: f"{x.month}-{x.day}")
    df = df.T
    df["state"] = df.index.map(n.buses.country)
    df = df[df.state.isin(states)].drop(columns="state")
    df = df.melt(ignore_index=False).reset_index()
    df["x"], df["y"] = df.Bus.map(n.buses.x), df.Bus.map(n.buses.y)

    fig = ff.create_hexbin_mapbox(
        data_frame=df,
        lat="y",
        lon="x",
        nx_hexagon=len(df.Bus.unique()),
        animation_frame="snapshot",
        color_continuous_scale="Cividis",
        min_count=1,
        agg_func=np.mean,
    )
    fig.update_layout(margin=dict(b=0, t=0, l=0, r=0))
    fig.update_layout(mapbox_style="carto-positron")
    fig.layout.sliders[0].pad.t = 20
    fig.layout.updatemenus[0].pad.t = 40

    return html.Div(children=[dcc.Graph(figure=fig)], id=GRAPHIC_LMP_MAP)


@app.callback(
    Output(GRAPHIC_LMP_MAP, "children"),
    Input(DROPDOWN_SELECT_STATE, "value"),
)
def plot_lmp_map_callback(states: list[str]) -> html.Div:
    return plot_lmp_map(NETWORK, states)


###
# APP LAYOUT
###

# tabs


@callback(
    Output(TABS_CONTENT, "children"),
    Input(TABS_UPDATE, "value"),
)
def render_content(tab):
    if tab == TAB_NODES:
        return html.Div(
            [
                html.H3("Spatial Resoluution"),
                html.Div(
                    children=[plot_map_callback()],
                    style={"width": "90%", "display": "inline-block"},
                ),
            ],
        )
    elif tab == TAB_DISPATCH:
        return html.Div(
            [
                html.H3("Dispatch and Load Results"),
                html.Div(
                    children=[
                        plot_dispatch_callback(
                            states=ALL_STATES,
                            resample="1h",
                            weeks=[TIMEFRAME.min().week, TIMEFRAME.max().week],
                        ),
                        plot_load_callback(
                            states=ALL_STATES,
                            resample="1h",
                            weeks=[TIMEFRAME.min().week, TIMEFRAME.max().week],
                        ),
                    ],
                    style={"width": "90%", "display": "inline-block"},
                ),
            ],
        )
    elif tab == TAB_EMISSIONS:
        return html.Div(
            [
                html.H3("Emission Results"),
                html.Div(
                    children=[
                        plot_emissions_fuel_callback(
                            states=ALL_STATES,
                            resample="1h",
                            weeks=[TIMEFRAME.min().week, TIMEFRAME.max().week],
                        ),
                        plot_accumulated_emissions_fuel_callback(
                            states=ALL_STATES,
                            weeks=[TIMEFRAME.min().week, TIMEFRAME.max().week],
                        ),
                        plot_emissions_state_callback(
                            states=ALL_STATES,
                            resample="1h",
                            weeks=[TIMEFRAME.min().week, TIMEFRAME.max().week],
                        ),
                        plot_accumulated_emissions_state_callback(
                            states=ALL_STATES,
                            weeks=[TIMEFRAME.min().week, TIMEFRAME.max().week],
                        ),
                    ],
                    style={"width": "90%", "display": "inline-block"},
                ),
            ],
        )
    elif tab == TAB_CF:
        return html.Div(
            [
                html.H3("Capacity Factor Results"),
                html.Div(
                    children=[plot_cf_callback(states=ALL_STATES)],
                    style={"width": "90%", "display": "inline-block"},
                ),
            ],
        )
    elif tab == TAB_LMP:
        return html.Div(
            [
                html.H3("Locational Marginal Price"),
                html.Div(
                    children=[
                        plot_lmp_timeseries_callback(
                            states=ALL_STATES,
                            resample="1h",
                            weeks=[TIMEFRAME.min().week, TIMEFRAME.max().week],
                        ),
                        plot_lmp_map_callback(states=ALL_STATES),
                    ],
                    style={"width": "90%", "display": "inline-block"},
                ),
            ],
        )


# layout

app.layout = html.Div(
    children=[
        html.H2("PyPSA-USA Dashboard"),
        state_dropdown(states=ALL_STATES),
        select_resample(),
        time_slider(NETWORK.snapshots),
        dcc.Tabs(
            id=TABS_UPDATE,
            value=TAB_DISPATCH,
            children=[
                dcc.Tab(label="Nodes", value=TAB_NODES),
                dcc.Tab(label="Dispatch", value=TAB_DISPATCH),
                dcc.Tab(label="Emissions", value=TAB_EMISSIONS),
                dcc.Tab(label="Capacity Factor", value=TAB_CF),
                dcc.Tab(label="LMP", value=TAB_LMP),
            ],
        ),
        html.Div(id=TABS_CONTENT),
    ],
    style={
        "width": "100%",
        "padding": "10px",
        "display": "inline-block",
        "vertical-align": "top",
    },
)

if __name__ == "__main__":
    app.run(debug=True)
