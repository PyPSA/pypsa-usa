"""
Dash app for exploring aggregated data.
"""

import logging
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import pypsa
from dash import Dash, Input, Output, callback, dcc, html

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.propagate = False

import calendar
from pathlib import Path
from typing import Dict, List

###
# IDS
###

# dropdowns
DROPDOWN_SELECT_SECTOR = "dropdown_select_sector"
DROPDOWN_SELECT_FUEL = "dropdown_select_fuel"
DROPDOWN_SELECT_YEAR = "dropdown_select_year"

# buttons
BUTTON_SELECT_ALL_SECTORS = "button_all_sectors"
BUTTON_SELECT_ALL_FUELS = "button_all_fuels"

# slider
SLIDER_SELECT_TIME = "slider_select_time"

# radio buttons
RADIO_BUTTON_RESAMPLE = "radio_button_resample"
RADIO_BUTTON_LOAD = "radio_button_load"
RADIO_BUTTON_GROUP_CARRIERS = "radio_button_group_carriers"
RADIO_BUTTON_PLOT_TYPE = "radio_button_plot_type"

# graphics
GRAPHIC_CONTAINER = "graphic_container"
GRAPHIC_MAP = "graphic_map"
GRAPHIC_TIMESERIES = "graphic_timeseries"

# tabs


###
# APP SETUP
###

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
logger.info("Reading configuration options")

# add logic to build network name
network_path = Path(
    "..",
    "results",
    "Default",
    "texas",
    "networks",
    "elec_s_20_ec_lv1.0_500SEG_E-G.nc",
    # "elec_s_40_ec_lv1.0_Ep-Co2L0.2_E.nc",
)
NETWORK = pypsa.Network(str(network_path))

shapes_path = Path(
    "..",
    "resources",
    "Default",
    # "Sector",
    "texas",
    "regions_onshore_s_20.geojson",
)
SHAPES = gpd.read_file(shapes_path).set_index("name")

TIMEFRAME = NETWORK.snapshots

CARRIERS = NETWORK.loads.carrier.unique()

INVESTMENT_YEARS = NETWORK.investment_periods.to_list()

try:  # sector coupled studies
    SECTORS = list({x.split("-")[0] for x in CARRIERS})
    FUELS = list({x.split("-")[1] for x in CARRIERS})

    SECTOR_NICE_NAMES = {
        "res": "Residential",
        "com": "Commercial",
        "ind": "Industrial",
        "trn": "Transportation",
    }

    FUEL_NICE_NAMES = {
        "heat": "Heating",
        "cool": "Cooling",
        "elec": "Electricity",
    }

except IndexError:  # power only studies
    SECTORS = ["pwr"]
    FUELS = ["elec"]

    SECTOR_NICE_NAMES = {
        "pwr": "Power",
    }

    FUEL_NICE_NAMES = {
        "elec": "Electricity",
    }

    NETWORK.buses.carrier = NETWORK.buses.carrier.map({"AC": "pwr-elec"})
    NETWORK.loads.carrier = NETWORK.loads.carrier.map({"AC": "pwr-elec"})

###
# INITIALIZATION
###

logger.info("Starting app")
app = Dash(external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "PyPSA-USA Dashboard"

###
# HELPER FUNCTIONS
###


def get_load_carriers(sector: str, fuels: list[str]) -> list[str]:
    """
    Gets all permutations of load carriers.
    """

    carriers = []
    if not sector == "trn":
        for fuel in fuels:
            carriers.append(f"{sector}-{fuel}")
            if fuel == "heat":
                carriers.append(f"{sector}-urban-{fuel}")
                carriers.append(f"{sector}-rural-{fuel}")
    else:
        for fuel in fuels:
            for vehicle in ("lgt", "med", "hvy", "bus"):
                carriers.append(f"{sector}-{fuel}-{vehicle}")
    return carriers


def get_load_names(n: pypsa.Network, carriers: list[str]) -> list[str]:
    """
    Gets names of loads.
    """
    return n.loads[n.loads.carrier.isin(carriers)].index.to_list()


def get_load_timeseries(n: pypsa.Network, loads: list[str], year: int) -> pd.DataFrame:
    """
    Gets timeseries of select carriers.
    """
    df = n.loads_t.p_set[loads].loc[year]
    df.index = pd.to_datetime(df.index)
    return df


def resample_load(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("24h").sum()


def filter_on_time(df: pd.DataFrame, doy: pd.Timestamp, year: int) -> pd.Series:
    day = datetime(year, 1, 1) + timedelta(doy - 1)
    assert isinstance(df.index, pd.DatetimeIndex)
    return df[df.index.date == day.date()]


def group_load_on_carriers(n: pypsa.Network, df: pd.DataFrame) -> pd.DataFrame:
    bus_2_carrier = n.buses.carrier.to_dict()
    return df.rename(columns=bus_2_carrier).T.groupby(level=0).sum().T


def group_sum_by_carrier(n: pypsa.Network, df: pd.Series) -> pd.Series:
    """
    Groups carriers to bus.
    """
    loads = df.copy()
    loads.name = "value"
    loads = loads.reset_index()
    loads["carrier"] = loads.Load.map(n.buses.carrier)
    loads["region"] = loads.apply(
        lambda row: row.Load.split(row.carrier)[0].rstrip(),
        axis=1,
    )
    loads = loads.drop(columns=["Load", "carrier"])
    return loads.groupby("region").sum()


def group_by_bus(n: pypsa.Network, df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups carriers to bus.
    """
    return df.rename(columns=n.loads.bus).T.groupby(level=0).sum().T


def format_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    load = df.melt(ignore_index=False).reset_index()
    load["hour"] = load.timestep.map(lambda x: x.hour)
    load = load.rename(columns={"Load": "region"})
    return load[["region", "value", "hour"]]


###
# CALLBACK FUNCTIONS
###


def time_slider(snapshots: pd.DatetimeIndex) -> html.Div:
    min_day = snapshots.min()[1].timetuple().tm_yday
    max_day = snapshots.max()[1].timetuple().tm_yday
    interval = round((max_day - min_day) / 50)
    return html.Div(
        children=[
            html.H4("Day To Plot"),
            dcc.Slider(
                id=SLIDER_SELECT_TIME,
                min=min_day,
                max=max_day,
                step=None,
                value=min_day,
                marks={
                    x: str(x) if (x % interval == 0) else ""
                    for x in range(min_day, max_day, 1)
                },
            ),
        ],
    )


def sector_dropdown(sector: str) -> html.Div:
    return html.Div(
        children=[
            html.H4("Sectors"),
            dcc.Dropdown(
                id=DROPDOWN_SELECT_SECTOR,
                options=SECTOR_NICE_NAMES,
                value=sector,
                multi=False,
                persistence=True,
            ),
        ],
    )


def investment_year_dropdown(investment_year: int) -> html.Div:
    return html.Div(
        children=[
            html.H4("Investment Year"),
            dcc.Dropdown(
                id=DROPDOWN_SELECT_YEAR,
                options=INVESTMENT_YEARS,
                value=investment_year,
                multi=False,
                persistence=True,
            ),
        ],
    )


def fuel_dropdown(fuels: list[str]) -> html.Div:
    return html.Div(
        children=[
            html.H4("Fuels"),
            dcc.Dropdown(
                id=DROPDOWN_SELECT_FUEL,
                options=FUEL_NICE_NAMES,
                value=fuels,
                multi=True,
                persistence=True,
            ),
            # html.Button(
            #     children=["Select All"],
            #     id=BUTTON_SELECT_ALL_FUELS,
            #     n_clicks=0,
            # ),
        ],
    )


# @app.callback(
#     Output(DROPDOWN_SELECT_FUEL, "value"),
#     Input(BUTTON_SELECT_ALL_FUELS, "n_clicks"),
# )
# def select_all_fuels(_: int) -> list[str]:
#     return FUELS


def group_carriers_radio() -> html.Div:
    return html.Div(
        children=[
            html.H4("Group Carriers"),
            dcc.RadioItems(
                id=RADIO_BUTTON_GROUP_CARRIERS,
                options=[
                    {"label": "True", "value": True},
                    {"label": "False", "value": False},
                ],
                value=True,
                inline=True,
            ),
        ],
    )


def plot_type_radio() -> html.Div:
    return html.Div(
        children=[
            html.H4("Plot Type"),
            dcc.RadioItems(
                id=RADIO_BUTTON_PLOT_TYPE,
                options=[
                    {"label": "Line", "value": "line"},
                    {"label": "Area", "value": "area"},
                ],
                value="line",
                inline=True,
            ),
        ],
    )


def plot_load(
    n: pypsa.Network,
    shapes: gpd.GeoDataFrame,
    sector: str,
    fuels: list[str],
    year: int,
    doy: pd.Timestamp,
    group_carriers: bool,
    plot_type: str,
) -> html.Div:

    carriers = get_load_carriers(sector, fuels)
    load_names = get_load_names(n, carriers)
    loads = get_load_timeseries(n, load_names, year)
    loads = filter_on_time(loads, doy, year)

    return html.Div(
        [
            html.Div(
                [plot_map(n, shapes, loads)],
                style={
                    "width": "48%",
                    "display": "inline-block",
                    "padding": "5px",
                },
            ),
            html.Div(
                [plot_timeseries(n, loads, plot_type, group_carriers)],
                style={
                    "width": "48%",
                    "display": "inline-block",
                    "padding": "5px",
                },
            ),
        ],
        id=GRAPHIC_CONTAINER,
    )


@app.callback(
    Output(GRAPHIC_CONTAINER, "children"),
    Input(DROPDOWN_SELECT_SECTOR, "value"),
    Input(DROPDOWN_SELECT_FUEL, "value"),
    Input(DROPDOWN_SELECT_YEAR, "value"),
    Input(SLIDER_SELECT_TIME, "value"),
    Input(RADIO_BUTTON_GROUP_CARRIERS, "value"),
    Input(RADIO_BUTTON_PLOT_TYPE, "value"),
)
def plot_load_callback(
    sector: str = SECTORS[0],
    fuels: list[str] = FUELS,
    year: int = INVESTMENT_YEARS[0],
    doy: pd.DatetimeIndex = NETWORK.snapshots.min()[1].timetuple().tm_yday,
    group_carriers: bool = True,
    plot_type: str = "line",
) -> html.Div:
    return plot_load(
        NETWORK,
        SHAPES,
        sector,
        fuels,
        year,
        doy,
        group_carriers,
        plot_type,
    )


def plot_map(n: pypsa.Network, shapes: gpd.GeoDataFrame, df: pd.DataFrame) -> html.Div:

    loads = df.sum()  # get daily total

    if not loads.empty:
        loads = group_sum_by_carrier(n, loads)
        gdf = shapes.join(loads)
    else:
        gdf = shapes.copy()
        gdf["value"] = 0

    fig = px.choropleth(
        gdf,
        geojson=gdf.geometry,
        locations=gdf.index,
        color="value",
        color_continuous_scale="Viridis",
        scope="usa",
    )

    title = "Load per Modelled Region"
    fig.update_layout(
        title=dict(text=title, font=dict(size=24)),
        # margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    return html.Div(children=[dcc.Graph(figure=fig)], id=GRAPHIC_MAP)


def plot_timeseries(
    n: pypsa.Network,
    df: pd.DataFrame,
    plot_type: str,
    group_carriers: bool,
) -> html.Div:

    load = group_by_bus(n, df)
    if group_carriers:
        load = group_load_on_carriers(n, load)
    load = format_timeseries(load)

    if plot_type == "area":
        fig = px.area(load, x="hour", y="value", color="region")
    else:
        fig = px.line(load, x="hour", y="value", color="region")

    title = "Loads [GW]"
    fig.update_layout(
        title=dict(text=title, font=dict(size=24)),
    )

    return html.Div(children=[dcc.Graph(figure=fig)], id=GRAPHIC_TIMESERIES)


###
# APP LAYOUT
###

app.layout = html.Div(
    children=[
        html.H2("PyPSA-USA Loads Dashboard"),
        time_slider(NETWORK.snapshots),
        html.Div(
            [
                html.Div(
                    [sector_dropdown(SECTORS[0])],
                    style={
                        "width": "15%",
                        "display": "inline-block",
                        "padding": "10px",
                    },
                ),
                html.Div(
                    [fuel_dropdown(FUELS)],
                    style={
                        "width": "25%",
                        "display": "inline-block",
                        "padding": "10px",
                    },
                ),
                html.Div(
                    [investment_year_dropdown(INVESTMENT_YEARS[0])],
                    style={
                        "width": "15%",
                        "display": "inline-block",
                        "padding": "10px",
                    },
                ),
                html.Div(
                    [group_carriers_radio()],
                    style={
                        "width": "15%",
                        "display": "inline-block",
                        "padding": "10px",
                    },
                ),
                html.Div(
                    [plot_type_radio()],
                    style={
                        "width": "15%",
                        "display": "inline-block",
                        "padding": "10px",
                    },
                ),
            ],
        ),
        plot_load_callback(),
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
