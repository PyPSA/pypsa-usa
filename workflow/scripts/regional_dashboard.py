"""Dash app for exploring regional disaggregated data"""

import pypsa

from dash import Dash, html, dcc, Input, Output, callback
import plotly.express as px
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import plotly.graph_objects as go

import logging 
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.propagate = False

from pathlib import Path

from typing import List, Dict

from summary import get_demand_timeseries, get_energy_timeseries
from plot_statistics import get_color_palette

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

# radio buttons
RADIO_BUTTON_RESAMPLE = "radio_button_resample"

# graphics 
GRAPHIC_MAP = "graphic_map"
GRAPHIC_DISPATCH = "graphic_dispatch"
GRAPHIC_LOAD = "graphic_load"
GRAPHIC_SOLAR_CF = "graphic_solar_cf"
GRAPHIC_WIND_CF = "graphic_wind_cf"
GRAPHIC_EMISSIONS = "graphic_emissions"
GRAPHIC_CAPACITY = "graphic_capacity"
GRAPHIC_GENERATION = "graphic_generation"

###
# APP SETUP
###

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
logger.info("Reading configuration options")
# add logic to build network name 
network_name = "elec_s_80_ec_lv1.0_RCo2L-SAFER-RPS_E.nc"

# read in network 
network_path = Path("..", "results", "western", "networks")
NETWORK = pypsa.Network(str(Path(network_path, network_name)))

# set up general data structures  
ALL_STATES = NETWORK.buses.country.unique()
NODES_GEOLOCATED = NETWORK.buses
geometry = [Point(xy) for xy in zip(NODES_GEOLOCATED["x"], NODES_GEOLOCATED["y"])]
NODES_GEOLOCATED = gpd.GeoDataFrame(NODES_GEOLOCATED, geometry=geometry, crs='EPSG:4326')[["geometry"]]

###
# INITIALIZATION
###

logger.info("Starting app")
app = Dash(external_stylesheets=external_stylesheets)
app.title = "PyPSA-USA Dashboard"

###
# CALLBACK FUNCTIONS
### 

# select states to include 

def state_dropdown(states: List[str]) -> html.Div:
    return html.Div(
        children=[
            html.H3("States to Include"),
            dcc.Dropdown(
                id=DROPDOWN_SELECT_STATE,
                options=states,
                value=states,
                multi=True,
                persistence=True
            ),
            html.Button(
                children=["Select All"],
                id=BUTTON_SELECT_ALL_STATES,
                n_clicks=0
            )
        ]
    )

@app.callback(
    Output(DROPDOWN_SELECT_STATE, "value"),
    Input(BUTTON_SELECT_ALL_STATES, "n_clicks"),
)
def select_all_countries(_: int) -> list[str]:
    return ALL_STATES

# plot map 

def plot_map(
    n: pypsa.Network,
    states: List[str]
) -> html.Div:
    
    nodes = n.buses[n.buses.country.isin(states)]
    
    usa_map = go.Figure()

    usa_map.add_trace(go.Scattermapbox(
        mode='markers',
        lon=nodes.x,
        lat=nodes.y,
        marker=dict(size=10, color='red'),
        text=nodes.index,
    ))

    # Update layout to include map
    usa_map.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lon=-95, lat=35),
            zoom=3
        ),
        margin=dict(l=0, r=0, t=0, b=0)
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

# dispatch 

def plot_dispatch(
    n: pypsa.Network,
    states: List[str],
    resample: str,
    timeframe: pd.date_range
) -> html.Div:
    energy_mix = get_energy_timeseries(n).mul(1e-3)  # MW -> GW

    # fix battery charge/discharge to only be positive
    if "battery" in energy_mix:
        col_rename = {
            "battery charger": "battery",
            "battery discharger": "battery",
        }
        energy_mix = energy_mix.rename(columns=col_rename)
        energy_mix = energy_mix.T.groupby(level=0).sum().T
        energy_mix["battery"] = energy_mix.battery.map(lambda x: max(0, x))

    energy_mix = energy_mix.rename(columns=n.carriers.nice_name)
    energy_mix["Demand"] = get_demand_timeseries(n).mul(1e-3)  # MW -> GW
    
    energy_mix = energy_mix.index.resample("4h")
    
    color_palette = get_color_palette(n)

    fig = px.area(
        energy_mix,
        x=energy_mix.index,
        y=[c for c in energy_mix.columns if c != "Demand"],
        color_discrete_map=color_palette,
    )
    fig.add_trace(
        go.Scatter(
            x=energy_mix.index,
            y=energy_mix.Demand,
            mode="lines",
            name="Demand",
            line_color="darkblue",
        ),
    )
    title = create_title("Dispatch [GW]", **wildcards)
    fig.update_layout(
        title=dict(text=title, font=dict(size=24)),
        xaxis_title="",
        yaxis_title="Power [GW]",
    )
    return fig

###
# APP LAYOUT 
### 

app.layout = html.Div(
    children=[
        # map section
        html.Div(
            children = [
                html.Div(
                    children=[
                        state_dropdown(states=ALL_STATES),
                    ],
                    style={"width": "30%", "padding": "20px", "display": "inline-block", "vertical-align":"top"},
                ),
                html.Div(
                    children = [
                        plot_map_callback()
                    ],
                    style={"width": "60%", "display": "inline-block"},
                )
            ]
        ),
        # dispatch and load section
        html.Div(
            children=[
                
            ]
        )
        
    ]
)

if __name__ == "__main__":
    app.run(debug=True)

