"""Dash app for exploring regional disaggregated data"""

import pypsa

from dash import Dash, html, dcc, Input, Output, callback
import plotly.express as px
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import logging 
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.propagate = False

from pathlib import Path

from typing import List, Dict

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
    
    usa_map = px.scatter_mapbox(
        nodes,
        lat=nodes.y,
        lon=nodes.x,
        hover_name = nodes.index
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

###
# APP LAYOUT 
### 

app.layout = html.Div(
    [
        html.Div(
            children = [
                state_dropdown(states=ALL_STATES)
            ],
            style={'width': '25%', 'display': 'inline-block'},
        ),
        html.Div(
            children = [
                plot_map_callback()
            ]
        )
    ]
)

if __name__ == "__main__":
    app.run(debug=True)

