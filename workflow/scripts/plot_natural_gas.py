import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd

import pypsa

from summary_natural_gas import (
    get_gas_demand,
    get_gas_processing,
    get_imports_exports,
    get_linepack,
    get_underground_storage
)
import constants

MWH_2_MMCF = constants.NG_MWH_2_MMCF

NETWORK = "./workflow/results/texas/networks/elec_s_40_ec_lv1.25_Co2L1.25_E-G.nc"
n = pypsa.Network(NETWORK)

# Initialize the Dash app
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(
    __name__, 
    external_stylesheets=external_stylesheets
)

app.layout = html.Div([
    html.H1("Natural Gas Results"),
    html.Div(
            [
                dcc.Graph(
                    id="NG_DEMAND",
                    figure=px.line(
                        get_gas_demand(n) * MWH_2_MMCF, 
                        title="Gas Demand",
                        labels={"snapshot":"", "value":"Demand (MMCF)", "carrier":"Source"}
                    )
                ),
            ], 
            className="ng_data"
    ),
    html.Div(
            [
                dcc.Graph(
                    id="NG_PROCESSING",
                    figure=px.line(
                        get_gas_processing(n) * MWH_2_MMCF, 
                        title="Gas Processing Capacity",
                        labels={"snapshot":"", "value":"Processing Capacity (MMCF)", "Generator":"State"}
                        )
                ),
            ], 
            className="ng_data"
    ),
    html.Div(
            [
                dcc.Graph(
                    id="NG_LINEPACK",
                    figure=px.line(
                        get_linepack(n) * MWH_2_MMCF, 
                        title="Gas Line Pack",
                        labels={"snapshot":"", "value":"Linepack Capacity (MMCF)", "Store":"State"}
                    )
                ),
            ], 
            className="ng_data"
    ),
    html.Div(
            [
                dcc.Graph(
                    id="NG_STORAGE",
                    figure=px.line(
                        get_underground_storage(n) * MWH_2_MMCF, 
                        title="Gas Underground Storage",
                        labels={"snapshot":"", "value":"Storage Capacity (MMCF)", "Store":"State"}
                        )
                ),
            ], 
            className="ng_data"
    ),
    html.Div(
            [
                dcc.Graph(
                    id="NG_TRADE_DOMESTIC",
                    figure=px.line(
                        get_imports_exports(n, international=False) * MWH_2_MMCF, 
                        title="Gas Trade Domestic",
                        labels={"snapshot":"", "value":"Volume (MMCF)", "Variable":"State"}
                    )
                ),
            ], 
            className="ng_data"
    ),
    html.Div(
            [
                dcc.Graph(
                    id="NG_TRADE_INTERNATIONAL",
                    figure=px.line(
                        get_imports_exports(n, international=True) * MWH_2_MMCF,  
                        title="Gas Trade International",
                        labels={"snapshot":"", "value":"Volume (MMCF)", "variable":"State"}
                    )
                ),
            ], 
            className="ng_data"
    ),

    ])

# Run the app
if __name__ == "__main__":
    
    # n = pypsa.Network()
    app.run_server(debug=True)