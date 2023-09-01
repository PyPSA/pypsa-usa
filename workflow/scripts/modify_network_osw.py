import logging
from itertools import product

import geopandas as gpd
import numpy as np
import pandas as pd

import pypsa
import scipy.sparse as sparse
import xarray as xr
from _helpers import configure_logging, update_p_nom_max

from shapely.prepared import prep
import pdb

idx = pd.IndexSlice

logger = logging.getLogger(__name__)

def add_hvdc_subsea(network, line_name, bus0, bus1):
    network.add("Link", 
                name= line_name, 
                bus0=bus0, 
                bus1=bus1,
                type="HVDC_VSC", 
                carrier = "DC",
                efficiency=1,
                p_nom=2000,
                p_min_pu=-1,
            )
    return network

def add_hvdc_overhead(network, line_name, bus0, bus1):
    network.add("Link",
            name = line_name,
            bus0=bus0, 
            bus1=bus1,
            type="HVDC_LCC", 
            carrier = "DC",
            efficiency=1,
            p_nom=3000,
            p_min_pu=-1,
        )
    return network

def add_hvac_500kv(network, bus0, bus1):
    network.add("Line", 
                name = line_name, 
                bus0=bus0, 
                bus1=bus1,
                r=0.815803,
                x=6.873208,
                s_nom=3200,
                type="500kvac",
        )
    return network


def add_osw_turbines():
    network.add("Generator", "Humboldt_OSW2", bus= "CISO-PGAE0 13", carrier="wind",
            p_nom=1000,
            marginal_cost=0,
            p_max_pu=1,
            )
    return network


def define_line_types(network):    
    network.line_types.loc["500kvac"] = pd.Series(
        [60, 0.0683, 0.335],
        index=["f_nom", "r_per_length", "x_per_length"],
        )
    return network

def load_osw_profiles():
    return osw_data

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("modify_network_osw")
    configure_logging(snakemake)

    # network = pypsa.Network(snakemake.input.network)

    # Load the network
    resources_folder= os.path.join(os.path.dirname(os.getcwd()), 'resources')
    network = pypsa.Network(os.path.join(resources_folder, 'western/elec_base_network_l_pp.nc'))


    network.buses[network.buses.balancing_area == '']

    network.buses[network.buses.sub_id == 36073]


    humboldt_offshore_bus_id = 2090012	
    humbolt_onshore_bus_id = 2020530

    #offshore bus
    network.buses[network.buses.index == humboldt_offshore_bus_id.__str__()]
    network.lines[network.lines.bus0 == humboldt_offshore_bus_id.__str__()]

    #onshore bus
    network.buses[network.buses.index == humbolt_onshore_bus_id.__str__()]
    network.lines[network.lines.bus1 == humbolt_onshore_bus_id.__str__()]
    network.transformers[network.transformers.bus0 == '2020531']

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.histplot(data=network.transformers, x="s_nom", hue="v_nom", multiple="stack")
    # plt.xlim(left=500)
    plt.ylim(top=100)



