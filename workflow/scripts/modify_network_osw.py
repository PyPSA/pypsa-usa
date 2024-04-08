import os

import pandas as pd
import pypsa

# Global Variables

# Del Norte Bus Data
del_norte_offshore_bus_id = 2090013
del_norte_onshore_bus_id = 2020531
del_norte_export_cable_id = 104181
del_norte_bus_loc = [-124.2125, 41.7072]

# Humboldt Bus Data
humboldt_offshore_bus_id = 2090012
humboldt_onshore_bus_id = 2020531
humboldt_export_cable_id = 104180
humboldt_bus_loc = [-124.6128, 40.8825]

# Cape Mendocino Bus Data
cape_mendocino_offshore_bus_id = 2090011
cape_mendocino_onshore_bus_id = 2020531
cape_mendocino_export_cable_id = 104179
cape_mendocino_bus_loc = [-124.6641, 40.2261]

# Morro Bay Bus Data
morro_bay_offshore_bus_id = 2090004
morro_bay_onshore_bus_id = 2023000
morro_bay_export_cable_id = 104172
morro_bay_bus_loc = [-121.7503, 35.6393]

# Diablo Canyon Bus Data
diablo_canyon_offshore_bus_id = 2090003
diablo_canyon_onshore_bus_id = 2026131
diablo_canyon_export_cable_id = 104171
diablo_canyon_bus_loc = [-121.2597, 35.2138]

# Humboldt Bus ID:
eureka_bus_id = "2020530"

# Round Mountain Bus ID:
round_mountain_bus_id = "2020316"

# Tesla Substation (Antioch is Closest Matching Substation)
tesla_substation_bus_id = "2021593"

# SF 345kB Substation:
sf_345kv_substation_bus_id = "2021181"

# Pittsburg 765kV Substation:
pittsburg_substation_bus_id = "2021641"


def add_osw_turbines(network, plant_name, capacity, pu_time_series):

    if pu_time_series.index.shape[0] != network.snapshots.shape[0]:
        # extend a pandas series to the length of the network.snapshots
        ts = pd.Series(index=network.snapshots)
        ts_len = ts.shape[0]
        ts.values[:] = pu_time_series.values[:ts_len]
        ts.fillna(0, inplace=True)
        pu_time_series = ts
    else:
        pu_time_series.index = network.snapshots

    network.add(
        "Generator",
        name=plant_name + "_osw",
        bus=eureka_bus_id,
        carrier="offwind_floating",
        p_nom=capacity,
        marginal_cost=0,
        capital_cost=1e9,
        p_max_pu=pu_time_series.values,
        efficiency=1,
        p_nom_extendable=False,
    )
    network.generators.loc[plant_name + "_osw", "weight"] = 1


# Load Offshore Wind Time Series Data
primary_path = os.path.join(
    os.getcwd(),
    "../repo_data/plants/Offshore_Wind_CEC_PLEXOS_2030.csv",
)
secondary_path = os.path.join(
    os.getcwd(),
    "repo_data/plants/Offshore_Wind_CEC_PLEXOS_2030.csv",
)
# Check if the primary path exists
if os.path.exists(primary_path):
    osw_ts_path = primary_path
else:
    osw_ts_path = secondary_path

osw_ts = pd.read_csv(
    osw_ts_path,
    index_col=0,
    parse_dates=True,
)


def build_OSW_base_configuration(network, osw_capacity):
    """
    Adding the initial buses, export cables, and transformers to the network.
    """
    # Add New Offshore Generators
    add_osw_turbines(
        network,
        "humboldt",
        capacity=osw_capacity,
        pu_time_series=osw_ts.Wind_Offshore_Humboldt,
    )


def build_OSW_500kV(network):
    # Alternative 1- 500 kV Overland Option
    # Add Fern Road Substation
    network.add(
        "Bus",
        name="fern_road_sub",
        x=network.buses.loc[round_mountain_bus_id].x + 0.001,
        y=network.buses.loc[round_mountain_bus_id].y + 0.001,
        v_nom=500,
        carrier="AC",
    )
    network.buses.loc["fern_road_sub", "sub_id"] = 503

    # Add 500 kV line from Humboldt Onshore Bus to Fern Road Substation
    add_hvac_500kv(
        network,
        line_name="humboldt_fern_road_500kv",
        bus0="humboldt_onshore_bus_500kv",
        bus1="fern_road_sub",
    )
    # Add transformer connecting Fern Road Substation to Round Mountain Bus
    network.add(
        "Transformer",
        name="fern_round_mountain_transformer",
        bus0="fern_road_sub",
        bus1=round_mountain_bus_id,
        s_nom=2000,
        type="Rail",
        x=10,  # revisit resistance and reactance values later
        r=0.1,
    )
    network.transformers.loc["fern_round_mountain_transformer", "carrier"] = "AC"

    # Alternative 1.1- 500 kV Overland Option strengthening COI
    # Add 500 kV line from Fern Road Substation to Tesla Substation
    network.add(
        "Bus",
        name="tesla_sub_500kv",
        x=network.buses.loc[tesla_substation_bus_id].x + 0.001,
        y=network.buses.loc[tesla_substation_bus_id].y + 0.001,
        v_nom=500,
        carrier="AC",
    )
    network.buses.loc["tesla_sub_500kv", "sub_id"] = 502

    add_hvac_500kv(
        network,
        line_name="fern_tesla_500kv",
        bus0="fern_road_sub",
        bus1="tesla_sub_500kv",
    )

    network.add(
        "Transformer",
        name="tesla_step_up_transformer",
        bus0="tesla_sub_500kv",
        bus1=tesla_substation_bus_id,
        s_nom=2000,
        type="Rail",
        x=10,  # revisit resistance and reactance values later
        r=0.1,
    )
    network.transformers.loc["tesla_step_up_transformer", "carrier"] = "AC"


# Alternative 2- HVDC LCC Overhead Option
def build_hvdc_overhead(network):
    network.add(
        "Bus",
        name="Pittsburg_500kV",
        x=network.buses.loc[pittsburg_substation_bus_id].x + 0.001,
        y=network.buses.loc[pittsburg_substation_bus_id].y + 0.001,
        v_nom=500,
        carrier="AC",
    )
    network.buses.loc["BayHub_500kV", "sub_id"] = 504

    network.add(
        "Transformer",
        name="Pittsburg_transformer",
        bus0="Pittsburg_500kV",
        bus1=pittsburg_substation_bus_id,
        s_nom=2000,
        type="Rail",
        x=10,  # revisit resistance and reactance values later
        r=0.1,
    )
    network.transformers.loc["Pittsburg_500kV", "carrier"] = "AC"

    add_hvdc_overhead(
        network,
        "HVDC_Humboldt_OverheadLink",
        "humboldt_onshore_bus_500kv",
        "Pittsburg_500kV",
    )


# Alternative 3- HVDC VSC Subsea Option
def build_hvdc_subsea(network):
    network.add(
        "Bus",
        name="BayHub_500kV",
        x=network.buses.loc[sf_345kv_substation_bus_id].x + 0.001,
        y=network.buses.loc[sf_345kv_substation_bus_id].y + 0.001,
        v_nom=500,
        carrier="AC",
    )
    network.buses.loc["BayHub_500kV", "sub_id"] = 504

    network.add(
        "Transformer",
        name="BayHub_transformer",
        bus0="BayHub_500kV",
        bus1=sf_345kv_substation_bus_id,
        s_nom=2000,
        type="Rail",
        x=10,  # revisit resistance and reactance values later
        r=0.1,
    )
    network.transformers.loc["BayHub_transformer", "carrier"] = "AC"

    add_hvdc_subsea(
        network,
        "HVDC_Humboldt_SubseaLink",
        "humboldt_onshore_bus_500kv",
        "BayHub_500kV",
    )


def add_hvdc_subsea(network, line_name, bus0, bus1):
    network.add(
        "Link",
        name=line_name,
        bus0=bus0,
        bus1=bus1,
        type="Rail",
        # type="HVDC_VSC",
        carrier="DC",
        efficiency=1,
        p_nom=2000,
        p_min_pu=-1,
    )


def add_hvdc_overhead(network, line_name, bus0, bus1):
    network.add(
        "Link",
        name=line_name,
        bus0=bus0,
        bus1=bus1,
        type="Rail",
        carrier="DC",
        efficiency=1,
        p_nom=3000,
        p_min_pu=-1,
    )


def add_hvac_500kv(network, line_name, bus0, bus1):
    network.add(
        "Line",
        name=line_name,
        bus0=bus0,
        bus1=bus1,
        r=2.8910114,
        x=84.8225115,
        s_nom=3200,
        type="Rail",
        carrier="AC",
    )
    network.lines.loc[line_name, "interconnect"] = "Western"
    network.lines.loc[line_name, "v_nom"] = 500


def add_export_array_module(
    network,
    name,
    export_cable_id,
    capacity,
    offshore_sub_location,
):

    # get export cable data
    export_cable_data = network.lines.loc[str(export_cable_id)]
    old_onshore_bus = network.buses.loc[export_cable_data.bus1]

    # Add off-shore Bus_id:
    network.add(
        "Bus",
        name=f"{name}_floating_sub",
        x=offshore_sub_location[0],
        y=offshore_sub_location[1],
        v_nom=230,
        carrier="AC",
    )
    network.buses.loc[f"{name}_floating_sub", "substation"] = False
    network.buses.loc[f"{name}_floating_sub", "balancing_area"] = "CISO-PGAE"
    network.buses.loc[f"{name}_floating_sub", "country"] = "US"
    network.buses.loc[f"{name}_floating_sub", "sub_id"] = 231

    # Add new onshore bus:
    network.add(
        "Bus",
        name=f"{name}_onshore_bus_230kv",
        x=old_onshore_bus.x + 0.001,
        y=old_onshore_bus.y + 0.001,
        v_nom=230,
        carrier="AC",
    )
    network.buses.loc[f"{name}_onshore_bus_230kv", "substation"] = False
    network.buses.loc[f"{name}_onshore_bus_230kv", "balancing_area"] = "CISO-PGAE"
    network.buses.loc[f"{name}_onshore_bus_230kv", "country"] = "US"
    network.buses.loc[f"{name}_onshore_bus_230kv", "sub_id"] = 232

    # Add new export cable
    network.add(
        "Line",
        name=f"{name}_export_cable",
        bus0=f"{name}_floating_sub",
        bus1=f"{name}_onshore_bus_230kv",
        s_nom=capacity,
        type="Rail",
        # type = 'export_cable',
        carrier="AC",
        x=10,  # revisit resistance and reactance values later
        r=0.1,
        s_nom_extendable=False,
    )
    network.lines.loc[f"{name}_export_cable", "v_nom"] = 230

    # Add new 500kV bus
    network.add(
        "Bus",
        name=f"{name}_onshore_bus_500kv",
        x=old_onshore_bus.x + 0.001,
        y=old_onshore_bus.y + 0.001,
        v_nom=500,
        carrier="AC",
    )
    network.buses.loc[f"{name}_onshore_bus_500kv", "substation"] = False
    network.buses.loc[f"{name}_onshore_bus_500kv", "balancing_area"] = "CISO-PGAE"
    network.buses.loc[f"{name}_onshore_bus_500kv", "country"] = "US"
    network.buses.loc[f"{name}_onshore_bus_500kv", "sub_id"] = 501

    # Add new transformer
    network.add(
        "Transformer",
        name=f"{name}_transformer",
        bus0=f"{name}_onshore_bus_230kv",
        bus1=f"{name}_onshore_bus_500kv",
        s_nom=capacity,
        type="Rail",
        x=10,  # revisit resistance and reactance values later
        r=0.1,
    )

    network.transformers.loc[f"{name}_transformer", "carrier"] = "AC"
