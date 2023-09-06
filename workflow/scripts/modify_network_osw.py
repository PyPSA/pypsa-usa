import pandas as pd
import pypsa
import pdb
import os


#Global Variables

#Del Norte Bus Data
del_norte_offshore_bus_id = 2090013
del_norte_onshore_bus_id = 2020531
del_norte_export_cable_id = 104181
del_norte_bus_loc = [-124.2125, 41.7072]

# Humboldt Bus Data
humboldt_offshore_bus_id = 2090012	
humboldt_onshore_bus_id = 2020531
humboldt_export_cable_id = 104180
humboldt_bus_loc = [-124.6128, 40.8825]

#Cape Mendocino Bus Data
cape_mendocino_offshore_bus_id = 2090011
cape_mendocino_onshore_bus_id = 2020531
cape_mendocino_export_cable_id = 104179
cape_mendocino_bus_loc = [-124.6641, 40.2261]

#Morro Bay Bus Data
morro_bay_offshore_bus_id = 2090004
morro_bay_onshore_bus_id = 2023000
morro_bay_export_cable_id = 104172
morro_bay_bus_loc = [-121.7503, 35.6393]

#Diablo Canyon Bus Data
diablo_canyon_offshore_bus_id = 2090003
diablo_canyon_onshore_bus_id = 2026131
diablo_canyon_export_cable_id = 104171
diablo_canyon_bus_loc = [-121.2597, 35.2138]

#Round Mountain Bus ID:
round_mountain_bus_id = '2020316'

#Tesla Substation (Antioch is Closest Matching Substation)
tesla_substation_bus_id = '2021593'


def define_line_types(network):    
    network.line_types.loc["500kvac"] = pd.Series(
        [60, 0.0683, 0.335],
        index=["f_nom", "r_per_length", "x_per_length"],
        )
    
    network.line_types.loc["export_cable"] = pd.Series(
        [60, 0.0683, 0.335],
        index=["f_nom", "r_per_length", "x_per_length"],
        )
    
    network.transformer_types.loc["Rail"] = pd.Series(
        [60, 0.0683, 0.335],
        index=["f_nom", "r_per_length", "x_per_length"],
        )
    return network


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

def add_hvac_500kv(network, line_name, bus0, bus1):
    network.add("Line", 
                name = line_name, 
                bus0=bus0, 
                bus1=bus1,
                r=2.8910114,
                x=84.8225115,
                s_nom=3200,
                type="500kvac",
                carrier = 'AC',
        )
    network.lines.loc[line_name, "interconnect"] = "Western"
    network.lines.loc[line_name, 'v_nom'] = 500

def add_osw_turbines(network, plant_name, capacity,  pu_time_series):

    if pu_time_series.index.shape[0] != network.snapshots.shape[0]:
        #extend a pandas series to the length of the network.snapshots
        ts = pd.Series(index=network.snapshots)
        ts.values[:8760] = pu_time_series.values
        ts.values[8760:] = pu_time_series.values[:24]
        pu_time_series = ts
    else:
        pu_time_series.index = network.snapshots

    network.add("Generator", 
                name= plant_name+"_osw", 
                bus= plant_name+"_floating_sub", 
                carrier= "wind",
                p_nom= capacity,
                marginal_cost=0,
                p_max_pu= pu_time_series.values,
                weight = 1,
                efficiency = 1,
                p_nom_extendable = False,

            )

def add_export_array_module(network, name, export_cable_id, 
                            capacity, offshore_sub_location):
    
    #get export cable data
    export_cable_data = network.lines.loc[str(export_cable_id)]
    old_onshore_bus = network.buses.loc[export_cable_data.bus1]

    # #Add off-shore Bus_id:
    # network.add("Bus",
    #             name = f'{name}_floating_sub',
    #             x = offshore_sub_location[0],
    #             y = offshore_sub_location[1],
    #             v_nom = 230,
    #             carrier = 'AC',
    #             )
    # network.buses.loc[f'{name}_floating_sub', 'substation'] = False
    # network.buses.loc[f'{name}_floating_sub', 'balancing_area'] = 'CISO-PGAE'
    # network.buses.loc[f'{name}_floating_sub', 'country'] = 'US'

    # #Add new onshore bus:
    # network.add("Bus",
    #             name = f'{name}_onshore_bus_230kv',
    #             x = old_onshore_bus.x + 0.001,
    #             y = old_onshore_bus.y + 0.001,
    #             v_nom = 230,
    #             carrier = 'AC',
    #             )
    # network.buses.loc[f'{name}_onshore_bus_230kv', 'substation'] = False
    # network.buses.loc[f'{name}_onshore_bus_230kv', 'balancing_area'] = 'CISO-PGAE'
    # network.buses.loc[f'{name}_onshore_bus_230kv', 'country'] = 'US'

    # #Add new export cable
    # network.add("Line",
    #             name = f'{name}_export_cable',
    #             bus0 = f'{name}_floating_sub',
    #             bus1 = f'{name}_onshore_bus_230kv',
    #             s_nom = capacity,
    #             type = 'export_cable',
    #             carrier = 'AC',
    #             x = 10, #revisit resistance and reactance values later
    #             r = 0.1,
    #             s_nom_extendable = False,
    #             )
    # network.lines.loc[f'{name}_export_cable', 'v_nom'] = 230

    
    # #Add new 500kV bus
    # network.add("Bus",
    #             name = f'{name}_onshore_bus_500kv',
    #             x = old_onshore_bus.x + 0.001,
    #             y = old_onshore_bus.y + 0.001,
    #             v_nom = 500,
    #             carrier = 'AC',
    #             )
    
    # network.buses.loc[f'{name}_onshore_bus_500kv', 'substation'] = False
    # network.buses.loc[f'{name}_onshore_bus_500kv', 'balancing_area'] = 'CISO-PGAE'
    # network.buses.loc[f'{name}_onshore_bus_500kv', 'country'] = 'US'

    # #Add new transformer
    # network.add("Transformer",
    #             name = f'{name}_transformer',
    #             bus0 = f'{name}_onshore_bus_230kv',
    #             bus1 = f'{name}_onshore_bus_500kv',
    #             s_nom = capacity,
    #             type = 'Rail',
    #             x = 10, #revisit resistance and reactance values later
    #             r = 0.1,
    #             )
    
    # network.transformers.loc[f'{name}_transformer', 'country'] = 'US'




# Load the network
# resources_folder= os.path.join(os.path.dirname(os.getcwd()), 'resources')
# network = pypsa.Network(os.path.join(resources_folder, 'western/elec_base_network_l_pp.nc'))

# Load Offshore Wind Time Series Data
# osw_ts = pd.read_csv(snakemake.input.osw_ts, index_col=0, parse_dates=True)
osw_ts = pd.read_csv('/Users/kamrantehranchi/Local_Documents/pypsa-usa/workflow/repo_data/Offshore_Wind_CEC_PLEXOS_2030.csv', 
                        index_col=0, 
                        parse_dates=True
                    )

def build_OSW_base_configuration(network):
    #define network lines
    define_line_types(network)

    # # Add Offshore Substations + Export Cables
    # add_export_array_module(network,
    #                         "humboldt",
    #                         humboldt_export_cable_id,
    #                         capacity = 2000,
    #                         offshore_sub_location = humboldt_bus_loc
    #                         )

    # # Add New Offshore Generators
    # add_osw_turbines(network,
    #                 "humboldt", 
    #                 capacity = 1607,  
    #                 pu_time_series = osw_ts.Wind_Offshore_Humboldt
    #             )
    print('test')


def build_OSW_500kV(network):
    #Alternative 1- 500 kV Overland Option
    # Add Fern Road Substation
    network.add("Bus",
                name = "fern_road_sub",
                x = network.buses.loc[round_mountain_bus_id].x + 0.001,
                y = network.buses.loc[round_mountain_bus_id].y + 0.001,
                v_nom = 500,
                carrier = 'AC',
                )

    # Add 500 kV line from Humboldt Onshore Bus to Fern Road Substation
    add_hvac_500kv(network,
                    line_name="humboldt_fern_road_500kv",
                    bus0 = "humboldt_onshore_bus_500kv",
                    bus1 = "fern_road_sub",
                    )
    # Add transformer connecting Fern Road Substation to Round Mountain Bus
    network.add("Transformer",
                name = "fern_round_mountain_transformer",
                bus0 = "fern_road_sub",
                bus1 = round_mountain_bus_id,
                s_nom = 2000,
                type = 'Rail',
                x = 10, #revisit resistance and reactance values later
                r = 0.1,
                )

    # Alternative 1.1- 500 kV Overland Option strengthening COI
    # Add 500 kV line from Fern Road Substation to Tesla Substation
    network.add("Bus",
                name= "tesla_sub_500kv",
                x = network.buses.loc[tesla_substation_bus_id].x + 0.001,
                y = network.buses.loc[tesla_substation_bus_id].y + 0.001,
                v_nom = 500,
                carrier= 'AC',
                )

    add_hvac_500kv(network,
                    line_name="fern_tesla_500kv",
                    bus0 = "fern_road_sub",
                    bus1 = "tesla_sub_500kv",
                    )

    network.add("Transformer",
                name = "tesla_step_up_transformer",
                bus0 = "tesla_sub_500kv",
                bus1 = tesla_substation_bus_id,
                s_nom = 2000,
                type = 'Rail',
                x = 10, #revisit resistance and reactance values later
                r = 0.1,
                )
    

# Alternative 2- HVDC LCC Overhead Option
def build_hvdc_overhead(network):
    None

# Alternative 3- HVDC VSC Subsea Option
def build_hvdc_subsea(network):
    None
