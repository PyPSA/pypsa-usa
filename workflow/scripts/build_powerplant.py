import os
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np

def load_eia_data_old(filter_region = None, base_dir = '..', version = ""):
    DATA_PATH = os.path.join(base_dir)
    REPO_DATA_PATH = os.path.join(base_dir)

    EIA__GEN_FILE = os.path.join(DATA_PATH,  f"3_1_Generator_{version}.xlsx" )
    EIA_PLANT_FILE = os.path.join(DATA_PATH, f"2___Plant_{version}.xlsx" )
    EIA_STORAGE_FILE = os.path.join(DATA_PATH, f"3_4_Energy_Storage_{version}.xlsx" )

    EIA_TECH_FILE = os.path.join(REPO_DATA_PATH,'eia_mappings', "eia_tech_mapping.csv" )
    EIA_FUEL_FILE = os.path.join(REPO_DATA_PATH,'eia_mappings', "eia_fuel_mapping.csv" )
    EIA_PRIMEMOVER_FILE = os.path.join(REPO_DATA_PATH,'eia_mappings',"eia_primemover_mapping.csv" )


    gen_cols = ["Plant Code", "Plant Name", "Generator ID", "Operating Year", "Nameplate Capacity (MW)","Summer Capacity (MW)", "Winter Capacity (MW)", "Minimum Load (MW)", "Energy Source 1", "Technology", "Status", "Prime Mover"]
    plant_cols = ["Plant Code",'NERC Region', 'Balancing Authority Code', "State", "Latitude" ,"Longitude"]
    storage_cols = ['Plant Code','Generator ID', 'Nameplate Energy Capacity (MWh)','Maximum Charge Rate (MW)', 'Maximum Discharge Rate (MW)','Storage Technology 1',]

    eia_data_operable = pd.read_excel(EIA__GEN_FILE, sheet_name="Operable", skiprows = 2, usecols=gen_cols) #, dtype=str)
    eia_storage = pd.read_excel(EIA_STORAGE_FILE, sheet_name="Operable", skiprows = 2, usecols=storage_cols)

    eia_loc = pd.read_excel(EIA_PLANT_FILE, sheet_name="Plant", skiprows = 2, usecols=plant_cols)
    eia_loc.rename(columns={'Plant Code': 'plant_id_eia'}, inplace=True)

    eia_tech_map = pd.read_csv(EIA_TECH_FILE, index_col = "Technology")
    eia_fuel_map = pd.read_csv(EIA_FUEL_FILE, index_col = "Energy Source 1")
    eia_primemover_map = pd.read_csv(EIA_PRIMEMOVER_FILE, index_col = "Prime Mover")
    tech_dict = dict(zip(eia_tech_map.index, eia_tech_map.primary_fuel.values))
    fuel_dict = dict(zip(eia_fuel_map.index, eia_fuel_map.primary_fuel.values))
    primemover_dict = dict(zip(eia_primemover_map.index, eia_primemover_map.prime_mover.values))

    #modify storage data
    eia_storage['plant_id_eia'] = eia_storage['Plant Code'].astype(int)
    eia_storage.rename(columns={'Plant Name': 'plant_name_eia','Generator ID':'generator_id','Nameplate Energy Capacity (MWh)':'energy_capacity_mwh','Maximum Charge Rate (MW)':'max_charge_rate_mw', 'Maximum Discharge Rate (MW)':'max_discharge_rate_mw'}, inplace=True)
    eia_storage.drop(columns = ['Plant Code',], inplace=True)
    eia_storage['energy_capacity_mwh'] = eia_storage['energy_capacity_mwh'].replace(' ', np.nan).astype(float)
    eia_storage['max_charge_rate_mw'] = eia_storage['max_charge_rate_mw'].replace(' ', np.nan).astype(float)
    eia_storage['max_discharge_rate_mw'] = eia_storage['max_discharge_rate_mw'].replace(' ', np.nan).astype(float)

    #modify data operable
    eia_data_operable.dropna(how='all', inplace=True)
    eia_data_operable['tech_type'] = eia_data_operable['Technology'].map(tech_dict)
    eia_data_operable['fuel_type'] = eia_data_operable['Energy Source 1'].map(fuel_dict)
    eia_data_operable['prime_mover'] = eia_data_operable['Prime Mover'].map(primemover_dict)
    eia_data_operable['plant_id_eia'] = eia_data_operable['Plant Code'].astype(int)
    eia_data_operable.rename(columns={'Plant Name': 'plant_name_eia','Generator ID':'generator_id', 'Nameplate Capacity (MW)':'capacity_mw','Summer Capacity (MW)':'summer_capacity_mw','Winter Capacity (MW)':'winter_capacity_mw','Minimum Load (MW)':'p_nom_min','Operating Year':'operating_year'}, inplace=True)
    eia_data_operable.drop(columns = ['Technology', 'Energy Source 1', 'Plant Code','Prime Mover'], inplace=True)
    eia_data_operable['capacity_mw'] = eia_data_operable['capacity_mw'].replace(' ', 0).fillna(0).astype(float)
    eia_data_operable['summer_capacity_mw'] = eia_data_operable['summer_capacity_mw'].replace(' ', 0).fillna(0).astype(float)
    eia_data_operable['winter_capacity_mw'] = eia_data_operable['winter_capacity_mw'].replace(' ', 0).fillna(0).astype(float)
    eia_data_operable['p_nom_min'] = eia_data_operable['p_nom_min'].replace(' ', 0).fillna(0).astype(float)
    eia_data_operable['operating_year'] = eia_data_operable['operating_year'].replace(' ', -1).fillna(-1).astype(int)

    # Merge locations and plant data
    eia_plants_locs = pd.merge(eia_data_operable, eia_loc, on='plant_id_eia', how='inner')
    if filter_region is not None:
        eia_plants_locs = eia_plants_locs[eia_plants_locs['NERC Region']== filter_region]
    eia_plants_locs.plant_id_eia =eia_plants_locs.plant_id_eia.astype(int)

    eia_plants_locs = pd.merge(eia_plants_locs, eia_storage, on=['plant_id_eia', 'generator_id'], how='left')


    return eia_data_operable, eia_storage, eia_loc, eia_plants_locs


def standardize_col_names(columns, prefix="", suffix=""):
    """Standardize column names by removing spaces, converting to lowercase, removing parentheses, and adding prefix and suffix."""
    return [prefix + col.lower().replace(" ", "_").replace("(", "").replace(")", "") + suffix for col in columns]

def convert_mixed_types_and_floats(df):
    """
    Convert columns with mixed types to string type, and columns with float types
    without decimal parts to integer type in a DataFrame.

    Parameters:
    - df: pandas.DataFrame - The DataFrame to process.

    Returns:
    - pandas.DataFrame: The DataFrame with mixed type columns converted to strings
      and float columns without decimals converted to integers.
    """
    # Attempt to standardize types where possible
    df = df.infer_objects()

    # Function to check and convert mixed type column to string
    def convert_if_mixed(col):
        # Detect if the column has mixed types (excluding NaN values)
        if not all(col.apply(type).eq(col.apply(type).iloc[0])):
            return col.astype(str)
        return col

    # Apply the conversion function to each column in the DataFrame
    df = df.apply(convert_if_mixed)
    return df

data_path = '../data/eia8602022'
version= "Y2022"
repo_data_path = os.path.join('../data/', 'eia_mappings')

sheets = {
    "gen": f"3_1_Generator_{version}.xlsx",
    "plant": f"2___Plant_{version}.xlsx",
    "storage": f"3_4_Energy_Storage_{version}.xlsx",
    "tech": "eia_tech_mapping.csv",
    "fuel": "eia_fuel_mapping.csv",
    "primemover": "eia_primemover_mapping.csv",
}
gen_cols = [
    'plant_code',
    'plant_name',
    'generator_id',
    'operating_year',
    'nameplate_capacity_mw',
    'summer_capacity_mw',
    'winter_capacity_mw',
    'minimum_load_mw',
    'energy_source_1',
    'technology',
    'status',
    'prime_mover',
    'operating_month',
    'operating_year',
    'planned_retirement_month',
    'planned_retirement_year',
    ]
locs_cols = [
    'plant_code',
    'nerc_region',
    'balancing_authority_code',
    'state',
    'latitude',
    'longitude']
storage_cols = [
    'plant_code',
    'generator_id',
    'nameplate_energy_capacity_mwh',
    'maximum_charge_rate_mw',
    'maximum_discharge_rate_mw',
    'storage_technology_1']

# Load plant data
eia_data_operable = pd.read_excel(os.path.join(data_path, sheets["gen"]), sheet_name="Operable", skiprows=1)
eia_data_operable.columns = standardize_col_names(eia_data_operable.columns)
eia_data_operable = eia_data_operable.loc[:,gen_cols]
eia_data_operable = convert_mixed_types_and_floats(eia_data_operable)

# Load Storage
eia_storage = pd.read_excel(os.path.join(data_path, sheets["storage"]), sheet_name="Operable", skiprows=1)
eia_storage.columns = standardize_col_names(eia_storage.columns)
eia_storage = eia_storage.loc[:,storage_cols]
eia_storage = convert_mixed_types_and_floats(eia_storage)

# Convert datatypes
eia_storage.nameplate_energy_capacity_mwh = eia_storage.nameplate_energy_capacity_mwh.str.replace(' ', '')
eia_storage.nameplate_energy_capacity_mwh = eia_storage.nameplate_energy_capacity_mwh.replace('', np.nan).astype(float)
eia_storage.maximum_charge_rate_mw = eia_storage.maximum_charge_rate_mw.replace(' ',np.nan).astype(float)
eia_storage.maximum_discharge_rate_mw = eia_storage.maximum_discharge_rate_mw.replace(' ',np.nan).astype(float)
eia_data_operable.summer_capacity_mw = eia_data_operable.summer_capacity_mw.replace(' ',np.nan).astype(float)
eia_data_operable.winter_capacity_mw = eia_data_operable.winter_capacity_mw.replace(' ',np.nan).astype(float)


# Merge storage and generator data
eia_data_operable = eia_data_operable.merge(eia_storage, on=['plant_code', 'generator_id'], how='outer')
eia_data_operable.dropna(subset='plant_code', inplace=True)
eia_data_operable



# Load sheet with gps locations
eia_locs = pd.read_excel(os.path.join(data_path, sheets["plant"]), sheet_name="Plant", skiprows=1)
eia_locs.columns = standardize_col_names(eia_locs.columns)
eia_locs = eia_locs.loc[:,locs_cols]
eia_locs = convert_mixed_types_and_floats(eia_locs)
eia_locs.loc[eia_locs.state.isin(['AK', 'HI']), 'nerc_region'] = 'non-conus'
eia_locs.loc[eia_locs.state.isin(['AK', 'HI']), 'balancing_authority_code'] = 'non-conus'
eia_locs

# Assign PyPSA Carrier Names, Fuel Types, and Prime Movers Names
# Load technology type and fuel type mapping files
eia_tech_map = pd.read_csv(os.path.join(repo_data_path, sheets["tech"]), index_col="Technology")
eia_fuel_map = pd.read_csv(os.path.join(repo_data_path, sheets["fuel"]), index_col="Energy Source 1")
eia_primemover_map = pd.read_csv(os.path.join(repo_data_path, sheets["primemover"]), index_col="Prime Mover")

# Map technologies, fuels, and prime movers
maps = {
    "carrier": (eia_data_operable["technology"], eia_tech_map["tech_type"]),
    "fuel_type": (eia_data_operable["energy_source_1"], eia_fuel_map["fuel_type"]),
    "fuel_name": (eia_data_operable["energy_source_1"], eia_fuel_map["fuel_name"]),
    "prime_mover_name": (eia_data_operable["prime_mover"], eia_primemover_map["prime_mover"]),
}
for col, (data_col, map_df) in maps.items():
    eia_data_operable[col] = data_col.map(dict(zip(map_df.index, map_df.values)))

# eia_data_operable.rename(columns={"technology": "tech_name"}, inplace=True)
eia_data_operable


# Assign Summer and Winter Capacity Derate Factors
eia_data_operable['summer_derate'] = 1 - ((eia_data_operable.nameplate_capacity_mw - eia_data_operable.summer_capacity_mw) / eia_data_operable.nameplate_capacity_mw)
eia_data_operable['winter_derate'] = 1 - ((eia_data_operable.nameplate_capacity_mw - eia_data_operable.winter_capacity_mw) / eia_data_operable.nameplate_capacity_mw)
eia_data_operable.summer_derate = eia_data_operable.summer_derate.clip(upper=1).clip(lower=0)
eia_data_operable.winter_derate = eia_data_operable.winter_derate.clip(upper=1).clip(lower=0)
eia_data_operable

#Examine carrier assignments
non_matching = eia_data_operable[eia_data_operable.carrier != eia_data_operable.fuel_type]
pivot = non_matching.pivot_table(index=['fuel_type','fuel_name'], columns=['carrier','prime_mover_name'], values='nameplate_capacity_mw', aggfunc='sum', fill_value=0)
pivot

eia_plants_locs = eia_data_operable.merge(eia_locs, on="plant_code", how="left")
eia_plants_locs.dropna(subset=['plant_code'], inplace= True)
eia_plants_locs

eia_data_operable[['carrier', 'nameplate_capacity_mw']].groupby('carrier').sum().plot(kind='bar', title='Total Nameplate Capacity by Technology Type (MW)')

egrid_unit = pd.read_excel('../data/egrid2022_data.xlsx', sheet_name='UNT22', skiprows=1)
egrid_gen = pd.read_excel('../data/egrid2022_data.xlsx', sheet_name='GEN22', skiprows=1)
egrid_unit = egrid_unit[['PNAME', 'ORISPL', 'UNITID', 'PRMVR', 'FUELU1', 'HTIAN']]
egrid_gen = egrid_gen[['ORISPL', 'GENID', 'GENNTAN']]

egrid_unit_gen = pd.merge(egrid_unit, egrid_gen, left_on=['ORISPL', 'UNITID'], right_on=['ORISPL', 'GENID'], how='inner')
egrid_unit_gen.dropna(subset=['HTIAN','GENNTAN'], inplace=True)
egrid_unit_gen['unit_avg_heat_rate'] = np.abs(egrid_unit_gen.HTIAN / egrid_unit_gen.GENNTAN)
egrid_unit_gen.unit_avg_heat_rate = egrid_unit_gen.unit_avg_heat_rate.replace(np.inf, np.nan)
egrid_unit_gen = egrid_unit_gen[~egrid_unit_gen.FUELU1.isin(['SUN', 'WND'])]
egrid_unit_gen.sort_values(by='unit_avg_heat_rate', ascending=False)

egrid_plant = pd.read_excel('../data/egrid2022_data.xlsx', sheet_name='PLNT22', skiprows=1)
egrid_plant = egrid_plant[['ORISPL','PLHTIAN', 'PLNGENAN']]
egrid_plant.dropna(subset=['PLHTIAN', 'PLNGENAN'], inplace=True)
egrid_plant['plant_avg_heat_rate'] = np.abs(egrid_plant.PLHTIAN / egrid_plant.PLNGENAN)

egrid_unit_gen_plt = pd.merge(egrid_unit_gen, egrid_plant, on='ORISPL', how='left')
egrid_unit_gen_plt.dropna(subset= ['unit_avg_heat_rate', 'plant_avg_heat_rate'], inplace= True)
egrid_unit_gen_plt = egrid_unit_gen_plt[['ORISPL', 'UNITID', 'GENID', 'PRMVR', 'FUELU1', 'unit_avg_heat_rate', 'plant_avg_heat_rate']]

egrid_unit_gen_plt['difference'] = (np.abs(egrid_unit_gen_plt.unit_avg_heat_rate - egrid_unit_gen_plt.plant_avg_heat_rate)).round(2)
egrid_unit_gen_plt['class_average'] = egrid_unit_gen_plt.groupby(['PRMVR','FUELU1'])['plant_avg_heat_rate'].transform('mean')
egrid_unit_gen_plt['class_difference_plant'] = (np.abs(egrid_unit_gen_plt.plant_avg_heat_rate - egrid_unit_gen_plt.class_average)/egrid_unit_gen_plt.class_average).mul(100).round(2)
egrid_unit_gen_plt['class_difference_unit'] = (np.abs(egrid_unit_gen_plt.unit_avg_heat_rate - egrid_unit_gen_plt.class_average)/egrid_unit_gen_plt.class_average).mul(100).round(2)
egrid_unit_gen_plt.sort_values(by='class_difference_unit', ascending=False).head(20)



egrid_unit_gen_plt.loc[np.abs(egrid_unit_gen_plt.class_difference_unit) > 300]

# if the class_difference avg heat rate is larger than 1000% then replace with class average
pct_worse_limit = 300
egrid_unit_gen_plt['egrid_heat_rate'] = egrid_unit_gen_plt.unit_avg_heat_rate
egrid_unit_gen_plt.loc[np.abs(egrid_unit_gen_plt.class_difference_unit) > pct_worse_limit, 'egrid_heat_rate'] = egrid_unit_gen_plt.loc[np.abs(egrid_unit_gen_plt.class_difference_unit) > pct_worse_limit, 'class_average']
egrid_unit_gen_plt.sort_values(by='egrid_heat_rate', ascending=False).head(20)


cec_plexos = pd.read_excel('../data/PLEXOS Export 2023-10-05 10-46.xlsx', sheet_name="Properties", skiprows=0)


cec_eia_mapper = pd.read_excel('../data/PLEXOS Export 2023-10-05 10-46.xlsx', sheet_name="CustomColumns", skiprows=0)
col_keep = ['object', 'value']
cec_eia_mapper = cec_eia_mapper[cec_eia_mapper.column == 'EIA ID'].loc[:,col_keep].set_index('value')
cec_eia_mapper

#Load ADS Data
ADS_PATH = os.path.join('../data/ads_2032_public_data/')
ads_thermal= pd.read_csv(ADS_PATH + '/Thermal_General_Info.csv',skiprows=1, )#encoding='unicode_escape')
ads_thermal = ads_thermal[['GeneratorName', ' Turbine Type', 'MustRun',
       'MinimumDownTime(hr)', 'MinimumUpTime(hr)', 'MaxUpTime(hr)', 'RampUp Rate(MW/minute)',
       'RampDn Rate(MW/minute)', 'Startup Cost Fixed($)', 'StartFuel(MMBTu)', 'Startup Time',
       'VOM Cost']]
ads_thermal.columns = standardize_col_names(ads_thermal.columns)
ads_thermal

ads_ioc= pd.read_csv(ADS_PATH + '/Thermal_IOCurve_Info.csv',skiprows=1, ).rename(columns={'Generator Name':'GeneratorName'})
ads_ioc = ads_ioc[['GeneratorName', 'IOMaxCap(MW)','IOMinCap(MW)', 'MinInput(MMBTu)', 'IncCap2(MW)', 'IncHR2(MMBTu/MWh)',
       'IncCap3(MW)', 'IncHR3(MMBTu/MWh)', 'IncCap4(MW)', 'IncHR4(MMBTu/MWh)',
       'IncCap5(MW)', 'IncHR5(MMBTu/MWh)', 'IncCap6(MW)', 'IncHR6(MMBTu/MWh)',
       'IncCap7(MW)', 'IncHR7(MMBTu/MWh)']]
ads_ioc['IncHR2(MMBTu/MWh)'] = ads_ioc['IncHR2(MMBTu/MWh)'].replace(0, np.nan)
ads_ioc.columns = standardize_col_names(ads_ioc.columns)

ads_ioc['inchr1mmbtu/mwh'] = ads_ioc.mininputmmbtu/ads_ioc.iomincapmw
ads_ioc.rename(
    columns={
        'inchr1mmbtu/mwh':'hr1',
        'inchr2mmbtu/mwh':'hr2',
        'inchr3mmbtu/mwh':'hr3',
        'inchr4mmbtu/mwh':'hr4',
        'inchr5mmbtu/mwh':'hr5',
        'inchr6mmbtu/mwh':'hr6',
        'inchr7mmbtu/mwh':'hr7',
        'iomincapmw':'x_1',
        'mininputmmbtu':'mmbtu_1'
        },
    inplace=True)

for i in range(2, 8):
    ads_ioc[f'x_{i}'] = ads_ioc[f'x_{i-1}'] + ads_ioc[f'inccap{i}mw']
    ads_ioc[f'mmbtu_{i}'] = ads_ioc[f'x_{i}'] * ads_ioc[f'hr{i}']

for i in range(0, ads_ioc.shape[0]):
    for j in range(2, 8):
        if ads_ioc[f'hr{j}'][i] == 0:
            ads_ioc[f'hr{j}'][i] = ads_ioc[f'hr{j-1}'][i]

ads_ioc


import numpy as np
def detail_linspace(x_values, y_values, num_points):
    # Arrays to hold the detailed linspace results
    x_detailed = np.array([])
    y_detailed = np.array([])

    for i in range(len(x_values) - 1):
        if x_values[i] == x_values[i+1]:
            continue
        # Generate linspace for x values
        x_segment = np.linspace(x_values[i], x_values[i+1], num_points, endpoint=False)

        # Calculate the slope of the segment
        slope = (y_values[i+1] - y_values[i]) / (x_values[i+1] - x_values[i])

        # Generate y values based on the slope and start point
        y_segment = slope * (x_segment - x_values[i]) + y_values[i]

        # Append the segment to the detailed arrays
        x_detailed = np.concatenate((x_detailed, x_segment))
        y_detailed = np.concatenate((y_detailed, y_segment))

    return x_detailed, y_detailed

import numpy as np
from scipy.optimize import minimize

# Define quadratic error function
def quadratic_error_function(params, x, y_true):
    a, b, c = params
    y_pred = a*x**2 + b*x + c
    return np.sum((y_true - y_pred)**2)

def linear_error_function(params, x, y_true):
    a, b = params
    y_pred = a*x + b
    return np.sum((y_true - y_pred)**2)

ads_ioc['linear_a'] = 0
ads_ioc['linear_b'] = 0
ads_ioc['quadratic_a'] = 0
ads_ioc['quadratic_b'] = 0
ads_ioc['quadratic_c'] = 0
ads_ioc['avg_hr'] = 0

for generator_index in range(ads_ioc.shape[0]):
    # generator_index = 0
    x_set_points = ads_ioc[['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7']].values[generator_index,:]
    y_vals_hr = ads_ioc[['hr1', 'hr2', 'hr3', 'hr4', 'hr5', 'hr6', 'hr7']].values[generator_index,:]
    y_vals = ads_ioc[['mmbtu_1', 'mmbtu_2', 'mmbtu_3', 'mmbtu_4', 'mmbtu_5', 'mmbtu_6', 'mmbtu_7']].values[generator_index,:]

    x_linspace, y_linspace = detail_linspace(x_set_points, y_vals, 10)

    initial_guess = [0.1, 0.1, 0.1]
    result_quad = minimize(quadratic_error_function, initial_guess, args=(x_linspace, y_linspace))

    initial_guess_lin = [0.1, 0.1]
    result_linear = minimize(linear_error_function, initial_guess_lin, args=(x_linspace, y_linspace))


    a_opt, b_opt, c_opt = result_quad.x
    # print(f"Quadratic parameters: a = {a_opt}, b = {b_opt}, c = {c_opt}")

    a_opt_lin, b_opt_lin = result_linear.x
    # print(f"Linear parameters: a = {a_opt_lin}, b = {b_opt_lin}")

    avg_hr = np.mean((a_opt_lin * x_linspace  + b_opt_lin) / x_linspace)
    # print(f"Average heat rate: {avg_hr}")

    ads_ioc.loc[generator_index, 'linear_a'] = a_opt_lin
    ads_ioc.loc[generator_index, 'linear_b'] = b_opt_lin
    ads_ioc.loc[generator_index, 'quadratic_a'] = a_opt
    ads_ioc.loc[generator_index, 'quadratic_b'] = b_opt
    ads_ioc.loc[generator_index, 'quadratic_c'] = c_opt
    ads_ioc.loc[generator_index, 'avg_hr'] = avg_hr

# Check for inf and nan values in avg_hr, and replace with nan.
# This is done so we can identify plants without data, then replace with averages later
print("# of np.inf in avg_hr: ", np.sum(abs(ads_ioc['avg_hr']) == np.inf))
print("# of np.nan in avg_hr: ", np.sum(abs(ads_ioc['avg_hr']) == np.nan))
ads_ioc['avg_hr'] = ads_ioc['avg_hr'].replace([np.inf, -np.inf], np.nan)

# Plotting IOC Results
generator_index = 102 #1050
x_set_points = ads_ioc[['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7']].values[generator_index,:]
y_vals_hr = ads_ioc[['hr1', 'hr2', 'hr3', 'hr4', 'hr5', 'hr6', 'hr7']].values[generator_index,:]
y_vals = ads_ioc[['mmbtu_1', 'mmbtu_2', 'mmbtu_3', 'mmbtu_4', 'mmbtu_5', 'mmbtu_6', 'mmbtu_7']].values[generator_index,:]
x_linspace, y_linspace = detail_linspace(x_set_points, y_vals, 10)

a_opt, b_opt, c_opt = ads_ioc.loc[generator_index, ['quadratic_a', 'quadratic_b', 'quadratic_c']]
a_opt_lin, b_opt_lin = ads_ioc.loc[generator_index, ['linear_a', 'linear_b']]

print('set points: ', x_set_points)
print('heat values: ',y_vals)
print('y_vals_hr: ', y_vals_hr)

ads_ioc.iloc[generator_index, :]

# Plotting IOC Results
import matplotlib.pyplot as plt

# Plot piecewise linear x and y values
plt.plot(x_linspace, y_linspace, 'o-', label='Piecewise Linear')

# Plot quadratic curve
x = np.linspace(min(x_set_points), max(x_set_points), 100)
y = a_opt * x**2 + b_opt * x + c_opt
plt.plot(x, y, label='Quadratic Curve')


# Plot linear term
y = a_opt_lin * x + b_opt_lin
plt.plot(x, y, label='Linear Curve')


# Set labels and title
plt.xlabel('Capacity [MW]')
plt.ylabel('Heat Input [MMBTU]')
plt.title('Piecewise Linear and Quadratic Curve')

plt.legend()
plt.title('Linear, Quadratic, and Piecewise Linear Heat Input-Output Curves')
plt.show()


# Plot piecewise linear x and y values
plt.plot(x_set_points, y_vals_hr, 'o-', label='Piecewise Linear')

# Plot quadratic curve
x = np.linspace(min(x_set_points), max(x_set_points), 100)
y = (a_opt * x**2 + b_opt * x + c_opt)/ x
plt.plot(x, y, label='Quadratic Curve')

# Plot linear term
y = (a_opt_lin * x + b_opt_lin) / x
plt.plot(x, y, label='Linear Curve')

# Set labels and title
plt.xlabel('Capacity [MW]')
plt.ylabel('Heat Rate [MMBTU/ MWh]')
plt.title('Piecewise Linear and Quadratic Curve')

plt.legend()
plt.title('Linear, Quadratic, and Piecewise Linear Heat Rate Curves')
plt.show()


# Plot piecewise linear x and y values
plt.plot(x_set_points, y_vals_hr, 'o-', label='Piecewise Linear')

# Plot quadratic curve
x = np.linspace(min(x_set_points), max(x_set_points), 100)
y = (a_opt * x**2 + b_opt * x + c_opt)/ x
plt.plot(x, y, label='Quadratic Curve')

# Plot linear term
y = (a_opt_lin * x + b_opt_lin) / x
plt.plot(x, y, label='Linear Curve')

y = np.mean((a_opt_lin * x + b_opt_lin) / x) * np.ones_like(x)
plt.plot(x[::5], y[::5], c='green', linestyle='--', label='Linear HR Curve Avg')

y = np.mean(y_vals_hr) * np.ones_like(x_set_points)
plt.plot(x_set_points[::5], y[::5], c='blue', linestyle='--', label='PWL HR Curve Avg')


# Set labels and title
plt.xlabel('Capacity [MW]')
plt.ylabel('Heat Rate [MMBTU/ MWh]')
plt.title('Piecewise Linear and Quadratic Curve')

plt.legend()
plt.title('Linear, Quadratic, and Piecewise Linear Heat Rate Curves')
plt.show()

# Merge ADS plant data with thermal IOC data
ads_thermal_ioc = pd.merge(ads_thermal, ads_ioc, on='generatorname', how='left')
ads_thermal_ioc.dropna(subset=['avg_hr'])
# we have heat rates for 13652 generators


#loading ads to match ads_name with generator key in order to link with ads thermal file
ads= pd.read_csv(ADS_PATH + '/GeneratorList.csv',skiprows=2, encoding='unicode_escape')
# ads = ads[ads['State'].isin(['NM', 'AZ', 'CA', 'WA', 'OR', 'ID', 'WY', 'MT', 'UT', 'SD', 'CO', 'NV', 'NE', '0', 'TX'])]
ads['Long Name'] = ads['Long Name'].astype(str)
ads['Name'] = ads['Name'].str.replace(" ", "")
ads['Name'] = ads['Name'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x).lower())
ads['Long Name'] = ads['Long Name'].str.replace(" ", "")
ads['Long Name'] = ads['Long Name'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x).lower())
ads['SubType'] = ads['SubType'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x).lower())
ads.rename({'Name': 'ads_name', 'Long Name': 'ads_long_name',
             'SubType': 'subtype','Commission Date':'commission_date',
             'Retirement Date':'retirement_date','Area Name':'balancing_area'},
               axis=1, inplace=True)
ads.rename(str.lower, axis='columns', inplace=True)
ads['long id'] = ads['long id'].astype(str)
ads = ads.loc[:, ~ads.columns.isin(['save to binary', 'county', 'city','zipcode', 'internalid'])]
ads_name_key_dict = dict(zip(ads['ads_name'], ads['generatorkey']))
ads.columns

ads_thermal_ioc['generator_name_alt'] = ads_thermal_ioc['generatorname'].str.replace(" ", "").str.lower().str.replace('_',"").str.replace('-','')
ads_thermal_ioc['generator_key'] = ads_thermal_ioc['generator_name_alt'].map(ads_name_key_dict)

# Identify Generators not in ads generator list that are in the IOC curve. This could potentially be matched with manual work.
ads_thermal_ioc[ads_thermal_ioc.generator_key.isna()]

# Merge ads thermal_IOC data with ads generator data
# Only keeping thermal plants for their heat rate and ramping data
ads_complete = ads_thermal_ioc.merge(ads, left_on='generator_key', right_on='generatorkey', how='left')
ads_complete.columns = standardize_col_names(ads_complete.columns, prefix='ads_')
ads_complete = ads_complete.loc[~ads_complete.ads_state.isin(['MX'])]
ads_complete

ads_complete.pivot_table(index=['ads_fueltype'], values='ads_avg_hr', aggfunc='mean').sort_values('ads_avg_hr', ascending=False)

#load mapping file to match the ads thermal to the eia_plants_locs file
eia_ads_mapper = pd.read_csv('../data/eia_mappings/eia_ads_generator_mapping_updated.csv')
eia_ads_mapper = eia_ads_mapper.loc[:,[
                                    'generatorkey', 'ads_name', 'plant_id_ads',
                                    'plant_id_eia', 'generator_id_ads']]
eia_ads_mapper.columns = standardize_col_names(eia_ads_mapper.columns, prefix='mapper_')
eia_ads_mapper.dropna(subset=['mapper_plant_id_eia'], inplace=True)
eia_ads_mapper.mapper_plant_id_eia = eia_ads_mapper.mapper_plant_id_eia.astype(int)
eia_ads_mapper.mapper_ads_name = eia_ads_mapper.mapper_ads_name.astype(str)
eia_ads_mapper.mapper_generatorkey = eia_ads_mapper.mapper_generatorkey.astype(int)
eia_ads_mapper

ads_complete.dropna(subset=['ads_generator_key'], inplace=True)
ads_complete.ads_generator_key = ads_complete.ads_generator_key.astype(int)
eia_ads_mapper.mapper_generatorkey = eia_ads_mapper.mapper_generatorkey.astype(int)

eia_ads_mapping = pd.merge(ads_complete, eia_ads_mapper, left_on= 'ads_generator_key', right_on= 'mapper_generatorkey', how='inner')
eia_ads_mapping

#ID MISSING ADS GENERATORS
# find missing ones not mapped
ads_missing = ads_complete[~ads_complete.ads_generator_key.isin(eia_ads_mapping.ads_generator_key)]
ads_missing

# Merge EIA and ADS Data
eia_plants_locs.plant_code = eia_plants_locs.plant_code.astype(int)

eia_ads_merged = pd.merge(
        left = eia_plants_locs,
        right = eia_ads_mapping,
        left_on=['plant_code','generator_id'],
        right_on=['mapper_plant_id_eia', 'mapper_generator_id_ads'],
        how='left')
eia_ads_merged.drop(columns=eia_ads_mapper.columns, inplace=True)
eia_ads_merged.drop(columns=['ads_x_1', 'ads_mmbtu_1', 'ads_inccap2mw', 'ads_hr2', 'ads_inccap3mw', 'ads_hr3', 'ads_inccap4mw', 'ads_hr4', 'ads_inccap5mw', 'ads_hr5', 'ads_inccap6mw', 'ads_hr6', 'ads_inccap7mw', 'ads_hr7', 'ads_hr1', 'ads_x_2', 'ads_mmbtu_2', 'ads_x_3', 'ads_mmbtu_3', 'ads_x_4', 'ads_mmbtu_4', 'ads_x_5', 'ads_mmbtu_5', 'ads_x_6', 'ads_mmbtu_6', 'ads_x_7', 'ads_mmbtu_7','ads_generator_name_alt', 'ads_generator_key', 'ads_generatorkey', 'ads_ads_name', 'ads_bus_id', 'ads_bus_name', 'ads_bus_kv', 'ads_unit_id', 'ads_generator_typeid', 'ads_subtype', 'ads_long_id', 'ads_ads_long_name',], inplace=True)
eia_ads_merged= eia_ads_merged.drop_duplicates(subset=['plant_code', 'plant_name', 'generator_id'], keep='first')
eia_ads_merged

# Merge EIA with egrid Data for heat-rates
plants_merged = pd.merge(eia_ads_merged, egrid_unit_gen_plt[['ORISPL', 'GENID','egrid_heat_rate']], left_on=['plant_code', 'generator_id'], right_on=['ORISPL', 'GENID'], how='left').drop(columns=['ORISPL', 'GENID'])
plants_merged

plants_merged.to_csv('../data/plants_merged.csv', index=False)

print(plants_merged.columns.tolist())
plants_merged

test = eia_ads_merged[['carrier','nameplate_capacity_mw','ads_avg_hr']].dropna(subset=['ads_avg_hr'])
test['p_nom_sum_carrier'] = test.groupby('carrier')['nameplate_capacity_mw'].transform('sum')
test['p_nom_weight'] = test['nameplate_capacity_mw'] / test['p_nom_sum_carrier']
test['weighted_hr'] = test['p_nom_weight'] * test['ads_avg_hr']
test.groupby('carrier')['weighted_hr'].sum().sort_values(ascending=False)

plants_merged['hr_diff'] = plants_merged['egrid_heat_rate'] - plants_merged['ads_avg_hr']
plants_merged.sort_values(by='hr_diff', ascending=False)

mixed_data_columns = eia_ads_merged.apply(lambda x: len(x.unique()) != x.nunique())
mixed_data_columns = mixed_data_columns[mixed_data_columns].index.tolist()
mixed_data_columns

eia_ads_thermal.pivot_table(index=['technology', 'carrier', 'fuel_type','fuel_name'], values='ads_avg_hr', aggfunc='mean').sort_values('ads_avg_hr', ascending=False)

eia_ads_thermal[eia_ads_thermal.carrier == 'nuclear']



