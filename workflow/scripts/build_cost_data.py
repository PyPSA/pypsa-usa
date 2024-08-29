"""
Combines all time independent cost data sources into a standard format.
"""

import logging
from typing import Dict, List, Union

import constants as const
import pandas as pd
import duckdb
from _helpers import mock_snakemake, calculate_annuity

logger = logging.getLogger(__name__)

def create_duckdb_instance(pudl_fn: str):
    duckdb.connect(database=":memory:", read_only=False)

    duckdb.query("INSTALL sqlite;")
    duckdb.query(
        f"""
        ATTACH '{pudl_fn}' (TYPE SQLITE);
        USE pudl;
        """,
    )

def load_pudl_atb_data():
    query = f"""
    WITH finance_cte AS (
        SELECT 
        wacc_real,
        technology_description,
        model_case_nrelatb,
        scenario_atb,
        projection_year,
        cost_recovery_period_years,
        report_year
        FROM core_nrelatb__yearly_projected_financial_cases_by_scenario
    )
    SELECT *
    FROM core_nrelatb__yearly_projected_cost_performance atb
    LEFT JOIN finance_cte AS finance
        ON atb.technology_description = finance.technology_description
            AND atb.model_case_nrelatb = finance.model_case_nrelatb
            AND atb.scenario_atb = finance.scenario_atb
            AND atb.projection_year = finance.projection_year
            AND atb.cost_recovery_period_years = finance.cost_recovery_period_years
            AND atb.report_year = finance.report_year
    WHERE atb.report_year = 2024
    """
    return duckdb.query(query).to_df()

def load_pudl_aeo_data():
    query = f"""
    SELECT *
    FROM core_eiaaeo__yearly_projected_fuel_cost_in_electric_sector_by_type aeo
    WHERE aeo.report_year = 2023
    """
    return duckdb.query(query).to_df()

def match_technology(row, tech_dict):
    for key, value in tech_dict.items():
        # Match technology and techdetail
        if row['technology_description'] == value.get('technology') and row['technology_description_detail_1'] == value.get('techdetail'):
            return key
        # Match technology and techdetail2
        elif row['technology_description'] == value.get('technology') and row['technology_description_detail_2'] == value.get('techdetail2'):
            return key
    
    return None 

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_cost_data", year=2030)
        rootpath = ".."
    else:
        rootpath = "."

    costs = snakemake.params.costs
    atb_params = costs.get("atb")
    aeo_params = costs.get("aeo")

    tech_year = snakemake.wildcards.year
    years = range(2021, 2051)
    tech_year = min(years, key=lambda x: abs(x - int(tech_year)))

    create_duckdb_instance(snakemake.input.pudl)

    # Import PUDLs ATB data
    pudl_atb = load_pudl_atb_data()
    pudl_atb['pypsa-name'] = pudl_atb.apply(match_technology, axis=1, tech_dict=const.ATB_TECH_MAPPER)
    pudl_atb = pudl_atb[pudl_atb['pypsa-name'].notnull()]

    # Group by pypsa-name and filter for correct cost recovery period
    pudl_atb = pudl_atb.groupby('pypsa-name').apply(lambda x: x[x['cost_recovery_period_years'] == const.ATB_TECH_MAPPER[x.name].get("crp", 30)]).reset_index(drop=True)

    #Filter for the correct year, scenario, and model case
    pudl_atb = pudl_atb[pudl_atb.projection_year == tech_year]
    pudl_atb = pudl_atb[pudl_atb.scenario_atb == atb_params.get("scenario", "Moderate")]
    pudl_atb = pudl_atb[pudl_atb.model_case_nrelatb == atb_params.get("model_case", "Market")]

    pudl_premelt = pudl_atb.copy()
    # Pivot Data
    cols = [
        'cost_recovery_period_years','capacity_factor', 'capex_per_kw', 'capex_overnight_per_kw',
       'capex_overnight_additional_per_kw', 'capex_grid_connection_per_kw',
       'capex_construction_finance_factor', 'fuel_cost_per_mwh',
       'heat_rate_mmbtu_per_mwh', 'heat_rate_penalty',
       'levelized_cost_of_energy_per_mwh', 'net_output_penalty',
       'opex_fixed_per_kw', 'opex_variable_per_mwh','wacc_real']
    #pivot such that cols all get moved to one column
    pudl_atb = pudl_atb.melt(id_vars='pypsa-name', value_vars=cols, var_name='parameter', value_name='value')

    # Impute emissions factor data
    # https://www.eia.gov/environment/emissions/co2_vol_mass.php
    # Units: [tCO2/MWh_thermal]
    emissions_data = [
        {'pypsa-name': 'coal', 'parameter': 'co2_emissions', 'value': 0.3453},
        {'pypsa-name': 'oil', 'parameter': 'co2_emissions', 'value': 0.34851},
        {'pypsa-name': 'geothermal', 'parameter': 'co2_emissions', 'value': 0.04029},
        {'pypsa-name': 'waste', 'parameter': 'co2_emissions', 'value': 0.1016},
        {'pypsa-name': 'gas', 'parameter': 'co2_emissions', 'value': 0.18058},
        {'pypsa-name': 'CCGT', 'parameter': 'co2_emissions', 'value': 0.18058},
        {'pypsa-name': 'OCGT', 'parameter': 'co2_emissions', 'value': 0.18058},
        {'pypsa-name': 'geothermal', 'parameter': 'heat_rate_mmbtu_per_mwh', 'value': 8881}, #AEO 2023
    ]
    # Impute Transmission Data
    # TEPCC 2023
    # WACC & Lifetime: https://emp.lbl.gov/publications/improving-estimates-transmission
    # Subsea costs: Purvins et al. (2018): https://doi.org/10.1016/j.jclepro.2018.03.095
    transmission_data = [
        {'pypsa-name': 'HVAC overhead', 'parameter': 'capex_per_mw_km', 'value': 2481.43},
        {'pypsa-name': 'HVAC overhead', 'parameter': 'cost_recovery_period_years', 'value': 60},
        {'pypsa-name': 'HVAC overhead', 'parameter': 'wacc_real', 'value': 0.044},
        {'pypsa-name': 'HVDC overhead', 'parameter': 'capex_per_mw_km', 'value': 1026.53},
        {'pypsa-name': 'HVDC overhead', 'parameter': 'cost_recovery_period_years', 'value': 60},
        {'pypsa-name': 'HVDC overhead', 'parameter': 'wacc_real', 'value': 0.044},
        {'pypsa-name': 'HVDC submarine', 'parameter': 'capex_per_mw_km', 'value': 504.141},
        {'pypsa-name': 'HVDC submarine', 'parameter': 'cost_recovery_period_years', 'value': 60},
        {'pypsa-name': 'HVDC submarine', 'parameter': 'wacc_real', 'value': 0.044},
        {'pypsa-name': 'HVDC inverter pair', 'parameter': 'capex_per_kw', 'value': 173.730},
        {'pypsa-name': 'HVDC inverter pair', 'parameter': 'cost_recovery_period_years', 'value': 60},
        {'pypsa-name': 'HVDC inverter pair', 'parameter': 'wacc_real', 'value': 0.044},
    ]
    pudl_atb = pd.concat([pudl_atb, pd.DataFrame(emissions_data), pd.DataFrame(transmission_data)], ignore_index=True)
    pudl_atb.drop_duplicates(subset=['pypsa-name', 'parameter'], inplace=True)

    # Load AEO Fuel Cost Data
    aeo = load_pudl_aeo_data()
    aeo = aeo[aeo.projection_year == tech_year]
    aeo = aeo[aeo.model_case_eiaaeo == aeo_params.get("scenario", "Reference")]
    cols = ['fuel_type_eiaaeo', 'fuel_cost_real_per_mmbtu_eiaaeo']
    aeo = aeo[cols]
    aeo = aeo.groupby('fuel_type_eiaaeo').mean()
    aeo['fuel_cost_real_per_mwhth'] = aeo['fuel_cost_real_per_mmbtu_eiaaeo'] * 3.412
    aeo = pd.melt(aeo.reset_index(), id_vars='fuel_type_eiaaeo', value_vars=['fuel_cost_real_per_mwhth'], var_name='parameter', value_name='value')
    aeo.rename(columns={'fuel_type_eiaaeo': 'pypsa-name'}, inplace=True)

    addnl_fuels = pd.DataFrame([
        {'pypsa-name': 'nuclear', 'parameter': 'fuel_cost_real_per_mwhth', 'value': 2.782},
        {'pypsa-name': 'biomass', 'parameter': 'fuel_cost_real_per_mwhth', 'value': 7.49},
    ])
    aeo = pd.concat([aeo, addnl_fuels], ignore_index=True)

    tech_fuel_map = {
        'CCGT': 'natural_gas',
        'OCGT': 'natural_gas',
        'CCGT-95CCS': 'natural_gas',
        'CCGT-97CCS': 'natural_gas',
        'coal-95CCS': 'coal',
        'coal-99CCS': 'coal',
        'SMR': 'nuclear',
    }
    tech_fuels = pd.DataFrame([
        {'pypsa-name': new_name, 'parameter': 'fuel_cost_real_per_mwhth', 'value': aeo.loc[aeo['pypsa-name'] == source_name, 'value'].values[0]}
        for new_name, source_name in tech_fuel_map.items()
    ])
    aeo = pd.concat([aeo, tech_fuels], ignore_index=True)
    pudl_atb = pd.concat([pudl_atb, aeo], ignore_index=True)



    # Calculate Annualized Costs and Marinal Costs
    # Apply: marginal_cost = opex_variable_per_mwh + fuel_cost_real_per_mwhth / efficiency
    pivot_atb = pudl_atb.pivot(index='pypsa-name', columns='parameter', values='value').reset_index()
    
    pivot_atb["efficiency"] = 3.412 / pivot_atb["heat_rate_mmbtu_per_mwh"]
    pivot_atb['fuel_cost'] = pivot_atb['fuel_cost_real_per_mwhth'] / pivot_atb['efficiency']
    pivot_atb['marginal_cost'] = pivot_atb['opex_variable_per_mwh'] + pivot_atb['fuel_cost']
    
    #Impute storage WACC from Utility Scale Solar. TODO: Revisit this assumption
    for x in [2, 4, 6, 8, 10]:
        pivot_atb.loc[pivot_atb['pypsa-name'] == f'{x}hr_battery_storage', 'wacc_real'] = pivot_atb.loc[pivot_atb['pypsa-name'] == 'solar', 'wacc_real'].values[0]
        pivot_atb.loc[pivot_atb['pypsa-name'] == f'{x}hr_battery_storage', 'efficiency'] = 0.85 # 2023 ATB

    pivot_atb["annualized_capex_per_mw"] = (
        (calculate_annuity(pivot_atb["cost_recovery_period_years"], pivot_atb["wacc_real"])
        * pivot_atb["capex_per_kw"]
        * 1) # change to nyears
    ) * 1e3

    pivot_atb["annualized_capex_per_mw_km"] = (
        (calculate_annuity(pivot_atb["cost_recovery_period_years"], pivot_atb["wacc_real"])
        * pivot_atb["capex_per_mw_km"]
        * 1) # change to nyears
    )

    # Calculate grid interrconnection costs per MW-KM
    # All land-based resources assume 1 mile of spur line
    # All offshore resources assume 30 km of subsea cable
    pivot_atb['capex_grid_connection_per_kw_km'] = pivot_atb['capex_grid_connection_per_kw'] / 1.609 
    pivot_atb.loc[pivot_atb['pypsa-name'].str.contains('offshore'), 'capex_grid_connection_per_kw_km'] = pivot_atb['capex_grid_connection_per_kw'] / 30

    pivot_atb["annualized_connection_capex_per_mw_km"] = (
        (calculate_annuity(pivot_atb["cost_recovery_period_years"], pivot_atb["wacc_real"])
        * pivot_atb["capex_grid_connection_per_kw_km"]
        * 1) # change to nyears
    )

    pivot_atb["annualized_capex_fom"] = pivot_atb["annualized_capex_per_mw"] + (pivot_atb["opex_fixed_per_kw"] * 1e3)
    pudl_atb = pivot_atb.melt(id_vars=['pypsa-name'], value_vars=pivot_atb.columns.difference(['pypsa-name']), var_name='parameter', value_name='value')

    # Export
    pudl_atb = pudl_atb.reset_index(drop=True)
    pudl_atb["value"] = pudl_atb["value"].round(3)
    pudl_atb.to_csv(snakemake.output.tech_costs, index=False)

