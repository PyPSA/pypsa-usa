import re

import duckdb
import numpy as np
import pandas as pd

def load_pudl_data():
    duckdb.connect(database=":memory:", read_only=False)

    duckdb.query("INSTALL sqlite;")
    duckdb.query(
        f"""
        ATTACH '{snakemake.input.pudl}' (TYPE SQLITE);
        USE pudl;
        """,
    )

    eia_data_operable = duckdb.query(
        # the pudl data is organised into a row per plant_id_eia, generator_id, and report_date
        # usually we want to get the most recent data for each plant_id_eia, generator_id
        # but sometimes the most recent data has null values, so we need to fill in with older data
        # this is why many of the columns are aggregated with array_agg and FILTER so we can get the most recent non-null value
        """
        WITH monthly_generators AS (
            SELECT
                plant_id_eia,
                generator_id,
                array_agg(out_eia__monthly_generators.unit_heat_rate_mmbtu_per_mwh ORDER BY out_eia__monthly_generators.report_date DESC) FILTER (WHERE out_eia__monthly_generators.unit_heat_rate_mmbtu_per_mwh IS NOT NULL)[1] AS unit_heat_rate_mmbtu_per_mwh
            FROM out_eia__monthly_generators
            WHERE operational_status = 'existing' AND report_date >= '2022-01-01'
            GROUP BY plant_id_eia, generator_id
        )
        SELECT
            out_eia__yearly_generators.plant_id_eia,
            out_eia__yearly_generators.generator_id,
            array_agg(out_eia__yearly_generators.plant_name_eia ORDER BY out_eia__yearly_generators.report_date DESC) FILTER (WHERE out_eia__yearly_generators.plant_name_eia IS NOT NULL)[1] AS plant_name_eia,
            array_agg(out_eia__yearly_generators.capacity_mw ORDER BY out_eia__yearly_generators.report_date DESC) FILTER (WHERE out_eia__yearly_generators.capacity_mw IS NOT NULL)[1] AS capacity_mw,
            array_agg(out_eia__yearly_generators.summer_capacity_mw ORDER BY out_eia__yearly_generators.report_date DESC) FILTER (WHERE out_eia__yearly_generators.summer_capacity_mw IS NOT NULL)[1] AS summer_capacity_mw,
            array_agg(out_eia__yearly_generators.winter_capacity_mw ORDER BY out_eia__yearly_generators.report_date DESC) FILTER (WHERE out_eia__yearly_generators.winter_capacity_mw IS NOT NULL)[1] AS winter_capacity_mw,
            array_agg(out_eia__yearly_generators.minimum_load_mw ORDER BY out_eia__yearly_generators.report_date DESC) FILTER (WHERE out_eia__yearly_generators.minimum_load_mw IS NOT NULL)[1] AS minimum_load_mw,
            array_agg(out_eia__yearly_generators.energy_source_code_1 ORDER BY out_eia__yearly_generators.report_date DESC) FILTER (WHERE out_eia__yearly_generators.energy_source_code_1 IS NOT NULL)[1] AS energy_source_code_1,
            array_agg(out_eia__yearly_generators.technology_description ORDER BY out_eia__yearly_generators.report_date DESC) FILTER (WHERE out_eia__yearly_generators.technology_description IS NOT NULL)[1] AS technology_description,
            arbitrary(out_eia__yearly_generators.operational_status) AS operational_status,
            array_agg(out_eia__yearly_generators.prime_mover_code ORDER BY out_eia__yearly_generators.report_date DESC) FILTER (WHERE out_eia__yearly_generators.prime_mover_code IS NOT NULL)[1] AS prime_mover_code,
            array_agg(out_eia__yearly_generators.planned_generator_retirement_date ORDER BY out_eia__yearly_generators.report_date DESC) FILTER (WHERE out_eia__yearly_generators.planned_generator_retirement_date IS NOT NULL)[1] AS planned_generator_retirement_date,
            array_agg(out_eia__yearly_generators.energy_storage_capacity_mwh ORDER BY out_eia__yearly_generators.report_date DESC) FILTER (WHERE out_eia__yearly_generators.energy_storage_capacity_mwh IS NOT NULL)[1] AS energy_storage_capacity_mwh,
            array_agg(out_eia__yearly_generators.generator_operating_date ORDER BY out_eia__yearly_generators.report_date DESC) FILTER (WHERE out_eia__yearly_generators.generator_operating_date IS NOT NULL)[1] AS generator_operating_date,
            array_agg(out_eia__yearly_generators.state ORDER BY out_eia__yearly_generators.report_date DESC) FILTER (WHERE out_eia__yearly_generators.state IS NOT NULL)[1] AS state,
            array_agg(out_eia__yearly_generators.latitude ORDER BY out_eia__yearly_generators.report_date DESC) FILTER (WHERE out_eia__yearly_generators.latitude IS NOT NULL)[1] AS latitude,
            array_agg(out_eia__yearly_generators.longitude ORDER BY out_eia__yearly_generators.report_date DESC) FILTER (WHERE out_eia__yearly_generators.longitude IS NOT NULL)[1] AS longitude,
            array_agg(core_eia860__scd_generators_energy_storage.max_charge_rate_mw ORDER BY core_eia860__scd_generators_energy_storage.report_date DESC) FILTER (WHERE core_eia860__scd_generators_energy_storage.max_charge_rate_mw IS NOT NULL)[1] AS max_charge_rate_mw,
            array_agg(core_eia860__scd_generators_energy_storage.max_discharge_rate_mw ORDER BY core_eia860__scd_generators_energy_storage.report_date DESC) FILTER (WHERE core_eia860__scd_generators_energy_storage.max_discharge_rate_mw IS NOT NULL)[1] AS max_discharge_rate_mw,
            array_agg(core_eia860__scd_generators_energy_storage.storage_technology_code_1 ORDER BY core_eia860__scd_generators_energy_storage.report_date DESC) FILTER (WHERE core_eia860__scd_generators_energy_storage.storage_technology_code_1 IS NOT NULL)[1] AS storage_technology_code_1,
            array_agg(core_eia860__scd_plants.nerc_region ORDER BY core_eia860__scd_plants.report_date DESC) FILTER (WHERE core_eia860__scd_plants.nerc_region IS NOT NULL)[1] AS nerc_region,
            array_agg(core_eia860__scd_plants.balancing_authority_code_eia ORDER BY core_eia860__scd_plants.report_date DESC) FILTER (WHERE core_eia860__scd_plants.balancing_authority_code_eia IS NOT NULL)[1] AS balancing_authority_code_eia,
            first(monthly_generators.unit_heat_rate_mmbtu_per_mwh) AS unit_heat_rate_mmbtu_per_mwh
        FROM out_eia__yearly_generators
        LEFT JOIN core_eia860__scd_generators_energy_storage ON out_eia__yearly_generators.plant_id_eia = core_eia860__scd_generators_energy_storage.plant_id_eia AND out_eia__yearly_generators.generator_id = core_eia860__scd_generators_energy_storage.generator_id
        LEFT JOIN core_eia860__scd_plants ON out_eia__yearly_generators.plant_id_eia = core_eia860__scd_plants.plant_id_eia
        LEFT JOIN monthly_generators ON out_eia__yearly_generators.plant_id_eia = monthly_generators.plant_id_eia AND out_eia__yearly_generators.generator_id = monthly_generators.generator_id
        WHERE out_eia__yearly_generators.operational_status = 'existing'
        GROUP BY out_eia__yearly_generators.plant_id_eia, out_eia__yearly_generators.generator_id
    """,
    ).to_df()

    heat_rates = duckdb.query(
        """
        WITH monthly_generators AS (
            SELECT
                plant_id_eia,
                generator_id,
                report_date,
                unit_heat_rate_mmbtu_per_mwh
            FROM out_eia__monthly_generators
            WHERE operational_status = 'existing' 
            AND report_date >= CURRENT_DATE - INTERVAL '2 years'
            AND unit_heat_rate_mmbtu_per_mwh IS NOT NULL
        )
        SELECT
            mg.plant_id_eia,
            mg.generator_id,
            mg.report_date,
            mg.unit_heat_rate_mmbtu_per_mwh,
            yg.plant_name_eia,
            yg.technology_description,
            yg.operational_status,
        FROM monthly_generators mg
        LEFT JOIN out_eia__yearly_generators yg ON mg.plant_id_eia = yg.plant_id_eia AND mg.generator_id = yg.generator_id
        LEFT JOIN core_eia860__scd_plants p ON mg.plant_id_eia = p.plant_id_eia
        WHERE yg.operational_status = 'existing'
        ORDER BY mg.report_date DESC
        """,
    ).to_df()

    return eia_data_operable, heat_rates


def set_non_conus(eia_data_operable):
    eia_data_operable.loc[eia_data_operable.state.isin(["AK", "HI"]), "nerc_region"] = (
        "non-conus"
    )
    eia_data_operable.loc[
        eia_data_operable.state.isin(["AK", "HI"]),
        "balancing_authority_code",
    ] = "non-conus"


def set_derates(eia_data_operable):
    eia_data_operable["summer_derate"] = 1 - (
        (eia_data_operable.capacity_mw - eia_data_operable.summer_capacity_mw)
        / eia_data_operable.capacity_mw
    )
    eia_data_operable["winter_derate"] = 1 - (
        (eia_data_operable.capacity_mw - eia_data_operable.winter_capacity_mw)
        / eia_data_operable.capacity_mw
    )
    eia_data_operable.summer_derate = eia_data_operable.summer_derate.clip(
        upper=1,
    ).clip(lower=0)
    eia_data_operable.winter_derate = eia_data_operable.winter_derate.clip(
        upper=1,
    ).clip(lower=0)


# Assign PyPSA Carrier Names, Fuel Types, and Prime Movers Names
eia_tech_map = pd.DataFrame(
    {
        "Technology": [
            "Petroleum Liquids",
            "Onshore Wind Turbine",
            "Conventional Hydroelectric",
            "Natural Gas Steam Turbine",
            "Conventional Steam Coal",
            "Natural Gas Fired Combined Cycle",
            "Natural Gas Fired Combustion Turbine",
            "Nuclear",
            "Hydroelectric Pumped Storage",
            "Natural Gas Internal Combustion Engine",
            "Solar Photovoltaic",
            "Geothermal",
            "Landfill Gas",
            "Batteries",
            "Wood/Wood Waste Biomass",
            "Coal Integrated Gasification Combined Cycle",
            "Other Gases",
            "Petroleum Coke",
            "Municipal Solid Waste",
            "Natural Gas with Compressed Air Storage",
            "All Other",
            "Other Waste Biomass",
            "Solar Thermal without Energy Storage",
            "Other Natural Gas",
            "Solar Thermal with Energy Storage",
            "Flywheels",
            "Offshore Wind Turbine",
        ],
        "tech_type": [
            "oil",
            "onwind",
            "hydro",
            "OCGT",
            "coal",
            "CCGT",
            "OCGT",
            "nuclear",
            "hydro",
            "OCGT",
            "solar",
            "geothermal",
            "biomass",
            "battery",
            "biomass",
            "coal",
            "other",
            "oil",
            "waste",
            "other",
            "other",
            "biomass",
            "solar",
            "other",
            "solar",
            "other",
            "offwind",
        ],
    },
)
eia_tech_map.set_index("Technology", inplace=True)
eia_fuel_map = pd.DataFrame(
    {
        "Energy Source 1": [
            "ANT",
            "BIT",
            "LIG",
            "SGC",
            "SUB",
            "WC",
            "RC",
            "DFO",
            "JF",
            "KER",
            "PC",
            "PG",
            "RFO",
            "SGP",
            "WO",
            "BFG",
            "NG",
            "H2",
            "OG",
            "AB",
            "MSW",
            "OBS",
            "WDS",
            "OBL",
            "SLW",
            "BLQ",
            "WDL",
            "LFG",
            "OBG",
            "SUN",
            "WND",
            "GEO",
            "WAT",
            "NUC",
            "PUR",
            "WH",
            "TDF",
            "MWH",
            "OTH",
        ],
        "fuel_type": [
            "coal",
            "coal",
            "coal",
            "coal",
            "coal",
            "coal",
            "coal",
            "oil",
            "oil",
            "oil",
            "oil",
            "oil",
            "oil",
            "oil",
            "oil",
            "gas",
            "gas",
            "gas",
            "gas",
            "waste",
            "waste",
            "waste",
            "waste",
            "biomass",
            "biomass",
            "biomass",
            "biomass",
            "biomass",
            "biomass",
            "solar",
            "wind",
            "geothermal",
            "hydro",
            "nuclear",
            "other",
            "other",
            "other",
            "other",
            "other",
        ],
        "fuel_name": [
            "Anthracite Coal",
            "Bituminous Coal",
            "Lignite Coal",
            "Coal-Derived Synthesis Gas",
            "Subbituminous Coal",
            "Waste/Other Coal",
            "Refined Coal",
            "Distillate Fuel Oil",
            "Jet Fuel",
            "Kerosene",
            "Petroleum Coke",
            "Gaseous Propane",
            "Residual Fuel Oil",
            "Synthesis Gas from Petroleum Coke",
            "Waste/Other Oil",
            "Blast Furnace Gas",
            "Natural Gas",
            "Hydrogen",
            "Other Gas",
            "Agricultural By-Products",
            "Municipal Solid Waste",
            "Other Biomass Solids",
            "Wood/Wood Waste Solids",
            "Other Biomass Liquids",
            "Sludge Waste",
            "Black Liquor",
            "Wood Waste Liquids excluding Black Liquor",
            "Landfill Gas",
            "Other Biomass Gas",
            "Solar",
            "Wind",
            "Geothermal",
            "Water",
            "Nuclear",
            "Purchased Steam",
            "Waste heat not directly attributed to a fuel source (undetermined)",
            "Tire-derived Fuels",
            "Energy Storage",
            "Other",
        ],
    },
)
eia_fuel_map.set_index("Energy Source 1", inplace=True)
eia_primemover_map = pd.DataFrame(
    {
        "Prime Mover": [
            "BA",
            "CE",
            "CP",
            "FW",
            "PS",
            "ES",
            "ST",
            "GT",
            "IC",
            "CA",
            "CT",
            "CS",
            "CC",
            "HA",
            "HB",
            "HK",
            "HY",
            "BT",
            "PV",
            "WT",
            "WS",
            "FC",
            "OT",
        ],
        "prime_mover": [
            "Energy Storage, Battery",
            "Energy Storage, Compressed Air",
            "Energy Storage, Concentrated Solar Power",
            "Energy Storage, Flywheel",
            "Energy Storage, Reversible Hydraulic Turbine (Pumped Storage)",
            "Energy Storage, Other",
            "Steam Turbine, including nuclear, geothermal and solar steam (does NOT include combined cycle)",
            "Combustion (Gas) Turbine",
            "Internal Combustion Engine",
            "Combined Cycle Steam Part",
            "Combined Cycle Combustion Turbine Part",
            "Combined Cycle Single Shaft",
            "Combined Cycle Total Unit (planned undetermined plants)",
            "Hydrokinetic, Axial Flow Turbine",
            "Hydrokinetic, Wave Buoy",
            "Hydrokinetic, Other",
            "Hydroelectric Turbine",
            "Turbines Used in a Binary Cycle (including those used for geothermal applications)",
            "Photovoltaic",
            "Wind Turbine, Onshore",
            "Wind Turbine, Offshore",
            "Fuel Cell",
            "Other",
        ],
    },
)
eia_primemover_map.set_index("Prime Mover", inplace=True)


def set_tech_fuels_primer_movers(eia_data_operable):
    # Map technologies, fuels, and prime movers
    maps = {
        "carrier": (
            eia_data_operable["technology_description"],
            eia_tech_map["tech_type"],
        ),
        "fuel_type": (
            eia_data_operable["energy_source_code_1"],
            eia_fuel_map["fuel_type"],
        ),
        "fuel_name": (
            eia_data_operable["energy_source_code_1"],
            eia_fuel_map["fuel_name"],
        ),
        "prime_mover_name": (
            eia_data_operable["prime_mover_code"],
            eia_primemover_map["prime_mover"],
        ),
    }
    for col, (data_col, map_df) in maps.items():
        eia_data_operable[col] = data_col.map(dict(zip(map_df.index, map_df.values)))


def standardize_col_names(columns, prefix="", suffix=""):
    """
    Standardize column names by removing spaces, converting to lowercase,
    removing parentheses, and adding prefix and suffix.
    """
    return [
        prefix
        + col.lower().replace(" ", "_").replace("(", "").replace(")", "")
        + suffix
        for col in columns
    ]


def merge_ads_data(eia_data_operable):
    ADS_PATH = snakemake.input.wecc_ads
    ads_thermal = pd.read_csv(
        ADS_PATH + "/Thermal_General_Info.csv",
        skiprows=1,
    )  # encoding='unicode_escape')
    ads_thermal = ads_thermal[
        [
            "GeneratorName",
            " Turbine Type",
            "MustRun",
            "MinimumDownTime(hr)",
            "MinimumUpTime(hr)",
            "MaxUpTime(hr)",
            "RampUp Rate(MW/minute)",
            "RampDn Rate(MW/minute)",
            "Startup Cost Fixed($)",
            "StartFuel(MMBTu)",
            "Startup Time",
            "VOM Cost",
        ]
    ]
    ads_thermal.columns = standardize_col_names(ads_thermal.columns)

    ads_ioc = pd.read_csv(
        ADS_PATH + "/Thermal_IOCurve_Info.csv",
        skiprows=1,
    ).rename(columns={"Generator Name": "GeneratorName"})
    ads_ioc = ads_ioc[
        [
            "GeneratorName",
            "IOMaxCap(MW)",
            "IOMinCap(MW)",
            "MinInput(MMBTu)",
        ]
    ]
    ads_ioc.columns = standardize_col_names(ads_ioc.columns)


    # Merge ADS plant data with thermal IOC data
    ads_thermal_ioc = pd.merge(ads_thermal, ads_ioc, on="generatorname", how="left")
    # ads_thermal_ioc.dropna(subset=["avg_hr"])

    # loading ads to match ads_name with generator key in order to link with ads thermal file
    ads = pd.read_csv(
        ADS_PATH + "/GeneratorList.csv",
        skiprows=2,
        encoding="unicode_escape",
    )
    # ads = ads[ads['State'].isin(['NM', 'AZ', 'CA', 'WA', 'OR', 'ID', 'WY', 'MT', 'UT', 'SD', 'CO', 'NV', 'NE', '0', 'TX'])]
    ads["Long Name"] = ads["Long Name"].astype(str)
    ads["Name"] = ads["Name"].str.replace(" ", "")
    ads["Name"] = ads["Name"].apply(lambda x: re.sub(r"[^a-zA-Z0-9]", "", x).lower())
    ads["Long Name"] = ads["Long Name"].str.replace(" ", "")
    ads["Long Name"] = ads["Long Name"].apply(
        lambda x: re.sub(r"[^a-zA-Z0-9]", "", x).lower(),
    )
    ads["SubType"] = ads["SubType"].apply(
        lambda x: re.sub(r"[^a-zA-Z0-9]", "", x).lower(),
    )
    ads.rename(
        {
            "Name": "ads_name",
            "Long Name": "ads_long_name",
            "SubType": "subtype",
            "Commission Date": "commission_date",
            "Retirement Date": "retirement_date",
            "Area Name": "balancing_area",
        },
        axis=1,
        inplace=True,
    )
    ads.rename(str.lower, axis="columns", inplace=True)
    ads["long id"] = ads["long id"].astype(str)
    ads = ads.loc[
        :,
        ~ads.columns.isin(
            ["save to binary", "county", "city", "zipcode", "internalid"],
        ),
    ]
    ads_name_key_dict = dict(zip(ads["ads_name"], ads["generatorkey"]))
    ads.columns

    ads_thermal_ioc["generator_name_alt"] = (
        ads_thermal_ioc["generatorname"]
        .str.replace(" ", "")
        .str.lower()
        .str.replace("_", "")
        .str.replace("-", "")
    )
    ads_thermal_ioc["generator_key"] = ads_thermal_ioc["generator_name_alt"].map(
        ads_name_key_dict,
    )

    # Identify Generators not in ads generator list that are in the IOC curve. This could potentially be matched with manual work.
    ads_thermal_ioc[ads_thermal_ioc.generator_key.isna()]

    # Merge ads thermal_IOC data with ads generator data
    # Only keeping thermal plants for their heat rate and ramping data
    ads_complete = ads_thermal_ioc.merge(
        ads,
        left_on="generator_key",
        right_on="generatorkey",
        how="left",
    )
    ads_complete.columns = standardize_col_names(ads_complete.columns, prefix="ads_")
    ads_complete = ads_complete.loc[~ads_complete.ads_state.isin(["MX"])]

    # load mapping file to match the ads thermal to the eia_plants_locs file
    eia_ads_mapper = pd.read_csv(snakemake.input.eia_ads_generator_mapping)
    eia_ads_mapper = eia_ads_mapper.loc[
        :,
        [
            "generatorkey",
            "ads_name",
            "plant_id_ads",
            "plant_id_eia",
            "generator_id_ads",
        ],
    ]
    eia_ads_mapper.columns = standardize_col_names(
        eia_ads_mapper.columns,
        prefix="mapper_",
    )
    eia_ads_mapper.dropna(subset=["mapper_plant_id_eia"], inplace=True)
    eia_ads_mapper.mapper_plant_id_eia = eia_ads_mapper.mapper_plant_id_eia.astype(int)
    eia_ads_mapper.mapper_ads_name = eia_ads_mapper.mapper_ads_name.astype(str)
    eia_ads_mapper.mapper_generatorkey = eia_ads_mapper.mapper_generatorkey.astype(int)

    ads_complete.dropna(subset=["ads_generator_key"], inplace=True)
    ads_complete.ads_generator_key = ads_complete.ads_generator_key.astype(int)
    eia_ads_mapper.mapper_generatorkey = eia_ads_mapper.mapper_generatorkey.astype(int)

    eia_ads_mapping = pd.merge(
        ads_complete,
        eia_ads_mapper,
        left_on="ads_generator_key",
        right_on="mapper_generatorkey",
        how="inner",
    )

    # Merge EIA and ADS Data
    eia_ads_merged = pd.merge(
        left=eia_data_operable,
        right=eia_ads_mapping,
        left_on=["plant_id_eia", "generator_id"],
        right_on=["mapper_plant_id_eia", "mapper_generator_id_ads"],
        how="left",
    )
    eia_ads_merged.drop(columns=eia_ads_mapper.columns, inplace=True)
    eia_ads_merged.drop(
        columns=[
            "ads_generator_name_alt",
            "ads_generator_key",
            "ads_generatorkey",
            "ads_ads_name",
            "ads_bus_id",
            "ads_bus_name",
            "ads_bus_kv",
            "ads_unit_id",
            "ads_generator_typeid",
            "ads_subtype",
            "ads_long_id",
            "ads_ads_long_name",
            "ads_state",
            "ads_btm",
            "ads_devstatus",
            "ads_retirement_date",
            "ads_commission_date",
            "ads_servicestatus",
        ],
        inplace=True,
    )
    eia_ads_merged = eia_ads_merged.drop_duplicates(
        subset=["plant_id_eia", "generator_id"],
        keep="first",
    )

    return eia_ads_merged



def add_missing_fuel_cost(plants, costs_fn):
    fuel_cost = pd.read_csv(costs_fn, index_col=0, skiprows=3)
    plants["fuel_cost"] = plants.fuel_type.map(fuel_cost.fuel_price_per_mmbtu)
    return plants


def impute_missing_plant_data(
    plants: pd.DataFrame,
    aggregation_fields: list[str],
    data_fields: list[str],
) -> pd.DataFrame:
    """
    Imputes missing data in the plants dataframe based on the average values of
    the data dataframe.
    """

    # Function to calculate weighted average
    def weighted_avg(df, values, weights):
        valid = df[values].notna()
        if valid.sum() == 0:
            return np.nan  # Return NaN if no valid entries
        return np.average(df[values][valid], weights=df[weights][valid])

    # Calculate the weighted averages excluding NaNs
    weighted_averages = (
        plants.groupby(aggregation_fields)
        .apply(
            lambda x: pd.Series(
                {field: weighted_avg(x, field, "p_nom") for field in data_fields},
            ),
        )
        .reset_index()
    )

    # Merge weighted averages back into the original DataFrame
    plants_merged = pd.merge(
        plants.reset_index(),
        weighted_averages,
        on=aggregation_fields,
        suffixes=("", "_weighted"),
    )

    # Fill NaN values using the weighted averages
    for field in data_fields:
        plants_merged[field] = plants_merged[field].fillna(
            plants_merged[f"{field}_weighted"],
        )

    # Drop the weighted average columns after filling NaNs
    plants_merged = plants_merged.drop(
        columns=[f"{field}_weighted" for field in data_fields],
    )
    plants_merged.set_index("generator_name", inplace=True)
    return plants_merged


def set_parameters(plants: pd.DataFrame):
    """
    Sets generator naming schemes, updates parameter names, and imputes missing data.
    """
    plants["generator_name"] = (
        plants.index.astype(str)
        + "_"
        + plants.plant_name_eia.astype(str)
        + "_"
        + plants.plant_id_eia.astype(str)
        + "_"
        + plants.generator_id.astype(str)
    )
    plants.set_index("generator_name", inplace=True)
    plants["p_nom"] = plants.pop("capacity_mw")
    plants["build_year"] = plants.pop("generator_operating_date").dt.year
    plants["heat_rate"] = plants.pop("unit_heat_rate_mmbtu_per_mwh")

    plants = add_missing_fuel_cost(
        plants,
        snakemake.input.fuel_costs,
    )  # National Avg used for start-up costs

    # Unit Commitment Parameters
    plants["start_up_cost"] = (
        plants.pop("ads_startup_cost_fixed$") + plants.ads_startfuelmmbtu * plants.fuel_cost
    )
    plants["min_down_time"] = plants.pop("ads_minimumdowntimehr")
    plants["min_up_time"] = plants.pop("ads_minimumuptimehr")

    # Ramp Limit Parameters
    plants["ramp_limit_up"] = (
        plants.pop("ads_rampup_ratemw/minute") / plants.p_nom * 60
    ).clip(
        lower=0,
        upper=1,
    )  # MW/min to p.u./hour
    plants["ramp_limit_down"] = (
        plants.pop("ads_rampdn_ratemw/minute") / plants.p_nom * 60
    ).clip(
        lower=0,
        upper=1,
    )  # MW/min to p.u./hour

    # Impute missing data based on average values of a given aggregation
    aggregation_fields = ["technology_description"]
    data_fields = [
        "start_up_cost",
        "min_down_time",
        "min_up_time",
        "ramp_limit_up",
        "ramp_limit_down",
    ]
    plants = impute_missing_plant_data(plants, aggregation_fields, data_fields)

    aggregation_fields = ["nerc_region", "technology_description"]
    data_fields = ["heat_rate"]
    plants = impute_missing_plant_data(plants, aggregation_fields, data_fields)

    plants["marginal_cost"] = (
        plants.heat_rate * plants.fuel_cost
    )  # (MMBTu/MW) * (USD/MMBTu) = USD/MW
    plants["efficiency"] = 1 / (
        plants["heat_rate"] / 3.412
    )  # MMBTu/MWh to MWh_electric/MWh_thermal
    return plants.reset_index()


def prepare_heat_rates(
    plants: pd.DataFrame,
    heat_rates: pd.DataFrame,
    cems_fn: str,
    crosswalk_fn: str,
):
    heat_rates['generator_name'] = (
        heat_rates['plant_name_eia'].astype(str) + '_' + 
        heat_rates['plant_id_eia'].astype(str) + '_' + 
        heat_rates['generator_id'].astype(str)
    )

    # Calculate mean and standard deviation for each generator
    heat_rate_stats_temporal = heat_rates.groupby(['generator_name'])['unit_heat_rate_mmbtu_per_mwh'].agg(['mean', 'std']).reset_index()
    heat_rate_stats_temporal['mean'] = heat_rate_stats_temporal['mean'].replace(np.inf, np.nan)
    heat_rate_stats_temporal.dropna(inplace=True)

    # Merge mean and std back to the original dataframe
    heat_rates = heat_rates.merge(
        heat_rate_stats_temporal,
        on=['generator_name'],
        how='left',
        suffixes=('', '_stats')
    )

    # Calculate the Z-score for each month's entry
    heat_rates['z_score'] = (heat_rates['unit_heat_rate_mmbtu_per_mwh'] - heat_rates['mean']) / heat_rates['std']

    # Filter out the outliers using Z-score
    threshold = 3
    filtered_heat_rates = heat_rates[np.abs(heat_rates['z_score']) <= threshold]

    # Further filter outliers using IQR for each generator
    def filter_outliers_iqr_grouped(df, group_column, value_column):
        def filter_outliers(group):
            Q1 = group[value_column].quantile(0.25)
            Q3 = group[value_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return group[(group[value_column] >= lower_bound) & (group[value_column] <= upper_bound)]
        return df.groupby(group_column).apply(filter_outliers).reset_index(drop=True)

    # Apply IQR filtering to each generator group
    filtered_heat_rates =  filter_outliers_iqr_grouped(filtered_heat_rates, 'technology_description', 'unit_heat_rate_mmbtu_per_mwh')
    average_heat_rates = filtered_heat_rates.groupby(['plant_id_eia', 'generator_id'])['unit_heat_rate_mmbtu_per_mwh'].mean().reset_index()
    plants.drop(columns = 'unit_heat_rate_mmbtu_per_mwh', inplace = True)
    plants = pd.merge(left=plants, right= average_heat_rates, on = ['plant_id_eia', 'generator_id'], how = 'left')

    cems_hr = pd.read_excel(cems_fn)[['Facility ID','Unit ID', 'Heat Input (mmBtu/MWh)']]
    crosswalk = pd.read_csv(crosswalk_fn)[['CAMD_PLANT_ID','CAMD_UNIT_ID', 'EIA_PLANT_ID', 'EIA_GENERATOR_ID']]
    cems_hr = pd.merge(cems_hr, crosswalk, left_on=['Facility ID','Unit ID'], right_on=['CAMD_PLANT_ID','CAMD_UNIT_ID'], how = 'inner')
    plants = pd.merge(cems_hr, plants, left_on=['EIA_PLANT_ID', 'EIA_GENERATOR_ID'], right_on=['plant_id_eia','generator_id'], how='right')
    plants.rename(columns={'Heat Input (mmBtu/MWh)':'heat_rate_'}, inplace=True)
    plants['heat_rate_'] = plants['heat_rate_']/1e3
    plants.heat_rate_.fillna(plants.unit_heat_rate_mmbtu_per_mwh)
    plants.unit_heat_rate_mmbtu_per_mwh = plants.pop('heat_rate_')
    plants.drop(columns=['Facility ID','Unit ID','CAMD_PLANT_ID','CAMD_UNIT_ID', 'EIA_PLANT_ID', 'EIA_GENERATOR_ID'], inplace=True)
    return plants


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_powerplants")
        rootpath = ".."
    else:
        rootpath = "."

    eia_data_operable, heat_rates = load_pudl_data()
    eia_data_operable = prepare_heat_rates(eia_data_operable, heat_rates, snakemake.input.cems, snakemake.input.epa_crosswalk)
    set_non_conus(eia_data_operable)
    set_derates(eia_data_operable)
    set_tech_fuels_primer_movers(eia_data_operable)
    eia_ads_merged = merge_ads_data(eia_data_operable)
    plants = set_parameters(eia_ads_merged)

    # temp throwing out plants without
    missing_locations = plants[plants.longitude.isna() | plants.latitude.isna()]
    print('Tossing out plants without locations:', missing_locations.shape[0])
    plants = plants[~plants.index.isin(missing_locations.index)]
    print(plants)

    plants.to_csv(snakemake.output.powerplants, index=False)