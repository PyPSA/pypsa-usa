"""Assimilates data on existing generator and storage resources from PUDL, CEMS, ADS, and other sources."""

import logging
import re

import duckdb
import numpy as np
import pandas as pd
from _helpers import configure_logging, weighted_avg

logger = logging.getLogger(__name__)


def initialize_duckdb():
    duckdb.connect(database=":memory:", read_only=False)
    duckdb.query("INSTALL httpfs;")


def load_eia_operable_data(parquet_path: str):
    """Queries the parquet files directly for operable plant data."""
    return duckdb.query(
        f"""
        WITH monthly_generators AS (
            SELECT
                plant_id_eia,
                generator_id,
                array_agg(unit_heat_rate_mmbtu_per_mwh ORDER BY report_date DESC) FILTER (WHERE unit_heat_rate_mmbtu_per_mwh IS NOT NULL)[1] AS unit_heat_rate_mmbtu_per_mwh
            FROM read_parquet('{parquet_path}/out_eia__monthly_generators.parquet')
            WHERE report_date >= '2023-01-01'
            GROUP BY plant_id_eia, generator_id
        )
        SELECT
            yg.plant_id_eia,
            yg.generator_id,
            array_agg(yg.plant_name_eia ORDER BY yg.report_date DESC) FILTER (WHERE yg.plant_name_eia IS NOT NULL)[1] AS plant_name_eia,
            array_agg(yg.capacity_mw ORDER BY yg.report_date DESC) FILTER (WHERE yg.capacity_mw IS NOT NULL)[1] AS capacity_mw,
            array_agg(yg.summer_capacity_mw ORDER BY yg.report_date DESC) FILTER (WHERE yg.summer_capacity_mw IS NOT NULL)[1] AS summer_capacity_mw,
            array_agg(yg.winter_capacity_mw ORDER BY yg.report_date DESC) FILTER (WHERE yg.winter_capacity_mw IS NOT NULL)[1] AS winter_capacity_mw,
            array_agg(yg.minimum_load_mw ORDER BY yg.report_date DESC) FILTER (WHERE yg.minimum_load_mw IS NOT NULL)[1] AS minimum_load_mw,
            array_agg(yg.energy_source_code_1 ORDER BY yg.report_date DESC) FILTER (WHERE yg.energy_source_code_1 IS NOT NULL)[1] AS energy_source_code_1,
            array_agg(yg.technology_description ORDER BY yg.report_date DESC) FILTER (WHERE yg.technology_description IS NOT NULL)[1] AS technology_description,
            array_agg(yg.operational_status ORDER BY yg.report_date DESC) FILTER (WHERE yg.operational_status IS NOT NULL)[1] AS operational_status,
            array_agg(yg.prime_mover_code ORDER BY yg.report_date DESC) FILTER (WHERE yg.prime_mover_code IS NOT NULL)[1] AS prime_mover_code,
            array_agg(yg.planned_generator_retirement_date ORDER BY yg.report_date DESC) FILTER (WHERE yg.planned_generator_retirement_date IS NOT NULL)[1] AS planned_generator_retirement_date,
            array_agg(yg.energy_storage_capacity_mwh ORDER BY yg.report_date DESC) FILTER (WHERE yg.energy_storage_capacity_mwh IS NOT NULL)[1] AS energy_storage_capacity_mwh,
            array_agg(yg.generator_operating_date ORDER BY yg.report_date DESC) FILTER (WHERE yg.generator_operating_date IS NOT NULL)[1] AS generator_operating_date,
            array_agg(yg.state ORDER BY yg.report_date DESC) FILTER (WHERE yg.state IS NOT NULL)[1] AS state,
            array_agg(yg.latitude ORDER BY yg.report_date DESC) FILTER (WHERE yg.latitude IS NOT NULL)[1] AS latitude,
            array_agg(yg.longitude ORDER BY yg.report_date DESC) FILTER (WHERE yg.longitude IS NOT NULL)[1] AS longitude,
            array_agg(ges.max_charge_rate_mw ORDER BY ges.report_date DESC) FILTER (WHERE ges.max_charge_rate_mw IS NOT NULL)[1] AS max_charge_rate_mw,
            array_agg(ges.max_discharge_rate_mw ORDER BY ges.report_date DESC) FILTER (WHERE ges.max_discharge_rate_mw IS NOT NULL)[1] AS max_discharge_rate_mw,
            array_agg(ges.storage_technology_code_1 ORDER BY ges.report_date DESC) FILTER (WHERE ges.storage_technology_code_1 IS NOT NULL)[1] AS storage_technology_code_1,
            array_agg(p.nerc_region ORDER BY p.report_date DESC) FILTER (WHERE p.nerc_region IS NOT NULL)[1] AS nerc_region,
            array_agg(p.balancing_authority_code_eia ORDER BY p.report_date DESC) FILTER (WHERE p.balancing_authority_code_eia IS NOT NULL)[1] AS balancing_authority_code_eia,
            array_agg(yg.current_planned_generator_operating_date ORDER BY yg.report_date DESC) FILTER (WHERE yg.current_planned_generator_operating_date IS NOT NULL)[1] AS current_planned_generator_operating_date,
            array_agg(yg.operational_status_code ORDER BY yg.report_date DESC) FILTER (WHERE yg.operational_status_code IS NOT NULL)[1] AS operational_status_code,
            array_agg(yg.generator_retirement_date ORDER BY yg.report_date DESC) FILTER (WHERE yg.generator_retirement_date IS NOT NULL)[1] AS generator_retirement_date,
            array_agg(yg.fuel_type_code_pudl ORDER BY yg.report_date DESC) FILTER (WHERE yg.fuel_type_code_pudl IS NOT NULL)[1] AS fuel_type_code_pudl,
            first(mg.unit_heat_rate_mmbtu_per_mwh) AS unit_heat_rate_mmbtu_per_mwh
        FROM read_parquet('{parquet_path}/out_eia__yearly_generators.parquet') yg
        LEFT JOIN read_parquet('{parquet_path}/core_eia860__scd_generators_energy_storage.parquet') ges
            ON yg.plant_id_eia = ges.plant_id_eia AND yg.generator_id = ges.generator_id
        LEFT JOIN read_parquet('{parquet_path}/core_eia860__scd_plants.parquet') p
            ON yg.plant_id_eia = p.plant_id_eia
        LEFT JOIN monthly_generators mg
            ON yg.plant_id_eia = mg.plant_id_eia AND yg.generator_id = mg.generator_id
        WHERE
            yg.operational_status_code IN ('RE','OP', 'SC', 'SB', 'CO' ,'U', 'V', 'TS', 'T')
            AND yg.report_date >= '2023-01-01'
        GROUP BY yg.plant_id_eia, yg.generator_id
    """,
    ).to_df()


def load_heat_rates_data(parquet_path: str, start_date: str, end_date: str):
    """Queries the parquet files for heat rate and fuel cost data within the specified date range."""
    query = f"""
    WITH monthly_generators AS (
        SELECT
            plant_id_eia,
            generator_id,
            report_date,
            unit_heat_rate_mmbtu_per_mwh,
            fuel_cost_per_mwh,
            fuel_cost_per_mmbtu
        FROM read_parquet('{parquet_path}/out_eia__monthly_generators.parquet')
        WHERE operational_status = 'existing'
        AND report_date BETWEEN '{start_date}' AND '{end_date}'
        AND unit_heat_rate_mmbtu_per_mwh IS NOT NULL
    )
    SELECT
        mg.plant_id_eia,
        mg.generator_id,
        mg.report_date,
        mg.unit_heat_rate_mmbtu_per_mwh,
        mg.fuel_cost_per_mwh,
        mg.fuel_cost_per_mmbtu,
        yg.plant_name_eia,
        yg.capacity_mw,
        yg.energy_source_code_1,
        yg.technology_description,
        yg.operational_status,
        yg.prime_mover_code,
        yg.state,
        p.nerc_region,
        p.balancing_authority_code_eia
    FROM monthly_generators mg
    LEFT JOIN read_parquet('{parquet_path}/out_eia__yearly_generators.parquet') yg
        ON mg.plant_id_eia = yg.plant_id_eia AND mg.generator_id = yg.generator_id
    LEFT JOIN read_parquet('{parquet_path}/core_eia860__scd_plants.parquet') p
        ON mg.plant_id_eia = p.plant_id_eia
    WHERE yg.operational_status = 'existing'
    ORDER BY mg.report_date DESC
    """
    return duckdb.query(query).to_df()


def set_non_conus(eia_data_operable):
    """Set NERC region and balancing authority code for non-CONUS plants."""
    eia_data_operable.loc[eia_data_operable.state.isin(["AK", "HI"]), "nerc_region"] = "non-conus"
    eia_data_operable.loc[
        eia_data_operable.state.isin(["AK", "HI"]),
        "balancing_authority_code",
    ] = "non-conus"


def set_derates(plants):
    plants["derate_summer_capacity"] = np.minimum(
        plants.summer_capacity_mw,
        plants.ads_maxcapmw.fillna(np.inf),
    )
    plants["derate_winter_capacity"] = np.minimum(
        plants.winter_capacity_mw,
        plants.ads_maxcapmw.fillna(np.inf),
    )

    plants["summer_derate"] = 1 - ((plants.p_nom - plants.derate_summer_capacity) / plants.p_nom)
    plants["winter_derate"] = 1 - ((plants.p_nom - plants.derate_winter_capacity) / plants.p_nom)
    plants.summer_derate = plants.summer_derate.clip(
        upper=1,
    ).clip(lower=0)
    plants.winter_derate = plants.winter_derate.clip(
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
eia_tech_map = eia_tech_map.set_index("Technology")
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
            "battery",
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
eia_fuel_map = eia_fuel_map.set_index("Energy Source 1")
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
eia_primemover_map = eia_primemover_map.set_index("Prime Mover")


def set_tech_fuels_primer_movers(eia_data_operable):
    """
    Maps technologies, fuels, and prime movers from EIA data to PyPSA carrier
    names.
    """
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
    return [prefix + col.lower().replace(" ", "_").replace("(", "").replace(")", "") + suffix for col in columns]


def merge_ads_data(eia_data_operable):
    """Merges WECC ADS Data into the prepared EIA Data."""
    path_ads = snakemake.input.wecc_ads
    ads_thermal = pd.read_csv(
        path_ads + "/Thermal_General_Info.csv",
        skiprows=1,
    )
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
        path_ads + "/Thermal_IOCurve_Info.csv",
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

    # loading ads to match ads_name with generator key in order to link with ads thermal file
    ads = pd.read_csv(
        path_ads + "/GeneratorList.csv",
        skiprows=2,
        encoding="unicode_escape",
    )
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
    ads = ads.rename(
        {
            "Name": "ads_name",
            "Long Name": "ads_long_name",
            "SubType": "subtype",
            "Commission Date": "commission_date",
            "Retirement Date": "retirement_date",
            "Area Name": "balancing_area",
        },
        axis=1,
    )
    ads = ads.rename(str.lower, axis="columns")
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
        ads_thermal_ioc["generatorname"].str.replace(" ", "").str.lower().str.replace("_", "").str.replace("-", "")
    )
    ads_thermal_ioc["generator_key"] = ads_thermal_ioc["generator_name_alt"].map(
        ads_name_key_dict,
    )

    # Identify Generators not in ads generator list that are in the IOC curve.
    # This could potentially be matched with manual work.
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
    eia_ads_mapper = eia_ads_mapper.dropna(subset=["mapper_plant_id_eia"])
    eia_ads_mapper.mapper_plant_id_eia = eia_ads_mapper.mapper_plant_id_eia.astype(int)
    eia_ads_mapper.mapper_ads_name = eia_ads_mapper.mapper_ads_name.astype(str)
    eia_ads_mapper.mapper_generatorkey = eia_ads_mapper.mapper_generatorkey.astype(int)

    ads_complete = ads_complete.dropna(subset=["ads_generator_key"])
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
    eia_ads_merged = eia_ads_merged.drop(columns=eia_ads_mapper.columns)
    eia_ads_merged = eia_ads_merged.drop(
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
    )
    eia_ads_merged = eia_ads_merged.drop_duplicates(
        subset=["plant_id_eia", "generator_id"],
        keep="first",
    )

    return eia_ads_merged


def impute_missing_plant_data(
    plants: pd.DataFrame,
    aggregation_fields: list[str],
    data_fields: list[str],
) -> pd.DataFrame:
    """
    Imputes missing data for the`data_fields` in the plants dataframe based on
    the average values of the  `aggregation_fields`.
    """
    # Calculate the weighted averages excluding NaNs
    weighted_averages = (
        plants.groupby(aggregation_fields)[plants.columns]
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
        if field in ["fuel_cost", "heat_rate"]:
            # need to properly assign weighted average to the entries which took their values
            # if the field has values equal to the _weighted column, then the source is the weighted average
            plants_merged[f"{field}_source"] = np.where(
                plants_merged[field] == plants_merged[f"{field}_weighted"],
                "weighted_average",
                plants_merged[f"{field}_source"],
            )
    # Drop the weighted average columns after filling NaNs
    plants_merged = plants_merged.drop(
        columns=[f"{field}_weighted" for field in data_fields],
    )
    return plants_merged.set_index("generator_name")


def set_parameters(plants: pd.DataFrame):
    """
    Sets generator naming schemes, updates parameter names, and imputes missing
    data.
    """
    plants = plants[plants.nerc_region.isin(["WECC", "TRE", "MRO", "SERC", "RFC", "NPCC"])]
    plants = plants.rename(
        {
            "fuel_cost_per_mwh_source": "fuel_cost_source",
            "unit_heat_rate_mmbtu_per_mwh_source": "heat_rate_source",
        },
        axis=1,
    )

    plants["generator_name"] = (
        plants.plant_name_eia.astype(str)
        + "_"
        + plants.plant_id_eia.astype(str)
        + "_"
        + plants.generator_id.astype(str)
    )
    plants = plants.set_index("generator_name")
    plants["p_nom"] = plants.pop("capacity_mw")
    plants["build_year"] = plants.pop("generator_operating_date").dt.year
    plants["heat_rate"] = plants.pop("unit_heat_rate_mmbtu_per_mwh")
    plants["vom"] = plants.pop("ads_vom_cost")
    plants["fuel_cost"] = plants.pop("fuel_cost_per_mmbtu")

    zero_mc_fuel_types = ["solar", "wind", "hydro", "geothermal", "battery"]
    plants.loc[plants.fuel_type.isin(zero_mc_fuel_types), "fuel_cost"] = 0
    plants = impute_missing_plant_data(
        plants,
        ["state", "fuel_name"],
        ["fuel_cost"],
    )
    plants = impute_missing_plant_data(
        plants,
        ["balancing_authority_code_eia", "fuel_name"],
        ["fuel_cost"],
    )
    plants = impute_missing_plant_data(
        plants,
        ["nerc_region", "fuel_name"],
        ["fuel_cost"],
    )

    plants = impute_missing_plant_data(plants, ["fuel_name"], ["fuel_cost"])
    plants = impute_missing_plant_data(plants, ["prime_mover_code"], ["fuel_cost"])
    plants.loc[plants.carrier.isin(["nuclear"]), "fuel_cost"] = np.float32(0.71)  # 2023 AEO

    # Unit Commitment Parameters
    plants["start_up_cost"] = plants.pop("ads_startup_cost_fixed$") + plants.ads_startfuelmmbtu * plants.fuel_cost
    plants["min_down_time"] = plants.pop("ads_minimumdowntimehr")
    plants["min_up_time"] = plants.pop("ads_minimumuptimehr")

    # Ramp Limit Parameters
    plants["ramp_limit_up"] = (plants.pop("ads_rampup_ratemw/minute") / plants.p_nom * 60).clip(
        lower=0,
        upper=1,
    )  # MW/min to p.u./hour
    plants["ramp_limit_down"] = (plants.pop("ads_rampdn_ratemw/minute") / plants.p_nom * 60).clip(
        lower=0,
        upper=1,
    )  # MW/min to p.u./hour

    # Impute missing data based on average values of a given aggregation
    data_fields = [
        "start_up_cost",
        "min_down_time",
        "min_up_time",
        "ramp_limit_up",
        "ramp_limit_down",
        "vom",
    ]
    plants = impute_missing_plant_data(plants, ["technology_description"], data_fields)
    plants = impute_missing_plant_data(plants, ["prime_mover_code"], data_fields)
    plants = impute_missing_plant_data(plants, ["carrier"], data_fields)

    # replace heat-rate above theoretical minimum with nan
    plants.loc[plants.heat_rate < 3.412, "heat_rate"] = np.nan
    plants.loc[
        plants.fuel_type.isin(["solar", "wind", "hydro", "battery"]),
        "heat_rate",
    ] = 3.412

    plants = impute_missing_plant_data(
        plants,
        ["nerc_region", "prime_mover_code"],
        ["heat_rate"],
    )
    plants = impute_missing_plant_data(
        plants,
        ["nerc_region", "technology_description"],
        ["heat_rate"],
    )
    plants = impute_missing_plant_data(
        plants,
        ["nerc_region", "prime_mover_code"],
        ["heat_rate"],
    )
    plants = impute_missing_plant_data(plants, ["prime_mover_code"], ["heat_rate"])
    plants = impute_missing_plant_data(
        plants,
        ["technology_description"],
        ["heat_rate"],
    )
    plants = impute_missing_plant_data(plants, ["carrier"], ["heat_rate"])

    plants["marginal_cost"] = plants.vom + (plants.fuel_cost * plants.heat_rate)  # (MMBTu/MW) * (USD/MMBTu) = USD/MW
    plants["efficiency"] = 1 / (plants["heat_rate"] / 3.412)  # MMBTu/MWh to MWh_electric/MWh_thermal

    set_derates(plants)

    plants["heat_rate_source"] = plants["heat_rate_source"].fillna("NA")
    plants["fuel_cost_source"] = plants["fuel_cost_source"].fillna("NA")

    # Check for missing heat rate data
    if plants["heat_rate"].isna().sum() > 0:
        logger.warning(
            "Missing {} heat rate records.".format(plants["heat_rate"].isna().sum()),
        )

    # Check for missing fuel cost data
    if plants["fuel_cost"].isna().sum() > 0:
        logger.warning(
            "Missing {} fuel cost records.".format(plants["fuel_cost"].isna().sum()),
        )

    # Remove all column names that start with "ads_" except ads_mustrun
    plants = plants.loc[:, ~plants.columns.str.startswith("ads_") | (plants.columns == "ads_mustrun")]

    # Round all numeric columns to 4 decimal places
    plants = plants.round(4)

    return plants.reset_index()


def filter_outliers_iqr_grouped(df, group_column, value_column):
    """Filter outliers using IQR for each generator group."""

    def filter_outliers(group):
        q1 = group[value_column].quantile(0.25)
        q3 = group[value_column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return group[(group[value_column] >= lower_bound) & (group[value_column] <= upper_bound)]

    return df.groupby(group_column)[df.columns].apply(filter_outliers).reset_index(drop=True)


def filter_outliers_zscore(temporal_data, target_field_name):
    """Filter outliers using Z-score."""
    # Calculate mean and standard deviation for each generator
    stats = temporal_data.groupby(["generator_name"])[target_field_name].agg(["mean", "std"]).reset_index()
    stats["mean"] = stats["mean"].replace(np.inf, np.nan)
    stats = stats.dropna()

    # Merge mean and std back to the original dataframe
    temporal_stats = temporal_data.merge(
        stats,
        on=["generator_name"],
        how="left",
        suffixes=("", "_stats"),
    )

    # Calculate the Z-score for each month's entry
    temporal_stats["z_score"] = (temporal_stats[target_field_name] - temporal_stats["mean"]) / temporal_stats["std"]

    # Filter out the outliers using Z-score
    threshold = 3
    filtered_temporal = temporal_stats[np.abs(temporal_stats["z_score"]) <= threshold]
    filtered_temporal = filtered_temporal.drop(columns=["mean", "std", "z_score"])
    return filtered_temporal


def merge_fc_hr_data(
    plants: pd.DataFrame,
    temporal_data: pd.DataFrame,
    target_field_name: str,
):
    temporal_data["generator_name"] = (
        temporal_data["plant_name_eia"].astype(str)
        + "_"
        + temporal_data["plant_id_eia"].astype(str)
        + "_"
        + temporal_data["generator_id"].astype(str)
    )

    # Apply Z-score filtering to each generator
    filtered_temporal = filter_outliers_zscore(temporal_data, target_field_name)

    # Apply IQR filtering to each generator group
    filtered_temporal = filter_outliers_iqr_grouped(
        filtered_temporal,
        "technology_description",
        target_field_name,
    )

    # Apply temporal average heat rates to plants dataframe
    temporal_average = (
        filtered_temporal.groupby(["plant_id_eia", "generator_id"])[target_field_name].mean().reset_index()
    )

    if target_field_name in plants.columns:
        plants = plants.drop(columns=[target_field_name])

    temporal_average[f"{target_field_name}_source"] = "pudl_reciepts"

    plants = pd.merge(
        left=plants,
        right=temporal_average,
        on=["plant_id_eia", "generator_id"],
        how="left",
    )
    return plants


def apply_cems_heat_rates(plants, crosswalk_fn, cems_fn):
    # Apply CEMS calculated heat rates
    cems_hr = pd.read_excel(cems_fn)[["Facility ID", "Unit ID", "Heat Input (mmBtu/MWh)"]]
    crosswalk = pd.read_csv(crosswalk_fn)[["CAMD_PLANT_ID", "CAMD_UNIT_ID", "EIA_PLANT_ID", "EIA_GENERATOR_ID"]]
    cems_hr = pd.merge(
        cems_hr,
        crosswalk,
        left_on=["Facility ID", "Unit ID"],
        right_on=["CAMD_PLANT_ID", "CAMD_UNIT_ID"],
        how="inner",
    )
    cems_hr["hr_source_cems"] = "cems"
    plants = pd.merge(
        cems_hr,
        plants,
        left_on=["EIA_PLANT_ID", "EIA_GENERATOR_ID"],
        right_on=["plant_id_eia", "generator_id"],
        how="right",
    )

    plants = plants.rename(columns={"Heat Input (mmBtu/MWh)": "heat_rate_"})
    plants.heat_rate_ = plants.heat_rate_.fillna(
        plants.unit_heat_rate_mmbtu_per_mwh,
    )  # First take CEMS, then use PUDL
    plants.unit_heat_rate_mmbtu_per_mwh = plants.pop("heat_rate_")

    plants.hr_source_cems = plants.hr_source_cems.fillna(
        "unit_heat_rate_mmbtu_per_mwh_source",
    )
    plants.unit_heat_rate_mmbtu_per_mwh_source = plants.pop("hr_source_cems")

    plants = plants.drop(
        columns=[
            "Facility ID",
            "Unit ID",
            "CAMD_PLANT_ID",
            "CAMD_UNIT_ID",
            "EIA_PLANT_ID",
            "EIA_GENERATOR_ID",
        ],
    )

    return plants


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_powerplants")
        rootpath = ".."
    else:
        rootpath = "."
    configure_logging(snakemake)

    weather_year = snakemake.params.renewable_weather_year[0]
    # Cap the year at 2023 if it's greater
    data_year = min(int(weather_year), 2023)
    start_date = f"{data_year}-01-01"
    end_date = f"{data_year + 1}-01-01"

    initialize_duckdb()
    eia_data_operable = load_eia_operable_data(snakemake.params.pudl_path)
    heat_rates = load_heat_rates_data(snakemake.params.pudl_path, start_date, end_date)

    eia_data_operable = merge_fc_hr_data(
        eia_data_operable,
        heat_rates,
        "unit_heat_rate_mmbtu_per_mwh",
    )
    eia_data_operable = merge_fc_hr_data(
        eia_data_operable,
        heat_rates,
        "fuel_cost_per_mwh",
    )
    eia_data_operable = merge_fc_hr_data(
        eia_data_operable,
        heat_rates,
        "fuel_cost_per_mmbtu",
    )
    eia_data_operable = apply_cems_heat_rates(
        eia_data_operable,
        snakemake.input.epa_crosswalk,
        snakemake.input.cems,
    )
    set_non_conus(eia_data_operable)
    set_tech_fuels_primer_movers(eia_data_operable)
    eia_ads_merged = merge_ads_data(eia_data_operable)
    plants = set_parameters(eia_ads_merged)

    # Throwing out plants without GPS data
    missing_locations = plants[plants.longitude.isna() | plants.latitude.isna()]
    logger.warning(
        f"Tossing out plants without locations: {missing_locations.shape[0]}",
    )
    # plants[plants.index.isin(missing_locations.index)].to_csv('missing_gps_pudl.csv')
    plants = plants[~plants.index.isin(missing_locations.index)]

    logger.info(f"Exporting Powerplants, with {plants.shape[0]} entries.")

    # Sort columns alphabetically for consistent diffing
    plants = plants.reindex(sorted(plants.columns), axis=1)
    # Sort rows by generator_name for consistent diffing
    plants = plants.sort_values("generator_name")

    plants.to_csv(snakemake.output.powerplants, index=False)
