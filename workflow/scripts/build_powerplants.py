import pandas as pd
import duckdb
import re
import numpy as np
from scipy.optimize import minimize


def load_pudl_data():
    duckdb.connect(database=":memory:", read_only=False)

    duckdb.query("INSTALL sqlite;")
    duckdb.query(
        f"""
        ATTACH '{snakemake.input.pudl}' (TYPE SQLITE);
        USE pudl;
        """
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
    """
    ).to_df()

    return eia_data_operable


def set_non_conus(eia_data_operable):
    eia_data_operable.loc[eia_data_operable.state.isin(["AK", "HI"]), "nerc_region"] = (
        "non-conus"
    )
    eia_data_operable.loc[
        eia_data_operable.state.isin(["AK", "HI"]), "balancing_authority_code"
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
        upper=1
    ).clip(lower=0)
    eia_data_operable.winter_derate = eia_data_operable.winter_derate.clip(
        upper=1
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
    }
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
    }
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
    }
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
    """Standardize column names by removing spaces, converting to lowercase, removing parentheses, and adding prefix and suffix."""
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
            "IncCap2(MW)",
            "IncHR2(MMBTu/MWh)",
            "IncCap3(MW)",
            "IncHR3(MMBTu/MWh)",
            "IncCap4(MW)",
            "IncHR4(MMBTu/MWh)",
            "IncCap5(MW)",
            "IncHR5(MMBTu/MWh)",
            "IncCap6(MW)",
            "IncHR6(MMBTu/MWh)",
            "IncCap7(MW)",
            "IncHR7(MMBTu/MWh)",
        ]
    ]
    ads_ioc["IncHR2(MMBTu/MWh)"] = ads_ioc["IncHR2(MMBTu/MWh)"].replace(0, np.nan)
    ads_ioc.columns = standardize_col_names(ads_ioc.columns)

    ads_ioc["inchr1mmbtu/mwh"] = ads_ioc.mininputmmbtu / ads_ioc.iomincapmw
    ads_ioc.rename(
        columns={
            "inchr1mmbtu/mwh": "hr1",
            "inchr2mmbtu/mwh": "hr2",
            "inchr3mmbtu/mwh": "hr3",
            "inchr4mmbtu/mwh": "hr4",
            "inchr5mmbtu/mwh": "hr5",
            "inchr6mmbtu/mwh": "hr6",
            "inchr7mmbtu/mwh": "hr7",
            "iomincapmw": "x_1",
            "mininputmmbtu": "mmbtu_1",
        },
        inplace=True,
    )

    for i in range(2, 8):
        ads_ioc[f"x_{i}"] = ads_ioc[f"x_{i-1}"] + ads_ioc[f"inccap{i}mw"]
        ads_ioc[f"mmbtu_{i}"] = ads_ioc[f"x_{i}"] * ads_ioc[f"hr{i}"]

    for i in range(0, ads_ioc.shape[0]):
        for j in range(2, 8):
            if ads_ioc[f"hr{j}"][i] == 0:
                ads_ioc[f"hr{j}"][i] = ads_ioc[f"hr{j-1}"][i]

    def detail_linspace(x_values, y_values, num_points):
        # Arrays to hold the detailed linspace results
        x_detailed = np.array([])
        y_detailed = np.array([])

        for i in range(len(x_values) - 1):
            if x_values[i] == x_values[i + 1]:
                continue
            # Generate linspace for x values
            x_segment = np.linspace(
                x_values[i], x_values[i + 1], num_points, endpoint=False
            )

            # Calculate the slope of the segment
            slope = (y_values[i + 1] - y_values[i]) / (x_values[i + 1] - x_values[i])

            # Generate y values based on the slope and start point
            y_segment = slope * (x_segment - x_values[i]) + y_values[i]

            # Append the segment to the detailed arrays
            x_detailed = np.concatenate((x_detailed, x_segment))
            y_detailed = np.concatenate((y_detailed, y_segment))

        return x_detailed, y_detailed

    # Define quadratic error function
    def quadratic_error_function(params, x, y_true):
        a, b, c = params
        y_pred = a * x**2 + b * x + c
        return np.sum((y_true - y_pred) ** 2)

    def linear_error_function(params, x, y_true):
        a, b = params
        y_pred = a * x + b
        return np.sum((y_true - y_pred) ** 2)

    ads_ioc["linear_a"] = 0
    ads_ioc["linear_b"] = 0
    ads_ioc["quadratic_a"] = 0
    ads_ioc["quadratic_b"] = 0
    ads_ioc["quadratic_c"] = 0
    ads_ioc["avg_hr"] = 0

    for generator_index in range(ads_ioc.shape[0]):
        # generator_index = 0
        x_set_points = ads_ioc[
            ["x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7"]
        ].values[generator_index, :]
        y_vals_hr = ads_ioc[["hr1", "hr2", "hr3", "hr4", "hr5", "hr6", "hr7"]].values[
            generator_index, :
        ]
        y_vals = ads_ioc[
            [
                "mmbtu_1",
                "mmbtu_2",
                "mmbtu_3",
                "mmbtu_4",
                "mmbtu_5",
                "mmbtu_6",
                "mmbtu_7",
            ]
        ].values[generator_index, :]

        x_linspace, y_linspace = detail_linspace(x_set_points, y_vals, 10)

        initial_guess = [0.1, 0.1, 0.1]
        result_quad = minimize(
            quadratic_error_function, initial_guess, args=(x_linspace, y_linspace)
        )

        initial_guess_lin = [0.1, 0.1]
        result_linear = minimize(
            linear_error_function, initial_guess_lin, args=(x_linspace, y_linspace)
        )

        a_opt, b_opt, c_opt = result_quad.x
        # print(f"Quadratic parameters: a = {a_opt}, b = {b_opt}, c = {c_opt}")

        a_opt_lin, b_opt_lin = result_linear.x
        # print(f"Linear parameters: a = {a_opt_lin}, b = {b_opt_lin}")

        avg_hr = np.mean((a_opt_lin * x_linspace + b_opt_lin) / x_linspace)
        # print(f"Average heat rate: {avg_hr}")

        ads_ioc.loc[generator_index, "linear_a"] = a_opt_lin
        ads_ioc.loc[generator_index, "linear_b"] = b_opt_lin
        ads_ioc.loc[generator_index, "quadratic_a"] = a_opt
        ads_ioc.loc[generator_index, "quadratic_b"] = b_opt
        ads_ioc.loc[generator_index, "quadratic_c"] = c_opt
        ads_ioc.loc[generator_index, "avg_hr"] = avg_hr

    # Check for inf and nan values in avg_hr, and replace with nan.
    # This is done so we can identify plants without data, then replace with averages later
    ads_ioc["avg_hr"] = ads_ioc["avg_hr"].replace([np.inf, -np.inf], np.nan)

    # Plotting IOC Results
    generator_index = 102  # 1050
    x_set_points = ads_ioc[["x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7"]].values[
        generator_index, :
    ]
    y_vals = ads_ioc[
        ["mmbtu_1", "mmbtu_2", "mmbtu_3", "mmbtu_4", "mmbtu_5", "mmbtu_6", "mmbtu_7"]
    ].values[generator_index, :]
    x_linspace, y_linspace = detail_linspace(x_set_points, y_vals, 10)

    a_opt, b_opt, c_opt = ads_ioc.loc[
        generator_index, ["quadratic_a", "quadratic_b", "quadratic_c"]
    ]
    a_opt_lin, b_opt_lin = ads_ioc.loc[generator_index, ["linear_a", "linear_b"]]

    # Merge ADS plant data with thermal IOC data
    ads_thermal_ioc = pd.merge(ads_thermal, ads_ioc, on="generatorname", how="left")
    ads_thermal_ioc.dropna(subset=["avg_hr"])

    # loading ads to match ads_name with generator key in order to link with ads thermal file
    ads = pd.read_csv(
        ADS_PATH + "/GeneratorList.csv", skiprows=2, encoding="unicode_escape"
    )
    # ads = ads[ads['State'].isin(['NM', 'AZ', 'CA', 'WA', 'OR', 'ID', 'WY', 'MT', 'UT', 'SD', 'CO', 'NV', 'NE', '0', 'TX'])]
    ads["Long Name"] = ads["Long Name"].astype(str)
    ads["Name"] = ads["Name"].str.replace(" ", "")
    ads["Name"] = ads["Name"].apply(lambda x: re.sub(r"[^a-zA-Z0-9]", "", x).lower())
    ads["Long Name"] = ads["Long Name"].str.replace(" ", "")
    ads["Long Name"] = ads["Long Name"].apply(
        lambda x: re.sub(r"[^a-zA-Z0-9]", "", x).lower()
    )
    ads["SubType"] = ads["SubType"].apply(
        lambda x: re.sub(r"[^a-zA-Z0-9]", "", x).lower()
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
            ["save to binary", "county", "city", "zipcode", "internalid"]
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
        ads_name_key_dict
    )

    # Identify Generators not in ads generator list that are in the IOC curve. This could potentially be matched with manual work.
    ads_thermal_ioc[ads_thermal_ioc.generator_key.isna()]

    # Merge ads thermal_IOC data with ads generator data
    # Only keeping thermal plants for their heat rate and ramping data
    ads_complete = ads_thermal_ioc.merge(
        ads, left_on="generator_key", right_on="generatorkey", how="left"
    )
    ads_complete.columns = standardize_col_names(ads_complete.columns, prefix="ads_")
    ads_complete = ads_complete.loc[~ads_complete.ads_state.isin(["MX"])]
    ads_complete

    ads_complete.pivot_table(
        index=["ads_fueltype"], values="ads_avg_hr", aggfunc="mean"
    ).sort_values("ads_avg_hr", ascending=False)

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
        eia_ads_mapper.columns, prefix="mapper_"
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
            "ads_x_1",
            "ads_mmbtu_1",
            "ads_inccap2mw",
            "ads_hr2",
            "ads_inccap3mw",
            "ads_hr3",
            "ads_inccap4mw",
            "ads_hr4",
            "ads_inccap5mw",
            "ads_hr5",
            "ads_inccap6mw",
            "ads_hr6",
            "ads_inccap7mw",
            "ads_hr7",
            "ads_hr1",
            "ads_x_2",
            "ads_mmbtu_2",
            "ads_x_3",
            "ads_mmbtu_3",
            "ads_x_4",
            "ads_mmbtu_4",
            "ads_x_5",
            "ads_mmbtu_5",
            "ads_x_6",
            "ads_mmbtu_6",
            "ads_x_7",
            "ads_mmbtu_7",
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
        ],
        inplace=True,
    )
    eia_ads_merged = eia_ads_merged.drop_duplicates(
        subset=["plant_id_eia", "generator_id"], keep="first"
    )

    return eia_ads_merged


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_powerplants")
        rootpath = ".."
    else:
        rootpath = "."

    eia_data_operable = load_pudl_data()
    set_non_conus(eia_data_operable)
    set_derates(eia_data_operable)
    set_tech_fuels_primer_movers(eia_data_operable)
    eia_ads_merged = merge_ads_data(eia_data_operable)
    eia_ads_merged.to_csv(snakemake.output.powerplants, index=False)
