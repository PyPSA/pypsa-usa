# Written by Kamran Tehranchi (Stanford University)
"""
This scripts stochastic samples for power systems models based on the Mean-Reversion Stochastic Process Method presented here: XXX placeholder for paper link XXX

"""
# %% Imports and functions


import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def import_profiles_from_folder(PATH_FILES):
    filelist = []
    for file in glob.glob(os.path.join(PATH_FILES, f"*.csv")):
        filelist.append(os.path.join(file))
    load_base = pd.read_csv([s for s in filelist if "loads" in s][0], index_col=0)
    solar_base = pd.read_csv([s for s in filelist if "solar" in s][0], index_col=0)
    wind_base = pd.read_csv([s for s in filelist if "wind" in s][0], index_col=0)
    return load_base, solar_base, wind_base


def import_profiles_from_network(PATH_NETWORK, num_clusters):
    """
    Imports the profiles from a pypsa network.

    Args:
        network_path (str): Path to the pypsa network.
        num_clusters (int): Number of clusters to be used in the network.

    Returns:
        pandas.DataFrame: Base load profile.
        pandas.DataFrame: Base solar profile.
        pandas.DataFrame: Base wind profile.
    """
    import pypsa

    network = pypsa.Network(PATH_NETWORK)
    # Export network to csv to be run in NodalStochastic Profile Generator
    network.loads_t.p_set.to_csv(
        os.path.join("resources/pypsa", f"loads_t_p_set_{num_clusters}.csv"),
    )

    # filter columns with solar in name
    solar_df = network.generators_t.p_max_pu.loc[
        :,
        network.generators_t.p_max_pu.columns.str.contains("solar"),
    ]
    solar_df.to_csv(
        os.path.join("resources/pypsa", f"solar_profiles_{num_clusters}.csv"),
    )

    # filter columns with wind in name and export as csv
    wind_df = network.generators_t.p_max_pu.loc[
        :,
        network.generators_t.p_max_pu.columns.str.contains("wind"),
    ]
    wind_df.to_csv(os.path.join("resources/pypsa", f"wind_profiles_{num_clusters}.csv"))

    load_base, solar_base, wind_base = import_profiles_from_folder("resources/pypsa")

    return load_base, solar_base, wind_base


def resample_and_group(base_data, area_mapping):
    """
    Resamples the input base data from hourly to daily, grouped by area id.

    Args:
        base_data (pandas.DataFrame): Input data to be resampled.
        area_mapping (dict): Dictionary mapping column names to area ids.

    Returns:
        pandas.DataFrame: Resampled data, grouped by day of year and area id.
    """
    base_area = base_data.groupby(
        base_data.columns.map(area_mapping.area_id),
        axis=1,
    ).mean()
    base_area["doy"] = timestamp_reference.day_of_year
    base_area_daily = base_area.groupby("doy").mean()
    base_area.drop(columns=["doy"], inplace=True)
    return base_area_daily, base_area


def create_allocation(base, mapping):
    """
    Creates an allocation of generation based on the mean of the base data.

    Args:
        base (pandas.DataFrame): Input data to be resampled.
        mapping (pandas.DataFrame): Dictionary mapping column names to area ids.

    Returns:
        pandas.DataFrame: Hourly allocation of generation based on the mean of the base data.
    """
    base_ = base.copy().reset_index(drop=True)
    total_area_hourly = base_.groupby(
        base_.columns.map(mapping.to_dict()["area_id"]),
        axis=1,
    ).sum()
    base_.columns = base.columns.map(mapping.to_dict()["area_id"])
    base_[base_ < 0.0001] = 0
    allocation = base_.apply(lambda x: x / total_area_hourly.loc[:, x.name], axis=0)
    allocation.fillna(0, inplace=True)
    return allocation


def define_solar_hours(solar_profile, timestamps):
    """
    Defines the first and last hour of the day for solar generation.

    Args:
        solar_profile (pandas.DataFrame): Solar generation profile.
        timestamps (pandas.DataFrame): Timestamps for the solar generation profile.

    Returns:
        pandas.DataFrame: First and last hour of the day for solar generation.
    """
    first_hour, last_hour, yesterday_last_hour = np.zeros((3, len(solar_profile)))
    last_selected_index = 0
    timestamps = timestamps.copy(deep=True)
    timestamps["solar_profile"] = pd.Series(solar_profile, name="solar_profile")

    for _, day_of_year_df in timestamps.groupby("day_of_year"):
        first_hour_idx = day_of_year_df.solar_profile.idxmax()
        last_hour_idx = day_of_year_df.solar_profile[::-1].idxmax()
        start_hour = day_of_year_df.solar_profile.idxmin()
        first_hour[first_hour_idx] = 1
        yesterday_last_hour[first_hour_idx] = (
            0 if start_hour < 23 else last_selected_index
        )
        last_hour[last_hour_idx] = 1
        last_selected_index = last_hour_idx

    return first_hour, last_hour, yesterday_last_hour


def assign_stochastic_mu(df_params, timestamp_reference, column_name="profile_type"):
    """
    Assigns a stochastic mu to each hour of the year based on the season and
    profile type.

    Args:
        df_params (pandas.DataFrame): Dataframe containing the parameters for the stochastic mu.
        timestamp_reference (pandas.DataFrame): Dataframe containing the timestamp reference.
        column_name (str): Column name to be used for the merge.

    Returns:
        pandas.DataFrame: Dataframe containing the stochastic mu for each hour of the year.
    """
    df_mu_hourly = pd.merge(
        left=timestamp_reference,
        right=df_params[[column_name, "area_id", "season", "mu"]],
        left_on="season_num",
        right_on="season",
        how="outer",
    )
    order = df_mu_hourly["area_id"].unique()
    df_mu_hourly_unstack = (
        df_mu_hourly.set_index(["timestamp", column_name, "area_id"])["mu"]
        .unstack()
        .reset_index()
    )
    df_mu_hourly_unstack.columns = df_mu_hourly_unstack.columns.map(str)
    df_mu_hourly_unstack = df_mu_hourly_unstack[np.append("timestamp", order)]
    return df_mu_hourly_unstack


def sampling_warmup(df, thresholds, rng, min_steps=50):
    """
    Performs a warmup of the sampling.

    Args:
        df (pandas.DataFrame): Dataframe containing the parameters for the ratio sample.
        thresholds (pandas.DataFrame): Dataframe containing the thresholds for the ratio sample.
        rng (numpy.random.Generator): Random number generator.
        min_steps (int): Minimum number of steps to be performed.

    Returns:
        pandas.DataFrame: Dataframe containing the first sample of the ratio and epsilon.
    """
    winter_params = df.query("season == 1")
    ratio = winter_params.mu
    for _ in range(min_steps):
        eps = rng_eps(winter_params, rng)
        ratio += winter_params.kai * (winter_params.mu - ratio) + eps
        ratio, eps, na = resample_ratio(
            ratio,
            winter_params,
            thresholds,
            eps,
            rng,
            True,
        )
    return ratio, eps


# idea: to better replicate inverter curtailment, i should retain out of threshold samples but maintain a seperate 'ratio' df that is the max(sample, 0) or min(sample, threshold))... this simplifies the need for re-sampling but retains proper distributional properties
def resample_ratio(
    original_ratio,
    params,
    thresholds,
    original_epsilon,
    rng,
    initialization,
):
    resample_count = 0
    param_dict = params.sampling_id.to_dict()
    while np.any(
        np.logical_or(original_ratio > thresholds, original_ratio < 0),
    ):  # if any values are out of bounds
        resampled_ratio = original_ratio.copy(deep=True)
        mask_to_resample = np.logical_or(
            original_ratio > thresholds,
            original_ratio < 0,
        )  # mask of values to resample
        resampled_epsilon = rng_eps(
            params[mask_to_resample.values],
            rng,
        )  # resample epsilon for values to resample
        resampled_ratio[mask_to_resample] = (
            original_ratio[mask_to_resample]
            - original_epsilon[mask_to_resample]
            + resampled_epsilon.values
        )  # resample ratio. subtract original epsilon and add new epsilon
        resample_count += 1

        if np.any(
            np.logical_or(resampled_ratio > thresholds, resampled_ratio < 0),
        ):  # if any values are still out of bounds

            in_bounds_mask = np.logical_and(
                mask_to_resample,
                np.logical_and(resampled_ratio < thresholds, resampled_ratio > 0),
            )

            if np.any(
                in_bounds_mask,
            ):  # if any values are in bounds (i.e. resampling was successful), then update original ratio and epsilon so that we reduce the number of resampled values in the next iteration
                if not initialization:
                    resampled_epsilon.index = resampled_epsilon.index.map(param_dict)
                original_ratio.loc[in_bounds_mask] = resampled_ratio[in_bounds_mask]
                original_epsilon.loc[in_bounds_mask] = resampled_epsilon[
                    in_bounds_mask
                ].values

        else:  # if no values are out of bounds, then set original ratio to resampled ratio, and we are done.
            original_ratio = resampled_ratio.copy(deep=True)
            original_epsilon[mask_to_resample] = resampled_epsilon.values

        if resample_count > 200:
            print(resampled_ratio[in_bounds_mask])
            raise ValueError("Max number of iterations reached")

    return original_ratio, original_epsilon, resample_count


def rng_eps(params, rng):
    return rng.standard_normal(len(params)) * params.sigma


################### Import Parameters ###################
area_mapping_solar = pd.read_csv("parameters/pypsa_area_mapping_solar.csv", index_col=0)
area_mapping_wind = pd.read_csv("parameters/pypsa_area_mapping_wind.csv", index_col=0)
load_parameters = pd.read_csv("parameters/pypsa_load_parameters.csv", index_col=0)
solar_parameters = pd.read_csv("parameters/pypsa_solar_parameters.csv", index_col=0)
wind_parameters = pd.read_csv("parameters/pypsa_wind_parameters.csv", index_col=0)

################### Parameters ###################
num_samples = 10
load_ratio_threshold = 5
solar_ratio_threshold = 5
wind_ratio_threshold = 100000
year = 2016
NETWORK_PATH = "/Users/kamrantehranchi/Local_Documents/pypsa-breakthroughenergy-usa/workflow/notebooks/elec_s_96_offwind_2.nc"
num_clusters = 96

################### Sampling ###################
# %%
load_base, solar_base, wind_base = import_profiles_from_network(
    NETWORK_PATH,
    num_clusters,
)
num_zones = {
    "load": load_base.shape[1],
    "solar": len(area_mapping_solar.area_id.unique()),
    "wind": len(area_mapping_wind.area_id.unique()),
}

Path("sampled_data").mkdir(parents=True, exist_ok=True)
PATH_SAMPLE = Path("sampled_data")
num_hours = load_base.shape[0]

solar_parameters = solar_parameters[
    solar_parameters.area_id.isin(area_mapping_solar.area_id.unique())
]
wind_parameters = wind_parameters[
    wind_parameters.area_id.isin(area_mapping_wind.area_id.unique())
]

# Define Sampling IDs
load_parameters["sampling_id"] = (
    load_parameters.index
    + "_"
    + load_parameters.timescale
    + "_"
    + load_parameters.area_id.astype(str)
)
solar_parameters["sampling_id"] = (
    solar_parameters.index
    + "_"
    + solar_parameters.timescale
    + "_"
    + solar_parameters.area_id.astype(str)
)
wind_parameters["sampling_id"] = (
    wind_parameters.index
    + "_"
    + wind_parameters.timescale
    + "_"
    + wind_parameters.area_id.astype(str)
)
parameters_concat = pd.concat(
    [load_parameters, solar_parameters, wind_parameters],
).reset_index()

# Create a timeseries index dataframe to reference when sampling
timestamp_reference = pd.DataFrame(
    {"timestamp": pd.period_range(year, periods=num_hours, freq="h")},
)
timestamp_reference = timestamp_reference.assign(
    month=timestamp_reference["timestamp"].dt.month,
    day=timestamp_reference["timestamp"].dt.day,
    hour=timestamp_reference["timestamp"].dt.hour,
    day_of_year=timestamp_reference["timestamp"].dt.dayofyear,
)
timestamp_reference["season"] = timestamp_reference["month"].map(
    {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "fall",
        10: "fall",
        11: "fall",
    },
)
timestamp_reference["season_num"] = timestamp_reference["season"].map(
    {"winter": 1, "spring": 2, "summer": 3, "fall": 4},
)

load_base.reset_index(inplace=True, drop=True)
solar_base.reset_index(inplace=True, drop=True)
wind_base.reset_index(inplace=True, drop=True)

# resample load, solar, and wind to daily and area level values.
solar_base_area_daily, solar_base_area = resample_and_group(
    solar_base,
    area_mapping_solar,
)
wind_base_area_daily, wind_base_area = resample_and_group(wind_base, area_mapping_wind)

# %% #Create Generation-Plant Allocations to Map from Area Stochastic Profiles to Plant level Profiles.
solar_allocation = create_allocation(solar_base, area_mapping_solar)
wind_allocation = create_allocation(wind_base, area_mapping_wind)

# %% #Define Solar Hours
daytime_mask = solar_base_area >= 0.0001
daytime_mask = daytime_mask.apply(lambda x: x.astype(int))
df_daytime_solar_tracking = pd.DataFrame(
    np.zeros_like(solar_base_area),
    index=solar_base_area.index,
    columns=solar_base_area.columns,
)
df_daytime_solar_tracking = df_daytime_solar_tracking.apply(
    lambda x: define_solar_hours(daytime_mask[x.name], timestamp_reference)[2],
)

# %%#Define Stochastic Mu Dataframes
cols_to_use = timestamp_reference.columns.difference(load_parameters.columns)
df_mu_load_hourly_unstack = assign_stochastic_mu(
    load_parameters.reset_index(),
    timestamp_reference[cols_to_use],
)
df_mu_solar_hourly_unstack = assign_stochastic_mu(
    solar_parameters.query("timescale == 'hourly'").reset_index(),
    timestamp_reference[cols_to_use],
)
df_mu_solar_daily_unstack = assign_stochastic_mu(
    solar_parameters.query(" timescale == 'daily' ").reset_index(),
    timestamp_reference,
)
df_mu_wind_hourly_unstack = assign_stochastic_mu(
    wind_parameters.query(" timescale == 'daily' ").reset_index(),
    timestamp_reference[cols_to_use],
)
df_mu_wind_daily_unstack = assign_stochastic_mu(
    wind_parameters.query(" timescale == 'daily' ").reset_index(),
    timestamp_reference[cols_to_use],
)

# %% Create Output NP Arrays
load_zones_samples = np.zeros([num_hours, num_zones["load"], num_samples])
solar_zones_samples = np.zeros([num_hours, num_zones["solar"], num_samples])
wind_zones_samples = np.zeros([num_hours, num_zones["wind"], num_samples])
solar_gen_samples = np.zeros([num_hours, solar_base.columns.shape[0], num_samples])
wind_gen_samples = np.zeros([num_hours, wind_base.columns.shape[0], num_samples])
ratio_samples_combined = np.zeros(
    [num_hours, parameters_concat.sampling_id.unique().shape[0], num_samples],
)

# %% Sampling Process
for sample_num in range(0, num_samples):
    print(f"Sample {sample_num}")
    rng = np.random.default_rng(seed=sample_num)
    start_time = time.time()
    resample_count = 0
    n_hours = timestamp_reference.shape[0]
    sampling_profile_names = parameters_concat.sampling_id.unique()
    solar_sampling_id_dict = dict(
        parameters_concat[parameters_concat.sampling_id.str.contains("solar")]
        .reset_index()[["sampling_id", "index"]]
        .values,
    )
    ratio_samples = pd.DataFrame(
        0,
        index=np.arange(n_hours),
        columns=sampling_profile_names,
    )
    epsilon_samples = pd.DataFrame(
        0,
        index=np.arange(n_hours),
        columns=sampling_profile_names,
    )

    thresholds = np.concatenate(
        [
            [load_ratio_threshold] * ratio_samples.columns.str.contains("load").sum(),
            [solar_ratio_threshold] * ratio_samples.columns.str.contains("solar").sum(),
            [wind_ratio_threshold] * ratio_samples.columns.str.contains("wind").sum(),
        ],
    )

    ratio_samples.iloc[0, :], epsilon_samples.iloc[0, :] = sampling_warmup(
        parameters_concat,
        thresholds,
        rng,
    )

    for i in range(1, n_hours):

        season_df = parameters_concat[
            parameters_concat.season == timestamp_reference.iloc[i].season_num
        ]
        p_eps = rng_eps(season_df, rng)
        ratio_samples.iloc[i, :] = (
            ratio_samples.iloc[i - 1, :]
            + season_df.kai.values
            * (season_df.mu.values - ratio_samples.iloc[i - 1, :])
            + p_eps.values
        )
        epsilon_samples.iloc[i, :] = p_eps.values

        if i % 24 != 0:
            ratio_samples.loc[i, ratio_samples.columns.str.contains("daily")] = (
                ratio_samples.loc[i - 1, ratio_samples.columns.str.contains("daily")]
            )
            epsilon_samples.loc[i, epsilon_samples.columns.str.contains("daily")] = (
                epsilon_samples.loc[
                    i - 1,
                    epsilon_samples.columns.str.contains("daily"),
                ]
            )

        previous_sunset_idx = df_daytime_solar_tracking.iloc[i].astype(int)
        if (previous_sunset_idx > 0).any():
            area_sunset_dict = dict(
                zip(
                    [
                        solar_sampling_id_dict.get(param_code)
                        for param_code in ratio_samples.columns
                        if "solar" in param_code
                    ],
                    previous_sunset_idx[previous_sunset_idx > 0].index,
                ),
            )
            previous_mu_vals = ratio_samples.loc[
                :,
                ratio_samples.columns.str.contains("solar"),
            ].iloc[i - 1]
            # previous_mu_vals = ratio_samples.loc[:,ratio_samples.columns.str.contains('solar')].iloc[i - 1].rename(index=solar_sampling_id_dict).loc[area_sunset_dict.keys()]
            solar_season = season_df[season_df.sampling_id.str.contains("solar")]
            solar_mu_vals, solar_kai_vals = solar_season[["mu", "kai"]][
                solar_season.sampling_id.isin(previous_mu_vals.index)
            ].values.T
            eps_filtered = (
                p_eps[solar_season.index]
                .rename(index=season_df.sampling_id.to_dict())
                .loc[previous_mu_vals.index]
                .values
            )
            ratio_samples.loc[i, previous_mu_vals.index] = (
                previous_mu_vals.values
                + solar_kai_vals * (solar_mu_vals - previous_mu_vals.values)
                + eps_filtered
            )

        if (ratio_samples.iloc[i] > thresholds).any() or (
            ratio_samples.iloc[i] < 0
        ).any():
            r0, eps = ratio_samples.iloc[i], epsilon_samples.iloc[i, :]
            ratios_new, eps_new, resample_count = resample_ratio(
                r0,
                season_df,
                thresholds,
                eps,
                rng,
                False,
            )
            ratio_samples.iloc[i], epsilon_samples.iloc[i, :], resample_count = (
                ratios_new,
                eps_new,
                resample_count,
            )
    print(f"Sampling time : {time.time() - start_time}")

    ratio_samples_combined[:, :, sample_num] = ratio_samples.values

    # Load Calculation
    sampled_load_ratio = ratio_samples.filter(like="load")
    sampled_load = (
        load_base
        * sampled_load_ratio.values
        / df_mu_load_hourly_unstack.iloc[:, 1:].values
    )
    sampled_load["timestamp"] = timestamp_reference.timestamp
    sampled_load.set_index("timestamp", inplace=True)
    sampled_load.columns = load_base.columns
    load_zones_samples[:, :, sample_num] = sampled_load.values

    # Solar Calculation
    sampled_hourly_solar_ratio = ratio_samples.filter(regex="solar.*hourly")
    sampled_daily_solar_ratio = ratio_samples.filter(regex="solar.*daily")
    sampled_solar = (
        solar_base_area.values
        * (
            0.333 * sampled_hourly_solar_ratio.values
            + 0.667 * sampled_daily_solar_ratio.values
        )
        / (
            df_mu_solar_hourly_unstack.iloc[:, 1:] * 0.333
            + df_mu_solar_daily_unstack.iloc[:, 1:] * 0.667
        )
    )
    sampled_solar[sampled_solar > 1] = 1  # capacity factor cannot be greater than 1

    sampled_solar_generatorlvl = (
        np.ones_like(solar_base.values) * solar_allocation.values
    )
    sampled_solar_generatorlvl = pd.DataFrame(
        sampled_solar_generatorlvl,
        index=sampled_solar.index,
        columns=solar_allocation.columns,
    )
    sampled_solar_generatorlvl = sampled_solar_generatorlvl.apply(
        lambda x: x * sampled_solar.loc[:, str(x.name)],
        axis=0,
    )

    solar_zones_samples[:, :, sample_num] = sampled_solar.values
    solar_gen_samples[:, :, sample_num] = sampled_solar_generatorlvl.values

    # Wind Calculation
    sampled_hourly_wind_ratio = ratio_samples.filter(regex="wind.*hourly")
    sampled_daily_wind_ratio = ratio_samples.filter(regex="wind.*daily")
    sampled_wind = (
        wind_base_area.values
        * (
            0.333 * sampled_hourly_wind_ratio.values
            + 0.667 * sampled_daily_wind_ratio.values
        )
        / (
            df_mu_wind_hourly_unstack.iloc[:, 1:] * 0.333
            + df_mu_wind_daily_unstack.iloc[:, 1:] * 0.667
        )
    )
    sampled_wind[sampled_wind > 1] = 1  # capacity factor cannot be greater than 1

    sampled_wind_generatorlvl = np.ones_like(wind_base.values) * wind_allocation.values
    sampled_wind_generatorlvl = pd.DataFrame(
        sampled_wind_generatorlvl,
        index=sampled_wind.index,
        columns=wind_allocation.columns,
    )
    sampled_wind_generatorlvl = sampled_wind_generatorlvl.apply(
        lambda x: x * sampled_wind.loc[:, str(x.name)],
        axis=0,
    )

    wind_zones_samples[:, :, sample_num] = sampled_wind.values
    wind_gen_samples[:, :, sample_num] = sampled_wind_generatorlvl.values

# %% Combining into Xarray Dataset
# Match headers across datasets
solar_cols = area_mapping_solar.drop("area_id", axis=1)
wind_cols = area_mapping_wind.drop("area_id", axis=1)
# remove words solar/wind/offwind from values in index to match column headers from load data. Used to match buses with eachother
solar_cols.index = solar_cols.index.str.replace(" solar", "")
wind_cols.index = wind_cols.index.str.replace(" wind", "")
wind_cols.index = wind_cols.index.str.replace(" offwind", "")

solar_cols["np_ind_solar"] = np.arange(0, len(solar_cols))
wind_cols["np_ind_wind"] = np.arange(0, len(wind_cols))
df_ind_match = pd.DataFrame(load_base.columns, columns=["load"])
df_ind_match["np_ind_load"] = np.arange(0, len(load_base.columns))
df_ind_match = df_ind_match.merge(
    solar_cols,
    left_on="load",
    right_index=True,
    how="outer",
).merge(wind_cols, left_on="load", right_index=True, how="outer")
df_ind_match.index = np.arange(0, len(df_ind_match))
df_bus_coordinates = df_ind_match["load"]
df_ind_match = df_ind_match.drop(columns=["load"])
df_ind_match_combinedarr = df_ind_match.where(
    df_ind_match.isnull(),
    np.column_stack([df_ind_match.index.values] * 3),
)

# Create index references for each dataset
solar_inds_combined = df_ind_match_combinedarr.np_ind_solar.values[
    ~np.isnan(df_ind_match_combinedarr.np_ind_solar.values)
].astype(int)
wind_inds_combined = df_ind_match_combinedarr.np_ind_wind.values[
    ~np.isnan(df_ind_match_combinedarr.np_ind_wind.values)
].astype(int)
load_inds_combined = df_ind_match_combinedarr.np_ind_load.values[
    ~np.isnan(df_ind_match_combinedarr.np_ind_load.values)
].astype(int)

wind_inds_orig = df_ind_match.np_ind_wind.dropna().astype(int).values
solar_inds_orig = df_ind_match.np_ind_solar.dropna().astype(int).values
load_inds_orig = df_ind_match.np_ind_load.dropna().astype(int).values

# Combine load solar wind data into one array for each sample
combined_gen_samples = np.zeros(
    [8784, 97, 3, num_samples],
)  # Dimensions are (t, node, profile type, samples)
combined_gen_samples[:, load_inds_combined, 0, :] = load_zones_samples[
    :,
    load_inds_orig,
    :,
]
combined_gen_samples[:, solar_inds_combined, 1, :] = solar_gen_samples[
    :,
    solar_inds_orig,
    :,
]
combined_gen_samples[:, wind_inds_combined, 2, :] = wind_gen_samples[
    :,
    wind_inds_orig,
    :,
]

# Create xarray dataset
da_bus_data = xr.DataArray(
    combined_gen_samples,
    coords={
        "timestamp": timestamp_reference.timestamp.dt.to_timestamp(),
        "bus": df_bus_coordinates.values,
        "profile_type": ["load", "solar", "wind"],
        "sample_num": np.arange(0, num_samples),
    },
    dims=["timestamp", "bus", "profile_type", "sample_num"],
)

################Combining zonal data into one array for each profile type #############################
# remove numeric from wind and solar columns
base_zones_solar = pd.DataFrame(
    solar_cols.index.str.replace("[0-9]", "").drop_duplicates(),
    columns=["solar"],
)
base_zones_wind = pd.DataFrame(
    wind_cols.index.str.replace("[0-9]", "").drop_duplicates(),
    columns=["wind"],
)

base_zones_wind = base_zones_wind[
    ~base_zones_wind.wind.str.contains("SDGE")
].reset_index(
    drop=True,
)  # temporary fix
df_zone_ind_match = base_zones_solar.reset_index(names="solar_ind").merge(
    base_zones_wind.reset_index(names="wind_ind"),
    left_on="solar",
    right_on="wind",
    how="outer",
)
# combine non nan values into new column
zone_coords = df_zone_ind_match.solar.combine_first(df_zone_ind_match.wind)
solar_zone_inds_combined = df_zone_ind_match.solar_ind.values[
    ~np.isnan(df_zone_ind_match.solar_ind.values)
].astype(int)
wind_zone_inds_combined = df_zone_ind_match.wind_ind.values[
    ~np.isnan(df_zone_ind_match.wind_ind.values)
].astype(int)
wind_zone_inds_orig = df_zone_ind_match.wind_ind.dropna().astype(int).values
solar_zone_inds_orig = df_zone_ind_match.solar_ind.dropna().astype(int).values

combined_zone_data = np.zeros(
    [8784, 22, 2, num_samples],
)  # Dimensions are (t, node, profile type, samples)
combined_zone_data[:, solar_zone_inds_combined, 0, :] = solar_zones_samples[
    :,
    solar_zone_inds_orig,
    :,
]
combined_zone_data[:, wind_zone_inds_combined, 1, :] = wind_zones_samples[
    :,
    wind_zone_inds_orig,
    :,
]

da_zone_data = xr.DataArray(
    combined_zone_data,
    coords={
        "timestamp": timestamp_reference.timestamp.dt.to_timestamp(),
        "zone": zone_coords.values,
        "profile_type": ["solar", "wind"],
        "sample_num": np.arange(0, num_samples),
    },
    dims=["timestamp", "zone", "profile_type", "sample_num"],
)

da_ratios = xr.DataArray(
    ratio_samples_combined,
    coords={
        "timestamp": timestamp_reference.timestamp.dt.to_timestamp(),
        "ratio_sample": parameters_concat.sampling_id.unique(),
        "sample_num": np.arange(0, num_samples),
    },
    dims=["timestamp", "ratio_sample", "sample_num"],
)

# # Saving files
da_zone_data.to_netcdf(os.path.join(os.getcwd(), PATH_SAMPLE, "sampled_zone_data.nc"))
da_bus_data.to_netcdf(os.path.join(os.getcwd(), PATH_SAMPLE, "sampled_bus_data.nc"))
da_ratios.to_netcdf(os.path.join(os.getcwd(), PATH_SAMPLE, "sampled_ratio_data.nc"))
np.save(
    os.path.join(os.getcwd(), PATH_SAMPLE, "combined_gen_samples.npy"),
    combined_gen_samples,
)
np.save(
    os.path.join(os.getcwd(), PATH_SAMPLE, "combined_ratio_samples.npy"),
    ratio_samples_combined,
)
