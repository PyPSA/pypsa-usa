# BY PyPSA-USA Authors
"""
Calculates for each network substation the installable capacity (based on land-
use) and the available generation time series (based on weather data).
"""

import functools
import logging
import time

import atlite
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from _helpers import configure_logging, get_snapshots
from dask.distributed import Client
from pypsa.geo import haversine
from shapely.geometry import LineString
from typing import List, Tuple, Dict, Union, Set
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_data(data):
    x = data.coords["x"].values  # Longitude
    y = data.coords["y"].values  # Latitude
    values = data.values

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(x, y, values, shading="auto", cmap="viridis")
    fig.colorbar(
        im,
        ax=ax,
        label="Value",
    )  # Add a colorbar to represent the value scale

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return fig, ax

# functions for WUS capacity factors
def get_buses(profile):
    buses = [int(x) for x in profile['bus'].values]
    return buses

def load_WUS_data(planning_horizon: int, base_path: str = "T:/WRFDownscaled/ec-earth3_r1i1p1f1_ssp370_bc/Annual_Solar_Wind") -> xr.Dataset:
    #Loads WUS data for given planning horizon#
    logger.info(f"Loading WUS data for planning horizon {planning_horizon}...")
    file_path = Path(base_path) / f"Solar_Wind_CFs_{planning_horizon}.nc"
    
    if not file_path.exists():
        raise FileNotFoundError(f"WUS data file not found at: {file_path}")
        
    return xr.open_dataset(file_path)

def create_blank_profile(pypsa_profile: xr.Dataset) -> xr.DataArray:
    #Create a blank xarray DataArray with the same structure as the input profile.
    cf = pypsa_profile['profile']
    return xr.DataArray(
        data=np.full_like(cf, fill_value=np.nan),
        coords=cf.coords,
        dims=cf.dims,
        name='empty_profile'
    )

def find_closest(lat: float, long: float, wus_latitude: np.ndarray, wus_longitude: np.ndarray) -> List[float]:
    #For given pypsa bus coordinate, find the closest latitude and longitude point in the WUS dataset.
    lat_idx = np.abs(wus_latitude - lat).argmin()
    long_idx = np.abs(wus_longitude - long).argmin()
    return [wus_latitude[lat_idx], wus_longitude[long_idx]]

def get_WUS_snapshot_subset(wus_cfs: xr.Dataset, snapshots: pd.DatetimeIndex) -> pd.DatetimeIndex:
    #Extract a subset of WUS timestamps that match the month-day-time pattern in pypsa snapshots.
    ss = pd.to_datetime(snapshots)
    wus = pd.to_datetime(wus_cfs['Times'].values)
    mdt_set = set(zip(ss.month, ss.day, ss.time))
    subset = wus[[(m, d, t) in mdt_set for m, d, t in zip(wus.month, wus.day, wus.time)]]
    return subset

def capitalize(s):
    return s[0].upper() + s[1:]

def return_wus_col(tech):
    if tech == "solar":
        return "Solar"
    if tech == "onwind":
        return "Wind"

def generate_wus_profile(
    pypsa_profile: xr.Dataset,
    wus_cfs: xr.Dataset,
    buses, # not sure what type this is
    bus_loc_data: pd.DataFrame,
    technology: str,
    snapshots: pd.DatetimeIndex
) -> xr.Dataset:
    """Generate WUS profiles for all buses in a PyPSA profile.
    
    Args:
        pypsa_profile: PyPSA profile dataset
        wus_cfs: WUS capacity factors dataset
        bus_loc_data: DataFrame with bus coordinates
        technology: Technology type (e.g., 'wind', 'solar')
        snapshots: DatetimeIndex of timestamps
        
    Returns:
        Updated profile with WUS capacity factors
    """
    logger.info("Extracting WUS snapshots...")
    wus_snapshots = get_WUS_snapshot_subset(wus_cfs, snapshots)
    wus_subset = wus_cfs.sel(Times=wus_snapshots)

    #logger.info("Determining buses in profile and extracting coordinates...")
    #profile_buses = get_buses(pypsa_profile)
    # Filter bus location data for relevant buses and remove duplicates
    #busLocProfile = bus_loc_data[bus_loc_data.index.isin(profile_buses)]
    #busLocProfile = busLocProfile[~busLocProfile.index.duplicated(keep='first')]
    
    logger.info("Creating blank profile...")
    new_profile = create_blank_profile(pypsa_profile)

    logger.info("Preparing WUS location data...")
    wus_latitude = wus_subset['lat'].values
    wus_longitude = wus_subset['lon'].values
    col = f"{return_wus_col(technology)}_CF"
    
    # Pre-compute set of available coordinates for faster lookups
    available_lats = set(wus_latitude)
    available_longs = set(wus_longitude)

    logger.info(f"Iterating through {len(new_profile['bus'].values)} buses and assigning profiles...")
    
    # Process buses in chunks for better performance
    for i, b in enumerate(new_profile['bus'].values):
        if b not in bus_loc_data.index:
            logger.warning(f"Bus {b} not found in location data, skipping")
            continue
        
        long, lat = bus_coords.loc[b]

        # Check if coordinates exist in WUS data, if not find closest
        if lat in available_lats and long in available_longs:
            closest = [lat, long]
        else:
            closest = find_closest(lat, long, wus_latitude, wus_longitude)

        # Assign bus profile from WUS at nearest location
        wus_data = wus_subset[col].sel(lon=closest[1], lat=closest[0], method='nearest')
        
        # Verify shape compatibility
        bus_profile = new_profile.loc[dict(bus=str(b))]
        if wus_data.shape != bus_profile.shape:
            logger.error(f"Shape mismatch for bus {b}: WUS data {wus_data.shape} vs profile {bus_profile.shape}")
            continue
            
        new_profile.loc[dict(bus=str(b))] = wus_data.values

    logger.info("Finalizing WUS profile...")
    wus_profile = pypsa_profile.copy()
    wus_profile['profile'] = new_profile
    
    return wus_profile

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_renewable_profiles",
            technology="solar",
            interconnect="texas",
        )
    configure_logging(snakemake)

    nprocesses = int(snakemake.threads)
    noprogress = snakemake.config["run"].get("disable_progressbar", True)
    noprogress = noprogress or not snakemake.config["atlite"]["show_progress"]
    params = snakemake.params.renewable[snakemake.wildcards.technology]
    resource = params["resource"]  # pv panel params / wind turbine params
    correction_factor = params.get("correction_factor", 1.0)
    capacity_per_sqkm = params["capacity_per_sqkm"]
    p_nom_max_meth = params.get("potential", "conservative")

    if isinstance(params.get("corine", {}), list):
        params["corine"] = {"grid_codes": params["corine"]}

    if correction_factor != 1.0:
        logger.info(f"correction_factor is set as {correction_factor}")

    if nprocesses > 1:
        client = Client(n_workers=nprocesses, threads_per_worker=1)
    else:
        client = None

    sns = get_snapshots(snakemake.params.snapshots[snakemake.wildcards.renewable_weather_years])
    logger.info(f'using cutout "{snakemake.input.cutout}"')
    cutout = atlite.Cutout(snakemake.input.cutout[0]).sel(
        time=sns,
    )  # Patch fix with [0] move expand in snakemake rule to add_elec for multiple cutouts

    regions = gpd.read_file(snakemake.input.regions)

    assert "x" in regions.columns and "y" in regions.columns, (
        f"List of regions in {snakemake.input.regions} is empty, please disable the corresponding renewable technology"
    )
    # do not pull up, set_index does not work if geo dataframe is empty
    regions = regions.set_index("name").rename_axis("bus")
    buses = regions.index

    res = params.get("excluder_resolution", 100)
    excluder = atlite.ExclusionContainer(crs=5070, res=res)

    if params["natura"]:
        excluder.add_raster(
            snakemake.input.natura,
            nodata=0,
            allow_no_overlap=True,
        )

    corine = params.get("corine", {})
    if "grid_codes" in corine:
        codes = corine["grid_codes"]
        excluder.add_raster(
            snakemake.input.corine,
            codes=codes,
            invert=True,
            # crs=4326
        )
    if corine.get("distance", 0.0) > 0.0:
        codes = corine["distance_grid_codes"]
        buffer = corine["distance"]
        excluder.add_raster(
            snakemake.input.corine,
            codes=codes,
            buffer=buffer,
            # crs=4326,
        )

    if params.get("cec", 0):
        excluder.add_raster(
            snakemake.input[f"cec_{snakemake.wildcards.technology}"],
            nodata=0,
            allow_no_overlap=True,
        )

    if params.get("boem_screen", 0):
        excluder.add_raster(
            snakemake.input["boem_osw"],
            invert=True,
            nodata=0,
            allow_no_overlap=True,
        )

    if params.get("max_depth"):
        # lambda not supported for atlite + multiprocessing
        # use named function np.greater with partially frozen argument instead
        # and exclude areas where: -max_depth > grid cell depth
        func = functools.partial(np.greater, -params["max_depth"])
        excluder.add_raster(
            snakemake.input.gebco,
            codes=func,
            nodata=-1000,
            # crs=4326,
        )

    if params.get("min_depth"):
        # lambda not supported for atlite + multiprocessing
        # use named function np.greater with partially frozen argument instead
        # and exclude areas where: -min_depth < grid cell depth
        func = functools.partial(np.less, -params["min_depth"])
        excluder.add_raster(
            snakemake.input.gebco,
            codes=func,
            nodata=-1000,
            # crs=4326,
        )

    if "min_shore_distance" in params:
        buffer = params["min_shore_distance"]
        excluder.add_geometry(snakemake.input.country_shapes, buffer=buffer)

    if "max_shore_distance" in params:
        buffer = params["max_shore_distance"]
        excluder.add_geometry(
            snakemake.input.country_shapes,
            buffer=buffer,
            invert=True,
        )

    logger.info("Calculate landuse availability...")
    start = time.time()

    kwargs = dict(nprocesses=nprocesses, disable_progressbar=noprogress)
    availability = cutout.availabilitymatrix(regions, excluder, **kwargs)

    duration = time.time() - start
    logger.info(f"Completed landuse availability calculation ({duration:2.2f}s)")

    # fig, ax = plt.subplots()
    # excluder.plot_shape_availability(regions, ax=ax)
    fig, ax = plot_data(availability.sum("bus"))
    ax.set_title(f"Availability of {snakemake.wildcards.technology} Technology")
    plt.savefig(snakemake.output.availability)
    plt.close(fig)

    area = cutout.grid.to_crs("EPSG: 5070").area / 1e6
    area = xr.DataArray(
        area.values.reshape(cutout.shape),
        [cutout.coords["y"], cutout.coords["x"]],
    )

    potential = capacity_per_sqkm * availability.sum("bus") * area
    func = getattr(cutout, resource.pop("method"))
    if client is not None:
        resource["dask_kwargs"] = {"scheduler": client}
    capacity_factor = correction_factor * func(capacity_factor=True, **resource)
    layout = capacity_factor * area * capacity_per_sqkm
    profile, capacities = func(
        matrix=availability.stack(spatial=["y", "x"]),
        layout=layout,
        index=buses,
        per_unit=True,
        return_capacity=True,
        **resource,
    )

    logger.info(f"Calculating maximal capacity per bus (method '{p_nom_max_meth}')")
    if p_nom_max_meth == "simple":
        p_nom_max = capacity_per_sqkm * availability @ area
    elif p_nom_max_meth == "conservative":
        max_cap_factor = capacity_factor.where(availability != 0).max(["x", "y"])
        p_nom_max = capacities / max_cap_factor
    else:
        raise AssertionError(
            f'Config key `potential` should be one of "simple" (default) or "conservative", not "{p_nom_max_meth}"',
        )

    logger.info("Calculate average distances.")
    layoutmatrix = (layout * availability).stack(spatial=["y", "x"])

    coords = cutout.grid[["x", "y"]]
    bus_coords = regions[["x", "y"]]

    average_distance = []
    centre_of_mass = []
    for bus in buses:
        row = layoutmatrix.sel(bus=bus).data
        nz_b = row != 0
        row = row[nz_b]
        co = coords[nz_b]
        distances = haversine(bus_coords.loc[bus], co)
        average_distance.append((distances * (row / row.sum())).sum())
        centre_of_mass.append(co.values.T @ (row / row.sum()))

    average_distance = xr.DataArray(average_distance, [buses])
    centre_of_mass = xr.DataArray(centre_of_mass, [buses, ("spatial", ["x", "y"])])

    ds = xr.merge(
        [
            (correction_factor * profile).rename("profile"),
            capacities.rename("weight"),
            p_nom_max.rename("p_nom_max"),
            potential.rename("potential"),
            average_distance.rename("average_distance"),
        ],
    )
    if snakemake.wildcards.technology.startswith("offwind"):
        logger.info("Calculate underwater fraction of connections.")
        offshore_shape = gpd.read_file(snakemake.input["offshore_shapes"]).unary_union
        underwater_fraction = []
        for bus in buses:
            p = centre_of_mass.sel(bus=bus).data
            line = LineString([p, regions.loc[bus, ["x", "y"]]])
            frac = line.intersection(offshore_shape).length / line.length
            underwater_fraction.append(frac)

        ds["underwater_fraction"] = xr.DataArray(underwater_fraction, [buses])

    # select only buses with some capacity and minimal capacity factor
    ds = ds.sel(
        bus=(
            (ds["profile"].mean("time") > params.get("min_p_max_pu", 0.0))
            & (ds["p_nom_max"] > params.get("min_p_nom_max", 0.0))
        ),
    )

    if "clip_p_max_pu" in params:
        min_p_max_pu = params["clip_p_max_pu"]
        ds["profile"] = ds["profile"].where(ds["profile"] >= min_p_max_pu, 0)

    # if cf_source is WUS, substitute in WUS capacity factors
    if snakemake.config["cf_source"] == "WUS":
        logger.info("CF source is set to WUS, beginning capacity factor substitution...")

        # get index of renewable weather year, then get that planning horizon
        index = snakemake.config['renewable_weather_years'].index(int(snakemake.wildcards.renewable_weather_years))
        wus_year = snakemake.config['scenario']['planning_horizons'][index]

        wus_data = load_WUS_data(wus_year)
        wus_profile = generate_wus_profile(ds,wus_data,buses,bus_coords,snakemake.wildcards.technology,sns)

        logger.info("Capacity factor substitution complete.")
        wus_profile.to_netcdf(snakemake.output.profile)

    # otherwise, leave renewable profile untouched
    else:
        logger.info("CF source is set to ERA5, no changes have been made to capacity factors.")
        ds.to_netcdf(snakemake.output.profile)
    if client is not None:
        client.shutdown()
