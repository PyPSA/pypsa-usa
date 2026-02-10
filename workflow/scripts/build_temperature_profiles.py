"""Build time series for air and soil temperatures per clustered model region."""

from tempfile import NamedTemporaryFile

import atlite
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client, LocalCluster


def load_cutout(cutout_files: str | list[str], time: None | pd.DatetimeIndex = None) -> atlite.Cutout:
    """
    Load and optionally combine multiple cutout files.

    Parameters
    ----------
    cutout_files : str or list of str
        Path to a single cutout file or a list of paths to multiple cutout files.
        If a list is provided, the cutouts will be concatenated along the time dimension.
    time : pd.DatetimeIndex, optional
        If provided, select only the specified times from the cutout.

    Returns
    -------
    atlite.Cutout
        Merged cutout with optional time selection applied.
    """
    if isinstance(cutout_files, str):
        cutout = atlite.Cutout(cutout_files)
    elif isinstance(cutout_files, list):
        cutout_da = [atlite.Cutout(c).data for c in cutout_files]
        combined_data = xr.concat(cutout_da, dim="time", data_vars="minimal")
        cutout = atlite.Cutout(NamedTemporaryFile().name, data=combined_data)

    if time is not None:
        cutout.data = cutout.data.sel(time=time)

    return cutout


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_temperature_profiles",
            interconnect="eastern",
            simpl=10,
            clusters="1m",
            scope="total",
        )

    nprocesses = int(snakemake.threads)
    cluster = LocalCluster(n_workers=nprocesses, threads_per_worker=1)
    client = Client(cluster, asynchronous=True)

    time = pd.date_range(freq="h", **snakemake.params.snapshots)

    cutout = load_cutout(snakemake.input.cutout, time=time)

    clustered_regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name").buffer(0)

    indicator_matrix = cutout.indicatormatrix(clustered_regions)

    pop_layout = xr.open_dataarray(snakemake.input.pop_layout)

    stacked_pop = pop_layout.stack(spatial=("y", "x"))
    M = indicator_matrix.T.dot(np.diag(indicator_matrix.dot(stacked_pop)))

    nonzero_sum = M.sum(axis=0, keepdims=True)
    nonzero_sum[nonzero_sum == 0.0] = 1.0
    M_tilde = M / nonzero_sum

    temp_air = cutout.temperature(
        matrix=M_tilde.T,
        index=clustered_regions.index,
        dask_kwargs=dict(scheduler=client),
        show_progress=False,
    )

    temp_air.to_netcdf(snakemake.output.temp_air)

    temp_soil = cutout.soil_temperature(
        matrix=M_tilde.T,
        index=clustered_regions.index,
        dask_kwargs=dict(scheduler=client),
        show_progress=False,
    )

    temp_soil.to_netcdf(snakemake.output.temp_soil)
