"""Build time series for air and soil temperatures per clustered model region."""

import atlite
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client, LocalCluster

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_temperature_profiles",
            interconnect="western",
            # simpl="",
            clusters=60,
            scope="total",
        )

    nprocesses = int(snakemake.threads)
    cluster = LocalCluster(n_workers=nprocesses, threads_per_worker=1)
    client = Client(cluster, asynchronous=True)

    time = pd.date_range(freq="h", **snakemake.params.snapshots)

    assert len(snakemake.input.cutout) == 1
    cutout = atlite.Cutout(snakemake.input.cutout[0]).sel(time=time)

    clustered_regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name").buffer(0).squeeze()

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
