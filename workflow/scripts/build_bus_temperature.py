"""
Build the hourly air temperature of each clustered bus, the area-weighted mean over its region.

Input of the temperature-dependent derate of thermal units (``conventional:
ambient_derate``, see ``ambient_derate.py``). Every cutout grid cell counts with
the share of its area that lies in the bus region; unlike
``build_temperature_profiles`` (population-weighted, for heating demand) this
needs no population layout.
"""

import logging

import geopandas as gpd
import numpy as np
from _helpers import configure_logging, get_snapshots, stamp_weather_calendar
from build_temperature_profiles import load_cutout
from scipy.sparse import csr_matrix, diags

logger = logging.getLogger(__name__)


def area_weights(indicator) -> csr_matrix:
    """
    Row-normalise an atlite indicator matrix (regions x cells) into area weights.

    ``indicator[i, j]`` is the share of grid cell ``j`` lying in region ``i``; each
    row then sums to one, so aggregating with it gives the area-weighted mean.
    """
    indicator = csr_matrix(indicator)
    total = np.asarray(indicator.sum(axis=1)).ravel()
    if (total <= 0).any():
        raise ValueError(f"{(total <= 0).sum()} bus region(s) do not overlap the weather cutout.")
    return csr_matrix(diags(1.0 / total) @ indicator)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_bus_temperature",
            interconnect="western",
            simpl="70",
            clusters="4m",
        )
    configure_logging(snakemake)

    cutout = load_cutout(snakemake.input.cutout, time=get_snapshots(snakemake.params.snapshots))
    regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name").buffer(0)

    temperature = cutout.temperature(
        matrix=area_weights(cutout.indicatormatrix(regions)),
        index=regions.index,
        show_progress=False,
    )
    # ambient_derate rejects a temperature profile that does not say which rule
    # laid the weather year onto the snapshots
    stamp_weather_calendar(temperature).to_netcdf(snakemake.output.temperature)
