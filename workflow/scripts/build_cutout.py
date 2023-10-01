
import logging

import atlite
import geopandas as gpd
import pandas as pd
from _helpers import configure_logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("build_cutout", cutout="era5-2019", interconnect="western")
    configure_logging(snakemake)

    cutout_params = snakemake.params.cutouts[snakemake.wildcards.cutout]

    snapshots = pd.date_range(freq="h", **snakemake.params.snapshots)
    time = [snapshots[0], snapshots[-1]]
    cutout_params["time"] = slice(*cutout_params.get("time", time))

    if {"x", "y", "bounds"}.isdisjoint(cutout_params):
        # Determine the bounds from bus regions with a buffer of two grid cells
        onshore = gpd.read_file(snakemake.input.regions_onshore)
        offshore = gpd.read_file(snakemake.input.regions_offshore)
        regions = pd.concat([onshore, offshore])
        d = max(cutout_params.get("dx", 0.25), cutout_params.get("dy", 0.25)) * 2
        cutout_params["bounds"] = regions.total_bounds + [-d, -d, d, d]
    elif {"x", "y"}.issubset(cutout_params):
        cutout_params["x"] = slice(*cutout_params["x"])
        cutout_params["y"] = slice(*cutout_params["y"])

    logging.info(f"Preparing cutout with parameters {cutout_params}.")
    features = cutout_params.pop("features", None)
    cutout = atlite.Cutout(snakemake.output[0], **cutout_params)
    cutout.prepare(features=features)


"""
import atlite
import logging
import pandas as pd
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

# x = [-126, -99] wecc
# y = [27, 50] wecc

x = [-110,-90] ercot
y = [24, 37] ercot

x = [-109 , -65] eastern
y = [23, 50] eastern

x = [-126, -65] #us
y = [23, 50] #us

cutout_params = snakemake.config["atlite"]["cutouts"]['era5']
snapshots = pd.date_range(freq="h",start ="2019-01-01", end= "2020-01-01", inclusive = 'left')
atlite_time = [snapshots[0], snapshots[-1]]

cutout = atlite.Cutout(
    path="western-usa-2019.nc",
    module="era5",
    x=slice(x[0], x[1]),
    y=slice(49.9096, 60.8479),
    time=atlite_time,
)

    cutout = atlite.Cutout(snakemake.output[0], **cutout_params)
    cutout.prepare(features=features)
"""