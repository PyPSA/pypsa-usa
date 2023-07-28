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