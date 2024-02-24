"""
Create cutouts with `atlite <https://atlite.readthedocs.io/en/latest/>`_.

For this rule to work you must have

- installed the `Copernicus Climate Data Store <https://cds.climate.copernicus.eu>`_ ``cdsapi`` package  (`install with `pip``) and
- registered and setup your CDS API key as described `on their website <https://cds.climate.copernicus.eu/api-how-to>`_.

.. seealso::
    For details on the weather data read the `atlite documentation <https://atlite.readthedocs.io/en/latest/>`_.
    If you need help specifically for creating cutouts `the corresponding section in the atlite documentation <https://atlite.readthedocs.io/en/latest/examples/create_cutout.html>`_ should be helpful.

**Relevant Settings**

.. code:: yaml

    atlite:
        nprocesses:
        cutouts:
            {cutout}:

.. seealso::
    Documentation of the configuration file ``config/config.yaml`` at
    :ref:`atlite_cf`

**Inputs**

None

**Outputs**

- ``cutouts/{cutout}``: weather data from the `ERA5 <https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5>`_
  reanalysis weather dataset satellite-based historic weather data with the following structure:

**Description**

Using the ERA5 cutout, the following parameters are accessible:

    ===================  ==========  ==========  =========================================================
    Field                Dimensions  Unit        Description
    ===================  ==========  ==========  =========================================================
    height               y, x        m           Surface elevation above sea level
    -------------------  ----------  ----------  ---------------------------------------------------------
    wnd100m              time, y, x  ms**-1      Wind speeds at 100 meters (regardless of direction)
    -------------------  ----------  ----------  ---------------------------------------------------------
    wnd_azimuth          time, y, x  ms**-1      100 metre U wind component
    -------------------  ----------  ----------  ---------------------------------------------------------
    roughness            y, x        m           Forecast surface roughness
                                                 (`roughness length <https://en.wikipedia.org/wiki/Roughness_length>`_)
    -------------------  ----------  ----------  ---------------------------------------------------------
    influx_toa           time, y, x  Wm**-2      Top of Earth's atmosphere TOA incident solar radiation
    -------------------  ----------  ----------  ---------------------------------------------------------
    influx_direct        time, y, x  Wm**-2      Total sky direct solar radiation at surface
    -------------------  ----------  ----------  ---------------------------------------------------------
    influx_diffuse       time, y, x  Wm**-2      Diffuse solar radiation at surface.
                                                 Surface solar radiation downwards minus
                                                 direct solar radiation.
    -------------------  ----------  ----------  ---------------------------------------------------------
    solar_altitude       time, y, x  rad
    -------------------  ----------  ----------  ---------------------------------------------------------
    solar_azimuth        time, y, x  rad
    -------------------  ----------  ----------  ---------------------------------------------------------
    temperature          time, y, x  K           Air temperature 2 meters above the surface.
    -------------------  ----------  ----------  ---------------------------------------------------------
    soil temperature     time, y, x  K           Soil temperature between 1 meters and 3 meters
                                                 depth (layer 4).
    -------------------  ----------  ----------  ---------------------------------------------------------
    influx_toa           time, y, x  Wm**-2      Top of Earth's atmosphere TOA incident solar radiation
    -------------------  ----------  ----------  ---------------------------------------------------------
    influx_direct        time, y, x  Wm**-2      Total sky direct solar radiation at surface
    -------------------  ----------  ----------  ---------------------------------------------------------
    runoff               time, y, x  m           `Runoff <https://en.wikipedia.org/wiki/Surface_runoff>`_
                                                 (volume per area)
    -------------------  ----------  ----------  ---------------------------------------------------------
    albedo               time, y, x  --          `Albedo <https://en.wikipedia.org/wiki/Albedo>`_
                                                 measure of diffuse reflection of solar radiation.
                                                 Calculated from relation between surface solar radiation
                                                 downwards (Jm**-2) and surface net solar radiation
                                                 (Jm**-2). Takes values between 0 and 1.
    ===================  ==========  ==========  =========================================================

The **USA Interconnect** weather data is shown below:

    .. image:: _static/cutouts/weather.png
        :scale: 80 %
"""

import logging

import atlite
import geopandas as gpd
import pandas as pd
from _helpers import configure_logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_cutout",
            cutout="era5_2019",
            interconnect="eastern",
        )
    configure_logging(snakemake)

    # data set and temporal patameters
    cutout_params = snakemake.params.cutouts[snakemake.wildcards.cutout]
    snapshots = pd.date_range(freq="h", **snakemake.params.snapshots)
    time = [snapshots[0], snapshots[-1]]
    cutout_params["time"] = slice(*cutout_params.get("time", time))

    # geographical extent parameters
    interconnect_params = snakemake.params.interconnects[
        snakemake.wildcards.interconnect
    ]

    if {"x", "y", "bounds"}.isdisjoint(interconnect_params):
        # Determine the bounds from bus regions with a buffer of two grid cells
        onshore = gpd.read_file(snakemake.input.regions_onshore)
        offshore = gpd.read_file(snakemake.input.regions_offshore)
        regions = pd.concat([onshore, offshore])
        d = (
            max(
                interconnect_params.get("dx", 0.25),
                interconnect_params.get("dy", 0.25),
            )
            * 2
        )
        interconnect_params["bounds"] = regions.total_bounds + [-d, -d, d, d]
    elif {"x", "y"}.issubset(interconnect_params):
        interconnect_params["x"] = slice(*interconnect_params["x"])
        interconnect_params["y"] = slice(*interconnect_params["y"])

    cutout_params.update(interconnect_params)

    logging.info(f"Preparing cutout with parameters {cutout_params}.")
    features = cutout_params.pop("features", None)
    cutout = atlite.Cutout(snakemake.output[0], **cutout_params)
    cutout.prepare(features=features)
