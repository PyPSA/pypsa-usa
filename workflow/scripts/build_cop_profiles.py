"""
Build coefficient of performance (COP) time series for air- or ground-sourced
heat pumps.

The COP is a function of the temperature difference between source and
sink.

The quadratic regression used is based on Staffell et al. (2012)
https://doi.org/10.1039/C2EE22653G.

COP ASHP = 6.81 - 0.121(Tsink - Tsource) + 0.000630(Tsink - Tsource)^2 for 15 <= DT <= 60
COP GSHP = 8.77 - 0.150(Tsink - Tsource) + 0.000734(Tsink - Tsource)^2 for 20 <= DT <= 60

A Tsink of 50-60C is recommended for how water temperatures
"""

import xarray as xr
from _helpers import WEATHER_CALENDAR_ATTR


def coefficient_of_performance(
    delta_t: xr.DataArray,
    source: str = "air",
) -> xr.DataArray:
    if source == "air":
        return 6.81 - 0.121 * delta_t + 0.000630 * delta_t**2
    elif source == "soil":
        return 8.77 - 0.150 * delta_t + 0.000734 * delta_t**2
    else:
        raise NotImplementedError("'source' must be one of  ['air', 'soil']")


def cop_profile(source_t: xr.DataArray, sink_t: float, source: str = "air") -> xr.DataArray:
    """
    COP time series for a source temperature profile and a sink temperature.

    The result keeps the stamp naming the rule the temperatures were laid on
    the model snapshots with (``_helpers.WEATHER_CALENDAR_ATTR``); arithmetic
    drops attributes, and ``build_heat`` checks the stamp.
    """
    cop = coefficient_of_performance(sink_t - source_t, source)
    if WEATHER_CALENDAR_ATTR in source_t.attrs:
        cop.attrs[WEATHER_CALENDAR_ATTR] = source_t.attrs[WEATHER_CALENDAR_ATTR]
    return cop


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_cop_profiles",
            # simpl="",
            clusters=60,
        )

    for area in ["total", "urban", "rural"]:
        for source in ["air", "soil"]:
            source_t = xr.open_dataarray(snakemake.input[f"temp_{source}_{area}"])

            cop = cop_profile(source_t, snakemake.params.heat_pump_sink_T, source)

            cop.to_netcdf(snakemake.output[f"cop_{source}_{area}"])
