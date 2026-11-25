"""
Temperature-dependent forced outage derate of thermal generators.

``conventional: ambient_derate`` with ``source: murphy2019`` sets the available
capacity of every thermal generator to

    p_max_pu(t) = 1 - FOR(T_bus(t)),

where ``FOR`` is the forced outage rate of the generator's prime mover at the
ambient air temperature of its bus (the area-weighted mean over the bus region,
``build_bus_temperature``). The curves are those of Murphy et al. (2019),
https://doi.org/10.1016/j.apenergy.2019.113513, as ReEDS uses them
(``reeds/input_processing/outage_rates.py``, ``inputs/plant_characteristics/
outage_forced_temperature_murphy2019.csv``): tabulated from -15 to 35 degC in
5 degC steps, interpolated linearly in between, extrapolated linearly beyond the
table with the slope of its two end points, and clipped to
``[0, AMBIENT_DERATE_MAX_FORCED_OUTAGE]``. The carrier to prime mover mapping is
``constants.AMBIENT_DERATE_PRIME_MOVER``.

The derate replaces the EIA-860 summer/winter derate of these carriers
(``add_electricity`` skips it) and the static ``ucap`` forced outage derate (the two
options are mutually exclusive): both describe the same thermal deficiency, so
stacking them would count it twice.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import check_weather_calendar
from constants import AMBIENT_DERATE_MAX_FORCED_OUTAGE, AMBIENT_DERATE_PRIME_MOVER

logger = logging.getLogger(__name__)

#: ``conventional: ambient_derate: source`` values: the implemented one first
AMBIENT_DERATE_SOURCES = ("murphy2019", "cpuc_servm")


def ambient_derate_enabled(conventional: dict | None, ucap: dict | None = None) -> bool:
    """
    Whether ``conventional: ambient_derate`` is on, after checking its settings.

    Raises when the settings ask for something that is not done: the reserved
    ``cpuc_servm`` source, an unknown source, stacking on the seasonal derate, or
    ``ucap`` at the same time.
    """
    settings = (conventional or {}).get("ambient_derate") or {}
    if not settings.get("enable", False):
        return False
    source = settings.get("source", AMBIENT_DERATE_SOURCES[0])
    if source == "cpuc_servm":
        raise NotImplementedError(
            "conventional.ambient_derate.source: cpuc_servm is reserved; the CPUC SERVM unit-specific hourly "
            "derate profiles are not retrieved or applied. Use source: murphy2019.",
        )
    if source != "murphy2019":
        raise ValueError(
            f"Unknown conventional.ambient_derate.source '{source}'; expected one of {AMBIENT_DERATE_SOURCES}.",
        )
    if not settings.get("replaces_seasonal_derate", True):
        raise ValueError(
            "conventional.ambient_derate.replaces_seasonal_derate must be true: the ambient derate replaces the "
            "EIA-860 seasonal derate and is never stacked on it.",
        )
    if (ucap or {}).get("enable", False):
        raise ValueError(
            "conventional.ambient_derate and ucap are mutually exclusive: both derate thermal capacity by its "
            "forced outage rate, so enabling both counts it twice. Disable one of them.",
        )
    return True


def load_outage_curves(path: str) -> pd.DataFrame:
    """Forced outage fraction, indexed by temperature (degC), one column per prime mover."""
    curves = pd.read_csv(path, comment="#").pivot(
        index="deg_celsius",
        columns="prime_mover",
        values="outage_frac",
    )
    return curves.sort_index().astype(float)


def forced_outage_rate(
    curve: pd.Series,
    temperature,
    cap: float = AMBIENT_DERATE_MAX_FORCED_OUTAGE,
) -> np.ndarray:
    """
    Forced outage rate of one prime mover at ``temperature`` (degC).

    Linear between the tabulated points of ``curve``; beyond them linear with the
    slope of the two end points (ReEDS ``extrapolate_forward_backward``); the result
    is clipped to ``[0, cap]``.
    """
    x = curve.index.to_numpy(dtype=float)
    y = curve.to_numpy(dtype=float)
    t = np.asarray(temperature, dtype=float)
    rate = np.interp(t, x, y)
    slope_low = (y[1] - y[0]) / (x[1] - x[0])
    slope_high = (y[-1] - y[-2]) / (x[-1] - x[-2])
    rate = np.where(t < x[0], y[0] + slope_low * (t - x[0]), rate)
    rate = np.where(t > x[-1], y[-1] + slope_high * (t - x[-1]), rate)
    return np.clip(rate, 0.0, cap)


def load_bus_temperature(path: str) -> pd.DataFrame:
    """
    Area-weighted air temperature (degC) per clustered bus.

    ``build_bus_temperature`` output: one weather year on the snapshots of
    ``config["snapshots"]`` (atlite converts the ERA5 Kelvin to degC).
    """
    da = xr.open_dataarray(path)
    check_weather_calendar(da, path)
    temperature = da.to_pandas()
    temperature.index = pd.DatetimeIndex(temperature.index)
    temperature.columns = temperature.columns.astype(str)
    return temperature


def align_to_snapshots(temperature: pd.DataFrame, snapshots: pd.Index) -> pd.DataFrame:
    """
    Lay the one weather year of ``temperature`` onto every investment period.

    Each period's timesteps must be the same calendar hours (month, day, hour) as
    the temperature index, in the same order: the snapshots and the temperature
    profile are both built from ``config["snapshots"]``.
    """
    temperature = temperature.sort_index()
    if isinstance(snapshots, pd.MultiIndex):
        periods = snapshots.unique(level=0)
        timesteps = [snapshots[snapshots.get_level_values(0) == period].get_level_values(1) for period in periods]
    else:
        timesteps = [pd.DatetimeIndex(snapshots)]

    def calendar(index: pd.DatetimeIndex) -> np.ndarray:
        index = pd.DatetimeIndex(index)
        return np.column_stack([index.month, index.day, index.hour, index.minute])

    for steps in timesteps:
        if len(steps) != len(temperature) or not (calendar(steps) == calendar(temperature.index)).all():
            raise ValueError(
                f"The temperature profile ({len(temperature)} steps from {temperature.index[0]}) does not cover the "
                f"same hours as the snapshots of an investment period ({len(steps)} steps from {steps[0]}). Rebuild "
                "it (build_bus_temperature) for the current snapshots.",
            )
    values = np.concatenate([temperature.to_numpy()] * len(timesteps))
    return pd.DataFrame(values, index=snapshots, columns=temperature.columns)


def bus_temperature(n: pypsa.Network, temperature: pd.DataFrame, buses: pd.Index) -> pd.DataFrame:
    """
    Temperature at each of ``buses``.

    The profile covers the onshore regions of the clustered network. A bus outside
    them (an external import bus, for example) takes the mean temperature of the
    buses it is linked to that have one, otherwise the mean of all buses.
    """
    known = buses.intersection(temperature.columns)
    out = temperature[known].copy()
    missing = buses.difference(temperature.columns)
    if missing.empty:
        return out[buses]

    branches = pd.concat(
        [n.links[["bus0", "bus1"]], n.lines[["bus0", "bus1"]]],
    )
    network_mean = temperature.mean(axis=1)
    fallback = []
    for bus in missing:
        neighbours = pd.Index(
            pd.concat([branches.bus1[branches.bus0 == bus], branches.bus0[branches.bus1 == bus]]).unique(),
        ).intersection(temperature.columns)
        if neighbours.empty:
            out[bus] = network_mean
            fallback.append(bus)
        else:
            out[bus] = temperature[neighbours].mean(axis=1)
    logger.warning(
        f"No temperature profile for bus(es) {list(missing)}; using the mean temperature of the buses they are "
        f"linked to{f', or of all buses for {fallback}' if fallback else ''}.",
    )
    return out[buses]


def _replace_columns(frame: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with the columns of ``new`` replaced or added, keeping its column dtype and name."""
    out = pd.concat([frame.drop(columns=new.columns, errors="ignore"), new], axis=1)
    out.columns = pd.Index(out.columns, dtype=frame.columns.dtype, name=frame.columns.name)
    return out


def apply_ambient_derate(
    n: pypsa.Network,
    curves: pd.DataFrame,
    temperature: pd.DataFrame,
    prime_movers: dict[str, str] = AMBIENT_DERATE_PRIME_MOVER,
) -> pd.Index:
    """
    Set ``p_max_pu(t) = 1 - FOR(T_bus(t))`` on every thermal generator.

    Parameters
    ----------
    n
        Network with all generators attached (existing fleet, new builds,
        retrofits, remote contracted units).
    curves
        :func:`load_outage_curves` output.
    temperature
        :func:`load_bus_temperature` output: one weather year, one column per bus.
    prime_movers
        Carrier to prime mover (a column of ``curves``); generators of other
        carriers are left alone.

    The time series REPLACES the generators' p_max_pu (static and time-varying),
    it does not multiply it. A non-zero minimum stable level is scaled by the same
    factor, ``p_min_pu(t) = p_min_pu * (1 - FOR(t))``, as ``apply_ucap`` does: a
    forced outage takes that share of the fleet away, minimum stable level
    included, and the unit's dispatch range must not become empty.

    Returns the derated generators.
    """
    gens = n.generators[n.generators.carrier.isin(list(prime_movers))]
    if gens.empty:
        logger.info("conventional.ambient_derate: no thermal generators to derate.")
        return gens.index

    needed = {prime_movers[carrier] for carrier in gens.carrier.unique()}
    missing = needed.difference(curves.columns)
    if missing:
        raise ValueError(f"No forced outage curve for prime mover(s) {sorted(missing)}.")

    temp = bus_temperature(n, align_to_snapshots(temperature, n.snapshots), pd.Index(gens.bus.unique()))

    # one curve evaluation per (prime mover, bus), shared by the units there
    keys = list(zip(gens.carrier.map(prime_movers), gens.bus))
    availability = {key: 1.0 - forced_outage_rate(curves[key[0]], temp[key[1]].to_numpy()) for key in set(keys)}
    p_max_pu = pd.DataFrame(
        np.column_stack([availability[key] for key in keys]),
        index=n.snapshots,
        columns=gens.index,
    )

    n.generators.loc[gens.index, "p_max_pu"] = 1.0
    n.generators_t.p_max_pu = _replace_columns(n.generators_t.p_max_pu, p_max_pu)

    p_min_pu = n.get_switchable_as_dense("Generator", "p_min_pu", inds=gens.index)
    with_min = p_min_pu.columns[(p_min_pu != 0).any()]
    if len(with_min):
        scaled = p_min_pu[with_min].mul(p_max_pu[with_min]).clip(upper=p_max_pu[with_min])
        n.generators_t.p_min_pu = _replace_columns(n.generators_t.p_min_pu, scaled)

    summary = p_max_pu.T.groupby(gens.carrier).mean().T.agg(["mean", "min"]).T.round(3)
    logger.info(
        f"conventional.ambient_derate: temperature-dependent p_max_pu on {len(gens)} generators "
        f"({len(with_min)} with their p_min_pu scaled alike):\n{summary}",
    )
    return gens.index
