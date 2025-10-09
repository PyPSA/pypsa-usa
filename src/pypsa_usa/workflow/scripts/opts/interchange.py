"""Adds constraints for electricity imports and exports."""

import logging

import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


def get_import_export_limit(n: pypsa.Network, timesteps_in_period: pd.Series) -> float:
    """Get the import or export limit for the network."""
    timesteps = pd.to_datetime(timesteps_in_period)
    load_names = n.loads[n.loads.carrier == "AC"].index.tolist()
    weights = n.snapshot_weightings.objective
    loads_t = n.loads_t["p_set"]
    loads_t = loads_t[loads_t.index.get_level_values("timestep").isin(timesteps)][load_names].mul(weights, axis=0)
    return loads_t.sum().sum()


def get_periods(n: pypsa.Network) -> pd.Series:
    """Get time periods from network snapshots.

    Parameters
    ----------
    n : pypsa.Network
        Network object containing snapshots
    period_type : str
        Type of period to return ('day', 'week', 'month', 'year')

    Returns
    -------
    pd.Series
        Series with datetime index and period values
    """
    timestamps = n.snapshots.get_level_values("timestep")

    periods = pd.DataFrame(index=timestamps)
    periods["day"] = timestamps.dayofyear
    periods["week"] = timestamps.isocalendar().week
    periods["month"] = timestamps.month
    periods["year"] = timestamps.year

    return periods


def add_interchange_constraints(n, config, direction):
    """Adds constraints for inter-regional energy flow."""
    assert direction in ["imports", "exports"], f"direction must be either imports or exports; received: {direction}"

    def _get_elec_import_links(n: pypsa.Network) -> list[str]:
        """Get all links for elec trade."""
        return n.links[n.links.carrier == "imports"].index.tolist()

    def _get_elec_export_links(n: pypsa.Network) -> list[str]:
        """Get all links for elec trade."""
        return n.links[n.links.carrier == "exports"].index.tolist()

    # default to 10% as unlimited exports can lead to weird behaviour
    if direction == "imports":
        volume_limit = config["electricity"].get("imports", {}).get("volume_limit", 10)
    elif direction == "exports":
        volume_limit = config["electricity"].get("exports", {}).get("volume_limit", 10)
    else:
        raise ValueError(f"direction must be either imports or exports; received: {direction}")

    if isinstance(volume_limit, str):
        if volume_limit == "inf":
            logger.info(f"Volume limit is infinite, skipping {direction} volume limit constraints")
            return
        else:
            raise ValueError(f"volume_limit must be a number or 'inf'; received: {volume_limit}")
    elif isinstance(volume_limit, int | float):
        if volume_limit < 0 or volume_limit > 100:
            raise ValueError(f"volume_limit must be between 0 and 100; received: {volume_limit}")
    else:
        raise ValueError(f"volume_limit must be a number or 'inf'; received: {volume_limit}")

    weights = n.snapshot_weightings.objective
    period = n.snapshots.get_level_values("period").unique().tolist()

    if direction == "imports":
        balancing_period = config["electricity"]["imports"].get("balancing_period", "month")
    elif direction == "exports":
        balancing_period = config["electricity"]["exports"].get("balancing_period", "month")
    else:
        raise ValueError(f"direction must be either imports or exports; received: {direction}")

    if direction == "imports":
        links = _get_elec_import_links(n)
    else:
        links = _get_elec_export_links(n)

    periods = get_periods(n)

    timesteps = n.snapshots.get_level_values("timestep")
    for year in periods["year"].unique():
        periods_in_year = periods[periods["year"] == year]
        for period in periods_in_year[balancing_period].unique():
            if balancing_period == "week":
                timesteps_in_period = timesteps[
                    (timesteps.year == year) & (timesteps.isocalendar().week == period)
                ].strftime("%Y-%m-%d %H:%M:%S")
            else:
                timesteps_in_period = timesteps[
                    (timesteps.year == year) & (getattr(timesteps, balancing_period) == period)
                ].strftime("%Y-%m-%d %H:%M:%S")

            lhs = (
                n.model["Link-p"]
                .mul(weights)
                .sel(period=year, Link=links)
                .sel(timestep=timesteps_in_period)  # Seperate cause slicing on multi-index is not supported
                .sum()
            )

            rhs = round(get_import_export_limit(n, timesteps_in_period) * volume_limit / 100, 2)

            n.model.add_constraints(
                lhs <= rhs,
                name=f"elec_trade_{direction}_{year}_{balancing_period}-{period}",
            )
