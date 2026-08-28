"""
Attach electricity demand and provide clustered sector-demand attachment.

The add_demand workflow rule always attaches the configured electricity
demand to the original network. Network simplification and clustering
therefore operate on electricity loads for both electricity-only and
sector studies.

The attach_sector_demand function is called by add_extra_components after
spatial clustering. It replaces the initial electricity loads with the
sector-specific demand profiles, before subsequent demand-dependent
network preparation.
"""

import logging
from pathlib import Path

import pandas as pd
import pypsa
from _helpers import (
    aggregate_sector_demand,
    compose_busmaps,
    configure_logging,
    get_multiindex_snapshots,
    mock_snakemake,
    read_busmap,
)
from constants_sector import (
    TRANSPORT_FUELS,
    SecCarriers,
    SecNames,
)

logger = logging.getLogger(__name__)


def attach_demand(n: pypsa.Network, df: pd.DataFrame, carrier: str, suffix: str):
    """
    Add bus-indexed demand profiles as network Load components.

    Parameters
    ----------
    n : pypsa.Network
        Network receiving the loads. Its snapshots must already be configured.
    df : pd.DataFrame
        Demand with timestamps as rows and existing bus identifiers as columns.
        Rows must correspond positionally to the network snapshots.
    carrier : str
        Carrier assigned to all loads created by this call.
    suffix : str
        Suffix appended to each bus identifier to construct the Load name.
        Bus assignments use the unmodified column identifiers.

    Returns
    -------
    None
        Loads are added to the supplied network in place.

    Raises
    ------
    AssertionError
        If the number of demand rows differs from the network snapshot count.

    Notes
    -----
    The input DataFrame index is converted to timestamps and then replaced
    by the network snapshot index in place. This supports investment-period
    MultiIndices while preserving the existing positional attachment.

    This helper validates row count, not timestamp equality. Callers requiring
    timestamp validation must perform it before attachment.
    """
    df.index = pd.to_datetime(df.index)
    assert len(df.index) == len(
        n.snapshots,
    ), "Demand time series length does not match network snapshots"

    df.index = n.snapshots
    n.madd(
        "Load",
        df.columns,
        suffix=suffix,
        bus=df.columns,
        p_set=df,
        carrier=carrier,
    )


def attach_sector_demand(n, demand_files, busmap_s, busmap_c):
    """
    Replace initial electricity loads with clustered sector demand.

    Parameters
    ----------
    n : pypsa.Network
        Spatially clustered network before temporal aggregation, trimming,
        or demand-response preparation. Existing loads must have carrier
        ``AC``, or the network must contain no loads.
    demand_files : str or list[str]
        Compact sector-demand pickle files produced by build_demand.
        Filenames follow ``{sector}_{end-use}.pkl``, for example
        ``residential_electricity.pkl`` or ``transport_light-duty.pkl``.
    busmap_s : str or path-like
        CSV mapping original buses to simplified buses, including transformer
        removal, substation aggregation, and optional simplification clustering.
    busmap_c : str or path-like
        CSV mapping simplified buses to final clustered buses. Null
        destinations represent buses explicitly removed during processing.

    Returns
    -------
    None
        Existing electricity Load components are removed and sector-specific
        loads are added to the supplied network in place.

    Raises
    ------
    ValueError
        If no demand files are supplied, existing loads have unexpected
        carriers, mappings are invalid or incomplete, final destinations are
        absent from the network, filenames are malformed, or timestamps differ.
    KeyError
        If a sector or end-use label is unknown, or a compact demand file
        lacks a required field.

    Notes
    -----
    The initial electricity demand is replaced, not added to the electrical
    components of sector demand. This avoids double counting.

    The AC Carrier itself is retained because network buses and other
    electrical components continue to use it.

    Original bus allocation factors, demand growth, unit conversion, and
    nodal rounding are applied before aggregation onto final buses. Load
    names use ``"{bus} {carrier}"``, matching the spatial aggregation naming
    convention expected by downstream sector builders.

    This function does not export the network or alter its snapshots.
    Changes are not transactional: an error while processing a later file
    can occur after electricity loads have been removed.
    """
    if isinstance(demand_files, str):
        demand_files = [demand_files]

    if not demand_files:
        raise ValueError("No sector demand files supplied")

    if not n.loads.carrier.eq("AC").all():
        raise ValueError(
            "Expected an electricity-only network before attaching sector demand; "
            "rebuild add_demand, simplify_network and cluster_network",
        )

    busmap = compose_busmaps(
        read_busmap(busmap_s),
        read_busmap(busmap_c),
    )

    missing = pd.Index(busmap.dropna().unique()).difference(n.buses.index)
    if len(missing):
        raise ValueError(
            f"Demand maps to absent clustered buses: {missing.tolist()[:10]}",
        )

    # Replace the initial electricity demand to avoid double counting.
    n.mremove("Load", n.loads.index.copy())

    for demand_file in demand_files:
        sector, end_use = Path(demand_file).stem.split("_")
        sec_name = SecNames[sector.upper()].value
        end_use = end_use.upper().replace("-", "_")

        if sector == "transport":
            sec_car = TRANSPORT_FUELS[end_use.lower()]
        else:
            sec_car = SecCarriers[end_use].value

        carrier = f"{sec_name}-{sec_car}"
        df = aggregate_sector_demand(
            pd.read_pickle(demand_file),
            busmap,
        )

        if not pd.DatetimeIndex(df.index).equals(
            n.snapshots.get_level_values("timestep"),
        ):
            raise ValueError(
                f"Demand timestamps do not match network snapshots: {demand_file}",
            )

        attach_demand(n, df, carrier, suffix=f" {carrier}")
        logger.info(
            "%s %s demand added to network",
            sector.upper(),
            end_use,
        )


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("add_demand", interconnect="western")

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    n.snapshots = get_multiindex_snapshots(
        snakemake.params.snapshots,
        snakemake.params.planning_horizons,
    )
    n.set_investment_periods(
        periods=snakemake.params.planning_horizons,
    )

    df = pd.read_csv(snakemake.input.demand, index_col=0)
    attach_demand(n, df, carrier="AC", suffix="")

    logger.info("Electricity demand added to network")
    n.export_to_netcdf(snakemake.output.network)
