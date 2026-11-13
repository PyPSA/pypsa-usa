"""
Constraints of components retrofitted from existing ones.

Natural gas storage and pipelines are retrofitted to hydrogen
(``build_hydrogen``), existing gas plants to hydrogen turbines or carbon capture
(``add_extra_components.attach_retrofits``). Each retrofit names the component
it is retrofitted from (``retrofit_from``) and its capacity per unit of that
component's capacity (``retrofit_factor``). The capacity of a generator is
measured by its fuel input (p_nom / efficiency), as that of the links
``add_sectors`` converts generators to.
"""

import logging

import pandas as pd
import pypsa
from xarray import DataArray

logger = logging.getLogger(__name__)

# components that can be retrofitted, and their capacity
CAPACITIES = {"Store": "e_nom", "Link": "p_nom", "Generator": "p_nom"}


def retrofitted(n: pypsa.Network) -> pd.Index:
    """Components that others are retrofitted from."""
    original = [_retrofits(n.components[c].static).retrofit_from for c in CAPACITIES]
    return pd.Index(sorted(set(pd.concat(original))), name="original")


def _retrofits(static: pd.DataFrame) -> pd.DataFrame:
    """Components retrofitted from another component."""
    if "retrofit_from" not in static:
        return static.iloc[:0].assign(retrofit_from=pd.Series(dtype=object))
    return static[static.retrofit_from.fillna("") != ""]


def _identity(names: pd.Index, dim: str) -> pd.Series:
    """Groups components by their own name, as ``dim``."""
    return pd.Series(names, index=pd.Index(names, name="name"), name=dim)


def add_retrofit_constraints(n: pypsa.Network) -> None:
    """
    Shares the capacity of each component with the components retrofitted from it.

    The capacity left and the capacity retrofitted add up to at most the
    existing one, so that the component may also retire:

        original + sum(retrofit / retrofit_factor) <= existing

    A component no longer extendable (built in an earlier myopic horizon) is
    left out: a retrofit has already been taken from the existing capacity, and
    an original keeps all of it. The original and its retrofits may be different
    components (a gas plant link and its hydrogen turbine generator).

    A reservoir's retrofitted share is one share of its energy and rates
    (``_add_reservoir_constraints``).
    """
    originals = retrofitted(n)
    if originals.empty:
        return

    lhs, existing = [], pd.Series(0.0, index=originals)
    for component, attr in CAPACITIES.items():
        static = n.components[component].static
        retrofits = _retrofits(static)
        own = static.index.intersection(originals)
        # the original of each retrofit, and of each original itself
        original = pd.concat([retrofits.retrofit_from, _identity(own, "original")]).rename("original")
        if original.empty:
            continue
        units = static.loc[original.index]
        # generators count by fuel input
        scale = 1 / units.efficiency if component == "Generator" else pd.Series(1.0, index=units.index)
        # retrofits count by capacity / retrofit_factor, originals by capacity
        factor = units.retrofit_factor.where(units.index.isin(retrofits.index), 1) if "retrofit_factor" in units else 1

        extendable = units.index[units[f"{attr}_extendable"]]
        own_extendable = extendable.intersection(own)
        # the existing capacity is the original's upper bound
        existing[own_extendable] += units.loc[own_extendable, f"{attr}_max"] * scale[own_extendable]
        if extendable.empty:
            continue
        capacity = n.model[f"{component}-{attr}"].sel(name=extendable)
        # sum each original with its retrofits
        expr = (capacity * DataArray((scale / factor)[extendable])).groupby(original[extendable]).sum()
        # the originals of each component are a subset of all: align them by name, not position
        lhs.append(expr.reindex(original=originals))

    n.model.add_constraints(sum(lhs[1:], lhs[0]), "<=", DataArray(existing), name="retrofit-capacity")
    logger.info(f"Shared the capacity of {len(originals)} components with the components retrofitted from them")

    _add_reservoir_constraints(n)


def _add_reservoir_constraints(n: pypsa.Network) -> None:
    """
    Shares the wells of reservoirs retrofitted to hydrogen between hydrogen and natural gas.

    ``retrofit-capacity`` shares the volume; this ties the rates to it. For each
    reservoir:

        h2_withdrawal / h2_withdrawal_whole == h2_energy / h2_energy_whole
        h2_injection <= h2_withdrawal
        gas_injection <= gas_withdrawal

    where ``*_whole`` is the whole reservoir retrofitted (``retrofit_factor``
    times the gas capacity). The wells go with the volume, so the withdrawal is
    fixed by the retrofitted share. Injection runs through the same wells, so it
    is capped by the withdrawal of its side.
    """
    stores = _retrofits(n.stores)
    links = _retrofits(n.links)
    if stores.empty or links.empty:
        return
    p_nom = n.model["Link-p_nom"]

    # hydrogen withdrawal: each discharger and the retrofitted store on its bus
    store_of_bus = pd.Series(stores.index, index=stores.bus)
    discharger = links[links.bus0.isin(store_of_bus.index)]
    store = discharger.bus0.map(store_of_bus).rename("store")
    reservoirs = stores.loc[store]
    whole_rate = discharger.retrofit_factor * n.links.loc[discharger.retrofit_from, "p_nom_max"].to_numpy()
    whole_energy = reservoirs.retrofit_factor * n.stores.loc[reservoirs.retrofit_from, "e_nom_max"].to_numpy()
    # rate == r * energy, with r = whole_rate / whole_energy (~1e-5 to 1e-3 per
    # hour), written as rate / sqrt(r) == sqrt(r) * energy so both coefficients
    # are better conditioned, as in opts.sector.add_gas_storage_contraction_constraints
    scale = (whole_rate.to_numpy() / whole_energy.to_numpy()) ** 0.5
    lhs = (p_nom.sel(name=discharger.index) / DataArray(pd.Series(scale, index=discharger.index))).groupby(
        store,
    ).sum() - (
        n.model["Store-e_nom"].sel(name=reservoirs.index) * DataArray(pd.Series(scale, index=reservoirs.index))
    ).groupby(_identity(reservoirs.index, "store")).sum()
    n.model.add_constraints(lhs, "==", 0, name="Store-retrofitted-withdrawal-share")

    # hydrogen injection, through the retrofitted wells
    discharger_of_bus = pd.Series(discharger.index, index=discharger.bus0)
    charger = n.links[(n.links.carrier == "h2_storage_charger") & n.links.bus1.isin(discharger_of_bus.index)]
    _add_injection_limit(n, charger.bus1.map(discharger_of_bus), "hydrogen")

    # gas injection, through the wells left to gas
    gas_bus = n.stores.loc[reservoirs.retrofit_from, "bus"]
    gas_links = n.links[n.links.carrier == "gas storage"]
    gas_discharge = gas_links[gas_links.bus0.isin(gas_bus)]
    discharge_of_bus = pd.Series(gas_discharge.index, index=gas_discharge.bus0)
    gas_charge = gas_links[gas_links.bus1.isin(discharge_of_bus.index)]
    _add_injection_limit(n, gas_charge.bus1.map(discharge_of_bus), "gas")


def _add_injection_limit(n: pypsa.Network, withdrawal: pd.Series, kind: str) -> None:
    """Keeps each injection link (the index) at most its reservoir's ``withdrawal`` link."""
    if withdrawal.empty:
        return
    p_nom = n.model["Link-p_nom"]
    pairs = withdrawal.rename("withdrawal").rename_axis("name")
    lhs = (
        p_nom.sel(name=pairs.index).groupby(pairs).sum()
        - p_nom.sel(name=pairs.unique()).groupby(_identity(pairs.unique(), "withdrawal")).sum()
    )
    n.model.add_constraints(lhs, "<=", 0, name=f"Link-retrofitted-{kind}-injection")
