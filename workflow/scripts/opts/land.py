import logging  # noqa: D100

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from opts._helpers import get_model_horizon

logger = logging.getLogger(__name__)


def _model_investment_periods(n: pypsa.Network) -> list[int]:
    """Investment periods covered by the model currently built on ``n``.

    Under myopic foresight ``solve_network`` builds one model per horizon
    (``snapshots=`` restricted to that period), so this is a single period.
    Returns an empty list when the model is not a multi-investment-period one,
    in which case no activity filtering is possible or needed.
    """
    periods = set(n.investment_periods)
    if not periods:
        return []
    try:
        horizon = get_model_horizon(n.model)
    except (AttributeError, KeyError):  # pragma: no cover - defensive
        return []
    return [p for p in horizon if p in periods]


def _occupied_by_fixed_capacity(n: pypsa.Network, fixed: pd.DataFrame) -> pd.Series:
    """Land already occupied by non-extendable generators, per (carrier, land_region).

    Only capacity that is *active* in the modelled horizon counts: a unit whose
    lifetime has elapsed (or that is not built yet) has released — or not yet
    taken — its land. Activity is evaluated per investment period with PyPSA's
    ``get_active_assets`` (build_year/lifetime), and the peak across the periods
    in the model is used, because the land-use constraint itself has no period
    dimension and must hold for the worst period it spans. For a myopic model
    (one period per solve) that peak *is* the horizon being solved.
    """
    if fixed.empty:
        return pd.Series(dtype=float)

    periods = _model_investment_periods(n)
    if not periods:
        return fixed.groupby(["carrier", "land_region"])["p_nom"].sum()

    component = n.components["Generator"]
    per_period = []
    for period in periods:
        active = component.get_active_assets(investment_period=period)
        active = active.reindex(fixed.index).fillna(False).astype(bool)
        per_period.append(
            fixed[active.to_numpy()].groupby(["carrier", "land_region"])["p_nom"].sum(),
        )

    occupied = pd.concat(per_period, axis=1).max(axis=1)
    return occupied.astype(float)


def add_land_use_constraints(n):
    """
    Adds constraint for land-use based on information from the generators
    table.

    Constraint is defined by land-use per carrier and land_region. The
    definition of land_region enables sub-bus level land-use
    constraints.

    The right-hand side is the developable potential of the (carrier,
    land_region) group — the ``max`` of ``p_nom_max`` over the group, since every
    member describes the same resource area — **net of capacity that is already
    standing on that land**. Concretely::

        sum_{g extendable in (c,z)} P_nom_g
            <= max_{g in (c,z)} p_nom_max_g - sum_{g fixed & active in (c,z)} p_nom_g

    Why the subtraction is needed, and why it is not double counting:

    * ``p_nom_max`` is a *gross* potential. It comes from the land-eligibility
      screens in ``build_renewable_profiles`` (``capacity_per_sqkm * availability @
      area``, or ``capacities / max_cap_factor``) and is never reduced by the
      plants that already occupy the site. ``add_electricity`` writes existing
      plant capacity into ``p_nom``/``p_nom_min`` of the *same* atlite generator
      and ``update_p_nom_max`` only ever raises ``p_nom_max`` up to
      ``p_nom_min``. So today's brownfield plants consume the same potential and
      must be subtracted too, not only model-built vintages.
    * Extendable assets stay on the left-hand side, so nothing is counted twice.
      An "existing" vintage split out by ``add_extra_components`` is extendable
      only when economic retirement is enabled, and then its ``p_nom_max`` is
      capped at its own ``p_nom``; when it is not extendable it lands in the
      subtraction instead. Either way each MW of standing capacity is charged to
      the land exactly once.
    * Under ``foresight: myopic``, ``solve_network.freeze_prior_periods`` sets
      ``p_nom = p_nom_opt`` and ``p_nom_extendable = False`` on every prior-period
      asset. Without the subtraction those MW drop out of the left-hand side
      while the right-hand side keeps the full potential, so the same land could
      be built on again in every horizon. Perfect foresight is unaffected in the
      common case: all vintages are extendable, share the ``land_region``, and are
      already summed on the left-hand side, so nothing is subtracted unless
      non-extendable capacity with a ``land_region`` is present — in which case it
      genuinely occupies land and the same accounting applies.
    """
    model = n.model
    with_region = n.generators.query("land_region != ''").rename_axis(index="name")

    if with_region.empty:
        return

    generators = with_region[with_region.p_nom_extendable]
    if generators.empty:
        return
    p_nom = n.model["Generator-p_nom"].loc[generators.index]

    grouper = pd.concat([generators.carrier, generators.land_region], axis=1)
    lhs = p_nom.groupby(grouper).sum()

    maximum = generators.groupby(["carrier", "land_region"])["p_nom_max"].max()
    maximum = maximum[np.isfinite(maximum)]

    # subtract previous builds (and pre-existing plants) from the maximum allowed
    occupied = _occupied_by_fixed_capacity(n, with_region[~with_region.p_nom_extendable])
    if not occupied.empty:
        maximum = maximum.sub(occupied.reindex(maximum.index).fillna(0.0))
        oversubscribed = maximum < 0
        if oversubscribed.any():
            logger.warning(
                "Existing capacity exceeds the developable potential for %d (carrier, land_region) "
                "group(s); their remaining land is clipped to 0 MW.",
                int(oversubscribed.sum()),
            )
            maximum = maximum.clip(lower=0.0)

    rhs = xr.DataArray(maximum).rename(dim_0="group")
    index = rhs.indexes["group"].intersection(lhs.indexes["group"])

    if not index.empty:
        logger.info("Adding land-use constraints")
        model.add_constraints(
            lhs.sel(group=index) <= rhs.loc[index],
            name="land_use_constraint",
        )
