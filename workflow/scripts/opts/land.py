import logging  # noqa: D100

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def add_land_use_constraints(n):
    """
    Adds constraint for land-use based on information from the generators
    table.

    Constraint is defined by land-use per carrier and land_region. The
    definition of land_region enables sub-bus level land-use
    constraints.
    """
    model = n.model
    generators = n.generators.query(
        "p_nom_extendable & land_region != '' ",
    ).rename_axis(index="Generator-ext")

    if generators.empty:
        return
    p_nom = n.model["Generator-p_nom"].loc[generators.index]

    grouper = pd.concat([generators.carrier, generators.land_region], axis=1)
    lhs = p_nom.groupby(grouper).sum()

    maximum = generators.groupby(["carrier", "land_region"])["p_nom_max"].max()
    maximum = maximum[np.isfinite(maximum)]

    rhs = xr.DataArray(maximum).rename(dim_0="group")
    index = rhs.indexes["group"].intersection(lhs.indexes["group"])

    if not index.empty:
        logger.info("Adding land-use constraints")
        model.add_constraints(
            lhs.sel(group=index) <= rhs.loc[index],
            name="land_use_constraint",
        )
