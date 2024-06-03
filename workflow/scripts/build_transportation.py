"""
Module for building transportation infrastructure.
"""

from typing import Optional

import pandas as pd
import pypsa
import xarray as xr
from add_electricity import load_costs


def build_transportation(
    n: pypsa.Network,
    costs_path: str,
) -> None:
    """
    Main funtion to interface with.
    """

    costs = load_costs(costs_path)
