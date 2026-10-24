"""Unit tests for formatting EIA natural gas responses in ``eia.py``.

The raw frames mirror the API v2 response; nothing here touches the network.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import eia

pytestmark = pytest.mark.fast


def test_gas_production_keeps_only_the_requested_year():
    """The inclusive ``end={year+1}-01`` bound also returns January of the next year."""
    raw = pd.DataFrame(
        {
            "period": ["2018-01", "2018-12", "2019-01", "2018-12"],
            "area-name": ["CALIFORNIA", "CALIFORNIA", "CALIFORNIA", "U.S."],
            "series-description": [
                "California Natural Gas Marketed Production (MMcf)",
                "California Natural Gas Marketed Production (MMcf)",
                "California Natural Gas Marketed Production (MMcf)",
                "U.S. Natural Gas Marketed Production (MMcf)",
            ],
            "value": [10.0, 12.0, 99.0, 100.0],
            "units": ["MMCF"] * 4,
        },
    )

    df = eia._GasProduction("market", 2018, "key").format_data(raw)

    assert (df.index.year == 2018).all()
    assert df[df.state == "CA"].value.tolist() == [10.0, 12.0]
