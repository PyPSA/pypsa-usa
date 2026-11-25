"""State lookup of the FERC 714 demand reader (``build_demand.ReadFERC714``)."""

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_demand import ReadFERC714
from constants import CODE_2_STATE, FIPS_2_STATE, STATE_FIPS_2_CODE

pytestmark = pytest.mark.fast


def test_fips_lookup_covers_the_states_and_dc():
    assert len(STATE_FIPS_2_CODE) == 51
    assert STATE_FIPS_2_CODE["06"] == "CA"
    assert STATE_FIPS_2_CODE["11"] == "DC"
    # every code has a full name the demand tables are keyed by
    assert set(STATE_FIPS_2_CODE.values()) <= set(CODE_2_STATE)
    # derived from the FIPS name table, so the two cannot drift apart
    assert STATE_FIPS_2_CODE.keys() == FIPS_2_STATE.keys()


def test_ferc714_demand_is_keyed_by_state_name(tmp_path):
    """The reader names the state columns without any census download (#394)."""
    hours = pd.date_range("2019-01-01", periods=8760, freq="h")
    raw = pd.concat(
        [
            pd.DataFrame(
                {
                    "datetime_utc": hours,
                    "state_id_fips": fips,
                    "demand_mwh": 1.0,
                    "scaled_demand_mwh": value,
                },
            )
            for fips, value in (("06", 10.0), ("48", 20.0))
        ],
    )
    fn = tmp_path / "ferc714.parquet"
    raw.to_parquet(fn)

    demand = ReadFERC714([str(fn)]).read_demand()

    assert sorted(demand.columns) == ["California", "Texas"]
    assert demand["California"].iloc[0] == 10.0
    assert demand["Texas"].iloc[0] == 20.0
