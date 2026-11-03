"""
Builds ``adp2024_h2_demand.csv`` from the EER ADP 2024 detailed output data.

The source workbook (``ADP2024_detailed_output_data.xlsx``, ~100 MB) is not
redistributed with PyPSA-USA. This script is run once, offline, to reduce it to
the annual state level hydrogen demand the workflow reads.

Source: Evolved Energy Research, Annual Decarbonization Perspective 2024,
https://www.evolved.energy/annual-decarbonization-perspectives, sheet
"Hydrogen source and use" (the "To" table, petajoules, higher heating value).

Kept: the baseline, current policy and central scenarios, and every end use but
"Power Generation" (the model builds power sector hydrogen use endogenously).
District of Columbia is folded into Maryland; Alaska and Hawaii are dropped.

Usage::

    python prepare_adp_h2_demand.py <path/to/ADP2024_detailed_output_data.xlsx>
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "scripts"))
from constants import STATE_2_CODE

SCENARIOS = ["baseline", "current policy", "central"]
PJ_2_MWH = 1e15 / 3.6e9


def prepare_adp_h2_demand(xlsx: str) -> pd.DataFrame:
    """Annual non-power hydrogen demand by scenario, state and year in MWh (HHV)."""
    raw = pd.read_excel(xlsx, sheet_name="Hydrogen source and use", header=1)
    df = raw[["To", "Run Name (clean name).1", "State.1", "Year", "value.1", "unit.1"]].dropna()
    df.columns = ["end_use", "scenario", "state", "year", "value", "unit"]

    assert (df.unit == "petajoules").all(), f"Unexpected units {df.unit.unique()}"

    df = df[df.scenario.isin(SCENARIOS) & (df.end_use != "Power Generation")]
    df = df[~df.state.isin(["alaska", "hawaii"])]
    df["state"] = df.state.replace("district of columbia", "maryland").str.title().map(STATE_2_CODE)
    assert df.state.notna().all(), "ADP state names not recognized"

    df = df.groupby(["scenario", "state", "year"], as_index=False)["value"].sum()
    df["demand_mwh"] = (df.pop("value") * PJ_2_MWH).round(1)
    df["year"] = df.year.astype(int)
    return df.sort_values(["scenario", "state", "year"]).reset_index(drop=True)


if __name__ == "__main__":
    demand = prepare_adp_h2_demand(sys.argv[1])
    demand.to_csv(Path(__file__).parent / "adp2024_h2_demand.csv", index=False)
