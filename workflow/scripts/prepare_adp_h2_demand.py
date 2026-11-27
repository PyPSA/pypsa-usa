"""
Builds ``repo_data/hydrogen/adp2024_h2_demand.csv`` from the EER ADP 2024 detailed output data.

The source workbook is not redistributed with PyPSA-USA, so this script is not
part of the workflow: it is run once, offline, and its output is committed.
Download ``ADP2024_detailed_output_data.xlsx`` (~100 MB) from Evolved Energy
Research, Annual Decarbonization Perspective 2024,
https://www.evolved.energy/annual-decarbonization-perspectives.

The demand is the "To" table of the "Hydrogen source and use" sheet
(petajoules, higher heating value). Kept: the baseline, current policy and
central scenarios, and every end use but "Power Generation" (the model builds
power sector hydrogen use endogenously). District of Columbia is folded into
Maryland; Alaska and Hawaii are dropped.

Usage::

    python workflow/scripts/prepare_adp_h2_demand.py <path/to/ADP2024_detailed_output_data.xlsx>
"""

import sys
from pathlib import Path

import pandas as pd
from constants import STATE_2_CODE

SCENARIOS = ["baseline", "current policy", "central"]
PJ_2_MWH = 1e15 / 3.6e9
OUTPUT = Path(__file__).resolve().parents[1] / "repo_data" / "hydrogen" / "adp2024_h2_demand.csv"


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
    prepare_adp_h2_demand(sys.argv[1]).to_csv(OUTPUT, index=False)
