"""
**Description**

State-level small-scale (behind-the-meter / rooftop) solar generation is
retrieved from the U.S. Energy Information Administration (EIA) API v2.

The EIA reports small-scale solar PV generation separately from utility-scale
plants.  This generation is already embedded as a demand reduction in the EIA
930 net-load data used elsewhere in the model.  Having it as an explicit input
allows the RPS constraint to credit it properly (see opts/policy.py).

**EIA series used**

- ``electricity/electric-power-operational-data``
  - ``fueltypeid = SUN``
  - ``sectorid = 98``  (small-scale, i.e. below 1 MW threshold)
  - ``frequency = annual``
  - ``facets[location]`` = US state abbreviations

**Outputs**

- ``data/eia/small_scale_solar.csv``

  Columns: ``state``, ``year``, ``generation_mwh``

  Values are annual generation in MWh, disaggregated by state.  Rows with
  zero or missing generation are dropped.

**API key**

Set ``api: eia: <YOUR_KEY>`` in ``config/config.api.yaml``.  A free key can be
obtained at https://www.eia.gov/opendata/.  Without a key the script falls back
to the bundled historical CSV stored in ``repo_data/policy_constraints/``.
"""

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# EIA API v2 endpoint for electric power operational data
_EIA_URL = "https://api.eia.gov/v2/electricity/electric-power-operational-data/data/"

# Sector 98 = small-scale solar (below 1 MW nameplate)
_SECTOR_ID = "98"
_FUEL_TYPE = "SUN"

# All CONUS state abbreviations covered by pypsa-usa
_STATES = [
    "AL",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]


def _fetch_from_eia(api_key: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Download small-scale solar annual generation from the EIA API v2."""
    records = []
    offset = 0
    page_size = 5000

    while True:
        params = {
            "api_key": api_key,
            "frequency": "annual",
            "data[0]": "generation",
            "facets[fueltypeid][]": _FUEL_TYPE,
            "facets[sectorid][]": _SECTOR_ID,
            "start": str(start_year),
            "end": str(end_year),
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": page_size,
        }

        response = requests.get(_EIA_URL, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()

        data = payload.get("response", {}).get("data", [])
        if not data:
            break

        records.extend(data)

        total = payload.get("response", {}).get("total", 0)
        offset += page_size
        if offset >= int(total):
            break

    if not records:
        raise ValueError(
            "EIA API returned no small-scale solar data. Check your API key and the date range.",
        )

    df = pd.DataFrame(records)
    # API returns location as state abbreviation, period as "YYYY"
    df = df.rename(columns={"location": "state", "period": "year", "generation": "generation_mwh"})
    df = df[["state", "year", "generation_mwh"]].copy()
    df["year"] = df["year"].astype(int)
    # EIA reports generation in thousand MWh; convert to MWh
    df["generation_mwh"] = pd.to_numeric(df["generation_mwh"], errors="coerce") * 1_000
    df = df.dropna(subset=["generation_mwh"])
    df = df[df["generation_mwh"] > 0]
    df = df[df["state"].isin(_STATES)]
    df = df.sort_values(["state", "year"]).reset_index(drop=True)

    logger.info(
        f"Retrieved {len(df)} state-year observations of small-scale solar from EIA API ({start_year}–{end_year}).",
    )
    return df


def _load_fallback(fallback_path: str) -> pd.DataFrame:
    """Load a pre-bundled CSV when no EIA API key is available."""
    logger.warning(
        "No EIA API key provided. Loading bundled small-scale solar data from "
        f"{fallback_path}. This data may not match your planning horizons exactly; "
        "the most recent available year will be used for future years.",
    )
    df = pd.read_csv(fallback_path, dtype={"state": str, "year": int, "generation_mwh": float})
    return df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_small_scale_solar")

    logging.basicConfig(level=logging.INFO)

    api_key = snakemake.params.get("eia_api", None)
    planning_horizons = snakemake.params.planning_horizons
    fallback_path = snakemake.input.fallback

    if api_key:
        # Fetch years from the earliest data year through the last planning horizon.
        # EIA small-scale solar data starts around 2014.
        start_year = 2014
        end_year = max(planning_horizons)
        df = _fetch_from_eia(api_key, start_year, end_year)
    else:
        df = _load_fallback(fallback_path)

    # For planning horizons beyond the latest data year, forward-fill with the
    # most recent observed value for each state.
    latest_year = df["year"].max()
    for horizon in planning_horizons:
        if horizon > latest_year:
            latest = df[df["year"] == latest_year][["state", "generation_mwh"]].copy()
            latest["year"] = horizon
            df = pd.concat([df, latest], ignore_index=True)

    # Keep only relevant years
    df = df[df["year"].isin(planning_horizons)]
    df = df.sort_values(["state", "year"]).reset_index(drop=True)

    output_path = Path(snakemake.output.small_scale_solar)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(
        f"Small-scale solar data written to {output_path} ({len(df)} state-year rows).",
    )
