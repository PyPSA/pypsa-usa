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
from eia import SmallScaleSolar

logger = logging.getLogger(__name__)


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
        df = SmallScaleSolar(start_year, end_year, api_key).get_data()
        logger.info(
            f"Retrieved {len(df)} state-year observations of small-scale solar "
            f"from EIA API ({start_year}–{end_year})."
        )
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
