"""
Retrieve industrial end-use load shapes from the EPRI Load Shape Library.

The shapes are requested from the JSON endpoints that the
`EPRI Load Shape Library <https://loadshape.epri.com/enduse>`_ page itself calls
when a shape is plotted, so the data are fetched by each user rather than
redistributed with PyPSA-USA. For every NERC region, industrial end use and
season, the average weekday and the average weekend shape are retrieved,
together with the end use's annual energy per watt of peak (the ``Scale
Multiplier``).

The shapes are hourly typical days, hour-ending, in local time. Weekday shapes
peak at 1; weekend shapes are scaled against the weekday peak, so their level
relative to the weekday is kept.

**Outputs**

- ``data/industry_load/epri_load_shapes.csv``
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests
from _helpers import configure_logging

logger = logging.getLogger(__name__)

ROOT = "https://loadshape.epri.com"

REGIONS = [
    "ECAR",
    "ERCOT",
    "MAAC",
    "MAIN",
    "MAPP",
    "NYPCC_NY",
    "NPCC_NE",
    "SERC_STV",
    "SERC_FL",
    "SPP",
    "WSCC_NWP",
    "WSCC_RA",
    "WSCC_CNV",
]

# EPRI request key: name used in the output
END_USES = {
    "indHVAC": "HVAC",
    "indLighting": "Lighting",
    "indMachineDrives": "MachineDrives",
    "indProcessHeating": "ProcessHeating",
}

SEASONS = ["Peak", "OffPeak"]
DAY_TYPES = ["AverageWeekday", "AverageWeekend"]

#: Pause between requests, to keep the load on the EPRI server light.
PAUSE_SECONDS = 0.2


def _request_region(region: str) -> str:
    """EPRI spells sub-regions with a slash (NPCC/NE), the output with an underscore."""
    return region.replace("_", "/", 1)


def _post(session: requests.Session, endpoint: str, data: dict, retries: int = 3):
    for attempt in range(retries):
        try:
            response = session.post(f"{ROOT}/{endpoint}", data=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_shape(session: requests.Session, region: str, end_use: str, season: str, day_type: str) -> list[float]:
    """24 hour-ending values of one typical-day shape."""
    data = _post(
        session,
        "GetEndUseLoadShapeJson",
        {
            "country": "USA",
            "region": _request_region(region),
            "season": season,
            "dayType": day_type,
            "endUse": end_use,
        },
    )
    if not data.get("DataFound") or len(data.get("Demand", [])) != 24:
        raise ValueError(f"EPRI returned no shape for {region} {end_use} {season} {day_type}")
    return [demand for _, demand in sorted(zip(data["Hour"], data["Demand"]))]


def fetch_scale(session: requests.Session, region: str, end_use: str) -> float:
    """Annual energy (Wh) of the end use per watt of peak demand."""
    data = _post(
        session,
        "GetAnnualWattHours",
        {
            "Country": "USA",
            "Region": _request_region(region),
            "EndUse": end_use,
            "PeakWatts": 1,
        },
    )
    return float(data)


def retrieve_load_shapes(session: requests.Session) -> pd.DataFrame:
    """All industrial shapes in the layout read by ``build_demand.ReadCliu``."""
    rows = []
    for region in REGIONS:
        for key, end_use in END_USES.items():
            scale = fetch_scale(session, region, key)
            time.sleep(PAUSE_SECONDS)
            for season in SEASONS:
                for day_type in DAY_TYPES:
                    shape = fetch_shape(session, region, key, season, day_type)
                    time.sleep(PAUSE_SECONDS)
                    row = {
                        "Region": region,
                        "Season": season,
                        "Day Type": day_type,
                        "Sector": "Industrial",
                        "End Use": end_use,
                        "Scale Multiplier": scale,
                    }
                    row.update({f"HR{hour}": round(value, 4) for hour, value in enumerate(shape, start=1)})
                    rows.append(row)
        logger.info(f"Retrieved EPRI industrial load shapes for {region}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_epri_load_shapes")

    configure_logging(snakemake)

    output = Path(snakemake.output[0])
    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading EPRI industrial load shapes from '{ROOT}'.")
    with requests.Session() as session:
        shapes = retrieve_load_shapes(session)
    shapes.to_csv(output, index=False)
