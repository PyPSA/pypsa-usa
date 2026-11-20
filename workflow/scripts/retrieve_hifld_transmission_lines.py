"""
Download the HIFLD transmission lines layer as one GeoJSON file.

The layer is served by an ArcGIS feature service
(``constants.HIFLD_TRANSMISSION_LINES_URL``), which returns at most a few thousand
features per request, so it is read in pages and the pages are joined. Features
are written unchanged (all attributes, WGS84). The download fails if the feature
count does not match the count the service reports.
"""

import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path

from _helpers import configure_logging

logger = logging.getLogger(__name__)

PAGE_SIZE = 2000
ATTEMPTS = 4


def _get_json(url: str, params: dict) -> dict:
    query = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(query, timeout=300) as response:
                data = json.load(response)
            if "error" in data:
                raise RuntimeError(data["error"])
            return data
        except Exception as err:
            if attempt == ATTEMPTS:
                raise
            logger.warning("Request failed (%s); retry %d of %d.", err, attempt, ATTEMPTS - 1)
            time.sleep(10 * attempt)


def download_layer(layer_url: str) -> dict:
    """All features of an ArcGIS feature layer as one GeoJSON FeatureCollection."""
    query_url = layer_url.rstrip("/") + "/query"
    expected = _get_json(query_url, {"where": "1=1", "returnCountOnly": "true", "f": "json"})["count"]
    oid = _get_json(layer_url, {"f": "json"})["objectIdField"]

    features = []
    while len(features) < expected:
        page = _get_json(
            query_url,
            {
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": 4326,
                "orderByFields": oid,
                "resultOffset": len(features),
                "resultRecordCount": PAGE_SIZE,
                "f": "geojson",
            },
        )
        batch = page.get("features", [])
        if not batch:
            break
        features += batch
        logger.info("Downloaded %d of %d features.", len(features), expected)

    if len(features) != expected:
        raise RuntimeError(f"Downloaded {len(features)} features, the service reports {expected}.")
    return {"type": "FeatureCollection", "features": features}


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_hifld_transmission_lines")
    configure_logging(snakemake)

    collection = download_layer(snakemake.params.url)
    output = Path(snakemake.output[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(collection), encoding="utf-8")
    logger.info("Wrote %d features to %s.", len(collection["features"]), output)
