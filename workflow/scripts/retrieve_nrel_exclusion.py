"""Download a single NREL land-access artifact (avail_*.nc or caps_*.nc)
from the Zenodo bundle into the path snakemake specified.

Driven by the `retrieve_nrel_exclusion_artifact` rule in rules/retrieve.smk.
"""

from pathlib import Path

from zenodo_downloader import ZenodoScenarioDownloader

if __name__ == "__main__":
    out_path = Path(snakemake.output[0])
    filename = out_path.name

    downloader = ZenodoScenarioDownloader()
    record_id = downloader.scenario_records["nrel_exclusion_v1"]
    downloader.download_to_path(record_id, filename, out_path)
