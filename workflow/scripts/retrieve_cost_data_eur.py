"""
Retrieves cost data for Europe.

This is a seperate rule due to the need for a wildcard argument
"""

import logging
from pathlib import Path

from _helpers import mock_snakemake, progress_retrieve

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_cost_data_eur", year=2030)
        rootpath = ".."
    else:
        rootpath = "."

    # get european template data
    version = snakemake.params.pypsa_costs_version
    tech_year = snakemake.wildcards.year
    csv = f"https://raw.githubusercontent.com/PyPSA/technology-data/{version}/outputs/costs_{tech_year}.csv"
    save_tech_data = snakemake.output.pypsa_technology_data
    if not Path(save_tech_data).exists():
        logger.info(f"Downloading PyPSA-Eur costs from '{csv}'")
        progress_retrieve(csv, save_tech_data)
