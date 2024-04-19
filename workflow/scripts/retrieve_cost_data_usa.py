"""
Retrieves cost data.
"""

import logging
from pathlib import Path

from _helpers import mock_snakemake, progress_retrieve

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_cost_data_usa")
        rootpath = ".."
    else:
        rootpath = "."

    # get nrel atb power generation data
    atb_year = 2023
    parquet = f"https://oedi-data-lake.s3.amazonaws.com/ATB/electricity/parquet/{atb_year}/ATBe.parquet"
    save_atb = snakemake.output.nrel_atb

    if not Path(save_atb).exists():
        logger.info(f"Downloading ATB costs from '{parquet}'")
        progress_retrieve(parquet, save_atb)

    """
    # get nrel atb transportation data
    xlsx = "https://atb-archive.nrel.gov/transportation/2020/files/2020_ATB_Data_VehFuels_Download.xlsx"
    save_atb_transport = snakemake.output.nrel_atb_transport

    if not Path(save_atb_transport).exists():
        logger.info(f"Downloading ATB transport costs from '{xlsx}'")
        progress_retrieve(xlsx, save_atb_transport)
        # urllib.request.urlretrieve(xlsx, save_atb_transport)
        # requests.get(xlsx, save_atb_transport)
    """
