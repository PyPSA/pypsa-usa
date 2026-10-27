"""
Download the ComStock building characteristics and annual results of one state.

The equipment loads and efficiencies in these results convert the ComStock
final-energy load profiles into useful energy (see ``build_eulp_loads``). The
file comes from the same ComStock release as the profiles ``retrieve_eulp``
downloads.
"""

import logging
from pathlib import Path

from _helpers import configure_logging
from retrieve_eulp import OediDownload

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_comstock_metadata", state="CO")
    configure_logging(snakemake)

    save = Path(snakemake.output[0])
    save.parent.mkdir(parents=True, exist_ok=True)
    OediDownload("com").download_metadata(snakemake.wildcards.state, str(save))
