import constants
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pypsa
from _helpers import configure_logging
from plotly.subplots import make_subplots
from summary_natural_gas import (
    get_gas_demand,
    get_gas_processing,
    get_imports_exports,
    get_linepack,
    get_underground_storage,
)

MWH_2_MMCF = constants.NG_MWH_2_MMCF

FIG_HEIGHT = 500


def figures_to_html(figs, filename):
    """
    Gets around subplots with plotly express issue.

    https://stackoverflow.com/a/58336718
    """
    with open(filename, "w") as html:
        html.write("<html><head></head><body>" + "\n")
        for fig in figs:
            inner_html = fig.to_html().split("<body>")[1].split("</body>")[0]
            html.write(inner_html)
        html.write("</body></html>" + "\n")


def main(n: pypsa.Network, write_file: str):
    """
    Creates interactive plots of key natural gas attributes.
    """

    figures = [
        px.line(
            get_gas_demand(n) / MWH_2_MMCF,
            title="Gas Demand",
            labels={"snapshot": "", "value": "Demand (MMCF)", "carrier": "Source"},
            height=FIG_HEIGHT,
        ),
        px.line(
            get_gas_processing(n) / MWH_2_MMCF,
            title="Gas Processing Capacity",
            labels={
                "snapshot": "",
                "value": "Processing Capacity (MMCF)",
                "Generator": "State",
            },
            height=FIG_HEIGHT,
        ),
        px.line(
            get_linepack(n) / MWH_2_MMCF,
            title="Gas Line Pack",
            labels={
                "snapshot": "",
                "value": "Linepack Capacity (MMCF)",
                "Store": "State",
            },
            height=FIG_HEIGHT,
        ),
        px.line(
            get_underground_storage(n) / MWH_2_MMCF,
            title="Gas Underground Storage",
            labels={
                "snapshot": "",
                "value": "Storage Capacity (MMCF)",
                "Store": "State",
            },
            height=FIG_HEIGHT,
        ),
        px.line(
            get_imports_exports(n, international=False) / MWH_2_MMCF,
            title="Gas Trade Domestic",
            labels={"snapshot": "", "value": "Volume (MMCF)", "Variable": "State"},
            height=FIG_HEIGHT,
        ),
        px.line(
            get_imports_exports(n, international=True) / MWH_2_MMCF,
            title="Gas Trade International",
            labels={"snapshot": "", "value": "Volume (MMCF)", "variable": "State"},
            height=FIG_HEIGHT,
        ),
    ]

    figures_to_html(figures, write_file)


# Run the app
if __name__ == "__main__":

    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_natural_gas",
            interconnect="texas",
            clusters=40,
            ll="v1.25",
            opts="Co2L1.25",
            sector="E-G",
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    main(n, snakemake.output["natural_gas.html"])
