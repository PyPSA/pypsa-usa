from pathlib import Path

import constants
import pandas as pd
import plotly.express as px
import pypsa
from _helpers import configure_logging
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


def make_plot_per_feature(df: pd.DataFrame, **kwargs) -> list[px.line]:

    title = kwargs.get("title", "")
    label = kwargs.get("label", {})

    figs = []

    for period, investment_period in enumerate(df.index.get_level_values(0).unique()):
        _title = title if period == 0 else ""
        figs.append(
            px.line(
                df[df.index.get_level_values("period") == investment_period].droplevel(
                    0,
                ),
                title=_title,
                labels=label,
                height=FIG_HEIGHT,
            ),
        )

    return figs


def make_plot_per_period(
    data: dict[str, pd.DataFrame],
    investment_period: int,
    **kwargs,
) -> list[px.line]:

    figs = []
    titles = kwargs.get("titles", {})
    labels = kwargs.get("labels", {})

    for feature, df in data.items():
        figs.append(
            px.line(
                df[df.index.get_level_values("period") == investment_period].droplevel(
                    0,
                ),
                title=titles[feature],
                labels=labels[feature],
                height=FIG_HEIGHT,
            ),
        )

    return figs


def main(n: pypsa.Network, write_files: dict[str, str]):
    """
    Creates interactive plots of key natural gas attributes.
    """

    data = {
        "demand": get_gas_demand(n) / MWH_2_MMCF,
        "processing": get_gas_processing(n) / MWH_2_MMCF,
        "linepack": get_linepack(n) / MWH_2_MMCF,
        "storage": get_underground_storage(n) / MWH_2_MMCF,
        "domestic_trade": get_imports_exports(n, international=False) / MWH_2_MMCF,
        "international_trade": get_imports_exports(n, international=True) / MWH_2_MMCF,
    }

    titles = {
        "demand": "Gas Demand",
        "processing": "Gas Processing Capacity",
        "linepack": "Gas Line Pack",
        "storage": "Gas Underground Storage",
        "domestic_trade": "Gas Trade Domestic",
        "international_trade": "Gas Trade International",
    }

    labels = {
        "demand": {"timestep": "", "value": "Demand (MMCF)", "carrier": "Source"},
        "processing": {
            "timestep": "",
            "value": "Processing Capacity (MMCF)",
            "Generator": "State",
        },
        "linepack": {
            "timestep": "",
            "value": "Linepack Capacity (MMCF)",
            "Store": "State",
        },
        "storage": {
            "timestep": "",
            "value": "Storage Capacity (MMCF)",
            "Store": "State",
        },
        "domestic_trade": {
            "timestep": "",
            "value": "Volume (MMCF)",
            "variable": "Direction",
        },
        "international_trade": {
            "timestep": "",
            "value": "Volume (MMCF)",
            "variable": "Direction",
        },
    }

    for feature, save_path in write_files.items():

        df = data[feature]
        title = titles[feature]
        label = labels[feature]

        figs = make_plot_per_feature(df, title=title, labels=label)

        figures_to_html(figs, save_path)

    # not tracked by snakemake cause accessing investment year in config is awk
    save_dir = Path(next(iter(write_files.values()))).parent

    for investment_period in n.investment_periods:

        figs = make_plot_per_period(
            data,
            investment_period,
            titles=titles,
            labels=labels,
        )

        save_file = Path(save_dir, f"natural_gas_{investment_period}.html")

        figures_to_html(figs, str(save_file))


if __name__ == "__main__":

    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_natural_gas",
            interconnect="texas",
            clusters=20,
            ll="v1.0",
            opts="500SEG",
            sector="E-G",
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)

    output_files = snakemake.output

    # {feature: save_file}
    # ie. {"demand": results/gas/natural_gas_demand.html}
    figures = {
        x.split("natural_gas_")[1].split(".")[0]: y for x, y in output_files.items()
    }

    assert figures, "No natural gas figures to create"

    main(n, figures)
