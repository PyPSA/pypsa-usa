import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import constants
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from _helpers import configure_logging
from summary_natural_gas import (
    get_gas_demand,
    get_gas_processing,
    get_imports_exports,
    get_linepack,
    get_underground_storage,
)

logger = logging.getLogger(__name__)


MWH_2_MMCF = constants.NG_MWH_2_MMCF

FIG_HEIGHT = 500


@dataclass
class PlottingData:
    name: str
    getter: callable
    plotter: callable
    nice_name: Optional[str] = None
    unit: Optional[str] = None
    converter: Optional[float] = None


def _group_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.T.groupby(level=0).sum().T


def _sum_state_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Sums state data together.
    """

    dfs = [y for _, y in data.items()]
    return pd.concat(dfs, axis=1)


def plot_gas(
    data: pd.DataFrame,
    title: str,
    units: str,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    General gas plotting function.
    """

    df = data.copy()

    periods = data.index.get_level_values("period").unique()

    n_rows = len(periods)

    fig, axs = plt.subplots(n_rows, 1)

    for i, period in enumerate(periods):
        period_data = df[df.index.get_level_values("period") == period].droplevel(
            "period",
        )
        if n_rows > 1:
            ax = axs[i]
        else:
            ax = axs
        period_data.plot(
            kind="line",
            ax=ax,
            title=title,
            xlabel="",
            ylabel=f"({units})",
        )

    return (fig, axs)


def plot_gas_trade(data: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """
    General gas trade plotting function.
    """
    pass


PLOTTING_META = [
    {
        "name": "demand",
        "nice_name": "Natural Gas Demand",
        "unit": "MMCF",
        "converter": MWH_2_MMCF,
        "getter": get_gas_demand,
        "plotter": plot_gas,
    },
    {
        "name": "processing",
        "nice_name": "Natural Gas Processed",
        "unit": "MMCF",
        "converter": MWH_2_MMCF,
        "getter": get_gas_processing,
        "plotter": plot_gas,
    },
    {
        "name": "linepack",
        "nice_name": "Natural Gas in Linepack",
        "unit": "MMCF",
        "converter": MWH_2_MMCF,
        "getter": get_linepack,
        "plotter": plot_gas,
    },
    {
        "name": "storage",
        "nice_name": "Natural Gas in Underground Storage",
        "unit": "MMCF",
        "converter": MWH_2_MMCF,
        "getter": get_underground_storage,
        "plotter": plot_gas,
    },
    {
        "name": "domestic_trade",
        "nice_name": "Natural Gas Traded Domestically",
        "unit": "MMCF",
        "converter": MWH_2_MMCF,
        "getter": partial(get_imports_exports, international=False),
        "plotter": plot_gas_trade,
    },
    {
        "name": "international_trade",
        "nice_name": "Natural Gas Traded Internationally",
        "unit": "MMCF",
        "converter": MWH_2_MMCF,
        "getter": partial(get_imports_exports, international=False),
        "plotter": plot_gas_trade,
    },
]

if __name__ == "__main__":

    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_natural_gas",
            simpl="12",
            opts="48SEG",
            clusters="6",
            ll="v1.0",
            sector_opts="",
            sector="E-G",
            planning_horizons="2030",
            interconnect="western",
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)

    output_files = snakemake.output

    states = n.buses[n.buses.reeds_state != ""].reeds_state.unique().tolist()
    states += ["system"]

    plotting_metadata = [PlottingData(**x) for x in PLOTTING_META]

    # hack to only read in the network once, but get images to all states independently
    # ie. "interconnect}/figures/s{{simpl}}_c{{clusters}}/l{{ll}}_{{opts}}_{{sector}}/system/natural_gas/%s.png"

    # {result_name: {state: save_path.png}}
    expected_figures = {}
    for output_file in output_files:
        p = Path(output_file)
        root_path = list(p.parts[1:-3])  # path up to the 'system/natural_gas/%s.png'
        figure_name = list(p.parts[-2:])  # path of 'natural_gas/%s.png'
        result = p.stem  # ie. 'demand'
        state_paths = {}
        for state in states:
            full_path = root_path + [state] + figure_name
            full_path = Path("/".join(full_path))
            state_paths[state] = full_path
        expected_figures[result] = state_paths

    for meta in plotting_metadata:

        if meta.name not in expected_figures:
            logger.warning(f"Not expecting {meta.name} natural gas chart")
            continue

        data = meta.getter(n)

        for state in states:

            if state == "system":
                state_data = _sum_state_data(data)
            else:
                state_data = data[state]

            state_data = _group_data(state_data).mul(meta.converter)  # convert units

            title = f"{state} {meta.nice_name}"
            units = meta.unit

            fig, axs = meta.plotter(state_data, title=title, units=units)
            save_path = expected_figures[meta.name][state]
            fig.tight_layout()
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path))
