"""Script used to compare outputs from multiple snakemake scenarios."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import yaml
from matplotlib import pyplot as plt


def get_carriers(n):
    carriers = n.carriers
    carriers["legend_name"] = carriers.nice_name
    carriers.loc["DC", "legend_name"] = "Transmission"
    carriers.loc["DC", "color"] = "#cf1dab"
    carriers.loc["battery", "legend_name"] = "Existing BESS"
    carriers = carriers.set_index("nice_name")
    carriers = carriers.sort_values(by="co2_emissions", ascending=False)
    return carriers


# Load scenarios from YAML configuration
def load_yaml_config(yaml_path):
    with open(yaml_path) as file:
        return yaml.safe_load(file)


# Load CSV data for all scenarios
def load_scenario_data(scenarios):
    data = {}
    for scenario in scenarios:
        scenario_name = scenario["name"]
        path = Path(scenario["path"])
        data[scenario_name] = {
            file.stem: pd.read_csv(file, index_col=[0, 1], header=[0, 1])
            for file in path.glob("statistics/statistics*.csv")
        }
    return data


# Process data to match the expected format
def process_data(data, alias_dict=None, new_order=None):
    stats = {}
    for scenario_name, files in data.items():
        stats[scenario_name] = files  # Placeholder for specific transformations

    if alias_dict:
        stats_with_alias = {}
        for scenario_name, df in stats.items():
            alias_name = alias_dict.get(scenario_name, scenario_name)
            stats_with_alias[alias_name] = df
        stats = stats_with_alias

    if new_order:
        stats = {key: stats[key] for key in new_order if key in stats}

    return stats


def prepare_combined_dataframe(
    stats,
    variable,
    carriers,
    include_link=False,
    as_pct=False,
    variable_units=None,
):
    factor_units = {"GW": 1e3, "GWh": 1e3, "%": 1}.get(variable_units, 1e9)

    data = []
    for scenario, df in stats.items():
        df = df["statistics"].fillna(0)

        tech_filter = ["Generator", "StorageUnit", "Link"]

        df = df.loc[df.index.get_level_values(0).isin(tech_filter), variable]
        df.index = df.index.get_level_values(1)
        df = df.reindex(carriers.index).dropna()

        if as_pct:
            df = ((df / df.sum()) * 100).round(2)

        for horizon in df.columns:
            df_horizon = df[horizon] / factor_units
            df_horizon = df_horizon.reset_index()
            df_horizon["Scenario"] = scenario
            df_horizon["horizon"] = horizon
            data.append(df_horizon.rename(columns={horizon: "statistics"}))

    combined_df = pd.concat(data, ignore_index=True)
    combined_df["scenario_name"] = combined_df["Scenario"].apply(
        lambda x: x.split("_")[0],
    )
    combined_df["trans_expansion"] = combined_df["Scenario"].apply(
        lambda x: x.split("_")[1],
    )
    combined_df.to_csv(figures_path / f"{variable}_comparison.csv")
    return combined_df


def plot_scenario_comparison(
    combined_df,
    carriers,
    variable,
    variable_units,
    title,
    figures_path,
    colors,
    include_link=False,
    reference_scenario=None,
):
    planning_horizons = combined_df["horizon"].unique()
    scenarios = combined_df["Scenario"].unique()
    if not include_link:
        combined_df = combined_df[~combined_df["nice_name"].isin(["Link", "Ac"])]

    fig, axes = plt.subplots(
        nrows=len(planning_horizons),
        ncols=1,
        figsize=(8, 1.2 * len(planning_horizons) + 0.2 * len(scenarios)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)  # Ensure axes is iterable for single horizon

    for ax, horizon in zip(axes, planning_horizons):
        horizon_df = combined_df[combined_df["horizon"] == horizon]
        y_positions = np.arange(len(scenarios))

        for j, scenario in enumerate(scenarios):
            scenario_df = horizon_df[horizon_df["Scenario"] == scenario]
            bottoms = np.zeros(len(y_positions))
            for tech in scenario_df["nice_name"].unique():
                values = scenario_df[scenario_df["nice_name"] == tech]["statistics"].values[0]
                ax.barh(
                    y_positions[j],
                    values,
                    left=bottoms[j],
                    color=colors[tech],
                    label=tech if j == 0 else "",
                )
                bottoms[j] += values

        ax.text(
            1.01,
            0.5,
            horizon,
            transform=ax.transAxes,
            va="center",
            rotation="vertical",
        )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(scenarios)
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.xlabel(f"{variable} [{variable_units}]")
    plt.subplots_adjust(hspace=0.5)

    carriers_plotted = carriers.loc[carriers.index.intersection(combined_df["nice_name"].unique())]
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[tech]) for tech in carriers_plotted.index]
    fig.legend(
        handles=legend_handles,
        labels=carriers_plotted.legend_name.tolist(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.4),
        ncol=4,
        title="Technologies",
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        figures_path / f"{variable}_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    if reference_scenario:
        _plot_reference_comparison(
            combined_df,
            reference_scenario,
            carriers,
            colors,
            figures_path,
            variable,
            horizon,
        )

    return


def _plot_reference_comparison(
    combined_df,
    reference_scenario,
    carriers,
    colors,
    figures_path,
    variable,
    horizon,
):
    combined_df = combined_df.set_index("Scenario")
    combined_df = combined_df.loc[combined_df.horizon == horizon]
    ref = combined_df.loc[reference_scenario].set_index("nice_name")
    combined_df = combined_df.reset_index().set_index("nice_name")
    for carrier in combined_df.index.unique():
        combined_df.loc[carrier, "statistics"] = (
            (combined_df.loc[carrier, "statistics"] - ref.loc[carrier, "statistics"]) / ref.statistics.sum() * 100
        )
    combined_df = combined_df.reset_index().set_index("Scenario")
    stacked_data = combined_df.reset_index().pivot(
        index="Scenario",
        columns="nice_name",
        values="statistics",
    )
    stacked_data.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 7),
        color=[colors[tech] for tech in stacked_data.columns],
        legend=False,
    )
    plt.ylabel("∆ Capacity[%]")
    plt.savefig(
        figures_path / f"{variable}_pct_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    combined_df.to_csv(figures_path / f"{variable}_pct_comparison.csv")

    # combined_df = combined_df.set_index(["Scenario", "horizon"])
    # ref = combined_df.loc[(reference_scenario, slice(None))]
    # combined_df = combined_df.reset_index()

    # for horizon in ref.index.get_level_values(0).unique():
    #     ref_stats = ref.loc[tech]["statistics"]
    #     combined_df["pct_diff"] = combined_df["statistics"] / ref_stats.sum() * 100 - 100

    # pivoted_data = combined_df.pivot(index="Scenario", columns="nice_name", values="pct_diff")
    # pivoted_data.plot(
    #     kind="bar",
    #     stacked=True,
    #     figsize=(10, 7),
    #     color=[colors[tech] for tech in pivoted_data.columns],
    #     legend=False,
    # )
    # plt.ylabel("∆ Capacity [%]")
    # plt.savefig(figures_path / f"{variable}_pct_comparison.png", dpi=300, bbox_inches="tight")


# Plot comparison
def scenario_comparison(
    stats,
    variable,
    variable_units,
    carriers,
    title,
    figures_path,
    include_link=False,
    as_pct=False,
    reference_scenario=None,
):
    combined_df = pd.DataFrame(
        columns=["Scenario", "horizon", "nice_name", "statistics"],
        index=[],
    )
    colors = carriers["color"]
    planning_horizons = stats[next(iter(stats.keys()))]["statistics"][variable].columns
    fig, axes = plt.subplots(
        nrows=len(planning_horizons),
        ncols=1,
        figsize=(8, 1.5 * len(planning_horizons) + 0.2 * len(stats)),
        sharex=True,
    )
    if variable_units in ["GW", "GWh"]:
        factor_units = 1e3
    elif variable_units in ["%"]:
        factor_units = 1
    else:
        factor_units = 1e9

    if len(planning_horizons) == 1:
        axes = [axes]

    for ax, horizon in zip(axes, planning_horizons):
        y_positions = np.arange(len(stats))
        for j, (scenario, df) in enumerate(stats.items()):
            df = df["statistics"].fillna(0)
            bottoms = np.zeros(len(y_positions))
            if include_link:
                df = df.loc[
                    df.index.get_level_values(0).isin(
                        ["Generator", "StorageUnit", "Link"],
                    ),
                    variable,
                ]
                df = df.loc[~(df.index.get_level_values(1) == "Ac")]
            else:
                df = df.loc[
                    df.index.get_level_values(0).isin(["Generator", "StorageUnit"]),
                    variable,
                ]
            df.index = df.index.get_level_values(1)
            df = df.reindex(carriers.index).dropna()
            if as_pct:
                df = ((df / df.sum()) * 100).round(2)
            for i, tech in enumerate(df.index.unique()):
                values = df.loc[tech, horizon] / factor_units
                ax.barh(
                    y_positions[j],
                    values,
                    left=bottoms[j],
                    color=colors[tech],
                    label=tech if j == 0 else "",
                )
                bottoms[j] += values

            df[["Scenario", "horizon"]] = scenario, horizon
            df = df.reset_index()
            df = df.rename(columns={horizon: "statistics"})
            combined_df = pd.concat(
                [combined_df, df[["Scenario", "nice_name", "statistics", "horizon"]]],
            )

        ax.text(
            1.01,
            0.5,
            f"{horizon}",
            transform=ax.transAxes,
            va="center",
            rotation="vertical",
        )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(stats.keys())
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.xlabel(f"{variable} [{variable_units}]")
    plt.subplots_adjust(hspace=0)
    carriers_plotted = carriers.loc[carriers.index.intersection(df.index.unique())]
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[tech]) for tech in carriers_plotted.index]
    fig.legend(
        handles=legend_handles,
        labels=carriers_plotted.legend_name.tolist(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.4),
        ncol=4,
        title="Technologies",
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        figures_path / f"{variable}_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    combined_df["scenario_name"] = combined_df["Scenario"].apply(
        lambda x: x.split("_")[0],
    )
    combined_df["trans_expansion"] = combined_df["Scenario"].apply(
        lambda x: x.split("_")[1],
    )
    combined_df.to_csv(figures_path / f"{variable}_comparison.csv")

    if reference_scenario:
        combined_df = combined_df.set_index("Scenario")
        combined_df = combined_df.query("horizon == @horizon").drop(
            columns="horizon",
        )  # only plot last horizon
        ref = combined_df.loc[reference_scenario].set_index("nice_name")
        combined_df = combined_df.reset_index().set_index("nice_name")
        for scenario in combined_df.index.unique():
            combined_df.loc[scenario, "statistics"] = (
                (combined_df.loc[scenario, "statistics"] - ref.loc[scenario, "statistics"]) / ref.statistics.sum() * 100
            )
        combined_df = combined_df.reset_index().set_index("Scenario")
        stacked_data = combined_df.reset_index().pivot(
            index="Scenario",
            columns="nice_name",
            values="statistics",
        )
        stacked_data.plot(
            kind="bar",
            stacked=True,
            figsize=(10, 7),
            color=[colors[tech] for tech in stacked_data.columns],
            legend=False,
        )
        plt.ylabel("∆ Capacity[%]")
        plt.savefig(
            figures_path / f"{variable}_pct_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        combined_df.to_csv(figures_path / f"{variable}_pct_comparison.csv")
    return combined_df


def plot_cost_comparison(
    stats,
    n,
    variable,
    variable_units,
    title,
    figures_path,
    reference_scenario=None,
):
    combined_df = pd.DataFrame(columns=["Scenario", "statistics"], index=[])
    for j, (scenario, stat) in enumerate(stats.items()):
        stat = stat["statistics"]
        combined_df = pd.concat(
            [
                combined_df,
                pd.DataFrame(
                    {
                        "Scenario": scenario,
                        "statistics": (
                            (stat["Capital Expenditure"].sum() + stat["Operational Expenditure"].sum())
                            * n.investment_period_weightings.objective.values
                        ).sum()
                        / 1e9,
                    },
                    index=[j],
                ),
            ],
        )

    combined_df.plot(
        kind="bar",
        x="Scenario",
        y="statistics",
        title="Total System Costs",
        legend=False,
    )
    plt.ylabel("Annualized System Costs [B$]")
    plt.savefig(
        figures_path / f"{variable}_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    if reference_scenario:
        combined_df = combined_df.set_index("Scenario")
        ref = combined_df.loc[reference_scenario]
        pct_df = (combined_df - ref) / combined_df * 100
        pct_df.plot(
            kind="bar",
            y="statistics",
            title="Total System Costs",
            legend=False,
        )
        plt.ylabel("∆ Annualized System Costs [%]")
        plt.savefig(
            figures_path / f"{variable}_pct_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        pct_df.to_csv(figures_path / f"{variable}_pct_comparison.csv")
        combined_df.to_csv(figures_path / f"{variable}_comparison.csv")


# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run scenario comparison script.")
    parser.add_argument(
        "yaml_name",
        type=str,
        help="Name of the YAML configuration file.",
    )
    args = parser.parse_args()

    yaml_name = args.yaml_name  # Name of the YAML file from command line argument
    yaml_path = Path.cwd() / yaml_name  # Path to the YAML file

    # Load and process data
    config = load_yaml_config(yaml_path)
    scenarios = config["scenarios"]
    raw_data = load_scenario_data(scenarios)

    alias_dict = config.get("alias_dict", None)
    new_order = config.get("new_order", None)
    reference_scenario = config.get("reference_scenario", None)

    figures_path = (
        Path.cwd() / f"results/{config.get('output_folder_name', 'scenario_comparison')}"
    )  # Directory to save the figures in the parent of cwd

    figures_path.mkdir(exist_ok=True)

    processed_data = process_data(raw_data, alias_dict, new_order)

    n = pypsa.Network(config["network"]["path"])
    # Example carrier setup
    carriers = get_carriers(n)
    carriers.to_csv(figures_path / "carriers.csv")

    # Example variable and title
    variable = "Optimal Capacity"
    variable_units = "GW"
    title = "Capacity Comparison"

    # Generate plots
    combined_df = prepare_combined_dataframe(
        processed_data,
        variable,
        carriers,
        as_pct=False,
        variable_units=variable_units,
    )
    plot_scenario_comparison(
        combined_df,
        carriers,
        variable,
        variable_units,
        title,
        figures_path,
        colors=carriers["color"],
        reference_scenario=reference_scenario,
    )

    # Example variable and title
    variable = "Supply"
    variable_units = "%"
    title = "Supply Comparison"

    # Generate plots
    scenario_comparison(
        processed_data,
        variable,
        variable_units,
        carriers,
        title,
        figures_path,
        as_pct=True,
    )

    # Example variable and title
    variable = "Capital Expenditure"
    variable_units = "$B"
    title = "CAPEX Comparison"

    # Generate plots
    scenario_comparison(
        processed_data,
        variable,
        variable_units,
        carriers,
        title,
        figures_path,
        as_pct=False,
    )

    # Example variable and title
    variable = "Operational Expenditure"
    variable_units = "$B"
    title = "OPEX Comparison"

    # Generate plots
    scenario_comparison(
        processed_data,
        variable,
        variable_units,
        carriers,
        title,
        figures_path,
        as_pct=False,
    )

    # Example variable and title
    variable = "System Costs"
    variable_units = "$B"
    title = "Scenario Comparison"
    plot_cost_comparison(
        processed_data,
        n,
        variable,
        variable_units,
        title,
        figures_path,
        reference_scenario,
    )
