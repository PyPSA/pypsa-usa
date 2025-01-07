"""
This script facilitates comparison of scenarios run in PyPSA-USA. It loads data from multiple scenarios, processes it for analysis, and generates comparative plots for system metrics such as optimal capacities, supply, and costs. The script is designed for users working with PyPSA networks and scenario data stored in a specific YAML configuration format.

Usage:
------

1. **Setup Configuration File**:
    - Create a YAML configuration file containing details about the scenarios, aliases, order of scenarios, and network data path. An example structure for the configuration file:
      ```yaml
      scenarios:
        - name: "Scenario 1"
          path: "path/to/scenario1/"
        - name: "Scenario 2"
          path: "path/to/scenario2/"
      alias_dict:
        "Scenario 1": "S1"
        "Scenario 2": "S2"
      new_order:
        - "S1"
        - "S2"
      reference_scenario: "S1"
      output_folder_name: "folder_name"
      network:
        path: "path/to/network/file.nc"
      ```

2. **Prepare Directory Structure**:
    - Place the YAML configuration file in the parent directory of the current working directory under `config/scenario_comparison.yaml`.
    - Ensure the scenario data contains statistics files (`statistics/statistics*.csv`).

3. **Execution**:
    - Run the script from its directory using a Python environment with the required libraries (`yaml`, `pandas`, `pypsa`, `matplotlib`, `numpy`, `seaborn`).

4. **Outputs**:
    - Plots will be saved in the `results/{output_folder_name}` directory in the parent of the current working directory. These include:
      - Bar charts comparing system metrics like "Optimal Capacity" or "System Costs."
      - Comparative percentage charts, if a reference scenario is specified.

5. **Adjusting Variables**:
    - Modify variables such as `variable`, `variable_units`, and `title` in the script to customize the metrics and plot titles.

Functions:
----------
- `get_carriers(n)`: Processes the carrier data from the PyPSA network for plotting.
- `load_yaml_config(yaml_path)`: Loads the YAML configuration file.
- `load_scenario_data(scenarios)`: Reads scenario CSV data from paths specified in the configuration.
- `process_data(data, alias_dict=None, new_order=None)`: Processes raw scenario data, applying aliases and ordering.
- `scenario_comparison(...)`: Generates horizontal bar charts comparing scenario data for a specific variable.
- `plot_cost_comparison(...)`: Plots a cost comparison across scenarios and optionally a percentage comparison relative to a reference scenario.

Dependencies:
-------------
- Libraries: `yaml`, `pandas`, `pypsa`, `matplotlib`, `numpy`, `seaborn`.
- Ensure all dependencies are installed in your Python environment before running the script.

Example:
--------
Run the script to generate comparison plots for energy system scenarios defined in the configuration file:
```bash
python script_name.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import seaborn as sns
import yaml
from matplotlib import pyplot as plt


def get_carriers(n):
    carriers = n.carriers
    carriers["legend_name"] = carriers.nice_name
    carriers.loc["DC", "legend_name"] = "Transmission"
    carriers.loc["DC", "color"] = "#cf1dab"
    carriers.loc["battery", "legend_name"] = "Existing BESS"
    carriers.set_index("nice_name", inplace=True)
    carriers.sort_values(by="co2_emissions", ascending=False, inplace=True)
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
    combined_df = pd.DataFrame(columns=["Scenario", "horizon", "nice_name", "statistics"], index=[])
    colors = carriers["color"]
    planning_horizons = stats[next(iter(stats.keys()))]["statistics"][variable].columns
    fig, axes = plt.subplots(
        nrows=len(planning_horizons),
        ncols=1,
        figsize=(8, 4.2 * len(planning_horizons)),
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
            bottoms = np.zeros(len(df.columns))
            if include_link:
                df = df.loc[df.index.get_level_values(0).isin(["Generator", "StorageUnit", "Link"]), variable]
                df = df.loc[~(df.index.get_level_values(1) == "Ac")]
            else:
                df = df.loc[df.index.get_level_values(0).isin(["Generator", "StorageUnit"]), variable]
            df.index = df.index.get_level_values(1)
            df = df.reindex(carriers.index).dropna()
            if as_pct:
                df = ((df / df.sum()) * 100).round(2)
            for i, tech in enumerate(df.index.unique()):
                values = df.loc[tech, horizon] / factor_units
                ax.barh(y_positions[j], values, left=bottoms[j], color=colors[tech], label=tech if j == 0 else "")
                bottoms[j] += values

            df[["Scenario", "horizon"]] = scenario, horizon
            df = df.reset_index()
            df.rename(columns={horizon: "statistics"}, inplace=True)
            combined_df = pd.concat([combined_df, df[["Scenario", "nice_name", "statistics", "horizon"]]])

        ax.text(1.01, 0.5, f"{horizon}", transform=ax.transAxes, va="center", rotation="vertical")
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
    plt.savefig(figures_path / f"{variable}_comparison.png", dpi=300, bbox_inches="tight")

    if reference_scenario:
        combined_df = combined_df.set_index("Scenario")
        combined_df = combined_df.query("horizon == @horizon").drop(columns="horizon")  # only plot last horizon
        ref = combined_df.loc[reference_scenario].set_index("nice_name")
        combined_df = combined_df.reset_index().set_index("nice_name")
        for scenario in combined_df.index.unique():
            combined_df.loc[scenario, "statistics"] = (
                (combined_df.loc[scenario, "statistics"] - ref.loc[scenario, "statistics"]) / ref.sum().values * 100
            )
        combined_df = combined_df.reset_index().set_index("Scenario")
        stacked_data = combined_df.reset_index().pivot(index="Scenario", columns="nice_name", values="statistics")
        stacked_data.plot(
            kind="bar",
            stacked=True,
            figsize=(10, 7),
            color=[colors[tech] for tech in stacked_data.columns],
            legend=False,
        )
        plt.ylabel("∆ Capacity[%]")
        plt.savefig(figures_path / f"{variable}_pct_comparison.png", dpi=300, bbox_inches="tight")


def plot_cost_comparison(stats, n, variable, variable_units, title, figures_path, reference_scenario=None):
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

    combined_df.plot(kind="bar", x="Scenario", y="statistics", title="Total System Costs", legend=False)
    plt.ylabel("Annualized System Costs [B$]")
    plt.savefig(figures_path / f"{variable}_comparison.png", dpi=300, bbox_inches="tight")

    if reference_scenario:
        combined_df = combined_df.set_index("Scenario")
        ref = combined_df.loc[reference_scenario]
        pct_df = (combined_df - ref) / combined_df * 100
        pct_df.plot(kind="bar", y="statistics", title="Total System Costs", legend=False)
        plt.ylabel("∆ Annualized System Costs [%]")
        plt.savefig(figures_path / f"{variable}_pct_comparison.png", dpi=300, bbox_inches="tight")


# Main execution
if __name__ == "__main__":
    yaml_path = Path.cwd().parent / "config/scenario_comparison.yaml"  # Path to the YAML file

    # Load and process data
    config = load_yaml_config(yaml_path)
    scenarios = config["scenarios"]
    raw_data = load_scenario_data(scenarios)

    alias_dict = config.get("alias_dict", None)
    new_order = config.get("new_order", None)
    reference_scenario = config.get("reference_scenario", None)

    figures_path = (
        Path.cwd().parent / f"results/{config.get('output_folder_name', 'scenario_comparison')}"
    )  # Directory to save the figures in the parent of cwd

    figures_path.mkdir(exist_ok=True)

    processed_data = process_data(raw_data, alias_dict, new_order)

    n = pypsa.Network(config["network"]["path"])
    # Example carrier setup
    carriers = get_carriers(n)

    # Example variable and title
    variable = "Optimal Capacity"
    variable_units = "GW"
    title = "Capacity Evolution Comparison"

    # Generate plots
    scenario_comparison(
        processed_data,
        variable,
        variable_units,
        carriers,
        title,
        figures_path,
        reference_scenario=reference_scenario,
    )

    # Example variable and title
    variable = "Supply"
    variable_units = "%"
    title = "Supply Evolution Comparison"

    # Generate plots
    scenario_comparison(processed_data, variable, variable_units, carriers, title, figures_path, as_pct=True)

    # Example variable and title
    variable = "System Costs"
    variable_units = "$B"
    title = "Scenario Comparison"
    plot_cost_comparison(processed_data, n, variable, variable_units, title, figures_path, reference_scenario)
