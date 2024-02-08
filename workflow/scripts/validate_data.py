import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)
from _helpers import configure_logging

sns.set_theme("paper", style="whitegrid")

rename_op = {'CCGT':'Natural gas', 'hydro':'Hydro', 'oil':'Oil', 'onwind':'Onshore wind', 'solar':'Solar', 'nuclear':'Nuclear','coal':'Coal', 'geothermal':'Other'}
selected_cols = ["Balancing Authority", 
                 "UTC Time at End of Hour",  
                "Net Generation (MW) from Natural Gas (Adjusted)",
                "Net Generation (MW) from Coal (Adjusted)",
                "Net Generation (MW) from Nuclear (Adjusted)",
                "Net Generation (MW) from All Petroleum Products (Adjusted)",
                "Net Generation (MW) from Hydropower and Pumped Storage (Adjusted)",
                "Net Generation (MW) from Solar (Adjusted)",
                "Net Generation (MW) from Wind (Adjusted)",
                "Net Generation (MW) from Other Fuel Sources (Adjusted)"]
rename_his = {'Net Generation (MW) from Natural Gas (Adjusted)':'Natural gas', 
            'Net Generation (MW) from Hydropower and Pumped Storage (Adjusted)':'Hydro', 
            'Net Generation (MW) from All Petroleum Products (Adjusted)':'Oil', 
            'Net Generation (MW) from Wind (Adjusted)':'Onshore wind', 
            'Net Generation (MW) from Solar (Adjusted)':'Solar', 
            'Net Generation (MW) from Nuclear (Adjusted)':'Nuclear',
            'Net Generation (MW) from Coal (Adjusted)':'Coal', 
            'Net Generation (MW) from Other Fuel Sources (Adjusted)':'Other'}
colors = ['purple','dimgray','brown','royalblue','chocolate','green','lightskyblue','crimson']
kwargs = dict(color=colors,legend=False, ylabel="Production [GW]", xlabel="")


def plot_graphs(n, csv_path_1, csv_path_2, save1, save2, save3):
    #plot a stacked plot for seasonal production
    #snapshot: January 2 - December 30 (inclusive)
    buses = get_buses(n)
    order = historic_df(csv_path_1, csv_path_2, buses)[1]
    optimized = optimized_df(n, order)
    historic = historic_df(csv_path_1, csv_path_2, buses)[0]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    optimized.resample('1D').sum().plot.area(ax=axes[0], **kwargs, title="Optimized")
    historic.resample('1D').sum().plot.area(ax=axes[1], **kwargs, title="Historic")

    diff = (optimized - historic).fillna(0).resample('1D').sum()
    diff.clip(lower=0).plot.area(
        ax=axes[2], **kwargs, title="$\Delta$ (Optimized - Historic)"
    )
    lim = axes[2].get_ylim()[1]
    diff.clip(upper=0).plot.area(ax=axes[2], **kwargs)
    axes[2].set_ylim(bottom=-lim, top=lim)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h[::-1],
        l[::-1],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        frameon=False,
        labelspacing=1,
    )
    fig.savefig(save1)

    # plot by carrier
    data = pd.concat([historic, optimized], keys=["Historic", "Optimized"], axis=1)
    data.columns.names = ["Kind", "Carrier"]
    fig, ax = plt.subplots(figsize=(6, 6))
    df = data.groupby(level=["Kind", "Carrier"], axis=1).sum().sum().unstack().T
    df = df / 1e3  # convert to TWh
    df.plot.barh(ax=ax, xlabel="Electricity Production [TWh]", ylabel="")
    ax.set_title("Electricity Production by Carriers")
    ax.grid(axis="y")
    fig.savefig(save2)

    # plot strongest deviations for each carrier
    fig, ax = plt.subplots(figsize=(6, 10))
    diff = (optimized - historic).sum()/1e3   # convert to TW
    diff = diff.dropna().sort_values()
    diff.plot.barh(
    xlabel="Optimized Production - Historic Production [TWh]", ax=ax
    )
    ax.set_title("Strongest Deviations")
    ax.grid(axis="y")
    fig.savefig(save3)

def optimized_df(n, order):
    """
    Create a DataFrame from the model output/optimized
    """
    # Drop Arizona since there is no Arizona balancing authority in the historical data
    #columns_to_drop = [col for col in n.generators_t["p"].columns if 'Arizona' in col]
    ba_carrier = n.generators_t["p"] #.drop(columns_to_drop, axis=1)
    optimized = ba_carrier.groupby(axis="columns", by=n.generators["carrier"]).sum().loc["2019-01-02 00:00:00":"2019-12-30 23:00:00"]
    # Combine CCGT and OCGT 
    optimized['CCGT'] = optimized['CCGT'] + optimized['OCGT']
    optimized_comb = optimized.drop(['OCGT'], axis=1)
    # Rename and rearrange the columns
    optimized = optimized_comb.rename(columns=rename_op)
    optimized = optimized.reindex(order, axis=1, level=1)
    # Convert to GW
    optimized = optimized / 1e3
    return optimized

def historic_df(csv_path_1, csv_path_2, buses):
    """
    Create a DataFrame from the csv files containing historical data
    """
    historic_first = pd.read_csv(csv_path_1,
        index_col = [0,1],
        header = 0,
        parse_dates=True,
        date_format="%m/%d/%Y %I:%M:%S %p",
        usecols = selected_cols)
    historic_second = pd.read_csv(csv_path_2,
        index_col = [0,1],
        header = 0,
        parse_dates=True,
        date_format="%m/%d/%Y %I:%M:%S %p",
        usecols = selected_cols)
    # Clean the data read from csv
    historic_first_df = historic_first.loc[buses].fillna(0).replace({',': ''}, regex=True).astype(float)
    historic_second_df = historic_second.loc[buses].fillna(0).replace({',': ''}, regex=True).astype(float)
    historic = pd.concat([historic_first_df, historic_second_df], axis=0).groupby(['UTC Time at End of Hour']).sum()
    # Consider negative values as 0 for stackplot purposes
    historic[historic < 0] = 0
    historic = historic.rename(columns=rename_his)
    order = (
        (historic.diff().abs().sum() / historic.sum()).sort_values().index
    )
    historic = historic.reindex(order, axis=1, level=1)
    historic = historic.loc["2019-01-02 00:00:00":"2019-12-30 23:00:00"] / 1e3
    return historic, order

def get_buses(n):
    buses = []
    for i in range(n.generators.bus.size):
        if n.generators.bus[i] not in buses:
            buses.append(n.generators.bus[i])
    buses_clean = [ba.split('0')[0] for ba in buses]
    buses_clean = [ba.split('-')[0] for ba in buses_clean]
    buses = list(OrderedDict.fromkeys(buses_clean))
    buses.pop(1)
    return buses

if __name__ == "__main__":  
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            'plot_validation_figures', 
            interconnect='western',
            clusters=30,
            ll='v1.25',
            opts='Co2L0.75',
        )
    configure_logging(snakemake)
    n = pypsa.Network(snakemake.input.network)
    csv_path_1 = snakemake.input.historic_first
    csv_path_2 = snakemake.input.historic_second
    plot_graphs(n, csv_path_1, csv_path_2, snakemake.output["seasonal_stacked_plot"], snakemake.output["carrier_production_bar"], snakemake.output["production_deviation_bar"])