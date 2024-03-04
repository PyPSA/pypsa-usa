import pandas as pd
import constants as const
from _helpers import mock_snakemake, configure_logging

"""
Todo- in build_fuel_prices create a pivoted dataframe where both BAs and States are in the columns. Rows are indexed to n.shapshots. 
"""

def prepare_eia(eia_fn: str, snapshots: pd.DatetimeIndex = None):
    """Cleans EIA fuel price data.

    returns:
    fuel_costs: pd.DataFrame
        Long format dataframe of state level, monthly fuel prices in units of $/MWh_thermal
    """
    fuel_prices = pd.read_csv(eia_fn)
    fuel_prices["dol_mwh_th"] = fuel_prices["value"] / const.NG_MCF_2_MWH
    
    fuel_prices['period'] = pd.to_datetime(fuel_prices.period, format="%Y-%m-%d")
    fuel_prices["month"] = fuel_prices['period'].dt.month
    fuel_prices.drop(columns=['series-description','period', 'units' , 'value'],inplace=True)


    year = snapshots[0].year
    fuel_prices['month'] = pd.to_datetime(fuel_prices['month'].astype(str) + '-' + str(year), format='%m-%Y').map(lambda dt: dt.replace(year=year))
    fuel_prices = fuel_prices.rename(columns={"month": "timestep"})
    fuel_prices = fuel_prices.pivot(index='timestep', columns="state", values='dol_mwh_th')
    fuel_prices = fuel_prices.reindex(snapshots)
    fuel_prices = fuel_prices.fillna(method='bfill').fillna(method='ffill')
    return fuel_prices


def prepare_caiso(caiso_fn: str, snapshots: pd.DatetimeIndex= None):
    caiso_ng = pd.read_csv(caiso_fn)
    caiso_ng["PRC"] = caiso_ng["PRC"] * const.NG_Dol_MMBTU_2_MWH
    caiso_ng.rename(columns={"PRC": "dol_mwh_th","Balancing Authority":"balancing_area"}, inplace=True)
    
    year = snapshots[0].year
    caiso_ng.day_of_year = pd.to_datetime(caiso_ng.day_of_year, format='%j').map(lambda dt: dt.replace(year=year))
    caiso_ng = caiso_ng.rename(columns={"day_of_year": "timestep"})
    caiso_ng = caiso_ng.pivot(index='timestep', columns="balancing_area", values='dol_mwh_th')
    caiso_ng = caiso_ng.reindex(snapshots)
    caiso_ng = caiso_ng.fillna(method='bfill').fillna(method='ffill')
    return caiso_ng


def main(snakemake):

    snapshot_config = snakemake.config["snapshots"]
    sns_start = pd.to_datetime(snapshot_config["start"])
    sns_end = pd.to_datetime(snapshot_config["end"])
    sns_inclusive = snapshot_config["inclusive"]

    snapshots= pd.date_range(
            freq="h",
            start=sns_start,
            end=sns_end,
            inclusive=sns_inclusive,
        )

    fuel_prices_caiso= prepare_caiso(snakemake.input.caiso_ng_prices, snapshots)

    fuel_prices_eia = prepare_eia(snakemake.input.eia_ng_prices, snapshots)

    fuel_prices = pd.concat([fuel_prices_caiso, fuel_prices_eia], axis=1)

    fuel_prices.to_csv(snakemake.output.ng_fuel_prices, index=False)        



if __name__ == '__main__':
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("build_fuel_prices", interconnect="western")
    configure_logging(snakemake)
    main(snakemake)