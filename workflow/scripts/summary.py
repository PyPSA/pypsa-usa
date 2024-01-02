"""Calcualtes summary files

Adapted from PyPSA-Eur summary statistics reporting script 
 - https://github.com/PyPSA/pypsa-eur/blob/master/scripts/make_summary.py
"""

import pypsa 
import pandas as pd
from _helpers import configure_logging


def get_energy_total(n: pypsa.Network):
    """Gets energy production totals"""

    def _get_energy_one_port(n: pypsa.Network, c: str) -> pd.DataFrame:
            return (
                c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0)
                .sum()
                .multiply(c.df.sign)
                .groupby(c.df.carrier)
                .sum()
            )
            
    def _get_energy_multi_port(n: pypsa.Network, c: str) -> pd.DataFrame:
        c_energies = pd.Series(0.0, c.df.carrier.unique())
        for port in [col[3:] for col in c.df.columns if col[:3] == "bus"]:
            totals = (
                c.pnl["p" + port]
                .multiply(n.snapshot_weightings.generators, axis=0)
                .sum()
            )
            # remove values where bus is missing (bug in nomopyomo)
            no_bus = c.df.index[c.df["bus" + port] == ""]
            totals.loc[no_bus] = float(
                n.component_attrs[c.name].loc["p" + port, "default"]
            )
            c_energies -= totals.groupby(c.df.carrier).sum()
        return c_energies
    
    energy = []
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        if c.name in ("Generator", "StorageUnit", "Store"):
            e = _get_energy_one_port(n, c)
        elif c.name in ("Link"):
            e = _get_energy_multi_port(n, c)
        else:
            continue
        energy.append(e)
        
    return energy

def get_energy_timeseries(n: pypsa.Network):
    """Gets timeseries energy production"""

    def _get_energy_one_port(n: pypsa.Network, c: str) -> pd.DataFrame:
            return (
                c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0)
                .T
                .sum()
                .T
                .multiply(c.df.sign)
                .groupby(c.df.carrier)
                .sum()
            )
            
    def _get_energy_multi_port(n: pypsa.Network, c: str) -> pd.DataFrame:
        c_energies = pd.Series(0.0, c.df.carrier.unique())
        for port in [col[3:] for col in c.df.columns if col[:3] == "bus"]:
            totals = (
                c.pnl["p" + port]
                .multiply(n.snapshot_weightings.generators, axis=0)
                .T
                .sum()
                .T
            )
            # remove values where bus is missing (bug in nomopyomo)
            no_bus = c.df.index[c.df["bus" + port] == ""]
            totals.loc[no_bus] = float(
                n.component_attrs[c.name].loc["p" + port, "default"]
            )
            c_energies -= totals.groupby(c.df.carrier).sum()
        return c_energies
    
    energy = []
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        if c.name in ("Generator", "StorageUnit", "Store"):
            e = _get_energy_one_port(n, c)
        elif c.name in ("Link"):
            e = _get_energy_multi_port(n, c)
        else:
            continue
        energy.append(e)
        
    return energy

def get_demand_timeseries(n: pypsa.Network):
    """Gets timeseries energy demand"""
    return pd.DataFrame(n.loads_t.p.sum(1).mul(1e-3)).rename(columns={0:"Demand"})

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            'plot_figures', 
            interconnect='texas',
            clusters=40,
            ll='v1.25',
            opts='Co2L1.25',
            sector="E-G"
        )
    configure_logging(snakemake)
    
    n = pypsa.Network(snakemake.input.network)
    get_energy_total(n)
    get_energy_timeseries(n)
    
    