"""Calcualtes summary files

Adapted from PyPSA-Eur summary statistics reporting script 
 - https://github.com/PyPSA/pypsa-eur/blob/master/scripts/make_summary.py
"""

import pypsa 
import pandas as pd
from _helpers import configure_logging
import logging
logger = logging.getLogger(__name__)


### 
# ENERGY SUPLPY
###

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
        
    return pd.concat(energy, axis=1)

def get_energy_timeseries(n: pypsa.Network) -> pd.DataFrame:
    """Gets timeseries energy production"""

    def _get_energy_one_port(n: pypsa.Network, c: str) -> pd.DataFrame:
            return (
                c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0)
                .multiply(c.df.sign)
                .T
                .groupby(c.df.carrier)
                .sum()
                .T
            )
            
    def _get_energy_multi_port(n: pypsa.Network, c: str) -> pd.DataFrame:
        c_energies = pd.DataFrame(index=n.snapshots, columns=c.df.carrier.unique()).fillna(0)
        for port in [col[3:] for col in c.df.columns if col[:3] == "bus"]:
            totals = (
                c.pnl["p" + port]
                .multiply(n.snapshot_weightings.generators, axis=0)
            )
            # remove values where bus is missing (bug in nomopyomo)
            no_bus = c.df.index[c.df["bus" + port] == ""]
            totals.loc[no_bus] = float(
                n.component_attrs[c.name].loc["p" + port, "default"]
            )
            c_energies += totals.T.groupby(c.df.carrier).sum().T
        return c_energies
    
    energy = []
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        # if c.name in ("Generator", "StorageUnit", "Store"):
        if c.name in ("Generator", "StorageUnit"):
            e = _get_energy_one_port(n, c)
        elif c.name in ("Link"):
            e = _get_energy_multi_port(n, c)
        else:
            continue
        energy.append(e)
        
    return pd.concat(energy, axis=1)

### 
# ENERGY DEMAND
###

def get_demand_timeseries(n: pypsa.Network) -> pd.DataFrame:
    """Gets timeseries energy demand"""
    return pd.DataFrame(n.loads_t.p.sum(1)).rename(columns={0:"Demand"})

### 
# ENERGY CAPACITY
###

def get_capacity_base(n: pypsa.Network) -> pd.DataFrame:
    """Gets starting capacities"""
    storage_pnom = n.storage_units.groupby(["bus", "carrier"]).p_nom.sum()
    generator_pnom = n.generators.groupby(["bus", "carrier"]).p_nom.sum()
    links_pnom = n.links.groupby(["bus1", "carrier"]).p_nom.sum()
    return pd.concat([generator_pnom, links_pnom, storage_pnom])

def get_capacity_greenfield(n: pypsa.Network, retirement_method = "economic") -> pd.DataFrame:
    """Gets optimal greenfield pnom capacity"""
    def _technical_retirement(n: pypsa.Network, component:str) -> pd.DataFrame:
        if component not in ("storage_units", "generators", "links"):
             logger.warning(f"{component} not in the set ('storage_units', 'generator', 'links')")
             return pd.DataFrame()
        else:
            gens = getattr(n, component)[["carrier", "bus"]].copy()
            gens_t_p = getattr(n, f"{component}_t")["p"]
            gens["p_max"] = gens.index.map(gens_t_p.max()).fillna(0)
            if component == "links":
                gens = gens[gens.carrier.isin(["battery charger", "battery discharger"])]
                return gens.groupby(["bus1", "carrier"]).p_nom_opt.sum()
            else:
                return gens.groupby(["bus", "carrier"]).p_nom_opt.sum()
    
    def _economic_retirement(n: pypsa.Network, component:str) -> pd.DataFrame:
        if component not in ("storage_units", "generators", "links"):
             logger.warning(f"{component} not in the set ('storage_units', 'generator', 'links')")
             return pd.DataFrame()
        else:
            p_nom_opt = getattr(n, component)
            if component == "links":
                p_nom_opt = p_nom_opt[p_nom_opt.carrier.isin(["battery charger", "battery discharger"])]
                return p_nom_opt.groupby(["bus1", "carrier"]).p_nom_opt.sum()
            else:
                return p_nom_opt.groupby(["bus", "carrier"]).p_nom_opt.sum()
    
    if retirement_method == "technical":
        return pd.concat([_technical_retirement(n, x) for x in ["generators", "storage_units", "links"]])
    elif retirement_method == "economic":
        return pd.concat([_economic_retirement(n, x) for x in ["generators", "storage_units", "links"]])
    else:
        logger.error(f"Retirement method must be one of 'technical' or 'economic'. Recieved {retirement_method}.")
        raise NotImplementedError

def get_capacity_brownfield(n: pypsa.Network, retirement_method = "economic", components:str = "all") -> pd.DataFrame:
    """Gets optimal brownfield pnom capacity"""
    
    def _technical_retirement(n: pypsa.Network, component:str) -> pd.DataFrame:
        if component not in ("storage_units", "generators", "links"):
             logger.warning(f"{component} not in the set ('storage_units', 'generator', 'links')")
             return pd.DataFrame()
        else:
            p_nom_opt = getattr(n, component)
            if component == "links":
                p_nom_opt = p_nom_opt[p_nom_opt.carrier.isin(["battery charger", "battery discharger"])]
                return p_nom_opt.groupby(["bus1", "carrier"]).p_nom_opt.sum()
            else:
                return p_nom_opt.groupby(["bus", "carrier"]).p_nom_opt.sum()
    
    def _economic_retirement(n: pypsa.Network, component:str) -> pd.DataFrame:
        if component not in ("storage_units", "generators", "links"):
             logger.warning(f"{component} not in the set ('storage_units', 'generators', 'links')")
             return pd.DataFrame()
        else:
            p_nom_opt = getattr(n, component)
            if component == "links":
                p_nom_opt = p_nom_opt[p_nom_opt.carrier.isin(["battery charger", "battery discharger"])]
                return p_nom_opt.groupby(["bus1", "carrier"]).p_nom_opt.sum()
            else:
                return p_nom_opt.groupby(["bus", "carrier"]).p_nom_opt.sum()
    
    if retirement_method == "technical":
        return pd.concat([_technical_retirement(n, x) for x in ["generators", "storage_units", "links"]])
    elif retirement_method == "economic":
        return pd.concat([_economic_retirement(n, x) for x in ["generators", "storage_units", "links"]])
    else:
        logger.error(f"Retirement method must be one of 'technical' or 'economic'. Recieved {retirement_method}.")
        raise NotImplementedError

### 
# EMISSIONS
###

def get_node_emissions_timeseries(n: pypsa.Network) -> pd.DataFrame:
    """Gets timeseries emissions per node"""
    
    emission_rates = n.carriers[n.carriers["co2_emissions"] != 0]["co2_emissions"]

    if emission_rates.empty:
        return pd.DataFrame(index=n.snapshots)
    
    emission_rates = n.carriers[n.carriers["co2_emissions"] != 0]["co2_emissions"]

    emitters = emission_rates.index
    generators = n.generators[n.generators.carrier.isin(emitters)]
    
    if generators.empty:
        return pd.DataFrame(index=n.snapshots, columns=n.buses.index).fillna(0)

    em_pu = generators.carrier.map(emission_rates) / generators.efficiency # TODO timeseries efficiency 
    return (
        n.generators_t.p[generators.index]
        .mul(em_pu)
        .groupby(n.generators.bus, axis=1)
        .sum()
    )

def get_tech_emissions_timeseries(n: pypsa.Network) -> pd.DataFrame:
    """Gets timeseries emissions per technology"""
    
    emission_rates = n.carriers[n.carriers["co2_emissions"] != 0]["co2_emissions"]

    if emission_rates.empty:
        return pd.DataFrame(index=n.snapshots)

    nice_names = n.carriers["nice_name"]
    emitters = emission_rates.index
    generators = n.generators[n.generators.carrier.isin(emitters)]

    if generators.empty:
        return pd.DataFrame(index=n.snapshots, columns=emitters).fillna(0).rename(columns=nice_names)
    
    em_pu = generators.carrier.map(emission_rates) / generators.efficiency # TODO timeseries efficiency 
    return (
        n.generators_t.p[generators.index]
        .mul(em_pu)
        .groupby(n.generators.carrier, axis=1)
        .sum()
        .rename(columns=nice_names)
    )


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
    
    