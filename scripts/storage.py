
import sys
sys.path.append(snakemake.config['subworkflow'] + "scripts/")
from add_electricity import load_costs, add_nice_carrier_names
from add_extra_components import attach_storageunits, attach_stores, attach_hydrogen_pipelines

import pypsa
import logging

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('add_extra_components', network='elec',
                                  simpl='', clusters=5)

    n = pypsa.Network(snakemake.input.network)
    Nyears = n.snapshot_weightings.stores.sum() / 8784
    costs = load_costs(
        snakemake.input.tech_costs, snakemake.config['costs'],
        snakemake.config['electricity'], Nyears
    )

    elec_config = snakemake.config['electricity']
    
    n.buses["country"] = "USA"
    attach_storageunits(n, costs, elec_config)
    attach_stores(n, costs, elec_config)
    attach_hydrogen_pipelines(n, costs, elec_config)

    add_nice_carrier_names(n, snakemake.config)

    n.export_to_netcdf(snakemake.output[0])
