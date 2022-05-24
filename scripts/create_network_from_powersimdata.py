import logging
import sys
import os

from powersimdata.input.export_data import export_to_pypsa
from powersimdata import Scenario
from yaml import load

from create_network_from_zenodo import add_custom_line_type

sys.path.append(os.path.join("pypsa-eur", "scripts"))
from add_electricity import load_costs, _add_missing_carriers_from_costs


def load_scenario(interconnect="Western"):
    """
    This code is copied from https://breakthrough-
    energy.github.io/docs/powersimdata/scenario.html#creating-a-scenario.
    """
    scenario = Scenario()
    scenario.set_grid(grid_model="usa_tamu", interconnect=interconnect)
    scenario.set_base_profile("demand", "vJan2021")
    scenario.set_base_profile("hydro", "vJan2021")
    scenario.set_base_profile("solar", "vJan2021")
    scenario.set_base_profile("wind", "vJan2021")

    return scenario


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    interconnect = snakemake.wildcards["interconnect"].title()
    interconnect = "USA" if interconnect == "Usa" else interconnect

    # Create scenario
    scenario = load_scenario(interconnect=interconnect)
    grid = scenario.get_base_grid()
    n = export_to_pypsa(scenario, add_substations=False, add_load_shedding=False)

    Nyears = n.snapshot_weightings.generators.sum() / 8760
    costs = load_costs(
        snakemake.input.tech_costs,
        snakemake.config["costs"],
        snakemake.config["electricity"],
        Nyears,
    )
    costs.rename(index={"onwind": "wind", "gas": "ng"}, inplace=True)

    allowed_carriers = snakemake.config["allowed_carriers"]
    committable_carriers = snakemake.config["committable_carriers"]
    extendable_carriers = snakemake.config["extendable_carriers"]
    n.generators = n.generators.assign(
        capital_cost=n.generators.carrier.map(costs.capital_cost),
        weight=1,
        p_nom_extendable=n.generators.carrier.isin(extendable_carriers),
        commmittable=n.generators.carrier.isin(committable_carriers),
    )
    n.mremove("Generator", n.generators.query("carrier not in @allowed_carriers").index)

    add_custom_line_type(n)
    n.lines = n.lines.assign(type="Rail")

    grid.bus2sub.to_csv(snakemake.output.bus2sub)
    grid.sub.to_csv(snakemake.output.sub)
    n.export_to_netcdf(snakemake.output.network)
