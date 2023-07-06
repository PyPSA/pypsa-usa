# Copyright 2021-2022 Martha Frysztacki (KIT)

import pypsa, pandas as pd, logging
from add_electricity import load_costs, _add_missing_carriers_from_costs

idx = pd.IndexSlice

def add_conventional_plants_from_file(
    n, fn_plants, conventional_carriers, extendable_carriers, costs):

    _add_missing_carriers_from_costs(n, costs, conventional_carriers)

    plants = pd.read_csv(fn_plants, dtype={"bus_id": str}, index_col=0).query(
        "bus_id in @n.buses.index"
    )
    plants.replace(["dfo"], ["oil"], inplace=True)

    for tech in conventional_carriers:
        tech_plants = plants.query("type == @tech")
        tech_plants.index = tech_plants.index.astype(str)

        logger.info(f"Adding {len(tech_plants)} {tech} generators to the network.")

        if tech in extendable_carriers:
            p_nom_extendable = True
        else:
            p_nom_extendable = False

        n.madd(
            "Generator",
            tech_plants.index,
            bus=tech_plants.bus_id.astype(str),
            p_nom=tech_plants.Pmax,
            p_nom_extendable=p_nom_extendable,
            marginal_cost=tech_plants.GenIOB * tech_plants.GenFuelCost,  #(MMBTu/MW) * (USD/MMBTu) = USD/MW
            marginal_cost_quadratic= tech_plants.GenIOC * tech_plants.GenFuelCost,
            carrier=tech_plants.type,
            weight=1.0,
            efficiency=costs.at[tech, "efficiency"],
        )

    return n


def add_renewable_plants_from_file(
    n, fn_plants, renewable_carriers, extendable_carriers, costs):

    _add_missing_carriers_from_costs(n, costs, renewable_carriers)

    plants = pd.read_csv(fn_plants, dtype={"bus_id": str}, index_col=0).query(
        "bus_id in @n.buses.index"
    )
    plants.replace(["wind_offshore"], ["offwind"], inplace=True)

    for tech in renewable_carriers:
        tech_plants = plants.query("type == @tech")
        tech_plants.index = tech_plants.index.astype(str)

        logger.info(f"Adding {len(tech_plants)} {tech} generators to the network.")

        if tech in ["wind", "offwind"]: 
            p = pd.read_csv(snakemake.input["wind"], index_col=0)
        else:
            p = pd.read_csv(snakemake.input[tech], index_col=0)
        intersection = set(p.columns).intersection(tech_plants.index) #filters by plants ID for the plants of type tech
        p = p[list(intersection)]

        # import pdb; pdb.set_trace()
        Nhours = len(n.snapshots)
        p = p.iloc[:Nhours,:]        #hotfix to fit 2016 renewable data to load data

        p.index = n.snapshots
        p.columns = p.columns.astype(str)

        if (tech_plants.Pmax == 0).any():
            # p_nom is the maximum of {Pmax, dispatch}
            p_nom = pd.concat([p.max(axis=0), tech_plants["Pmax"]], axis=1).max(axis=1)
            p_max_pu = (p[p_nom.index] / p_nom).fillna(0)  # some values remain 0
        else:
            p_nom = tech_plants.Pmax
            p_max_pu = p[tech_plants.index] / p_nom

        if tech in extendable_carriers:
            p_nom_extendable = True
        else:
            p_nom_extendable = False

        n.madd(
            "Generator",
            tech_plants.index,
            bus=tech_plants.bus_id,
            p_nom_min=p_nom,
            p_nom=p_nom,
            marginal_cost=tech_plants.GenIOB * tech_plants.GenFuelCost, #(MMBTu/MW) * (USD/MMBTu) = USD/MW
            marginal_cost_quadratic = tech_plants.GenIOC * tech_plants.GenFuelCost, 
            capital_cost=costs.at[tech, "capital_cost"],
            p_max_pu=p_max_pu,
            p_nom_extendable=p_nom_extendable,
            carrier=tech,
            weight=1.0,
            efficiency=costs.at[tech, "efficiency"],
        )

    # hack to remove generators without capacity (required for SEG to work)
    # shouldn't exist, in fact...
    import pdb;pdb.set_trace()
    p_max_pu_norm = n.generators_t.p_max_pu.max()
    remove_g = p_max_pu_norm[p_max_pu_norm == 0.0].index
    logger.info(
        f"removing {len(remove_g)} {tech} generators {remove_g} with no renewable potential."
    )
    n.mremove("Generator", remove_g)

    return n

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    n = pypsa.Network(snakemake.input.network)

    # attach load costs
    Nhours = len(n.snapshots)
    Nyears = n.snapshot_weightings.generators.sum() / Nhours
    costs = load_costs(
        snakemake.input.tech_costs,
        snakemake.config["costs"],
        snakemake.config["electricity"],
        Nyears,
    )
    import pdb;pdb.set_trace()
    # should renaming technologies move to config.yaml?
    costs = costs.rename(index={"onwind": "wind", "OCGT": "ng"})

    interconnect = snakemake.wildcards.interconnect
    # interconnect in raw data given with an uppercase first letter
    if interconnect != "usa":
        interconnect = interconnect[0].upper() + interconnect[1:]

    # add renewable generators
    renewable_carriers = list(
        set(snakemake.config["allowed_carriers"]).intersection(
            set(["wind", "solar", "offwind", "hydro"])
        )
    )
    n = add_renewable_plants_from_file(
        n,
        snakemake.input["plants"],
        renewable_carriers,
        snakemake.config["extendable_carriers"],
        costs,
    )

    # add conventional generators
    conventional_carriers = list(
        set(snakemake.config["allowed_carriers"]).intersection(
            set(["coal", "ng", "nuclear", "oil", "geothermal"])
        )
    )
    n = add_conventional_plants_from_file(
        n,
        snakemake.input["plants"],
        conventional_carriers,
        snakemake.config["extendable_carriers"],
        costs,
    )

    # export network
    n.export_to_netcdf(snakemake.output.network)
