"""
Build population layouts for all clustered model regions as total as well as
split by urban and rural population.
"""

import atlite
import geopandas as gpd
import pandas as pd
import xarray as xr

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_clustered_population_layouts",
            interconnect="texas",
            clusters=20,
        )

    cutout = atlite.Cutout(snakemake.input.cutout)

    clustered_regions = (
        gpd.read_file(snakemake.input.regions_onshore)
        .set_index("name")
        .buffer(0)
        .squeeze()
    )

    I = cutout.indicatormatrix(clustered_regions)

    pop = {}
    for item in ["total", "urban", "rural"]:
        pop_layout = xr.open_dataarray(snakemake.input[f"pop_layout_{item}"])
        pop[item] = I.dot(pop_layout.stack(spatial=("y", "x")))

    pop = pd.DataFrame(pop, index=clustered_regions.index)

    pop["country"] = pop.index.map(lambda x: x.split(" ")[0])

    # get fraction of each clustering area population at each node
    country_population = pop.total.groupby(pop.country).sum()
    pop["fraction_per_node"] = pop.total / pop.country.map(country_population)

    # get fraction of popuation that is classified as urban or rural
    pop["urban_fraction"] = pop.urban / pop.total
    pop["rural_fraction"] = pop.rural / pop.total

    pop.to_csv(snakemake.output.clustered_pop_layout)
