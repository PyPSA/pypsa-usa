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
            interconnect="western",
            # simpl="",
            clusters=60,
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

    pop["ba"] = pop.index.map(lambda x: x.split(" ")[0])
    ba_population = pop.total.groupby(pop.ba).sum()
    pop["fraction"] = pop.total / pop.ba.map(ba_population)

    pop.to_csv(snakemake.output.clustered_pop_layout)
