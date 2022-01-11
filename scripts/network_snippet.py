# Copyright 2021-2022 Martha Frysztacki (KIT)

import pypsa

prep = pypsa.Network(snakemake.input[0])

snapshots = prep.snapshots[0:24]
n = prep.copy(snapshots=snapshots)

n.export_to_netcdf(snakemake.output[0])
