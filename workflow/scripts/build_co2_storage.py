# import necessary Python modules
import logging

import geopandas
import pandas


def build_co2_storage(regions_onshore_geojson, co2_storage_geojson, output_csv, logger):
    # get PyPSA-USA network nodes and CO2 storage information at a county level
    if logger is not None:
        logger.info("Calculate CO2 storage potentials and costs")
    regions_onshore = geopandas.read_file(regions_onshore_geojson)
    co2_storage = geopandas.read_file(co2_storage_geojson)

    # create data frame to store aggregated CO2 storage potential and average cost for each node
    data_frame = pandas.DataFrame(columns=["node", "potential [MtCO2]", "cost [USD/tCO2]"])

    # iterate through PyPSA-USA network nodes
    for i in range(len(regions_onshore)):
        # get node name and geometry
        region_onshore = regions_onshore.iloc[i]
        node_name = region_onshore["name"]
        node_geometry = geopandas.GeoSeries(region_onshore["geometry"])
        node_geometry.crs = co2_storage.crs

        # iterate through CO2 storage information at a county level
        co2_storage_potential = 0
        co2_storage_cost_sum = 0
        for j in range(len(co2_storage)):
            # get county geometry
            co2_storage_county = co2_storage.iloc[j]
            co2_storage_geometry = geopandas.GeoSeries(co2_storage_county["geometry"])

            # calculate proportion of intersection between node geometry and county geometry
            intersection = co2_storage_geometry.intersection(node_geometry.iloc[0])
            proportion = float(intersection.area.iloc[0]) / float(co2_storage_geometry.area.iloc[0])

            # calculate CO2 storage potential and cost
            potential = co2_storage_county["Capacity_All_Mt"] * proportion
            co2_storage_potential += potential
            co2_storage_cost_sum += float(co2_storage_county["StorageCost_USDperTonCO2"]) * potential

            # calculate average CO2 storage cost for the node
            if co2_storage_potential > 0:
                co2_storage_cost_average = co2_storage_cost_sum / co2_storage_potential
            else:
                co2_storage_potential = 0
                co2_storage_cost_average = 0

        # add aggregated CO2 storage potential and average cost for the node into data frame
        if logger is not None:
            if co2_storage_potential == 0:
                logger.info("Node '%s' has no CO2 storage potential" % node_name)
            else:
                logger.info(
                    f"Node '{node_name}' has an aggregated CO2 storage potential of {co2_storage_potential:0.1f} MtCO2 with an average cost of {co2_storage_cost_average:0.2f} USD/tCO2"
                )
        data_frame.loc[len(data_frame)] = [node_name, co2_storage_potential, co2_storage_cost_average]

    # write data frame into a CSV file
    if logger is not None:
        logger.info("Save CO2 storage potentials and costs into CSV file '%s'" % output_csv)
    data_frame.set_index("node", inplace=True)
    data_frame.to_csv(output_csv)


if __name__ == "__main__":
    # build and save CO2 storage potentials and costs
    if "snakemake" in globals():
        build_co2_storage(
            snakemake.input["regions_onshore"], snakemake.input["co2_storage"], snakemake.output["co2_storage"], logger
        )
    else:
        logger = logging.getLogger(__name__)
        build_co2_storage("regions_onshore_s75.geojson", "co2_storage.geojson", "co2_storage.csv", logger)
