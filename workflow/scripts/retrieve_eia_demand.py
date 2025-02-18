"""Retrieves data from the EIA API."""

from _helpers import configure_logging
from demand_scalers import AeoEnergyScalerApi, AeoVmtScalerApi

# def get_energy_demand(api: str, years: int, scenario: str = "reference") -> pd.DataFrame:
#     """
#     Get sector yearly END-USE ENERGY growth rates from AEO at a NATIONAL
#     level.

#     |      | residential | commercial  | industrial  | transport  | units |
#     |----- |-------------|-------------|-------------|------------|-------|
#     | 2018 |     ###     |     ###     |     ###     |     ###    |  ###  |
#     | 2019 |     ###     |     ###     |     ###     |     ###    |  ###  |
#     | 2020 |     ###     |     ###     |     ###     |     ###    |  ###  |
#     | ...  |             |             |             |            |       |
#     | 2049 |     ###     |     ###     |     ###     |     ###    |  ###  |
#     | 2050 |     ###     |     ###     |     ###     |     ###    |  ###  |
#     """

#     def get_sector_data(years: list[int], sector: str) -> pd.DataFrame:
#         """Function to piece togehter historical and projected values."""
#         start_year = min(years)
#         end_year = max(years)

#         data = []

#         if start_year < 2024:
#             data.append(
#                 EnergyDemand(sector=sector, year=start_year, api=api).get_data(),
#             )
#         if end_year >= 2024:
#             data.append(
#                 EnergyDemand(sector=sector, year=end_year, api=api, scenario=scenario).get_data(),
#             )
#         return pd.concat(data)

#     sectors = ("residential", "commercial", "industry", "transport")

#     df = pd.DataFrame(
#         index=years,
#     )

#     for sector in sectors:
#         sector_data = get_sector_data(years, sector).sort_index()
#         df[sector] = sector_data.value

#     df["units"] = "quads"
#     return df


# def get_transport_demand(api: str, years: int, scenario: str = "reference") -> pd.DataFrame:
#     """
#     Get sector yearly END-USE ENERGY growth rates from AEO at a NATIONAL
#     level.

#     |      | light_duty | med_duty  | heavy_duty  | bus  | units |
#     |----- |------------|-----------|-------------|------|-------|
#     | 2018 |     ###    |    ###    |     ###     | ###  |  ###  |
#     | 2019 |     ###    |    ###    |     ###     | ###  |  ###  |
#     | 2020 |     ###    |    ###    |     ###     | ###  |  ###  |
#     | ...  |            |           |             |      |       |
#     | 2049 |     ###    |    ###    |     ###     | ###  |  ###  |
#     | 2050 |     ###    |    ###    |     ###     | ###  |  ###  |
#     """

#     def get_historical_value(year: int, sector: str) -> float:
#         """Returns single year value at a time."""
#         return TransportationDemand(vehicle=sector, year=year, api=api).get_data(pivot=True).values[0][0]

#     def get_future_values(
#         year: int,
#         sector: str,
#         scenario: str,
#     ) -> pd.DataFrame:
#         """Returns all values from 2024 onwards."""
#         return TransportationDemand(
#             vehicle=sector,
#             year=year,
#             api=api,
#             scenario=scenario,
#         ).get_data()

#     vehicles = ("light_duty", "med_duty", "heavy_duty", "bus")

#     df = pd.DataFrame(
#         columns=["light_duty", "med_duty", "heavy_duty", "bus"],
#         index=years,
#     )

#     for year in sorted(YEARS):
#         if year < 2024:
#             for vehicle in vehicles:
#                 df.at[year, vehicle] = get_historical_value(
#                     year,
#                     vehicle,
#                 )

#     for vehicle in vehicles:
#         aeo = get_future_values(max(YEARS), vehicle, scenario)
#         for year in YEARS:
#             if year < 2024:
#                 continue
#             df.at[year, vehicle] = aeo.at[year, "value"]

#     df["units"] = "thousand VMT"
#     return df

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "retrieve_eia_demand",
            scenario="reference",
            source="transport",
        )
    configure_logging(snakemake)

    source = snakemake.wildcards.source
    scenario = snakemake.wildcards.scenario
    api = snakemake.params.api
    save = snakemake.output

    if source == "energy":
        df = AeoEnergyScalerApi(api, scenario).get_projections()
    elif source == "transport":
        df = AeoVmtScalerApi(api, scenario).get_projections()
    elif source == "electricity":
        raise NotImplementedError
    else:
        raise ValueError

    df = df.reset_index(names="YEAR")

    df.to_csv(save, index=False)
