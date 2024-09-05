"""
Builds End-Use initial stock data.
"""

# to supress warning in water heat xlsx
# UserWarning: Print area cannot be set to Defined name: data!$A:$J
import warnings

warnings.simplefilter("ignore")

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pypsa
import yaml
from build_heat import combined_heat
from constants import STATE_2_CODE, STATES_CENSUS_DIVISION_MAPPER
from eia import TransportationFuelUse

logger = logging.getLogger(__name__)

CODE_2_STATE = {v: k for k, v in STATE_2_CODE.items()}
"""
Hardcoded build years based on building year constructed starting from 2000
https://www.eia.gov/consumption/commercial/data/2018/bc/pdf/b6.pdf
"""
CECS_BUILD_YEARS = {
    # year_built: percent of stock
    2018: 9.1,
    2009: 15.6,
    2000: 50.0,  # assumed from replacement from 2000 data
}
"""
Hardcoded appliance build years from residenital energy consumption survey
https://www.eia.gov/consumption/residential/data/2020/hc/pdf/HC%206.1.pdf
"""
RECS_BUILD_YEARS = {
    # year_built: percent of stock
    2020: 10.8,
    2018: 13.8,
    2015: 21.7,
    2010: 18.0,
    2005: 11.9,
    2000: 19.1,
}


class Recs:
    """
    Processes state level residential energy consumption survey stock data.

    Provide folder to any/all of these files from. Do not change file name!
    https://www.eia.gov/consumption/residential/data/2020/index.php?view=state#hc
    - Highlights for space heating in U.S. homes by state, 2020
    - Highlights for air conditioning in U.S. homes by state, 2020
    - Highlights for space heating fuel in U.S. homes by state, 2020
    - Highlights for water heating in U.S. homes by state, 2020
    """

    file_mapper = {
        "aircon_stock": "State Air Conditioning",
        "space_heat_stock": "State Space Heating",
        # "water_heat": "State Water Heating",
        "space_heat_fuel": "State Space Heating Fuels",
        "water_heat_fuel": "State Water Heating Fuels",
    }

    column_mapper = {
        "aircon_stock": {
            "Unnamed: 1": "total_stock",
            "Uses air-conditioning equipment": "ac_stock",
            "Unnamed: 3": "ac_percent",
            "Uses central air-conditioning unitb": "ac_central_stock",
            "Unnamed: 5": "ac_central_percent",
            "Uses individual air-conditioning unitc": "ac_individual_stock",
            "Unnamed: 7": "ac_individual_percent",
            "Unnamed: 8": "dehumidifier_stock",
            "Unnamed: 9": "dehumidifier_percent",
            "Unnamed: 10": "ceiling_fan_stock",
            "Unnamed: 11": "ceiling_fan_percent",
        },
        "space_heat_stock": {
            "Totala: 1": "total_stock",
            "Furnace": "furnace_stock",
            "Unnamed: 3": "furnace_percent",
            "Central heat pump": "central_hp_stock",
            "Unnamed: 5": "central_hp_percent",
            "Steam or hot water boiler": "boiler_stock",
            "Unnamed: 7": "boiler_percent",
            "Unnamed: 8": "secondary_heater_stock",
            "Unnamed: 9": "secondary_heater_percent",
            "Unnamed: 10": "humidifier_stock",
            "Unnamed: 11": "humidifier_percent",
        },
        "space_heat_fuel": {
            "Unnamed: 1": "total_stock",
            "Electricity": "electricity_stock",
            "Unnamed: 3": "electricity_percent",
            "Natural gas": "gas_stock",
            "Unnamed: 5": "gas_percent",
            "Propane": "propane_stock",
            "Unnamed: 7": "propane_percent",
            "Fuel oil or kerosene": "lpg_stock",
            "Unnamed: 9": "lpg_percent",
        },
        "water_heat_fuel": {
            "Unnamed: 1": "total_stock",
            "Electricity": "electricity_stock",
            "Unnamed: 3": "electricity_percent",
            "Natural gas": "gas_stock",
            "Unnamed: 5": "gas_percent",
            "Propane": "propane_stock",
            "Unnamed: 7": "propane_percent",
            "Fuel oil or kerosene": "lpg_stock",
            "Unnamed: 9": "lpg_percent",
        },
    }

    def __init__(self, root_dir: str) -> None:
        self.dir = root_dir

    @staticmethod
    def _valid_name(s: str):
        assert s in [
            "aircon_stock",
            "space_heat_stock",
            "water_heat_fuel",
            "space_heat_fuel",
        ]

    def _read(self, stock: str) -> pd.DataFrame:
        """
        Reads in the data.
        """
        f = Path(self.dir, f"{self.file_mapper[stock]}.xlsx")

        df = (
            pd.read_excel(
                f,
                sheet_name="data",
                skiprows=4,
                index_col=0,
                skipfooter=2,
            )
            .rename(columns=self.column_mapper[stock], index={"All homes": "USA"})
            .dropna()
            .astype(str)  # avoids downcasting object dtype error
            .replace("Q", np.nan)  # > 50% RSE or n < 10
            .replace("N", np.nan)  # No households
            .astype(float)
        )
        df.index = df.index.map(lambda x: x.strip())
        return df

    def _get_data(self, stock: str, fillna: bool = False) -> pd.DataFrame:
        """
        Formats data.
        """
        self._valid_name(stock)
        df = self._read(stock)
        if fillna:
            return self._fill_missing(df)
        else:
            return df

    def get_percentage(self, stock: str) -> pd.DataFrame:
        """
        Gets percentage of stock per state.
        """
        df = self._get_data(stock, fillna=True)
        return df[[x for x in df.columns if x.endswith("percent")]]

    def get_absolute(self, stock: str) -> pd.DataFrame:
        """
        Gets raw stock values per state.
        """
        df = self._get_data(stock, fillna=False)
        return df[[x for x in df.columns if x.endswith("stock")]]

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values with USA average.
        """
        columns = df.columns
        for col in columns:
            df[col] = df[col].fillna(df.at["USA", col])
        return df


class Cecs:
    """
    Processes census level commercial energy consumption survey stock data.

    Provide path to excel files of 'c7.xlsx', 'c8.xlsx', 'c9.xlsx' from:
    - https://www.eia.gov/consumption/commercial/data/2018/index.php?view=consumption#major
    """

    # Percentages from Page 7 from Table C7 at
    # https://www.eia.gov/consumption/commercial/data/2018/index.php?view=consumption#major
    usa_avg_space_heating = {
        "Electricity": 0.313,  # 1856 / 5918
        "Natural gas": 0.396,  # 2344 / 5918
        "Fuel oil": 0.038,  # 222 / 5918
        "District heat": 0.013,  # 78 / 5918
        "Propane": 0.057,  # 338 / 5918
    }
    usa_avg_water_heating = {
        "Electricity": 0.471,  # 2785 / 5918
        "Natural gas": 0.319,  # 1885 / 5918
        "Fuel oil": 0.012,  # 72 / 5918
        "District heat": 0.006,  # 38 / 5918
        "Propane": 0.025,  # 145 / 5918
    }
    usa_avg_cooling = {
        "Electricity": 0.775,  # 4584 / 5918
        "Natural gas": 0.0,  # 4 / 5918
        "District chilled": 0.01,  # 55 / 5918
    }

    census_name_map = {
        "NewEngland": "new_england",
        "Middle Atlantic": "mid_atlantic",
        "EastNorthCentral": "east_north_central",
        "WestNorthCentral": "west_north_central",
        "SouthAtlantic": "south_atlantic",
        "EastSouthCentral": "east_south_central",
        "WestSouthcentral": "west_south_central",
        "Mountain": "mountain",
        "Pacific": "pacific",
    }

    state_2_census_division = STATES_CENSUS_DIVISION_MAPPER
    code_2_state = CODE_2_STATE

    def __init__(self, root_dir: Path | str) -> None:
        self.dir = root_dir

    @staticmethod
    def _valid_name(s: str):
        assert s in ["aircon_fuel", "space_heat_fuel", "water_heat_fuel"]

    def _read(self, fuel: str) -> pd.DataFrame:
        """
        Reads in the data.
        """

        skip_rows = self._get_skip_rows(fuel)

        dfs = []

        for f_name in ("c7", "c8", "c9"):

            f = Path(self.dir, f"{f_name}.xlsx")

            df = (
                pd.read_excel(
                    f,
                    sheet_name="data",
                    skiprows=skip_rows,
                    index_col=0,
                    skipfooter=2,
                    usecols="A:D",
                )
                .dropna()
                .astype(str)  # avoids downcasting object dtype error
                .replace("Q", np.nan)  # > 50% RSE or n < 10
                .replace("N", np.nan)  # No households
                .astype(float)
            )
            df = df.rename(columns={x: x.replace("\n", "") for x in df.columns}).rename(
                columns=self.census_name_map,
            )
            dfs.append(df)

        return pd.concat(dfs, axis=1, join="outer")

    @staticmethod
    def _get_skip_rows(fuel: str) -> list[int]:
        """
        Gets rows to skip when reading in data files.
        """

        keep_rows = [3, 4]

        match fuel:
            case "space_heat_fuel":  # primary energy source
                keep_rows.extend([163, 164, 165, 166, 167])  # excludes 'other'
            case "water_heat_fuel":  # primary and secondary energy source
                keep_rows.extend([174, 175, 176, 177, 178])
            case "aircon_fuel":  # primary and secondary energy source
                keep_rows.extend([170, 171, 172])
            case _:
                raise NotImplementedError

        # 307 is number of lines in the excel file
        return [x for x in range(0, 307) if x not in keep_rows]

    def _get_data(
        self,
        fuel: str,
        as_percent: bool = True,
        fillna: bool = False,
        by_state: bool = True,
    ) -> pd.DataFrame:
        """
        Formats data.
        """
        self._valid_name(fuel)
        data = self._read(fuel)

        if as_percent:
            df = data.T
            df = df.div(df["All buildings"], axis=0)
            df = df.drop(columns=["All buildings"]).T
        else:
            df = data

        if by_state:
            df = self._expand_by_state(df).T
        else:
            df = df.T

        if fillna:
            return self._fill_missing(df, fuel)
        else:
            return df

    def _expand_by_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps census division to state.
        """

        states = pd.DataFrame(index=df.index)
        for state, census_division in self.state_2_census_division.items():
            try:
                states[self.code_2_state[state]] = df[census_division]
            except KeyError:
                logger.info(f"No Census Division for state {state}")
        return states

    def _fill_missing(self, df: pd.DataFrame, fuel: str) -> pd.DataFrame:
        """
        Fills missing values with USA average.
        """
        match fuel:
            case "space_heat_fuel":
                fill_values = self.usa_avg_space_heating
            case "water_heat_fuel":
                fill_values = self.usa_avg_water_heating
            case "aircon_fuel":
                fill_values = self.usa_avg_cooling
            case _:
                raise NotImplementedError

        for col in df.columns:
            df[col] = df[col].fillna(fill_values[col])

        return df

    """
    The two following functions are included for consistencey with the Recs class
    """

    def get_percentage(self, fuel: str, by_state: bool = True) -> pd.DataFrame:
        """
        Gets percentage of stock at a national level.
        """
        return (
            self._get_data(fuel, as_percent=True, by_state=by_state, fillna=True)
            .mul(100)
            .round(2)
        )

    def get_absolute(self, fuel: str, by_state: bool = True) -> pd.DataFrame:
        """
        Gets raw stock values at national level.
        """
        return self._get_data(fuel, as_percent=False, by_state=by_state, fillna=False)


def _already_retired(build_year: int, lifetime: int, year: int) -> bool:
    """
    Checks if brownfield capacity should already be retired.

    remove any exiting brownfield that already exceeds its lifetime '<='
    instead of '<' to follow pypsa convention. See folling link
    https://pypsa.readthedocs.io/en/latest/examples/multi-investment-optimisation.html#Multi-Investment-Optimization
    """

    if (build_year + lifetime) <= year:
        return True
    else:
        return False


def _get_marginal_cost(
    n: pypsa.Network,
    names: list[str],
    fuel: Optional[str] = None,
) -> float | pd.DataFrame:
    """
    Gets marginal cost from the investable link.

    If dyanmic costs are applied, returns the marginal cost dataframe.
    Else, returns the static cost associated with the first name in the
    list
    """

    df = pd.DataFrame(index=n.links_t.marginal_cost.index)

    try:
        for name in names:
            df[name] = n.links_t.marginal_cost[name]
        return df
    except KeyError:
        logger.info(f"No dynamic cost found for {name}")
        if fuel:
            return n.links.at[fuel, "marginal_cost"]
        else:
            logger.warning(f"No fuel costs applied for {name}")


###
# Public methods
###


def get_residential_stock(root_dir: str, load: str) -> pd.DataFrame:
    """
    Gets residential stock values as a percetange.

    Pass folder of data from the residential energy consumption survey
    """

    recs = Recs(root_dir)

    match load:
        case "space_heating":
            df = recs.get_percentage("space_heat_fuel")
        case "water_heating":
            df = recs.get_percentage("water_heat_fuel")
        case "cooling":
            # dont have fuels, so take ac stock percentage and convert to electricity
            df = recs.get_percentage("aircon_stock")
            df = df.rename(
                columns={x: x.replace("ac_", "electricity_") for x in df.columns},
            )
        case _:
            raise NotImplementedError

    fuels = ["electricity", "gas", "lpg"]

    df = df.rename(columns={x: x.split("_percent")[0] for x in df.columns})
    df = df[[x for x in fuels if x in df.columns]]
    return df


def get_commercial_stock(root_dir: Path | str, fuel: str) -> pd.DataFrame:
    """
    Gets commercial fuel values as a percetange.
    """

    cecs = Cecs(root_dir)

    match fuel:
        case "space_heating":
            df = cecs.get_percentage("space_heat_fuel")
        case "water_heating":
            df = cecs.get_percentage("water_heat_fuel")
        case "cooling":
            df = cecs.get_percentage("aircon_fuel")
        case _:
            raise NotImplementedError

    fuels = {"Electricity": "electricity", "Natural gas": "gas", "Fuel oil": "lpg"}

    df = df[[x for x in fuels if x in df.columns]]
    return df.rename(columns=fuels)


def get_transport_stock(api: str, year: int) -> pd.DataFrame:
    """
    Gets exisiting transport stock by fuel consumption.

    Note: 'gas' in the return dataframe is natural gas! 'lpg' is motor
    gasoline and disel
    """

    def _get_data(api: str, year: int) -> pd.DataFrame:

        dfs = []

        for vehicle in ("light_duty", "med_duty", "heavy_duty", "bus"):
            df = TransportationFuelUse(vehicle, year, api).get_data(
                pivot=False,
            )  # note: cannot pivot this due to a bug
            df = df[df.index == year].reset_index()
            df["vehicle"] = vehicle

            # needed since transit bus is a subset of bus category
            if vehicle == "bus":
                df["series-description"] = df["series-description"].str.replace(
                    "Transit Bus",
                    "Total",
                )

            dfs.append(
                df.pivot(
                    index="series-description",
                    columns="vehicle",
                    values="value",
                ),
            )

        return pd.concat(dfs, axis=1).fillna(0)

    def get_percentage(api: str, year: int) -> pd.DataFrame:
        """
        Gets percentage of stock at a national level.
        """
        df = get_absolute(api, year)

        for col in df.columns:
            df[col] = df[col].div(df.at["Total", col])

        df = df.drop(index=["Total"])

        return df.mul(100).round(2)

    def get_absolute(api: str, year: int) -> pd.DataFrame:
        """
        Gets raw stock values at national level.
        """
        return _get_data(api, year).round(2)

    df = get_percentage(api, year).T
    df = (
        df.rename(
            columns={
                "Distillate Fuel Oil": "lpg",
                "Electricity": "electricity",
                "Ethanol": "ethanol",
                "Hydrogen": "hydrogen",
                "Motor Gasoline": "lpg",
                "Natural Gas": "gas",
                "Propane": "propane",
                "E85": "e85",
            },
        )
        .T.groupby(level=0)
        .sum()
    )

    return df.loc[["electricity", "lpg", "gas"]]


###
# brownfield application
###

"""
Brownfield is added as a percentage of max load. For example, if we are adding
brownfield capacity for 2030, we take max load in modelled 2030 year, use the
growth multiplier to determine how much load has increased from 2023, then
apply the brownfield capacity to the 2023 value.

For example.
> Max load in 2030 is 1GW
> Growth multiplier is 0.8 -> Max load in 2023 is 0.80GW
> Browfield capacity of natural gas is 50%
> Applied capacity is 0.40GW with a build year of 2023
"""


def _get_brownfield_template_df(
    n: pypsa.Network,
    fuel: str,
    sector: str,
    subsector: Optional[str] = None,
) -> None:
    """
    Gets a dataframe in the following form.

    |     | bus1                  | name   | suffix         | state | p_max     |
    |-----|-----------------------|--------|----------------|-------|-----------|
    | 0   | p480 0 com-urban-heat | p480 0 | com-urban-heat | TX    | 90.0544   |
    | 1   | p600 0 com-urban-heat | p600 0 | com-urban-heat | TX    | 716.606   |
    | 2   | p610 0 com-urban-heat | p610 0 | com-urban-heat | TX    | 1999.486  |
    | ... | ...                   | ...    | ...            | ...   | ...       |
    """

    assert fuel in ("cool", "elec", "heat", "lpg", "space-heat", "water-heat")

    if subsector:
        loads = n.loads[
            (n.loads.carrier.str.endswith(f"{fuel}-{subsector}"))
            & (n.loads.carrier.str.startswith(sector))
        ]
    else:
        loads = n.loads[
            (n.loads.carrier.str.endswith(fuel))
            & (n.loads.carrier.str.startswith(sector))
        ]

    df = n.loads_t.p_set[loads.index].max().to_frame(name="p_max")

    df["state"] = df.index.map(n.loads.bus).map(n.buses.STATE)
    df = df.reset_index(names="bus1")
    df["name"] = df.bus1.map(lambda x: x.split(f" {sector}")[0])
    df["suffix"] = [
        bus.split(name)[1].strip() for (bus, name) in df[["bus1", "name"]].values
    ]

    return df[["bus1", "name", "suffix", "state", "p_max"]]


def add_transport_brownfield(
    n: pypsa.Network,
    vehicle: str,
    growth_multiplier: float,
    ratios: pd.DataFrame,
    costs: pd.DataFrame,
) -> None:
    """
    Adds existing stock to transportation sector.
    """

    def add_brownfield_ev(
        n: pypsa.Network,
        df: pd.DataFrame,
        vehicle: str,
        ratios: pd.DataFrame,
        costs: pd.DataFrame,
    ) -> None:

        match vehicle:
            case "lgt":
                costs_name = "Light Duty Cars BEV 300"
                ratio_name = "light_duty"
            case "med":
                costs_name = "Medium Duty Trucks BEV"
                ratio_name = "med_duty"
            case "hvy":
                costs_name = "Heavy Duty Trucks BEV"
                ratio_name = "heavy_duty"
            case "bus":
                costs_name = "Buses BEV"
                ratio_name = "bus"
            case _:
                raise NotImplementedError

        # dont bother adding in extra for less than 0.5% market share
        if ratios.at["electricity", ratio_name] < 0.5:
            logger.info(f"No Brownfield for {costs_name}")
            return

        # 1000s to convert:
        #  miles/MWh -> k-miles/MWh
        efficiency = costs.at[costs_name, "efficiency"] / 1000
        lifetime = costs.at[costs_name, "lifetime"]

        df["bus0"] = df.name + " trn-elec-veh"
        df["carrier"] = f"trn-elec-{vehicle}"

        df["ratio"] = ratios.at["electricity", ratio_name]
        df["p_nom"] = df.p_max.mul(df.ratio).div(100)  # div to convert from %

        # roll back vehicle stock in 5 year segments
        step = 5  # years
        periods = int(lifetime // step)

        start_year = n.investment_periods[0]
        start_year = start_year if start_year <= 2023 else 2023

        for period in range(1, periods + 1):

            build_year = start_year - period * step
            percent = step / lifetime  # given as a ratio

            if _already_retired(build_year, lifetime, start_year):
                continue

            vehicles = df.copy()

            vehicles["name"] = (
                vehicles.name + f" existing_{build_year} " + vehicles.carrier
            )
            vehicles["p_nom"] = vehicles.p_nom.mul(percent).round(2)
            vehicles = vehicles.set_index("name")

            n.madd(
                "Link",
                vehicles.index,
                bus0=vehicles.bus0,
                bus1=vehicles.bus1,
                carrier=vehicles.carrier,
                efficiency=efficiency,
                capital_cost=0,
                p_nom_extendable=False,
                p_nom=vehicles.p_nom,
                lifetime=lifetime,
                build_year=build_year,
            )

    def add_brownfield_lpg(
        n: pypsa.Network,
        df: pd.DataFrame,
        vehicle: str,
        ratios: pd.DataFrame,
        costs: pd.DataFrame,
    ) -> None:

        # existing stock efficiencies taken from 2016 EFS Technology data
        # This is consistent with where future efficiencies are taken from
        # https://data.nrel.gov/submissions/93
        # https://www.nrel.gov/docs/fy18osti/70485.pdf

        match vehicle:
            case "lgt":
                costs_name = "Light Duty Cars ICEV"
                ratio_name = "light_duty"
                # efficiency = 25.9  # mpg
                efficiency = 5  # mpg
            case "med":
                costs_name = "Medium Duty Trucks ICEV"
                ratio_name = "med_duty"
                # efficiency = 16.35  # mpg
                efficiency = 3  # mpg
            case "hvy":
                costs_name = "Heavy Duty Trucks ICEV"
                ratio_name = "heavy_duty"
                # efficiency = 5.44  # mpg
                efficiency = 1  # mpg
            case "bus":
                costs_name = "Buses ICEV"
                ratio_name = "bus"
                # efficiency = 3.67  # mpg
                efficiency = 0.75  # mpg
            case _:
                raise NotImplementedError

        # dont bother adding in extra for less than 0.5% market share
        if ratios.at["lpg", ratio_name] < 0.5:
            logger.info(f"No Brownfield for {costs_name}")
            return

        # same assumption when building future efficiences in 'build_sector_costs.py'
        # Assumptions from https://www.nrel.gov/docs/fy18osti/70485.pdf
        wh_per_gallon = 33700  # footnote 24

        # mpg -> miles/wh -> miles/MWh -> k miles / MWH
        efficiency *= (1 / wh_per_gallon) * 1000000 / 1000
        lifetime = costs.at[costs_name, "lifetime"]

        df["bus0"] = df.name + " trn-lpg-veh"
        df["carrier"] = f"trn-lpg-{vehicle}"

        df["ratio"] = ratios.at["lpg", ratio_name]
        df["p_nom"] = df.p_max.mul(df.ratio).div(100)  # div to convert from %

        marginal_cost = _get_marginal_cost(n, df.bus1.to_list())

        # roll back vehicle stock in 5 year segments
        step = 5  # years
        periods = int(lifetime // step)

        start_year = n.investment_periods[0]
        start_year = start_year if start_year <= 2023 else 2023

        for period in range(1, periods + 1):

            build_year = start_year - period * step
            percent = step / lifetime  # given as a ratio

            if _already_retired(build_year, lifetime, start_year):
                continue

            vehicles = df.copy()

            vehicles["name"] = (
                vehicles.name + f" existing_{build_year} " + vehicles.carrier
            )
            vehicles["p_nom"] = vehicles.p_nom.mul(percent).round(2)
            vehicles = vehicles.set_index("name")

            if isinstance(marginal_cost, pd.DataFrame):
                mc = marginal_cost.copy()
                name_mapper = vehicles["bus1"].to_dict()
                mc = mc.rename(columns={v: k for k, v in name_mapper.items()})
            else:
                mc = marginal_cost
                assert isinstance(mc, (float, int))

            n.madd(
                "Link",
                vehicles.index,
                bus0=vehicles.bus0,
                bus1=vehicles.bus1,
                carrier=vehicles.carrier,
                efficiency=efficiency,
                capital_cost=0,
                p_nom_extendable=False,
                p_nom=vehicles.p_nom,
                lifetime=lifetime,
                build_year=build_year,
                marginal_cost=mc,
            )

    # ev brownfield
    df = _get_brownfield_template_df(n, "elec", "trn", vehicle)
    df["p_nom"] = df.p_max.mul(growth_multiplier)
    add_brownfield_ev(n, df, vehicle, ratios, costs)

    # lpg brownfield
    df = _get_brownfield_template_df(n, "lpg", "trn", vehicle)
    df["p_nom"] = df.p_max.mul(growth_multiplier)
    add_brownfield_lpg(n, df, vehicle, ratios, costs)


def add_service_brownfield(
    n: pypsa.Network,
    sector: str,
    fuel: str,
    growth_multiplier: float,
    ratios: pd.DataFrame,
    costs: pd.DataFrame,
) -> None:
    """
    Adds existing stock to res/com sector.
    """

    def add_brownfield_gas_furnace(
        n: pypsa.Network,
        df: pd.DataFrame,
        sector: str,
        ratios: pd.DataFrame,
        costs: pd.DataFrame,
    ) -> None:

        # existing efficiency values taken from:
        # https://www.eia.gov/analysis/studies/buildings/equipcosts/pdf/full.pdf

        # will give approximate installed capacity percentage by year
        if sector == "res":
            installed_capacity = RECS_BUILD_YEARS
            lifetime = costs.at["Residential Gas-Fired Furnaces", "lifetime"]
            efficiency = 0.80
        elif sector == "com":
            installed_capacity = CECS_BUILD_YEARS
            lifetime = costs.at["Commercial Gas-Fired Furnaces", "lifetime"]
            efficiency = 0.80

        efficiency2 = costs.at["gas", "co2_emissions"]

        df["bus0"] = df.state + " gas"
        df["bus2"] = df.state + f" {sector}-co2"

        # remove 'heat' or 'cool' ect.. from suffix
        df["carrier"] = df.suffix.map(lambda x: "-".join(x.split("-")[:-1]))
        df["carrier"] = df.carrier + "-gas-furnace"

        df["ratio"] = df.state.map(ratios.gas)
        df["p_nom"] = df.p_max.mul(df.ratio).div(100)  # div to convert from %

        marginal_cost_names = [
            x.replace("heat", "gas-furnace") for x in df.bus1.to_list()
        ]
        marginal_cost = _get_marginal_cost(n, marginal_cost_names)

        start_year = n.investment_periods[0]

        for build_year, percent in installed_capacity.items():

            if _already_retired(build_year, lifetime, start_year):
                continue

            furnaces = df.copy()

            furnaces["name"] = (
                furnaces.name + f" existing_{build_year} " + furnaces.carrier
            )
            furnaces["p_nom"] = furnaces.p_nom.mul(percent).div(100).round(2)
            furnaces = furnaces.set_index("name")

            if isinstance(marginal_cost, pd.DataFrame):
                mc = marginal_cost.copy()
                name_mapper = (
                    furnaces["bus1"].str.replace("-heat", "-gas-furnace").to_dict()
                )
                mc = mc.rename(columns={v: k for k, v in name_mapper.items()})
            else:
                mc = marginal_cost
                assert isinstance(mc, (float, int))

            n.madd(
                "Link",
                furnaces.index,
                bus0=furnaces.bus0,
                bus1=furnaces.bus1,
                bus2=furnaces.bus2,
                carrier=furnaces.carrier,
                efficiency=efficiency,
                efficiency2=efficiency2,
                capital_cost=0,
                p_nom_extendable=False,
                p_nom=furnaces.p_nom,
                lifetime=lifetime,
                build_year=build_year,
                marginal_cost=mc,
            )

    def add_brownfield_oil(
        n: pypsa.Network,
        df: pd.DataFrame,
        sector: str,
        ratios: pd.DataFrame,
        costs: pd.DataFrame,
    ) -> None:

        # existing efficiency values taken from:
        # https://www.eia.gov/analysis/studies/buildings/equipcosts/pdf/full.pdf

        # will give approximate installed capacity percentage by year
        if sector == "res":
            installed_capacity = RECS_BUILD_YEARS
            lifetime = costs.at["Residential Oil-Fired Furnaces", "lifetime"]
            efficiency = 0.83
        elif sector == "com":
            installed_capacity = CECS_BUILD_YEARS
            lifetime = costs.at["Commercial Oil-Fired Furnaces", "lifetime"]
            efficiency = 0.81

        efficiency2 = costs.at["oil", "co2_emissions"]

        df["bus0"] = df.state + " oil"
        df["bus2"] = df.state + f" {sector}-co2"

        # remove 'heat' or 'cool' ect.. from suffix
        df["carrier"] = df.suffix.map(lambda x: "-".join(x.split("-")[:-1]))
        df["carrier"] = df.carrier + "-lpg-furnace"

        df["ratio"] = df.state.map(ratios.lpg)
        df["p_nom"] = df.p_max.mul(df.ratio).div(100)  # div to convert from %

        marginal_cost_names = [
            x.replace("heat", "lpg-furnace") for x in df.bus1.to_list()
        ]
        marginal_cost = _get_marginal_cost(n, marginal_cost_names)

        start_year = n.investment_periods[0]
        start_year = start_year if start_year <= 2023 else 2023

        for build_year, percent in installed_capacity.items():

            if _already_retired(build_year, lifetime, start_year):
                continue

            furnaces = df.copy()

            furnaces["name"] = (
                furnaces.name + f" existing_{build_year} " + furnaces.carrier
            )
            furnaces["p_nom"] = furnaces.p_nom.mul(percent).div(100).round(2)
            furnaces = furnaces.set_index("name")

            if isinstance(marginal_cost, pd.DataFrame):
                mc = marginal_cost.copy()
                name_mapper = (
                    furnaces["bus1"].str.replace("-heat", "-lpg-furnace").to_dict()
                )
                mc = mc.rename(columns={v: k for k, v in name_mapper.items()})
            else:
                mc = marginal_cost
                assert isinstance(mc, (float, int))

            n.madd(
                "Link",
                furnaces.index,
                bus0=furnaces.bus0,
                bus1=furnaces.bus1,
                bus2=furnaces.bus2,
                carrier=furnaces.carrier,
                efficiency=efficiency,
                efficiency2=efficiency2,
                capital_cost=0,
                p_nom_extendable=False,
                p_nom=furnaces.p_nom,
                lifetime=lifetime,
                build_year=build_year,
                marginal_cost=mc,
            )

    def add_brownfield_elec_furnace(
        n: pypsa.Network,
        df: pd.DataFrame,
        sector: str,
        ratios: pd.DataFrame,
        costs: pd.DataFrame,
    ) -> None:

        # existing efficiency values taken from:
        # https://www.eia.gov/analysis/studies/buildings/equipcosts/pdf/full.pdf

        # will give approximate installed capacity percentage by year
        if sector == "res":
            installed_capacity = RECS_BUILD_YEARS
            lifetime = costs.at["Residential Electric Resistance Heaters", "lifetime"]
            efficiency = 1.0
        elif sector == "com":
            installed_capacity = CECS_BUILD_YEARS
            lifetime = costs.at["Commercial Electric Resistance Heaters", "lifetime"]
            efficiency = 1.0

        df["bus0"] = df.name  # central electricity bus

        # remove 'heat' or 'cool' ect.. from suffix
        df["carrier"] = df.suffix.map(lambda x: "-".join(x.split("-")[:-1]))
        df["carrier"] = df.carrier + "-elec-furnace"

        df["ratio"] = df.state.map(ratios.electricity)
        df["p_nom"] = df.p_max.mul(df.ratio).div(100)  # div to convert from %

        start_year = n.investment_periods[0]
        start_year = start_year if start_year <= 2023 else 2023

        for build_year, percent in installed_capacity.items():

            if _already_retired(build_year, lifetime, start_year):
                continue

            furnaces = df.copy()
            furnaces["name"] = (
                furnaces.name + f" existing_{build_year} " + furnaces.carrier
            )
            furnaces["p_nom"] = furnaces.p_nom.mul(percent).div(100).round(2)
            furnaces = furnaces.set_index("name")

            n.madd(
                "Link",
                furnaces.index,
                bus0=furnaces.bus0,
                bus1=furnaces.bus1,
                carrier=furnaces.carrier,
                efficiency=efficiency,
                capital_cost=0,
                p_nom_extendable=False,
                p_nom=furnaces.p_nom,
                lifetime=lifetime,
                build_year=build_year,
            )

    def add_brownfield_heat_pump(
        n: pypsa.Network,
        df: pd.DataFrame,
        sector: str,
        ratios: pd.DataFrame,
        costs: pd.DataFrame,
    ) -> None:
        """
        Need to pull in existing COP profiles.
        """
        pass

    def add_brownfield_aircon(
        n: pypsa.Network,
        df: pd.DataFrame,
        sector: str,
        ratios: pd.DataFrame,
        costs: pd.DataFrame,
    ) -> None:
        # existing efficiency values taken from:
        # https://www.eia.gov/analysis/studies/buildings/equipcosts/pdf/full.pdf

        # will give approximate installed capacity percentage by year
        if sector == "res":
            installed_capacity = RECS_BUILD_YEARS
            lifetime = costs.at["Residential Central Air Conditioner", "lifetime"]
            efficiency = 3.17  # 12.4 SEER2 converted to COP
        elif sector == "com":
            installed_capacity = CECS_BUILD_YEARS
            lifetime = costs.at["Commercial Rooftop Air Conditioners", "lifetime"]
            efficiency = 3.11  # 10.6 EER converted to COP

        df["bus0"] = df.name  # central electricity bus

        # remove 'heat' or 'cool' ect.. from suffix
        df["carrier"] = df.suffix.map(lambda x: "-".join(x.split("-")[:-1]))
        df["carrier"] = df.carrier + "-air-con"

        df["ratio"] = df.state.map(ratios.electricity)
        df["p_nom"] = df.p_max.mul(df.ratio).div(100)  # div to convert from %

        start_year = n.investment_periods[0]
        start_year = start_year if start_year <= 2023 else 2023

        for build_year, percent in installed_capacity.items():

            if _already_retired(build_year, lifetime, start_year):
                continue

            aircon = df.copy()
            aircon["name"] = aircon.name + f" existing_{build_year} " + aircon.carrier
            aircon["p_nom"] = aircon.p_nom.mul(percent).div(100).round(2)
            aircon = aircon.set_index("name")

            n.madd(
                "Link",
                aircon.index,
                bus0=aircon.bus0,
                bus1=aircon.bus1,
                carrier=aircon.carrier,
                efficiency=efficiency,
                capital_cost=0,
                p_nom_extendable=False,
                p_nom=aircon.p_nom,
                lifetime=lifetime,
                build_year=build_year,
            )

    assert sector in ("res", "com")

    match fuel:
        case "space_heating":
            load = "space-heat"
        case "water_heating":
            load = "water-heat"
        case "heating":
            load = "heat"
        case "cooling":
            load = "cool"
        case _:
            raise NotImplementedError

    df = _get_brownfield_template_df(n, load, sector)
    df["p_nom"] = df.p_max.mul(growth_multiplier)

    if load == "heat":
        add_brownfield_gas_furnace(n, df, sector, ratios, costs)
        add_brownfield_oil(n, df, sector, ratios, costs)
        add_brownfield_elec_furnace(n, df, sector, ratios, costs)
    elif load == "cool":
        add_brownfield_aircon(n, df, sector, ratios, costs)
    elif load == "space-heat":
        add_brownfield_gas_furnace(n, df, sector, ratios, costs)
        add_brownfield_oil(n, df, sector, ratios, costs)
        add_brownfield_elec_furnace(n, df, sector, ratios, costs)
    elif load == "water-heat":
        pass
    else:
        raise NotImplementedError

    # need to add in logic to pull eff profile from new builds
    # add_brownfield_heat_pump(n, df, sector, ratios, costs)


if __name__ == "__main__":
    print(get_residential_stock("./../repo_data/sectors/residential_stock", "cooling"))

    # with open("./../config/config.api.yaml") as file:
    #     yaml_data = yaml.safe_load(file)
    # api = yaml_data["api"]["eia"]

    # print(get_transport_stock(api, 2024))
