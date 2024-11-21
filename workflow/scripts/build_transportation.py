"""
Module for building transportation infrastructure.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import pypsa
from build_heat import _get_dynamic_marginal_costs, get_link_marginal_costs

logger = logging.getLogger(__name__)

from constants_sector import (
    AirTransport,
    BoatTransport,
    RailTransport,
    RoadTransport,
    Transport,
)


def build_transportation(
    n: pypsa.Network,
    costs: pd.DataFrame,
    air: bool = True,
    rail: bool = True,
    boat: bool = True,
    dynamic_pricing: bool = False,
    eia: Optional[str] = None,  # for dynamic pricing
    year: Optional[int] = None,  # for dynamic pricing
) -> None:
    """
    Main funtion to interface with.
    """

    road_suffix = Transport.ROAD.value

    for fuel in ("elec", "lpg"):
        if fuel == "elec":
            add_ev_infrastructure(n, road_suffix)  # attaches at node level
        else:
            add_lpg_infrastructure(n, road_suffix, costs)  # attaches at state level

    if dynamic_pricing:
        assert eia
        assert year
        lpg_cost = _get_dynamic_marginal_costs(n, "lpg", eia, year)
    else:
        logger.warning("Marginal lpg cost set to zero :(")
        lpg_cost = 0  # TODO: No static cost found :(

    road_vehicles = [x.value for x in RoadTransport]
    for vehicle in road_vehicles:
        add_elec_vehicle(n, road_suffix, vehicle, costs)
        add_lpg_vehicle(n, road_suffix, vehicle, costs, lpg_cost)

    if air:
        air_suffix = Transport.AIR.value
        add_lpg_infrastructure(n, air_suffix, costs)

        air_vehicles = [x.value for x in AirTransport]
        for vehicle in air_vehicles:
            add_air(n, air_suffix, vehicle, costs)

    if rail:
        rail_suffix = Transport.RAIL.value
        add_lpg_infrastructure(n, rail_suffix, costs)

        rail_vehicles = [x.value for x in RailTransport]
        for vehicle in rail_vehicles:
            add_rail(n, rail_suffix, vehicle, costs)

    if boat:
        boat_suffix = Transport.BOAT.value
        add_lpg_infrastructure(n, boat_suffix, costs)

        boat_vehicles = [x.value for x in BoatTransport]
        for vehicle in boat_vehicles:
            add_boat(n, boat_suffix, vehicle, costs)


def add_ev_infrastructure(
    n: pypsa.Network,
    vehicle: str,
) -> None:
    """
    Adds bus that all EVs attach to at a node level.
    """

    nodes = n.buses[n.buses.carrier == "AC"]

    n.madd(
        "Bus",
        nodes.index,
        suffix=f" trn-elec-{vehicle}",
        x=nodes.x,
        y=nodes.y,
        country=nodes.country,
        state=nodes.STATE,
        carrier=f"trn-elec-{vehicle}",
    )

    n.madd(
        "Link",
        nodes.index,
        suffix=f" trn-elec-{vehicle}",
        bus0=nodes.index,
        bus1=nodes.index + f" trn-elec-{vehicle}",
        carrier=f"trn-elec-{vehicle}",
        efficiency=1,
        capital_cost=0,
        p_nom_extendable=True,
        lifetime=np.inf,
    )


def add_lpg_infrastructure(
    n: pypsa.Network,
    vehicle: str,
    costs: Optional[pd.DataFrame] = None,
) -> None:
    """
    Adds lpg connections for vehicle type.
    """

    nodes = n.buses[n.buses.carrier == "AC"]

    n.madd(
        "Bus",
        nodes.index,
        suffix=f" trn-lpg-{vehicle}",
        x=nodes.x,
        y=nodes.y,
        country=nodes.country,
        state=nodes.state,
        carrier=f"trn-lpg-{vehicle}",
    )

    nodes["bus0"] = nodes.STATE + " oil"

    if isinstance(costs, pd.DataFrame):
        try:
            efficiency2 = costs.at["oil", "co2_emissions"]
        except KeyError:
            efficiency2 = 0
    else:
        efficiency2 = 0

    n.madd(
        "Link",
        nodes.index,
        suffix=f" trn-lpg-{vehicle}",
        bus0=nodes.bus0,
        bus1=nodes.index + f" trn-lpg-{vehicle}",
        bus2=nodes.STATE + " trn-co2",
        carrier=f"trn-lpg-{vehicle}",
        efficiency=1,
        efficiency2=efficiency2,
        capital_cost=0,
        p_nom_extendable=True,
        lifetime=np.inf,
    )


def add_elec_vehicle(
    n: pypsa.Network,
    vehicle: str,
    mode: str,
    costs: pd.DataFrame,
) -> None:
    """
    Adds electric vehicle to the network.

    Available technology types are:
    - Buses BEV
    - Heavy Duty Trucks BEV
    - Light Duty Cars BEV 100
    - Light Duty Cars BEV 200
    - Light Duty Cars BEV 300
    - Light Duty Cars PHEV 25
    - Light Duty Cars PHEV 50
    - Light Duty Trucks BEV 100
    - Light Duty Trucks BEV 200
    - Light Duty Trucks BEV 300
    - Light Duty Trucks PHEV 25
    - Light Duty Trucks PHEV 50
    - Medium Duty Trucks BEV
    """

    match mode:
        case RoadTransport.LIGHT.value:
            costs_name = "Light Duty Cars BEV 300"
        case RoadTransport.MEDIUM.value:
            costs_name = "Medium Duty Trucks BEV"
        case RoadTransport.HEAVY.value:
            costs_name = "Heavy Duty Trucks BEV"
        case RoadTransport.BUS.value:
            costs_name = "Buses BEV"
        case _:
            raise NotImplementedError

    # 1000s to convert:
    #  $/mile -> $/k-miles
    #  miles/MWh -> k-miles/MWh
    capex = costs.at[costs_name, "capital_cost"] * 1000
    efficiency = costs.at[costs_name, "efficiency"] / 1000
    lifetime = costs.at[costs_name, "lifetime"]

    carrier_name = f"trn-elec-{vehicle}-{mode}"

    loads = n.loads[n.loads.carrier == carrier_name]

    vehicles = pd.DataFrame(index=loads.bus)
    vehicles.index = vehicles.index.map(
        lambda x: x.split(f" trn-elec-{vehicle}-{mode}")[0],
    )
    vehicles["bus0"] = vehicles.index + f" trn-elec-{vehicle}"
    vehicles["bus1"] = vehicles.index + f" trn-elec-{vehicle}-{mode}"
    vehicles["carrier"] = f"trn-elec-{vehicle}-{mode}"

    n.madd(
        "Link",
        vehicles.index,
        suffix=f" trn-elec-{vehicle}-{mode}",
        bus0=vehicles.bus0,
        bus1=vehicles.bus1,
        carrier=vehicles.carrier,
        efficiency=efficiency,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
        marginal_cost=0,
    )


def add_lpg_vehicle(
    n: pypsa.Network,
    vehicle: str,
    mode: str,
    costs: pd.DataFrame,
    marginal_cost: Optional[pd.DataFrame | float] = None,
) -> None:
    """
    Adds electric vehicle to the network.

    Available technology types are:
    - Light Duty Cars ICEV
    - Light Duty Trucks ICEV
    - Medium Duty Trucks ICEV
    - Heavy Duty Trucks ICEV
    - Buses ICEV
    """

    match mode:
        case RoadTransport.LIGHT.value:
            costs_name = "Light Duty Cars ICEV"
        case RoadTransport.MEDIUM.value:
            costs_name = "Medium Duty Trucks ICEV"
        case RoadTransport.HEAVY.value:
            costs_name = "Heavy Duty Trucks ICEV"
        case RoadTransport.BUS.value:
            costs_name = "Buses ICEV"
        case _:
            raise NotImplementedError

    # 1000s to convert:
    #  $/mile -> $/k-miles
    #  miles/MWh -> k-miles/MWh

    capex = costs.at[costs_name, "capital_cost"] * 1000
    efficiency = costs.at[costs_name, "efficiency"] / 1000
    lifetime = costs.at[costs_name, "lifetime"]

    carrier_name = f"trn-lpg-{vehicle}-{mode}"

    loads = n.loads[n.loads.carrier == carrier_name]

    vehicles = pd.DataFrame(index=loads.bus)
    vehicles["state"] = vehicles.index.map(n.buses.STATE)
    vehicles.index = vehicles.index.map(
        lambda x: x.split(f" trn-lpg-{vehicle}-{mode}")[0],
    )
    vehicles["bus0"] = vehicles.index + f" trn-lpg-{vehicle}"
    vehicles["bus1"] = vehicles.index + f" trn-lpg-{vehicle}-{mode}"
    vehicles["carrier"] = f"trn-lpg-{vehicle}-{mode}"

    if isinstance(marginal_cost, pd.DataFrame):
        assert "state" in vehicles.columns
        mc = get_link_marginal_costs(n, vehicles, marginal_cost)
    elif isinstance(marginal_cost, (int, float)):
        mc = marginal_cost
    elif isinstance(marginal_cost, None):
        mc = 0
    else:
        raise TypeError

    n.madd(
        "Link",
        vehicles.index,
        suffix=f" trn-lpg-{vehicle}-{mode}",
        bus0=vehicles.bus0,
        bus1=vehicles.bus1,
        carrier=vehicles.carrier,
        efficiency=efficiency,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
        marginal_cost=mc,
    )


def add_air(
    n: pypsa.Network,
    vehicle: str,
    mode: str,
    costs: pd.DataFrame,
) -> None:
    """
    Adds air transport to the model.

    NOTE: COSTS ARE CURRENTLY HARD CODED IN FROM 2030. Look at travel indicators here:
    - https://www.eia.gov/outlooks/aeo/data/browser/
    """

    # Assumptions from https://www.nrel.gov/docs/fy18osti/70485.pdf
    wh_per_gallon = 33700  # footnote 24

    capex = 1
    # efficiency = costs.at[costs_name, "efficiency"] / 1000
    #  (seat miles / gallon) * ( 1 gal / 33700 wh) * (1k seat mile / 1000 seat miles) * (1000 * 1000 Wh / MWh)
    efficiency = 76.5 / wh_per_gallon / 1000 * 1000 * 1000
    lifetime = 25

    loads = n.loads[(n.loads.carrier.str.contains("trn-")) & (n.loads.carrier.str.contains(f"{vehicle}-{mode}"))]

    vehicles = pd.DataFrame(index=loads.bus)
    vehicles.index = vehicles.index.map(lambda x: x.split(f" trn-")[0])
    vehicles["bus0"] = vehicles.index + f" trn-lpg-{vehicle}"
    vehicles["bus1"] = vehicles.index + f" trn-lpg-{vehicle}-{mode}"
    vehicles["carrier"] = f"trn-lpg-{vehicle}-{mode}"

    n.madd(
        "Link",
        vehicles.index,
        suffix=f" trn-lpg-{vehicle}-{mode}",
        bus0=vehicles.bus0,
        bus1=vehicles.bus1,
        carrier=vehicles.carrier,
        efficiency=efficiency,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
    )


def add_boat(
    n: pypsa.Network,
    vehicle: str,
    mode: str,
    costs: pd.DataFrame,
) -> None:
    """
    Adds boat transport to the model.

    NOTE: COSTS ARE CURRENTLY HARD CODED IN FROM 2030. Look at travel indicators here:
    - https://www.eia.gov/outlooks/aeo/data/browser/
    """

    # efficiency = costs.at[costs_name, "efficiency"] / 1000
    # base efficiency is 5 ton miles per thousand Btu
    # 1 kBTU / 0.000293 MWh
    efficiency = 5 / 0.000293 / 1000
    lifetime = 25
    capex = 1

    loads = n.loads[(n.loads.carrier.str.contains("trn-")) & (n.loads.carrier.str.contains(f"{vehicle}-{mode}"))]

    vehicles = pd.DataFrame(index=loads.bus)
    vehicles.index = vehicles.index.map(lambda x: x.split(f" trn-")[0])
    vehicles["bus0"] = vehicles.index + f" trn-lpg-{vehicle}"
    vehicles["bus1"] = vehicles.index + f" trn-lpg-{vehicle}-{mode}"
    vehicles["carrier"] = f"trn-lpg-{vehicle}-{mode}"

    n.madd(
        "Link",
        vehicles.index,
        suffix=f" trn-lpg-{vehicle}-{mode}",
        bus0=vehicles.bus0,
        bus1=vehicles.bus1,
        carrier=vehicles.carrier,
        efficiency=efficiency,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
    )


def add_rail(
    n: pypsa.Network,
    vehicle: str,
    mode: str,
    costs: pd.DataFrame,
) -> None:
    """
    Adds rail (shipping and passenger) transport to the model.

    NOTE: COSTS ARE CURRENTLY HARD CODED IN FROM 2030. Look at travel indicators here:
    - https://www.eia.gov/outlooks/aeo/data/browser/
    """

    match mode:
        case RailTransport.SHIPPING.value:
            # efficiency = costs.at[costs_name, "efficiency"] / 1000
            # base efficiency is 3.4 ton miles per thousand Btu
            # 1 kBTU / 0.000293 MWh
            efficiency = 3.4 / 0.000293 / 1000
            lifetime = 25
            capex = 1
        case RailTransport.PASSENGER.value:
            # efficiency = costs.at[costs_name, "efficiency"] / 1000
            # base efficiency is 1506 BTU / Passenger Mile
            # https://www.amtrak.com/content/dam/projects/dotcom/english/public/documents/environmental1/Amtrak-Sustainability-Report-FY21.pdf
            efficiency = 1506 / 3.412e6 * 1000  # MWh / k passenger miles
            lifetime = 25
            capex = 1
        case _:
            logger.warning(f"No cost params set for {mode}")
            efficiency = 1
            lifetime = 1
            capex = 0

    loads = n.loads[(n.loads.carrier.str.contains("trn-")) & (n.loads.carrier.str.contains(f"{vehicle}-{mode}"))]

    vehicles = pd.DataFrame(index=loads.bus)
    vehicles.index = vehicles.index.map(lambda x: x.split(f" trn-")[0])
    vehicles["bus0"] = vehicles.index + f" trn-lpg-{vehicle}"
    vehicles["bus1"] = vehicles.index + f" trn-lpg-{vehicle}-{mode}"
    vehicles["carrier"] = f"trn-lpg-{vehicle}-{mode}"

    n.madd(
        "Link",
        vehicles.index,
        suffix=f" trn-lpg-{vehicle}-{mode}",
        bus0=vehicles.bus0,
        bus1=vehicles.bus1,
        carrier=vehicles.carrier,
        efficiency=efficiency,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
    )


def apply_exogenous_ev_policy(n: pypsa.Network, policy: pd.DataFrame) -> None:
    """
    If transport investment is exogenous, applies policy to control loads.

    At this point, all road-vehicle loads are represented as if the entire load is
    met through that fuel type (ie. if there is elec and lpg load, there will
    be 2x the amount of load in the system). This function will adjust the
    amount of load attributed to each fuel.

    The EFS ev policies come from:
    - Figure 6.3 at https://www.nrel.gov/docs/fy18osti/71500.pdf
    - Sheet 6.3 at https://data.nrel.gov/submissions/90
    """

    vehicle_mapper = {
        "light_duty": RoadTransport.LIGHT.value,
        "med_duty": RoadTransport.MEDIUM.value,
        "heavy_duty": RoadTransport.HEAVY.value,
        "bus": RoadTransport.BUS.value,
    }

    abrev = Transport.ROAD.value

    adjusted_loads = []

    for vehicle in policy.columns:  # name of vehicle type
        for period in n.investment_periods:
            ev_share = policy.at[period, vehicle]
            for fuel in ("elec", "lpg"):

                # adjust load value
                load_names = [x for x in n.loads.index if x.endswith(f"trn-{fuel}-{abrev}-{vehicle_mapper[vehicle]}")]
                df = n.loads_t.p_set.loc[period,][load_names]
                multiplier = ev_share if fuel == "elec" else (100 - ev_share)
                df *= multiplier / 100  # divide by 100 to get rid of percent

                # reapply period index level
                df["period"] = period
                df = df.set_index(["period", df.index])  # df.index is snapshots

                adjusted_loads.append(df)

    adjusted = pd.concat(adjusted_loads, axis=1)

    adjusted_loads = adjusted.columns
    n.loads_t.p_set[adjusted_loads] = adjusted[adjusted_loads]
