"""Module for building transportation infrastructure."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import pypsa
from build_heat import get_link_marginal_costs
from constants_sector import (
    AirTransport,
    BoatTransport,
    RailTransport,
    RoadTransport,
    Transport,
)

logger = logging.getLogger(__name__)


def build_transportation(
    n: pypsa.Network,
    costs: pd.DataFrame,
    air: bool = True,
    rail: bool = True,
    boat: bool = True,
    must_run_evs: bool | None = None,  # for endogenous EV investment
    dr_config: dict[str, Any] | None = None,
) -> None:
    """Main funtion to interface with."""
    road_suffix = Transport.ROAD.value

    add_ev_infrastructure(n, road_suffix)  # attaches at node level
    add_lpg_infrastructure(n, road_suffix, costs)  # attaches at state level

    # lpg costs tracked at state level
    lpg_cost = 0

    road_vehicles = [x.value for x in RoadTransport]
    for vehicle in road_vehicles:
        add_elec_vehicle(n, road_suffix, vehicle, costs)
        add_lpg_vehicle(n, road_suffix, vehicle, costs, lpg_cost)
    constrain_charing_rates(n, must_run_evs)

    # demand response must happen after exogenous/endogenous split
    if dr_config:
        add_transport_dr(n, road_suffix, dr_config)

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
    """Adds bus that all EVs attach to at a node level."""
    nodes = n.buses[n.buses.carrier == "AC"]

    n.madd(
        "Bus",
        nodes.index,
        suffix=f" trn-elec-{vehicle}",
        x=nodes.x,
        y=nodes.y,
        country=nodes.country,
        STATE=nodes.STATE,
        STATE_NAME=nodes.STATE_NAME,
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
        build_year=n.investment_periods[0],
    )


def add_lpg_infrastructure(
    n: pypsa.Network,
    vehicle: str,
    costs: pd.DataFrame | None = None,
) -> None:
    """Adds lpg connections for vehicle type."""
    nodes = n.buses[n.buses.carrier == "AC"]

    n.madd(
        "Bus",
        nodes.index,
        suffix=f" trn-lpg-{vehicle}",
        x=nodes.x,
        y=nodes.y,
        country=nodes.country,
        state=nodes.STATE,
        carrier=f"trn-lpg-{vehicle}",
    )

    nodes["bus0"] = nodes.STATE + " lpg"

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
        build_year=n.investment_periods[0],
    )


def add_transport_dr(n: pypsa.Network, vehicle: str, dr_config: dict[str, Any]) -> None:
    """Attachs DR infrastructure at load location."""
    shift = dr_config.get("shift", 0)
    marginal_cost_storage = dr_config.get("marginal_cost", 0)

    if shift == 0:
        logger.info(f"DR not applied to {vehicle} as allowable sift is {shift}")
        return

    if marginal_cost_storage == 0:
        logger.warning(f"No cost applied to demand response for {vehicle}")

    df = n.buses[n.buses.carrier == f"trn-elec-{vehicle}"]
    df["carrier"] = df.carrier + "-dr"

    lifetime = np.inf
    build_year = n.investment_periods[0]

    # two buses for forward and backwards load shifting

    n.madd(
        "Bus",
        df.index,
        suffix="-fwd-dr",
        x=df.x,
        y=df.y,
        carrier=df.carrier,
        unit="MWh",
        STATE=df.STATE,
        STATE_NAME=df.STATE_NAME,
    )

    n.madd(
        "Bus",
        df.index,
        suffix="-bck-dr",
        x=df.x,
        y=df.y,
        carrier=df.carrier,
        unit="MWh",
        STATE=df.STATE,
        STATE_NAME=df.STATE_NAME,
    )

    # seperate charging/discharging links to follow conventions

    n.madd(
        "Link",
        df.index,
        suffix="-fwd-dr-charger",
        bus0=df.index,
        bus1=df.index + "-fwd-dr",
        carrier=df.carrier,
        p_nom_extendable=False,
        p_nom=np.inf,
        lifetime=lifetime,
        build_year=build_year,
    )

    n.madd(
        "Link",
        df.index,
        suffix="-fwd-dr-discharger",
        bus0=df.index + "-fwd-dr",
        bus1=df.index,
        carrier=df.carrier,
        p_nom_extendable=False,
        p_nom=np.inf,
        lifetime=lifetime,
        build_year=build_year,
    )

    n.madd(
        "Link",
        df.index,
        suffix="-bck-dr-charger",
        bus0=df.index,
        bus1=df.index + "-bck-dr",
        carrier=df.carrier,
        p_nom_extendable=False,
        p_nom=np.inf,
        lifetime=lifetime,
        build_year=build_year,
    )

    n.madd(
        "Link",
        df.index,
        suffix="-bck-dr-discharger",
        bus0=df.index + "-bck-dr",
        bus1=df.index,
        carrier=df.carrier,
        p_nom_extendable=False,
        p_nom=np.inf,
        lifetime=lifetime,
        build_year=build_year,
    )

    # backward stores have positive marginal cost storage and postive e
    # forward stores have negative marginal cost storage and negative e

    n.madd(
        "Store",
        df.index,
        suffix="-bck-dr",
        bus=df.index + "-bck-dr",
        e_cyclic=True,
        e_nom_extendable=False,
        e_nom=np.inf,
        e_min_pu=0,
        e_max_pu=1,
        carrier=df.carrier,
        marginal_cost_storage=marginal_cost_storage,
        lifetime=lifetime,
        build_year=build_year,
        standing_loss=0,
    )

    n.madd(
        "Store",
        df.index,
        suffix="-fwd-dr",
        bus=df.index + "-fwd-dr",
        e_cyclic=True,
        e_nom_extendable=False,
        e_nom=np.inf,
        e_min_pu=-1,
        e_max_pu=0,
        carrier=df.carrier,
        marginal_cost_storage=marginal_cost_storage * (-1),
        lifetime=lifetime,
        build_year=build_year,
        standing_loss=0,
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
    build_year = n.investment_periods[0]

    carrier_name = f"trn-{vehicle}-{mode}"

    loads = n.loads[n.loads.carrier == carrier_name]

    vehicles = pd.DataFrame(index=loads.bus)
    vehicles.index = vehicles.index.map(
        lambda x: x.split(f" trn-{vehicle}-{mode}")[0],
    )
    vehicles["bus0"] = vehicles.index + f" trn-elec-{vehicle}"
    vehicles["bus1"] = vehicles.index + f" trn-{vehicle}-{mode}"
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
        build_year=build_year,
        marginal_cost=0,
    )


def add_lpg_vehicle(
    n: pypsa.Network,
    vehicle: str,
    mode: str,
    costs: pd.DataFrame,
    marginal_cost: pd.DataFrame | float | None = None,
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
    build_year = n.investment_periods[0]

    carrier_name = f"trn-{vehicle}-{mode}"

    loads = n.loads[n.loads.carrier == carrier_name]

    vehicles = pd.DataFrame(index=loads.bus)
    vehicles["state"] = vehicles.index.map(n.buses.STATE)
    vehicles.index = vehicles.index.map(
        lambda x: x.split(f" trn-{vehicle}-{mode}")[0],
    )
    vehicles["bus0"] = vehicles.index + f" trn-lpg-{vehicle}"
    vehicles["bus1"] = vehicles.index + f" trn-{vehicle}-{mode}"
    vehicles["carrier"] = f"trn-lpg-{vehicle}-{mode}"

    if isinstance(marginal_cost, pd.DataFrame):
        assert "state" in vehicles.columns
        mc = get_link_marginal_costs(n, vehicles, marginal_cost)
    elif isinstance(marginal_cost, int | float):
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
        build_year=build_year,
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
    build_year = n.investment_periods[0]

    loads = n.loads[(n.loads.carrier.str.startswith("trn-")) & (n.loads.carrier.str.endswith(f"{vehicle}-{mode}"))]

    vehicles = pd.DataFrame(index=loads.bus)
    vehicles.index = vehicles.index.map(lambda x: x.split(" trn-")[0])
    vehicles["bus0"] = vehicles.index + f" trn-lpg-{vehicle}"
    vehicles["bus1"] = vehicles.index + f" trn-{vehicle}-{mode}"
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
        build_year=build_year,
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
    build_year = n.investment_periods[0]

    loads = n.loads[(n.loads.carrier.str.startswith("trn-")) & (n.loads.carrier.str.endswith(f"{vehicle}-{mode}"))]

    vehicles = pd.DataFrame(index=loads.bus)
    vehicles.index = vehicles.index.map(lambda x: x.split(" trn-")[0])
    vehicles["bus0"] = vehicles.index + f" trn-lpg-{vehicle}"
    vehicles["bus1"] = vehicles.index + f" trn-{vehicle}-{mode}"
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
        build_year=build_year,
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
            build_year = n.investment_periods[0]
        case RailTransport.PASSENGER.value:
            # efficiency = costs.at[costs_name, "efficiency"] / 1000
            # base efficiency is 1506 BTU / Passenger Mile
            # https://www.amtrak.com/content/dam/projects/dotcom/english/public/documents/environmental1/Amtrak-Sustainability-Report-FY21.pdf
            efficiency = 1506 / 3.412e6 * 1000  # MWh / k passenger miles
            lifetime = 25
            capex = 1
            build_year = n.investment_periods[0]
        case _:
            logger.warning(f"No cost params set for {mode}")
            efficiency = 1
            lifetime = 1
            capex = 1
            build_year = n.investment_periods[0]

    loads = n.loads[(n.loads.carrier.str.startswith("trn-")) & (n.loads.carrier.str.endswith(f"{vehicle}-{mode}"))]

    vehicles = pd.DataFrame(index=loads.bus)
    vehicles.index = vehicles.index.map(lambda x: x.split(" trn-")[0])
    vehicles["bus0"] = vehicles.index + f" trn-lpg-{vehicle}"
    vehicles["bus1"] = vehicles.index + f" trn-{vehicle}-{mode}"
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
        build_year=build_year,
    )


def constrain_charing_rates(n: pypsa.Network, must_run_evs: bool) -> None:
    """Applies limits to p_min_pu/p_max_pu on links.

    must_run_evs:
        True
            Both the p_min_pu and p_max_pu of only EVs are set to match the load profile
        False
            p_max_pu of both EVs and LPGs are set to match the load profile.
            p_min_pu is left unchanged (ie. zero)
    """
    links = n.links[n.links.carrier.str.contains("-veh") & ~n.links.carrier.str.endswith("-veh")]
    evs = links[links.carrier.str.contains("-elec-")]
    lpgs = links[links.carrier.str.contains("-lpg-")]

    # these must be done seperatly, as they share bus1 names
    ev_mapper = evs.reset_index().set_index("bus1")["Link"].to_dict()
    lpg_mapper = lpgs.reset_index().set_index("bus1")["Link"].to_dict()

    assert len(evs) + len(lpgs) == len(links)

    p_max_pu_evs = n.loads_t["p_set"][evs.bus1.tolist()]
    p_max_pu_evs = p_max_pu_evs.rename(columns=ev_mapper)

    if must_run_evs:
        p_min_pu = n.loads_t["p_set"][evs.bus1.tolist()]
        p_min_pu = p_min_pu.rename(columns=ev_mapper)
        p_max_pu = p_max_pu_evs.copy()
    else:
        p_min_pu = pd.DataFrame()
        p_max_pu_lpg = n.loads_t["p_set"][lpgs.bus1.tolist()]
        p_max_pu_lpg = p_max_pu_lpg.rename(columns=lpg_mapper)
        p_max_pu = pd.concat([p_max_pu_evs, p_max_pu_lpg], axis=1)

    # normalize to get profiles
    # add small buffer for computation issues while solving
    if not p_min_pu.empty:
        p_min_pu = (p_min_pu - p_min_pu.min()) / (p_min_pu.max() - p_min_pu.min())
        p_min_pu = p_min_pu.fillna(0)  # if uniform profile, p_min_pu will be nan
        p_min_pu = p_min_pu.sub(0.01).clip(lower=0).round(2)
        n.links_t["p_min_pu"] = pd.concat([n.links_t["p_min_pu"], p_min_pu], axis=1)

    p_max_pu = (p_max_pu - p_max_pu.min()) / (p_max_pu.max() - p_max_pu.min())
    p_max_pu = p_max_pu.fillna(1)  # if uniform profile, p_max_pu will be nan
    p_max_pu = p_max_pu = p_max_pu.add(0.01).clip(upper=1).round(2)
    n.links_t["p_max_pu"] = pd.concat([n.links_t["p_max_pu"], p_max_pu], axis=1)
