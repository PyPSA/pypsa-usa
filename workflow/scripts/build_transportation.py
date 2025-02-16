"""
Module for building transportation infrastructure.
"""

import logging
from typing import Any, Optional

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
    exogenous: bool = False,
    air: bool = True,
    rail: bool = True,
    boat: bool = True,
    dynamic_pricing: bool = False,
    eia: Optional[str] = None,  # for dynamic pricing
    year: Optional[int] = None,  # for dynamic pricing
    must_run_evs: Optional[bool] = None,  # for endogenous EV investment
    dr_config: Optional[dict[str, Any]] = None,
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

    if not exogenous:
        assert isinstance(must_run_evs, bool)
        apply_endogenous_road_investments(n, must_run_evs)

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
        state=nodes.STATE,
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


def add_transport_dr(n: pypsa.Network, vehicle: str, dr_config: dict[str, Any]) -> None:
    """Attachs DR infrastructure at load location"""
    shift = dr_config.get("shift", 0)
    marginal_cost_storage = dr_config.get("marginal_cost", 0)

    if shift == 0:
        logger.info(f"DR not applied to {vehicle} as allowable sift is {shift}")
        return

    if marginal_cost_storage == 0:
        logger.warning(f"No cost applied to demand response for {vehicle}")

    df = n.buses[n.buses.carrier == f"trn-elec-{vehicle}"]
    df["carrier"] = df.carrier + "-dr"

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
    vehicles.index = vehicles.index.map(lambda x: x.split(" trn-")[0])
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
    vehicles.index = vehicles.index.map(lambda x: x.split(" trn-")[0])
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
    vehicles.index = vehicles.index.map(lambda x: x.split(" trn-")[0])
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


def _create_endogenous_buses(n: pypsa.Network) -> None:
    """Creats new bus for grouped endogenous vehicle load"""
    buses = n.buses[
        n.buses.carrier.str.startswith("trn")
        & n.buses.carrier.str.contains("veh")
        & ~n.buses.carrier.str.endswith("veh")
    ].copy()
    buses["veh"] = buses.carrier.map(lambda x: x.split("-")[-1])
    buses["name"] = buses.country + " trn-veh-" + buses.veh
    buses["carrier"] = "trn-veh-" + buses.veh
    buses = buses.drop_duplicates(subset="name")
    buses = buses.set_index("name")

    n.madd(
        "Bus",
        buses.index,
        x=buses.x,
        y=buses.y,
        carrier=buses.carrier,
        STATE=buses.STATE,
        STATE_NAME=buses.STATE_NAME,
        unit=buses.unit,
        interconnect=buses.interconnect,
        state=buses.state,
        country=buses.country,
        reeds_ba=buses.reeds_ba,
        reeds_state=buses.reeds_state,
        nerc_reg=buses.nerc_reg,
        trans_reg=buses.trans_reg,
    )


def _create_endogenous_loads(n: pypsa.Network) -> None:
    """Creates aggregated vehicle load

    - Removes LPG load
    - Transfers EV load to central bus
    """
    loads = n.loads[n.loads.carrier.str.startswith("trn") & n.loads.carrier.str.contains("veh")]
    to_remove = [x for x in loads.index if "-lpg-" in x]
    to_shift = [x for x in loads.index if "-elec-" in x]
    assert (len(to_remove) + len(to_shift)) == len(loads)

    new_name_mapper = {x: x.replace("-elec", "") for x in to_shift}
    new_names = [x for _, x in new_name_mapper.items()]

    # remove LPG loads
    n.mremove(
        "Load",
        to_remove,
    )

    # rename elec loads to general loads
    n.loads_t["p_set"] = n.loads_t["p_set"].rename(columns=new_name_mapper)
    n.loads = n.loads.rename(index=new_name_mapper)

    # transfer elec load buses to general bus
    n.loads.loc[new_names, "bus"] = n.loads.loc[new_names, "bus"].map(new_name_mapper)
    n.loads.loc[new_names, "carrier"] = n.loads.loc[new_names, "carrier"].map(lambda x: x.replace("trn-elec-", "trn-"))
    n.loads.loc[new_names, "carrier"] = n.loads.loc[new_names, "carrier"].map(lambda x: x.replace("trn-lpg-", "trn-"))


def _create_endogenous_links(n: pypsa.Network) -> None:
    """Creates links for LPG and EV to load bus

    Just involves transfering bus1 from exogenous load bus to endogenous load bus
    """
    slicer = (
        n.links.carrier.str.startswith("trn")
        & n.links.carrier.str.contains("veh")
        & ~n.links.carrier.str.endswith("veh")
    )
    n.links.loc[slicer, "bus1"] = n.links.loc[slicer, "bus1"].map(lambda x: x.replace("trn-elec-", "trn-"))
    n.links.loc[slicer, "bus1"] = n.links.loc[slicer, "bus1"].map(lambda x: x.replace("trn-lpg-", "trn-"))


def _remove_exogenous_buses(n: pypsa.Network) -> None:
    """Removes buses that are used for exogenous vehicle loads"""
    # this is super awkward filtering :(
    buses = n.buses[
        (n.buses.index.str.contains("trn-elec-veh") | n.buses.index.str.contains("trn-lpg-veh"))
        & ~(n.buses.index.str.endswith("-veh") | n.buses.index.str.endswith("-veh-store"))
    ].index.to_list()
    n.mremove("Bus", buses)


def _constrain_charing_rates(n: pypsa.Network, must_run_evs: bool) -> None:
    """Applies limits to p_min_pu/p_max_pu on links

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
        p_min_pu = p_min_pu.sub(0.01).clip(lower=0).round(2)
        n.links_t["p_min_pu"] = pd.concat([n.links_t["p_min_pu"], p_min_pu], axis=1)

    p_max_pu = (p_max_pu - p_max_pu.min()) / (p_max_pu.max() - p_max_pu.min())
    p_max_pu = p_max_pu = p_max_pu.add(0.01).clip(upper=1).round(2)
    n.links_t["p_max_pu"] = pd.concat([n.links_t["p_max_pu"], p_max_pu], axis=1)


def apply_endogenous_road_investments(n: pypsa.Network, must_run_evs: bool = False) -> None:
    """Merges EV and LPG load into a single load.

    This function will do the following:
    - Create a new bus that the combined load will apply to
    - Shift the EV load to the new bus. This load will retain the EV profile and full
      magnitude, as this represents all vehicle load in the system. The load is renamed.
    - Remove the LPG load
    - Create two links. 1) from EV to new bus for aggregate load. 2) from LPG to new
      bus for aggregate load
    - Costrain the new EV link to match the vehicle load profile
      (ie. p_nom_min(ev) = p_nom_max(ev) = p_nom_t(load))
    """
    _create_endogenous_buses(n)
    _create_endogenous_loads(n)
    _create_endogenous_links(n)
    _remove_exogenous_buses(n)
    _constrain_charing_rates(n, must_run_evs)


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
