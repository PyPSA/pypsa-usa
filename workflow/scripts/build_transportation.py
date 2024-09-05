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

    for fuel in ("elec", "lpg"):
        if fuel == "elec":
            add_ev_infrastructure(n)  # attaches at node level
        else:
            add_lpg_infrastructure(n, "veh", costs)  # attaches at state level

    if dynamic_pricing:
        assert eia
        assert year
        lpg_cost = _get_dynamic_marginal_costs(n, "lpg", eia, year)
    else:
        logger.warning("Marginal lpg cost set to zero :(")
        lpg_cost = 0  # TODO: No static cost found :(

    for vehicle in ("lgt", "med", "hvy", "bus"):
        add_elec_vehicle(n, vehicle, costs)
        add_lpg_vehicle(n, vehicle, costs, lpg_cost)

    if air:
        add_lpg_infrastructure(n, "air", costs)
        add_air(n, costs)

    if rail:
        add_lpg_infrastructure(n, "rail", costs)
        add_rail(n, costs)

    if boat:
        add_lpg_infrastructure(n, "boat", costs)
        add_boat(n, costs)


def add_ev_infrastructure(n: pypsa.Network) -> None:
    """
    Adds bus that all EVs attach to at a node level.
    """

    nodes = n.buses[n.buses.carrier == "AC"]

    n.madd(
        "Bus",
        nodes.index,
        suffix=" trn-elec-veh",
        x=nodes.x,
        y=nodes.y,
        country=nodes.country,
        state=nodes.STATE,
        carrier="trn-elec-veh",
    )

    n.madd(
        "Link",
        nodes.index,
        suffix=" trn-elec-infra",
        bus0=nodes.index,
        bus1=nodes.index + " trn-elec-veh",
        carrier="trn-elec-veh",
        efficiency=1,
        capital_cost=0,
        p_nom_extendable=True,
        lifetime=np.inf,
    )


def add_lpg_infrastructure(
    n: pypsa.Network,
    vehicle: str,
    costs: Optional[pd.DataFrame] = pd.DataFrame(),
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
        carrier=f"trn-lpg",
    )

    nodes["bus0"] = nodes.STATE + " oil"

    try:
        efficiency2 = costs.at["oil", "co2_emissions"]
    except KeyError:
        efficiency2 = 0

    n.madd(
        "Link",
        nodes.index,
        suffix=f" trn-lpg-{vehicle}",
        bus0=nodes.bus0,
        bus1=nodes.index + f" trn-lpg-{vehicle}",
        bus2=nodes.STATE + " trn-co2",
        carrier=f"trn-{vehicle}",
        efficiency=1,
        efficiency2=efficiency2,
        capital_cost=0,
        p_nom_extendable=True,
        lifetime=np.inf,
    )


def add_elec_vehicle(
    n: pypsa.Network,
    vehicle: str,
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

    match vehicle:
        case "lgt":
            costs_name = "Light Duty Cars BEV 300"
        case "med":
            costs_name = "Medium Duty Trucks BEV"
        case "hvy":
            costs_name = "Heavy Duty Trucks BEV"
        case "bus":
            costs_name = "Buses BEV"
        case _:
            raise NotImplementedError

    # 1000s to convert:
    #  $/mile -> $/k-miles
    #  miles/MWh -> k-miles/MWh
    capex = costs.at[costs_name, "capital_cost"] * 1000
    efficiency = costs.at[costs_name, "efficiency"] / 1000
    lifetime = costs.at[costs_name, "lifetime"]

    carrier_name = f"trn-elec-{vehicle}"

    loads = n.loads[n.loads.carrier == carrier_name]

    vehicles = pd.DataFrame(index=loads.bus)
    vehicles.index = vehicles.index.map(lambda x: x.split(f" trn-elec-{vehicle}")[0])
    vehicles["bus0"] = vehicles.index + " trn-elec-veh"
    vehicles["bus1"] = vehicles.index + f" trn-elec-{vehicle}"
    vehicles["carrier"] = f"trn-elec-{vehicle}"

    n.madd(
        "Link",
        vehicles.index,
        suffix=f" trn-elec-{vehicle}",
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

    match vehicle:
        case "lgt":
            costs_name = "Light Duty Cars ICEV"
        case "med":
            costs_name = "Medium Duty Trucks ICEV"
        case "hvy":
            costs_name = "Heavy Duty Trucks ICEV"
        case "bus":
            costs_name = "Buses ICEV"
        case _:
            raise NotImplementedError

    # 1000s to convert:
    #  $/mile -> $/k-miles
    #  miles/MWh -> k-miles/MWh

    capex = costs.at[costs_name, "capital_cost"] * 1000
    efficiency = costs.at[costs_name, "efficiency"] / 1000
    lifetime = costs.at[costs_name, "lifetime"]

    carrier_name = f"trn-lpg-{vehicle}"

    loads = n.loads[n.loads.carrier == carrier_name]

    vehicles = pd.DataFrame(index=loads.bus)
    vehicles["state"] = vehicles.index.map(n.buses.STATE)
    vehicles.index = vehicles.index.map(lambda x: x.split(f" trn-lpg-{vehicle}")[0])
    vehicles["bus0"] = vehicles.index + " trn-lpg-veh"
    vehicles["bus1"] = vehicles.index + f" trn-lpg-{vehicle}"
    vehicles["carrier"] = f"trn-lpg-{vehicle}"

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
        suffix=f" trn-lpg-{vehicle}",
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

    loads = n.loads[
        (n.loads.carrier.str.contains("trn-"))
        & (n.loads.carrier.str.contains("air-psg"))
    ]

    vehicles = pd.DataFrame(index=loads.bus)
    vehicles.index = vehicles.index.map(lambda x: x.split(f" trn-")[0])
    vehicles["bus0"] = vehicles.index + " trn-lpg-air"
    vehicles["bus1"] = vehicles.index + f" trn-lpg-air-psg"
    vehicles["carrier"] = f"trn-lpg-air"

    n.madd(
        "Link",
        vehicles.index,
        suffix=f" trn-lpg-air-psg",
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

    loads = n.loads[
        (n.loads.carrier.str.contains("trn-"))
        & (n.loads.carrier.str.contains("boat-ship"))
    ]

    vehicles = pd.DataFrame(index=loads.bus)
    vehicles.index = vehicles.index.map(lambda x: x.split(f" trn-")[0])
    vehicles["bus0"] = vehicles.index + " trn-lpg-boat"
    vehicles["bus1"] = vehicles.index + f" trn-lpg-boat-ship"
    vehicles["carrier"] = f"trn-lpg-boat"

    n.madd(
        "Link",
        vehicles.index,
        suffix=f" trn-lpg-boat-ship",
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
    costs: pd.DataFrame,
) -> None:
    """
    Adds rail (shipping and passenger) transport to the model.

    NOTE: COSTS ARE CURRENTLY HARD CODED IN FROM 2030. Look at travel indicators here:
    - https://www.eia.gov/outlooks/aeo/data/browser/
    """

    def add_rail_shipping(
        n: pypsa.Network,
        costs: pd.DataFrame,
    ) -> None:

        # efficiency = costs.at[costs_name, "efficiency"] / 1000
        # base efficiency is 3.4 ton miles per thousand Btu
        # 1 kBTU / 0.000293 MWh
        efficiency = 3.4 / 0.000293 / 1000
        lifetime = 25
        capex = 1

        loads = n.loads[
            (n.loads.carrier.str.contains("trn-"))
            & (n.loads.carrier.str.contains("rail-ship"))
        ]

        vehicles = pd.DataFrame(index=loads.bus)
        vehicles.index = vehicles.index.map(lambda x: x.split(f" trn-")[0])
        vehicles["bus0"] = vehicles.index + " trn-lpg-rail"
        vehicles["bus1"] = vehicles.index + f" trn-lpg-rail-ship"
        vehicles["carrier"] = f"trn-lpg-rail"

        n.madd(
            "Link",
            vehicles.index,
            suffix=f" trn-lpg-rail-ship",
            bus0=vehicles.bus0,
            bus1=vehicles.bus1,
            carrier=vehicles.carrier,
            efficiency=efficiency,
            capital_cost=capex,
            p_nom_extendable=True,
            lifetime=lifetime,
        )

    def add_rail_passenger(
        n: pypsa.Network,
        costs: pd.DataFrame,
    ) -> None:

        # efficiency = costs.at[costs_name, "efficiency"] / 1000
        # base efficiency is 1506 BTU / Passenger Mile
        # https://www.amtrak.com/content/dam/projects/dotcom/english/public/documents/environmental1/Amtrak-Sustainability-Report-FY21.pdf
        efficiency = 1506 / 3.412e6 * 1000  # MWh / k passenger miles
        lifetime = 25
        capex = 1

        loads = n.loads[
            (n.loads.carrier.str.contains("trn-"))
            & (n.loads.carrier.str.contains("rail-psg"))
        ]

        vehicles = pd.DataFrame(index=loads.bus)
        vehicles.index = vehicles.index.map(lambda x: x.split(f" trn-")[0])
        vehicles["bus0"] = vehicles.index + " trn-lpg-rail"
        vehicles["bus1"] = vehicles.index + f" trn-lpg-rail-psg"
        vehicles["carrier"] = f"trn-lpg-rail"

        n.madd(
            "Link",
            vehicles.index,
            suffix=f" trn-lpg-rail-psg",
            bus0=vehicles.bus0,
            bus1=vehicles.bus1,
            carrier=vehicles.carrier,
            efficiency=efficiency,
            capital_cost=capex,
            p_nom_extendable=True,
            lifetime=lifetime,
        )

    add_rail_shipping(n, costs)
    add_rail_passenger(n, costs)


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
        "light_duty": "lgt",
        "med_duty": "med",
        "heavy_duty": "hvy",
        "bus": "bus",
    }

    adjusted_loads = []

    for vehicle in policy.columns:  # name of vehicle type
        for period in n.investment_periods:
            ev_share = policy.at[period, vehicle]
            for fuel in ("elec", "lpg"):

                # adjust load value
                load_names = [
                    x
                    for x in n.loads.index
                    if x.endswith(f"trn-{fuel}-{vehicle_mapper[vehicle]}")
                ]
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
