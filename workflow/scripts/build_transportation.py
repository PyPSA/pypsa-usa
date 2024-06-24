"""
Module for building transportation infrastructure.
"""

import numpy as np
import pandas as pd
import pypsa
from add_electricity import load_costs


def build_transportation(
    n: pypsa.Network,
    costs_path: str,
) -> None:
    """
    Main funtion to interface with.
    """

    costs = load_costs(costs_path)

    for vehicle in ("lgt", "med", "hvy", "bus"):
        add_elec_vehicle(n, vehicle, costs)
        add_lpg_vehicle(n, vehicle, costs)


def add_ev_infrastructure(n: pypsa.Network) -> None:
    """
    Adds node that all EVs attach to.
    """

    nodes = n.buses[n.buses.carrier == "AC"]

    n.madd(
        "Bus",
        nodes.index,
        suffix=" trn-elec",
        x=nodes.x,
        y=nodes.y,
        country=nodes.country,
        state=nodes.state,
        carrier="trn-elec",
    )

    n.madd(
        "Link",
        nodes.index,
        suffix=" trn-elec",
        bus0=nodes.index,
        bus1=nodes.index + " trn-elec",
        carrier="trn-elec",
        efficiency=1,
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

    capex = costs.at[costs_name, "capex"]
    efficiency = costs.at[costs_name, "efficiency"]
    lifetime = costs.at[costs_name, "lifetime"]

    carrier_name = f"trn-elec-{vehicle}"

    loads = n.loads[n.loads.carrier == carrier_name]

    vehicles = pd.DataFrame(index=loads.bus)
    vehicles["bus0"] = vehicles.index
    vehicles["bus1"] = vehicles.index.map(lambda x: f"{x} trn-elec-{vehicle}")
    vehicles["carrier"] = carrier_name

    n.madd(
        "Link",
        vehicles.index,
        suffix=f" trn-elec",
        bus0=vehicles.bus0,
        bus1=vehicles.bus1,
        carrier=vehicles.carrier,
        efficiency=efficiency,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
    )


def add_lpg_vehicle(
    n: pypsa.Network,
    vehicle: str,
    costs: pd.DataFrame,
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

    capex = costs.at[costs_name, "capex"]
    efficiency = costs.at[costs_name, "efficiency"]
    lifetime = costs.at[costs_name, "lifetime"]

    carrier_name = f"trn-lpg-{vehicle}"

    loads = n.loads[n.loads.carrier == carrier_name]

    vehicles = pd.DataFrame(index=loads.bus)
    vehicles["bus0"] = vehicles.index
    vehicles["bus1"] = vehicles.index.map(lambda x: f"{x} trn-lpg-{vehicle}")
    vehicles["carrier"] = carrier_name

    n.madd(
        "Link",
        vehicles.index,
        suffix=f" trn-lpg",
        bus0=vehicles.bus0,
        bus1=vehicles.bus1,
        carrier=vehicles.carrier,
        efficiency=efficiency,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
    )
