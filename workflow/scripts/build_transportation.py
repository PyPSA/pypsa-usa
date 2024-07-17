"""
Module for building transportation infrastructure.
"""

from typing import Optional

import numpy as np
import pandas as pd
import pypsa
from add_electricity import load_costs


def build_transportation(
    n: pypsa.Network,
    costs: pd.DataFrame,
) -> None:
    """
    Main funtion to interface with.
    """

    for fuel in ("elec", "lpg"):
        if fuel == "elec":
            add_ev_infrastructure(n)  # attaches at node level
        else:
            add_fossil_infrastructure(n, fuel, costs)  # attaches at state level

    for vehicle in ("lgt", "med", "hvy", "bus"):
        add_elec_vehicle(n, vehicle, costs)
        add_lpg_vehicle(n, vehicle, costs)


def add_ev_infrastructure(n: pypsa.Network) -> None:
    """
    Adds bus that all EVs attach to at a node level.
    """

    nodes = n.buses[n.buses.carrier == "AC"]

    n.madd(
        "Bus",
        nodes.index,
        suffix=" trn-elec",
        x=nodes.x,
        y=nodes.y,
        country=nodes.country,
        state=nodes.STATE,
        carrier="trn-elec",
    )

    n.madd(
        "Link",
        nodes.index,
        suffix=" trn elec-infra",
        bus0=nodes.index,
        bus1=nodes.index + " trn-elec",
        carrier="trn-elec",
        efficiency=1,
        capital_cost=0,
        p_nom_extendable=True,
        lifetime=np.inf,
    )


def add_fossil_infrastructure(
    n: pypsa.Network,
    carrier: str,
    costs: Optional[pd.DataFrame] = pd.DataFrame(),
) -> None:
    """
    Adds bus that all fossil vehicles attach to at a state level.
    """

    nodes = n.buses[n.buses.carrier == "AC"]

    n.madd(
        "Bus",
        nodes.index,
        suffix=f" trn-{carrier}",
        x=nodes.x,
        y=nodes.y,
        country=nodes.country,
        state=nodes.state,
        carrier=f"trn-{carrier}",
    )

    # alings oil and lpg
    corrected_carrier = carrier if not carrier == "lpg" else "oil"

    nodes["bus0"] = nodes.STATE + f" {corrected_carrier}"

    try:
        efficiency2 = costs.at["oil", "co2_emissions"]
    except KeyError:
        efficiency2 = 0

    n.madd(
        "Link",
        nodes.index,
        suffix=f" trn {carrier}-infra",
        bus0=nodes.bus0,
        bus1=nodes.index + f" trn-{carrier}",
        bus2=nodes.STATE + " trn-co2",
        carrier=f"trn-{carrier}",
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
    vehicles["bus0"] = vehicles.index + " trn-elec"
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

    # 1000s to convert:
    #  $/mile -> $/k-miles
    #  miles/MWh -> k-miles/MWh

    capex = costs.at[costs_name, "capital_cost"] * 1000
    efficiency = costs.at[costs_name, "efficiency"] / 1000
    lifetime = costs.at[costs_name, "lifetime"]

    carrier_name = f"trn-lpg-{vehicle}"

    loads = n.loads[n.loads.carrier == carrier_name]

    vehicles = pd.DataFrame(index=loads.bus)
    vehicles.index = vehicles.index.map(lambda x: x.split(f" trn-lpg-{vehicle}")[0])
    vehicles["bus0"] = vehicles.index + " trn-lpg"
    vehicles["bus1"] = vehicles.index + f" trn-lpg-{vehicle}"
    vehicles["carrier"] = f"trn-lpg-{vehicle}"

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
    )


def apply_exogenous_ev_policy(n: pypsa.Network, policy: pd.DataFrame) -> None:
    """
    If transport investment is exogenous, applies policy to control loads.

    At this point, all vehicle loads are represented as if the entire load is
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