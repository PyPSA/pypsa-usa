import logging  # noqa: D100
from typing import Any

import pandas as pd
import pypsa
from constants import NG_MWH_2_MMCF
from eia import Trade

logger = logging.getLogger(__name__)


def add_demand_response_constraint(n, config, sector_study):
    """Add demand response capacity constraint."""

    def add_capacity_constraint(
        n: pypsa.Network,
        shift: float,  # per_unit
    ):
        """Add limit on deferable load. No need for snapshot weights."""
        dr_links = n.links[n.links.carrier == "demand_response"].copy()

        if dr_links.empty:
            logger.info("No demand response links identified.")
            return

        deferrable_links = dr_links[dr_links.index.str.endswith("-discharger")]
        deferrable_loads = n.loads[n.loads.bus.isin(deferrable_links.bus1)]

        lhs = n.model["Link-p"].loc[:, deferrable_links.index].groupby(deferrable_links.bus1).sum()
        rhs = n.loads_t["p_set"][deferrable_loads.index].mul(shift).round(2)
        rhs.columns.name = "bus1"
        rhs = rhs.rename(columns={x: x.strip(" AC") for x in rhs})

        # force rhs to be same order as lhs
        # idk why but coordinates were not aligning and this gets around that
        bus_order = lhs.vars.bus1.data
        rhs = rhs[bus_order.tolist()]

        n.model.add_constraints(lhs <= rhs.T, name="demand_response_capacity")

    def add_sector_capacity_constraint(
        n: pypsa.Network,
        shift: float,  # per_unit
    ):
        """Add limit on deferable load. No need for snapshot weights."""
        dr_links = n.links[n.links.carrier == "demand_response"].copy()

        if dr_links.empty:
            logger.info("No demand response links identified.")
            return

        inflow_links = dr_links[dr_links.index.str.endswith("-discharger")]
        inflow = n.model["Link-p"].loc[:, inflow_links.index].groupby(inflow_links.bus1).sum()
        inflow = inflow.rename({"bus1": "Bus"})  # align coordinate names

        outflow_links = n.links[n.links.bus0.isin(inflow_links.bus1) & ~n.links.carrier.str.endswith("-dr")]
        outflow = n.model["Link-p"].loc[:, outflow_links.index].groupby(outflow_links.bus0).sum()
        outflow = outflow.rename({"bus0": "Bus"})  # align coordinate names

        lhs = outflow.mul(shift) - inflow
        rhs = 0

        n.model.add_constraints(
            lhs >= rhs,
            name="demand_response_capacity",
        )

    dr_config = config["electricity"].get("demand_response", {})

    shift = dr_config.get("shift", 0)

    # seperate, as the electrical constraint can directly apply to the load,
    # while the sector constraint has to apply to the power flows out of the bus
    if sector_study:
        fn = add_sector_capacity_constraint
    else:
        fn = add_capacity_constraint

    if isinstance(shift, str):
        if shift == "inf":
            pass
        else:
            logger.error(f"Unknown arguement of {shift} for DR")
            raise ValueError(shift)
    elif isinstance(shift, int | float):
        if shift < 0.001:
            logger.info("Demand response not enabled")
        else:
            fn(n, shift)
    else:
        logger.error(f"Unknown arguement of {shift} for DR")
        raise ValueError(shift)


def add_sector_co2_constraints(n, config):
    """
    Adds sector co2 constraints.

    Parameters
    ----------
        n : pypsa.Network
        config : dict
    """

    def apply_state_limit(n: pypsa.Network, year: int, state: str, value: float, sector: str | None = None):
        if sector:
            stores = n.stores[
                (n.stores.index.str.startswith(state))
                & ((n.stores.index.str.endswith(f"{sector}-co2")) | (n.stores.index.str.endswith(f"{sector}-ch4")))
            ].index
            name = f"GlobalConstraint-co2_limit-{year}-{state}-{sector}"
            if stores.empty:
                log_statement = f"No co2 stores found for {state} {year} {sector}"
            else:
                log_statement = f"Adding {state} {sector} co2 Limit in {year} of"
        else:
            stores = n.stores[
                (n.stores.index.str.startswith(state))
                & ((n.stores.index.str.endswith("-co2")) | (n.stores.index.str.endswith("-ch4")))
            ].index
            name = f"GlobalConstraint-co2_limit-{year}-{state}"
            if stores.empty:
                log_statement = f"No co2 stores found for {state} {year}"
            else:
                log_statement = f"Adding {state} co2 Limit in {year} of"

        if stores.empty:
            logger.warning(log_statement)
            return

        lhs = n.model["Store-e"].loc[:, stores].sel(snapshot=n.snapshots[-1]).sum(dim="Store")
        rhs = value  # value in T CO2

        n.model.add_constraints(lhs <= rhs, name=name)

        logger.info(f"{log_statement} {rhs * 1e-6} MMT CO2")

    def apply_national_limit(n: pypsa.Network, year: int, value: float, sector: str | None = None):
        """For every snapshot, sum of co2 and ch4 must be less than limit."""
        if sector:
            stores = n.stores[
                ((n.stores.index.str.endswith(f"{sector}-co2")) | (n.stores.index.str.endswith(f"{sector}-ch4")))
            ].index
            name = f"co2_limit-{year}-{sector}"
            log_statement = f"Adding national {sector} co2 Limit in {year} of"
        else:
            stores = n.stores[((n.stores.index.str.endswith("-co2")) | (n.stores.index.str.endswith("-ch4")))].index
            name = f"co2_limit-{year}"
            log_statement = f"Adding national co2 Limit in {year} of"

        if stores.empty:
            logger.warning(f"No co2 stores found for USA {year} {sector}")
            return

        lhs = n.model["Store-e"].loc[:, stores].sel(snapshot=n.snapshots[-1]).sum(dim="Store")
        rhs = value  # value in T CO2

        n.model.add_constraints(lhs <= rhs, name=name)

        logger.info(f"{log_statement} {rhs * 1e-6} MMT CO2")

    try:
        f = config["sector"]["co2"]["policy"]
    except KeyError:
        logger.error("No co2 policy constraint file found")
        return

    df = pd.read_csv(f)

    if df.empty:
        logger.warning("No co2 policies applied")
        return

    sectors = df.sector.unique()

    for sector in sectors:
        df_sector = df[df.sector == sector]
        states = df_sector.state.unique()

        for state in states:
            df_state = df_sector[df_sector.state == state]
            years = [x for x in df_state.year.unique() if x in n.investment_periods]

            if not years:
                logger.warning(
                    f"No co2 policies applied for {sector} due to no defined years",
                )
                continue

            for year in years:
                df_limit = df_state[df_state.year == year].reset_index(drop=True)
                assert df_limit.shape[0] == 1

                # results calcualted in T CO2, policy given in MMT CO2
                value = df_limit.loc[0, "co2_limit_mmt"] * 1e6

                if state.lower() == "all":
                    if sector == "all":
                        apply_national_limit(n, year, value)
                    else:
                        apply_national_limit(n, year, value, sector)
                else:
                    if sector == "all":
                        apply_state_limit(n, year, state, value)
                    else:
                        apply_state_limit(n, year, state, value, sector)


def add_cooling_heat_pump_constraints(n, config):
    """
    Adds constraints to the cooling heat pumps.

    These constraints allow HPs to be used to meet both heating and cooling
    demand within a single timeslice while respecting capacity limits.
    Since we are aggregating (and not modelling individual units)
    this should be fine.

    Two seperate constraints are added:
    - Constrains the cooling HP capacity to equal the heating HP capacity. Since the
    cooling hps do not have a capital cost, this will not effect objective cost
    - Constrains the total generation of Heating and Cooling HPs at each time slice
    to be less than or equal to the max generation of the heating HP. Note, that both
    the cooling and heating HPs have the same COP
    """

    def add_hp_capacity_constraint(n, hp_type):
        assert hp_type in ("ashp", "gshp")

        heating_hps = n.links[n.links.index.str.endswith(hp_type)].index
        if heating_hps.empty:
            return
        cooling_hps = n.links[n.links.index.str.endswith(f"{hp_type}-cool")].index

        assert len(heating_hps) == len(cooling_hps)

        lhs = n.model["Link-p_nom"].loc[heating_hps] - n.model["Link-p_nom"].loc[cooling_hps]
        rhs = 0

        n.model.add_constraints(lhs == rhs, name=f"Link-{hp_type}_cooling_capacity")

    def add_hp_generation_constraint(n, hp_type):
        heating_hps = n.links[n.links.index.str.endswith(hp_type)].index
        if heating_hps.empty:
            return
        cooling_hps = n.links[n.links.index.str.endswith(f"{hp_type}-cooling")].index

        heating_hp_p = n.model["Link-p"].loc[:, heating_hps]
        cooling_hp_p = n.model["Link-p"].loc[:, cooling_hps]

        heating_hps_cop = n.links_t["efficiency"][heating_hps]
        cooling_hps_cop = n.links_t["efficiency"][cooling_hps]

        heating_hps_gen = heating_hp_p.mul(heating_hps_cop)
        cooling_hps_gen = cooling_hp_p.mul(cooling_hps_cop)

        lhs = heating_hps_gen + cooling_hps_gen

        heating_hp_p_nom = n.model["Link-p_nom"].loc[heating_hps]
        max_gen = heating_hp_p_nom.mul(heating_hps_cop)

        rhs = max_gen

        n.model.add_constraints(lhs <= rhs, name=f"Link-{hp_type}_cooling_generation")

    for hp_type in ("ashp", "gshp"):
        add_hp_capacity_constraint(n, hp_type)
        add_hp_generation_constraint(n, hp_type)


def add_gshp_capacity_constraint(n, config, snakemake):
    """
    Constrains gshp capacity based on population and ashp installations.

    This constraint should be added if rural/urban sectors are combined into
    a single total area. In this case, we need to constrain how much gshp capacity
    can be added to the system.

    For example:
    - If ratio is 0.75 urban and 0.25 rural
    - We want to enforce that at max, only 0.33 unit of GSHP can be installed for every unit of ASHP
    - The constraint is: [ASHP - (urban / rural) * GSHP >= 0]
    - ie. for every unit of GSHP, we need to install 3 units of ASHP
    """
    pop = pd.read_csv(snakemake.input.pop_layout)
    pop["urban_rural_fraction"] = (pop.urban_fraction / pop.rural_fraction).round(2)
    fraction = pop.set_index("name")["urban_rural_fraction"].to_dict()

    ashp = n.links[n.links.index.str.endswith("ashp")].copy()
    gshp = n.links[n.links.index.str.endswith("gshp")].copy()
    if gshp.empty:
        return

    assert len(ashp) == len(gshp)

    gshp["node"] = gshp.bus0.str.split(" ").str[0]
    gshp["urban_rural_fraction"] = gshp.node.map(fraction)

    ashp_capacity = n.model["Link-p_nom"].loc[ashp.index]
    gshp_capacity = n.model["Link-p_nom"].loc[gshp.index]
    gshp_multiplier = gshp["urban_rural_fraction"]

    lhs = ashp_capacity - gshp_capacity.mul(gshp_multiplier.values)
    rhs = 0

    n.model.add_constraints(lhs >= rhs, name="Link-gshp_capacity_ratio")


def add_ng_import_export_limits(n, config, gwh=False):
    def _format_link_name(s: str) -> str:
        states = s.split("-")
        return f"{states[0]} {states[1]} gas"

    def _format_data(
        prod: pd.DataFrame,
        link_suffix: str | None = None,
    ) -> pd.DataFrame:
        df = prod.copy()
        df["link"] = df.state.map(_format_link_name)
        if link_suffix:
            df["link"] = df.link + link_suffix

        # convert mmcf to MWh
        df["value"] = df["value"] * NG_MWH_2_MMCF
        if gwh:
            df["value"] = df["value"] / 1000

        return df[["link", "value"]].rename(columns={"value": "rhs"}).set_index("link")

    def add_import_limits(n, data, constraint, multiplier=None):
        """Sets gas import limit over each year."""
        assert constraint in ("max", "min")

        if not multiplier:
            multiplier = 1

        weights = n.snapshot_weightings.objective

        links = n.links[(n.links.carrier == "gas trade") & (n.links.bus0.str.endswith(" gas trade"))].index.to_list()

        for year in n.investment_periods:
            for link in links:
                try:
                    rhs = data.at[link, "rhs"] * multiplier
                except KeyError:
                    # logger.warning(f"Can not set gas import limit for {link}")
                    continue
                lhs = n.model["Link-p"].mul(weights).sel(snapshot=year, Link=link).sum()

                if constraint == "min":
                    n.model.add_constraints(
                        lhs >= rhs,
                        name=f"ng_limit_import_min-{year}-{link}",
                    )
                else:
                    n.model.add_constraints(
                        lhs <= rhs,
                        name=f"ng_limit_import_max-{year}-{link}",
                    )

    def add_export_limits(n, data, constraint, multiplier=None):
        """Sets maximum export limit over the year."""
        assert constraint in ("max", "min")

        if not multiplier:
            multiplier = 1

        weights = n.snapshot_weightings.objective

        links = n.links[(n.links.carrier == "gas trade") & (n.links.bus0.str.endswith(" gas"))].index.to_list()

        for year in n.investment_periods:
            for link in links:
                try:
                    rhs = data.at[link, "rhs"] * multiplier
                except KeyError:
                    # logger.warning(f"Can not set gas import limit for {link}")
                    continue
                lhs = n.model["Link-p"].mul(weights).sel(snapshot=year, Link=link).sum()

                if constraint == "min":
                    n.model.add_constraints(
                        lhs >= rhs,
                        name=f"ng_limit_export_min-{year}-{link}",
                    )
                else:
                    n.model.add_constraints(
                        lhs <= rhs,
                        name=f"ng_limit_export_max-{year}-{link}",
                    )

    api = config["api"]["eia"]
    year = pd.to_datetime(config["snapshots"]["start"]).year

    # get limits

    import_min = config["sector"]["natural_gas"]["imports"].get("min", 1)
    import_max = config["sector"]["natural_gas"]["imports"].get("max", 1)
    export_min = config["sector"]["natural_gas"]["exports"].get("min", 1)
    export_max = config["sector"]["natural_gas"]["exports"].get("max", 1)

    # to avoid numerical issues, ensure there is a gap between min/max constraints
    if import_max == "inf":
        pass
    elif abs(import_max - import_min) < 0.0001:
        import_min -= 0.001
        import_max += 0.001
        if import_min < 0:
            import_min = 0

    if export_max == "inf":
        pass
    elif abs(export_max - export_min) < 0.0001:
        export_min -= 0.001
        export_max += 0.001
        if export_min < 0:
            export_min = 0

    # import and export dataframes contain the same information, just in different formats
    # ie. imports from one S1 -> S2 are the same as exports from S2 -> S1
    # we use the exports direction to set limits

    convert_2_gwh = True

    # add domestic limits

    trade = Trade("gas", False, "exports", year, api).get_data()
    trade = _format_data(trade, " trade", convert_2_gwh)

    add_import_limits(n, trade, "min", import_min)
    add_export_limits(n, trade, "min", export_min)

    if not import_max == "inf":
        add_import_limits(n, trade, "max", import_max)
    if not export_max == "inf":
        add_export_limits(n, trade, "max", export_max)

    # add international limits

    trade = Trade("gas", True, "exports", year, api).get_data()
    trade = _format_data(trade, " trade", convert_2_gwh)

    add_import_limits(n, trade, "min", import_min)
    add_export_limits(n, trade, "min", export_min)

    if not import_max == "inf":
        add_import_limits(n, trade, "max", import_max)
    if not export_max == "inf":
        add_export_limits(n, trade, "max", export_max)


def add_water_heater_constraints(n, config):
    """Adds constraint so energy to meet water demand must flow through store."""
    links = n.links[(n.links.index.str.contains("-water-")) & (n.links.index.str.contains("-discharger"))]

    link_names = links.index
    store_names = [x.replace("-discharger", "") for x in links.index]

    for period in n.investment_periods:
        # first snapshot does not respect constraint
        e_previous = n.model["Store-e"].loc[period, store_names]
        e_previous = e_previous.roll(timestep=1)
        e_previous = e_previous.mul(n.snapshot_weightings.stores.loc[period])

        p_current = n.model["Link-p"].loc[period, link_names]
        p_current = p_current.mul(n.snapshot_weightings.objective.loc[period])

        lhs = e_previous - p_current
        rhs = 0

        n.model.add_constraints(lhs >= rhs, name=f"water_heater-{period}")


def add_sector_demand_response_constraints(n, config):
    """
    Add demand response equations for individual sectors.

    These constraints are applied at the sector/carrier level. They are
    fundamentally the same as the power sector constraints, tho.
    """

    def add_capacity_constraint(
        n: pypsa.Network,
        sector: str,
        shift: float,  # as a percentage
        carrier: str | None = None,
    ):
        """Adds limit on deferable load.

        No need to multiply out snapshot weights here
        """
        dr_links = n.links[n.links.carrier.str.endswith("-dr") & n.links.carrier.str.startswith(f"{sector}-")].copy()
        constraint_name = f"demand_response_capacity-{sector}"

        if carrier:
            dr_links = dr_links[dr_links.carrier.str.contains(f"-{carrier}-")].copy()
            constraint_name = f"demand_response_capacity-{sector}-{carrier}"

        if dr_links.empty:
            return

        if sector != "trn":
            deferrable_links = dr_links[dr_links.index.str.endswith("-dr-discharger")]

            deferrable_loads = deferrable_links.bus1.unique().tolist()

            lhs = n.model["Link-p"].loc[:, deferrable_links.index].groupby(deferrable_links.bus1).sum()
            rhs = n.loads_t["p_set"][deferrable_loads].mul(shift).div(100).round(2)  # div cause percentage input
            rhs.columns.name = "bus1"

            # force rhs to be same order as lhs
            # idk why but coordinates were not aligning and this gets around that
            bus_order = lhs.vars.bus1.data
            rhs = rhs[bus_order]

            n.model.add_constraints(lhs <= rhs, name=constraint_name)

        # transport dr is at the aggregation bus
        # sum all outgoing capacity and apply the capacity limit to that
        else:
            inflow_links = dr_links[dr_links.index.str.endswith("-dr-discharger")]
            inflow = n.model["Link-p"].loc[:, inflow_links.index].groupby(inflow_links.bus1).sum()
            inflow = inflow.rename({"bus1": "Bus"})  # align coordinate names

            outflow_links = n.links[n.links.bus0.isin(inflow_links.bus1) & ~n.links.carrier.str.endswith("-dr")]
            outflow = n.model["Link-p"].loc[:, outflow_links.index].groupby(outflow_links.bus0).sum()
            outflow = outflow.rename({"bus0": "Bus"})  # align coordinate names

            lhs = outflow.mul(shift).div(100) - inflow
            rhs = 0

            n.model.add_constraints(
                lhs >= rhs,
                name=constraint_name,
            )

    # helper to manage capacity constraint between non-carrier and carrier

    def _apply_constraint(
        n: pypsa.Network,
        sector: str,
        cfg: dict[str, Any],
        carrier: str | None = None,
    ):
        shift = cfg.get("shift", 0)

        if isinstance(shift, str):
            if shift == "inf":
                pass
            else:
                logger.info(f"Unknown arguement of {shift} for {sector} DR")
                raise ValueError(shift)
        elif isinstance(shift, int | float):
            if shift < 0.001:
                logger.info(f"Demand response not enabled for {sector}")
            else:
                add_capacity_constraint(n, sector, shift, carrier)
        else:
            logger.info(f"Unknown arguement of {shift} for {sector} DR")
            raise ValueError(shift)

    # demand response addition starts here

    sectors = ["res", "com", "ind", "trn"]

    for sector in sectors:
        if sector in ["res", "com"]:
            dr_config = config["sector"]["service_sector"].get("demand_response", {})
        elif sector == "trn":
            dr_config = config["sector"]["transport_sector"].get("demand_response", {})
        elif sector == "ind":
            dr_config = config["sector"]["industrial_sector"].get("demand_response", {})
        else:
            raise ValueError

        if not dr_config:
            continue

        by_carrier = dr_config.get("by_carrier", False)

        if by_carrier:
            for carrier, carrier_config in dr_config.items():
                # hacky check to make sure only carriers get passed in
                # the actual constraint should check this as well
                if carrier in ("elec", "heat", "space-heat", "water-heat", "cool"):
                    _apply_constraint(n, sector, carrier_config, carrier)
        else:
            _apply_constraint(n, sector, dr_config)


def add_ev_generation_constraint(n, config, snakemake):
    """Adds a limit to the maximum generation from EVs per mode and year.

    The constraint is:
    - (EV_gen * eff) / dem <= policy (where policy is a percentage giving max gen)
    - EV_gen <= dem * policy / eff

    This is an approximation based on average EV efficiency for that investmenet period. This
    is needed as EV production will be different than LPG production for the same unit input.

    Default limits taken from:
    - (Fig ES2) https://www.nrel.gov/docs/fy18osti/71500.pdf
    - (Sheet 6.3 - high case) https://data.nrel.gov/submissions/90
    """
    mode_mapper = {
        "light_duty": "lgt",
        "med_duty": "med",
        "heavy_duty": "hvy",
        "bus": "bus",
    }

    policy = pd.read_csv(snakemake.input.ev_policy, index_col=0)

    for mode in policy.columns:
        evs = n.links[n.links.carrier == f"trn-elec-veh-{mode_mapper[mode]}"].index
        dem_names = n.loads[n.loads.carrier == f"trn-veh-{mode_mapper[mode]}"].index
        dem = n.loads_t["p_set"][dem_names]

        for investment_period in n.investment_periods:
            ratio = policy.at[investment_period, mode] / 100  # input is percentage
            eff = n.links.loc[evs].efficiency.mean()
            lhs = n.model["Link-p"].loc[investment_period].sel(Link=evs).sum()
            rhs = dem.loc[investment_period].sum().sum() * ratio / eff

            # scale the constraint to help numerically with other constraints
            rhs = rhs / 1000
            lhs = lhs.div(1000)

            n.model.add_constraints(lhs <= rhs, name=f"Link-ev_gen_{mode}_{investment_period}")
