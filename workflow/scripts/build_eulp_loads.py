"""
Aggregate the ResStock or ComStock building types of one state into hourly loads.

Electricity is final energy; space heating, water heating and cooling are
useful energy (see :class:`eulp.Eulp`). ResStock reports the delivered loads.
ComStock reports final energy per fuel only, so each fuel's profile is scaled
by a state-level useful-energy factor,

    factor = useful energy / final energy,

summed over the state's baseline buildings from the equipment loads, energies
and efficiencies in the ComStock annual results. The factor is an annual,
stock-weighted value: temperature-dependent COPs are not resolved within the
year.
"""

import logging

import pandas as pd
from _helpers import configure_logging
from eulp import Eulp

logger = logging.getLogger(__name__)

J_TO_KWH = 1 / 3.6e6

# ComStock does not report the output of fuel-fired and electric-resistance
# service water heaters; use the thermal efficiencies of its prototype storage
# water heaters (ASHRAE 90.1 minimum for gas, resistance at unity).
FUEL_WATER_HEATER_EFFICIENCY = 0.80
ELECTRIC_WATER_HEATER_EFFICIENCY = 1.0
# fuel-fired heating coils without a reported efficiency
FUEL_COIL_EFFICIENCY = 0.80
# steam- or fuel-driven absorption chillers (district_heating and other_fuel cooling)
ABSORPTION_COP = 1.0
# electric cooling of a state without DX equipment and no chiller COP (existing commercial AC)
DX_COP_FALLBACK = 3.11

EQUIPMENT = [
    "boiler_load..j",
    "boiler_gas..j",
    "boiler_electric..j",
    "boiler_other_fuel..j",
    "gas_coil_average_efficiency",
    "dx_heating_total_dx_load..j",
    "dx_heating_total_dx_electric..j",
    "dx_heating_total_supplemental_electric..j",
    "dx_heating_total_supplemental_load_electric..j",
    "dx_heating_total_supplemental_gas..j",
    "dx_heating_total_supplemental_load_gas..j",
    "vrf_heating_total_supplemental_electric..j",
    "vrf_heating_total_supplemental_load_electric..j",
    "vrf_heating_total_supplemental_gas..j",
    "vrf_heating_total_supplemental_load_gas..j",
    "wa_hp_heating_load..j",
    "wa_hp_heating_electric..j",
    "heat_pump_heating_total_load..j",
    "heat_pump_heating_total_electric..j",
    "dx_cooling_load..j",
    "dx_cooling_electric..j",
    "wa_hp_cooling_load..j",
    "wa_hp_cooling_electric..j",
    "heat_pump_cooling_total_load..j",
    "heat_pump_cooling_total_electric..j",
    "chiller_average_cop..cop",
    "hp_water_heater_total_output..j",
    "hp_water_heater_total_electric..j",
]


def assert_delivered_loads(path: str) -> None:
    """ResStock releases without delivered-load columns would give zero heating and cooling."""
    delivered = {c for cols in Eulp._res_useful_groups.values() for c in cols}
    missing = delivered.difference(pd.read_csv(path, nrows=0).columns)
    assert not missing, f"ResStock file {path} has no delivered-load columns {sorted(missing)}"


def _final(column: str) -> str:
    """Annual-results column of a timeseries column (same name without the unit)."""
    return column.removesuffix(".kwh")


def read_comstock_metadata(path: str) -> pd.DataFrame:
    """Building weight, final energy (kWh) and equipment parameters (kWh or efficiency)."""
    final = [_final(c) for cols in Eulp._com_useful_groups.values() for c in cols]
    df = pd.read_parquet(path, columns=["in.hvac_heat_type", "weight", *final, *(f"out.params.{c}" for c in EQUIPMENT)])
    df = df.rename(columns=lambda c: c.removeprefix("out.params."))
    numeric = df.columns.difference(["in.hvac_heat_type"])
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    joules = [c for c in df.columns if c.endswith("..j")]
    df[joules] = df[joules] * J_TO_KWH
    return df


def comstock_useful_energy(df: pd.DataFrame, state_dx_cop: float) -> pd.DataFrame:
    """Useful energy (kWh per building) behind each final-energy timeseries column."""
    col = "out.{}.energy_consumption.kwh".format
    final = {c: df[_final(c)] for cols in Eulp._com_useful_groups.values() for c in cols}
    u = pd.DataFrame(index=df.index)

    boiler_fuel = df["boiler_gas..j"] + df["boiler_electric..j"] + df["boiler_other_fuel..j"]
    boiler_eff = (df["boiler_load..j"] / boiler_fuel.where(boiler_fuel > 0)).fillna(0.0)
    coil_eff = df["gas_coil_average_efficiency"].where(df["gas_coil_average_efficiency"] > 0, FUEL_COIL_EFFICIENCY)

    # fuel-fired heating: boilers at their simulated efficiency, the rest through fuel coils
    gas_supplemental = df["dx_heating_total_supplemental_gas..j"] + df["vrf_heating_total_supplemental_gas..j"]
    gas_coils = (final[col("natural_gas.heating")] - df["boiler_gas..j"] - gas_supplemental).clip(lower=0.0)
    u[col("natural_gas.heating")] = (
        df["boiler_gas..j"] * boiler_eff
        + gas_coils * coil_eff
        + df["dx_heating_total_supplemental_load_gas..j"]
        + df["vrf_heating_total_supplemental_load_gas..j"]
    )
    other_coils = (final[col("other_fuel.heating")] - df["boiler_other_fuel..j"]).clip(lower=0.0)
    u[col("other_fuel.heating")] = df["boiler_other_fuel..j"] * boiler_eff + other_coils * coil_eff
    u[col("district_heating.heating")] = final[col("district_heating.heating")]

    # electric heating: heat-pump loads at their COP, the remainder is resistance.
    # A water-source heat pump lifts boiler loop heat already counted under the
    # boiler fuel, so only its compressor work is added here.
    hp_electric = df["wa_hp_heating_electric..j"] + df["heat_pump_heating_total_electric..j"]
    hp_load = df["wa_hp_heating_load..j"] + df["heat_pump_heating_total_load..j"]
    hp_useful = hp_load.where(df["in.hvac_heat_type"] != "WSHP", hp_electric)
    supplemental = df["dx_heating_total_supplemental_electric..j"] + df["vrf_heating_total_supplemental_electric..j"]
    resistance = (
        final[col("electricity.heating")]
        - df["dx_heating_total_dx_electric..j"]
        - supplemental
        - hp_electric
        - df["boiler_electric..j"]
    ).clip(lower=0.0)
    u[col("electricity.heating")] = (
        df["dx_heating_total_dx_load..j"]
        + df["dx_heating_total_supplemental_load_electric..j"]
        + df["vrf_heating_total_supplemental_load_electric..j"]
        + hp_useful
        + df["boiler_electric..j"] * boiler_eff
        + resistance
    )

    # electric cooling: DX and heat-pump loads at their COP; the remainder is
    # chillers at their average COP, else the building's or the state's DX COP
    dx_cop = (df["dx_cooling_load..j"] / df["dx_cooling_electric..j"].where(df["dx_cooling_electric..j"] > 0)).fillna(
        0.0
    )
    hp_cool_electric = df["wa_hp_cooling_electric..j"] + df["heat_pump_cooling_total_electric..j"]
    remainder = (final[col("electricity.cooling")] - df["dx_cooling_electric..j"] - hp_cool_electric).clip(lower=0.0)
    remainder_cop = df["chiller_average_cop..cop"].where(df["chiller_average_cop..cop"] > 0, dx_cop)
    remainder_cop = remainder_cop.where(remainder_cop > 0, state_dx_cop)
    u[col("electricity.cooling")] = (
        df["dx_cooling_load..j"]
        + df["wa_hp_cooling_load..j"]
        + df["heat_pump_cooling_total_load..j"]
        + remainder * remainder_cop
    )
    u[col("district_cooling.cooling")] = final[col("district_cooling.cooling")]
    u[col("district_heating.cooling")] = final[col("district_heating.cooling")] * ABSORPTION_COP
    u[col("other_fuel.cooling")] = final[col("other_fuel.cooling")] * ABSORPTION_COP

    electric_water = final[col("electricity.water_systems")]
    hpwh_electric = df["hp_water_heater_total_electric..j"].clip(upper=electric_water)
    hpwh_output = df["hp_water_heater_total_output..j"].where(hpwh_electric > 0, 0.0)
    u[col("electricity.water_systems")] = (
        electric_water - hpwh_electric
    ) * ELECTRIC_WATER_HEATER_EFFICIENCY + hpwh_output
    u[col("natural_gas.water_systems")] = final[col("natural_gas.water_systems")] * FUEL_WATER_HEATER_EFFICIENCY
    u[col("other_fuel.water_systems")] = final[col("other_fuel.water_systems")] * FUEL_WATER_HEATER_EFFICIENCY
    u[col("district_heating.water_systems")] = final[col("district_heating.water_systems")]
    return u


def comstock_factors(df: pd.DataFrame) -> pd.Series:
    """Weighted useful / final energy of each ComStock timeseries column for one state.

    A column with no final energy in the state gets a factor of one; it scales
    a profile that is zero throughout.
    """
    w = df["weight"]
    dx_electric = (df["dx_cooling_electric..j"] * w).sum()
    state_dx_cop = (df["dx_cooling_load..j"] * w).sum() / dx_electric if dx_electric > 0 else DX_COP_FALLBACK
    useful = comstock_useful_energy(df, state_dx_cop).mul(w, axis=0).sum()
    final = pd.Series({c: (df[_final(c)] * w).sum() for c in useful.index})
    return useful.div(final.where(final > 0)).fillna(1.0)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_eulp_loads", stock="com", state="CO")
    configure_logging(snakemake)

    stock = snakemake.wildcards.stock
    state = snakemake.wildcards.state

    factors = None
    if stock == "com":
        factors = comstock_factors(read_comstock_metadata(snakemake.input.metadata))
        logger.info(f"ComStock useful-energy factors for {state}:\n{factors.round(3).to_string()}")
    else:
        for f in snakemake.input.profiles:
            assert_delivered_loads(f)

    profiles = [Eulp(f, stock=stock, factors=factors) for f in snakemake.input.profiles]
    loads = sum(profiles[1:], profiles[0])

    logger.info(f"{stock} {state} annual loads (GWh): {loads.data.sum().div(1e3).round(1).to_dict()}")
    loads.to_csv(snakemake.output[0])
