"""
Build a realistic small clustered network for end-to-end ERM testing.

Run this script to regenerate test_network.nc:
    python build_test_network.py

The .nc file is gitignored (*.nc in root .gitignore) and must be
regenerated locally before running tests that depend on it.

Network: 8 buses, 3 investment periods (2030, 2040, 2050), 24 h per period.
Covers two NERC regions (WECC, RFC) with a mix of existing and new-build
generators, storage, AC lines, and DC links — mirroring the structure of
a real pypsa-usa clustered solve network.
"""

import os
import sys

import numpy as np
import pandas as pd
import pypsa

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from _helpers import get_multiindex_snapshots

OUTPUT = os.path.join(os.path.dirname(__file__), "test_network.nc")


def build():
    invest_periods = [2030, 2040, 2050]
    sns_cfg = {"start": "2030-01-01 00:00", "end": "2030-01-01 23:00", "inclusive": "both"}

    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(sns_cfg, invest_periods)
    n.set_investment_periods(periods=invest_periods)

    # ------------------------------------------------------------------
    # Buses: 4 WECC (CA-ish), 4 RFC (Midwest-ish)
    # ------------------------------------------------------------------
    buses = {
        "CA_N": dict(x=-122, y=40, interconnect="western", nerc_reg="WECC", reeds_zone="CA_N", reeds_state="CA"),
        "CA_S": dict(x=-118, y=34, interconnect="western", nerc_reg="WECC", reeds_zone="CA_S", reeds_state="CA"),
        "NV": dict(x=-117, y=39, interconnect="western", nerc_reg="WECC", reeds_zone="NV", reeds_state="NV"),
        "AZ": dict(x=-112, y=33, interconnect="western", nerc_reg="WECC", reeds_zone="AZ", reeds_state="AZ"),
        "IL": dict(x=-89, y=40, interconnect="eastern", nerc_reg="RFC", reeds_zone="IL", reeds_state="IL"),
        "IN": dict(x=-86, y=40, interconnect="eastern", nerc_reg="RFC", reeds_zone="IN", reeds_state="IN"),
        "OH": dict(x=-83, y=40, interconnect="eastern", nerc_reg="RFC", reeds_zone="OH", reeds_state="OH"),
        "MI": dict(x=-84, y=44, interconnect="eastern", nerc_reg="RFC", reeds_zone="MI", reeds_state="MI"),
    }
    for name, attrs in buses.items():
        n.add("Bus", name, carrier="AC", x=attrs["x"], y=attrs["y"])

    # Custom attributes must be set after add() — PyPSA ignores unknown kwargs
    for name, attrs in buses.items():
        n.buses.loc[name, "country"] = "US"
        n.buses.loc[name, "interconnect"] = attrs["interconnect"]
        n.buses.loc[name, "nerc_reg"] = attrs["nerc_reg"]
        n.buses.loc[name, "reeds_zone"] = attrs["reeds_zone"]
        n.buses.loc[name, "reeds_state"] = attrs["reeds_state"]

    # ------------------------------------------------------------------
    # Carriers
    # ------------------------------------------------------------------
    carriers = {
        "onwind": 0,
        "offwind": 0,
        "solar": 0,
        "nuclear": 0,
        "gas": 0.19,
        "coal": 0.34,
        "battery": 0,
        "AC": 0,
        "DC": 0,
    }
    for carrier, co2 in carriers.items():
        n.add("Carrier", carrier, co2_emissions=co2)

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------
    sns = n.snapshots
    n_hrs = 24  # hours per period

    def flat(val):
        return pd.Series(val, index=sns)

    # Realistic load: morning/evening peaks, tiled across periods
    hour_load = np.array(
        [
            0.65,
            0.62,
            0.60,
            0.59,
            0.60,
            0.63,
            0.72,
            0.82,
            0.88,
            0.90,
            0.91,
            0.92,
            0.93,
            0.94,
            0.95,
            0.97,
            0.99,
            1.00,
            0.97,
            0.93,
            0.87,
            0.80,
            0.74,
            0.68,
        ],
    )
    load_profile = pd.Series(np.tile(hour_load, len(invest_periods)), index=sns)

    # Solar: midday peak, tiled
    solar_raw = np.maximum(0, np.sin(np.linspace(0, np.pi, n_hrs)))
    solar_raw = solar_raw / solar_raw.max()
    solar_profile = pd.Series(np.tile(solar_raw, len(invest_periods)), index=sns)

    # Wind: modest variation
    wind_raw = 0.35 + 0.25 * np.sin(np.linspace(0, 2 * np.pi, n_hrs) + 1.5)
    wind_profile = pd.Series(np.tile(wind_raw, len(invest_periods)), index=sns)

    # ------------------------------------------------------------------
    # Existing (non-extendable) generators — retiring or long-lived
    # ------------------------------------------------------------------

    # Coal: retires before 2050
    coal_plants = [
        ("coal_IL", "IL", 1200, 2000, 25),  # (name, bus, p_nom, build_year, lifetime)
        ("coal_IN", "IN", 800, 2000, 25),
        ("coal_OH", "OH", 600, 2005, 25),
    ]
    for name, bus, pnom, by, lt in coal_plants:
        n.add(
            "Generator",
            name,
            bus=bus,
            carrier="coal",
            p_nom=pnom,
            p_nom_extendable=False,
            marginal_cost=25,
            p_max_pu=0.85,
            build_year=by,
            lifetime=lt,
        )

    # Gas (CCGT): persists through all periods
    gas_existing = [
        ("gas_CA_N", "CA_N", 800, 2015, 40),
        ("gas_CA_S", "CA_S", 1200, 2010, 40),
        ("gas_IL", "IL", 600, 2012, 40),
        ("gas_MI", "MI", 500, 2008, 40),
    ]
    for name, bus, pnom, by, lt in gas_existing:
        n.add(
            "Generator",
            name,
            bus=bus,
            carrier="gas",
            p_nom=pnom,
            p_nom_extendable=False,
            marginal_cost=45,
            p_max_pu=0.90,
            build_year=by,
            lifetime=lt,
        )

    # Nuclear: long-lived, zero-emissions
    n.add(
        "Generator",
        "nuclear_IL",
        bus="IL",
        carrier="nuclear",
        p_nom=1100,
        p_nom_extendable=False,
        marginal_cost=10,
        p_max_pu=0.92,
        build_year=1990,
        lifetime=80,
    )

    # ------------------------------------------------------------------
    # New-build (extendable) generators
    # ------------------------------------------------------------------

    # Wind (onshore)
    wind_sites = [
        ("wind_NV", "NV", 0.42),
        ("wind_AZ", "AZ", 0.38),
        ("wind_IL", "IL", 0.40),
        ("wind_IN", "IN", 0.45),
        ("wind_OH", "OH", 0.35),
        ("wind_MI", "MI", 0.38),
    ]
    for name, bus, cf_scale in wind_sites:
        n.add(
            "Generator",
            name,
            bus=bus,
            carrier="onwind",
            p_nom=0,
            p_nom_extendable=True,
            p_nom_max=3000,
            capital_cost=1200,
            marginal_cost=0.5,
            p_max_pu=wind_profile * cf_scale / wind_profile.mean() * cf_scale,
            build_year=2030,
            lifetime=25,
        )

    # Solar (utility)
    solar_sites = [
        ("solar_CA_N", "CA_N", 1.00),
        ("solar_CA_S", "CA_S", 1.10),
        ("solar_NV", "NV", 1.15),
        ("solar_AZ", "AZ", 1.20),
        ("solar_IL", "IL", 0.90),
        ("solar_IN", "IN", 0.88),
    ]
    for name, bus, scale in solar_sites:
        n.add(
            "Generator",
            name,
            bus=bus,
            carrier="solar",
            p_nom=0,
            p_nom_extendable=True,
            p_nom_max=5000,
            capital_cost=900,
            marginal_cost=0.1,
            p_max_pu=(solar_profile * scale).clip(upper=1.0),
            build_year=2030,
            lifetime=30,
        )

    # New gas (peakers): available as firm capacity backstop
    for bus in ["CA_N", "CA_S", "IL", "OH"]:
        n.add(
            "Generator",
            f"gas_new_{bus}",
            bus=bus,
            carrier="gas",
            p_nom=0,
            p_nom_extendable=True,
            p_nom_max=5000,
            capital_cost=600,
            marginal_cost=55,
            p_max_pu=0.95,
            build_year=2030,
            lifetime=30,
        )

    # Extendable nuclear: firm zero-emission capacity needed for zero-emission ERM tests.
    # p_max_pu=0.92 means it counts as nearly-firm capacity credit in every snapshot.
    # High capital cost ensures it isn't built unless the ERM constraint requires it.
    for bus in list(buses.keys()):
        n.add(
            "Generator",
            f"nuclear_new_{bus}",
            bus=bus,
            carrier="nuclear",
            p_nom=0,
            p_nom_extendable=True,
            p_nom_max=10000,
            capital_cost=7000,
            marginal_cost=8,
            p_max_pu=0.92,
            build_year=2030,
            lifetime=60,
        )

    # ------------------------------------------------------------------
    # Storage (batteries)
    # ------------------------------------------------------------------
    battery_sites = ["CA_N", "CA_S", "NV", "IL", "OH"]
    for bus in battery_sites:
        n.add(
            "StorageUnit",
            f"battery_{bus}",
            bus=bus,
            carrier="battery",
            p_nom=0,
            p_nom_extendable=True,
            p_nom_max=2000,
            capital_cost=250,
            marginal_cost=1,
            efficiency_store=0.92,
            efficiency_dispatch=0.92,
            standing_loss=0.002,
            max_hours=4,
            build_year=2030,
            lifetime=15,
        )

    # ------------------------------------------------------------------
    # Loads
    # ------------------------------------------------------------------
    peak_mw = {
        "CA_N": 4000,
        "CA_S": 6000,
        "NV": 1200,
        "AZ": 2000,
        "IL": 3500,
        "IN": 2000,
        "OH": 2500,
        "MI": 2200,
    }
    for bus, peak in peak_mw.items():
        n.add("Load", f"load_{bus}", bus=bus, p_set=load_profile * peak)

    # ------------------------------------------------------------------
    # AC Lines (intra-interconnect)
    # ------------------------------------------------------------------
    ac_lines = [
        # WECC
        ("line_CAN_CAS", "CA_N", "CA_S", 3000, 0.05),
        ("line_CAN_NV", "CA_N", "NV", 1500, 0.08),
        ("line_CAS_AZ", "CA_S", "AZ", 1500, 0.08),
        ("line_NV_AZ", "NV", "AZ", 800, 0.10),
        # RFC
        ("line_IL_IN", "IL", "IN", 2000, 0.06),
        ("line_IL_OH", "IL", "OH", 1500, 0.07),
        ("line_IN_OH", "IN", "OH", 1200, 0.07),
        ("line_OH_MI", "OH", "MI", 1000, 0.08),
    ]
    for name, b0, b1, snom, x in ac_lines:
        n.add(
            "Line",
            name,
            bus0=b0,
            bus1=b1,
            s_nom=snom,
            s_nom_extendable=True,
            s_nom_min=snom,
            x=x,
            r=x * 0.1,
            capital_cost=200,
            build_year=2030,
            lifetime=60,
        )

    # ------------------------------------------------------------------
    # DC Link (inter-interconnect HVDC tie)
    # ------------------------------------------------------------------
    n.add(
        "Link",
        "hvdc_WECC_RFC",
        bus0="NV",
        bus1="IL",
        carrier="DC",
        p_nom=1000,
        p_nom_extendable=True,
        p_nom_min=1000,
        capital_cost=800,
        efficiency=0.97,
        build_year=2030,
        lifetime=60,
    )

    # prepare_brownfield accesses these columns via df.loc[idx].col — they must exist
    for col, default in [
        ("heat_rate", np.nan),
        ("fuel_cost", 0.0),
        ("vom_cost", 0.0),
        ("carrier_base", ""),
        ("land_region", ""),
        ("p_min_pu", 0.0),
    ]:
        if col not in n.generators.columns:
            n.generators[col] = default

    n.export_to_netcdf(OUTPUT)
    print(f"Saved {OUTPUT}  ({os.path.getsize(OUTPUT) / 1024:.0f} kB)")


if __name__ == "__main__":
    build()
