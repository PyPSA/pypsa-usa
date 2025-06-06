# By PyPSA-USA Authors

import copy
import hashlib
import logging
import re
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import requests
import yaml
from snakemake.utils import update_config

REGION_COLS = ["geometry", "name", "x", "y", "country"]


def configure_logging(snakemake, skip_handlers=False):
    """
    Configure the basic behaviour for the logging module.

    Note: Must only be called once from the __main__ section of a script.

    The setup includes printing log messages to STDERR and to a log file defined
    by either (in priority order): snakemake.log.python, snakemake.log[0] or "logs/{rulename}.log".
    Additional keywords from logging.basicConfig are accepted via the snakemake configuration
    file under snakemake.config.logging.

    Parameters
    ----------
    snakemake : snakemake object
        Your snakemake object containing a snakemake.config and snakemake.log.
    skip_handlers : True | False (default)
        Do (not) skip the default handlers created for redirecting output to STDERR and file.
    """
    kwargs = snakemake.config.get("logging", dict()).copy()
    kwargs.setdefault("level", "INFO")

    if skip_handlers is False:
        fallback_path = Path(__file__).parent.joinpath(
            "..",
            "logs",
            f"{snakemake.rule}.log",
        )
        logfile = snakemake.log.get(
            "python",
            snakemake.log[0] if snakemake.log else fallback_path,
        )
        kwargs.update(
            {
                "handlers": [
                    # Prefer the 'python' log, otherwise take the first log for each
                    # Snakemake rule
                    logging.FileHandler(logfile),
                    logging.StreamHandler(),
                ],
            },
        )
    logging.basicConfig(**kwargs)


def setup_custom_logger(name):
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    # logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


def load_network(import_name=None, custom_components=None):
    """
    Helper for importing a pypsa.Network with additional custom components.

    Parameters
    ----------
    import_name : str
        As in pypsa.Network(import_name)
    custom_components : dict
        Dictionary listing custom components.
        For using ``snakemake.config['override_components']``
        in ``config.yaml`` define:

        .. code:: yaml

            override_components:
                ShadowPrice:
                    component: ["shadow_prices","Shadow price for a global constraint.",np.nan]

    Attributes
    ----------
                    name: ["string","n/a","n/a","Unique name","Input (required)"]
                    value: ["float","n/a",0.,"shadow value","Output"]

    Returns
    -------
    pypsa.Network
    """
    from pypsa.descriptors import Dict

    override_components = None
    override_component_attrs = None

    if custom_components is not None:
        override_components = pypsa.components.components.copy()
        override_component_attrs = Dict(
            {k: v.copy() for k, v in pypsa.components.component_attrs.items()},
        )
        for k, v in custom_components.items():
            override_components.loc[k] = v["component"]
            override_component_attrs[k] = pd.DataFrame(
                columns=["type", "unit", "default", "description", "status"],
            )
            for attr, val in v["attributes"].items():
                override_component_attrs[k].loc[attr] = val

    return pypsa.Network(
        import_name=import_name,
        override_components=override_components,
        override_component_attrs=override_component_attrs,
    )


def pdbcast(v, h):
    return pd.DataFrame(
        v.values.reshape((-1, 1)) * h.values,
        index=v.index,
        columns=h.index,
    )


def calculate_annuity(n, r):
    """
    Calculate the annuity factor for an asset with lifetime n years and.

    discount rate of r, e.g. annuity(20, 0.05) * 20 = 1.6
    """
    if isinstance(r, pd.Series):
        return pd.Series(1 / n, index=r.index).where(
            r == 0,
            r / (1.0 - 1.0 / (1.0 + r) ** n),
        )
    elif r > 0:
        return r / (1.0 - 1.0 / (1.0 + r) ** n)
    else:
        return 1 / n


def load_costs(tech_costs: str) -> pd.DataFrame:
    df = pd.read_csv(tech_costs)
    return df.pivot(index="pypsa-name", columns="parameter", values="value").fillna(0)


def load_network_for_plots(fn, tech_costs, config, combine_hydro_ps=True):
    import pypsa
    from add_electricity import load_costs, update_transmission_costs

    n = pypsa.Network(fn)

    n.loads["carrier"] = n.loads.bus.map(n.buses.carrier) + " load"
    n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)

    n.links["carrier"] = n.links.bus0.map(n.buses.carrier) + "-" + n.links.bus1.map(n.buses.carrier)
    n.lines["carrier"] = "AC line"
    n.transformers["carrier"] = "AC transformer"

    n.lines["s_nom"] = n.lines["s_nom_min"]
    n.links["p_nom"] = n.links["p_nom_min"]

    if combine_hydro_ps:
        n.storage_units.loc[
            n.storage_units.carrier.isin({"PHS", "hydro"}),
            "carrier",
        ] = "hydro+PHS"

    # if the carrier was not set on the heat storage units
    # bus_carrier = n.storage_units.bus.map(n.buses.carrier)
    # n.storage_units.loc[bus_carrier == "heat","carrier"] = "water tanks"

    num_years = n.snapshot_weightings.loc[n.investment_periods[0]].objective.sum() / 8760.0
    costs = load_costs(tech_costs, config["costs"], config["electricity"], num_years)
    update_transmission_costs(n, costs)

    return n


def is_transport_model(transmission_network):
    match transmission_network:
        case "reeds":
            return True
        case "tamu":
            return False
        case _:
            return ValueError(
                "transmission network not specified correctly. Check config",
            )


def update_p_nom_max(n):
    # if extendable carriers (solar/onwind/...) have capacity >= 0,
    # e.g. existing assets from the OPSD project are included to the network,
    # the installed capacity might exceed the expansion limit.
    # Hence, we update the assumptions.

    n.generators.p_nom_max = n.generators[["p_nom_min", "p_nom_max"]].max(1)


def aggregate_p_nom(n):
    return pd.concat(
        [
            n.generators.groupby("carrier").p_nom_opt.sum(),
            n.storage_units.groupby("carrier").p_nom_opt.sum(),
            n.links.groupby("carrier").p_nom_opt.sum(),
            n.loads_t.p.groupby(n.loads.carrier, axis=1).sum().mean(),
        ],
    )


def aggregate_p(n):
    return pd.concat(
        [
            n.generators_t.p.sum().groupby(n.generators.carrier).sum(),
            n.storage_units_t.p.sum().groupby(n.storage_units.carrier).sum(),
            n.stores_t.p.sum().groupby(n.stores.carrier).sum(),
            -n.loads_t.p.sum().groupby(n.loads.carrier).sum(),
        ],
    )


def aggregate_e_nom(n):
    return pd.concat(
        [
            (n.storage_units["p_nom_opt"] * n.storage_units["max_hours"]).groupby(n.storage_units["carrier"]).sum(),
            n.stores["e_nom_opt"].groupby(n.stores.carrier).sum(),
        ],
    )


def aggregate_p_curtailed(n):
    return pd.concat(
        [
            (
                (n.generators_t.p_max_pu.sum().multiply(n.generators.p_nom_opt) - n.generators_t.p.sum())
                .groupby(n.generators.carrier)
                .sum()
            ),
            ((n.storage_units_t.inflow.sum() - n.storage_units_t.p.sum()).groupby(n.storage_units.carrier).sum()),
        ],
    )


def aggregate_costs(n, flatten=False, opts=None, existing_only=False):
    components = dict(
        Link=("p_nom", "p0"),
        Generator=("p_nom", "p"),
        StorageUnit=("p_nom", "p"),
        Store=("e_nom", "p"),
        Line=("s_nom", None),
        Transformer=("s_nom", None),
    )

    costs = {}
    for c, (p_nom, p_attr) in zip(
        n.iterate_components(components.keys(), skip_empty=False),
        components.values(),
    ):
        if c.df.empty:
            continue
        if not existing_only:
            p_nom += "_opt"
        costs[(c.list_name, "capital")] = (c.df[p_nom] * c.df.capital_cost).groupby(c.df.carrier).sum()
        if p_attr is not None:
            p = c.pnl[p_attr].sum()
            if c.name == "StorageUnit":
                p = p.loc[p > 0]
            costs[(c.list_name, "marginal")] = (p * c.df.marginal_cost).groupby(c.df.carrier).sum()
    costs = pd.concat(costs)

    if flatten:
        assert opts is not None
        conv_techs = opts["conv_techs"]

        costs = costs.reset_index(level=0, drop=True)
        costs = costs["capital"].add(
            costs["marginal"].rename({t: t + " marginal" for t in conv_techs}),
            fill_value=0.0,
        )

    return costs


def progress_retrieve(url, file):
    import urllib

    from progressbar import ProgressBar

    pbar = ProgressBar(0, 100)

    def dlProgress(count, block_size, total_size):
        pbar.update(int(count * block_size * 100 / total_size))

    urllib.request.urlretrieve(url, file, reporthook=dlProgress)


def get_aggregation_strategies(aggregation_strategies):
    # default aggregation strategies that cannot be defined in .yaml format must be specified within
    # the function, otherwise (when defaults are passed in the function's definition) they get lost
    # when custom values are specified in the config.

    import numpy as np
    from pypsa.clustering.spatial import _make_consense

    bus_strategies = dict(country=_make_consense("Bus", "country"))
    bus_strategies.update(aggregation_strategies.get("buses", {}))

    generator_strategies = {"build_year": lambda x: 0, "lifetime": lambda x: np.inf}
    generator_strategies.update(aggregation_strategies.get("generators", {}))

    return bus_strategies, generator_strategies


def export_network_for_gis_mapping(n, output_path):
    # Creating GIS Table for Mapping Lines in QGIS
    lines_gis = n.lines.copy()
    lines_gis["latitude1"] = n.buses.loc[lines_gis.bus0].y.values
    lines_gis["longitude1"] = n.buses.loc[lines_gis.bus0].x.values
    lines_gis["latitude2"] = n.buses.loc[lines_gis.bus1].y.values
    lines_gis["longitude2"] = n.buses.loc[lines_gis.bus1].x.values
    lines_gis["v_nom"] = n.buses.loc[lines_gis.bus0].v_nom.values
    lines_gis["wkt_geom"] = (
        "LINESTRING ("
        + lines_gis.longitude1.astype(str)
        + " "
        + lines_gis.latitude1.astype(str)
        + ", "
        + lines_gis.longitude2.astype(str)
        + " "
        + lines_gis.latitude2.astype(str)
        + ")"
    )

    lines_gis.to_csv(output_path + "_lines_GIS.csv")

    # Creating GIS Table for Mapping Buses in QGIS
    buses_gis = n.buses.copy()
    buses_gis.to_csv(output_path + "_buses_GIS.csv")


def mock_snakemake(rulename, **wildcards):
    """
    Function is expected to be executed from the 'scripts'-directory of '
    the snakemake project. It returns a snakemake.script.Snakemake object,
    based on the Snakefile.

    If a rule has wildcards, you have to specify them in **wildcards.

    Parameters
    ----------
    rulename: str
        name of the rule for which the snakemake object should be generated
    **wildcards:
        keyword arguments fixing the wildcards. Only necessary if wildcards are
        needed.
    """
    import os

    import snakemake as sm
    from packaging.version import Version, parse
    from pypsa.descriptors import Dict
    from snakemake.script import Snakemake

    script_dir = Path(__file__).parent.resolve()
    assert Path.cwd().resolve() == script_dir, (
        f"mock_snakemake has to be run from the repository scripts directory {script_dir}"
    )
    os.chdir(script_dir.parent)
    for p in sm.SNAKEFILE_CHOICES:
        if os.path.exists(p):
            snakefile = p
            break
    kwargs = dict(rerun_triggers=[]) if parse(sm.__version__) > Version("7.7.0") else {}
    workflow = sm.Workflow(snakefile, overwrite_configfiles=[], **kwargs)
    workflow.include(snakefile)
    workflow.global_resources = {}
    rule = workflow.get_rule(rulename)
    dag = sm.dag.DAG(workflow, rules=[rule])
    wc = Dict(wildcards)
    job = sm.jobs.Job(rule, dag, wc)

    def make_accessable(*ios):
        for io in ios:
            for i in range(len(io)):
                io[i] = os.path.abspath(io[i])

    make_accessable(job.input, job.output, job.log)
    snakemake = Snakemake(
        job.input,
        job.output,
        job.params,
        job.wildcards,
        job.threads,
        job.resources,
        job.log,
        job.dag.workflow.config,
        job.rule.name,
        None,
    )
    # create log and output dir if not existent
    for path in list(snakemake.log) + list(snakemake.output):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    os.chdir(script_dir)
    return snakemake


def validate_checksum(file_path, zenodo_url=None, checksum=None):
    """
    Validate file checksum against provided or Zenodo-retrieved checksum.
    Calculates the hash of a file using 64KB chunks. Compares it against a
    given checksum or one from a Zenodo URL.

    Parameters
    ----------
    file_path : str
        Path to the file for checksum validation.
    zenodo_url : str, optional
        URL of the file on Zenodo to fetch the checksum.
    checksum : str, optional
        Checksum (format 'hash_type:checksum_value') for validation.

    Raises
    ------
    AssertionError
        If the checksum does not match, or if neither `checksum` nor `zenodo_url` is provided.


    Examples
    --------
    >>> validate_checksum("/path/to/file", checksum="md5:abc123...")
    >>> validate_checksum(
    ...     "/path/to/file",
    ...     zenodo_url="https://zenodo.org/record/12345/files/example.txt",
    ... )

    If the checksum is invalid, an AssertionError will be raised.
    """
    assert checksum or zenodo_url, "Either checksum or zenodo_url must be provided"
    if zenodo_url:
        checksum = get_checksum_from_zenodo(zenodo_url)
    hash_type, checksum = checksum.split(":")
    hasher = hashlib.new(hash_type)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):  # 64kb chunks
            hasher.update(chunk)
    calculated_checksum = hasher.hexdigest()
    assert calculated_checksum == checksum, (
        "Checksum is invalid. This may be due to an incomplete download. Delete the file and re-execute the rule."
    )


def get_checksum_from_zenodo(file_url):
    parts = file_url.split("/")
    record_id = parts[parts.index("record") + 1]
    filename = parts[-1]

    response = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=30)
    response.raise_for_status()
    data = response.json()

    for file in data["files"]:
        if file["key"] == filename:
            return file["checksum"]
    return None


### Config Related Helpers ###


def set_scenario_config(snakemake):
    scenario = snakemake.config["run"].get("scenarios", {})
    if scenario.get("enable") and "run" in snakemake.wildcards.keys():
        try:
            with open(scenario["file"]) as f:
                scenario_config = yaml.safe_load(f)
        except FileNotFoundError:
            # fallback for mock_snakemake
            script_dir = Path(__file__).parent.resolve()
            root_dir = script_dir.parent
            with open(root_dir / scenario["file"]) as f:
                scenario_config = yaml.safe_load(f)
        update_config(snakemake.config, scenario_config[snakemake.wildcards.run])


def update_config_with_sector_opts(config, sector_opts):
    from packaging.version import parse
    from snakemake.utils import update_config

    for o in sector_opts.split("-"):
        if o.startswith("CF+"):
            l_ = o.split("+")[1:]
            update_config(config, parse(l_))


def get_opt(opts, expr, flags=None):
    """
    Return the first option matching the regular expression.

    The regular expression is case-insensitive by default.
    """
    if flags is None:
        flags = re.IGNORECASE
    for o in opts:
        match = re.match(expr, o, flags=flags)
        if match:
            return match.group(0)
    return None


def find_opt(opts, expr):
    """Return if available the float after the expression."""
    for o in opts:
        if expr in o:
            m = re.findall(r"m?\d+(?:[\.p]\d+)?", o)
            if len(m) > 0:
                return True, float(m[-1].replace("p", ".").replace("m", "-"))
            else:
                return True, None
    return False, None


def update_config_from_wildcards(config, w, inplace=True):
    """Parses configuration settings from wildcards and updates the config."""
    from packaging.version import parse

    if not inplace:
        config = copy.deepcopy(config)

    if w.get("opts"):
        opts = w.opts.split("-")

        if nhours := get_opt(opts, r"^\d+(h|seg)$"):
            config["clustering"]["temporal"]["resolution_elec"] = nhours

        co2l_enable, co2l_value = find_opt(opts, "Co2L")
        if co2l_enable:
            config["electricity"]["co2limit_enable"] = True
            if co2l_value is not None:
                config["electricity"]["co2limit"] = co2l_value * config["electricity"]["co2base"]

        gasl_enable, gasl_value = find_opt(opts, "CH4L")
        if gasl_enable:
            config["electricity"]["gaslimit_enable"] = True
            if gasl_value is not None:
                config["electricity"]["gaslimit"] = gasl_value * 1e6

        if "Ept" in opts:
            config["costs"]["emission_prices"]["co2_monthly_prices"] = True

        ep_enable, ep_value = find_opt(opts, "Ep")
        if ep_enable:
            config["costs"]["emission_prices"]["enable"] = True
            if ep_value is not None:
                config["costs"]["emission_prices"]["co2"] = ep_value

        if "ATK" in opts:
            config["autarky"]["enable"] = True
            if "ATKc" in opts:
                config["autarky"]["by_country"] = True

        attr_lookup = {
            "p": "p_nom_max",
            "e": "e_nom_max",
            "c": "capital_cost",
            "m": "marginal_cost",
        }
        for o in opts:
            flags = ["+e", "+p", "+m", "+c"]
            if all(flag not in o for flag in flags):
                continue
            carrier, attr_factor = o.split("+")
            attr = attr_lookup[attr_factor[0]]
            factor = float(attr_factor[1:])
            if not isinstance(config["adjustments"]["electricity"], dict):
                config["adjustments"]["electricity"] = dict()
            update_config(
                config["adjustments"]["electricity"],
                {attr: {carrier: factor}},
            )

    if w.get("sector_opts"):
        opts = w.sector_opts.split("-")

        if "T" in opts:
            config["sector"]["transport"] = True

        if "H" in opts:
            config["sector"]["heating"] = True

        if "B" in opts:
            config["sector"]["biomass"] = True

        if "I" in opts:
            config["sector"]["industry"] = True

        if "A" in opts:
            config["sector"]["agriculture"] = True

        if "TCT" in opts:
            config["solving"]["constraints"]["TCT"] = True

        eq_value = get_opt(opts, r"^EQ+\d*\.?\d+(c|)")
        for o in opts:
            if eq_value is not None:
                config["solving"]["constraints"]["EQ"] = eq_value
            elif "EQ" in o:
                config["solving"]["constraints"]["EQ"] = True
            break

        if "BAU" in opts:
            config["solving"]["constraints"]["BAU"] = True

        if "SAFE" in opts:
            config["solving"]["constraints"]["SAFE"] = True

        if nhours := get_opt(opts, r"^\d+(h|sn|seg)$"):
            config["clustering"]["temporal"]["resolution_sector"] = nhours

        if "decentral" in opts:
            config["sector"]["electricity_transmission_grid"] = False

        if "noH2network" in opts:
            config["sector"]["H2_network"] = False

        if "nowasteheat" in opts:
            config["sector"]["use_fischer_tropsch_waste_heat"] = False
            config["sector"]["use_methanolisation_waste_heat"] = False
            config["sector"]["use_haber_bosch_waste_heat"] = False
            config["sector"]["use_methanation_waste_heat"] = False
            config["sector"]["use_fuel_cell_waste_heat"] = False
            config["sector"]["use_electrolysis_waste_heat"] = False

        if "nodistrict" in opts:
            config["sector"]["district_heating"]["progress"] = 0.0

        dg_enable, dg_factor = find_opt(opts, "dist")
        if dg_enable:
            config["sector"]["electricity_distribution_grid"] = True
            if dg_factor is not None:
                config["sector"]["electricity_distribution_grid_cost_factor"] = dg_factor

        if "biomasstransport" in opts:
            config["sector"]["biomass_transport"] = True

        _, maxext = find_opt(opts, "linemaxext")
        if maxext is not None:
            config["lines"]["max_extension"] = maxext * 1e3
            config["links"]["max_extension"] = maxext * 1e3

        _, co2l_value = find_opt(opts, "Co2L")
        if co2l_value is not None:
            config["co2_budget"] = float(co2l_value)

        if co2_distribution := get_opt(opts, r"^(cb)\d+(\.\d+)?(ex|be)$"):
            config["co2_budget"] = co2_distribution

        if co2_budget := get_opt(opts, r"^(cb)\d+(\.\d+)?$"):
            config["co2_budget"] = float(co2_budget[2:])

        attr_lookup = {
            "p": "p_nom_max",
            "e": "e_nom_max",
            "c": "capital_cost",
            "m": "marginal_cost",
        }
        for o in opts:
            flags = ["+e", "+p", "+m", "+c"]
            if all(flag not in o for flag in flags):
                continue
            carrier, attr_factor = o.split("+")
            attr = attr_lookup[attr_factor[0]]
            factor = float(attr_factor[1:])
            if not isinstance(config["adjustments"]["sector"], dict):
                config["adjustments"]["sector"] = dict()
            update_config(config["adjustments"]["sector"], {attr: {carrier: factor}})

        _, sdr_value = find_opt(opts, "sdr")
        if sdr_value is not None:
            config["costs"]["social_discountrate"] = sdr_value / 100

        _, seq_limit = find_opt(opts, "seq")
        if seq_limit is not None:
            config["sector"]["co2_sequestration_potential"] = seq_limit

        # any config option can be represented in wildcard
        for o in opts:
            if o.startswith("CF+"):
                infix = o.split("+")[1:]
                update_config(config, parse(infix))

    if not inplace:
        return config


def get_scenarios(run):
    scenario_config = run.get("scenarios", {})
    if run["name"] and scenario_config.get("enable"):
        fn = Path(scenario_config["file"])
        if fn.exists():
            scenarios = yaml.safe_load(fn.read_text())
            if run["name"] == "all":
                run["name"] = list(scenarios.keys())
            return scenarios
    return {}


def get_rdir(run):
    scenario_config = run.get("scenarios", {})
    if run["name"] and scenario_config.get("enable"):
        rdir = "{run}/"
    elif run["name"]:
        rdir = run["name"] + "/"
    else:
        rdir = ""

    prefix = run.get("prefix", "")
    if prefix:
        rdir = f"{prefix}/{rdir}"

    return rdir


def get_run_path(fn, dir, rdir, shared_resources):
    """
    Dynamically provide paths based on shared resources and filename.

    Use this function for snakemake rule inputs or outputs that should be
    optionally shared across runs or created individually for each run.

    Parameters
    ----------
    fn : str
        The filename for the path to be generated.
    dir : str
        The base directory.
    rdir : str
        Relative directory for non-shared resources.
    shared_resources : str or bool
        Specifies which resources should be shared.
        - If string is "base", special handling for shared "base" resources (see notes).
        - If random string other than "base", this folder is used instead of the `rdir` keyword.
        - If boolean, directly specifies if the resource is shared.

    Returns
    -------
    str
        Full path where the resource should be stored.

    Notes
    -----
    Special case for "base" allows no wildcards other than "technology", "year"
    and "scope" and excludes filenames starting with "networks/elec" or
    "add_electricity". All other resources are shared.
    """
    if shared_resources == "base":
        pattern = r"\{([^{}]+)\}"
        existing_wildcards = set(re.findall(pattern, fn))
        irrelevant_wildcards = {"technology", "year", "scope", "kind"}
        no_relevant_wildcards = not existing_wildcards - irrelevant_wildcards
        no_elec_rule = not fn.startswith("networks/elec") and not fn.startswith(
            "add_electricity",
        )
        is_shared = no_relevant_wildcards and no_elec_rule
        rdir = "" if is_shared else rdir
    elif isinstance(shared_resources, str):
        rdir = shared_resources + "/"
    elif isinstance(shared_resources, bool):
        rdir = "" if shared_resources else rdir
    else:
        raise ValueError(
            "shared_resources must be a boolean, str, or 'base' for special handling.",
        )

    return f"{dir}{rdir}{fn}"


def path_provider(dir, rdir, shared_resources):
    """
    Returns a partial function that dynamically provides paths based on shared
    resources and the filename.

    Returns
    -------
    partial function
        A partial function that takes a filename as input and
        returns the path to the file based on the shared_resources parameter.
    """
    return partial(get_run_path, dir=dir, rdir=rdir, shared_resources=shared_resources)


def get_snapshots(
    snapshots: dict[str, str],
    drop_leap_day: bool = True,
    freq: str = "h",
    **kwargs,
) -> pd.date_range:
    """
    Returns pandas DateTimeIndex potentially without leap days.

    Taken from PyPSA-Eur implementation
    """
    time = pd.date_range(freq=freq, **snapshots, **kwargs)
    if drop_leap_day and time.is_leap_year.any():
        time = time[~((time.month == 2) & (time.day == 29))]

    return time


def weighted_avg(df, values, weights):
    """
    Return the weighted average of a DataFrame column(s) `values` with weights
    `weights`.
    """
    valid = df[values].notna()
    if valid.sum() == 0:
        return np.nan  # Return NaN if no valid entries
    return np.average(df[values][valid], weights=df[weights][valid])


def get_multiindex_snapshots(
    sns_config: dict[str, str],
    invest_periods: list[int],
) -> pd.MultiIndex:
    sns = pd.DatetimeIndex([])
    for year in invest_periods:
        sns = sns.append(
            get_snapshots(sns_config).map(lambda x: x.replace(year=year)),
        )
    return pd.MultiIndex.from_arrays([sns.year, sns])
