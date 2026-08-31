"""Config-declared registry for GODEEEP capacity-factor (CF) files.

The GODEEEP CF retrieval used to be spread across ``build_renewable_profiles.py``
(filename/record-key construction, wind-height defaulting, year selection) and
``zenodo_downloader.py`` (a hardcoded ``record_id`` table that returned ``None``
when a lookup missed). A miss therefore surfaced hundreds of lines later as a
``TypeError`` on a ``None`` path, or — worse — silently resolved to a different
year/hub-height than the one requested.

This module replaces that with a single declarative registry read from
``config["godeeep_cf_registry"]``:

* every source (local Oak mirror, Zenodo) declares which ``(dataset key, year)``
  pairs it actually holds,
* :func:`resolve_cf` walks the sources IN CONFIG ORDER and the first source that
  holds the requested pair wins,
* an unresolvable request raises :class:`CfNotAvailableError` naming the dataset
  key, the requested year and the available years per source.

There are NO fallback paths: no default hub height, no "closest year", no
substitution of an unscreened file. Every failure is loud and names what is
missing. :func:`validate_godeeep_cf_config` runs the same resolution eagerly at
snakemake parse time so a bad config fails before any rule executes.

The module is deliberately stdlib-only so it stays importable from the
``Snakefile`` at parse time (``workflow/rules/common.smk`` puts
``workflow/scripts`` on ``sys.path``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

__all__ = [
    "CfNotAvailableError",
    "CfResolution",
    "CfSource",
    "TechSpec",
    "cf_filename",
    "dataset_key",
    "godeeep_tech_spec",
    "load_sources",
    "parse_years",
    "resolve_cf",
    "resolve_scenario",
    "resolve_weather_year",
    "validate_godeeep_cf_config",
]

#: Hub heights with published GODEEEP wind CFs. "_80m" is deliberately absent —
#: no such dataset exists, and defaulting to it silently produced wrong profiles.
VALID_WIND_HEIGHTS = ("_100m", "_125m")

#: pypsa-usa technology wildcards backed by the GODEEEP wind CFs.
WIND_TECHNOLOGIES = ("onwind", "offwind", "offwind_floating")

#: pypsa-usa technology wildcards backed by the GODEEEP solar CFs.
SOLAR_TECHNOLOGIES = ("solar",)

#: All pypsa-usa technology wildcards this registry can resolve.
GODEEEP_TECHNOLOGIES = SOLAR_TECHNOLOGIES + WIND_TECHNOLOGIES

#: Registry key in the merged snakemake config.
REGISTRY_KEY = "godeeep_cf_registry"

_YEAR_RANGE_RE = re.compile(r"^\s*(\d{4})\s*-\s*(\d{4})\s*$")

_KNOWN_SOURCE_KINDS = ("local", "zenodo")


class CfNotAvailableError(Exception):
    """A requested GODEEEP CF dataset/year is not declared by any config source.

    Raised instead of returning ``None`` or falling back to a different year,
    hub height or screening variant. The message always names the dataset key,
    the requested year and what each source actually offers.
    """


@dataclass(frozen=True)
class TechSpec:
    """The GODEEEP-side identity of a pypsa-usa technology wildcard."""

    technology: str
    """GODEEEP technology family: ``"solar"`` or ``"wind"``."""

    wind_height: str
    """Hub-height suffix (``"_100m"``/``"_125m"``); empty string for solar."""

    def __iter__(self):
        # Tuple-unpackable so call sites can keep the original
        # ``technology, wind_height = ...`` shape from build_renewable_profiles.
        return iter((self.technology, self.wind_height))

    @property
    def tech_dir(self) -> str:
        """Directory component of the local mirror layout (e.g. ``wind_125m``)."""
        return f"{self.technology}{self.wind_height}"


@dataclass(frozen=True)
class CfSource:
    """One entry of ``config["godeeep_cf_registry"]["sources"]``.

    A source declares the ``(dataset key -> years)`` availability it can serve
    plus the kind-specific coordinates needed to actually fetch a file: a
    filesystem ``root``/``layout`` for ``kind == "local"``, a Zenodo record id
    per dataset key for ``kind == "zenodo"``.
    """

    kind: str
    """``"local"`` or ``"zenodo"``."""

    years: Mapping[str, tuple[int, ...]]
    """Dataset key -> the years this source holds."""

    root: str | None = None
    """Filesystem root of a ``local`` source."""

    layout: str = "{scenario}/{tech_dir}/{filename}"
    """Path template of a ``local`` source, relative to ``root``."""

    records: Mapping[str, str] = field(default_factory=dict)
    """Dataset key -> Zenodo record id, for a ``zenodo`` source."""

    copy_local: bool = False
    """Copy rather than symlink the retrieved file (``local`` sources only)."""

    name: str = ""
    """Human-readable label used in error messages."""

    def has(self, key: str, year: int) -> bool:
        """Whether this source declares ``year`` for dataset ``key``."""
        return int(year) in self.years.get(key, ())

    def years_for(self, key: str) -> tuple[int, ...]:
        """The years this source declares for dataset ``key`` (may be empty)."""
        return self.years.get(key, ())

    @classmethod
    def from_config(cls, raw: Mapping, index: int) -> CfSource:
        """Build a source from its raw config mapping.

        Parameters
        ----------
        raw
            One entry of ``godeeep_cf_registry: sources:``.
        index
            Position in the source list, used for error messages.

        Raises
        ------
        ValueError
            If the entry is malformed (unknown kind, missing root, unparseable
            years, zenodo dataset without a record id).
        """
        where = f"godeeep_cf_registry.sources[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where} must be a mapping, got {type(raw).__name__}.")

        kind = raw.get("kind")
        if kind not in _KNOWN_SOURCE_KINDS:
            raise ValueError(
                f"{where}.kind is {kind!r}; expected one of {list(_KNOWN_SOURCE_KINDS)}.",
            )

        datasets = raw.get("datasets") or {}
        if not isinstance(datasets, Mapping):
            raise ValueError(f"{where}.datasets must be a mapping of dataset key -> years.")

        years: dict[str, tuple[int, ...]] = {}
        records: dict[str, str] = {}
        for key, spec in datasets.items():
            if isinstance(spec, Mapping):
                if "years" not in spec:
                    raise ValueError(f"{where}.datasets.{key} is missing the 'years' key.")
                raw_years = spec["years"]
                record = spec.get("record", spec.get("record_id"))
            else:
                raw_years = spec
                record = None
            try:
                years[key] = tuple(parse_years(raw_years))
            except ValueError as exc:
                raise ValueError(f"{where}.datasets.{key}: {exc}") from exc
            if kind == "zenodo":
                if record is None:
                    raise ValueError(
                        f"{where}.datasets.{key} is a zenodo dataset but declares no 'record' id.",
                    )
                records[key] = str(record)

        root = raw.get("root")
        if kind == "local":
            if not root:
                raise ValueError(f"{where}.root is required for a local source.")
            root = str(root)

        return cls(
            kind=kind,
            years=years,
            root=root,
            layout=str(raw.get("layout") or "{scenario}/{tech_dir}/{filename}"),
            records=records,
            copy_local=bool(raw.get("copy_local", False)),
            name=str(raw.get("name") or (root if kind == "local" else "zenodo")),
        )


@dataclass(frozen=True)
class CfResolution:
    """A fully resolved GODEEEP CF request — where exactly the file comes from."""

    kind: str
    """Source kind that won: ``"local"`` or ``"zenodo"``."""

    dataset_key: str
    """Registry key, e.g. ``wind_125m_historical_compressed``."""

    scenario: str
    """Climate scenario, e.g. ``historical`` / ``rcp85cooler``."""

    technology: str
    """GODEEEP technology family: ``solar`` / ``wind``."""

    wind_height: str
    """Hub-height suffix; empty for solar."""

    year: int
    """The requested (and confirmed available) year."""

    filename: str
    """Published file name inside the source."""

    path: str | None = None
    """Absolute path of the file for a ``local`` source, else ``None``."""

    record_id: str | None = None
    """Zenodo record id for a ``zenodo`` source, else ``None``."""

    copy_local: bool = False
    """Retrieve rule should copy rather than symlink a ``local`` hit."""

    source_index: int = 0
    """Position of the winning source in the configured list."""

    @property
    def location(self) -> str:
        """Source-specific location string (local path or ``zenodo:<record>``)."""
        return self.path if self.kind == "local" else f"zenodo:{self.record_id}"


def parse_years(spec) -> list[int]:
    """Parse a year availability spec into an explicit, sorted list of years.

    Accepts an inclusive range string (``"1980-2022"``) or an explicit sequence
    of years (``[2012]``, ``["2030", 2040]``). Anything else raises so a typo in
    the config cannot degrade into an empty — silently unavailable — year set.

    Raises
    ------
    ValueError
        If ``spec`` is not a range string or a sequence of four-digit years.
    """
    if isinstance(spec, bool):
        raise ValueError(f"invalid year spec {spec!r}; expected 'YYYY-YYYY' or a list of years.")
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, str):
        match = _YEAR_RANGE_RE.match(spec)
        if not match:
            raise ValueError(
                f"invalid year range {spec!r}; expected an inclusive range like '1980-2022'.",
            )
        start, end = int(match.group(1)), int(match.group(2))
        if end < start:
            raise ValueError(f"invalid year range {spec!r}; end year {end} precedes start year {start}.")
        return list(range(start, end + 1))
    if isinstance(spec, Sequence) and not isinstance(spec, str | bytes):
        years: list[int] = []
        for item in spec:
            if isinstance(item, bool) or not isinstance(item, int | str):
                raise ValueError(f"invalid year {item!r} in {spec!r}; expected an integer year.")
            try:
                years.append(int(item))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid year {item!r} in {spec!r}; expected an integer year.") from exc
        return sorted(set(years))
    raise ValueError(f"invalid year spec {spec!r}; expected 'YYYY-YYYY' or a list of years.")


def dataset_key(technology: str, wind_height: str, scenario: str) -> str:
    """Registry key for a ``(technology, hub height, scenario)`` triple.

    Reproduces the ``cf_record_key`` string ``build_renewable_profiles.py`` used
    to build inline, which is also the key space of the legacy
    ``ZenodoScenarioDownloader.scenario_records`` table — e.g.
    ``solar_historical_compressed``, ``wind_125m_rcp85cooler_compressed``.
    Compressed CF records are split by ``(tech, scenario)``, not by year: one
    record holds 2030/2040/2050 for the same scenario.
    """
    return f"{technology}{wind_height}_{scenario}_compressed"


def cf_filename(technology: str, wind_height: str, scenario: str, year: int) -> str:
    """Published file name of a compressed GODEEEP CF file.

    Identical for every scenario — the scenario lives in the containing record /
    directory, never in the file name. Examples:
    ``solar_gen_cf_2019_compressed.nc``,
    ``wind_gen_cf_2019_125m_compressed.nc``.

    ``scenario`` is accepted (and unused) so callers can pass the full request
    tuple and so a future scenario-dependent naming change has one place to go.
    """
    del scenario  # part of the request tuple, but not of the published name
    return f"{technology}_gen_cf_{year}{wind_height}_compressed.nc"


def godeeep_tech_spec(technology: str, config: Mapping) -> TechSpec:
    """Map a pypsa-usa technology wildcard onto its GODEEEP identity.

    Unlike the code this replaces there is NO ``"_100m"`` default: an absent or
    invalid ``godeeep_wind_height`` raises rather than quietly picking a hub
    height the configured source may not even hold (issue #803).

    Raises
    ------
    ValueError
        If ``technology`` is not a GODEEEP-backed carrier, or if
        ``godeeep_wind_height`` is missing/invalid for a wind carrier.
    """
    if technology in SOLAR_TECHNOLOGIES:
        return TechSpec("solar", "")
    if technology not in WIND_TECHNOLOGIES:
        raise ValueError(
            f"Invalid technology {technology!r} for the GODEEEP CF registry; "
            f"choose one of {list(GODEEEP_TECHNOLOGIES)}.",
        )

    wind_height = config.get("godeeep_wind_height")
    if wind_height is None:
        raise ValueError(
            f"godeeep_wind_height is not set, but technology {technology!r} needs a GODEEEP wind hub "
            f"height. Set it to one of {list(VALID_WIND_HEIGHTS)} (there is no default: '_100m' data "
            "is only published on the local Oak mirror, '_125m' on both Oak and Zenodo).",
        )
    wind_height = str(wind_height)
    if wind_height not in VALID_WIND_HEIGHTS:
        raise ValueError(
            f"godeeep_wind_height {wind_height!r} is not a published GODEEEP hub height; "
            f"valid values are {list(VALID_WIND_HEIGHTS)} (note '_80m' has no data).",
        )
    return TechSpec("wind", wind_height)


def resolve_scenario(config: Mapping) -> str:
    """First entry of ``renewable_scenarios``, with a clear error when unusable.

    Replaces the bare ``config["renewable_scenarios"][0]`` indexing in
    ``build_renewable_profiles.py`` and ``build_electricity.smk``, which raised
    an unattributed ``KeyError``/``IndexError`` at parse time.

    Raises
    ------
    ValueError
        If ``renewable_scenarios`` is missing, not a list, or empty.
    """
    if "renewable_scenarios" not in config:
        raise ValueError(
            "renewable_scenarios is not set. It is REQUIRED when renewable.dataset == 'godeeep'; "
            "set it to a one-element list, e.g. renewable_scenarios: ['historical'].",
        )
    scenarios = config["renewable_scenarios"]
    if isinstance(scenarios, str):
        raise ValueError(
            f"renewable_scenarios must be a list, got the string {scenarios!r}; "
            f"write renewable_scenarios: [{scenarios!r}].",
        )
    if not isinstance(scenarios, Sequence) or not scenarios:
        raise ValueError(
            f"renewable_scenarios must be a non-empty list of scenario names, got {scenarios!r}.",
        )
    return str(scenarios[0])


def resolve_weather_year(config: Mapping, planning_horizon=None) -> int:
    """The CF year implied by the configured scenario.

    Historical runs take the year from ``renewable_weather_years``; future
    scenarios take it from the ``{planning_horizon}`` wildcard, exactly as
    ``build_renewable_profiles.py`` did.

    Raises
    ------
    ValueError
        If the scenario is historical and ``renewable_weather_years`` is
        missing/empty/non-integer, or if a future scenario is requested without
        a ``planning_horizon``.
    """
    scenario = resolve_scenario(config)
    if scenario == "historical":
        if "renewable_weather_years" not in config:
            raise ValueError(
                "renewable_weather_years is not set, but renewable_scenarios == ['historical'] takes "
                "the GODEEEP CF year from it; set e.g. renewable_weather_years: [2019].",
            )
        years = config["renewable_weather_years"]
        if not isinstance(years, Sequence) or isinstance(years, str) or not years:
            raise ValueError(
                f"renewable_weather_years must be a non-empty list of years, got {years!r}.",
            )
        try:
            return int(years[0])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"renewable_weather_years[0] must be an integer year, got {years[0]!r}.",
            ) from exc
    if planning_horizon is None:
        raise ValueError(
            f"renewable_scenarios == [{scenario!r}] is a future GODEEEP scenario, so the CF year comes "
            "from the {planning_horizon} wildcard — but none was supplied. Either pass a planning "
            "horizon or set renewable_scenarios: ['historical'].",
        )
    try:
        return int(planning_horizon)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"planning_horizon must be an integer year, got {planning_horizon!r}.") from exc


def load_sources(config: Mapping) -> list[CfSource]:
    """Parse ``config["godeeep_cf_registry"]["sources"]`` in declaration order.

    Raises
    ------
    ValueError
        If the registry block is missing or any source entry is malformed.
    """
    registry = config.get(REGISTRY_KEY)
    if registry is None:
        raise ValueError(
            f"{REGISTRY_KEY} is not configured. GODEEEP CF retrieval is declared in config: add a "
            f"{REGISTRY_KEY}: sources: list (see workflow/repo_data/config/config.common.yaml).",
        )
    if not isinstance(registry, Mapping):
        raise ValueError(f"{REGISTRY_KEY} must be a mapping with a 'sources' list, got {type(registry).__name__}.")

    raw_sources = registry.get("sources")
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, str | bytes) or not raw_sources:
        raise ValueError(
            f"{REGISTRY_KEY}.sources must be a non-empty ordered list of sources, got {raw_sources!r}.",
        )

    default_copy_local = bool(registry.get("copy_local", False))
    sources: list[CfSource] = []
    for index, raw in enumerate(raw_sources):
        source = CfSource.from_config(raw, index)
        if isinstance(raw, Mapping) and "copy_local" not in raw:
            source = replace(source, copy_local=default_copy_local)
        sources.append(source)
    return sources


def _availability_report(sources: Sequence[CfSource], key: str) -> str:
    lines = []
    for source in sources:
        if key not in source.years:
            available = "(dataset key not declared)"
        elif not source.years[key]:
            # Declared on purpose with no years, e.g. a Zenodo record id that is
            # not (yet) published — say so rather than pretending it is unknown.
            available = "(declared, but no years published)"
        else:
            available = _format_years(source.years[key])
        lines.append(f"    - [{source.kind}] {source.name}: {available}")
    return "\n".join(lines) if lines else "    (no sources configured)"


def _format_years(years: Sequence[int]) -> str:
    """Render a year tuple compactly, collapsing contiguous runs to a range."""
    ordered = sorted({int(y) for y in years})
    if not ordered:
        return "(none)"
    if len(ordered) > 2 and ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]}-{ordered[-1]} (inclusive)"
    return ", ".join(str(y) for y in ordered)


def resolve_cf(config: Mapping, technology: str, planning_horizon=None) -> CfResolution:
    """Resolve a GODEEEP CF request to exactly one source, or raise.

    Walks ``godeeep_cf_registry.sources`` in configured order and returns the
    first source declaring the requested ``(dataset key, year)``. There is no
    fallback after resolution: the winning source must serve that exact file.

    Parameters
    ----------
    config
        Merged snakemake config.
    technology
        pypsa-usa technology wildcard (``solar``, ``onwind``, ``offwind``,
        ``offwind_floating``).
    planning_horizon
        Planning-horizon wildcard; used as the CF year for future scenarios and
        ignored for ``historical``.

    Returns
    -------
    CfResolution
        Source kind, location, dataset key, year and file name.

    Raises
    ------
    ValueError
        On a malformed config (bad wind height, unknown technology, malformed
        registry, missing scenario/weather-year keys).
    CfNotAvailableError
        If no configured source declares the requested dataset key and year.
    """
    spec = godeeep_tech_spec(technology, config)
    scenario = resolve_scenario(config)
    year = resolve_weather_year(config, planning_horizon)
    key = dataset_key(spec.technology, spec.wind_height, scenario)
    filename = cf_filename(spec.technology, spec.wind_height, scenario, year)
    sources = load_sources(config)

    for index, source in enumerate(sources):
        if not source.has(key, year):
            continue
        if source.kind == "local":
            relative = source.layout.format(
                scenario=scenario,
                tech_dir=spec.tech_dir,
                technology=spec.technology,
                wind_height=spec.wind_height,
                year=year,
                filename=filename,
                dataset_key=key,
            )
            path = str(Path(source.root) / relative)
            return CfResolution(
                kind="local",
                dataset_key=key,
                scenario=scenario,
                technology=spec.technology,
                wind_height=spec.wind_height,
                year=year,
                filename=filename,
                path=path,
                copy_local=source.copy_local,
                source_index=index,
            )
        record_id = source.records.get(key)
        if record_id is None:
            raise ValueError(
                f"godeeep_cf_registry.sources[{index}] declares years for {key!r} but no zenodo record id.",
            )
        return CfResolution(
            kind="zenodo",
            dataset_key=key,
            scenario=scenario,
            technology=spec.technology,
            wind_height=spec.wind_height,
            year=year,
            filename=filename,
            record_id=str(record_id),
            source_index=index,
        )

    raise CfNotAvailableError(
        f"No configured GODEEEP CF source provides dataset {key!r} for year {year}.\n"
        f"  requested: technology={technology!r} scenario={scenario!r} "
        f"hub_height={spec.wind_height or 'n/a'!r} file={filename!r}\n"
        f"  available years per source:\n{_availability_report(sources, key)}\n"
        f"  Fix: request a year listed above, add the dataset/year to "
        f"{REGISTRY_KEY}.sources, or point a local source at a mirror that holds it.",
    )


def _technologies_to_validate(config: Mapping) -> list[str]:
    """The GODEEEP-backed carriers present in the config (deduped by dataset)."""
    renewable = config.get("renewable") or {}
    if isinstance(renewable, Mapping):
        techs = [tech for tech in GODEEEP_TECHNOLOGIES if tech in renewable]
        if techs:
            return techs
    return ["solar", "onwind"]


def _years_to_validate(config: Mapping, scenario: str, problems: list[str]) -> list[int]:
    if scenario == "historical":
        years = config.get("renewable_weather_years")
        if not years or isinstance(years, str) or not isinstance(years, Sequence):
            problems.append(
                "renewable_weather_years must be a non-empty list of years when "
                f"renewable_scenarios == ['historical'] (got {years!r}).",
            )
            return []
        return [years[0]]
    horizons = (config.get("scenario") or {}).get("planning_horizons")
    if not horizons or isinstance(horizons, str) or not isinstance(horizons, Sequence):
        problems.append(
            f"scenario.planning_horizons must be a non-empty list when renewable_scenarios == [{scenario!r}]: "
            "future GODEEEP scenarios take the CF year from the planning horizon "
            f"(got {horizons!r}).",
        )
        return []
    return list(horizons)


def validate_godeeep_cf_config(config: Mapping) -> list[CfResolution]:
    """Fail fast, and all at once, on a broken GODEEEP CF configuration.

    Intended to run at snakemake parse time. Every configured
    ``scenario x year x technology`` combination is resolved eagerly; ALL
    problems (missing keys, invalid hub height, malformed sources, unresolvable
    combinations) are collected and reported in a single error rather than one
    per run.

    A config whose ``renewable.dataset`` is not ``godeeep`` is skipped.

    Returns
    -------
    list of CfResolution
        Every combination that resolved, for logging/debugging.

    Raises
    ------
    CfNotAvailableError
        If any check failed; the message enumerates all failures.
    """
    renewable = config.get("renewable")
    if isinstance(renewable, Mapping) and renewable.get("dataset", "godeeep") != "godeeep":
        return []

    problems: list[str] = []
    resolved: list[CfResolution] = []

    try:
        sources = load_sources(config)
    except ValueError as exc:
        sources = []
        problems.append(str(exc))

    try:
        scenario = resolve_scenario(config)
    except ValueError as exc:
        problems.append(str(exc))
        scenario = None

    technologies = _technologies_to_validate(config)
    specs: dict[str, TechSpec] = {}
    for technology in technologies:
        try:
            specs[technology] = godeeep_tech_spec(technology, config)
        except ValueError as exc:
            problems.append(str(exc))

    if scenario is not None and sources and specs:
        years = _years_to_validate(config, scenario, problems)
        seen: set[tuple[str, int]] = set()
        for year in years:
            try:
                year = int(year)
            except (TypeError, ValueError):
                problems.append(f"CF year {year!r} for scenario {scenario!r} is not an integer.")
                continue
            for technology, spec in specs.items():
                key = dataset_key(spec.technology, spec.wind_height, scenario)
                if (key, year) in seen:
                    continue
                seen.add((key, year))
                horizon = None if scenario == "historical" else year
                try:
                    resolved.append(resolve_cf(config, technology, horizon))
                except (CfNotAvailableError, ValueError) as exc:
                    problems.append(str(exc))

    if problems:
        # The three wind carriers share one TechSpec, so a bad hub height (etc.)
        # produces the same message once per carrier — report each problem once.
        problems = list(dict.fromkeys(problems))
        bullets = "\n\n".join(f"  ({i + 1}) {problem}" for i, problem in enumerate(problems))
        raise CfNotAvailableError(
            f"Invalid GODEEEP capacity-factor configuration — {len(problems)} problem(s):\n\n{bullets}\n\n"
            f"All GODEEEP CF retrieval is declared under '{REGISTRY_KEY}' in the config; there are no "
            "fallback paths (no default hub height, no nearest-year substitution).",
        )
    return resolved
