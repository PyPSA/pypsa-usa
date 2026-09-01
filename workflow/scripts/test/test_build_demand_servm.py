"""Tests for the CPUC SERVM demand profile reader."""

import logging

import numpy as np
import pandas as pd
import pytest
from _helpers import get_multiindex_snapshots
from build_demand import Context, ReadServm

HOURS = ReadServm.HOURS_PER_YEAR

# Component layout copied from the published file. IID/LADWP/NCNC carry the six
# universal components; PGE/SCE/SDGE additionally carry the EV, BTM-storage,
# climate-change and data-centre components.
SIMPLE_COMPONENTS = ("Load", "Modified Load", "Net Load", "BTMPV", "AAEE", "AAFS")
SIMPLE_UNITS = ("", "", "", "R", "R", "R")
FULL_COMPONENTS = (
    "Load",
    "Modified Load",
    "Net Load",
    "BTMPV",
    "AAEE",
    "EV",
    "BTMStorageShapeDischarge",
    "AAFS",
    "BTMStorageShapeCharge",
    "CLIM_CHG_nou",
    "CLIM_CHG_gen",
    "DATA_CEN",
)
FULL_UNITS = ("", "", "", "R", "R", "R", "R", "R", "R", "R", "R", "R")

REGION_LAYOUT = (
    ("IID", SIMPLE_COMPONENTS, SIMPLE_UNITS),
    ("LADWP", SIMPLE_COMPONENTS, SIMPLE_UNITS),
    ("NCNC", SIMPLE_COMPONENTS, SIMPLE_UNITS),
    ("PGE", FULL_COMPONENTS, FULL_UNITS),
    ("SCE", FULL_COMPONENTS, FULL_UNITS),
    ("SDGE", FULL_COMPONENTS, FULL_UNITS),
)

# Literal first two header rows of the real file, for the leading calendar block.
# The seventh column carries the stray ('Region', 'Unit Type') labels.
INDEX_HEADER_0 = ["", "", "", "", "", "", "Region"]
INDEX_HEADER_1 = ["", "", "", "", "", "", "Unit Type"]
INDEX_HEADER_2 = [
    "Weather Year",
    "Season",
    "Month",
    "Day",
    "Day of Month",
    "Hour",
    "Hour of Day",
]


def _column_layout(drop=()):
    """(region, unit, component) triples in published order, minus `drop`."""
    columns = []
    for region, components, units in REGION_LAYOUT:
        for component, unit in zip(components, units, strict=True):
            if (region, component) in drop:
                continue
            columns.append((region, unit, component))
    return columns


def _value(column_index: int, weather_year: int, hour: int, offset: int = 0) -> int:
    """Deterministic, collision-free cell value."""
    return offset + column_index * 100_000 + (weather_year - 2000) * 10_000 + hour


def write_servm_fixture(
    path,
    weather_years=(2000, 2001),
    offset: int = 0,
    drop=(),
):
    """Write a synthetic SERVM CSV carrying the real three-row header quirks."""
    columns = _column_layout(drop=drop)
    n_columns = len(columns)

    header_lines = [
        ",".join(INDEX_HEADER_0 + [region for region, _, _ in columns]),
        ",".join(INDEX_HEADER_1 + [unit for _, unit, _ in columns]),
        ",".join(INDEX_HEADER_2 + [component for _, _, component in columns]),
    ]

    hours = np.arange(HOURS)
    blocks = []
    for weather_year in weather_years:
        index_block = pd.DataFrame(
            {
                "Weather Year": weather_year,
                "Season": "Winter",
                "Month": 1,
                "Day": 1,
                "Day of Month": 1,
                "Hour": hours + 1,
                "Hour of Day": (hours % 24) + 1,
            },
        )
        data = np.empty((HOURS, n_columns), dtype=np.int64)
        for column_index in range(n_columns):
            data[:, column_index] = _value(column_index, weather_year, hours, offset)
        blocks.append(pd.concat([index_block, pd.DataFrame(data)], axis=1))

    body = pd.concat(blocks, ignore_index=True)
    with open(path, "w") as f:
        f.write("\n".join(header_lines) + "\n")
        body.to_csv(f, header=False, index=False, lineterminator="\n")
    return path


def make_snapshots(planning_horizons, base_year=2019):
    """Snapshots exactly as the pipeline builds them."""
    return get_multiindex_snapshots(
        {
            "start": f"{base_year}-01-01 00:00",
            "end": f"{base_year}-12-31 23:00",
            "inclusive": "both",
        },
        planning_horizons,
    )


def column_index_of(region: str, component: str, drop=()) -> int:
    columns = _column_layout(drop=drop)
    return columns.index(
        next(c for c in columns if c[0] == region and c[2] == component),
    )


@pytest.fixture(scope="module")
def servm_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("servm") / "HourlyLoad_CA_Regions_V2025E_2224_Mon_2028.csv"
    return str(write_servm_fixture(path))


@pytest.fixture(scope="module")
def servm_demand(servm_file):
    reader = ReadServm(
        servm_file,
        planning_horizons=[2028],
        servm_weather_years=[2000],
        snapshots=make_snapshots([2028]),
    )
    return reader.read_demand()


def test_multiheader_parse_recovers_region_and_component(servm_demand):
    """The three-row header resolves to regions on columns, components on subsector."""
    assert list(servm_demand.columns) == list(ReadServm.REGIONS)
    assert servm_demand.index.names == ["snapshot", "sector", "subsector", "fuel"]

    subsectors = set(servm_demand.index.get_level_values("subsector"))
    assert "Net Load" in subsectors
    assert set(servm_demand.index.get_level_values("sector")) == {"all"}
    assert set(servm_demand.index.get_level_values("fuel")) == {"electricity"}
    assert len(servm_demand) == HOURS * len(set(FULL_COMPONENTS) | set(SIMPLE_COMPONENTS))


def test_weather_year_filter_selects_correct_block(servm_file):
    """The chosen weather year selects its own 8760-row block, not the first one."""
    snapshots = make_snapshots([2028])
    column = column_index_of("SCE", "Net Load")

    values = {}
    for weather_year in (2000, 2001):
        reader = ReadServm(
            servm_file,
            planning_horizons=[2028],
            servm_weather_years=[weather_year],
            snapshots=snapshots,
        )
        demand = reader.read_demand()
        # hour 0 of the strip lands PST_TO_UTC_SHIFT hours into the year
        stamp = snapshots.get_level_values(1)[ReadServm.PST_TO_UTC_SHIFT]
        values[weather_year] = demand.loc[(stamp, "all", "Net Load", "electricity"), "SCE"]

    assert values[2000] == _value(column, 2000, 0)
    assert values[2001] == _value(column, 2001, 0)
    assert values[2000] != values[2001]


def test_all_components_preserved_on_subsector_level(servm_demand):
    """Every published component survives; EV stays absent where CPUC omits it."""
    subsectors = set(servm_demand.index.get_level_values("subsector"))
    assert subsectors == set(SIMPLE_COMPONENTS) | set(FULL_COMPONENTS)

    ev = servm_demand.xs("EV", level="subsector")
    assert ev[["PGE", "SCE", "SDGE"]].notna().all().all()
    assert ev[["IID", "LADWP", "NCNC"]].isna().all().all()

    net_load = servm_demand.xs("Net Load", level="subsector")
    assert net_load.notna().all().all()


def test_net_load_is_the_default_subsector(servm_file):
    """Context filters to Net Load without main() knowing about SERVM."""
    assert ReadServm.default_subsector == "Net Load"

    reader = ReadServm(
        servm_file,
        planning_horizons=[2028],
        servm_weather_years=[2000],
        snapshots=make_snapshots([2028]),
    )

    class _RecordingWriter:
        def __init__(self):
            self.kwargs = None

        def dissagregate_demand(self, df, zone, **kwargs):
            self.kwargs = dict(kwargs, zone=zone)
            return df

    writer = _RecordingWriter()
    Context(reader, writer).prepare_demand()

    assert writer.kwargs["subsector"] == "Net Load"
    assert writer.kwargs["zone"] == "servm"


def test_explicit_subsector_overrides_the_default(servm_file):
    """An explicit subsector= argument still wins over the reader's default."""
    reader = ReadServm(
        servm_file,
        planning_horizons=[2028],
        servm_weather_years=[2000],
        snapshots=make_snapshots([2028]),
    )

    class _RecordingWriter:
        def __init__(self):
            self.kwargs = None

        def dissagregate_demand(self, df, zone, **kwargs):
            self.kwargs = kwargs
            return df

    writer = _RecordingWriter()
    Context(reader, writer).prepare_demand(subsector="BTMPV")

    assert writer.kwargs["subsector"] == "BTMPV"


def test_snapshots_align_with_leap_model_year(servm_file):
    """A leap planning horizon keeps the network's own (Feb-29-free) snapshots."""
    snapshots = make_snapshots([2028])
    timesteps = snapshots.get_level_values(1)
    assert len(timesteps) == HOURS  # the pipeline drops Feb 29

    reader = ReadServm(
        servm_file,
        planning_horizons=[2028],
        servm_weather_years=[2000],
        snapshots=snapshots,
    )
    demand = reader.read_demand()

    stamps = pd.DatetimeIndex(demand.index.get_level_values("snapshot"))
    assert stamps.year.unique().tolist() == [2028]
    assert not ((stamps.month == 2) & (stamps.day == 29)).any()
    assert stamps.max() == pd.Timestamp("2028-12-31 23:00")

    net_load = demand.xs("Net Load", level="subsector")
    got = pd.DatetimeIndex(net_load.index.get_level_values("snapshot"))
    pd.testing.assert_index_equal(got, timesteps, check_names=False)


def test_pst_shift_rolls_by_eight(servm_file):
    """PST (UTC-8) with no DST: hour 0 of the strip is 08:00 UTC."""
    snapshots = make_snapshots([2028])
    timesteps = snapshots.get_level_values(1)
    column = column_index_of("IID", "Net Load")

    reader = ReadServm(
        servm_file,
        planning_horizons=[2028],
        servm_weather_years=[2000],
        snapshots=snapshots,
    )
    demand = reader.read_demand().xs("Net Load", level="subsector")

    shift = ReadServm.PST_TO_UTC_SHIFT
    assert demand.loc[(timesteps[shift], "all", "electricity"), "IID"] == _value(column, 2000, 0)
    # the tail of the strip wraps onto the first hours of the year
    assert demand.loc[(timesteps[0], "all", "electricity"), "IID"] == _value(column, 2000, HOURS - shift)


def test_rejects_unsupported_planning_horizon(tmp_path):
    with pytest.raises(ValueError, match="unsupported year"):
        ReadServm(
            str(tmp_path / "HourlyLoad_2027.csv"),
            planning_horizons=[2027],
            servm_weather_years=[2019],
            snapshots=make_snapshots([2027]),
        )


def test_rejects_unsupported_weather_year(tmp_path):
    with pytest.raises(ValueError, match="supports weather years"):
        ReadServm(
            str(tmp_path / "HourlyLoad_2028.csv"),
            planning_horizons=[2028],
            servm_weather_years=[1999],
            snapshots=make_snapshots([2028]),
        )


def test_multiple_weather_years_raises_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError, match="stochastic scenarios"):
        ReadServm(
            str(tmp_path / "HourlyLoad_2028.csv"),
            planning_horizons=[2028],
            servm_weather_years=[2018, 2019],
            snapshots=make_snapshots([2028]),
        )


def test_mismatched_renewable_weather_years_warns(tmp_path, caplog):
    path = tmp_path / "HourlyLoad_CA_Regions_V2025E_2224_Mon_2028.csv"
    path.touch()

    with caplog.at_level(logging.WARNING, logger="build_demand"):
        ReadServm(
            str(path),
            planning_horizons=[2028],
            servm_weather_years=[2019],
            renewable_weather_years=[2012],
            snapshots=make_snapshots([2028]),
        )

    assert "does not match renewable_weather_years" in caplog.text


def test_matched_renewable_weather_years_do_not_warn(tmp_path, caplog):
    path = tmp_path / "HourlyLoad_CA_Regions_V2025E_2224_Mon_2028.csv"
    path.touch()

    with caplog.at_level(logging.WARNING, logger="build_demand"):
        ReadServm(
            str(path),
            planning_horizons=[2028],
            servm_weather_years=[2019],
            renewable_weather_years=[2019],
            snapshots=make_snapshots([2028]),
        )

    assert "does not match renewable_weather_years" not in caplog.text


def test_missing_net_load_column_raises(tmp_path):
    """A CPUC layout change that drops a region's Net Load must fail loudly."""
    path = tmp_path / "HourlyLoad_CA_Regions_V2025E_2224_Mon_2028.csv"
    write_servm_fixture(path, weather_years=(2000,), drop=(("SDGE", "Net Load"),))

    reader = ReadServm(
        str(path),
        planning_horizons=[2028],
        servm_weather_years=[2000],
        snapshots=make_snapshots([2028]),
    )
    with pytest.raises(ValueError, match="missing the 'Net Load' column"):
        reader.read_demand()


def test_files_indexed_by_basename_year_not_order(tmp_path):
    """Files are matched to horizons by their filename year, whatever the order."""
    file_2026 = tmp_path / "HourlyLoad_CA_Regions_V2025E_2224_Mon_2026.csv"
    file_2028 = tmp_path / "HourlyLoad_CA_Regions_V2025E_2224_Mon_2028.csv"
    write_servm_fixture(file_2026, weather_years=(2000,), offset=0)
    write_servm_fixture(file_2028, weather_years=(2000,), offset=1_000_000_000)

    # deliberately reversed relative to the planning horizons
    reader = ReadServm(
        [str(file_2028), str(file_2026)],
        planning_horizons=[2026, 2028],
        servm_weather_years=[2000],
        snapshots=make_snapshots([2026, 2028]),
    )
    assert reader.files == {2026: str(file_2026), 2028: str(file_2028)}

    demand = reader.read_demand().xs("Net Load", level="subsector")
    column = column_index_of("PGE", "Net Load")
    shift = ReadServm.PST_TO_UTC_SHIFT

    hour_2026 = pd.Timestamp("2026-01-01 00:00") + pd.Timedelta(hours=shift)
    hour_2028 = pd.Timestamp("2028-01-01 00:00") + pd.Timedelta(hours=shift)
    assert demand.loc[(hour_2026, "all", "electricity"), "PGE"] == _value(column, 2000, 0)
    assert demand.loc[(hour_2028, "all", "electricity"), "PGE"] == _value(column, 2000, 0, offset=1_000_000_000)


def test_missing_file_for_planning_horizon_raises(tmp_path):
    path = tmp_path / "HourlyLoad_CA_Regions_V2025E_2224_Mon_2026.csv"
    path.touch()

    with pytest.raises(ValueError, match=r"No SERVM load file provided for planning horizon"):
        ReadServm(
            str(path),
            planning_horizons=[2026, 2028],
            servm_weather_years=[2019],
            snapshots=make_snapshots([2026, 2028]),
        )


def test_snapshots_are_required(tmp_path):
    path = tmp_path / "HourlyLoad_CA_Regions_V2025E_2224_Mon_2028.csv"
    path.touch()

    with pytest.raises(ValueError, match="requires the network snapshots"):
        ReadServm(
            str(path),
            planning_horizons=[2028],
            servm_weather_years=[2019],
        )
