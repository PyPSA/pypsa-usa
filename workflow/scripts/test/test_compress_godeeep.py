"""Unit tests for nrel_exclusion/compress_godeeep.py (the uint8 CF quantizer).

``compress_one`` produced every file in the Oak registry
($OAK/GoDEEEP_Capacity_Factors_compressed), so its encoding contract is what the
registry readers decode against. These tests pin that contract on a miniature
grid shaped like the real thing (float ``capacity_factor`` over Time x
south_north x west_east, XLAT/XLONG aux vars carrying ``_FillValue``):

  * uint8 with scale_factor 1/254 and _FillValue 255, so xarray auto-decodes,
  * round-trip error bounded by the quantization step, NaN in -> NaN out,
  * coordinates, variable attributes and provenance survive the rewrite,
  * compressing the same input twice yields byte-identical uint8 payloads —
    the determinism the Phase 1 pilot leaned on when it proved the regenerated
    2012 files matched the published Zenodo artifacts.
"""

import os
import sys

import netCDF4
import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from nrel_exclusion.compress_godeeep import FILL, SCALE, compress_one

pytestmark = pytest.mark.fast

N_TIME, N_SOUTH_NORTH, N_WEST_EAST = 7, 3, 4
TOL = 1.0 / 254.0
PROJECTION = "LambertConformal(stand_lon=-97.0, moad_cen_lat=39.0)"


def make_raw_cf() -> np.ndarray:
    """Miniature capacity-factor field: full 0-1 range, a NaN row and a lone NaN cell."""
    rng = np.random.default_rng(20260831)
    cf = rng.random((N_TIME, N_SOUTH_NORTH, N_WEST_EAST)).astype(np.float32)
    cf[0, 0, 0] = 0.0  # exact endpoints must survive quantization
    cf[0, 0, 1] = 1.0
    cf[0, 0, 2] = 0.5
    cf[1, 2, :] = np.nan  # an offshore/out-of-domain row
    cf[3, 1, 2] = np.nan  # and a single dead cell
    return cf


def write_raw(path, cf: np.ndarray) -> None:
    """Write a raw-GODEEEP-shaped NetCDF (dims, aux vars and attrs mirror the real files)."""
    lat = np.linspace(31.0, 49.0, N_SOUTH_NORTH * N_WEST_EAST, dtype=np.float32).reshape(N_SOUTH_NORTH, N_WEST_EAST)
    lon = np.linspace(-124.0, -67.0, N_SOUTH_NORTH * N_WEST_EAST, dtype=np.float32).reshape(N_SOUTH_NORTH, N_WEST_EAST)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", N_TIME)
        ds.createDimension("south_north", N_SOUTH_NORTH)
        ds.createDimension("west_east", N_WEST_EAST)

        # Aux vars carrying _FillValue: it can only be set at create time, which is
        # the branch compress_one has to get right.
        for name, values in (("XLAT", lat), ("XLONG", lon)):
            var = ds.createVariable(name, "f4", ("south_north", "west_east"), fill_value=np.float32(np.nan))
            var[:] = values
            var.units = "degrees"

        ds.createVariable("Time", "i8", ("Time",))[:] = np.arange(N_TIME, dtype=np.int64)
        ds.createVariable("south_north", "i8", ("south_north",))[:] = np.arange(N_SOUTH_NORTH, dtype=np.int64)
        ds.createVariable("west_east", "i8", ("west_east",))[:] = np.arange(N_WEST_EAST, dtype=np.int64)

        cf_var = ds.createVariable(
            "capacity_factor",
            "f4",
            ("Time", "south_north", "west_east"),
            fill_value=np.float32(np.nan),
        )
        cf_var.set_auto_mask(False)
        cf_var[:] = cf
        cf_var.projection = PROJECTION
        cf_var.coordinates = "XLAT XLONG"
        cf_var.missing_value = np.float32(-999.0)  # dropped by compress_one's skip list
        ds.title = "synthetic GODEEEP capacity factors"


@pytest.fixture
def raw_cf() -> np.ndarray:
    return make_raw_cf()


@pytest.fixture
def raw_path(tmp_path, raw_cf):
    path = tmp_path / "solar_gen_cf_2012.nc"
    write_raw(path, raw_cf)
    return path


@pytest.fixture
def compressed_path(tmp_path, raw_path):
    out = tmp_path / "solar_gen_cf_2012_compressed.nc"
    compress_one(str(raw_path), str(out), chunk_t=3)  # 7 hours / 3 => a partial final chunk
    return out


def read_packed(path) -> np.ndarray:
    """The stored uint8 payload, with auto-scaling and auto-masking switched off."""
    with netCDF4.Dataset(path) as ds:
        var = ds.variables["capacity_factor"]
        var.set_auto_scale(False)
        var.set_auto_mask(False)
        return np.asarray(var[:])


def test_encoding_is_uint8_scaled_by_1_over_254(compressed_path):
    with netCDF4.Dataset(compressed_path) as ds:
        var = ds.variables["capacity_factor"]
        assert var.dtype == np.uint8
        assert var.getncattr("scale_factor") == pytest.approx(1.0 / 254.0)
        assert var.getncattr("scale_factor") == SCALE
        assert var.getncattr("add_offset") == pytest.approx(0.0)
        assert int(var.getncattr("_FillValue")) == int(FILL) == 255
        assert var.dimensions == ("Time", "south_north", "west_east")


def test_roundtrip_error_within_quantization_step(compressed_path, raw_cf):
    with netCDF4.Dataset(compressed_path) as ds:
        decoded = ds.variables["capacity_factor"][:]  # auto-decoded to float, 255 -> masked
    decoded = np.ma.filled(decoded, np.nan).astype(np.float64)

    valid = ~np.isnan(raw_cf)
    err = np.abs(decoded[valid] - raw_cf.astype(np.float64)[valid]).max()
    assert err <= TOL + 1e-9
    # The endpoints of the 0-1 range are representable exactly.
    assert decoded[0, 0, 0] == pytest.approx(0.0, abs=1e-9)
    assert decoded[0, 0, 1] == pytest.approx(1.0, abs=1e-6)


def test_nan_survives_as_fill_255(compressed_path, raw_cf):
    packed = read_packed(compressed_path)
    nan_in = np.isnan(raw_cf)
    assert np.array_equal(packed == int(FILL), nan_in)

    with netCDF4.Dataset(compressed_path) as ds:
        decoded = ds.variables["capacity_factor"][:]
    decoded = np.ma.filled(decoded, np.nan)
    assert np.array_equal(np.isnan(decoded), nan_in)
    assert nan_in.sum() == N_WEST_EAST + 1  # the NaN row plus the dead cell


def test_coords_and_attrs_preserved(compressed_path, raw_path):
    with netCDF4.Dataset(raw_path) as src, netCDF4.Dataset(compressed_path) as dst:
        assert {name: len(dim) for name, dim in dst.dimensions.items()} == {
            "Time": N_TIME,
            "south_north": N_SOUTH_NORTH,
            "west_east": N_WEST_EAST,
        }
        for name in ("XLAT", "XLONG", "Time", "south_north", "west_east"):
            assert np.array_equal(np.asarray(src.variables[name][:]), np.asarray(dst.variables[name][:])), name
            assert dst.variables[name].dtype == src.variables[name].dtype, name

        # _FillValue is re-created (not copied as a plain attribute) on aux vars.
        assert np.isnan(dst.variables["XLAT"].getncattr("_FillValue"))
        assert dst.variables["XLAT"].getncattr("units") == "degrees"

        cf_var = dst.variables["capacity_factor"]
        assert cf_var.getncattr("projection") == PROJECTION
        assert cf_var.getncattr("coordinates") == "XLAT XLONG"
        # The source's float encoding attributes must not leak onto the uint8 field.
        assert "missing_value" not in cf_var.ncattrs()


def test_provenance_attrs_stamped(compressed_path, raw_path):
    with netCDF4.Dataset(compressed_path) as ds:
        assert ds.getncattr("compression_source") == str(raw_path)
        assert ds.getncattr("compression_encoding") == "uint8 scale_factor=1/254 zlib=4"


def test_compressing_twice_is_deterministic(tmp_path, raw_path):
    """Two runs over the same input must agree bit-for-bit on the packed payload."""
    first = tmp_path / "first.nc"
    second = tmp_path / "second.nc"
    compress_one(str(raw_path), str(first), chunk_t=3)
    compress_one(str(raw_path), str(second), chunk_t=5)  # chunking must not change the payload

    assert read_packed(first).tobytes() == read_packed(second).tobytes()
