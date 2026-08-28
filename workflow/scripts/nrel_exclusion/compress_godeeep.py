"""
Compress a raw GODEEEP capacity-factor NetCDF by encoding the 0-1 float CF field
as uint8 (scale_factor = 1/254, fill_value = 255) with chunked zlib.

Produces a ~350 MB file from a ~4.4 GB input (~12x) with max absolute error
~0.002 — below the quantization step. All coordinate variables and variable
attributes are preserved so downstream consumers can open the file with xarray
and have CF auto-decoded back to float.
"""

import argparse
import os
import time
from pathlib import Path

import netCDF4
import numpy as np

DEFAULT_CHUNK_T = 500  # hours per time chunk
SCALE = np.float32(1.0 / 254.0)
FILL = np.uint8(255)


def compress_one(src_path: str, out_path: str, chunk_t: int = DEFAULT_CHUNK_T) -> None:
    print(f"[compress] source: {src_path} ({os.path.getsize(src_path) / 1e9:.2f} GB)", flush=True)
    t_start = time.time()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with netCDF4.Dataset(src_path) as srcf, netCDF4.Dataset(out_path, "w") as dstf:
        # Copy dimensions with original names (keep "Time" so downstream code is unchanged)
        for name, dim in srcf.dimensions.items():
            dstf.createDimension(name, len(dim))

        # Copy all coordinate/auxiliary variables except the main CF field.
        # _FillValue must be set at create time (not via setncattr), so pass it
        # explicitly if present and skip it in the attribute-copy loop.
        for name, var in srcf.variables.items():
            if name == "capacity_factor":
                continue
            fill = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
            dst_var = dstf.createVariable(name, var.dtype, var.dimensions, fill_value=fill)
            dst_var[:] = var[:]
            for attr in var.ncattrs():
                if attr == "_FillValue":
                    continue
                dst_var.setncattr(attr, var.getncattr(attr))

        src_cf = srcf.variables["capacity_factor"]
        time_count, ns_count, ew_count = src_cf.shape

        cfv = dstf.createVariable(
            "capacity_factor",
            "u1",
            src_cf.dimensions,
            zlib=True,
            complevel=4,
            chunksizes=(min(chunk_t, time_count), ns_count, ew_count),
            fill_value=FILL,
        )
        cfv.setncattr("scale_factor", SCALE)
        cfv.setncattr("add_offset", np.float32(0.0))
        for attr in src_cf.ncattrs():
            if attr in ("_FillValue", "scale_factor", "add_offset", "missing_value"):
                continue
            cfv.setncattr(attr, src_cf.getncattr(attr))
        # We quantize ourselves; tell the library not to re-scale on write.
        cfv.set_auto_scale(False)
        cfv.set_auto_mask(False)

        for t0 in range(0, time_count, chunk_t):
            t1 = min(t0 + chunk_t, time_count)
            chunk = src_cf[t0:t1, :, :]
            mask = np.isnan(chunk)
            quant = np.clip(np.round(chunk * 254.0), 0, 254).astype(np.uint8)
            quant[mask] = FILL
            cfv[t0:t1, :, :] = quant
            print(f"[compress]   wrote hours {t0}:{t1}", flush=True)

        # Provenance
        dstf.setncattr("compression_source", src_path)
        dstf.setncattr("compression_encoding", "uint8 scale_factor=1/254 zlib=4")

    elapsed = time.time() - t_start
    out_size = os.path.getsize(out_path)
    src_size = os.path.getsize(src_path)
    print(
        f"[compress] wrote {out_path}: {out_size / 1e6:.1f} MB ({src_size / out_size:.1f}x smaller) in {elapsed:.1f}s",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Raw GODEEEP .nc file.")
    ap.add_argument("--output", required=True, help="Destination compressed .nc file.")
    ap.add_argument(
        "--chunk-t",
        type=int,
        default=DEFAULT_CHUNK_T,
        help="Hours per time chunk during read/write (controls peak memory).",
    )
    args = ap.parse_args()
    compress_one(args.input, args.output, chunk_t=args.chunk_t)


if __name__ == "__main__":
    main()
