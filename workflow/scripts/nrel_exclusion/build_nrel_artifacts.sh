#!/bin/bash
# Regenerate NREL availability + caps NetCDFs for each (tech, access) combo,
# applying the CEC BaseScreen and BOEM OSW filters as configured below.
#
# Outputs land in data/nrel_exclusion/derived/, named with suffixes that match
# rules/build_electricity.smk's input lambdas (e.g. avail_solar_reference_cec.nc,
# avail_offwind_floating_reference_boem.nc).
#
# Idempotent: files already on disk are skipped. Remove them to force rebuild.
#
# Usage:
#   bash scripts/nrel_exclusion/build_nrel_artifacts.sh                              # reference, all 4 techs
#   bash scripts/nrel_exclusion/build_nrel_artifacts.sh limited                       # different access
#   TECHS="solar onwind" bash scripts/nrel_exclusion/build_nrel_artifacts.sh          # subset
#   APPLY_CEC=0 APPLY_BOEM=0 bash scripts/nrel_exclusion/build_nrel_artifacts.sh      # bare NREL (no overlays)

set -euo pipefail

# ---- knobs (keep in sync with config/godeeep/config.nrel_smoke.yaml) ----
APPLY_CEC="${APPLY_CEC:-1}"
APPLY_BOEM="${APPLY_BOEM:-1}"
ACCESS="${1:-reference}"
TECHS="${TECHS:-solar onwind offwind offwind_floating}"
INTERCONNECT="${INTERCONNECT:-western}"
BUS_SUBDIR="${BUS_SUBDIR:-nrel_smoke_rcp45cooler_2030_reference}"

# ---- paths ----
REPO=/home/groups/iazevedo/asia/pypsa-usa/workflow
PY=/home/groups/iazevedo/asia/miniforge3/envs/pypsa-usa/bin/python
SCRIPTS="$REPO/scripts/nrel_exclusion"
OUT="$REPO/data/nrel_exclusion/derived"

# Any raw GODEEEP file works for --godeeep: the Lambert Conformal grid is
# identical across (scenario, year, tech). Solar/2030 is a convenient pick.
GODEEEP_REF="${GODEEEP_REF:-/scratch/groups/iazevedo/asia/godeeep/aggregate_by_county/solar_rcp45cooler_2020_2059/solar_gen_cf_2030.nc}"

BUS_DIR="$REPO/resources/godeeep/geospatial/$BUS_SUBDIR/$INTERCONNECT"
ONSHORE_SHAPES="$BUS_DIR/regions_onshore.geojson"
OFFSHORE_SHAPES="$BUS_DIR/regions_offshore.geojson"
OFFSHORE_EEZ="$BUS_DIR/offshore_shapes.geojson"

# Avoid OpenBLAS blowing past the shared-node RLIMIT_NPROC.
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p "$OUT"

build_suffix() {
    local tech=$1 s=""
    if [ "$APPLY_CEC" = 1 ] && { [ "$tech" = onwind ] || [ "$tech" = solar ]; }; then
        s+="_cec"
    fi
    if [ "$APPLY_BOEM" = 1 ] && [[ "$tech" == offwind* ]]; then
        s+="_boem"
    fi
    echo "$s"
}

build_flags() {
    local tech=$1 f=""
    if [ "$APPLY_CEC" = 1 ] && { [ "$tech" = onwind ] || [ "$tech" = solar ]; }; then
        f+=" --apply-cec-basescreen"
    fi
    if [ "$APPLY_BOEM" = 1 ] && [[ "$tech" == offwind* ]]; then
        f+=" --apply-boem-osw"
    fi
    echo "$f"
}

echo "config: APPLY_CEC=$APPLY_CEC  APPLY_BOEM=$APPLY_BOEM  ACCESS=$ACCESS"
echo "techs : $TECHS"
echo "buses : $BUS_DIR"
echo "output: $OUT"

for tech in $TECHS; do
    suffix=$(build_suffix "$tech")
    flags=$(build_flags "$tech")
    avail="$OUT/avail_${tech}_${ACCESS}${suffix}.nc"
    caps="$OUT/caps_${tech}_${ACCESS}${suffix}.nc"

    echo
    echo "===================================================================="
    echo " $tech / $ACCESS  (flags:${flags:- none})"
    echo "===================================================================="

    if [ -f "$avail" ]; then
        echo "[skip] $avail already exists"
    else
        echo "[run ] avail → $avail"
        $PY "$SCRIPTS/build_nrel_availability.py" \
            --tech "$tech" --access "$ACCESS" \
            --godeeep "$GODEEEP_REF" \
            $flags \
            --output "$avail"
    fi

    if [ -f "$caps" ]; then
        echo "[skip] $caps already exists"
    else
        echo "[run ] caps  → $caps"
        extra=""
        if [[ "$tech" == offwind* ]]; then
            extra="--offshore-shapes $OFFSHORE_SHAPES --offshore-eez-shape $OFFSHORE_EEZ"
        fi
        $PY "$SCRIPTS/build_nrel_bus_capacities.py" \
            --tech "$tech" --access "$ACCESS" \
            --onshore-shapes "$ONSHORE_SHAPES" \
            $extra \
            $flags \
            --output "$caps"
    fi
done

echo
echo "===== artifacts in $OUT (access=$ACCESS) ====="
ls -la "$OUT"/*"_${ACCESS}"*.nc 2>/dev/null || echo "  (none)"
