#!/usr/bin/env bash
# build_optimized.sh — Build prgm_opt.exe with -O2 (no debug flags)
#
# Usage: bash build_optimized.sh
#        COMPILER=gfortran bash build_optimized.sh
#
# Produces:  clm-ml-fortran/offline_executable/prgm_opt.exe
# The default Makefile uses -C -Ktrap=fp (bounds+trap); this strips those
# flags and adds -O2 for a fair performance measurement.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/offline_executable"
OBJ_DIR="${BUILD_DIR}/obj"

# ── Compiler selection ────────────────────────────────────────────────────────
# Override with:  COMPILER=gfortran bash build_optimized.sh
COMPILER="${COMPILER:-auto}"

if [ "${COMPILER}" = "auto" ]; then
    if command -v nvfortran &>/dev/null; then
        COMPILER=nvfortran
    elif command -v gfortran &>/dev/null; then
        COMPILER=gfortran
    else
        echo "ERROR: no Fortran compiler found (tried nvfortran, gfortran)." >&2
        echo "Set COMPILER=<your_compiler> and re-run." >&2
        exit 1
    fi
fi

echo "Using compiler: ${COMPILER}"

# ── NetCDF path ───────────────────────────────────────────────────────────────
if [ -z "${NETCDF_PATH:-}" ]; then
    # Try common module-load locations
    for candidate in \
        /usr/local/netcdf \
        /opt/netcdf \
        "$(nc-config --prefix 2>/dev/null || true)"; do
        if [ -d "${candidate}/lib" ]; then
            export NETCDF_PATH="${candidate}"
            break
        fi
    done
fi

if [ -z "${NETCDF_PATH:-}" ]; then
    echo "WARNING: NETCDF_PATH not set and could not be auto-detected." >&2
    echo "  Set it with:  export NETCDF_PATH=/path/to/netcdf" >&2
    echo "  Proceeding anyway; build may fail if NetCDF is not on the default path." >&2
    NETCDF_PATH="/usr"   # last-resort fallback
fi

echo "NETCDF_PATH: ${NETCDF_PATH}"

# ── Build flags ───────────────────────────────────────────────────────────────
# Remove bounds-checking and floating-point trap flags; add -O2.
LIB_NETCDF="${NETCDF_PATH}/lib"
MOD_NETCDF="${NETCDF_PATH}/include"

case "${COMPILER}" in
    nvfortran)
        FFLAGS="-O2"
        LDFLAGS="-L${LIB_NETCDF} -I${MOD_NETCDF} -lnetcdff -lblas -lm -L/usr/lib64"
        ;;
    gfortran)
        FFLAGS="-O2 -fno-range-check -ffree-line-length-none"
        LDFLAGS="-L${LIB_NETCDF} -I${MOD_NETCDF} -lnetcdff -lnetcdf -lblas -lm -L/usr/lib64"
        ;;
    *)
        FFLAGS="-O2"
        LDFLAGS="-L${LIB_NETCDF} -I${MOD_NETCDF} -lnetcdff -lblas -lm"
        ;;
esac

CMPLR="${COMPILER} ${FFLAGS} ${LDFLAGS}"

# ── Write a temporary Makefile override ──────────────────────────────────────
# We reuse the existing Makefile but override the cmplr variable.
cd "${BUILD_DIR}"

# Clean previous objects so the optimized version is a fresh build
echo "Cleaning previous object files ..."
make clean 2>/dev/null || true

echo "Building prgm_opt.exe with flags: ${FFLAGS} ..."
make prgm.exe "cmplr=${CMPLR}"

# Rename so it coexists with the debug build
mv prgm.exe prgm_opt.exe
echo ""
echo "Success: ${BUILD_DIR}/prgm_opt.exe"
echo ""
echo "Test with:"
echo "  cd ${BUILD_DIR}"
echo "  ./prgm_opt.exe < nl.CHATS7.05.2007"
