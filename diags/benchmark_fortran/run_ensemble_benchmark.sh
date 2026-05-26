#!/usr/bin/env bash
# run_ensemble_benchmark.sh — Parameter ensemble benchmark for Fortran CLM-ML.
#
# Fortran cannot batch parameter samples the way JAX vmap can.
# Each "parameter sample" = 1 run of prgm.exe with the same parameters.
# (Fortran has no equivalent of jax.vmap over parameter vectors.)
#
# For N=1,8,32,128,512,1024,2048 samples:
#   Sequential: run N instances one after another; total = N × single_run_time.
#               The per-sample cost is constant across all N (no batching benefit).
#   Parallel:   run N instances simultaneously (bash &); wall clock = single_run_time
#               independent of N (if enough cores), showing the parallelism ceiling.
#
# Because large N (512, 1024, 2048) would take very long sequentially, we use
# extrapolation for N>32 sequential: single_run_ms × N.
# For parallel runs, we cap at min(N, MAX_PARALLEL_JOBS) simultaneous processes.
#
# Each run: stop_n=6 (1 warmup + 5 timed steps) at CHATS7.
#
# Usage:
#   cd clm-ml-fortran/benchmark
#   bash run_ensemble_benchmark.sh
#
# Output:
#   benchmark/results/ensemble_benchmark_fortran.csv
#   benchmark/results/ensemble_timing_raw.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXE_DIR="${REPO_ROOT}/offline_executable"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

RAW_LOG="${RESULTS_DIR}/ensemble_timing_raw.txt"
CSV_OUT="${RESULTS_DIR}/ensemble_benchmark_fortran.csv"
> "${RAW_LOG}"

# ── Maximum simultaneous processes (set to number of cores, or a safe limit) ──
# Adjust this to match the number of physical CPU cores on the benchmark node.
# Exceeding core count causes slowdowns from context-switching.
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-$(nproc 2>/dev/null || echo 8)}"
echo "Max parallel jobs: ${MAX_PARALLEL_JOBS}" | tee -a "${RAW_LOG}"

# ── Locate executable ─────────────────────────────────────────────────────────
if [ -f "${EXE_DIR}/prgm_opt.exe" ]; then
    EXE="${EXE_DIR}/prgm_opt.exe"
    echo "Using optimized build: prgm_opt.exe"
elif [ -f "${EXE_DIR}/prgm.exe" ]; then
    EXE="${EXE_DIR}/prgm.exe"
    echo "WARNING: using debug build prgm.exe (slower). Run build_optimized.sh first." >&2
else
    echo "ERROR: no executable found in ${EXE_DIR}." >&2
    exit 1
fi

# ── Locate input files ────────────────────────────────────────────────────────
CLM_FILE=$(ls "${REPO_ROOT}/input_files/clm5_0/CHATS7/CHATS_A15.clm2.h1."*.nc 2>/dev/null | head -1)
SOIL_FILE=$(ls "${REPO_ROOT}/input_files/clm5_0/CHATS7/CHATS_soil_moisture_correction_"*.nc 2>/dev/null | head -1)
if [ -z "${CLM_FILE}" ] || [ -z "${SOIL_FILE}" ]; then
    echo "ERROR: input files not found under ${REPO_ROOT}/input_files/clm5_0/CHATS7/" >&2
    exit 1
fi
REL_CLM=$(python3 -c "import os; print(os.path.relpath('${CLM_FILE}', '${EXE_DIR}'))")
REL_SOIL=$(python3 -c "import os; print(os.path.relpath('${SOIL_FILE}', '${EXE_DIR}'))")

# ── Config ────────────────────────────────────────────────────────────────────
# N values to benchmark.  Large N sequential runs are extrapolated to save time.
N_SIZES=(1 8 32 128 512 1024 2048)
# N_SIZES above this threshold use extrapolation for sequential (too slow to run).
SEQ_MAX_N=32
N_STEPS=6       # total steps per run
N_TIMED=5       # timed steps (exclude warmup step)

echo "Benchmark config: N_STEPS=${N_STEPS}, N_TIMED=${N_TIMED}" | tee -a "${RAW_LOG}"
echo "SEQ_MAX_N=${SEQ_MAX_N} (N>${SEQ_MAX_N} sequential is extrapolated)" | tee -a "${RAW_LOG}"
echo "" | tee -a "${RAW_LOG}"

# ── Helper: write namelist ────────────────────────────────────────────────────
write_nl() {
    local outdir="$1"
    local stop_n="$2"
    cat <<EOF
&clmML_inparm
tower_name       = 'CHATS7'
start_ymd        = 20070501
start_tod        = 0
stop_option      = 'nsteps'
stop_n           = ${stop_n}
clm_start_ymd    = 20070401
clm_start_tod    = 0
fin_tower        = '../input_files/tower-forcing/CHATS7/2007-05.nc'
fin_clm          = '${REL_CLM}'
clm_phys         = 'CLM5_0'
fin_soil_adjust  = '${REL_SOIL}'
nlev_soil_adjust = 3
dirout           = '${outdir}'
met_type         = 3
dpai_min         = 0.01D0
pftcon_val       = 1
/
EOF
}

# ── Step 1: measure single-run baseline (used for extrapolation) ──────────────
echo "=== Step 1: single-run baseline (3 repetitions for averaging) ===" | tee -a "${RAW_LOG}"
baseline_times=()
for rep in 1 2 3; do
    outdir="../output_files/bench_ens_baseline_r${rep}/"
    mkdir -p "${EXE_DIR}/../output_files/bench_ens_baseline_r${rep}"
    nl=$(write_nl "${outdir}" "${N_STEPS}")
    t0=$(date +%s%N)
    echo "${nl}" | "${EXE}" > /dev/null 2>&1
    t1=$(date +%s%N)
    elapsed_s=$(awk "BEGIN { printf \"%.6f\", $(( t1 - t0 ))/1e9 }")
    baseline_times+=("${elapsed_s}")
    echo "  rep${rep}: ${elapsed_s}s" | tee -a "${RAW_LOG}"
done

# Use the MAX of the 3 reps as baseline (not mean) to avoid warm-cache bias.
# Subsequent repetitions read from filesystem cache and run faster than a cold
# first call; this biases the mean downward.  The max is a conservative estimate
# closer to what a cold N×sequential run will actually cost.
baseline_mean_s=$(python3 -c "
vals = [${baseline_times[0]}, ${baseline_times[1]}, ${baseline_times[2]}]
print(f'{max(vals):.6f}')
")
# ms per timed step (single run, cold-start estimate)
baseline_ms_per_step=$(awk "BEGIN { printf \"%.4f\", ${baseline_mean_s}*1000/${N_TIMED} }")
echo "  Baseline (max of 3 reps, cold-start): ${baseline_mean_s}s  ->  ${baseline_ms_per_step} ms/step" | tee -a "${RAW_LOG}"
echo "" | tee -a "${RAW_LOG}"

# ── CSV header ────────────────────────────────────────────────────────────────
# backend: fortran_seq (sequential), fortran_par (parallel)
echo "backend,N,run_wall_s,ms_per_sample,notes" > "${CSV_OUT}"

# ── Step 2: sequential and parallel benchmarks for each N ────────────────────
for N in "${N_SIZES[@]}"; do
    echo "====== N = ${N} ======" | tee -a "${RAW_LOG}"

    # ── Sequential ────────────────────────────────────────────────────────────
    if [ "${N}" -le "${SEQ_MAX_N}" ]; then
        # Actually run N instances sequentially
        echo "  Sequential: running ${N} instances ..." | tee -a "${RAW_LOG}"
        t0_seq=$(date +%s%N)
        for idx in $(seq 1 "${N}"); do
            outdir="../output_files/bench_ens_seq_N${N}_i${idx}/"
            mkdir -p "${EXE_DIR}/../output_files/bench_ens_seq_N${N}_i${idx}"
            nl=$(write_nl "${outdir}" "${N_STEPS}")
            echo "${nl}" | "${EXE}" > /dev/null 2>&1
        done
        t1_seq=$(date +%s%N)
        seq_wall_s=$(awk "BEGIN { printf \"%.6f\", $(( t1_seq - t0_seq ))/1e9 }")
        seq_ms_per_sample=$(awk "BEGIN { printf \"%.4f\", $(( t1_seq - t0_seq ))/1e6/${N}/${N_TIMED} }")
        seq_notes="measured"
    else
        # Extrapolate: single_run_time × N
        seq_wall_s=$(awk "BEGIN { printf \"%.6f\", ${baseline_mean_s}*${N} }")
        seq_ms_per_sample="${baseline_ms_per_step}"
        seq_notes="extrapolated"
        echo "  Sequential: extrapolated (${N} × ${baseline_mean_s}s = ${seq_wall_s}s)" | tee -a "${RAW_LOG}"
    fi
    echo "    seq wall: ${seq_wall_s}s  |  ms/sample: ${seq_ms_per_sample}  [${seq_notes}]" | tee -a "${RAW_LOG}"
    echo "fortran_seq,${N},${seq_wall_s},${seq_ms_per_sample},${seq_notes}" >> "${CSV_OUT}"

    # ── Parallel ──────────────────────────────────────────────────────────────
    # Cap simultaneous jobs to MAX_PARALLEL_JOBS
    actual_par=$(( N < MAX_PARALLEL_JOBS ? N : MAX_PARALLEL_JOBS ))

    if [ "${actual_par}" -eq "${N}" ]; then
        par_notes="measured_full_parallel"
        echo "  Parallel: launching ${N} simultaneous instances ..." | tee -a "${RAW_LOG}"
    else
        par_notes="measured_capped_at_${actual_par}"
        echo "  Parallel: capped at ${actual_par}/${N} simultaneous (MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}) ..." | tee -a "${RAW_LOG}"
    fi

    t0_par=$(date +%s%N)

    # Launch in batches of MAX_PARALLEL_JOBS
    launched=0
    active_pids=()
    for idx in $(seq 1 "${N}"); do
        outdir="../output_files/bench_ens_par_N${N}_i${idx}/"
        mkdir -p "${EXE_DIR}/../output_files/bench_ens_par_N${N}_i${idx}"
        nl=$(write_nl "${outdir}" "${N_STEPS}")
        echo "${nl}" | "${EXE}" > /dev/null 2>&1 &
        active_pids+=($!)
        (( launched++ )) || true

        # If we've hit MAX_PARALLEL_JOBS, wait for all current batch to finish
        # before launching more (wave-based batching)
        if [ "${#active_pids[@]}" -ge "${MAX_PARALLEL_JOBS}" ]; then
            for pid in "${active_pids[@]}"; do
                wait "${pid}"
            done
            active_pids=()
        fi
    done

    # Wait for any remaining
    for pid in "${active_pids[@]}"; do
        wait "${pid}"
    done

    t1_par=$(date +%s%N)
    par_wall_s=$(awk "BEGIN { printf \"%.6f\", $(( t1_par - t0_par ))/1e9 }")
    # parallel ms/sample = wall_clock / n_timed_steps (all N samples in that time)
    # ... but if capped, adjust for the number of waves
    if [ "${actual_par}" -eq "${N}" ]; then
        par_ms_per_sample=$(awk "BEGIN { printf \"%.4f\", $(( t1_par - t0_par ))/1e6/${N_TIMED} }")
    else
        # with waves, ms/sample = wall/(n_waves × n_timed) with n_waves = ceil(N/batch)
        par_ms_per_sample=$(python3 -c "
import math
n_waves = math.ceil(${N} / ${actual_par})
ms = $(( t1_par - t0_par )) / 1e6 / (n_waves * ${N_TIMED})
print(f'{ms:.4f}')
")
    fi

    echo "    par wall: ${par_wall_s}s  |  ms/sample: ${par_ms_per_sample}  [${par_notes}]" | tee -a "${RAW_LOG}"
    echo "fortran_par,${N},${par_wall_s},${par_ms_per_sample},${par_notes}" >> "${CSV_OUT}"
    echo "" | tee -a "${RAW_LOG}"
done

echo ""
echo "=== DONE ===" | tee -a "${RAW_LOG}"
echo "Raw log:  ${RAW_LOG}"
echo "CSV:      ${CSV_OUT}"
echo ""
echo "Contents of ${CSV_OUT}:"
cat "${CSV_OUT}"
