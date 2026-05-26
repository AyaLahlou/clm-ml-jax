# Fortran CLM-ML Benchmark Suite

Timing scripts for the Fortran CLM-ML model, designed to produce CSV outputs
that can be merged into the JAX benchmark plots in `diags/plot_benchmarks.py`
and `diags/plot_ensemble_benchmark.py`.

## Prerequisites

1. A working build of `prgm.exe` (see Build section below).
2. NetCDF libraries available (`NETCDF_PATH` environment variable set).
3. Input data files in the expected locations relative to `offline_executable/`:
   - `../input_files/tower-forcing/CHATS7/2007-05.nc`
   - `../input_files/clm5_0/CHATS7/CHATS_A15.clm2.h1.*.nc`
   - `../input_files/clm5_0/CHATS7/CHATS_soil_moisture_correction_*.nc`
4. `bash`, `python3` (with `numpy`), `awk`, `date` — standard on most Linux HPC nodes.

## Build

```bash
cd clm-ml-fortran/benchmark
bash build_optimized.sh
```

This builds `../offline_executable/prgm_opt.exe` with `-O2` (no bounds checking).
The default Makefile builds with `-C -Ktrap=fp` (debug flags) — the optimized
build is 2–5× faster for benchmarking.

If your system uses `gfortran` instead of `nvfortran`, edit `build_optimized.sh`
and set `COMPILER=gfortran`.

## Workflow

Run the scripts in order:

```bash
# Step 1: baseline single-site timing (sanity check + warmup measurement)
bash run_single_timing.sh

# Step 2: multisite benchmark (N=1..32, sequential then parallel)
bash run_multisite_benchmark.sh

# Step 3: ensemble (parameter sample) benchmark (N=1..2048)
bash run_ensemble_benchmark.sh

# Step 4: parse all timing logs → CSV files
python3 parse_results.py
```

Results CSVs are written to `benchmark/results/`:
- `multisite_benchmark_fortran.csv`
- `ensemble_benchmark_fortran.csv`

## Merging with JAX plots

Copy the CSVs back to the JAX repo and re-run the plot scripts with `--fortran`:

```bash
# On the JAX machine, from the project root:
python diags/plot_benchmarks.py \
    --fortran path/to/multisite_benchmark_fortran.csv

python diags/plot_ensemble_benchmark.py \
    --fortran path/to/ensemble_benchmark_fortran.csv
```

## Expected timings (reference)

| Configuration       | Approx. wall time              |
|---------------------|-------------------------------|
| Single-site, 1 day  | ~3.6 s (48 timesteps, -O2)     |
| Single-site, 5 steps| ~0.37 s (after 1-step warmup)  |
| run_single_timing   | ~5-10 min (3 warm runs × 1day) |
| run_multisite       | ~10-20 min                     |
| run_ensemble        | ~15-30 min                     |

Actual timings vary by CPU. The scripts print progress to stdout.

## Notes on methodology

- **Warmup**: The first timestep always includes JIT-like initialization (file I/O,
  array allocation). All timing scripts use `stop_option='nsteps'` with 
  `stop_n=6` (1 warmup + 5 timed) so per-step times exclude initialization.
- **Parallel runs**: "Parallel Fortran" means N independent `prgm.exe` processes
  running simultaneously on the same node (bash `&` background jobs). This is the
  Fortran equivalent of JAX vmap — it exploits multi-core but not SIMD/GPU batching.
- **Sequential Fortran**: Running N instances one after another. The per-sample
  cost is constant (no batching benefit), so this is the correct baseline for
  comparing against JAX sequential execution.
- **Output directory isolation**: Each parallel run writes to its own subdirectory
  under `output_files/` to avoid file conflicts.
