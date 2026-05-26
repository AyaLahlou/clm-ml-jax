# CLM-JAX Diagnostics

Scripts for gradient validation, parameter calibration, performance benchmarking,
sensitivity analysis, and visualization. These are standalone Python scripts — not
pytest tests — designed to be run directly from the project root.

All scripts require the full model to be installed (`pip install -e .` from project root)
and input data to be present (see `input_files/`).

## Setup

```bash
# Activate environment
conda activate clm-ml-jax

# Load CUDA (on HPC nodes)
module load cuda12.8/toolkit/12.8.61

# Expose JAX's bundled CUDA libraries
SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$(find $SITE_PKGS/nvidia -name "*.so*" \
  -exec dirname {} \; | sort -u | tr '\n' ':')$LD_LIBRARY_PATH

# Verify GPU
python -c "import jax; print(jax.devices())"
```

Most scripts are invoked from the **project root**:

```bash
python diags/<script>.py
```

Some scripts set `cd src &&` in their docstring — follow the usage note in each file's
module docstring.

---

## Shared Infrastructure

These two modules are imported by most other scripts in this directory. They handle
model initialization, warmup, and observational data loading so individual scripts
can focus on the specific computation they are testing.

### `expt_init.py`

Initializes all CLM module-level singletons, runs one warmup timestep, and exports:

| Symbol | Type | Description |
|--------|------|-------------|
| `forward_fn` | callable | Scalar forward pass: `mlcanopy_inst → loss (H + LE sum)` |
| `mlcanopy_inst` | NamedTuple | Initialized canopy state after warmup |
| `grid` | `GridInfo` | Single-patch grid dimensions |

**Import pattern used by other scripts:**

```python
from diags.expt_init import forward_fn, mlcanopy_inst, grid
```

> **Note:** importing `expt_init` has side effects — it initializes all CLM
> singletons and runs one warmup timestep. Import it once at the top of your script.

### `expt_load_obs.py`

Loads AmeriFlux observational data for CHATS7 (May 2007) and exports:

| Symbol | Description |
|--------|-------------|
| `obs_gpp` | Observed GPP time series (µmol CO₂ m⁻² s⁻¹) |
| `obs_h` | Observed sensible heat flux (W m⁻²) |
| `obs_le` | Observed latent heat flux (W m⁻²) |
| `obs_times` | Fractional day-of-year timestamps |

### `oracle_table.py`

Generates reference parameter sets ("oracle" values) for calibration benchmarking.
Outputs a table of known-good parameter combinations with corresponding model outputs.

---

## Gradient Validation & Debugging

Scripts for verifying that `jax.grad` produces correct and finite gradients through
the model. Use these when you modify physics or add new parameters and need to confirm
differentiability.

### Quick checks (start here)

| Script | What it does | Runtime |
|--------|-------------|---------|
| `quick_grad_check.py` | JIT-compiled `jax.grad` smoke test; prints gradient norms | ~30 s |
| `full_grad_check.py` | Compares `jax.grad` vs finite differences for a set of parameters | ~5 min |
| `fd_grad_check.py` | Pure finite-difference gradient validation (no JAX AD) | ~10 min |
| `le_h_grad_check.py` | Gradient check specifically for latent/sensible heat fluxes | ~5 min |
| `check_10param_grads.py` | 10-parameter gradient check with heatmap output | ~20 min |

**Usage:**

```bash
python diags/quick_grad_check.py          # fastest sanity check
python diags/full_grad_check.py           # full AD vs FD comparison
python diags/le_h_grad_check.py           # H and LE gradient verification
python diags/check_10param_grads.py       # multi-parameter gradient table
```

### Stomatal conductance checks

| Script | What it does |
|--------|-------------|
| `check_g1_medlyn.py` | Verifies gradient through the Medlyn G1 parameter |
| `check_g1_medlyn_fixed.py` | Same with numerical fix applied (reference for debugging) |

### Targeted debugging (for contributors investigating NaN gradients)

These scripts were written during development to isolate specific gradient failures.
They are useful as templates when adding new differentiable computations.

| Script | Target |
|--------|--------|
| `debug_nan_grads.py` | Locates where NaN first appears in the backward pass |
| `debug_grad_isolation.py` | Isolates which submodule produces NaN |
| `debug_g1_grad_isolation.py` | Targets G1 stomatal parameter gradient path |
| `debug_g1_fast.py` | Streamlined G1 gradient test (faster iteration) |
| `bisect_nan_grads.py` | Binary search to find the timestep that introduces NaN |
| `debug_bisect_grad.py` | Bisection-based gradient isolation |
| `debug_tridiag.py` | Tests gradient through the tridiagonal solver |
| `debug_tridiag_vjp.py` | Custom VJP experiments for the tridiagonal solver |
| `debug_custom_vjp.py` | Template for writing custom `jax.custom_vjp` rules |
| `debug_second_loop.py` | Targets the second Runge-Kutta integration loop |
| `debug_grad_no_fps.py` | Gradient check with checkpoint (`remat`) disabled |
| `debug_recompile.py` | Tracks JAX recompilation events during iteration |
| `isolate_grad_path.py` | Minimal reproducer for a specific gradient path |
| `grad_mode_comparison.py` | Compares forward-mode and reverse-mode AD results |

### Kernel-level gradient tests

These test differentiability of individual physics kernels in isolation, independent
of the full model stack. Useful for verifying a new kernel before integrating it.

| Script | Kernel tested |
|--------|-------------|
| `test_ci_solver_grad.py` | `_ci_solver_scan`: CO₂ intercellular concentration solver |
| `test_photo_kernel_grad.py` | Full leaf photosynthesis kernel |
| `test_bisect_ift.py` | Bisection solver for IFT (optimal stomatal conductance) |

**Usage:**

```bash
python diags/test_ci_solver_grad.py       # tests d(ci)/d(apar) vs FD
python diags/test_photo_kernel_grad.py    # tests photosynthesis kernel grads
python diags/test_bisect_ift.py          # tests IFT bisection differentiability
```

> These files use a `test_` prefix for historical reasons — they are **not** pytest
> tests and will not be collected by the test suite (pytest's `testpaths = tests`).

---

## Parameter Calibration & Optimization

Scripts for recovering physiological parameters from flux tower observations using
gradient-based and gradient-free optimization. These demonstrate the scientific value
of differentiable programming in land surface modeling.

| Script | Algorithm | Parameters | Notes |
|--------|-----------|-----------|-------|
| `calibration_demo.py` | Adam + Nelder-Mead | `alpha_sw` (radiation scale) | Best starting point; includes comparison plot |
| `optimize_params.py` | Adam | `vcmaxpft`, `iota_SPA` | Phase 1: synthetic observations |
| `calibration_vcmax_iota.py` | Adam | Vcmax₂₅, iota (water-use efficiency) | Targeted two-parameter calibration |
| `multipar_calibration.py` | Adam | 5 physiological + forcing scales | Full multi-parameter run |
| `multipar_calibration_laxscan.py` | Adam + `lax.scan` | Same as above | Faster via JAX scan; use for long runs |
| `multipar_calibration_singlestep.py` | Adam | 5 parameters | Single-timestep loss; fast iteration |
| `minimal_calibration.py` | Adam | Single parameter | Minimal example for understanding the loop |
| `calibration_nm_only.py` | Nelder-Mead | Arbitrary | Gradient-free baseline; no JAX AD required |

**Recommended sequence for new users:**

```bash
# 1. Understand the setup
python diags/calibration_demo.py          # ~5 min; produces figures/calibration_convergence.png

# 2. Single-parameter optimization
python diags/minimal_calibration.py       # ~2 min

# 3. Full multi-parameter run
python diags/multipar_calibration_singlestep.py   # ~15 min
```

**Outputs:** Results are written to `diags/output/` as JSON files and to
`diags/figures/` as PNG plots.

---

## Performance Benchmarking

Scripts for measuring wall-clock time, JIT compile time, and GPU vs CPU speedup.
Run these before and after significant code changes to track performance regressions.

### Core benchmarks

| Script | Measures | Notes |
|--------|---------|-------|
| `benchmark_gpu.py` | JIT compile time + steady-state throughput on GPU | Start here |
| `benchmark_multisite.py` | Single-site batching performance (vmap over N columns) | |
| `benchmark_ensemble.py` | Ensemble (vmap over N parameter sets) GPU vs CPU | |
| `benchmark_backward_pass.py` | Cost of `jax.grad` vs forward pass | |
| `benchmark_laxscan.py` | `lax.scan` vs Python loop overhead | |
| `benchmark_cpu_compile.py` | JAX XLA compilation time (CPU only) | |

### CPU-scale sweeps (for batch allocation planning)

| Script | Ensemble size |
|--------|-------------|
| `benchmark_ensemble_cpu_512.py` | N=512 |
| `benchmark_ensemble_cpu_1024.py` | N=1024 |
| `benchmark_ensemble_cpu_2048.py` | N=2048 |

### Precision and compilation

| Script | What it measures |
|--------|----------------|
| `precision_roofline.py` | Float32 vs Float64 performance roofline |
| `benchmark_precision.py` | Numerical error vs speed tradeoff |
| `aot_compile.py` | Ahead-of-time (AOT) JAX compilation time |
| `time_jax_run.py` | Simple wall-clock timing harness |
| `ad_vs_fd_scaling.py` | Scaling of AD vs finite-difference cost with parameter count |

**Usage:**

```bash
python diags/benchmark_gpu.py             # GPU timing; run on compute node
python diags/benchmark_ensemble.py        # vmap ensemble scaling
python diags/ad_vs_fd_scaling.py          # AD speedup vs FD for N=1..100 params
```

### Fortran reference benchmarks (`benchmark_fortran/`)

Shell scripts for timing the original Fortran model on HPC nodes:

| Script | Purpose |
|--------|---------|
| `run_single_timing.sh` | Single-point Fortran run with `time` |
| `run_ensemble_benchmark.sh` | SLURM batch job for N-site ensemble |
| `run_multisite_benchmark.sh` | Multi-site sweep |
| `build_optimized.sh` | Compile Fortran with `-O3 -march=native` |
| `parse_results.py` | Aggregate timing logs into CSV |

---

## Sensitivity Analysis

Scripts for computing how model outputs (GPP, H, LE) respond to perturbations in
physiological parameters and environmental forcing.

| Script | Method | Parameters |
|--------|--------|-----------|
| `sensitivity_analysis.py` | Jacobian via `jax.jacrev` | 5 scale factors (Vcmax₂₅, T_air, SW, q, LAI) |
| `sensitivity_analysis_v2.py` | Same + improved plotting | Same |
| `param_sensitivity.py` | First-order sensitivity indices | PFT parameters |
| `temporal_jacobian.py` | Time-varying Jacobian (48 half-hourly steps) | Vcmax₂₅, G1 |

**Usage:**

```bash
python diags/sensitivity_analysis.py      # full Jacobian; ~10 min
python diags/sensitivity_analysis_v2.py   # same + scatter plots
python diags/temporal_jacobian.py         # time-varying sensitivity; ~30 min
```

**Outputs:** CSV files and PNG figures in `diags/figures/`.

---

## Visualization

Standalone plotting scripts. Most read CSV or JSON output from the calibration,
benchmarking, or sensitivity scripts above. Pass `--help` for options.

| Script | Input | Output |
|--------|-------|--------|
| `plot_validation.py` | `output_files/` vs `output_files/validation_files/` | JAX vs Fortran comparison panels |
| `plot_benchmarks.py` | Benchmark timing CSV | GPU/CPU timing bar charts |
| `plot_ensemble_benchmark.py` | `figures/ensemble_benchmark.csv` | Scaling plots |
| `plot_grad_check.py` | Gradient check results | AD vs FD scatter |
| `plot_multipar_calibration.py` | `output/multipar_calibration_results.json` | Loss curve + parameter trajectory |
| `plot_10param_grads.py` | `check_10param_grads.py` output | Gradient magnitude heatmap |
| `plot_paper_figure.py` | Pre-computed CSVs | Publication-quality figures |
| `plot_precision_benchmark.py` | `benchmark_precision.py` output | Float32 vs Float64 trade-off |
| `plot_bug_module_heatmap.py` | `output/bug_taxonomy_reference.md` | Bug distribution by module |
| `plot_bug_taxonomy.py` | Same | Bug category bar chart |
| `plot_repair_timeline.py` | CHANGELOG.md (parsed) | Session-by-session repair timeline |

**Usage:**

```bash
# After running benchmark_ensemble.py:
python diags/plot_ensemble_benchmark.py

# Compare JAX vs Fortran output (after a full run):
python diags/plot_validation.py

# Replot from existing CSV without re-running:
python diags/sensitivity_analysis.py --plot-only
```

---

## Generated Output

| Path | Contents |
|------|---------|
| `diags/figures/` | PNG/PDF plots from all scripts above |
| `diags/output/multipar_calibration_results.json` | Full calibration history (all steps) |
| `diags/output/multipar_calibration_singlestep_results.json` | Single-step calibration results |
| `diags/output/agentic_pipeline_analysis.md` | Analysis of differentiability work across 46 development sessions |
| `diags/output/bug_taxonomy_reference.md` | Reference card categorizing gradient bugs by type |
| `diags/hle_discrepancy_report.md` | Detailed analysis of H/LE flux discrepancies vs Fortran reference |

---

## Additional Debug Scripts in `src/offline_executable/`

These scripts live alongside the model entry point and operate on the full initialization
flow (namelist → InitializeRealize → ModelAdvance):

| Script | Purpose | Usage |
|--------|---------|-------|
| `debug_physics.py` | Traces physics values (LAI, fluxes, temperature) through one timestep | `python src/offline_executable/debug_physics.py` |
| `test_grad.py` | Smoke test for `jax.grad` through the full 1-timestep forward pass | `python src/offline_executable/test_grad.py` |
| `compare_outputs.py` | Systematic column-by-column comparison of JAX vs Fortran output files | `python src/offline_executable/compare_outputs.py` |
| `profile_baseline.py` | Wall-clock profiler for `MLCanopyFluxes` sub-functions | `python src/offline_executable/profile_baseline.py` |

---

## Notes for Contributors

- **Before adding a new differentiable parameter**, run `quick_grad_check.py` before and
  after your change to confirm you haven't introduced NaN gradients.
- **Before committing physics changes**, run `compare_outputs.py` to verify parity with
  the Fortran reference within the expected tolerance.
- **For new calibration targets**, copy `minimal_calibration.py` as a starting template —
  it is the simplest end-to-end example of the optimization loop.
- **For new sensitivity analyses**, copy `sensitivity_analysis_v2.py` — it has the cleanest
  Jacobian computation pattern with proper output labeling.
