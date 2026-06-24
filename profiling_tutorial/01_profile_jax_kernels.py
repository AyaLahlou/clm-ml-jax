"""
GPU Profiling Tutorial — JAX + NVTX + Nsight Systems
=====================================================

This script shows how to annotate JAX code with NVTX ranges so that
Nsight Systems (nsys) and Nsight Compute (ncu) can pinpoint:

  1. GPU gaps  — stretches of time where the GPU is completely idle
                 while the CPU is busy (Python overhead, data transfer,
                 JIT compilation, etc.)
  2. Kernel bottlenecks — individual CUDA kernels that consume the
                          most wall time

HOW TO RUN
----------
# Step 1 — profile with Nsight Systems (timeline view)
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=reports/clm_profile \
    python profiling_tutorial/01_profile_jax_kernels.py

# Step 2 — open the report
nsys-ui reports/clm_profile.nsys-rep

# Step 3 — drill into a single kernel with Nsight Compute
ncu --set full \
    --kernel-name-base function \
    --launch-skip 5 --launch-count 1 \
    --output reports/clm_kernel \
    python profiling_tutorial/01_profile_jax_kernels.py

# Step 4 — open kernel report
ncu-ui reports/clm_kernel.ncu-rep

WHAT TO LOOK FOR IN NSIGHT SYSTEMS
-----------------------------------
* White gaps between CUDA rows → GPU stalls (the target to eliminate)
* Long green bars in the NVTX row → which CLM phase owns that gap
* Back-to-back kernels with no gap → healthy compute
* cudaMemcpy / H2D transfers → data not pre-staged on GPU
* "JIT compile" gaps → first-run XLA overhead (use JAX cache to skip)

See the README in this directory for a guided walkthrough.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

# ── Enable 64-bit floats to match CLM-ML production config ──────────────────
jax.config.update("jax_enable_x64", True)

# ── Optional: cupy-based nvtx if nvtx package is unavailable ────────────────
try:
    import nvtx                       # pip install nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False

    class _FakeRange:                 # no-op shim so the script runs without nvtx
        def __enter__(self): return self
        def __exit__(self, *_): pass

    class nvtx:                       # type: ignore[no-redef]
        @staticmethod
        def annotate(msg, color="blue", domain=None):
            return _FakeRange()

        @staticmethod
        def push_range(msg, color="blue", domain=None): pass

        @staticmethod
        def pop_range(): pass


# ─────────────────────────────────────────────────────────────────────────────
# Simulated CLM-ML workload
#   We replicate the compute shapes used in the real canopy model so the
#   profiling numbers are realistic.
# ─────────────────────────────────────────────────────────────────────────────
N_COLS   = 2048   # ensemble / spatial columns (matches a vmap batch)
N_LAYERS = 40     # canopy layers
N_ITER   = 50     # Runge-Kutta / solver iterations

rng = jax.random.PRNGKey(42)

# ── Synthetic inputs (already on GPU after the first JIT) ───────────────────
def _make_inputs():
    """Create representative canopy state arrays."""
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    return {
        "tleaf":   jax.random.normal(k1, (N_COLS, N_LAYERS)) + 300.0,   # K
        "par":     jax.random.uniform(k2, (N_COLS, N_LAYERS)) * 800.0,  # W m-2
        "vpd":     jax.random.uniform(k3, (N_COLS,)) * 3.0,             # kPa
        "wind":    jax.random.uniform(k4, (N_COLS, N_LAYERS)) * 5.0,    # m s-1
    }


# ── Canopy sub-kernels ───────────────────────────────────────────────────────

@jax.jit
def solar_radiation(par: jnp.ndarray) -> jnp.ndarray:
    """Beer-Lambert light extinction through canopy layers."""
    k_ext = 0.5
    lai_per_layer = 0.1
    tau = jnp.exp(-k_ext * lai_per_layer * jnp.arange(N_LAYERS))
    return par * tau[None, :]   # (N_COLS, N_LAYERS)


@jax.jit
def leaf_photosynthesis(tleaf: jnp.ndarray, par: jnp.ndarray, vpd: jnp.ndarray) -> jnp.ndarray:
    """Farquhar–Ball–Berry photosynthesis (simplified)."""
    vcmax25 = 60.0
    q10 = 2.0
    tleaf_c = tleaf - 273.15
    vcmax = vcmax25 * q10 ** ((tleaf_c - 25.0) / 10.0)
    a_gross = vcmax * par / (par + 100.0)
    gs = 0.1 + 8.0 * a_gross * (1.0 / (vpd[:, None] + 0.1))
    return a_gross * gs   # (N_COLS, N_LAYERS)


@jax.jit
def canopy_turbulence(wind: jnp.ndarray) -> jnp.ndarray:
    """Exponential wind profile inside canopy."""
    alpha = 2.5
    wind_top = wind[:, -1:]
    attenuation = jnp.exp(-alpha * (1.0 - jnp.linspace(0, 1, N_LAYERS)[None, :]))
    return wind_top * attenuation   # (N_COLS, N_LAYERS)


@jax.jit
def soil_temperature_solver(tleaf: jnp.ndarray) -> jnp.ndarray:
    """Tridiagonal Thomas algorithm for soil temperature (N_ITER steps)."""
    t = jnp.mean(tleaf, axis=1)   # (N_COLS,)

    def step(t, _):
        return t + 0.01 * (jnp.roll(t, 1) - 2 * t + jnp.roll(t, -1)), None

    t_final, _ = jax.lax.scan(step, t, None, length=N_ITER)
    return t_final   # (N_COLS,)


# ── Gap-inducing patterns (intentional, for teaching) ───────────────────────

def _gap_numpy_roundtrip(arr: jnp.ndarray) -> jnp.ndarray:
    """
    BAD PATTERN — pulls data back to CPU, processes in NumPy, then sends
    it back to GPU.  Creates a visible H2D/D2H gap in Nsight.
    """
    cpu = np.array(arr)                 # D2H copy — GPU stalls here
    cpu = np.clip(cpu, 0, None)         # CPU work
    return jnp.array(cpu)              # H2D copy


def _gap_python_loop(par: jnp.ndarray) -> jnp.ndarray:
    """
    BAD PATTERN — Python loop over columns.  Each iteration dispatches a
    tiny kernel, leaving the GPU starved between dispatches.
    """
    results = []
    for i in range(min(16, N_COLS)):    # keep it short so the demo finishes
        results.append(solar_radiation(par[i:i+1]))
    return jnp.concatenate(results, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Main profiling demonstration
# ─────────────────────────────────────────────────────────────────────────────

def warmup(inputs: dict):
    """
    Run all JIT-compiled functions once so XLA compilation finishes before
    the profiler window opens.  This prevents the huge one-time compile gap
    from polluting timing data.
    """
    with nvtx.annotate("warmup / JIT compile", color="gray"):
        par_lit  = solar_radiation(inputs["par"])
        _        = leaf_photosynthesis(inputs["tleaf"], par_lit, inputs["vpd"])
        _        = canopy_turbulence(inputs["wind"])
        _        = soil_temperature_solver(inputs["tleaf"])
        jax.block_until_ready(par_lit)   # flush async dispatch queue


def run_optimized(inputs: dict):
    """
    Good pattern: all operations stay on-device, no CPU round-trips.
    In Nsight you should see back-to-back CUDA kernels with no gaps.
    """
    with nvtx.annotate("optimized_run", color="green"):

        with nvtx.annotate("solar_radiation", color="yellow"):
            par_lit = solar_radiation(inputs["par"])

        with nvtx.annotate("leaf_photosynthesis", color="lime"):
            flux = leaf_photosynthesis(inputs["tleaf"], par_lit, inputs["vpd"])

        with nvtx.annotate("canopy_turbulence", color="cyan"):
            u = canopy_turbulence(inputs["wind"])

        with nvtx.annotate("soil_temperature_solver", color="orange"):
            t_soil = soil_temperature_solver(inputs["tleaf"])

        jax.block_until_ready((flux, u, t_soil))

    return flux, u, t_soil


def run_with_gaps(inputs: dict):
    """
    Bad pattern: intentional CPU ↔ GPU round-trips and a Python loop.
    In Nsight the CUDA timeline will have visible white gaps.
    """
    with nvtx.annotate("run_with_gaps", color="red"):

        with nvtx.annotate("gap: numpy_roundtrip", color="red"):
            par_lit = solar_radiation(inputs["par"])
            par_lit = _gap_numpy_roundtrip(par_lit)   # intentional gap

        with nvtx.annotate("gap: python_loop_over_cols", color="red"):
            _ = _gap_python_loop(inputs["par"])        # intentional gap

        jax.block_until_ready(par_lit)


def benchmark(label: str, fn, *args, repeats: int = 5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    mean_ms = 1e3 * sum(times) / len(times)
    print(f"  {label:<35s} {mean_ms:7.1f} ms  (avg over {repeats} runs)")


def main():
    print("=" * 60)
    print("CLM-ML JAX — GPU Profiling Tutorial")
    print("=" * 60)

    print("\n[1] Building inputs on GPU …")
    with nvtx.annotate("build_inputs", color="blue"):
        inputs = _make_inputs()

    print("[2] Warming up JIT compilation …")
    warmup(inputs)
    print("    Done. XLA compilation complete.\n")

    print("[3] Benchmarking …")
    benchmark("optimized (no gaps)",    run_optimized, inputs)
    benchmark("with_gaps (roundtrips)", run_with_gaps, inputs)

    print("\n[4] Running final annotated pass for Nsight capture …")
    nvtx.push_range("nsight_capture_window", color="white")

    for i in range(3):
        with nvtx.annotate(f"iteration_{i}", color="blue"):
            run_optimized(inputs)

    with nvtx.annotate("gap_demo", color="red"):
        run_with_gaps(inputs)

    nvtx.pop_range()

    print("\nDone.  Open the .nsys-rep report in Nsight Systems UI.")
    print("See profiling_tutorial/README.md for what to look for.\n")


if __name__ == "__main__":
    main()
