"""
benchmark_cpu_compile.py — CPU XLA compilation time vs ensemble size N.

Measures how long JAX/XLA takes to compile jax.jit(jax.vmap(forward_multi_cpu))
for each N in N_VALUES, using a per-N SIGALRM timeout.  When compilation
exceeds TIMEOUT_S the result is recorded as a timeout and the loop advances.

Purpose: characterise the CPU compilation scaling bottleneck.  XLA on CPU
unrolls vmap into a flat O(N × model_ops) graph, making compile time grow
super-linearly with N.  The GPU backend tiles the batch dimension instead,
keeping compile time roughly O(model_ops) regardless of N.

This is a new benchmark added alongside the existing throughput benchmark
(ensemble_benchmark.csv).  It answers: "at what N does CPU vmap become
impractical due to compile time alone?"

N_VALUES: [1, 8, 32, 128, 512, 1024, 2048]  (same as ensemble_benchmark.py)
TIMEOUT_S: 3600 per N  (1 hour — generous; N>=512 expected to time out)

Output:
  diags/figures/cpu_compile_time.csv  (compile_s, status, run_ms, ms_per_sample)

Usage (from project root):
    CLM_ML_NO_CHECKPOINT=1 python diags/benchmark_cpu_compile.py
"""
from __future__ import annotations

import csv
import os
import signal
import sys
import time
from pathlib import Path

# ── Use a separate cache dir so we measure clean compilation, not cache hits ───
_BENCH_CACHE = "/burg-archive/home/al4385/.cache/jax_compile_cache_cpu_compile_bench"
os.environ["JAX_COMPILATION_CACHE_DIR"] = _BENCH_CACHE
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "99999"  # effectively disabled
os.makedirs(_BENCH_CACHE, exist_ok=True)
print(f"JAX cache dir (compile bench, separate): {_BENCH_CACHE}", flush=True)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import multilayer_canopy.MLpftconMod as _pftcon_mod
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs,
    jax, jnp, MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst, compute_gpp, compute_le, compute_h,
)

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_VALUES  = [128, 512, 1024, 2048]  # N=1,8,32 measured in job 7578601
TIMEOUT_S = 3600    # 1 hour per N
REPEATS   = 3       # post-compile throughput repeats (only when compile succeeds)

# ── CPU device setup ──────────────────────────────────────────────────────────
cpu_dev = jax.devices("cpu")[0]
print(f"CPU device: {cpu_dev}", flush=True)

_orig_pftcon     = _pftcon_mod.MLpftcon
_p               = grid.p
_n               = grid.ncan
_canopystate_inst = _mlcf_kwargs["canopystate_inst"]
_mlcf_kwargs_base = {k: v for k, v in _mlcf_kwargs.items()
                     if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst",
                                  "canopystate_inst")}

# Move all state to CPU
atm2lnd_cpu  = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_dev), atm2lnd_inst)
watm_cpu     = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_dev), wateratm2lndbulk_inst)
mlcanopy_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_dev), mlcanopy_inst)
canopy_cpu   = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_dev), _canopystate_inst)
vcmaxpft_cpu = jax.device_put(_orig_pftcon.vcmaxpft, cpu_dev)

_kwargs_cpu = {}
for k, v in _mlcf_kwargs_base.items():
    try:
        _kwargs_cpu[k] = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, cpu_dev) if hasattr(x, "shape") else x, v)
    except Exception:
        _kwargs_cpu[k] = v


def forward_multi_cpu(scales):
    vc = scales[0] * vcmaxpft_cpu
    mc = canopy_cpu._replace(
        elai_patch=scales[4] * jnp.asarray(canopy_cpu.elai_patch, dtype=jnp.float64),
        esai_patch=scales[4] * jnp.asarray(canopy_cpu.esai_patch, dtype=jnp.float64))
    ma = atm2lnd_cpu._replace(
        forc_t_downscaled_col     = scales[1] * atm2lnd_cpu.forc_t_downscaled_col,
        forc_solad_downscaled_col = scales[2] * atm2lnd_cpu.forc_solad_downscaled_col,
        forc_solai_grc            = scales[2] * atm2lnd_cpu.forc_solai_grc)
    mw = watm_cpu._replace(forc_q_downscaled_col=scales[3] * watm_cpu.forc_q_downscaled_col)
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_cpu, atm2lnd_inst=ma,
        wateratm2lndbulk_inst=mw, canopystate_inst=mc,
        vcmaxpft_jax=vc, **_kwargs_cpu)
    return jnp.array([compute_gpp(inst, _p, _n), compute_h(inst, _p, _n), compute_le(inst, _p, _n)])


# ── Single-sample JIT warmup ──────────────────────────────────────────────────
print("\nCompiling single-sample CPU JIT...", flush=True)
_fwd_jit = jax.jit(forward_multi_cpu)
t0 = time.perf_counter()
_ = _fwd_jit(jax.device_put(jnp.ones(5, dtype=jnp.float64), cpu_dev))
jax.effects_barrier()
single_compile_s = time.perf_counter() - t0
print(f"  single JIT compile: {single_compile_s:.1f}s", flush=True)

t0 = time.perf_counter()
_ = _fwd_jit(jax.device_put(jnp.ones(5, dtype=jnp.float64), cpu_dev))
jax.effects_barrier()
single_run_s = time.perf_counter() - t0
print(f"  single forward (post-JIT): {single_run_s*1000:.1f}ms", flush=True)

# ── Parameter samples ─────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=42)
all_thetas = rng.uniform(0.8, 1.2, size=(max(N_VALUES), 5)).astype(np.float64)

# ── Timeout handler ───────────────────────────────────────────────────────────
class _CompileTimeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise _CompileTimeout()

signal.signal(signal.SIGALRM, _alarm_handler)

# ── Results accumulator ───────────────────────────────────────────────────────
rows = []

# ── Main benchmark loop ───────────────────────────────────────────────────────
for N in N_VALUES:
    print(f"\n{'='*60}", flush=True)
    print(f"N = {N}  (timeout = {TIMEOUT_S}s)", flush=True)

    thetas_N  = jax.device_put(jnp.array(all_thetas[:N]), cpu_dev)
    # Fresh jit+vmap for each N — ensures no cache reuse across N values
    batched_fn = jax.jit(jax.vmap(forward_multi_cpu))

    # ── Compilation timing with timeout ──────────────────────────────────────
    signal.alarm(TIMEOUT_S)
    try:
        t0 = time.perf_counter()
        out = batched_fn(thetas_N)
        jax.effects_barrier()
        compile_s = time.perf_counter() - t0
        signal.alarm(0)   # cancel pending alarm
        status = "ok"
        print(f"  compile+first call: {compile_s:.1f}s  [OK]", flush=True)
    except _CompileTimeout:
        compile_s = float(TIMEOUT_S)
        status    = f"timeout>{TIMEOUT_S}s"
        print(f"  TIMED OUT after {TIMEOUT_S}s", flush=True)

    # ── Post-compile throughput (only if compile succeeded) ───────────────────
    run_ms        = float("nan")
    ms_per_sample = float("nan")
    if status == "ok":
        times = []
        for i in range(REPEATS):
            t0 = time.perf_counter()
            _ = batched_fn(thetas_N)
            jax.effects_barrier()
            times.append(time.perf_counter() - t0)
            print(f"  repeat {i+1}/{REPEATS}: {times[-1]*1000:.1f}ms", flush=True)
        run_ms        = float(np.mean(times)) * 1000.0
        ms_per_sample = run_ms / N
        print(f"  throughput: {run_ms:.1f}ms total  {ms_per_sample:.2f}ms/sample", flush=True)

    rows.append({
        "N":             N,
        "compile_s":     f"{compile_s:.1f}",
        "status":        status,
        "run_ms":        f"{run_ms:.1f}" if not np.isnan(run_ms) else "nan",
        "ms_per_sample": f"{ms_per_sample:.2f}" if not np.isnan(ms_per_sample) else "nan",
    })

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
print("=== CPU vmap compilation time summary ===", flush=True)
print(f"  {'N':>6}  {'compile_s':>12}  {'status':>20}  {'run_ms':>10}  {'ms/sample':>10}", flush=True)
print("  " + "-" * 65, flush=True)
for r in rows:
    print(f"  {r['N']:>6}  {r['compile_s']:>12}  {r['status']:>20}  {r['run_ms']:>10}  {r['ms_per_sample']:>10}", flush=True)

# ── Write CSV (append if file exists, so partial runs accumulate) ─────────────
csv_path = FIGURES_DIR / "cpu_compile_time.csv"
fieldnames = ["N", "compile_s", "status", "run_ms", "ms_per_sample"]
write_header = not csv_path.exists()
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerows(rows)
print(f"\nCSV {'created' if write_header else 'appended'}: {csv_path}", flush=True)

print("\n=== benchmark_cpu_compile.py complete ===", flush=True)
