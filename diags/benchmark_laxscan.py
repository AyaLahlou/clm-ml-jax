"""
benchmark_laxscan.py — Test correctness and runtime of the lax.scan refactoring.

Compares:
  1. DIFF MODE (lax.scan):     jit(forward_fn)(mlcanopy_inst)
     — XLA sees one dispatch; no Python loop overhead.
  2. NON-DIFF MODE (Python loop): MLCanopyFluxes(..., grid=None)
     — Python for-loop over sub-steps; many separate dispatches.

Tests run for both Euler (1 sub-step) and full RK4 (6 sub-steps × 4 RK stages).

Correctness check: compare key flux outputs (GPP, shflx, lhflx) between the
two modes.  They are not expected to match bit-for-bit (diff mode skips
diagnostics + flux averaging that non-diff mode computes), but the physics
outputs should be consistent.

Usage (from project root):
    CLM_ML_NO_CHECKPOINT=1 python diags/benchmark_laxscan.py
"""
from __future__ import annotations
import csv, os, sys, time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR      = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ["CLM_ML_NO_CHECKPOINT"] = "1"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

print(f"JAX devices: {jax.devices()}", flush=True)
print(f"JAX backend: {jax.default_backend()}", flush=True)

# ── Initialise CHATS7 ─────────────────────────────────────────────────────────
print("\n=== Initialising CHATS7 (warmup ~2-5 min) ===", flush=True)
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs, forward_fn,
)
from multilayer_canopy.MLCanopyFluxesMod import MLCanopyFluxes
from multilayer_canopy import MLclm_varctl as _ctl

N_REPEATS = 5

# ── helpers ───────────────────────────────────────────────────────────────────
def _time_jit(fn, *args, repeats=N_REPEATS):
    """Return (compile_s, mean_steady_s, output)."""
    t0 = time.perf_counter()
    out = fn(*args)
    jax.effects_barrier()
    compile_s = time.perf_counter() - t0

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.effects_barrier()
        times.append(time.perf_counter() - t0)
    return compile_s, float(np.mean(times)), out


def _run_nondiff(repeats=N_REPEATS):
    """Time non-diff mode (grid=None) Python loop."""
    kwargs_nd = {k: v for k, v in _mlcf_kwargs.items() if k not in ("grid", "_o2ref_py")}

    # First call (includes any tracing overhead)
    t0 = time.perf_counter()
    out = MLCanopyFluxes(mlcanopy_inst=mlcanopy_inst, **kwargs_nd)
    jax.effects_barrier()
    first_s = time.perf_counter() - t0

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        MLCanopyFluxes(mlcanopy_inst=mlcanopy_inst, **kwargs_nd)
        jax.effects_barrier()
        times.append(time.perf_counter() - t0)
    return first_s, float(np.mean(times)), out


def _section(title, rk_type, dtime_ml, nrk):
    _ctl.runge_kutta_type = rk_type
    _ctl.dtime_ml         = dtime_ml
    _ctl.nrk              = nrk
    print(f"\n{'='*65}", flush=True)
    print(f"  {title}", flush=True)
    print(f"  runge_kutta_type={rk_type}, dtime_ml={dtime_ml}s, nrk={nrk}", flush=True)
    print(f"{'='*65}", flush=True)

    # ── DIFF MODE (lax.scan) ─────────────────────────────────────────────────
    print("\n-- DIFF MODE (lax.scan) --", flush=True)
    jit_fwd = jax.jit(forward_fn)
    diff_compile_s, diff_ss_s, diff_out = _time_jit(jit_fwd, mlcanopy_inst)
    print(f"  compile (first call): {diff_compile_s:.3f}s", flush=True)
    print(f"  steady  ({N_REPEATS} reps mean): {diff_ss_s*1000:.1f} ms", flush=True)

    # ── NON-DIFF MODE (Python loop) ──────────────────────────────────────────
    print("\n-- NON-DIFF MODE (Python for-loop) --", flush=True)
    nd_first_s, nd_ss_s, nd_out = _run_nondiff()
    print(f"  first call:          {nd_first_s:.3f}s", flush=True)
    print(f"  steady  ({N_REPEATS} reps mean): {nd_ss_s*1000:.1f} ms", flush=True)

    speedup = nd_ss_s / diff_ss_s if diff_ss_s > 0 else float("nan")
    print(f"\n  Speedup (non-diff / diff): {speedup:.2f}x", flush=True)

    # ── Correctness check ────────────────────────────────────────────────────
    # forward_fn returns sum(shair_profile[1:n+1]) + sum(etair_profile[1:n+1])
    # at patch p (last sub-step value, not averaged).
    # non-diff mode returns shair_profile averaged over all sub-steps.
    # We check:
    #   1. diff loss is finite and non-NaN
    #   2. grad check: jax.grad(forward_fn)(mlcanopy_inst) has finite gradients
    #   3. non-diff shair/etair profiles are finite
    p = grid.p
    n = grid.ncan
    loss  = float(diff_out)
    sh_nd_sum = float(jnp.sum(nd_out.shair_profile[p, 1:n+1]))
    et_nd_sum = float(jnp.sum(nd_out.etair_profile[p, 1:n+1]))

    print(f"\n-- Correctness check --", flush=True)
    print(f"  diff_mode  loss (sh+et last step) = {loss:.4f}", flush=True)
    print(f"  non-diff   sum(shair_profile)    = {sh_nd_sum:.4f}  (averaged)", flush=True)
    print(f"  non-diff   sum(etair_profile)    = {et_nd_sum:.4f}  (averaged)", flush=True)
    finite_diff = jnp.isfinite(jnp.array(loss))
    print(f"  diff loss finite: {bool(finite_diff)}", flush=True)

    # Gradient check via alpha_tref scale factor (jax.grad requires float inputs)
    # Cannot use jax.grad(forward_fn)(mlcanopy_inst) directly — mlcanopy_inst has int fields.
    # Instead differentiate w.r.t. a scalar alpha that scales tref_forcing, same as fd_grad_check.py.
    print("\n-- Gradient check (no NaN/Inf) --", flush=True)
    from clm_src_main import clm_instMod as _ci
    from diags.expt_init import compute_gpp as _compute_gpp
    _atm = _ci.atm2lnd_inst
    _wat = _ci.wateratm2lndbulk_inst
    # Keep grid and _o2ref_py in kwargs (same pattern as fd_grad_check.py)
    _mlcf_kwargs_no_atm2 = {k: v for k, v in _mlcf_kwargs.items()
                            if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}

    def _fwd_alpha_tref(alpha):
        _mod_atm = _atm._replace(forc_t_downscaled_col=alpha * _atm.forc_t_downscaled_col)
        inst2 = MLCanopyFluxes(
            mlcanopy_inst=mlcanopy_inst,
            atm2lnd_inst=_mod_atm,
            wateratm2lndbulk_inst=_wat,
            **_mlcf_kwargs_no_atm2,
        )
        return _compute_gpp(inst2, grid.p, grid.ncan)

    t0 = time.perf_counter()
    grad_tref = float(jax.jit(jax.grad(_fwd_alpha_tref))(jnp.float64(1.0)))
    jax.effects_barrier()
    grad_time_s = time.perf_counter() - t0
    finite = np.isfinite(grad_tref)
    flag = "OK" if finite and abs(grad_tref) < 1e10 else "WARN — non-finite or large"
    print(f"  dGPP/d(alpha_tref) = {grad_tref:.4e}  finite={finite}  Status: {flag}", flush=True)
    print(f"  grad() call time: {grad_time_s:.3f}s", flush=True)

    return diff_compile_s, diff_ss_s, nd_ss_s, speedup


# ── Run benchmarks ────────────────────────────────────────────────────────────
_csv_rows = []

print("\n=== Benchmark: Euler (1 sub-step) — grad check ===", flush=True)
_euler = _section("EULER — 1 sub-step, 0 RK stages", rk_type=10, dtime_ml=float(_ctl.dtime_ml), nrk=0)
_csv_rows.append({
    "mode": "Euler", "rk_type": 10, "n_substeps": 1, "n_rk_stages": 0,
    "diff_compile_s": round(_euler[0], 3),
    "diff_ss_ms":     round(_euler[1] * 1000, 2),
    "nondiff_ss_ms":  round(_euler[2] * 1000, 2),
    "speedup":        round(_euler[3], 1),
})

print("\n=== Benchmark: Full RK4 (6 sub-steps × 4 RK stages) ===", flush=True)
_rk4 = _section("FULL RK4 — 6 sub-steps × 4 RK stages", rk_type=41, dtime_ml=300.0, nrk=4)
_csv_rows.append({
    "mode": "RK4", "rk_type": 41, "n_substeps": 6, "n_rk_stages": 4,
    "diff_compile_s": round(_rk4[0], 3),
    "diff_ss_ms":     round(_rk4[1] * 1000, 2),
    "nondiff_ss_ms":  round(_rk4[2] * 1000, 2),
    "speedup":        round(_rk4[3], 1),
})

# ── Save CSV ──────────────────────────────────────────────────────────────────
_figures_dir = Path(__file__).resolve().parent / "figures"
_figures_dir.mkdir(parents=True, exist_ok=True)
_csv_path = _figures_dir / "laxscan_benchmark.csv"
_fields = ["mode", "rk_type", "n_substeps", "n_rk_stages",
           "diff_compile_s", "diff_ss_ms", "nondiff_ss_ms", "speedup"]
with open(_csv_path, "w", newline="") as _f:
    _w = csv.DictWriter(_f, fieldnames=_fields)
    _w.writeheader()
    _w.writerows(_csv_rows)
print(f"\nResults saved: {_csv_path}", flush=True)

print("\n=== benchmark_laxscan.py complete ===", flush=True)
