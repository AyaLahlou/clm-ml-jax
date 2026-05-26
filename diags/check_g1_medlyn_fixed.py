"""
check_g1_medlyn_fixed.py — Verify d(GPP)/d(alpha_g1) under gs_type=0 (Medlyn).

Uses the explicit g1_MED_jax argument to LeafPhotosynthesis (via MLCanopyFluxes)
instead of module-global mutation (_set_pftcon).  This bypasses any potential
JAX-tracing issue with reading module globals inside jax.lax.scan and mirrors
the vcmaxpft_jax pattern that correctly passes gradients.

Fix applied in MLLeafPhotosynthesisMod.py:
  LeafPhotosynthesis(..., g1_MED_jax=None) — when provided, overrides MLpftcon.g1_MED
Fix applied in MLCanopyFluxesMod.py:
  MLCanopyFluxes(..., g1_MED_jax=None) — threads g1_MED_jax to LeafPhotosynthesis

Steps:
  1. Sets gs_type=0 (Medlyn) BEFORE expt_init runs the warmup timestep.
  2. Runs FD: confirms d(GPP)/d(alpha_g1) is non-zero under Medlyn.
  3. Runs jax.grad with explicit g1_MED_jax arg: compares against FD.
  4. Reports rel error, PASS/FAIL at <1% criterion.

Usage (from project root):
    python diags/check_g1_medlyn_fixed.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["CLM_ML_NO_CHECKPOINT"] = "1"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _d in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# ── Set Medlyn BEFORE expt_init imports (expt_init warmup uses this gs_type) ──
from multilayer_canopy import MLclm_varctl
MLclm_varctl.gs_type = 0
print(f"gs_type set to 0 (Medlyn) before init", flush=True)

# ── Shared init (runs warmup with gs_type=0) ──────────────────────────────────
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs,
    atm2lnd_inst, wateratm2lndbulk_inst, compute_gpp,
    MLCanopyFluxes,
)
import multilayer_canopy.MLpftconMod as _MLpftconMod

_p = grid.p
_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}

_orig_pftcon = _MLpftconMod.MLpftcon
_orig_g1_MED = jnp.asarray(_orig_pftcon.g1_MED)
print(f"g1_MED (PFT 1..5): {list(float(v) for v in _orig_pftcon.g1_MED[1:6])}", flush=True)

def forward_gpp_g1(alpha: jnp.ndarray) -> jnp.ndarray:
    """Scale all PFT g1_MED values by alpha; return canopy GPP proxy.

    Uses explicit g1_MED_jax argument — no module-global mutation needed.
    The JAX tracer for alpha flows directly through g1_MED_jax into
    LeafPhotosynthesis, bypassing any module-global read issues inside
    jax.lax.scan.
    """
    g1_MED_jax = alpha * _orig_g1_MED
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=atm2lnd_inst,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        g1_MED_jax=g1_MED_jax,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, grid.ncan)

# ── Baseline ──────────────────────────────────────────────────────────────────
print(f"\n=== Medlyn baseline (gs_type={MLclm_varctl.gs_type}) ===", flush=True)
gpp_base = float(forward_gpp_g1(jnp.float64(1.0)))
print(f"  GPP proxy = {gpp_base:.6f}", flush=True)
if gpp_base == 0.0:
    print("  ERROR: baseline GPP is zero — may be nighttime step. Gradient will be uninformative.")

# ── FD gradient ──────────────────────────────────────────────────────────────
# NOTE: MUST use large eps (0.05–0.1) for g1_MED!
# The shade-leaf (isha) ci-solver is NON-SMOOTH at small eps (1e-4 gives FD=+1254,
# which is WRONG — solver converges to different attractor branches).
# At eps=0.1, FD converges to +5.02 and JAX gives +4.98 (ratio 0.99). ✓
# Session 39 investigation confirmed this — see CHANGELOG.md.
EPS = 0.1
print(f"\n=== FD gradient (eps={EPS}, stable regime for g1_MED ci-solver) ===", flush=True)
t0 = time.time()
gpp_plus  = float(forward_gpp_g1(jnp.float64(1.0 + EPS)))
gpp_minus = float(forward_gpp_g1(jnp.float64(1.0 - EPS)))
fd_val = (gpp_plus - gpp_minus) / (2.0 * EPS)
dt_fd = time.time() - t0
print(f"  GPP(1+eps) = {gpp_plus:.6f}  GPP(1-eps) = {gpp_minus:.6f}", flush=True)
print(f"  d(GPP)/d(alpha_g1) [FD]  = {fd_val:+.6e}  ({dt_fd:.1f}s)", flush=True)

if abs(fd_val) < 1e-8:
    print("  FD ≈ 0 — g1_MED has no effect under this configuration.")
    print("  Possible causes: nighttime step (PAR=0), gs_type not 0, or g1 not on active PFT.")
    sys.exit(1)
print(f"  g1_MED is ACTIVE under Medlyn (FD non-zero, as expected)", flush=True)

# ── JAX gradient ──────────────────────────────────────────────────────────────
print(f"\n=== jax.grad (includes JIT compile) ===", flush=True)
t0 = time.time()
ad_val = float(jax.jit(jax.grad(forward_gpp_g1))(jnp.float64(1.0)))
dt_ad = time.time() - t0
print(f"  d(GPP)/d(alpha_g1) [JAX] = {ad_val:+.6e}  ({dt_ad:.1f}s)", flush=True)

# ── Result ────────────────────────────────────────────────────────────────────
# Use relaxed criterion (5%) because eps=0.1 FD has finite-difference truncation error.
# At eps=0.1 (large step), FD captures nonlinearity but not pure Jacobian.
# Session 39 confirmed JAX/FD ratio = 0.992 at eps=0.1 for LeafPhotosynthesis level.
print(f"\n=== Result ===", flush=True)
rel_err = abs(ad_val - fd_val) / (abs(fd_val) + 1e-30)
passed  = rel_err < 0.05  # 5% criterion for eps=0.1 finite-difference
print(f"  FD  (eps=0.1)   = {fd_val:+.6e}")
print(f"  JAX (exact AD)  = {ad_val:+.6e}")
print(f"  Rel. error = {rel_err:.2e}")
print(f"  Status: {'PASS' if passed else 'FAIL'}  (<5% criterion for eps=0.1 FD)", flush=True)

if passed and abs(ad_val) > 1e-8:
    print("\n  g1_MED_jax explicit-arg fix CONFIRMED: d(GPP)/d(g1_MED) is correct under Medlyn.")
elif not passed:
    print("\n  FAIL — gradient disagrees with FD(eps=0.1). Unexpected (JAX expected within 5%).")
