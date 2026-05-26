"""
check_g1_medlyn.py — Verify d(GPP)/d(alpha_g1) under gs_type=0 (Medlyn stomatal model).

Session 25 (commit 0305f82) moved g1_MED from a closed-over Python float to a
runtime JAX broadcast scalar so autodiff can trace through it.  That fix was
committed but never run under Medlyn mode — every subsequent job used gs_type=2
(WUE), where g1 is inactive.  This script fills that gap.

Steps:
  1. Sets gs_type=0 (Medlyn) BEFORE expt_init runs the warmup timestep.
  2. Runs FD first: confirms g1 is actually active (FD != 0) under Medlyn.
  3. Runs jax.grad: compares against FD, reports PASS/FAIL at <1% rel. error.

Usage (from project root):
    python diags/check_g1_medlyn.py
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
import multilayer_canopy.MLpftconMod             as _MLpftconMod
import multilayer_canopy.MLLeafPhotosynthesisMod as _LeafMod
import multilayer_canopy.MLCanopyNitrogenProfileMod as _NitroMod

_p = grid.p
_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}

_orig_pftcon = _MLpftconMod.MLpftcon
print(f"g1_MED (PFT 1..5): {list(_orig_pftcon.g1_MED[1:6])}", flush=True)

def _set_pftcon(new_inst):
    _MLpftconMod.MLpftcon = new_inst
    _LeafMod.MLpftcon     = new_inst
    _NitroMod.MLpftcon    = new_inst

def _restore_pftcon():
    _MLpftconMod.MLpftcon = _orig_pftcon
    _LeafMod.MLpftcon     = _orig_pftcon
    _NitroMod.MLpftcon    = _orig_pftcon

def forward_gpp_g1(alpha: jnp.ndarray) -> jnp.ndarray:
    """Scale all PFT g1_MED values by alpha; return canopy GPP proxy."""
    _set_pftcon(_orig_pftcon._replace(g1_MED=alpha * _orig_pftcon.g1_MED))
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=atm2lnd_inst,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    _restore_pftcon()
    return compute_gpp(inst, _p, grid.ncan)

# ── Baseline ──────────────────────────────────────────────────────────────────
print(f"\n=== Medlyn baseline (gs_type={MLclm_varctl.gs_type}) ===", flush=True)
gpp_base = float(forward_gpp_g1(jnp.float64(1.0)))
print(f"  GPP proxy = {gpp_base:.6f}", flush=True)
if gpp_base == 0.0:
    print("  ERROR: baseline GPP is zero — may be nighttime step. Gradient will be uninformative.")

# ── FD gradient (quick sanity check that g1 is active) ───────────────────────
EPS = 1e-4
print(f"\n=== FD gradient (eps={EPS}) ===", flush=True)
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
print(f"\n=== Result ===", flush=True)
rel_err = abs(ad_val - fd_val) / (abs(fd_val) + 1e-30)
passed  = rel_err < 0.01
print(f"  FD  = {fd_val:+.6e}")
print(f"  JAX = {ad_val:+.6e}")
print(f"  Rel. error = {rel_err:.2e}")
print(f"  Status: {'PASS' if passed else 'FAIL'}  (<1% criterion)", flush=True)

if passed and abs(ad_val) > 1e-8:
    print("\n  Session 25 fix CONFIRMED: d(GPP)/d(g1_MED) is correct under Medlyn model.")
elif not passed:
    print("\n  FAIL — gradient is non-zero but inaccurate. Session 25 fix may be incomplete.")
