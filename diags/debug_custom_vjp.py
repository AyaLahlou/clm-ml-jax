"""
Experiment: Diagnose and verify custom_vjp fix for tridiag_2eq.

Three tests:
  1. FD at multiple epsilons — confirm FD stability (rules out nonlinearity/bad eps)
  2. flux_profile_type=0 (well-mixed, no tridiag_2eq) — if JAX≈FD, confirms custom_vjp is culprit
  3. jacfwd vs jacrev comparison — jacfwd bypasses custom_vjp; if jacfwd≈FD≠jacrev → custom_vjp buggy

Usage:
    cd src && python ../diags/debug_custom_vjp.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs, jax, jnp, MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst, compute_gpp,
)
from multilayer_canopy import MLclm_varctl

_p = grid.p
_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}


def forward_gpp_sw(alpha: jnp.ndarray) -> jnp.ndarray:
    modified_atm = atm2lnd_inst._replace(
        forc_solad_downscaled_col = alpha * atm2lnd_inst.forc_solad_downscaled_col,
        forc_solai_grc            = alpha * atm2lnd_inst.forc_solai_grc,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, grid.ncan)


def forward_gpp_tref(alpha: jnp.ndarray) -> jnp.ndarray:
    modified_atm = atm2lnd_inst._replace(
        forc_t_downscaled_col = alpha * atm2lnd_inst.forc_t_downscaled_col,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, grid.ncan)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: FD at multiple epsilons — check FD stability
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 1: FD stability across multiple epsilons")
print("="*70)
print(f"{'eps':>10}  {'FD(alpha_sw)':>16}  {'FD(alpha_tref)':>16}")
print("-" * 48)
fd_sw_vals = {}
fd_tref_vals = {}
for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    f_plus_sw  = float(forward_gpp_sw(jnp.float64(1.0 + eps)))
    f_minus_sw = float(forward_gpp_sw(jnp.float64(1.0 - eps)))
    fd_sw      = (f_plus_sw - f_minus_sw) / (2 * eps)

    f_plus_t   = float(forward_gpp_tref(jnp.float64(1.0 + eps)))
    f_minus_t  = float(forward_gpp_tref(jnp.float64(1.0 - eps)))
    fd_tref    = (f_plus_t - f_minus_t) / (2 * eps)

    fd_sw_vals[eps]   = fd_sw
    fd_tref_vals[eps] = fd_tref
    print(f"{eps:>10.0e}  {fd_sw:>16.6f}  {fd_tref:>16.6f}")

# Check stability (variation across epsilons)
fd_sw_list   = list(fd_sw_vals.values())
fd_tref_list = list(fd_tref_vals.values())
sw_spread   = max(fd_sw_list) - min(fd_sw_list)
tref_spread = max(fd_tref_list) - min(fd_tref_list)
print(f"\nSpread in FD(alpha_sw):   {sw_spread:.4e}")
print(f"Spread in FD(alpha_tref): {tref_spread:.4e}")
if sw_spread < 0.01 * abs(fd_sw_list[2]) and tref_spread < 0.01 * abs(fd_tref_list[2]):
    print("→ FD is STABLE: FD values are the reference, JAX custom_vjp is the culprit")
else:
    print("→ FD is UNSTABLE: choose eps carefully before trusting FD")

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: jacfwd vs jacrev vs FD
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 2: jacfwd vs jacrev vs FD (eps=1e-4)")
print("="*70)
EPS = 1e-4
fd_sw   = (float(forward_gpp_sw(jnp.float64(1.0+EPS))) - float(forward_gpp_sw(jnp.float64(1.0-EPS)))) / (2*EPS)
fd_tref = (float(forward_gpp_tref(jnp.float64(1.0+EPS))) - float(forward_gpp_tref(jnp.float64(1.0-EPS)))) / (2*EPS)

t0 = time.time()
jacrev_sw = float(jax.jit(jax.grad(forward_gpp_sw))(jnp.float64(1.0)))
print(f"jacrev dGPP/d(alpha_sw)   = {jacrev_sw:.6e}  ({time.time()-t0:.1f}s)", flush=True)

t0 = time.time()
jacrev_tref = float(jax.jit(jax.grad(forward_gpp_tref))(jnp.float64(1.0)))
print(f"jacrev dGPP/d(alpha_tref) = {jacrev_tref:.6e}  ({time.time()-t0:.1f}s)", flush=True)

# jacfwd using jvp
t0 = time.time()
_, jacfwd_sw = jax.jit(lambda a: jax.jvp(forward_gpp_sw, (a,), (jnp.ones_like(a),)))(jnp.float64(1.0))
jacfwd_sw = float(jacfwd_sw)
print(f"jacfwd dGPP/d(alpha_sw)   = {jacfwd_sw:.6e}  ({time.time()-t0:.1f}s)", flush=True)

t0 = time.time()
_, jacfwd_tref = jax.jit(lambda a: jax.jvp(forward_gpp_tref, (a,), (jnp.ones_like(a),)))(jnp.float64(1.0))
jacfwd_tref = float(jacfwd_tref)
print(f"jacfwd dGPP/d(alpha_tref) = {jacfwd_tref:.6e}  ({time.time()-t0:.1f}s)", flush=True)

print(f"\n{'Method':<12}  {'alpha_sw':>14}  {'alpha_tref':>14}")
print("-"*44)
print(f"{'jacrev':<12}  {jacrev_sw:>14.6e}  {jacrev_tref:>14.6e}")
print(f"{'jacfwd':<12}  {jacfwd_sw:>14.6e}  {jacfwd_tref:>14.6e}")
print(f"{'FD(1e-4)':<12}  {fd_sw:>14.6e}  {fd_tref:>14.6e}")

rel_err_rev_sw   = abs(jacrev_sw   - fd_sw)   / (abs(fd_sw)   + 1e-30)
rel_err_fwd_sw   = abs(jacfwd_sw   - fd_sw)   / (abs(fd_sw)   + 1e-30)
rel_err_rev_tref = abs(jacrev_tref - fd_tref) / (abs(fd_tref) + 1e-30)
rel_err_fwd_tref = abs(jacfwd_tref - fd_tref) / (abs(fd_tref) + 1e-30)

print(f"\nRel error vs FD:")
print(f"  jacrev alpha_sw:   {rel_err_rev_sw:.3e}  {'PASS' if rel_err_rev_sw<0.01 else 'FAIL'}")
print(f"  jacfwd alpha_sw:   {rel_err_fwd_sw:.3e}  {'PASS' if rel_err_fwd_sw<0.01 else 'FAIL'}")
print(f"  jacrev alpha_tref: {rel_err_rev_tref:.3e}  {'PASS' if rel_err_rev_tref<0.01 else 'FAIL'}")
print(f"  jacfwd alpha_tref: {rel_err_fwd_tref:.3e}  {'PASS' if rel_err_fwd_tref<0.01 else 'FAIL'}")

if (rel_err_rev_sw > 0.01 and rel_err_fwd_sw < 0.01):
    print("\n→ DIAGNOSIS CONFIRMED: jacfwd≈FD but jacrev≠FD → custom_vjp (backward) is buggy")
elif (rel_err_rev_sw < 0.01 and rel_err_fwd_sw < 0.01):
    print("\n→ FIXED: Both jacrev and jacfwd agree with FD!")
else:
    print("\n→ INCONCLUSIVE or forward pass itself has issues (jacfwd also disagrees)")

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: flux_profile_type=0 (well-mixed, no tridiag_2eq custom_vjp)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 3: flux_profile_type=0 (well-mixed, bypasses tridiag_2eq)")
print("="*70)
orig_fpt = MLclm_varctl.flux_profile_type
MLclm_varctl.flux_profile_type = 0
print(f"  flux_profile_type set to 0 (was {orig_fpt})", flush=True)

t0 = time.time()
jacrev_sw_wm = float(jax.jit(jax.grad(forward_gpp_sw))(jnp.float64(1.0)))
print(f"  jacrev dGPP/d(alpha_sw)   [well-mixed] = {jacrev_sw_wm:.6e}  ({time.time()-t0:.1f}s)", flush=True)

t0 = time.time()
jacrev_tref_wm = float(jax.jit(jax.grad(forward_gpp_tref))(jnp.float64(1.0)))
print(f"  jacrev dGPP/d(alpha_tref) [well-mixed] = {jacrev_tref_wm:.6e}  ({time.time()-t0:.1f}s)", flush=True)

# FD with well-mixed
fd_sw_wm = (float(forward_gpp_sw(jnp.float64(1.0+EPS))) - float(forward_gpp_sw(jnp.float64(1.0-EPS)))) / (2*EPS)
fd_tref_wm = (float(forward_gpp_tref(jnp.float64(1.0+EPS))) - float(forward_gpp_tref(jnp.float64(1.0-EPS)))) / (2*EPS)

err_sw_wm   = abs(jacrev_sw_wm   - fd_sw_wm)   / (abs(fd_sw_wm)   + 1e-30)
err_tref_wm = abs(jacrev_tref_wm - fd_tref_wm) / (abs(fd_tref_wm) + 1e-30)
print(f"\n  FD(well-mixed) alpha_sw   = {fd_sw_wm:.6e}")
print(f"  FD(well-mixed) alpha_tref = {fd_tref_wm:.6e}")
print(f"\n  Rel error (jacrev vs FD), well-mixed:")
print(f"    alpha_sw:   {err_sw_wm:.3e}  {'PASS' if err_sw_wm<0.01 else 'FAIL'}")
print(f"    alpha_tref: {err_tref_wm:.3e}  {'PASS' if err_tref_wm<0.01 else 'FAIL'}")

if err_sw_wm < 0.01 and err_tref_wm < 0.01:
    print("\n→ CONFIRMED: well-mixed (no custom_vjp) passes → custom_vjp in tridiag_2eq is the culprit")
else:
    print("\n→ INCONCLUSIVE: well-mixed also fails → bug is outside tridiag_2eq custom_vjp")

# Restore
MLclm_varctl.flux_profile_type = orig_fpt

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  jacrev alpha_sw:   rel_err vs FD = {rel_err_rev_sw:.3e}  {'PASS' if rel_err_rev_sw<0.01 else 'FAIL'}")
print(f"  jacrev alpha_tref: rel_err vs FD = {rel_err_rev_tref:.3e}  {'PASS' if rel_err_rev_tref<0.01 else 'FAIL'}")
