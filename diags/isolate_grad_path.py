"""
Gradient path isolation: find where JAX vs FD diverges.

Tests dGPP/d(alpha_sw) at each stage:
  Stage 1: d(sum(apar_leaf))/d(alpha_sw)  — tests solar radiation path
  Stage 2: d(sum(agross_leaf))/d(alpha_sw) — tests full photo path

Usage: cd /burg-archive/home/al4385/clm-ml-jax && python diags/isolate_grad_path.py
"""
from __future__ import annotations
import os, sys, time
os.environ['CLM_ML_NO_CHECKPOINT'] = '1'

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs, jax, jnp, MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst, compute_gpp,
)
from multilayer_canopy.MLclm_varpar import isun, isha

_p = grid.p
_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}
EPS = 1e-4

# ── Stage 1: d(sum(apar_leaf))/d(alpha_sw) ───────────────────────────────────
print("\n" + "="*70)
print("STAGE 1: sum(apar_leaf) vs alpha_sw")
print("="*70)

def forward_apar_sum(alpha: jnp.ndarray) -> jnp.ndarray:
    modified_atm = atm2lnd_inst._replace(
        forc_solad_downscaled_col=alpha * atm2lnd_inst.forc_solad_downscaled_col,
        forc_solai_grc           =alpha * atm2lnd_inst.forc_solai_grc,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    # Use dpai-weighted sum to mask inactive layers (dpai=0 → no spval contamination)
    # Inactive layers have spval=1e36 in apar_leaf; plain sum causes FD cancellation
    dpai = inst.dpai_profile[_p, 1:grid.ncan+1]  # shape [ncan], 0 for inactive
    apar = inst.apar_leaf[_p, 1:grid.ncan+1, isun:isha+1]  # shape [ncan, 2]
    return jnp.sum(dpai[:, None] * apar)

t0 = time.time()
jax_grad_apar = float(jax.jit(jax.grad(forward_apar_sum))(jnp.float64(1.0)))
print(f"JAX grad d(sum(apar))/d(alpha_sw) = {jax_grad_apar:.6e}  ({time.time()-t0:.1f}s)", flush=True)

a_plus = float(forward_apar_sum(jnp.float64(1.0 + EPS)))
a_minus = float(forward_apar_sum(jnp.float64(1.0 - EPS)))
fd_grad_apar = (a_plus - a_minus) / (2 * EPS)
print(f"FD  grad d(sum(apar))/d(alpha_sw) = {fd_grad_apar:.6e}", flush=True)
rel1 = abs(jax_grad_apar - fd_grad_apar) / (abs(fd_grad_apar) + 1e-30)
print(f"Rel error: {rel1:.3e}  {'PASS' if rel1 < 0.01 else 'FAIL'}", flush=True)
baseline_apar = float(forward_apar_sum(jnp.float64(1.0)))
print(f"Baseline sum(apar) = {baseline_apar:.4f}", flush=True)
print(f"Expected (linear): FD ≈ sum(apar)/1.0 = {baseline_apar:.4f}", flush=True)

# ── Stage 2: d(sum(agross_leaf))/d(alpha_sw) via LeafPhotosynthesis only ─────
print("\n" + "="*70)
print("STAGE 2: sum(agross_leaf) vs alpha_sw (after LeafPhotosynthesis)")
print("="*70)

def forward_agross_sum(alpha: jnp.ndarray) -> jnp.ndarray:
    modified_atm = atm2lnd_inst._replace(
        forc_solad_downscaled_col=alpha * atm2lnd_inst.forc_solad_downscaled_col,
        forc_solai_grc           =alpha * atm2lnd_inst.forc_solai_grc,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    # Use dpai-weighted sum to mask inactive layers (same reason as Stage 1)
    dpai = inst.dpai_profile[_p, 1:grid.ncan+1]
    agross = inst.agross_leaf[_p, 1:grid.ncan+1, isun:isha+1]
    return jnp.sum(dpai[:, None] * agross)

t0 = time.time()
jax_grad_agross = float(jax.jit(jax.grad(forward_agross_sum))(jnp.float64(1.0)))
print(f"JAX grad d(sum(agross))/d(alpha_sw) = {jax_grad_agross:.6e}  ({time.time()-t0:.1f}s)", flush=True)

b_plus = float(forward_agross_sum(jnp.float64(1.0 + EPS)))
b_minus = float(forward_agross_sum(jnp.float64(1.0 - EPS)))
fd_grad_agross = (b_plus - b_minus) / (2 * EPS)
print(f"FD  grad d(sum(agross))/d(alpha_sw) = {fd_grad_agross:.6e}", flush=True)
rel2 = abs(jax_grad_agross - fd_grad_agross) / (abs(fd_grad_agross) + 1e-30)
print(f"Rel error: {rel2:.3e}  {'PASS' if rel2 < 0.01 else 'FAIL'}", flush=True)

# ── Stage 3: d(GPP)/d(alpha_sw) (with weighted average) ──────────────────────
print("\n" + "="*70)
print("STAGE 3: GPP (fracsun-weighted agross) vs alpha_sw")
print("="*70)

def forward_gpp_sw(alpha: jnp.ndarray) -> jnp.ndarray:
    modified_atm = atm2lnd_inst._replace(
        forc_solad_downscaled_col=alpha * atm2lnd_inst.forc_solad_downscaled_col,
        forc_solai_grc           =alpha * atm2lnd_inst.forc_solai_grc,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, grid.ncan)

t0 = time.time()
jax_grad_gpp = float(jax.jit(jax.grad(forward_gpp_sw))(jnp.float64(1.0)))
print(f"JAX grad dGPP/d(alpha_sw) = {jax_grad_gpp:.6e}  ({time.time()-t0:.1f}s)", flush=True)

c_plus = float(forward_gpp_sw(jnp.float64(1.0 + EPS)))
c_minus = float(forward_gpp_sw(jnp.float64(1.0 - EPS)))
fd_grad_gpp = (c_plus - c_minus) / (2 * EPS)
print(f"FD  grad dGPP/d(alpha_sw) = {fd_grad_gpp:.6e}", flush=True)
rel3 = abs(jax_grad_gpp - fd_grad_gpp) / (abs(fd_grad_gpp) + 1e-30)
print(f"Rel error: {rel3:.3e}  {'PASS' if rel3 < 0.01 else 'FAIL'}", flush=True)

# ── Baseline comparison ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("BASELINE VALUES")
print("="*70)
inst0 = MLCanopyFluxes(
    mlcanopy_inst=mlcanopy_inst,
    atm2lnd_inst=atm2lnd_inst,
    wateratm2lndbulk_inst=wateratm2lndbulk_inst,
    **_mlcf_kwargs_no_atm,
)
dpai0 = inst0.dpai_profile[_p, 1:5]
print(f"dpai_profile[p, 1:5] = {dpai0}")
print(f"apar_leaf[p, 1:5, isun] (raw) = {inst0.apar_leaf[_p, 1:5, isun]}")
print(f"apar_leaf[p, 1:5, isun] (dpai-masked) = {dpai0 * inst0.apar_leaf[_p, 1:5, isun]}")
print(f"agross_leaf[p, 1:5, isun] (raw) = {inst0.agross_leaf[_p, 1:5, isun]}")
print(f"fracsun_profile[p, 1:5] = {inst0.fracsun_profile[_p, 1:5]}")
print(f"dpai_profile[p, 1:5] = {inst0.dpai_profile[_p, 1:5]}")
print(f"GPP = {float(compute_gpp(inst0, _p, grid.ncan)):.4f}", flush=True)

print("\n=== isolate_grad_path.py complete ===", flush=True)
