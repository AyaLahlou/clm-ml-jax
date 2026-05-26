"""
Direct photosynthesis kernel gradient test.

Tests d(agross)/d(apar) for a single canopy layer by calling
the vmapped photosynthesis kernel directly — bypasses MLCanopyFluxes
and SolarRadiation, so any gradient error here is PURELY in the
photosynthesis kernel.

Supports gs_type 0 (Medlyn), 1 (Ball-Berry), and 2 (WUE).

If this test PASSES: the bug is NOT in the photosynthesis kernel.
If this test FAILS: the bug IS in the photosynthesis kernel.

Usage:
    cd /burg-archive/home/al4385/clm-ml-jax
    CLM_ML_NO_CHECKPOINT=1 python diags/test_photo_kernel_grad.py
"""
from __future__ import annotations
import os, sys
os.environ['CLM_ML_NO_CHECKPOINT'] = '1'

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diags.expt_init import (
    mlcanopy_inst, grid, jax, jnp,
)
from multilayer_canopy.MLclm_varpar import isun, isha
from multilayer_canopy.MLclm_varctl import gs_type, colim_type
from multilayer_canopy.MLLeafPhotosynthesisMod import (
    _get_vmapped_photo_kernel, _get_vmapped_photo_kernel_wue,
    _make_leaf_photo_kernel, _fth25_py,
)
from multilayer_canopy import MLclm_varpar as _vpar
from multilayer_canopy.MLclm_varcon import (
    vcmaxha_noacclim, vcmaxhd_noacclim, vcmaxse_noacclim,
    jmaxha_noacclim, jmaxhd_noacclim, jmaxse_noacclim,
    rdhd, rdse,
)
from clm_src_main.pftconMod import pftcon as _pftcon
from multilayer_canopy.MLpftconMod import MLpftcon as _MLpftcon
from clm_src_main.PatchType import patch
import numpy as np

_p    = grid.p
_ncan = grid.ncan

EPS = 1e-4

print("\n" + "="*70)
print("PHOTOSYNTHESIS KERNEL GRADIENT TEST")
print(f"  gs_type={gs_type}, colim_type={colim_type}, ncan={_ncan}")
print("="*70)

# ── Extract baseline leaf-level data from warmed-up state ─────────────────────
_sl = slice(1, _ncan + 1)

# PFT for patch p
pft_idx = int(patch.itype[_p])

# Photosynthesis parameters (acclim_type==0: module-level constants)
vcmaxha    = vcmaxha_noacclim
vcmaxhd    = vcmaxhd_noacclim
vcmaxse    = vcmaxse_noacclim
vcmaxc     = _fth25_py(vcmaxhd_noacclim, vcmaxse_noacclim)
jmaxha     = jmaxha_noacclim
jmaxhd     = jmaxhd_noacclim
jmaxse     = jmaxse_noacclim
jmaxc      = _fth25_py(jmaxhd_noacclim, jmaxse_noacclim)
rdc        = _fth25_py(rdhd, rdse)

c3psn_val  = float(np.asarray(_pftcon.c3psn)[pft_idx])
is_c3_bool = round(c3psn_val) == 1

# WUE params
iota_pft  = float(np.asarray(_MLpftcon.iota_SPA)[pft_idx])
gsmin_pft = float(np.asarray(_MLpftcon.gsmin_SPA)[pft_idx])

# Medlyn/BB params (for gs_type 0/1)
g0_val = float(np.asarray(_MLpftcon.g0_MED)[pft_idx]) if gs_type == 0 else float(np.asarray(_MLpftcon.g0_BB)[pft_idx])
g1_val = float(np.asarray(_MLpftcon.g1_MED)[pft_idx]) if gs_type == 0 else float(np.asarray(_MLpftcon.g1_BB)[pft_idx])

o2ref_py  = float(mlcanopy_inst.o2ref_forcing[_p])
pref_py   = float(mlcanopy_inst.pref_forcing[_p])

print(f"  pft_idx={pft_idx}, is_c3={is_c3_bool}, iota_pft={iota_pft:.3f}, gsmin_pft={gsmin_pft:.4f}")
print(f"  o2ref={o2ref_py:.2f} mmol/mol, pref={pref_py:.1f} Pa")

# ── Build the vmapped kernel ──────────────────────────────────────────────────
if gs_type in (0, 1):
    vmapped = _get_vmapped_photo_kernel(
        is_c3=is_c3_bool,
        c3psn_pft_val=c3psn_val,
        vcmaxha=vcmaxha, vcmaxhd=vcmaxhd,
        vcmaxse=vcmaxse, vcmaxc=vcmaxc,
        jmaxha=jmaxha, jmaxhd=jmaxhd,
        jmaxse=jmaxse, jmaxc=jmaxc,
        rdc=rdc,
        g0_val=g0_val, g1_val=g1_val,
        o2ref_p=o2ref_py,
    )
elif gs_type == 2:
    vmapped = _get_vmapped_photo_kernel_wue(
        is_c3=is_c3_bool,
        c3psn_pft_val=c3psn_val,
        vcmaxha=vcmaxha, vcmaxhd=vcmaxhd,
        vcmaxse=vcmaxse, vcmaxc=vcmaxc,
        jmaxha=jmaxha, jmaxhd=jmaxhd,
        jmaxse=jmaxse, jmaxc=jmaxc,
        rdc=rdc,
        iota_pft=iota_pft, gsmin_pft=gsmin_pft,
    )
else:
    raise ValueError(f"Unsupported gs_type={gs_type}")

# ── Extract baseline leaf-level data from warmed-up state ────────────────────
for il, leaf_label in [(isun, 'isun'), (isha, 'isha')]:
    print(f"\n--- Leaf type: {leaf_label} ---")

    dpai_arr   = np.asarray(mlcanopy_inst.dpai_profile[_p, _sl])        # (ncan,)
    tleaf_arr  = np.asarray(mlcanopy_inst.tleaf_leaf[_p, _sl, il])      # (ncan,)
    vcmax25_arr= np.asarray(mlcanopy_inst.vcmax25_leaf[_p, _sl, il])
    jmax25_arr = np.asarray(mlcanopy_inst.jmax25_leaf[_p, _sl, il])
    rd25_arr   = np.asarray(mlcanopy_inst.rd25_leaf[_p, _sl, il])
    kp25_arr   = np.asarray(mlcanopy_inst.kp25_leaf[_p, _sl, il])
    eair_arr   = np.asarray(mlcanopy_inst.eair_profile[_p, _sl])
    apar_arr   = np.asarray(mlcanopy_inst.apar_leaf[_p, _sl, il])       # (ncan,)
    gbc_arr    = np.asarray(mlcanopy_inst.gbc_leaf[_p, _sl, il])
    gbv_arr    = np.asarray(mlcanopy_inst.gbv_leaf[_p, _sl, il])
    cair_arr   = np.asarray(mlcanopy_inst.cair_profile[_p, _sl])

    # Convert to JAX arrays
    dpai_j   = jnp.array(dpai_arr)
    tleaf_j  = jnp.array(tleaf_arr)
    vcmax25_j= jnp.array(vcmax25_arr)
    jmax25_j = jnp.array(jmax25_arr)
    rd25_j   = jnp.array(rd25_arr)
    kp25_j   = jnp.array(kp25_arr)
    eair_j   = jnp.array(eair_arr)
    apar_j   = jnp.array(apar_arr)
    gbc_j    = jnp.array(gbc_arr)
    gbv_j    = jnp.array(gbv_arr)
    cair_j   = jnp.array(cair_arr)

    def forward_agross_kernel(apar_scale: jnp.ndarray) -> jnp.ndarray:
        """Scale all apar values by apar_scale, return sum(agross)."""
        if gs_type in (0, 1):
            layer_out = vmapped(
                dpai_j, tleaf_j, vcmax25_j, jmax25_j, rd25_j, kp25_j,
                eair_j, apar_j * apar_scale, gbc_j, gbv_j, cair_j,
            )
        else:  # WUE
            layer_out = vmapped(
                dpai_j, tleaf_j, vcmax25_j, jmax25_j, rd25_j, kp25_j,
                eair_j, apar_j * apar_scale, gbc_j, gbv_j, cair_j,
                jnp.float64(o2ref_py), jnp.float64(pref_py),
            )
        # layer_out[15] = agross (0-indexed from tuple, see kernel return)
        # For WUE: (kc,ko,cp,vcmax,jmax,rd,kp,lesat,ceair,je,gs,ci,ac,aj,ap,agross,anet,cs)
        #                                                                     idx=15
        # For Medlyn/BB: same structure, agross is at index 10 (agross_val)
        # Both kernels return same 18-tuple:
        # (kc,ko,cp,vcmax,jmax,rd,kp,lesat,ceair,je,gs,ci,ac,aj,ap,agross,anet,cs)
        # → agross is index 15 for ALL kernel types
        agross = layer_out[15]  # agross is at index 15 for all kernels
        return jnp.sum(agross)

    # Baseline
    baseline_agross = float(forward_agross_kernel(jnp.float64(1.0)))
    print(f"  baseline sum(agross) = {baseline_agross:.6f}", flush=True)

    # JAX gradient
    import time
    t0 = time.time()
    jax_grad = float(jax.jit(jax.grad(forward_agross_kernel))(jnp.float64(1.0)))
    print(f"  JAX grad d(sum(agross))/d(scale) = {jax_grad:.6e}  ({time.time()-t0:.1f}s)", flush=True)

    # FD
    f_plus  = float(forward_agross_kernel(jnp.float64(1.0 + EPS)))
    f_minus = float(forward_agross_kernel(jnp.float64(1.0 - EPS)))
    fd_grad = (f_plus - f_minus) / (2.0 * EPS)
    print(f"  FD  grad d(sum(agross))/d(scale) = {fd_grad:.6e}", flush=True)

    rel = abs(jax_grad - fd_grad) / (abs(fd_grad) + 1e-30)
    print(f"  Rel error: {rel:.3e}  {'PASS' if rel < 0.01 else 'FAIL'}", flush=True)

    # Also show baseline sum(apar) for comparison
    print(f"  baseline sum(apar) = {float(jnp.sum(apar_j)):.6f}", flush=True)

    # Layer-by-layer breakdown for first few layers
    print(f"\n  Layer-by-layer (first 5 non-zero apar):", flush=True)
    print(f"  {'ic':>3}  {'apar':>10}  {'jax_grad_ic':>12}  {'fd_grad_ic':>12}  {'rel_err':>10}", flush=True)
    count = 0
    for ic in range(_ncan):
        _apar_ic = float(apar_j[ic])
        if _apar_ic < 1.0:
            continue  # skip near-zero apar layers
        count += 1
        if count > 5:
            break

        def _single_layer(s, _ic=ic):
            _a = apar_j.at[_ic].set(apar_j[_ic] * s)
            if gs_type in (0, 1):
                out = vmapped(dpai_j, tleaf_j, vcmax25_j, jmax25_j, rd25_j, kp25_j, eair_j, _a, gbc_j, gbv_j, cair_j)
            else:
                out = vmapped(dpai_j, tleaf_j, vcmax25_j, jmax25_j, rd25_j, kp25_j, eair_j, _a, gbc_j, gbv_j, cair_j, jnp.float64(o2ref_py), jnp.float64(pref_py))
            return out[15][_ic]  # agross at index 15 for all kernels

        _g_ic = float(jax.grad(_single_layer)(jnp.float64(1.0)))
        _fd_ic = (float(_single_layer(jnp.float64(1.0+EPS))) - float(_single_layer(jnp.float64(1.0-EPS)))) / (2*EPS)
        _rel_ic = abs(_g_ic - _fd_ic) / (abs(_fd_ic) + 1e-30)
        print(f"  {ic+1:>3}  {_apar_ic:>10.4f}  {_g_ic:>12.6e}  {_fd_ic:>12.6e}  {_rel_ic:>10.3e}", flush=True)

print("\n=== test_photo_kernel_grad.py complete ===", flush=True)
