"""
debug_second_loop.py — Direct test of second loop gradient.

Tests whether the second loop of LeafPhotosynthesis correctly propagates
d(agross_leaf)/d(g1_MED) from gs_leaf (set by first loop) through to agross_leaf.

Strategy:
1. Run a forward pass to get a valid mlcanopy_inst state (after first loop ran)
2. Directly manipulate gs_leaf by scaling it
3. Run ONLY the second loop logic manually
4. Check d(agross)/d(scale_gs) via JAX vs FD

This isolates whether the second loop quadratic correctly propagates gradients.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

os.environ["CLM_ML_NO_CHECKPOINT"] = "1"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _d in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from multilayer_canopy import MLclm_varctl
MLclm_varctl.gs_type = 0
print(f"gs_type = {MLclm_varctl.gs_type} (Medlyn)", flush=True)

from diags.expt_init import mlcanopy_inst, grid
from clm_src_main.PatchType import patch as _patch
import multilayer_canopy.MLpftconMod as _MLpftconMod

_p    = grid.p
_ncan = grid.ncan
_pft  = int(np.asarray(_patch.itype)[_p])
_orig_pftcon = _MLpftconMod.MLpftcon

from multilayer_canopy.MLclm_varpar import isun, isha
from multilayer_canopy.MLMathToolsMod import quadratic
from multilayer_canopy.MLLeafPhotosynthesisMod import _RealizedRate
from multilayer_canopy.MLclm_varctl import gspot_type, acclim_type
from multilayer_canopy.MLclm_varcon import dh2o_to_dco2

print(f"patch={_p}, pft={_pft}, ncan={_ncan}", flush=True)
print(f"gspot_type={gspot_type}, acclim_type={acclim_type}", flush=True)

# Run first loop to get a valid mlcanopy_inst (with gs_leaf set by Medlyn kernel)
from multilayer_canopy.MLLeafPhotosynthesisMod import LeafPhotosynthesis
from diags.expt_init import compute_gpp, _mlcf_kwargs

from clm_src_main import clm_driver as _clm_driver_mod
_filt = _clm_driver_mod.filter
_num = int(_filt.num_exposedvegp)
_filt_list = [int(_filt.exposedvegp[i]) for i in range(_num)]
_o2ref_py = float(mlcanopy_inst.o2ref_forcing[_p])
_orig_g1_MED = jnp.asarray(_orig_pftcon.g1_MED)

# Run first loop (sun) to get a state where gs_leaf[isun] is set
inst_after_first_loop = LeafPhotosynthesis(
    _num, _filt_list, isun, mlcanopy_inst,
    grid=grid, _o2ref_py=_o2ref_py, g1_MED_jax=_orig_g1_MED,
)
print("First loop done. Checking gs_leaf[isun]:", flush=True)
gs_sun = np.asarray(inst_after_first_loop.gs_leaf[_p, :, isun])
dpai = np.asarray(inst_after_first_loop.dpai_profile[_p])
active = dpai > 0
print(f"  gs_leaf[isun, active] range: {float(np.min(gs_sun[active])):.4f}..{float(np.max(gs_sun[active])):.4f}", flush=True)

# Now test d(GPP_second_loop)/d(scale_gs)
# The second loop computes agross_sl from gleaf_sl = 1/(1/gbc + 1.6/gs_new_sl)
# where gs_new_sl = max(gs_leaf * fpsi, gsmin)
#
# Test: scale gs_leaf by alpha, run second loop manually, check d(agross)/d(alpha)

_sl = slice(1, _ncan + 1)
_psi50 = float(_orig_pftcon.psi50_gs[_pft])
_shape_gs = float(_orig_pftcon.shape_gs[_pft])
_gsmin = jnp.asarray(_orig_pftcon.gsmin_SPA[_pft])

# Precompute fpsi (shape: ncan,)
if gspot_type == 1:
    lwp_arr = inst_after_first_loop.lwp_leaf[_p, _sl, isun]  # shape (ncan,)
    fpsi = 1.0 / (1.0 + (np.asarray(lwp_arr) / _psi50) ** _shape_gs)
else:
    fpsi = np.ones(_ncan)

print(f"fpsi shape: {fpsi.shape}, active shape: {active[_sl].shape}", flush=True)
active_sl_bool = active[_sl]  # shape (ncan,)
print(f"fpsi[active] range: {float(np.min(fpsi[active_sl_bool])):.4f}..{float(np.max(fpsi[active_sl_bool])):.4f}", flush=True)

# Extract second-loop inputs from inst_after_first_loop
_gbc_sl = inst_after_first_loop.gbc_leaf[_p, _sl, isun]
_gbv_sl = inst_after_first_loop.gbv_leaf[_p, _sl, isun]
_eair_sl = inst_after_first_loop.eair_profile[_p, _sl]
_lesat_sl = inst_after_first_loop.leaf_esat_leaf[_p, _sl, isun]
_vcmax_sl = inst_after_first_loop.vcmax_leaf[_p, _sl, isun]
_je_sl   = inst_after_first_loop.je_leaf[_p, _sl, isun]
_rd_sl   = inst_after_first_loop.rd_leaf[_p, _sl, isun]
_kc_sl   = inst_after_first_loop.kc_leaf[_p, _sl, isun]
_ko_sl   = inst_after_first_loop.ko_leaf[_p, _sl, isun]
_cp_sl   = inst_after_first_loop.cp_leaf[_p, _sl, isun]
_cair_sl = inst_after_first_loop.cair_profile[_p, _sl]
_apar_sl = inst_after_first_loop.apar_leaf[_p, _sl, isun]
_o2ref   = inst_after_first_loop.o2ref_forcing[_p]
_gs_sl_base = inst_after_first_loop.gs_leaf[_p, _sl, isun]  # from first loop

_c3psn_val = 1.0  # is_c3=True
active_sl = np.asarray(inst_after_first_loop.dpai_profile[_p, _sl]) > 0
_fpsi_sl = jnp.asarray(fpsi)  # shape (ncan,) — already sliced above
_dpai_sl  = mlcanopy_inst.dpai_profile[_p, _sl]  # pre-extracted, used in both fw_second_loop and fw_full_chain

EPS = 1e-4

print("\n=== Test: d(agross_sum)/d(scale_gs) directly in second loop ===", flush=True)

def fw_second_loop(scale_gs: jnp.ndarray) -> jnp.ndarray:
    """Scale gs_leaf by scale_gs, run second loop quadratic, return sum(agross*dpai)."""
    _gs_sl = scale_gs * _gs_sl_base
    gs_new_sl = jnp.maximum(_gs_sl * _fpsi_sl, _gsmin)

    # Guard for inactive layers
    _gs_safe  = jnp.maximum(gs_new_sl, jnp.asarray(1.0e-30))
    _gbc_safe = jnp.maximum(_gbc_sl,   jnp.asarray(1.0e-30))
    gleaf_sl  = 1.0 / (1.0 / _gbc_safe + dh2o_to_dco2 / _gs_safe)

    # C3 quadratic (second loop recomputation)
    _ko_safe2 = jnp.maximum(_ko_sl, jnp.asarray(1.0e-30))
    b0_sl  = _kc_sl * (1.0 + _o2ref / _ko_safe2)
    bq_sl  = -(_cair_sl + b0_sl) - (_vcmax_sl - _rd_sl) / gleaf_sl
    cq_sl  = _vcmax_sl * (_cair_sl - _cp_sl) - _rd_sl * (_cair_sl + b0_sl)
    r1_sl, r2_sl = quadratic(1.0 / gleaf_sl, bq_sl, cq_sl)
    ac_sl  = jnp.minimum(r1_sl, r2_sl) + _rd_sl

    b0j_sl = 2.0 * _cp_sl
    bqj_sl = -(_cair_sl + b0j_sl) - (_je_sl / 4.0 - _rd_sl) / gleaf_sl
    cqj_sl = (_je_sl / 4.0) * (_cair_sl - _cp_sl) - _rd_sl * (_cair_sl + b0j_sl)
    r1j_sl, r2j_sl = quadratic(1.0 / gleaf_sl, bqj_sl, cqj_sl)
    aj_sl  = jnp.minimum(r1j_sl, r2j_sl) + _rd_sl
    ap_sl  = jnp.zeros_like(ac_sl)

    agross_sl = _RealizedRate(_c3psn_val, ac_sl, aj_sl, ap_sl)

    # Weighted sum over active layers (use pre-extracted constant)
    return jnp.sum(jnp.where(active_sl_bool, agross_sl * _dpai_sl, 0.0))

ONE = jnp.float64(1.0)

base = float(fw_second_loop(ONE))
print(f"  agross*dpai sum (second loop) = {base:.6f}", flush=True)
fd_plus  = float(fw_second_loop(ONE + EPS))
fd_minus = float(fw_second_loop(ONE - EPS))
fd_val = (fd_plus - fd_minus) / (2.0 * EPS)
print(f"  FD  d(agross_sum)/d(scale_gs) = {fd_val:+.6e}", flush=True)

t0 = time.time()
ad_val = float(jax.jit(jax.grad(fw_second_loop))(ONE))
print(f"  JAX d(agross_sum)/d(scale_gs) = {ad_val:+.6e}  ({time.time()-t0:.1f}s)", flush=True)
print(f"  Ratio JAX/FD = {ad_val/fd_val if abs(fd_val) > 1e-10 else float('nan'):.4f}", flush=True)

# Now chain: test d(agross)/d(alpha) through the full chain:
# alpha → kernel g1_val → gs_leaf → scale × gs_leaf → second loop → agross

print("\n=== Test: full chain alpha → g1 → kernel gs → second loop agross ===", flush=True)
from multilayer_canopy.MLLeafPhotosynthesisMod import (
    _get_vmapped_photo_kernel_acclim,
    _get_vmapped_photo_kernel,
    _fth25_py, _fth25,
)
from multilayer_canopy.MLclm_varcon import (
    vcmaxha_acclim, vcmaxhd_acclim, jmaxha_acclim, jmaxhd_acclim, rdha, rdhd, rdse,
)
from clm_src_main.clm_varcon import tfrz

rdc = _fth25_py(rdhd, rdse)
_o2ref_cache_key = float(mlcanopy_inst.o2ref_forcing[_p])

ta_c = jnp.clip(mlcanopy_inst.tacclim_forcing[_p] - tfrz, 11.0, 35.0)
vcmaxse = 668.39 - 1.07 * ta_c
jmaxse  = 659.70 - 0.75 * ta_c
vcmaxc  = _fth25(vcmaxhd_acclim, vcmaxse)
jmaxc   = _fth25(jmaxhd_acclim,  jmaxse)

vmapped = _get_vmapped_photo_kernel_acclim(
    is_c3=True, c3psn_pft_val=1.0,
    vcmaxha=vcmaxha_acclim, vcmaxhd=vcmaxhd_acclim,
    jmaxha=jmaxha_acclim,   jmaxhd=jmaxhd_acclim,
    rdc=rdc, o2ref_p=_o2ref_cache_key,
)

_tleaf_sl = mlcanopy_inst.tleaf_leaf[_p, _sl, isun]
_vcmax25  = mlcanopy_inst.vcmax25_leaf[_p, _sl, isun]
_jmax25   = mlcanopy_inst.jmax25_leaf[_p, _sl, isun]
_rd25     = mlcanopy_inst.rd25_leaf[_p, _sl, isun]
_kp25     = mlcanopy_inst.kp25_leaf[_p, _sl, isun]
_eair     = mlcanopy_inst.eair_profile[_p, _sl]
_apar     = mlcanopy_inst.apar_leaf[_p, _sl, isun]
_gbc      = mlcanopy_inst.gbc_leaf[_p, _sl, isun]
_gbv      = mlcanopy_inst.gbv_leaf[_p, _sl, isun]
_cair     = mlcanopy_inst.cair_profile[_p, _sl]

g0_jax = jnp.float64(float(_orig_pftcon.g0_MED[_pft]))

def fw_full_chain(g1_val_x: jnp.ndarray) -> jnp.ndarray:
    """g1 → kernel gs → second loop agross."""
    outs = vmapped(
        _dpai_sl, _tleaf_sl, _vcmax25, _jmax25, _rd25, _kp25,
        _eair, _apar, _gbc, _gbv, _cair,
        g0_jax, g1_val_x,
        vcmaxse, vcmaxc, jmaxse, jmaxc,
    )
    gs_from_kernel = outs[10]  # _gs

    # Second loop (copy of fw_second_loop but using gs_from_kernel)
    gs_new_sl = jnp.maximum(gs_from_kernel * _fpsi_sl, _gsmin)
    _gs_safe  = jnp.maximum(gs_new_sl, jnp.asarray(1.0e-30))
    _gbc_safe = jnp.maximum(_gbc_sl,   jnp.asarray(1.0e-30))
    gleaf_sl  = 1.0 / (1.0 / _gbc_safe + dh2o_to_dco2 / _gs_safe)

    _ko_safe2 = jnp.maximum(_ko_sl, jnp.asarray(1.0e-30))
    b0_sl  = _kc_sl * (1.0 + _o2ref / _ko_safe2)
    bq_sl  = -(_cair_sl + b0_sl) - (_vcmax_sl - _rd_sl) / gleaf_sl
    cq_sl  = _vcmax_sl * (_cair_sl - _cp_sl) - _rd_sl * (_cair_sl + b0_sl)
    r1_sl, r2_sl = quadratic(1.0 / gleaf_sl, bq_sl, cq_sl)
    ac_sl  = jnp.minimum(r1_sl, r2_sl) + _rd_sl

    b0j_sl = 2.0 * _cp_sl
    bqj_sl = -(_cair_sl + b0j_sl) - (_je_sl / 4.0 - _rd_sl) / gleaf_sl
    cqj_sl = (_je_sl / 4.0) * (_cair_sl - _cp_sl) - _rd_sl * (_cair_sl + b0j_sl)
    r1j_sl, r2j_sl = quadratic(1.0 / gleaf_sl, bqj_sl, cqj_sl)
    aj_sl  = jnp.minimum(r1j_sl, r2j_sl) + _rd_sl
    ap_sl  = jnp.zeros_like(ac_sl)

    agross_sl = _RealizedRate(1.0, ac_sl, aj_sl, ap_sl)
    return jnp.sum(jnp.where(active_sl_bool, agross_sl * _dpai_sl, 0.0))

g1_jax = jnp.float64(float(_orig_pftcon.g1_MED[_pft]))

base2 = float(fw_full_chain(g1_jax))
print(f"  agross*dpai sum (full chain) = {base2:.6f}", flush=True)

fd_plus2  = float(fw_full_chain(g1_jax + g1_jax * EPS))
fd_minus2 = float(fw_full_chain(g1_jax - g1_jax * EPS))
fd_val2   = (fd_plus2 - fd_minus2) / (2.0 * float(g1_jax) * EPS)
print(f"  FD  d(agross)/d(g1) [full chain] = {fd_val2:+.6e}", flush=True)

t0 = time.time()
ad_val2 = float(jax.jit(jax.grad(fw_full_chain))(g1_jax))
print(f"  JAX d(agross)/d(g1) [full chain] = {ad_val2:+.6e}  ({time.time()-t0:.1f}s)", flush=True)
print(f"  Ratio JAX/FD = {ad_val2/fd_val2 if abs(fd_val2)>1e-10 else float('nan'):.4f}", flush=True)

print("\n=== debug_second_loop.py done ===", flush=True)
