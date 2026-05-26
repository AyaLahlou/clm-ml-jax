"""
debug_g1_grad_isolation.py — Isolate d(GPP)/d(g1) at each stage of the pipeline.

Tests gradient at 3 levels:
  Level 1: Kernel only (vmapped, single call) — d(agross_kernel)/d(g1)
  Level 2: First loop of LeafPhotosynthesis only — d(agross_after_first_loop)/d(g1)
  Level 3: Full second loop included — d(agross_after_second_loop)/d(g1)
  Level 4: Full MLCanopyFluxes — d(GPP)/d(g1)

Each level compares JAX grad vs central FD.
"""
from __future__ import annotations
import os, sys
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

# ── Set Medlyn BEFORE init ──────────────────────────────────────────────────
from multilayer_canopy import MLclm_varctl
MLclm_varctl.gs_type = 0
print(f"gs_type = {MLclm_varctl.gs_type} (Medlyn)", flush=True)

# ── Shared init ──────────────────────────────────────────────────────────────
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs,
    atm2lnd_inst, wateratm2lndbulk_inst, compute_gpp,
    MLCanopyFluxes,
)
import multilayer_canopy.MLpftconMod as _MLpftconMod
from multilayer_canopy.MLclm_varpar import isun, isha

_p    = grid.p
_ncan = grid.ncan
_orig_pftcon = _MLpftconMod.MLpftcon
_orig_g1_MED = jnp.asarray(_orig_pftcon.g1_MED)

from clm_src_main.PatchType import patch as _patch
_pft = int(np.asarray(_patch.itype)[_p])
print(f"patch={_p}, pft={_pft}, ncan={_ncan}", flush=True)
print(f"g1_MED[pft={_pft}] = {float(_orig_g1_MED[_pft]):.4f}", flush=True)

EPS = 1e-4
ONE = jnp.float64(1.0)

_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}

# =============================================================================
# Level 4: Full MLCanopyFluxes — d(GPP)/d(alpha_g1)
# =============================================================================
print("\n" + "="*70, flush=True)
print("Level 4: Full MLCanopyFluxes", flush=True)
print("="*70, flush=True)

def forward_full(alpha: jnp.ndarray) -> jnp.ndarray:
    g1_MED_jax = alpha * _orig_g1_MED
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=atm2lnd_inst,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        g1_MED_jax=g1_MED_jax,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, _ncan)

gpp_base = float(forward_full(ONE))
print(f"  GPP baseline = {gpp_base:.6f}", flush=True)

fd_plus  = float(forward_full(ONE + EPS))
fd_minus = float(forward_full(ONE - EPS))
fd_val4  = (fd_plus - fd_minus) / (2.0 * EPS)
print(f"  FD  d(GPP)/d(alpha) = {fd_val4:+.6e}", flush=True)

ad_val4 = float(jax.jit(jax.grad(forward_full))(ONE))
print(f"  JAX d(GPP)/d(alpha) = {ad_val4:+.6e}", flush=True)
print(f"  Ratio JAX/FD = {ad_val4/fd_val4 if abs(fd_val4)>1e-10 else float('nan'):.4f}", flush=True)

# =============================================================================
# Level 3: First loop + Second loop (agross after second loop, before any further)
#          Using LeafPhotosynthesis directly on mlcanopy_inst
# =============================================================================
print("\n" + "="*70, flush=True)
print("Level 3: LeafPhotosynthesis only (sun), second loop agross", flush=True)
print("="*70, flush=True)

from multilayer_canopy.MLLeafPhotosynthesisMod import LeafPhotosynthesis

# Build the filter as used in _physics_step_fn
from clm_src_main import clm_driver as _clm_driver_mod
_filt = _clm_driver_mod.filter
_num = int(_filt.num_exposedvegp)
_filt_list = [int(_filt.exposedvegp[i]) for i in range(_num)]

def forward_leaf_photo(alpha: jnp.ndarray) -> jnp.ndarray:
    """Run LeafPhotosynthesis (sun + shade) with scaled g1. Return sum(agross_leaf)."""
    g1_MED_jax = alpha * _orig_g1_MED
    _o2ref_py = float(mlcanopy_inst.o2ref_forcing[_p])
    inst_sun = LeafPhotosynthesis(
        _num, _filt_list, isun, mlcanopy_inst,
        grid=grid, _o2ref_py=_o2ref_py, g1_MED_jax=g1_MED_jax,
    )
    inst_both = LeafPhotosynthesis(
        _num, _filt_list, isha, inst_sun,
        grid=grid, _o2ref_py=_o2ref_py, g1_MED_jax=g1_MED_jax,
    )
    # Read agross_leaf: sum over layers and sun+shade
    return compute_gpp(inst_both, _p, _ncan)

gpp3 = float(forward_leaf_photo(ONE))
print(f"  agross sum baseline = {gpp3:.6f}", flush=True)

fd_plus3  = float(forward_leaf_photo(ONE + EPS))
fd_minus3 = float(forward_leaf_photo(ONE - EPS))
fd_val3   = (fd_plus3 - fd_minus3) / (2.0 * EPS)
print(f"  FD  d(agross_sum)/d(alpha) = {fd_val3:+.6e}", flush=True)

ad_val3 = float(jax.jit(jax.grad(forward_leaf_photo))(ONE))
print(f"  JAX d(agross_sum)/d(alpha) = {ad_val3:+.6e}", flush=True)
print(f"  Ratio JAX/FD = {ad_val3/fd_val3 if abs(fd_val3)>1e-10 else float('nan'):.4f}", flush=True)

# =============================================================================
# Level 2: First loop ONLY (agross from first loop, before second loop overwrites)
# =============================================================================
print("\n" + "="*70, flush=True)
print("Level 2: First loop only (gs_leaf from kernel, not second-loop agross)", flush=True)
print("="*70, flush=True)

# To access first-loop output only, we check gs_leaf (output of first loop that
# second loop uses as input), and also agross_leaf which first loop sets then
# second loop overwrites. To get first-loop agross, we need to intercept.
# Approach: set gspot_type=0 temporarily (fpsi=1), so second loop doesn't change things.
# Actually easier: just check gs_leaf after first loop runs.

from multilayer_canopy.MLLeafPhotosynthesisMod import (
    _get_vmapped_photo_kernel_acclim,
    _get_vmapped_photo_kernel,
)
from multilayer_canopy import MLclm_varctl as _ml_ctl

# Need to run the kernel directly with the actual inputs that are present
# after initialization (after SolarRadiation, CanopyNitrogenProfile, etc.)
# The simplest approach: run LeafPhotosynthesis but measure gs_leaf (first loop output)
# by setting gspot_type=0 temporarily so second loop doesn't apply fpsi

_orig_gspot = _ml_ctl.gspot_type
_ml_ctl.gspot_type = 0
print(f"  Temporarily set gspot_type=0 (fpsi=1) so second loop doesn't change gs", flush=True)

def forward_gs_leaf(alpha: jnp.ndarray) -> jnp.ndarray:
    """Run LeafPhotosynthesis, return sum of gs_leaf (first loop output)."""
    g1_MED_jax = alpha * _orig_g1_MED
    _o2ref_py = float(mlcanopy_inst.o2ref_forcing[_p])
    inst_sun = LeafPhotosynthesis(
        _num, _filt_list, isun, mlcanopy_inst,
        grid=grid, _o2ref_py=_o2ref_py, g1_MED_jax=g1_MED_jax,
    )
    inst_both = LeafPhotosynthesis(
        _num, _filt_list, isha, inst_sun,
        grid=grid, _o2ref_py=_o2ref_py, g1_MED_jax=g1_MED_jax,
    )
    # With gspot_type=0, fpsi=1 so gs_leaf after second loop = kernel gs_leaf
    # And agross_leaf after second loop is recomputed from kernel gs_leaf
    return compute_gpp(inst_both, _p, _ncan)

gpp2 = float(forward_gs_leaf(ONE))
print(f"  GPP baseline (gspot=0) = {gpp2:.6f}", flush=True)

fd_plus2  = float(forward_gs_leaf(ONE + EPS))
fd_minus2 = float(forward_gs_leaf(ONE - EPS))
fd_val2   = (fd_plus2 - fd_minus2) / (2.0 * EPS)
print(f"  FD  d(GPP)/d(alpha) [gspot=0] = {fd_val2:+.6e}", flush=True)

ad_val2 = float(jax.jit(jax.grad(forward_gs_leaf))(ONE))
print(f"  JAX d(GPP)/d(alpha) [gspot=0] = {ad_val2:+.6e}", flush=True)
print(f"  Ratio JAX/FD = {ad_val2/fd_val2 if abs(fd_val2)>1e-10 else float('nan'):.4f}", flush=True)

_ml_ctl.gspot_type = _orig_gspot
print(f"  Restored gspot_type={_orig_gspot}", flush=True)

# =============================================================================
# Level 1: Kernel only — d(_gs)/d(g1_val) directly
# =============================================================================
print("\n" + "="*70, flush=True)
print("Level 1: vmapped kernel only — d(sum_gs)/d(g1_val)", flush=True)
print("="*70, flush=True)

# Extract current mlcanopy_inst arrays as they'd be seen by the kernel
# (after SolarRadiation+CanopyNitrogenProfile, but those haven't run here,
#  so we use the warmup state)
from multilayer_canopy.MLLeafPhotosynthesisMod import _fth25_py, _fth25

_acclim = _ml_ctl.acclim_type
_pft_np = np.asarray(_patch.itype)
from clm_src_main.pftconMod import pftcon as _main_pftcon
_c3psn_arr = np.asarray(_main_pftcon.c3psn)

pft_p = int(_pft_np[_p])
c3psn_val = float(_c3psn_arr[pft_p])
is_c3 = round(c3psn_val) == 1

# Temperature response constants
from multilayer_canopy.MLclm_varcon import (
    vcmaxha_noacclim, vcmaxhd_noacclim, vcmaxse_noacclim,
    jmaxha_noacclim, jmaxhd_noacclim, jmaxse_noacclim,
    vcmaxha_acclim, vcmaxhd_acclim, jmaxha_acclim, jmaxhd_acclim,
    rdha, rdhd, rdse,
)
from clm_src_main.clm_varcon import tfrz

rdc = _fth25_py(rdhd, rdse)
_o2ref_cache_key = float(mlcanopy_inst.o2ref_forcing[_p])

_sl = slice(1, _ncan + 1)

if _acclim == 0:
    vcmaxse = vcmaxse_noacclim; jmaxse = jmaxse_noacclim
    vcmaxha = vcmaxha_noacclim; vcmaxhd = vcmaxhd_noacclim
    jmaxha  = jmaxha_noacclim;  jmaxhd  = jmaxhd_noacclim
    vcmaxc  = _fth25_py(vcmaxhd, vcmaxse)
    jmaxc   = _fth25_py(jmaxhd,  jmaxse)
    vmapped = _get_vmapped_photo_kernel(
        is_c3=is_c3, c3psn_pft_val=c3psn_val,
        vcmaxha=vcmaxha, vcmaxhd=vcmaxhd, vcmaxse=vcmaxse, vcmaxc=vcmaxc,
        jmaxha=jmaxha,   jmaxhd=jmaxhd,   jmaxse=jmaxse,  jmaxc=jmaxc,
        rdc=rdc, o2ref_p=_o2ref_cache_key,
    )
    def _call_vmapped(g1_val):
        g0_val = _orig_g1_MED[pft_p] * 0.0  # g0 ≈ 0 for Medlyn
        # Actually use correct g0
        g0_val = jnp.asarray(float(_orig_pftcon.g0_MED[pft_p]))
        return vmapped(
            mlcanopy_inst.dpai_profile[_p, _sl],
            mlcanopy_inst.tleaf_leaf[_p, _sl, isun],
            mlcanopy_inst.vcmax25_leaf[_p, _sl, isun],
            mlcanopy_inst.jmax25_leaf[_p, _sl, isun],
            mlcanopy_inst.rd25_leaf[_p, _sl, isun],
            mlcanopy_inst.kp25_leaf[_p, _sl, isun],
            mlcanopy_inst.eair_profile[_p, _sl],
            mlcanopy_inst.apar_leaf[_p, _sl, isun],
            mlcanopy_inst.gbc_leaf[_p, _sl, isun],
            mlcanopy_inst.gbv_leaf[_p, _sl, isun],
            mlcanopy_inst.cair_profile[_p, _sl],
            g0_val,
            g1_val,
        )
else:
    from multilayer_canopy.MLLeafPhotosynthesisMod import tfrz
    ta_c = jnp.clip(mlcanopy_inst.tacclim_forcing[_p] - tfrz, 11.0, 35.0)
    vcmaxse = 668.39 - 1.07 * ta_c
    jmaxse  = 659.70 - 0.75 * ta_c
    vcmaxha = vcmaxha_acclim; vcmaxhd = vcmaxhd_acclim
    jmaxha  = jmaxha_acclim;  jmaxhd  = jmaxhd_acclim
    vcmaxc  = _fth25(vcmaxhd, vcmaxse)
    jmaxc   = _fth25(jmaxhd,  jmaxse)
    vmapped = _get_vmapped_photo_kernel_acclim(
        is_c3=is_c3, c3psn_pft_val=c3psn_val,
        vcmaxha=vcmaxha, vcmaxhd=vcmaxhd,
        jmaxha=jmaxha,   jmaxhd=jmaxhd,
        rdc=rdc, o2ref_p=_o2ref_cache_key,
    )
    def _call_vmapped(g1_val):
        g0_val = jnp.asarray(float(_orig_pftcon.g0_MED[pft_p]))
        return vmapped(
            mlcanopy_inst.dpai_profile[_p, _sl],
            mlcanopy_inst.tleaf_leaf[_p, _sl, isun],
            mlcanopy_inst.vcmax25_leaf[_p, _sl, isun],
            mlcanopy_inst.jmax25_leaf[_p, _sl, isun],
            mlcanopy_inst.rd25_leaf[_p, _sl, isun],
            mlcanopy_inst.kp25_leaf[_p, _sl, isun],
            mlcanopy_inst.eair_profile[_p, _sl],
            mlcanopy_inst.apar_leaf[_p, _sl, isun],
            mlcanopy_inst.gbc_leaf[_p, _sl, isun],
            mlcanopy_inst.gbv_leaf[_p, _sl, isun],
            mlcanopy_inst.cair_profile[_p, _sl],
            g0_val,
            g1_val,
            vcmaxse, vcmaxc, jmaxse, jmaxc,
        )

print(f"  acclim_type={_acclim}, is_c3={is_c3}, pft={pft_p}", flush=True)
print(f"  gspot_type={_ml_ctl.gspot_type}", flush=True)

g1_val_ref = _orig_g1_MED[pft_p]
print(f"  g1_val_ref = {float(g1_val_ref):.4f}", flush=True)

def forward_kernel(g1_val: jnp.ndarray) -> jnp.ndarray:
    """Return sum of gs_leaf (output[10]) over active layers."""
    outs = _call_vmapped(g1_val)
    gs_arr  = outs[10]   # _gs per layer
    agr_arr = outs[15]   # _agross per layer
    dpai_sl = mlcanopy_inst.dpai_profile[_p, _sl]
    active  = dpai_sl > 0.0
    # Return sum of agross weighted by dpai (like GPP proxy)
    return jnp.sum(jnp.where(active, agr_arr * dpai_sl, 0.0))

base_kernel = float(forward_kernel(g1_val_ref))
print(f"  kernel agross*dpai sum = {base_kernel:.6f}", flush=True)

fd_pk = float(forward_kernel(g1_val_ref + g1_val_ref * EPS))
fd_mk = float(forward_kernel(g1_val_ref - g1_val_ref * EPS))
fd_val1 = (fd_pk - fd_mk) / (2.0 * g1_val_ref * EPS)
print(f"  FD  d(sum_agross)/d(g1_val) = {fd_val1:+.6e}", flush=True)

ad_val1 = float(jax.jit(jax.grad(forward_kernel))(g1_val_ref))
# Note: jax.grad gives d(output)/d(input), so this is d/d(g1_val)
print(f"  JAX d(sum_agross)/d(g1_val) = {ad_val1:+.6e}", flush=True)
print(f"  Ratio JAX/FD = {ad_val1/fd_val1 if abs(fd_val1)>1e-10 else float('nan'):.4f}", flush=True)

# Also check d(sum_gs)/d(g1_val)
def forward_kernel_gs(g1_val: jnp.ndarray) -> jnp.ndarray:
    outs = _call_vmapped(g1_val)
    gs_arr  = outs[10]
    dpai_sl = mlcanopy_inst.dpai_profile[_p, _sl]
    active  = dpai_sl > 0.0
    return jnp.sum(jnp.where(active, gs_arr * dpai_sl, 0.0))

fd_gs_p = float(forward_kernel_gs(g1_val_ref + g1_val_ref * EPS))
fd_gs_m = float(forward_kernel_gs(g1_val_ref - g1_val_ref * EPS))
fd_gs_val = (fd_gs_p - fd_gs_m) / (2.0 * g1_val_ref * EPS)
ad_gs_val = float(jax.jit(jax.grad(forward_kernel_gs))(g1_val_ref))
print(f"\n  kernel gs check:", flush=True)
print(f"  FD  d(sum_gs)/d(g1_val) = {fd_gs_val:+.6e}", flush=True)
print(f"  JAX d(sum_gs)/d(g1_val) = {ad_gs_val:+.6e}", flush=True)
print(f"  Ratio JAX/FD = {ad_gs_val/fd_gs_val if abs(fd_gs_val)>1e-10 else float('nan'):.4f}", flush=True)

print("\n=== Summary ===", flush=True)
print(f"  Level 1 (kernel agross): FD={fd_val1:+.4e}  JAX={ad_val1:+.4e}", flush=True)
print(f"  Level 1 (kernel gs):     FD={fd_gs_val:+.4e}  JAX={ad_gs_val:+.4e}", flush=True)
print(f"  Level 2 (gspot=0 GPP):   FD={fd_val2:+.4e}  JAX={ad_val2:+.4e}", flush=True)
print(f"  Level 3 (full LP, GPP):  FD={fd_val3:+.4e}  JAX={ad_val3:+.4e}", flush=True)
print(f"  Level 4 (MLCanopyFlux):  FD={fd_val4:+.4e}  JAX={ad_val4:+.4e}", flush=True)
print("\n=== debug_g1_grad_isolation.py done ===", flush=True)
