"""
debug_g1_fast.py — Fast isolation test for d(GPP)/d(g1_MED) at kernel level.

Tests ONLY the vmapped kernel (Level 1), not the full MLCanopyFluxes pipeline.
This avoids the slow lax.scan JIT compilation.

Expected: both FD and JAX should be positive (increasing g1 → more CO2 uptake).
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

# ── Set Medlyn BEFORE init ──────────────────────────────────────────────────
from multilayer_canopy import MLclm_varctl
MLclm_varctl.gs_type = 0
print(f"gs_type = {MLclm_varctl.gs_type} (Medlyn)", flush=True)

# ── Shared init ──────────────────────────────────────────────────────────────
from diags.expt_init import (
    mlcanopy_inst, grid,
    atm2lnd_inst, wateratm2lndbulk_inst, compute_gpp,
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
g1_val_ref = float(_orig_g1_MED[_pft])
g0_val_ref = float(_orig_pftcon.g0_MED[_pft])
print(f"g1_MED[{_pft}] = {g1_val_ref:.4f}, g0_MED[{_pft}] = {g0_val_ref:.6f}", flush=True)

EPS = 1e-4

# ── Get vmapped kernel ────────────────────────────────────────────────────────
from multilayer_canopy.MLLeafPhotosynthesisMod import (
    _get_vmapped_photo_kernel_acclim,
    _get_vmapped_photo_kernel,
    _fth25_py, _fth25,
)
from multilayer_canopy.MLclm_varcon import (
    vcmaxha_noacclim, vcmaxhd_noacclim, vcmaxse_noacclim,
    jmaxha_noacclim, jmaxhd_noacclim, jmaxse_noacclim,
    vcmaxha_acclim, vcmaxhd_acclim, jmaxha_acclim, jmaxhd_acclim,
    rdha, rdhd, rdse,
)
from clm_src_main.clm_varcon import tfrz

_acclim = MLclm_varctl.acclim_type
from clm_src_main.pftconMod import pftcon as _main_pftcon
_c3psn_arr = np.asarray(_main_pftcon.c3psn)
c3psn_val = float(_c3psn_arr[_pft])
is_c3 = round(c3psn_val) == 1
rdc = _fth25_py(rdhd, rdse)
_o2ref_cache_key = float(mlcanopy_inst.o2ref_forcing[_p])

_sl = slice(1, _ncan + 1)
print(f"acclim_type={_acclim}, is_c3={is_c3}", flush=True)

# Extract current state arrays (warmup state, after LeafPhotosynthesis ran)
dpai_arr   = mlcanopy_inst.dpai_profile[_p, _sl]
tleaf_arr  = mlcanopy_inst.tleaf_leaf[_p, _sl, isun]
vcmax25_arr = mlcanopy_inst.vcmax25_leaf[_p, _sl, isun]
jmax25_arr  = mlcanopy_inst.jmax25_leaf[_p, _sl, isun]
rd25_arr   = mlcanopy_inst.rd25_leaf[_p, _sl, isun]
kp25_arr   = mlcanopy_inst.kp25_leaf[_p, _sl, isun]
eair_arr   = mlcanopy_inst.eair_profile[_p, _sl]
apar_arr   = mlcanopy_inst.apar_leaf[_p, _sl, isun]
gbc_arr    = mlcanopy_inst.gbc_leaf[_p, _sl, isun]
gbv_arr    = mlcanopy_inst.gbv_leaf[_p, _sl, isun]
cair_arr   = mlcanopy_inst.cair_profile[_p, _sl]

# Print key forcing values for diagnosis
active = np.asarray(dpai_arr) > 0
print(f"Active layers: {np.sum(active)}/{_ncan}", flush=True)
print(f"apar[active] range: {float(jnp.min(apar_arr[active])):.2f}..{float(jnp.max(apar_arr[active])):.2f}", flush=True)
print(f"cair[active] range: {float(jnp.min(cair_arr[active])):.2f}..{float(jnp.max(cair_arr[active])):.2f}", flush=True)
print(f"gbc[active] range: {float(jnp.min(gbc_arr[active])):.4f}..{float(jnp.max(gbc_arr[active])):.4f}", flush=True)

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
    def _call_vmapped(g1_val_x, g0_val_x):
        return vmapped(
            dpai_arr, tleaf_arr, vcmax25_arr, jmax25_arr, rd25_arr, kp25_arr,
            eair_arr, apar_arr, gbc_arr, gbv_arr, cair_arr,
            g0_val_x, g1_val_x,
        )
else:
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
    def _call_vmapped(g1_val_x, g0_val_x):
        return vmapped(
            dpai_arr, tleaf_arr, vcmax25_arr, jmax25_arr, rd25_arr, kp25_arr,
            eair_arr, apar_arr, gbc_arr, gbv_arr, cair_arr,
            g0_val_x, g1_val_x,
            vcmaxse, vcmaxc, jmaxse, jmaxc,
        )

g0_jax = jnp.float64(g0_val_ref)
g1_jax = jnp.float64(g1_val_ref)

# Run baseline
outs_base = _call_vmapped(g1_jax, g0_jax)
gs_base  = np.asarray(outs_base[10])
agr_base = np.asarray(outs_base[15])
ci_base  = np.asarray(outs_base[11])
anet_base = np.asarray(outs_base[16])

print(f"\nKernel baseline values (active layers):", flush=True)
active_np = np.asarray(dpai_arr) > 0
print(f"  gs   range: {float(np.min(gs_base[active_np])):.4f}..{float(np.max(gs_base[active_np])):.4f}", flush=True)
print(f"  ci   range: {float(np.min(ci_base[active_np])):.1f}..{float(np.max(ci_base[active_np])):.1f}", flush=True)
print(f"  anet range: {float(np.min(anet_base[active_np])):.2f}..{float(np.max(anet_base[active_np])):.2f}", flush=True)
print(f"  agross range: {float(np.min(agr_base[active_np])):.2f}..{float(np.max(agr_base[active_np])):.2f}", flush=True)

# ── Test d(agross_sum)/d(g1_val) ─────────────────────────────────────────────
print("\n=== Kernel: d(sum_agross*dpai)/d(g1_val) ===", flush=True)

def fw_agross(g1_val_x: jnp.ndarray) -> jnp.ndarray:
    outs = _call_vmapped(g1_val_x, g0_jax)
    agr  = outs[15]
    return jnp.sum(jnp.where(active_np, agr * np.asarray(dpai_arr), 0.0))

agr_base_sum = float(fw_agross(g1_jax))
agr_plus  = float(fw_agross(g1_jax + g1_jax * EPS))
agr_minus = float(fw_agross(g1_jax - g1_jax * EPS))
fd_agr = (agr_plus - agr_minus) / (2.0 * float(g1_jax) * EPS)
print(f"  agross*dpai sum = {agr_base_sum:.6f}", flush=True)
print(f"  FD  d(agross_sum)/d(g1) = {fd_agr:+.6e}", flush=True)

t0 = time.time()
ad_agr = float(jax.jit(jax.grad(fw_agross))(g1_jax))
print(f"  JAX d(agross_sum)/d(g1) = {ad_agr:+.6e}  ({time.time()-t0:.1f}s)", flush=True)
print(f"  Ratio JAX/FD = {ad_agr/fd_agr if abs(fd_agr)>1e-10 else float('nan'):.4f}", flush=True)

# ── Test d(gs_sum)/d(g1_val) ─────────────────────────────────────────────────
print("\n=== Kernel: d(sum_gs*dpai)/d(g1_val) ===", flush=True)

def fw_gs(g1_val_x: jnp.ndarray) -> jnp.ndarray:
    outs = _call_vmapped(g1_val_x, g0_jax)
    gs   = outs[10]
    return jnp.sum(jnp.where(active_np, gs * np.asarray(dpai_arr), 0.0))

gs_base_sum = float(fw_gs(g1_jax))
gs_plus  = float(fw_gs(g1_jax + g1_jax * EPS))
gs_minus = float(fw_gs(g1_jax - g1_jax * EPS))
fd_gs = (gs_plus - gs_minus) / (2.0 * float(g1_jax) * EPS)
print(f"  gs*dpai sum = {gs_base_sum:.6f}", flush=True)
print(f"  FD  d(gs_sum)/d(g1) = {fd_gs:+.6e}", flush=True)

t0 = time.time()
ad_gs = float(jax.jit(jax.grad(fw_gs))(g1_jax))
print(f"  JAX d(gs_sum)/d(g1) = {ad_gs:+.6e}  ({time.time()-t0:.1f}s)", flush=True)
print(f"  Ratio JAX/FD = {ad_gs/fd_gs if abs(fd_gs)>1e-10 else float('nan'):.4f}", flush=True)

# ── Test d(ci_sum)/d(g1_val) — IFT sanity check ──────────────────────────────
print("\n=== Kernel: d(sum_ci*dpai)/d(g1_val) ===", flush=True)

def fw_ci(g1_val_x: jnp.ndarray) -> jnp.ndarray:
    outs = _call_vmapped(g1_val_x, g0_jax)
    ci   = outs[11]
    return jnp.sum(jnp.where(active_np, ci * np.asarray(dpai_arr), 0.0))

ci_base_sum = float(fw_ci(g1_jax))
ci_plus  = float(fw_ci(g1_jax + g1_jax * EPS))
ci_minus = float(fw_ci(g1_jax - g1_jax * EPS))
fd_ci = (ci_plus - ci_minus) / (2.0 * float(g1_jax) * EPS)
print(f"  ci*dpai sum = {ci_base_sum:.4f}", flush=True)
print(f"  FD  d(ci_sum)/d(g1) = {fd_ci:+.6e}", flush=True)

t0 = time.time()
ad_ci = float(jax.jit(jax.grad(fw_ci))(g1_jax))
print(f"  JAX d(ci_sum)/d(g1) = {ad_ci:+.6e}  ({time.time()-t0:.1f}s)", flush=True)
print(f"  Ratio JAX/FD = {ad_ci/fd_ci if abs(fd_ci)>1e-10 else float('nan'):.4f}", flush=True)

# ── LeafPhotosynthesis only (both loops) ─────────────────────────────────────
print("\n=== LeafPhotosynthesis (sun+sha, both loops): d(GPP)/d(alpha) ===", flush=True)
from multilayer_canopy.MLLeafPhotosynthesisMod import LeafPhotosynthesis
from clm_src_main import clm_driver as _clm_driver_mod
_filt = _clm_driver_mod.filter
_num = int(_filt.num_exposedvegp)
_filt_list = [int(_filt.exposedvegp[i]) for i in range(_num)]
_o2ref_py = float(mlcanopy_inst.o2ref_forcing[_p])

def fw_lp_gpp(alpha: jnp.ndarray) -> jnp.ndarray:
    g1_MED_jax = alpha * _orig_g1_MED
    inst_sun = LeafPhotosynthesis(
        _num, _filt_list, isun, mlcanopy_inst,
        grid=grid, _o2ref_py=_o2ref_py, g1_MED_jax=g1_MED_jax,
    )
    inst_both = LeafPhotosynthesis(
        _num, _filt_list, isha, inst_sun,
        grid=grid, _o2ref_py=_o2ref_py, g1_MED_jax=g1_MED_jax,
    )
    return compute_gpp(inst_both, _p, _ncan)

ONE = jnp.float64(1.0)
gpp_lp = float(fw_lp_gpp(ONE))
print(f"  GPP baseline = {gpp_lp:.6f}", flush=True)
print(f"  Scanning FD over multiple epsilons:", flush=True)
for eps_v in [0.1, 0.01, 0.001, 1e-4, 1e-5]:
    fd_p = float(fw_lp_gpp(jnp.float64(1.0 + eps_v)))
    fd_m = float(fw_lp_gpp(jnp.float64(1.0 - eps_v)))
    fd_v = (fd_p - fd_m) / (2.0 * eps_v)
    print(f"    eps={eps_v:.0e}: f(1+e)={fd_p:.4f}, f(1-e)={fd_m:.4f}, FD={fd_v:+.4e}", flush=True)
t0 = time.time()
ad_lp = float(jax.jit(jax.grad(fw_lp_gpp))(ONE))
print(f"  JAX d(GPP)/d(alpha) [LP only] = {ad_lp:+.6e}  ({time.time()-t0:.1f}s)", flush=True)
# Use eps=0.1 as the stable FD estimate
fd_stable = (float(fw_lp_gpp(jnp.float64(1.1))) - float(fw_lp_gpp(jnp.float64(0.9)))) / 0.2
print(f"  FD (eps=0.1, stable) = {fd_stable:+.6e}", flush=True)
print(f"  Ratio JAX/FD(eps=0.1) = {ad_lp/fd_stable if abs(fd_stable)>1e-10 else float('nan'):.4f}", flush=True)


# ── Deeper diagnosis: split isun vs isha GPP contributions ───────────────────
print("\n=== DIAGNOSIS: isun only, isha only, both calls separately ===", flush=True)

def fw_isun_only(alpha: jnp.ndarray) -> jnp.ndarray:
    """Only isun GPP after isun LeafPhotosynthesis call."""
    g1_MED_jax = alpha * _orig_g1_MED
    inst_sun = LeafPhotosynthesis(
        _num, _filt_list, isun, mlcanopy_inst,
        grid=grid, _o2ref_py=_o2ref_py, g1_MED_jax=g1_MED_jax,
    )
    # Read isun agross directly (second loop output)
    agross_sun = inst_sun.agross_leaf[_p, 1:_ncan + 1, isun]
    dpai = inst_sun.dpai_profile[_p, 1:_ncan + 1]
    fracsun = inst_sun.fracsun_profile[_p, 1:_ncan + 1]
    return jnp.sum(agross_sun * fracsun * dpai)

def fw_isha_only(alpha: jnp.ndarray) -> jnp.ndarray:
    """Only isha GPP after isha LeafPhotosynthesis call."""
    g1_MED_jax = alpha * _orig_g1_MED
    inst_sun = LeafPhotosynthesis(
        _num, _filt_list, isun, mlcanopy_inst,
        grid=grid, _o2ref_py=_o2ref_py, g1_MED_jax=g1_MED_jax,
    )
    inst_both = LeafPhotosynthesis(
        _num, _filt_list, isha, inst_sun,
        grid=grid, _o2ref_py=_o2ref_py, g1_MED_jax=g1_MED_jax,
    )
    # Read isha agross
    agross_sha = inst_both.agross_leaf[_p, 1:_ncan + 1, isha]
    dpai = inst_both.dpai_profile[_p, 1:_ncan + 1]
    fracsun = inst_both.fracsun_profile[_p, 1:_ncan + 1]
    return jnp.sum(agross_sha * (1.0 - fracsun) * dpai)

def fw_isun_passthrough(alpha: jnp.ndarray) -> jnp.ndarray:
    """isun agross read from inst_both (after isha call passes it through)."""
    g1_MED_jax = alpha * _orig_g1_MED
    inst_sun = LeafPhotosynthesis(
        _num, _filt_list, isun, mlcanopy_inst,
        grid=grid, _o2ref_py=_o2ref_py, g1_MED_jax=g1_MED_jax,
    )
    inst_both = LeafPhotosynthesis(
        _num, _filt_list, isha, inst_sun,
        grid=grid, _o2ref_py=_o2ref_py, g1_MED_jax=g1_MED_jax,
    )
    # Read isun agross from inst_both (should be same as inst_sun's)
    agross_sun = inst_both.agross_leaf[_p, 1:_ncan + 1, isun]
    dpai = inst_both.dpai_profile[_p, 1:_ncan + 1]
    fracsun = inst_both.fracsun_profile[_p, 1:_ncan + 1]
    return jnp.sum(agross_sun * fracsun * dpai)

# Check baseline values
inst_sun_base = LeafPhotosynthesis(_num, _filt_list, isun, mlcanopy_inst,
    grid=grid, _o2ref_py=_o2ref_py, g1_MED_jax=jnp.asarray(_orig_pftcon.g1_MED))
inst_both_base = LeafPhotosynthesis(_num, _filt_list, isha, inst_sun_base,
    grid=grid, _o2ref_py=_o2ref_py, g1_MED_jax=jnp.asarray(_orig_pftcon.g1_MED))
_sl = slice(1, _ncan + 1)
active_np2 = np.asarray(inst_both_base.dpai_profile[_p, _sl]) > 0
gs_sha = np.asarray(inst_both_base.gs_leaf[_p, _sl, isha])
agross_sha_np = np.asarray(inst_both_base.agross_leaf[_p, _sl, isha])
agross_sun_np = np.asarray(inst_both_base.agross_leaf[_p, _sl, isun])
fracsun_np = np.asarray(inst_both_base.fracsun_profile[_p, _sl])
dpai_np = np.asarray(inst_both_base.dpai_profile[_p, _sl])
print(f"  gs_sha[active] range: {float(np.min(gs_sha[active_np2])):.4f}..{float(np.max(gs_sha[active_np2])):.4f}", flush=True)
print(f"  agross_sha[active] range: {float(np.min(agross_sha_np[active_np2])):.4f}..{float(np.max(agross_sha_np[active_np2])):.4f}", flush=True)
print(f"  agross_sun[active] range: {float(np.min(agross_sun_np[active_np2])):.4f}..{float(np.max(agross_sun_np[active_np2])):.4f}", flush=True)
print(f"  fracsun[active] range: {float(np.min(fracsun_np[active_np2])):.4f}..{float(np.max(fracsun_np[active_np2])):.4f}", flush=True)
gpp_isun_base = float(np.sum(agross_sun_np * fracsun_np * dpai_np))
gpp_isha_base = float(np.sum(agross_sha_np * (1.0 - fracsun_np) * dpai_np))
print(f"  GPP isun portion = {gpp_isun_base:.4f}, isha portion = {gpp_isha_base:.4f}", flush=True)

# Debug: check fw_isha_only shape around alpha=1
print(f"\n  Scanning fw_isha_only over alpha range:", flush=True)
for eps_v in [0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6]:
    val_p = float(fw_isha_only(jnp.float64(1.0 + eps_v)))
    val_m = float(fw_isha_only(jnp.float64(1.0 - eps_v)))
    fd_v = (val_p - val_m) / (2.0 * eps_v)
    print(f"    eps={eps_v:.0e}: f(1+e)={val_p:.4f}, f(1-e)={val_m:.4f}, FD={fd_v:+.4e}", flush=True)

print(f"\n  Scanning fw_isun_only over alpha range:", flush=True)
for eps_v in [0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6]:
    val_p = float(fw_isun_only(jnp.float64(1.0 + eps_v)))
    val_m = float(fw_isun_only(jnp.float64(1.0 - eps_v)))
    fd_v = (val_p - val_m) / (2.0 * eps_v)
    print(f"    eps={eps_v:.0e}: f(1+e)={val_p:.4f}, f(1-e)={val_m:.4f}, FD={fd_v:+.4e}", flush=True)

# Check how many shade layers have anet > 0 (active in ci-solver sense)
print(f"\n  Checking anet for shade layers:", flush=True)
anet_sha_np = np.asarray(inst_both_base.anet_leaf[_p, _sl, isha])
print(f"  anet_sha[active] range: {float(np.min(anet_sha_np[active_np2])):.4f}..{float(np.max(anet_sha_np[active_np2])):.4f}", flush=True)
n_pos = int(np.sum(anet_sha_np[active_np2] > 0))
print(f"  anet_sha > 0 (ci-solver active): {n_pos}/{int(np.sum(active_np2))} layers", flush=True)
anet_sun_np = np.asarray(inst_both_base.anet_leaf[_p, _sl, isun])
n_pos_sun = int(np.sum(anet_sun_np[active_np2] > 0))
print(f"  anet_sun > 0: {n_pos_sun}/{int(np.sum(active_np2))} layers", flush=True)

def fw_isha_direct(alpha: jnp.ndarray) -> jnp.ndarray:
    """isha GPP using mlcanopy_inst directly (no isun call first)."""
    g1_MED_jax = alpha * _orig_g1_MED
    # Call isha directly on the warmup state
    inst_isha = LeafPhotosynthesis(
        _num, _filt_list, isha, mlcanopy_inst,
        grid=grid, _o2ref_py=_o2ref_py, g1_MED_jax=g1_MED_jax,
    )
    agross_sha = inst_isha.agross_leaf[_p, 1:_ncan + 1, isha]
    dpai = inst_isha.dpai_profile[_p, 1:_ncan + 1]
    fracsun = inst_isha.fracsun_profile[_p, 1:_ncan + 1]
    return jnp.sum(agross_sha * (1.0 - fracsun) * dpai)

print(f"\n  Scanning fw_isha_direct (no isun call) over alpha range:", flush=True)
for eps_v in [0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6]:
    val_p = float(fw_isha_direct(jnp.float64(1.0 + eps_v)))
    val_m = float(fw_isha_direct(jnp.float64(1.0 - eps_v)))
    fd_v = (val_p - val_m) / (2.0 * eps_v)
    print(f"    eps={eps_v:.0e}: f(1+e)={val_p:.4f}, f(1-e)={val_m:.4f}, FD={fd_v:+.4e}", flush=True)
ad_v_direct = float(jax.jit(jax.grad(fw_isha_direct))(ONE))
print(f"  JAX d(isha_direct)/d(alpha) = {ad_v_direct:+.4e}", flush=True)

for name, fw_fn in [
    ("isun only (after isun call)",    fw_isun_only),
    ("isha only (after both calls)",   fw_isha_only),
    ("isun passthrough (from both)",   fw_isun_passthrough),
]:
    base_v = float(fw_fn(ONE))
    fd_p   = float(fw_fn(ONE + EPS))
    fd_m   = float(fw_fn(ONE - EPS))
    fd_v   = (fd_p - fd_m) / (2.0 * EPS)
    ad_v   = float(jax.jit(jax.grad(fw_fn))(ONE))
    ratio  = ad_v / fd_v if abs(fd_v) > 1e-10 else float('nan')
    print(f"  [{name}]  FD={fd_v:+.4e}  JAX={ad_v:+.4e}  ratio={ratio:.4f}", flush=True)

print("\n=== debug_g1_fast.py done ===", flush=True)
