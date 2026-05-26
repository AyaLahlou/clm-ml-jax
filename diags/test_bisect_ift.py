"""
Standalone test of _bisect_gs_ift vs _bisect_gs_jax gradient correctness.

Uses real warmed-up model state for a single canopy layer.
Runs WITHOUT needing a full MLCanopyFluxes forward pass — just tests
the bisection root-finder itself.

Usage:
    cd /burg-archive/home/al4385/clm-ml-jax
    CLM_ML_NO_CHECKPOINT=1 python diags/test_bisect_ift.py
"""
from __future__ import annotations
import os, sys
os.environ['CLM_ML_NO_CHECKPOINT'] = '1'

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diags.expt_init import mlcanopy_inst, grid, jax, jnp
from multilayer_canopy.MLclm_varpar import isun
from multilayer_canopy.MLclm_varcon import (
    vcmaxha_noacclim as vcmaxha, vcmaxhd_noacclim as vcmaxhd,
    vcmaxse_noacclim as vcmaxse, jmaxha_noacclim as jmaxha,
    jmaxhd_noacclim as jmaxhd, jmaxse_noacclim as jmaxse,
    rdhd, rdse, kc25, ko25, cp25, kcha, koha, cpha, rdha,
)
from clm_src_main.clm_varcon import tfrz
from multilayer_canopy.MLclm_varcon import rgas, dh2o_to_dco2, vpd_min_MED
from multilayer_canopy.MLclm_varctl import gs_type, colim_type
from multilayer_canopy.MLLeafPhotosynthesisMod import (
    _StomataEfficiencyJax, _bisect_gs_jax, _bisect_gs_ift, _fth25_py,
    _ft, _fth,
)
from multilayer_canopy.MLpftconMod import MLpftcon
from clm_src_main.PatchType import patch
from multilayer_canopy.MLMathToolsMod import quadratic
from multilayer_canopy.MLWaterVaporMod import SatVap_py
import numpy as np

_p = grid.p

EPS = 1e-4

print("\n" + "="*60)
print("BISECT IFT GRADIENT TEST")
print(f"  gs_type={gs_type}, colim_type={colim_type}")
print("="*60)

if gs_type != 2:
    print("  gs_type != 2: WUE not active, skipping.")
    sys.exit(0)

# ── Extract single-layer state ────────────────────────────────────────────────
ic = 5
il = isun

tleaf   = float(mlcanopy_inst.tleaf_leaf[_p, ic, il])
vcmax25 = float(mlcanopy_inst.vcmax25_leaf[_p, ic, il])
jmax25  = float(mlcanopy_inst.jmax25_leaf[_p, ic, il])
rd25    = float(mlcanopy_inst.rd25_leaf[_p, ic, il])
kp25    = float(mlcanopy_inst.kp25_leaf[_p, ic, il])
apar    = float(mlcanopy_inst.apar_leaf[_p, ic, il])
gbc     = float(mlcanopy_inst.gbc_leaf[_p, ic, il])
gbv     = float(mlcanopy_inst.gbv_leaf[_p, ic, il])
cair    = float(mlcanopy_inst.cair_profile[_p, ic])
dpai    = float(mlcanopy_inst.dpai_profile[_p, ic])
eair    = float(mlcanopy_inst.eair_profile[_p, ic])
o2ref   = float(mlcanopy_inst.o2ref_forcing[_p])
pref    = float(mlcanopy_inst.pref_forcing[_p])

pft_idx   = int(patch.itype[_p])
iota_pft  = float(np.asarray(MLpftcon.iota_SPA)[pft_idx])
gsmin_pft = float(np.asarray(MLpftcon.gsmin_SPA)[pft_idx])

print(f"\nLayer ic={ic}, il=isun")
print(f"  tleaf={tleaf:.2f}K, vcmax25={vcmax25:.2f}, apar={apar:.2f}")
print(f"  iota_pft={iota_pft:.3f}, gsmin_pft={gsmin_pft:.4f}")

# ── Build se_kwargs for given apar ────────────────────────────────────────────
from multilayer_canopy.MLclm_varcon import (
    colim_c3a, colim_c4a, colim_c4b, qe_c4, theta_j, phi_psII,
)

def make_ci_kwargs_and_je(apar_val):
    """Build ci_kwargs (with je computed from apar_val)."""
    t25 = tfrz + 25.0
    tl = jnp.float64(tleaf)
    kc_v  = kc25 * _ft(tl, kcha)
    ko_v  = ko25 * _ft(tl, koha)
    cp_v  = cp25 * _ft(tl, cpha)
    vcmax_v = (jnp.float64(vcmax25)
               * _ft(tl, vcmaxha)
               * _fth(tl, vcmaxhd, vcmaxse, _fth25_py(vcmaxhd, vcmaxse)))
    jmax_v  = (jnp.float64(jmax25)
               * _ft(tl, jmaxha)
               * _fth(tl, jmaxhd, jmaxse, _fth25_py(jmaxhd, jmaxse)))
    rd_v    = (jnp.float64(rd25)
               * _ft(tl, rdha)
               * _fth(tl, rdhd, rdse, _fth25_py(rdhd, rdse)))
    kp_v    = jnp.float64(0.0)

    qabs = 0.5 * phi_psII * apar_val
    bq_j = -(qabs + jmax_v)
    cq_j = qabs * jmax_v
    r1j, r2j = quadratic(theta_j, bq_j, cq_j)
    je_v = jnp.minimum(r1j, r2j)

    lesat_v = jnp.float64(SatVap_py(tleaf))
    ceair_v = jnp.minimum(jnp.float64(eair), lesat_v)

    ci_kw = dict(
        is_c3=True, dpai_ic=jnp.float64(dpai),
        gbc_ic=jnp.float64(gbc), cair_ic=jnp.float64(cair),
        vcmax_ic=vcmax_v, je_ic=je_v, kp_ic=kp_v, rd_ic=rd_v,
        kc_ic=kc_v, ko_ic=ko_v, cp_ic=cp_v,
        o2ref_p=jnp.float64(o2ref), apar_ic=apar_val,
        c3psn_pft_val=1.0,
    )
    se_kw = dict(
        iota=iota_pft, pref_p=jnp.float64(pref),
        eair_ic=jnp.float64(eair), gbv_ic=jnp.float64(gbv),
        lesat_ic=lesat_v, **ci_kw,
    )
    return ci_kw, se_kw

apar_j = jnp.float64(apar)
ci_kw0, se_kw0 = make_ci_kwargs_and_je(apar_j)

# ── Test gs_opt from both solvers ─────────────────────────────────────────────
gs_old = float(_bisect_gs_jax(gsmin_pft, 2.0, se_kw0))
gs_new = float(_bisect_gs_ift(gsmin_pft, 2.0, se_kw0))
print(f"\ngs_old (bisect_jax) = {gs_old:.6f}")
print(f"gs_new (bisect_ift) = {gs_new:.6f}")
print(f"  Forward values match: {abs(gs_old - gs_new) < 1e-8}")

# ── Compare d(gs*)/d(apar) ────────────────────────────────────────────────────
def gs_from_apar_old(a):
    _, se_kw = make_ci_kwargs_and_je(a)
    return _bisect_gs_jax(gsmin_pft, 2.0, se_kw)

def gs_from_apar_new(a):
    _, se_kw = make_ci_kwargs_and_je(a)
    return _bisect_gs_ift(gsmin_pft, 2.0, se_kw)

print("\n--- d(gs*)/d(apar) comparison ---")

import time
t0 = time.time()
old_jax = float(jax.grad(gs_from_apar_old)(apar_j))
print(f"bisect_jax JAX grad: {old_jax:.4e}  ({time.time()-t0:.1f}s)")

t0 = time.time()
new_jax = float(jax.grad(gs_from_apar_new)(apar_j))
print(f"bisect_ift JAX grad: {new_jax:.4e}  ({time.time()-t0:.1f}s)")

a_plus  = float(gs_from_apar_new(apar_j * (1 + EPS)))
a_minus = float(gs_from_apar_new(apar_j * (1 - EPS)))
fd_grad = (a_plus - a_minus) / (2 * EPS * float(apar_j))
print(f"FD grad:             {fd_grad:.4e}")

rel_old = abs(old_jax - fd_grad) / (abs(fd_grad) + 1e-30)
rel_new = abs(new_jax - fd_grad) / (abs(fd_grad) + 1e-30)
print(f"\nbisect_jax rel err: {rel_old:.3e}  {'PASS' if rel_old < 0.01 else 'FAIL'}")
print(f"bisect_ift rel err: {rel_new:.3e}  {'PASS' if rel_new < 0.01 else 'FAIL'}")

# ── IFT reference ─────────────────────────────────────────────────────────────
gs_opt_j = jnp.float64(gs_new)
df_dgs   = float(jax.grad(lambda gs: _StomataEfficiencyJax(gs, **se_kw0))(gs_opt_j))
df_dapar = float(jax.grad(lambda a: _StomataEfficiencyJax(gs_opt_j, **make_ci_kwargs_and_je(a)[1]))(apar_j))
ift_grad = -df_dapar / (df_dgs if abs(df_dgs) > 1e-15 else 1e-15)
print(f"IFT reference:       {ift_grad:.4e}  (should match bisect_ift & FD)")

print("\n=== test_bisect_ift.py complete ===", flush=True)
