"""
Quick ci-solver gradient test.

Tests whether d(ci_root)/d(apar) from jax.grad matches finite differences.
This directly tests whether the secant-method ci solver is differentiable.

Usage:
    cd /burg-archive/home/al4385/clm-ml-jax
    CLM_ML_NO_CHECKPOINT=1 python diags/test_ci_solver_grad.py
"""
from __future__ import annotations
import os, sys
os.environ['CLM_ML_NO_CHECKPOINT'] = '1'
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

print('JAX devices:', jax.devices())

# Load the ci solver
from multilayer_canopy.MLLeafPhotosynthesisMod import _ci_solver_scan, _CiFuncPure_jax

# Load real state from warmed-up model
from diags.expt_init import mlcanopy_inst, grid, jax, jnp
from multilayer_canopy.MLclm_varpar import isun, isha
from multilayer_canopy.MLclm_varcon import (
    vcmaxha_noacclim as vcmaxha, vcmaxhd_noacclim as vcmaxhd,
    vcmaxse_noacclim as vcmaxse, jmaxha_noacclim as jmaxha,
    jmaxhd_noacclim as jmaxhd, jmaxse_noacclim as jmaxse,
    rdhd, rdse, kc25, ko25, cp25, kcha, koha, cpha, rdha,
)
from clm_src_main.clm_varcon import tfrz
from multilayer_canopy.MLclm_varcon import rgas
from multilayer_canopy.MLLeafPhotosynthesisMod import _fth25_py
from multilayer_canopy.MLpftconMod import MLpftcon
from clm_src_main.PatchType import patch
from multilayer_canopy.MLMathToolsMod import quadratic
from multilayer_canopy.MLWaterVaporMod import SatVap_py
from multilayer_canopy.MLclm_varcon import dh2o_to_dco2, vpd_min_MED
from multilayer_canopy.MLclm_varctl import gs_type, colim_type

_p = grid.p
_ncan = grid.ncan
EPS = 1e-4

print(f"gs_type={gs_type}, colim_type={colim_type}, ncan={_ncan}")
print(f"isun={isun}, isha={isha}")

# Test for a single canopy layer (layer 5, sunlit)
ic = 5
il = isun

# Extract real values from warmed-up state
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

print(f"\nLayer ic={ic}, il=isun")
print(f"  tleaf={tleaf:.2f}K, vcmax25={vcmax25:.2f}, apar={apar:.2f}")
print(f"  gbc={gbc:.4f}, gbv={gbv:.4f}, cair={cair:.2f}, o2ref={o2ref:.2f}")

pft_idx = int(patch.itype[_p])
g0_p = float(np.asarray(MLpftcon.g0_MED)[pft_idx])
g1_p = float(np.asarray(MLpftcon.g1_MED)[pft_idx])

# Compute temperature responses
tleaf_j = jnp.float64(tleaf)
vcmax25_j = jnp.float64(vcmax25)
jmax25_j = jnp.float64(jmax25)
rd25_j = jnp.float64(rd25)
kp25_j = jnp.float64(kp25)

def _exp_temp(ha, val25, tl):
    t25 = tfrz + 25.0
    return val25 * jnp.exp(ha * (tl - t25) / (t25 * rgas * tl))

kc_ic = _exp_temp(kcha, kc25, tleaf_j)
ko_ic = _exp_temp(koha, ko25, tleaf_j)
cp_ic = _exp_temp(cpha, cp25, tleaf_j)
vcmax_ic = vcmax25_j * jnp.exp(vcmaxha * (tleaf_j - (tfrz+25.0)) / ((tfrz+25.0) * rgas * tleaf_j)) / (1.0 + jnp.exp((-vcmaxhd + vcmaxse * tleaf_j) / (rgas * tleaf_j))) / _fth25_py(vcmaxhd, vcmaxse)
jmax_ic = jmax25_j * jnp.exp(jmaxha * (tleaf_j - (tfrz+25.0)) / ((tfrz+25.0) * rgas * tleaf_j)) / (1.0 + jnp.exp((-jmaxhd + jmaxse * tleaf_j) / (rgas * tleaf_j))) / _fth25_py(jmaxhd, jmaxse)
rd_ic = rd25_j * jnp.exp(rdha * (tleaf_j - (tfrz+25.0)) / ((tfrz+25.0) * rgas * tleaf_j)) / (1.0 + jnp.exp((-rdhd + rdse * tleaf_j) / (rgas * tleaf_j))) / _fth25_py(rdhd, rdse)
kp_ic = jnp.float64(0.0)

# Electron transport (je) — compute with real apar
def compute_je(apar_val):
    t25 = tfrz + 25.0
    ta_c = float(tleaf) - tfrz
    phiPSII = max(0.0, 0.352 + 0.022 * ta_c - 3.4e-4 * ta_c**2)
    phiPSII = min(phiPSII, 1.0)
    theta_j = jnp.float64(0.9)
    phi_val = phiPSII * apar_val
    r1, r2 = quadratic(theta_j, -(phi_val + jmax_ic), phi_val * jmax_ic)
    return jnp.minimum(r1, r2)

# Saturation vapor
lesat_ic = jnp.float64(SatVap_py(float(tleaf_j)))
ceair_ic = jnp.float64(eair)

# Build func_kwargs
def build_fkw(apar_val):
    je = compute_je(apar_val)
    return dict(
        apar_ic=apar_val, kc_ic=kc_ic, ko_ic=ko_ic, cp_ic=cp_ic,
        vcmax_ic=vcmax_ic, jmax_ic=jmax_ic, rd_ic=rd_ic, kp_ic=kp_ic,
        je_ic=je, dpai_ic=float(dpai),
        gbc_ic=jnp.float64(gbc), gbv_ic=jnp.float64(gbv),
        cair_ic=jnp.float64(cair), lesat_ic=lesat_ic, ceair_ic=ceair_ic,
        c3psn_pft_val=1.0, is_c3=True, g0_p=float(g0_p), g1_p=float(g1_p),
        o2ref_p=float(o2ref),
    )

# Baseline ci root
ci0 = jnp.float64(0.7 * cair)
ci1 = jnp.float64(0.99 * cair)
apar_j = jnp.float64(apar)

fkw_base = build_fkw(apar_j)
ci_root = _ci_solver_scan(ci0, ci1, fkw_base)
F_at_root = float(_CiFuncPure_jax(ci_root, **fkw_base))
print(f"\nci_root = {float(ci_root):.4f} umol/mol")
print(f"F(ci_root) = {F_at_root:.2e}  (should be ~0 at convergence)")

# d(ci_root)/d(apar) comparison
def get_ci_root(apar_val):
    fkw = build_fkw(apar_val)
    return _ci_solver_scan(ci0, ci1, fkw)

print(f"\n--- d(ci_root)/d(apar) ---")
jax_grad_ci = float(jax.grad(get_ci_root)(apar_j))
f_plus  = float(get_ci_root(apar_j * (1+EPS)))
f_minus = float(get_ci_root(apar_j * (1-EPS)))
fd_grad_ci = (f_plus - f_minus) / (2*EPS*float(apar_j))
rel_err = abs(jax_grad_ci - fd_grad_ci) / (abs(fd_grad_ci) + 1e-30)
print(f"  JAX: {jax_grad_ci:.4e}")
print(f"  FD:  {fd_grad_ci:.4e}")
print(f"  Rel err: {rel_err:.3e}  {'PASS' if rel_err < 0.01 else 'FAIL'}")

# For comparison: IFT gradient
# d(ci_root)/d(apar) = -(dF/dapar) / (dF/dci)  at ci_root
dF_dci  = float(jax.grad(lambda ci: _CiFuncPure_jax(ci, **fkw_base))(ci_root))
dF_dapar = float(jax.grad(lambda a: _CiFuncPure_jax(ci_root, **build_fkw(a)))(apar_j))
ift_grad = -dF_dapar / (dF_dci if abs(dF_dci) > 1e-15 else 1e-15)
print(f"  IFT: {ift_grad:.4e}")
print(f"  (IFT = implicit function theorem = exact d(ci*)/d(apar))")

print("\n=== test_ci_solver_grad.py complete ===")
