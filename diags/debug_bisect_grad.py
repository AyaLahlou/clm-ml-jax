"""
Minimal gradient check for _bisect_gs_ift fix.
Tests that d(gs_ift)/d(tleaf) is finite, not exploding.
Runs on CPU without warmup.

Usage (from project root):
    python diags/debug_bisect_grad.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path

os.environ['CLM_ML_NO_CHECKPOINT'] = '1'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multilayer_canopy.MLLeafPhotosynthesisMod import (
    _bisect_gs_ift, _CiFuncGsJax, _ft, _fth, SatVap, _fth25,
)
from multilayer_canopy.MLclm_varcon import (
    kc25, ko25, cp25, kcha, koha, cpha,
    vcmaxha_noacclim as vcmaxha, vcmaxha_acclim,
    vcmaxhd_noacclim as vcmaxhd, vcmaxhd_acclim,
    vcmaxse_noacclim as vcmaxse, vcmaxse_acclim,
    jmaxha_noacclim as jmaxha, jmaxhd_noacclim as jmaxhd, jmaxse_noacclim as jmaxse,
    rdha, rdhd, rdse,
    phi_psII, theta_j,
)
from multilayer_canopy.MLMathToolsMod import quadratic
from multilayer_canopy.MLpftconMod import MLpftcon

print(f"JAX devices: {jax.devices()}")
pft = 6
gsmin = float(MLpftcon.gsmin_SPA[pft])
iota  = float(MLpftcon.iota_SPA[pft])
print(f"PFT {pft}: gsmin={gsmin:.4f}, iota={iota:.4f}")

# Build scaling factors at 25°C
vcmaxc = _fth25(vcmaxhd, vcmaxse)
jmaxc  = _fth25(jmaxhd,  jmaxse)
rdc    = _fth25(rdhd,    rdse)
print(f"vcmaxc={vcmaxc:.4f}, jmaxc={jmaxc:.4f}, rdc={rdc:.4f}")


def forward_gs(tleaf, apar=500.0, vcmax25=60.0, jmax25=120.0, rd25=1.5,
               gbc=0.1, gbv=0.1, cair=40.0, eair=2000.0, pref=101325.0, o2ref=20900.0):
    kc_v   = kc25  * _ft(tleaf, kcha)
    ko_v   = ko25  * _ft(tleaf, koha)
    cp_v   = cp25  * _ft(tleaf, cpha)
    vcmax_v = jnp.float64(vcmax25) * _ft(tleaf, vcmaxha) * _fth(tleaf, vcmaxhd, vcmaxse, vcmaxc)
    jmax_v  = jnp.float64(jmax25)  * _ft(tleaf, jmaxha)  * _fth(tleaf, jmaxhd,  jmaxse,  jmaxc)
    rd_v    = jnp.float64(rd25)    * _ft(tleaf, rdha)     * _fth(tleaf, rdhd,    rdse,    rdc)
    lesat_v, _ = SatVap(tleaf)
    qabs  = 0.5 * phi_psII * jnp.float64(apar)
    r1j, r2j = quadratic(theta_j, -(qabs + jmax_v), qabs * jmax_v)
    je_v  = jnp.minimum(r1j, r2j)
    ci_kw = dict(is_c3=True, dpai_ic=jnp.float64(1.0), gbc_ic=jnp.float64(gbc),
                 cair_ic=jnp.float64(cair), vcmax_ic=vcmax_v, je_ic=je_v,
                 kp_ic=jnp.float64(0.0), rd_ic=rd_v, kc_ic=kc_v, ko_ic=ko_v, cp_ic=cp_v,
                 o2ref_p=jnp.float64(o2ref), apar_ic=jnp.float64(apar), c3psn_pft_val=jnp.float64(1.0))
    se_kw = dict(iota=iota, pref_p=jnp.float64(pref), eair_ic=jnp.float64(eair),
                 gbv_ic=jnp.float64(gbv), lesat_ic=lesat_v, **ci_kw)
    return _bisect_gs_ift(jnp.float64(gsmin), jnp.float64(2.0), se_kw)


def forward_agross(tleaf, apar=500.0):
    kc_v   = kc25  * _ft(tleaf, kcha)
    ko_v   = ko25  * _ft(tleaf, koha)
    cp_v   = cp25  * _ft(tleaf, cpha)
    vcmax25, jmax25, rd25 = 60.0, 120.0, 1.5
    vcmax_v = jnp.float64(vcmax25) * _ft(tleaf, vcmaxha) * _fth(tleaf, vcmaxhd, vcmaxse, vcmaxc)
    jmax_v  = jnp.float64(jmax25)  * _ft(tleaf, jmaxha)  * _fth(tleaf, jmaxhd,  jmaxse,  jmaxc)
    rd_v    = jnp.float64(rd25)    * _ft(tleaf, rdha)     * _fth(tleaf, rdhd,    rdse,    rdc)
    lesat_v, _ = SatVap(tleaf)
    qabs  = 0.5 * phi_psII * jnp.float64(apar)
    r1j, r2j = quadratic(theta_j, -(qabs + jmax_v), qabs * jmax_v)
    je_v  = jnp.minimum(r1j, r2j)
    ci_kw = dict(is_c3=True, dpai_ic=jnp.float64(1.0), gbc_ic=jnp.float64(0.1),
                 cair_ic=jnp.float64(40.0), vcmax_ic=vcmax_v, je_ic=je_v,
                 kp_ic=jnp.float64(0.0), rd_ic=rd_v, kc_ic=kc_v, ko_ic=ko_v, cp_ic=cp_v,
                 o2ref_p=jnp.float64(20900.0), apar_ic=jnp.float64(apar), c3psn_pft_val=jnp.float64(1.0))
    se_kw = dict(iota=iota, pref_p=jnp.float64(101325.0), eair_ic=jnp.float64(2000.0),
                 gbv_ic=jnp.float64(0.1), lesat_ic=lesat_v, **ci_kw)
    gs = _bisect_gs_ift(jnp.float64(gsmin), jnp.float64(2.0), se_kw)
    _, _, _, _, agross, _, _ = _CiFuncGsJax(gs, **ci_kw)
    return agross


eps = 1e-3
T0  = jnp.float64(303.15)

print()
for apar, label in [(500.0, "BRIGHT (apar=500)"), (1.0, "DIM   (apar=1)"), (1e-6, "DARK  (apar=1e-6)")]:
    print(f"=== {label} ===")
    gs0  = forward_gs(T0, apar)
    print(f"  gs     = {float(gs0):.6e}")

    try:
        grad_gs_jax = float(jax.jit(jax.grad(lambda t: forward_gs(t, apar)))(T0))
    except Exception as e:
        grad_gs_jax = float('nan')
        print(f"  JAX grad error: {e}")

    gs_p = forward_gs(T0 + eps, apar)
    gs_m = forward_gs(T0 - eps, apar)
    grad_gs_fd = float((gs_p - gs_m) / (2*eps))

    rel = abs(grad_gs_jax - grad_gs_fd) / (abs(grad_gs_fd) + 1e-30)
    ok  = "PASS" if rel < 0.01 else "FAIL"
    print(f"  d(gs)/dT JAX={grad_gs_jax:.3e}  FD={grad_gs_fd:.3e}  rel={rel:.2e}  {ok}")
    print()

print("=== agross gradient check (BRIGHT) ===")
ag0 = forward_agross(T0, 500.0)
print(f"  agross = {float(ag0):.6e}")
grad_ag_jax = float(jax.jit(jax.grad(lambda t: forward_agross(t, 500.0)))(T0))
ag_p = forward_agross(T0 + eps, 500.0)
ag_m = forward_agross(T0 - eps, 500.0)
grad_ag_fd  = float((ag_p - ag_m) / (2*eps))
rel = abs(grad_ag_jax - grad_ag_fd) / (abs(grad_ag_fd) + 1e-30)
ok  = "PASS" if rel < 0.01 else "FAIL"
print(f"  d(agross)/dT JAX={grad_ag_jax:.3e}  FD={grad_ag_fd:.3e}  rel={rel:.2e}  {ok}")

print("\n=== debug_bisect_grad.py complete ===")
