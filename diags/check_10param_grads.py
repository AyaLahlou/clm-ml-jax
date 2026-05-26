"""
check_10param_grads.py — Per-parameter JAX-vs-FD gradient check for all 10
parameters used in the multi-parameter calibration experiment.

Adam diverged to NaN after step 1 in job 7578886. This script pinpoints which
parameters produce NaN or inaccurate JAX gradients by testing each one
independently (all other scale factors held at 1.0).

Parameters tested (indices match multipar_calibration.py theta vector):
  0  alpha_sw    — shortwave radiation (direct + diffuse)
  1  alpha_tref  — air temperature
  2  alpha_vcmax — global Vcmax25 (vcmaxpft_jax explicit arg)
  3  alpha_iota  — WUE efficiency iota_SPA (module-global mutation)
  4  alpha_q     — specific humidity
  5  alpha_pbot  — atmospheric pressure
  6  alpha_lwrad — longwave radiation
  7  alpha_u     — wind u-component
  8  alpha_v     — wind v-component
  9  alpha_pco2  — CO2 partial pressure

Outputs:
  console table with PASS/FAIL/NaN/INACT status
  diags/figures/check_10param_grads.csv
"""
from __future__ import annotations

import csv
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
import numpy as np

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared model init ─────────────────────────────────────────────────────────
print("=== check_10param_grads.py: loading model ===", flush=True)
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs,
    MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst,
    compute_gpp,
)
import multilayer_canopy.MLpftconMod             as _MLpftconMod
import multilayer_canopy.MLLeafPhotosynthesisMod  as _LeafMod
import multilayer_canopy.MLCanopyNitrogenProfileMod as _NitroMod

_p    = grid.p
_ncan = grid.ncan

_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}

_orig_pftcon = _MLpftconMod.MLpftcon


def _set_pftcon(new_inst):
    _MLpftconMod.MLpftcon = new_inst
    _LeafMod.MLpftcon     = new_inst
    _NitroMod.MLpftcon    = new_inst


def _restore_pftcon():
    _MLpftconMod.MLpftcon = _orig_pftcon
    _LeafMod.MLpftcon     = _orig_pftcon
    _NitroMod.MLpftcon    = _orig_pftcon


# ── Per-parameter forward functions ──────────────────────────────────────────
# Each function perturbs only its own parameter; all others stay at 1.0.

def forward_sw(alpha_i: jnp.ndarray) -> jnp.ndarray:
    """alpha_sw: scale shortwave (direct + diffuse)."""
    modified_atm = atm2lnd_inst._replace(
        forc_solad_downscaled_col=alpha_i * atm2lnd_inst.forc_solad_downscaled_col,
        forc_solai_grc           =alpha_i * atm2lnd_inst.forc_solai_grc,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, _ncan)


def forward_tref(alpha_i: jnp.ndarray) -> jnp.ndarray:
    """alpha_tref: scale air temperature."""
    modified_atm = atm2lnd_inst._replace(
        forc_t_downscaled_col=alpha_i * atm2lnd_inst.forc_t_downscaled_col,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, _ncan)


def forward_vcmax(alpha_i: jnp.ndarray) -> jnp.ndarray:
    """alpha_vcmax: scale vcmaxpft array (explicit JAX arg to bypass JIT cache)."""
    vcmaxpft_jax = alpha_i * _orig_pftcon.vcmaxpft
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=atm2lnd_inst,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        vcmaxpft_jax=vcmaxpft_jax,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, _ncan)


def forward_iota(alpha_i: jnp.ndarray) -> jnp.ndarray:
    """alpha_iota: scale iota_SPA via module-global mutation."""
    _set_pftcon(_orig_pftcon._replace(iota_SPA=alpha_i * _orig_pftcon.iota_SPA))
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=atm2lnd_inst,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    _restore_pftcon()
    return compute_gpp(inst, _p, _ncan)


def forward_q(alpha_i: jnp.ndarray) -> jnp.ndarray:
    """alpha_q: scale specific humidity."""
    modified_wat = wateratm2lndbulk_inst._replace(
        forc_q_downscaled_col=alpha_i * wateratm2lndbulk_inst.forc_q_downscaled_col,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=atm2lnd_inst,
        wateratm2lndbulk_inst=modified_wat,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, _ncan)


def forward_pbot(alpha_i: jnp.ndarray) -> jnp.ndarray:
    """alpha_pbot: scale atmospheric pressure."""
    modified_atm = atm2lnd_inst._replace(
        forc_pbot_downscaled_col=alpha_i * atm2lnd_inst.forc_pbot_downscaled_col,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, _ncan)


def forward_lwrad(alpha_i: jnp.ndarray) -> jnp.ndarray:
    """alpha_lwrad: scale longwave radiation."""
    modified_atm = atm2lnd_inst._replace(
        forc_lwrad_downscaled_col=alpha_i * atm2lnd_inst.forc_lwrad_downscaled_col,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, _ncan)


def forward_u(alpha_i: jnp.ndarray) -> jnp.ndarray:
    """alpha_u: scale wind u-component."""
    modified_atm = atm2lnd_inst._replace(
        forc_u_grc=alpha_i * atm2lnd_inst.forc_u_grc,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, _ncan)


def forward_v(alpha_i: jnp.ndarray) -> jnp.ndarray:
    """alpha_v: scale wind v-component."""
    modified_atm = atm2lnd_inst._replace(
        forc_v_grc=alpha_i * atm2lnd_inst.forc_v_grc,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, _ncan)


def forward_pco2(alpha_i: jnp.ndarray) -> jnp.ndarray:
    """alpha_pco2: scale CO2 partial pressure."""
    modified_atm = atm2lnd_inst._replace(
        forc_pco2_grc=alpha_i * atm2lnd_inst.forc_pco2_grc,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, _ncan)


# ── Parameter registry ────────────────────────────────────────────────────────
PARAMS = [
    (0,  "alpha_sw",    forward_sw),
    (1,  "alpha_tref",  forward_tref),
    (2,  "alpha_vcmax", forward_vcmax),
    (3,  "alpha_iota",  forward_iota),
    (4,  "alpha_q",     forward_q),
    (5,  "alpha_pbot",  forward_pbot),
    (6,  "alpha_lwrad", forward_lwrad),
    (7,  "alpha_u",     forward_u),
    (8,  "alpha_v",     forward_v),
    (9,  "alpha_pco2",  forward_pco2),
]

EPS = 1e-4
ONE = jnp.float64(1.0)

# ── Print baseline ────────────────────────────────────────────────────────────
print("\n=== Baseline GPP (alpha_i=1 for all i) ===", flush=True)
t0 = time.time()
gpp_baseline = float(forward_sw(ONE))
print(f"  GPP baseline = {gpp_baseline:.6f}  ({time.time()-t0:.2f}s)", flush=True)
if gpp_baseline == 0.0:
    print("  WARNING: GPP is zero — likely nighttime step; gradients may be trivially zero.", flush=True)

# ── Run per-parameter checks ──────────────────────────────────────────────────
print("\n=== Per-parameter gradient check (JAX vs central FD, eps=1e-4) ===\n", flush=True)

rows = []  # (idx, name, grad_jax, grad_fd, rel_err, status, jax_nan, fd_nan, time_jax, time_fd)

for idx, name, fwd_fn in PARAMS:
    print(f"--- [{idx}] {name} ---", flush=True)

    # JAX gradient
    t_jax0 = time.time()
    try:
        grad_jax_val = float(jax.jit(jax.grad(fwd_fn))(ONE))
    except Exception as exc:
        print(f"  JAX grad ERROR: {exc}", flush=True)
        grad_jax_val = float("nan")
    t_jax = time.time() - t_jax0
    jax_nan = np.isnan(grad_jax_val)
    print(f"  JAX grad = {grad_jax_val:.6e}  ({t_jax:.2f}s)  {'NaN!' if jax_nan else ''}",
          flush=True)

    # FD gradient (central differences)
    t_fd0 = time.time()
    try:
        f_plus  = float(fwd_fn(ONE + EPS))
        f_minus = float(fwd_fn(ONE - EPS))
        grad_fd_val = (f_plus - f_minus) / (2.0 * EPS)
    except Exception as exc:
        print(f"  FD grad ERROR: {exc}", flush=True)
        grad_fd_val = float("nan")
    t_fd = time.time() - t_fd0
    fd_nan = np.isnan(grad_fd_val)
    print(f"  FD  grad = {grad_fd_val:.6e}  ({t_fd:.2f}s)  {'NaN!' if fd_nan else ''}",
          flush=True)

    # Status classification
    if jax_nan or fd_nan:
        status = "NaN"
        rel_err = float("nan")
    elif abs(grad_jax_val) < 1e-10 and abs(grad_fd_val) < 1e-10:
        status = "INACT"
        rel_err = 0.0
    else:
        rel_err = abs(grad_jax_val - grad_fd_val) / (abs(grad_fd_val) + 1e-30)
        status = "PASS" if rel_err < 0.01 else "FAIL"

    print(f"  rel_err = {rel_err:.2e}  ->  {status}\n", flush=True)

    rows.append({
        "idx":       idx,
        "param":     name,
        "grad_jax":  grad_jax_val,
        "grad_fd":   grad_fd_val,
        "rel_err":   rel_err,
        "status":    status,
        "jax_nan":   jax_nan,
        "fd_nan":    fd_nan,
        "time_jax_s": round(t_jax, 2),
        "time_fd_s":  round(t_fd, 2),
    })

# ── Summary table ─────────────────────────────────────────────────────────────
print("=" * 90, flush=True)
print(f"{'idx':<4}  {'param':<14}  {'JAX grad':>14}  {'FD grad':>14}  {'rel err':>10}  {'status':>6}  {'NaN?':>4}", flush=True)
print("-" * 90, flush=True)
for r in rows:
    nan_flag = ("J" if r["jax_nan"] else " ") + ("F" if r["fd_nan"] else " ")
    rel_str  = "nan" if np.isnan(r["rel_err"]) else f"{r['rel_err']:.2e}"
    print(
        f"{r['idx']:<4}  {r['param']:<14}  {r['grad_jax']:>14.4e}  "
        f"{r['grad_fd']:>14.4e}  {rel_str:>10}  {r['status']:>6}  {nan_flag:>4}",
        flush=True,
    )
print("=" * 90, flush=True)

n_pass  = sum(1 for r in rows if r["status"] == "PASS")
n_fail  = sum(1 for r in rows if r["status"] == "FAIL")
n_nan   = sum(1 for r in rows if r["status"] == "NaN")
n_inact = sum(1 for r in rows if r["status"] == "INACT")
print(f"\nSummary: {n_pass} PASS  |  {n_fail} FAIL  |  {n_nan} NaN  |  {n_inact} INACT  (of {len(rows)} params)", flush=True)

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = FIGURES_DIR / "check_10param_grads.csv"
fieldnames = ["idx", "param", "grad_jax", "grad_fd", "rel_err", "status",
              "jax_nan", "fd_nan", "time_jax_s", "time_fd_s"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"\nCSV saved: {csv_path}", flush=True)

print("\n=== check_10param_grads.py complete ===", flush=True)
