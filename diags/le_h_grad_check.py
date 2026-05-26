"""
LE and H gradient check — CLM-ml-jax.

Verifies that jax.grad produces accurate gradients for Latent Heat (LE)
and Sensible Heat (H) through the full CLM-ml-jax column by comparing
against central finite differences (ε = 1e-4).

Outputs tested:
  - LE: dpai-weighted sum of lhleaf_leaf (sun+shade) over canopy layers
        (compute_le in expt_init.py)
  - H:  dpai-weighted sum of shleaf_leaf (sun+shade) over canopy layers
        (compute_h in expt_init.py)

Parameters tested:
  - alpha_sw:    scale on shortwave radiation forcing
  - alpha_tref:  scale on air temperature forcing
  - alpha_iota:  scale on iota_SPA (WUE efficiency, gs_type=2)
  - alpha_vcmax: scale on vcmaxpft (max carboxylation at 25°C)
  - alpha_g1:    scale on g1_MED (Medlyn slope; INACT for gs_type=2)

Pass criterion: relative error < 1% (same as fd_grad_check.py for GPP).

Usage (from project root):
    cd src && python ../diags/le_h_grad_check.py

Output:
  - Console table: parameter, jax.grad vs FD, relative error, PASS/FAIL
  - diags/figures/le_h_grad_check.png: bar chart
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

# ── Shared init ────────────────────────────────────────────────────────────────
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs, jax, jnp, MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst, compute_le, compute_h,
)
import multilayer_canopy.MLpftconMod              as _MLpftconMod
import multilayer_canopy.MLLeafPhotosynthesisMod  as _LeafMod
import multilayer_canopy.MLCanopyNitrogenProfileMod as _NitroMod

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_p    = grid.p
_ncan = grid.ncan

_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}

# ── MLpftcon injection helpers ────────────────────────────────────────────────
_orig_pftcon = _MLpftconMod.MLpftcon


def _set_pftcon(new_inst):
    _MLpftconMod.MLpftcon = new_inst
    _LeafMod.MLpftcon     = new_inst
    _NitroMod.MLpftcon    = new_inst


def _restore_pftcon():
    _MLpftconMod.MLpftcon = _orig_pftcon
    _LeafMod.MLpftcon     = _orig_pftcon
    _NitroMod.MLpftcon    = _orig_pftcon


# ── Forward functions — LE ─────────────────────────────────────────────────────

def _run(modified_atm=None, modified_ml=None, vcmaxpft_jax=None):
    return MLCanopyFluxes(
        mlcanopy_inst=modified_ml or mlcanopy_inst,
        atm2lnd_inst=modified_atm or atm2lnd_inst,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        vcmaxpft_jax=vcmaxpft_jax,
        **_mlcf_kwargs_no_atm,
    )


def forward_le_sw(alpha):
    inst = _run(modified_atm=atm2lnd_inst._replace(
        forc_solad_downscaled_col=alpha * atm2lnd_inst.forc_solad_downscaled_col,
        forc_solai_grc           =alpha * atm2lnd_inst.forc_solai_grc,
    ))
    return compute_le(inst, _p, _ncan)


def forward_le_tref(alpha):
    inst = _run(modified_atm=atm2lnd_inst._replace(
        forc_t_downscaled_col=alpha * atm2lnd_inst.forc_t_downscaled_col,
    ))
    return compute_le(inst, _p, _ncan)


def forward_le_g1(alpha):
    _set_pftcon(_orig_pftcon._replace(g1_MED=alpha * _orig_pftcon.g1_MED))
    inst = _run()
    _restore_pftcon()
    return compute_le(inst, _p, _ncan)


def forward_le_iota(alpha):
    _set_pftcon(_orig_pftcon._replace(iota_SPA=alpha * _orig_pftcon.iota_SPA))
    inst = _run()
    _restore_pftcon()
    return compute_le(inst, _p, _ncan)


def forward_le_vcmax(alpha):
    vcmaxpft_jax = alpha * _orig_pftcon.vcmaxpft
    inst = _run(vcmaxpft_jax=vcmaxpft_jax)
    return compute_le(inst, _p, _ncan)


# ── Forward functions — H ──────────────────────────────────────────────────────

def forward_h_sw(alpha):
    inst = _run(modified_atm=atm2lnd_inst._replace(
        forc_solad_downscaled_col=alpha * atm2lnd_inst.forc_solad_downscaled_col,
        forc_solai_grc           =alpha * atm2lnd_inst.forc_solai_grc,
    ))
    return compute_h(inst, _p, _ncan)


def forward_h_tref(alpha):
    inst = _run(modified_atm=atm2lnd_inst._replace(
        forc_t_downscaled_col=alpha * atm2lnd_inst.forc_t_downscaled_col,
    ))
    return compute_h(inst, _p, _ncan)


def forward_h_g1(alpha):
    _set_pftcon(_orig_pftcon._replace(g1_MED=alpha * _orig_pftcon.g1_MED))
    inst = _run()
    _restore_pftcon()
    return compute_h(inst, _p, _ncan)


def forward_h_iota(alpha):
    _set_pftcon(_orig_pftcon._replace(iota_SPA=alpha * _orig_pftcon.iota_SPA))
    inst = _run()
    _restore_pftcon()
    return compute_h(inst, _p, _ncan)


def forward_h_vcmax(alpha):
    vcmaxpft_jax = alpha * _orig_pftcon.vcmaxpft
    inst = _run(vcmaxpft_jax=vcmaxpft_jax)
    return compute_h(inst, _p, _ncan)


# ── Baseline ──────────────────────────────────────────────────────────────────
from multilayer_canopy.MLclm_varctl import gs_type as _gs_type
print("\n=== Baseline outputs ===", flush=True)
_inst0 = _run()
print(f"  LE proxy (baseline) = {float(compute_le(_inst0, _p, _ncan)):.4f} W/m2", flush=True)
print(f"  H  proxy (baseline) = {float(compute_h(_inst0,  _p, _ncan)):.4f} W/m2", flush=True)
_gs_name = {0: "Medlyn (gs_type=0)", 1: "Ball-Berry (gs_type=1)", 2: "WUE (gs_type=2)"}.get(
    _gs_type, f"unknown ({_gs_type})")
print(f"  Stomatal model: {_gs_name}", flush=True)
del _inst0

# ── Compute JAX gradients ─────────────────────────────────────────────────────
EPS = 1e-4

PARAM_FNS = [
    ("alpha_sw",   forward_le_sw,   forward_h_sw),
    ("alpha_tref", forward_le_tref, forward_h_tref),
    ("alpha_g1",   forward_le_g1,   forward_h_g1),
    ("alpha_iota", forward_le_iota, forward_h_iota),
    ("alpha_vcmax",forward_le_vcmax,forward_h_vcmax),
]

print("\n=== Computing JAX + FD gradients for LE ===", flush=True)
le_results = []
for name, fn_le, _ in PARAM_FNS:
    t0 = time.time()
    jax_val = float(jax.jit(jax.grad(fn_le))(jnp.float64(1.0)))
    t_jax = time.time() - t0
    t0 = time.time()
    fp = float(fn_le(jnp.float64(1.0 + EPS)))
    fm = float(fn_le(jnp.float64(1.0 - EPS)))
    fd_val = (fp - fm) / (2 * EPS)
    t_fd = time.time() - t0
    print(f"  dLE/d({name:12s})  JAX={jax_val:+.4e} ({t_jax:.0f}s)  FD={fd_val:+.4e} ({t_fd:.0f}s)", flush=True)
    le_results.append((name, jax_val, fd_val))

print("\n=== Computing JAX + FD gradients for H ===", flush=True)
h_results = []
for name, _, fn_h in PARAM_FNS:
    t0 = time.time()
    jax_val = float(jax.jit(jax.grad(fn_h))(jnp.float64(1.0)))
    t_jax = time.time() - t0
    t0 = time.time()
    fp = float(fn_h(jnp.float64(1.0 + EPS)))
    fm = float(fn_h(jnp.float64(1.0 - EPS)))
    fd_val = (fp - fm) / (2 * EPS)
    t_fd = time.time() - t0
    print(f"  dH/d({name:12s})   JAX={jax_val:+.4e} ({t_jax:.0f}s)  FD={fd_val:+.4e} ({t_fd:.0f}s)", flush=True)
    h_results.append((name, jax_val, fd_val))


# ── Summary table ─────────────────────────────────────────────────────────────
def _summarize(label, results):
    print(f"\n=== {label} gradient accuracy ===")
    print(f"{'Parameter':<15}  {'JAX':>12}  {'FD':>12}  {'Rel err':>10}  Status")
    print("-" * 65)
    all_pass = True
    for name, jax_val, fd_val in results:
        both_tiny = abs(jax_val) < 1e-6 and abs(fd_val) < 1e-6
        if both_tiny:
            rel_err = 0.0; status = "INACT"
        else:
            rel_err = abs(jax_val - fd_val) / (abs(fd_val) + 1e-30)
            status = "PASS" if rel_err < 0.01 else "FAIL"
            if status == "FAIL":
                all_pass = False
        print(f"  {name:<13}  {jax_val:>12.4e}  {fd_val:>12.4e}  {rel_err:>10.2e}  {status}")
    print(f"\n  → {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return all_pass

pass_le = _summarize("LE", le_results)
pass_h  = _summarize("H",  h_results)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
names = [r[0] for r in le_results]
x = np.arange(len(names))
w = 0.35

for row, (results, lbl) in enumerate([(le_results, "LE"), (h_results, "H")]):
    jax_vals = np.abs([r[1] for r in results])
    fd_vals  = np.abs([r[2] for r in results])
    both_tiny = [(abs(r[1]) < 1e-6 and abs(r[2]) < 1e-6) for r in results]
    rel_errs = [
        0.0 if bt else abs(r[1] - r[2]) / (abs(r[2]) + 1e-30)
        for r, bt in zip(results, both_tiny)
    ]
    colors = ["gray" if bt else ("green" if re < 0.01 else "red")
              for bt, re in zip(both_tiny, rel_errs)]

    axes[row, 0].bar(x - w/2, np.maximum(jax_vals, 1e-12), w, label="jax.grad", color="steelblue")
    axes[row, 0].bar(x + w/2, np.maximum(fd_vals,  1e-12), w, label="FD", color="coral", alpha=0.8)
    axes[row, 0].set_yscale("log")
    axes[row, 0].set_xticks(x); axes[row, 0].set_xticklabels(names, fontsize=8)
    axes[row, 0].set_title(f"|d{lbl}/dα|: JAX vs FD"); axes[row, 0].legend(fontsize=8)

    axes[row, 1].bar(x, rel_errs, color=colors)
    axes[row, 1].axhline(0.01, color="k", linestyle="--", lw=1.2, label="1% threshold")
    axes[row, 1].set_xticks(x); axes[row, 1].set_xticklabels(names, fontsize=8)
    axes[row, 1].set_title(f"d{lbl}/dα relative error (gray=INACT)")
    axes[row, 1].legend(fontsize=8)

fig.suptitle("CLM-ml-jax: LE and H gradient check — jax.grad vs central FD", fontsize=12)
fig.tight_layout()
out = FIGURES_DIR / "le_h_grad_check.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {out}")
print(f"\n{'ALL OUTPUTS PASS' if (pass_le and pass_h) else 'FAILURES DETECTED'}")
