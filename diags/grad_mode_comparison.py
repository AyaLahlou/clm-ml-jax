"""
Gradient mode comparison: jacrev vs FD with flux_profile_type=0 vs 1.

Diagnoses whether the 15%/76% disagreement seen in Exp 2 (fd_grad_check.py)
is caused by an incorrect custom_vjp in ImplicitFluxProfileSolution.

NOTE: jacfwd (jax.jvp) is impossible — JAX raises TypeError when applied to a
custom_vjp function. The only isolation test available is switching the turbulence
solver off (flux_profile_type=0, well-mixed) to remove the custom_vjp from the
gradient path entirely.

Strategy
--------
  flux_profile_type=1 : ImplicitFluxProfileSolution active  (uses custom_vjp)
  flux_profile_type=0 : well-mixed turbulence               (no custom_vjp)

For each mode, compute dGPP/d(alpha_sw) and dGPP/d(alpha_tair) via:
  jacrev : jax.jit(jax.grad(f))(1.0)    — reverse-mode autodiff
  FD     : central difference at 5 epsilons

Interpretation
--------------
  If jacrev ≈ FD at type=0 but jacrev ≠ FD at type=1
      → custom_vjp in ImplicitFluxProfileSolution is the confirmed bug.
  If jacrev ≠ FD at both modes
      → bug is elsewhere (NaN gradients, stop_gradient, or other cause).
  If jacrev ≈ FD at both modes
      → gradients are correct; the earlier failure was a FD epsilon issue.

Usage (from project root):
    cd src && python ../diags/grad_mode_comparison.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Shared init ───────────────────────────────────────────────────────────────
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs, jax, jnp, MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst, compute_gpp,
)
from multilayer_canopy import MLclm_varctl

_p = grid.p

# Build kwargs without atm2lnd_inst so we can pass it as a traced arg
_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}


# ── Forward functions ─────────────────────────────────────────────────────────

def forward_gpp_sw(alpha: jnp.ndarray) -> jnp.ndarray:
    """Scale beam+diffuse SW by alpha via atm2lnd_inst, return GPP proxy."""
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


def forward_gpp_tref(alpha: jnp.ndarray) -> jnp.ndarray:
    """Scale air temperature by alpha via atm2lnd_inst, return GPP proxy."""
    modified_atm = atm2lnd_inst._replace(
        forc_t_downscaled_col=alpha * atm2lnd_inst.forc_t_downscaled_col,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, grid.ncan)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fd_central(f, eps: float) -> float:
    """Central finite difference at alpha=1."""
    a = jnp.float64(1.0)
    return float((f(a + eps) - f(a - eps)) / (2.0 * eps))


def jacrev_grad(f) -> tuple[float, float]:
    """Reverse-mode autodiff. Returns (value, wall_time_s)."""
    t0 = time.time()
    g = float(jax.jit(jax.grad(f))(jnp.float64(1.0)))
    return g, time.time() - t0


def rel_err(a, b):
    return abs(a - b) / (abs(b) + 1e-30)


# ── Modes to test ─────────────────────────────────────────────────────────────

MODES = [
    (1, "flux_profile_type=1 (implicit solver, custom_vjp ACTIVE)"),
    (0, "flux_profile_type=0 (well-mixed,       custom_vjp ABSENT)"),
]

PARAMS = [
    ("alpha_sw",   forward_gpp_sw),
    ("alpha_tair", forward_gpp_tref),
]

EPSILONS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

# ── Main loop ─────────────────────────────────────────────────────────────────

all_results: dict = {}   # {mode_type: {param_name: {jacrev, fd: {eps: val}}}}

for mode_type, mode_label in MODES:
    print(f"\n{'='*80}", flush=True)
    print(f"MODE: {mode_label}", flush=True)
    print(f"{'='*80}", flush=True)

    # Switch turbulence solver mode
    MLclm_varctl.flux_profile_type = mode_type

    mode_results: dict = {}

    for pname, f in PARAMS:
        print(f"\n  --- {pname} ---", flush=True)

        rev, t_rev = jacrev_grad(f)
        print(f"    jacrev = {rev:.6e}  ({t_rev:.1f}s)", flush=True)

        fd_vals: dict = {}
        for eps in EPSILONS:
            t0 = time.time()
            fd = fd_central(f, eps)
            t_fd = time.time() - t0
            fd_vals[eps] = fd
            print(f"    FD eps={eps:.0e} = {fd:.6e}  ({t_fd:.1f}s)", flush=True)

        mode_results[pname] = dict(jacrev=rev, fd=fd_vals)

    all_results[mode_type] = mode_results


# ── Comparison table ──────────────────────────────────────────────────────────

print("\n" + "=" * 90)
print("COMPARISON TABLE")
print("=" * 90)

for mode_type, mode_label in MODES:
    print(f"\n  {mode_label}")
    r_mode = all_results[mode_type]
    for pname in [p for p, _ in PARAMS]:
        r = r_mode[pname]
        rev = r["jacrev"]
        fd_d = r["fd"]
        best_eps = min(EPSILONS, key=lambda e: rel_err(rev, fd_d[e]))
        best_fd  = fd_d[best_eps]
        print(f"    {pname}:  jacrev={rev:.4e}  best_FD={best_fd:.4e} (eps={best_eps:.0e})  "
              f"rel_err={rel_err(rev, best_fd):.3e}")


# ── Conclusions ───────────────────────────────────────────────────────────────

AGREE_THRESH = 0.05   # 5% tolerance

print("\n" + "=" * 90)
print("CONCLUSIONS")
print("=" * 90)

agree: dict = {}   # {(mode_type, pname): bool}

for mode_type, mode_label in MODES:
    r_mode = all_results[mode_type]
    print(f"\n  {mode_label}")
    for pname in [p for p, _ in PARAMS]:
        r = r_mode[pname]
        rev = r["jacrev"]
        fd_d = r["fd"]
        best_eps = min(EPSILONS, key=lambda e: rel_err(rev, fd_d[e]))
        best_fd  = fd_d[best_eps]
        err = rel_err(rev, best_fd)
        ok = err < AGREE_THRESH
        agree[(mode_type, pname)] = ok
        status = "AGREE" if ok else f"DISAGREE (rel_err={err:.3e})"
        print(f"    {pname}: jacrev vs best FD → {status}", flush=True)

print("\n  --- Overall diagnosis ---", flush=True)
for pname in [p for p, _ in PARAMS]:
    ok0 = agree.get((0, pname), False)
    ok1 = agree.get((1, pname), False)
    if ok0 and not ok1:
        print(f"  {pname}: jacrev≈FD at type=0 but fails at type=1")
        print(f"    >>> CONFIRMED: custom_vjp in ImplicitFluxProfileSolution is the bug.")
    elif ok0 and ok1:
        print(f"  {pname}: jacrev≈FD in both modes — gradients are correct.")
        print(f"    >>> The earlier failure at eps=1e-4 was a FD epsilon issue.")
    elif not ok0 and not ok1:
        print(f"  {pname}: jacrev≠FD in both modes — bug is NOT in custom_vjp.")
        print(f"    >>> Look for stop_gradient, NaN sources, or wrong output scalar.")
    else:
        print(f"  {pname}: jacrev fails at type=0 but passes at type=1 — unexpected.")
        print(f"    >>> The implicit solver is actually improving gradient accuracy.")

print("\n=== grad_mode_comparison.py complete ===", flush=True)
