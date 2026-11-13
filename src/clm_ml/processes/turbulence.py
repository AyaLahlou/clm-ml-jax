"""Canopy turbulence: the roughness sublayer (RSL) theory of Harman and Finnigan (2007, 2008)
for the Obukhov length, displacement height, friction velocity, wind speed profile and the
aerodynamic conductances between the canopy layers.

One patch. Layers are 0-based slots on axis 0 (``clm_ml.state.PatchLayers``): layers
``0 .. n_within-1`` are within the canopy, ``n_within .. n_layers-1`` above it; the
results in padding slots (``>= n_layers``) are finite but meaningless (the caller keeps its
own values there).

The RSL function psihat comes from a lookup table (:class:`RSLTables`, read from the
package's ``clm_ml/data/psihat.nc`` by ``clm_ml.io.rsl_tables``) that every routine needing
it takes as an argument.

Fortran reference: CLM-ml v2 ``MLCanopyTurbulenceMod.F90``. Bonan et al. (2018) Agric. For.
Meteorol. 250-251, 64-89 (appendix A); Harman and Finnigan (2007, 2008); Harman (2012) for
beta in unstable conditions (Bonan et al. 2026, eq. 2).

Reads and writes are the arguments and results; ``Reads``/``Writes`` in the docstrings name
the CLM-ml v2 (``mlcanopy_type``) fields they correspond to.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from clm_ml.processes.constants import (
    LcL_max,
    LcL_min,
    Pr0,
    Pr1,
    Pr2,
    beta_neutral_max,
    c2,
    cd,
    cr,
    grav,
    mmdry,
    mmh2o,
    ra_max,
    rpi,
    vkc,
    z0mg,
)
from clm_ml.processes.layers import pad_with_top
from clm_ml.processes.math import bisection_traceable
from clm_ml.state import PatchLayers

# Harman (2012) beta in unstable conditions, CHATS fit (Bonan et al. 2026, eq. 2; Fortran
# MLclm_varcon ``aH12``), as Python floats: they enter every iteration of the Obukhov solver
_AH12_0: float = 0.89
_AH12_1: float = -0.07
_AH12_2: float = 2.19
#: Secant steps of the Obukhov-length solver (numerics_and_regression.md §4.1)
OBU_ITERATIONS = 25


class PsihatTable(NamedTuple):
    """The RSL function psihat (unscaled, Bonan et al. 2018 eq. A21) on a grid."""

    zdt: Array  # (nZ,) (z - h) / (h - d), decreasing
    dtl: Array  # (nL,) (h - d) / L, increasing
    psi: Array  # (nZ, nL) psihat


class RSLTables(NamedTuple):
    """The psihat lookup tables for momentum and scalars."""

    momentum: PsihatTable
    scalar: PsihatTable


#: How the psihat tables are searched, by name (model.yaml ``numerics.table_search``) -> the
#: ``jnp.searchsorted`` method: ``loop`` is a binary search as a compiled loop (fastest on the
#: CPU), ``unrolled`` the same binary search without the loop, ``compare_all`` a comparison
#: with every grid value. All return the same indices. On a GPU every loop iteration is a
#: separate command, and these searches were 89 % of a step's loop iterations
#: (docs/design/gpu_progress.md, decision 114).
TABLE_SEARCH = {"loop": "scan", "unrolled": "scan_unrolled", "compare_all": "compare_all"}


# ---------------------------------------------------------------------------
# Monin-Obukhov stability functions (Bonan et al. 2018, eqs. A10-A13)
# ---------------------------------------------------------------------------


def phim_monin_obukhov(zeta):
    """Monin-Obukhov phi for momentum: ``(1 - 16 zeta)^(-1/4)`` if ``zeta < 0``, else
    ``1 + 5 zeta``.

    Fortran reference: MLCanopyTurbulenceMod.F90::phim_monin_obukhov (Bonan et al. 2018,
    eq. A10).
    """
    zeta = jnp.asarray(zeta)
    # jnp.maximum avoids sqrt(negative) NaN in the stable (zeta>0) branch gradient
    unstable = 1.0 / jnp.sqrt(jnp.sqrt(jnp.maximum(1.0 - 16.0 * zeta, 1.0e-10)))
    stable = 1.0 + 5.0 * zeta
    return jnp.where(zeta < 0.0, unstable, stable)


def phic_monin_obukhov(zeta):
    """Monin-Obukhov phi for scalars: ``(1 - 16 zeta)^(-1/2)`` if ``zeta < 0``, else
    ``1 + 5 zeta``.

    Fortran reference: MLCanopyTurbulenceMod.F90::phic_monin_obukhov (Bonan et al. 2018,
    eq. A11).
    """
    zeta = jnp.asarray(zeta)
    # jnp.maximum avoids sqrt(negative) NaN in the stable (zeta>0) branch gradient
    unstable = 1.0 / jnp.sqrt(jnp.maximum(1.0 - 16.0 * zeta, 1.0e-10))
    stable = 1.0 + 5.0 * zeta
    return jnp.where(zeta < 0.0, unstable, stable)


def psim_monin_obukhov(zeta):
    """Monin-Obukhov psi for momentum: with ``x = (1 - 16 zeta)^(1/4)``,
    ``2 log((1+x)/2) + log((1+x^2)/2) - 2 atan(x) + pi/2`` if ``zeta < 0``, else
    ``-5 zeta``.

    Fortran reference: MLCanopyTurbulenceMod.F90::psim_monin_obukhov (Bonan et al. 2018,
    eq. A12).
    """
    zeta = jnp.asarray(zeta)
    # The unstable branch is evaluated (and differentiated) for zeta >= 0 too: the floor
    # keeps its gradient finite there (0 * finite instead of 0 * inf). Forward values of the
    # selected branch are unchanged.
    x = jnp.sqrt(jnp.sqrt(jnp.maximum(1.0 - 16.0 * zeta, 1.0e-10)))
    unstable = (
        2.0 * jnp.log((1.0 + x) / 2.0)
        + jnp.log((1.0 + x * x) / 2.0)
        - 2.0 * jnp.arctan(x)
        + rpi * 0.5
    )
    stable = -5.0 * zeta
    return jnp.where(zeta < 0.0, unstable, stable)


def psic_monin_obukhov(zeta):
    """Monin-Obukhov psi for scalars: with ``x = (1 - 16 zeta)^(1/4)``,
    ``2 log((1+x^2)/2)`` if ``zeta < 0``, else ``-5 zeta``.

    Fortran reference: MLCanopyTurbulenceMod.F90::psic_monin_obukhov (Bonan et al. 2018,
    eq. A13).
    """
    zeta = jnp.asarray(zeta)
    # Same floor as psim_monin_obukhov
    x = jnp.sqrt(jnp.sqrt(jnp.maximum(1.0 - 16.0 * zeta, 1.0e-10)))
    unstable = 2.0 * jnp.log((1.0 + x * x) / 2.0)
    stable = -5.0 * zeta
    return jnp.where(zeta < 0.0, unstable, stable)


# ---------------------------------------------------------------------------
# RSL psihat lookup
# ---------------------------------------------------------------------------


def lookup_psihat(zdt, dtl, table: PsihatTable, table_search: str = "loop"):
    """Bilinear interpolation of the unscaled psihat ``A[(z-h)/(h-d), (h-d)/L]`` in
    ``table`` (multiplied by ``c1`` it is psihat). Outside the grid, both neighbours are the
    edge value (weights 1/2 each, as Fortran). ``table_search``: how the grids are searched
    (:data:`TABLE_SEARCH`; the same indices either way).

    Fortran reference: MLCanopyTurbulenceMod.F90::LookupPsihat (Bonan et al. 2018,
    eq. A21).
    """
    dtl = jnp.asarray(dtl)
    zdt = jnp.asarray(zdt)
    dtLg, zdtg, pg = (jnp.asarray(a) for a in (table.dtl, table.zdt, table.psi))
    nL = dtLg.shape[0]
    nZ = zdtg.shape[0]
    method = TABLE_SEARCH[table_search]

    # (h - d)/L bracket (grid increasing)
    at_bnd_L = (dtl <= dtLg[0]) | (dtl > dtLg[-1])
    L2_raw = jnp.clip(jnp.searchsorted(dtLg, dtl, side="right", method=method), 0, nL - 1)
    L1_raw = jnp.maximum(L2_raw - 1, 0)
    # jnp.maximum avoids select op -> prevents XLA select_divide_fusion bug
    denom_L = jnp.maximum(dtLg[L2_raw] - dtLg[L1_raw], 1.0e-30)
    wL1 = jnp.where(at_bnd_L, 0.5, (dtLg[L2_raw] - dtl) / denom_L)
    wL2 = 1.0 - wL1
    L1 = jnp.where(dtl <= dtLg[0], 0, jnp.where(dtl > dtLg[-1], nL - 1, L1_raw))
    L2 = jnp.where(dtl <= dtLg[0], 0, jnp.where(dtl > dtLg[-1], nL - 1, L2_raw))

    # (z - h)/(h - d) bracket (grid decreasing: searched as its negative)
    at_bnd_Z = (zdt > zdtg[0]) | (zdt < zdtg[-1])
    Z1_raw = jnp.clip(jnp.searchsorted(-zdtg, -zdt, side="right", method=method) - 1, 0, nZ - 2)
    Z2_raw = Z1_raw + 1
    # jnp.maximum avoids select op -> prevents XLA select_divide_fusion bug
    denom_Z = jnp.maximum(zdtg[Z1_raw] - zdtg[Z2_raw], 1.0e-30)
    wZ1 = jnp.where(at_bnd_Z, 0.5, (zdt - zdtg[Z2_raw]) / denom_Z)
    wZ2 = 1.0 - wZ1
    Z1 = jnp.where(zdt > zdtg[0], 0, jnp.where(zdt < zdtg[-1], nZ - 1, Z1_raw))
    Z2 = jnp.where(zdt > zdtg[0], 0, jnp.where(zdt < zdtg[-1], nZ - 1, Z2_raw))

    return (
        wZ1 * wL1 * pg[Z1, L1]
        + wZ2 * wL1 * pg[Z2, L1]
        + wZ1 * wL2 * pg[Z1, L2]
        + wZ2 * wL2 * pg[Z2, L2]
    )


# ---------------------------------------------------------------------------
# Canopy-top Prandtl number, RSL psi functions, beta
# ---------------------------------------------------------------------------


def prandtl_schmidt(beta_neutral, LcL, sparse_canopy: bool):
    """Prandtl (= Schmidt) number at the canopy top, ``Pr0 + Pr1 tanh(Pr2 Lc/L)``; for a
    sparse canopy blended towards 1 by ``beta_neutral / beta_neutral_max``.

    Fortran reference: MLCanopyTurbulenceMod.F90::GetPrSc (Bonan et al. 2018, eqs. A25,
    A34).
    """
    PrSc = Pr0 + Pr1 * jnp.tanh(Pr2 * LcL)
    if sparse_canopy:
        f = beta_neutral / beta_neutral_max
        PrSc = (1.0 - f) * 1.0 + f * PrSc
    return PrSc


def psi_rsl(za, hc, disp, obu, beta, PrSc, tables: RSLTables, table_search: str = "loop"):
    """RSL-modified stability functions from the canopy top ``hc`` to height ``za`` (m)
    for displacement height ``disp``, Obukhov length ``obu``, ``beta = u*/u(h)`` and
    Prandtl number ``PrSc``. Returns ``(psim, psic, psim2, psim_hat2)``: psi for momentum
    (including ``vkc / beta``) and scalars, and the Monin-Obukhov psi and psihat for
    momentum at ``hc``.

    Fortran reference: MLCanopyTurbulenceMod.F90::GetPsiRSL (Bonan et al. 2018, appendix
    A2: eqs. A16, A19, A21, 19, 20).
    """
    dt = hc - disp
    zdt_za = (za - hc) / dt
    zdt_hc = 0.0  # (hc - hc) / dt
    dtL = dt / obu

    # Momentum
    phim = phim_monin_obukhov((hc - disp) / obu)
    c1_m = (1.0 - vkc / (2.0 * beta * phim)) * jnp.exp(0.5 * c2)
    psim_hat1 = lookup_psihat(zdt_za, dtL, tables.momentum, table_search) * c1_m
    psim_hat2 = lookup_psihat(zdt_hc, dtL, tables.momentum, table_search) * c1_m
    psim1 = psim_monin_obukhov((za - disp) / obu)
    psim2 = psim_monin_obukhov((hc - disp) / obu)
    psim = -psim1 + psim2 + psim_hat1 - psim_hat2 + vkc / beta

    # Scalars
    phic = phic_monin_obukhov((hc - disp) / obu)
    c1_c = (1.0 - PrSc * vkc / (2.0 * beta * phic)) * jnp.exp(0.5 * c2)
    psic_hat1 = lookup_psihat(zdt_za, dtL, tables.scalar, table_search) * c1_c
    psic_hat2 = lookup_psihat(zdt_hc, dtL, tables.scalar, table_search) * c1_c
    psic1 = psic_monin_obukhov((za - disp) / obu)
    psic2 = psic_monin_obukhov((hc - disp) / obu)
    psic = -psic1 + psic2 + psic_hat1 - psic_hat2

    return psim, psic, psim2, psim_hat2


def psi_rsl_at(za, hc, disp, obu, beta, PrSc, tables: RSLTables, table_search: str = "loop"):
    """:func:`psi_rsl` at many heights (vector ``za``), the other arguments shared."""
    f = functools.partial(psi_rsl, table_search=table_search)
    return jax.vmap(f, in_axes=(0, None, None, None, None, None, None))(
        za, hc, disp, obu, beta, PrSc, tables
    )


def beta_stability(beta_neutral, LcL):
    """``beta = u*/u(h)`` for stability ``Lc/L`` from its neutral value: the root of
    ``beta^4 + 16 Lc/L beta_n^4 beta^2 - beta_n^4 = 0`` (unstable, ``Lc/L <= 0``) or of
    ``5 Lc/L beta^3 + beta - beta_n = 0`` (stable, Cardano's formula).

    Fortran reference: MLCanopyTurbulenceMod.F90::GetBeta (Bonan et al. 2018, eqs. A22-A23).
    Fortran's check of the root is left out (it cannot run inside compiled code).
    """
    # Unstable: beta^2 = (-b + sqrt(b^2 - 4c)) / 2
    bb_u = 16.0 * LcL * beta_neutral**4
    cc_u = -(beta_neutral**4)
    disc_u = jnp.maximum(bb_u * bb_u - 4.0 * cc_u, 1.0e-30)
    beta_u = jnp.sqrt(jnp.maximum((-bb_u + jnp.sqrt(disc_u)) / 2.0, 1.0e-30))

    # Stable: a beta^3 + beta - beta_n = 0 with a = 5 Lc/L (the floor keeps the discarded
    # branch finite for Lc/L <= 0; jnp.maximum avoids the XLA select_divide_fusion bug)
    aa_s = jnp.maximum(5.0 * LcL, 1.0e-10)
    dd_s = -beta_neutral
    qq_s = jnp.sqrt(jnp.maximum((27.0 * aa_s**2 * dd_s) ** 2 + 108.0 * aa_s**3, 1.0e-30))
    rr_s = jnp.maximum(0.5 * (qq_s + 27.0 * aa_s**2 * dd_s), 1e-30) ** (1.0 / 3.0)
    beta_s = -rr_s / (3.0 * aa_s) + 1.0 / rr_s

    return jnp.where(LcL <= 0.0, beta_u, beta_s)


# ---------------------------------------------------------------------------
# Obukhov length
# ---------------------------------------------------------------------------


class ObuInputs(NamedTuple):
    """What the Obukhov length depends on (one patch)."""

    Lc: Array  # canopy density length scale, m
    ztop: Array  # canopy height, m
    lai: Array
    sai: Array
    zref: Array  # reference height, m
    uref: Array  # wind speed at zref, m/s
    thref: Array  # potential temperature at zref, K
    thvref: Array  # virtual potential temperature at zref, K
    qref: Array  # specific humidity at zref, kg/kg
    rhomol: Array  # molar density, mol/m3
    taf: Array  # air temperature at the canopy top, K
    qaf: Array  # specific humidity at the canopy top, kg/kg


class ObuTerms(NamedTuple):
    """The canopy-top quantities that follow from a trial Obukhov length."""

    obu: Array  # Obukhov length, limited to Lc/LcL_max .. Lc/LcL_min, m
    beta: Array  # u*/u(h)
    zdisp: Array  # displacement height, m
    PrSc: Array  # Prandtl number at the canopy top
    ustar: Array  # friction velocity, m/s
    gac_to_hc: Array  # aerodynamic conductance from the canopy top to zref, mol/m2/s
    obu_new: Array  # the Obukhov length these give, m


def _signed_floor(x, eps):
    """``x`` with ``|x|`` at least ``eps``, as ``(sign, |x|)``: dividing by this instead of
    a ``jnp.where`` avoids the XLA select_divide_fusion bug and NaN gradients."""
    return jnp.where(x < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0)), jnp.maximum(jnp.abs(x), eps)


def obukhov_terms(
    obu_val, inp: ObuInputs, tables: RSLTables, sparse_canopy: bool, table_search: str = "loop"
) -> ObuTerms:
    """Beta, displacement height, Prandtl number, u*, T*, q* and the Obukhov length they
    give, for a trial Obukhov length ``obu_val``.

    Fortran reference: MLCanopyTurbulenceMod.F90::ObuFunc (Bonan et al. 2018, eqs. A1-A9,
    Harman 2012 beta). Fortran's check ``zref > zdisp`` is left out (compiled code).
    """
    eps = jnp.asarray(1e-30)
    obu_cur = jnp.where(
        obu_val >= 0.0,
        jnp.maximum(obu_val, inp.Lc / LcL_max),
        jnp.minimum(obu_val, inp.Lc / LcL_min),
    )
    obu_sign, obu_abs = _signed_floor(obu_cur, eps)
    LcL = inp.Lc * obu_sign / obu_abs

    # Neutral beta, and beta with the Harman (2012) blend in unstable conditions
    z0mg_safe = jnp.maximum(z0mg, eps)
    c1_n = (vkc / jnp.log((inp.ztop + z0mg_safe) / z0mg_safe)) ** 2
    beta_neutral = jnp.minimum(
        jnp.sqrt(c1_n + cr * (inp.lai + inp.sai)), jnp.asarray(beta_neutral_max)
    )
    beta_HF = beta_stability(beta_neutral, LcL)
    beta_norsl = beta_stability(jnp.asarray(vkc / 2.0), LcL)
    beta = jnp.where(
        LcL > _AH12_1,
        beta_HF,
        beta_norsl + (beta_HF - beta_norsl) / (1.0 + _AH12_0 * jnp.abs(LcL - _AH12_1) ** _AH12_2),
    )

    # Displacement height
    hc_minus_d = beta**2 * inp.Lc
    if sparse_canopy:
        hc_minus_d = hc_minus_d * (1.0 - jnp.exp(-0.25 * (inp.lai + inp.sai) / beta**2))
    hc_minus_d = jnp.minimum(hc_minus_d, jnp.asarray(inp.ztop))
    zdisp = inp.ztop - hc_minus_d

    PrSc = prandtl_schmidt(beta_neutral, LcL, sparse_canopy)
    psim, psic, _, _ = psi_rsl(inp.zref, inp.ztop, zdisp, obu_cur, beta, PrSc, tables, table_search)

    # u*, T*, q*, and the conductance from the canopy top to zref
    zlog = jnp.log(jnp.maximum((inp.zref - zdisp) / jnp.maximum(inp.ztop - zdisp, eps), eps))
    dm_sign, dm_abs = _signed_floor(zlog + psim, eps)
    dc_sign, dc_abs = _signed_floor(zlog + psic, eps)
    ustar = inp.uref * vkc * dm_sign / dm_abs
    tstar = (inp.thref - inp.taf) * vkc * dc_sign / dc_abs
    qstar = (inp.qref - inp.qaf) * vkc * dc_sign / dc_abs
    gac_to_hc = inp.rhomol * vkc * ustar * dc_sign / dc_abs

    # Obukhov length
    tvstar = tstar + 0.61 * inp.thref * qstar
    tv_sign, tv_abs = _signed_floor(tvstar, eps)
    obu_new = ustar**2 * inp.thvref * tv_sign / (vkc * grav * tv_abs)
    return ObuTerms(obu_cur, beta, zdisp, PrSc, ustar, gac_to_hc, obu_new)


def _obu_residual(obu, inp, tables, sparse_canopy, table_search="loop"):
    """``obu_new(obu) - obu``: zero at the Obukhov length."""
    return obukhov_terms(obu, inp, tables, sparse_canopy, table_search).obu_new - obu


def _secant_step(tables, sparse_canopy, table_search, _, carry):
    """One secant step (``fori_loop`` body; patch inputs in the carry)."""
    x0, x1, f0, f1, inp = carry
    df_sign, df_abs = _signed_floor(f1 - f0, jnp.asarray(1e-30))
    # Identical residuals (f1 == f0, typically both round-off after convergence) leave the
    # secant step undefined: take no step. Dividing by the 1e-30 floor instead throws a
    # converged iterate far away (e.g. Obukhov length 73.6 m -> -128 m).
    dx = jnp.where(f1 - f0 == 0.0, 0.0, -f1 * (x1 - x0) * df_sign / df_abs)
    x_new = x1 + dx
    return x1, x_new, f1, _obu_residual(x_new, inp, tables, sparse_canopy, table_search), inp


def obukhov_length(
    inp: ObuInputs, tables: RSLTables, sparse_canopy: bool, table_search: str = "loop"
):
    """The Obukhov length (m): ``OBU_ITERATIONS`` secant steps from 100 m and -100 m on
    ``obu_new(obu) - obu``. The forward value is the solver's answer; its gradient is the
    implicit-function one, ``-dF/dtheta / dF/dobu`` (``dF/dobu`` by central differences
    with a 1 m step, the correction skipped where ``|dF/dobu| <= 1e-6``), applied as
    ``obu - (step - stop_gradient(step))`` so it never changes the forward value
    (numerics_and_regression.md §2). Differentiating through the iterations instead gives
    exploding gradients (|J|^25, ~1e144 observed).

    Fortran reference: MLCanopyTurbulenceMod.F90::GetObu (Fortran solves with the hybrid
    secant/Brent ``MLMathToolsMod::hybrid``, tol = 0.1 m; 25 fixed secant steps here,
    numerics_and_regression.md §4.1).
    """
    inp = ObuInputs(*(jnp.asarray(v) for v in inp))
    obu0, obu1 = jnp.asarray(100.0), jnp.asarray(-100.0)
    carry = (
        obu0,
        obu1,
        _obu_residual(obu0, inp, tables, sparse_canopy, table_search),
        _obu_residual(obu1, inp, tables, sparse_canopy, table_search),
        inp,
    )
    step = functools.partial(_secant_step, tables, sparse_canopy, table_search)
    obu = jax.lax.stop_gradient(jax.lax.fori_loop(0, OBU_ITERATIONS, step, carry)[1])

    # Implicit-function gradient: the residual's dependence on the inputs, divided by its
    # (constant) slope in obu
    f0 = _obu_residual(obu, inp, tables, sparse_canopy, table_search)
    delta = jnp.asarray(1.0, dtype=obu.dtype)  # obu is O(100-10000 m)
    df_dobu = jax.lax.stop_gradient(
        (
            _obu_residual(obu + delta, inp, tables, sparse_canopy, table_search)
            - _obu_residual(obu - delta, inp, tables, sparse_canopy, table_search)
        )
        / (2.0 * delta)
    )
    apply = jnp.abs(df_dobu) > jnp.asarray(1e-6)
    newton = jnp.where(apply, f0, jnp.zeros_like(f0)) / jnp.where(
        apply, df_dobu, -jnp.ones_like(df_dobu)
    )
    return obu - (newton - jax.lax.stop_gradient(newton))


# ---------------------------------------------------------------------------
# Roughness length, wind profile, conductances
# ---------------------------------------------------------------------------


def _z0m_residual(trial, obu, hc_minus_d, exp1, exp2, psim_hc):
    """Residual whose root is z0m (Fortran ``RoughnessLength``, function ``z0m_func``)."""
    psim_t = psim_monin_obukhov(trial / obu)
    return hc_minus_d * exp1 * jnp.exp(-psim_hc + psim_t) * exp2 - trial


def roughness_length(
    ztop, zdisp, obu, beta, PrSc, zref, tables: RSLTables, table_search: str = "loop"
):
    """Roughness length for momentum (m), by bisection on ``[ztop, 0]`` with the Fortran
    stopping rule (bracket narrower than 0.001 m, at most 20 halvings; the last midpoint).

    Fortran reference: MLCanopyTurbulenceMod.F90::RoughnessLength (Bonan et al. 2018,
    eq. A14 solved for z0m).
    """
    _, _, psim_hc, psim_hat_hc = psi_rsl(zref, ztop, zdisp, obu, beta, PrSc, tables, table_search)
    hc_minus_d = ztop - zdisp
    exp1 = jnp.exp(-vkc / beta)
    exp2 = jnp.exp(psim_hat_hc)
    return bisection_traceable(
        _z0m_residual,
        ztop,
        jnp.asarray(0.0),
        tol=0.001,
        itmax=20,
        c_init=0.0,
        args=(obu, hc_minus_d, exp1, exp2, psim_hc),
    )


def wind_profile(
    layers: PatchLayers, ztop, zdisp, obu, beta, PrSc, ustar, lm_over_beta, tables,
    table_search: str = "loop",
):  # fmt: skip
    """Wind speed (m/s) of every layer slot (layer,) and at the canopy top: the RSL profile
    above the canopy, ``u(h) exp((z - h) / (lm/beta))`` within it.

    Fortran reference: MLCanopyTurbulenceMod.F90::WindProfile (Bonan et al. 2018, eqs. 14,
    A15).
    """
    zs = pad_with_top(layers, layers.zs)

    zs_above = jnp.where(layers.above_canopy, zs, jnp.take(zs, layers.n_layers - 1))
    psim, _, _, _ = psi_rsl_at(zs_above, ztop, zdisp, obu, beta, PrSc, tables, table_search)
    zlog_m = jnp.log(jnp.maximum((zs_above - zdisp) / jnp.maximum(ztop - zdisp, 1e-30), 1e-30))
    wind_above = ustar / vkc * (zlog_m + psim)

    uaf = ustar / jnp.maximum(beta, 1e-30)
    zs_within = jnp.where(layers.in_canopy, zs, ztop)
    wind_within = uaf * jnp.exp((zs_within - ztop) / lm_over_beta)
    return jnp.where(layers.above_canopy, wind_above, wind_within), uaf


#: Aerodynamic conductance between the ground and the lowest layer (model.yaml
#: ``processes.turbulence.ground_conductance``; Fortran ``HF_extension_type`` 1 and 2)
GROUND_CONDUCTANCE = ("extend_hf", "log_profile")


class Conductances(NamedTuple):
    gac: Array  # (layer,) aerodynamic conductance from each layer to the next, mol/m2/s
    gac0: Array  # () from the ground to the lowest layer, mol/m2/s
    kc_eddy: Array  # (layer,) eddy diffusivity, m2/s


def aerodynamic_conductance(
    layers: PatchLayers,
    ztop,
    zdisp,
    obu,
    beta,
    PrSc,
    ustar,
    rhomol,
    zref,
    wind_bottom,
    lm_over_beta,
    tables: RSLTables,
    ground_conductance: str,
    table_search: str = "loop",
) -> Conductances:
    """Aerodynamic conductances of every layer slot: above the canopy from the RSL profile
    (the top layer to ``zref``), within it from the exponential profile, its top layer the
    series of the two (below and above ``ztop``); the ground's by ``ground_conductance``
    (``extend_hf``: the exponential profile down to ``0.1 z0mg``; ``log_profile``: a log
    wind profile to ``z0mg`` with the lowest layer's wind ``wind_bottom`` at least
    0.1 m/s). Resistances are limited to ``ra_max``. Eddy diffusivity from each conductance
    and the distance to the next level.

    Fortran reference: MLCanopyTurbulenceMod.F90::AerodynamicConductance (Bonan et al. 2018,
    eqs. 24-27, A30-A31). Fortran's error checks are left out (compiled code).
    """
    if ground_conductance not in GROUND_CONDUCTANCE:
        raise ValueError(f"ground_conductance {ground_conductance!r} not one of "
                         f"{GROUND_CONDUCTANCE}")  # fmt: skip
    eps = jnp.asarray(1e-30)
    zs = pad_with_top(layers, layers.zs)
    is_top_layer = layers.index == layers.n_layers - 1
    zs_next = jnp.where(is_top_layer, zref, jnp.concatenate([zs[1:], zs[-1:]]))

    # Above the canopy: layer i spans zs(i) to zs(i+1), the top layer to zref
    z_psi = jnp.concatenate([zs, jnp.stack([zref, ztop])])
    _, psic_z, _, _ = psi_rsl_at(z_psi, ztop, zdisp, obu, beta, PrSc, tables, table_search)
    psic_zs, psic_zref, psic_ztop = psic_z[:-2], psic_z[-2], psic_z[-1]
    psic_next = jnp.where(is_top_layer, psic_zref, jnp.concatenate([psic_zs[1:], psic_zs[-1:]]))
    psic_d = psic_next - psic_zs
    zlog_c = jnp.log(jnp.maximum((zs_next - zdisp) / jnp.maximum(zs - zdisp, eps), eps))
    dc_sign, dc_abs = _signed_floor(zlog_c + psic_d, eps)
    gac_above = rhomol * vkc * ustar * dc_sign / dc_abs

    # ztop to the lowest layer above the canopy
    zs_first_above = jnp.take(zs, layers.n_within)
    psic_d = jnp.take(psic_zs, layers.n_within) - psic_ztop
    zlog_c = jnp.log(jnp.maximum((zs_first_above - zdisp) / jnp.maximum(ztop - zdisp, eps), eps))
    dc_sign, dc_abs = _signed_floor(zlog_c + psic_d, eps)
    gac_above_foliage = rhomol * vkc * ustar * dc_sign / dc_abs

    # Within the canopy: layer i spans zs(i) to zs(i+1)
    bu_safe = jnp.maximum(beta * ustar, eps)
    zl = jnp.where(layers.in_canopy, zs, ztop) - ztop
    zu = jnp.where(layers.in_canopy, jnp.concatenate([zs[1:], zs[-1:]]), ztop) - ztop
    res = PrSc / bu_safe * (jnp.exp(-zl / lm_over_beta) - jnp.exp(-zu / lm_over_beta))
    gac_within = rhomol / jnp.maximum(jnp.abs(res), eps)

    # The top layer within the canopy: zs to ztop in series with ztop to the next layer
    zl = jnp.take(zs, layers.n_within - 1) - ztop
    res = PrSc / bu_safe * (jnp.exp(-zl / lm_over_beta) - 1.0)
    gac_below_foliage = rhomol / jnp.maximum(jnp.abs(res), eps)
    gac_top = 1.0 / (1.0 / gac_below_foliage + 1.0 / gac_above_foliage)

    gac = jnp.where(
        layers.above_canopy,
        gac_above,
        jnp.where(layers.index == layers.n_within - 1, gac_top, gac_within),
    )

    # Ground to the lowest layer
    zs1 = layers.zs[0]
    if ground_conductance == "extend_hf":
        zl = 0.1 * z0mg - ztop
        zu = zs1 - ztop
        res = PrSc / bu_safe * (jnp.exp(-zl / lm_over_beta) - jnp.exp(-zu / lm_over_beta))
        gac0 = rhomol / jnp.maximum(jnp.abs(res), eps)
    else:
        zlog_m = jnp.log(jnp.maximum(zs1 / jnp.maximum(z0mg, eps), eps))
        zlog_m_safe = jnp.maximum(jnp.abs(zlog_m), eps)
        ustar_g = jnp.maximum(wind_bottom, 0.1) * vkc / zlog_m_safe
        gac0 = rhomol * vkc * ustar_g / zlog_m_safe

    # Resistances at most ra_max
    gac0 = rhomol / jnp.minimum(rhomol / gac0, ra_max)
    gac = rhomol / jnp.minimum(rhomol / gac, ra_max)
    kc_eddy = gac / rhomol * (zs_next - zs)
    return Conductances(gac=gac, gac0=gac0, kc_eddy=kc_eddy)


# ---------------------------------------------------------------------------
# The process
# ---------------------------------------------------------------------------


class Turbulence(NamedTuple):
    Lc: Array  # () canopy density length scale, m
    taf: Array  # () air temperature at the canopy top, K
    qaf: Array  # () specific humidity at the canopy top, kg/kg
    obu: Array  # () Obukhov length, m
    beta: Array  # () u*/u(h)
    zdisp: Array  # () displacement height, m
    PrSc: Array  # () Prandtl number at the canopy top
    ustar: Array  # () friction velocity, m/s
    gac_to_hc: Array  # () aerodynamic conductance, canopy top to zref, mol/m2/s
    z0m: Array  # () roughness length for momentum, m
    uaf: Array  # () wind speed at the canopy top, m/s
    eta: Array  # () ztop / (lm/beta); Fortran warns if >= eta_max
    wind: Array  # (layer,) wind speed, m/s
    mflx: Array  # (layer,) momentum flux at the layer tops, m2/s2
    gac: Array  # (layer,) aerodynamic conductance to the next level, mol/m2/s
    gac0: Array  # () aerodynamic conductance, ground to the lowest layer, mol/m2/s
    kc_eddy: Array  # (layer,) eddy diffusivity, m2/s


def harman_finnigan(
    layers: PatchLayers,
    tair: Array,
    eair: Array,
    pref,
    ztop,
    lai,
    sai,
    zref,
    uref,
    thref,
    thvref,
    qref,
    rhomol,
    tables: RSLTables,
    *,
    sparse_canopy: bool,
    ground_conductance: str,
    table_search: str = "loop",
) -> Turbulence:
    """Canopy turbulence from the roughness sublayer theory of Harman and Finnigan (2008):
    the canopy-top air (from the layer air temperature ``tair`` and vapour pressure ``eair``,
    (layer,)), the Obukhov length, roughness length, wind and momentum flux profiles and the
    aerodynamic conductances. Forcing at ``zref``: ``pref`` (Pa), ``uref``, ``thref``,
    ``thvref``, ``qref``, ``rhomol``. ``table_search``: how the psihat tables are searched
    (:data:`TABLE_SEARCH`).

    Fortran reference: MLCanopyTurbulenceMod.F90::HF2008 (Bonan et al. 2018).
    Reads: ``tair_profile``, ``eair_profile``, ``pref_forcing``, ``ztop_canopy``,
    ``lai_canopy``, ``sai_canopy``, ``zref_forcing``, ``uref_forcing``, ``thref_forcing``,
    ``thvref_forcing``, ``qref_forcing``, ``rhomol_forcing``, the layer grid. Writes:
    ``Lc_canopy``, ``taf_canopy``, ``qaf_canopy``, ``obu_canopy``, ``beta_canopy``,
    ``zdisp_canopy``, ``PrSc_canopy``, ``ustar_canopy``, ``gac_to_hc_canopy``,
    ``z0m_canopy``, ``uaf_canopy``, ``wind_profile``, ``mflx_profile``, ``gac_profile``,
    ``gac0_soil``, ``kc_eddy_profile``.
    """
    Lc = ztop / (cd * (lai + sai))

    # Temperature and specific humidity at the canopy top
    eair_top = eair[layers.n_within - 1]
    taf = tair[layers.n_within - 1]
    qaf = mmh2o / mmdry * eair_top / (pref - (1.0 - mmh2o / mmdry) * eair_top)

    inp = ObuInputs(Lc, ztop, lai, sai, zref, uref, thref, thvref, qref, rhomol, taf, qaf)
    obu = obukhov_length(inp, tables, sparse_canopy, table_search)
    t = obukhov_terms(obu, inp, tables, sparse_canopy, table_search)
    # Rounded as stored (Fortran's ustar(p), beta(p), ...; the legacy code wrote them to the
    # container): otherwise XLA may fold e.g. (a / b) / beta in u(h) = u*/beta into
    # a / (b beta), 1 ulp off (Phase 7 decision C15)
    t = jax.lax.optimization_barrier(t)
    z0m = roughness_length(ztop, t.zdisp, t.obu, t.beta, t.PrSc, zref, tables, table_search)

    # Mixing length lm and lm/beta
    lm = 2.0 * t.beta**3 * Lc
    eta = t.beta / lm * ztop
    lm_over_beta = ztop / eta

    wind, uaf = wind_profile(
        layers, ztop, t.zdisp, t.obu, t.beta, t.PrSc, t.ustar, lm_over_beta, tables, table_search
    )

    # Momentum flux at the layer tops
    zw = pad_with_top(layers, layers.zw[1:])
    mflx = jnp.where(
        zw > ztop,
        -(t.ustar**2),
        -(t.ustar**2) * jnp.exp(2.0 * (jnp.minimum(zw, ztop) - ztop) / lm_over_beta),
    )

    g = aerodynamic_conductance(
        layers, ztop, t.zdisp, t.obu, t.beta, t.PrSc, t.ustar, rhomol, zref, wind[0],
        lm_over_beta, tables, ground_conductance, table_search,
    )  # fmt: skip
    return Turbulence(
        Lc=Lc, taf=taf, qaf=qaf, obu=t.obu, beta=t.beta, zdisp=t.zdisp, PrSc=t.PrSc,
        ustar=t.ustar, gac_to_hc=t.gac_to_hc, z0m=z0m, uaf=uaf, eta=eta, wind=wind,
        mflx=mflx, gac=g.gac, gac0=g.gac0, kc_eddy=g.kc_eddy,
    )  # fmt: skip


#: Roughness sublayer schemes by name (model.yaml ``processes.turbulence.roughness_sublayer``; Fortran
#: ``turb_type``)
ROUGHNESS_SUBLAYER = {"harman_finnigan": harman_finnigan}
