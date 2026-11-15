"""Leaf photosynthesis and stomatal conductance: Farquhar (C3) or Collatz (C4) photosynthesis
with the stomatal conductance of the Medlyn or Ball-Berry model or of water-use-efficiency
optimization, then reduced for leaf water stress.

One patch. Layers are 0-based slots on axis 0 (``clm_ml.state.PatchLayers``); leaves (0
sunlit, 1 shaded, or any number) on axis 1 of the leaf inputs and of the results: the sunlit
and shaded leaves are one computation over the leaf axis. Layers without plant area get zero
(conductance, rates) except as noted; results in padding slots (``>= n_layers``) are finite
but meaningless.

Iterative solvers (both run a fixed number of steps so they compile; deviations from Fortran
in ``numerics_and_regression.md`` §4.2-4.3): the intercellular CO2 of the Medlyn and
Ball-Berry models by 40 secant steps, the WUE-optimal conductance by 20 bisection halvings.
Each ends with a Newton step whose derivative is held constant, which gives the
implicit-function gradient (and shifts the forward value by the remaining residual; §2).

Fortran reference: CLM-ml v2 ``MLLeafPhotosynthesisMod.F90``. Bonan et al. (2014) Geosci.
Model Dev. 7, 2193-2222; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling,
chapters 11-12; Bonan et al. (2021) Agric. For. Meteorol. 306, 108435, supplemental eq. (14)
(water stress).

Reads and writes are the arguments and results; ``Reads``/``Writes`` in the docstrings name
the CLM-ml v2 (``mlcanopy_type``) fields they correspond to.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from clm_ml.processes.constants import (
    colim_c3a,
    colim_c4a,
    colim_c4b,
    cp25,
    cpha,
    dh2o_to_dco2,
    jmaxha_acclim,
    jmaxha_noacclim,
    jmaxhd_acclim,
    jmaxhd_noacclim,
    jmaxse_noacclim,
    kc25,
    kcha,
    ko25,
    koha,
    phi_psII,
    qe_c4,
    rdha,
    rdhd,
    rdse,
    rgas,
    rh_min_BB,
    tfrz,
    theta_j,
    vcmaxha_acclim,
    vcmaxha_noacclim,
    vcmaxhd_acclim,
    vcmaxhd_noacclim,
    vcmaxse_noacclim,
    vpd_min_MED,
)
from clm_ml.processes.layers import pad_with_top
from clm_ml.processes.math import quadratic
from clm_ml.processes.water_vapor import sat_vap
from clm_ml.state import PatchLayers

#: Stomatal conductance models by name (model.yaml ``processes.stomatal_conductance.scheme``; Fortran
#: ``gs_type`` 0, 1, 2)
STOMATAL_CONDUCTANCE = ("medlyn", "ball_berry", "wue")
#: Root finders of the WUE optimization by name (model.yaml ``processes.stomatal_conductance.wue_solver``; Fortran
#: ``gs_solver`` 1 Brent, 2 bisection). Brent's method is registered but not supported
#: (Phase 7 decision C16, open issue 18 in ``global_input.md``).
WUE_SOLVER = ("bisection", "brent")

#: Secant steps of the intercellular-CO2 solver (Fortran ``hybrid``'s ``itmax``)
CI_ITERATIONS = 40
#: Bisection halvings of the WUE optimization
GS_HALVINGS = 20
#: Upper bracket of the WUE optimization, mol H2O/m2/s (Fortran ``gs2``)
GS_UPPER = 2.0
#: Increase of gs in the marginal water-use-efficiency check, mol H2O/m2/s (Fortran ``delta``)
WUE_DELTA = 0.001

_EPS = 1.0e-30  # floor of denominators (finite gradients in layers without plant area)


# ---------------------------------------------------------------------------
# Temperature responses
# ---------------------------------------------------------------------------


def ft(tl, ha):
    """Arrhenius temperature response, 1 at 25 C, of leaf temperature ``tl`` (K) for the
    activation energy ``ha`` (J/mol).

    Fortran reference: MLLeafPhotosynthesisMod.F90::ft.
    """
    return jnp.exp(ha / (rgas * (tfrz + 25.0)) * (1.0 - (tfrz + 25.0) / tl))


def fth(tl, hd, se, c):
    """High-temperature deactivation of leaf temperature ``tl`` (K) for the deactivation
    energy ``hd`` (J/mol) and entropy term ``se`` (J/mol/K), scaled by ``c`` (:func:`fth25`).

    Fortran reference: MLLeafPhotosynthesisMod.F90::fth.
    """
    return c / (1.0 + jnp.exp((-hd + se * tl) / (rgas * tl)))


def fth25(hd, se):
    """Scaling factor that makes :func:`fth` 1 at 25 C.

    Fortran reference: MLLeafPhotosynthesisMod.F90::fth25.
    """
    t25 = tfrz + 25.0
    return 1.0 + jnp.exp((-hd + se * t25) / (rgas * t25))


def _fth25_float(hd: float, se: float) -> float:
    """:func:`fth25` of constants, as a Python float (a constant of the compiled program)."""
    t25 = tfrz + 25.0
    return 1.0 + math.exp((-hd + se * t25) / (rgas * t25))


class TemperatureParameters(NamedTuple):
    """Activation (``ha``) and deactivation (``hd``) energies, J/mol, entropy terms (``se``),
    J/mol/K, and deactivation scaling factors (``c``) of vcmax, jmax and respiration."""

    vcmaxha: float
    vcmaxhd: float
    vcmaxse: Array | float
    vcmaxc: Array | float
    jmaxha: float
    jmaxhd: float
    jmaxse: Array | float
    jmaxc: Array | float
    rdc: float


def temperature_parameters(acclimation: bool, tacclim) -> TemperatureParameters:
    """The temperature-response parameters, with acclimation to the air temperature
    ``tacclim`` (K; Kattge and Knorr 2007, entropy terms from ``tacclim`` limited to
    11-35 C) or without (then Python floats: constants of the compiled program).

    Fortran reference: MLLeafPhotosynthesisMod.F90::LeafPhotosynthesis (``acclim_type``).
    """
    if acclimation:
        vcmaxha, jmaxha, vcmaxhd, jmaxhd = (
            vcmaxha_acclim, jmaxha_acclim, vcmaxhd_acclim, jmaxhd_acclim
        )  # fmt: skip
        ta_c = jnp.clip(tacclim - tfrz, 11.0, 35.0)
        vcmaxse = 668.39 - 1.07 * ta_c
        jmaxse = 659.70 - 0.75 * ta_c
        vcmaxc = fth25(vcmaxhd, vcmaxse)
        jmaxc = fth25(jmaxhd, jmaxse)
    else:
        vcmaxha, jmaxha, vcmaxhd, jmaxhd = (
            vcmaxha_noacclim, jmaxha_noacclim, vcmaxhd_noacclim, jmaxhd_noacclim
        )  # fmt: skip
        vcmaxse, jmaxse = vcmaxse_noacclim, jmaxse_noacclim
        vcmaxc = _fth25_float(vcmaxhd, vcmaxse)
        jmaxc = _fth25_float(jmaxhd, jmaxse)
    return TemperatureParameters(
        vcmaxha=vcmaxha,
        vcmaxhd=vcmaxhd,
        vcmaxse=vcmaxse,
        vcmaxc=vcmaxc,
        jmaxha=jmaxha,
        jmaxhd=jmaxhd,
        jmaxse=jmaxse,
        jmaxc=jmaxc,
        rdc=_fth25_float(rdhd, rdse),
    )


class LeafRates(NamedTuple):
    """Photosynthetic parameters at leaf temperature."""

    kc: Array  # Michaelis-Menten constant for CO2, umol/mol
    ko: Array  # Michaelis-Menten constant for O2, mmol/mol
    cp: Array  # CO2 compensation point, umol/mol
    vcmax: Array  # maximum carboxylation rate, umol/m2/s
    jmax: Array  # maximum electron transport rate, umol/m2/s
    rd: Array  # respiration, umol CO2/m2/s
    kp: Array  # C4 initial slope of the CO2 response, mol/m2/s (0 for C3)


def leaf_rates(tleaf, vcmax25, jmax25, rd25, kp25, temp: TemperatureParameters, c3: bool):
    """:class:`LeafRates` at leaf temperature ``tleaf`` (K) from their values at 25 C (C4:
    Q10 responses of vcmax, respiration and kp).

    Fortran reference: MLLeafPhotosynthesisMod.F90::LeafPhotosynthesis.
    """
    kc = kc25 * ft(tleaf, kcha)
    ko = ko25 * ft(tleaf, koha)
    cp = cp25 * ft(tleaf, cpha)
    vcmax = vcmax25 * ft(tleaf, temp.vcmaxha) * fth(tleaf, temp.vcmaxhd, temp.vcmaxse, temp.vcmaxc)
    jmax = jmax25 * ft(tleaf, temp.jmaxha) * fth(tleaf, temp.jmaxhd, temp.jmaxse, temp.jmaxc)
    rd = rd25 * ft(tleaf, rdha) * fth(tleaf, rdhd, rdse, temp.rdc)
    kp = jnp.zeros_like(tleaf)
    if not c3:
        t1 = jnp.exp(jnp.log(jnp.asarray(2.0)) * ((tleaf - (tfrz + 25.0)) / 10.0))
        t2 = 1.0 + jnp.exp(jnp.asarray(0.2) * ((tfrz + 15.0) - tleaf))
        t3 = 1.0 + jnp.exp(jnp.asarray(0.3) * (tleaf - (tfrz + 40.0)))
        t4 = 1.0 + jnp.exp(jnp.asarray(1.3) * (tleaf - (tfrz + 55.0)))
        vcmax = vcmax25 * t1 / (t2 * t3)
        rd = rd25 * t1 / t4
        kp = kp25 * t1
    return LeafRates(kc=kc, ko=ko, cp=cp, vcmax=vcmax, jmax=jmax, rd=rd, kp=kp)


def electron_transport(apar, jmax):
    """Electron transport rate, umol/m2/s, for absorbed PAR ``apar`` (umol photons/m2/s): the
    smaller root of ``theta_j J^2 - (I + jmax) J + I jmax = 0`` with ``I = 0.5 phi_psII apar``.

    Fortran reference: MLLeafPhotosynthesisMod.F90::LeafPhotosynthesis.
    """
    qabs = 0.5 * phi_psII * apar
    r1, r2 = quadratic(theta_j, -(qabs + jmax), qabs * jmax)
    return jnp.minimum(r1, r2)


def realized_rate(ac, aj, ap, *, c3: bool, colimitation: bool):
    """Gross photosynthesis, umol CO2/m2/s, from the Rubisco- (``ac``), RuBP-regeneration-
    (``aj``) and product- or CO2-limited (``ap``) rates: their minimum (C3: of ``ac``,
    ``aj``), or with ``colimitation`` the smaller roots of the co-limitation quadratics
    (``ac`` with ``aj``, then, C4, the result with ``ap``).

    Fortran reference: MLLeafPhotosynthesisMod.F90::RealizedRate.
    """
    if not colimitation:
        return jnp.minimum(ac, aj) if c3 else jnp.minimum(jnp.minimum(ac, aj), ap)
    r1, r2 = quadratic(colim_c3a if c3 else colim_c4a, -(ac + aj), ac * aj)
    ai = jnp.minimum(r1, r2)
    if c3:
        return ai
    r1, r2 = quadratic(colim_c4b, -(ai + ap), ai * ap)
    return jnp.minimum(r1, r2)


# ---------------------------------------------------------------------------
# Photosynthesis for a given intercellular CO2 or stomatal conductance
# ---------------------------------------------------------------------------


class Biochemistry(NamedTuple):
    """What photosynthesis depends on besides CO2 and conductance (leaf rates at leaf
    temperature, electron transport, absorbed PAR, O2)."""

    rates: LeafRates
    je: Array
    apar: Array
    o2ref: Array


def _gross_rates_at_ci(ci, b: Biochemistry, c3: bool):
    """``(ac, aj, ap)`` for the intercellular CO2 ``ci`` (umol/mol)."""
    r = b.rates
    if c3:
        ac = r.vcmax * jnp.maximum(ci - r.cp, 0.0) / (ci + r.kc * (1.0 + b.o2ref / r.ko))
        aj = b.je * jnp.maximum(ci - r.cp, 0.0) / (4.0 * ci + 8.0 * r.cp)
        ap = jnp.zeros_like(ci)
    else:
        ac = jnp.broadcast_to(r.vcmax, jnp.shape(ci))
        aj = jnp.broadcast_to(qe_c4 * b.apar, jnp.shape(ci))
        ap = r.kp * jnp.maximum(ci, 0.0)
    return ac, aj, ap


def _net_rates_at_ci(ci, b: Biochemistry, c3: bool, colimitation: bool):
    """``(ac, aj, ap, agross, anet)`` for the intercellular CO2 ``ci``, rates at least 0."""
    ac, aj, ap = _gross_rates_at_ci(ci, b, c3)
    agross = realized_rate(ac, aj, ap, c3=c3, colimitation=colimitation)
    ac = jnp.maximum(ac, 0.0)
    aj = jnp.maximum(aj, 0.0)
    ap = jnp.maximum(ap, 0.0)
    agross = jnp.maximum(agross, 0.0)
    return ac, aj, ap, agross, agross - b.rates.rd


class AtGs(NamedTuple):
    """Photosynthesis for a given stomatal conductance."""

    ci: Array  # intercellular CO2, umol/mol
    ac: Array
    aj: Array
    ap: Array
    agross: Array
    anet: Array
    cs: Array  # CO2 at the leaf surface, umol/mol


def photosynthesis_at_gs(gs, gbc, cair, b: Biochemistry, c3: bool, colimitation: bool) -> AtGs:
    """Photosynthesis for the stomatal conductance ``gs`` (mol H2O/m2/s), boundary layer
    conductance ``gbc`` (mol CO2/m2/s) and air CO2 ``cair`` (umol/mol): the rates are the
    roots of the photosynthesis equations combined with the diffusion equation
    ``An = (cair - ci) gleaf``.

    Fortran reference: MLLeafPhotosynthesisMod.F90::CiFuncGs.
    """
    r = b.rates
    gbc_safe = jnp.maximum(gbc, _EPS)
    gs_safe = jnp.maximum(gs, _EPS)
    gleaf = 1.0 / (1.0 / gbc_safe + dh2o_to_dco2 / gs_safe)
    if c3:
        a0 = r.vcmax
        b0 = r.kc * (1.0 + b.o2ref / jnp.maximum(r.ko, _EPS))
        aq = 1.0 / gleaf
        bq = -(cair + b0) - (a0 - r.rd) / gleaf
        cq = a0 * (cair - r.cp) - r.rd * (cair + b0)
        r1, r2 = quadratic(aq, bq, cq)
        ac = jnp.minimum(r1, r2) + r.rd

        a0j = b.je / 4.0
        b0j = 2.0 * r.cp
        aqj = 1.0 / gleaf
        bqj = -(cair + b0j) - (a0j - r.rd) / gleaf
        cqj = a0j * (cair - r.cp) - r.rd * (cair + b0j)
        r1j, r2j = quadratic(aqj, bqj, cqj)
        aj = jnp.minimum(r1j, r2j) + r.rd
        ap = jnp.zeros_like(ac)
    else:
        ac = r.vcmax
        aj = qe_c4 * b.apar
        ap = r.kp * (cair * gleaf + r.rd) / (gleaf + r.kp)
    agross = realized_rate(ac, aj, ap, c3=c3, colimitation=colimitation)
    anet = agross - r.rd
    cs = jnp.maximum(cair - anet / gbc_safe, 1.0)
    ci = cair - anet / gleaf
    return AtGs(ci=ci, ac=ac, aj=aj, ap=ap, agross=agross, anet=anet, cs=cs)


# ---------------------------------------------------------------------------
# Medlyn and Ball-Berry: intercellular CO2 by a secant solver
# ---------------------------------------------------------------------------


class _Leaf(NamedTuple):
    """Leaf and air conditions of the stomatal models."""

    gbc: Array
    gbv: Array
    cair: Array
    eair: Array
    ceair: Array  # vapor pressure of the air constrained for the stomatal models, Pa
    leaf_esat: Array  # saturation vapor pressure at leaf temperature, Pa


def _ci_residual(ci, leaf: _Leaf, b, g0, g1, model, c3, colimitation):
    """Change of ``ci`` after one pass of photosynthesis, the stomatal model and diffusion
    (zero at the solution; zero where net photosynthesis is negative).

    Fortran reference: MLLeafPhotosynthesisMod.F90::CiFunc.
    """
    ac, aj, ap, agross, anet = _net_rates_at_ci(ci, b, c3, colimitation)
    gbc_safe = jnp.maximum(leaf.gbc, _EPS)
    cs = jnp.maximum(leaf.cair - anet / gbc_safe, 1.0)
    if model == "ball_berry":
        term = anet / cs
        bq = leaf.gbv - g0 - g1 * term
        lesat_safe = jnp.maximum(leaf.leaf_esat, _EPS)
        cq = -leaf.gbv * (g0 + g1 * term * leaf.ceair / lesat_safe)
        r1, r2 = quadratic(1.0, bq, cq)
        gs = jnp.where(anet > 0.0, jnp.maximum(r1, r2), g0)
    else:  # medlyn
        vpd_term = jnp.maximum(leaf.leaf_esat - leaf.ceair, vpd_min_MED) * 0.001
        term = dh2o_to_dco2 * anet / cs
        gbv_safe = jnp.maximum(leaf.gbv, _EPS)
        bq = -(2.0 * (g0 + term) + (g1 * term) ** 2 / (gbv_safe * vpd_term))
        cq = g0 * g0 + (2.0 * g0 + term * (1.0 - g1 * g1 / vpd_term)) * term
        r1, r2 = quadratic(1.0, bq, cq)
        gs = jnp.where(anet > 0.0, jnp.maximum(r1, r2), g0)
    gleaf = 1.0 / (1.0 / gbc_safe + dh2o_to_dco2 / jnp.maximum(gs, _EPS))
    ci_dif = leaf.cair - anet / gleaf - ci
    return jnp.where(anet < 0.0, 0.0, ci_dif)


def _secant(f, x0, x1, n_iter):
    """``n_iter`` secant steps from ``x0``, ``x1`` (no step where two residuals are equal:
    ``numerics_and_regression.md`` §5)."""
    f0 = f(x0)
    f1 = f(x1)

    def body(carry, _):
        x0, x1, f0, f1 = carry
        # sign * max(|df|, eps): no select as a denominator (XLA select_divide_fusion)
        df = f1 - f0
        df_abs_safe = jnp.maximum(jnp.abs(df), jnp.asarray(_EPS))
        df_sign = jnp.where(df < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
        dx = jnp.where(df == 0.0, 0.0, -f1 * (x1 - x0) * df_sign / df_abs_safe)
        x_new = x1 + dx
        return (x1, x_new, f1, f(x_new)), None

    (_, x, _, _), _ = jax.lax.scan(body, (x0, x1, f0, f1), None, length=n_iter)
    return x


def solve_ci(f, x0, x1, n_iter: int = CI_ITERATIONS):
    """The root of ``f`` by :func:`_secant`, then one Newton step with the derivative (central
    difference, 1e-4) held constant: the implicit-function gradient (where the derivative
    exceeds 1e-8 in magnitude).

    Fortran reference: MLMathToolsMod.F90::hybrid (tol = 0.1; here fixed steps,
    ``numerics_and_regression.md`` §4.3).
    """
    x0 = jax.lax.stop_gradient(_secant(f, x0, x1, n_iter))
    fx = f(x0)
    delta = jnp.asarray(1e-4, dtype=x0.dtype)
    dfdx = jax.lax.stop_gradient((f(x0 + delta) - f(x0 - delta)) / (2.0 * delta))
    apply = jnp.abs(dfdx) > jnp.asarray(1e-8, dtype=x0.dtype)
    safe_f = jnp.where(apply, fx, jnp.zeros_like(fx))
    safe_denom = jnp.where(apply, dfdx, jnp.ones_like(dfdx))
    return x0 - safe_f / safe_denom


def _ci_model(leaf: _Leaf, b: Biochemistry, g0, g1, model, c3, colimitation):
    """Medlyn or Ball-Berry: ``(gs, AtGs)`` without water stress.

    Fortran reference: MLLeafPhotosynthesisMod.F90::LeafPhotosynthesis (``gs_type`` 0, 1:
    ``hybrid`` with ``CiFunc``).
    """
    ci0 = (0.7 if c3 else 0.4) * leaf.cair
    ci1 = ci0 * 0.99

    def f(ci):
        return _ci_residual(ci, leaf, b, g0, g1, model, c3, colimitation)

    ci = solve_ci(f, ci0, ci1)
    ac, aj, ap, agross, anet = _net_rates_at_ci(ci, b, c3, colimitation)
    cs = jnp.maximum(leaf.cair - anet / leaf.gbc, 1.0)
    if model == "ball_berry":
        term = anet / cs
        bq = leaf.gbv - g0 - g1 * term
        lesat_safe = jnp.maximum(leaf.leaf_esat, _EPS)
        cq = -leaf.gbv * (g0 + g1 * term * leaf.ceair / lesat_safe)
        r1, r2 = quadratic(1.0, bq, cq)
        gs = jnp.where(anet > 0.0, jnp.maximum(r1, r2), g0)
    else:  # medlyn
        vpd_term = jnp.maximum(leaf.leaf_esat - leaf.ceair, vpd_min_MED) * 0.001
        term = dh2o_to_dco2 * anet / cs
        gbv_vpd = jnp.maximum(leaf.gbv * vpd_term, _EPS)
        bq = -(2.0 * (g0 + term) + (g1 * term) ** 2 / gbv_vpd)
        cq = g0 * g0 + (2.0 * g0 + term * (1.0 - g1 * g1 / vpd_term)) * term
        r1, r2 = quadratic(1.0, bq, cq)
        gs = jnp.where(anet > 0.0, jnp.maximum(r1, r2), g0)
    return gs, AtGs(ci=ci, ac=ac, aj=aj, ap=ap, agross=agross, anet=anet, cs=cs)


# ---------------------------------------------------------------------------
# Water-use-efficiency optimization: stomatal conductance by bisection
# ---------------------------------------------------------------------------


def stomata_efficiency(gs, leaf: _Leaf, b: Biochemistry, iota, pref, c3, colimitation):
    """Marginal water-use-efficiency check: the increase of net photosynthesis from
    ``gs - delta`` to ``gs`` minus ``iota delta vpd / pref`` (positive while opening the
    stomata gains more carbon than the water-use efficiency ``iota`` requires; vpd at the leaf
    surface, at least ``vpd_min_MED``).

    Fortran reference: MLLeafPhotosynthesisMod.F90::StomataEfficiency (Bonan et al. 2014,
    eqs. A16-A18).
    """
    an_low = photosynthesis_at_gs(gs - WUE_DELTA, leaf.gbc, leaf.cair, b, c3, colimitation).anet
    an_high = photosynthesis_at_gs(gs, leaf.gbc, leaf.cair, b, c3, colimitation).anet
    lesat_safe = jnp.maximum(leaf.leaf_esat, _EPS)
    denom = jnp.maximum((leaf.gbv + gs) * lesat_safe, _EPS)
    hs = (leaf.gbv * leaf.eair + gs * lesat_safe) / denom
    vpd = jnp.maximum(lesat_safe - hs * lesat_safe, vpd_min_MED)
    return (an_high - an_low) - iota * WUE_DELTA * (vpd / jnp.maximum(pref, _EPS))


def solve_gs_bisection(f, gs_min, shape, n_iter: int = GS_HALVINGS):
    """WUE-optimal stomatal conductance (array of ``shape``): where ``f`` changes sign between
    ``gs_min`` and :data:`GS_UPPER`, the midpoint after ``n_iter`` halvings, then one Newton
    step with the derivative (central difference, 1e-4) held constant (the implicit-function
    gradient, where the derivative exceeds 1e-6 in magnitude); elsewhere (low light)
    ``gs_min``, whose gradient is that of ``gs_min`` (decision 119).

    Fortran reference: MLLeafPhotosynthesisMod.F90::StomataOptimization and
    MLMathToolsMod.F90::bisection (tol = 0.001; here fixed halvings,
    ``numerics_and_regression.md`` §4.2).
    """
    fa = f(gs_min)
    fb = f(GS_UPPER)
    bracket_ok = fa * fb < 0.0

    def body(_, carry):
        a, b, f_a = carry
        mid = 0.5 * (a + b)
        f_mid = f(mid)
        same_sign = f_a * f_mid > 0.0  # root in [mid, b]
        return (jnp.where(same_sign, mid, a), jnp.where(same_sign, b, mid),
                jnp.where(same_sign, f_mid, f_a))  # fmt: skip

    init = (jnp.broadcast_to(gs_min, shape), jnp.full(shape, GS_UPPER), fa)
    a, b, _ = jax.lax.fori_loop(0, n_iter, body, init)
    # Without a root the conductance is gs_min itself, with its gradient (decision 119)
    gs0 = jnp.where(bracket_ok, jax.lax.stop_gradient(0.5 * (a + b)), gs_min)

    f0 = f(gs0)
    delta = jnp.asarray(1e-4, dtype=gs0.dtype)
    dfdgs = jax.lax.stop_gradient((f(gs0 + delta) - f(gs0 - delta)) / (2.0 * delta))
    # Only near a bracketed root: at gs_min without a root, f can be O(0.01) and the Newton
    # step would extrapolate far away
    apply = bracket_ok & (jnp.abs(dfdgs) > jnp.asarray(1e-6, dtype=gs0.dtype))
    safe_f0 = jnp.where(apply, f0, jnp.zeros_like(f0))
    safe_denom = jnp.where(apply, dfdgs, jnp.ones_like(dfdgs))
    return gs0 - safe_f0 / safe_denom


def brent(*args, **kwargs):
    """Brent's method for the WUE optimization: registered but not supported (Phase 7
    decision C16; open issue 18 in ``global_input.md``). The JAX translation never had a
    traceable Brent (``wue_solver: brent`` silently ran the bisection until Phase 7.2).

    Fortran reference: MLLeafPhotosynthesisMod.F90::StomataOptimization (``gs_solver = 1``,
    ``MLMathToolsMod.F90::zbrent``).
    """
    raise NotImplementedError(
        "photosynthesis: wue_solver brent is not supported (Phase 7 decision C16; see the "
        "plan, 'After Phase 8: unsupported variants'); use bisection"
    )


def _wue_model(leaf: _Leaf, b: Biochemistry, iota, gs_min, pref, c3, colimitation, wue_solver):
    """WUE optimization: ``(gs, AtGs)`` without water stress.

    Fortran reference: MLLeafPhotosynthesisMod.F90::StomataOptimization.
    """
    if wue_solver != "bisection":
        brent()

    def f(gs):
        return stomata_efficiency(gs, leaf, b, iota, pref, c3, colimitation)

    gs = solve_gs_bisection(f, gs_min, jnp.shape(leaf.cair))
    return gs, photosynthesis_at_gs(gs, leaf.gbc, leaf.cair, b, c3, colimitation)


# ---------------------------------------------------------------------------
# The process
# ---------------------------------------------------------------------------


class Photosynthesis(NamedTuple):
    """Leaf photosynthesis and stomatal conductance (layer, leaf); zero in layers without
    plant area unless noted."""

    kc: Array
    ko: Array
    cp: Array
    vcmax: Array
    jmax: Array
    rd: Array
    kp: Array
    leaf_esat: Array  # saturation vapor pressure at leaf temperature, Pa
    ceair: Array  # vapor pressure of the air constrained for the stomatal models, Pa
    je: Array  # electron transport rate, umol/m2/s
    gspot: Array  # stomatal conductance without water stress, mol H2O/m2/s
    gs: Array  # stomatal conductance, mol H2O/m2/s
    ci: Array  # intercellular CO2, umol/mol
    ac: Array  # Rubisco-limited gross photosynthesis, umol CO2/m2/s
    aj: Array  # RuBP-regeneration-limited gross photosynthesis, umol CO2/m2/s
    ap: Array  # product- (C3) or CO2-limited (C4) gross photosynthesis, umol CO2/m2/s
    agross: Array  # gross photosynthesis, umol CO2/m2/s
    anet: Array  # net photosynthesis, umol CO2/m2/s
    cs: Array  # CO2 at the leaf surface, umol/mol
    hs: Array  # fractional humidity at the leaf surface
    vpd: Array  # vapor pressure deficit at the leaf surface, Pa


def leaf_photosynthesis(
    layers: PatchLayers,
    dpai: Array,
    eair: Array,
    cair: Array,
    tleaf: Array,
    vcmax25: Array,
    jmax25: Array,
    rd25: Array,
    kp25: Array,
    apar: Array,
    gbc: Array,
    gbv: Array,
    lwp: Array,
    o2ref,
    pref,
    tacclim,
    *,
    iota,
    gs_min,
    psi50_gs,
    shape_gs,
    g0=None,
    g1=None,
    c3: bool,
    stomatal_conductance: str = "wue",
    water_stress: bool = True,
    colimitation: bool = True,
    acclimation: bool = True,
    wue_solver: str = "bisection",
) -> Photosynthesis:
    """Photosynthesis and stomatal conductance (layer, leaf) of the leaves of every layer.

    First the unstressed conductance ``gspot`` and photosynthesis by the
    ``stomatal_conductance`` model (one of :data:`STOMATAL_CONDUCTANCE`; WUE by
    ``wue_solver``, one of :data:`WUE_SOLVER`); then, with ``water_stress``, the conductance
    reduced by leaf water potential ``lwp`` (MPa), ``gs = max(gspot / (1 + (lwp /
    psi50_gs)^shape_gs), gs_min)`` (no reduction for ``lwp >= 0``), and photosynthesis recomputed for it, with the humidity
    and vapor pressure deficit at the leaf surface. Soil water does not limit vcmax
    (Fortran's ``btran`` is 1).

    Inputs per layer: ``dpai`` (m2/m2), ``eair`` (Pa), ``cair`` (umol/mol); per layer and
    leaf: ``tleaf`` (K), the 25 C values ``vcmax25``, ``jmax25``, ``rd25``, ``kp25``,
    ``apar`` (umol photons/m2/s), boundary layer conductances ``gbc``, ``gbv``
    (mol/m2/s), ``lwp``; ``o2ref`` (mmol/mol), ``pref`` (Pa), the acclimation temperature
    ``tacclim`` (K). The PFT's parameters: WUE ``iota`` (umol CO2/mol H2O), minimum
    conductance ``gs_min`` (mol H2O/m2/s), water-stress ``psi50_gs`` (MPa) and ``shape_gs``,
    and for the Medlyn and Ball-Berry models their ``g0``, ``g1``
    (:func:`stomatal_parameters`). Padding slots take the top layer's inputs.

    Fortran reference: MLLeafPhotosynthesisMod.F90::LeafPhotosynthesis (called once per
    leaf).
    Reads: ``dpai_profile``, ``eair_profile``, ``cair_profile``, ``tleaf_leaf``,
    ``vcmax25_leaf``, ``jmax25_leaf``, ``rd25_leaf``, ``kp25_leaf``, ``apar_leaf``,
    ``gbc_leaf``, ``gbv_leaf``, ``lwp_leaf``, ``o2ref_forcing``, ``pref_forcing``,
    ``tacclim_forcing``, the PFT's ``iota_SPA``, ``gsmin_SPA``, ``psi50_gs``, ``shape_gs``,
    ``g0_MED``/``g1_MED`` or ``g0_BB``/``g1_BB``.
    Writes: the ``*_leaf`` fields of :class:`Photosynthesis`; ``g0_canopy``, ``g1_canopy``
    (Medlyn, Ball-Berry: the model's parameters) and ``btran_soil`` (1) by the adapter.
    """
    if stomatal_conductance not in STOMATAL_CONDUCTANCE:
        raise ValueError(
            f"photosynthesis: stomatal_conductance {stomatal_conductance!r} not one of "
            f"{STOMATAL_CONDUCTANCE}"
        )
    if wue_solver not in WUE_SOLVER:
        raise ValueError(f"photosynthesis: wue_solver {wue_solver!r} not one of {WUE_SOLVER}")

    def _layer(x):  # per layer -> (layer, leaf)
        return jnp.broadcast_to(pad_with_top(layers, x)[:, None], tleaf.shape)

    def _leaf(x):
        return pad_with_top(layers, x)

    dpai, eair, cair = _layer(dpai), _layer(eair), _layer(cair)
    tleaf, vcmax25, jmax25, rd25, kp25, apar, gbc, gbv, lwp = (
        _leaf(x) for x in (tleaf, vcmax25, jmax25, rd25, kp25, apar, gbc, gbv, lwp)
    )
    active = dpai > 0.0

    # --- Unstressed conductance and photosynthesis ---------------------------------------
    # Layers without plant area take 1 (finite arithmetic; results zeroed below)
    gbc1 = jnp.where(active, gbc, 1.0)
    gbv1 = jnp.where(active, gbv, 1.0)
    cair1 = jnp.where(active, cair, 1.0)
    eair1 = jnp.where(active, eair, 1.0)
    apar1 = jnp.where(active, apar, 1.0)

    temp = temperature_parameters(acclimation, tacclim)
    rates = leaf_rates(tleaf, vcmax25, jmax25, rd25, kp25, temp, c3)
    leaf_esat, _ = sat_vap(tleaf)
    ceair = jnp.minimum(eair1, leaf_esat)
    if stomatal_conductance == "ball_berry":
        # hs stays above zero at low humidity
        ceair = jnp.maximum(ceair, rh_min_BB * leaf_esat)
    je = electron_transport(apar1, rates.jmax)
    bio = Biochemistry(rates=rates, je=je, apar=apar1, o2ref=o2ref)
    leaf = _Leaf(gbc=gbc1, gbv=gbv1, cair=cair1, eair=eair1, ceair=ceair, leaf_esat=leaf_esat)
    if stomatal_conductance == "wue":
        gs, at = _wue_model(leaf, bio, iota, gs_min, pref, c3, colimitation, wue_solver)
    else:
        gs, at = _ci_model(leaf, bio, g0, g1, stomatal_conductance, c3, colimitation)

    def _zero(x):
        return jnp.where(active, x, 0.0)

    rates = LeafRates(*(_zero(x) for x in rates))
    leaf_esat, ceair, je, gspot = _zero(leaf_esat), _zero(ceair), _zero(je), _zero(gs)
    at = AtGs(*(_zero(x) for x in at))

    # --- Water stress (Bonan et al. 2021, supplemental eq. 14) ----------------------------
    if water_stress:
        # no stress at or above zero water potential (dew makes it slightly positive): the
        # power of a negative base has no derivative in the exponent, and no value for an
        # exponent that is not a whole number (decision 121)
        fpsi = 1.0 / (1.0 + jnp.maximum(lwp / psi50_gs, 0.0) ** shape_gs)
    else:
        fpsi = jnp.ones_like(gspot)
    gs_new = jnp.maximum(gspot * fpsi, gs_min)
    stressed = _stressed_photosynthesis(gs_new, gbc, cair, rates, je, apar, o2ref, c3,
                                        colimitation)  # fmt: skip
    hs_denom = jnp.maximum((gbv + gs_new) * leaf_esat, _EPS)
    hs = (gbv * eair + gs_new * leaf_esat) / hs_denom
    vpd = jnp.maximum(leaf_esat - hs * leaf_esat, 0.1)

    def _stressed(new, old):
        return jnp.where(active, new, old)

    return Photosynthesis(
        *rates,
        leaf_esat=leaf_esat,
        ceair=ceair,
        je=je,
        gspot=gspot,
        gs=_stressed(gs_new, gspot),
        ci=_stressed(stressed.ci, at.ci),
        ac=_stressed(stressed.ac, at.ac),
        aj=_stressed(stressed.aj, at.aj),
        ap=_stressed(stressed.ap, at.ap),
        agross=_stressed(stressed.agross, at.agross),
        anet=_stressed(stressed.anet, at.anet),
        cs=_stressed(stressed.cs, at.cs),
        hs=_zero(hs),
        vpd=_zero(vpd),
    )


def stomatal_parameters(params, model: str):
    """``(g0, g1)`` of the Medlyn or Ball-Berry model from the PFT's ``params``.

    Fortran reference: MLLeafPhotosynthesisMod.F90::LeafPhotosynthesis (``g0``, ``g1``).
    """
    if model == "medlyn":
        return params.g0_medlyn, params.g1_medlyn
    return params.g0_ball_berry, params.g1_ball_berry


def _stressed_photosynthesis(gs, gbc, cair, r: LeafRates, je, apar, o2ref, c3, colimitation):
    """Photosynthesis for the water-stressed conductance ``gs`` (:func:`photosynthesis_at_gs`
    with the leaf rates of the layers without plant area at zero).

    Fortran reference: MLLeafPhotosynthesisMod.F90::CiFuncGs.
    """
    gs_safe = jnp.maximum(gs, _EPS)
    gbc_safe = jnp.maximum(gbc, _EPS)
    gleaf = 1.0 / (1.0 / gbc_safe + dh2o_to_dco2 / gs_safe)
    if c3:
        # ko is 0 in layers without plant area
        b0 = r.kc * (1.0 + o2ref / jnp.maximum(r.ko, jnp.asarray(_EPS)))
        bq = -(cair + b0) - (r.vcmax - r.rd) / gleaf
        cq = r.vcmax * (cair - r.cp) - r.rd * (cair + b0)
        r1, r2 = quadratic(1.0 / gleaf, bq, cq)
        ac = jnp.minimum(r1, r2) + r.rd
        b0j = 2.0 * r.cp
        bqj = -(cair + b0j) - (je / 4.0 - r.rd) / gleaf
        cqj = (je / 4.0) * (cair - r.cp) - r.rd * (cair + b0j)
        r1j, r2j = quadratic(1.0 / gleaf, bqj, cqj)
        aj = jnp.minimum(r1j, r2j) + r.rd
        ap = jnp.zeros_like(ac)
    else:
        ac = r.vcmax
        aj = qe_c4 * apar
        ap = r.kp * (cair * gleaf + r.rd) / (gleaf + r.kp)
    agross = realized_rate(ac, aj, ap, c3=c3, colimitation=colimitation)
    anet = agross - r.rd
    cs = jnp.maximum(cair - anet / gbc_safe, 1.0)
    ci = cair - anet / gleaf
    return AtGs(ci=ci, ac=ac, aj=aj, ap=ap, agross=agross, anet=anet, cs=cs)
