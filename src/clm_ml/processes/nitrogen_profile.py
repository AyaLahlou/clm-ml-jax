"""Canopy profile of leaf nitrogen and photosynthetic capacity: Vcmax, Jmax, leaf respiration
and the C4 CO2 slope at 25 C decline exponentially with the plant area above, from their
canopy-top values (Bonan et al. 2014, eqs. A1-A2; Bonan et al. 2021, supplement eqs. 6-12).

One patch. Layers are 0-based slots on axis 0; leaves 0 sunlit, 1 shaded on axis 1. Every
slot is computed; slots without plant area (``dpai == 0``) and the padding get 0.

Fortran reference: CLM-ml v2 ``MLCanopyNitrogenProfileMod.F90``.

Reads and writes are the arguments and results; ``Reads``/``Writes`` in the docstrings name
the CLM-ml v2 (``mlcanopy_type``) fields they correspond to.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from clm_ml.processes.constants import (
    jmax25_to_vcmax25_noacclim,
    kp25_to_vcmax25_c4,
    rd25_to_vcmax25_c3,
    rd25_to_vcmax25_c4,
    tfrz,
)
from clm_ml.state import PatchLayers


class TopRates(NamedTuple):
    """Canopy-top leaf rates at 25 C."""

    vcmax25: Array  # maximum carboxylation rate, umol/m2/s
    jmax25: Array  # C3 maximum electron transport rate, umol/m2/s (0 for C4)
    rd25: Array  # leaf respiration rate, umol CO2/m2/s
    kp25: Array  # C4 initial slope of the CO2 response curve, mol/m2/s (0 for C3)


def canopy_top_rates(vcmax25top, tacclim, *, c3: bool, acclimation: bool) -> TopRates:
    """Canopy-top rates from the PFT's ``vcmax25top``: Jmax/Vcmax is a constant, or with
    temperature acclimation ``2.59 - 0.035 clamp(tacclim - tfrz, 11, 35)`` (C).

    Fortran reference: MLCanopyNitrogenProfileMod.F90::CanopyNitrogenProfile.
    Reads: ``vcmaxpft`` (parameters), ``tacclim_forcing``.
    """
    if acclimation:
        j2v = 2.59 - 0.035 * jnp.clip(tacclim - tfrz, 11.0, 35.0)
    else:
        j2v = jmax25_to_vcmax25_noacclim
    if c3:
        return TopRates(vcmax25top, j2v * vcmax25top, rd25_to_vcmax25_c3 * vcmax25top, 0.0)
    return TopRates(
        vcmax25top, 0.0, rd25_to_vcmax25_c4 * vcmax25top, kp25_to_vcmax25_c4 * vcmax25top
    )


def decay_coefficient(vcmax25top, kn_val: float):
    """Leaf nitrogen decay coefficient: ``exp(0.00963 vcmax25top - 2.43)`` when ``kn_val``
    is negative (the default), else ``kn_val``.

    Fortran reference: MLCanopyNitrogenProfileMod.F90::CanopyNitrogenProfile.
    """
    if kn_val < 0.0:
        return jnp.exp(0.00963 * vcmax25top - 2.43)
    if kn_val > 0.0:
        return jnp.asarray(kn_val)
    raise ValueError("decay_coefficient: kn_val must not be 0")


def nitrogen_fractions(kn, pai_above, dpai, kb, tbi, clump, *, vary_with_height: bool):
    """Nitrogen integrated over each layer and its sunlit part (elementwise):
    ``fn = exp(-kn pai_above) (1 - exp(-kn dpai)) / kn`` and, with ``vary_with_height``
    false (Fortran ``leaf_optics_type = 0``),
    ``fn_sun = clump / (kn + kb clump) exp(-kn pai_above) tbi (1 - exp(-(kn + kb clump) dpai))``
    (``None`` otherwise). The denominator has a 1e-30 floor.

    Fortran reference: MLCanopyNitrogenProfileMod.F90::CanopyNitrogenProfile.
    """
    exp_kn_above = jnp.exp(-kn * pai_above)
    fn = exp_kn_above * (1.0 - jnp.exp(-kn * dpai)) / kn
    if vary_with_height:
        return fn, None
    denom_safe = jnp.maximum(kn + kb * clump, 1.0e-30)  # jnp.maximum: no select_divide_fusion
    fn_sun = clump / denom_safe * exp_kn_above * tbi * (1.0 - jnp.exp(-denom_safe * dpai))
    return fn, fn_sun


def nitrogen_scale(fn, fn_sun, dpai, fracsun, *, vary_with_height: bool):
    """Nitrogen scaling factors of the sunlit and shaded leaves (elementwise) from
    :func:`nitrogen_fractions`: ``nscale_sun = fn_sun / (fracsun dpai)``,
    ``nscale_sha = (fn - fn_sun) / ((1 - fracsun) dpai)``, or both ``fn / dpai`` with
    ``vary_with_height``. Denominators have a 1e-30 floor (no division by 0 where the
    caller discards the result).

    Fortran reference: MLCanopyNitrogenProfileMod.F90::CanopyNitrogenProfile.
    """
    dpai_safe = jnp.maximum(dpai, 1.0e-30)  # jnp.maximum avoids the select_divide_fusion bug
    if vary_with_height:
        nscale = fn / dpai_safe
        return nscale, nscale
    nscale_sun = fn_sun / (jnp.maximum(fracsun, 1.0e-30) * dpai_safe)
    nscale_sha = (fn - fn_sun) / (jnp.maximum(1.0 - fracsun, 1.0e-30) * dpai_safe)
    return nscale_sun, nscale_sha


class NitrogenProfile(NamedTuple):
    vcmax25_leaf: Array  # (layer, leaf) umol/m2/s
    jmax25_leaf: Array  # (layer, leaf) umol/m2/s
    rd25_leaf: Array  # (layer, leaf) umol CO2/m2/s
    kp25_leaf: Array  # (layer, leaf) mol/m2/s
    vcmax25_profile: Array  # (layer,) layer mean of the leaves
    jmax25_profile: Array  # (layer,)
    rd25_profile: Array  # (layer,)
    kp25_profile: Array  # (layer,)
    integration_error: Array  # () sum(vcmax25_profile dpai) minus its analytical value


def canopy_nitrogen_profile(
    layers: PatchLayers,
    dpai: Array,
    kb: Array,
    tbi: Array,
    fracsun: Array,
    lai,
    sai,
    tacclim,
    vcmax25top,
    clump,
    *,
    c3: bool,
    acclimation: bool,
    kn_val: float,
    vary_with_height: bool,
) -> NitrogenProfile:
    """Leaf rates at 25 C of each layer's sunlit and shaded leaves and their layer means
    (``vcmax25_profile = vcmax25top fn / dpai``, the sunlit/shaded mean written without the
    sunlit fraction so that it is exact where ``fracsun`` is 0 or 1), from the top layer
    down; and the error of the canopy integral of ``vcmax25_profile`` against
    ``vcmax25top (1 - exp(-kn (lai + sai))) / kn`` (Fortran stops above 1e-6; the caller
    records it).

    Fortran reference: MLCanopyNitrogenProfileMod.F90::CanopyNitrogenProfile.
    Reads: ``dpai_profile``, ``kb_profile``, ``tbi_profile``, ``fracsun_profile``,
    ``lai_canopy``, ``sai_canopy``, ``tacclim_forcing``, ``ncan_canopy``; ``vcmaxpft``,
    ``clump_fac`` (parameters). Writes: ``vcmax25_leaf``, ``jmax25_leaf``, ``rd25_leaf``,
    ``kp25_leaf``, ``vcmax25_profile``, ``jmax25_profile``, ``rd25_profile``,
    ``kp25_profile``; the check ``NITROGEN``.
    """
    top = canopy_top_rates(vcmax25top, tacclim, c3=c3, acclimation=acclimation)
    kn = decay_coefficient(vcmax25top, kn_val)

    # From the top slot down (Fortran: do ic = ncan, 1, -1), as the cumulative sum runs
    td = slice(None, None, -1)
    active = layers.in_use[td]
    dpai_v, kb_v, tbi_v, fs_v = dpai[td], kb[td], tbi[td], fracsun[td]
    has_pai = active & (dpai_v > 0.0)
    dpai_safe = jnp.maximum(dpai_v, 1.0e-30)  # jnp.maximum avoids the select_divide_fusion bug
    # Plant area above each layer; padding (fill values) zeroed before the sum
    dpai_cumsum = jnp.where(active, dpai_v, 0.0)
    pai_above_v = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dpai_cumsum[:-1])])

    # fn and fn_sun are masked before they are combined (combined unmasked, the shaded
    # leaves' values compile 1 ulp differently in the chunk)
    fn, fn_sun = nitrogen_fractions(
        kn, pai_above_v, dpai_v, kb_v, tbi_v, clump, vary_with_height=vary_with_height
    )
    fn = jnp.where(has_pai, fn, 0.0)
    if vary_with_height:
        nscale_sun = nscale_sha = jnp.where(has_pai, fn / dpai_safe, 0.0)
    else:
        fn_sun = jnp.where(has_pai, fn_sun, 0.0)
        fn_sha = fn - fn_sun
        fs_safe = jnp.maximum(fs_v, 1.0e-30)
        fsha_safe = jnp.maximum(1.0 - fs_v, 1.0e-30)
        nscale_sun = jnp.where(has_pai, fn_sun / (fs_safe * dpai_safe), 0.0)
        nscale_sha = jnp.where(has_pai, fn_sha / (fsha_safe * dpai_safe), 0.0)

    def _leaves(rate):
        sun = jnp.where(has_pai, rate * nscale_sun, 0.0)[td]
        sha = jnp.where(has_pai, rate * nscale_sha, 0.0)[td]
        return jnp.stack([sun, sha], axis=-1)

    def _mean(rate):
        return jnp.where(has_pai, rate * fn / dpai_safe, 0.0)[td]

    vcmax25_profile = _mean(top.vcmax25)
    numerical = jnp.sum(vcmax25_profile * dpai)
    analytical = top.vcmax25 * (1.0 - jnp.exp(-kn * (lai + sai))) / kn
    return NitrogenProfile(
        *(_leaves(r) for r in top),
        vcmax25_profile,
        _mean(top.jmax25),
        _mean(top.rd25),
        _mean(top.kp25),
        numerical - analytical,
    )
