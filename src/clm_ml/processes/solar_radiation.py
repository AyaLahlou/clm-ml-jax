"""Solar radiation: canopy optical properties and two-stream radiative transfer integrated
over each canopy layer, for the visible and near-infrared bands.

One patch. Layers are 0-based slots on axis 0 (``clm_ml.state.PatchLayers``; leaves in the
leafy layers ``first_leafy .. n_within - 1``); levels are the ground (0) and the layers;
bands are 0-based (0 visible, 1 near infrared); leaves 0 sunlit, 1 shaded. The scheme is a
variant chosen by name (``SOLAR_RADIATION``, model.yaml ``processes.radiation.solar``):
``two_stream`` (Fortran ``light_type = 2``, CLM-ml v2's default); ``norman`` (Fortran
``light_type = 1``) is registered but not supported (Phase 7 decision D8: its JAX translation
was never validated and stops at the first CHATS7 step; the translation is in the git history,
``src/clm_ml/legacy/solar_radiation.py`` at commit 1f1b651, for the fix after Phase 8: C34).

Fortran reference: CLM-ml v2 ``MLSolarRadiationMod.F90::SolarRadiation``, ``TwoStream``.
Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, chapter 14; Bonan et al.
(2021) Agric. For. Meteorol. 306, 108435, supplement eqs. (1)-(5).

Reads and writes are the arguments and results; ``Reads``/``Writes`` in the docstrings name
the CLM-ml v2 (``mlcanopy_type``) fields they correspond to.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from clm_ml.processes.constants import J_to_umol, chil_max, chil_min, kb_max, rpi
from clm_ml.processes.layers import keep, leafy_or_top_leafy, pad_with_top
from clm_ml.state import PatchLayers

# ---------------------------------------------------------------------------
# Canopy optical properties
# ---------------------------------------------------------------------------


class CanopyOptics(NamedTuple):
    """Optical properties of each layer slot; 0 outside the leafy layers."""

    rho: Array  # (layer, band) reflectance of leaves and stems
    tau: Array  # (layer, band) transmittance of leaves and stems
    omega: Array  # (layer, band) scattering coefficient
    clump: Array  # (layer,) foliage clumping index
    kb: Array  # (layer,) direct beam extinction coefficient
    tb: Array  # (layer,) direct beam transmittance through the layer
    td: Array  # (layer,) diffuse transmittance through the layer
    tbi: Array  # (layer,) direct beam transmittance onto the layer (from the canopy top)
    tbi_ground: Array  # () direct beam transmittance onto the ground
    fracsun: Array  # (layer,) sunlit fraction
    avmu: Array  # (layer,) average inverse diffuse optical depth per unit plant area
    betad: Array  # (layer, band) upscatter parameter for diffuse radiation
    betab: Array  # (layer, band) upscatter parameter for direct beam radiation
    sunlit_fraction_error: Array  # () 1 if a leafy layer has fracsun outside (0, 1), else 0


def canopy_optics(
    solar_zen,
    dlai: Array,
    dsai: Array,
    dpai: Array,
    layers: PatchLayers,
    rhol: Array,
    taul: Array,
    rhos: Array,
    taus: Array,
    leaf_angle_index,
    clumping_index,
    unroll: bool = False,
) -> CanopyOptics:
    """Optical properties of the canopy layers for the solar zenith angle ``solar_zen``.

    Leaf and stem optics (``rhol``, ``taul``, ``rhos``, ``taus``; per band) are weighted by
    the layer's leaf and stem area (Fortran ``leaf_optics_type = 0``, constant with height);
    Ross-Goudriaan leaf angle distribution from ``leaf_angle_index``; diffuse transmittance
    by 9-angle integration. ``unroll``: the loop over layers written out (``math.tridiag``).

    Fortran reference: MLSolarRadiationMod.F90::SolarRadiation (optical properties).
    Reads: ``solar_zen_forcing``, ``dlai_profile``, ``dsai_profile``, ``dpai_profile``, the
    PFT's optical parameters. Writes: ``kb_profile``, ``tb_profile``, ``td_profile``,
    ``tbi_profile`` (level 0: ``tbi_ground``), ``fracsun_profile``; the sunlit-fraction check.
    """
    leafy = layers.leafy
    cos_zen = jnp.cos(solar_zen)

    dpai_v = pad_with_top(layers, dpai)  # padding: the top layer's (0)
    # jnp.maximum avoids select op -> prevents XLA select_divide_fusion bug
    dpai_safe = jnp.maximum(dpai_v, 1.0e-30)
    wl = (pad_with_top(layers, dlai) / dpai_safe)[:, None]
    ws = (pad_with_top(layers, dsai) / dpai_safe)[:, None]

    # Reflectance, transmittance, scattering (layer, band)
    r_v = jnp.maximum(rhol * wl + rhos * ws, 1.0e-6)
    t_v = jnp.maximum(taul * wl + taus * ws, 1.0e-6)
    om_v = r_v + t_v

    # Leaf angle distribution (one PFT: the same in every layer)
    chil = jnp.clip(leaf_angle_index, chil_min, chil_max)
    chil = jnp.where(jnp.abs(chil) <= 0.01, 0.01, chil)

    # Ross-Goudriaan phi1, phi2, gdir
    p1 = 0.5 - 0.633 * chil - 0.330 * chil * chil
    p2 = 0.877 * (1.0 - 2.0 * p1)
    gd = p1 + p2 * cos_zen

    # Direct beam extinction coefficient. jnp.maximum avoids a NaN gradient: cos_zen = 0 at
    # the horizon gives gd/0 = inf in the unused branch of jnp.minimum (0*inf in backward).
    cos_zen_safe = jnp.maximum(cos_zen, 1.0e-10)
    kb_s = jnp.minimum(gd / cos_zen_safe, kb_max)
    kb_v = jnp.broadcast_to(kb_s, dpai_v.shape)

    cf = clumping_index

    # Direct beam transmittance through a single layer
    tb_v = jnp.exp(-kb_v * dpai_v * cf)

    # Diffuse transmittance through a single layer (9-angle integration)
    td_v = jnp.zeros_like(dpai_v)
    for j in range(1, 10):
        angle = (5.0 + (j - 1) * 10.0) * rpi / 180.0
        gdirj = p1 + p2 * jnp.cos(jnp.asarray(angle))
        td_v = td_v + (
            jnp.exp(-gdirj / jnp.cos(jnp.asarray(angle)) * dpai_v * cf)
            * jnp.sin(jnp.asarray(angle))
            * jnp.cos(jnp.asarray(angle))
        )
    td_v = td_v * 2.0 * (10.0 * rpi / 180.0)

    # Direct beam transmittance onto each layer, from the top: tbi(ic) = tbi(ic+1) tb(ic+1),
    # multiplied in Fortran's order. Slots above the leafy layers multiply by 1 (exact).
    tb_eff = jnp.where(leafy, tb_v, 1.0)

    def _tbi_step(carry, tb_above):
        return carry * tb_above, carry

    _, tbi_v = jax.lax.scan(_tbi_step, jnp.ones(()), tb_eff, reverse=True, unroll=unroll or 1)

    # Sunlit fraction. When kb*dpai = 0 the exp term is 0 too, so fracsun = 0.
    kb_dp_safe = jnp.maximum(kb_v * dpai_v, 1.0e-30)
    fracsun_v = tbi_v / kb_dp_safe * (1.0 - jnp.exp(-kb_v * cf * dpai_v))
    # Fortran stops if fracsun or fracsha is not positive (recorded, checked on the host)
    bad = jnp.any(leafy & ((fracsun_v <= 0.0) | ((1.0 - fracsun_v) <= 0.0)))

    # Two-stream avmu
    p1_safe = jnp.maximum(p1, 1.0e-10)
    p2_safe = jnp.maximum(p2, 1.0e-10)
    avmu_s = (1.0 - p1_safe / p2_safe * jnp.log((p1_safe + p2_safe) / p1_safe)) / p2_safe

    # betad, betab (layer, band)
    betad_v = 0.5 / om_v * (r_v + t_v + (r_v - t_v) * ((1.0 + chil) / 2.0) ** 2)
    tmp0 = gd + p2 * cos_zen
    tmp1 = p1 * cos_zen
    # both non-negative physically; jnp.maximum guards the divisions and the log
    tmp0_safe = jnp.maximum(tmp0, 1.0e-10)
    tmp1_safe = jnp.maximum(tmp1, 1.0e-10)
    tmp2 = 1.0 - tmp1_safe / tmp0_safe * jnp.log((tmp1_safe + tmp0_safe) / tmp1_safe)
    asu = 0.5 * om_v * gd / tmp0_safe * tmp2
    betab_v = (1.0 + avmu_s * kb_s) / (om_v * avmu_s * kb_s) * asu

    def _leafy(v):
        return keep(leafy, v, 0.0)

    return CanopyOptics(
        rho=_leafy(r_v),
        tau=_leafy(t_v),
        omega=_leafy(om_v),
        clump=jnp.where(leafy, cf, 0.0),
        kb=_leafy(kb_v),
        tb=_leafy(tb_v),
        td=_leafy(td_v),
        tbi=_leafy(tbi_v),
        tbi_ground=tbi_v[layers.first_leafy] * tb_v[layers.first_leafy],
        fracsun=_leafy(fracsun_v),
        avmu=jnp.where(leafy, avmu_s, 0.0),
        betad=_leafy(betad_v),
        betab=_leafy(betab_v),
        sunlit_fraction_error=jnp.where(bad, 1.0, 0.0),
    )


# ---------------------------------------------------------------------------
# Radiative transfer
# ---------------------------------------------------------------------------


class SolarFluxes(NamedTuple):
    """Solar radiation fluxes per band (W/m2); layer values 0 outside the leafy layers."""

    swbeam: Array  # (level, band) direct beam onto the ground (0) and each layer
    swdwn: Array  # (level, band) downward diffuse onto the ground and each layer
    swupw: Array  # (level, band) upward diffuse from the ground and each layer
    swleaf: Array  # (layer, leaf, band) absorbed per unit area of sunlit/shaded leaf
    swsoi: Array  # (band,) absorbed by the ground
    swveg: Array  # (band,) absorbed by the vegetation
    swvegsun: Array  # (band,) absorbed by sunlit leaves
    swvegsha: Array  # (band,) absorbed by shaded leaves
    albcan: Array  # (band,) canopy albedo
    error: Array  # (band,) energy conservation error (Fortran stops above 1e-6)


def two_stream(
    optics: CanopyOptics,
    dpai: Array,
    swskyb: Array,
    swskyd: Array,
    albsoib: Array,
    albsoid: Array,
    layers: PatchLayers,
    unroll: bool = False,
) -> SolarFluxes:
    """Two-stream radiative transfer integrated over each canopy layer, both bands at once.

    A bottom-to-top sweep solves each leafy layer for unit direct beam and unit diffuse
    radiation from above, with the albedo of what lies below (the ground, then the layers
    beneath); a top-to-bottom sweep then propagates the sky's direct beam ``swskyb`` and
    diffuse ``swskyd`` radiation down. Both sweeps are ``lax.scan`` loops over every slot:
    slots outside the leafy layers get the top leafy layer's inputs and pass the sweep's
    values through unchanged (their results are discarded). ``unroll``: the sweeps written
    out instead of compiled loops (``math.tridiag``).

    Fortran reference: MLSolarRadiationMod.F90::TwoStream.
    Reads: ``swskyb_forcing``, ``swskyd_forcing``, ``albsoib_soil``, ``albsoid_soil``,
    ``dpai_profile`` and the optical properties. Writes: ``swbeam_profile``,
    ``swdwn_profile``, ``swupw_profile``, ``swleaf_leaf``, ``swsoi_soil``, ``swveg_canopy``,
    ``swvegsun_canopy``, ``swvegsha_canopy``, ``albcan_canopy``; the solar check.
    """
    leafy = layers.leafy

    def _in(x):
        return leafy_or_top_leafy(layers, x)

    # Bottom-to-top sweep
    def _upward(carry, x):
        albb, albd = carry
        is_leafy, om_ic, av_ic, kb_ic, cf_ic, dp_ic, bd_ic, bb_ic, tbi_ic = x

        b = (1.0 - (1.0 - bd_ic) * om_ic) / av_ic
        c = bd_ic * om_ic / av_ic
        # jnp.maximum guards sqrt(0) and /0 in backward pass (NaN-safe grads)
        h = jnp.sqrt(jnp.maximum(b * b - c * c, 1.0e-30))
        u = (h - b - c) / (2.0 * h)
        v = (h + b + c) / (2.0 * h)
        # Guard h²−kb² denominator for NaN-safe grads when h ≈ kb
        _h2_kb2 = h * h - kb_ic * kb_ic
        _h2_kb2_sign = jnp.where(_h2_kb2 < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
        _h2_kb2_safe = _h2_kb2_sign * jnp.maximum(jnp.abs(_h2_kb2), 1.0e-30)
        d = om_ic * kb_ic / _h2_kb2_safe  # unitb=1
        g1 = (bb_ic * kb_ic - b * bb_ic - c * (1.0 - bb_ic)) * d
        g2 = ((1.0 - bb_ic) * kb_ic + c * bb_ic + b * (1.0 - bb_ic)) * d
        s1 = jnp.exp(-h * cf_ic * dp_ic)
        s2 = jnp.exp(-kb_ic * cf_ic * dp_ic)

        # Direct beam terms (unitb = 1)
        num1 = v * (g1 + g2 * albd + albb) * s2
        num2 = g2 * (u + v * albd) * s1
        den1 = v * (v + u * albd) / s1
        den2 = u * (u + v * albd) * s1
        # Guard Wronskian-like denominator for NaN-safe backward pass
        _d1d2 = den1 - den2
        _d1d2_sgn = jnp.where(_d1d2 < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
        _d1d2_saf = _d1d2_sgn * jnp.maximum(jnp.abs(_d1d2), 1.0e-30)
        n2b = (num1 - num2) / _d1d2_saf
        n1b = (g2 - n2b * u) / v

        # Guard kb±h denominators — degenerate when kb≈h (already h≠0 from sqrt guard)
        _kbph = kb_ic + h  # always > 0
        _kbmh = kb_ic - h
        _kbmh_sgn = jnp.where(_kbmh < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
        _kbmh_saf = _kbmh_sgn * jnp.maximum(jnp.abs(_kbmh), 1.0e-30)
        _2kb_saf = jnp.maximum(2.0 * kb_ic, 1.0e-30)
        a1b = (
            -g1 * (1.0 - s2 * s2) / _2kb_saf
            + n1b * u * (1.0 - s2 * s1) / _kbph
            + n2b * v * (1.0 - s2 / s1) / _kbmh_saf
        ) * tbi_ic
        a2b = (
            g2 * (1.0 - s2 * s2) / _2kb_saf
            - n1b * v * (1.0 - s2 * s1) / _kbph
            - n2b * u * (1.0 - s2 / s1) / _kbmh_saf
        ) * tbi_ic

        iupwb0_ic = -g1 + n1b * u + n2b * v
        iupwb_ic = -g1 * s2 + n1b * u * s1 + n2b * v / s1
        idwnb_ic = g2 * s2 - n1b * v * s1 - n2b * u / s1
        abs_b = (1.0 - s2) - iupwb0_ic + iupwb_ic - idwnb_ic
        abs_b_sun = (1.0 - om_ic) * ((1.0 - s2) + cf_ic / av_ic * (a1b + a2b))

        # Diffuse terms (unitd = 1) — reuse guards from direct-beam block
        num1 = (u + v * albd) * s1
        n2d = num1 / _d1d2_saf
        n1d = -(1.0 + n2d * u) / v

        a1d = (n1d * u * (1.0 - s2 * s1) / _kbph + n2d * v * (1.0 - s2 / s1) / _kbmh_saf) * tbi_ic
        a2d = (-n1d * v * (1.0 - s2 * s1) / _kbph - n2d * u * (1.0 - s2 / s1) / _kbmh_saf) * tbi_ic

        iupwd0_ic = n1d * u + n2d * v
        iupwd_ic = n1d * u * s1 + n2d * v / s1
        idwnd_ic = -n1d * v * s1 - n2d * u / s1
        abs_d = 1.0 - iupwd0_ic + iupwd_ic - idwnd_ic
        abs_d_sun = (1.0 - om_ic) * cf_ic / av_ic * (a1d + a2d)

        out = (
            iupwb0_ic,
            idwnb_ic,
            abs_b_sun,
            abs_b - abs_b_sun,
            iupwd0_ic,
            idwnd_ic,
            abs_d_sun,
            abs_d - abs_d_sun,
        )
        # Albedos seen by the next layer up
        carry = (jnp.where(is_leafy, iupwb0_ic, albb), jnp.where(is_leafy, iupwd0_ic, albd))
        return carry, out

    col = lambda a: a[:, None]  # noqa: E731  (layer value shared by both bands)
    kb_in, cf_in, dp_in = col(_in(optics.kb)), col(_in(optics.clump)), col(_in(dpai))
    xs_up = (
        leafy,
        _in(optics.omega),
        col(_in(optics.avmu)),
        kb_in,
        cf_in,
        dp_in,
        _in(optics.betad),
        _in(optics.betab),
        col(_in(optics.tbi)),
    )
    _, (iupwb0, idwnb, iabsb_sun, iabsb_sha, iupwd0, idwnd, iabsd_sun, iabsd_sha) = jax.lax.scan(
        _upward, (albsoib, albsoid), xs_up, unroll=unroll or 1
    )

    # Top-to-bottom sweep
    def _downward(carry, x):
        dir_, dif = carry
        (is_leafy, iu_d0, iu_b0, ab_sun, ab_sha, ad_sun, ad_sha, id_b, id_d, fs, dp_ic, kb_ic,
         cf_ic) = x  # fmt: skip
        swupw_ic = iu_d0 * dif + iu_b0 * dir_
        # jnp.maximum avoids select op -> prevents XLA select_divide_fusion bug
        fs_safe = jnp.maximum(fs, 1.0e-30)
        fsha_safe = jnp.maximum(1.0 - fs, 1.0e-30)
        dpai_safe = jnp.maximum(dp_ic, 1.0e-30)
        sun = (ab_sun * dir_ + ad_sun * dif) / (fs_safe * dpai_safe)
        sha = (ab_sha * dir_ + ad_sha * dif) / (fsha_safe * dpai_safe)
        dif_new = dir_ * id_b + dif * id_d
        dir_new = dir_ * jnp.exp(-kb_ic * cf_ic * dp_ic)
        dir_new = jnp.where(is_leafy, dir_new, dir_)
        dif_new = jnp.where(is_leafy, dif_new, dif)
        return (dir_new, dif_new), (dir_, dif, swupw_ic, sun, sha)

    fs_in = col(_in(optics.fracsun))
    xs_down = (leafy, iupwd0, iupwb0, iabsb_sun, iabsb_sha, iabsd_sun, iabsd_sha, idwnb, idwnd,
               fs_in, dp_in, kb_in, cf_in)  # fmt: skip
    (dir_, dif), (beam_v, dwn_v, upw_v, sun_v, sha_v) = jax.lax.scan(
        _downward, (swskyb, swskyd), xs_down, reverse=True, unroll=unroll or 1
    )

    # Ground
    swsoi = dir_ * (1.0 - albsoib) + dif * (1.0 - albsoid)

    # Canopy albedo
    top = layers.n_within - 1
    suminc = swskyb + swskyd
    sumref = iupwb0[top] * swskyb + iupwd0[top] * swskyd
    suminc_safe = jnp.maximum(suminc, 1.0e-30)
    albcan = jnp.where(suminc > 0.0, sumref / suminc_safe, 0.0)

    # Absorption by the vegetation, summed bottom to top as in Fortran (other slots add 0)
    sun_abs = jnp.where(leafy[:, None], sun_v * fs_in * dp_in, 0.0)
    sha_abs = jnp.where(leafy[:, None], sha_v * (1.0 - fs_in) * dp_in, 0.0)

    def _sum(carry, x):
        veg, vsun, vsha = carry
        sun, sha = x
        return (veg + sun + sha, vsun + sun, vsha + sha), None

    zero = jnp.zeros_like(swskyb)
    (swveg, swvegsun, swvegsha), _ = jax.lax.scan(
        _sum, (zero, zero, zero), (sun_abs, sha_abs), unroll=unroll or 1
    )

    # Conservation check (Fortran stops; recorded and checked on the host)
    sumabs = swveg + swsoi
    error = suminc - (sumabs + albcan * suminc)

    def _levels(v, ground):
        return jnp.concatenate([ground[None], jnp.where(leafy[:, None], v, 0.0)])

    swleaf = jnp.stack(
        [jnp.where(leafy[:, None], sun_v, 0.0), jnp.where(leafy[:, None], sha_v, 0.0)], axis=1
    )
    return SolarFluxes(
        swbeam=_levels(beam_v, dir_),
        swdwn=_levels(dwn_v, dif),
        swupw=_levels(upw_v, albsoid * dif + albsoib * dir_),
        swleaf=swleaf,
        swsoi=swsoi,
        swveg=swveg,
        swvegsun=swvegsun,
        swvegsha=swvegsha,
        albcan=albcan,
        error=error,
    )


def norman(*args, **kwargs):
    """Norman (1979) radiative transfer: registered but not supported (Phase 7 decision D8).

    Fortran reference: MLSolarRadiationMod.F90::Norman. The legacy JAX translation (never
    validated; stops at the first CHATS7 step with a conservation error) is in the git
    history (``src/clm_ml/legacy/solar_radiation.py`` at 1f1b651) for the fix after Phase 8.
    """
    raise NotImplementedError(
        "solar_radiation: norman is not supported (Phase 7 decision D8; see the plan, "
        "'After Phase 8: unsupported variants'); use two_stream"
    )


#: Radiative transfer variants by name (model.yaml ``processes.radiation.solar``).
SOLAR_RADIATION = {"two_stream": two_stream, "norman": norman}


def absorbed_par(swleaf: Array) -> Array:
    """PAR absorbed per unit leaf area, umol photons/m2/s (layer, leaf), from the visible
    band of ``swleaf`` (W/m2).

    Fortran reference: MLSolarRadiationMod.F90::SolarRadiation (``apar``).
    """
    return swleaf[..., 0] * J_to_umol
