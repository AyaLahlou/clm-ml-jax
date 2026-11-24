"""
Model state of one patch: 0-based arrays without a patch axis.

A time step is ``state -> state`` given forcing and parameters. :class:`State` holds the
prognostic variables carried from one step to the next (the ``lax.scan`` carry of a chunk, and
what a restart stores); :class:`Diagnostics` the step's results that outputs and checks read;
:class:`StepState` both. :class:`PatchLayers` is the patch's fixed layer grid.

Axes (all 0-based):

- ``layer`` (``max_layers``): canopy-air layers from the ground up; layers ``0 .. n_layers-1``
  are in use, the rest is padding (``PatchLayers`` masks).
- ``level`` (``max_layers + 1``): the ground (0) and the layers above it (``i`` = layer
  ``i - 1``); for fluxes at the ground and every layer, and layer-top heights.
- ``leaf`` (2): 0 sunlit, 1 shaded.
- ``band`` (2): 0 visible, 1 near-infrared.
- ``soil_level`` (``nlevgrnd``): soil layers from the top down.

Arrays are JAX arrays (a pytree), so ``jax.grad`` differentiates with respect to them.
Batched runs add a leading patch axis to every field.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class CanopyState(NamedTuple):
    """Canopy air, leaves and ground surface."""

    air_temperature: Array  # (layer,) K
    vapor_pressure: Array  # (layer,) Pa
    co2: Array  # (layer,) umol mol-1
    leaf_temperature: Array  # (layer, leaf) K
    leaf_water_potential: Array  # (layer, leaf) MPa
    intercepted_water: Array  # (layer,) kg m-2
    ground_temperature: Array  # () K
    # Air temperature that photosynthesis acclimates to: by default a 10-day running mean of
    # the forcing air temperature, updated each step (differs from CLM-ml v2: a long run cannot
    # know its whole-run mean in advance); or the whole-run mean (Fortran CLM-ml v2, set once
    # at initialization; validation only).
    acclimation_temperature: Array  # () K
    # Steps averaged into the running mean so far (its weight is 1 / min(steps, period)).
    acclimation_steps: Array  # ()


class SoilState(NamedTuple):
    """The soil column under the patch (one per patch: ``soil.columns: per_patch``)."""

    temperature: Array  # (soil_level,) K


class State(NamedTuple):
    """The prognostic variables: everything one step passes to the next."""

    canopy: CanopyState
    soil: SoilState


class Diagnostics(NamedTuple):
    """A step's results read by the output variables (``clm_ml.outputs``), the legacy ``.out``
    files and the conservation checks; nothing else. Names are CLM-ml v2's (``mlcanopy_type``),
    shapes 0-based; the outputs give them readable names. Canopy totals are per m2 ground."""

    # Forcing at the reference height (interpolated to the step)
    tref_forcing: Array  # () air temperature, K
    pref_forcing: Array  # () air pressure, Pa
    rhomol_forcing: Array  # () molar density, mol m-3
    swskyb_forcing: Array  # (band,) direct beam shortwave, W m-2
    swskyd_forcing: Array  # (band,) diffuse shortwave, W m-2
    solar_zen_forcing: Array  # () solar zenith angle, radians
    # Vegetation (prescribed; the plant area per layer)
    lai_canopy: Array  # () leaf area index, m2 m-2
    sai_canopy: Array  # () stem area index (with sai_floor_canopy), m2 m-2
    sai_floor_canopy: Array  # () stem area added by the per-layer minimum plant area, m2 m-2
    laisun_canopy: Array  # () sunlit leaf area index, m2 m-2
    laisha_canopy: Array  # () shaded leaf area index, m2 m-2
    dpai_profile: Array  # (layer,) plant area index of the layer, m2 m-2
    fracsun_profile: Array  # (layer,) sunlit fraction of the layer
    # Canopy totals
    albcan_canopy: Array  # (band,) canopy albedo
    swveg_canopy: Array  # (band,) absorbed shortwave: vegetation, W m-2
    swvegsun_canopy: Array  # (band,) absorbed shortwave: sunlit canopy, W m-2
    swvegsha_canopy: Array  # (band,) absorbed shortwave: shaded canopy, W m-2
    lwup_canopy: Array  # () upward longwave above the canopy, W m-2
    rnet_canopy: Array  # () net radiation, W m-2
    shflx_canopy: Array  # () total sensible heat flux, W m-2
    lhflx_canopy: Array  # () total latent heat flux, W m-2
    stflx_air_canopy: Array  # () canopy air storage heat flux, W m-2
    stflx_veg_canopy: Array  # () vegetation heat storage flux, W m-2
    ustar_canopy: Array  # () friction velocity, m s-1
    beta_canopy: Array  # () u*/U(h) at the canopy top
    shveg_canopy: Array  # () sensible heat: vegetation, W m-2
    shvegsun_canopy: Array  # () sensible heat: sunlit canopy, W m-2
    shvegsha_canopy: Array  # () sensible heat: shaded canopy, W m-2
    lhveg_canopy: Array  # () latent heat: vegetation, W m-2
    lhvegsun_canopy: Array  # () latent heat: sunlit canopy, W m-2
    lhvegsha_canopy: Array  # () latent heat: shaded canopy, W m-2
    trveg_canopy: Array  # () transpiration, mol H2O m-2 s-1
    evveg_canopy: Array  # () evaporation of intercepted water, mol H2O m-2 s-1
    gppveg_canopy: Array  # () GPP: vegetation, umol CO2 m-2 s-1
    gppvegsun_canopy: Array  # () GPP: sunlit canopy, umol CO2 m-2 s-1
    gppvegsha_canopy: Array  # () GPP: shaded canopy, umol CO2 m-2 s-1
    vcmax25veg_canopy: Array  # () Vcmax25: total canopy, umol m-2 s-1
    vcmax25sun_canopy: Array  # () Vcmax25: sunlit canopy, umol m-2 s-1
    vcmax25sha_canopy: Array  # () Vcmax25: shaded canopy, umol m-2 s-1
    gsveg_canopy: Array  # () stomatal conductance: canopy, mol H2O m-2 s-1
    gsvegsun_canopy: Array  # () stomatal conductance: sunlit canopy, mol H2O m-2 s-1
    gsvegsha_canopy: Array  # () stomatal conductance: shaded canopy, mol H2O m-2 s-1
    windveg_canopy: Array  # () wind speed: vegetation (leaf-area weighted), m s-1
    windvegsun_canopy: Array  # () wind speed: sunlit canopy, m s-1
    windvegsha_canopy: Array  # () wind speed: shaded canopy, m s-1
    tlveg_canopy: Array  # () leaf temperature: vegetation, K
    tlvegsun_canopy: Array  # () leaf temperature: sunlit canopy, K
    tlvegsha_canopy: Array  # () leaf temperature: shaded canopy, K
    taveg_canopy: Array  # () air temperature: vegetation, K
    tavegsun_canopy: Array  # () air temperature: sunlit canopy, K
    tavegsha_canopy: Array  # () air temperature: shaded canopy, K
    fracminlwp_canopy: Array  # () fraction of canopy at the minimum leaf water potential
    # Soil surface
    rnsoi_soil: Array  # () net radiation, W m-2
    shsoi_soil: Array  # () sensible heat flux, W m-2
    lhsoi_soil: Array  # () latent heat flux, W m-2
    gsoi_soil: Array  # () soil heat flux, W m-2
    btran_soil: Array  # () soil wetness factor for stomatal conductance
    psis_soil: Array  # () weighted soil water potential, MPa
    # Profiles
    wind_profile: Array  # (layer,) wind speed, m s-1
    shair_profile: Array  # (layer,) air sensible heat flux, W m-2
    etair_profile: Array  # (layer,) air water vapor flux, mol H2O m-2 s-1
    mflx_profile: Array  # (layer,) momentum flux, m2 s-2
    gac_profile: Array  # (layer,) aerodynamic conductance for scalars, mol m-2 s-1
    lsc_profile: Array  # (layer,) leaf-specific conductance, mmol H2O m-2 leaf s-1 MPa-1
    lwp_mean_profile: Array  # (layer,) leaf water potential, sunlit/shaded mean, MPa
    td_profile: Array  # (layer,) diffuse transmittance (checked: > 1 in very thin layers)
    lwdwn_profile: Array  # (level,) downward longwave at the ground and each layer, W m-2
    lwupw_profile: Array  # (level,) upward longwave at the ground and each layer, W m-2
    swbeam_profile: Array  # (level, band) direct beam shortwave, W m-2
    swdwn_profile: Array  # (level, band) downward diffuse shortwave, W m-2
    swupw_profile: Array  # (level, band) upward diffuse shortwave, W m-2
    # Leaves (per m2 leaf)
    apar_leaf: Array  # (layer, leaf) absorbed PAR, umol photon m-2 s-1
    rnleaf_leaf: Array  # (layer, leaf) net radiation, W m-2
    shleaf_leaf: Array  # (layer, leaf) sensible heat flux, W m-2
    lhleaf_leaf: Array  # (layer, leaf) latent heat flux, W m-2
    anet_leaf: Array  # (layer, leaf) net photosynthesis, umol CO2 m-2 s-1
    gs_leaf: Array  # (layer, leaf) stomatal conductance, mol H2O m-2 s-1
    vcmax25_leaf: Array  # (layer, leaf) Vcmax at 25 C, umol m-2 s-1
    tleaf_hist_leaf: Array  # (layer, leaf) leaf temperature (output), K
    lwp_hist_leaf: Array  # (layer, leaf) leaf water potential (output), MPa
    # Largest error of each Fortran conservation check in the step
    # (``clm_ml.checks.BALANCE_CHECKS``), checked on the host
    balance_error: Array  # (check,)


class StepState(NamedTuple):
    """One step's result: the prognostic state and the diagnostics."""

    state: State
    diagnostics: Diagnostics


class PatchLayers(NamedTuple):
    """The patch's layer grid, fixed at setup (counts and heights; masks as properties)."""

    n_layers: Array  # () layers in use (CLM-ml v2 ncan)
    n_within: Array  # () layers within the canopy (CLM-ml v2 ntop: layers 0 .. n_within-1)
    first_leafy: Array  # () lowest layer with leaves (CLM-ml v2 nbot - 1)
    dz: Array  # (layer,) layer thickness, m
    zs: Array  # (layer,) height at the layer's middle, m
    zw: Array  # (level,) height of the ground (0) and the layers' tops, m

    @property
    def index(self) -> Array:
        return jnp.arange(self.dz.shape[-1])

    @property
    def in_use(self) -> Array:
        return self.index < self.n_layers

    @property
    def in_canopy(self) -> Array:
        return self.index < self.n_within

    @property
    def above_canopy(self) -> Array:
        return (self.index >= self.n_within) & (self.index < self.n_layers)

    @property
    def leafy(self) -> Array:
        return (self.index >= self.first_leafy) & (self.index < self.n_within)
