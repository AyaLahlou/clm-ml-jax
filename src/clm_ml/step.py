"""One time step of one patch on the 0-based step state (Phase 7.2, process 10c).

``step(state, inputs, constants, params, options) -> StepState``: the soil routines of
CLM-ml v2's ``clm_driver`` around the multilayer canopy's ``MLCanopyFluxes``, built from the
processes in ``clm_ml.processes`` (soil, plant area and hydraulics, the canopy sub-steps, the
averages, totals and the sunlit/shaded merge). Pure: everything the step reads is an argument
and everything it changes is returned.

- :class:`State` (``clm_ml.state``): the prognostic variables carried from step to step.
- :class:`StepInputs`: this step's forcing and land data (tower meteorology, the CLM history
  record, the forcing records of the previous and next steps for the time interpolation) and
  its time.
- :class:`PatchConstants`: the patch's values that every step reads and none changes: the
  layer grid, PFT and soil properties, location (set up once; slots the step does not
  compute hold ``spval``, Phase 7.4b).
- :class:`StepOptions`: the static choices (code paths, step lengths).

Fortran reference: CLM-ml v2 ``clm_driver.F90::clm_drv`` and
``MLCanopyFluxesMod.F90::MLCanopyFluxes`` (with ``GetCLMVar``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from clm_ml.checks import (  # the conservation checks' indices
    BALANCE_CHECKS,
    CANOPY_ENERGY_1,
    PLANT_AREA,
    SOIL_TEMPERATURE,
)
from clm_ml.processes import canopy, soil
from clm_ml.processes.atm_forcing import ForcingRecord, clm_forcing_to_ml
from clm_ml.processes.layers import keep
from clm_ml.processes.leaf_heat_capacity import leaf_heat_capacity
from clm_ml.processes.plant_hydraulics import plant_resistance, soil_resistance
from clm_ml.state import CanopyState, Diagnostics, PatchLayers, SoilState, State, StepState


class StepOptions(NamedTuple):
    """The step's static choices (hashable: an argument of compiled functions)."""

    substep: canopy.SubstepOptions  # the canopy sub-step's choices
    step_seconds: int  # model time step (s)
    layer_pai_floor: float  # minimum plant area of a leafy layer (m2/m2; issue 14)
    flux_profile_type: int  # canopy totals: fluxes at the top layer (1)
    n_soil_adjust: int  # soil levels multiplied by the history's adjustment factor
    n_soil: int  # hydrologically active soil levels (nlevsoi)
    # Orbital parameters for the solar zenith angle (clm_varorb, fixed for the run)
    eccen: float
    mvelpp: float
    lambm0: float
    obliqr: float
    #: jax.checkpoint on each sub-step too (model.yaml numerics.checkpoint_substeps, which
    #: follows the device): gradients keep one sub-step's intermediate values instead of a
    #: step's (less memory; much slower on the CPU)
    checkpoint_substeps: bool = False

    @property
    def n_substeps(self) -> int:
        return int(self.step_seconds / self.substep.dtime)


class StepInputs(NamedTuple):
    """One step's forcing, land data and time (0-based bands, no patch axis)."""

    calday: Array  # () calendar day at the end of the step
    calday_beg: Array  # () calendar day at the beginning of the step
    # Tower (or gridcell) meteorology of this step, as TowerMetMod.TowerMet
    uref: Array  # () wind speed, m/s
    tref: Array  # () air temperature, K
    qref: Array  # () specific humidity, kg/kg
    pref: Array  # () air pressure, Pa
    lwsky: Array  # () atmospheric longwave, W/m2
    rain: Array  # () rainfall, kg H2O/m2/s
    snow: Array  # () snowfall, kg H2O/m2/s
    solad: Array  # (band,) direct beam solar radiation, W/m2
    solai: Array  # (band,) diffuse solar radiation, W/m2
    pco2: Array  # () CO2 partial pressure, Pa
    po2: Array  # () O2 partial pressure, Pa
    # CLM history record of this step (clmDataMod.ClmHistoryRecord)
    elai: Array  # () leaf area index, m2/m2
    esai: Array  # () stem area index, m2/m2
    h2osoi: Array  # (n_soil,) volumetric soil water
    h2osoi_factor: Array  # () soil water adjustment factor
    # Forcing records of the previous and next steps (3-point time interpolation)
    forcing_bef: ForcingRecord
    forcing_next: ForcingRecord


class MetInputs(NamedTuple):
    """One step's time and meteorology, per gridcell (the :class:`StepInputs` fields that
    every patch of a gridcell shares). A chunk holds them stacked along a leading step axis,
    then a gridcell axis (``calday``/``calday_beg`` may have no gridcell axis)."""

    calday: Array
    calday_beg: Array
    uref: Array
    tref: Array
    qref: Array
    pref: Array
    lwsky: Array
    rain: Array
    snow: Array
    solad: Array  # (band,)
    solai: Array  # (band,)
    pco2: Array
    po2: Array
    forcing_bef: ForcingRecord
    forcing_next: ForcingRecord


class LandInputs(NamedTuple):
    """One step's prescribed land data, per patch (the :class:`StepInputs` fields of the CLM
    history record or the land-data files). A chunk holds them stacked along a leading step
    axis, then a patch axis."""

    elai: Array
    esai: Array
    h2osoi: Array  # (n_soil,)
    h2osoi_factor: Array  # 1 when the history has no adjustment factor


def step_inputs(met: MetInputs, land: LandInputs) -> StepInputs:
    """A patch's :class:`StepInputs` from its gridcell's meteorology and its land data."""
    return StepInputs(**met._asdict(), **land._asdict())


class PatchConstants(NamedTuple):
    """The patch's values every step reads and no step changes (0-based, no patch axis)."""

    layers: PatchLayers
    # Leaf and stem area fractions of the layers
    dlai_frac: Array  # (layer,) (dlai_frac_profile)
    dsai_frac: Array  # (layer,) (dsai_frac_profile)
    zref: Array  # () reference height, m (zref_forcing)
    ztop: Array  # () canopy height, m (ztop_canopy)
    root_biomass: Array  # () fine root biomass, g/m2 (root_biomass_canopy)
    rootfr: Array  # (n_soil,) root fraction of each soil level (soilstate.rootfr_patch)
    # The soil column (soil levels 0-based from the top)
    dz: Array  # (soil_level,) level thickness, m (site.dz_col)
    z: Array  # (soil_level,) level depth, m (site.z_col)
    zi: Array  # (soil_level,) depth of the level's bottom interface, m (site.zi_col)
    soil_dz: Array  # () depth of the first level's middle below the surface, m (soil_dz_soil)
    nbedrock: Array  # () 1-based index of the deepest level above bedrock (site.nbedrock_col)
    watsat: Array  # (soil_level,) porosity
    sucsat: Array  # (soil_level,) saturated suction, mm
    bsw: Array  # (soil_level,) Clapp and Hornberger b
    hksat: Array  # (soil_level,) saturated hydraulic conductivity, mm/s
    tkmg: Array  # (soil_level,) thermal conductivity of the soil minerals
    tkdry: Array  # (soil_level,) thermal conductivity of dry soil
    csol: Array  # (soil_level,) heat capacity of the soil solids
    albsat: Array  # (band,) saturated soil albedo of the soil colour
    albdry: Array  # (band,) dry soil albedo of the soil colour
    latdeg: Array  # () latitude, degrees
    londeg: Array  # () longitude, degrees


def interpolation_days(calday, calday_beg, options: StepOptions) -> tuple:
    """Calendar days of the previous, current and next forcing records (centred in their
    steps for the 3-point interpolation; 0 without interpolation).

    Fortran reference: MLCanopyFluxesMod.F90::MLCanopyFluxes (lines 105-130).
    """
    days = options.step_seconds / 86400.0
    if options.substep.interpolation == "none":
        return 0.0, 0.0, 0.0
    cur = 0.5 * (calday + calday_beg)
    return cur - days, cur, cur + days


def substep_days(calday_beg, options: StepOptions) -> Array:
    """Calendar day of each sub-step (the sub-step's middle for the 3-point interpolation).

    Fortran reference: MLCanopyFluxesMod.F90::MLCanopyFluxes (lines 263-270).
    """
    offset = 0.0 if options.substep.interpolation == "none" else 0.5
    dt = options.substep.dtime
    return jnp.stack(
        [
            jnp.asarray(calday_beg + (float(k) - offset) * (dt / 86400.0), dtype=jnp.float64)
            for k in range(1, options.n_substeps + 1)
        ]
    )


def solar_zenith(calday_beg, latdeg, londeg, options: StepOptions) -> Array:
    """Solar zenith angle at the beginning of the step (cosine at least 0.01).

    Fortran reference: MLCanopyFluxesMod.F90::GetCLMVar (lines 92-110).
    """
    from clm_ml.processes.constants import rpi as pi
    from clm_ml.processes.orbit import shr_orb_cosz, shr_orb_decl

    declin, _ = shr_orb_decl(
        calday_beg, options.eccen, options.mvelpp, options.lambm0, options.obliqr
    )
    lat = latdeg * pi / 180.0
    lon = londeg * pi / 180.0
    coszen = shr_orb_cosz(calday_beg, lat, lon, declin)
    return jnp.arccos(jnp.maximum(jnp.asarray(0.01), jnp.asarray(coszen)))


def _fill(like: Array) -> Array:
    """``spval`` shaped as ``like``: the value of slots the step does not compute (above the
    top layer in use, soil levels below bedrock), as the legacy containers' fill (C26)."""
    return jnp.full(jnp.shape(like), canopy.SPVAL, dtype=jnp.float64)


def _record(errors: Array, check: int, err) -> Array:
    """``errors`` with ``|err|`` (largest element) recorded for ``check``."""
    return errors.at[check].max(jax.lax.stop_gradient(jnp.max(jnp.abs(jnp.asarray(err)))))


def soil_level_ice(ice, n_soil: int):
    """The ice of each hydrologically active soil level, ``(n_soil,)``: level j's own
    (index j, 0-based), which stops root uptake from a frozen level. The legacy translation
    took level j + 1's (open issue 19, fixed 2026-11-01, decision 95).

    Fortran reference: MLPlantHydraulicsMod.F90::SoilResistance (``h2osoi_ice(c,j)``)."""
    return ice[:n_soil]


def step(
    state: State,
    inputs: StepInputs,
    constants: PatchConstants,
    params,
    tables,
    *,
    options: StepOptions,
    c3: bool,
) -> StepState:
    """One time step of one patch: the step's results and the state after it.

    In the order of ``clm_drv``: the CLM history record's LAI, SAI and soil water; soil
    albedo, evaporative resistance, thermal and hydraulic properties; the canopy's forcing,
    solar zenith angle, plant area, soil and plant resistance, leaf heat capacity and soil
    surface humidity; the canopy sub-steps (``canopy.substep``; a ``lax.scan``, with
    ``jax.checkpoint`` on each sub-step if ``options.checkpoint_substeps``); the averages
    over the sub-steps, the canopy totals, the sunlit/shaded merge; the soil temperature. The conservation checks of the step are
    recorded in ``diagnostics.balance_error`` (checked on the host). ``params``: the patch's
    PFT parameters (``CanopyParams``); ``tables``: the psihat lookup tables
    (``clm_ml.processes.turbulence.RSLTables``, one copy for all patches and steps of a chunk);
    ``c3`` (static): the photosynthetic pathway.

    Fortran reference: clm_driver.F90::clm_drv; MLCanopyFluxesMod.F90::MLCanopyFluxes,
    GetCLMVar.
    """
    k = constants
    lay = k.layers
    use = lay.in_use
    t_soil = state.soil.temperature
    n_soil = options.n_soil
    zero = jnp.zeros((), t_soil.dtype)

    # CLM history record: LAI, SAI (patch) and soil water (column) — clmDataMod::clmData
    water = soil.soil_water_from_history(
        inputs.h2osoi, inputs.h2osoi_factor, k.dz, k.watsat, k.nbedrock,
        n_soil=n_soil, n_adjust=options.n_soil_adjust,
    )  # fmt: skip
    liq, ice = water.h2osoi_liq, water.h2osoi_ice

    # Soil albedo, evaporative resistance, thermal and hydraulic properties (no snow, no
    # surface water: clm_drv sets them to 0)
    albgrd, albgri = soil.soil_albedo(water.h2osoi_vol[0], k.albsat, k.albdry)
    _, soilresis = soil.soil_evaporative_resistance(
        liq[0], ice[0], t_soil[0], k.dz[0], k.watsat[0], k.sucsat[0], k.bsw[0]
    )
    thermal = soil.soil_thermal_properties(
        t_soil, liq, ice, zero, zero, k.dz, k.z, k.zi, k.watsat, k.tkmg, k.tkdry, k.csol,
        k.nbedrock,
    )  # fmt: skip
    hk_l, smp_l = soil.soil_hydraulic_properties(
        liq[:n_soil], k.dz[:n_soil], k.watsat[:n_soil], k.hksat[:n_soil], k.sucsat[:n_soil],
        k.bsw[:n_soil], k.nbedrock, _fill(k.dz[:n_soil]), _fill(k.dz[:n_soil]),
    )  # fmt: skip

    # The canopy's forcing of this step — GetCLMVar
    u, t, q, p, co2, solad, solai, lw = clm_forcing_to_ml(
        inputs.uref, zero, inputs.tref, inputs.qref, inputs.pref, inputs.pco2,
        inputs.solad, inputs.solai, inputs.lwsky,
    )  # fmt: skip
    forcing_cur = ForcingRecord(u, t, q, p, co2, solad, solai, lw)
    o2ref = inputs.po2 / inputs.pref * 1.0e3  # Pa -> mmol/mol
    solar_zen = solar_zenith(inputs.calday_beg, k.latdeg, k.londeg, options)

    # Plant area of the layers for the step's LAI and SAI
    area = canopy.plant_area(
        lay, k.dlai_frac, k.dsai_frac, inputs.elai, inputs.esai, options.layer_pai_floor
    )
    fill_layer = _fill(lay.dz)
    dlai = keep(use, area.dlai, fill_layer)
    dsai = keep(use, area.dsai, fill_layer)
    dpai = keep(use, area.dpai, fill_layer)
    errors = _record(jnp.zeros(len(BALANCE_CHECKS), t_soil.dtype), PLANT_AREA, area.error)

    # Plant hydraulics and leaf heat capacity (every layer slot)
    roots = soil_resistance(
        hk_l, smp_l, k.dz[:n_soil], k.rootfr, soil_level_ice(ice, n_soil), k.nbedrock, area.lai,
        k.root_biomass, params.root_radius, params.root_density, params.root_resistivity,
    )  # fmt: skip
    lsc = plant_resistance(dpai, roots.rsoil, params.plant_conductance)
    cpleaf = leaf_heat_capacity(dpai, params.sla_top)
    rhg = canopy.soil_surface_humidity(smp_l[0], t_soil[0])

    # The canopy sub-steps
    times = tuple(
        jnp.asarray(d, dtype=jnp.float64)
        for d in interpolation_days(inputs.calday, inputs.calday_beg, options)
    )
    sub_inputs = canopy.SubstepInputs(
        layers=lay, forcing_bef=inputs.forcing_bef, forcing_cur=forcing_cur,
        forcing_next=inputs.forcing_next, time_bef=times[0], time_cur=times[1],
        time_next=times[2], zref=k.zref, solar_zen=solar_zen, dlai=dlai, dsai=dsai, dpai=dpai,
        albsoib=albgrd, albsoid=albgri, lai=area.lai, sai=area.sai,
        tacclim=state.canopy.acclimation_temperature, ztop=k.ztop, o2ref=o2ref,
        qflx_rain=inputs.rain, qflx_snow=inputs.snow, rhg=rhg, soilres=soilresis,
        soil_t=t_soil[0], soil_dz=k.soil_dz, soil_tk=thermal.thk[0], cpleaf=cpleaf, lsc=lsc,
        psis=roots.psis,
    )  # fmt: skip
    c = state.canopy
    n_slots = lay.dz.shape[-1]
    start = canopy.initial_substep(
        n_slots, canopy.stage_slots(options.substep.integrator), len(BALANCE_CHECKS)
    )._replace(
        tair_profile=c.air_temperature,
        eair_profile=c.vapor_pressure,
        cair_profile=c.co2,
        tleaf_leaf=c.leaf_temperature,
        lwp_leaf=c.leaf_water_potential,
        h2ocan_profile=c.intercepted_water,
        tg_soil=c.ground_temperature,
        balance_error=errors,
    )
    sub_options = options.substep

    def body(carry, calday):
        return (
            canopy.substep(carry, calday, sub_inputs, params, tables, options=sub_options, c3=c3),
            None,
        )

    if options.checkpoint_substeps:
        body = jax.checkpoint(body)
    carry = (start, canopy.zero_sums(canopy.substep_fluxes(start, lay)))
    (end, sums), _ = jax.lax.scan(body, carry, substep_days(inputs.calday_beg, options))

    # Averages over the sub-steps (layers in use, levels up to the top layer in use; the
    # other slots keep the sub-step values) — MLTimeStepFluxIntegration
    means = canopy.average(sums, options.n_substeps)
    up_to_top = jnp.arange(lay.dz.shape[0] + 1) <= lay.n_layers

    # Canopy totals (the averages through a barrier, as Fortran stores them first: C19)
    leaves = canopy.CanopyLeaves(
        dpai=dpai, fracsun=end.fracsun_profile, fwet=end.fwet_profile, fdry=end.fdry_profile,
        wind=end.wind_profile, tair=end.tair_profile, vcmax25=end.vcmax25_profile,
        tleaf=end.tleaf_leaf, lwp=end.lwp_leaf, vcmax25_leaf=end.vcmax25_leaf,
    )  # fmt: skip
    totals = canopy.canopy_totals(
        lay,
        jax.lax.optimization_barrier(means),
        leaves,
        area.lai,
        area.sai,
        flux_profile_type=options.flux_profile_type,
    )
    errors = end.balance_error
    for i in range(4):
        errors = _record(errors, CANOPY_ENERGY_1 + i, totals.errors[i])

    # Sunlit and shaded leaves merged to their layer mean (the state is prognostic)
    merged = canopy.merge_sun_shade(lay, dpai, end.fracsun_profile, end.tleaf_leaf, end.lwp_leaf)

    # Soil temperature — MLSoilTemperatureMod::SoilTemperature (thermal properties of
    # before the canopy fluxes: unchanged in between, decision C11)
    gsoi = means.gsoi
    t_soil_new, soil_error = soil.soil_temperature(
        t_soil, thermal.tk, thermal.cv, k.z, gsoi, options.step_seconds
    )
    errors = _record(errors, SOIL_TEMPERATURE, soil_error)

    new_state = State(
        canopy=CanopyState(
            air_temperature=end.tair_profile,
            vapor_pressure=end.eair_profile,
            co2=end.cair_profile,
            leaf_temperature=keep(use, merged.tleaf, end.tleaf_leaf),
            leaf_water_potential=keep(use, merged.lwp, end.lwp_leaf),
            intercepted_water=end.h2ocan_profile,
            ground_temperature=end.tg_soil,
            acclimation_temperature=c.acclimation_temperature,
            acclimation_steps=c.acclimation_steps,
        ),
        soil=SoilState(temperature=t_soil_new),
    )
    diagnostics = Diagnostics(
        tref_forcing=end.tref_forcing,
        pref_forcing=end.pref_forcing,
        rhomol_forcing=end.rhomol_forcing,
        swskyb_forcing=means.swskyb,
        swskyd_forcing=means.swskyd,
        solar_zen_forcing=solar_zen,
        lai_canopy=area.lai,
        sai_canopy=area.sai,
        sai_floor_canopy=area.sai_floor,
        laisun_canopy=totals.laisun,
        laisha_canopy=totals.laisha,
        dpai_profile=dpai,
        fracsun_profile=end.fracsun_profile,
        albcan_canopy=totals.albcan,
        swveg_canopy=totals.swveg,
        swvegsun_canopy=totals.swvegsun,
        swvegsha_canopy=totals.swvegsha,
        lwup_canopy=means.lwup,
        rnet_canopy=totals.rnet,
        shflx_canopy=totals.shflx,
        lhflx_canopy=totals.lhflx,
        stflx_air_canopy=totals.stflx_air,
        stflx_veg_canopy=totals.stflx_veg,
        ustar_canopy=means.ustar,
        beta_canopy=means.beta,
        shveg_canopy=totals.shveg,
        shvegsun_canopy=totals.shvegsun,
        shvegsha_canopy=totals.shvegsha,
        lhveg_canopy=totals.lhveg,
        lhvegsun_canopy=totals.lhvegsun,
        lhvegsha_canopy=totals.lhvegsha,
        trveg_canopy=totals.trveg,
        evveg_canopy=totals.evveg,
        gppveg_canopy=totals.gppveg,
        gppvegsun_canopy=totals.gppvegsun,
        gppvegsha_canopy=totals.gppvegsha,
        vcmax25veg_canopy=totals.vcmax25veg,
        vcmax25sun_canopy=totals.vcmax25sun,
        vcmax25sha_canopy=totals.vcmax25sha,
        gsveg_canopy=totals.gsveg,
        gsvegsun_canopy=totals.gsvegsun,
        gsvegsha_canopy=totals.gsvegsha,
        windveg_canopy=totals.windveg,
        windvegsun_canopy=totals.windvegsun,
        windvegsha_canopy=totals.windvegsha,
        tlveg_canopy=totals.tlveg,
        tlvegsun_canopy=totals.tlvegsun,
        tlvegsha_canopy=totals.tlvegsha,
        taveg_canopy=totals.taveg,
        tavegsun_canopy=totals.tavegsun,
        tavegsha_canopy=totals.tavegsha,
        fracminlwp_canopy=totals.fracminlwp,
        rnsoi_soil=means.rnsoi,
        shsoi_soil=means.shsoi,
        lhsoi_soil=means.lhsoi,
        gsoi_soil=gsoi,
        btran_soil=end.btran_soil,
        psis_soil=roots.psis,
        wind_profile=end.wind_profile,
        shair_profile=keep(use, means.shair, end.shair_profile),
        etair_profile=keep(use, means.etair, end.etair_profile),
        mflx_profile=keep(use, means.mflx, end.mflx_profile),
        gac_profile=keep(use, means.gac, end.gac_profile),
        lsc_profile=lsc,
        lwp_mean_profile=keep(use, totals.lwp_mean, fill_layer),
        td_profile=end.td_profile,
        lwdwn_profile=keep(up_to_top, means.lwdwn, end.lwdwn_profile),
        lwupw_profile=keep(up_to_top, means.lwupw, end.lwupw_profile),
        swbeam_profile=keep(up_to_top, means.swbeam, end.swbeam_profile),
        swdwn_profile=keep(up_to_top, means.swdwn, end.swdwn_profile),
        swupw_profile=keep(up_to_top, means.swupw, end.swupw_profile),
        apar_leaf=end.apar_leaf,
        rnleaf_leaf=means.rnleaf,
        shleaf_leaf=means.shleaf,
        lhleaf_leaf=means.lhleaf,
        anet_leaf=means.anet,
        gs_leaf=means.gs,
        vcmax25_leaf=end.vcmax25_leaf,
        tleaf_hist_leaf=keep(use, merged.tleaf_hist, _fill(end.tleaf_leaf)),
        lwp_hist_leaf=keep(use, merged.lwp_hist, _fill(end.tleaf_leaf)),
        balance_error=errors,
    )
    return StepState(state=new_state, diagnostics=diagnostics)
