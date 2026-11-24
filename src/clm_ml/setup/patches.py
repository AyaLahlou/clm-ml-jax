"""Every patch's constants and initial state: :func:`build_patches`.

From the domain (``clm_ml.domain.Domain``: gridcells, patches, soil columns), the parameters
(``clm_ml.params.Params``), the model options (``clm_ml.config.ModelConfig``) and the first
step's forcing, with the Fortran setup's arithmetic (``clm_ml.setup.soil``,
``clm_ml.setup.canopy``). Arrays have a leading patch axis.

Fortran reference: CLM-ml v2 ``CLMml_driver.F90`` (``TowerVeg``, ``SoilInit``,
``init_acclim``), ``initVerticalMod.F90``, ``SoilStateInitTimeConstMod.F90``,
``SurfaceAlbedoMod.F90``, ``MLinitVerticalMod.F90``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from clm_ml.setup import canopy, soil
from clm_ml.state import CanopyState, PatchLayers, SoilState, State
from clm_ml.step import PatchConstants


class PatchSetup(NamedTuple):
    """The patches set up (NumPy, leading patch axis)."""

    constants: PatchConstants
    state: State  # before the first step
    c3: np.ndarray  # (P,) bool: C3 photosynthesis (static per group of patches)
    pft: np.ndarray  # (P,) int: the PFT (row of the parameter table)
    soil_depth: np.ndarray  # (P, nlevgrnd) soil node depths, m (output metadata)


def patch_heights(domain) -> np.ndarray:
    """The forcing (reference) height of each patch, m: the patch's own measurement height
    when the domain has one per patch (grid runs), else its gridcell's (30 m if missing, as
    Fortran ``TowerMetMod``)."""
    p = domain.patches
    if p.measurement_height is not None:
        return np.asarray(p.measurement_height, dtype=float)
    h = np.asarray(domain.gridcells.measurement_height, dtype=float)[p.gridcell_index]
    return np.where(np.isnan(h) | (np.round(h) == -999), 30.0, h)


def build_patches(domain, params, config, first: dict, acclimation_steps: float = 0.0):
    """The :class:`PatchSetup` of every patch of ``domain``.

    ``first``: per patch (P,) values of the first step: ``uref`` (m/s), ``tref`` (K),
    ``qref`` (kg/kg), ``pref`` (Pa), ``pco2`` (Pa) of the forcing; ``tsoi`` (P, nlevgrnd) the
    initial soil temperature (K); ``tacclim`` (P,) the initial acclimation temperature (K).
    ``acclimation_steps``: the steps the initial acclimation temperature counts for (0: the
    first step's update sets it to that step's air temperature, which it already is)."""
    pt, cols = domain.patches, domain.columns
    pft = np.asarray(pt.pft_index, dtype=int)
    gi, ci = np.asarray(pt.gridcell_index, dtype=int), np.asarray(pt.column_index, dtype=int)
    n_p = len(pft)
    layers_cfg = config.canopy_layers
    max_layers = int(layers_cfg.max_layers)

    # vegetation (TowerVeg, getPADparameters)
    htop = canopy.canopy_heights(pt.canopy_height, pft)
    root = np.asarray(pt.root_biomass, dtype=float)
    if not (root > 0.0).all():  # NaN: missing
        bad = [int(i) for i in np.flatnonzero(~(root > 0.0))]
        raise ValueError(f"patches {bad}: root biomass must be given and positive (TowerVeg)")
    lai_b, sai_b = canopy.area_profile_parameters(
        pt.leaf_area_profile_beta, pt.stem_area_profile_beta, pft
    )
    own = np.asarray(pt.n_layers_within, dtype=int) > 0
    counts = layers_cfg.layer_counts  # model.yaml: explicit counts, or from the spacing (0)
    model_within, model_above = (0, 0) if isinstance(counts, str) else (counts.within, counts.above)
    within = np.where(own, pt.n_layers_within, model_within).astype(int)
    above = np.where(own, pt.n_layers_above, model_above).astype(int)
    zref = patch_heights(domain)
    grid = canopy.vertical_structure(
        zref, htop, lai_b, sai_b, within, above, max_layers, float(layers_cfg.dz_tall),
        float(layers_cfg.dz_short), float(layers_cfg.tall_canopy_threshold),
        float(layers_cfg.merge_thin_layers_below),
    )  # fmt: skip

    # soil (initVertical, SoilStateInitTimeConst, SurfaceAlbedoInitTimeConst)
    nlevsoi, nlevgrnd = soil.NLEVSOI, soil.NLEVGRND
    z, dz, zi = soil.soil_layers()
    nbed_col = soil.bedrock_index(np.asarray(cols.bedrock_depth, dtype=float), zi, nlevsoi)
    # a missing organic matter content (NaN) is passed on as -999
    organic = np.where(np.isnan(cols.organic_matter), -999.0, cols.organic_matter)
    props = soil.soil_properties(
        np.asarray(cols.sand_percent, dtype=float), np.asarray(cols.clay_percent, dtype=float),
        organic, list(cols.texture_class), z, nlevsoi,
    )  # fmt: skip
    albsat, albdry = soil.soil_albedos(cols.color_class)
    beta = np.asarray(params.values["root_profile_beta"], dtype=float)[pft]
    nbed = nbed_col[ci]
    rootfr = soil.root_fractions(beta, zi, nbed, nlevsoi)

    def per_patch(a):
        return np.broadcast_to(a, (n_p,) + np.shape(a)).copy()

    g = domain.gridcells
    constants = PatchConstants(
        layers=PatchLayers(
            n_layers=grid["n_layers"],
            n_within=grid["n_within"],
            first_leafy=grid["first_leafy"],
            dz=grid["dz"],
            zs=grid["zs"],
            zw=grid["zw"],
        ),
        dlai_frac=grid["dlai_frac"],
        dsai_frac=grid["dsai_frac"],
        zref=zref.astype(float),
        ztop=htop.astype(float),
        root_biomass=root,
        rootfr=rootfr,
        dz=per_patch(dz),
        z=per_patch(z),
        zi=per_patch(zi[1:]),
        soil_dz=per_patch(z[0] - zi[0]),
        nbedrock=nbed.astype(np.int32),
        **{k: props[k][ci] for k in ("watsat", "sucsat", "bsw", "hksat", "tkmg", "tkdry", "csol")},
        albsat=albsat[ci],
        albdry=albdry[ci],
        latdeg=np.asarray(g.latitude, dtype=float)[gi],
        londeg=np.asarray(g.longitude, dtype=float)[gi],
    )

    # initial state (initVerticalProfiles, SoilInit, init_acclim)
    init = canopy.initial_canopy_state(
        grid["n_layers"], max_layers, first["uref"], np.zeros(n_p), first["tref"],
        first["qref"], first["pref"], first["pco2"],
    )  # fmt: skip
    state = State(
        canopy=CanopyState(
            air_temperature=init["air_temperature"],
            vapor_pressure=init["vapor_pressure"],
            co2=init["co2"],
            leaf_temperature=init["leaf_temperature"],
            leaf_water_potential=init["leaf_water_potential"],
            intercepted_water=init["intercepted_water"],
            ground_temperature=init["ground_temperature"],
            acclimation_temperature=np.asarray(first["tacclim"], dtype=float),
            acclimation_steps=np.full(n_p, float(acclimation_steps)),
        ),
        soil=SoilState(temperature=np.asarray(first["tsoi"], dtype=float)[:, :nlevgrnd]),
    )
    c3 = np.round(np.asarray(params.values["c3_photosynthesis"], dtype=float)[pft]) == 1
    return PatchSetup(constants, state, c3, pft, per_patch(z))
