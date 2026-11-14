"""
Model output: named variables with units, per patch and per gridcell.

The variables below are computed from a step's results (:func:`step_fields`: the 0-based
diagnostics, prognostic state and layer grid; Phase 7.6) inside the compiled chunk, where
output streams accumulate them. Each variable says which legacy ``.out`` file column it
corresponds to, so results can be compared with earlier runs and with the Fortran model.

Dimensions: ``time``, ``patch``, ``layer`` (canopy-air layers, bottom first, padded to
``max_layers`` with NaN), ``leaf_class`` (sunlit, shaded), ``leaf_group`` (canopy total,
sunlit, shaded), ``band`` (vis, nir), ``soil_layer``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from clm_ml.io import netcdf
from clm_ml.params import PFT_NAMES

# Physical constants used only to derive output quantities (as in CLMml_driver.output).
_MMH2O, _MMDRY = 18.02e-3, 28.97e-3  # kg/mol, as in MLclm_varcon


@dataclasses.dataclass(frozen=True)
class VariableSpec:
    name: str
    units: str
    long_name: str
    dims: tuple[str, ...]  # after (time, patch)
    legacy: str  # where the value appears in the legacy .out files
    compute: Callable[["_Raw"], np.ndarray]  # -> (time, patch, *dims)


def step_fields(step, layers) -> dict:
    """One patch's step results as named 0-based fields: ``diagnostics.<name>``
    (:class:`clm_ml.state.Diagnostics`, CLM-ml names), ``state.<name>`` (the prognostic
    state; ``state.soil_temperature``) and ``layers.<name>`` (:class:`clm_ml.state.
    PatchLayers`). These are the raw fields a validation stream can name and the regression
    reference records."""
    out = {f"diagnostics.{n}": v for n, v in step.diagnostics._asdict().items()
           if n != "balance_error"}  # fmt: skip
    out.update({f"state.{n}": v for n, v in step.state.canopy._asdict().items()})
    out["state.soil_temperature"] = step.state.soil.temperature
    out.update({f"layers.{n}": getattr(layers, n) for n in layers._fields})
    return out


def field_names() -> tuple:
    """The names :func:`step_fields` gives."""
    from clm_ml.state import CanopyState, Diagnostics, PatchLayers

    return (
        tuple(f"diagnostics.{n}" for n in Diagnostics._fields if n != "balance_error")
        + tuple(f"state.{n}" for n in CanopyState._fields)
        + ("state.soil_temperature",)
        + tuple(f"layers.{n}" for n in PatchLayers._fields)
    )


#: Fields the variables name by their CLM-ml names besides the diagnostics
_ALIASES = {"tair_profile": "state.air_temperature", "eair_profile": "state.vapor_pressure",
            "zs_profile": "layers.zs", "dz_profile": "layers.dz", "zw_profile": "layers.zw"}  # fmt: skip
#: Fields on the level axis (the ground, then the top of each layer): the variables take
#: the layers' tops
_LEVEL = {"zw_profile", "lwdwn_profile", "lwupw_profile", "swbeam_profile", "swdwn_profile",
          "swupw_profile"}  # fmt: skip


class _Raw:
    """Step fields (:func:`step_fields`, leading axes (time, patch)), with helpers for
    slicing profiles to the layers in use.

    ``xp`` is the array module: ``numpy`` for recorded fields, or ``jax.numpy`` inside a
    compiled chunk (output streams accumulate the variables during the scan). The variable
    definitions below use ``r.xp``, so each is written once for both."""

    def __init__(self, raw: Mapping[str, np.ndarray], max_layers: int, xp=np):
        self.raw = raw
        self.max_layers = max_layers
        self.xp = xp
        self.ncan = raw["layers.n_layers"].astype(int)  # (nt, np)
        self.ntop = raw["layers.n_within"].astype(int)
        self.first_leafy = raw["layers.first_leafy"].astype(int)

    def __getitem__(self, name: str) -> np.ndarray:
        value = self.raw[_ALIASES.get(name, f"diagnostics.{name}")]
        return value[:, :, 1:] if name in _LEVEL else value

    def layer(self, values: np.ndarray, *, within_canopy_only: bool = False) -> np.ndarray:
        """Profile (nt, np, max_layers, ...) with NaN above the top layer (or above the
        canopy, if ``within_canopy_only``)."""
        xp = self.xp
        out = xp.asarray(values[:, :, : self.max_layers], dtype=float)
        top = self.ntop if within_canopy_only else self.ncan
        k = xp.arange(self.max_layers)
        invalid = k[None, None, :] >= top[:, :, None]
        invalid = invalid.reshape(invalid.shape + (1,) * (out.ndim - 3))
        return xp.where(invalid, xp.nan, out)

    def leaf(self, values: np.ndarray) -> np.ndarray:
        """Leaf profile (nt, np, max_layers, 2: sunlit, shaded), NaN in layers without
        leaves."""
        xp = self.xp
        out = self.layer(values, within_canopy_only=True)
        has_leaf = self.layer(self["dpai_profile"], within_canopy_only=True) > 0  # NaN -> False
        return xp.where(has_leaf[..., None], out, xp.nan)

    def by_group(self, total: str, sun: str, sha: str, band: int | None = None) -> np.ndarray:
        parts = [self[total], self[sun], self[sha]]
        if band is not None:
            parts = [p[..., band] for p in parts]
        return self.xp.stack(parts, axis=-1)

    def at_layer(self, values: np.ndarray, index: np.ndarray) -> np.ndarray:
        """``values`` (nt, np, layer) at the 0-based layer ``index`` (nt, np)."""
        return self.xp.take_along_axis(values, index[:, :, None], axis=2)[:, :, 0]

    def latvap_tref(self) -> np.ndarray:
        """Molar latent heat of vaporization at the reference air temperature (J/mol), as
        clm_ml.processes.water_vapor.lat_vap."""
        from clm_ml.processes.constants import hsub, hvap, mmh2o, tfrz

        t = self["tref_forcing"]
        return self.xp.where(t > tfrz, hvap, hsub) * mmh2o


def _shortwave_up(r: _Raw) -> np.ndarray:
    sky = r["swskyb_forcing"] + r["swskyd_forcing"]
    return (r["albcan_canopy"] * sky).sum(axis=-1)


def _mid_layer(r: _Raw) -> np.ndarray:
    """The 0-based mid-canopy layer of the legacy aux.out (1-based ``max(1, nbot + (ntop -
    nbot + 1) / 2 - 1)``)."""
    nbot, ntop = r.first_leafy + 1, r.ntop
    return r.xp.maximum(1, nbot + (ntop - nbot + 1) // 2 - 1) - 1


def _specific_humidity(r: _Raw) -> np.ndarray:
    e = r.layer(r["eair_profile"])
    p = r["pref_forcing"][:, :, None]
    ratio = _MMH2O / _MMDRY
    return ratio * e / (p - (1.0 - ratio) * e)


V = VariableSpec
VARIABLES: tuple[VariableSpec, ...] = (
    # --- canopy-scale fluxes (flux.out) -----------------------------------------------
    V("net_radiation", "W m-2", "Net radiation above the canopy", (), "flux.out col 2",
      lambda r: r["rnet_canopy"]),
    V("canopy_air_storage_flux", "W m-2", "Heat storage in canopy air", (), "flux.out col 3",
      lambda r: r["stflx_air_canopy"]),
    V("sensible_heat_flux", "W m-2", "Sensible heat flux above the canopy", (), "flux.out col 4",
      lambda r: r["shflx_canopy"]),
    V("latent_heat_flux", "W m-2", "Latent heat flux above the canopy", (), "flux.out col 5",
      lambda r: r["lhflx_canopy"]),
    V("gpp", "umol CO2 m-2 s-1", "Gross primary production", (), "flux.out col 6",
      lambda r: r["gppveg_canopy"]),
    V("friction_velocity", "m s-1", "Friction velocity", (), "flux.out col 7",
      lambda r: r["ustar_canopy"]),
    V("shortwave_up", "W m-2", "Reflected shortwave radiation", (), "flux.out col 8",
      _shortwave_up),
    V("longwave_up", "W m-2", "Upward longwave radiation above the canopy", (), "flux.out col 9",
      lambda r: r["lwup_canopy"]),
    V("canopy_top_air_temperature", "K", "Air temperature at the canopy top layer", (),
      "flux.out col 10", lambda r: r.at_layer(r["tair_profile"], r.ntop - 1)),
    V("ground_heat_flux", "W m-2", "Soil heat flux (positive into soil)", (), "flux.out col 11",
      lambda r: r["gsoi_soil"]),
    V("soil_net_radiation", "W m-2", "Net radiation at the soil surface", (), "flux.out col 12",
      lambda r: r["rnsoi_soil"]),
    V("soil_sensible_heat_flux", "W m-2", "Sensible heat flux from the soil", (),
      "flux.out col 13", lambda r: r["shsoi_soil"]),
    V("soil_latent_heat_flux", "W m-2", "Latent heat flux from the soil", (), "flux.out col 14",
      lambda r: r["lhsoi_soil"]),
    V("transpiration_latent_heat_flux", "W m-2", "Latent heat flux from transpiration", (),
      "flux.out col 15", lambda r: r["trveg_canopy"] * r.latvap_tref()),
    V("canopy_evaporation_latent_heat_flux", "W m-2",
      "Latent heat flux from evaporation of intercepted water", (), "flux.out col 16",
      lambda r: r["evveg_canopy"] * r.latvap_tref()),
    V("rsl_beta", "1", "Roughness-sublayer beta: friction velocity / wind speed at canopy top",
      (), "flux.out col 17", lambda r: r["beta_canopy"]),
    V("vegetation_storage_flux", "W m-2", "Heat storage in vegetation", (), "flux.out col 18",
      lambda r: r["stflx_veg_canopy"]),
    # --- sunlit/shaded canopy totals (fsun.out) ---------------------------------------
    V("solar_zenith_angle", "degree", "Solar zenith angle", (), "fsun.out col 1",
      lambda r: r.xp.degrees(r["solar_zen_forcing"])),
    V("incident_visible_radiation", "W m-2", "Incident visible solar radiation", (),
      "fsun.out col 2", lambda r: r["swskyb_forcing"][..., 0] + r["swskyd_forcing"][..., 0]),
    V("leaf_area_index", "m2 m-2", "Leaf area index", (), "fsun.out col 3 (with sai)",
      lambda r: r["lai_canopy"]),
    V("stem_area_index", "m2 m-2", "Stem area index (prescribed + stem_area_added)", (),
      "fsun.out col 3 (with lai)", lambda r: r["sai_canopy"]),
    V("stem_area_added", "m2 m-2",
      "Stem area added so every leafy layer has min_layer_plant_area (issue 14)", (), "",
      lambda r: r["sai_floor_canopy"]),
    V("leaf_area_index_by_leaf_class", "m2 m-2", "Sunlit and shaded leaf area index",
      ("leaf_class",), "fsun.out cols 4-5",
      lambda r: r.xp.stack([r["laisun_canopy"], r["laisha_canopy"]], axis=-1)),
    V("absorbed_visible_radiation", "W m-2", "Visible radiation absorbed by vegetation",
      ("leaf_group",), "fsun.out cols 6-8",
      lambda r: r.by_group("swveg_canopy", "swvegsun_canopy", "swvegsha_canopy", band=0)),
    V("gpp_by_leaf_group", "umol CO2 m-2 s-1", "Gross primary production",
      ("leaf_group",), "fsun.out cols 9-11",
      lambda r: r.by_group("gppveg_canopy", "gppvegsun_canopy", "gppvegsha_canopy")),
    V("vegetation_latent_heat_flux", "W m-2", "Latent heat flux from vegetation",
      ("leaf_group",), "fsun.out cols 12-14",
      lambda r: r.by_group("lhveg_canopy", "lhvegsun_canopy", "lhvegsha_canopy")),
    V("vegetation_sensible_heat_flux", "W m-2", "Sensible heat flux from vegetation",
      ("leaf_group",), "fsun.out cols 15-17",
      lambda r: r.by_group("shveg_canopy", "shvegsun_canopy", "shvegsha_canopy")),
    V("canopy_vcmax25", "umol m-2 s-1", "Canopy Vcmax at 25 C", ("leaf_group",),
      "fsun.out cols 18-20",
      lambda r: r.by_group("vcmax25veg_canopy", "vcmax25sun_canopy", "vcmax25sha_canopy")),
    V("canopy_stomatal_conductance", "mol H2O m-2 s-1", "Canopy stomatal conductance",
      ("leaf_group",), "fsun.out cols 21-23",
      lambda r: r.by_group("gsveg_canopy", "gsvegsun_canopy", "gsvegsha_canopy")),
    V("leaf_wind_speed", "m s-1", "Wind speed at leaves (leaf-area weighted)",
      ("leaf_group",), "fsun.out cols 24-26",
      lambda r: r.by_group("windveg_canopy", "windvegsun_canopy", "windvegsha_canopy")),
    V("mean_leaf_temperature", "K", "Leaf temperature (leaf-area weighted)",
      ("leaf_group",), "fsun.out cols 27-29",
      lambda r: r.by_group("tlveg_canopy", "tlvegsun_canopy", "tlvegsha_canopy")),
    V("leaf_air_temperature", "K", "Air temperature at leaves (leaf-area weighted)",
      ("leaf_group",), "fsun.out cols 30-32",
      lambda r: r.by_group("taveg_canopy", "tavegsun_canopy", "tavegsha_canopy")),
    # --- plant water (aux.out) ----------------------------------------------------------
    V("soil_water_stress", "1", "Soil water stress factor (btran)", (), "aux.out col 1",
      lambda r: r["btran_soil"]),
    V("top_leaf_specific_conductance", "mmol H2O m-2 s-1 MPa-1",
      "Leaf-specific conductance, canopy top layer", (), "aux.out col 2",
      lambda r: r.at_layer(r["lsc_profile"], r.ntop - 1)),
    V("soil_water_potential", "MPa", "Weighted soil water potential", (), "aux.out col 3",
      lambda r: r["psis_soil"]),
    V("top_leaf_water_potential", "MPa", "Mean leaf water potential, canopy top layer", (),
      "aux.out col 4", lambda r: r.at_layer(r["lwp_mean_profile"], r.ntop - 1)),
    V("mid_leaf_water_potential", "MPa", "Mean leaf water potential, mid-canopy layer", (),
      "aux.out col 5", lambda r: r.at_layer(r["lwp_mean_profile"], _mid_layer(r))),
    V("fraction_at_minimum_leaf_water_potential", "1",
      "Fraction of canopy at minimum leaf water potential", (), "aux.out col 6",
      lambda r: r["fracminlwp_canopy"]),
    # --- vertical profiles (profile.out, fluxprofile.out) -------------------------------
    V("layer_height", "m", "Height of layer center", ("layer",), "profile.out col 2",
      lambda r: r.layer(r["zs_profile"])),
    V("layer_top_height", "m", "Height of layer top interface", ("layer",),
      "fluxprofile.out col 2", lambda r: r.layer(r["zw_profile"])),
    V("plant_area_density", "m2 m-3", "Plant area density", ("layer",), "profile.out col 4",
      lambda r: r.layer(r["dpai_profile"], within_canopy_only=True)
      / r.layer(r["dz_profile"], within_canopy_only=True)),
    V("sunlit_fraction", "1", "Sunlit fraction of leaves", ("layer",), "profile.out col 3",
      lambda r: r.layer(r["fracsun_profile"], within_canopy_only=True)),
    V("air_temperature", "K", "Air temperature", ("layer",), "profile.out col 26",
      lambda r: r.layer(r["tair_profile"])),
    V("vapor_pressure", "Pa", "Water vapor pressure", ("layer",), "profile.out (via col 27)",
      lambda r: r.layer(r["eair_profile"])),
    V("specific_humidity", "kg kg-1", "Specific humidity", ("layer",),
      "profile.out col 27 (g/kg there)", _specific_humidity),
    V("wind_speed", "m s-1", "Wind speed", ("layer",), "profile.out col 25",
      lambda r: r.layer(r["wind_profile"])),
    V("aerodynamic_resistance", "s m-1", "Aerodynamic resistance between layers", ("layer",),
      "profile.out col 28",
      lambda r: r["rhomol_forcing"][:, :, None] / r.layer(r["gac_profile"])),
    V("sensible_heat_flux_profile", "W m-2", "Sensible heat flux at layer top", ("layer",),
      "fluxprofile.out col 3", lambda r: r.layer(r["shair_profile"])),
    V("latent_heat_flux_profile", "W m-2", "Latent heat flux at layer top", ("layer",),
      "fluxprofile.out col 4",
      lambda r: r.layer(r["etair_profile"]) * r.latvap_tref()[:, :, None]),
    V("momentum_flux_profile", "m2 s-2", "Momentum flux at layer top", ("layer",),
      "fluxprofile.out col 5", lambda r: r.layer(r["mflx_profile"])),
    V("direct_shortwave_down_profile", "W m-2", "Direct beam solar radiation", ("layer", "band"),
      "fluxprofile.out cols 6-7", lambda r: r.layer(r["swbeam_profile"])),
    V("diffuse_shortwave_down_profile", "W m-2", "Downward diffuse solar radiation",
      ("layer", "band"), "fluxprofile.out cols 8-9",
      lambda r: r.layer(r["swdwn_profile"])),
    V("diffuse_shortwave_up_profile", "W m-2", "Upward diffuse solar radiation",
      ("layer", "band"), "fluxprofile.out cols 10-11",
      lambda r: r.layer(r["swupw_profile"])),
    V("longwave_down_profile", "W m-2", "Downward longwave radiation (within canopy only)",
      ("layer",), "fluxprofile.out col 12 (0 above canopy there)",
      lambda r: r.layer(r["lwdwn_profile"], within_canopy_only=True)),
    V("longwave_up_profile", "W m-2", "Upward longwave radiation (within canopy only)",
      ("layer",), "fluxprofile.out col 13 (0 above canopy there)",
      lambda r: r.layer(r["lwupw_profile"], within_canopy_only=True)),
    # --- leaf profiles, sunlit and shaded (profile.out) ---------------------------------
    V("leaf_net_radiation", "W m-2", "Leaf net radiation (per leaf area)",
      ("layer", "leaf_class"), "profile.out cols 7-8", lambda r: r.leaf(r["rnleaf_leaf"])),
    V("leaf_sensible_heat_flux", "W m-2", "Leaf sensible heat flux (per leaf area)",
      ("layer", "leaf_class"), "profile.out cols 9-10", lambda r: r.leaf(r["shleaf_leaf"])),
    V("leaf_latent_heat_flux", "W m-2", "Leaf latent heat flux (per leaf area)",
      ("layer", "leaf_class"), "profile.out cols 11-12", lambda r: r.leaf(r["lhleaf_leaf"])),
    V("leaf_net_photosynthesis", "umol CO2 m-2 s-1", "Leaf net photosynthesis (per leaf area)",
      ("layer", "leaf_class"), "profile.out cols 13-14", lambda r: r.leaf(r["anet_leaf"])),
    V("leaf_absorbed_par", "umol photon m-2 s-1", "Leaf absorbed PAR (per leaf area)",
      ("layer", "leaf_class"), "profile.out cols 15-16", lambda r: r.leaf(r["apar_leaf"])),
    V("leaf_stomatal_conductance", "mol H2O m-2 s-1", "Leaf stomatal conductance",
      ("layer", "leaf_class"), "profile.out cols 17-18", lambda r: r.leaf(r["gs_leaf"])),
    V("leaf_water_potential", "MPa", "Leaf water potential", ("layer", "leaf_class"),
      "profile.out cols 19-20", lambda r: r.leaf(r["lwp_hist_leaf"])),
    V("leaf_temperature", "K", "Leaf temperature", ("layer", "leaf_class"),
      "profile.out cols 21-22", lambda r: r.leaf(r["tleaf_hist_leaf"])),
    V("leaf_vcmax25", "umol m-2 s-1", "Leaf Vcmax at 25 C", ("layer", "leaf_class"),
      "profile.out cols 23-24", lambda r: r.leaf(r["vcmax25_leaf"])),
)  # fmt: skip

VARIABLES_BY_NAME: Mapping[str, VariableSpec] = {v.name: v for v in VARIABLES}
# Variables that cannot be averaged across patches with different layer grids.
_PATCH_ONLY_DIMS = {"layer"}

#: Gridcell fields written with a grid run's gridcell output (decision 108): the file name,
#: units and long name of clm_ml.domain.Gridcells' area, land_fraction, vegetated_fraction
GRIDCELL_AREA = {
    "area": ("area", "km2", "Gridcell area (CLM history area)"),
    "land_fraction": ("landfrac", "1", "Land fraction of the gridcell (CLM history landfrac)"),
    "vegetated_fraction": (
        "vegetated_fraction", "1",
        "Fraction of the land covered by PFTs 1-16 (the patches run; surface data). A total: "
        "value * area * landfrac * vegetated_fraction",
    ),
}  # fmt: skip

COORDINATES = {
    "leaf_class": ["sunlit", "shaded"],
    "leaf_group": ["canopy", "sunlit", "shaded"],
    "band": ["vis", "nir"],
}


@dataclasses.dataclass(frozen=True)
class Outputs:
    """Per-patch output: ``data[name]`` has shape ``(time, patch, ...)``.

    Variables are also available as attributes (``outputs.latent_heat_flux``).
    """

    data: Mapping[str, np.ndarray]
    time: np.ndarray  # datetime64, end of each time step
    patch_gridcell: np.ndarray  # (patch,) int
    patch_weight: np.ndarray  # (patch,)
    patch_pft: tuple[str, ...]
    gridcell_names: tuple[str, ...]
    soil_temperature: np.ndarray | None = None  # (time, patch, soil_layer) K
    soil_depth: np.ndarray | None = None  # (patch, soil_layer) m, node depth
    raw: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict, repr=False)
    # Area-weighted gridcell sums formed at every step inside the run (Phase 6 step 3),
    # ``(time, gridcell, ...)``: used for the gridcell level, so min/max are right there
    gridcell_data: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict, repr=False)
    # Grid runs: the gridcells' latitude-longitude map (clm_ml.domain.LatLonMap), on which
    # the gridcell level is written (decision 108); None: per gridcell
    gridcell_map: object = None
    # Grid runs: {Gridcells field: (gridcell,) values} of GRIDCELL_AREA, written on the map
    gridcell_area: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict, repr=False)

    def __getattr__(self, name: str) -> np.ndarray:
        data = object.__getattribute__(self, "data")
        if name in data:
            return data[name]
        raise AttributeError(name)

    @classmethod
    def from_records(cls, records: Mapping[str, np.ndarray], meta: Mapping) -> "Outputs":
        """One output stream's records (``{"time": ..., variable: (record, patch, ...)}``).
        ``meta`` is :func:`domain_meta` plus ``soil_depth`` ``(patch, soil_layer)``. Raw
        step fields (``diagnostics.<name>`` etc., validation; :func:`step_fields`) go to
        ``raw``."""
        soil = records.get("soil_temperature")
        gridcell = "gridcell:"
        return cls(
            data={
                n: np.asarray(v, dtype=float)
                for n, v in records.items()
                if n not in ("time", "soil_temperature") and "." not in n and ":" not in n
            },
            gridcell_data={
                n[len(gridcell) :]: np.asarray(v, dtype=float)
                for n, v in records.items()
                if n.startswith(gridcell)
            },
            time=np.asarray(records["time"], dtype="datetime64[ns]"),
            patch_gridcell=np.asarray(meta["patch_gridcell"]),
            patch_weight=np.asarray(meta["patch_weight"], dtype=float),
            patch_pft=tuple(meta["patch_pft"]),
            gridcell_names=tuple(meta["gridcell_names"]),
            soil_temperature=None if soil is None else np.asarray(soil, dtype=float),
            soil_depth=np.asarray(meta["soil_depth"]),
            raw={n: v for n, v in records.items() if "." in n},
            gridcell_map=meta.get("gridcell_map"),
            gridcell_area={n: v for n, v in meta.get("gridcell_area", {}).items() if v is not None},
        )

    def split(self, file_per: str, run_start: dt.datetime) -> list:
        """``[(label, Outputs), ...]``, one per output file: the whole run (labelled by the run
        start date), or per year / month of the records (by the start of their interval)."""
        if file_per == "run":
            return [(f"{run_start:%Y-%m-%d}", self)]
        # A record's time is the end of its interval; one microsecond earlier is inside it.
        inside = self.time - np.timedelta64(1, "us")
        fmt = "%Y" if file_per == "year" else "%Y-%m"
        labels = np.array([dt.datetime.fromisoformat(str(t)[:26]).strftime(fmt) for t in inside])
        return [(label, self.take(labels == label)) for label in dict.fromkeys(labels)]

    def take(self, mask: np.ndarray) -> "Outputs":
        """The records selected by a boolean ``mask`` over time."""
        return dataclasses.replace(
            self,
            data={n: v[mask] for n, v in self.data.items()},
            time=self.time[mask],
            soil_temperature=None if self.soil_temperature is None else self.soil_temperature[mask],
            raw={n: v[mask] for n, v in self.raw.items()},
            gridcell_data={n: v[mask] for n, v in self.gridcell_data.items()},
        )

    def select(self, names: Sequence[str]) -> "Outputs":
        unknown = sorted(set(names) - set(self.data))
        if unknown:
            raise KeyError(f"unknown output variable(s) {unknown}; see clm_ml.outputs.VARIABLES")
        return dataclasses.replace(
            self,
            data={n: self.data[n] for n in names},
            gridcell_data={n: v for n, v in self.gridcell_data.items() if n in names},
        )

    def gridcell_mean(self) -> dict[str, np.ndarray]:
        """Area-weighted mean over each gridcell's patches, for variables without a
        ``layer`` dimension (patches on a gridcell have different layer grids).
        Returns ``{name: array (time, gridcell, ...)}``. Values summed over the gridcell's
        patches at every step inside the run (``gridcell_data``) are used when present (the
        only right ones for min and max); otherwise the patch records are averaged."""
        n_g = len(self.gridcell_names)
        out = {}
        for name, values in self.data.items():
            if _PATCH_ONLY_DIMS & set(VARIABLES_BY_NAME[name].dims):
                continue
            if name in self.gridcell_data:
                out[name] = self.gridcell_data[name]
                continue
            w = self.patch_weight.reshape((1, -1) + (1,) * (values.ndim - 2))
            summed = np.zeros((values.shape[0], n_g) + values.shape[2:])
            np.add.at(summed, (slice(None), self.patch_gridcell), values * w)
            out[name] = summed
        return out

    def to_xarray(self, level: str = "patch"):
        """``xarray.Dataset`` with units and long names. ``level`` is ``"patch"`` or
        ``"gridcell"`` (area-weighted means; profile variables are omitted). A grid run's
        gridcell level is on its latitude-longitude map (:meth:`_latlon_xarray`)."""
        import xarray as xr

        if level == "gridcell" and self.gridcell_map is not None:
            return self._latlon_xarray()
        if level == "gridcell":
            data, unit_dim = self.gridcell_mean(), "gridcell"
        elif level == "patch":
            data, unit_dim = self.data, "patch"
        else:
            raise ValueError(f"level must be 'patch' or 'gridcell', got {level!r}")

        variables = {}
        for name, values in data.items():
            spec = VARIABLES_BY_NAME[name]
            variables[name] = xr.Variable(
                ("time", unit_dim) + spec.dims,
                values,
                attrs={"units": spec.units, "long_name": spec.long_name, "legacy": spec.legacy},
            )
        coords = {"time": self.time, **{k: v for k, v in COORDINATES.items()}}
        if level == "patch":
            coords.update(
                gridcell=("patch", self.patch_gridcell),
                weight=("patch", self.patch_weight),
                pft=("patch", list(self.patch_pft)),
            )
            if self.soil_temperature is not None:
                variables["soil_temperature"] = xr.Variable(
                    ("time", "patch", "soil_layer"),
                    self.soil_temperature,
                    attrs={"units": "K", "long_name": "Soil temperature"},
                )
                coords["soil_depth"] = (("patch", "soil_layer"), self.soil_depth, {"units": "m"})
        else:
            coords["gridcell"] = list(self.gridcell_names)
        ds = xr.Dataset(variables, coords=coords)
        ds.attrs["title"] = "CLM-ML-JAX multilayer canopy model output"
        ds.attrs["time_convention"] = "time is the end of each output interval"
        return ds

    def _latlon_xarray(self):
        """The gridcell level of a grid run on its latitude-longitude map (decision 108):
        variables ``(time, ..., lat, lon)``, NaN where no gridcell was run (ocean, ice,
        unvegetated land, outside the region), ``run_mask`` (1 where one was) and the
        gridcells' area, land fraction and vegetated fraction (``GRIDCELL_AREA``)."""
        import xarray as xr

        m = self.gridcell_map
        rows, cols = m.index[:, 0], m.index[:, 1]
        shape = (m.lat.size, m.lon.size)
        variables = {}
        for name, values in self.gridcell_mean().items():
            spec = VARIABLES_BY_NAME[name]
            per_cell = np.moveaxis(values, 1, -1)  # (time, ..., gridcell)
            mapped = np.full(per_cell.shape[:-1] + shape, np.nan)
            mapped[..., rows, cols] = per_cell
            variables[name] = xr.Variable(
                ("time",) + spec.dims + ("lat", "lon"),
                mapped,
                attrs={"units": spec.units, "long_name": spec.long_name, "legacy": spec.legacy},
            )
        mask = np.zeros(shape, dtype=np.int8)
        mask[rows, cols] = 1
        variables["run_mask"] = xr.Variable(
            ("lat", "lon"), mask, attrs={"long_name": "1: a gridcell of the run"}
        )
        for field, values in self.gridcell_area.items():
            name, units, long_name = GRIDCELL_AREA[field]
            mapped = np.full(shape, np.nan)
            mapped[rows, cols] = values
            variables[name] = xr.Variable(
                ("lat", "lon"), mapped, attrs={"units": units, "long_name": long_name}
            )
        coords = {
            "time": self.time,
            **COORDINATES,
            "lat": ("lat", m.lat, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", m.lon, {"units": "degrees_east", "long_name": "longitude"}),
        }
        ds = xr.Dataset(variables, coords=coords)
        ds.attrs["title"] = "CLM-ML-JAX multilayer canopy model output"
        ds.attrs["time_convention"] = "time is the end of each output interval"
        return ds

    def to_netcdf(
        self, path: Path, levels: Sequence[str] = ("patch", "gridcell"), append: bool = False
    ) -> list[Path]:
        """Write one file per requested level (``*_patch.nc``, ``*_gridcell.nc``); with
        ``append``, add the records to the existing files along ``time`` (a file continued
        after a restart)."""
        import os

        import xarray as xr

        written = []
        for level in levels:
            out = Path(path).with_name(f"{Path(path).stem}_{level}.nc")
            ds = self.to_xarray(level)
            with netcdf.LOCK:  # no forcing read while a file is written (O31)
                if append and out.exists():
                    with xr.open_dataset(out) as old:
                        ds = xr.concat(
                            [old.load(), ds], dim="time", data_vars="minimal", coords="minimal",
                            compat="override",
                        )  # fmt: skip
                    tmp = out.with_name(out.name + ".tmp")
                    ds.to_netcdf(tmp)
                    os.replace(tmp, out)
                else:
                    ds.to_netcdf(out)
            written.append(out)
        return written


def domain_meta(domain) -> dict:
    """Patch and gridcell metadata the output files carry (``gridcell_map``: a grid run's
    :class:`clm_ml.domain.LatLonMap`, or None)."""
    return {
        "patch_gridcell": domain.patches.gridcell_index.tolist(),
        "patch_weight": domain.patches.weight.tolist(),
        "patch_pft": [PFT_NAMES[i] for i in domain.patches.pft_index],
        "gridcell_names": list(domain.gridcells.names),
        "gridcell_map": domain.gridcells.latlon,
        "gridcell_area": {n: getattr(domain.gridcells, n) for n in GRIDCELL_AREA},
    }


class StreamFileWriter:
    """Writes one output stream's netCDF files as their periods complete (``file_per``
    run, year or month), so a long run holds at most one file's records in memory.

    Files are ``{directory}/{tower}_{stream}_{label}_{level}.nc``."""

    def __init__(self, stream, variables, meta, directory: Path, tower: str, run_start):
        self.stream, self.variables, self.meta = stream, tuple(variables), meta
        self.directory, self.tower, self.run_start = Path(directory), tower, run_start
        self._label = None
        self._records: list = []
        self._started: set = set()  # labels whose file holds records written before a restart
        self.written: list[Path] = []

    def _label_of(self, end) -> str:
        if self.stream.file_per == "run":
            return f"{self.run_start:%Y-%m-%d}"
        inside = end - dt.timedelta(microseconds=1)  # the record's interval ends at `end`
        return inside.strftime("%Y" if self.stream.file_per == "year" else "%Y-%m")

    def add(self, records) -> None:
        """Records ``[(key, end_time, {variable: value}), ...]`` in time order."""
        for record in records:
            label = self._label_of(record[1])
            if self._label is not None and label != self._label:
                self._flush()
            self._label = label
            self._records.append(record)

    def close(self) -> None:
        if self._records:
            self._flush()

    def state(self) -> dict:
        """For a restart: the completed records are written to their file first (a restart
        holds no output history, Phase 6 review R6); the run continuing from the restart
        appends to that file. Returns the file in progress."""
        if self._records:
            self._flush()
            self._started.add(self._label)
        return {"label": self._label, "started": sorted(self._started)}

    def restore(self, saved: dict) -> None:
        self._label, self._records = saved["label"], []
        self._started = set(saved.get("started", ()))

    def _flush(self) -> None:
        outputs = Outputs.from_records(records_to_arrays(self._records, self.variables), self.meta)
        if outputs.data or outputs.soil_temperature is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{self.tower}_{self.stream.name}_{self._label}.nc"
            files = outputs.to_netcdf(path, self.stream.level, append=self._label in self._started)
            self.written += [f for f in files if f not in self.written]
        self._records = []


def records_to_arrays(records, variables) -> dict:
    """``[(key, end, {variable: value}), ...]`` -> ``{"time": ..., variable: stacked}``."""
    arrays = {"time": np.array([end for _, end, _ in records], dtype="datetime64[ns]")}
    for name in variables:
        arrays[name] = np.stack([v[name] for _, _, v in records])
    return arrays
