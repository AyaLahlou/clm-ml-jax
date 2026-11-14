"""
Physical parameters per plant functional type (PFT): the contents of ``params.yaml``.

Every patch looks up its PFT's row, so parameters are shared by all patches of the same
PFT and gradients come back per PFT. All values are differentiable.

A params file is **complete** (decision in ``docs/design/decisions.md``): it lists every
parameter of every PFT 1-16, with nothing inherited from another file, so a run's parameters
are all in the one file it names::

    units: {vcmax25: umol m-2 s-1, ...}
    pft:
      needleleaf_evergreen_temperate_tree:
        vcmax25: 62.5
        ...                         # every parameter
      ...                           # every PFT 1-16

Only the canopy roughness ratios (``z0m_to_height_ratio``, ``displacement_to_height_ratio``),
which only grid runs use, may be left out; a grid run then refuses the file. To make a new
parameter set, copy a complete file (``clm-ml params chats -o mine.yaml``) and edit it.

Compsets (``clm_ml/compsets/params/``, copied into a folder by ``clm-ml new``): ``clm5_defaults`` (CLM5 / CLM-ml v1 values),
``chats`` (``clm5_defaults`` with the CHATS walnut orchard values; Bonan, Burns & Patton
2026, Table 2) and ``clm6``.

Every params file is YAML in this one format. To use a CLM parameter file (netCDF, e.g. the
CTSM 6.0 file), convert it once (``clm-ml params ctsm60_params.nc -o params_clm6.yaml``,
:mod:`clm_ml.params_convert`); the compset ``clm6`` is that conversion of the CTSM 6.0 file.
"""

from __future__ import annotations

import dataclasses
from importlib import resources
from pathlib import Path
from typing import Mapping, NamedTuple, Optional

import numpy as np

from clm_ml.io.yaml_schema import ConfigError, read_yaml


class CanopyParams(NamedTuple):
    """Parameters of one patch's PFT (JAX scalars; spectral fields are ``[vis, nir]``, band 0
    visible, band 1 near-infrared, as everywhere in the model)."""

    # Photosynthesis and stomatal conductance
    vcmax25: object
    iota: object
    gs_min: object
    g0_ball_berry: object
    g1_ball_berry: object
    g0_medlyn: object
    g1_medlyn: object
    psi50_gs: object
    shape_gs: object
    # Plant hydraulics and roots
    plant_conductance: object
    plant_capacitance: object
    root_radius: object
    root_density: object
    root_resistivity: object
    # Leaf structure and radiation
    leaf_dimension: object
    leaf_angle_index: object
    sla_top: object
    leaf_emissivity: object
    clumping_index: object
    leaf_reflectance: object
    leaf_transmittance: object
    stem_reflectance: object
    stem_transmittance: object


#: CLM PFT names, in CLM index order (index 0 = not_vegetated). Only PFTs 1-16 have
#: multilayer canopy parameters; CLM's crop PFTs (17-78) are not supported.
PFT_NAMES: tuple[str, ...] = (
    "not_vegetated",
    "needleleaf_evergreen_temperate_tree",
    "needleleaf_evergreen_boreal_tree",
    "needleleaf_deciduous_boreal_tree",
    "broadleaf_evergreen_tropical_tree",
    "broadleaf_evergreen_temperate_tree",
    "broadleaf_deciduous_tropical_tree",
    "broadleaf_deciduous_temperate_tree",
    "broadleaf_deciduous_boreal_tree",
    "broadleaf_evergreen_shrub",
    "broadleaf_deciduous_temperate_shrub",
    "broadleaf_deciduous_boreal_shrub",
    "c3_arctic_grass",
    "c3_non-arctic_grass",
    "c4_grass",
    "c3_crop",
    "c3_irrigated",
)


@dataclasses.dataclass(frozen=True)
class ParamSpec:
    name: str  # params.yaml / Params field name
    units: str
    description: str
    legacy_table: str  # "MLpftcon" (multilayer_canopy.MLpftconMod) or "pftcon" (clm_src_main)
    legacy_field: str
    spectral: bool = False  # [vis, nir] pair
    # Not needed by every run (e.g. only grid runs use the roughness ratios): a missing value
    # is an error only where it is used
    optional: bool = False


PARAM_SPECS: tuple[ParamSpec, ...] = (
    # Photosynthesis and stomatal conductance
    ParamSpec("vcmax25", "umol m-2 s-1", "Maximum carboxylation rate at 25 C, canopy top",
              "MLpftcon", "vcmaxpft"),
    ParamSpec("iota", "umol CO2 mol-1 H2O", "Marginal water-use efficiency (WUE stomatal model)",
              "MLpftcon", "iota_SPA"),
    ParamSpec("gs_min", "mol H2O m-2 s-1", "Minimum stomatal conductance",
              "MLpftcon", "gsmin_SPA"),
    ParamSpec("g0_ball_berry", "mol H2O m-2 s-1", "Ball-Berry minimum leaf conductance",
              "MLpftcon", "g0_BB"),
    ParamSpec("g1_ball_berry", "-", "Ball-Berry slope", "MLpftcon", "g1_BB"),
    ParamSpec("g0_medlyn", "mol H2O m-2 s-1", "Medlyn minimum leaf conductance",
              "MLpftcon", "g0_MED"),
    ParamSpec("g1_medlyn", "kPa0.5", "Medlyn slope", "MLpftcon", "g1_MED"),
    ParamSpec("psi50_gs", "MPa", "Leaf water potential at 50% loss of stomatal conductance",
              "MLpftcon", "psi50_gs"),
    ParamSpec("shape_gs", "-", "Shape parameter for stomatal closure with leaf water potential",
              "MLpftcon", "shape_gs"),
    ParamSpec("c3_photosynthesis", "1", "1 for C3 photosynthesis, 0 for C4 (structural)",
              "pftcon", "c3psn"),
    # Plant hydraulics and roots
    ParamSpec("plant_conductance", "mmol H2O m-2 s-1 MPa-1",
              "Stem hydraulic conductance per unit leaf area", "MLpftcon", "gplant_SPA"),
    ParamSpec("plant_capacitance", "mmol H2O m-2 MPa-1", "Plant capacitance per unit leaf area",
              "MLpftcon", "capac_SPA"),
    ParamSpec("root_radius", "m", "Fine root radius", "MLpftcon", "root_radius_SPA"),
    ParamSpec("root_density", "g m-3", "Fine root density (biomass per root volume)",
              "MLpftcon", "root_density_SPA"),
    ParamSpec("root_resistivity", "MPa s g mmol-1 H2O", "Hydraulic resistivity of root tissue",
              "MLpftcon", "root_resist_SPA"),
    ParamSpec("root_profile_beta", "-", "Root profile shape parameter", "pftcon", "rootprof_beta"),
    # Leaf structure and radiation
    ParamSpec("leaf_dimension", "m", "Leaf dimension (for boundary-layer conductance)",
              "pftcon", "dleaf"),
    ParamSpec("leaf_angle_index", "-", "Leaf angle departure from spherical (chi_l)",
              "pftcon", "xl"),
    ParamSpec("sla_top", "m2 gC-1", "Specific leaf area at canopy top", "pftcon", "slatop"),
    ParamSpec("leaf_emissivity", "-", "Leaf emissivity", "MLpftcon", "emleaf"),
    ParamSpec("clumping_index", "-", "Foliage clumping index", "MLpftcon", "clump_fac"),
    ParamSpec("leaf_reflectance", "-", "Leaf reflectance [vis, nir]", "pftcon", "rhol", True),
    ParamSpec("leaf_transmittance", "-", "Leaf transmittance [vis, nir]", "pftcon", "taul", True),
    ParamSpec("stem_reflectance", "-", "Stem reflectance [vis, nir]", "pftcon", "rhos", True),
    ParamSpec("stem_transmittance", "-", "Stem transmittance [vis, nir]", "pftcon", "taus", True),
    # Canopy roughness (CLM z0mr, displar): only the forcing height of grid runs uses them
    # (30 m + (z0m + displacement) above the ground, CLM's rule); not CLM-ml physics parameters
    ParamSpec("z0m_to_height_ratio", "-", "Momentum roughness length / canopy height (CLM z0mr)",
              "", "z0mr", optional=True),
    ParamSpec("displacement_to_height_ratio", "-",
              "Displacement height / canopy height (CLM displar)", "", "displar", optional=True),
)  # fmt: skip

SPECS_BY_NAME: Mapping[str, ParamSpec] = {s.name: s for s in PARAM_SPECS}


@dataclasses.dataclass(frozen=True)
class Params:
    """Per-PFT parameter table.

    ``values[name]`` has shape ``(n_pft,)``, or ``(n_pft, 2)`` for spectral [vis, nir]
    parameters, with rows in :data:`PFT_NAMES` order. PFTs without values are NaN.
    """

    values: Mapping[str, np.ndarray]
    source: str = ""  # file the table was loaded from (for provenance)

    def __getattr__(self, name: str) -> np.ndarray:
        if name in SPECS_BY_NAME:
            return self.values[name]
        raise AttributeError(name)

    @classmethod
    def from_yaml(cls, path_or_name: Path | str) -> "Params":
        """Load a complete params file (a path, or a compset name such as ``"chats"``)."""
        path = _resolve(path_or_name)
        return cls.from_data(read_yaml(path), path.name)

    @classmethod
    def from_data(cls, data: dict, source: str) -> "Params":
        """A complete params table from a params file's contents (``units``, ``pft``)."""
        merged = _check_data(data, source)
        params = cls._from_merged(merged, source)
        missing = params.missing_for(list(range(1, len(PFT_NAMES))))
        if missing:
            shown = ", ".join(missing[:8]) + (
                f" and {len(missing) - 8} more" if len(missing) > 8 else ""
            )
            raise ConfigError(
                f"{source}: a params file must give every parameter of every PFT 1-16 "
                f"(no defaults are filled in); missing: {shown}"
            )
        return params

    @classmethod
    def _from_merged(cls, merged: dict[str, dict[str, object]], source: str) -> "Params":
        values = {
            spec.name: np.full((len(PFT_NAMES), 2) if spec.spectral else len(PFT_NAMES), np.nan)
            for spec in PARAM_SPECS
        }
        for pft, entries in merged.items():
            row = pft_index(pft, source)
            for name, value in entries.items():
                spec = SPECS_BY_NAME.get(name)
                if spec is None:
                    raise ConfigError(
                        f"{source}: pft.{pft}.{name}: unknown parameter; "
                        f"allowed: {sorted(SPECS_BY_NAME)}"
                    )
                values[name][row] = _check_value(spec, value, f"{source}: pft.{pft}.{name}")
        return cls(values=values, source=source)

    def get(self, pft: str, name: str) -> np.ndarray:
        return self.values[name][pft_index(pft, "get()")]

    def set(self, pft: str, **changes) -> "Params":
        """Return a copy with values changed for one PFT."""
        row = pft_index(pft, "set()")
        values = {k: v.copy() for k, v in self.values.items()}
        for name, value in changes.items():
            if name not in SPECS_BY_NAME:
                raise ConfigError(f"set(): unknown parameter {name!r}")
            values[name][row] = value
        return Params(values=values, source=self.source)

    def canopy_params(self, pft: str):
        """The parameters of one PFT as the processes take them
        (:class:`CanopyParams`, JAX scalars; spectral ``[vis, nir]``)."""
        import jax.numpy as jnp

        row = pft_index(pft, "canopy_params()")
        return CanopyParams(**{
            name: jnp.asarray(self.values[name][row], dtype=jnp.float64)
            for name in CanopyParams._fields
        })  # fmt: skip

    def missing_for(self, pft_rows: list[int], names=None) -> list[str]:
        """Parameters with no value (NaN) for any of the given PFT rows: the required ones,
        or those in ``names``."""
        names = names or [s.name for s in PARAM_SPECS if not s.optional]
        return sorted(
            f"{PFT_NAMES[row]}.{name}"
            for name in names
            for row in pft_rows
            if np.isnan(self.values[name][row]).any()
        )

    def describe(self, pft_rows: Optional[list[int]] = None) -> str:
        rows = pft_rows if pft_rows is not None else range(1, len(PFT_NAMES))
        lines = []
        for row in rows:
            lines.append(PFT_NAMES[row])
            for spec in PARAM_SPECS:
                value = self.values[spec.name][row]
                shown = f"[{value[0]:g}, {value[1]:g}]" if spec.spectral else f"{value:g}"
                lines.append(f"  {spec.name:22s} {shown:>16s}  {spec.units}")
        return "\n".join(lines)


def pft_index(name: str, source: str) -> int:
    try:
        return PFT_NAMES.index(name)
    except ValueError:
        raise ConfigError(
            f"{source}: unknown PFT {name!r}; allowed: {', '.join(PFT_NAMES[1:])}"
        ) from None


def builtin_params_path(name: str) -> Path:
    return Path(str(resources.files("clm_ml") / "compsets" / "params" / f"{name}.yaml"))


def _check_value(spec: ParamSpec, value, where: str):
    if spec.spectral:
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            raise ConfigError(f"{where}: expected [vis, nir], got {value!r}")
        for v in value:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ConfigError(f"{where}: expected numbers, got {value!r}")
        return [float(v) for v in value]
    if spec.name == "c3_photosynthesis":
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true or false, got {value!r}")
        return float(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: expected a number, got {value!r}")
    return float(value)


def _resolve(path_or_name: Path | str) -> Path:
    text = str(path_or_name)
    if not text.endswith((".yaml", ".yml")) and "/" not in text:
        return builtin_params_path(text)
    return Path(text)


def _check_data(data: dict, name: str) -> dict:
    """``{pft: {name: value}}`` of a params file's contents, checked for unknown keys."""
    if "extends" in data:
        raise ConfigError(
            f"{name}: 'extends' is not supported: a params file gives every parameter of "
            f"every PFT itself. Write a complete copy to edit with "
            f"`clm-ml params {data['extends']} -o {Path(name).name}`"
        )
    unknown = sorted(set(data) - {"pft", "units", "description"})
    if unknown:
        hint = (
            "; a CLM parameter file is converted with `clm-ml params FILE.nc -o FILE.yaml`"
            if "clm_parameters" in unknown
            else ""
        )
        raise ConfigError(f"{name}: unknown key(s) {unknown}; allowed: pft, units{hint}")
    pfts = data.get("pft") or {}
    if not isinstance(pfts, dict):
        raise ConfigError(f"{name}: pft: expected a mapping of PFT names to parameters")
    for pft, entries in pfts.items():
        pft_index(pft, name)
        if not isinstance(entries, dict):
            raise ConfigError(f"{name}: pft.{pft}: expected a mapping of parameters")
    return {pft: dict(entries) for pft, entries in pfts.items()}
