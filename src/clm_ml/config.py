"""
Model configuration: the contents of ``model.yaml``.

These settings choose how each process is represented (parameterizations) and the numerics. They are *static*: changing any of
them changes the compiled program, and none of them are differentiable. Numeric values
that should be calibrated belong in ``params.yaml`` instead.

Each option's comment names the Fortran ``MLclm_varctl`` variable and code it replaces.
:data:`FORTRAN_CODES` holds that mapping in both directions.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal, Union

from clm_ml.io.yaml_schema import ConfigError, from_dict, option, read_yaml, render, to_dict


@dataclasses.dataclass(frozen=True)
class LayerCounts:
    """Explicit numbers of canopy-air layers."""

    within: int = option(doc="Layers within the canopy.", fortran="nlayer_within")
    above: int = option(doc="Layers above the canopy, up to the forcing height.",
                        fortran="nlayer_above")  # fmt: skip


@dataclasses.dataclass(frozen=True)
class PhotosynthesisOptions:
    """Leaf photosynthesis (Farquhar) and the leaf nitrogen profile through the canopy."""

    colimitation: bool = option(
        True,
        "Co-limitation (smoothed minimum) of the Rubisco, light and product limited rates; false "
        "takes the plain minimum.",
        "colim_type 1/0",
    )
    temperature_acclimation: Literal["running_mean", "run_mean", "none"] = option(
        "running_mean",
        "Acclimation of the photosynthetic temperature responses (Kattge and Knorr 2007) to a "
        "running mean of air temperature carried in the state (as CLM), to the mean over the "
        "whole run (Fortran CLM-ml v2; needs all forcing before the run starts: validation "
        "only), or none.",
        "acclim_type 1/0; run_mean = init_acclim (CLMml_driver)",
    )
    acclimation_days: float = option(
        10.0,
        "Length of the acclimation running mean (days).",
        when={"temperature_acclimation": "running_mean"},
    )
    nitrogen_profile: Union[Literal["computed"], float] = option(
        "computed",
        "Decay coefficient of leaf nitrogen (Vcmax25) through the canopy: computed from Vcmax25 "
        "at the canopy top, exp(0.00963 Vcmax25 - 2.43) (Bonan et al. 2021), or a number.",
        "kn_val (-999 = computed)",
    )


@dataclasses.dataclass(frozen=True)
class StomatalConductanceOptions:
    """Stomatal conductance."""

    scheme: Literal["wue", "medlyn", "ball_berry"] = option(
        "wue",
        "Water-use-efficiency optimization (iota) with the plant hydraulics constraint (CLM-ml "
        "v2), or the Medlyn or Ball-Berry relation (their g0, g1 in params.yaml).",
        "gs_type 2/0/1",
    )
    water_stress: bool = option(
        True,
        "Reduce stomatal conductance as leaf water potential falls, gs = max(gs / (1 + "
        "(psi_leaf/psi50_gs)^shape_gs), gs_min) (Bonan et al. 2021, supplement eq. 14).",
        "gspot_type 1/0",
    )
    wue_solver: Literal["bisection", "brent"] = option(
        "bisection",
        "Root finder of the WUE optimization: bisection (CLM-ml v2). brent is refused (not "
        "translated as traceable code; decisions.md).",
        "gs_solver 2/1",
        when={"scheme": "wue"},
    )


@dataclasses.dataclass(frozen=True)
class RadiationOptions:
    """Radiative transfer through the canopy."""

    solar: Literal["two_stream", "norman"] = option(
        "two_stream",
        "Shortwave radiation: two-stream, integrated over each layer. norman is refused (not "
        "yet supported).",
        "light_type 2/1",
    )
    longwave: Literal["norman"] = option("norman", "Longwave radiation: Norman (the only one).",
                                         "longwave_type 1")  # fmt: skip
    leaf_optics_vary_with_height: bool = option(
        False,
        "Leaf optical properties varying with height in the canopy (not supported).",
        "leaf_optics_type 0/1",
    )


@dataclasses.dataclass(frozen=True)
class TurbulenceOptions:
    """Turbulence and aerodynamic conductances above and within the canopy, and the leaf
    boundary layer."""

    roughness_sublayer: Literal["harman_finnigan"] = option(
        "harman_finnigan",
        "Harman and Finnigan roughness sublayer with the psihat tables (the only one).",
        "turb_type 1",
    )
    sparse_canopy: bool = option(
        True,
        "Harman-Finnigan adjustments for sparse canopies: the displacement height and the "
        "canopy-top Prandtl number depend on the plant area index.",
        "sparse_canopy_type 1/0",
    )
    ground_conductance: Literal["log_profile", "extend_hf"] = option(
        "log_profile",
        "Aerodynamic conductance of the lowest canopy layer to the ground: a log wind profile "
        "(CLM-ml v2), or the Harman-Finnigan exponential profile extended down to the ground.",
        "HF_extension_type 2/1",
    )
    leaf_boundary_layer: Literal["free_convection", "laminar", "turbulent"] = option(
        "free_convection",
        "Leaf boundary-layer conductance: forced plus free convection (CLM-ml v2), or laminar or "
        "turbulent forced convection only.",
        "gb_type 3/1/2",
    )


@dataclasses.dataclass(frozen=True)
class CanopyAirOptions:
    """Air temperature and vapour in the canopy layers."""

    flux_profile_solution: Literal["implicit", "well_mixed"] = option(
        "implicit",
        "Coupled implicit solution of air temperature, vapour and leaf temperature in every "
        "layer. well_mixed is refused (Fortran calls it legacy, not supported).",
        "flux_profile_type 1/0",
    )


@dataclasses.dataclass(frozen=True)
class ProcessOptions:
    """How each process is represented: the parameterizations (schemes) and their options.
    The defaults are CLM-ml v2 as run for CHATS (Bonan et al. 2026), validated against
    Fortran."""

    photosynthesis: PhotosynthesisOptions = option(PhotosynthesisOptions(), "")
    stomatal_conductance: StomatalConductanceOptions = option(StomatalConductanceOptions(), "")
    radiation: RadiationOptions = option(RadiationOptions(), "")
    turbulence: TurbulenceOptions = option(TurbulenceOptions(), "")
    canopy_air: CanopyAirOptions = option(CanopyAirOptions(), "")


@dataclasses.dataclass(frozen=True)
class NumericsOptions:
    """Time stepping, solvers and run control."""

    step_seconds: int = option(
        1800,
        "Model time step (s). It must divide the tower forcing interval (run.yaml "
        "forcing.meteorology.interval_seconds); a shorter step interpolates the forcing.",
        "",
    )
    substep_seconds: float = option(
        300.0,
        "Canopy sub-step (s): each step is split into sub-steps, forcing interpolated within "
        "the step, output averaged over them.",
        "dtime_ml",
    )
    integrator: Literal[
        "euler",
        "rk2_trapezoid",
        "rk2_midpoint",
        "rk2_ralston",
        "rk3_heun",
        "rk3_ralston",
        "rk3_kutta",
        "rk4",
    ] = option(
        "rk4",
        "Time integration of the sub-step state (air temperature, vapour, leaf temperature, "
        "leaf water potential, intercepted water, ground temperature): 4th-order Runge-Kutta "
        "in CLM-ml v2 (v1 used Euler).",
        "runge_kutta_type",
    )
    wetted_fraction: Literal["implicit", "explicit"] = option(
        "implicit",
        "Wetted fraction of the leaves in a sub-step: implicit (of the intercepted water left "
        "at the end of the sub-step, so leaves cannot evaporate more water than they hold; "
        "decision 122) or explicit (of the water at the start of each Runge-Kutta stage, as "
        "Fortran CLM-ml v2: almost dry leaves alternate between wet and dry and their water "
        "turns negative; for comparison with Fortran).",
        "",
    )
    water_stress_derivative: Literal["stabilized", "exact"] = option(
        "stabilized",
        "Derivative of the water stress of the stomata (gradients only; forward results are "
        "the same either way). The stomata take the stress of the leaf water potential at "
        "the start of each Runge-Kutta stage; near the stress threshold a leaf alternates "
        "between open and closed, and the exact derivative of that grows without bound over "
        "days. stabilized: the derivative of the potential consistent with the transpiration "
        "it gives (bounded; 30-day mean derivatives of stressed patches within about 6 % of "
        "a finite change instead of 35-360 % off; no cost), an approximation: not the exact "
        "derivative of the forward values at a stressed step. exact: the exact derivative "
        "(decision 124).",
        "",
    )
    obukhov_derivative: Literal["exact", "stabilized"] = option(
        "exact",
        "Derivative of the Obukhov length (gradients only; forward results are the same "
        "either way). Each Runge-Kutta stage solves the Obukhov length from the canopy-top "
        "air at its start and then gives new air; in stability transitions this loop "
        "overshoots from stage to stage, and the exact derivative of that grows without "
        "bound over days (common in regimes like the wet-season cerrado, rare at a tower "
        "like CHATS7). exact: the exact derivative. stabilized: where the loop overshoots, "
        "the derivative of the length consistent with the air the stage gives (bounded; "
        "monthly-mean derivatives within about 12 % of a finite change; the gradient takes "
        "1.7 times as long and half as much memory again), an approximation: not the exact "
        "derivative of the forward values at an affected step (decision 124).",
        "",
    )
    checkpoint_steps: bool = option(
        True,
        "Recompute each step when taking gradients instead of storing its intermediate values "
        "(jax.checkpoint, as jax-gcm): much less memory for gradients over a chunk, some more "
        "compute. Results are the same either way.",
        "",
    )
    checkpoint_substeps: Literal["auto", "always", "never"] = option(
        "auto",
        "Whether gradients also recompute each sub-step of a step (a second jax.checkpoint, "
        "inside the step): the gradient keeps the intermediate values of one sub-step instead "
        "of a whole step. always: about a fifth of the memory with many patches, at the same "
        "speed on a GPU (4 096 patches do not fit on a 40 GB GPU without it) but six times "
        "slower on the CPU (18 patches). never: the reverse. auto: always on a GPU, never on "
        "the CPU. Results are the same either way (decision 125).",
        "",
    )
    table_search: Literal["auto", "loop", "unrolled", "compare_all"] = option(
        "auto",
        "How the psihat lookup tables are searched: loop (a binary search as a compiled loop, "
        "fastest on the CPU), unrolled (the same binary search without the loop, fastest on a "
        "GPU, where every loop iteration is a separate command) or compare_all. auto: "
        "unrolled on a GPU, loop on the CPU. The table indices are the same either way "
        "(decision 114).",
        "",
        hidden=True,
    )
    solver_loops: Literal["auto", "loop", "unrolled"] = option(
        "auto",
        "How the loops over canopy layers (shortwave sweeps, longwave and flux-profile "
        "tridiagonal solves) and the Obukhov and roughness-length iterations are compiled: "
        "loop (compiled loops: small programs, fastest on the CPU) or unrolled (the same "
        "operations written out one after another: faster on a GPU, about three times the "
        "compile time). auto: unrolled on a GPU, loop on the CPU. Results agree to round-off; "
        "the stomatal solvers, sub-steps and soil always stay loops (decision 117).",
        "",
        hidden=True,
    )
    on_patch_error: Literal["stop", "exclude"] = option(
        "stop",
        "A patch that fails a conservation check or whose state becomes unhealthy: stop the "
        "run naming it (as Fortran), or exclude it from then on (global runs; its output is "
        "fill values and gridcell sums use the remaining weights).",
        "",
    )
    max_failed_patch_fraction: float = option(
        0.01,
        "With exclude: stop the run when more than this fraction of patches has failed.",
        "",
    )
    ignore_checks: tuple[str, ...] = option(
        (),
        "Conservation checks that neither stop the run nor exclude a patch (their largest "
        "error is still reported), by (part of) their name in clm_ml.checks.BALANCE_CHECKS, "
        "e.g. 'CanopyFluxesDiagnostics 3'. Normally empty: every check passes for CHATS7 and "
        "the 1-degree grid runs.",
        "",
    )


@dataclasses.dataclass(frozen=True)
class CanopyLayerOptions:
    """The vertical layers of the canopy air space, from the ground to the forcing height.
    A patch's own layer counts (domain.yaml patches[].canopy_layers) take precedence over
    these."""

    max_layers: int = option(
        50,
        "Layer slots of every patch (one compiled program for all patches; unused slots are "
        "masked). CHATS7 uses 46.",
        "nlevmlcan",
    )
    layer_counts: Union[Literal["from_spacing"], LayerCounts] = option(
        "from_spacing",
        "Numbers of layers within and above the canopy: from_spacing (layers of dz_tall or "
        "dz_short thickness), or explicit counts {within: N, above: M} (5-10 within the canopy "
        "suffice for fluxes, Bonan et al. 2021).",
        "nlayer_within, nlayer_above (0 = from spacing)",
    )
    dz_tall: float = option(
        0.5,
        "Layer thickness (m) for canopies taller than tall_canopy_threshold.",
        "dz_tall",
        when={"layer_counts": "from_spacing"},
    )
    dz_short: float = option(
        0.1,
        "Layer thickness (m) for shorter canopies.",
        "dz_short",
        when={"layer_counts": "from_spacing"},
    )
    tall_canopy_threshold: float = option(
        2.0,
        "Canopy height (m) above which dz_tall is used.",
        "dz_param",
        when={"layer_counts": "from_spacing"},
    )
    merge_thin_layers_below: float = option(
        0.01,
        "At setup: layers holding less than this fraction of the canopy's plant area are "
        "emptied and their area given to the other layers.",
        "dpai_min",
    )
    minimum_leafy_layer_plant_area: float = option(
        0.01,
        "At every step: every leafy layer keeps at least this plant area (m2 m-2), the missing "
        "part added as stem area: thinner layers transmit more diffuse radiation than they "
        "receive (decisions.md; CLM-ml v1's dpai_min). 0 reproduces CLM-ml v2.",
        "",
    )


@dataclasses.dataclass(frozen=True)
class SoilOptions:
    """The soil under the patches (the CLM soil layer structure: 20 hydrologically active
    levels, 25 in all)."""

    columns: Literal["per_patch", "shared"] = option(
        "per_patch",
        "per_patch: each patch has its own soil column (option A). shared: patches on a "
        "gridcell share one column (option B; not yet implemented).",
        "",
    )


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Contents of ``model.yaml``; the defaults reproduce CLM-ml v2/CHATS."""

    processes: ProcessOptions = ProcessOptions()
    numerics: NumericsOptions = NumericsOptions()
    canopy_layers: CanopyLayerOptions = CanopyLayerOptions()
    soil: SoilOptions = SoilOptions()

    def __post_init__(self):
        if self.soil.columns == "shared":
            raise ConfigError(
                "model.yaml: soil.columns: 'shared' (option B) is planned but not implemented; "
                "use 'per_patch'"
            )
        if self.processes.photosynthesis.acclimation_days <= 0:
            raise ConfigError(
                "model.yaml: processes.photosynthesis.acclimation_days must be positive"
            )
        if self.numerics.substep_seconds <= 0:
            raise ConfigError("model.yaml: numerics.substep_seconds must be positive")
        if not 0.0 <= self.numerics.max_failed_patch_fraction <= 1.0:
            raise ConfigError("model.yaml: numerics.max_failed_patch_fraction must be in [0, 1]")
        if self.canopy_layers.max_layers < 2:
            raise ConfigError("model.yaml: canopy_layers.max_layers must be at least 2")
        counts = self.canopy_layers.layer_counts
        if isinstance(counts, LayerCounts) and (counts.within < 1 or counts.above < 1):
            raise ConfigError("model.yaml: canopy_layers.layer_counts: within and above >= 1")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ModelConfig":
        path = Path(path)
        return cls.from_dict(read_yaml(path), source=path.name)

    @classmethod
    def from_dict(cls, data: dict, source: str = "model.yaml") -> "ModelConfig":
        return from_dict(cls, data, source=source)

    def to_yaml(self, path: Path | str, generated: str = "") -> None:
        """Write the complete configuration, every key with its explanation
        (:func:`model_yaml_text`)."""
        Path(path).write_text(model_yaml_text(self, generated))

    def replace(self, **dotted_changes) -> "ModelConfig":
        """Return a copy with options changed, e.g.
        ``config.replace(**{"processes.stomatal_conductance.scheme": "medlyn"})``."""
        data = to_dict(self)
        for dotted, value in dotted_changes.items():
            *sections, name = dotted.split(".")
            node = data
            for part in sections:
                if not isinstance(node.get(part), dict):
                    raise ConfigError(f"replace: unknown option {dotted!r}")
                node = node[part]
            if not sections or name not in node:
                raise ConfigError(f"replace: unknown option {dotted!r}")
            node[name] = value
        return ModelConfig.from_dict(data, source="replace()")

    def to_fortran_codes(self, interpolation: str) -> dict:
        """Values for the legacy ``MLclm_varctl`` module variables.

        ``interpolation`` comes from ``run.yaml`` (``forcing.meteorology.interpolation``), because it
        describes the forcing data rather than the processes.
        """
        ps, n, c = self.processes, self.numerics, self.canopy_layers
        ph, gs, rad, turb = ps.photosynthesis, ps.stomatal_conductance, ps.radiation, ps.turbulence
        rk = INTEGRATOR_CODES[n.integrator]
        counts = c.layer_counts if isinstance(c.layer_counts, LayerCounts) else None
        kn = ph.nitrogen_profile
        return {
            "gs_type": _GS_TYPE[gs.scheme],
            "gspot_type": int(gs.water_stress),
            "gs_solver": {"brent": 1, "bisection": 2}[gs.wue_solver],
            "colim_type": int(ph.colimitation),
            "acclim_type": int(ph.temperature_acclimation != "none"),
            "kn_val": -999.0 if kn == "computed" else float(kn),
            "turb_type": 1,
            "sparse_canopy_type": int(turb.sparse_canopy),
            "HF_extension_type": {"extend_hf": 1, "log_profile": 2}[turb.ground_conductance],
            "flux_profile_type": {"well_mixed": 0, "implicit": 1}[
                ps.canopy_air.flux_profile_solution
            ],
            "gb_type": {"laminar": 1, "turbulent": 2, "free_convection": 3}[
                turb.leaf_boundary_layer
            ],
            "light_type": {"norman": 1, "two_stream": 2}[rad.solar],
            "leaf_optics_type": int(rad.leaf_optics_vary_with_height),
            "longwave_type": 1,
            "dtime_ml": float(n.substep_seconds),
            "runge_kutta_type": rk,
            "nrk": rk // 10,
            "dz_tall": float(c.dz_tall),
            "dz_short": float(c.dz_short),
            "dz_param": float(c.tall_canopy_threshold),
            "nlayer_above": counts.above if counts else 0,
            "nlayer_within": counts.within if counts else 0,
            "dpai_min": float(c.merge_thin_layers_below),
            "layer_pai_floor": float(c.minimum_leafy_layer_plant_area),
            "max_layers": int(c.max_layers),
            "met_type": {"none": 0, "three_point": 3}[interpolation],
        }

    def non_default_options(self) -> list[str]:
        """Dotted names of options that differ from the CLM-ml v2/CHATS defaults.

        The JAX translation has been validated against Fortran only with the defaults;
        other options are translated but not yet validated, so ``clm-ml check`` lists them.
        """
        return [
            k for k in changed_options(self)
            if k.split(".")[0] in ("processes", "numerics")
            and k not in ("numerics.checkpoint_steps", "numerics.checkpoint_substeps",
                          "numerics.table_search", "numerics.solver_loops",
                          "numerics.water_stress_derivative", "numerics.obukhov_derivative")
            and not (k == "processes.photosynthesis.temperature_acclimation"
                     and self.processes.photosynthesis.temperature_acclimation == "run_mean")
        ]  # fmt: skip


def changed_options(config: ModelConfig) -> list[str]:
    """Dotted names of the settings that differ from the defaults (nested sections
    flattened; a whole sub-section, e.g. explicit layer counts, counts as one setting)."""

    def flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict) and k != "layer_counts":
                out.update(flat(v, key + "."))
            else:
                out[key] = v
        return out

    default, mine = flat(to_dict(ModelConfig())), flat(to_dict(config))
    return [k for k, v in mine.items() if v != default.get(k)]


#: Header of model.yaml in an experiment folder
HEADER = """\
model.yaml: how each process is represented (parameterizations) and the numerics. Part of an experiment folder (run.yaml,
domain.yaml, model.yaml, params.yaml). Every setting is listed; the defaults are CLM-ml v2 as
run for CHATS (Bonan et al. 2026), the configuration validated against Fortran. Changing
anything here recompiles the model; none of these settings is differentiable (calibrated
numbers belong in params.yaml). {differs} {generated}"""


def model_yaml_text(config: ModelConfig, generated: str = "") -> str:
    """The complete model.yaml of ``config``: every section and key, each with its
    explanation, its choices and the Fortran variable it replaces, as comments
    (:func:`clm_ml.io.yaml_schema.render`). The header lists the settings that differ from
    the defaults."""
    changed = changed_options(config)
    differs = (
        "Differs from the defaults in: " + ", ".join(changed) + "."
        if changed
        else "All settings are at their defaults."
    )
    return render(config, HEADER.format(differs=differs, generated=generated).strip())


_GS_TYPE = {"medlyn": 0, "ball_berry": 1, "wue": 2}
INTEGRATOR_CODES = {
    "euler": 10,
    "rk2_trapezoid": 21,
    "rk2_midpoint": 22,
    "rk2_ralston": 23,
    "rk3_heun": 31,
    "rk3_ralston": 32,
    "rk3_kutta": 33,
    "rk4": 41,
}

#: Fortran namelist/varctl codes -> readable model.yaml settings (used by ``clm-ml convert``).
FORTRAN_CODES = {
    "gs_type": ("processes.stomatal_conductance.scheme", {v: k for k, v in _GS_TYPE.items()}),
    "light_type": ("processes.radiation.solar", {1: "norman", 2: "two_stream"}),
    "runge_kutta_type": ("numerics.integrator", {v: k for k, v in INTEGRATOR_CODES.items()}),
    "flux_profile_type": (
        "processes.canopy_air.flux_profile_solution",
        {0: "well_mixed", 1: "implicit"},
    ),
    "gb_type": (
        "processes.turbulence.leaf_boundary_layer",
        {1: "laminar", 2: "turbulent", 3: "free_convection"},
    ),
}
