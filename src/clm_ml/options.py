"""The multilayer canopy's options (Phase 7.4; from the legacy ``MLclm_varctl`` and
``clm_ml.legacy.canopy``): :class:`MLOptions` (CLM-ml v2's option codes, built from model.yaml)
and :func:`substep_options` (the codes as the sub-step's named, static choices).

Fortran reference: CLM-ml v2 ``MLclm_varctl.F90``.
"""

from __future__ import annotations

from typing import NamedTuple

from clm_ml.config import INTEGRATOR_CODES
from clm_ml.processes import canopy


class MLOptions(NamedTuple):
    """The multilayer canopy's run-control options as CLM-ml v2's ``MLclm_varctl`` integer
    codes (model.yaml ``ModelConfig.to_fortran_codes``): a hashable tuple of Python numbers.
    Defaults are the Fortran defaults (CLM-ml v2, the CHATS7 configuration except
    ``met_type``)."""

    gs_type: int = 2
    gspot_type: int = 1
    gs_solver: int = 2
    colim_type: int = 1
    acclim_type: int = 1
    kn_val: float = -999.0
    turb_type: int = 1
    sparse_canopy_type: int = 1
    HF_extension_type: int = 2
    flux_profile_type: int = 1
    gb_type: int = 3
    light_type: int = 2
    leaf_optics_type: int = 0
    longwave_type: int = 1
    dtime_ml: float = 300.0
    runge_kutta_type: int = 41
    dz_tall: float = 0.5
    dz_short: float = 0.1
    dz_param: float = 2.0
    nlayer_above: int = 0
    nlayer_within: int = 0
    mlcan_to_clm: int = 0
    met_type: int = 0
    dpai_min: float = 0.01
    # Minimum plant area of every leafy layer at every time step (m2/m2), added as stem area
    # (not Fortran CLM-ml v2; Phase 6b, issue 14): the 9-angle diffuse transmittance exceeds 1
    # below ~0.005, which makes the longwave equations unstable. 0: CLM-ml v2 as in Fortran.
    # Value: CLM-ml v1's dpai_min ("minimum plant area index to be considered a vegetation
    # layer", m2/m2), applied per step instead of removing layers at setup.
    layer_pai_floor: float = 0.01
    # Layer slots of every canopy profile array (model.yaml canopy_layers.max_layers; Fortran:
    # the parameter nlevmlcan = 100). Patches use 1..ncan; the rest is padding.
    max_layers: int = 50

    #: Fields used only to set up the canopy layers (``initVerticalStructure``, and
    #: ``max_layers`` for the array shapes), never by the time step.
    SETUP_ONLY = (
        "dz_tall",
        "dz_short",
        "dz_param",
        "nlayer_above",
        "nlayer_within",
        "dpai_min",
        "max_layers",
    )

    def step_options(self) -> "MLOptions":
        """These options with the setup-only fields (:attr:`SETUP_ONLY`) at their defaults:
        what the time step receives (Phase 6 step 2). Compiled functions take the options as a
        static argument, so patches and models that differ only in their layer settings share
        one compiled program."""
        return self._replace(**{f: self._field_defaults[f] for f in self.SETUP_ONLY})

    @property
    def nrk(self) -> int:
        """Number of Runge-Kutta stages — Fortran: runge_kutta_type/10."""
        return self.runge_kutta_type // 10

    @classmethod
    def from_codes(cls, codes: dict) -> "MLOptions":
        """From ``ModelConfig.to_fortran_codes`` values (other keys, e.g. ``nrk``, ignored)."""
        return cls(**{k: type(getattr(cls(), k))(v) for k, v in codes.items() if k in cls._fields})


DEFAULT_OPTIONS = MLOptions()

#: Fortran code -> integrator name (model.yaml numerics.integrator)
INTEGRATOR = {code: name for name, code in INTEGRATOR_CODES.items()}
#: Fortran codes -> model.yaml names
_MET_TYPE = {0: "none", 3: "three_point"}
_GROUND = {1: "extend_hf", 2: "log_profile"}
_REGIME = {1: "laminar", 2: "turbulent", 3: "free_convection"}
_STOMATAL = {0: "medlyn", 1: "ball_berry", 2: "wue"}
_WUE_SOLVER = {1: "brent", 2: "bisection"}


def substep_options(
    opts,
    table_search: str = "loop",
    unroll_loops: bool = False,
    wetted_fraction: str = "implicit",
    water_stress_derivative: str = "stabilized",
    obukhov_derivative: str = "exact",
) -> canopy.SubstepOptions:
    """The sub-step's choices from the legacy options (``MLOptions``), ``table_search``
    (how the psihat tables are searched), ``unroll_loops`` (the solver loops written out),
    ``wetted_fraction`` (implicit, or explicit as Fortran) and the derivative rules
    ``water_stress_derivative`` and ``obukhov_derivative`` (decision 124; none is a Fortran
    option), with the legacy
    routines' checks: Norman solar radiation and the well-mixed flux-profile solution are
    registered but not supported (Phase 7 decision D8: ``clm_ml.run`` refuses them), the other codes must be
    valid."""
    if opts.light_type == 1:
        raise NotImplementedError("Norman solar radiation (light_type 1) is not supported")
    if opts.flux_profile_type in (0, -1):
        raise NotImplementedError("the well-mixed flux-profile solution is not supported")
    checks = (("met_type", opts.met_type, _MET_TYPE), ("light_type", opts.light_type, (2,)),
              ("leaf_optics_type", opts.leaf_optics_type, (0,)),
              ("acclim_type", opts.acclim_type, (0, 1)), ("longwave_type", opts.longwave_type, (1,)),
              ("turb_type", opts.turb_type, (1,)),
              ("HF_extension_type", opts.HF_extension_type, _GROUND),
              ("gb_type", opts.gb_type, _REGIME), ("gs_type", opts.gs_type, _STOMATAL),
              ("gs_solver", opts.gs_solver, _WUE_SOLVER), ("gspot_type", opts.gspot_type, (0, 1)),
              ("colim_type", opts.colim_type, (0, 1)),
              ("flux_profile_type", opts.flux_profile_type, (1,)),
              ("runge_kutta_type", opts.runge_kutta_type, INTEGRATOR))  # fmt: skip
    for name, code, valid in checks:
        if code not in valid:
            raise ValueError(f"{name} {code} not one of {tuple(valid)}")
    return canopy.SubstepOptions(
        interpolation=_MET_TYPE[opts.met_type],
        acclimation=bool(opts.acclim_type == 1),
        kn_val=float(opts.kn_val),
        sparse_canopy=bool(opts.sparse_canopy_type == 1),
        ground_conductance=_GROUND[opts.HF_extension_type],
        regime=_REGIME[opts.gb_type],
        stomatal_conductance=_STOMATAL[opts.gs_type],
        water_stress=bool(opts.gspot_type == 1),
        colimitation=bool(opts.colim_type == 1),
        wue_solver=_WUE_SOLVER[opts.gs_solver],
        dtime=float(opts.dtime_ml),
        integrator=INTEGRATOR[opts.runge_kutta_type],
        table_search=table_search,
        unroll_loops=unroll_loops,
        wetted_fraction=wetted_fraction,
        water_stress_derivative=water_stress_derivative,
        obukhov_derivative=obukhov_derivative,
    )
