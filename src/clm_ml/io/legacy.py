"""The legacy ASCII output: the six ``.out`` files of CLM-ml v2 (``flux``, ``aux``,
``profile``, ``fsun``, ``fluxprofile``, ``soiltemp``) of one patch, written step by step from
the step's fields in the legacy 1-based layout (run.yaml ``output.legacy_ascii``, for the
comparison with the Fortran output, ``tests/test_fortran_python_comparison.py``).

Fortran reference: CLM-ml v2 ``CLMml_driver.F90::output``.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import IO

import numpy as np


def _variables() -> tuple:
    from clm_ml.outputs import field_names

    return field_names()


#: Legacy name of each state and layer-grid field
_LEGACY_NAMES = {
    "state.air_temperature": "tair_profile", "state.vapor_pressure": "eair_profile",
    "state.co2": "cair_profile", "state.leaf_temperature": "tleaf_leaf",
    "state.leaf_water_potential": "lwp_leaf", "state.intercepted_water": "h2ocan_profile",
    "state.ground_temperature": "tg_soil", "layers.n_layers": "ncan_canopy",
    "layers.n_within": "ntop_canopy", "layers.dz": "dz_profile", "layers.zs": "zs_profile",
    "layers.zw": "zw_profile",
}  # fmt: skip
#: Legacy fields on the level axis (index 0 the ground: no padding in front)
_LEVEL = {"lwdwn_profile", "lwupw_profile", "zw_profile"}
_LEVEL_BAND = {"swbeam_profile", "swdwn_profile", "swupw_profile"}
_SPVAL = 1.0e36


def legacy_fields(fields: dict) -> tuple:
    """One step's 0-based fields (``clm_ml.outputs.step_fields`` names, one patch) in the
    legacy layout the writer reads: ``(mlcanopy, temperature)`` namespaces of arrays with a
    patch axis of length 1 and 1-based layer, leaf and band axes (``spval`` in front)."""
    ml = {}
    for key, value in fields.items():
        v = np.asarray(value)
        if key == "layers.first_leafy":
            ml["nbot_canopy"] = (v + 1)[None]
            continue
        if key.startswith("diagnostics."):
            name = key.split(".", 1)[1]
        elif key in _LEGACY_NAMES:
            name = _LEGACY_NAMES[key]
        else:
            continue
        if name in _LEVEL or v.ndim == 0:
            pads = []
        elif name in _LEVEL_BAND:
            pads = [(0, 0), (1, 0)]
        elif name.endswith("_leaf"):
            pads = [(1, 0), (1, 0)]
        else:  # (layer,) profiles and (band,) arrays
            pads = [(1, 0)]
        if pads:
            v = np.pad(v, pads + [(0, 0)] * (v.ndim - len(pads)), constant_values=_SPVAL)
        ml[name] = v[None]
    soil = np.pad(np.asarray(fields["state.soil_temperature"]), (1, 0), constant_values=_SPVAL)
    return SimpleNamespace(**ml), SimpleNamespace(t_soisno_col=soil[None])


TAGS = ("flux", "aux", "profile", "fsun", "fluxprofile", "soiltemp")


class LegacyAsciiWriter:
    """The six ``.out`` files of one patch: ``{directory}/{tower}_{yyyy}-{mm}_{tag}.out``
    (the run's start year and month). :meth:`write` appends one step."""

    def __init__(self, directory: Path, tower: str, year: int, month: int, *, met_type: int,
                 step_seconds: int, soil_z, append: bool = False):  # fmt: skip
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"  # a run continuing from a restart appends
        self.files = [
            open(directory / f"{tower}_{year:04d}-{month:02d}_{tag}.out", mode) for tag in TAGS
        ]
        self.met_type, self.step_seconds, self.soil_z = met_type, step_seconds, soil_z

    def write(self, calday: float, fields: dict) -> None:
        """One step: ``fields`` (``clm_ml.outputs.step_fields`` names, 0-based, one patch)."""
        ml, temp = legacy_fields(fields)
        flux, aux, profile, fsun, fluxprofile, soiltemp = self.files
        output(calday, flux, aux, profile, fsun, fluxprofile, soiltemp, ml, temp, self.soil_z,
               met_type=self.met_type, dtstep=self.step_seconds)  # fmt: skip

    def close(self) -> None:
        for f in self.files:
            f.close()


def output(
    curr_calday: float,
    nout1: IO,
    nout2: IO,
    nout3: IO,
    nout4: IO,
    nout5: IO,
    nout6: IO,
    mlcan,
    temperature_inst,
    soil_z,
    *,
    met_type: int,
    dtstep: int,
) -> None:
    """
    Write per-timestep model output to six ASCII files.

    Mirrors Fortran subroutine ``output`` (private to ``CLMml_driver``).

    Output files and their contents
    --------------------------------
    - ``nout1`` (``*_flux.out``): Canopy and soil energy fluxes,
      GPP, wind, albedo, and storage terms (18 columns).
    - ``nout2`` (``*_aux.out``): Leaf water potential and soil
      moisture stress (6 columns).
    - ``nout3`` (``*_profile.out``): Vertical profiles of leaf and
      air-layer state variables; above-canopy layers use
      ``missing_value`` for leaf quantities; within-canopy layers
      with ``dpai > 0`` include per-leaf-area fluxes (28 columns).
    - ``nout4`` (``*_fsun.out``): Sunlit/shaded canopy fluxes and
      bulk canopy properties (32 columns).
    - ``nout5`` (``*_fluxprofile.out``): Vertical flux profiles of
      sensible heat, latent heat (converted from mol H2O/m2/s),
      momentum, and shortwave/longwave radiation (13 columns).
    - ``nout6`` (``*_soiltemp.out``): Soil layer depths and
      temperatures for the first 10 layers (21 columns).

    Time stamp adjustment (Fortran lines 64-72):
    - ``met_type == 0``: time at end of time step — ``curr_calday``.
    - ``met_type == 3``: time centred in time step —
      ``curr_calday - 0.5 * dtstep / 86400``.
    - ``met_type == 2``: fatal error (not supported).

    All patch-level quantities are read at ``p = 1`` (single-patch
    tower site).

    Args:
        curr_calday: Current calendar day (1.000 = 0Z 1 January).
        nout1–nout6: Open Python file objects for the six output
            streams (replace Fortran unit numbers).
        mlcan: The step's ``mlcanopy`` fields of the legacy view (attributes, legacy layout).
        temperature_inst: Its ``temperature`` fields (``t_soisno_col``).
        soil_z: Soil node depths (0-based levels), m.
        dtstep: The model step (s).
    """
    from clm_ml.processes.constants import mmdry, mmh2o
    from clm_ml.processes.water_vapor import lat_vap

    ivis, inir = 1, 2  # legacy band indices
    isun, isha = 1, 2  # legacy leaf indices

    missing_value: float = -999.0
    zero_value: float = 0.0

    p: int = 0  # Single-patch tower site — 0-based Python indexing (Fortran used 1)

    # ------------------------------------------------------------------
    # Time stamp — Fortran lines 66-74
    # ------------------------------------------------------------------
    if met_type == 0:
        # Time at end of timestep — Fortran line 68
        time_stamp = curr_calday
    elif met_type == 3:
        # Time centred in timestep — Fortran line 71
        time_stamp = curr_calday - 0.5 * dtstep / 86400.0
    elif met_type == 2:
        # Time at end of timestep (not supported) — Fortran lines 73-74
        raise ValueError("met_type 2 is not supported")
    else:
        time_stamp = curr_calday

    # ------------------------------------------------------------------
    # nout1: flux.out — canopy and soil fluxes — Fortran lines 77-95
    # write(nout1,'(f12.7,17f10.3)') time_stamp, 17 variables
    # ------------------------------------------------------------------
    swup = float(mlcan.albcan_canopy[p, ivis]) * (
        float(mlcan.swskyb_forcing[p, ivis]) + float(mlcan.swskyd_forcing[p, ivis])
    ) + float(mlcan.albcan_canopy[p, inir]) * (
        float(mlcan.swskyb_forcing[p, inir]) + float(mlcan.swskyd_forcing[p, inir])
    )

    lhflx_tr = float(mlcan.trveg_canopy[p]) * lat_vap(float(mlcan.tref_forcing[p]))
    lhflx_ev = float(mlcan.evveg_canopy[p]) * lat_vap(float(mlcan.tref_forcing[p]))

    ic_top = int(mlcan.ntop_canopy[p])
    tair = float(mlcan.tair_profile[p, ic_top])

    nout1.write(
        f"{time_stamp:12.7f}"
        + _fmt10(mlcan.rnet_canopy[p])
        + _fmt10(mlcan.stflx_air_canopy[p])
        + _fmt10(mlcan.shflx_canopy[p])
        + _fmt10(mlcan.lhflx_canopy[p])
        + _fmt10(mlcan.gppveg_canopy[p])
        + _fmt10(mlcan.ustar_canopy[p])
        + _fmt10(swup)
        + _fmt10(mlcan.lwup_canopy[p])
        + _fmt10(tair)
        + _fmt10(mlcan.gsoi_soil[p])
        + _fmt10(mlcan.rnsoi_soil[p])
        + _fmt10(mlcan.shsoi_soil[p])
        + _fmt10(mlcan.lhsoi_soil[p])
        + _fmt10(lhflx_tr)
        + _fmt10(lhflx_ev)
        + _fmt10(mlcan.beta_canopy[p])
        + _fmt10(mlcan.stflx_veg_canopy[p])
        + "\n"
    )

    # ------------------------------------------------------------------
    # nout4: fsun.out — sunlit/shaded fluxes — Fortran lines 98-113
    # write(nout4,'(32f10.3)') 32 variables
    # ------------------------------------------------------------------
    nout4.write(
        _fmt10(float(mlcan.solar_zen_forcing[p]) * 180.0 / math.pi)
        + _fmt10(float(mlcan.swskyb_forcing[p, ivis]) + float(mlcan.swskyd_forcing[p, ivis]))
        + _fmt10(float(mlcan.lai_canopy[p]) + float(mlcan.sai_canopy[p]))
        + _fmt10(mlcan.laisun_canopy[p])
        + _fmt10(mlcan.laisha_canopy[p])
        + _fmt10(mlcan.swveg_canopy[p, ivis])
        + _fmt10(mlcan.swvegsun_canopy[p, ivis])
        + _fmt10(mlcan.swvegsha_canopy[p, ivis])
        + _fmt10(mlcan.gppveg_canopy[p])
        + _fmt10(mlcan.gppvegsun_canopy[p])
        + _fmt10(mlcan.gppvegsha_canopy[p])
        + _fmt10(mlcan.lhveg_canopy[p])
        + _fmt10(mlcan.lhvegsun_canopy[p])
        + _fmt10(mlcan.lhvegsha_canopy[p])
        + _fmt10(mlcan.shveg_canopy[p])
        + _fmt10(mlcan.shvegsun_canopy[p])
        + _fmt10(mlcan.shvegsha_canopy[p])
        + _fmt10(mlcan.vcmax25veg_canopy[p])
        + _fmt10(mlcan.vcmax25sun_canopy[p])
        + _fmt10(mlcan.vcmax25sha_canopy[p])
        + _fmt10(mlcan.gsveg_canopy[p])
        + _fmt10(mlcan.gsvegsun_canopy[p])
        + _fmt10(mlcan.gsvegsha_canopy[p])
        + _fmt10(mlcan.windveg_canopy[p])
        + _fmt10(mlcan.windvegsun_canopy[p])
        + _fmt10(mlcan.windvegsha_canopy[p])
        + _fmt10(mlcan.tlveg_canopy[p])
        + _fmt10(mlcan.tlvegsun_canopy[p])
        + _fmt10(mlcan.tlvegsha_canopy[p])
        + _fmt10(mlcan.taveg_canopy[p])
        + _fmt10(mlcan.tavegsun_canopy[p])
        + _fmt10(mlcan.tavegsha_canopy[p])
        + "\n"
    )

    # ------------------------------------------------------------------
    # nout2: aux.out — leaf water potential / soil stress — Fortran lines 116-120
    # write(nout2,'(f10.4,5f10.3)') 6 variables
    # ------------------------------------------------------------------
    top = int(mlcan.ntop_canopy[p])
    mid = max(
        1,
        int(mlcan.nbot_canopy[p])
        + (int(mlcan.ntop_canopy[p]) - int(mlcan.nbot_canopy[p]) + 1) // 2
        - 1,
    )

    nout2.write(
        f"{float(mlcan.btran_soil[p]):10.4f}"
        + _fmt10(mlcan.lsc_profile[p, top])
        + _fmt10(mlcan.psis_soil[p])
        + _fmt10(mlcan.lwp_mean_profile[p, top])
        + _fmt10(mlcan.lwp_mean_profile[p, mid])
        + _fmt10(mlcan.fracminlwp_canopy[p])
        + "\n"
    )

    # ------------------------------------------------------------------
    # nout3: profile.out — vertical profiles — Fortran lines 122-176
    # write(nout3,'(f12.7,27f10.3)') time_stamp + 27 variables
    # ------------------------------------------------------------------

    def _qair(ic_: int) -> float:
        """Specific humidity (g/kg) from vapour pressure profile."""
        e = float(mlcan.eair_profile[p, ic_])
        pr = float(mlcan.pref_forcing[p])
        return 1000.0 * (mmh2o / mmdry) * e / (pr - (1.0 - mmh2o / mmdry) * e)

    def _eair_kpa(ic_: int) -> float:
        """Vapour pressure (kPa)."""
        return float(mlcan.eair_profile[p, ic_]) / 1000.0

    def _ra(ic_: int) -> float:
        """Aerodynamic resistance (s/m)."""
        return float(mlcan.rhomol_forcing[p]) / float(mlcan.gac_profile[p, ic_])

    def _lad(ic_: int) -> float:
        """Plant area density (m2/m3)."""
        return float(mlcan.dpai_profile[p, ic_]) / float(mlcan.dz_profile[p, ic_])

    def _profile_line(ic_: int, is_above: bool) -> str:
        """Build one nout3 record (28 columns)."""
        tair_ = float(mlcan.tair_profile[p, ic_])
        qair_ = _qair(ic_)
        ra_ = _ra(ic_)
        mv = missing_value
        zv = zero_value

        if is_above:
            # Above-canopy: all leaf quantities are missing — Fortran lines 131-149
            return (
                f"{time_stamp:12.7f}"
                + _fmt10(mlcan.zs_profile[p, ic_])
                + _fmt10(zv)
                + _fmt10(zv)
                + _fmt10(zv)
                + _fmt10(zv)
                + (_fmt10(mv) * 18)
                + _fmt10(float(mlcan.wind_profile[p, ic_]))
                + _fmt10(tair_)
                + _fmt10(qair_)
                + _fmt10(ra_)
                + "\n"
            )
        else:
            dpai_ = float(mlcan.dpai_profile[p, ic_])
            lad_ = _lad(ic_)
            frac_ = float(mlcan.fracsun_profile[p, ic_])
            if dpai_ > 0.0:
                # Leaf layer — Fortran lines 155-170
                return (
                    f"{time_stamp:12.7f}"
                    + _fmt10(mlcan.zs_profile[p, ic_])
                    + _fmt10(frac_)
                    + _fmt10(lad_)
                    + _fmt10(lad_ * frac_)
                    + _fmt10(lad_ * (1.0 - frac_))
                    + _fmt10(mlcan.rnleaf_leaf[p, ic_, isun])
                    + _fmt10(mlcan.rnleaf_leaf[p, ic_, isha])
                    + _fmt10(mlcan.shleaf_leaf[p, ic_, isun])
                    + _fmt10(mlcan.shleaf_leaf[p, ic_, isha])
                    + _fmt10(mlcan.lhleaf_leaf[p, ic_, isun])
                    + _fmt10(mlcan.lhleaf_leaf[p, ic_, isha])
                    + _fmt10(mlcan.anet_leaf[p, ic_, isun])
                    + _fmt10(mlcan.anet_leaf[p, ic_, isha])
                    + _fmt10(mlcan.apar_leaf[p, ic_, isun])
                    + _fmt10(mlcan.apar_leaf[p, ic_, isha])
                    + _fmt10(mlcan.gs_leaf[p, ic_, isun])
                    + _fmt10(mlcan.gs_leaf[p, ic_, isha])
                    + _fmt10(mlcan.lwp_hist_leaf[p, ic_, isun])
                    + _fmt10(mlcan.lwp_hist_leaf[p, ic_, isha])
                    + _fmt10(mlcan.tleaf_hist_leaf[p, ic_, isun])
                    + _fmt10(mlcan.tleaf_hist_leaf[p, ic_, isha])
                    + _fmt10(mlcan.vcmax25_leaf[p, ic_, isun])
                    + _fmt10(mlcan.vcmax25_leaf[p, ic_, isha])
                    + _fmt10(float(mlcan.wind_profile[p, ic_]))
                    + _fmt10(tair_)
                    + _fmt10(qair_)
                    + _fmt10(ra_)
                    + "\n"
                )
            else:
                # Non-leaf within-canopy layer — Fortran lines 172-185
                return (
                    f"{time_stamp:12.7f}"
                    + _fmt10(mlcan.zs_profile[p, ic_])
                    + _fmt10(frac_)
                    + _fmt10(zv)
                    + _fmt10(zv)
                    + _fmt10(zv)
                    + (_fmt10(mv) * 18)
                    + _fmt10(float(mlcan.wind_profile[p, ic_]))
                    + _fmt10(tair_)
                    + _fmt10(qair_)
                    + _fmt10(ra_)
                    + "\n"
                )

    # Above-canopy layers — Fortran: do ic = ncan, ntop+1, -1
    # Python range stop is exclusive, so use ntop (not ntop+1) to include ic=ntop+1
    for ic in range(int(mlcan.ncan_canopy[p]), int(mlcan.ntop_canopy[p]), -1):
        nout3.write(_profile_line(ic, is_above=True))

    # Within-canopy layers — Fortran: do ic = ntop, 1, -1
    for ic in range(int(mlcan.ntop_canopy[p]), 0, -1):
        nout3.write(_profile_line(ic, is_above=False))

    # ------------------------------------------------------------------
    # nout5: fluxprofile.out — vertical flux profiles — Fortran lines 188-202
    # write(nout5,'(f12.7,26f10.3)') time_stamp + 12 variables (13 cols total)
    # ------------------------------------------------------------------
    _ntop_fp = int(mlcan.ntop_canopy[p])
    for ic in range(int(mlcan.ncan_canopy[p]), 0, -1):  # Fortran: do ic = ncan, 1, -1
        shf = float(mlcan.shair_profile[p, ic])
        lhf = float(mlcan.etair_profile[p, ic]) * lat_vap(float(mlcan.tref_forcing[p]))
        mflx = float(mlcan.mflx_profile[p, ic])
        # LW profiles are only computed for within-canopy layers (0..ntop).
        # Above-canopy layers retain spval; write 0.0 to match Fortran behavior.
        lwdwn_ic = float(mlcan.lwdwn_profile[p, ic]) if ic <= _ntop_fp else 0.0
        lwupw_ic = float(mlcan.lwupw_profile[p, ic]) if ic <= _ntop_fp else 0.0
        nout5.write(
            f"{time_stamp:12.7f}"
            + _fmt10(mlcan.zw_profile[p, ic])
            + _fmt10(shf)
            + _fmt10(lhf)
            + _fmt10(mflx)
            + _fmt10(mlcan.swbeam_profile[p, ic, ivis])
            + _fmt10(mlcan.swbeam_profile[p, ic, inir])
            + _fmt10(mlcan.swdwn_profile[p, ic, ivis])
            + _fmt10(mlcan.swdwn_profile[p, ic, inir])
            + _fmt10(mlcan.swupw_profile[p, ic, ivis])
            + _fmt10(mlcan.swupw_profile[p, ic, inir])
            + _fmt10(lwdwn_ic)
            + _fmt10(lwupw_ic)
            + "\n"
        )

    # ------------------------------------------------------------------
    # nout6: soiltemp.out — soil temperature — Fortran lines 204-207
    # write(nout6,'(f12.7,20f10.3)') time_stamp, (z(1,ic), t_soisno(1,ic), ic=1,10)
    # ------------------------------------------------------------------
    soiltemp_cols = ""
    for ic in range(1, 11):  # Fortran: ic=1,10
        soiltemp_cols += _fmt10(soil_z[ic - 1])  # 0-based soil level
        soiltemp_cols += _fmt10(temperature_inst.t_soisno_col[0, ic])  # patch 0, 1-based level
    nout6.write(f"{time_stamp:12.7f}" + soiltemp_cols + "\n")


# ---------------------------------------------------------------------------
# Private formatting helper
# ---------------------------------------------------------------------------


def _fmt10(val) -> str:
    """
    Format a scalar as a 10-character field with 3 decimal places,
    matching Fortran ``f10.3``.
    """
    return f"{float(val):10.3f}"
