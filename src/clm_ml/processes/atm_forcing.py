"""Atmospheric forcing at the reference height for one model sub-step: time interpolation of
the CLM-step forcing and the derived air properties.

One patch; shortwave bands are 0-based (0 visible, 1 near infrared). The forcing of the
previous, current and next CLM steps (``bef``, ``cur``, ``nxt``) is interpolated to the
sub-step's calendar day with the variant chosen by name (``TIME_INTERPOLATION``, model.yaml
``forcing interpolation``: ``none`` = Fortran ``met_type = 0``, ``three_point`` =
``met_type = 3``). Fortran's ``met_type = 2`` (two-point) stops the model in CLM-ml v2 and is
not provided.

Fortran reference: CLM-ml v2 ``MLGetAtmForcingMod.F90::GetAtmForcing``,
``TimeInterpolation3``; ``MLCanopyFluxesMod.F90::GetCLMVar`` (``clm_forcing_to_ml``).

Reads and writes are the arguments and results; ``Reads``/``Writes`` in the docstrings name
the CLM-ml v2 (``mlcanopy_type``) fields they correspond to.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from clm_ml.processes.constants import cpd, cpw, lapse_rate, mmdry, mmh2o, rgas, wind_forc_min


class ForcingRecord(NamedTuple):
    """Atmospheric forcing of one patch at one CLM step (or interpolated to a sub-step)."""

    uref: Array  # wind speed (m/s)
    tref: Array  # air temperature (K)
    qref: Array  # specific humidity (kg/kg)
    pref: Array  # air pressure (Pa)
    co2ref: Array  # atmospheric CO2 (umol/mol)
    swskyb: Array  # (band,) direct beam solar radiation (W/m2)
    swskyd: Array  # (band,) diffuse solar radiation (W/m2)
    lwsky: Array  # atmospheric longwave radiation (W/m2)


class AtmForcing(NamedTuple):
    """Forcing at the reference height for one sub-step (CLM-ml v2 ``<name>_forcing`` fields)."""

    uref: Array  # wind speed, at least ``wind_forc_min`` (m/s)
    tref: Array  # air temperature (K)
    qref: Array  # specific humidity (kg/kg)
    pref: Array  # air pressure (Pa)
    co2ref: Array  # atmospheric CO2 (umol/mol)
    swskyb: Array  # (band,) direct beam solar radiation (W/m2)
    swskyd: Array  # (band,) diffuse solar radiation (W/m2)
    lwsky: Array  # atmospheric longwave radiation (W/m2)
    eref: Array  # vapor pressure (Pa)
    rhomol: Array  # molar density (mol/m3)
    rhoair: Array  # air density (kg/m3)
    mmair: Array  # molecular mass of air (kg/mol)
    cpair: Array  # specific heat of air at constant pressure (J/mol/K)
    thref: Array  # potential temperature (K)
    thvref: Array  # virtual potential temperature (K)


def clm_forcing_to_ml(u, v, t, q, pbot, pco2, solad, solai, lwrad):
    """CLM forcing of one patch as the multilayer canopy's forcing record values.

    Wind speed is the resultant of the east and north components; CO2 goes from partial
    pressure (Pa) to mole fraction (umol/mol). ``solad``, ``solai`` are passed through.
    Shared by the current-step record and the previous/next-step records, so all three use
    identical arithmetic.

    Fortran reference: MLCanopyFluxesMod.F90::GetCLMVar.
    Returns ``(uref, tref, qref, pref, co2ref, swskyb, swskyd, lwsky)``.
    """
    uref = jnp.sqrt(u**2 + v**2)
    co2ref = pco2 / pbot * 1.0e6  # Pa -> umol/mol
    return uref, t, q, pbot, co2ref, solad, solai, lwrad


def time_interpolation3(x0, x1, x2, t0, t1, t2, tx):
    """Piecewise linear interpolation through ``(t0, x0)``, ``(t1, x1)``, ``(t2, x2)`` at
    ``tx``: the lower segment when ``tx < t1``, else the upper one (``tx`` may be traced).

    Fortran reference: MLGetAtmForcingMod.F90::TimeInterpolation3.
    """
    b1_lo = (x1 - x0) / (t1 - t0)  # slope, lower segment
    b0_lo = x0 - b1_lo * t0  # intercept, lower segment
    b1_hi = (x2 - x1) / (t2 - t1)  # slope, upper segment
    b0_hi = x1 - b1_hi * t1  # intercept, upper segment
    b1 = jnp.where(tx < t1, b1_lo, b1_hi)
    b0 = jnp.where(tx < t1, b0_lo, b0_hi)
    return b0 + b1 * tx


def _no_interpolation(bef, cur, nxt, t_bef, t_cur, t_next, t_ml):
    """The current CLM step's values (Fortran ``met_type = 0``)."""
    return cur


def _three_point(bef, cur, nxt, t_bef, t_cur, t_next, t_ml):
    """Every field by ``time_interpolation3`` (Fortran ``met_type = 3``)."""
    return jax.tree_util.tree_map(
        lambda x0, x1, x2: time_interpolation3(x0, x1, x2, t_bef, t_cur, t_next, t_ml),
        bef,
        cur,
        nxt,
    )


#: Time interpolation variants by name (model.yaml ``interpolation``).
TIME_INTERPOLATION = {"none": _no_interpolation, "three_point": _three_point}


def atm_forcing(
    bef: ForcingRecord,
    cur: ForcingRecord,
    nxt: ForcingRecord,
    zref,
    t_bef,
    t_cur,
    t_next,
    t_ml,
    *,
    interpolation: str,
) -> AtmForcing:
    """Forcing at the reference height for the sub-step at calendar day ``t_ml``.

    The CLM-step records ``bef``, ``cur``, ``nxt`` (calendar days ``t_bef``, ``t_cur``,
    ``t_next``) are interpolated (``TIME_INTERPOLATION[interpolation]``), the wind speed is
    held at ``wind_forc_min`` or more, and the air properties are derived at the reference
    height ``zref`` (m).

    Fortran reference: MLGetAtmForcingMod.F90::GetAtmForcing.
    Reads: the forcing window (``*_bef/_cur/_next_forcing``), ``zref_forcing``.
    Writes: ``uref, tref, qref, pref, co2ref, swskyb, swskyd, lwsky, eref, rhomol, rhoair,
    mmair, cpair, thref, thvref`` (``_forcing``).
    """
    f = TIME_INTERPOLATION[interpolation](bef, cur, nxt, t_bef, t_cur, t_next, t_ml)
    eps = mmh2o / mmdry  # ratio of molecular masses
    one_minus_eps = 1.0 - eps
    uref = jnp.maximum(wind_forc_min, f.uref)
    t, q, p = f.tref, f.qref, f.pref
    eref = q * p / (eps + one_minus_eps * q)
    rhomol = p / (rgas * t)
    rhoair = rhomol * mmdry * (1.0 - one_minus_eps * eref / p)
    mmair = rhoair / rhomol
    cpair = cpd * (1.0 + (cpw / cpd - 1.0) * q) * mmair
    thref = t + lapse_rate * zref
    thvref = thref * (1.0 + 0.61 * q)
    return AtmForcing(
        uref, t, q, p, f.co2ref, f.swskyb, f.swskyd, f.lwsky,
        eref, rhomol, rhoair, mmair, cpair, thref, thvref,
    )  # fmt: skip
