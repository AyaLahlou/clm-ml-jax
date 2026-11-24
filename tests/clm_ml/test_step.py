"""clm_ml.step: the time and sun of one step.

The step itself is checked end to end (bit-for-bit regression test). Here: the calendar days
of the forcing records and sub-steps, and the solar zenith angle, against the Fortran
driver's formulas.
"""

import math

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from clm_ml import step as S  # noqa: E402
from clm_ml.processes.canopy import SubstepOptions  # noqa: E402


def _options(interpolation="three_point", step_seconds=1800, dtime=300.0):
    sub = SubstepOptions(
        interpolation=interpolation, acclimation=False, kn_val=-999.0, sparse_canopy=False,
        ground_conductance="log_profile", regime="laminar", stomatal_conductance="wue",
        water_stress=True, colimitation=True, wue_solver="bisection", dtime=dtime,
        integrator="rk4",
    )  # fmt: skip
    return S.StepOptions(
        substep=sub, step_seconds=step_seconds, layer_pai_floor=0.005, flux_profile_type=1,
        n_soil_adjust=0, n_soil=20, eccen=0.0167, mvelpp=4.94,
        lambm0=-0.0327, obliqr=0.409,
    )  # fmt: skip


def test_interpolation_days_centred():
    beg = 121.5 - 1800 / 86400
    bef, cur, nxt = S.interpolation_days(121.5, beg, _options())
    assert cur == 0.5 * (121.5 + beg)
    assert bef == cur - 1800 / 86400 and nxt == cur + 1800 / 86400
    assert S.interpolation_days(121.5, 121.4, _options("none")) == (0.0, 0.0, 0.0)


def test_substep_days():
    opts = _options()
    assert opts.n_substeps == 6
    days = np.asarray(S.substep_days(121.0, opts))
    expected = [121.0 + (k - 0.5) * (300.0 / 86400.0) for k in range(1, 7)]
    np.testing.assert_array_equal(days, expected)
    days = np.asarray(S.substep_days(121.0, _options("none")))
    np.testing.assert_array_equal(days, [121.0 + k * (300.0 / 86400.0) for k in range(1, 7)])


def test_solar_zenith_matches_formula():
    opts = _options()
    calday, lat, lon = 140.75, 38.49, -121.84
    pi = 3.141592654
    lambm = opts.lambm0 + (calday - 80.5) * 2.0 * pi / 365.0
    lmm = lambm - opts.mvelpp
    e, sinl = opts.eccen, math.sin(lmm)
    lamb = lambm + e * (2.0 * sinl + e * (1.25 * math.sin(2.0 * lmm)
                                          + e * ((13.0 / 12.0) * math.sin(3.0 * lmm) - 0.25 * sinl)))  # fmt: skip
    declin = math.asin(math.sin(opts.obliqr) * math.sin(lamb))
    la, lo = lat * pi / 180.0, lon * pi / 180.0
    cosz = math.sin(la) * math.sin(declin) - math.cos(la) * math.cos(declin) * math.cos(
        calday * 2.0 * pi + lo
    )
    expected = math.acos(max(0.01, cosz))
    assert abs(float(S.solar_zenith(calday, lat, lon, opts)) - expected) < 1e-12
    # at night the cosine is floored at 0.01
    night = float(S.solar_zenith(140.25, lat, lon, opts))
    assert night <= math.acos(0.01) + 1e-12


def test_soil_level_ice_is_the_levels_own():
    """Root uptake stops in a frozen level, read from that level (Fortran
    SoilResistance h2osoi_ice(c,j)), not from the level below it."""
    ice = np.zeros(25)
    ice[3] = 1.0  # level 4 (0-based index 3) frozen
    out = np.asarray(S.soil_level_ice(ice, 20))
    assert out.shape == (20,) and out[3] == 1.0 and out.sum() == 1.0
