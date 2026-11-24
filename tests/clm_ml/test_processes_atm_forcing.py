"""clm_ml.processes.atm_forcing, checked against the interpolation it
describes."""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.processes import atm_forcing  # noqa: E402
from clm_ml.processes.constants import wind_forc_min  # noqa: E402


def _record(x):
    return atm_forcing.ForcingRecord(
        x, x, x, x, x, jnp.array([x, 2 * x]), jnp.array([x, 3 * x]), x
    )  # fmt: skip


@pytest.mark.parametrize("t_ml, expected", [(1.0, 10.0), (2.0, 20.0), (3.0, 40.0), (1.5, 15.0),
                                            (2.5, 30.0)])  # fmt: skip
def test_three_point_interpolation_is_piecewise_linear(t_ml, expected):
    bef, cur, nxt = _record(10.0), _record(20.0), _record(40.0)
    f = atm_forcing.TIME_INTERPOLATION["three_point"](bef, cur, nxt, 1.0, 2.0, 3.0, t_ml)
    np.testing.assert_allclose(f.tref, expected, rtol=1e-14)
    np.testing.assert_allclose(f.swskyb, [expected, 2 * expected], rtol=1e-14)
    np.testing.assert_allclose(f.swskyd, [expected, 3 * expected], rtol=1e-14)


def test_derived_air_properties():
    rec = atm_forcing.ForcingRecord(
        jnp.asarray(0.1), jnp.asarray(293.15), jnp.asarray(0.01), jnp.asarray(1.0e5),
        jnp.asarray(383.0), jnp.zeros(2), jnp.zeros(2), jnp.asarray(350.0),
    )  # fmt: skip
    f = atm_forcing.atm_forcing(rec, rec, rec, 23.0, 0.0, 1.0, 2.0, 1.0, interpolation="none")
    assert float(f.uref) == wind_forc_min  # wind below the minimum is raised to it
    # ideal gas: rho = p M / (R T) with the moist-air molecular mass
    np.testing.assert_allclose(f.rhoair, 1.0e5 * f.mmair / (8.31446 * 293.15), rtol=1e-12)
    assert 0.0285 < float(f.mmair) < 0.02897  # moist air is lighter than dry air
    np.testing.assert_allclose(f.thref, 293.15 + 0.0098 * 23.0, rtol=1e-14)
