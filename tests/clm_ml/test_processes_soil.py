"""clm_ml.processes.soil."""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.processes import soil  # noqa: E402


def test_soil_temperature_keeps_a_uniform_profile_without_heat_flux():
    n = 6
    z = np.cumsum(np.full(n, 0.2))
    t, err = soil.soil_temperature(
        jnp.full(n, 283.0), jnp.full(n, 1.5).at[-1].set(0.0), jnp.full(n, 2.0e6), z, 0.0, 1800.0
    )
    np.testing.assert_allclose(np.asarray(t), 283.0, rtol=0, atol=1e-10)
    assert abs(float(err)) < 1e-9
