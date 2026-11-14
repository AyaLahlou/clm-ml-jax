"""Accuracy of the model's derivatives where the solvers' implicit-function gradients are
hardest (decision 119; ``docs/design/gradient_progress.md`` round 2): the CHATS7 morning
transition, with Obukhov lengths of 10-15 m at the limit of the roughness-sublayer
functions (Lc/L = 1) and leaves at the minimum stomatal conductance. Forward-mode
derivatives of the fluxes of four steps against central finite differences.

Gradient tier (``tests/gradient_tier.py``): compiles two chunks and a forward derivative.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pytestmark = [pytest.mark.gradient, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[1]
NAMELIST = ROOT / "data" / "namelists" / "nl.CHATS7.1day"
FIRST, STEPS = 25, 4  # run steps 25-28: 05:30-07:30 local time, sunrise
VARIABLES = ("latent_heat_flux", "sensible_heat_flux", "gpp", "friction_velocity")
PARAMETERS = ("vcmax25", "gs_min", "leaf_emissivity", "iota")


@pytest.fixture(scope="module")
def derivatives():
    """``(ad, fd)``, each (parameter, variable, step): the derivative of the step values
    with respect to each of :data:`PARAMETERS` by ``jax.jvp`` and by central differences
    (0.01 % of the parameter)."""
    from clm_ml.experiment import load_experiment
    from clm_ml.io.namelist import convert_namelist
    from clm_ml.model import Model
    from clm_ml.output_streams import StreamSpec

    tmp = Path(tempfile.mkdtemp())
    convert_namelist(NAMELIST, tmp)
    model = Model.from_experiment(load_experiment(tmp))
    params = model.default_params()
    state, _ = model.run_chunk(model.initial_state(), model.forcing(1, FIRST - 1), 1)
    forcing = model.forcing(FIRST, STEPS)
    stream = StreamSpec("s", "step", "mean", VARIABLES)

    def values(scale):  # the parameters multiplied by 1 + scale[k]
        p = params._replace(
            **{name: getattr(params, name) * (1.0 + scale[k]) for k, name in enumerate(PARAMETERS)}
        )
        _, out = model.run_chunk(state, forcing, FIRST, p, streams=[stream])
        return jnp.stack([out["s"][v][:, 0] for v in VARIABLES])

    zero = jnp.zeros(len(PARAMETERS))
    values(zero)  # fills the host caches outside a trace (O76)
    eye = jnp.eye(len(PARAMETERS))
    ad = np.asarray(jax.jit(jax.vmap(lambda d: jax.jvp(values, (zero,), (d,))[1]))(eye))
    run = jax.jit(values)
    h = 1e-4
    fd = np.array([(np.asarray(run(h * d)) - np.asarray(run(-h * d))) / (2 * h) for d in eye])
    return ad, fd


@pytest.mark.parametrize("k", range(len(PARAMETERS)), ids=PARAMETERS)
def test_parameter_derivatives_match_finite_differences(derivatives, k):
    ad, fd = derivatives
    assert np.isfinite(ad).all()
    for v, name in enumerate(VARIABLES):
        scale = np.abs(fd[k, v]).max()
        assert scale > 0.0, name
        # 1e-3 of the largest value of the four steps: the finite difference is good to
        # ~1e-5; the errors this test is for were 5 % to 50 %
        np.testing.assert_allclose(ad[k, v], fd[k, v], rtol=0, atol=1e-3 * scale, err_msg=name)
