"""Running-mean acclimation temperature: the mean of all steps so far during
the first period, then a running mean with weight 1 / (steps per period)."""

import types

import jax.numpy as jnp
import numpy as np

from clm_ml.driver import advance_acclimation
from clm_ml.state import CanopyState, SoilState, State


def _state(mean, steps):
    """One patch's state (scalars per patch; profiles of 2 layers, unused here)."""
    z, leaf = jnp.zeros(2), jnp.zeros((2, 2))
    canopy = CanopyState(z, z, z, leaf, leaf, z, jnp.array(0.0), jnp.array(mean), jnp.array(steps))
    return State(canopy=canopy, soil=SoilState(temperature=z))


def _run(days, mode="running_mean"):
    return types.SimpleNamespace(
        acclimation=mode,
        acclimation_days=days,
        step_seconds=86400,  # one step per day
    )


def test_mean_of_all_steps_during_the_first_period_then_running():
    temps = [280.0, 290.0, 300.0, 310.0, 320.0]
    run = _run(days=3)  # period: 3 steps
    state = _state(temps[0], 1.0)  # the first step initializes the mean
    means = [temps[0]]
    for t in temps[1:]:
        state = advance_acclimation(run, state, t)
        means.append(float(state.canopy.acclimation_temperature))
    expected = [280.0, 285.0, 290.0]  # plain means of steps 1..3
    expected.append(290.0 + (310.0 - 290.0) / 3)  # then weight 1/3
    expected.append(expected[-1] + (320.0 - expected[-1]) / 3)
    np.testing.assert_allclose(means, expected, rtol=1e-14)
    assert float(state.canopy.acclimation_steps) == 5.0


def test_run_mean_leaves_the_state_unchanged():
    state = _state(288.0, 1.0)
    assert advance_acclimation(_run(3, "run_mean"), state, 300.0) is state
