"""Derivatives on real 1° patches, where two defects of the time stepping made them useless:
almost dry leaves that alternated between wet and dry (derivatives of 1e35 after a day of
drizzle on the Great Plains) and short canopies whose stable turbulence collapsed and
recovered from stage to stage (the daily-mean derivative 40 % off in the cerrado). One day
of each box: the forward-mode derivative of latent heat with respect to ``iota`` stays
bounded, and its daily mean agrees with a ±1 % change of ``iota``.

Gradient tier (``tests/gradient_tier.py``). Needs the 1° global input files on NCAR's GLADE
file system (skipped elsewhere).
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from tests.regression.test_grid_setup import needs_data  # noqa: E402

pytestmark = [pytest.mark.gradient, pytest.mark.slow, needs_data]

ROOT = Path(__file__).resolve().parents[1]
#: box, first day, largest relative error of the daily-mean derivative (worst patch; measured
#: 0.007 and 0.023; before the two decisions: exploding and 0.40)
CASES = {
    "great_plains_july": ("{lat: [35, 37], lon: [-100, -98]}", "2000-07-01", 0.03),
    "cerrado_january": ("{lat: [-12, -10], lon: [-52, -50]}", "2000-01-01", 0.05),
}


@pytest.mark.parametrize("case", CASES)
def test_daily_mean_derivative_matches_a_finite_change(case):
    from clm_ml.experiment import load_experiment
    from clm_ml.model import Model
    from clm_ml.output_streams import StreamSpec

    box, start, tolerance = CASES[case]
    run = load_experiment(
        ROOT / "configs" / "grid_1deg_2000",
        [f"domain.grid.region={box}", "run.execution.devices=1", f"run.time.start={start}"],
    )
    model = Model.from_experiment(run)
    state, params, forcing = model.initial_state(), model.default_params(), model.forcing(1, 47)
    stream = StreamSpec("s", "step", "mean", ("latent_heat_flux",))

    def latent_heat(e):  # (patch, step), iota multiplied by 1 + e
        p = params._replace(iota=params.iota * (1.0 + e))
        _, out = model.run_chunk(state, forcing, 1, p, streams=[stream])
        return out["s"]["latent_heat_flux"].T

    latent_heat(0.0)  # fills the host caches outside a trace
    ad = np.asarray(jax.jit(lambda e: jax.jvp(latent_heat, (e,), (jnp.ones(()),))[1])(0.0))
    run_ = jax.jit(latent_heat)
    fd = (np.asarray(run_(0.01)) - np.asarray(run_(-0.01))) / 0.02
    assert np.isfinite(ad).all()
    assert np.abs(ad).max() < 1e4  # W m-2 per unit relative change; exploding ones reached 1e35
    ad_mean, fd_mean = ad.mean(axis=1), fd.mean(axis=1)
    error = np.abs(ad_mean - fd_mean) / np.maximum(np.abs(fd_mean), 0.5)
    assert error.max() < tolerance, (error.max(), int(error.argmax()))
