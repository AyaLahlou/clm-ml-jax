"""clm_ml.model.Model: a compiled chunk in this process, differentiable with respect to the
parameters in reverse mode (``jax.grad``) and forward mode (``jax.jvp``; decision 104), both
checked against a finite difference; the gradient with a checkpoint on each sub-step
(``numerics.checkpoint_substeps``, decision 125). Slow: compiles two gradients and a forward
derivative."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = """
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, {src!r}); sys.path.insert(0, {root!r})
import jax
import jax.numpy as jnp
from clm_ml.io.namelist import convert_namelist
from clm_ml.experiment import load_experiment
from clm_ml.model import Model
from clm_ml.output_streams import StreamSpec

tmp = Path(tempfile.mkdtemp())
convert_namelist(Path({namelist!r}), tmp)
model = Model.from_experiment(load_experiment(tmp))
state, params = model.initial_state(), model.default_params()
first, n = 1, 8
forcing = model.forcing(first, n)
stream = StreamSpec("s", "step", "mean", ("latent_heat_flux",))

def loss(p):
    _, out = model.run_chunk(state, forcing, first, p, streams=[stream])
    return out["s"]["latent_heat_flux"].mean()

# vcmax25 is per PFT (the table, Phase 6): the derivative along a 0.1 % change of every
# entry (only the patch's PFT matters) against a centred finite difference
d = 1e-3 * params.vcmax25
g = float((jax.grad(loss)(params).vcmax25 * d).sum())
fd = (float(loss(params._replace(vcmax25=params.vcmax25 + d)))
      - float(loss(params._replace(vcmax25=params.vcmax25 - d)))) / 2
# forward mode along the same direction (decision 104: tridiag_2eq through
# custom_linear_solve; a custom_vjp rule refused jvp)
tangent = jax.tree.map(jnp.zeros_like, params)._replace(vcmax25=d)
_, jvp = jax.jvp(loss, (params,), (tangent,))
# the same gradient with a checkpoint on each sub-step too (decision 125; never on the CPU
# unless asked for)
other = Model.from_experiment(load_experiment(tmp, ["model.numerics.checkpoint_substeps=always"]))

def loss_substeps(p):
    _, out = other.run_chunk(state, forcing, first, p, streams=[stream])
    return out["s"]["latent_heat_flux"].mean()

gs = float((jax.grad(loss_substeps)(params).vcmax25 * d).sum())
print("RESULT" + json.dumps({{"grad": g, "fd": fd, "jvp": float(jvp), "grad_substeps": gs}}))
"""


@pytest.fixture(scope="module")
def chunk_derivatives():
    """d(mean latent heat over 8 steps) along a 0.1 % change of vcmax25: reverse mode,
    forward mode and a centred finite difference (one subprocess, one model)."""
    script = SCRIPT.format(
        src=str(ROOT / "src"),
        root=str(ROOT),
        namelist=str(ROOT / "data" / "namelists" / "nl.CHATS7.50steps"),
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("RESULT"))
    return json.loads(line[len("RESULT") :])


@pytest.mark.slow
@pytest.mark.gradient
def test_model_chunk_gradient_matches_finite_difference(chunk_derivatives):
    result = chunk_derivatives
    assert result["grad"] != 0.0
    assert abs(result["grad"] - result["fd"]) <= 0.05 * abs(result["fd"]), result


@pytest.mark.slow
@pytest.mark.gradient
def test_model_chunk_forward_mode_matches_reverse_mode(chunk_derivatives):
    """Forward and reverse mode differentiate the same compiled chunk: equal up to round-off
    amplified through the model (~1e-8 relative on CHATS7), far below the finite difference's
    error (~1 %)."""
    result = chunk_derivatives
    assert result["jvp"] != 0.0
    assert abs(result["jvp"] - result["grad"]) <= 1e-6 * abs(result["grad"]), result
    assert abs(result["jvp"] - result["fd"]) <= 0.05 * abs(result["fd"]), result


@pytest.mark.slow
@pytest.mark.gradient
def test_model_chunk_gradient_same_with_substep_checkpoint(chunk_derivatives):
    """A checkpoint on each sub-step changes what the gradient stores, not its value."""
    result = chunk_derivatives
    assert abs(result["grad_substeps"] - result["grad"]) <= 1e-9 * abs(result["grad"]), result
