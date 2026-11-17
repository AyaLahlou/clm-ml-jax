"""clm_ml.model.Model: a compiled chunk in this process, differentiable with respect to the
parameters in reverse mode (``jax.grad``) and forward mode (``jax.jvp``; decision 104), both
checked against a finite difference; the gradient with a checkpoint on each sub-step
(``numerics.checkpoint_substeps``, decision 125); the derivative rules of decision 124
(``numerics.water_stress_derivative``, ``numerics.obukhov_derivative``): the same forward
values, derivatives that stay close to the finite difference where their feedbacks are weak.
The first three tests use the exact derivatives (``water_stress_derivative: exact``).
Slow: compiles four gradients and three forward derivatives."""

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
# the exact derivatives of the forward values (the default replaces the water stress's by a
# bounded approximation, decision 124: checked further down)
EXACT = ["model.numerics.water_stress_derivative=exact"]
model = Model.from_experiment(load_experiment(tmp, EXACT))
state, params = model.initial_state(), model.default_params()
first, n = 1, 8
forcing = model.forcing(first, n)
stream = StreamSpec("s", "step", "mean", ("latent_heat_flux",))

def steps_of(m):
    def steps(p):
        _, out = m.run_chunk(state, forcing, first, p, streams=[stream])
        return out["s"]["latent_heat_flux"][:, 0]
    return steps

def loss(p):
    return steps_of(model)(p).mean()

# vcmax25 is per PFT (the table, Phase 6): the derivative along a 0.1 % change of every
# entry (only the patch's PFT matters) against a centred finite difference
d = 1e-3 * params.vcmax25
g = float((jax.grad(loss)(params).vcmax25 * d).sum())
up, down = (steps_of(model)(params._replace(vcmax25=params.vcmax25 + e)) for e in (d, -d))
fd_steps = (up - down) / 2
fd = float(fd_steps.mean())
# forward mode along the same direction (decision 104: tridiag_2eq through
# custom_linear_solve; a custom_vjp rule refused jvp)
tangent = jax.tree.map(jnp.zeros_like, params)._replace(vcmax25=d)
_, jvp = jax.jvp(loss, (params,), (tangent,))
# the same gradient with a checkpoint on each sub-step too (decision 125; never on the CPU
# unless asked for)
other = Model.from_experiment(
    load_experiment(tmp, EXACT + ["model.numerics.checkpoint_substeps=always"])
)
gs = float((jax.grad(lambda p: steps_of(other)(p).mean())(params).vcmax25 * d).sum())
values = steps_of(model)(params)
res = {{"grad": g, "fd": fd, "jvp": float(jvp), "grad_substeps": gs,
        "fd_steps": [float(v) for v in fd_steps]}}

# the derivative rules of decision 124: the defaults (the water stress stabilized), and
# the Obukhov length stabilized too
for name, override in (("default", []),
                       ("obukhov", ["model.numerics.obukhov_derivative=stabilized"])):
    m = Model.from_experiment(load_experiment(tmp, override))
    # the forward run (a program with derivatives rounds differently: not compared)
    res[name + "_same_values"] = bool(jnp.all(steps_of(m)(params) == values))
    _, j = jax.jvp(steps_of(m), (params,), (tangent,))
    res[name + "_jvp_steps"] = [float(x) for x in j]
    res[name + "_grad"] = float(
        (jax.grad(lambda p, m=m: steps_of(m)(p).mean())(params).vcmax25 * d).sum()
    )
print("RESULT" + json.dumps(res))
"""


@pytest.fixture(scope="module")
def chunk_derivatives():
    """d(mean latent heat over 8 steps) along a 0.1 % change of vcmax25: reverse mode,
    forward mode and a centred finite difference, with the exact derivatives and with the
    derivative rules of decision 124 (one subprocess)."""
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


@pytest.mark.slow
@pytest.mark.gradient
def test_derivative_rules_keep_the_forward_values(chunk_derivatives):
    """The derivative rules of decision 124 change derivatives only: every step's value is
    bit for bit the same with the water stress stabilized (the default) or exact, and with
    the Obukhov length stabilized."""
    result = chunk_derivatives
    assert result["default_same_values"] and result["obukhov_same_values"], result


@pytest.mark.slow
@pytest.mark.gradient
@pytest.mark.parametrize("rules", ["default", "obukhov"])
def test_derivative_rules_stay_close_to_the_finite_difference(chunk_derivatives, rules):
    """The rules are approximations where their feedbacks act and exact elsewhere. On these
    eight CHATS7 steps (an afternoon into the evening: mild water stress at first, no
    overshooting stability loop) each step's derivative stays within 2 % of the largest
    step's finite difference (measured: 0.6 % at the first step, from the water-stress
    rule; the mean over the steps, whose terms cancel to 0.6 % of their size, is 9 % off:
    not compared). Reverse mode gives the forward-mode mean."""
    result = chunk_derivatives
    fd, jvp = result["fd_steps"], result[rules + "_jvp_steps"]
    largest = max(abs(v) for v in fd)
    assert max(abs(a - b) for a, b in zip(jvp, fd)) <= 0.02 * largest, result
    mean = sum(jvp) / len(jvp)
    assert abs(result[rules + "_grad"] - mean) <= 1e-6 * abs(mean), result
