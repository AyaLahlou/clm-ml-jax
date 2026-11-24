"""
Parameter ensemble: jax.vmap of a compiled chunk over a batch of CanopyParams.

A member with the default parameters must reproduce the single run, and every member must
give finite results. (A 32-member, one-day vcmax25 ensemble runs in ~11 s on CPU; the members
differ in the daytime as expected.)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

HERE = Path(__file__).resolve().parent

SCRIPT = """
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, {src!r}); sys.path.insert(0, {root!r})
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from clm_ml.io.namelist import convert_namelist
from clm_ml.experiment import load_experiment
from clm_ml.model import Model

tmp = Path(tempfile.mkdtemp())
convert_namelist(Path({namelist!r}), tmp)
model = Model.from_experiment(load_experiment(tmp))
first, n = 1, 8
state, forcing, table = model.initial_state(), model.forcing(first, n), model.default_params()
factors = jnp.array([1.0, 0.8, 0.9, 1.2])
ens = jax.tree.map(lambda x: jnp.broadcast_to(x, (4,) + jnp.shape(x)), table)
ens = ens._replace(vcmax25=table.vcmax25[None] * factors[:, None])
final = jax.vmap(lambda p: model.run_chunk(state, forcing, first, p)[0])(ens)
single, _ = model.run_chunk(state, forcing, first)
diff = max(
    float(np.max(np.abs(np.asarray(a)[0] - np.asarray(b))) / (np.max(np.abs(np.asarray(b))) or 1))
    for a, b in zip(jax.tree.leaves(final), jax.tree.leaves(single))
)
finite = all(bool(np.isfinite(np.asarray(a)).all()) for a in jax.tree.leaves(final))
print("RESULT" + json.dumps({{"member0_vs_single": diff, "finite": finite}}))
"""


def test_parameter_ensemble():
    root = HERE.parents[1]
    script = SCRIPT.format(
        src=str(root / "src"),
        root=str(root),
        namelist=str(root / "data" / "namelists" / "nl.CHATS7.50steps"),
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("RESULT"))
    result = json.loads(line[len("RESULT") :])
    assert result["finite"], "non-finite ensemble state"
    assert result["member0_vs_single"] <= 1e-13, result
