"""
Several models of the same tower in one Python process (plan Phase 5a, decision 20): each
model holds its own options and parameters, so a model with Medlyn stomatal conductance
created after a WUE model does not change the WUE model's results, and gives the same
results as when it is alone in its process. A model of another tower (a copy of CHATS7
under another name) is accepted since Phase 6 step 2 (per-site data in the containers): it
gives CHATS7's results and leaves the first model's results unchanged.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = """
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, {src!r}); sys.path.insert(0, {root!r})
import yaml
from tests.experiment_tools import edit
from clm_ml.io.namelist import convert_namelist
from clm_ml.experiment import load_experiment
from clm_ml.model import Model
from clm_ml.output_streams import StreamSpec

tmp = Path(tempfile.mkdtemp())
convert_namelist(Path({namelist!r}), tmp)
stream = StreamSpec("s", "step", "mean", ("latent_heat_flux", "gpp"))

def run(model):
    first, n = 1, 4
    _, out = model.run_chunk(model.initial_state(), model.forcing(first, n), first,
                             streams=[stream])
    return {{k: [float(x) for x in v.ravel()] for k, v in out["s"].items()}}

result = {{}}
models = {{}}
for name in {models!r}:
    overrides = ["model.processes.stomatal_conductance.scheme=" + name]
    models[name] = Model.from_experiment(load_experiment(tmp, overrides))
    result[name] = run(models[name])
if len({models!r}) > 1:
    result["wue_again"] = run(Model.from_experiment(load_experiment(tmp)))
    towers = yaml.safe_load(open({towers!r}))
    other = Path(tempfile.mkdtemp()) / "other"
    convert_namelist(Path({namelist!r}), other)
    edit(other, domain=lambda d: d.update(towers={{"OTHER": towers["CHATS7"]}}))
    result["other_tower"] = run(Model.from_experiment(load_experiment(other)))
    result["wue_after_other"] = run(models["wue"])
print("RESULT" + json.dumps(result))
"""


def _run(models):
    script = SCRIPT.format(
        src=str(ROOT / "src"),
        namelist=str(ROOT / "data" / "namelists" / "nl.CHATS7.1day"),
        towers=str(ROOT / "src" / "clm_ml" / "compsets" / "towers" / "chats7.yaml"),
        root=str(ROOT),
        models=list(models),
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("RESULT"))
    return json.loads(line[len("RESULT") :])


def test_two_models_of_one_tower_in_one_process():
    together = _run(["wue", "medlyn"])
    alone = _run(["medlyn"])
    for var in ("latent_heat_flux", "gpp"):
        # the second model does not disturb the first
        np.testing.assert_array_equal(together["wue_again"][var], together["wue"][var])
        # a model gives the same results with or without another model in its process
        np.testing.assert_array_equal(together["medlyn"][var], alone["medlyn"][var])
    # the options are the models' own: the schemes differ
    assert not np.allclose(together["medlyn"]["latent_heat_flux"],
                           together["wue"]["latent_heat_flux"])  # fmt: skip
    for var in ("latent_heat_flux", "gpp"):
        # another tower in the process: a copy of CHATS7 gives CHATS7's results, and the first
        # model's results do not change
        np.testing.assert_array_equal(together["other_tower"][var], together["wue"][var])
        np.testing.assert_array_equal(together["wue_after_other"][var], together["wue"][var])
