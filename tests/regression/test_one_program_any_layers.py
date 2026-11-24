"""
One compiled program serves patches with different layer counts.

Two CHATS7 models in one process with the same ``max_layers`` (50): the default layer
spacing (46 layers: 20 within the canopy, 26 above) and explicit counts (10 within, 10
above: 20 layers). The second model's chunk is run through the first model's compiled chunk
(its state, forcing and constants as inputs) and must give the second model's own results; the
first model's program must not compile again (layer counts are data, not compile-time
constants).
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
sys.path.insert(0, {src!r})
from clm_ml.model import Model
import numpy as np
from clm_ml import output_streams as S
from clm_ml.io.namelist import convert_namelist
from clm_ml.experiment import load_experiment
from clm_ml.outputs import VARIABLES

tmp = Path(tempfile.mkdtemp())
convert_namelist(Path({namelist!r}), tmp)
names = tuple(v.name for v in VARIABLES)
stream = S.StreamSpec("s", "step", "instantaneous", names)
first, n = 12, 6

a = Model.from_experiment(load_experiment(tmp))
b = Model.from_experiment(load_experiment(
    tmp, ["model.canopy_layers.layer_counts={{within: 10, above: 10}}"]))
layers = [int(np.asarray(m.constants().layers.n_layers).max()) for m in (a, b)]

def chunk(model):
    return model.run_chunk(model.initial_state(), model.forcing(first, n), first,
                           streams=[stream])

chunk(a)  # creates a's program
runner = next(iter(a._runners.values()))
_, own = chunk(b)  # b through its own program
compiled_before = runner._cache_size()
_, via_a = b.run_chunk(b.initial_state(), b.forcing(first, n), first, streams=[stream],
                       program=a)  # b's inputs (state, forcing, constants) through a's program
out = {{
    "layers": layers,
    "compiled_before": compiled_before,
    "compiled_after": runner._cache_size(),
    "own": {{k: np.asarray(v).tolist() for k, v in own["s"].items() if ":" not in k}},
    "via_a": {{k: np.asarray(v).tolist() for k, v in via_a["s"].items() if ":" not in k}},
}}
print("RESULT" + json.dumps(out))
"""


def test_one_compiled_program_for_different_layer_counts():
    script = SCRIPT.format(
        src=str(ROOT / "src"),
        namelist=str(ROOT / "data" / "namelists" / "nl.CHATS7.1day"),
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("RESULT"))
    result = json.loads(line[len("RESULT") :])
    assert result["layers"] == [46, 20]
    assert result["compiled_after"] == result["compiled_before"]  # no new compilation
    for name, own in result["own"].items():
        own = np.asarray(own, dtype=float)
        via = np.asarray(result["via_a"][name], dtype=float)
        np.testing.assert_array_equal(np.isnan(via), np.isnan(own), err_msg=name)
        # Two separately compiled programs may round differently: within 1e-9 of each
        # variable's magnitude, as the padding test (measured: ~1e-11, the twin-noise level),
        # plus 1e-12 absolute for variables that are zero to round-off (night-time GPP)
        ok = np.isfinite(own)
        scale = float(np.max(np.abs(own[ok]), initial=0.0))
        assert np.max(np.abs(via[ok] - own[ok]), initial=0.0) <= 1e-9 * scale + 1e-12, name
