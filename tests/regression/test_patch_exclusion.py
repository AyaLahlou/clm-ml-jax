"""
Phase 6 step 3 test of decision 25: a patch that fails.

One gridcell with two patches (the CHATS7 walnut, weight 0.6, and a needleleaf evergreen tree,
weight 0.4). The needleleaf patch is made to fail by starting the run (a restart at the run's
start) with its air temperatures set to NaN, so its state is non-finite after the first
chunk. (A NaN parameter such as ``vcmax25`` does not make it fail: the photosynthesis guards
close the stomata.) ``clm_ml.driver.run_streams``, against the same start without NaN:

- ``on_patch_error="stop"`` stops the run naming the failing patch;
- ``"exclude"`` (allowed fraction 0.5) completes: the needleleaf outputs are fill values, the
  walnut outputs equal those of the run where nothing fails, and the gridcell values are the
  walnut's alone (the weights renormalized) at every step, also before the failure is
  detected at the end of the chunk;
- ``"exclude"`` with an allowed fraction of 0.1 stops (1 of 2 patches failed).
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
import jax, jax.numpy as jnp
import numpy as np
from clm_ml import driver as D
from clm_ml.io.namelist import convert_namelist
from clm_ml.output_streams import StreamSpec
from clm_ml.experiment import load_experiment

tmp = Path(tempfile.mkdtemp())
convert_namelist(Path({namelist!r}), tmp)
tower = yaml.safe_load(open({towers!r}))["CHATS7"]
needle = {{"pft": "needleleaf_evergreen_temperate_tree", "weight": 0.4, "canopy_height": 15.0,
          "root_biomass": 800.0, "canopy_layers": {{"within": 8, "above": 6}}}}
tower["patches"] = [dict(tower["patches"][0], weight=0.6), needle]
edit(tmp, domain=lambda d: d.update(towers={{"MIXED": tower}}))
run = load_experiment(tmp)
sim = D.simulation(run).laid_out()
stream = StreamSpec("s", "step", "mean", ("latent_heat_flux", "gpp"))
good = jax.tree.map(jnp.asarray, sim.state0)  # before the first step
c = good.canopy
bad = good._replace(canopy=c._replace(air_temperature=c.air_temperature.at[1].set(jnp.nan)))
start = sim.start

def run_with(state, policy, fraction):
    records = []
    try:
        D.run_streams(sim, [stream], lambda i, r: records.extend(r),
                      restart={{"time": start, "state": state, "collectors": []}},
                      levels=[("patch", "gridcell")], on_patch_error=policy,
                      max_failed_patch_fraction=fraction)
    except RuntimeError as exc:  # clm_ml.checks.ModelError
        return {{"stopped": str(exc)}}
    out = {{}}
    for key in ("latent_heat_flux", "gridcell:latent_heat_flux"):
        out[key] = np.stack([v[key] for _, _, v in records]).tolist()
    return out

result = {{
    "normal": run_with(good, "stop", 0.0),
    "stop": run_with(bad, "stop", 0.0),
    "exclude": run_with(bad, "exclude", 0.5),
    "too_many": run_with(bad, "exclude", 0.1),
}}
print("RESULT" + json.dumps(result))
"""


def test_failing_patch_stop_or_exclude():
    script = SCRIPT.format(
        src=str(ROOT / "src"),
        namelist=str(ROOT / "data" / "namelists" / "nl.CHATS7.1day"),
        towers=str(ROOT / "src" / "clm_ml" / "compsets" / "towers" / "chats7.yaml"),
        root=str(ROOT),
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("RESULT"))
    r = json.loads(line[len("RESULT") :])
    assert "excluding MIXED#1" in run.stdout

    normal, excl = r["normal"], r["exclude"]
    assert "stopped" not in normal and "stopped" not in excl
    assert "MIXED#1" in r["stop"]["stopped"]  # stop: the failing patch is named
    assert "1 of 2 patches failed" in r["too_many"]["stopped"]

    lh = np.asarray(excl["latent_heat_flux"], dtype=float)  # (record, patch)
    lh_normal = np.asarray(normal["latent_heat_flux"], dtype=float)
    assert np.all(np.isnan(lh[:, 1]))  # the needleleaf is excluded: fill values
    assert np.all(np.isfinite(lh_normal))
    np.testing.assert_array_equal(lh[:, 0], lh_normal[:, 0])  # the walnut is unaffected
    grid = np.asarray(excl["gridcell:latent_heat_flux"], dtype=float)[:, 0]
    np.testing.assert_allclose(grid, lh_normal[:, 0], rtol=1e-12, atol=1e-12)
    # in the normal run the gridcell is the weighted sum of both patches
    grid_normal = np.asarray(normal["gridcell:latent_heat_flux"], dtype=float)[:, 0]
    np.testing.assert_allclose(grid_normal, 0.6 * lh_normal[:, 0] + 0.4 * lh_normal[:, 1],
                               rtol=1e-12, atol=1e-9)  # fmt: skip
