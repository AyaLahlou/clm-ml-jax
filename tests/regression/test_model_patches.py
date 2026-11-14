"""
Phase 6 step 3 test of the in-process model (``clm_ml.model.Model``) with several patches.

One gridcell with two patches of different PFTs (the CHATS7 walnut, weight 0.6, and a
needleleaf evergreen tree, weight 0.4), the step batched over the patches:

- the parameters are a table with one row per PFT (``Model.pft_names``);
- the gradient of the gridcell's mean latent heat flux with respect to the patch weights is
  each patch's mean latent heat flux (the gridcell value is the weighted sum of the patches);
- (slow) the gradient with respect to the PFT table is per PFT: the needleleaf row's
  ``vcmax25`` gradient matches a finite difference.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = """
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, {src!r}); sys.path.insert(0, {root!r})
import yaml
from tests.experiment_tools import edit
import jax
import numpy as np
from clm_ml.model import Model
from clm_ml.output_streams import StreamSpec
from clm_ml.io.namelist import convert_namelist
from clm_ml.experiment import load_experiment

tmp = Path(tempfile.mkdtemp())
convert_namelist(Path({namelist!r}), tmp)
tower = yaml.safe_load(open({towers!r}))["CHATS7"]
walnut = dict(tower["patches"][0], weight=0.6)
needle = {{"pft": "needleleaf_evergreen_temperate_tree", "weight": 0.4, "canopy_height": 15.0,
          "root_biomass": 800.0, "canopy_layers": {{"within": 8, "above": 6}}}}
tower["patches"] = [walnut, needle]
edit(tmp, domain=lambda d: d.update(towers={{"MIXED": tower}}))

model = Model.from_experiment(load_experiment(tmp))
state, params, weights = model.initial_state(), model.default_params(), model.weights()
first, n = 36, 2  # midday
forcing = model.forcing(first, n)
stream = StreamSpec("s", "step", "mean", ("latent_heat_flux",))

def chunk(p, w):
    _, out = model.run_chunk(state, forcing, first, p, streams=[stream], weights=w)
    return out["s"]

def gridcell_lh(p, w):
    return chunk(p, w)["gridcell:latent_heat_flux"].mean()

out = chunk(params, weights)
result = {{
    "pft_names": list(model.pft_names),
    "vcmax25_rows": np.asarray(params.vcmax25).tolist(),
    "patch_lh_mean": np.asarray(out["latent_heat_flux"]).mean(axis=0).tolist(),
    "weights": np.asarray(weights).tolist(),
    "dw": np.asarray(jax.grad(gridcell_lh, argnums=1)(params, weights)).tolist(),
}}
if {gradient!r}:
    g = jax.grad(gridcell_lh)(params, weights)
    result["dvcmax25"] = np.asarray(g.vcmax25).tolist()
    eps = 1e-3 * float(params.vcmax25[1])
    up = params._replace(vcmax25=params.vcmax25.at[1].add(eps))
    dn = params._replace(vcmax25=params.vcmax25.at[1].add(-eps))
    result["fd_needle"] = (float(gridcell_lh(up, weights)) - float(gridcell_lh(dn, weights))) / (2 * eps)
print("RESULT" + json.dumps(result))
"""


def _run(gradient: bool) -> dict:
    script = SCRIPT.format(
        src=str(ROOT / "src"),
        namelist=str(ROOT / "data" / "namelists" / "nl.CHATS7.1day"),
        towers=str(ROOT / "src" / "clm_ml" / "compsets" / "towers" / "chats7.yaml"),
        root=str(ROOT),
        gradient=gradient,
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("RESULT"))
    return json.loads(line[len("RESULT") :])


def _check_common(r: dict) -> None:
    assert r["pft_names"] == [
        "broadleaf_deciduous_temperate_tree",
        "needleleaf_evergreen_temperate_tree",
    ]
    assert len(r["vcmax25_rows"]) == 2
    np.testing.assert_allclose(r["weights"], [0.6, 0.4])
    # d(sum_p w_p x_p)/d(w_p) = x_p
    np.testing.assert_allclose(r["dw"], r["patch_lh_mean"], rtol=1e-12)
    assert abs(r["patch_lh_mean"][0] - r["patch_lh_mean"][1]) > 1e-3


@pytest.mark.regression
def test_model_patches_weights_gradient():
    _check_common(_run(gradient=False))


@pytest.mark.slow
def test_model_patches_pft_gradient():
    r = _run(gradient=True)
    _check_common(r)
    g = r["dvcmax25"]
    assert np.all(np.isfinite(g)) and g[0] != 0.0 and g[1] != 0.0
    assert abs(g[1] - r["fd_needle"]) <= 0.05 * abs(r["fd_needle"]), r
