"""
Phase 6 step 2 test: one compiled program serves sites that differ in every per-site field.

Site B is CHATS7's forcing and vegetation under another name with a different latitude and
longitude, bedrock depth, soil colour class and explicit layer counts (towers.yaml), so every
per-site value the time step reads (``SiteType.site_type`` and the layer counts) differs from
CHATS7 (site A). In one process:

- B's chunk run through A's compiled program (B's state, forcing and constants as inputs)
  equals B's own run, and A's program does not compile again: the site data are inputs of the
  program, not constants baked into it (a mix-up, e.g. B reading A's bedrock depth, would show
  as a difference);
- B's own run in this process equals B alone in its own process (A's setup does not leak
  into B);
- B differs from A (the site data take effect).
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

SITE_B = {
    "latitude": 45.0,  # CHATS7: 38.49
    "longitude": -100.0,  # CHATS7: -121.84
    "bedrock_depth": 1.0,  # m; CHATS7: 2.0
    "color_class": 5,  # CHATS7: 15
    "n_layers_within": 10,  # CHATS7: 20 (default spacing)
    "n_layers_above": 8,  # CHATS7: 26
}

SCRIPT = """
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, {src!r}); sys.path.insert(0, {root!r})
import yaml
from tests.experiment_tools import edit
import numpy as np
from clm_ml.model import Model
from clm_ml import output_streams as S
from clm_ml.io.namelist import convert_namelist
from clm_ml.experiment import load_experiment

site_b = {site_b!r}
root = Path(tempfile.mkdtemp())
tmp, tmp_b = root / "a", root / "b"
convert_namelist(Path({namelist!r}), tmp)
convert_namelist(Path({namelist!r}), tmp_b)
tower = yaml.safe_load(open({towers!r}))["CHATS7"]
tower.update(latitude=site_b["latitude"], longitude=site_b["longitude"])
tower["soil"].update(bedrock_depth=site_b["bedrock_depth"], color_class=site_b["color_class"])
tower["patches"][0].update(canopy_layers={{"within": site_b["n_layers_within"],
                                          "above": site_b["n_layers_above"]}})
edit(tmp_b, domain=lambda d: d.update(towers={{"SITEB": tower}}))

names = ("latent_heat_flux", "sensible_heat_flux", "gpp", "soil_temperature",
         "net_radiation")
stream = S.StreamSpec("s", "step", "instantaneous", names)
first, n = 12, 6

def own(model):
    _, out = model.run_chunk(model.initial_state(), model.forcing(first, n), first,
                             streams=[stream])
    return {{k: np.asarray(v).tolist() for k, v in out["s"].items()}}

def site(model):
    k = model.constants()  # PatchConstants, (patch, ...)
    return {{"bedrock": int(np.asarray(k.nbedrock)[0]),
             "albsat_vis": float(np.asarray(k.albsat)[0, 0]),
             "lat": float(np.asarray(k.latdeg)[0]),
             "layers": int(np.asarray(k.layers.n_layers).max())}}

result = {{}}
if {alone!r}:
    b = Model.from_experiment(load_experiment(tmp_b))
    result["b_alone"] = own(b)
else:
    a = Model.from_experiment(load_experiment(tmp))
    b = Model.from_experiment(load_experiment(tmp_b))
    result["site_a"], result["site_b"] = site(a), site(b)
    result["a"] = own(a)
    runner = next(iter(a._runners.values()))
    result["b"] = own(b)
    result["compiled_before"] = runner._cache_size()
    _, via = b.run_chunk(b.initial_state(), b.forcing(first, n), first, streams=[stream],
                         program=a)  # b's inputs (state, forcing, constants) through a's program
    result["b_via_a"] = {{k: np.asarray(v).tolist() for k, v in via["s"].items()}}
    result["compiled_after"] = runner._cache_size()
print("RESULT" + json.dumps(result))
"""


def _run(alone: bool) -> dict:
    script = SCRIPT.format(
        src=str(ROOT / "src"),
        namelist=str(ROOT / "data" / "namelists" / "nl.CHATS7.1day"),
        towers=str(ROOT / "src" / "clm_ml" / "compsets" / "towers" / "chats7.yaml"),
        root=str(ROOT),
        site_b=SITE_B,
        alone=alone,
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("RESULT"))
    return json.loads(line[len("RESULT") :])


def test_one_compiled_program_for_different_sites():
    together, alone = _run(alone=False), _run(alone=True)
    # the two sites differ in every per-site field
    a, b = together["site_a"], together["site_b"]
    from clm_ml.setup.tables import ALBSAT

    assert a == {"bedrock": 12, "albsat_vis": ALBSAT[0][14], "lat": 38.49, "layers": 46}
    assert b["albsat_vis"] == ALBSAT[0][SITE_B["color_class"] - 1]
    assert b["lat"] == SITE_B["latitude"]
    assert b["bedrock"] < a["bedrock"]
    assert b["layers"] == SITE_B["n_layers_within"] + SITE_B["n_layers_above"]

    assert together["compiled_after"] == together["compiled_before"]  # no new compilation
    for name, own in together["b"].items():
        own = np.asarray(own, dtype=float)
        # B in this process equals B alone (A's setup does not leak into B)
        np.testing.assert_array_equal(own, np.asarray(alone["b_alone"][name]), err_msg=name)
        # B through A's program equals B's own run: separately compiled programs may round
        # differently (as in test_one_program_any_layers), within 1e-9 of the magnitude
        via = np.asarray(together["b_via_a"][name], dtype=float)
        np.testing.assert_array_equal(np.isnan(via), np.isnan(own), err_msg=name)
        ok = np.isfinite(own)
        scale = float(np.max(np.abs(own[ok]), initial=0.0))
        assert np.max(np.abs(via[ok] - own[ok]), initial=0.0) <= 1e-9 * scale + 1e-12, name
        # the site data take effect
        if name in ("net_radiation", "soil_temperature", "latent_heat_flux"):
            assert not np.allclose(own, np.asarray(together["a"][name]), rtol=1e-6), name
