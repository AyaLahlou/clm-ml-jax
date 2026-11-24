"""
The setup as pure functions (``clm_ml.setup``) equals a recorded reference setup bit for bit:
every patch's constants (``clm_ml.step.PatchConstants``) and state before the first step, and
the tower forcing read by the readers (``clm_ml.tower_forcing``).

Seven patches on three gridcells with every towers.yaml input varied (as
``test_vector_setup``): two forcing heights, four PFTs (C3 and C4), sand/clay given or from
the texture class, organic matter, soil colour, bedrock depth, canopy height given or the PFT
default, root biomass, leaf/stem profiles given or the PFT default, explicit or default layer
counts. The reference (``reference/setup_7patches.npz``) holds the values of an independent
earlier implementation of the setup: each patch's constants and state after its first-step
initialization, and 48 steps of tower forcing. The soil hydraulic and thermal values of the
tower given only a texture class (G3) differ from the reference on purpose: they are their
own class's values, where the reference took the next class's
(``tests/clm_ml/test_setup_soil.py``); every other value is compared bit for bit. The
forcing's ``soil_water`` is the reference's ``soil_water_clm50`` (CLM soil layer structure
only).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = """
import sys, tempfile, copy
from pathlib import Path
ROOT = Path({root!r}); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import numpy as np, yaml, jax
jax.config.update("jax_enable_x64", True)
from clm_ml.io.namelist import convert_namelist
from clm_ml.experiment import load_experiment
from clm_ml.setup.patches import build_patches
from clm_ml.setup.towers import gridcell_sources, first_values
from tests.experiment_tools import chats7_tower, edit
tmp = Path(tempfile.mkdtemp())
convert_namelist(ROOT / "data/namelists/nl.CHATS7.1day", tmp)
chats = chats7_tower()
walnut = chats["patches"][0]
def tower(lat, lon, ht, soil, patches):
    t = copy.deepcopy(chats); t.update(latitude=lat, longitude=lon, measurement_height=ht); t["soil"].update(soil); t["patches"] = patches; return t
towers = {{
  "G1": tower(38.49, -121.84, 23.0, {{}}, [dict(walnut, weight=0.5),
        {{"pft": "needleleaf_evergreen_temperate_tree", "weight": 0.3, "canopy_height": 15.0, "root_biomass": 800.0, "canopy_layers": {{"within": 8, "above": 6}}}},
        {{"pft": "c4_grass", "weight": 0.2, "canopy_height": 0.5, "root_biomass": 400.0, "canopy_layers": {{"within": 5, "above": 10}}}}]),
  "G2": tower(45.0, -100.0, 23.0, {{"bedrock_depth": 1.0, "color_class": 5, "texture": {{"given_as": "percentages", "sand_percent": 40.0, "clay_percent": 20.0}}, "organic_matter": 20.0}},
        [dict(walnut, weight=0.7, canopy_height=12.0, root_biomass=600.0, area_profiles={{"leaf_beta": [3.0, 1.5], "stem_beta": [2.0, 1.5]}}),
         {{"pft": "broadleaf_deciduous_boreal_tree", "weight": 0.3, "root_biomass": 500.0, "canopy_layers": {{"within": 10, "above": 4}}}}]),
  "G3": tower(10.0, 20.0, 21.0, {{"texture": {{"given_as": "class", "texture_class": "loam"}}, "bedrock_depth": 3.0, "color_class": 10}},
        [dict(walnut, weight=0.6), {{"pft": "needleleaf_evergreen_temperate_tree", "weight": 0.4, "canopy_height": 9.0, "root_biomass": 700.0}}]),
}}
edit(tmp, domain=lambda d: d.update(towers=towers))
run = load_experiment(tmp)

# the setup first, in a fresh process (no module state from other runs)
sources = gridcell_sources(run)
first = first_values(run, sources)
gi = run.domain.patches.gridcell_index
per_patch = {{k: v[gi] for k, v in first.items()}}
new = build_patches(run.domain, run.params, run.config, per_patch, acclimation_steps=1.0)
chunks = [s.chunk(0, 48) for s in sources]

ref = np.load(ROOT / "tests/regression/reference/setup_7patches.npz")
bad = n_cmp = 0
def compare(name, a, b):
    global bad, n_cmp
    n_cmp += 1
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
        bad += 1
        print("DIFF", name, a.shape, b.shape, a.dtype, b.dtype)
# G3 gives a texture class only; its columns' hydraulic and thermal values come from their
# own class (the reference took the next class's)
ISSUE21 = {{"constants.watsat", "constants.sucsat", "constants.bsw", "constants.hksat",
           "constants.tkmg", "constants.tkdry"}}
g3 = np.asarray(gi) == 2
for name, tree in (("constants", new.constants), ("state", new.state)):
    for path, b in jax.tree_util.tree_leaves_with_path(tree):
        key = name + jax.tree_util.keystr(path)
        if key in ISSUE21:
            compare(key, ref[key][~g3], np.asarray(b)[~g3])  # every other column unchanged
            if np.array_equal(ref[key][g3], np.asarray(b)[g3]):
                bad += 1; print("G3 TEXTURE VALUES EQUAL THE REFERENCE", key)
        else:
            compare(key, ref[key], b)
print("c3 equal:", ref["c3"].tolist() == new.c3.tolist())
xbad = 0
# the reference's names: soil_water is its soil_water_clm50, and its soil_water_clm45 (the
# CLM4.5 layout) is all fill values
OLD_NAMES = {{"soil_water": "soil_water_clm50"}}
for g, chunk in enumerate(chunks):
    if not np.all(ref[f"forcing{{g}}.clm_history.soil_water_clm45"] == 1e36):
        xbad += 1; print("FORCING DIFF", g, "soil_water_clm45 not fill")
    for part in ("tower", "clm_history"):
        for k, v in vars(getattr(chunk, part)).items():
            key = f"forcing{{g}}.{{part}}.{{OLD_NAMES.get(k, k)}}"
            if (v is None) != (key not in ref.files) or (v is not None and not np.array_equal(ref[key], v)):
                xbad += 1; print("FORCING DIFF", key)
print("compared", n_cmp)
print("setup fields differing:", bad, "; forcing fields differing:", xbad)
"""


def test_setup_equals_legacy_setup():
    run = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(root=str(ROOT))], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr[-3000:]
    out = run.stdout
    assert "setup fields differing: 0 ; forcing fields differing: 0" in out, out[-3000:]
    assert "c3 equal: True" in out, out[-3000:]
    assert "compared 38" in out, out[-3000:]  # 28 constant and 10 state arrays (P, ...)
