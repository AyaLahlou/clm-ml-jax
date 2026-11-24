"""
C3 and C4 patches in one compiled program.

One gridcell with two C3 patches (the CHATS7 walnut, weight 0.5, and a needleleaf tree, 0.2)
and a C4 patch (C4 grass, 0.5 m, explicit layer counts, 0.3), listed walnut, grass,
needleleaf. They form two groups inside one program (contiguous slots: the C3 patches, then
the C4 patch), each running only its own photosynthesis pathway. The run of the gridcell
matches each patch run alone, and the gridcell value is the area-weighted sum. (With a C3
group of one patch, scattering the groups' results into the scan's state would pass this
test but be wrong for larger groups.)
"""

from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]
VARS = ["latent_heat_flux", "sensible_heat_flux", "gpp", "net_radiation"]


def _execute(tmp: Path, name: str, towers: dict):
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from clm_ml.experiment import load_experiment
    from tests.experiment_tools import make_experiment

    work = tmp / name
    output = {
        "directory": str(work / "out"),
        "streams": [{"name": "step", "frequency": "step", "statistic": "mean",
                     "variables": VARS, "level": ["patch", "gridcell"]}],
    }  # fmt: skip
    make_experiment(work, namelist=ROOT / "data" / "namelists" / "nl.CHATS7.1day", run={"output": output},
                    domain=lambda d: d.update(towers=towers))  # fmt: skip
    return load_experiment(work).execute(write=False, quiet=True)["step"]


def _close(actual, expected, name):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    scale = float(np.max(np.abs(expected)))
    assert np.max(np.abs(actual - expected)) <= 1e-9 * scale + 1e-12, name


def test_c3_and_c4_patches_in_one_program(tmp_path):
    sys.path.insert(0, str(ROOT))
    from tests.experiment_tools import chats7_tower

    chats = chats7_tower()
    walnut = dict(chats["patches"][0], weight=0.5)
    grass = {"pft": "c4_grass", "weight": 0.3, "canopy_height": 0.5, "root_biomass": 400.0,
             "canopy_layers": {"within": 5, "above": 10}}  # fmt: skip
    needle = {"pft": "needleleaf_evergreen_temperate_tree", "weight": 0.2,
              "canopy_height": 15.0, "root_biomass": 800.0, "canopy_layers": {"within": 8, "above": 6}}  # fmt: skip
    mixed = copy.deepcopy(chats)
    mixed["patches"] = [walnut, grass, needle]
    jobs = {"mixed": {"MIXED": mixed}}
    for name, patch in (("C3", walnut), ("C4", grass), ("C3B", needle)):
        t = copy.deepcopy(chats)
        t["patches"] = [dict(patch, weight=1.0)]
        jobs[name] = {name: t}
    # one after the other: runs are in this process, and netCDF is not
    # thread-safe
    out = {k: _execute(tmp_path, k, v) for k, v in jobs.items()}
    shutil.rmtree(tmp_path, ignore_errors=True)

    mixed = out["mixed"]
    for name in VARS:
        _close(mixed.data[name][:, 0], out["C3"].data[name][:, 0], f"{name} C3")
        _close(mixed.data[name][:, 1], out["C4"].data[name][:, 0], f"{name} C4")
        _close(mixed.data[name][:, 2], out["C3B"].data[name][:, 0], f"{name} C3B")
        expected = (
            0.5 * out["C3"].data[name][:, 0]
            + 0.3 * out["C4"].data[name][:, 0]
            + 0.2 * out["C3B"].data[name][:, 0]
        )
        _close(mixed.gridcell_mean()[name][:, 0], expected, f"{name} gridcell")
    # the C4 pathway takes effect: daytime GPP differs from the C3 patch's
    assert not np.allclose(mixed.data["gpp"][:, 0], mixed.data["gpp"][:, 1], rtol=0.1)
