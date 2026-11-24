"""
Patches spread over several devices (``execution.devices``,
``jax.shard_map``), on CPU with simulated devices (``--xla_force_host_platform_device_count``).

A domain of three gridcells and four patches, C3 and C4: ``SITEA`` (a walnut and a C4 grass),
``SITEB`` (a walnut, another location and soil) and ``SITEC`` (a needleleaf tree). Spread over
four devices (one gridcell per device block, one C3 and one C4 slot per block, the fourth
block all padding), it gives the same patch and gridcell outputs, in domain order, as on one
device.
"""

from __future__ import annotations

import copy
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]
VARS = ["latent_heat_flux", "sensible_heat_flux", "gpp", "net_radiation"]


def _towers() -> dict:
    sys.path.insert(0, str(ROOT))
    from tests.experiment_tools import chats7_tower

    chats = chats7_tower()
    walnut = chats["patches"][0]
    grass = {"pft": "c4_grass", "weight": 0.3, "canopy_height": 0.5, "root_biomass": 400.0,
             "canopy_layers": {"within": 5, "above": 10}}  # fmt: skip
    needle = {"pft": "needleleaf_evergreen_temperate_tree", "weight": 1.0, "canopy_height": 15.0,
              "root_biomass": 800.0, "canopy_layers": {"within": 8, "above": 6}}  # fmt: skip
    a = copy.deepcopy(chats)
    a["patches"] = [dict(walnut, weight=0.7), grass]
    b = copy.deepcopy(chats)
    b.update(latitude=45.0, longitude=-100.0)
    b["soil"].update(bedrock_depth=1.0, color_class=5)
    b["patches"] = [dict(walnut, weight=1.0, canopy_layers={"within": 10, "above": 8})]
    c = copy.deepcopy(chats)
    c["patches"] = [needle]
    return {"SITEA": a, "SITEB": b, "SITEC": c}


def _execute(tmp: Path, devices: int):
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from clm_ml.experiment import load_experiment
    from tests.experiment_tools import make_experiment

    work = tmp / f"devices{devices}"
    output = {
        "directory": str(work / "out"),
        "streams": [{"name": "step", "frequency": "step", "statistic": "mean",
                     "variables": VARS, "level": ["patch", "gridcell"]}],
    }  # fmt: skip
    make_experiment(work, namelist=ROOT / "data" / "namelists" / "nl.CHATS7.1day", domain=lambda d: d.update(towers=_towers()),
                    run={"execution": {"devices": devices}, "output": output})  # fmt: skip
    return load_experiment(work).execute(write=False, quiet=True)["step"]


_SCRIPT = """
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, {root!r})
from tests.regression.test_devices import _execute
out = _execute(Path({tmp!r}), {devices})
np.savez({npz!r}, patch_gridcell=np.asarray(out.patch_gridcell),
         **{{n: np.asarray(out.data[n]) for n in {names!r}}},
         **{{"g_" + n: np.asarray(out.gridcell_mean()[n]) for n in {names!r}}})
"""


def _execute_in_process(tmp: Path, devices: int) -> dict:
    """``_execute`` in a fresh Python process with four CPU devices (XLA reads the flag when
    it starts; runs are in-process)."""
    import subprocess

    npz = tmp / f"out{devices}.npz"
    env = dict(os.environ)
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    ).strip()
    script = _SCRIPT.format(root=str(ROOT), tmp=str(tmp), devices=devices, npz=str(npz), names=VARS)
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr[-3000:]
    with np.load(npz) as data:
        return {k: data[k] for k in data.files}


def test_patches_spread_over_devices(tmp_path):
    one, four = _execute_in_process(tmp_path, 1), _execute_in_process(tmp_path, 4)
    shutil.rmtree(tmp_path, ignore_errors=True)
    assert list(four["patch_gridcell"]) == [0, 0, 1, 2]
    for name in VARS:
        a, b = one[name], four[name]
        assert a.shape == b.shape == (48, 4), name
        scale = float(np.max(np.abs(a)))
        assert np.max(np.abs(b - a)) <= 1e-9 * scale + 1e-12, name
        ga, gb = one["g_" + name], four["g_" + name]
        assert ga.shape == gb.shape == (48, 3), name
        assert np.max(np.abs(gb - ga)) <= 1e-9 * scale + 1e-12, name
