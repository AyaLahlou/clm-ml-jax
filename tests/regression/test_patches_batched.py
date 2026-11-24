"""
Many gridcells and patches in one run.

A domain of two gridcells and three patches, every per-patch field different:

- ``SITEA`` (CHATS7's location and soil) with two patches: the CHATS7 walnut (weight 0.6) and
  a needleleaf evergreen tree (weight 0.4) with its own canopy height, root biomass, leaf and
  stem profiles and explicit layer counts;
- ``SITEB`` (another latitude, longitude, bedrock depth, soil colour and texture) with one
  walnut patch with explicit layer counts.

All three read CHATS7's forcing files. One run of the domain (the step batched over the
patches) is compared with each patch run alone (a one-patch tower):

- every patch's output equals its run alone (round-off: separately compiled programs);
- the gridcell outputs are the area-weighted sums of the patches, for the mean and for the
  maximum (accumulated at every step inside the run, so the hourly maximum of the gridcell is
  the maximum of the weighted sum, not the weighted sum of the maxima);
- listing the gridcells and patches in another order reorders the outputs only.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]
VARS = ["latent_heat_flux", "sensible_heat_flux", "gpp", "net_radiation"]


def _towers() -> dict:
    import sys

    sys.path.insert(0, str(ROOT))
    from tests.experiment_tools import chats7_tower

    chats = chats7_tower()
    walnut = dict(chats["patches"][0])
    needle = {
        "pft": "needleleaf_evergreen_temperate_tree",
        "weight": 0.4,
        "canopy_height": 15.0,
        "root_biomass": 800.0,
        "area_profiles": {"leaf_beta": [3.0, 1.5], "stem_beta": [2.0, 1.5]},
        "canopy_layers": {"within": 8, "above": 6},
    }
    a = copy.deepcopy(chats)
    a["patches"] = [dict(walnut, weight=0.6), needle]
    b = copy.deepcopy(chats)
    b.update(latitude=45.0, longitude=-100.0)
    b["soil"].update(
        bedrock_depth=1.0,
        color_class=5,
        texture={"given_as": "percentages", "sand_percent": 40.0, "clay_percent": 20.0},
    )
    b["patches"] = [dict(walnut, weight=1.0, canopy_layers={"within": 10, "above": 8})]
    return {"SITEA": a, "SITEB": b}


def _execute(tmp: Path, name: str, towers: dict, order: list):
    """Run the domain ``order`` (tower names in ``towers``) for one day in ``tmp/name``."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from clm_ml.experiment import load_experiment
    from tests.experiment_tools import make_experiment

    work = tmp / name
    output = {
        "directory": str(work / "out"),
        "streams": [
            {"name": "step", "frequency": "step", "statistic": "mean", "variables": VARS,
             "level": ["patch", "gridcell"]},
            {"name": "hmax", "frequency": "hourly", "statistic": "max",
             "variables": ["sensible_heat_flux"], "level": ["gridcell"]},
        ],
    }  # fmt: skip
    make_experiment(work, namelist=ROOT / "data" / "namelists" / "nl.CHATS7.1day",
                    domain=lambda d: d.update(towers={t: towers[t] for t in order}),
                    run={"output": output})  # fmt: skip
    return load_experiment(work).execute(write=False, quiet=True)


def _close(actual, expected, name):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, name
    scale = float(np.max(np.abs(expected)))
    assert np.max(np.abs(actual - expected)) <= 1e-9 * scale + 1e-12, name


def test_patches_batched(tmp_path):
    towers = _towers()
    singles = {}  # each patch alone: a one-patch tower
    for g, patches in (("SITEA", [0, 1]), ("SITEB", [0])):
        for k in patches:
            t = copy.deepcopy(towers[g])
            t["patches"] = [dict(t["patches"][k], weight=1.0)]
            singles[f"{g}{k}"] = t
    jobs = {
        "batch": (towers, ["SITEA", "SITEB"]),
        "permuted": (
            {"SITEB": towers["SITEB"],
             "SITEA": dict(towers["SITEA"], patches=towers["SITEA"]["patches"][::-1])},
            ["SITEB", "SITEA"],
        ),
        **{name: ({name: t}, [name]) for name, t in singles.items()},
    }  # fmt: skip
    # one after the other: runs are in this process, and netCDF is not
    # thread-safe
    out = {k: _execute(tmp_path, k, *v) for k, v in jobs.items()}
    shutil.rmtree(tmp_path, ignore_errors=True)

    batch = out["batch"]["step"]
    order = ["SITEA0", "SITEA1", "SITEB0"]  # the batch's patches, in domain order
    assert list(batch.patch_gridcell) == [0, 0, 1]
    np.testing.assert_allclose(batch.patch_weight, [0.6, 0.4, 1.0])
    for name in VARS:
        # every patch equals its run alone
        for p, single in enumerate(order):
            _close(batch.data[name][:, p], out[single]["step"].data[name][:, 0], f"{name} {p}")
        # the patches differ (each patch's own data take effect)
        assert not np.allclose(batch.data[name][:, 0], batch.data[name][:, 1])
        assert not np.allclose(batch.data[name][:, 0], batch.data[name][:, 2])
        # gridcell means: the area-weighted sums of the patches
        alone = {k: out[k]["step"].data[name][:, 0] for k in order}
        grid = batch.gridcell_mean()[name]
        _close(grid[:, 0], 0.6 * alone["SITEA0"] + 0.4 * alone["SITEA1"], f"{name} SITEA")
        _close(grid[:, 1], alone["SITEB0"], f"{name} SITEB")

    # hourly maximum of the gridcell: max over the hour's steps of the weighted sum
    sh = {k: out[k]["step"].data["sensible_heat_flux"][:, 0] for k in order}
    site_a = 0.6 * sh["SITEA0"] + 0.4 * sh["SITEA1"]  # per step
    hmax = out["batch"]["hmax"].gridcell_mean()["sensible_heat_flux"]
    expected = site_a.reshape(-1, 2).max(axis=1)  # 30-min steps from 00:00: two per hour
    _close(hmax[:, 0], expected, "hourly max SITEA")
    # (not the weighted sum of the patches' maxima, which differs when they peak apart)

    # another order of gridcells and patches reorders the outputs only
    perm = out["permuted"]["step"]
    for name in VARS:
        np.testing.assert_array_equal(perm.data[name][:, [2, 1, 0]], batch.data[name], name)
        _close(perm.gridcell_mean()[name][:, [1, 0]], batch.gridcell_mean()[name], name)
