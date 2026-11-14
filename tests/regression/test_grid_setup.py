"""
Grid runs (Phase 6b; setup without prototypes since Phase 7.4): the setup and the runs of a
grid run equal those of tower runs whose files hold the grid run's data, and thin layers keep
their minimum plant area (issue 14).

Needs the Phase 6b input files on NCAR's GLADE file system (skipped elsewhere).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]
HIST = (
    "/glade/derecho/scratch/slevis/archive/wetlands.n04.ctsm5.4.044_f09BgcCrop_flat_174_Hist/"
    "lnd/hist/wetlands.n04.ctsm5.4.044_f09BgcCrop_flat_174_Hist.clm2.h0a"
)
TRENDY = "/glade/campaign/cgd/tss/projects/TRENDY2026/inputs"
SURFACE = (
    "/glade/campaign/cesm/cesmdata/inputdata/lnd/clm2/surfdata_esmf/ctsm5.4.0/"
    "surfdata_0.9x1.25_hist_2000_16pfts_c251022.nc"
)

needs_data = pytest.mark.skipif(
    not (Path(SURFACE).exists() and Path(TRENDY).exists() and Path(f"{HIST}.2000-07.nc").exists()),
    reason="Phase 6b input files (NCAR GLADE) not available",
)


def grid_run_files(tmp: Path, region: dict, start: str, days: float = 1, co2: dict | None = None,
                   output: dict | None = None) -> Path:  # fmt: skip
    """A small grid experiment folder (``tmp/grid``) from the grid_1deg_2000 compset: 10 + 10
    canopy layers, CO2 constant (``co2``, default 383 ppm), hourly latent heat and GPP, Gregorian
    calendar."""
    import sys

    sys.path.insert(0, str(ROOT))
    from tests.experiment_tools import make_experiment

    def run(d):
        # Gregorian (the compset's is noleap, decision 109): the tower runs compared with it
        d["time"].update(start=start, length=f"{days} days", calendar="gregorian")
        d["forcing"]["gridded"]["co2"] = dict({"source": "constant"}, **(co2 or {"ppm": 383}))
        d["output"] = output or {
            "directory": str(tmp / "out"),
            "streams": [{"name": "h", "frequency": "hourly",
                         "variables": ["latent_heat_flux", "gpp"]}],
        }  # fmt: skip

    return make_experiment(
        tmp / "grid", compset="grid_1deg_2000", run=run,
        domain={"grid": {"surface_data": SURFACE, "land_mask": f"{HIST}.2000-01.nc",
                         "region": region}},
        model=lambda d: d["numerics"].update(on_patch_error="stop", max_failed_patch_fraction=0.01),
    )  # fmt: skip


#: A grid run and tower runs of its patches whose files hold the gridded data (meteorology
#: at the model step, vegetation and soil water as the grid run forms them, initial soil
#: temperature), in ``tmp``: the folders ``grid`` and ``tower``
WRITE_RUNS = """
import sys, yaml
from pathlib import Path
ROOT = Path({root!r}); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from tests.regression.test_grid_setup import grid_run_files
from clm_ml.experiment import load_experiment
from tests.regression import grid_tower_files as grid_setup
from clm_ml.setup.grid import grid_inputs
from clm_ml.params import PFT_NAMES
tmp = Path({tmp!r})
path = grid_run_files(tmp, {region}, "2000-07-01T00:00")
run = load_experiment(path)
d = run.domain
inputs = grid_inputs(run)
print("patches", d.n_patches, "gridcells", d.n_gridcells, "C4", int((d.patches.pft_index == 14).sum()))
print("snow", float(np.max(inputs.met_at(list(range(run.n_steps)))["snow"])))
towers = {{}}
for p in range(d.n_patches):
    g = int(d.patches.gridcell_index[p])
    tw, hi = grid_setup.write_tower_files(inputs, g, run.n_steps, tmp / "towers" / str(p),
                                          grid_setup.SOIL_LEVELS, lai_patch=p)
    (tmp / "towers" / f"P{{p}}_met.nc").symlink_to(tw); (tmp / "towers" / f"P{{p}}_hist.nc").symlink_to(hi)
    c = d.columns
    towers[f"P{{p}}"] = {{"latitude": float(d.gridcells.latitude[g]), "longitude": float(d.gridcells.longitude[g]),
        "measurement_height": float(d.patches.measurement_height[p]),
        "soil": {{"texture": {{"given_as": "percentages", "sand_percent": float(c.sand_percent[p]),
                              "clay_percent": float(c.clay_percent[p])}},
                 "organic_matter": float(c.organic_matter[p]), "color_class": int(c.color_class[p]),
                 "bedrock_depth": float(c.bedrock_depth[p])}},
        "patches": [{{"pft": PFT_NAMES[int(d.patches.pft_index[p])], "weight": 1.0,
                     "canopy_height": float(d.patches.canopy_height[p]),
                     "root_biomass": float(d.patches.root_biomass[p])}}]}}
import dataclasses
from clm_ml.domain import DomainFile
from clm_ml.experiment import write_experiment
from clm_ml.io.yaml_schema import from_dict
from clm_ml.run import ForcingSection
domain = from_dict(DomainFile, {{"type": "towers", "towers": towers}}, source="towers", context="towers")
forcing = from_dict(ForcingSection, {{
    "meteorology": {{"files": str(tmp / "towers" / "{{tower}}_met.nc"), "interval_seconds": 1800}},
    "clm_history": {{"file": str(tmp / "towers" / "{{tower}}_hist.nc"), "start": "2000-07-01T00:00"}}}},
    source="forcing", context="towers")
write_experiment(tmp / "tower", dataclasses.replace(run.run_file, forcing=forcing), domain, run.config,
                 run.params, generated="test", params_about="the grid run's")
print("written")
"""

#: Both runs' setups: every patch's constants, initial state and chunk inputs (the
#: meteorology gathered to the patches, the land data as the chunk forms it), saved
SETUP_SCRIPT = """
import sys
from pathlib import Path
ROOT = Path({root!r}); sys.path.insert(0, str(ROOT / "src"))
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from clm_ml.experiment import load_experiment
from clm_ml import driver
tmp = Path({tmp!r})
sim = driver.simulation(load_experiment(tmp / {run_file!r}))  # domain order
xg, xp = sim.read(0, 48)
if sim.land_fn is not None:
    xw, land = xp
    gidx = jax.numpy.asarray(sim.runner_gidx)
    xp = jax.jit(jax.vmap(lambda w: sim.land_fn(w, land, gidx)))(xw)
gi = np.asarray(sim.gridcell_index)
out = {{}}
xg = {{k: v for k, v in xg._asdict().items() if k not in ("calday", "calday_beg")}}
trees = (("xg", xg, gi, 1), ("xp", xp, None, 1), ("constants", sim.constants, None, 0),
         ("state", sim.state0, None, 0))
for name, tree, idx, axis in trees:
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        a = np.asarray(leaf)
        out[name + jax.tree_util.keystr(path)] = np.take(a, idx, axis=1) if idx is not None else a
out["c3"] = np.asarray(sim.c3)
np.savez(tmp / {out!r}, **out)
print("saved")
"""

#: Both runs' outputs
RUN_SCRIPT = """
import sys
from pathlib import Path
ROOT = Path({root!r}); sys.path.insert(0, str(ROOT / "src"))
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from clm_ml.experiment import load_experiment
tmp = Path({tmp!r})
runs, outs = {{}}, {{}}
for name in ("grid", "tower"):
    runs[name] = load_experiment(tmp / name)
    outs[name] = runs[name].execute(write=name == "grid")["h"]  # the grid run's files
    np.savez(tmp / f"{{name}}_out.npz", **{{v: np.asarray(outs[name].data[v]) for v in ("latent_heat_flux", "gpp")}})
# the grid run's gridcell file on the latitude-longitude map (decision 108)
import xarray as xr
(path,) = sorted((tmp / "out").glob("*_gridcell.nc"))
ds, g = xr.load_dataset(path), runs["grid"].domain.gridcells
i, j = g.latlon.index[:, 0], g.latlon.index[:, 1]
np.savez(tmp / "grid_map.npz", mapped=ds["latent_heat_flux"].values[:, i, j],
         mean=outs["grid"].gridcell_mean()["latent_heat_flux"], map_lat=ds["lat"].values[i],
         map_lon=ds["lon"].values[j], lat=g.latitude, lon=g.longitude,
         n_run=int(ds["run_mask"].sum()), n_finite=int(np.isfinite(ds["latent_heat_flux"][0]).sum()),
         area=ds["area"].values[i, j], landfrac=ds["landfrac"].values[i, j],
         vegetated=ds["vegetated_fraction"].values[i, j], domain_area=g.area,
         domain_landfrac=g.land_fraction, domain_vegetated=g.vegetated_fraction)
print("ran")
"""


def _python(script):
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-4000:]
    return run.stdout


def _setups(tmp_path, region):
    import numpy as np

    out = _python(WRITE_RUNS.format(root=str(ROOT), tmp=str(tmp_path), region=region))
    for run_file, name in (("grid", "grid_in.npz"), ("tower", "tower_in.npz")):
        _python(SETUP_SCRIPT.format(root=str(ROOT), tmp=str(tmp_path), run_file=run_file, out=name))
    return out, np.load(tmp_path / "grid_in.npz"), np.load(tmp_path / "tower_in.npz")


@needs_data
def test_grid_setup_equals_tower_setup_of_its_data(tmp_path):
    """The setup of a grid run (``clm_ml.setup.grid``: every patch from its gridcell's gridded
    data) equals the setup of tower runs of its patches whose files hold that data, bit for
    bit: every patch's constants and initial state, its C3/C4 flag, and every chunk input of
    the first 48 steps (the meteorology exactly; vegetation and soil water to the last bit:
    XLA fuses the grid run's interpolation, NumPy does not). A few 1° gridcells (several
    PFTs, C3 and C4) on 1 July 2000 (no snow: tower files have no snowfall), CO2 constant at
    the tower value (383 ppm)."""
    import numpy as np

    out, gin, tin = _setups(tmp_path, {"lat": [38.5, 40.5], "lon": [-90.0, -88.5]})
    assert "snow 0.0" in out, out
    assert sorted(gin.files) == sorted(tin.files)
    differing = [
        k for k in tin.files
        if not (np.allclose(gin[k], tin[k], rtol=1e-15, atol=0.0) if k.startswith("xp")
                else np.array_equal(gin[k], tin[k], equal_nan=True))
    ]  # fmt: skip
    assert not differing, differing


@needs_data
def test_grid_run_equals_tower_runs_of_its_data(tmp_path):
    """A day of a grid run against tower runs of its patches whose files hold the gridded
    data: the outputs are equal to round-off (<= 1e-10 relative; measured <= 2e-13) through
    the night. After sunrise they differ (up to ~10 W m-2 in latent heat by evening): these
    1° patches amplify round-off (a 1e-15 relative change of LAI alone moves a tower run by
    ~8 W m-2 within hours; open issue in docs/design/global_input.md), and the two
    programs (vegetation and soil water formed inside the chunk vs read per step) round
    differently. The grid run's gridcell file has every gridcell at its latitude and
    longitude (decision 108)."""
    import numpy as np

    _python(WRITE_RUNS.format(root=str(ROOT), tmp=str(tmp_path),
                              region={"lat": [39.5, 40.5], "lon": [-90.0, -88.0]}))  # fmt: skip
    _python(RUN_SCRIPT.format(root=str(ROOT), tmp=str(tmp_path)))
    g, t = np.load(tmp_path / "grid_out.npz"), np.load(tmp_path / "tower_out.npz")
    for v in ("latent_heat_flux", "gpp"):
        night = np.max(np.abs(g[v][:12] - t[v][:12])) / np.max(np.abs(t[v]))  # 00-12 UTC
        assert night <= 1e-10, (v, night)
        assert np.max(np.abs(g[v] - t[v])) <= 0.1 * np.max(np.abs(t[v])), v
    # decision 108: the gridcell file is on the latitude-longitude grid of the surface data
    m = np.load(tmp_path / "grid_map.npz")
    np.testing.assert_array_equal(m["mapped"], m["mean"])
    np.testing.assert_array_equal(m["map_lat"], m["lat"])
    np.testing.assert_allclose(np.mod(m["map_lon"], 360.0), m["lon"], rtol=0, atol=1e-9)
    assert m["n_run"] == m["n_finite"] == m["lat"].size
    for name in ("area", "landfrac", "vegetated"):  # for totals: area * landfrac * vegetated
        np.testing.assert_array_equal(m[name], m[f"domain_{name}"])
    assert (m["area"] > 0).all() and (m["landfrac"] > 0).all() and (m["landfrac"] <= 1).all()
    assert (m["vegetated"] > 0).all() and (m["vegetated"] <= 1 + 1e-12).all()


THIN_LAYER_SCRIPT = """
import sys
from pathlib import Path
ROOT = Path({root!r}); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import yaml, numpy as np
from tests.regression.test_grid_setup import grid_run_files
from clm_ml.experiment import load_experiment
tmp = Path({tmp!r})
output = {{"directory": str(tmp / "out"), "streams": [{{"name": "s", "frequency": "step",
    "statistic": "instantaneous", "level": ["patch"], "variables": ["diagnostics.td_profile",
    "diagnostics.dpai_profile", "state.leaf_temperature", "stem_area_added", "stem_area_index",
    "leaf_area_index"]}}]}}
path = grid_run_files(tmp, {{"lat": [41.5, 42.5], "lon": [-109.0, -108.5]}}, "2000-01-01T00:00",
                      days=16, co2={{"ppm": 369}}, output=output)
run = load_experiment(path)
pft = [str(x) for x in run.domain.patches.pft_index]
out = run.execute(write=False)["s"]
d = {{k: np.asarray(v) for k, v in {{**out.data, **out.raw}}.items()}}
td, dpai = d["diagnostics.td_profile"], d["diagnostics.dpai_profile"]
tl = d["state.leaf_temperature"]  # (time, patch, layer, leaf), 0-based
leafy = (dpai > 0) & (dpai < 1e30)
print("max td of leafy layers", float(td[leafy].max()))
print("min dpai of leafy layers", float(dpai[leafy].min()))
print("leaf T range", float(tl[leafy].min()), float(tl[leafy].max()))
print("stem area added max by patch", np.round(d["stem_area_added"].max(axis=0), 4).tolist(), pft)
"""


@needs_data
def test_thin_layers_keep_minimum_plant_area(tmp_path):
    """Issue 14: a stems-only shrub (LAI 0, SAI 0.108) in Wyoming, January 2000, whose bottom
    layer held 0.0013 m2 m-2: CLM-ml's 9-angle diffuse transmittance exceeded 1 there and
    the leaf temperature ran away at a calm sunrise (13 January). With every leafy layer
    kept at minimum_leafy_layer_plant_area (0.01, added as stems) the run completes (any failed check stops
    the run, the canopy energy balance below freezing included: issue 12), every leafy layer transmits less than it receives, and leaf temperatures stay
    physical."""
    out = _python(THIN_LAYER_SCRIPT.format(root=str(ROOT), tmp=str(tmp_path)))
    value = lambda key: [float(x) for x in out.split(key)[1].split("\n")[0].split()]  # noqa: E731
    assert value("max td of leafy layers")[0] < 1.0, out
    assert value("min dpai of leafy layers")[0] >= 0.01 - 1e-12, out
    t_min, t_max = value("leaf T range")
    assert 200.0 < t_min and t_max < 330.0, out
    added = out.split("stem area added max by patch")[1].split("]")[0]
    assert max(float(x) for x in added.strip(" [").split(",")) > 0.0, out
