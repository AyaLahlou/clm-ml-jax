"""
A configurable model step with interpolated forcing, end to end: CHATS7 forced
by 6-hourly means of its own half-hourly data, run at a 30-minute model step, against the
normal half-hourly-forced run. The forcing is interpolated back to 30 minutes
(clm_ml.forcing_interpolation), so the daily incoming shortwave is conserved; the fluxes differ
because variability within 6 hours (clouds, the exact timing of temperature changes) is lost.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]
TOWER_FILE = ROOT / "data" / "input_files" / "tower-forcing" / "CHATS7" / "2007-05.nc"
DAYS = 3


def _six_hourly_file(path: Path) -> None:
    import netCDF4 as nc

    with nc.Dataset(TOWER_FILE) as src, nc.Dataset(path, "w") as dst:
        n = src.dimensions["time"].size // 12
        for name, dim in src.dimensions.items():
            dst.createDimension(name, n if name == "time" else dim.size)
        for name, var in src.variables.items():
            out = dst.createVariable(name, var.dtype, var.dimensions)
            out.setncatts({a: var.getncattr(a) for a in var.ncattrs()})
            values = var[:]
            if name == "time":
                values = values[::12]  # each record starts its 6-hour interval
            elif "time" in var.dimensions:
                values = values.reshape((n, 12) + values.shape[1:]).mean(axis=1)
            out[:] = values


def _run(workdir: Path, six_hourly: bool):
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from clm_ml.experiment import load_experiment
    from tests.experiment_tools import make_experiment

    run = {"time": {"start": "2007-05-01T00:00", "length": f"{DAYS} days"}}
    run["output"] = {
        "directory": str(workdir / "out"),
        "streams": [{"name": "daily", "frequency": "daily", "statistic": "mean",
                     "variables": ["latent_heat_flux", "sensible_heat_flux", "gpp",
                                   "incident_visible_radiation", "net_radiation"]}],
    }  # fmt: skip
    if six_hourly:
        _six_hourly_file(workdir / "six_hourly.nc")
        # the forcing data interval (the model step stays 1800 s, model.yaml)
        run["forcing"] = {"meteorology": {"files": [str(workdir / "six_hourly.nc")],
                                          "interval_seconds": 21600}}  # fmt: skip
    make_experiment(workdir, namelist=ROOT / "data" / "namelists" / "nl.CHATS7.1day", run=run)
    return load_experiment(workdir).execute(quiet=True)["daily"]


def test_six_hourly_forcing_at_a_half_hourly_step(tmp_path):
    (tmp_path / "half").mkdir()
    (tmp_path / "six").mkdir()
    half = _run(tmp_path / "half", six_hourly=False)
    six = _run(tmp_path / "six", six_hourly=True)
    assert half.time.shape == six.time.shape == (DAYS,)
    # Daily incoming shortwave is conserved by the coszen interpolation (the visible part is
    # a fixed fraction of the total in TowerMetSolarRad only approximately, hence 1 %).
    np.testing.assert_allclose(
        six.incident_visible_radiation, half.incident_visible_radiation, rtol=0.01
    )
    # Daily means stay close; loose bounds, since sub-6-hour variability is not recoverable.
    # Energy fluxes: within 10 % of the daily net radiation (sensible heat, the small
    # residual of the energy balance, moves most: up to ~11 W m-2). GPP: within 15 %.
    rnet = np.abs(half.data["net_radiation"][:, 0]).max()
    bounds = {
        "net_radiation": 0.1 * rnet,
        "latent_heat_flux": 0.1 * rnet,
        "sensible_heat_flux": 0.1 * rnet,
        "gpp": 0.15 * np.abs(half.data["gpp"][:, 0]).max(),
    }
    report = []
    for name, bound in bounds.items():
        a, b = half.data[name][:, 0], six.data[name][:, 0]
        report.append(f"{name}: half-hourly {np.round(a, 2)} vs 6-hourly {np.round(b, 2)}")
        assert np.all(np.abs(b - a) <= bound), "\n".join(report)
    print("\n".join(report))
