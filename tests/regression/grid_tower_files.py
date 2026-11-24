"""Tower-format files holding one gridcell's gridded data (test helper): the grid run and
tower runs of these files must agree
(``test_grid_setup.py``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Soil layer counts (nlevsoi, nlevgrnd) of the CLM soil layer structure
SOIL_LEVELS = (20, 25)


def write_tower_files(
    inputs, g: int, n_steps: int, out_dir: Path, levels: tuple, lai_patch=None
) -> tuple:
    """A tower forcing file and a CLM history file (30-min records, float64) holding gridcell
    ``g``'s gridded data for run steps ``0 ... n_steps - 1``: the derived meteorology at each
    step (``TBOT``, ``PSRF``, ``QBOT``, ``WIND``, ``FLDS``, ``FSDS_DIRECT``/``FSDS_DIFFUSE``,
    ``PRECTmms`` = rain), and ``ELAI``/``ESAI`` (of patch ``lai_patch``, default the
    gridcell's first), ``H2OSOI`` and ``TSOI`` as the grid run sees them at each step.
    The tower path then reads exactly the grid run's values (CO2 and snow excepted: tower
    files have neither; tower runs use 383 ppm and no snow)."""
    import netCDF4 as nc

    nlevsoi, nlevgrnd = levels
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    met = inputs.met_at(list(range(n_steps)))
    d = inputs.domain
    p = int(np.flatnonzero(d.patches.gridcell_index == g)[0]) if lai_patch is None else lai_patch
    dt_days = inputs.step_seconds / 86400.0
    origin = f"{inputs.start:%Y-%m-%d %H:%M:%S}"
    tower = out_dir / f"tower_g{g}.nc"
    with nc.Dataset(tower, "w") as ds:
        ds.createDimension("time", n_steps)
        ds.createDimension("lat", 1)
        ds.createDimension("lon", 1)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = f"days since {origin}"
        t[:] = np.arange(n_steps) * dt_days
        for name, value in (("LATIXY", d.gridcells.latitude[g]),
                            ("LONGXY", d.gridcells.longitude[g])):  # fmt: skip
            ds.createVariable(name, "f8", ("lat", "lon"))[:] = value
        columns = {
            "TBOT": met["tref"], "PSRF": met["pref"], "QBOT": met["qref"], "WIND": met["uref"],
            "FLDS": met["lwsky"], "FSDS_DIRECT": met["solad"][0] * 2.0,
            "FSDS_DIFFUSE": met["solai"][0] * 2.0, "PRECTmms": met["rain"],
        }  # fmt: skip
        for name, v in columns.items():
            ds.createVariable(name, "f8", ("time", "lat", "lon"))[:] = v[:, g][:, None, None]
    # vegetation and soil water at every step, as the compiled chunk forms them
    m1, m2, w1, w2 = inputs.vegetation_weights(0, n_steps)
    h, lb, f1, f2 = inputs.soil_window(0, n_steps, n_records=None)
    from clm_ml.grid_forcing import exposed_area

    elai, esai, h2o = [], [], []
    for k in range(n_steps):  # as land_record
        lai = w1[k] * inputs.lai[m1[k], p] + w2[k] * inputs.lai[m2[k], p]
        sai = w1[k] * inputs.sai[m1[k], p] + w2[k] * inputs.sai[m2[k], p]
        e_l, e_s = exposed_area(np.array(lai), np.array(sai))
        elai.append(float(e_l))
        esai.append(float(e_s))
        h2o.append(h[lb[k], g] * f1[k] + h[lb[k] + 1, g] * f2[k])
    tsoi = inputs.soil_at_first_step("TSOI")[g]
    hist = out_dir / f"history_g{g}_p{p}.nc"
    with nc.Dataset(hist, "w") as ds:
        ds.createDimension("time", n_steps)
        ds.createDimension("lndgrid", 1)
        ds.createDimension("levsoi", nlevsoi)
        ds.createDimension("levgrnd", nlevgrnd)
        ds.createDimension("hist_interval", 2)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = f"days since {origin}"
        t[:] = (np.arange(n_steps) + 1) * dt_days  # each slice ends at its step's end
        ds.createVariable("time_bounds", "f8", ("time", "hist_interval"))[:] = np.stack(
            [np.arange(n_steps) * dt_days, (np.arange(n_steps) + 1) * dt_days], 1
        )
        ds.createVariable("levgrnd", "f8", ("levgrnd",))[:] = np.arange(nlevgrnd)
        ds.createVariable("ELAI", "f8", ("time", "lndgrid"))[:] = np.array(elai)[:, None]
        ds.createVariable("ESAI", "f8", ("time", "lndgrid"))[:] = np.array(esai)[:, None]
        ds.createVariable("H2OSOI", "f8", ("time", "levsoi", "lndgrid"))[:] = np.array(h2o)[
            :, :, None
        ]
        ds.createVariable("TSOI", "f8", ("time", "levgrnd", "lndgrid"))[:] = np.repeat(
            tsoi[None, :, None], n_steps, axis=0
        )
    return tower, hist
