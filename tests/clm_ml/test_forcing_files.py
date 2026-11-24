"""Forcing split in time over several files: the CHATS7 month split into daily
tower files reads the same records as the single file; gaps are rejected; land-data files
split in time read as one series."""

import datetime as dt
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)  # before any model module is imported

from clm_ml.forcing_interpolation import read_land_data  # noqa: E402
from clm_ml.io.tower import (  # noqa: E402
    TowerFiles,
    TowerMetError,
    read_tower_met_series,
    tower_first_record,
)

ROOT = Path(__file__).resolve().parents[2]
TOWER_FILE = ROOT / "data" / "input_files" / "tower-forcing" / "CHATS7" / "2007-05.nc"


def _split(path: Path, out_dir: Path, records_per_file: int, skip=()) -> list:
    """Write the tower file as consecutive files of ``records_per_file`` records."""
    import netCDF4 as nc

    paths = []
    with nc.Dataset(path) as src:
        n = src.dimensions["time"].size
        for i, lo in enumerate(range(0, n, records_per_file)):
            if i in skip:
                continue
            hi = min(lo + records_per_file, n)
            out = out_dir / f"part{i:03d}.nc"
            with nc.Dataset(out, "w") as dst:
                for name, dim in src.dimensions.items():
                    dst.createDimension(name, hi - lo if name == "time" else dim.size)
                for name, var in src.variables.items():
                    v = dst.createVariable(name, var.dtype, var.dimensions)
                    v.setncatts({a: var.getncattr(a) for a in var.ncattrs()})
                    v[:] = var[lo:hi] if "time" in var.dimensions else var[:]
            paths.append(str(out))
    return paths


def test_split_files_read_as_the_single_file(tmp_path):
    paths = _split(TOWER_FILE, tmp_path, 48)
    series = TowerFiles.open(paths, 1800)
    start = dt.datetime(2007, 5, 3, 12)
    first = tower_first_record(str(TOWER_FILE), start, 1800)
    assert series.first_record(start) == first
    records = list(range(first - 50, first + 60))  # spans three files
    assert series.read(records) == read_tower_met_series(str(TOWER_FILE), records)


def test_gap_between_files_is_rejected(tmp_path):
    paths = _split(TOWER_FILE, tmp_path, 48, skip=(2,))
    with pytest.raises(TowerMetError, match="without gaps"):
        TowerFiles.open(paths, 1800)


def _land_file(path: Path, days, values) -> None:
    import netCDF4 as nc

    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", len(days))
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "days since 2007-01-01 00:00:00"
        t[:] = days
        ds.createVariable("LAI", "f8", ("time",))[:] = values


def test_land_data_split_in_time(tmp_path):
    _land_file(tmp_path / "a.nc", np.arange(0, 120), np.arange(0.0, 120))
    _land_file(tmp_path / "b.nc", np.arange(120, 365), np.arange(120.0, 365))
    files = [str(tmp_path / "b.nc"), str(tmp_path / "a.nc")]  # order does not matter
    starts, values, end = read_land_data(
        files, ["LAI"], dt.datetime(2007, 4, 29), dt.datetime(2007, 5, 3)
    )
    days = [(t - dt.datetime(2007, 1, 1)).days for t in starts]
    assert days == list(range(days[0], days[0] + len(days)))  # contiguous across the files
    np.testing.assert_array_equal(values["LAI"], days)
    _land_file(tmp_path / "c.nc", [100.0], [0.0])  # repeats a record of a.nc
    with pytest.raises(ValueError, match="repeat"):
        read_land_data(files + [str(tmp_path / "c.nc")], ["LAI"], starts[0], starts[-1])
