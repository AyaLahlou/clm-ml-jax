"""Prescribed land data from daily/monthly files: linear interpolation between
interval midpoints (irregular intervals, held at the ends) and the windowed file reader."""

import datetime as dt

import numpy as np
import pytest

from clm_ml.forcing_interpolation import linear_between_midpoints, read_land_data

MONTHS = [dt.datetime(2007, m, 1) for m in (4, 5, 6)]  # 30, 31, 30 days


def test_monthly_values_at_midpoints_and_between():
    values = np.array([1.0, 2.0, 3.0])
    mid_may = dt.datetime(2007, 5, 16, 12)  # May's midpoint (31 days)
    half = 1800
    at = [mid_may - dt.timedelta(seconds=half / 2), dt.datetime(2007, 5, 1)]
    out = linear_between_midpoints(MONTHS, values, at, half)
    assert out[0] == pytest.approx(2.0)  # the step's midpoint is May's midpoint
    # 1 May 00:15: between April's midpoint (16 Apr 00:00) and May's (16 May 12:00)
    t = (dt.datetime(2007, 5, 1, 0, 15) - dt.datetime(2007, 4, 16)).total_seconds()
    span = (mid_may - dt.datetime(2007, 4, 16)).total_seconds()
    assert out[1] == pytest.approx(1.0 + t / span)


def test_last_interval_end_sets_its_midpoint():
    # a window ending at May: May's midpoint needs June 1 (31 days), not April's length
    at = [dt.datetime(2007, 5, 16, 12) - dt.timedelta(seconds=900)]
    out = linear_between_midpoints(MONTHS[:2], np.array([1.0, 2.0]), at, 1800, MONTHS[2])
    assert out[0] == pytest.approx(2.0)


def test_held_beyond_the_ends_and_extra_axes_kept():
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])  # (time, level)
    at = [dt.datetime(2007, 3, 1), dt.datetime(2007, 6, 29)]
    out = linear_between_midpoints(MONTHS, values, at, 1800)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[0], [1.0, 10.0])
    np.testing.assert_allclose(out[1], [3.0, 30.0])  # June's interval assumed 31 days


def test_rejects_unordered_records():
    with pytest.raises(ValueError, match="increase"):
        linear_between_midpoints(MONTHS[::-1], np.zeros(3), MONTHS[:1], 1800)


def test_reader_returns_only_the_window(tmp_path):
    import netCDF4 as nc

    path = tmp_path / "daily.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", 365)
        ds.createDimension("lndgrid", 1)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "days since 2007-01-01 00:00:00"
        t[:] = np.arange(365)
        ds.createVariable("LAI", "f8", ("time", "lndgrid"))[:] = np.arange(365.0)[:, None]
    starts, values, end = read_land_data(
        str(path), ["LAI"], dt.datetime(2007, 5, 1), dt.datetime(2007, 5, 3)
    )
    assert starts[0] <= dt.datetime(2007, 4, 30) and starts[-1] > dt.datetime(2007, 5, 3)
    assert end == starts[-1] + dt.timedelta(days=1)
    assert len(starts) <= 6 and values["LAI"].shape == (len(starts),)
    at = [dt.datetime(2007, 5, 1, 12) - dt.timedelta(minutes=15)]
    # daily midpoint of 1 May (day 120) is 12:00 -> value 120
    assert linear_between_midpoints(starts, values["LAI"], at, 1800, end)[0] == pytest.approx(120.0)
    with pytest.raises(ValueError, match="SAI"):
        read_land_data(str(path), ["SAI"], dt.datetime(2007, 5, 1), dt.datetime(2007, 5, 3))
