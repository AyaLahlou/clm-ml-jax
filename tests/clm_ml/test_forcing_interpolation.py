"""Forcing interpolation to the model step: 6-hourly means made from the CHATS7
half-hourly data and interpolated back to 30 minutes conserve daily shortwave and
precipitation, put no sunlight at night, and interpolate temperature linearly."""

import datetime as dt
import math
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)  # before any model module is imported

from clm_ml.forcing_interpolation import interpolate_records, mean_cosz_function  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TOWER_FILE = ROOT / "data" / "input_files" / "tower-forcing" / "CHATS7" / "2007-05.nc"
LAT, LON = math.radians(38.49), math.radians(-121.84)  # CHATS7
START = dt.datetime(2007, 5, 1)
HALF, SIX = 1800, 21600
DAYS = 4


@pytest.fixture(scope="module")
def data():
    from clm_ml.io.tower import read_tower_met_series, tower_first_record

    first = tower_first_record(str(TOWER_FILE), START, HALF)
    raws = read_tower_met_series(str(TOWER_FILE), [first + k for k in range(DAYS * 48)])
    half = {f: np.array([getattr(r, f) for r in raws]) for f in raws[0]._fields}
    six = {f: v.reshape(-1, 12).mean(axis=1) for f, v in half.items()}  # 6-hourly means
    six_starts = [START + dt.timedelta(hours=6 * i) for i in range(DAYS * 4)]
    steps = [START + dt.timedelta(seconds=HALF * k) for k in range(DAYS * 48)]
    cosz = mean_cosz_function(LAT, LON, 2007)
    back = interpolate_records(six, six_starts, SIX, steps, HALF, cosz)
    return half, six, back, steps, cosz


def test_daily_shortwave_and_precipitation_are_conserved(data):
    half, six, back, _, _ = data
    for name in ("fsdsbot", "prect"):
        np.testing.assert_allclose(
            back[name].reshape(DAYS, 48).mean(1), six[name].reshape(DAYS, 4).mean(1), rtol=1e-12
        )
        # and each 6-hour mean, too
        np.testing.assert_allclose(back[name].reshape(-1, 12).mean(1), six[name], rtol=1e-12)


def test_no_sunlight_at_night_and_a_diurnal_cycle(data):
    half, _, back, steps, cosz = data
    night = np.array([cosz(s, HALF) == 0.0 for s in steps])
    assert night.any() and not night.all()
    assert np.all(back["fsdsbot"][night] == 0.0)
    # The shape follows the sun: it matches the original half-hourly data much better than
    # holding each 6-hour mean constant (0.92 vs 0.79; clouds within an interval cannot be
    # recovered from 6-hourly data).
    observed = np.maximum(half["fsdsbot"], 0)
    r_coszen = np.corrcoef(back["fsdsbot"], observed)[0, 1]
    r_constant = np.corrcoef(np.repeat(data[1]["fsdsbot"], 12), observed)[0, 1]
    assert r_coszen > 0.9 and r_coszen > r_constant + 0.1


def test_temperature_is_linear_between_record_midpoints(data):
    _, six, back, _, _ = data
    t = back["tbot"]
    # step 6 of each 6-hour interval ends at the interval midpoint +15 min; between two
    # midpoints the values are on a straight line
    mids = np.arange(DAYS * 4) * 12 + 5.5  # record midpoints in step-index units
    k = np.arange(len(t))
    np.testing.assert_allclose(t, np.interp(k, mids, six["tbot"]), rtol=1e-12)
