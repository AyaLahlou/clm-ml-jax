"""Model calendars: exact Gregorian and noleap date arithmetic, as CTSM's clock."""

import datetime as dt

import pytest

from clm_ml.calendar import add_seconds, seconds_between, step_time
from clm_ml.grid import satellite_phenology_date


@pytest.mark.parametrize("start", [19010101, 20000101, 20070501, 19991231])
def test_gregorian_dates_are_exact_over_150_years(start):
    s = dt.datetime(start // 10000, start // 100 % 100, start % 100)
    for k in (0, 47, 48 * 30, 48 * 366, 48 * 365 * 4 + 7, int(150 * 365.2425) * 48 + 13):
        t = step_time(s, 1800, k, "GREGORIAN")
        real = s + dt.timedelta(seconds=k * 1800)
        assert (t.year, t.month, t.day) == (real.year, real.month, real.day)
        assert t.tod == real.hour * 3600 + real.minute * 60


def test_noleap_years_have_365_days_and_no_29_february():
    a = dt.datetime(2000, 2, 28, 23, 30)
    assert add_seconds(a, 1800, "NOLEAP") == dt.datetime(2000, 3, 1)
    assert (
        seconds_between(dt.datetime(2000, 1, 1), dt.datetime(2001, 1, 1), "NOLEAP") == 365 * 86400
    )
    assert add_seconds(dt.datetime(2000, 1, 1), 150 * 365 * 86400, "NOLEAP") == dt.datetime(
        2150, 1, 1
    )
    t = step_time(dt.datetime(2000, 2, 28), 1800, 48, "NOLEAP")  # the step ending 1 March
    assert (t.month, t.day, t.tod) == (3, 1, 0)
    assert t.calday == 60.0  # no leap day in the calendar day either
    with pytest.raises(ValueError, match="noleap"):
        seconds_between(dt.datetime(2000, 2, 29), dt.datetime(2000, 3, 1), "NOLEAP")


def test_vegetation_weights_use_the_end_of_the_step():
    # CTSM: the leaf area of a step comes from the weights at the step's end
    s = dt.datetime(2000, 2, 28, 23, 30)
    assert satellite_phenology_date(s, 1800) == dt.datetime(2000, 2, 29)
    assert satellite_phenology_date(s, 1800, "NOLEAP") == dt.datetime(2000, 3, 1)
