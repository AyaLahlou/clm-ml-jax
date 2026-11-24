"""The model calendar: exact date arithmetic in the proleptic Gregorian or the
365-day (noleap) calendar, calendar days, and each model step's time.

Calendars are named as in CLM (``GREGORIAN``, ``NOLEAP``; run.yaml ``time.calendar``:
``gregorian``, ``noleap``, :data:`CALENDARS`).

Fortran reference: CLM-ml v2 ``clm_time_manager.F90`` (``isleap``, ``get_curr_date``,
``get_curr_calday``), with exact calendar arithmetic as CTSM's ESMF clock.
"""

from __future__ import annotations

import datetime
from typing import NamedTuple

#: run.yaml time.calendar -> calendar name
CALENDARS = {"gregorian": "GREGORIAN", "noleap": "NOLEAP"}

_MDAYCUM = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)
_MDAYLEAPCUM = (0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366)


def isleap(year: int, calendar: str) -> bool:
    """Whether ``year`` is a leap year of ``calendar`` (never in ``NOLEAP``)."""
    if calendar.strip() != "GREGORIAN":
        return False
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _noleap_seconds(t: datetime.datetime) -> int:
    """Seconds of date-time ``t`` since 0001-01-01 in the 365-day calendar."""
    if t.month == 2 and t.day == 29:
        raise ValueError(f"{t:%Y-%m-%d} does not exist in the noleap calendar")
    days = (t.year - 1) * 365 + _MDAYCUM[t.month - 1] + t.day - 1
    return days * 86400 + t.hour * 3600 + t.minute * 60 + t.second


def add_seconds(t: datetime.datetime, seconds: float, calendar: str = "GREGORIAN"):
    """Date-time ``seconds`` after ``t`` in ``calendar``: exact calendar arithmetic, as
    CTSM's ESMF clock (proleptic Gregorian, or 365-day years with the months of a normal
    year). Noleap dates are ordinary dates that skip 29 February."""
    if calendar.strip() == "GREGORIAN":
        return t + datetime.timedelta(seconds=seconds)
    total = _noleap_seconds(t) + seconds
    days, sec = divmod(total, 86400)
    days, sec = int(days), float(sec)
    years, doy = divmod(days, 365)
    month = next(m for m in range(12, 0, -1) if _MDAYCUM[m - 1] <= doy)
    out = datetime.datetime(years + 1, month, doy - _MDAYCUM[month - 1] + 1)
    return out + datetime.timedelta(seconds=sec)


def seconds_between(a: datetime.datetime, b: datetime.datetime, calendar: str = "GREGORIAN"):
    """Seconds from ``a`` to ``b`` in ``calendar``."""
    if calendar.strip() == "GREGORIAN":
        return (b - a).total_seconds()
    return float(_noleap_seconds(b) - _noleap_seconds(a)) + ((b.microsecond - a.microsecond) * 1e-6)


def calday_of(yr: int, mon: int, day: int, tod: int, calendar: str = "GREGORIAN") -> float:
    """Calendar day (1.000 = 0Z 1 January) of a date, with the Gregorian hack of Fortran
    ``get_curr_calday``: ``366 < calday <= 367`` becomes ``calday - 1`` (calendar days in
    [1, 366] for the solar declination)."""
    cum = _MDAYLEAPCUM if isleap(yr, calendar) else _MDAYCUM
    calday = float(cum[mon - 1]) + float(day) + float(tod) / 86400.0
    if 366.0 < calday <= 367.0 and calendar.strip() == "GREGORIAN":
        calday -= 1.0
    if calday < 1.0 or calday > 366.0:
        raise ValueError(f"calendar day {calday} of {yr}-{mon}-{day} out of bounds")
    return calday


class StepTime(NamedTuple):
    """Time information for one model time step."""

    nstep: int  # 1-based time step number
    dtime: int  # time step length (s)
    year: int  # date at the end of the step
    month: int
    day: int
    tod: int  # seconds past 0Z at the end of the step
    calday: float  # calendar day at the end of the step
    calday_beg: float  # calendar day at the beginning of the step


def date_after(start: datetime.datetime, nsecs: float, calendar: str) -> tuple:
    """``(yr, mon, day, tod)`` ``nsecs`` seconds after ``start``."""
    t = add_seconds(start, nsecs, calendar)
    return t.year, t.month, t.day, t.hour * 3600 + t.minute * 60 + t.second


def step_time(start: datetime.datetime, dtime: int, nstep: int, calendar: str) -> StepTime:
    """Time information for step ``nstep`` (1-based) of a run starting at ``start`` with
    steps of ``dtime`` seconds."""
    yr, mon, day, tod = date_after(start, nstep * dtime, calendar)
    calday = calday_of(yr, mon, day, tod, calendar)
    calday_beg = calday_of(*date_after(start, (nstep - 1) * dtime, calendar), calendar)
    return StepTime(nstep, dtime, yr, mon, day, tod, calday, calday_beg)
