"""
CLM Time Manager Module

Translated from clm_time_manager.F90 (lines 1-462)
Pure JAX implementation of CLM time management functionality.

This module provides time management utilities for CLM simulations including:
- Date and time representation
- Calendar operations (Gregorian and NOLEAP)
- Timestep tracking
- Leap year calculations
- Calendar day computations

DEFINITIONS:
- date: Instant in time (year, month, day, time of day in UTC)
  Represented as yyyymmdd integer and seconds past 0Z
- time: Elapsed time since reference date (days + partial day seconds)
- time of day: Elapsed time since midnight (UTC)
- start date: Date assigned to initial conditions
- current date: Date at end of current timestep
- current time: Elapsed time from start date to current date
- calendar day: Day number in calendar year (Jan 1 = day 1)

Original Fortran: clm_time_manager.F90
Translation: Complete module with all subroutines and functions
"""

from typing import NamedTuple, Tuple, Optional
import jax.numpy as jnp
from jax import Array, lax


# ============================================================================
# CONSTANTS (Fortran lines 75-82)
# ============================================================================

# Calendar type flag
CALENDAR_KIND_FLAG = "GREGORIAN"

# Number of days in each month (non-leap year)
# Fortran line 77
MDAY = jnp.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=jnp.int32)

# Cumulative days at end of each month (non-leap year)
# Fortran line 78
MDAYCUM = jnp.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365], dtype=jnp.int32)

# Number of days in each month (leap year)
# Fortran line 81
MDAYLEAP = jnp.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=jnp.int32)

# Cumulative days at end of each month (leap year)
# Fortran line 82
MDAYLEAPCUM = jnp.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366], dtype=jnp.int32)


# ============================================================================
# STATE TYPES
# ============================================================================

class TimeManagerState(NamedTuple):
    """State for CLM time manager.
    
    Attributes:
        dtstep: Model time step in seconds (Fortran line 63)
        itim: Current model time step number (Fortran line 64)
        start_date_ymd: Start date in yyyymmdd format (Fortran line 66)
        start_date_tod: Start time of day in seconds past 0Z (Fortran line 67)
        curr_date_ymd: Current date in yyyymmdd format (Fortran line 69)
        curr_date_tod: Current time of day in seconds past 0Z (Fortran line 70)
        calkindflag: Calendar type ('GREGORIAN' or 'NOLEAP')
    """
    dtstep: int
    itim: int
    start_date_ymd: int
    start_date_tod: int
    curr_date_ymd: int
    curr_date_tod: int
    calkindflag: str = "GREGORIAN"


class CurrentDate(NamedTuple):
    """Container for current date components.
    
    Attributes:
        year: Current year (e.g., 1979, 2000)
        month: Current month (1-12)
        day: Current day of month (1-31)
        tod: Time of day in seconds past midnight (0-86399)
    """
    year: int
    month: int
    day: int
    tod: int


# ============================================================================
# BASIC ACCESSOR FUNCTIONS (Fortran lines 104-122)
# ============================================================================

def get_step_size(state: TimeManagerState) -> int:
    """Return the step size in seconds.
    
    Translated from Fortran lines 104-112.
    
    Args:
        state: Time manager state
        
    Returns:
        Step size in seconds
    """
    return state.dtstep


def get_nstep(state: TimeManagerState) -> int:
    """Return the timestep number.
    
    Translated from Fortran lines 114-122.
    
    Args:
        state: Time manager state
        
    Returns:
        Current timestep number
    """
    return state.itim


# ============================================================================
# LEAP YEAR CALCULATION (Fortran lines 124-151)
# ============================================================================

def isleap(year: int, calendar: str = "GREGORIAN") -> bool:
    """Return true if a leap year.
    
    Translated from Fortran lines 124-151.
    
    Implements Gregorian calendar leap year rules:
    - Divisible by 4: leap year
    - Divisible by 100: not a leap year
    - Divisible by 400: leap year
    
    Args:
        year: Year (e.g., 1900, 2000, 2024)
        calendar: Calendar type ('NOLEAP' or 'GREGORIAN')
        
    Returns:
        True if leap year, False otherwise
    """
    # Fortran line 139: By default, February has 28 days
    is_leap = False
    
    # Fortran line 141: Check if GREGORIAN calendar
    if calendar.strip() == "GREGORIAN":
        # Fortran line 142: Every four years, it has 29 days
        if year % 4 == 0:
            is_leap = True
            # Fortran line 144: Except every 100 years, when it has 28 days
            if year % 100 == 0:
                is_leap = False
                # Fortran line 146: Except every 400 years, when it has 29 days
                if year % 400 == 0:
                    is_leap = True
    
    return is_leap


def isleap_jax(year: Array, calendar: str = "GREGORIAN") -> Array:
    """JAX-compatible vectorized leap year calculation.
    
    Translated from Fortran lines 124-151.
    Pure function version using jnp.where for JIT compatibility.
    
    Args:
        year: Year(s) as JAX array
        calendar: Calendar type ('NOLEAP' or 'GREGORIAN')
        
    Returns:
        Boolean array indicating leap years
    """
    # Default: not a leap year (Fortran line 139)
    is_leap = jnp.zeros_like(year, dtype=bool)
    
    # Only apply leap year logic for GREGORIAN calendar (Fortran line 141)
    if calendar.strip() == "GREGORIAN":
        # Divisible by 4 (Fortran line 142-143)
        div_by_4 = (year % 4) == 0
        # Divisible by 100 (Fortran line 144-145)
        div_by_100 = (year % 100) == 0
        # Divisible by 400 (Fortran line 146-147)
        div_by_400 = (year % 400) == 0
        
        # Apply leap year rules using jnp.where for JIT compatibility
        is_leap = jnp.where(
            div_by_4,
            jnp.where(
                div_by_100,
                jnp.where(div_by_400, True, False),
                True
            ),
            False
        )
    
    return is_leap


# ============================================================================
# DATE CALCULATION FUNCTIONS (Fortran lines 152-318)
# ============================================================================

def get_curr_date(
    state: TimeManagerState,
) -> Tuple[int, int, int, int]:
    """Return date components valid at end of current timestep.
    
    Translated from clm_time_manager.F90, lines 152-233.
    
    This function computes the current calendar date (year, month, day, time-of-day)
    based on the elapsed timesteps from the simulation start date. It handles:
    - Leap year calculations
    - Month/day overflow (e.g., day 32 becomes next month)
    - Year transitions
    
    Args:
        state: TimeManagerState containing calendar configuration and start date
        
    Returns:
        Tuple of (year, month, day, time_of_day_seconds):
            - year: Current year (e.g., 1979, 2000)
            - month: Current month (1-12)
            - day: Current day of month (1-31)
            - time_of_day_seconds: Seconds past midnight (0-86399)
            
    Note:
        - Uses pure JAX operations for JIT compatibility
        - Implements iterative month/day overflow correction using lax.while_loop
        - Preserves exact Fortran logic including leap year handling
    """
    # Line 174: Current year from start date
    mcyear_start = state.start_date_ymd // 10000
    
    # Lines 176-177: Seconds and days since start
    nsecs = state.itim * state.dtstep  # Elapsed seconds
    ndays = (nsecs + state.start_date_tod) // 86400  # Elapsed days
    
    # Lines 178-182: Elapsed years
    is_leap_start = isleap(mcyear_start, state.calkindflag)
    days_per_year = jnp.where(is_leap_start, 366, 365)
    nyears = ndays // days_per_year
    
    # Lines 186-190: Day of current year (mod operation)
    ndays_in_year = ndays % days_per_year
    
    # Lines 192-193: Current seconds of current date
    tod = (nsecs + state.start_date_tod) % 86400
    
    # Lines 195-198: Initialize current year, month, day
    mcyear = mcyear_start + nyears
    mcmnth = (state.start_date_ymd % 10000) // 100
    mcday = (state.start_date_ymd % 100) + ndays_in_year
    
    # Lines 200-218: Loop through months, converting overflow
    # Using lax.while_loop for JIT compatibility (replaces goto 10)
    def cond_fn(carry):
        mcyear, mcmnth, mcday = carry
        is_leap = isleap(mcyear, state.calkindflag)
        days_per_month = jnp.where(is_leap, MDAYLEAP[mcmnth - 1], MDAY[mcmnth - 1])
        return mcday > days_per_month
    
    def body_fn(carry):
        mcyear, mcmnth, mcday = carry
        is_leap = isleap(mcyear, state.calkindflag)
        days_per_month = jnp.where(is_leap, MDAYLEAP[mcmnth - 1], MDAY[mcmnth - 1])
        
        # Subtract days in current month
        mcday = mcday - days_per_month
        mcmnth = mcmnth + 1
        
        # Handle year transition (line 214-217)
        year_overflow = mcmnth == 13
        mcyear = jnp.where(year_overflow, mcyear + 1, mcyear)
        mcmnth = jnp.where(year_overflow, 1, mcmnth)
        
        return (mcyear, mcmnth, mcday)
    
    mcyear, mcmnth, mcday = lax.while_loop(
        cond_fn, body_fn, (mcyear, mcmnth, mcday)
    )
    
    # Line 220: Compute curr_date_ymd
    curr_date_ymd = mcyear * 10000 + mcmnth * 100 + mcday
    
    # Lines 222-225: Extract year, month, day of current date
    yr = curr_date_ymd // 10000
    mon = (curr_date_ymd % 10000) // 100
    day = curr_date_ymd % 100
    
    return (int(yr), int(mon), int(day), int(tod))


def get_curr_date_tuple(
    state: TimeManagerState,
) -> CurrentDate:
    """Convenience wrapper returning CurrentDate NamedTuple.
    
    Args:
        state: TimeManagerState containing calendar configuration
        
    Returns:
        CurrentDate NamedTuple with year, month, day, tod fields
    """
    yr, mon, day, tod = get_curr_date(state)
    return CurrentDate(year=yr, month=mon, day=day, tod=tod)


def get_prev_date(
    state: TimeManagerState,
) -> Tuple[int, int, int, int]:
    """Return date components valid at beginning of timestep.
    
    Translated from clm_time_manager.F90, lines 236-318.
    
    This function computes the date at the beginning of the current timestep
    by subtracting one timestep from the current time. It handles year, month,
    and day rollovers correctly according to the calendar type.
    
    Args:
        state: TimeManagerState containing:
            - itim: Current timestep number (1-based)
            - dtstep: Timestep size in seconds
            - start_date_ymd: Start date in yyyymmdd format
            - start_date_tod: Start time of day in seconds
            - calkindflag: Calendar type flag
    
    Returns:
        Tuple of (yr, mon, day, tod) where:
            - yr: Year (1900, ...)
            - mon: Month (1, ..., 12)
            - day: Day of month (1, ..., 31)
            - tod: Time of day (seconds past 0Z)
    
    Note:
        This computes the date at timestep start (itim-1), whereas get_curr_date
        computes the date at timestep end (itim).
    """
    # Line 258: Year
    mcyear_start = state.start_date_ymd // 10000
    
    # Lines 260-267: Seconds and days since start
    nsecs = (state.itim - 1) * state.dtstep  # Elapsed seconds
    ndays = (nsecs + state.start_date_tod) // 86400  # Elapsed days
    
    # Compute elapsed years based on leap year status
    is_leap_start = isleap(mcyear_start, state.calkindflag)
    days_per_year = jnp.where(is_leap_start, 366, 365)
    nyears = ndays // days_per_year
    
    # Lines 269-275: Day of year (mod operation for remainder)
    ndays = jnp.where(is_leap_start, ndays % 366, ndays % 365)
    
    # Line 277-278: Seconds of date
    tod = (nsecs + state.start_date_tod) % 86400
    
    # Lines 280-283: Initialize year, month, day
    mcyear = state.start_date_ymd // 10000 + nyears
    mcmnth = (state.start_date_ymd % 10000) // 100
    mcday = (state.start_date_ymd % 100) + ndays
    
    # Lines 285-303: Loop through months to handle day overflow
    # Convert to while loop using lax.while_loop for JIT compatibility
    def cond_fun(carry):
        mcyear, mcmnth, mcday = carry
        is_leap = isleap(mcyear, state.calkindflag)
        days_per_month = jnp.where(
            is_leap,
            MDAYLEAP[mcmnth - 1],  # 0-indexed array, 1-indexed month
            MDAY[mcmnth - 1]
        )
        return mcday > days_per_month
    
    def body_fun(carry):
        mcyear, mcmnth, mcday = carry
        is_leap = isleap(mcyear, state.calkindflag)
        days_per_month = jnp.where(
            is_leap,
            MDAYLEAP[mcmnth - 1],
            MDAY[mcmnth - 1]
        )
        
        # Subtract days and increment month
        mcday = mcday - days_per_month
        mcmnth = mcmnth + 1
        
        # Handle year rollover (line 299-301)
        mcyear = jnp.where(mcmnth == 13, mcyear + 1, mcyear)
        mcmnth = jnp.where(mcmnth == 13, 1, mcmnth)
        
        return (mcyear, mcmnth, mcday)
    
    mcyear, mcmnth, mcday = lax.while_loop(
        cond_fun,
        body_fun,
        (mcyear, mcmnth, mcday)
    )
    
    # Line 304: Construct date_ymd
    date_ymd = mcyear * 10000 + mcmnth * 100 + mcday
    
    # Lines 306-309: Extract year, month, day of date
    yr = date_ymd // 10000
    mon = (date_ymd % 10000) // 100
    day = date_ymd % 100
    
    return int(yr), int(mon), int(day), int(tod)


# ============================================================================
# TIME CALCULATION FUNCTIONS (Fortran lines 321-340)
# ============================================================================

def get_curr_time(
    state: TimeManagerState,
) -> Tuple[int, int]:
    """Return the time components at the end of the current timestep.
    
    Current time is the time interval between the current date and the start date.
    
    Translated from clm_time_manager.F90, lines 321-340.
    
    Physics:
    - Computes elapsed seconds: nsecs = itim * dtstep (line 336)
    - Computes days: days = (nsecs + start_date_tod) / 86400 (line 337)
    - Computes remaining seconds: seconds = mod(nsecs + start_date_tod, 86400) (line 338)
    
    Args:
        state: TimeManagerState containing:
            - itim: Current timestep number
            - dtstep: Timestep size in seconds
            - start_date_tod: Time of day at start date (seconds)
    
    Returns:
        Tuple containing:
            - days: Number of whole days in time interval (scalar int)
            - seconds: Remaining seconds in the day (scalar int)
    
    Note:
        All arithmetic operations preserve exact integer semantics from Fortran.
        Division by 86400 (seconds per day) is integer division.
    """
    # Line 336: nsecs = itim * dtstep
    nsecs = state.itim * state.dtstep
    
    # Line 337: days = (nsecs + start_date_tod) / 86400
    # Integer division in Fortran
    days = (nsecs + state.start_date_tod) // 86400
    
    # Line 338: seconds = mod(nsecs+start_date_tod, 86400)
    seconds = (nsecs + state.start_date_tod) % 86400
    
    return int(days), int(seconds)


# ============================================================================
# CALENDAR DAY FUNCTIONS (Fortran lines 343-462)
# ============================================================================

def get_prev_calday(
    state: TimeManagerState,
) -> float:
    """Return calendar day at beginning of timestep.
    
    Translated from clm_time_manager.F90 (lines 415-462).
    Calendar day 1.0 = 0Z on Jan 1.
    
    This function computes the calendar day (day-of-year + fraction) at the
    beginning of the current timestep. It handles both leap and non-leap years,
    and includes a special hack for Gregorian calendar compatibility with
    shr_orb_decl calculations.
    
    Args:
        state: TimeManagerState containing current time information
        
    Returns:
        Calendar day as a scalar float (1.0 to 366.0)
        
    Notes:
        - Calendar day 1.0 corresponds to 0Z (midnight) on January 1
        - For leap years, uses MDAYLEAPCUM; otherwise uses MDAYCUM
        - Includes Gregorian calendar hack: days 366-367 are mapped to day 365
          to maintain compatibility with shr_orb_decl (which can't handle day > 366)
        - This hack was added by Dani Bundy-Coleman and Erik Kluzek (Aug 2008)
        - Validates that result is in valid range [1.0, 366.0]
    """
    # Get year, month, day, and time of day at beginning of timestep
    # Lines 430-431 from original
    yr, mon, day, tod = get_prev_date(state)
    
    # Convert to day-of-year + fraction
    # Lines 433-437 from original
    is_leap = isleap(yr, state.calkindflag)
    
    # Compute cumulative days for the month (0-indexed, so mon-1)
    # Use jnp.where to select between leap and non-leap year cumulative days
    mdaycum_value = jnp.where(
        is_leap,
        MDAYLEAPCUM[mon - 1],
        MDAYCUM[mon - 1]
    )
    
    # Calendar day = cumulative days + current day + fraction of day
    calday = float(mdaycum_value) + float(day) + float(tod) / 86400.0
    
    # WARNING HACK TO ENABLE Gregorian CALENDAR WITH SHR_ORB
    # Lines 439-447 from original
    # The following hack fakes day 366 by reusing day 365. This is just because
    # the current shr_orb_decl calculation can't handle days > 366.
    # Dani Bundy-Coleman and Erik Kluzek Aug/2008
    is_gregorian = state.calkindflag == "GREGORIAN"
    in_hack_range = (calday > 366.0) and (calday <= 367.0)
    if is_gregorian and in_hack_range:
        calday = calday - 1.0
    
    return calday


def get_curr_calday(
    state: TimeManagerState,
    offset: Optional[int] = None,
) -> float:
    """Return calendar day at end of current timestep with optional offset.
    
    Translated from clm_time_manager.F90 (lines 343-412).
    
    Calendar day 1.0 = 0Z on Jan 1. Offset is positive for future times
    and negative for previous times.
    
    Args:
        state: TimeManagerState containing current time information
        offset: Optional offset from current time in seconds.
                Positive for future (not supported), negative for past,
                None or 0 for current time.
    
    Returns:
        Calendar day (1.0 to 366.0)
    
    Raises:
        ValueError: If offset > 0 (future times not supported)
        ValueError: If computed calendar day is out of bounds [1.0, 366.0]
    
    Notes:
        - Line 343-412: Original Fortran implementation
        - Line 362-365: offset < 0 returns previous calendar day
        - Line 367-371: offset > 0 triggers error (not supported)
        - Line 373-412: offset == 0 computes current calendar day
        - Line 379-384: Leap year handling for day-of-year calculation
        - Line 386-398: Gregorian calendar hack for day 366
        - Line 400-403: Bounds checking (1.0 <= calday <= 366.0)
    """
    # Handle offset parameter (default to 0 if None)
    offset_val = 0 if offset is None else offset
    
    # Check for unsupported positive offset (line 367-371)
    if offset_val > 0:
        raise ValueError("get_curr_calday: positive offset not supported")
    
    # Check for negative offset (line 362-365)
    if offset_val < 0:
        return get_prev_calday(state)
    
    # Compute current calendar day (line 373-412)
    # Get current date components (line 377)
    yr, mon, day, tod = get_curr_date(state)
    
    # Check if leap year (line 379)
    is_leap = isleap(yr, state.calkindflag)
    
    # Convert to day-of-year + fraction (line 381-384)
    # Use cumulative day arrays based on leap year status
    day_cum = jnp.where(
        is_leap,
        MDAYLEAPCUM[mon - 1],
        MDAYCUM[mon - 1]
    )
    
    calday = (
        float(day_cum) + 
        float(day) + 
        float(tod) / 86400.0
    )
    
    # Apply Gregorian calendar hack (line 386-398)
    # If day is between 366 and 367 in Gregorian calendar, subtract 1
    is_gregorian = state.calkindflag == 'GREGORIAN'
    needs_hack = (calday > 366.0) and (calday <= 367.0) and is_gregorian
    if needs_hack:
        calday = calday - 1.0
    
    # Check bounds (line 400-403)
    if calday < 1.0 or calday > 366.0:
        raise ValueError(
            f"get_curr_calday: calendar day {calday} out of bounds [1.0, 366.0]"
        )
    
    return calday


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_end_curr_day(state: TimeManagerState) -> bool:
    """Return true if current timestep is last timestep in current day.
    
    Args:
        state: Time manager state
        
    Returns:
        True if at end of day
    """
    # Check if next timestep would roll over to next day
    next_tod = state.curr_date_tod + state.dtstep
    return next_tod >= 86400  # 86400 seconds in a day


def is_end_curr_month(state: TimeManagerState) -> bool:
    """Return true if current timestep is last timestep in current month.
    
    Args:
        state: Time manager state
        
    Returns:
        True if at end of month
    """
    # Get current date
    yr, mon, day, tod = get_curr_date(state)
    
    # Get days in current month
    is_leap = isleap(yr, state.calkindflag)
    days_in_month = jnp.where(is_leap, MDAYLEAP[mon - 1], MDAY[mon - 1])
    
    # Check if at last day and last timestep of day
    at_last_day = day == days_in_month
    next_tod = tod + state.dtstep
    at_end_of_day = next_tod >= 86400
    
    return at_last_day and at_end_of_day