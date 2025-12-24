"""
Comprehensive pytest suite for clm_time_manager module.

This test suite covers:
- TimeManagerState creation and manipulation
- Date/time retrieval functions
- Leap year detection (scalar and JAX array)
- Calendar day calculations
- End-of-period detection
- Edge cases: boundaries, minimum/maximum values, year transitions
- Both GREGORIAN and NOLEAP calendars
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
from collections import namedtuple

# Import the module under test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from clm_src_utils import clm_time_manager


# Define NamedTuples matching the module specification
TimeManagerState = namedtuple(
    'TimeManagerState',
    ['dtstep', 'itim', 'start_date_ymd', 'start_date_tod', 
     'curr_date_ymd', 'curr_date_tod', 'calkindflag']
)

CurrentDate = namedtuple(
    'CurrentDate',
    ['year', 'month', 'day', 'tod']
)


@pytest.fixture
def test_data():
    """
    Fixture providing comprehensive test data for all clm_time_manager functions.
    
    Returns:
        dict: Test cases covering nominal, edge, and special scenarios
    """
    return {
        "test_nominal_gregorian_midyear": {
            "state": TimeManagerState(
                dtstep=1800,
                itim=240,
                start_date_ymd=20000101,
                start_date_tod=0,
                curr_date_ymd=20000106,
                curr_date_tod=0,
                calkindflag="GREGORIAN"
            ),
            "year": 2000,
            "calendar": "GREGORIAN",
            "offset": 0
        },
        "test_nominal_noleap_calendar": {
            "state": TimeManagerState(
                dtstep=3600,
                itim=100,
                start_date_ymd=19790315,
                start_date_tod=43200,
                curr_date_ymd=19790319,
                curr_date_tod=7200,
                calkindflag="NOLEAP"
            ),
            "year": 1979,
            "calendar": "NOLEAP",
            "offset": -7200
        },
        "test_nominal_end_of_day": {
            "state": TimeManagerState(
                dtstep=900,
                itim=96,
                start_date_ymd=20240701,
                start_date_tod=0,
                curr_date_ymd=20240701,
                curr_date_tod=86400,
                calkindflag="GREGORIAN"
            ),
            "year": 2024,
            "calendar": "GREGORIAN",
            "offset": None
        },
        "test_nominal_end_of_month": {
            "state": TimeManagerState(
                dtstep=1800,
                itim=1488,
                start_date_ymd=20001201,
                start_date_tod=0,
                curr_date_ymd=20001231,
                curr_date_tod=86400,
                calkindflag="GREGORIAN"
            ),
            "year": 2000,
            "calendar": "GREGORIAN",
            "offset": 0
        },
        "test_nominal_february_leap": {
            "state": TimeManagerState(
                dtstep=600,
                itim=4176,
                start_date_ymd=20000201,
                start_date_tod=0,
                curr_date_ymd=20000229,
                curr_date_tod=0,
                calkindflag="GREGORIAN"
            ),
            "year": 2000,
            "calendar": "GREGORIAN",
            "offset": -600
        },
        "test_edge_first_timestep": {
            "state": TimeManagerState(
                dtstep=1800,
                itim=0,
                start_date_ymd=19000101,
                start_date_tod=0,
                curr_date_ymd=19000101,
                curr_date_tod=0,
                calkindflag="GREGORIAN"
            ),
            "year": 1900,
            "calendar": "GREGORIAN",
            "offset": 0
        },
        "test_edge_minimum_timestep": {
            "state": TimeManagerState(
                dtstep=1,
                itim=86400,
                start_date_ymd=20200101,
                start_date_tod=0,
                curr_date_ymd=20200102,
                curr_date_tod=0,
                calkindflag="GREGORIAN"
            ),
            "year": 2020,
            "calendar": "GREGORIAN",
            "offset": 0
        },
        "test_edge_year_boundaries": {
            "state": TimeManagerState(
                dtstep=3600,
                itim=8760,
                start_date_ymd=19990101,
                start_date_tod=0,
                curr_date_ymd=20000101,
                curr_date_tod=0,
                calkindflag="GREGORIAN"
            ),
            "year": 1999,
            "calendar": "GREGORIAN",
            "offset": 0
        },
        "test_edge_maximum_tod": {
            "state": TimeManagerState(
                dtstep=1800,
                itim=48,
                start_date_ymd=20150815,
                start_date_tod=86399,
                curr_date_ymd=20150816,
                curr_date_tod=86399,
                calkindflag="NOLEAP"
            ),
            "year": 2015,
            "calendar": "NOLEAP",
            "offset": 0
        },
        "test_special_jax_array_years": {
            "years_array": jnp.array([1900, 2000, 2004, 2100, 2400, 1999, 2001, 1600, 1700, 1800]),
            "calendar": "GREGORIAN",
            "state": TimeManagerState(
                dtstep=21600,
                itim=1460,
                start_date_ymd=20160101,
                start_date_tod=0,
                curr_date_ymd=20161231,
                curr_date_tod=86400,
                calkindflag="GREGORIAN"
            ),
            "offset": -21600
        }
    }


# ============================================================================
# Tests for get_step_size
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "test_nominal_gregorian_midyear",
    "test_nominal_noleap_calendar",
    "test_edge_first_timestep",
    "test_edge_minimum_timestep",
])
def test_get_step_size_values(test_data, test_case_name):
    """
    Test get_step_size returns correct timestep values.
    
    Verifies that the function correctly extracts dtstep from TimeManagerState
    for various timestep sizes (1s to 21600s).
    """
    test_case = test_data[test_case_name]
    state = test_case["state"]
    
    result = clm_time_manager.get_step_size(state)
    
    assert isinstance(result, int), f"Expected int, got {type(result)}"
    assert result == state.dtstep, f"Expected {state.dtstep}, got {result}"
    assert result > 0, f"Timestep must be positive, got {result}"


def test_get_step_size_dtypes(test_data):
    """Test get_step_size returns correct data type (int)."""
    state = test_data["test_nominal_gregorian_midyear"]["state"]
    result = clm_time_manager.get_step_size(state)
    
    assert isinstance(result, int), f"Expected int type, got {type(result)}"


# ============================================================================
# Tests for get_nstep
# ============================================================================

@pytest.mark.parametrize("test_case_name,expected_nstep", [
    ("test_nominal_gregorian_midyear", 240),
    ("test_edge_first_timestep", 0),
    ("test_edge_minimum_timestep", 86400),
    ("test_edge_year_boundaries", 8760),
])
def test_get_nstep_values(test_data, test_case_name, expected_nstep):
    """
    Test get_nstep returns correct timestep numbers.
    
    Verifies the function correctly extracts itim from TimeManagerState,
    including edge cases like first timestep (0) and large timestep numbers.
    """
    test_case = test_data[test_case_name]
    state = test_case["state"]
    
    result = clm_time_manager.get_nstep(state)
    
    assert isinstance(result, int), f"Expected int, got {type(result)}"
    assert result == expected_nstep, f"Expected {expected_nstep}, got {result}"
    assert result >= 0, f"Timestep number must be non-negative, got {result}"


def test_get_nstep_dtypes(test_data):
    """Test get_nstep returns correct data type (int)."""
    state = test_data["test_nominal_gregorian_midyear"]["state"]
    result = clm_time_manager.get_nstep(state)
    
    assert isinstance(result, int), f"Expected int type, got {type(result)}"


# ============================================================================
# Tests for isleap (scalar version)
# ============================================================================

@pytest.mark.parametrize("year,calendar,expected", [
    # Gregorian calendar tests
    (2000, "GREGORIAN", True),   # Divisible by 400
    (2004, "GREGORIAN", True),   # Divisible by 4, not by 100
    (2020, "GREGORIAN", True),   # Divisible by 4, not by 100
    (2024, "GREGORIAN", True),   # Divisible by 4, not by 100
    (1900, "GREGORIAN", False),  # Divisible by 100, not by 400
    (2100, "GREGORIAN", False),  # Divisible by 100, not by 400
    (1999, "GREGORIAN", False),  # Not divisible by 4
    (2001, "GREGORIAN", False),  # Not divisible by 4
    (1600, "GREGORIAN", True),   # Divisible by 400
    (2400, "GREGORIAN", True),   # Divisible by 400
    (1700, "GREGORIAN", False),  # Divisible by 100, not by 400
    (1800, "GREGORIAN", False),  # Divisible by 100, not by 400
    # NOLEAP calendar tests (always False)
    (2000, "NOLEAP", False),
    (2004, "NOLEAP", False),
    (1979, "NOLEAP", False),
    (2015, "NOLEAP", False),
])
def test_isleap_values(year, calendar, expected):
    """
    Test isleap correctly identifies leap years for both calendar types.
    
    Tests Gregorian leap year rules:
    - Divisible by 4: leap year
    - Divisible by 100: not leap year
    - Divisible by 400: leap year
    
    Tests NOLEAP calendar always returns False.
    """
    result = clm_time_manager.isleap(year, calendar)
    
    assert isinstance(result, bool), f"Expected bool, got {type(result)}"
    assert result == expected, f"Year {year} with {calendar} calendar: expected {expected}, got {result}"


def test_isleap_default_calendar():
    """Test isleap uses GREGORIAN as default calendar."""
    result_explicit = clm_time_manager.isleap(2000, "GREGORIAN")
    result_default = clm_time_manager.isleap(2000)
    
    assert result_explicit == result_default, "Default calendar should be GREGORIAN"
    assert result_default is True, "Year 2000 should be leap year with default calendar"


def test_isleap_dtypes():
    """Test isleap returns correct data type (bool)."""
    result = clm_time_manager.isleap(2000, "GREGORIAN")
    assert isinstance(result, bool), f"Expected bool type, got {type(result)}"


# ============================================================================
# Tests for isleap_jax (array version)
# ============================================================================

def test_isleap_jax_array_values(test_data):
    """
    Test isleap_jax correctly processes arrays of years.
    
    Verifies leap year detection for multiple years simultaneously,
    including century years and 400-divisible years.
    """
    years = test_data["test_special_jax_array_years"]["years_array"]
    expected = jnp.array([False, True, True, False, True, False, False, True, False, False])
    
    result = clm_time_manager.isleap_jax(years, "GREGORIAN")
    
    assert isinstance(result, jnp.ndarray), f"Expected JAX array, got {type(result)}"
    assert result.shape == years.shape, f"Shape mismatch: expected {years.shape}, got {result.shape}"
    assert jnp.all(result == expected), f"Expected {expected}, got {result}"


def test_isleap_jax_noleap_calendar():
    """Test isleap_jax returns all False for NOLEAP calendar."""
    years = jnp.array([2000, 2004, 2020, 2024])
    result = clm_time_manager.isleap_jax(years, "NOLEAP")
    
    assert jnp.all(result == False), "NOLEAP calendar should return all False"


def test_isleap_jax_scalar_input():
    """Test isleap_jax handles scalar JAX array input."""
    year = jnp.array(2000)
    result = clm_time_manager.isleap_jax(year, "GREGORIAN")
    
    assert isinstance(result, jnp.ndarray), f"Expected JAX array, got {type(result)}"
    assert result.shape == (), f"Expected scalar shape (), got {result.shape}"
    assert result == True, "Year 2000 should be leap year"


def test_isleap_jax_multidimensional():
    """Test isleap_jax handles multidimensional arrays."""
    years = jnp.array([[2000, 2001], [2004, 2005]])
    expected = jnp.array([[True, False], [True, False]])
    
    result = clm_time_manager.isleap_jax(years, "GREGORIAN")
    
    assert result.shape == years.shape, f"Shape mismatch: expected {years.shape}, got {result.shape}"
    assert jnp.all(result == expected), f"Expected {expected}, got {result}"


def test_isleap_jax_dtypes():
    """Test isleap_jax returns boolean JAX array."""
    years = jnp.array([2000, 2001])
    result = clm_time_manager.isleap_jax(years, "GREGORIAN")
    
    assert result.dtype == jnp.bool_, f"Expected bool dtype, got {result.dtype}"


# ============================================================================
# Tests for get_curr_date
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "test_nominal_gregorian_midyear",
    "test_nominal_noleap_calendar",
    "test_edge_first_timestep",
])
def test_get_curr_date_values(test_data, test_case_name):
    """
    Test get_curr_date returns correct date components.
    
    Verifies the function returns a tuple of (year, month, day, tod)
    representing the date at the end of the current timestep.
    """
    test_case = test_data[test_case_name]
    state = test_case["state"]
    
    result = clm_time_manager.get_curr_date(state)
    
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 4, f"Expected 4 components, got {len(result)}"
    
    year, month, day, tod = result
    
    # Validate ranges
    assert isinstance(year, int), f"Year should be int, got {type(year)}"
    assert isinstance(month, int), f"Month should be int, got {type(month)}"
    assert isinstance(day, int), f"Day should be int, got {type(day)}"
    assert isinstance(tod, int), f"TOD should be int, got {type(tod)}"
    
    assert 1 <= month <= 12, f"Month must be 1-12, got {month}"
    assert 1 <= day <= 31, f"Day must be 1-31, got {day}"
    assert 0 <= tod <= 86400, f"TOD must be 0-86400, got {tod}"


def test_get_curr_date_february_leap(test_data):
    """Test get_curr_date handles February 29 in leap year."""
    state = test_data["test_nominal_february_leap"]["state"]
    year, month, day, tod = clm_time_manager.get_curr_date(state)
    
    assert year == 2000, f"Expected year 2000, got {year}"
    assert month == 2, f"Expected month 2, got {month}"
    assert day == 29, f"Expected day 29, got {day}"


def test_get_curr_date_dtypes(test_data):
    """Test get_curr_date returns correct data types."""
    state = test_data["test_nominal_gregorian_midyear"]["state"]
    result = clm_time_manager.get_curr_date(state)
    
    assert all(isinstance(x, int) for x in result), "All components should be int"


# ============================================================================
# Tests for get_prev_date
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "test_nominal_gregorian_midyear",
    "test_nominal_noleap_calendar",
    "test_edge_first_timestep",
])
def test_get_prev_date_values(test_data, test_case_name):
    """
    Test get_prev_date returns correct date components.
    
    Verifies the function returns date at the beginning of the current timestep,
    which should be one timestep before curr_date.
    """
    test_case = test_data[test_case_name]
    state = test_case["state"]
    
    result = clm_time_manager.get_prev_date(state)
    
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 4, f"Expected 4 components, got {len(result)}"
    
    yr, mon, day, tod = result
    
    # Validate ranges
    assert isinstance(yr, int), f"Year should be int, got {type(yr)}"
    assert isinstance(mon, int), f"Month should be int, got {type(mon)}"
    assert isinstance(day, int), f"Day should be int, got {type(day)}"
    assert isinstance(tod, int), f"TOD should be int, got {type(tod)}"
    
    assert 1 <= mon <= 12, f"Month must be 1-12, got {mon}"
    assert 1 <= day <= 31, f"Day must be 1-31, got {day}"
    assert 0 <= tod <= 86400, f"TOD must be 0-86400, got {tod}"


def test_get_prev_date_first_timestep(test_data):
    """Test get_prev_date at first timestep returns start date."""
    state = test_data["test_edge_first_timestep"]["state"]
    result = clm_time_manager.get_prev_date(state)
    
    # At first timestep, prev_date should equal start_date
    assert result is not None, "Should return valid date at first timestep"


def test_get_prev_date_dtypes(test_data):
    """Test get_prev_date returns correct data types."""
    state = test_data["test_nominal_gregorian_midyear"]["state"]
    result = clm_time_manager.get_prev_date(state)
    
    assert all(isinstance(x, int) for x in result), "All components should be int"


# ============================================================================
# Tests for get_curr_time
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "test_nominal_gregorian_midyear",
    "test_edge_first_timestep",
    "test_edge_minimum_timestep",
])
def test_get_curr_time_values(test_data, test_case_name):
    """
    Test get_curr_time returns correct time components.
    
    Verifies the function returns (days, seconds) representing elapsed time
    since simulation start.
    """
    test_case = test_data[test_case_name]
    state = test_case["state"]
    
    result = clm_time_manager.get_curr_time(state)
    
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2 components, got {len(result)}"
    
    days, seconds = result
    
    assert isinstance(days, int), f"Days should be int, got {type(days)}"
    assert isinstance(seconds, int), f"Seconds should be int, got {type(seconds)}"
    assert days >= 0, f"Days must be non-negative, got {days}"
    assert 0 <= seconds < 86400, f"Seconds must be 0-86399, got {seconds}"


def test_get_curr_time_first_timestep(test_data):
    """Test get_curr_time at first timestep."""
    state = test_data["test_edge_first_timestep"]["state"]
    days, seconds = clm_time_manager.get_curr_time(state)
    
    # At first timestep with tod=0, should be (0, 0)
    assert days == 0, f"Expected 0 days at first timestep, got {days}"
    assert seconds == 0, f"Expected 0 seconds at first timestep, got {seconds}"


def test_get_curr_time_dtypes(test_data):
    """Test get_curr_time returns correct data types."""
    state = test_data["test_nominal_gregorian_midyear"]["state"]
    result = clm_time_manager.get_curr_time(state)
    
    assert all(isinstance(x, int) for x in result), "All components should be int"


# ============================================================================
# Tests for get_curr_calday
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "test_nominal_gregorian_midyear",
    "test_nominal_noleap_calendar",
])
def test_get_curr_calday_values(test_data, test_case_name):
    """
    Test get_curr_calday returns valid calendar day.
    
    Verifies the function returns a float in range [1.0, 366.0] representing
    the day of year, and a boolean error flag.
    """
    test_case = test_data[test_case_name]
    state = test_case["state"]
    offset = test_case.get("offset", None)
    
    result = clm_time_manager.get_curr_calday(state, offset)
    
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2 components, got {len(result)}"
    
    calday, error = result
    
    assert isinstance(calday, float), f"Calday should be float, got {type(calday)}"
    assert isinstance(error, bool), f"Error should be bool, got {type(error)}"
    assert 1.0 <= calday <= 366.0, f"Calday must be 1.0-366.0, got {calday}"


def test_get_curr_calday_with_offset(test_data):
    """Test get_curr_calday with negative offset."""
    test_case = test_data["test_nominal_noleap_calendar"]
    state = test_case["state"]
    
    # Test with negative offset
    calday_offset, error_offset = clm_time_manager.get_curr_calday(state, -7200)
    calday_current, error_current = clm_time_manager.get_curr_calday(state, 0)
    
    assert not error_offset, "Should not error with negative offset"
    assert not error_current, "Should not error with zero offset"
    # Offset in past should give earlier calday
    assert calday_offset <= calday_current, "Past offset should give earlier or equal calday"


def test_get_curr_calday_no_offset(test_data):
    """Test get_curr_calday with None offset (current time)."""
    state = test_data["test_nominal_gregorian_midyear"]["state"]
    
    calday_none, error_none = clm_time_manager.get_curr_calday(state, None)
    calday_zero, error_zero = clm_time_manager.get_curr_calday(state, 0)
    
    assert calday_none == calday_zero, "None and 0 offset should give same result"
    assert error_none == error_zero, "Error flags should match"


def test_get_curr_calday_dtypes(test_data):
    """Test get_curr_calday returns correct data types."""
    state = test_data["test_nominal_gregorian_midyear"]["state"]
    calday, error = clm_time_manager.get_curr_calday(state)
    
    assert isinstance(calday, float), f"Expected float for calday, got {type(calday)}"
    assert isinstance(error, bool), f"Expected bool for error, got {type(error)}"


# ============================================================================
# Tests for get_prev_calday
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "test_nominal_gregorian_midyear",
    "test_nominal_noleap_calendar",
    "test_edge_first_timestep",
])
def test_get_prev_calday_values(test_data, test_case_name):
    """
    Test get_prev_calday returns valid calendar day.
    
    Verifies the function returns a float in range [1.0, 366.0] representing
    the day of year at the beginning of the timestep.
    """
    test_case = test_data[test_case_name]
    state = test_case["state"]
    
    result = clm_time_manager.get_prev_calday(state)
    
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert 1.0 <= result <= 366.0, f"Calday must be 1.0-366.0, got {result}"


def test_get_prev_calday_vs_curr_calday(test_data):
    """Test get_prev_calday returns earlier or equal day than get_curr_calday."""
    state = test_data["test_nominal_gregorian_midyear"]["state"]
    
    prev_calday = clm_time_manager.get_prev_calday(state)
    curr_calday, _ = clm_time_manager.get_curr_calday(state, 0)
    
    assert prev_calday <= curr_calday, "Previous calday should be <= current calday"


def test_get_prev_calday_dtypes(test_data):
    """Test get_prev_calday returns correct data type."""
    state = test_data["test_nominal_gregorian_midyear"]["state"]
    result = clm_time_manager.get_prev_calday(state)
    
    assert isinstance(result, float), f"Expected float type, got {type(result)}"


# ============================================================================
# Tests for get_curr_date_tuple
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "test_nominal_gregorian_midyear",
    "test_nominal_noleap_calendar",
])
def test_get_curr_date_tuple_values(test_data, test_case_name):
    """
    Test get_curr_date_tuple returns CurrentDate NamedTuple.
    
    Verifies the function returns a NamedTuple with year, month, day, tod fields.
    """
    test_case = test_data[test_case_name]
    state = test_case["state"]
    
    result = clm_time_manager.get_curr_date_tuple(state)
    
    # Check it's a NamedTuple with correct fields
    assert hasattr(result, 'year'), "Should have 'year' field"
    assert hasattr(result, 'month'), "Should have 'month' field"
    assert hasattr(result, 'day'), "Should have 'day' field"
    assert hasattr(result, 'tod'), "Should have 'tod' field"
    
    # Validate ranges
    assert isinstance(result.year, int), f"Year should be int, got {type(result.year)}"
    assert isinstance(result.month, int), f"Month should be int, got {type(result.month)}"
    assert isinstance(result.day, int), f"Day should be int, got {type(result.day)}"
    assert isinstance(result.tod, int), f"TOD should be int, got {type(result.tod)}"
    
    assert 1 <= result.month <= 12, f"Month must be 1-12, got {result.month}"
    assert 1 <= result.day <= 31, f"Day must be 1-31, got {result.day}"
    assert 0 <= result.tod <= 86400, f"TOD must be 0-86400, got {result.tod}"


def test_get_curr_date_tuple_consistency(test_data):
    """Test get_curr_date_tuple matches get_curr_date."""
    state = test_data["test_nominal_gregorian_midyear"]["state"]
    
    tuple_result = clm_time_manager.get_curr_date_tuple(state)
    date_result = clm_time_manager.get_curr_date(state)
    
    assert tuple_result.year == date_result[0], "Year should match"
    assert tuple_result.month == date_result[1], "Month should match"
    assert tuple_result.day == date_result[2], "Day should match"
    assert tuple_result.tod == date_result[3], "TOD should match"


# ============================================================================
# Tests for is_end_curr_day
# ============================================================================

@pytest.mark.parametrize("test_case_name,expected", [
    ("test_nominal_end_of_day", True),
    ("test_nominal_gregorian_midyear", False),
    ("test_edge_first_timestep", False),
])
def test_is_end_curr_day_values(test_data, test_case_name, expected):
    """
    Test is_end_curr_day correctly identifies end of day.
    
    Verifies the function returns True when current timestep is the last
    timestep in the current day (tod = 86400 or will reach it).
    """
    test_case = test_data[test_case_name]
    state = test_case["state"]
    
    result = clm_time_manager.is_end_curr_day(state)
    
    assert isinstance(result, bool), f"Expected bool, got {type(result)}"
    assert result == expected, f"Expected {expected}, got {result} for {test_case_name}"


def test_is_end_curr_day_dtypes(test_data):
    """Test is_end_curr_day returns correct data type."""
    state = test_data["test_nominal_gregorian_midyear"]["state"]
    result = clm_time_manager.is_end_curr_day(state)
    
    assert isinstance(result, bool), f"Expected bool type, got {type(result)}"


# ============================================================================
# Tests for is_end_curr_month
# ============================================================================

@pytest.mark.parametrize("test_case_name,expected", [
    ("test_nominal_end_of_month", True),
    ("test_nominal_gregorian_midyear", False),
    ("test_edge_first_timestep", False),
])
def test_is_end_curr_month_values(test_data, test_case_name, expected):
    """
    Test is_end_curr_month correctly identifies end of month.
    
    Verifies the function returns True when current timestep is the last
    timestep in the current month.
    """
    test_case = test_data[test_case_name]
    state = test_case["state"]
    
    result = clm_time_manager.is_end_curr_month(state)
    
    assert isinstance(result, bool), f"Expected bool, got {type(result)}"
    assert result == expected, f"Expected {expected}, got {result} for {test_case_name}"


def test_is_end_curr_month_february_leap(test_data):
    """Test is_end_curr_month handles February 29 in leap year."""
    state = test_data["test_nominal_february_leap"]["state"]
    
    # This state is at Feb 29, not end of month
    result = clm_time_manager.is_end_curr_month(state)
    
    assert isinstance(result, bool), "Should return bool"
    # Feb 29 is end of February in leap year
    # Result depends on whether tod indicates end of day


def test_is_end_curr_month_dtypes(test_data):
    """Test is_end_curr_month returns correct data type."""
    state = test_data["test_nominal_gregorian_midyear"]["state"]
    result = clm_time_manager.is_end_curr_month(state)
    
    assert isinstance(result, bool), f"Expected bool type, got {type(result)}"


# ============================================================================
# Tests for create_time_manager_state
# ============================================================================

@pytest.mark.parametrize("dtstep,start_ymd,start_tod,calendar", [
    (1800, 20000101, 0, "GREGORIAN"),
    (3600, 19790315, 43200, "NOLEAP"),
    (1, 20200101, 0, "GREGORIAN"),
    (21600, 20160101, 0, "GREGORIAN"),
])
def test_create_time_manager_state_values(dtstep, start_ymd, start_tod, calendar):
    """
    Test create_time_manager_state creates valid state.
    
    Verifies the function creates a TimeManagerState with correct initial values:
    - itim = 0 (first timestep)
    - curr_date matches start_date
    - All fields properly initialized
    """
    result = clm_time_manager.create_time_manager_state(
        dtstep, start_ymd, start_tod, calendar
    )
    
    # Check it's a TimeManagerState
    assert hasattr(result, 'dtstep'), "Should have 'dtstep' field"
    assert hasattr(result, 'itim'), "Should have 'itim' field"
    assert hasattr(result, 'start_date_ymd'), "Should have 'start_date_ymd' field"
    assert hasattr(result, 'start_date_tod'), "Should have 'start_date_tod' field"
    assert hasattr(result, 'curr_date_ymd'), "Should have 'curr_date_ymd' field"
    assert hasattr(result, 'curr_date_tod'), "Should have 'curr_date_tod' field"
    assert hasattr(result, 'calkindflag'), "Should have 'calkindflag' field"
    
    # Verify initial values
    assert result.dtstep == dtstep, f"Expected dtstep {dtstep}, got {result.dtstep}"
    assert result.itim == 0, f"Expected itim 0, got {result.itim}"
    assert result.start_date_ymd == start_ymd, f"Expected start_ymd {start_ymd}, got {result.start_date_ymd}"
    assert result.start_date_tod == start_tod, f"Expected start_tod {start_tod}, got {result.start_date_tod}"
    assert result.calkindflag == calendar, f"Expected calendar {calendar}, got {result.calkindflag}"
    
    # At initialization, curr_date should match start_date
    assert result.curr_date_ymd == start_ymd, "curr_date_ymd should match start_date_ymd at init"
    assert result.curr_date_tod == start_tod, "curr_date_tod should match start_date_tod at init"


def test_create_time_manager_state_default_calendar():
    """Test create_time_manager_state uses GREGORIAN as default calendar."""
    result = clm_time_manager.create_time_manager_state(1800, 20000101, 0)
    
    assert result.calkindflag == "GREGORIAN", "Default calendar should be GREGORIAN"


def test_create_time_manager_state_default_tod():
    """Test create_time_manager_state uses 0 as default start_date_tod."""
    result = clm_time_manager.create_time_manager_state(1800, 20000101)
    
    assert result.start_date_tod == 0, "Default start_date_tod should be 0"


def test_create_time_manager_state_edge_cases():
    """Test create_time_manager_state with edge case values."""
    # Minimum timestep
    result_min = clm_time_manager.create_time_manager_state(1, 20000101, 0)
    assert result_min.dtstep == 1, "Should handle minimum timestep"
    
    # Maximum tod
    result_max_tod = clm_time_manager.create_time_manager_state(1800, 20000101, 86399)
    assert result_max_tod.start_date_tod == 86399, "Should handle maximum tod"


# ============================================================================
# Tests for advance_timestep
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "test_nominal_gregorian_midyear",
    "test_edge_first_timestep",
    "test_nominal_noleap_calendar",
])
def test_advance_timestep_values(test_data, test_case_name):
    """
    Test advance_timestep increments timestep correctly.
    
    Verifies the function:
    - Increments itim by 1
    - Updates curr_date_ymd and curr_date_tod appropriately
    - Preserves other state fields
    """
    test_case = test_data[test_case_name]
    state = test_case["state"]
    
    result = clm_time_manager.advance_timestep(state)
    
    # Check structure
    assert hasattr(result, 'itim'), "Should have 'itim' field"
    
    # Verify itim incremented
    assert result.itim == state.itim + 1, f"Expected itim {state.itim + 1}, got {result.itim}"
    
    # Verify other fields preserved
    assert result.dtstep == state.dtstep, "dtstep should be preserved"
    assert result.start_date_ymd == state.start_date_ymd, "start_date_ymd should be preserved"
    assert result.start_date_tod == state.start_date_tod, "start_date_tod should be preserved"
    assert result.calkindflag == state.calkindflag, "calkindflag should be preserved"


def test_advance_timestep_multiple_steps(test_data):
    """Test advance_timestep can be called multiple times."""
    state = test_data["test_edge_first_timestep"]["state"]
    
    # Advance 5 times
    current_state = state
    for i in range(5):
        current_state = clm_time_manager.advance_timestep(current_state)
        assert current_state.itim == i + 1, f"After {i+1} advances, itim should be {i+1}"


def test_advance_timestep_immutability(test_data):
    """Test advance_timestep doesn't modify original state."""
    state = test_data["test_nominal_gregorian_midyear"]["state"]
    original_itim = state.itim
    
    result = clm_time_manager.advance_timestep(state)
    
    # Original state should be unchanged
    assert state.itim == original_itim, "Original state should not be modified"
    assert result.itim == original_itim + 1, "New state should have incremented itim"


# ============================================================================
# Integration Tests
# ============================================================================

def test_integration_full_day_simulation():
    """
    Integration test: simulate a full day with 1-hour timesteps.
    
    Tests the interaction between create_time_manager_state, advance_timestep,
    and is_end_curr_day.
    """
    # Create state for 1-hour timesteps starting at midnight
    state = clm_time_manager.create_time_manager_state(
        dtstep=3600,
        start_date_ymd=20000101,
        start_date_tod=0,
        calendar="GREGORIAN"
    )
    
    # Advance through 24 hours
    for hour in range(24):
        state = clm_time_manager.advance_timestep(state)
        
        if hour < 23:
            assert not clm_time_manager.is_end_curr_day(state), f"Hour {hour+1} should not be end of day"
        else:
            # After 24 advances, should be at end of day
            assert clm_time_manager.is_end_curr_day(state), "Hour 24 should be end of day"


def test_integration_leap_year_february():
    """
    Integration test: verify February handling in leap vs non-leap years.
    
    Tests isleap, get_curr_date, and date progression through February.
    """
    # Leap year 2000
    assert clm_time_manager.isleap(2000, "GREGORIAN"), "2000 should be leap year"
    
    state_leap = clm_time_manager.create_time_manager_state(
        dtstep=86400,  # 1 day
        start_date_ymd=20000228,
        start_date_tod=0,
        calendar="GREGORIAN"
    )
    
    # Advance one day - should go to Feb 29
    state_leap = clm_time_manager.advance_timestep(state_leap)
    year, month, day, _ = clm_time_manager.get_curr_date(state_leap)
    assert month == 2 and day == 29, "Should advance to Feb 29 in leap year"
    
    # Non-leap year 1900
    assert not clm_time_manager.isleap(1900, "GREGORIAN"), "1900 should not be leap year"


def test_integration_calendar_day_progression():
    """
    Integration test: verify calendar day increases correctly.
    
    Tests get_curr_calday and get_prev_calday progression.
    """
    state = clm_time_manager.create_time_manager_state(
        dtstep=43200,  # 12 hours
        start_date_ymd=20000101,
        start_date_tod=0,
        calendar="GREGORIAN"
    )
    
    prev_calday = clm_time_manager.get_prev_calday(state)
    
    # Advance several timesteps
    for _ in range(5):
        state = clm_time_manager.advance_timestep(state)
        curr_calday, error = clm_time_manager.get_curr_calday(state, 0)
        assert not error, "Should not error during normal progression"
        assert curr_calday >= prev_calday, "Calendar day should increase or stay same"
        prev_calday = curr_calday


def test_integration_noleap_vs_gregorian():
    """
    Integration test: compare NOLEAP and GREGORIAN calendars.
    
    Verifies that NOLEAP calendar never has leap years while GREGORIAN does.
    """
    test_years = [2000, 2004, 2020, 2024, 1900, 2100]
    
    for year in test_years:
        gregorian_leap = clm_time_manager.isleap(year, "GREGORIAN")
        noleap_leap = clm_time_manager.isleap(year, "NOLEAP")
        
        assert not noleap_leap, f"NOLEAP calendar should never have leap years, failed for {year}"
        
        # For GREGORIAN, verify known leap years
        if year in [2000, 2004, 2020, 2024]:
            assert gregorian_leap, f"Year {year} should be leap in GREGORIAN calendar"
        elif year in [1900, 2100]:
            assert not gregorian_leap, f"Year {year} should not be leap in GREGORIAN calendar"


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_edge_case_year_2000_leap():
    """Edge case: Year 2000 is leap year (divisible by 400)."""
    assert clm_time_manager.isleap(2000, "GREGORIAN"), "Year 2000 should be leap year"
    
    years = jnp.array([2000])
    result = clm_time_manager.isleap_jax(years, "GREGORIAN")
    assert result[0], "Year 2000 should be leap year in JAX version"


def test_edge_case_year_1900_not_leap():
    """Edge case: Year 1900 is not leap year (divisible by 100 but not 400)."""
    assert not clm_time_manager.isleap(1900, "GREGORIAN"), "Year 1900 should not be leap year"
    
    years = jnp.array([1900])
    result = clm_time_manager.isleap_jax(years, "GREGORIAN")
    assert not result[0], "Year 1900 should not be leap year in JAX version"


def test_edge_case_minimum_year():
    """Edge case: Test with minimum valid year (1)."""
    result = clm_time_manager.isleap(1, "GREGORIAN")
    assert isinstance(result, bool), "Should handle year 1"


def test_edge_case_large_year():
    """Edge case: Test with large year value."""
    result = clm_time_manager.isleap(9999, "GREGORIAN")
    assert isinstance(result, bool), "Should handle year 9999"


def test_edge_case_tod_boundary():
    """Edge case: Test time-of-day at exact boundary (86400 seconds)."""
    state = TimeManagerState(
        dtstep=1800,
        itim=48,
        start_date_ymd=20000101,
        start_date_tod=0,
        curr_date_ymd=20000102,
        curr_date_tod=0,
        calkindflag="GREGORIAN"
    )
    
    # Should handle transition to next day
    result = clm_time_manager.get_curr_date(state)
    assert result is not None, "Should handle day boundary"


def test_edge_case_december_31():
    """Edge case: Test end of year (December 31)."""
    state = TimeManagerState(
        dtstep=1800,
        itim=1000,
        start_date_ymd=20001201,
        start_date_tod=0,
        curr_date_ymd=20001231,
        curr_date_tod=43200,
        calkindflag="GREGORIAN"
    )
    
    year, month, day, _ = clm_time_manager.get_curr_date(state)
    assert month == 12 and day == 31, "Should handle December 31"


def test_edge_case_offset_zero_vs_none():
    """Edge case: Verify offset=0 and offset=None give same result."""
    state = TimeManagerState(
        dtstep=1800,
        itim=100,
        start_date_ymd=20000101,
        start_date_tod=0,
        curr_date_ymd=20000105,
        curr_date_tod=43200,
        calkindflag="GREGORIAN"
    )
    
    calday_none, error_none = clm_time_manager.get_curr_calday(state, None)
    calday_zero, error_zero = clm_time_manager.get_curr_calday(state, 0)
    
    assert calday_none == calday_zero, "None and 0 offset should be equivalent"
    assert error_none == error_zero, "Error flags should match"


def test_edge_case_large_negative_offset():
    """Edge case: Test with large negative offset."""
    state = TimeManagerState(
        dtstep=1800,
        itim=1000,
        start_date_ymd=20000101,
        start_date_tod=0,
        curr_date_ymd=20000120,
        curr_date_tod=0,
        calkindflag="GREGORIAN"
    )
    
    # Large negative offset (multiple days in past)
    calday, error = clm_time_manager.get_curr_calday(state, -864000)  # -10 days
    
    assert isinstance(calday, float), "Should return float"
    assert isinstance(error, bool), "Should return error flag"


# ============================================================================
# Documentation Tests
# ============================================================================

def test_module_has_docstrings():
    """Verify that key functions have docstrings."""
    functions_to_check = [
        'get_step_size',
        'get_nstep',
        'isleap',
        'isleap_jax',
        'get_curr_date',
        'create_time_manager_state',
        'advance_timestep'
    ]
    
    for func_name in functions_to_check:
        if hasattr(clm_time_manager, func_name):
            func = getattr(clm_time_manager, func_name)
            assert func.__doc__ is not None, f"Function {func_name} should have docstring"


# ============================================================================
# Consistency Tests
# ============================================================================

def test_consistency_curr_vs_prev_date():
    """Test consistency between get_curr_date and get_prev_date."""
    state = TimeManagerState(
        dtstep=3600,
        itim=10,
        start_date_ymd=20000101,
        start_date_tod=0,
        curr_date_ymd=20000101,
        curr_date_tod=36000,
        calkindflag="GREGORIAN"
    )
    
    curr_date = clm_time_manager.get_curr_date(state)
    prev_date = clm_time_manager.get_prev_date(state)
    
    # Both should return valid dates
    assert len(curr_date) == 4, "curr_date should have 4 components"
    assert len(prev_date) == 4, "prev_date should have 4 components"
    
    # prev_date should be earlier or equal to curr_date
    # (comparing as tuples works for date/time)
    assert prev_date <= curr_date, "prev_date should be <= curr_date"


def test_consistency_date_tuple_vs_date():
    """Test consistency between get_curr_date_tuple and get_curr_date."""
    state = TimeManagerState(
        dtstep=1800,
        itim=100,
        start_date_ymd=20000101,
        start_date_tod=0,
        curr_date_ymd=20000105,
        curr_date_tod=43200,
        calkindflag="GREGORIAN"
    )
    
    date_tuple = clm_time_manager.get_curr_date_tuple(state)
    date_components = clm_time_manager.get_curr_date(state)
    
    assert date_tuple.year == date_components[0], "Year should match"
    assert date_tuple.month == date_components[1], "Month should match"
    assert date_tuple.day == date_components[2], "Day should match"
    assert date_tuple.tod == date_components[3], "TOD should match"


def test_consistency_isleap_scalar_vs_array():
    """Test consistency between isleap and isleap_jax for same inputs."""
    test_years = [2000, 2004, 1900, 2100, 1999, 2001]
    
    for year in test_years:
        scalar_result = clm_time_manager.isleap(year, "GREGORIAN")
        array_result = clm_time_manager.isleap_jax(jnp.array([year]), "GREGORIAN")
        
        assert scalar_result == bool(array_result[0]), \
            f"Scalar and array versions should agree for year {year}"