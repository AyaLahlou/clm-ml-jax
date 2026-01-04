"""
Comprehensive pytest suite for controlMod.control function.

This module tests the control function which configures CLM-ML simulation runs
by processing namelist inputs and tower data to generate control configuration
parameters including time stepping, directory paths, and tower site selection.

Test Coverage:
- Nominal cases: Single/multiple towers, ndays/nsteps options, various dates
- Edge cases: Boundary time-of-day values, minimum simulation lengths
- Special cases: Mixed timesteps, large tower counts, date boundaries
- Output validation: Shapes, types, values, and constraints
"""

import sys
from pathlib import Path
from typing import NamedTuple

import pytest
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from offline_driver.controlMod import (
    control,
    ControlConfig,
    NamelistInput,
    TowerData,
    ControlError,
)


@pytest.fixture
def test_data():
    """
    Fixture providing comprehensive test data for control function.
    
    Returns:
        dict: Test cases with inputs and metadata for various scenarios
    """
    return {
        "test_nominal_single_tower_ndays": {
            "inputs": {
                "namelist": NamelistInput(
                    tower_name="US-Ha1",
                    start_ymd=20150701,
                    start_tod=43200,
                    stop_option="ndays",
                    stop_n=30,
                    clm_start_ymd=20150701,
                    clm_start_tod=0
                ),
                "tower_data": TowerData(
                    ntower=1,
                    tower_id=("US-Ha1",),
                    tower_time=(30,)
                )
            },
            "metadata": {
                "type": "nominal",
                "description": "Standard single tower configuration with 30-day simulation starting at noon UTC"
            }
        },
        "test_nominal_multiple_towers_nsteps": {
            "inputs": {
                "namelist": NamelistInput(
                    tower_name="US-Var",
                    start_ymd=20180415,
                    start_tod=0,
                    stop_option="nsteps",
                    stop_n=2880,
                    clm_start_ymd=20180401,
                    clm_start_tod=0
                ),
                "tower_data": TowerData(
                    ntower=5,
                    tower_id=("US-Ha1", "US-Var", "US-UMB", "US-MMS", "US-WCr"),
                    tower_time=(30, 30, 60, 30, 30)
                )
            },
            "metadata": {
                "type": "nominal",
                "description": "Multiple tower sites with nsteps option, 2880 steps (30 days at 30-min intervals)"
            }
        },
        "test_nominal_leap_year_date": {
            "inputs": {
                "namelist": NamelistInput(
                    tower_name="US-MMS",
                    start_ymd=20200229,
                    start_tod=21600,
                    stop_option="ndays",
                    stop_n=10,
                    clm_start_ymd=20200229,
                    clm_start_tod=21600
                ),
                "tower_data": TowerData(
                    ntower=3,
                    tower_id=("US-Ha1", "US-MMS", "US-WCr"),
                    tower_time=(30, 30, 60)
                )
            },
            "metadata": {
                "type": "nominal",
                "description": "Leap year date (Feb 29) with 6-hour offset start time"
            }
        },
        "test_nominal_year_boundary": {
            "inputs": {
                "namelist": NamelistInput(
                    tower_name="US-UMB",
                    start_ymd=20191231,
                    start_tod=82800,
                    stop_option="ndays",
                    stop_n=5,
                    clm_start_ymd=20191231,
                    clm_start_tod=0
                ),
                "tower_data": TowerData(
                    ntower=2,
                    tower_id=("US-UMB", "US-Ha1"),
                    tower_time=(30, 30)
                )
            },
            "metadata": {
                "type": "nominal",
                "description": "Year boundary crossing (Dec 31) with late evening start time (23:00 UTC)"
            }
        },
        "test_nominal_long_simulation": {
            "inputs": {
                "namelist": NamelistInput(
                    tower_name="US-WCr",
                    start_ymd=20160101,
                    start_tod=0,
                    stop_option="ndays",
                    stop_n=365,
                    clm_start_ymd=20160101,
                    clm_start_tod=0
                ),
                "tower_data": TowerData(
                    ntower=1,
                    tower_id=("US-WCr",),
                    tower_time=(30,)
                )
            },
            "metadata": {
                "type": "nominal",
                "description": "Full year simulation (365 days) starting at midnight UTC on Jan 1"
            }
        },
        "test_edge_minimum_stop_n": {
            "inputs": {
                "namelist": NamelistInput(
                    tower_name="US-Ha1",
                    start_ymd=20170615,
                    start_tod=0,
                    stop_option="ndays",
                    stop_n=1,
                    clm_start_ymd=20170615,
                    clm_start_tod=0
                ),
                "tower_data": TowerData(
                    ntower=1,
                    tower_id=("US-Ha1",),
                    tower_time=(30,)
                )
            },
            "metadata": {
                "type": "edge",
                "description": "Minimum simulation length (1 day)"
            }
        },
        "test_edge_boundary_start_tod_zero": {
            "inputs": {
                "namelist": NamelistInput(
                    tower_name="US-Var",
                    start_ymd=20190801,
                    start_tod=0,
                    stop_option="nsteps",
                    stop_n=48,
                    clm_start_ymd=20190801,
                    clm_start_tod=0
                ),
                "tower_data": TowerData(
                    ntower=2,
                    tower_id=("US-Var", "US-MMS"),
                    tower_time=(30, 30)
                )
            },
            "metadata": {
                "type": "edge",
                "description": "Start time at exact midnight (0 seconds past 0Z)"
            }
        },
        "test_edge_boundary_start_tod_max": {
            "inputs": {
                "namelist": NamelistInput(
                    tower_name="US-UMB",
                    start_ymd=20210310,
                    start_tod=86400,
                    stop_option="ndays",
                    stop_n=7,
                    clm_start_ymd=20210310,
                    clm_start_tod=86400
                ),
                "tower_data": TowerData(
                    ntower=1,
                    tower_id=("US-UMB",),
                    tower_time=(60,)
                )
            },
            "metadata": {
                "type": "edge",
                "description": "Start time at maximum boundary (86400 seconds = end of day)"
            }
        },
        "test_special_hourly_timestep": {
            "inputs": {
                "namelist": NamelistInput(
                    tower_name="US-MMS",
                    start_ymd=20140920,
                    start_tod=10800,
                    stop_option="nsteps",
                    stop_n=720,
                    clm_start_ymd=20140920,
                    clm_start_tod=0
                ),
                "tower_data": TowerData(
                    ntower=4,
                    tower_id=("US-Ha1", "US-MMS", "US-UMB", "US-Var"),
                    tower_time=(60, 60, 60, 60)
                )
            },
            "metadata": {
                "type": "special",
                "description": "All towers with hourly (60-minute) time steps, 720 steps = 30 days"
            }
        },
        "test_special_mixed_timesteps_large_ntower": {
            "inputs": {
                "namelist": NamelistInput(
                    tower_name="US-WCr",
                    start_ymd=20221105,
                    start_tod=64800,
                    stop_option="ndays",
                    stop_n=90,
                    clm_start_ymd=20221101,
                    clm_start_tod=0
                ),
                "tower_data": TowerData(
                    ntower=8,
                    tower_id=("US-Ha1", "US-Var", "US-UMB", "US-MMS", "US-WCr", "US-NR1", "US-ARM", "US-Bo1"),
                    tower_time=(30, 30, 60, 30, 15, 30, 60, 30)
                )
            },
            "metadata": {
                "type": "special",
                "description": "Large number of towers (8) with mixed time steps (15, 30, 60 minutes)"
            }
        }
    }


@pytest.mark.parametrize("test_case_name", [
    "test_nominal_single_tower_ndays",
    "test_nominal_multiple_towers_nsteps",
    "test_nominal_leap_year_date",
    "test_nominal_year_boundary",
    "test_nominal_long_simulation",
    "test_edge_minimum_stop_n",
    "test_edge_boundary_start_tod_zero",
    "test_edge_boundary_start_tod_max",
    "test_special_hourly_timestep",
    "test_special_mixed_timesteps_large_ntower"
])
def test_control_output_type(test_data, test_case_name):
    """
    Test that control function returns ControlConfig namedtuple.
    
    Verifies that the output is a ControlConfig instance with all required fields.
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    result = control(inputs["namelist"], inputs["tower_data"])
    
    assert isinstance(result, ControlConfig), \
        f"Expected ControlConfig, got {type(result)}"
    
    # Verify all fields are present
    expected_fields = [
        'ntim', 'clm_start_ymd', 'clm_start_tod', 'diratm', 'dirclm',
        'dirout', 'dirin', 'start_date_ymd', 'start_date_tod', 'dtstep',
        'tower_num', 'clm_phys'
    ]
    for field in expected_fields:
        assert hasattr(result, field), \
            f"ControlConfig missing required field: {field}"


@pytest.mark.parametrize("test_case_name", [
    "test_nominal_single_tower_ndays",
    "test_nominal_multiple_towers_nsteps",
    "test_nominal_leap_year_date",
    "test_nominal_year_boundary",
    "test_nominal_long_simulation",
])
def test_control_nominal_values(test_data, test_case_name):
    """
    Test control function with nominal input cases.
    
    Verifies that:
    - Output dates match input dates
    - Time-of-day values are preserved
    - Tower number is correctly identified
    - Time step is calculated from tower_time
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    namelist = inputs["namelist"]
    tower_data = inputs["tower_data"]
    
    result = control(namelist, tower_data)
    
    # Verify date/time values are preserved
    assert result.start_date_ymd == namelist.start_ymd, \
        f"start_date_ymd mismatch: expected {namelist.start_ymd}, got {result.start_date_ymd}"
    
    assert result.start_date_tod == namelist.start_tod, \
        f"start_date_tod mismatch: expected {namelist.start_tod}, got {result.start_date_tod}"
    
    assert result.clm_start_ymd == namelist.clm_start_ymd, \
        f"clm_start_ymd mismatch: expected {namelist.clm_start_ymd}, got {result.clm_start_ymd}"
    
    assert result.clm_start_tod == namelist.clm_start_tod, \
        f"clm_start_tod mismatch: expected {namelist.clm_start_tod}, got {result.clm_start_tod}"
    
    # Verify tower is found in tower_data
    assert namelist.tower_name in tower_data.tower_id, \
        f"Tower {namelist.tower_name} not found in tower_data"
    
    tower_idx = tower_data.tower_id.index(namelist.tower_name)
    assert result.tower_num == tower_idx + 1, \
        f"tower_num should be {tower_idx + 1} (1-based), got {result.tower_num}"
    
    # Verify dtstep is calculated from tower_time (minutes to seconds)
    expected_dtstep = tower_data.tower_time[tower_idx] * 60
    assert result.dtstep == expected_dtstep, \
        f"dtstep mismatch: expected {expected_dtstep}, got {result.dtstep}"


@pytest.mark.parametrize("test_case_name", [
    "test_edge_minimum_stop_n",
    "test_edge_boundary_start_tod_zero",
    "test_edge_boundary_start_tod_max",
])
def test_control_edge_cases(test_data, test_case_name):
    """
    Test control function with edge case inputs.
    
    Verifies behavior at boundary conditions:
    - Minimum stop_n value (1)
    - Time-of-day at 0 seconds
    - Time-of-day at 86400 seconds
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    namelist = inputs["namelist"]
    tower_data = inputs["tower_data"]
    
    result = control(namelist, tower_data)
    
    # Should not raise exceptions
    assert result is not None, "Function should return valid result for edge cases"
    
    # Verify constraints are satisfied
    assert result.ntim >= 1, f"ntim must be >= 1, got {result.ntim}"
    assert 0 <= result.start_date_tod <= 86400, \
        f"start_date_tod must be in [0, 86400], got {result.start_date_tod}"
    assert 0 <= result.clm_start_tod <= 86400, \
        f"clm_start_tod must be in [0, 86400], got {result.clm_start_tod}"
    assert result.dtstep >= 1, f"dtstep must be >= 1, got {result.dtstep}"
    assert result.tower_num >= 1, f"tower_num must be >= 1, got {result.tower_num}"


@pytest.mark.parametrize("test_case_name", [
    "test_special_hourly_timestep",
    "test_special_mixed_timesteps_large_ntower",
])
def test_control_special_cases(test_data, test_case_name):
    """
    Test control function with special configurations.
    
    Verifies:
    - Handling of multiple towers with various timesteps
    - Correct tower identification in large tower arrays
    - Proper timestep calculation for different intervals
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    namelist = inputs["namelist"]
    tower_data = inputs["tower_data"]
    
    result = control(namelist, tower_data)
    
    # Verify tower is correctly identified
    tower_idx = tower_data.tower_id.index(namelist.tower_name)
    assert result.tower_num == tower_idx + 1, \
        f"tower_num incorrect for {namelist.tower_name} in large array"
    
    # Verify dtstep matches the specific tower's time interval
    expected_dtstep = tower_data.tower_time[tower_idx] * 60
    assert result.dtstep == expected_dtstep, \
        f"dtstep should be {expected_dtstep} for tower {namelist.tower_name}"
    
    # Verify dtstep divides evenly into a day
    assert 86400 % result.dtstep == 0, \
        f"dtstep {result.dtstep} must divide evenly into 86400 seconds"


def test_control_ntim_calculation_ndays(test_data):
    """
    Test ntim calculation for 'ndays' stop_option.
    
    Verifies that ntim is correctly calculated as:
    ntim = (stop_n * 86400) / dtstep
    """
    test_case = test_data["test_nominal_single_tower_ndays"]
    inputs = test_case["inputs"]
    namelist = inputs["namelist"]
    tower_data = inputs["tower_data"]
    
    result = control(namelist, tower_data)
    
    # Calculate expected ntim
    tower_idx = tower_data.tower_id.index(namelist.tower_name)
    dtstep = tower_data.tower_time[tower_idx] * 60
    expected_ntim = (namelist.stop_n * 86400) // dtstep
    
    assert result.ntim == expected_ntim, \
        f"ntim mismatch for ndays: expected {expected_ntim}, got {result.ntim}"


def test_control_ntim_calculation_nsteps(test_data):
    """
    Test ntim calculation for 'nsteps' stop_option.
    
    Verifies that ntim equals stop_n when stop_option is 'nsteps'.
    """
    test_case = test_data["test_nominal_multiple_towers_nsteps"]
    inputs = test_case["inputs"]
    namelist = inputs["namelist"]
    tower_data = inputs["tower_data"]
    
    result = control(namelist, tower_data)
    
    assert result.ntim == namelist.stop_n, \
        f"ntim should equal stop_n for nsteps: expected {namelist.stop_n}, got {result.ntim}"


def test_control_directory_paths(test_data):
    """
    Test that directory paths are properly set in output.
    
    Verifies that all directory path fields are non-empty strings.
    """
    test_case = test_data["test_nominal_single_tower_ndays"]
    inputs = test_case["inputs"]
    
    result = control(inputs["namelist"], inputs["tower_data"])
    
    # Check all directory paths are strings
    assert isinstance(result.diratm, str), "diratm must be a string"
    assert isinstance(result.dirclm, str), "dirclm must be a string"
    assert isinstance(result.dirout, str), "dirout must be a string"
    assert isinstance(result.dirin, str), "dirin must be a string"
    
    # Check paths are non-empty
    assert len(result.diratm) > 0, "diratm should not be empty"
    assert len(result.dirclm) > 0, "dirclm should not be empty"
    assert len(result.dirout) > 0, "dirout should not be empty"
    assert len(result.dirin) > 0, "dirin should not be empty"


def test_control_clm_phys_version(test_data):
    """
    Test that CLM physics version is set correctly.
    
    Verifies that clm_phys is one of the valid options: 'CLM4_5' or 'CLM5_0'.
    """
    test_case = test_data["test_nominal_single_tower_ndays"]
    inputs = test_case["inputs"]
    
    result = control(inputs["namelist"], inputs["tower_data"])
    
    valid_versions = ['CLM4_5', 'CLM5_0']
    assert result.clm_phys in valid_versions, \
        f"clm_phys must be one of {valid_versions}, got {result.clm_phys}"


def test_control_date_format_validation(test_data):
    """
    Test that date values maintain yyyymmdd format.
    
    Verifies that date fields are 8-digit integers representing valid dates.
    """
    test_case = test_data["test_nominal_leap_year_date"]
    inputs = test_case["inputs"]
    
    result = control(inputs["namelist"], inputs["tower_data"])
    
    # Check date formats (should be 8-digit integers)
    assert 10000000 <= result.start_date_ymd <= 99999999, \
        f"start_date_ymd should be 8-digit yyyymmdd format, got {result.start_date_ymd}"
    
    assert 10000000 <= result.clm_start_ymd <= 99999999, \
        f"clm_start_ymd should be 8-digit yyyymmdd format, got {result.clm_start_ymd}"


def test_control_tower_not_found():
    """
    Test behavior when tower_name is not in tower_data.
    
    Verifies that function handles missing tower gracefully (should raise error or return None).
    """
    namelist = NamelistInput(
        tower_name="XX-XXX",  # Non-existent tower
        start_ymd=20150701,
        start_tod=0,
        stop_option="ndays",
        stop_n=1,
        clm_start_ymd=20150701,
        clm_start_tod=0
    )
    
    tower_data = TowerData(
        ntower=2,
        tower_id=("US-Ha1", "US-Var"),
        tower_time=(30, 30)
    )
    
    # Should raise ValueError or similar
    with pytest.raises((ValueError, KeyError, IndexError, ControlError)):
        control(namelist, tower_data)


def test_control_consistency_across_calls(test_data):
    """
    Test that control function produces consistent results for identical inputs.
    
    Verifies deterministic behavior by calling function multiple times.
    """
    test_case = test_data["test_nominal_single_tower_ndays"]
    inputs = test_case["inputs"]
    
    result1 = control(inputs["namelist"], inputs["tower_data"])
    result2 = control(inputs["namelist"], inputs["tower_data"])
    
    # All fields should be identical
    assert result1 == result2, "Function should produce consistent results for same inputs"


def test_control_all_integer_fields_positive(test_data):
    """
    Test that all integer fields have positive values where required.
    
    Verifies constraints on integer fields (ntim, dtstep, tower_num > 0).
    """
    test_case = test_data["test_nominal_multiple_towers_nsteps"]
    inputs = test_case["inputs"]
    
    result = control(inputs["namelist"], inputs["tower_data"])
    
    assert result.ntim > 0, f"ntim must be positive, got {result.ntim}"
    assert result.dtstep > 0, f"dtstep must be positive, got {result.dtstep}"
    assert result.tower_num > 0, f"tower_num must be positive, got {result.tower_num}"


def test_control_tower_num_within_bounds(test_data):
    """
    Test that tower_num is within valid range [1, ntower].
    
    Verifies that tower_num doesn't exceed the number of available towers.
    """
    test_case = test_data["test_special_mixed_timesteps_large_ntower"]
    inputs = test_case["inputs"]
    tower_data = inputs["tower_data"]
    
    result = control(inputs["namelist"], tower_data)
    
    assert 1 <= result.tower_num <= tower_data.ntower, \
        f"tower_num {result.tower_num} must be in range [1, {tower_data.ntower}]"


@pytest.mark.parametrize("stop_option", ["ndays", "nsteps"])
def test_control_stop_option_variants(test_data, stop_option):
    """
    Test control function with both stop_option values.
    
    Verifies that function handles both 'ndays' and 'nsteps' correctly.
    """
    # Use appropriate test case based on stop_option
    if stop_option == "ndays":
        test_case = test_data["test_nominal_single_tower_ndays"]
    else:
        test_case = test_data["test_nominal_multiple_towers_nsteps"]
    
    inputs = test_case["inputs"]
    result = control(inputs["namelist"], inputs["tower_data"])
    
    assert result is not None, f"Function should handle stop_option='{stop_option}'"
    assert result.ntim > 0, f"ntim should be positive for stop_option='{stop_option}'"


def test_control_dtstep_divides_day(test_data):
    """
    Test that dtstep always divides evenly into 86400 seconds (1 day).
    
    This is a critical constraint for time stepping in the simulation.
    """
    for test_case_name in test_data.keys():
        test_case = test_data[test_case_name]
        inputs = test_case["inputs"]
        
        result = control(inputs["namelist"], inputs["tower_data"])
        
        assert 86400 % result.dtstep == 0, \
            f"dtstep {result.dtstep} must divide evenly into 86400 for {test_case_name}"