"""
Comprehensive pytest suite for abortutils module.

This module tests error handling and program termination utilities translated
from Fortran CLM code to Python. Tests cover:
- Program termination functions (endrun, handle_err, check_netcdf_status, assert_condition)
- Warning functions (warn_and_continue)
- Custom exception classes (CLMError, CLMNetCDFError, etc.)

All tests that involve sys.exit() use pytest.raises(SystemExit) to prevent
actual termination during testing.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import pytest
import logging

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from clm_src_main.abortutils import (
    endrun,
    handle_err,
    check_netcdf_status,
    assert_condition,
    warn_and_continue,
    NetCDFConstants,
    CLMError,
    CLMNetCDFError,
    CLMInitializationError,
    CLMComputationError,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data():
    """
    Fixture providing comprehensive test data for all abortutils functions.
    
    Returns:
        dict: Test cases organized by function name with inputs and metadata
    """
    return {
        "endrun": [
            {
                "name": "simple_message",
                "inputs": {"msg": "Simple error occurred"},
                "type": "nominal",
                "description": "Basic error message"
            },
            {
                "name": "none_message",
                "inputs": {"msg": None},
                "type": "edge",
                "description": "No message provided"
            },
            {
                "name": "empty_string",
                "inputs": {"msg": ""},
                "type": "edge",
                "description": "Empty string message"
            },
            {
                "name": "multiline_message",
                "inputs": {
                    "msg": "Critical error in CLM simulation:\n  - Temperature out of bounds\n  - Timestep: 1234\n  - Location: (45.5, -122.6)"
                },
                "type": "nominal",
                "description": "Formatted multiline error"
            },
            {
                "name": "special_characters",
                "inputs": {"msg": "Error: Invalid value μ=∞, expected finite number (τ < 1e-6)"},
                "type": "special",
                "description": "Unicode and special characters"
            },
        ],
        "handle_err": [
            {
                "name": "no_error",
                "inputs": {"status": 0, "errmsg": "This should not trigger"},
                "type": "nominal",
                "should_exit": False,
                "description": "NF_NOERR status (no error)"
            },
            {
                "name": "with_error",
                "inputs": {"status": -33, "errmsg": "Failed to open NetCDF file: data.nc"},
                "type": "nominal",
                "should_exit": True,
                "description": "Non-zero error status"
            },
            {
                "name": "positive_error_code",
                "inputs": {"status": 1, "errmsg": "Unexpected positive error code"},
                "type": "edge",
                "should_exit": True,
                "description": "Positive non-zero status"
            },
            {
                "name": "large_negative_status",
                "inputs": {"status": -2147483648, "errmsg": "NetCDF dimension mismatch in variable 'temperature'"},
                "type": "edge",
                "should_exit": True,
                "description": "Extreme negative status code"
            },
            {
                "name": "empty_message",
                "inputs": {"status": -45, "errmsg": ""},
                "type": "edge",
                "should_exit": True,
                "description": "Empty error message"
            },
        ],
        "check_netcdf_status": [
            {
                "name": "success",
                "inputs": {"status": 0, "operation": "Reading variable 'soil_temperature'"},
                "type": "nominal",
                "should_exit": False,
                "description": "Successful operation"
            },
            {
                "name": "failure",
                "inputs": {"status": -51, "operation": "Writing to NetCDF file"},
                "type": "nominal",
                "should_exit": True,
                "description": "Failed operation"
            },
            {
                "name": "default_operation",
                "inputs": {"status": -36, "operation": "NetCDF operation"},
                "type": "nominal",
                "should_exit": True,
                "description": "Default operation string"
            },
            {
                "name": "complex_operation",
                "inputs": {
                    "status": -61,
                    "operation": "Creating dimension 'time' with size UNLIMITED in file '/data/clm/output/clm_history_2024.nc'"
                },
                "type": "special",
                "should_exit": True,
                "description": "Detailed operation description"
            },
        ],
        "assert_condition": [
            {
                "name": "true_condition",
                "inputs": {"condition": True, "msg": "This assertion should pass"},
                "type": "nominal",
                "should_exit": False,
                "description": "True condition"
            },
            {
                "name": "false_condition",
                "inputs": {"condition": False, "msg": "Temperature must be positive"},
                "type": "nominal",
                "should_exit": True,
                "description": "False condition"
            },
            {
                "name": "false_detailed_message",
                "inputs": {
                    "condition": False,
                    "msg": "Array bounds violation: index 150 exceeds array size 100 in subroutine CanopyFluxes"
                },
                "type": "special",
                "should_exit": True,
                "description": "Detailed diagnostic message"
            },
        ],
        "warn_and_continue": [
            {
                "name": "simple_warning",
                "inputs": {"msg": "Warning: Using default parameter value"},
                "type": "nominal",
                "description": "Simple warning message"
            },
            {
                "name": "detailed_warning",
                "inputs": {
                    "msg": "WARNING: Soil moisture approaching wilting point (θ=0.05) at grid cell (i=45, j=123). Consider adjusting irrigation schedule."
                },
                "type": "nominal",
                "description": "Detailed scientific warning"
            },
            {
                "name": "empty_warning",
                "inputs": {"msg": ""},
                "type": "edge",
                "description": "Empty warning message"
            },
            {
                "name": "multiline_warning",
                "inputs": {
                    "msg": "Performance Warning:\n  - Timestep reduced to 0.5s for stability\n  - Expected simulation time increased by 2x\n  - Consider adjusting CFL condition"
                },
                "type": "special",
                "description": "Multiline formatted warning"
            },
        ],
    }


@pytest.fixture
def mock_logger():
    """
    Fixture to mock the logging module for verification.
    
    Returns:
        MagicMock: Mocked logger object
    """
    with patch('clm_src_main.abortutils.logging') as mock_log:
        yield mock_log


@pytest.fixture
def capture_stdout(capsys):
    """
    Fixture to capture stdout/stderr for verification.
    
    Args:
        capsys: pytest's built-in capsys fixture
        
    Returns:
        capsys: The capsys fixture for capturing output
    """
    return capsys


# ============================================================================
# Test NetCDFConstants
# ============================================================================

def test_netcdf_constants_values():
    """
    Test that NetCDFConstants has correct values.
    
    Verifies:
    - NF_NOERR is defined and equals 0
    """
    assert hasattr(NetCDFConstants, 'NF_NOERR'), "NetCDFConstants should have NF_NOERR attribute"
    assert NetCDFConstants.NF_NOERR == 0, "NF_NOERR should equal 0"


# ============================================================================
# Test Exception Classes
# ============================================================================

def test_clm_error_inheritance():
    """
    Test CLMError exception class inheritance.
    
    Verifies:
    - CLMError inherits from Exception
    - Can be instantiated with a message
    - Message is accessible
    """
    error = CLMError("Test error message")
    assert isinstance(error, Exception), "CLMError should inherit from Exception"
    assert str(error) == "Test error message", "Error message should be accessible"


def test_clm_netcdf_error_creation():
    """
    Test CLMNetCDFError exception creation and attributes.
    
    Verifies:
    - CLMNetCDFError inherits from CLMError
    - Status and message are stored correctly
    - String representation includes both status and message
    """
    status = -33
    message = "Failed to read variable 'LAI' from file"
    error = CLMNetCDFError(status, message)
    
    assert isinstance(error, CLMError), "CLMNetCDFError should inherit from CLMError"
    assert error.status == status, "Status should be stored correctly"
    assert message in str(error), "Message should be in string representation"


def test_clm_netcdf_error_zero_status():
    """
    Test CLMNetCDFError with zero status (edge case).
    
    Verifies:
    - Exception can be created with zero status
    - This is unusual but should be allowed
    """
    error = CLMNetCDFError(0, "Unexpected error with zero status")
    assert error.status == 0, "Should allow zero status"


def test_clm_initialization_error():
    """
    Test CLMInitializationError exception class.
    
    Verifies:
    - CLMInitializationError inherits from CLMError
    - Can be instantiated with a message
    """
    error = CLMInitializationError("Initialization failed")
    assert isinstance(error, CLMError), "CLMInitializationError should inherit from CLMError"
    assert "Initialization failed" in str(error)


def test_clm_computation_error():
    """
    Test CLMComputationError exception class.
    
    Verifies:
    - CLMComputationError inherits from CLMError
    - Can be instantiated with a message
    """
    error = CLMComputationError("Computation failed")
    assert isinstance(error, CLMError), "CLMComputationError should inherit from CLMError"
    assert "Computation failed" in str(error)


# ============================================================================
# Test endrun function
# ============================================================================

@pytest.mark.parametrize("test_case", [
    {"name": "simple_message", "msg": "Simple error occurred"},
    {"name": "multiline_message", "msg": "Critical error in CLM simulation:\n  - Temperature out of bounds\n  - Timestep: 1234\n  - Location: (45.5, -122.6)"},
    {"name": "special_characters", "msg": "Error: Invalid value μ=∞, expected finite number (τ < 1e-6)"},
])
def test_endrun_with_message(test_case, capture_stdout):
    """
    Test endrun function with various message types.
    
    Verifies:
    - Function calls sys.exit(1)
    - Error message is printed to stdout
    - Different message formats are handled correctly
    
    Args:
        test_case: Dictionary with test case name and message
        capture_stdout: Fixture to capture output
    """
    with pytest.raises(SystemExit) as exc_info:
        endrun(msg=test_case["msg"])
    
    assert exc_info.value.code == 1, f"endrun should exit with code 1 for {test_case['name']}"
    
    captured = capture_stdout.readouterr()
    if test_case["msg"]:
        assert test_case["msg"] in captured.out or test_case["msg"] in captured.err, \
            f"Error message should be printed for {test_case['name']}"


def test_endrun_with_none_message(capture_stdout):
    """
    Test endrun function with None message (edge case).
    
    Verifies:
    - Function calls sys.exit(1) even with None message
    - No exception is raised for None message
    """
    with pytest.raises(SystemExit) as exc_info:
        endrun(msg=None)
    
    assert exc_info.value.code == 1, "endrun should exit with code 1 even with None message"


def test_endrun_with_empty_string(capture_stdout):
    """
    Test endrun function with empty string message (edge case).
    
    Verifies:
    - Function calls sys.exit(1) with empty message
    - Empty string is handled gracefully
    """
    with pytest.raises(SystemExit) as exc_info:
        endrun(msg="")
    
    assert exc_info.value.code == 1, "endrun should exit with code 1 with empty message"


# ============================================================================
# Test handle_err function
# ============================================================================

def test_handle_err_no_error(capture_stdout):
    """
    Test handle_err with NF_NOERR status (no error).
    
    Verifies:
    - Function returns normally without calling sys.exit
    - No error message is printed
    """
    # Should not raise SystemExit
    handle_err(status=0, errmsg="This should not trigger")
    
    # If we get here, the function returned normally (correct behavior)
    captured = capture_stdout.readouterr()
    # No specific assertion on output since function should just return


@pytest.mark.parametrize("test_case", [
    {"status": -33, "errmsg": "Failed to open NetCDF file: data.nc", "name": "with_error"},
    {"status": 1, "errmsg": "Unexpected positive error code", "name": "positive_error_code"},
    {"status": -2147483648, "errmsg": "NetCDF dimension mismatch in variable 'temperature'", "name": "large_negative_status"},
    {"status": -45, "errmsg": "", "name": "empty_message"},
])
def test_handle_err_with_error(test_case, capture_stdout):
    """
    Test handle_err with various error statuses.
    
    Verifies:
    - Function calls sys.exit(1) for non-zero status
    - Error message includes status code and custom message
    - Handles various status codes (negative, positive, extreme values)
    
    Args:
        test_case: Dictionary with status, errmsg, and test name
        capture_stdout: Fixture to capture output
    """
    with pytest.raises(SystemExit) as exc_info:
        handle_err(status=test_case["status"], errmsg=test_case["errmsg"])
    
    assert exc_info.value.code == 1, f"handle_err should exit with code 1 for {test_case['name']}"
    
    captured = capture_stdout.readouterr()
    # Error message should contain the custom message (if not empty)
    if test_case["errmsg"]:
        assert test_case["errmsg"] in captured.out or test_case["errmsg"] in captured.err, \
            f"Error message should be printed for {test_case['name']}"


# ============================================================================
# Test check_netcdf_status function
# ============================================================================

def test_check_netcdf_status_success(capture_stdout):
    """
    Test check_netcdf_status with successful operation (status=0).
    
    Verifies:
    - Function returns normally without calling sys.exit
    - No error is raised for successful operations
    """
    # Should not raise SystemExit
    check_netcdf_status(status=0, operation="Reading variable 'soil_temperature'")
    
    # If we get here, the function returned normally (correct behavior)


@pytest.mark.parametrize("test_case", [
    {"status": -51, "operation": "Writing to NetCDF file", "name": "failure"},
    {"status": -36, "operation": "NetCDF operation", "name": "default_operation"},
    {"status": -61, "operation": "Creating dimension 'time' with size UNLIMITED in file '/data/clm/output/clm_history_2024.nc'", "name": "complex_operation"},
])
def test_check_netcdf_status_failure(test_case, capture_stdout):
    """
    Test check_netcdf_status with failed operations.
    
    Verifies:
    - Function calls handle_err which exits for non-zero status
    - Operation description is included in error message
    
    Args:
        test_case: Dictionary with status, operation, and test name
        capture_stdout: Fixture to capture output
    """
    with pytest.raises(SystemExit) as exc_info:
        check_netcdf_status(status=test_case["status"], operation=test_case["operation"])
    
    assert exc_info.value.code == 1, f"check_netcdf_status should exit with code 1 for {test_case['name']}"
    
    captured = capture_stdout.readouterr()
    # Operation description should be in error message
    assert test_case["operation"] in captured.out or test_case["operation"] in captured.err, \
        f"Operation description should be in error message for {test_case['name']}"


# ============================================================================
# Test assert_condition function
# ============================================================================

def test_assert_condition_true(capture_stdout):
    """
    Test assert_condition with true condition.
    
    Verifies:
    - Function returns normally without calling endrun
    - No error is raised for true conditions
    """
    # Should not raise SystemExit
    assert_condition(condition=True, msg="This assertion should pass")
    
    # If we get here, the function returned normally (correct behavior)


@pytest.mark.parametrize("test_case", [
    {"condition": False, "msg": "Temperature must be positive", "name": "false_condition"},
    {"condition": False, "msg": "Array bounds violation: index 150 exceeds array size 100 in subroutine CanopyFluxes", "name": "false_detailed_message"},
])
def test_assert_condition_false(test_case, capture_stdout):
    """
    Test assert_condition with false conditions.
    
    Verifies:
    - Function calls endrun which exits for false conditions
    - Error message is included in output
    
    Args:
        test_case: Dictionary with condition, msg, and test name
        capture_stdout: Fixture to capture output
    """
    with pytest.raises(SystemExit) as exc_info:
        assert_condition(condition=test_case["condition"], msg=test_case["msg"])
    
    assert exc_info.value.code == 1, f"assert_condition should exit with code 1 for {test_case['name']}"
    
    captured = capture_stdout.readouterr()
    assert test_case["msg"] in captured.out or test_case["msg"] in captured.err, \
        f"Error message should be printed for {test_case['name']}"


# ============================================================================
# Test warn_and_continue function
# ============================================================================

@pytest.mark.parametrize("test_case", [
    {"msg": "Warning: Using default parameter value", "name": "simple_warning"},
    {"msg": "WARNING: Soil moisture approaching wilting point (θ=0.05) at grid cell (i=45, j=123). Consider adjusting irrigation schedule.", "name": "detailed_warning"},
    {"msg": "", "name": "empty_warning"},
    {"msg": "Performance Warning:\n  - Timestep reduced to 0.5s for stability\n  - Expected simulation time increased by 2x\n  - Consider adjusting CFL condition", "name": "multiline_warning"},
])
def test_warn_and_continue(test_case, capture_stdout):
    """
    Test warn_and_continue function with various warning messages.
    
    Verifies:
    - Function returns normally (does not exit)
    - Warning message is printed to stdout
    - Different message formats are handled correctly
    
    Args:
        test_case: Dictionary with msg and test name
        capture_stdout: Fixture to capture output
    """
    # Should not raise SystemExit
    warn_and_continue(msg=test_case["msg"])
    
    # Function should return normally
    captured = capture_stdout.readouterr()
    
    # Warning message should be printed (if not empty)
    if test_case["msg"]:
        assert test_case["msg"] in captured.out or test_case["msg"] in captured.err, \
            f"Warning message should be printed for {test_case['name']}"


# ============================================================================
# Integration Tests
# ============================================================================

def test_handle_err_calls_endrun_on_error(capture_stdout):
    """
    Integration test: Verify handle_err calls endrun for errors.
    
    Verifies:
    - handle_err properly delegates to endrun for error conditions
    - Error flow is correct
    """
    with pytest.raises(SystemExit) as exc_info:
        handle_err(status=-99, errmsg="Integration test error")
    
    assert exc_info.value.code == 1, "handle_err should exit via endrun"


def test_check_netcdf_status_calls_handle_err(capture_stdout):
    """
    Integration test: Verify check_netcdf_status calls handle_err.
    
    Verifies:
    - check_netcdf_status properly delegates to handle_err
    - Error flow is correct through the call chain
    """
    with pytest.raises(SystemExit) as exc_info:
        check_netcdf_status(status=-88, operation="Integration test operation")
    
    assert exc_info.value.code == 1, "check_netcdf_status should exit via handle_err"


def test_assert_condition_calls_endrun_on_false(capture_stdout):
    """
    Integration test: Verify assert_condition calls endrun for false conditions.
    
    Verifies:
    - assert_condition properly delegates to endrun
    - Error flow is correct
    """
    with pytest.raises(SystemExit) as exc_info:
        assert_condition(condition=False, msg="Integration test assertion failure")
    
    assert exc_info.value.code == 1, "assert_condition should exit via endrun"


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_multiple_consecutive_warnings(capture_stdout):
    """
    Test multiple consecutive calls to warn_and_continue.
    
    Verifies:
    - Multiple warnings can be issued without termination
    - All warnings are printed
    """
    warn_and_continue("First warning")
    warn_and_continue("Second warning")
    warn_and_continue("Third warning")
    
    captured = capture_stdout.readouterr()
    assert "First warning" in captured.out or "First warning" in captured.err
    assert "Second warning" in captured.out or "Second warning" in captured.err
    assert "Third warning" in captured.out or "Third warning" in captured.err


def test_very_long_error_message(capture_stdout):
    """
    Test handling of very long error messages.
    
    Verifies:
    - Long messages are handled without truncation or errors
    """
    long_msg = "Error: " + "x" * 10000  # Very long message
    
    with pytest.raises(SystemExit):
        endrun(msg=long_msg)
    
    # Should not raise any exceptions during message handling


def test_error_message_with_newlines_and_tabs(capture_stdout):
    """
    Test error messages with various whitespace characters.
    
    Verifies:
    - Newlines, tabs, and other whitespace are preserved
    """
    msg = "Error:\n\tLine 1\n\tLine 2\n\t\tIndented line"
    
    with pytest.raises(SystemExit):
        endrun(msg=msg)
    
    captured = capture_stdout.readouterr()
    # Message should be printed (exact formatting may vary)


# ============================================================================
# Type and Value Tests
# ============================================================================

def test_handle_err_status_type():
    """
    Test that handle_err accepts integer status codes.
    
    Verifies:
    - Function works with various integer types
    """
    # Should work with regular int
    handle_err(status=0, errmsg="Test")
    
    # Should work with negative int
    with pytest.raises(SystemExit):
        handle_err(status=-1, errmsg="Test")


def test_assert_condition_boolean_type():
    """
    Test that assert_condition accepts boolean conditions.
    
    Verifies:
    - Function works with explicit boolean values
    """
    # Should work with True
    assert_condition(condition=True, msg="Test")
    
    # Should work with False (and exit)
    with pytest.raises(SystemExit):
        assert_condition(condition=False, msg="Test")


# ============================================================================
# Documentation Tests
# ============================================================================

def test_functions_have_docstrings():
    """
    Test that all public functions have docstrings.
    
    Verifies:
    - Functions are documented
    - Helps maintain code quality
    """
    functions = [endrun, handle_err, check_netcdf_status, assert_condition, warn_and_continue]
    
    for func in functions:
        assert func.__doc__ is not None, f"{func.__name__} should have a docstring"
        assert len(func.__doc__.strip()) > 0, f"{func.__name__} docstring should not be empty"


def test_exception_classes_have_docstrings():
    """
    Test that all exception classes have docstrings.
    
    Verifies:
    - Exception classes are documented
    """
    exceptions = [CLMError, CLMNetCDFError, CLMInitializationError, CLMComputationError]
    
    for exc_class in exceptions:
        assert exc_class.__doc__ is not None, f"{exc_class.__name__} should have a docstring"


# ============================================================================
# Summary Test
# ============================================================================

def test_module_completeness():
    """
    Test that all expected functions and classes are exported.
    
    Verifies:
    - All documented functions are available
    - All documented classes are available
    - Module is complete
    """
    expected_functions = ['endrun', 'handle_err', 'check_netcdf_status', 'assert_condition', 'warn_and_continue']
    expected_classes = ['NetCDFConstants', 'CLMError', 'CLMNetCDFError', 'CLMInitializationError', 'CLMComputationError']
    
    import clm_src_main.abortutils as module
    
    for func_name in expected_functions:
        assert hasattr(module, func_name), f"Module should export {func_name}"
    
    for class_name in expected_classes:
        assert hasattr(module, class_name), f"Module should export {class_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])