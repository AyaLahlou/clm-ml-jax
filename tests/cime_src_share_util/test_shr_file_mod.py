"""
Comprehensive pytest suite for shr_file_mod file unit management functions.

This module tests the JAX/Python implementation of file unit allocation and
deallocation functions, including:
- create_initial_file_unit_state: Initialize file unit state
- shr_file_get_unit: Allocate file units (auto or specific)
- shr_file_free_unit: Free allocated file units

Tests cover nominal cases, edge cases (boundaries, reserved units), and
error conditions (invalid units, already allocated, not in use).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cime_src_share_util.shr_file_mod import (
    FileUnitState,
    FreeUnitResult,
    create_initial_file_unit_state,
    shr_file_free_unit,
    shr_file_free_unit_jit,
    shr_file_get_unit,
    shr_file_get_unit_jit,
)


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load test data for file unit management functions.
    
    Returns:
        Dictionary containing test cases with inputs and expected outputs
        for all file unit management functions.
    """
    return {
        "test_cases": [
            {
                "name": "test_create_initial_state",
                "function": "create_initial_file_unit_state",
                "inputs": {},
                "expected_output": {
                    "unit_tag_shape": [100],
                    "unit_tag_all_false": True,
                    "unit_tag_dtype": "bool",
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Tests creation of initial file unit state with all units available",
                    "edge_cases": [],
                },
            },
            {
                "name": "test_get_unit_auto_allocation_from_empty",
                "function": "shr_file_get_unit",
                "inputs": {
                    "state": {"unit_tag": [False] * 100},
                    "unit": None,
                },
                "expected_output": {
                    "allocated_unit": 99,
                    "success": True,
                    "unit_tag_at_99": True,
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Tests automatic unit allocation from empty state, should return highest available unit (99)",
                    "edge_cases": [],
                },
            },
            {
                "name": "test_get_specific_unit_available",
                "function": "shr_file_get_unit",
                "inputs": {
                    "state": {"unit_tag": [False] * 100},
                    "unit": 42,
                },
                "expected_output": {
                    "allocated_unit": 42,
                    "success": True,
                    "unit_tag_at_42": True,
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Tests requesting a specific available unit number",
                    "edge_cases": [],
                },
            },
            {
                "name": "test_get_unit_boundary_min",
                "function": "shr_file_get_unit",
                "inputs": {
                    "state": {"unit_tag": [False] * 100},
                    "unit": 10,
                },
                "expected_output": {
                    "allocated_unit": 10,
                    "success": True,
                    "unit_tag_at_10": True,
                },
                "metadata": {
                    "type": "edge",
                    "description": "Tests allocation at minimum valid unit boundary (SHR_FILE_MIN_UNIT = 10)",
                    "edge_cases": ["boundary_min"],
                },
            },
            {
                "name": "test_get_unit_boundary_max",
                "function": "shr_file_get_unit",
                "inputs": {
                    "state": {"unit_tag": [False] * 100},
                    "unit": 99,
                },
                "expected_output": {
                    "allocated_unit": 99,
                    "success": True,
                    "unit_tag_at_99": True,
                },
                "metadata": {
                    "type": "edge",
                    "description": "Tests allocation at maximum valid unit boundary (SHR_FILE_MAX_UNIT = 99)",
                    "edge_cases": ["boundary_max"],
                },
            },
            {
                "name": "test_get_unit_reserved_stdin",
                "function": "shr_file_get_unit",
                "inputs": {
                    "state": {"unit_tag": [False] * 100},
                    "unit": 5,
                },
                "expected_output": {
                    "allocated_unit": -1,
                    "success": False,
                    "unit_tag_unchanged": True,
                },
                "metadata": {
                    "type": "edge",
                    "description": "Tests attempting to allocate reserved unit 5 (stdin), should fail",
                    "edge_cases": ["reserved_unit"],
                },
            },
            {
                "name": "test_get_unit_already_allocated",
                "function": "shr_file_get_unit",
                "inputs": {
                    "state": {
                        "unit_tag": [False] * 75 + [True] + [False] * 24
                    },
                    "unit": 75,
                },
                "expected_output": {
                    "allocated_unit": -1,
                    "success": False,
                    "unit_tag_at_75": True,
                },
                "metadata": {
                    "type": "edge",
                    "description": "Tests attempting to allocate a unit that is already in use",
                    "edge_cases": ["already_allocated"],
                },
            },
            {
                "name": "test_get_unit_auto_with_partial_allocation",
                "function": "shr_file_get_unit",
                "inputs": {
                    "state": {
                        "unit_tag": [False] * 90 + [True] * 10
                    },
                    "unit": None,
                },
                "expected_output": {
                    "allocated_unit": 89,
                    "success": True,
                    "unit_tag_at_89": True,
                },
                "metadata": {
                    "type": "special",
                    "description": "Tests auto-allocation when high units (90-99) are taken, should return 89",
                    "edge_cases": [],
                },
            },
            {
                "name": "test_free_unit_success",
                "function": "shr_file_free_unit",
                "inputs": {
                    "state": {
                        "unit_tag": [False] * 75 + [True] + [False] * 24
                    },
                    "unit": 75,
                },
                "expected_output": {
                    "error_code": 0,
                    "error_msg": "",
                    "unit_tag_at_75": False,
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Tests successfully freeing an allocated unit",
                    "edge_cases": [],
                },
            },
            {
                "name": "test_free_unit_reserved",
                "function": "shr_file_free_unit",
                "inputs": {
                    "state": {
                        "unit_tag": [True] + [False] * 4 + [True, True] + [False] * 93
                    },
                    "unit": 6,
                },
                "expected_output": {
                    "error_code": 2,
                    "error_msg": "reserved unit",
                    "unit_tag_at_6": True,
                },
                "metadata": {
                    "type": "edge",
                    "description": "Tests attempting to free reserved unit 6 (stdout), should fail with error code 2",
                    "edge_cases": ["reserved_unit"],
                },
            },
            {
                "name": "test_free_unit_not_in_use",
                "function": "shr_file_free_unit",
                "inputs": {
                    "state": {"unit_tag": [False] * 100},
                    "unit": 50,
                },
                "expected_output": {
                    "error_code": 3,
                    "error_msg": "unit not in use",
                    "unit_tag_at_50": False,
                },
                "metadata": {
                    "type": "edge",
                    "description": "Tests attempting to free a unit that is not currently in use, should fail with error code 3",
                    "edge_cases": ["not_in_use"],
                },
            },
            {
                "name": "test_free_unit_invalid_negative",
                "function": "shr_file_free_unit",
                "inputs": {
                    "state": {"unit_tag": [False] * 100},
                    "unit": -5,
                },
                "expected_output": {
                    "error_code": 1,
                    "error_msg": "invalid unit number",
                    "unit_tag_unchanged": True,
                },
                "metadata": {
                    "type": "edge",
                    "description": "Tests attempting to free an invalid negative unit number, should fail with error code 1",
                    "edge_cases": ["invalid_negative"],
                },
            },
            {
                "name": "test_free_unit_invalid_too_large",
                "function": "shr_file_free_unit",
                "inputs": {
                    "state": {"unit_tag": [False] * 100},
                    "unit": 150,
                },
                "expected_output": {
                    "error_code": 1,
                    "error_msg": "invalid unit number",
                    "unit_tag_unchanged": True,
                },
                "metadata": {
                    "type": "edge",
                    "description": "Tests attempting to free a unit number beyond valid range (>99), should fail with error code 1",
                    "edge_cases": ["invalid_too_large"],
                },
            },
        ]
    }


def create_file_unit_state(unit_tag_list: List[bool]) -> FileUnitState:
    """
    Helper function to create FileUnitState from a list of booleans.
    
    Args:
        unit_tag_list: List of 100 boolean values indicating unit allocation
        
    Returns:
        FileUnitState namedtuple with unit_tag as JAX array
    """
    return FileUnitState(unit_tag=jnp.array(unit_tag_list, dtype=bool))


# ============================================================================
# Tests for create_initial_file_unit_state
# ============================================================================


def test_create_initial_state_shape():
    """
    Test that create_initial_file_unit_state returns correct shape.
    
    The unit_tag array should have shape (100,) representing units 0-99.
    """
    state = create_initial_file_unit_state()
    
    assert isinstance(state, FileUnitState), "Should return FileUnitState namedtuple"
    assert state.unit_tag.shape == (100,), f"Expected shape (100,), got {state.unit_tag.shape}"


def test_create_initial_state_dtype():
    """
    Test that create_initial_file_unit_state returns correct dtype.
    
    The unit_tag array should be boolean type.
    """
    state = create_initial_file_unit_state()
    
    assert state.unit_tag.dtype == bool, f"Expected bool dtype, got {state.unit_tag.dtype}"


def test_create_initial_state_values():
    """
    Test that create_initial_file_unit_state initializes all units as available.
    
    All elements in unit_tag should be False, indicating no units are in use.
    """
    state = create_initial_file_unit_state()
    
    assert jnp.all(~state.unit_tag), "All units should be available (False) initially"
    assert jnp.sum(state.unit_tag) == 0, "No units should be allocated initially"


# ============================================================================
# Tests for shr_file_get_unit
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        "test_get_unit_auto_allocation_from_empty",
        "test_get_specific_unit_available",
        "test_get_unit_boundary_min",
        "test_get_unit_boundary_max",
    ],
)
def test_shr_file_get_unit_nominal_cases(test_data, test_case):
    """
    Test nominal cases for shr_file_get_unit.
    
    Tests successful allocation scenarios including:
    - Auto-allocation from empty state
    - Specific unit allocation
    - Boundary values (min=10, max=99)
    """
    case = next(tc for tc in test_data["test_cases"] if tc["name"] == test_case)
    
    # Create input state
    state = create_file_unit_state(case["inputs"]["state"]["unit_tag"])
    unit = case["inputs"]["unit"]
    
    # Call function
    new_state, allocated_unit, success = shr_file_get_unit(state, unit)
    
    # Verify return types
    assert isinstance(new_state, FileUnitState), "Should return FileUnitState"
    assert isinstance(allocated_unit, (int, jnp.integer)), "Should return int"
    assert isinstance(success, (bool, jnp.bool_)), "Should return bool"
    
    # Verify expected outputs
    expected = case["expected_output"]
    assert allocated_unit == expected["allocated_unit"], (
        f"Expected allocated_unit={expected['allocated_unit']}, got {allocated_unit}"
    )
    assert bool(success) == expected["success"], (
        f"Expected success={expected['success']}, got {success}"
    )
    
    # Verify specific unit was marked as allocated
    if "unit_tag_at_99" in expected:
        assert bool(new_state.unit_tag[99]) == expected["unit_tag_at_99"], (
            f"Unit 99 should be {expected['unit_tag_at_99']}"
        )
    if "unit_tag_at_42" in expected:
        assert bool(new_state.unit_tag[42]) == expected["unit_tag_at_42"], (
            f"Unit 42 should be {expected['unit_tag_at_42']}"
        )
    if "unit_tag_at_10" in expected:
        assert bool(new_state.unit_tag[10]) == expected["unit_tag_at_10"], (
            f"Unit 10 should be {expected['unit_tag_at_10']}"
        )
    if "unit_tag_at_89" in expected:
        assert bool(new_state.unit_tag[89]) == expected["unit_tag_at_89"], (
            f"Unit 89 should be {expected['unit_tag_at_89']}"
        )


@pytest.mark.parametrize(
    "test_case",
    [
        "test_get_unit_reserved_stdin",
        "test_get_unit_already_allocated",
    ],
)
def test_shr_file_get_unit_edge_cases(test_data, test_case):
    """
    Test edge cases for shr_file_get_unit.
    
    Tests failure scenarios including:
    - Attempting to allocate reserved units (5, 6)
    - Attempting to allocate already-allocated units
    """
    case = next(tc for tc in test_data["test_cases"] if tc["name"] == test_case)
    
    # Create input state
    state = create_file_unit_state(case["inputs"]["state"]["unit_tag"])
    unit = case["inputs"]["unit"]
    
    # Call function
    new_state, allocated_unit, success = shr_file_get_unit(state, unit)
    
    # Verify expected outputs
    expected = case["expected_output"]
    assert allocated_unit == expected["allocated_unit"], (
        f"Expected allocated_unit={expected['allocated_unit']}, got {allocated_unit}"
    )
    assert bool(success) == expected["success"], (
        f"Expected success={expected['success']}, got {success}"
    )
    
    # Verify state unchanged for failed allocations
    if "unit_tag_unchanged" in expected and expected["unit_tag_unchanged"]:
        assert jnp.array_equal(new_state.unit_tag, state.unit_tag), (
            "State should be unchanged for failed allocation"
        )
    
    # Verify specific unit status
    if "unit_tag_at_75" in expected:
        assert bool(new_state.unit_tag[75]) == expected["unit_tag_at_75"], (
            f"Unit 75 should be {expected['unit_tag_at_75']}"
        )


def test_shr_file_get_unit_auto_with_partial_allocation(test_data):
    """
    Test auto-allocation with partial unit allocation.
    
    When high units (90-99) are already allocated, auto-allocation should
    find the next available unit (89).
    """
    case = next(
        tc
        for tc in test_data["test_cases"]
        if tc["name"] == "test_get_unit_auto_with_partial_allocation"
    )
    
    # Create input state
    state = create_file_unit_state(case["inputs"]["state"]["unit_tag"])
    unit = case["inputs"]["unit"]
    
    # Call function
    new_state, allocated_unit, success = shr_file_get_unit(state, unit)
    
    # Verify expected outputs
    expected = case["expected_output"]
    assert allocated_unit == expected["allocated_unit"], (
        f"Expected allocated_unit={expected['allocated_unit']}, got {allocated_unit}"
    )
    assert bool(success) == expected["success"], (
        f"Expected success={expected['success']}, got {success}"
    )
    assert bool(new_state.unit_tag[89]) == expected["unit_tag_at_89"], (
        f"Unit 89 should be {expected['unit_tag_at_89']}"
    )


def test_shr_file_get_unit_state_immutability():
    """
    Test that shr_file_get_unit does not modify the input state.
    
    The function should return a new state object, leaving the original
    state unchanged (functional programming principle).
    """
    state = create_initial_file_unit_state()
    original_unit_tag = state.unit_tag.copy()
    
    # Allocate a unit
    new_state, allocated_unit, success = shr_file_get_unit(state, 50)
    
    # Verify original state unchanged
    assert jnp.array_equal(state.unit_tag, original_unit_tag), (
        "Original state should not be modified"
    )
    
    # Verify new state is different
    assert not jnp.array_equal(new_state.unit_tag, state.unit_tag), (
        "New state should be different from original"
    )


def test_shr_file_get_unit_multiple_allocations():
    """
    Test multiple sequential unit allocations.
    
    Verifies that multiple units can be allocated and tracked correctly.
    """
    state = create_initial_file_unit_state()
    
    # Allocate unit 50
    state, unit1, success1 = shr_file_get_unit(state, 50)
    assert success1 and unit1 == 50, "First allocation should succeed"
    
    # Allocate unit 60
    state, unit2, success2 = shr_file_get_unit(state, 60)
    assert success2 and unit2 == 60, "Second allocation should succeed"
    
    # Verify both units are marked as allocated
    assert bool(state.unit_tag[50]) and bool(state.unit_tag[60]), (
        "Both units should be marked as allocated"
    )
    
    # Try to allocate unit 50 again (should fail)
    state, unit3, success3 = shr_file_get_unit(state, 50)
    assert not success3 and unit3 == -1, "Re-allocation should fail"


def test_shr_file_get_unit_jit_compilation():
    """
    Test that JIT-compiled version produces same results as regular version.
    
    Verifies that shr_file_get_unit_jit behaves identically to shr_file_get_unit.
    """
    state = create_initial_file_unit_state()
    
    # Test with specific unit
    new_state_regular, unit_regular, success_regular = shr_file_get_unit(state, 42)
    new_state_jit, unit_jit, success_jit = shr_file_get_unit_jit(state, 42)
    
    assert unit_regular == unit_jit, "JIT and regular should return same unit"
    assert success_regular == success_jit, "JIT and regular should return same success"
    assert jnp.array_equal(new_state_regular.unit_tag, new_state_jit.unit_tag), (
        "JIT and regular should produce same state"
    )
    
    # Test with auto-allocation
    state2 = create_initial_file_unit_state()
    new_state_regular2, unit_regular2, success_regular2 = shr_file_get_unit(state2, None)
    new_state_jit2, unit_jit2, success_jit2 = shr_file_get_unit_jit(state2, None)
    
    assert unit_regular2 == unit_jit2, "JIT and regular should return same auto-allocated unit"
    assert success_regular2 == success_jit2, "JIT and regular should return same success"
    assert jnp.array_equal(new_state_regular2.unit_tag, new_state_jit2.unit_tag), (
        "JIT and regular should produce same state for auto-allocation"
    )


# ============================================================================
# Tests for shr_file_free_unit
# ============================================================================


def test_shr_file_free_unit_success(test_data):
    """
    Test successful unit deallocation.
    
    Verifies that an allocated unit can be successfully freed.
    """
    case = next(
        tc for tc in test_data["test_cases"] if tc["name"] == "test_free_unit_success"
    )
    
    # Create input state
    state = create_file_unit_state(case["inputs"]["state"]["unit_tag"])
    unit = case["inputs"]["unit"]
    
    # Call function
    result = shr_file_free_unit(state, unit)
    
    # Verify return type
    assert isinstance(result, FreeUnitResult), "Should return FreeUnitResult"
    
    # Verify expected outputs
    expected = case["expected_output"]
    assert result.error_code == expected["error_code"], (
        f"Expected error_code={expected['error_code']}, got {result.error_code}"
    )
    assert result.error_msg == expected["error_msg"], (
        f"Expected error_msg='{expected['error_msg']}', got '{result.error_msg}'"
    )
    assert bool(result.state.unit_tag[75]) == expected["unit_tag_at_75"], (
        f"Unit 75 should be {expected['unit_tag_at_75']}"
    )


@pytest.mark.parametrize(
    "test_case",
    [
        "test_free_unit_reserved",
        "test_free_unit_not_in_use",
        "test_free_unit_invalid_negative",
        "test_free_unit_invalid_too_large",
    ],
)
def test_shr_file_free_unit_error_cases(test_data, test_case):
    """
    Test error cases for shr_file_free_unit.
    
    Tests various failure scenarios:
    - Attempting to free reserved units (0, 5, 6)
    - Attempting to free units not in use
    - Invalid unit numbers (negative, too large)
    """
    case = next(tc for tc in test_data["test_cases"] if tc["name"] == test_case)
    
    # Create input state
    state = create_file_unit_state(case["inputs"]["state"]["unit_tag"])
    unit = case["inputs"]["unit"]
    
    # Call function
    result = shr_file_free_unit(state, unit)
    
    # Verify expected outputs
    expected = case["expected_output"]
    assert result.error_code == expected["error_code"], (
        f"Expected error_code={expected['error_code']}, got {result.error_code}"
    )
    assert result.error_msg == expected["error_msg"], (
        f"Expected error_msg='{expected['error_msg']}', got '{result.error_msg}'"
    )
    
    # Verify state unchanged for errors
    if "unit_tag_unchanged" in expected and expected["unit_tag_unchanged"]:
        assert jnp.array_equal(result.state.unit_tag, state.unit_tag), (
            "State should be unchanged for error cases"
        )
    
    # Verify specific unit status
    if "unit_tag_at_6" in expected:
        assert bool(result.state.unit_tag[6]) == expected["unit_tag_at_6"], (
            f"Unit 6 should be {expected['unit_tag_at_6']}"
        )
    if "unit_tag_at_50" in expected:
        assert bool(result.state.unit_tag[50]) == expected["unit_tag_at_50"], (
            f"Unit 50 should be {expected['unit_tag_at_50']}"
        )


def test_shr_file_free_unit_state_immutability():
    """
    Test that shr_file_free_unit does not modify the input state.
    
    The function should return a new state object, leaving the original
    state unchanged (functional programming principle).
    """
    # Create state with unit 50 allocated
    state = create_file_unit_state([False] * 50 + [True] + [False] * 49)
    original_unit_tag = state.unit_tag.copy()
    
    # Free the unit
    result = shr_file_free_unit(state, 50)
    
    # Verify original state unchanged
    assert jnp.array_equal(state.unit_tag, original_unit_tag), (
        "Original state should not be modified"
    )
    
    # Verify new state is different
    assert not jnp.array_equal(result.state.unit_tag, state.unit_tag), (
        "New state should be different from original"
    )


def test_shr_file_free_unit_allocate_free_cycle():
    """
    Test allocate-free-allocate cycle.
    
    Verifies that a unit can be allocated, freed, and allocated again.
    """
    state = create_initial_file_unit_state()
    
    # Allocate unit 50
    state, unit, success = shr_file_get_unit(state, 50)
    assert success and unit == 50, "Allocation should succeed"
    assert bool(state.unit_tag[50]), "Unit 50 should be allocated"
    
    # Free unit 50
    result = shr_file_free_unit(state, 50)
    assert result.error_code == 0, "Free should succeed"
    assert not bool(result.state.unit_tag[50]), "Unit 50 should be freed"
    
    # Allocate unit 50 again
    state2, unit2, success2 = shr_file_get_unit(result.state, 50)
    assert success2 and unit2 == 50, "Re-allocation should succeed"
    assert bool(state2.unit_tag[50]), "Unit 50 should be allocated again"


def test_shr_file_free_unit_jit_compilation():
    """
    Test that JIT-compiled version produces same results as regular version.
    
    Verifies that shr_file_free_unit_jit behaves identically to shr_file_free_unit.
    """
    # Create state with unit 50 allocated
    state = create_file_unit_state([False] * 50 + [True] + [False] * 49)
    
    # Test regular version
    result_regular = shr_file_free_unit(state, 50)
    
    # Test JIT version
    result_jit = shr_file_free_unit_jit(state, 50)
    
    # Compare results
    assert result_regular.error_code == result_jit.error_code, (
        "JIT and regular should return same error_code"
    )
    assert result_regular.error_msg == result_jit.error_msg, (
        "JIT and regular should return same error_msg"
    )
    assert jnp.array_equal(result_regular.state.unit_tag, result_jit.state.unit_tag), (
        "JIT and regular should produce same state"
    )


def test_shr_file_free_unit_error_code_meanings():
    """
    Test that error codes have correct meanings.
    
    Verifies the error code mapping:
    - 0: success
    - 1: invalid unit number
    - 2: reserved unit
    - 3: unit not in use
    """
    state = create_file_unit_state([False] * 50 + [True] + [False] * 49)
    
    # Error code 0: success
    result = shr_file_free_unit(state, 50)
    assert result.error_code == 0, "Should return error_code 0 for success"
    assert result.error_msg == "", "Should return empty error_msg for success"
    
    # Error code 1: invalid unit number
    result = shr_file_free_unit(state, -5)
    assert result.error_code == 1, "Should return error_code 1 for invalid unit"
    assert "invalid" in result.error_msg.lower(), "Error message should mention 'invalid'"
    
    result = shr_file_free_unit(state, 150)
    assert result.error_code == 1, "Should return error_code 1 for out-of-range unit"
    
    # Error code 2: reserved unit
    result = shr_file_free_unit(state, 5)
    assert result.error_code == 2, "Should return error_code 2 for reserved unit"
    assert "reserved" in result.error_msg.lower(), "Error message should mention 'reserved'"
    
    # Error code 3: unit not in use
    result = shr_file_free_unit(state, 25)
    assert result.error_code == 3, "Should return error_code 3 for unit not in use"
    assert "not in use" in result.error_msg.lower(), "Error message should mention 'not in use'"


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_unit_lifecycle():
    """
    Test complete lifecycle of file unit management.
    
    Tests initialization, multiple allocations, freeing, and re-allocation
    in a realistic usage scenario.
    """
    # Initialize state
    state = create_initial_file_unit_state()
    assert jnp.sum(state.unit_tag) == 0, "Should start with no allocated units"
    
    # Allocate several units
    units_to_allocate = [15, 25, 35, 45, 55]
    for unit_num in units_to_allocate:
        state, allocated, success = shr_file_get_unit(state, unit_num)
        assert success and allocated == unit_num, f"Should allocate unit {unit_num}"
    
    assert jnp.sum(state.unit_tag) == len(units_to_allocate), (
        f"Should have {len(units_to_allocate)} units allocated"
    )
    
    # Free some units
    units_to_free = [25, 45]
    for unit_num in units_to_free:
        result = shr_file_free_unit(state, unit_num)
        assert result.error_code == 0, f"Should successfully free unit {unit_num}"
        state = result.state
    
    assert jnp.sum(state.unit_tag) == len(units_to_allocate) - len(units_to_free), (
        f"Should have {len(units_to_allocate) - len(units_to_free)} units allocated"
    )
    
    # Re-allocate freed units
    for unit_num in units_to_free:
        state, allocated, success = shr_file_get_unit(state, unit_num)
        assert success and allocated == unit_num, f"Should re-allocate unit {unit_num}"
    
    assert jnp.sum(state.unit_tag) == len(units_to_allocate), (
        f"Should have {len(units_to_allocate)} units allocated again"
    )


def test_reserved_units_protection():
    """
    Test that reserved units (0, 5, 6) are properly protected.
    
    Verifies that reserved units cannot be allocated or freed.
    """
    state = create_initial_file_unit_state()
    reserved_units = [0, 5, 6]
    
    for unit_num in reserved_units:
        # Try to allocate reserved unit
        new_state, allocated, success = shr_file_get_unit(state, unit_num)
        assert not success, f"Should not allocate reserved unit {unit_num}"
        assert allocated == -1, f"Should return -1 for reserved unit {unit_num}"
        
        # Try to free reserved unit
        result = shr_file_free_unit(state, unit_num)
        assert result.error_code in [1, 2], (
            f"Should return error for reserved unit {unit_num}"
        )


def test_boundary_units():
    """
    Test allocation and freeing at boundary values.
    
    Tests units at the minimum (10) and maximum (99) valid range.
    """
    state = create_initial_file_unit_state()
    
    # Test minimum boundary (10)
    state, unit, success = shr_file_get_unit(state, 10)
    assert success and unit == 10, "Should allocate minimum unit 10"
    
    result = shr_file_free_unit(state, 10)
    assert result.error_code == 0, "Should free minimum unit 10"
    
    # Test maximum boundary (99)
    state = result.state
    state, unit, success = shr_file_get_unit(state, 99)
    assert success and unit == 99, "Should allocate maximum unit 99"
    
    result = shr_file_free_unit(state, 99)
    assert result.error_code == 0, "Should free maximum unit 99"


def test_auto_allocation_exhaustion():
    """
    Test auto-allocation behavior when units are exhausted.
    
    Allocates all valid units (10-99) and verifies that subsequent
    auto-allocation fails gracefully.
    """
    state = create_initial_file_unit_state()
    
    # Allocate all valid units (10-99)
    for unit_num in range(10, 100):
        state, allocated, success = shr_file_get_unit(state, unit_num)
        assert success, f"Should allocate unit {unit_num}"
    
    # Try auto-allocation when all units are taken
    state, allocated, success = shr_file_get_unit(state, None)
    assert not success, "Auto-allocation should fail when all units are taken"
    assert allocated == -1, "Should return -1 when no units available"


def test_state_consistency_after_operations():
    """
    Test that state remains consistent after various operations.
    
    Verifies that the unit_tag array accurately reflects the allocation
    status after multiple operations.
    """
    state = create_initial_file_unit_state()
    
    # Allocate units 20, 30, 40
    allocated_units = []
    for unit_num in [20, 30, 40]:
        state, allocated, success = shr_file_get_unit(state, unit_num)
        if success:
            allocated_units.append(allocated)
    
    # Verify only allocated units are marked
    for i in range(100):
        if i in allocated_units:
            assert bool(state.unit_tag[i]), f"Unit {i} should be marked as allocated"
        else:
            assert not bool(state.unit_tag[i]), f"Unit {i} should not be marked as allocated"
    
    # Free unit 30
    result = shr_file_free_unit(state, 30)
    state = result.state
    allocated_units.remove(30)
    
    # Verify state consistency after freeing
    for i in range(100):
        if i in allocated_units:
            assert bool(state.unit_tag[i]), f"Unit {i} should still be marked as allocated"
        else:
            assert not bool(state.unit_tag[i]), f"Unit {i} should not be marked as allocated"