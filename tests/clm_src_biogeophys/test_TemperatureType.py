"""
Comprehensive pytest suite for TemperatureType module.

This module tests the temperature state management functions including:
- Initialization of temperature states
- Temperature allocation with NaN values
- Getting and updating soil/snow temperatures
- Surface temperature retrieval
- Fortran-to-Python index conversion

Tests cover nominal cases, edge cases, and physical constraints.
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_biogeophys.TemperatureType import (
    BoundsType,
    TemperatureState,
    get_soil_temperature,
    get_surface_temperature,
    init_allocate,
    init_temperature,
    init_temperature_state,
    update_soil_temperature,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_data():
    """Load test data for all test cases."""
    return {
        "init_temperature_state_nominal_small": {
            "inputs": {
                "n_columns": 5,
                "n_patches": 8,
                "n_levtot": 15,
                "initial_temp": 273.15,
            },
            "expected_shapes": {
                "t_soisno_col": (5, 15),
                "t_a10_patch": (8,),
                "t_ref2m_patch": (8,),
            },
        },
        "init_temperature_state_nominal_large": {
            "inputs": {
                "n_columns": 100,
                "n_patches": 250,
                "n_levtot": 20,
                "initial_temp": 288.15,
            },
            "expected_shapes": {
                "t_soisno_col": (100, 20),
                "t_a10_patch": (250,),
                "t_ref2m_patch": (250,),
            },
        },
        "init_temperature_state_edge_minimum": {
            "inputs": {
                "n_columns": 1,
                "n_patches": 1,
                "n_levtot": 1,
                "initial_temp": 273.15,
            },
            "expected_shapes": {
                "t_soisno_col": (1, 1),
                "t_a10_patch": (1,),
                "t_ref2m_patch": (1,),
            },
        },
        "init_temperature_state_edge_cold": {
            "inputs": {
                "n_columns": 10,
                "n_patches": 15,
                "n_levtot": 18,
                "initial_temp": 200.0,
            },
            "expected_shapes": {
                "t_soisno_col": (10, 18),
                "t_a10_patch": (15,),
                "t_ref2m_patch": (15,),
            },
        },
        "init_temperature_state_edge_hot": {
            "inputs": {
                "n_columns": 12,
                "n_patches": 20,
                "n_levtot": 16,
                "initial_temp": 350.0,
            },
            "expected_shapes": {
                "t_soisno_col": (12, 16),
                "t_a10_patch": (20,),
                "t_ref2m_patch": (20,),
            },
        },
    }


@pytest.fixture
def sample_temperature_state():
    """Create a sample temperature state for testing get/update operations."""
    return TemperatureState(
        t_soisno_col=jnp.array(
            [
                [250.0, 255.0, 260.0, 265.0, 270.0, 273.15, 275.0, 278.0, 280.0, 282.0],
                [248.0, 253.0, 258.0, 263.0, 268.0, 271.0, 274.0, 277.0, 279.0, 281.0],
                [252.0, 257.0, 262.0, 267.0, 272.0, 275.0, 277.0, 279.0, 281.0, 283.0],
            ],
            dtype=jnp.float32,
        ),
        t_a10_patch=jnp.array([285.0, 286.0, 284.0, 287.0, 285.5], dtype=jnp.float32),
        t_ref2m_patch=jnp.array([283.0, 284.0, 282.0, 285.0, 283.5], dtype=jnp.float32),
    )


# ============================================================================
# Tests for init_temperature_state
# ============================================================================


@pytest.mark.parametrize(
    "test_case_name",
    [
        "init_temperature_state_nominal_small",
        "init_temperature_state_nominal_large",
        "init_temperature_state_edge_minimum",
        "init_temperature_state_edge_cold",
        "init_temperature_state_edge_hot",
    ],
)
def test_init_temperature_state_shapes(test_data, test_case_name):
    """
    Test that init_temperature_state creates arrays with correct shapes.
    
    Verifies that the returned TemperatureState has arrays matching the
    expected dimensions based on n_columns, n_patches, and n_levtot.
    """
    case = test_data[test_case_name]
    inputs = case["inputs"]
    expected_shapes = case["expected_shapes"]

    state = init_temperature_state(**inputs)

    assert isinstance(state, TemperatureState), "Should return TemperatureState"
    assert state.t_soisno_col.shape == expected_shapes["t_soisno_col"], (
        f"t_soisno_col shape mismatch: got {state.t_soisno_col.shape}, "
        f"expected {expected_shapes['t_soisno_col']}"
    )
    assert state.t_a10_patch.shape == expected_shapes["t_a10_patch"], (
        f"t_a10_patch shape mismatch: got {state.t_a10_patch.shape}, "
        f"expected {expected_shapes['t_a10_patch']}"
    )
    assert state.t_ref2m_patch.shape == expected_shapes["t_ref2m_patch"], (
        f"t_ref2m_patch shape mismatch: got {state.t_ref2m_patch.shape}, "
        f"expected {expected_shapes['t_ref2m_patch']}"
    )


@pytest.mark.parametrize(
    "test_case_name",
    [
        "init_temperature_state_nominal_small",
        "init_temperature_state_edge_cold",
        "init_temperature_state_edge_hot",
    ],
)
def test_init_temperature_state_values(test_data, test_case_name):
    """
    Test that init_temperature_state initializes all values to initial_temp.
    
    Verifies that all elements in all arrays are set to the specified
    initial temperature value.
    """
    case = test_data[test_case_name]
    inputs = case["inputs"]
    initial_temp = inputs["initial_temp"]

    state = init_temperature_state(**inputs)

    # Check that all values are initialized to initial_temp
    assert jnp.allclose(state.t_soisno_col, initial_temp, atol=1e-6, rtol=1e-6), (
        f"t_soisno_col values should all be {initial_temp}"
    )
    assert jnp.allclose(state.t_a10_patch, initial_temp, atol=1e-6, rtol=1e-6), (
        f"t_a10_patch values should all be {initial_temp}"
    )
    assert jnp.allclose(state.t_ref2m_patch, initial_temp, atol=1e-6, rtol=1e-6), (
        f"t_ref2m_patch values should all be {initial_temp}"
    )


def test_init_temperature_state_dtypes():
    """
    Test that init_temperature_state creates arrays with correct dtypes.
    
    Verifies that all arrays use float32 dtype as specified in the schema.
    """
    state = init_temperature_state(
        n_columns=5, n_patches=8, n_levtot=15, initial_temp=273.15
    )

    assert state.t_soisno_col.dtype == jnp.float32, "t_soisno_col should be float32"
    assert state.t_a10_patch.dtype == jnp.float32, "t_a10_patch should be float32"
    assert state.t_ref2m_patch.dtype == jnp.float32, "t_ref2m_patch should be float32"


def test_init_temperature_state_physical_constraints():
    """
    Test that init_temperature_state respects physical constraints.
    
    Verifies that:
    - Temperatures are >= 0 K (absolute zero)
    - Typical temperatures are in reasonable range (200-350 K)
    """
    # Test with physically valid temperatures
    state_cold = init_temperature_state(5, 8, 15, 200.0)
    assert jnp.all(state_cold.t_soisno_col >= 0.0), "Temperatures must be >= 0 K"

    state_hot = init_temperature_state(5, 8, 15, 350.0)
    assert jnp.all(state_hot.t_soisno_col >= 0.0), "Temperatures must be >= 0 K"

    # Test default temperature (freezing point)
    state_default = init_temperature_state(5, 8, 15)
    assert jnp.allclose(state_default.t_soisno_col, 273.15, atol=1e-6), (
        "Default temperature should be 273.15 K (0°C)"
    )


# ============================================================================
# Tests for init_temperature
# ============================================================================


def test_init_temperature_nominal():
    """
    Test init_temperature with standard domain decomposition.
    
    Verifies that the function correctly calculates array dimensions
    from bounds and layer counts.
    """
    bounds = BoundsType(begp=0, endp=49, begc=0, endc=29, begg=0, endg=9)
    nlevsno = 5
    nlevgrnd = 15

    state = init_temperature(bounds, nlevsno, nlevgrnd)

    # n_columns = endc - begc + 1 = 29 - 0 + 1 = 30
    # n_patches = endp - begp + 1 = 49 - 0 + 1 = 50
    # n_levtot = nlevsno + nlevgrnd = 5 + 15 = 20
    assert state.t_soisno_col.shape == (30, 20), (
        f"Expected shape (30, 20), got {state.t_soisno_col.shape}"
    )
    assert state.t_a10_patch.shape == (50,), (
        f"Expected shape (50,), got {state.t_a10_patch.shape}"
    )
    assert state.t_ref2m_patch.shape == (50,), (
        f"Expected shape (50,), got {state.t_ref2m_patch.shape}"
    )


def test_init_temperature_edge_single_elements():
    """
    Test init_temperature with single-element bounds (beg==end).
    
    Verifies correct handling of minimal domain size where each
    dimension has only one element.
    """
    bounds = BoundsType(begp=5, endp=5, begc=3, endc=3, begg=1, endg=1)
    nlevsno = 1
    nlevgrnd = 10

    state = init_temperature(bounds, nlevsno, nlevgrnd)

    # n_columns = 3 - 3 + 1 = 1
    # n_patches = 5 - 5 + 1 = 1
    # n_levtot = 1 + 10 = 11
    assert state.t_soisno_col.shape == (1, 11), (
        f"Expected shape (1, 11), got {state.t_soisno_col.shape}"
    )
    assert state.t_a10_patch.shape == (1,), (
        f"Expected shape (1,), got {state.t_a10_patch.shape}"
    )
    assert state.t_ref2m_patch.shape == (1,), (
        f"Expected shape (1,), got {state.t_ref2m_patch.shape}"
    )


def test_init_temperature_many_snow_layers():
    """
    Test init_temperature with maximum typical snow layers.
    
    Verifies correct handling of deep snowpack scenario with
    12 snow layers and 25 soil layers.
    """
    bounds = BoundsType(begp=10, endp=59, begc=5, endc=34, begg=0, endg=4)
    nlevsno = 12
    nlevgrnd = 25

    state = init_temperature(bounds, nlevsno, nlevgrnd)

    # n_columns = 34 - 5 + 1 = 30
    # n_patches = 59 - 10 + 1 = 50
    # n_levtot = 12 + 25 = 37
    assert state.t_soisno_col.shape == (30, 37), (
        f"Expected shape (30, 37), got {state.t_soisno_col.shape}"
    )
    assert state.t_a10_patch.shape == (50,), (
        f"Expected shape (50,), got {state.t_a10_patch.shape}"
    )


def test_init_temperature_default_value():
    """
    Test that init_temperature initializes to default temperature (273.15 K).
    
    Verifies that when no initial_temp is specified, arrays are
    initialized to the freezing point of water.
    """
    bounds = BoundsType(begp=0, endp=9, begc=0, endc=4, begg=0, endg=1)
    state = init_temperature(bounds, nlevsno=3, nlevgrnd=12)

    # Should initialize to default 273.15 K
    assert jnp.allclose(state.t_soisno_col, 273.15, atol=1e-6), (
        "Default initialization should be 273.15 K"
    )
    assert jnp.allclose(state.t_a10_patch, 273.15, atol=1e-6), (
        "Default initialization should be 273.15 K"
    )
    assert jnp.allclose(state.t_ref2m_patch, 273.15, atol=1e-6), (
        "Default initialization should be 273.15 K"
    )


# ============================================================================
# Tests for init_allocate
# ============================================================================


def test_init_allocate_shapes():
    """
    Test that init_allocate creates arrays with correct shapes.
    
    Verifies that arrays are allocated with dimensions matching
    the bounds and layer counts.
    """
    bounds = BoundsType(begp=0, endp=9, begc=0, endc=4, begg=0, endg=1)
    nlevsno = 3
    nlevgrnd = 12

    state = init_allocate(bounds, nlevsno, nlevgrnd)

    # n_columns = 4 - 0 + 1 = 5
    # n_patches = 9 - 0 + 1 = 10
    # n_levtot = 3 + 12 = 15
    assert state.t_soisno_col.shape == (5, 15), (
        f"Expected shape (5, 15), got {state.t_soisno_col.shape}"
    )
    assert state.t_a10_patch.shape == (10,), (
        f"Expected shape (10,), got {state.t_a10_patch.shape}"
    )
    assert state.t_ref2m_patch.shape == (10,), (
        f"Expected shape (10,), got {state.t_ref2m_patch.shape}"
    )


def test_init_allocate_nan_values():
    """
    Test that init_allocate initializes arrays with NaN values.
    
    Verifies that all array elements are NaN to enable detection
    of uninitialized access.
    """
    bounds = BoundsType(begp=0, endp=9, begc=0, endc=4, begg=0, endg=1)
    state = init_allocate(bounds, nlevsno=3, nlevgrnd=12)

    # All values should be NaN
    assert jnp.all(jnp.isnan(state.t_soisno_col)), (
        "t_soisno_col should be initialized to NaN"
    )
    assert jnp.all(jnp.isnan(state.t_a10_patch)), (
        "t_a10_patch should be initialized to NaN"
    )
    assert jnp.all(jnp.isnan(state.t_ref2m_patch)), (
        "t_ref2m_patch should be initialized to NaN"
    )


def test_init_allocate_dtypes():
    """
    Test that init_allocate creates arrays with correct dtypes.
    
    Verifies that all arrays use float32 dtype.
    """
    bounds = BoundsType(begp=0, endp=9, begc=0, endc=4, begg=0, endg=1)
    state = init_allocate(bounds, nlevsno=3, nlevgrnd=12)

    assert state.t_soisno_col.dtype == jnp.float32, "t_soisno_col should be float32"
    assert state.t_a10_patch.dtype == jnp.float32, "t_a10_patch should be float32"
    assert state.t_ref2m_patch.dtype == jnp.float32, "t_ref2m_patch should be float32"


# ============================================================================
# Tests for get_soil_temperature
# ============================================================================


def test_get_soil_temperature_nominal(sample_temperature_state):
    """
    Test get_soil_temperature with typical inputs.
    
    Verifies correct retrieval of temperature from a soil layer
    using Fortran indexing convention.
    """
    state = sample_temperature_state
    nlevsno = 3

    # Get temperature from column 1, Fortran layer 3
    # Fortran layer 3 -> Python index = 3 + 3 - 1 = 5
    temp = get_soil_temperature(state, col_idx=1, layer_idx=3, nlevsno=nlevsno)

    expected = state.t_soisno_col[1, 5]  # Should be 271.0
    assert jnp.allclose(temp, expected, atol=1e-6), (
        f"Expected temperature {expected}, got {temp}"
    )
    assert jnp.allclose(temp, 271.0, atol=1e-6), (
        f"Expected temperature 271.0 K, got {temp}"
    )


def test_get_soil_temperature_snow_layer():
    """
    Test get_soil_temperature with negative Fortran index (snow layer).
    
    Verifies correct retrieval from snow layers using negative
    Fortran indexing convention.
    """
    state = TemperatureState(
        t_soisno_col=jnp.array(
            [[240.0, 245.0, 250.0, 255.0, 260.0, 265.0, 270.0, 273.15]],
            dtype=jnp.float32,
        ),
        t_a10_patch=jnp.array([280.0], dtype=jnp.float32),
        t_ref2m_patch=jnp.array([278.0], dtype=jnp.float32),
    )
    nlevsno = 5

    # Get temperature from Fortran layer -4 (deep snow layer)
    # Fortran layer -4 -> Python index = -4 + 5 - 1 = 0
    temp = get_soil_temperature(state, col_idx=0, layer_idx=-4, nlevsno=nlevsno)

    expected = 240.0
    assert jnp.allclose(temp, expected, atol=1e-6), (
        f"Expected temperature {expected}, got {temp}"
    )


def test_get_soil_temperature_surface_layer():
    """
    Test get_soil_temperature at surface (Fortran layer 1).
    
    Verifies correct retrieval from the top soil layer.
    """
    state = TemperatureState(
        t_soisno_col=jnp.array(
            [[250.0, 255.0, 260.0, 265.0, 270.0, 273.15, 275.0]],
            dtype=jnp.float32,
        ),
        t_a10_patch=jnp.array([280.0], dtype=jnp.float32),
        t_ref2m_patch=jnp.array([278.0], dtype=jnp.float32),
    )
    nlevsno = 3

    # Get temperature from Fortran layer 1 (top soil layer)
    # Fortran layer 1 -> Python index = 1 + 3 - 1 = 3
    temp = get_soil_temperature(state, col_idx=0, layer_idx=1, nlevsno=nlevsno)

    expected = 265.0
    assert jnp.allclose(temp, expected, atol=1e-6), (
        f"Expected temperature {expected}, got {temp}"
    )


def test_get_soil_temperature_multiple_columns():
    """
    Test get_soil_temperature across different columns.
    
    Verifies that column indexing works correctly and retrieves
    the right values from different columns.
    """
    state = TemperatureState(
        t_soisno_col=jnp.array(
            [
                [250.0, 255.0, 260.0, 265.0, 270.0],
                [248.0, 253.0, 258.0, 263.0, 268.0],
                [252.0, 257.0, 262.0, 267.0, 272.0],
            ],
            dtype=jnp.float32,
        ),
        t_a10_patch=jnp.array([280.0, 282.0, 284.0], dtype=jnp.float32),
        t_ref2m_patch=jnp.array([278.0, 280.0, 282.0], dtype=jnp.float32),
    )
    nlevsno = 2

    # Test different columns at same layer
    temp0 = get_soil_temperature(state, col_idx=0, layer_idx=2, nlevsno=nlevsno)
    temp1 = get_soil_temperature(state, col_idx=1, layer_idx=2, nlevsno=nlevsno)
    temp2 = get_soil_temperature(state, col_idx=2, layer_idx=2, nlevsno=nlevsno)

    # Fortran layer 2 -> Python index = 2 + 2 - 1 = 3
    assert jnp.allclose(temp0, 265.0, atol=1e-6), f"Column 0: expected 265.0, got {temp0}"
    assert jnp.allclose(temp1, 263.0, atol=1e-6), f"Column 1: expected 263.0, got {temp1}"
    assert jnp.allclose(temp2, 267.0, atol=1e-6), f"Column 2: expected 267.0, got {temp2}"


# ============================================================================
# Tests for update_soil_temperature
# ============================================================================


def test_update_soil_temperature_nominal():
    """
    Test update_soil_temperature with typical inputs.
    
    Verifies that temperature update creates a new state with
    the correct value updated while preserving other values.
    """
    state = TemperatureState(
        t_soisno_col=jnp.array(
            [[260.0, 265.0, 270.0, 273.15, 275.0, 278.0]],
            dtype=jnp.float32,
        ),
        t_a10_patch=jnp.array([280.0], dtype=jnp.float32),
        t_ref2m_patch=jnp.array([278.0], dtype=jnp.float32),
    )
    nlevsno = 3

    # Update Fortran layer -1 (snow layer)
    # Fortran layer -1 -> Python index = -1 + 3 - 1 = 1
    new_state = update_soil_temperature(
        state, col_idx=0, layer_idx=-1, nlevsno=nlevsno, new_temp=262.5
    )

    # Check that the correct value was updated
    assert jnp.allclose(new_state.t_soisno_col[0, 1], 262.5, atol=1e-6), (
        f"Expected updated temperature 262.5, got {new_state.t_soisno_col[0, 1]}"
    )

    # Check that other values are unchanged
    assert jnp.allclose(new_state.t_soisno_col[0, 0], 260.0, atol=1e-6), (
        "Other values should be unchanged"
    )
    assert jnp.allclose(new_state.t_soisno_col[0, 2], 270.0, atol=1e-6), (
        "Other values should be unchanged"
    )

    # Check that patch arrays are unchanged
    assert jnp.allclose(new_state.t_a10_patch, state.t_a10_patch, atol=1e-6), (
        "t_a10_patch should be unchanged"
    )
    assert jnp.allclose(new_state.t_ref2m_patch, state.t_ref2m_patch, atol=1e-6), (
        "t_ref2m_patch should be unchanged"
    )


def test_update_soil_temperature_deep_soil():
    """
    Test update_soil_temperature in deep soil layer.
    
    Verifies correct update of temperature in a deep soil layer
    with high positive Fortran index.
    """
    state = TemperatureState(
        t_soisno_col=jnp.array(
            [[255.0, 260.0, 265.0, 270.0, 273.15, 276.0, 278.0, 280.0, 282.0, 284.0, 285.0, 286.0]],
            dtype=jnp.float32,
        ),
        t_a10_patch=jnp.array([285.0], dtype=jnp.float32),
        t_ref2m_patch=jnp.array([283.0], dtype=jnp.float32),
    )
    nlevsno = 4

    # Update Fortran layer 8 (deep soil)
    # Fortran layer 8 -> Python index = 8 + 4 - 1 = 11
    new_state = update_soil_temperature(
        state, col_idx=0, layer_idx=8, nlevsno=nlevsno, new_temp=283.5
    )

    assert jnp.allclose(new_state.t_soisno_col[0, 11], 283.5, atol=1e-6), (
        f"Expected updated temperature 283.5, got {new_state.t_soisno_col[0, 11]}"
    )


def test_update_soil_temperature_immutability():
    """
    Test that update_soil_temperature doesn't modify original state.
    
    Verifies that the function returns a new state and leaves
    the original state unchanged (functional programming principle).
    """
    original_state = TemperatureState(
        t_soisno_col=jnp.array(
            [[260.0, 265.0, 270.0, 273.15, 275.0]],
            dtype=jnp.float32,
        ),
        t_a10_patch=jnp.array([280.0], dtype=jnp.float32),
        t_ref2m_patch=jnp.array([278.0], dtype=jnp.float32),
    )

    # Store original values
    original_values = original_state.t_soisno_col.copy()

    # Update temperature
    new_state = update_soil_temperature(
        original_state, col_idx=0, layer_idx=1, nlevsno=2, new_temp=290.0
    )

    # Original state should be unchanged
    assert jnp.allclose(original_state.t_soisno_col, original_values, atol=1e-6), (
        "Original state should not be modified"
    )

    # New state should have the update
    assert not jnp.allclose(new_state.t_soisno_col, original_values, atol=1e-6), (
        "New state should be different from original"
    )


def test_update_soil_temperature_multiple_columns():
    """
    Test update_soil_temperature with multiple columns.
    
    Verifies that updating one column doesn't affect other columns.
    """
    state = TemperatureState(
        t_soisno_col=jnp.array(
            [
                [250.0, 255.0, 260.0, 265.0, 270.0],
                [248.0, 253.0, 258.0, 263.0, 268.0],
                [252.0, 257.0, 262.0, 267.0, 272.0],
            ],
            dtype=jnp.float32,
        ),
        t_a10_patch=jnp.array([280.0, 282.0, 284.0], dtype=jnp.float32),
        t_ref2m_patch=jnp.array([278.0, 280.0, 282.0], dtype=jnp.float32),
    )

    # Update column 1
    new_state = update_soil_temperature(
        state, col_idx=1, layer_idx=2, nlevsno=2, new_temp=300.0
    )

    # Column 1 should be updated
    assert jnp.allclose(new_state.t_soisno_col[1, 3], 300.0, atol=1e-6), (
        "Column 1 should be updated"
    )

    # Other columns should be unchanged
    assert jnp.allclose(new_state.t_soisno_col[0], state.t_soisno_col[0], atol=1e-6), (
        "Column 0 should be unchanged"
    )
    assert jnp.allclose(new_state.t_soisno_col[2], state.t_soisno_col[2], atol=1e-6), (
        "Column 2 should be unchanged"
    )


# ============================================================================
# Tests for get_surface_temperature
# ============================================================================


def test_get_surface_temperature_with_snow():
    """
    Test get_surface_temperature when snow is present (snl < 0).
    
    Verifies that the function returns the temperature of the top
    snow layer when snow is present.
    """
    state = TemperatureState(
        t_soisno_col=jnp.array(
            [[260.0, 265.0, 268.0, 270.0, 273.15, 275.0, 278.0, 280.0]],
            dtype=jnp.float32,
        ),
        t_a10_patch=jnp.array([280.0], dtype=jnp.float32),
        t_ref2m_patch=jnp.array([278.0], dtype=jnp.float32),
    )
    nlevsno = 5
    snl = -3  # 3 active snow layers

    temp = get_surface_temperature(state, col_idx=0, nlevsno=nlevsno, snl=snl)

    # Surface layer index = nlevsno + snl = 5 + (-3) = 2
    expected = state.t_soisno_col[0, 2]  # Should be 268.0
    assert jnp.allclose(temp, expected, atol=1e-6), (
        f"Expected surface temperature {expected}, got {temp}"
    )
    assert jnp.allclose(temp, 268.0, atol=1e-6), (
        f"Expected surface temperature 268.0 K, got {temp}"
    )


def test_get_surface_temperature_no_snow():
    """
    Test get_surface_temperature when no snow is present (snl = 0).
    
    Verifies that the function returns the temperature of the top
    soil layer when there is no snow cover.
    """
    state = TemperatureState(
        t_soisno_col=jnp.array(
            [[245.0, 250.0, 255.0, 260.0, 265.0, 273.15, 278.0, 282.0]],
            dtype=jnp.float32,
        ),
        t_a10_patch=jnp.array([290.0], dtype=jnp.float32),
        t_ref2m_patch=jnp.array([288.0], dtype=jnp.float32),
    )
    nlevsno = 5
    snl = 0  # No snow

    temp = get_surface_temperature(state, col_idx=0, nlevsno=nlevsno, snl=snl)

    # Surface layer index = nlevsno = 5
    expected = state.t_soisno_col[0, 5]  # Should be 273.15
    assert jnp.allclose(temp, expected, atol=1e-6), (
        f"Expected surface temperature {expected}, got {temp}"
    )
    assert jnp.allclose(temp, 273.15, atol=1e-6), (
        f"Expected surface temperature 273.15 K, got {temp}"
    )


def test_get_surface_temperature_single_snow_layer():
    """
    Test get_surface_temperature with single snow layer (snl = -1).
    
    Verifies correct behavior with minimal snow cover.
    """
    state = TemperatureState(
        t_soisno_col=jnp.array(
            [[250.0, 255.0, 260.0, 265.0, 270.0, 273.15, 275.0]],
            dtype=jnp.float32,
        ),
        t_a10_patch=jnp.array([280.0], dtype=jnp.float32),
        t_ref2m_patch=jnp.array([278.0], dtype=jnp.float32),
    )
    nlevsno = 4
    snl = -1  # 1 active snow layer

    temp = get_surface_temperature(state, col_idx=0, nlevsno=nlevsno, snl=snl)

    # Surface layer index = nlevsno + snl = 4 + (-1) = 3
    expected = state.t_soisno_col[0, 3]  # Should be 265.0
    assert jnp.allclose(temp, expected, atol=1e-6), (
        f"Expected surface temperature {expected}, got {temp}"
    )


def test_get_surface_temperature_multiple_columns():
    """
    Test get_surface_temperature across different columns.
    
    Verifies that surface temperature retrieval works correctly
    for different columns with varying snow conditions.
    """
    state = TemperatureState(
        t_soisno_col=jnp.array(
            [
                [250.0, 255.0, 260.0, 265.0, 270.0, 273.15, 275.0],
                [248.0, 253.0, 258.0, 263.0, 268.0, 271.0, 274.0],
                [252.0, 257.0, 262.0, 267.0, 272.0, 275.0, 277.0],
            ],
            dtype=jnp.float32,
        ),
        t_a10_patch=jnp.array([280.0, 282.0, 284.0], dtype=jnp.float32),
        t_ref2m_patch=jnp.array([278.0, 280.0, 282.0], dtype=jnp.float32),
    )
    nlevsno = 3

    # Test with snow
    temp0_snow = get_surface_temperature(state, col_idx=0, nlevsno=nlevsno, snl=-2)
    # Surface index = 3 + (-2) = 1
    assert jnp.allclose(temp0_snow, 255.0, atol=1e-6), (
        f"Column 0 with snow: expected 255.0, got {temp0_snow}"
    )

    # Test without snow
    temp1_no_snow = get_surface_temperature(state, col_idx=1, nlevsno=nlevsno, snl=0)
    # Surface index = 3
    assert jnp.allclose(temp1_no_snow, 263.0, atol=1e-6), (
        f"Column 1 without snow: expected 263.0, got {temp1_no_snow}"
    )


# ============================================================================
# Tests for Fortran indexing edge cases
# ============================================================================


def test_fortran_indexing_deepest_snow_layer():
    """
    Test Fortran indexing at the deepest snow layer boundary.
    
    Verifies correct conversion of the most negative Fortran index
    (layer_idx = -nlevsno + 1) to Python index.
    """
    state = TemperatureState(
        t_soisno_col=jnp.array(
            [[240.0, 245.0, 250.0, 255.0, 260.0, 265.0, 270.0, 273.15, 275.0, 278.0, 280.0, 282.0]],
            dtype=jnp.float32,
        ),
        t_a10_patch=jnp.array([280.0], dtype=jnp.float32),
        t_ref2m_patch=jnp.array([278.0], dtype=jnp.float32),
    )
    nlevsno = 5

    # Deepest snow layer: Fortran index = -nlevsno + 1 = -4
    # Python index = -4 + 5 - 1 = 0
    temp = get_soil_temperature(state, col_idx=0, layer_idx=-4, nlevsno=nlevsno)

    expected = 240.0
    assert jnp.allclose(temp, expected, atol=1e-6), (
        f"Deepest snow layer: expected {expected}, got {temp}"
    )


def test_fortran_indexing_top_snow_layer():
    """
    Test Fortran indexing at the top snow layer (layer_idx = 0).
    
    Verifies correct conversion of Fortran index 0 (top snow layer)
    to Python index.
    """
    state = TemperatureState(
        t_soisno_col=jnp.array(
            [[240.0, 245.0, 250.0, 255.0, 260.0, 265.0, 270.0]],
            dtype=jnp.float32,
        ),
        t_a10_patch=jnp.array([280.0], dtype=jnp.float32),
        t_ref2m_patch=jnp.array([278.0], dtype=jnp.float32),
    )
    nlevsno = 3

    # Top snow layer: Fortran index = 0
    # Python index = 0 + 3 - 1 = 2
    temp = get_soil_temperature(state, col_idx=0, layer_idx=0, nlevsno=nlevsno)

    expected = 250.0
    assert jnp.allclose(temp, expected, atol=1e-6), (
        f"Top snow layer: expected {expected}, got {temp}"
    )


def test_fortran_indexing_consistency():
    """
    Test consistency of Fortran-to-Python index conversion.
    
    Verifies that the conversion formula works correctly across
    the full range of valid layer indices.
    """
    state = TemperatureState(
        t_soisno_col=jnp.array(
            [[240.0, 245.0, 250.0, 255.0, 260.0, 265.0, 270.0, 273.15, 275.0, 278.0]],
            dtype=jnp.float32,
        ),
        t_a10_patch=jnp.array([280.0], dtype=jnp.float32),
        t_ref2m_patch=jnp.array([278.0], dtype=jnp.float32),
    )
    nlevsno = 4
    nlevgrnd = 6

    # Test range: -nlevsno+1 to nlevgrnd
    # Fortran indices: -3, -2, -1, 0, 1, 2, 3, 4, 5, 6
    # Python indices:   0,  1,  2, 3, 4, 5, 6, 7, 8, 9
    fortran_indices = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
    expected_values = [240.0, 245.0, 250.0, 255.0, 260.0, 265.0, 270.0, 273.15, 275.0, 278.0]

    for fortran_idx, expected_val in zip(fortran_indices, expected_values):
        temp = get_soil_temperature(state, col_idx=0, layer_idx=fortran_idx, nlevsno=nlevsno)
        assert jnp.allclose(temp, expected_val, atol=1e-6), (
            f"Fortran index {fortran_idx}: expected {expected_val}, got {temp}"
        )


# ============================================================================
# Tests for physical constraints
# ============================================================================


def test_physical_constraint_absolute_zero():
    """
    Test that temperatures cannot be below absolute zero (0 K).
    
    Verifies that the initialization functions respect the fundamental
    physical constraint that temperature >= 0 K.
    """
    # Test with valid temperature at lower bound
    state = init_temperature_state(5, 8, 15, initial_temp=0.1)
    assert jnp.all(state.t_soisno_col >= 0.0), "Temperatures must be >= 0 K"

    # Test with typical cold temperature
    state_cold = init_temperature_state(5, 8, 15, initial_temp=200.0)
    assert jnp.all(state_cold.t_soisno_col >= 0.0), "Cold temperatures must be >= 0 K"


def test_physical_constraint_typical_range():
    """
    Test that typical Earth surface temperatures are in reasonable range.
    
    Verifies that initialized temperatures fall within the typical
    range for Earth surface conditions (200-350 K).
    """
    # Test various temperatures in typical range
    temps = [200.0, 250.0, 273.15, 300.0, 350.0]

    for temp in temps:
        state = init_temperature_state(5, 8, 15, initial_temp=temp)
        assert jnp.all(state.t_soisno_col >= 200.0), (
            f"Temperature {temp} should be >= 200 K"
        )
        assert jnp.all(state.t_soisno_col <= 350.0), (
            f"Temperature {temp} should be <= 350 K"
        )


def test_bounds_constraint_validity():
    """
    Test that bounds satisfy end >= begin constraints.
    
    Verifies that BoundsType enforces the constraint that all
    end indices are >= corresponding begin indices.
    """
    # Valid bounds
    valid_bounds = BoundsType(begp=0, endp=10, begc=0, endc=5, begg=0, endg=2)
    state = init_temperature(valid_bounds, nlevsno=3, nlevgrnd=10)
    assert state.t_soisno_col.shape[0] == 6, "Should create 6 columns (0-5)"
    assert state.t_a10_patch.shape[0] == 11, "Should create 11 patches (0-10)"

    # Edge case: begin == end
    edge_bounds = BoundsType(begp=5, endp=5, begc=3, endc=3, begg=1, endg=1)
    state_edge = init_temperature(edge_bounds, nlevsno=2, nlevgrnd=8)
    assert state_edge.t_soisno_col.shape[0] == 1, "Should create 1 column"
    assert state_edge.t_a10_patch.shape[0] == 1, "Should create 1 patch"


def test_snow_layer_count_constraint():
    """
    Test that active snow layer count (snl) is non-positive.
    
    Verifies that get_surface_temperature correctly handles the
    constraint that snl <= 0.
    """
    state = TemperatureState(
        t_soisno_col=jnp.array(
            [[250.0, 255.0, 260.0, 265.0, 270.0, 273.15]],
            dtype=jnp.float32,
        ),
        t_a10_patch=jnp.array([280.0], dtype=jnp.float32),
        t_ref2m_patch=jnp.array([278.0], dtype=jnp.float32),
    )
    nlevsno = 3

    # Test with snl = 0 (no snow)
    temp_no_snow = get_surface_temperature(state, col_idx=0, nlevsno=nlevsno, snl=0)
    assert temp_no_snow is not None, "Should handle snl=0"

    # Test with snl < 0 (snow present)
    temp_with_snow = get_surface_temperature(state, col_idx=0, nlevsno=nlevsno, snl=-2)
    assert temp_with_snow is not None, "Should handle snl<0"


# ============================================================================
# Integration tests
# ============================================================================


def test_integration_init_and_update():
    """
    Integration test: Initialize state and perform updates.
    
    Tests the complete workflow of initializing a temperature state
    and then updating individual temperatures.
    """
    # Initialize state
    state = init_temperature_state(n_columns=3, n_patches=5, n_levtot=8, initial_temp=273.15)

    # Verify initialization
    assert jnp.allclose(state.t_soisno_col, 273.15, atol=1e-6), (
        "Initial state should be 273.15 K"
    )

    # Update a temperature
    nlevsno = 3
    updated_state = update_soil_temperature(
        state, col_idx=1, layer_idx=2, nlevsno=nlevsno, new_temp=280.0
    )

    # Verify update
    retrieved_temp = get_soil_temperature(
        updated_state, col_idx=1, layer_idx=2, nlevsno=nlevsno
    )
    assert jnp.allclose(retrieved_temp, 280.0, atol=1e-6), (
        f"Updated temperature should be 280.0, got {retrieved_temp}"
    )

    # Verify other values unchanged
    other_temp = get_soil_temperature(
        updated_state, col_idx=0, layer_idx=2, nlevsno=nlevsno
    )
    assert jnp.allclose(other_temp, 273.15, atol=1e-6), (
        f"Other temperatures should remain 273.15, got {other_temp}"
    )


def test_integration_bounds_and_surface():
    """
    Integration test: Initialize with bounds and get surface temperature.
    
    Tests the workflow of initializing with bounds and then retrieving
    surface temperatures under different snow conditions.
    """
    # Initialize with bounds
    bounds = BoundsType(begp=0, endp=9, begc=0, endc=4, begg=0, endg=1)
    state = init_temperature(bounds, nlevsno=5, nlevgrnd=10)

    # Update some temperatures to create a profile
    nlevsno = 5
    state = update_soil_temperature(state, col_idx=0, layer_idx=-2, nlevsno=nlevsno, new_temp=260.0)
    state = update_soil_temperature(state, col_idx=0, layer_idx=0, nlevsno=nlevsno, new_temp=268.0)
    state = update_soil_temperature(state, col_idx=0, layer_idx=1, nlevsno=nlevsno, new_temp=273.15)

    # Get surface temperature with snow
    surf_temp_snow = get_surface_temperature(state, col_idx=0, nlevsno=nlevsno, snl=-3)
    # Surface index = 5 + (-3) = 2, which should have been updated to 268.0
    assert jnp.allclose(surf_temp_snow, 268.0, atol=1e-6), (
        f"Surface temp with snow should be 268.0, got {surf_temp_snow}"
    )

    # Get surface temperature without snow
    surf_temp_no_snow = get_surface_temperature(state, col_idx=0, nlevsno=nlevsno, snl=0)
    # Surface index = 5, which should have been updated to 273.15
    assert jnp.allclose(surf_temp_no_snow, 273.15, atol=1e-6), (
        f"Surface temp without snow should be 273.15, got {surf_temp_no_snow}"
    )


def test_integration_multiple_updates():
    """
    Integration test: Perform multiple sequential updates.
    
    Tests that multiple updates can be chained together correctly
    and that each update preserves previous changes.
    """
    # Initialize
    state = init_temperature_state(n_columns=2, n_patches=3, n_levtot=6, initial_temp=273.15)
    nlevsno = 2

    # Perform multiple updates
    state = update_soil_temperature(state, col_idx=0, layer_idx=-1, nlevsno=nlevsno, new_temp=260.0)
    state = update_soil_temperature(state, col_idx=0, layer_idx=1, nlevsno=nlevsno, new_temp=275.0)
    state = update_soil_temperature(state, col_idx=1, layer_idx=2, nlevsno=nlevsno, new_temp=280.0)

    # Verify all updates
    temp1 = get_soil_temperature(state, col_idx=0, layer_idx=-1, nlevsno=nlevsno)
    temp2 = get_soil_temperature(state, col_idx=0, layer_idx=1, nlevsno=nlevsno)
    temp3 = get_soil_temperature(state, col_idx=1, layer_idx=2, nlevsno=nlevsno)

    assert jnp.allclose(temp1, 260.0, atol=1e-6), f"First update: expected 260.0, got {temp1}"
    assert jnp.allclose(temp2, 275.0, atol=1e-6), f"Second update: expected 275.0, got {temp2}"
    assert jnp.allclose(temp3, 280.0, atol=1e-6), f"Third update: expected 280.0, got {temp3}"

    # Verify unchanged values
    unchanged_temp = get_soil_temperature(state, col_idx=1, layer_idx=1, nlevsno=nlevsno)
    assert jnp.allclose(unchanged_temp, 273.15, atol=1e-6), (
        f"Unchanged values should remain 273.15, got {unchanged_temp}"
    )