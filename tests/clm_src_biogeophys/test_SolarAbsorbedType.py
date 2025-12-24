"""
Comprehensive pytest suite for SolarAbsorbedType module.

This test suite validates the solar radiation absorption state management
functions including initialization, allocation, and update operations for
patch-level solar radiation data in the CLM biogeophysics module.

Test Coverage:
- Initialization functions (init_solar_abs_state, init_allocate_solar_abs, init, init_allocate)
- State update operations (update_fsa_patch)
- Array shapes and data types
- Physical constraints (solar radiation values)
- Edge cases (single patch, large domains, boundary conditions)
- Nominal cases (typical grid sizes and radiation values)
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from clm_src_biogeophys.SolarAbsorbedType import (
    BoundsType,
    SolarAbsState,
    init_solar_abs_state,
    init_allocate_solar_abs,
    init_allocate,
    init,
    update_fsa_patch,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data():
    """
    Load test data for SolarAbsorbedType module.
    
    Returns:
        dict: Test cases with inputs and expected behaviors
    """
    return {
        "test_cases": [
            {
                "name": "test_init_solar_abs_state_single_patch",
                "function": "init_solar_abs_state",
                "inputs": {"n_patches": 1},
                "expected_behavior": {
                    "fsa_patch_shape": [1],
                    "fsa_patch_dtype": "float",
                    "contains_nan": True
                },
            },
            {
                "name": "test_init_solar_abs_state_typical_grid",
                "function": "init_solar_abs_state",
                "inputs": {"n_patches": 100},
                "expected_behavior": {
                    "fsa_patch_shape": [100],
                    "fsa_patch_dtype": "float",
                    "contains_nan": True
                },
            },
            {
                "name": "test_init_allocate_solar_abs_large_domain",
                "function": "init_allocate_solar_abs",
                "inputs": {"n_patches": 10000},
                "expected_behavior": {
                    "fsa_patch_shape": [10000],
                    "fsa_patch_dtype": "float",
                    "contains_nan": True
                },
            },
            {
                "name": "test_init_allocate_with_bounds_typical",
                "function": "init_allocate",
                "inputs": {
                    "this": {"fsa_patch": [None]},
                    "bounds": {"begp": 0, "endp": 49}
                },
                "expected_behavior": {
                    "fsa_patch_shape": [50],
                    "fsa_patch_dtype": "float",
                    "contains_nan": True
                },
            },
            {
                "name": "test_init_allocate_with_bounds_single_patch",
                "function": "init_allocate",
                "inputs": {
                    "this": {"fsa_patch": [None]},
                    "bounds": {"begp": 5, "endp": 5}
                },
                "expected_behavior": {
                    "fsa_patch_shape": [1],
                    "fsa_patch_dtype": "float",
                    "contains_nan": True
                },
            },
            {
                "name": "test_init_with_bounds_zero_based",
                "function": "init",
                "inputs": {"bounds": {"begp": 0, "endp": 999}},
                "expected_behavior": {
                    "fsa_patch_shape": [1000],
                    "fsa_patch_dtype": "float",
                    "contains_nan": True
                },
            },
            {
                "name": "test_update_fsa_patch_clear_sky_noon",
                "function": "update_fsa_patch",
                "inputs": {
                    "state": {"fsa_patch": [0.0, 0.0, 0.0, 0.0, 0.0]},
                    "fsa_patch": [850.5, 862.3, 845.0, 870.2, 855.8]
                },
                "expected_behavior": {
                    "fsa_patch_values": [850.5, 862.3, 845.0, 870.2, 855.8]
                },
            },
            {
                "name": "test_update_fsa_patch_zero_nighttime",
                "function": "update_fsa_patch",
                "inputs": {
                    "state": {"fsa_patch": [500.0, 500.0, 500.0]},
                    "fsa_patch": [0.0, 0.0, 0.0]
                },
                "expected_behavior": {
                    "fsa_patch_values": [0.0, 0.0, 0.0]
                },
            },
            {
                "name": "test_update_fsa_patch_maximum_toa",
                "function": "update_fsa_patch",
                "inputs": {
                    "state": {"fsa_patch": [0.0, 0.0]},
                    "fsa_patch": [1361.0, 1360.8]
                },
                "expected_behavior": {
                    "fsa_patch_values": [1361.0, 1360.8]
                },
            },
            {
                "name": "test_update_fsa_patch_mixed_conditions",
                "function": "update_fsa_patch",
                "inputs": {
                    "state": {"fsa_patch": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]},
                    "fsa_patch": [0.0, 125.5, 450.3, 678.9, 892.1, 1050.0, 234.7, 15.2]
                },
                "expected_behavior": {
                    "fsa_patch_values": [0.0, 125.5, 450.3, 678.9, 892.1, 1050.0, 234.7, 15.2]
                },
            },
        ]
    }


@pytest.fixture
def sample_bounds():
    """
    Create sample BoundsType instances for testing.
    
    Returns:
        dict: Dictionary of named BoundsType instances
    """
    return {
        "single_patch": BoundsType(begp=0, endp=0),
        "small_domain": BoundsType(begp=0, endp=9),
        "typical_domain": BoundsType(begp=0, endp=99),
        "large_domain": BoundsType(begp=0, endp=9999),
        "offset_domain": BoundsType(begp=10, endp=59),
    }


@pytest.fixture
def sample_solar_abs_states():
    """
    Create sample SolarAbsState instances for testing.
    
    Returns:
        dict: Dictionary of named SolarAbsState instances
    """
    return {
        "uninitialized_small": SolarAbsState(fsa_patch=jnp.full(5, jnp.nan)),
        "zeros_small": SolarAbsState(fsa_patch=jnp.zeros(5)),
        "typical_values": SolarAbsState(fsa_patch=jnp.array([850.0, 860.0, 845.0, 870.0, 855.0])),
        "nighttime": SolarAbsState(fsa_patch=jnp.zeros(10)),
        "mixed_conditions": SolarAbsState(fsa_patch=jnp.array([0.0, 125.5, 450.3, 678.9, 892.1])),
    }


# ============================================================================
# Test init_solar_abs_state
# ============================================================================

@pytest.mark.parametrize("n_patches,expected_shape", [
    (1, (1,)),
    (10, (10,)),
    (100, (100,)),
    (1000, (1000,)),
    (10000, (10000,)),
])
def test_init_solar_abs_state_shapes(n_patches, expected_shape):
    """
    Test that init_solar_abs_state creates arrays with correct shapes.
    
    Validates that the function properly allocates arrays matching the
    requested number of patches for various domain sizes.
    """
    state = init_solar_abs_state(n_patches)
    
    assert isinstance(state, SolarAbsState), \
        f"Expected SolarAbsState, got {type(state)}"
    assert state.fsa_patch.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {state.fsa_patch.shape}"


def test_init_solar_abs_state_contains_nan():
    """
    Test that init_solar_abs_state initializes arrays with NaN values.
    
    NaN values indicate uninitialized state that requires subsequent
    update before use in calculations.
    """
    state = init_solar_abs_state(50)
    
    assert jnp.all(jnp.isnan(state.fsa_patch)), \
        "Expected all values to be NaN for uninitialized state"


def test_init_solar_abs_state_dtype():
    """
    Test that init_solar_abs_state creates arrays with correct dtype.
    
    Validates that arrays use floating-point type suitable for
    representing solar radiation values.
    """
    state = init_solar_abs_state(10)
    
    assert jnp.issubdtype(state.fsa_patch.dtype, jnp.floating), \
        f"Expected floating point dtype, got {state.fsa_patch.dtype}"


@pytest.mark.parametrize("n_patches", [1, 5, 100, 10000])
def test_init_solar_abs_state_independence(n_patches):
    """
    Test that multiple calls to init_solar_abs_state create independent arrays.
    
    Ensures that each initialization creates a new array instance without
    shared memory that could cause unintended side effects.
    """
    state1 = init_solar_abs_state(n_patches)
    state2 = init_solar_abs_state(n_patches)
    
    # Both should be NaN but be different array instances
    assert jnp.all(jnp.isnan(state1.fsa_patch))
    assert jnp.all(jnp.isnan(state2.fsa_patch))
    assert state1.fsa_patch is not state2.fsa_patch, \
        "Expected independent array instances"


# ============================================================================
# Test init_allocate_solar_abs
# ============================================================================

@pytest.mark.parametrize("n_patches,expected_shape", [
    (1, (1,)),
    (50, (50,)),
    (1000, (1000,)),
    (10000, (10000,)),
])
def test_init_allocate_solar_abs_shapes(n_patches, expected_shape):
    """
    Test that init_allocate_solar_abs creates arrays with correct shapes.
    
    Validates proper allocation for various domain sizes from single
    patch to large global simulations.
    """
    state = init_allocate_solar_abs(n_patches)
    
    assert isinstance(state, SolarAbsState), \
        f"Expected SolarAbsState, got {type(state)}"
    assert state.fsa_patch.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {state.fsa_patch.shape}"


def test_init_allocate_solar_abs_contains_nan():
    """
    Test that init_allocate_solar_abs initializes with NaN values.
    
    Verifies that allocated arrays are properly initialized to NaN
    to indicate uninitialized state.
    """
    state = init_allocate_solar_abs(100)
    
    assert jnp.all(jnp.isnan(state.fsa_patch)), \
        "Expected all values to be NaN for allocated but uninitialized state"


def test_init_allocate_solar_abs_dtype():
    """
    Test that init_allocate_solar_abs creates arrays with correct dtype.
    """
    state = init_allocate_solar_abs(25)
    
    assert jnp.issubdtype(state.fsa_patch.dtype, jnp.floating), \
        f"Expected floating point dtype, got {state.fsa_patch.dtype}"


# ============================================================================
# Test init_allocate
# ============================================================================

@pytest.mark.parametrize("begp,endp,expected_size", [
    (0, 0, 1),      # Single patch (begp == endp)
    (0, 9, 10),     # Small domain
    (0, 49, 50),    # Typical domain
    (0, 999, 1000), # Large domain
    (5, 5, 1),      # Single patch with offset
    (10, 59, 50),   # Offset domain
])
def test_init_allocate_shapes(begp, endp, expected_size):
    """
    Test that init_allocate creates arrays with correct shapes based on bounds.
    
    Validates that array size equals (endp - begp + 1) for inclusive
    index ranges, covering various boundary conditions.
    """
    bounds = BoundsType(begp=begp, endp=endp)
    initial_state = SolarAbsState(fsa_patch=jnp.array([]))
    
    state = init_allocate(initial_state, bounds)
    
    assert isinstance(state, SolarAbsState), \
        f"Expected SolarAbsState, got {type(state)}"
    assert state.fsa_patch.shape == (expected_size,), \
        f"Expected shape ({expected_size},), got {state.fsa_patch.shape}"


def test_init_allocate_contains_nan():
    """
    Test that init_allocate initializes arrays with NaN values.
    """
    bounds = BoundsType(begp=0, endp=49)
    initial_state = SolarAbsState(fsa_patch=jnp.array([]))
    
    state = init_allocate(initial_state, bounds)
    
    assert jnp.all(jnp.isnan(state.fsa_patch)), \
        "Expected all values to be NaN after init_allocate"


def test_init_allocate_boundary_equality():
    """
    Test init_allocate edge case where begp equals endp (single patch).
    
    This boundary condition represents the minimum valid domain size
    and tests proper handling of inclusive range endpoints.
    """
    bounds = BoundsType(begp=5, endp=5)
    initial_state = SolarAbsState(fsa_patch=jnp.array([]))
    
    state = init_allocate(initial_state, bounds)
    
    assert state.fsa_patch.shape == (1,), \
        f"Expected single element array when begp==endp, got shape {state.fsa_patch.shape}"
    assert jnp.isnan(state.fsa_patch[0]), \
        "Expected NaN value in single-element array"


def test_init_allocate_dtype():
    """
    Test that init_allocate creates arrays with correct dtype.
    """
    bounds = BoundsType(begp=0, endp=19)
    initial_state = SolarAbsState(fsa_patch=jnp.array([]))
    
    state = init_allocate(initial_state, bounds)
    
    assert jnp.issubdtype(state.fsa_patch.dtype, jnp.floating), \
        f"Expected floating point dtype, got {state.fsa_patch.dtype}"


# ============================================================================
# Test init
# ============================================================================

@pytest.mark.parametrize("begp,endp,expected_size", [
    (0, 0, 1),
    (0, 9, 10),
    (0, 99, 100),
    (0, 999, 1000),
    (10, 109, 100),
])
def test_init_shapes(begp, endp, expected_size):
    """
    Test that init creates arrays with correct shapes based on bounds.
    
    Validates full initialization workflow from bounds to allocated
    and initialized state arrays.
    """
    bounds = BoundsType(begp=begp, endp=endp)
    
    state = init(bounds)
    
    assert isinstance(state, SolarAbsState), \
        f"Expected SolarAbsState, got {type(state)}"
    assert state.fsa_patch.shape == (expected_size,), \
        f"Expected shape ({expected_size},), got {state.fsa_patch.shape}"


def test_init_contains_nan():
    """
    Test that init initializes arrays with NaN values.
    """
    bounds = BoundsType(begp=0, endp=99)
    
    state = init(bounds)
    
    assert jnp.all(jnp.isnan(state.fsa_patch)), \
        "Expected all values to be NaN after init"


def test_init_zero_based_indexing():
    """
    Test init with zero-based indexing (begp=0).
    
    Validates proper handling of zero-based array indexing convention
    commonly used in Python/NumPy/JAX.
    """
    bounds = BoundsType(begp=0, endp=999)
    
    state = init(bounds)
    
    assert state.fsa_patch.shape == (1000,), \
        f"Expected 1000 elements for zero-based [0, 999] range, got {state.fsa_patch.shape}"


def test_init_dtype():
    """
    Test that init creates arrays with correct dtype.
    """
    bounds = BoundsType(begp=0, endp=49)
    
    state = init(bounds)
    
    assert jnp.issubdtype(state.fsa_patch.dtype, jnp.floating), \
        f"Expected floating point dtype, got {state.fsa_patch.dtype}"


# ============================================================================
# Test update_fsa_patch
# ============================================================================

@pytest.mark.parametrize("initial_values,new_values", [
    ([0.0, 0.0, 0.0], [850.5, 862.3, 845.0]),
    ([500.0, 500.0], [0.0, 0.0]),
    ([100.0, 200.0], [1361.0, 1360.8]),
    ([0.0] * 5, [0.0, 125.5, 450.3, 678.9, 892.1]),
])
def test_update_fsa_patch_values(initial_values, new_values):
    """
    Test that update_fsa_patch correctly updates state with new values.
    
    Validates that the update operation properly replaces old values
    with new solar radiation measurements across various scenarios.
    """
    initial_state = SolarAbsState(fsa_patch=jnp.array(initial_values))
    new_fsa_patch = jnp.array(new_values)
    
    updated_state = update_fsa_patch(initial_state, new_fsa_patch)
    
    assert isinstance(updated_state, SolarAbsState), \
        f"Expected SolarAbsState, got {type(updated_state)}"
    assert jnp.allclose(updated_state.fsa_patch, new_fsa_patch, atol=1e-6, rtol=1e-6), \
        f"Expected values {new_values}, got {updated_state.fsa_patch}"


def test_update_fsa_patch_clear_sky_noon():
    """
    Test update with typical clear-sky solar noon values.
    
    Validates handling of realistic solar radiation values during
    peak daytime conditions (~850-870 W/m²).
    """
    initial_state = SolarAbsState(fsa_patch=jnp.zeros(5))
    new_values = jnp.array([850.5, 862.3, 845.0, 870.2, 855.8])
    
    updated_state = update_fsa_patch(initial_state, new_values)
    
    assert jnp.allclose(updated_state.fsa_patch, new_values, atol=1e-6, rtol=1e-6), \
        "Clear sky noon values not correctly updated"
    assert jnp.all(updated_state.fsa_patch >= 0.0), \
        "Solar radiation values should be non-negative"
    assert jnp.all(updated_state.fsa_patch <= 1400.0), \
        "Solar radiation values should not exceed solar constant"


def test_update_fsa_patch_zero_nighttime():
    """
    Test update with zero values representing nighttime conditions.
    
    Validates proper handling of zero solar radiation during nighttime,
    an important edge case for diurnal cycle simulations.
    """
    initial_state = SolarAbsState(fsa_patch=jnp.array([500.0, 500.0, 500.0]))
    new_values = jnp.zeros(3)
    
    updated_state = update_fsa_patch(initial_state, new_values)
    
    assert jnp.allclose(updated_state.fsa_patch, 0.0, atol=1e-6), \
        "Nighttime zero values not correctly updated"


def test_update_fsa_patch_maximum_toa():
    """
    Test update with maximum physically realistic values (solar constant).
    
    Validates handling of extreme but valid values at top of atmosphere
    (~1361 W/m²), testing upper physical bounds.
    """
    initial_state = SolarAbsState(fsa_patch=jnp.zeros(2))
    new_values = jnp.array([1361.0, 1360.8])
    
    updated_state = update_fsa_patch(initial_state, new_values)
    
    assert jnp.allclose(updated_state.fsa_patch, new_values, atol=1e-6, rtol=1e-6), \
        "Maximum TOA values not correctly updated"
    assert jnp.all(updated_state.fsa_patch <= 1400.0), \
        "Values should not exceed reasonable solar constant bounds"


def test_update_fsa_patch_mixed_conditions():
    """
    Test update with heterogeneous spatial conditions.
    
    Validates handling of realistic spatial variability including
    nighttime, dawn/dusk, partial cloud, and clear sky conditions
    across different patches.
    """
    initial_state = SolarAbsState(fsa_patch=jnp.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]))
    new_values = jnp.array([0.0, 125.5, 450.3, 678.9, 892.1, 1050.0, 234.7, 15.2])
    
    updated_state = update_fsa_patch(initial_state, new_values)
    
    assert jnp.allclose(updated_state.fsa_patch, new_values, atol=1e-6, rtol=1e-6), \
        "Mixed condition values not correctly updated"
    assert jnp.all(updated_state.fsa_patch >= 0.0), \
        "All solar radiation values should be non-negative"


def test_update_fsa_patch_shape_preservation():
    """
    Test that update_fsa_patch preserves array shape.
    
    Ensures that the update operation maintains the original array
    dimensions without reshaping or broadcasting errors.
    """
    initial_state = SolarAbsState(fsa_patch=jnp.zeros(100))
    new_values = jnp.ones(100) * 850.0
    
    updated_state = update_fsa_patch(initial_state, new_values)
    
    assert updated_state.fsa_patch.shape == initial_state.fsa_patch.shape, \
        "Shape should be preserved during update"


def test_update_fsa_patch_dtype_preservation():
    """
    Test that update_fsa_patch preserves array dtype.
    
    Validates that the update operation maintains floating-point
    precision without unintended type conversions.
    """
    initial_state = SolarAbsState(fsa_patch=jnp.zeros(10, dtype=jnp.float32))
    new_values = jnp.ones(10, dtype=jnp.float32) * 850.0
    
    updated_state = update_fsa_patch(initial_state, new_values)
    
    assert updated_state.fsa_patch.dtype == initial_state.fsa_patch.dtype, \
        "Dtype should be preserved during update"


# ============================================================================
# Test Physical Constraints
# ============================================================================

@pytest.mark.parametrize("n_patches", [1, 10, 100, 1000])
def test_physical_constraint_non_negative(n_patches):
    """
    Test that solar radiation values remain non-negative after updates.
    
    Validates physical constraint that absorbed solar radiation cannot
    be negative, testing across various domain sizes.
    """
    state = init_solar_abs_state(n_patches)
    # Update with typical positive values
    new_values = jnp.ones(n_patches) * 850.0
    updated_state = update_fsa_patch(state, new_values)
    
    assert jnp.all(updated_state.fsa_patch >= 0.0), \
        "Solar radiation values must be non-negative"


def test_physical_constraint_solar_constant():
    """
    Test that solar radiation values don't exceed solar constant.
    
    Validates physical upper bound that surface solar radiation should
    not exceed top-of-atmosphere solar constant (~1361 W/m²).
    """
    state = init_solar_abs_state(10)
    # Test with values at and slightly below solar constant
    new_values = jnp.array([1361.0, 1360.0, 1350.0, 1300.0, 1200.0, 
                            1100.0, 1000.0, 900.0, 800.0, 700.0])
    updated_state = update_fsa_patch(state, new_values)
    
    assert jnp.all(updated_state.fsa_patch <= 1400.0), \
        "Solar radiation should not significantly exceed solar constant"


def test_physical_constraint_typical_range():
    """
    Test that typical surface solar radiation values are in expected range.
    
    Validates that common surface values (0-1050 W/m²) are properly
    handled, accounting for atmospheric absorption.
    """
    state = init_solar_abs_state(20)
    # Generate typical surface values
    new_values = jnp.linspace(0.0, 1050.0, 20)
    updated_state = update_fsa_patch(state, new_values)
    
    assert jnp.all(updated_state.fsa_patch >= 0.0), \
        "Values should be non-negative"
    assert jnp.all(updated_state.fsa_patch <= 1050.0), \
        "Typical surface values should not exceed 1050 W/m²"


# ============================================================================
# Test Edge Cases
# ============================================================================

def test_edge_case_single_patch_workflow():
    """
    Test complete workflow with single patch (minimum domain size).
    
    Validates that all operations work correctly for the edge case
    of a single-patch domain, testing minimum valid configuration.
    """
    # Initialize
    state = init_solar_abs_state(1)
    assert state.fsa_patch.shape == (1,)
    assert jnp.isnan(state.fsa_patch[0])
    
    # Update
    new_value = jnp.array([850.0])
    updated_state = update_fsa_patch(state, new_value)
    assert jnp.allclose(updated_state.fsa_patch[0], 850.0, atol=1e-6)


def test_edge_case_large_domain_workflow():
    """
    Test complete workflow with large domain (global simulation scale).
    
    Validates that all operations scale properly to large domains
    (10,000 patches) representative of global climate simulations.
    """
    n_patches = 10000
    
    # Initialize
    state = init_solar_abs_state(n_patches)
    assert state.fsa_patch.shape == (n_patches,)
    assert jnp.all(jnp.isnan(state.fsa_patch))
    
    # Update with spatially varying values
    new_values = jnp.linspace(0.0, 1000.0, n_patches)
    updated_state = update_fsa_patch(state, new_values)
    assert jnp.allclose(updated_state.fsa_patch, new_values, atol=1e-6, rtol=1e-6)


def test_edge_case_bounds_equality():
    """
    Test bounds edge case where begp equals endp.
    
    Validates proper handling of the boundary condition where
    beginning and ending indices are equal (single patch).
    """
    bounds = BoundsType(begp=10, endp=10)
    state = init(bounds)
    
    assert state.fsa_patch.shape == (1,), \
        "Expected single element when begp == endp"
    assert jnp.isnan(state.fsa_patch[0]), \
        "Expected NaN initialization"


def test_edge_case_zero_radiation():
    """
    Test edge case of zero solar radiation (nighttime).
    
    Validates proper handling of zero values, which occur during
    nighttime and are a common edge case in diurnal simulations.
    """
    state = init_solar_abs_state(10)
    zero_values = jnp.zeros(10)
    
    updated_state = update_fsa_patch(state, zero_values)
    
    assert jnp.allclose(updated_state.fsa_patch, 0.0, atol=1e-10), \
        "Zero values should be exactly preserved"


# ============================================================================
# Test Data Type Handling
# ============================================================================

@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_dtype_compatibility(dtype):
    """
    Test that functions work with different floating-point precisions.
    
    Validates compatibility with both single and double precision
    floating-point types commonly used in scientific computing.
    """
    state = SolarAbsState(fsa_patch=jnp.zeros(10, dtype=dtype))
    new_values = jnp.ones(10, dtype=dtype) * 850.0
    
    updated_state = update_fsa_patch(state, new_values)
    
    assert updated_state.fsa_patch.dtype == dtype, \
        f"Expected dtype {dtype}, got {updated_state.fsa_patch.dtype}"


# ============================================================================
# Test Integration Scenarios
# ============================================================================

def test_integration_full_initialization_workflow():
    """
    Test complete initialization workflow from bounds to updated state.
    
    Validates the full sequence: bounds definition -> initialization ->
    allocation -> update, ensuring all steps work together correctly.
    """
    # Step 1: Define bounds
    bounds = BoundsType(begp=0, endp=99)
    
    # Step 2: Initialize
    state = init(bounds)
    assert state.fsa_patch.shape == (100,)
    assert jnp.all(jnp.isnan(state.fsa_patch))
    
    # Step 3: Update with realistic values
    new_values = jnp.linspace(0.0, 900.0, 100)
    updated_state = update_fsa_patch(state, new_values)
    
    # Step 4: Verify final state
    assert jnp.allclose(updated_state.fsa_patch, new_values, atol=1e-6, rtol=1e-6)
    assert jnp.all(updated_state.fsa_patch >= 0.0)
    assert jnp.all(updated_state.fsa_patch <= 1400.0)


def test_integration_diurnal_cycle_simulation():
    """
    Test simulation of diurnal cycle with multiple updates.
    
    Validates that state can be repeatedly updated to simulate
    changing solar radiation throughout a day.
    """
    n_patches = 5
    state = init_solar_abs_state(n_patches)
    
    # Simulate diurnal cycle: night -> dawn -> noon -> dusk -> night
    time_steps = [
        jnp.zeros(n_patches),                          # Midnight
        jnp.ones(n_patches) * 50.0,                    # Dawn
        jnp.ones(n_patches) * 850.0,                   # Noon
        jnp.ones(n_patches) * 200.0,                   # Dusk
        jnp.zeros(n_patches),                          # Midnight
    ]
    
    for new_values in time_steps:
        state = update_fsa_patch(state, new_values)
        assert jnp.all(state.fsa_patch >= 0.0), \
            "Values should remain non-negative throughout cycle"
        assert jnp.all(state.fsa_patch <= 1400.0), \
            "Values should not exceed solar constant"
    
    # Final state should be nighttime (zeros)
    assert jnp.allclose(state.fsa_patch, 0.0, atol=1e-6)


def test_integration_spatial_heterogeneity():
    """
    Test handling of spatially heterogeneous solar radiation patterns.
    
    Validates that the module correctly handles realistic spatial
    variability in solar radiation across patches (e.g., due to
    clouds, topography, latitude).
    """
    n_patches = 20
    state = init_solar_abs_state(n_patches)
    
    # Create spatially varying pattern: some patches cloudy, some clear
    new_values = jnp.array([
        0.0, 0.0, 0.0,           # Nighttime patches
        150.0, 200.0, 180.0,     # Dawn patches
        400.0, 450.0, 420.0,     # Cloudy patches
        850.0, 870.0, 860.0,     # Clear sky patches
        900.0, 920.0, 910.0,     # High elevation clear patches
        300.0, 250.0, 280.0,     # Dusk patches
        50.0, 30.0               # Late dusk patches
    ])
    
    updated_state = update_fsa_patch(state, new_values)
    
    assert jnp.allclose(updated_state.fsa_patch, new_values, atol=1e-6, rtol=1e-6)
    assert jnp.all(updated_state.fsa_patch >= 0.0)
    assert jnp.all(updated_state.fsa_patch <= 1400.0)
    
    # Verify spatial variability is preserved
    assert jnp.std(updated_state.fsa_patch) > 0.0, \
        "Spatial heterogeneity should be preserved"


# ============================================================================
# Test Error Conditions and Robustness
# ============================================================================

def test_robustness_nan_handling_in_update():
    """
    Test that NaN values in input are properly handled during update.
    
    Validates behavior when updating with NaN values, which might
    occur in error conditions or uninitialized data.
    """
    state = init_solar_abs_state(5)
    # Create array with some NaN values
    new_values = jnp.array([850.0, jnp.nan, 860.0, jnp.nan, 870.0])
    
    updated_state = update_fsa_patch(state, new_values)
    
    # NaN values should be preserved in output
    assert jnp.isnan(updated_state.fsa_patch[1])
    assert jnp.isnan(updated_state.fsa_patch[3])
    # Non-NaN values should be correctly updated
    assert jnp.allclose(updated_state.fsa_patch[0], 850.0, atol=1e-6)
    assert jnp.allclose(updated_state.fsa_patch[2], 860.0, atol=1e-6)
    assert jnp.allclose(updated_state.fsa_patch[4], 870.0, atol=1e-6)


def test_robustness_empty_to_allocated():
    """
    Test transition from empty state to allocated state.
    
    Validates that init_allocate properly handles transition from
    an empty/uninitialized state to a fully allocated state.
    """
    # Start with empty state
    empty_state = SolarAbsState(fsa_patch=jnp.array([]))
    assert empty_state.fsa_patch.shape == (0,)
    
    # Allocate with bounds
    bounds = BoundsType(begp=0, endp=49)
    allocated_state = init_allocate(empty_state, bounds)
    
    assert allocated_state.fsa_patch.shape == (50,)
    assert jnp.all(jnp.isnan(allocated_state.fsa_patch))


# ============================================================================
# Test Parametrized Test Data
# ============================================================================

def test_parametrized_test_data_coverage(test_data):
    """
    Test that parametrized test data covers all expected scenarios.
    
    Validates that the test data fixture includes comprehensive
    coverage of nominal, edge, and special cases.
    """
    test_cases = test_data["test_cases"]
    
    # Check we have expected number of test cases
    assert len(test_cases) == 10, \
        f"Expected 10 test cases, got {len(test_cases)}"
    
    # Check coverage of different functions
    functions_covered = set(tc["function"] for tc in test_cases)
    expected_functions = {
        "init_solar_abs_state",
        "init_allocate_solar_abs",
        "init_allocate",
        "init",
        "update_fsa_patch"
    }
    assert functions_covered == expected_functions, \
        f"Expected coverage of {expected_functions}, got {functions_covered}"
    
    # Check we have mix of test types
    test_types = [tc["metadata"].get("type", "unknown") for tc in test_cases]
    assert "nominal" in test_types, "Should have nominal test cases"
    assert "edge" in test_types, "Should have edge test cases"
    assert "special" in test_types, "Should have special test cases"


@pytest.mark.parametrize("test_case_idx", range(10))
def test_parametrized_execution(test_data, test_case_idx):
    """
    Execute all parametrized test cases from test data.
    
    This test dynamically executes each test case defined in the
    test data fixture, validating expected behaviors.
    """
    test_case = test_data["test_cases"][test_case_idx]
    function_name = test_case["function"]
    inputs = test_case["inputs"]
    expected = test_case["expected_behavior"]
    
    # Execute based on function name
    if function_name == "init_solar_abs_state":
        state = init_solar_abs_state(**inputs)
        assert state.fsa_patch.shape == tuple(expected["fsa_patch_shape"])
        if expected.get("contains_nan"):
            assert jnp.all(jnp.isnan(state.fsa_patch))
            
    elif function_name == "init_allocate_solar_abs":
        state = init_allocate_solar_abs(**inputs)
        assert state.fsa_patch.shape == tuple(expected["fsa_patch_shape"])
        if expected.get("contains_nan"):
            assert jnp.all(jnp.isnan(state.fsa_patch))
            
    elif function_name == "init_allocate":
        # Convert dict inputs to proper types
        initial_state = SolarAbsState(fsa_patch=jnp.array([]))
        bounds = BoundsType(**inputs["bounds"])
        state = init_allocate(initial_state, bounds)
        assert state.fsa_patch.shape == tuple(expected["fsa_patch_shape"])
        if expected.get("contains_nan"):
            assert jnp.all(jnp.isnan(state.fsa_patch))
            
    elif function_name == "init":
        bounds = BoundsType(**inputs["bounds"])
        state = init(bounds)
        assert state.fsa_patch.shape == tuple(expected["fsa_patch_shape"])
        if expected.get("contains_nan"):
            assert jnp.all(jnp.isnan(state.fsa_patch))
            
    elif function_name == "update_fsa_patch":
        initial_state = SolarAbsState(fsa_patch=jnp.array(inputs["state"]["fsa_patch"]))
        new_values = jnp.array(inputs["fsa_patch"])
        updated_state = update_fsa_patch(initial_state, new_values)
        expected_values = jnp.array(expected["fsa_patch_values"])
        assert jnp.allclose(updated_state.fsa_patch, expected_values, atol=1e-6, rtol=1e-6)