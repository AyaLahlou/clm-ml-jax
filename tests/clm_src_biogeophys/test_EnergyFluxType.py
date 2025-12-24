"""
Comprehensive pytest suite for EnergyFluxType module.

This module tests the energy flux state initialization and update functions
for the CLM biogeophysics component, including:
- State allocation with NaN initialization
- State initialization with zero values
- State updates with partial and full field modifications
- Physical constraints and edge cases
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_biogeophys.EnergyFluxType import (
    BoundsType,
    EnergyFluxState,
    init,
    init_allocate,
    init_energyflux_state,
    update_energyflux_state,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_data():
    """
    Load and provide test data for all test cases.
    
    Returns:
        dict: Test data organized by function and test case name
    """
    return {
        "init_allocate": {
            "single_patch": {
                "bounds": BoundsType(begp=0, endp=0, begc=0, endc=0, begg=0, endg=0),
                "expected_size": 1,
            },
            "multiple_patches": {
                "bounds": BoundsType(begp=0, endp=99, begc=0, endc=49, begg=0, endg=24),
                "expected_size": 100,
            },
            "large_domain": {
                "bounds": BoundsType(begp=0, endp=9999, begc=0, endc=4999, begg=0, endg=999),
                "expected_size": 10000,
            },
        },
        "init_energyflux_state": {
            "minimal": {"n_patches": 1, "expected_size": 1},
            "typical": {"n_patches": 50, "expected_size": 50},
            "large": {"n_patches": 1000, "expected_size": 1000},
        },
        "update_energyflux_state": {
            "all_fields_positive": {
                "n_patches": 3,
                "updates": {
                    "eflx_sh_tot_patch": jnp.array([150.5, 200.3, 175.8]),
                    "eflx_lh_tot_patch": jnp.array([85.2, 120.7, 95.4]),
                    "eflx_lwrad_out_patch": jnp.array([400.0, 425.5, 410.2]),
                    "taux_patch": jnp.array([0.05, 0.08, 0.06]),
                    "tauy_patch": jnp.array([0.03, 0.04, 0.035]),
                },
            },
            "negative_sensible_heat": {
                "n_patches": 4,
                "updates": {
                    "eflx_sh_tot_patch": jnp.array([-50.2, -75.8, -30.5, -100.0]),
                    "eflx_lh_tot_patch": jnp.array([20.5, 15.3, 25.8, 18.2]),
                    "eflx_lwrad_out_patch": jnp.array([350.0, 340.0, 360.0, 345.0]),
                    "taux_patch": None,
                    "tauy_patch": None,
                },
            },
            "partial_update": {
                "n_patches": 2,
                "initial": {
                    "eflx_sh_tot_patch": jnp.array([100.0, 150.0]),
                    "eflx_lh_tot_patch": jnp.array([50.0, 75.0]),
                    "eflx_lwrad_out_patch": jnp.array([400.0, 420.0]),
                    "taux_patch": jnp.array([0.05, 0.06]),
                    "tauy_patch": jnp.array([0.03, 0.04]),
                },
                "updates": {
                    "eflx_sh_tot_patch": None,
                    "eflx_lh_tot_patch": jnp.array([60.5, 80.2]),
                    "eflx_lwrad_out_patch": None,
                    "taux_patch": None,
                    "tauy_patch": jnp.array([0.035, 0.045]),
                },
            },
            "extreme_values": {
                "n_patches": 5,
                "updates": {
                    "eflx_sh_tot_patch": jnp.array([1000.0, -500.0, 0.0, 0.001, -0.001]),
                    "eflx_lh_tot_patch": jnp.array([800.0, -200.0, 0.0, 0.0001, -0.0001]),
                    "eflx_lwrad_out_patch": jnp.array([600.0, 100.0, 0.001, 500.0, 450.0]),
                    "taux_patch": jnp.array([5.0, -5.0, 0.0, 1e-5, -1e-5]),
                    "tauy_patch": jnp.array([3.0, -3.0, 0.0, 1e-5, -1e-5]),
                },
            },
            "zero_radiation": {
                "n_patches": 3,
                "updates": {
                    "eflx_sh_tot_patch": jnp.array([120.0, 130.0, 110.0]),
                    "eflx_lh_tot_patch": jnp.array([55.0, 60.0, 52.0]),
                    "eflx_lwrad_out_patch": jnp.array([0.001, 0.0, 450.0]),
                    "taux_patch": jnp.array([0.06, 0.055, 0.052]),
                    "tauy_patch": jnp.array([0.032, 0.031, 0.029]),
                },
            },
        },
    }


@pytest.fixture
def sample_bounds():
    """Provide a sample BoundsType for general testing."""
    return BoundsType(begp=0, endp=9, begc=0, endc=4, begg=0, endg=2)


@pytest.fixture
def sample_state():
    """Provide a sample EnergyFluxState for testing updates."""
    n_patches = 5
    return EnergyFluxState(
        eflx_sh_tot_patch=jnp.zeros(n_patches),
        eflx_lh_tot_patch=jnp.zeros(n_patches),
        eflx_lwrad_out_patch=jnp.zeros(n_patches),
        taux_patch=jnp.zeros(n_patches),
        tauy_patch=jnp.zeros(n_patches),
    )


# ============================================================================
# Tests for init_allocate
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    ["single_patch", "multiple_patches", "large_domain"],
    ids=["single_patch", "100_patches", "10000_patches"],
)
def test_init_allocate_shapes(test_data, test_case):
    """
    Test that init_allocate creates arrays with correct shapes.
    
    Verifies that all fields in the returned EnergyFluxState have the
    expected size based on the bounds (endp - begp + 1).
    """
    data = test_data["init_allocate"][test_case]
    bounds = data["bounds"]
    expected_size = data["expected_size"]
    
    state = init_allocate(bounds)
    
    assert isinstance(state, EnergyFluxState), "Should return EnergyFluxState"
    assert state.eflx_sh_tot_patch.shape == (expected_size,), \
        f"eflx_sh_tot_patch shape mismatch"
    assert state.eflx_lh_tot_patch.shape == (expected_size,), \
        f"eflx_lh_tot_patch shape mismatch"
    assert state.eflx_lwrad_out_patch.shape == (expected_size,), \
        f"eflx_lwrad_out_patch shape mismatch"
    assert state.taux_patch.shape == (expected_size,), \
        f"taux_patch shape mismatch"
    assert state.tauy_patch.shape == (expected_size,), \
        f"tauy_patch shape mismatch"


def test_init_allocate_nan_initialization(test_data):
    """
    Test that init_allocate initializes all arrays with NaN values.
    
    This represents an uninitialized state where values must be set
    before use.
    """
    data = test_data["init_allocate"]["multiple_patches"]
    bounds = data["bounds"]
    
    state = init_allocate(bounds)
    
    assert jnp.all(jnp.isnan(state.eflx_sh_tot_patch)), \
        "eflx_sh_tot_patch should be NaN"
    assert jnp.all(jnp.isnan(state.eflx_lh_tot_patch)), \
        "eflx_lh_tot_patch should be NaN"
    assert jnp.all(jnp.isnan(state.eflx_lwrad_out_patch)), \
        "eflx_lwrad_out_patch should be NaN"
    assert jnp.all(jnp.isnan(state.taux_patch)), \
        "taux_patch should be NaN"
    assert jnp.all(jnp.isnan(state.tauy_patch)), \
        "tauy_patch should be NaN"


def test_init_allocate_dtypes(sample_bounds):
    """
    Test that init_allocate creates arrays with correct data types.
    
    All arrays should be JAX arrays with float dtype.
    """
    state = init_allocate(sample_bounds)
    
    assert isinstance(state.eflx_sh_tot_patch, jnp.ndarray), \
        "Should be JAX array"
    assert jnp.issubdtype(state.eflx_sh_tot_patch.dtype, jnp.floating), \
        "Should be float dtype"
    assert isinstance(state.eflx_lh_tot_patch, jnp.ndarray), \
        "Should be JAX array"
    assert isinstance(state.eflx_lwrad_out_patch, jnp.ndarray), \
        "Should be JAX array"
    assert isinstance(state.taux_patch, jnp.ndarray), \
        "Should be JAX array"
    assert isinstance(state.tauy_patch, jnp.ndarray), \
        "Should be JAX array"


# ============================================================================
# Tests for init (wrapper function)
# ============================================================================


def test_init_wrapper_equivalence(sample_bounds):
    """
    Test that init() produces the same result as init_allocate().
    
    The init function should be a wrapper that calls init_allocate.
    """
    state_init = init(sample_bounds)
    state_allocate = init_allocate(sample_bounds)
    
    assert state_init.eflx_sh_tot_patch.shape == state_allocate.eflx_sh_tot_patch.shape
    assert state_init.eflx_lh_tot_patch.shape == state_allocate.eflx_lh_tot_patch.shape
    assert state_init.eflx_lwrad_out_patch.shape == state_allocate.eflx_lwrad_out_patch.shape
    assert state_init.taux_patch.shape == state_allocate.taux_patch.shape
    assert state_init.tauy_patch.shape == state_allocate.tauy_patch.shape


# ============================================================================
# Tests for init_energyflux_state
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    ["minimal", "typical", "large"],
    ids=["1_patch", "50_patches", "1000_patches"],
)
def test_init_energyflux_state_shapes(test_data, test_case):
    """
    Test that init_energyflux_state creates arrays with correct shapes.
    
    Verifies that all fields have the expected size based on n_patches.
    """
    data = test_data["init_energyflux_state"][test_case]
    n_patches = data["n_patches"]
    expected_size = data["expected_size"]
    
    state = init_energyflux_state(n_patches)
    
    assert isinstance(state, EnergyFluxState), "Should return EnergyFluxState"
    assert state.eflx_sh_tot_patch.shape == (expected_size,)
    assert state.eflx_lh_tot_patch.shape == (expected_size,)
    assert state.eflx_lwrad_out_patch.shape == (expected_size,)
    assert state.taux_patch.shape == (expected_size,)
    assert state.tauy_patch.shape == (expected_size,)


def test_init_energyflux_state_zero_initialization(test_data):
    """
    Test that init_energyflux_state initializes all arrays with zeros.
    
    This represents a properly initialized state with default values.
    """
    data = test_data["init_energyflux_state"]["typical"]
    n_patches = data["n_patches"]
    
    state = init_energyflux_state(n_patches)
    
    assert jnp.allclose(state.eflx_sh_tot_patch, 0.0), \
        "eflx_sh_tot_patch should be zero"
    assert jnp.allclose(state.eflx_lh_tot_patch, 0.0), \
        "eflx_lh_tot_patch should be zero"
    assert jnp.allclose(state.eflx_lwrad_out_patch, 0.0), \
        "eflx_lwrad_out_patch should be zero"
    assert jnp.allclose(state.taux_patch, 0.0), \
        "taux_patch should be zero"
    assert jnp.allclose(state.tauy_patch, 0.0), \
        "tauy_patch should be zero"


def test_init_energyflux_state_dtypes():
    """
    Test that init_energyflux_state creates arrays with correct data types.
    """
    state = init_energyflux_state(10)
    
    assert isinstance(state.eflx_sh_tot_patch, jnp.ndarray)
    assert jnp.issubdtype(state.eflx_sh_tot_patch.dtype, jnp.floating)
    assert isinstance(state.eflx_lh_tot_patch, jnp.ndarray)
    assert isinstance(state.eflx_lwrad_out_patch, jnp.ndarray)
    assert isinstance(state.taux_patch, jnp.ndarray)
    assert isinstance(state.tauy_patch, jnp.ndarray)


def test_init_energyflux_state_minimum_size():
    """
    Test edge case: minimum valid patch count (1).
    """
    state = init_energyflux_state(1)
    
    assert state.eflx_sh_tot_patch.shape == (1,)
    assert jnp.allclose(state.eflx_sh_tot_patch, 0.0)


# ============================================================================
# Tests for update_energyflux_state
# ============================================================================


def test_update_all_fields_positive_fluxes(test_data):
    """
    Test updating all fields with typical positive flux values.
    
    Simulates daytime conditions with heat and moisture fluxes to atmosphere.
    """
    data = test_data["update_energyflux_state"]["all_fields_positive"]
    n_patches = data["n_patches"]
    updates = data["updates"]
    
    # Create initial state
    initial_state = init_energyflux_state(n_patches)
    
    # Update all fields
    updated_state = update_energyflux_state(
        initial_state,
        eflx_sh_tot_patch=updates["eflx_sh_tot_patch"],
        eflx_lh_tot_patch=updates["eflx_lh_tot_patch"],
        eflx_lwrad_out_patch=updates["eflx_lwrad_out_patch"],
        taux_patch=updates["taux_patch"],
        tauy_patch=updates["tauy_patch"],
    )
    
    assert jnp.allclose(updated_state.eflx_sh_tot_patch, updates["eflx_sh_tot_patch"], atol=1e-6)
    assert jnp.allclose(updated_state.eflx_lh_tot_patch, updates["eflx_lh_tot_patch"], atol=1e-6)
    assert jnp.allclose(updated_state.eflx_lwrad_out_patch, updates["eflx_lwrad_out_patch"], atol=1e-6)
    assert jnp.allclose(updated_state.taux_patch, updates["taux_patch"], atol=1e-6)
    assert jnp.allclose(updated_state.tauy_patch, updates["tauy_patch"], atol=1e-6)


def test_update_negative_sensible_heat(test_data):
    """
    Test updating with negative sensible heat flux.
    
    Negative sensible heat represents heat flux from atmosphere to surface,
    typical of stable nighttime conditions.
    """
    data = test_data["update_energyflux_state"]["negative_sensible_heat"]
    n_patches = data["n_patches"]
    updates = data["updates"]
    
    initial_state = init_energyflux_state(n_patches)
    
    updated_state = update_energyflux_state(
        initial_state,
        eflx_sh_tot_patch=updates["eflx_sh_tot_patch"],
        eflx_lh_tot_patch=updates["eflx_lh_tot_patch"],
        eflx_lwrad_out_patch=updates["eflx_lwrad_out_patch"],
        taux_patch=updates["taux_patch"],
        tauy_patch=updates["tauy_patch"],
    )
    
    # Verify negative values are preserved
    assert jnp.all(updated_state.eflx_sh_tot_patch < 0), \
        "Negative sensible heat should be preserved"
    assert jnp.allclose(updated_state.eflx_sh_tot_patch, updates["eflx_sh_tot_patch"], atol=1e-6)
    assert jnp.allclose(updated_state.eflx_lh_tot_patch, updates["eflx_lh_tot_patch"], atol=1e-6)


def test_update_partial_fields_only(test_data):
    """
    Test partial update where only some fields are modified.
    
    Fields with None values should preserve their existing values.
    """
    data = test_data["update_energyflux_state"]["partial_update"]
    initial_values = data["initial"]
    updates = data["updates"]
    
    # Create state with initial values
    initial_state = EnergyFluxState(**initial_values)
    
    # Update only some fields
    updated_state = update_energyflux_state(
        initial_state,
        eflx_sh_tot_patch=updates["eflx_sh_tot_patch"],
        eflx_lh_tot_patch=updates["eflx_lh_tot_patch"],
        eflx_lwrad_out_patch=updates["eflx_lwrad_out_patch"],
        taux_patch=updates["taux_patch"],
        tauy_patch=updates["tauy_patch"],
    )
    
    # Fields that were None should be unchanged
    assert jnp.allclose(updated_state.eflx_sh_tot_patch, initial_values["eflx_sh_tot_patch"], atol=1e-6), \
        "eflx_sh_tot_patch should be unchanged (None passed)"
    assert jnp.allclose(updated_state.eflx_lwrad_out_patch, initial_values["eflx_lwrad_out_patch"], atol=1e-6), \
        "eflx_lwrad_out_patch should be unchanged (None passed)"
    assert jnp.allclose(updated_state.taux_patch, initial_values["taux_patch"], atol=1e-6), \
        "taux_patch should be unchanged (None passed)"
    
    # Fields that were updated should have new values
    assert jnp.allclose(updated_state.eflx_lh_tot_patch, updates["eflx_lh_tot_patch"], atol=1e-6), \
        "eflx_lh_tot_patch should be updated"
    assert jnp.allclose(updated_state.tauy_patch, updates["tauy_patch"], atol=1e-6), \
        "tauy_patch should be updated"


def test_update_extreme_values(test_data):
    """
    Test updating with extreme but physically valid values.
    
    Includes very large fluxes, near-zero values, and negative values
    to ensure the function handles the full physical range.
    """
    data = test_data["update_energyflux_state"]["extreme_values"]
    n_patches = data["n_patches"]
    updates = data["updates"]
    
    initial_state = init_energyflux_state(n_patches)
    
    updated_state = update_energyflux_state(
        initial_state,
        eflx_sh_tot_patch=updates["eflx_sh_tot_patch"],
        eflx_lh_tot_patch=updates["eflx_lh_tot_patch"],
        eflx_lwrad_out_patch=updates["eflx_lwrad_out_patch"],
        taux_patch=updates["taux_patch"],
        tauy_patch=updates["tauy_patch"],
    )
    
    # Verify extreme values are preserved
    assert jnp.allclose(updated_state.eflx_sh_tot_patch, updates["eflx_sh_tot_patch"], atol=1e-6)
    assert jnp.allclose(updated_state.eflx_lh_tot_patch, updates["eflx_lh_tot_patch"], atol=1e-6)
    assert jnp.allclose(updated_state.eflx_lwrad_out_patch, updates["eflx_lwrad_out_patch"], atol=1e-6)
    assert jnp.allclose(updated_state.taux_patch, updates["taux_patch"], atol=1e-6)
    assert jnp.allclose(updated_state.tauy_patch, updates["tauy_patch"], atol=1e-6)
    
    # Verify range of values
    assert jnp.max(updated_state.eflx_sh_tot_patch) == 1000.0, \
        "Should handle large positive sensible heat"
    assert jnp.min(updated_state.eflx_sh_tot_patch) == -500.0, \
        "Should handle large negative sensible heat"


def test_update_zero_radiation_constraint(test_data):
    """
    Test longwave radiation at physical boundary (zero and near-zero).
    
    Longwave radiation must be >= 0 due to Stefan-Boltzmann law.
    Tests that zero and near-zero values are handled correctly.
    """
    data = test_data["update_energyflux_state"]["zero_radiation"]
    n_patches = data["n_patches"]
    updates = data["updates"]
    
    initial_state = init_energyflux_state(n_patches)
    
    updated_state = update_energyflux_state(
        initial_state,
        eflx_sh_tot_patch=updates["eflx_sh_tot_patch"],
        eflx_lh_tot_patch=updates["eflx_lh_tot_patch"],
        eflx_lwrad_out_patch=updates["eflx_lwrad_out_patch"],
        taux_patch=updates["taux_patch"],
        tauy_patch=updates["tauy_patch"],
    )
    
    # Verify all radiation values are non-negative
    assert jnp.all(updated_state.eflx_lwrad_out_patch >= 0), \
        "Longwave radiation must be non-negative"
    assert jnp.allclose(updated_state.eflx_lwrad_out_patch, updates["eflx_lwrad_out_patch"], atol=1e-6)


def test_update_preserves_state_immutability(sample_state):
    """
    Test that update_energyflux_state doesn't modify the input state.
    
    The function should return a new state object, not modify the original.
    """
    original_sh = sample_state.eflx_sh_tot_patch.copy()
    new_values = jnp.array([100.0, 200.0, 150.0, 175.0, 125.0])
    
    updated_state = update_energyflux_state(
        sample_state,
        eflx_sh_tot_patch=new_values,
    )
    
    # Original state should be unchanged
    assert jnp.allclose(sample_state.eflx_sh_tot_patch, original_sh, atol=1e-6), \
        "Original state should not be modified"
    
    # Updated state should have new values
    assert jnp.allclose(updated_state.eflx_sh_tot_patch, new_values, atol=1e-6), \
        "Updated state should have new values"


def test_update_dtypes_preserved():
    """
    Test that update_energyflux_state preserves array data types.
    """
    state = init_energyflux_state(5)
    new_values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    updated_state = update_energyflux_state(
        state,
        eflx_sh_tot_patch=new_values,
    )
    
    assert isinstance(updated_state.eflx_sh_tot_patch, jnp.ndarray)
    assert jnp.issubdtype(updated_state.eflx_sh_tot_patch.dtype, jnp.floating)


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_bounds_with_offset_indices():
    """
    Test init_allocate with non-zero starting indices.
    
    The function should correctly calculate array size from begp/endp
    regardless of the actual index values.
    """
    bounds = BoundsType(begp=100, endp=109, begc=50, endc=54, begg=10, endg=12)
    state = init_allocate(bounds)
    
    expected_size = 10  # 109 - 100 + 1
    assert state.eflx_sh_tot_patch.shape == (expected_size,)


def test_update_with_all_none():
    """
    Test update_energyflux_state when all update values are None.
    
    Should return a state identical to the input.
    """
    initial_state = init_energyflux_state(3)
    
    updated_state = update_energyflux_state(
        initial_state,
        eflx_sh_tot_patch=None,
        eflx_lh_tot_patch=None,
        eflx_lwrad_out_patch=None,
        taux_patch=None,
        tauy_patch=None,
    )
    
    assert jnp.allclose(updated_state.eflx_sh_tot_patch, initial_state.eflx_sh_tot_patch)
    assert jnp.allclose(updated_state.eflx_lh_tot_patch, initial_state.eflx_lh_tot_patch)
    assert jnp.allclose(updated_state.eflx_lwrad_out_patch, initial_state.eflx_lwrad_out_patch)
    assert jnp.allclose(updated_state.taux_patch, initial_state.taux_patch)
    assert jnp.allclose(updated_state.tauy_patch, initial_state.tauy_patch)


def test_namedtuple_structure():
    """
    Test that BoundsType and EnergyFluxState are proper NamedTuples.
    """
    bounds = BoundsType(begp=0, endp=9, begc=0, endc=4, begg=0, endg=2)
    
    # Test field access
    assert bounds.begp == 0
    assert bounds.endp == 9
    assert hasattr(bounds, '_fields')
    
    state = init_energyflux_state(5)
    assert hasattr(state, 'eflx_sh_tot_patch')
    assert hasattr(state, 'eflx_lh_tot_patch')
    assert hasattr(state, 'eflx_lwrad_out_patch')
    assert hasattr(state, 'taux_patch')
    assert hasattr(state, 'tauy_patch')


def test_physical_units_consistency():
    """
    Test that updated values maintain physical unit consistency.
    
    This is a documentation test to verify the expected units:
    - Heat fluxes: W/m²
    - Wind stress: kg/m/s²
    """
    state = init_energyflux_state(3)
    
    # Typical values in correct units
    updated_state = update_energyflux_state(
        state,
        eflx_sh_tot_patch=jnp.array([150.0, 200.0, 175.0]),  # W/m²
        eflx_lh_tot_patch=jnp.array([85.0, 120.0, 95.0]),    # W/m²
        eflx_lwrad_out_patch=jnp.array([400.0, 425.0, 410.0]),  # W/m²
        taux_patch=jnp.array([0.05, 0.08, 0.06]),  # kg/m/s²
        tauy_patch=jnp.array([0.03, 0.04, 0.035]),  # kg/m/s²
    )
    
    # Verify values are in reasonable physical ranges
    assert jnp.all(jnp.abs(updated_state.eflx_sh_tot_patch) < 2000), \
        "Sensible heat flux should be < 2000 W/m² for Earth conditions"
    assert jnp.all(jnp.abs(updated_state.eflx_lh_tot_patch) < 1000), \
        "Latent heat flux should be < 1000 W/m² for typical conditions"
    assert jnp.all(updated_state.eflx_lwrad_out_patch < 700), \
        "Longwave radiation should be < 700 W/m² for Earth surface temps"
    assert jnp.all(jnp.abs(updated_state.taux_patch) < 10), \
        "Wind stress should be < 10 kg/m/s² for extreme conditions"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])