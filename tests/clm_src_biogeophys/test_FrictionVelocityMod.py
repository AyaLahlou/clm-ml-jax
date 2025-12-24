"""
Comprehensive pytest suite for FrictionVelocityMod module.

This test suite validates the friction velocity initialization and update functions
for atmospheric boundary layer calculations in the CLM model.

Test Coverage:
- init_frictionvel_type: Initialize friction velocity state for n patches
- init_allocate: Allocate arrays based on patch bounds
- init_friction_velocity: Convenience wrapper for initialization
- update_frictionvel_type: Immutable state updates following JAX paradigm

Physical Constraints Tested:
- Heights >= 0 m
- Wind speeds >= 0 m/s
- Friction velocities >= 0 m/s
- Bounds relationship: endp >= begp
- Array dimension consistency
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_biogeophys.FrictionVelocityMod import (
    BoundsType,
    FrictionVelType,
    init_allocate,
    init_friction_velocity,
    init_frictionvel_type,
    update_frictionvel_type,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_data():
    """
    Load test data for FrictionVelocityMod functions.
    
    Returns:
        dict: Test cases organized by function name with inputs and expected outputs.
    """
    return {
        "init_frictionvel_type": [
            {
                "name": "single_patch",
                "inputs": {"n_patches": 1},
                "expected_shapes": {
                    "forc_hgt_u_patch": (1,),
                    "u10_clm_patch": (1,),
                    "fv_patch": (1,),
                },
            },
            {
                "name": "multiple_patches",
                "inputs": {"n_patches": 100},
                "expected_shapes": {
                    "forc_hgt_u_patch": (100,),
                    "u10_clm_patch": (100,),
                    "fv_patch": (100,),
                },
            },
            {
                "name": "large_domain",
                "inputs": {"n_patches": 1000},
                "expected_shapes": {
                    "forc_hgt_u_patch": (1000,),
                    "u10_clm_patch": (1000,),
                    "fv_patch": (1000,),
                },
            },
        ],
        "init_allocate": [
            {
                "name": "contiguous_bounds",
                "inputs": {"bounds": BoundsType(begp=0, endp=49)},
                "expected_shapes": {
                    "forc_hgt_u_patch": (50,),
                    "u10_clm_patch": (50,),
                    "fv_patch": (50,),
                },
            },
            {
                "name": "single_patch_bounds",
                "inputs": {"bounds": BoundsType(begp=5, endp=5)},
                "expected_shapes": {
                    "forc_hgt_u_patch": (1,),
                    "u10_clm_patch": (1,),
                    "fv_patch": (1,),
                },
            },
            {
                "name": "large_offset_bounds",
                "inputs": {"bounds": BoundsType(begp=1000, endp=1999)},
                "expected_shapes": {
                    "forc_hgt_u_patch": (1000,),
                    "u10_clm_patch": (1000,),
                    "fv_patch": (1000,),
                },
            },
        ],
        "update_frictionvel_type": [
            {
                "name": "all_fields_update",
                "inputs": {
                    "state": FrictionVelType(
                        forc_hgt_u_patch=jnp.array([10.0, 10.0, 10.0]),
                        u10_clm_patch=jnp.array([5.0, 5.0, 5.0]),
                        fv_patch=jnp.array([0.4, 0.4, 0.4]),
                    ),
                    "forc_hgt_u_patch": jnp.array([30.0, 25.0, 20.0]),
                    "u10_clm_patch": jnp.array([8.5, 7.2, 6.1]),
                    "fv_patch": jnp.array([0.65, 0.58, 0.52]),
                },
                "expected": {
                    "forc_hgt_u_patch": jnp.array([30.0, 25.0, 20.0]),
                    "u10_clm_patch": jnp.array([8.5, 7.2, 6.1]),
                    "fv_patch": jnp.array([0.65, 0.58, 0.52]),
                },
            },
            {
                "name": "partial_update",
                "inputs": {
                    "state": FrictionVelType(
                        forc_hgt_u_patch=jnp.array([10.0, 15.0]),
                        u10_clm_patch=jnp.array([4.5, 5.5]),
                        fv_patch=jnp.array([0.35, 0.42]),
                    ),
                    "forc_hgt_u_patch": None,
                    "u10_clm_patch": jnp.array([6.8, 7.3]),
                    "fv_patch": None,
                },
                "expected": {
                    "forc_hgt_u_patch": jnp.array([10.0, 15.0]),
                    "u10_clm_patch": jnp.array([6.8, 7.3]),
                    "fv_patch": jnp.array([0.35, 0.42]),
                },
            },
            {
                "name": "zero_values",
                "inputs": {
                    "state": FrictionVelType(
                        forc_hgt_u_patch=jnp.array([10.0, 20.0, 30.0, 40.0]),
                        u10_clm_patch=jnp.array([5.0, 6.0, 7.0, 8.0]),
                        fv_patch=jnp.array([0.4, 0.5, 0.6, 0.7]),
                    ),
                    "forc_hgt_u_patch": jnp.array([0.0, 10.0, 20.0, 30.0]),
                    "u10_clm_patch": jnp.array([0.0, 0.0, 5.0, 10.0]),
                    "fv_patch": jnp.array([0.0, 0.0, 0.0, 0.5]),
                },
                "expected": {
                    "forc_hgt_u_patch": jnp.array([0.0, 10.0, 20.0, 30.0]),
                    "u10_clm_patch": jnp.array([0.0, 0.0, 5.0, 10.0]),
                    "fv_patch": jnp.array([0.0, 0.0, 0.0, 0.5]),
                },
            },
            {
                "name": "extreme_values",
                "inputs": {
                    "state": FrictionVelType(
                        forc_hgt_u_patch=jnp.array([10.0, 20.0]),
                        u10_clm_patch=jnp.array([5.0, 6.0]),
                        fv_patch=jnp.array([0.4, 0.5]),
                    ),
                    "forc_hgt_u_patch": jnp.array([500.0, 1000.0]),
                    "u10_clm_patch": jnp.array([50.0, 75.0]),
                    "fv_patch": jnp.array([5.0, 7.5]),
                },
                "expected": {
                    "forc_hgt_u_patch": jnp.array([500.0, 1000.0]),
                    "u10_clm_patch": jnp.array([50.0, 75.0]),
                    "fv_patch": jnp.array([5.0, 7.5]),
                },
            },
            {
                "name": "very_small_values",
                "inputs": {
                    "state": FrictionVelType(
                        forc_hgt_u_patch=jnp.array([10.0, 20.0, 30.0]),
                        u10_clm_patch=jnp.array([5.0, 6.0, 7.0]),
                        fv_patch=jnp.array([0.4, 0.5, 0.6]),
                    ),
                    "forc_hgt_u_patch": jnp.array([0.01, 0.1, 1.0]),
                    "u10_clm_patch": jnp.array([0.001, 0.01, 0.1]),
                    "fv_patch": jnp.array([0.0001, 0.001, 0.01]),
                },
                "expected": {
                    "forc_hgt_u_patch": jnp.array([0.01, 0.1, 1.0]),
                    "u10_clm_patch": jnp.array([0.001, 0.01, 0.1]),
                    "fv_patch": jnp.array([0.0001, 0.001, 0.01]),
                },
            },
        ],
    }


@pytest.fixture
def bounds_test_cases():
    """
    Fixture providing various BoundsType test cases.
    
    Returns:
        list: BoundsType instances covering edge cases.
    """
    return [
        BoundsType(begp=0, endp=0),  # Single patch at zero
        BoundsType(begp=0, endp=99),  # Zero-based contiguous
        BoundsType(begp=10, endp=10),  # Single patch with offset
        BoundsType(begp=100, endp=199),  # Offset contiguous
        BoundsType(begp=1000, endp=2999),  # Large offset
    ]


# ============================================================================
# Tests for init_frictionvel_type
# ============================================================================


@pytest.mark.parametrize(
    "n_patches,expected_shape",
    [
        (1, (1,)),
        (10, (10,)),
        (100, (100,)),
        (1000, (1000,)),
    ],
)
def test_init_frictionvel_type_shapes(n_patches, expected_shape):
    """
    Test that init_frictionvel_type creates arrays with correct shapes.
    
    Verifies that all three fields (forc_hgt_u_patch, u10_clm_patch, fv_patch)
    have the expected shape matching n_patches.
    """
    result = init_frictionvel_type(n_patches)
    
    assert isinstance(result, FrictionVelType), "Result must be FrictionVelType"
    assert result.forc_hgt_u_patch.shape == expected_shape, (
        f"forc_hgt_u_patch shape mismatch: {result.forc_hgt_u_patch.shape} != {expected_shape}"
    )
    assert result.u10_clm_patch.shape == expected_shape, (
        f"u10_clm_patch shape mismatch: {result.u10_clm_patch.shape} != {expected_shape}"
    )
    assert result.fv_patch.shape == expected_shape, (
        f"fv_patch shape mismatch: {result.fv_patch.shape} != {expected_shape}"
    )


def test_init_frictionvel_type_nan_initialization(test_data):
    """
    Test that init_frictionvel_type initializes all arrays with NaN values.
    
    NaN initialization is the standard pattern for uninitialized scientific
    computing arrays, making it easy to detect unset values.
    """
    for case in test_data["init_frictionvel_type"]:
        result = init_frictionvel_type(**case["inputs"])
        
        assert jnp.all(jnp.isnan(result.forc_hgt_u_patch)), (
            f"forc_hgt_u_patch should be all NaN for {case['name']}"
        )
        assert jnp.all(jnp.isnan(result.u10_clm_patch)), (
            f"u10_clm_patch should be all NaN for {case['name']}"
        )
        assert jnp.all(jnp.isnan(result.fv_patch)), (
            f"fv_patch should be all NaN for {case['name']}"
        )


def test_init_frictionvel_type_dtypes():
    """
    Test that init_frictionvel_type creates arrays with correct data types.
    
    All arrays should be float64 for numerical precision in atmospheric
    boundary layer calculations.
    """
    result = init_frictionvel_type(10)
    
    assert result.forc_hgt_u_patch.dtype == jnp.float64, (
        f"forc_hgt_u_patch dtype should be float64, got {result.forc_hgt_u_patch.dtype}"
    )
    assert result.u10_clm_patch.dtype == jnp.float64, (
        f"u10_clm_patch dtype should be float64, got {result.u10_clm_patch.dtype}"
    )
    assert result.fv_patch.dtype == jnp.float64, (
        f"fv_patch dtype should be float64, got {result.fv_patch.dtype}"
    )


@pytest.mark.parametrize("n_patches", [1, 2, 5, 10, 50, 100, 500, 1000])
def test_init_frictionvel_type_various_sizes(n_patches):
    """
    Test init_frictionvel_type with various domain sizes.
    
    Ensures the function works correctly across a range of realistic
    domain sizes from single patch to large regional domains.
    """
    result = init_frictionvel_type(n_patches)
    
    assert isinstance(result, FrictionVelType)
    assert result.forc_hgt_u_patch.shape == (n_patches,)
    assert result.u10_clm_patch.shape == (n_patches,)
    assert result.fv_patch.shape == (n_patches,)


# ============================================================================
# Tests for init_allocate
# ============================================================================


def test_init_allocate_shapes(test_data):
    """
    Test that init_allocate creates arrays with correct shapes based on bounds.
    
    Array size should be (endp - begp + 1) to include both endpoints.
    """
    for case in test_data["init_allocate"]:
        result = init_allocate(**case["inputs"])
        expected_shapes = case["expected_shapes"]
        
        assert result.forc_hgt_u_patch.shape == expected_shapes["forc_hgt_u_patch"], (
            f"forc_hgt_u_patch shape mismatch for {case['name']}"
        )
        assert result.u10_clm_patch.shape == expected_shapes["u10_clm_patch"], (
            f"u10_clm_patch shape mismatch for {case['name']}"
        )
        assert result.fv_patch.shape == expected_shapes["fv_patch"], (
            f"fv_patch shape mismatch for {case['name']}"
        )


def test_init_allocate_nan_initialization(bounds_test_cases):
    """
    Test that init_allocate initializes all arrays with NaN values.
    
    Verifies NaN initialization across various bounds configurations.
    """
    for bounds in bounds_test_cases:
        result = init_allocate(bounds)
        
        assert jnp.all(jnp.isnan(result.forc_hgt_u_patch)), (
            f"forc_hgt_u_patch should be all NaN for bounds {bounds}"
        )
        assert jnp.all(jnp.isnan(result.u10_clm_patch)), (
            f"u10_clm_patch should be all NaN for bounds {bounds}"
        )
        assert jnp.all(jnp.isnan(result.fv_patch)), (
            f"fv_patch should be all NaN for bounds {bounds}"
        )


def test_init_allocate_size_calculation():
    """
    Test that init_allocate correctly calculates array size from bounds.
    
    Size should be (endp - begp + 1) to include both boundary patches.
    """
    test_cases = [
        (BoundsType(begp=0, endp=0), 1),
        (BoundsType(begp=0, endp=9), 10),
        (BoundsType(begp=5, endp=14), 10),
        (BoundsType(begp=100, endp=199), 100),
        (BoundsType(begp=1000, endp=1999), 1000),
    ]
    
    for bounds, expected_size in test_cases:
        result = init_allocate(bounds)
        actual_size = result.forc_hgt_u_patch.shape[0]
        
        assert actual_size == expected_size, (
            f"Size mismatch for bounds {bounds}: {actual_size} != {expected_size}"
        )


def test_init_allocate_dtypes(bounds_test_cases):
    """
    Test that init_allocate creates arrays with correct data types.
    
    All arrays should be float64 for numerical precision.
    """
    for bounds in bounds_test_cases:
        result = init_allocate(bounds)
        
        assert result.forc_hgt_u_patch.dtype == jnp.float64
        assert result.u10_clm_patch.dtype == jnp.float64
        assert result.fv_patch.dtype == jnp.float64


# ============================================================================
# Tests for init_friction_velocity
# ============================================================================


def test_init_friction_velocity_wrapper_equivalence(bounds_test_cases):
    """
    Test that init_friction_velocity produces same results as init_allocate.
    
    init_friction_velocity is a convenience wrapper and should delegate
    to init_allocate, producing identical results.
    """
    for bounds in bounds_test_cases:
        result_wrapper = init_friction_velocity(bounds)
        result_direct = init_allocate(bounds)
        
        # Check shapes match
        assert result_wrapper.forc_hgt_u_patch.shape == result_direct.forc_hgt_u_patch.shape
        assert result_wrapper.u10_clm_patch.shape == result_direct.u10_clm_patch.shape
        assert result_wrapper.fv_patch.shape == result_direct.fv_patch.shape
        
        # Check both are NaN
        assert jnp.all(jnp.isnan(result_wrapper.forc_hgt_u_patch))
        assert jnp.all(jnp.isnan(result_wrapper.u10_clm_patch))
        assert jnp.all(jnp.isnan(result_wrapper.fv_patch))


def test_init_friction_velocity_return_type(bounds_test_cases):
    """
    Test that init_friction_velocity returns correct type.
    
    Should return FrictionVelType instance.
    """
    for bounds in bounds_test_cases:
        result = init_friction_velocity(bounds)
        assert isinstance(result, FrictionVelType), (
            f"Result should be FrictionVelType for bounds {bounds}"
        )


# ============================================================================
# Tests for update_frictionvel_type
# ============================================================================


def test_update_frictionvel_type_all_fields(test_data):
    """
    Test updating all fields in FrictionVelType.
    
    Verifies that when all fields are provided, they are all updated correctly.
    """
    case = test_data["update_frictionvel_type"][0]  # all_fields_update
    inputs = case["inputs"]
    expected = case["expected"]
    
    result = update_frictionvel_type(
        inputs["state"],
        forc_hgt_u_patch=inputs["forc_hgt_u_patch"],
        u10_clm_patch=inputs["u10_clm_patch"],
        fv_patch=inputs["fv_patch"],
    )
    
    assert jnp.allclose(result.forc_hgt_u_patch, expected["forc_hgt_u_patch"], atol=1e-6, rtol=1e-6), (
        "forc_hgt_u_patch values don't match expected"
    )
    assert jnp.allclose(result.u10_clm_patch, expected["u10_clm_patch"], atol=1e-6, rtol=1e-6), (
        "u10_clm_patch values don't match expected"
    )
    assert jnp.allclose(result.fv_patch, expected["fv_patch"], atol=1e-6, rtol=1e-6), (
        "fv_patch values don't match expected"
    )


def test_update_frictionvel_type_partial_update(test_data):
    """
    Test partial updates with some fields set to None.
    
    Verifies that None fields preserve original values while provided
    fields are updated. This tests the immutable update pattern.
    """
    case = test_data["update_frictionvel_type"][1]  # partial_update
    inputs = case["inputs"]
    expected = case["expected"]
    
    result = update_frictionvel_type(
        inputs["state"],
        forc_hgt_u_patch=inputs["forc_hgt_u_patch"],
        u10_clm_patch=inputs["u10_clm_patch"],
        fv_patch=inputs["fv_patch"],
    )
    
    # Check that unchanged fields are preserved
    assert jnp.allclose(result.forc_hgt_u_patch, expected["forc_hgt_u_patch"], atol=1e-6, rtol=1e-6), (
        "forc_hgt_u_patch should be preserved when None"
    )
    # Check that updated field has new values
    assert jnp.allclose(result.u10_clm_patch, expected["u10_clm_patch"], atol=1e-6, rtol=1e-6), (
        "u10_clm_patch should be updated"
    )
    # Check that unchanged fields are preserved
    assert jnp.allclose(result.fv_patch, expected["fv_patch"], atol=1e-6, rtol=1e-6), (
        "fv_patch should be preserved when None"
    )


def test_update_frictionvel_type_zero_values(test_data):
    """
    Test updating with zero values (calm conditions, ground level).
    
    Zero values are physically valid for calm conditions and should be
    handled correctly.
    """
    case = test_data["update_frictionvel_type"][2]  # zero_values
    inputs = case["inputs"]
    expected = case["expected"]
    
    result = update_frictionvel_type(
        inputs["state"],
        forc_hgt_u_patch=inputs["forc_hgt_u_patch"],
        u10_clm_patch=inputs["u10_clm_patch"],
        fv_patch=inputs["fv_patch"],
    )
    
    assert jnp.allclose(result.forc_hgt_u_patch, expected["forc_hgt_u_patch"], atol=1e-6, rtol=1e-6)
    assert jnp.allclose(result.u10_clm_patch, expected["u10_clm_patch"], atol=1e-6, rtol=1e-6)
    assert jnp.allclose(result.fv_patch, expected["fv_patch"], atol=1e-6, rtol=1e-6)


def test_update_frictionvel_type_extreme_values(test_data):
    """
    Test updating with extreme but physically valid values.
    
    Tests hurricane-force winds and high altitude conditions to ensure
    the function handles extreme atmospheric conditions.
    """
    case = test_data["update_frictionvel_type"][3]  # extreme_values
    inputs = case["inputs"]
    expected = case["expected"]
    
    result = update_frictionvel_type(
        inputs["state"],
        forc_hgt_u_patch=inputs["forc_hgt_u_patch"],
        u10_clm_patch=inputs["u10_clm_patch"],
        fv_patch=inputs["fv_patch"],
    )
    
    assert jnp.allclose(result.forc_hgt_u_patch, expected["forc_hgt_u_patch"], atol=1e-6, rtol=1e-6)
    assert jnp.allclose(result.u10_clm_patch, expected["u10_clm_patch"], atol=1e-6, rtol=1e-6)
    assert jnp.allclose(result.fv_patch, expected["fv_patch"], atol=1e-6, rtol=1e-6)


def test_update_frictionvel_type_very_small_values(test_data):
    """
    Test updating with very small but non-zero values.
    
    Tests numerical precision handling for near-calm conditions with
    values approaching machine epsilon.
    """
    case = test_data["update_frictionvel_type"][4]  # very_small_values
    inputs = case["inputs"]
    expected = case["expected"]
    
    result = update_frictionvel_type(
        inputs["state"],
        forc_hgt_u_patch=inputs["forc_hgt_u_patch"],
        u10_clm_patch=inputs["u10_clm_patch"],
        fv_patch=inputs["fv_patch"],
    )
    
    assert jnp.allclose(result.forc_hgt_u_patch, expected["forc_hgt_u_patch"], atol=1e-6, rtol=1e-6)
    assert jnp.allclose(result.u10_clm_patch, expected["u10_clm_patch"], atol=1e-6, rtol=1e-6)
    assert jnp.allclose(result.fv_patch, expected["fv_patch"], atol=1e-6, rtol=1e-6)


def test_update_frictionvel_type_immutability():
    """
    Test that update_frictionvel_type follows immutable update pattern.
    
    Original state should not be modified; a new state should be returned.
    This is critical for JAX functional programming paradigm.
    """
    original_state = FrictionVelType(
        forc_hgt_u_patch=jnp.array([10.0, 20.0]),
        u10_clm_patch=jnp.array([5.0, 6.0]),
        fv_patch=jnp.array([0.4, 0.5]),
    )
    
    # Store original values
    orig_height = original_state.forc_hgt_u_patch.copy()
    orig_wind = original_state.u10_clm_patch.copy()
    orig_fv = original_state.fv_patch.copy()
    
    # Update state
    new_state = update_frictionvel_type(
        original_state,
        forc_hgt_u_patch=jnp.array([30.0, 40.0]),
        u10_clm_patch=jnp.array([8.0, 9.0]),
        fv_patch=jnp.array([0.7, 0.8]),
    )
    
    # Original state should be unchanged
    assert jnp.allclose(original_state.forc_hgt_u_patch, orig_height, atol=1e-6, rtol=1e-6), (
        "Original state was modified (forc_hgt_u_patch)"
    )
    assert jnp.allclose(original_state.u10_clm_patch, orig_wind, atol=1e-6, rtol=1e-6), (
        "Original state was modified (u10_clm_patch)"
    )
    assert jnp.allclose(original_state.fv_patch, orig_fv, atol=1e-6, rtol=1e-6), (
        "Original state was modified (fv_patch)"
    )
    
    # New state should have updated values
    assert jnp.allclose(new_state.forc_hgt_u_patch, jnp.array([30.0, 40.0]), atol=1e-6, rtol=1e-6)
    assert jnp.allclose(new_state.u10_clm_patch, jnp.array([8.0, 9.0]), atol=1e-6, rtol=1e-6)
    assert jnp.allclose(new_state.fv_patch, jnp.array([0.7, 0.8]), atol=1e-6, rtol=1e-6)


def test_update_frictionvel_type_no_updates():
    """
    Test updating with all None values (no actual updates).
    
    When all update parameters are None, the returned state should be
    identical to the input state.
    """
    original_state = FrictionVelType(
        forc_hgt_u_patch=jnp.array([10.0, 20.0, 30.0]),
        u10_clm_patch=jnp.array([5.0, 6.0, 7.0]),
        fv_patch=jnp.array([0.4, 0.5, 0.6]),
    )
    
    result = update_frictionvel_type(
        original_state,
        forc_hgt_u_patch=None,
        u10_clm_patch=None,
        fv_patch=None,
    )
    
    assert jnp.allclose(result.forc_hgt_u_patch, original_state.forc_hgt_u_patch, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(result.u10_clm_patch, original_state.u10_clm_patch, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(result.fv_patch, original_state.fv_patch, atol=1e-6, rtol=1e-6)


def test_update_frictionvel_type_return_type():
    """
    Test that update_frictionvel_type returns correct type.
    
    Should always return FrictionVelType instance.
    """
    state = FrictionVelType(
        forc_hgt_u_patch=jnp.array([10.0]),
        u10_clm_patch=jnp.array([5.0]),
        fv_patch=jnp.array([0.4]),
    )
    
    result = update_frictionvel_type(state, u10_clm_patch=jnp.array([6.0]))
    
    assert isinstance(result, FrictionVelType), "Result should be FrictionVelType"


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_bounds_type_constraints():
    """
    Test BoundsType constraint: endp >= begp.
    
    Verifies that valid bounds satisfy the constraint and that the
    constraint is documented.
    """
    valid_bounds = [
        BoundsType(begp=0, endp=0),
        BoundsType(begp=0, endp=10),
        BoundsType(begp=5, endp=5),
        BoundsType(begp=100, endp=200),
    ]
    
    for bounds in valid_bounds:
        assert bounds.endp >= bounds.begp, (
            f"Bounds constraint violated: endp ({bounds.endp}) < begp ({bounds.begp})"
        )


def test_physical_constraints_non_negative():
    """
    Test that physical quantities are non-negative.
    
    Heights, wind speeds, and friction velocities must be >= 0 for
    physical realism.
    """
    state = FrictionVelType(
        forc_hgt_u_patch=jnp.array([0.0, 10.0, 100.0]),
        u10_clm_patch=jnp.array([0.0, 5.0, 50.0]),
        fv_patch=jnp.array([0.0, 0.4, 4.0]),
    )
    
    assert jnp.all(state.forc_hgt_u_patch >= 0), "Heights must be non-negative"
    assert jnp.all(state.u10_clm_patch >= 0), "Wind speeds must be non-negative"
    assert jnp.all(state.fv_patch >= 0), "Friction velocities must be non-negative"


def test_array_dimension_consistency():
    """
    Test that all arrays in FrictionVelType have consistent dimensions.
    
    All three fields should have the same shape for a given state.
    """
    sizes = [1, 10, 100, 1000]
    
    for size in sizes:
        state = init_frictionvel_type(size)
        
        assert state.forc_hgt_u_patch.shape == state.u10_clm_patch.shape, (
            f"Shape mismatch between forc_hgt_u_patch and u10_clm_patch for size {size}"
        )
        assert state.u10_clm_patch.shape == state.fv_patch.shape, (
            f"Shape mismatch between u10_clm_patch and fv_patch for size {size}"
        )


@pytest.mark.parametrize(
    "field_name,update_value",
    [
        ("forc_hgt_u_patch", jnp.array([15.0, 25.0])),
        ("u10_clm_patch", jnp.array([7.5, 8.5])),
        ("fv_patch", jnp.array([0.55, 0.65])),
    ],
)
def test_update_single_field(field_name, update_value):
    """
    Test updating each field individually.
    
    Parametrized test to verify that each field can be updated independently
    while preserving other fields.
    """
    state = FrictionVelType(
        forc_hgt_u_patch=jnp.array([10.0, 20.0]),
        u10_clm_patch=jnp.array([5.0, 6.0]),
        fv_patch=jnp.array([0.4, 0.5]),
    )
    
    kwargs = {field_name: update_value}
    result = update_frictionvel_type(state, **kwargs)
    
    # Check that the updated field has new values
    assert jnp.allclose(getattr(result, field_name), update_value, atol=1e-6, rtol=1e-6), (
        f"Field {field_name} was not updated correctly"
    )
    
    # Check that other fields are preserved
    for other_field in ["forc_hgt_u_patch", "u10_clm_patch", "fv_patch"]:
        if other_field != field_name:
            assert jnp.allclose(
                getattr(result, other_field),
                getattr(state, other_field),
                atol=1e-6,
                rtol=1e-6,
            ), f"Field {other_field} should be preserved when updating {field_name}"


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_workflow():
    """
    Integration test: Initialize, allocate, and update workflow.
    
    Tests a complete workflow of:
    1. Creating bounds
    2. Initializing state
    3. Updating with realistic values
    4. Verifying final state
    """
    # Step 1: Create bounds
    bounds = BoundsType(begp=0, endp=9)
    
    # Step 2: Initialize state
    state = init_friction_velocity(bounds)
    
    # Verify initialization
    assert state.forc_hgt_u_patch.shape == (10,)
    assert jnp.all(jnp.isnan(state.forc_hgt_u_patch))
    
    # Step 3: Update with realistic atmospheric values
    updated_state = update_frictionvel_type(
        state,
        forc_hgt_u_patch=jnp.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]),
        u10_clm_patch=jnp.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        fv_patch=jnp.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]),
    )
    
    # Step 4: Verify final state
    assert not jnp.any(jnp.isnan(updated_state.forc_hgt_u_patch)), "Heights should not be NaN"
    assert not jnp.any(jnp.isnan(updated_state.u10_clm_patch)), "Wind speeds should not be NaN"
    assert not jnp.any(jnp.isnan(updated_state.fv_patch)), "Friction velocities should not be NaN"
    
    assert jnp.all(updated_state.forc_hgt_u_patch >= 0), "Heights must be non-negative"
    assert jnp.all(updated_state.u10_clm_patch >= 0), "Wind speeds must be non-negative"
    assert jnp.all(updated_state.fv_patch >= 0), "Friction velocities must be non-negative"


def test_multiple_sequential_updates():
    """
    Test multiple sequential updates to the same state.
    
    Verifies that the state can be updated multiple times in sequence,
    with each update building on the previous one.
    """
    # Initialize
    state = init_frictionvel_type(3)
    
    # First update: set heights
    state = update_frictionvel_type(
        state,
        forc_hgt_u_patch=jnp.array([10.0, 20.0, 30.0]),
    )
    assert jnp.allclose(state.forc_hgt_u_patch, jnp.array([10.0, 20.0, 30.0]), atol=1e-6, rtol=1e-6)
    assert jnp.all(jnp.isnan(state.u10_clm_patch))
    assert jnp.all(jnp.isnan(state.fv_patch))
    
    # Second update: set wind speeds
    state = update_frictionvel_type(
        state,
        u10_clm_patch=jnp.array([5.0, 6.0, 7.0]),
    )
    assert jnp.allclose(state.forc_hgt_u_patch, jnp.array([10.0, 20.0, 30.0]), atol=1e-6, rtol=1e-6)
    assert jnp.allclose(state.u10_clm_patch, jnp.array([5.0, 6.0, 7.0]), atol=1e-6, rtol=1e-6)
    assert jnp.all(jnp.isnan(state.fv_patch))
    
    # Third update: set friction velocities
    state = update_frictionvel_type(
        state,
        fv_patch=jnp.array([0.4, 0.5, 0.6]),
    )
    assert jnp.allclose(state.forc_hgt_u_patch, jnp.array([10.0, 20.0, 30.0]), atol=1e-6, rtol=1e-6)
    assert jnp.allclose(state.u10_clm_patch, jnp.array([5.0, 6.0, 7.0]), atol=1e-6, rtol=1e-6)
    assert jnp.allclose(state.fv_patch, jnp.array([0.4, 0.5, 0.6]), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])