"""
Comprehensive pytest suite for WaterStateType module.

This module tests the initialization, allocation, update, and query functions
for water state management in the CLM biogeophysics component.

Test Coverage:
- Initialization functions with various domain sizes
- Bounds-based allocation with NaN and zero initialization
- Immutable state updates
- Water mass calculations (total and soil-only)
- Edge cases: minimum dimensions, zero values, negative fills
- Shape and dtype validation
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_biogeophys.WaterStateType import (
    Bounds,
    WaterState,
    get_soil_water_mass,
    get_total_water_mass,
    init_allocate_water_state,
    init_water_state_from_bounds,
    init_waterstate,
    update_waterstate,
)


@pytest.fixture
def test_data():
    """
    Load test data for WaterStateType functions.
    
    Returns:
        dict: Test cases with inputs, expected outputs, and metadata
    """
    return {
        "test_cases": [
            {
                "name": "test_init_waterstate_nominal_small_domain",
                "function": "init_waterstate",
                "inputs": {
                    "ncols": 5,
                    "npatches": 10,
                    "nlevgrnd": 15,
                    "nlevsno": 5,
                    "fill_value": 0.0,
                },
                "expected_shapes": {
                    "bw_col": (5, 5),
                    "h2osno_col": (5,),
                    "h2osoi_liq_col": (5, 20),
                    "h2osoi_ice_col": (5, 20),
                    "h2osoi_vol_col": (5, 15),
                    "h2osfc_col": (5,),
                    "q_ref2m_patch": (10,),
                    "frac_sno_eff_col": (5,),
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Small domain with typical layer counts, zero initialization",
                },
            },
            {
                "name": "test_init_waterstate_nominal_large_domain",
                "function": "init_waterstate",
                "inputs": {
                    "ncols": 100,
                    "npatches": 500,
                    "nlevgrnd": 20,
                    "nlevsno": 10,
                    "fill_value": 1.5,
                },
                "expected_shapes": {
                    "bw_col": (100, 10),
                    "h2osno_col": (100,),
                    "h2osoi_liq_col": (100, 30),
                    "h2osoi_ice_col": (100, 30),
                    "h2osoi_vol_col": (100, 20),
                    "h2osfc_col": (100,),
                    "q_ref2m_patch": (500,),
                    "frac_sno_eff_col": (100,),
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Large domain with non-zero initialization value",
                },
            },
            {
                "name": "test_init_waterstate_edge_minimum_dimensions",
                "function": "init_waterstate",
                "inputs": {
                    "ncols": 1,
                    "npatches": 1,
                    "nlevgrnd": 1,
                    "nlevsno": 1,
                    "fill_value": 0.0,
                },
                "expected_shapes": {
                    "bw_col": (1, 1),
                    "h2osno_col": (1,),
                    "h2osoi_liq_col": (1, 2),
                    "h2osoi_ice_col": (1, 2),
                    "h2osoi_vol_col": (1, 1),
                    "h2osfc_col": (1,),
                    "q_ref2m_patch": (1,),
                    "frac_sno_eff_col": (1,),
                },
                "metadata": {
                    "type": "edge",
                    "description": "Minimum valid dimensions (all = 1)",
                },
            },
            {
                "name": "test_init_waterstate_edge_negative_fill",
                "function": "init_waterstate",
                "inputs": {
                    "ncols": 10,
                    "npatches": 20,
                    "nlevgrnd": 10,
                    "nlevsno": 5,
                    "fill_value": -999.0,
                },
                "expected_shapes": {
                    "bw_col": (10, 5),
                    "h2osno_col": (10,),
                    "h2osoi_liq_col": (10, 15),
                    "h2osoi_ice_col": (10, 15),
                    "h2osoi_vol_col": (10, 10),
                    "h2osfc_col": (10,),
                    "q_ref2m_patch": (20,),
                    "frac_sno_eff_col": (10,),
                },
                "metadata": {
                    "type": "edge",
                    "description": "Negative fill value (sentinel/missing data pattern)",
                },
            },
        ]
    }


@pytest.fixture
def allocate_test_data():
    """Test data for init_allocate_water_state function."""
    return {
        "test_cases": [
            {
                "name": "test_init_allocate_water_state_nominal_with_nan",
                "inputs": {
                    "bounds": Bounds(begp=0, endp=50, begc=0, endc=25, begg=0, endg=10),
                    "nlevsno": 5,
                    "nlevgrnd": 15,
                    "use_nan": True,
                },
                "expected_shapes": {
                    "bw_col": (25, 5),
                    "h2osno_col": (25,),
                    "h2osoi_liq_col": (25, 20),
                    "h2osoi_ice_col": (25, 20),
                    "h2osoi_vol_col": (25, 15),
                    "h2osfc_col": (25,),
                    "q_ref2m_patch": (50,),
                    "frac_sno_eff_col": (25,),
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Allocate with NaN initialization",
                },
            },
            {
                "name": "test_init_allocate_water_state_nominal_with_zeros",
                "inputs": {
                    "bounds": Bounds(begp=10, endp=110, begc=5, endc=55, begg=0, endg=20),
                    "nlevsno": 8,
                    "nlevgrnd": 25,
                    "use_nan": False,
                },
                "expected_shapes": {
                    "bw_col": (50, 8),
                    "h2osno_col": (50,),
                    "h2osoi_liq_col": (50, 33),
                    "h2osoi_ice_col": (50, 33),
                    "h2osoi_vol_col": (50, 25),
                    "h2osfc_col": (50,),
                    "q_ref2m_patch": (100,),
                    "frac_sno_eff_col": (50,),
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Allocate with zero initialization, non-zero starting indices",
                },
            },
            {
                "name": "test_init_allocate_water_state_edge_single_element",
                "inputs": {
                    "bounds": Bounds(begp=0, endp=1, begc=0, endc=1, begg=0, endg=1),
                    "nlevsno": 1,
                    "nlevgrnd": 1,
                    "use_nan": False,
                },
                "expected_shapes": {
                    "bw_col": (1, 1),
                    "h2osno_col": (1,),
                    "h2osoi_liq_col": (1, 2),
                    "h2osoi_ice_col": (1, 2),
                    "h2osoi_vol_col": (1, 1),
                    "h2osfc_col": (1,),
                    "q_ref2m_patch": (1,),
                    "frac_sno_eff_col": (1,),
                },
                "metadata": {
                    "type": "edge",
                    "description": "Single element domain with minimum layers",
                },
            },
        ]
    }


@pytest.fixture
def update_test_data():
    """Test data for update_waterstate function."""
    return {
        "state": WaterState(
            bw_col=jnp.array([[100.0, 150.0, 200.0], [120.0, 160.0, 210.0]]),
            h2osno_col=jnp.array([50.0, 75.0]),
            h2osoi_liq_col=jnp.array(
                [[10.0, 15.0, 20.0, 25.0], [12.0, 18.0, 22.0, 28.0]]
            ),
            h2osoi_ice_col=jnp.array(
                [[5.0, 8.0, 10.0, 12.0], [6.0, 9.0, 11.0, 13.0]]
            ),
            h2osoi_vol_col=jnp.array([[0.3, 0.35], [0.32, 0.38]]),
            h2osfc_col=jnp.array([2.0, 3.0]),
            q_ref2m_patch=jnp.array([0.008, 0.01, 0.012]),
            frac_sno_eff_col=jnp.array([0.6, 0.7]),
        ),
        "updates": {
            "h2osno_col": jnp.array([60.0, 80.0]),
            "frac_sno_eff_col": jnp.array([0.65, 0.75]),
        },
    }


@pytest.fixture
def mass_calculation_test_data():
    """Test data for water mass calculation functions."""
    return {
        "state": WaterState(
            bw_col=jnp.array([[100.0, 150.0], [120.0, 160.0], [110.0, 155.0]]),
            h2osno_col=jnp.array([50.0, 75.0, 60.0]),
            h2osoi_liq_col=jnp.array(
                [[10.0, 15.0, 20.0], [12.0, 18.0, 22.0], [11.0, 16.0, 21.0]]
            ),
            h2osoi_ice_col=jnp.array(
                [[5.0, 8.0, 10.0], [6.0, 9.0, 11.0], [5.5, 8.5, 10.5]]
            ),
            h2osoi_vol_col=jnp.array([[0.3], [0.32], [0.31]]),
            h2osfc_col=jnp.array([2.0, 3.0, 2.5]),
            q_ref2m_patch=jnp.array([0.008, 0.01]),
            frac_sno_eff_col=jnp.array([0.6, 0.7, 0.65]),
        ),
        "nlevsno": 2,
    }


# ============================================================================
# Tests for init_waterstate
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        "test_init_waterstate_nominal_small_domain",
        "test_init_waterstate_nominal_large_domain",
        "test_init_waterstate_edge_minimum_dimensions",
        "test_init_waterstate_edge_negative_fill",
    ],
)
def test_init_waterstate_shapes(test_data, test_case):
    """
    Test that init_waterstate creates arrays with correct shapes.
    
    Verifies that all WaterState fields have the expected dimensions
    based on input parameters (ncols, npatches, nlevgrnd, nlevsno).
    """
    case = next(tc for tc in test_data["test_cases"] if tc["name"] == test_case)
    inputs = case["inputs"]
    expected_shapes = case["expected_shapes"]
    
    result = init_waterstate(**inputs)
    
    assert isinstance(result, WaterState), "Result should be a WaterState instance"
    
    for field_name, expected_shape in expected_shapes.items():
        field_value = getattr(result, field_name)
        assert (
            field_value.shape == expected_shape
        ), f"{field_name} shape mismatch: got {field_value.shape}, expected {expected_shape}"


@pytest.mark.parametrize(
    "test_case",
    [
        "test_init_waterstate_nominal_small_domain",
        "test_init_waterstate_nominal_large_domain",
        "test_init_waterstate_edge_negative_fill",
    ],
)
def test_init_waterstate_values(test_data, test_case):
    """
    Test that init_waterstate initializes arrays with correct fill values.
    
    Verifies that all array elements are set to the specified fill_value.
    """
    case = next(tc for tc in test_data["test_cases"] if tc["name"] == test_case)
    inputs = case["inputs"]
    fill_value = inputs["fill_value"]
    
    result = init_waterstate(**inputs)
    
    # Check all fields are filled with the correct value
    for field_name in result._fields:
        field_value = getattr(result, field_name)
        assert jnp.allclose(
            field_value, fill_value, atol=1e-10
        ), f"{field_name} not initialized to {fill_value}"


def test_init_waterstate_dtypes(test_data):
    """
    Test that init_waterstate creates arrays with correct data types.
    
    All arrays should be float32 or float64 for JAX compatibility.
    """
    case = test_data["test_cases"][0]
    inputs = case["inputs"]
    
    result = init_waterstate(**inputs)
    
    for field_name in result._fields:
        field_value = getattr(result, field_name)
        assert jnp.issubdtype(
            field_value.dtype, jnp.floating
        ), f"{field_name} should be floating point type, got {field_value.dtype}"


def test_init_waterstate_immutability():
    """
    Test that WaterState is immutable (NamedTuple behavior).
    
    Attempting to modify fields should raise an AttributeError.
    """
    state = init_waterstate(ncols=5, npatches=10, nlevgrnd=15, nlevsno=5)
    
    with pytest.raises(AttributeError):
        state.h2osno_col = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])


# ============================================================================
# Tests for init_allocate_water_state
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        "test_init_allocate_water_state_nominal_with_nan",
        "test_init_allocate_water_state_nominal_with_zeros",
        "test_init_allocate_water_state_edge_single_element",
    ],
)
def test_init_allocate_water_state_shapes(allocate_test_data, test_case):
    """
    Test that init_allocate_water_state creates arrays with correct shapes.
    
    Verifies that array dimensions are correctly computed from Bounds object.
    """
    case = next(
        tc for tc in allocate_test_data["test_cases"] if tc["name"] == test_case
    )
    inputs = case["inputs"]
    expected_shapes = case["expected_shapes"]
    
    result = init_allocate_water_state(**inputs)
    
    assert isinstance(result, WaterState), "Result should be a WaterState instance"
    
    for field_name, expected_shape in expected_shapes.items():
        field_value = getattr(result, field_name)
        assert (
            field_value.shape == expected_shape
        ), f"{field_name} shape mismatch: got {field_value.shape}, expected {expected_shape}"


def test_init_allocate_water_state_nan_initialization(allocate_test_data):
    """
    Test that init_allocate_water_state correctly initializes with NaN.
    
    When use_nan=True, all array elements should be NaN for detecting
    uninitialized data.
    """
    case = allocate_test_data["test_cases"][0]  # NaN case
    inputs = case["inputs"]
    
    result = init_allocate_water_state(**inputs)
    
    for field_name in result._fields:
        field_value = getattr(result, field_name)
        assert jnp.all(
            jnp.isnan(field_value)
        ), f"{field_name} should be all NaN when use_nan=True"


def test_init_allocate_water_state_zero_initialization(allocate_test_data):
    """
    Test that init_allocate_water_state correctly initializes with zeros.
    
    When use_nan=False, all array elements should be 0.0.
    """
    case = allocate_test_data["test_cases"][1]  # Zero case
    inputs = case["inputs"]
    
    result = init_allocate_water_state(**inputs)
    
    for field_name in result._fields:
        field_value = getattr(result, field_name)
        assert jnp.allclose(
            field_value, 0.0, atol=1e-10
        ), f"{field_name} should be all zeros when use_nan=False"


def test_init_allocate_water_state_bounds_calculation():
    """
    Test that array sizes are correctly calculated from Bounds indices.
    
    Array size should be (end - beg) for both columns and patches.
    """
    bounds = Bounds(begp=10, endp=30, begc=5, endc=15, begg=0, endg=5)
    nlevsno = 3
    nlevgrnd = 10
    
    result = init_allocate_water_state(bounds, nlevsno, nlevgrnd, use_nan=False)
    
    expected_ncols = bounds.endc - bounds.begc
    expected_npatches = bounds.endp - bounds.begp
    
    assert result.h2osno_col.shape[0] == expected_ncols, "Column count mismatch"
    assert result.q_ref2m_patch.shape[0] == expected_npatches, "Patch count mismatch"


# ============================================================================
# Tests for init_water_state_from_bounds
# ============================================================================


def test_init_water_state_from_bounds_shapes():
    """
    Test that init_water_state_from_bounds creates correctly shaped arrays.
    
    This function should behave similarly to init_allocate_water_state
    with use_nan=False.
    """
    bounds = Bounds(begp=0, endp=20, begc=0, endc=10, begg=0, endg=5)
    nlevsno = 5
    nlevgrnd = 15
    
    result = init_water_state_from_bounds(bounds, nlevsno, nlevgrnd)
    
    expected_ncols = bounds.endc - bounds.begc
    expected_npatches = bounds.endp - bounds.begp
    
    assert result.bw_col.shape == (expected_ncols, nlevsno)
    assert result.h2osno_col.shape == (expected_ncols,)
    assert result.h2osoi_liq_col.shape == (expected_ncols, nlevsno + nlevgrnd)
    assert result.h2osoi_ice_col.shape == (expected_ncols, nlevsno + nlevgrnd)
    assert result.h2osoi_vol_col.shape == (expected_ncols, nlevgrnd)
    assert result.h2osfc_col.shape == (expected_ncols,)
    assert result.q_ref2m_patch.shape == (expected_npatches,)
    assert result.frac_sno_eff_col.shape == (expected_ncols,)


def test_init_water_state_from_bounds_zero_initialization():
    """
    Test that init_water_state_from_bounds initializes with zeros.
    
    Default behavior should initialize all arrays to 0.0.
    """
    bounds = Bounds(begp=0, endp=10, begc=0, endc=5, begg=0, endg=2)
    nlevsno = 3
    nlevgrnd = 8
    
    result = init_water_state_from_bounds(bounds, nlevsno, nlevgrnd)
    
    for field_name in result._fields:
        field_value = getattr(result, field_name)
        assert jnp.allclose(
            field_value, 0.0, atol=1e-10
        ), f"{field_name} should be initialized to zero"


# ============================================================================
# Tests for update_waterstate
# ============================================================================


def test_update_waterstate_partial_update(update_test_data):
    """
    Test that update_waterstate correctly updates specified fields.
    
    Only fields in the updates dict should change; others should remain
    unchanged.
    """
    state = update_test_data["state"]
    updates = update_test_data["updates"]
    
    result = update_waterstate(state, **updates)
    
    # Check updated fields
    assert jnp.allclose(
        result.h2osno_col, updates["h2osno_col"], atol=1e-10
    ), "h2osno_col not updated correctly"
    assert jnp.allclose(
        result.frac_sno_eff_col, updates["frac_sno_eff_col"], atol=1e-10
    ), "frac_sno_eff_col not updated correctly"
    
    # Check unchanged fields
    assert jnp.allclose(
        result.bw_col, state.bw_col, atol=1e-10
    ), "bw_col should not change"
    assert jnp.allclose(
        result.h2osoi_liq_col, state.h2osoi_liq_col, atol=1e-10
    ), "h2osoi_liq_col should not change"
    assert jnp.allclose(
        result.h2osoi_ice_col, state.h2osoi_ice_col, atol=1e-10
    ), "h2osoi_ice_col should not change"
    assert jnp.allclose(
        result.h2osoi_vol_col, state.h2osoi_vol_col, atol=1e-10
    ), "h2osoi_vol_col should not change"
    assert jnp.allclose(
        result.h2osfc_col, state.h2osfc_col, atol=1e-10
    ), "h2osfc_col should not change"
    assert jnp.allclose(
        result.q_ref2m_patch, state.q_ref2m_patch, atol=1e-10
    ), "q_ref2m_patch should not change"


def test_update_waterstate_immutability(update_test_data):
    """
    Test that update_waterstate returns a new instance.
    
    The original state should remain unchanged (immutable update pattern).
    """
    state = update_test_data["state"]
    updates = update_test_data["updates"]
    
    original_h2osno = state.h2osno_col.copy()
    
    result = update_waterstate(state, **updates)
    
    # Original state should be unchanged
    assert jnp.allclose(
        state.h2osno_col, original_h2osno, atol=1e-10
    ), "Original state was modified"
    
    # Result should be different
    assert not jnp.allclose(
        result.h2osno_col, original_h2osno, atol=1e-10
    ), "Result should have updated values"


def test_update_waterstate_no_updates():
    """
    Test that update_waterstate with no updates returns equivalent state.
    
    When no updates are provided, the returned state should be identical
    to the input state.
    """
    state = init_waterstate(ncols=5, npatches=10, nlevgrnd=15, nlevsno=5, fill_value=1.0)
    
    result = update_waterstate(state)
    
    for field_name in state._fields:
        original_value = getattr(state, field_name)
        result_value = getattr(result, field_name)
        assert jnp.allclose(
            original_value, result_value, atol=1e-10
        ), f"{field_name} changed without update"


def test_update_waterstate_multiple_fields():
    """
    Test that update_waterstate can update multiple fields simultaneously.
    """
    state = init_waterstate(ncols=3, npatches=5, nlevgrnd=10, nlevsno=3, fill_value=0.0)
    
    updates = {
        "h2osno_col": jnp.array([10.0, 20.0, 30.0]),
        "h2osfc_col": jnp.array([1.0, 2.0, 3.0]),
        "frac_sno_eff_col": jnp.array([0.5, 0.6, 0.7]),
    }
    
    result = update_waterstate(state, **updates)
    
    assert jnp.allclose(result.h2osno_col, updates["h2osno_col"], atol=1e-10)
    assert jnp.allclose(result.h2osfc_col, updates["h2osfc_col"], atol=1e-10)
    assert jnp.allclose(result.frac_sno_eff_col, updates["frac_sno_eff_col"], atol=1e-10)


# ============================================================================
# Tests for get_total_water_mass
# ============================================================================


def test_get_total_water_mass_shape(mass_calculation_test_data):
    """
    Test that get_total_water_mass returns correct shape.
    
    Output should be 1D array with length equal to number of columns.
    """
    state = mass_calculation_test_data["state"]
    
    result = get_total_water_mass(state)
    
    expected_shape = (state.h2osno_col.shape[0],)
    assert (
        result.shape == expected_shape
    ), f"Shape mismatch: got {result.shape}, expected {expected_shape}"


def test_get_total_water_mass_calculation(mass_calculation_test_data):
    """
    Test that get_total_water_mass correctly sums all water components.
    
    Total mass should equal sum of:
    - h2osno_col (snow water)
    - h2osfc_col (surface water)
    - h2osoi_liq_col (liquid water in all layers)
    - h2osoi_ice_col (ice in all layers)
    """
    state = mass_calculation_test_data["state"]
    
    result = get_total_water_mass(state)
    
    # Manual calculation
    expected = (
        state.h2osno_col
        + state.h2osfc_col
        + jnp.sum(state.h2osoi_liq_col, axis=1)
        + jnp.sum(state.h2osoi_ice_col, axis=1)
    )
    
    assert jnp.allclose(
        result, expected, atol=1e-6, rtol=1e-6
    ), "Total water mass calculation incorrect"


def test_get_total_water_mass_zero_state():
    """
    Test get_total_water_mass with all-zero state.
    
    Should return zeros for all columns.
    """
    state = init_waterstate(ncols=5, npatches=10, nlevgrnd=15, nlevsno=5, fill_value=0.0)
    
    result = get_total_water_mass(state)
    
    assert jnp.allclose(result, 0.0, atol=1e-10), "Zero state should have zero total mass"


def test_get_total_water_mass_positive_values():
    """
    Test that get_total_water_mass returns non-negative values.
    
    Water mass should always be >= 0 for physical realism.
    """
    state = init_waterstate(ncols=10, npatches=20, nlevgrnd=15, nlevsno=5, fill_value=10.0)
    
    result = get_total_water_mass(state)
    
    assert jnp.all(result >= 0.0), "Total water mass should be non-negative"


# ============================================================================
# Tests for get_soil_water_mass
# ============================================================================


def test_get_soil_water_mass_shape(mass_calculation_test_data):
    """
    Test that get_soil_water_mass returns correct shape.
    
    Output should be 1D array with length equal to number of columns.
    """
    state = mass_calculation_test_data["state"]
    nlevsno = mass_calculation_test_data["nlevsno"]
    
    result = get_soil_water_mass(state, nlevsno)
    
    expected_shape = (state.h2osno_col.shape[0],)
    assert (
        result.shape == expected_shape
    ), f"Shape mismatch: got {result.shape}, expected {expected_shape}"


def test_get_soil_water_mass_excludes_snow():
    """
    Test that get_soil_water_mass excludes snow layers.
    
    Should only sum water from soil layers (indices >= nlevsno).
    """
    # Create state with distinct values in snow vs soil layers
    state = WaterState(
        bw_col=jnp.array([[100.0, 150.0], [120.0, 160.0]]),
        h2osno_col=jnp.array([50.0, 75.0]),
        h2osoi_liq_col=jnp.array(
            [[1.0, 2.0, 10.0, 15.0], [3.0, 4.0, 12.0, 18.0]]
        ),  # First 2 are snow
        h2osoi_ice_col=jnp.array(
            [[0.5, 1.0, 5.0, 8.0], [0.6, 1.2, 6.0, 9.0]]
        ),  # First 2 are snow
        h2osoi_vol_col=jnp.array([[0.3, 0.35], [0.32, 0.38]]),
        h2osfc_col=jnp.array([2.0, 3.0]),
        q_ref2m_patch=jnp.array([0.008, 0.01]),
        frac_sno_eff_col=jnp.array([0.6, 0.7]),
    )
    nlevsno = 2
    
    result = get_soil_water_mass(state, nlevsno)
    
    # Manual calculation: only soil layers (indices 2 and 3)
    expected = jnp.array([
        10.0 + 15.0 + 5.0 + 8.0,  # Column 0: soil liquid + soil ice
        12.0 + 18.0 + 6.0 + 9.0,  # Column 1: soil liquid + soil ice
    ])
    
    assert jnp.allclose(
        result, expected, atol=1e-6, rtol=1e-6
    ), "Soil water mass should exclude snow layers"


def test_get_soil_water_mass_zero_snow():
    """
    Test get_soil_water_mass with zero snow layers.
    
    When snow layers have zero water, soil mass should still be calculated
    correctly from soil layers only.
    """
    state = WaterState(
        bw_col=jnp.array([[0.0, 0.0], [0.0, 0.0]]),
        h2osno_col=jnp.array([0.0, 0.0]),
        h2osoi_liq_col=jnp.array(
            [[0.0, 0.0, 25.0, 30.0], [0.0, 0.0, 28.0, 32.0]]
        ),
        h2osoi_ice_col=jnp.array(
            [[0.0, 0.0, 15.0, 18.0], [0.0, 0.0, 16.0, 19.0]]
        ),
        h2osoi_vol_col=jnp.array([[0.35, 0.4], [0.38, 0.42]]),
        h2osfc_col=jnp.array([0.0, 0.0]),
        q_ref2m_patch=jnp.array([0.005]),
        frac_sno_eff_col=jnp.array([0.0, 0.0]),
    )
    nlevsno = 2
    
    result = get_soil_water_mass(state, nlevsno)
    
    expected = jnp.array([
        25.0 + 30.0 + 15.0 + 18.0,  # Column 0
        28.0 + 32.0 + 16.0 + 19.0,  # Column 1
    ])
    
    assert jnp.allclose(
        result, expected, atol=1e-6, rtol=1e-6
    ), "Soil water mass incorrect with zero snow"


def test_get_soil_water_mass_positive_values():
    """
    Test that get_soil_water_mass returns non-negative values.
    
    Soil water mass should always be >= 0 for physical realism.
    """
    state = init_waterstate(ncols=10, npatches=20, nlevgrnd=15, nlevsno=5, fill_value=5.0)
    
    result = get_soil_water_mass(state, nlevsno=5)
    
    assert jnp.all(result >= 0.0), "Soil water mass should be non-negative"


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_bounds_with_zero_gridcells():
    """
    Test Bounds creation with default zero gridcell indices.
    
    begg and endg are optional and default to 0.
    """
    bounds = Bounds(begp=0, endp=10, begc=0, endc=5)
    
    assert bounds.begg == 0, "Default begg should be 0"
    assert bounds.endg == 0, "Default endg should be 0"


def test_waterstate_field_access():
    """
    Test that WaterState fields can be accessed by name.
    
    NamedTuple should allow both attribute and index access.
    """
    state = init_waterstate(ncols=2, npatches=3, nlevgrnd=5, nlevsno=2, fill_value=1.0)
    
    # Attribute access
    assert hasattr(state, "bw_col"), "Should have bw_col attribute"
    assert hasattr(state, "h2osno_col"), "Should have h2osno_col attribute"
    
    # Index access
    assert state[0] is state.bw_col, "Index 0 should be bw_col"
    assert state[1] is state.h2osno_col, "Index 1 should be h2osno_col"


def test_large_domain_memory_efficiency():
    """
    Test that large domains can be initialized without memory errors.
    
    This is a smoke test for memory allocation with realistic domain sizes.
    """
    # Realistic large domain
    ncols = 10000
    npatches = 50000
    nlevgrnd = 25
    nlevsno = 10
    
    try:
        state = init_waterstate(
            ncols=ncols,
            npatches=npatches,
            nlevgrnd=nlevgrnd,
            nlevsno=nlevsno,
            fill_value=0.0,
        )
        assert state is not None, "Large domain initialization failed"
    except MemoryError:
        pytest.skip("Insufficient memory for large domain test")


def test_consistency_between_init_functions():
    """
    Test that different initialization functions produce consistent results.
    
    init_waterstate and init_allocate_water_state should create equivalent
    structures when given equivalent parameters.
    """
    ncols = 10
    npatches = 20
    nlevgrnd = 15
    nlevsno = 5
    
    bounds = Bounds(begp=0, endp=npatches, begc=0, endc=ncols, begg=0, endg=5)
    
    state1 = init_waterstate(ncols, npatches, nlevgrnd, nlevsno, fill_value=0.0)
    state2 = init_allocate_water_state(bounds, nlevsno, nlevgrnd, use_nan=False)
    
    # Check shapes match
    for field_name in state1._fields:
        shape1 = getattr(state1, field_name).shape
        shape2 = getattr(state2, field_name).shape
        assert shape1 == shape2, f"{field_name} shape mismatch between init functions"


def test_update_with_invalid_field():
    """
    Test that update_waterstate handles invalid field names gracefully.
    
    Should raise TypeError when trying to update non-existent field.
    """
    state = init_waterstate(ncols=5, npatches=10, nlevgrnd=15, nlevsno=5)
    
    with pytest.raises(TypeError):
        update_waterstate(state, invalid_field=jnp.array([1.0, 2.0, 3.0]))


# ============================================================================
# Documentation Tests
# ============================================================================


def test_waterstate_docstring():
    """
    Test that WaterState has proper documentation.
    
    NamedTuple should have accessible field information.
    """
    assert hasattr(WaterState, "_fields"), "WaterState should have _fields attribute"
    assert len(WaterState._fields) == 8, "WaterState should have 8 fields"


def test_bounds_docstring():
    """
    Test that Bounds has proper documentation.
    
    NamedTuple should have accessible field information.
    """
    assert hasattr(Bounds, "_fields"), "Bounds should have _fields attribute"
    assert len(Bounds._fields) == 6, "Bounds should have 6 fields"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])