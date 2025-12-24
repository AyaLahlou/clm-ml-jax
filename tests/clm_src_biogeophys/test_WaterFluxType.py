"""
Comprehensive pytest suite for WaterFluxType module.

This module tests the water flux state management functions including:
- Initialization of water flux types
- Array allocation with bounds
- State updates for evaporation/condensation fluxes

Tests cover nominal cases, edge cases, and special conditions for scientific
computing with JAX arrays in land surface modeling.
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_biogeophys.WaterFluxType import (
    BoundsType,
    WaterFluxType,
    init,
    init_allocate,
    init_waterflux_type,
    update_qflx_evap_tot_patch,
)


@pytest.fixture
def test_data():
    """
    Load test data for WaterFluxType functions.
    
    Returns:
        dict: Test cases organized by function name with inputs and metadata.
    """
    return {
        "init_waterflux_type": [
            {
                "name": "test_init_waterflux_type_single_patch",
                "inputs": {"n_patches": 1},
                "metadata": {
                    "type": "nominal",
                    "description": "Initialize water flux for single patch domain",
                },
            },
            {
                "name": "test_init_waterflux_type_typical_domain",
                "inputs": {"n_patches": 100},
                "metadata": {
                    "type": "nominal",
                    "description": "Initialize water flux for typical 100-patch domain",
                },
            },
            {
                "name": "test_init_waterflux_type_large_domain",
                "inputs": {"n_patches": 10000},
                "metadata": {
                    "type": "special",
                    "description": "Initialize water flux for large-scale simulation with 10k patches",
                },
            },
        ],
        "init_allocate": [
            {
                "name": "test_init_allocate_small_bounds",
                "inputs": {
                    "waterflux_state": WaterFluxType(qflx_evap_tot_patch=None),
                    "bounds": BoundsType(begp=1, endp=5),
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Allocate arrays for small domain with 5 patches",
                },
            },
            {
                "name": "test_init_allocate_single_patch_bounds",
                "inputs": {
                    "waterflux_state": WaterFluxType(qflx_evap_tot_patch=None),
                    "bounds": BoundsType(begp=1, endp=1),
                },
                "metadata": {
                    "type": "edge",
                    "description": "Allocate for minimum valid domain (single patch)",
                },
            },
            {
                "name": "test_init_allocate_offset_bounds",
                "inputs": {
                    "waterflux_state": WaterFluxType(qflx_evap_tot_patch=None),
                    "bounds": BoundsType(begp=50, endp=150),
                },
                "metadata": {
                    "type": "special",
                    "description": "Allocate with non-zero starting index (Fortran 1-based convention)",
                },
            },
            {
                "name": "test_init_allocate_large_bounds",
                "inputs": {
                    "waterflux_state": WaterFluxType(qflx_evap_tot_patch=None),
                    "bounds": BoundsType(begp=1, endp=5000),
                },
                "metadata": {
                    "type": "special",
                    "description": "Allocate for large-scale regional simulation with 5000 patches",
                },
            },
        ],
        "init": [
            {
                "name": "test_init_with_bounds_typical",
                "inputs": {
                    "waterflux_state": WaterFluxType(qflx_evap_tot_patch=None),
                    "bounds": BoundsType(begp=1, endp=50),
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Initialize complete state with typical 50-patch domain",
                },
            },
        ],
        "update_qflx_evap_tot_patch": [
            {
                "name": "test_update_qflx_typical_evaporation",
                "inputs": {
                    "waterflux": WaterFluxType(
                        qflx_evap_tot_patch=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
                    ),
                    "qflx_evap_tot_patch": jnp.array(
                        [0.0001, 0.0025, 0.005, 0.0015, 0.0008]
                    ),
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Update with typical positive evaporation fluxes during daytime",
                },
            },
            {
                "name": "test_update_qflx_condensation",
                "inputs": {
                    "waterflux": WaterFluxType(
                        qflx_evap_tot_patch=jnp.array([0.0, 0.0, 0.0])
                    ),
                    "qflx_evap_tot_patch": jnp.array([-0.0005, -0.0012, -0.0003]),
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Update with negative fluxes representing condensation/dew formation",
                },
            },
            {
                "name": "test_update_qflx_zero_flux",
                "inputs": {
                    "waterflux": WaterFluxType(
                        qflx_evap_tot_patch=jnp.array([0.0025, 0.003, 0.0015, 0.002])
                    ),
                    "qflx_evap_tot_patch": jnp.array([0.0, 0.0, 0.0, 0.0]),
                },
                "metadata": {
                    "type": "edge",
                    "description": "Update with zero flux (no evaporation or condensation)",
                },
            },
            {
                "name": "test_update_qflx_extreme_conditions",
                "inputs": {
                    "waterflux": WaterFluxType(
                        qflx_evap_tot_patch=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    ),
                    "qflx_evap_tot_patch": jnp.array(
                        [0.0095, -0.0098, 0.0, 0.0001, -0.0001, 0.005]
                    ),
                },
                "metadata": {
                    "type": "edge",
                    "description": "Update with extreme but physically valid fluxes near typical bounds",
                },
            },
            {
                "name": "test_update_qflx_mixed_magnitudes",
                "inputs": {
                    "waterflux": WaterFluxType(
                        qflx_evap_tot_patch=jnp.array(
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        )
                    ),
                    "qflx_evap_tot_patch": jnp.array(
                        [1e-6, 0.001, 0.005, -1e-6, -0.001, 0.0, 0.009, -0.007]
                    ),
                },
                "metadata": {
                    "type": "special",
                    "description": "Update with wide range of magnitudes from micro-scale to near-maximum",
                },
            },
            {
                "name": "test_update_qflx_very_small_fluxes",
                "inputs": {
                    "waterflux": WaterFluxType(
                        qflx_evap_tot_patch=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
                    ),
                    "qflx_evap_tot_patch": jnp.array(
                        [1e-10, -1e-10, 5e-11, -3e-11, 0.0]
                    ),
                },
                "metadata": {
                    "type": "edge",
                    "description": "Update with extremely small fluxes near numerical precision limits",
                },
            },
        ],
    }


# ============================================================================
# Tests for init_waterflux_type
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(tc, id=tc["name"])
        for tc in [
            {
                "name": "test_init_waterflux_type_single_patch",
                "inputs": {"n_patches": 1},
                "metadata": {"type": "nominal"},
            },
            {
                "name": "test_init_waterflux_type_typical_domain",
                "inputs": {"n_patches": 100},
                "metadata": {"type": "nominal"},
            },
            {
                "name": "test_init_waterflux_type_large_domain",
                "inputs": {"n_patches": 10000},
                "metadata": {"type": "special"},
            },
        ]
    ],
)
def test_init_waterflux_type_shapes(test_case):
    """
    Test that init_waterflux_type returns correct array shapes.
    
    Verifies that the initialized WaterFluxType has qflx_evap_tot_patch
    array with shape matching the requested number of patches.
    """
    n_patches = test_case["inputs"]["n_patches"]
    result = init_waterflux_type(n_patches)
    
    assert isinstance(result, WaterFluxType), (
        f"Expected WaterFluxType, got {type(result)}"
    )
    assert result.qflx_evap_tot_patch.shape == (n_patches,), (
        f"Expected shape ({n_patches},), got {result.qflx_evap_tot_patch.shape}"
    )


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(tc, id=tc["name"])
        for tc in [
            {
                "name": "test_init_waterflux_type_single_patch",
                "inputs": {"n_patches": 1},
            },
            {
                "name": "test_init_waterflux_type_typical_domain",
                "inputs": {"n_patches": 100},
            },
        ]
    ],
)
def test_init_waterflux_type_values(test_case):
    """
    Test that init_waterflux_type initializes arrays with NaN values.
    
    Verifies that all elements in the initialized array are NaN,
    indicating uninitialized state ready for allocation.
    """
    n_patches = test_case["inputs"]["n_patches"]
    result = init_waterflux_type(n_patches)
    
    assert jnp.all(jnp.isnan(result.qflx_evap_tot_patch)), (
        "Expected all NaN values in initialized array"
    )


def test_init_waterflux_type_dtypes():
    """
    Test that init_waterflux_type creates arrays with correct data types.
    
    Verifies that the qflx_evap_tot_patch array uses float32 dtype
    for efficient computation in JAX.
    """
    result = init_waterflux_type(10)
    
    assert result.qflx_evap_tot_patch.dtype == jnp.float32, (
        f"Expected float32 dtype, got {result.qflx_evap_tot_patch.dtype}"
    )


def test_init_waterflux_type_minimum_patches():
    """
    Test init_waterflux_type with minimum valid patch count (edge case).
    
    Verifies that the function handles the minimum domain size of 1 patch
    correctly without errors.
    """
    result = init_waterflux_type(1)
    
    assert result.qflx_evap_tot_patch.shape == (1,), (
        "Failed to initialize minimum domain size"
    )
    assert jnp.isnan(result.qflx_evap_tot_patch[0]), (
        "Expected NaN for single patch initialization"
    )


# ============================================================================
# Tests for init_allocate
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(tc, id=tc["name"])
        for tc in [
            {
                "name": "test_init_allocate_small_bounds",
                "inputs": {
                    "waterflux_state": WaterFluxType(qflx_evap_tot_patch=None),
                    "bounds": BoundsType(begp=1, endp=5),
                },
                "expected_size": 5,
            },
            {
                "name": "test_init_allocate_single_patch_bounds",
                "inputs": {
                    "waterflux_state": WaterFluxType(qflx_evap_tot_patch=None),
                    "bounds": BoundsType(begp=1, endp=1),
                },
                "expected_size": 1,
            },
            {
                "name": "test_init_allocate_offset_bounds",
                "inputs": {
                    "waterflux_state": WaterFluxType(qflx_evap_tot_patch=None),
                    "bounds": BoundsType(begp=50, endp=150),
                },
                "expected_size": 101,
            },
            {
                "name": "test_init_allocate_large_bounds",
                "inputs": {
                    "waterflux_state": WaterFluxType(qflx_evap_tot_patch=None),
                    "bounds": BoundsType(begp=1, endp=5000),
                },
                "expected_size": 5000,
            },
        ]
    ],
)
def test_init_allocate_shapes(test_case):
    """
    Test that init_allocate creates arrays with correct shapes.
    
    Verifies that the allocated array size matches the bounds specification,
    accounting for Fortran 1-based indexing (size = endp - begp + 1).
    """
    waterflux_state = test_case["inputs"]["waterflux_state"]
    bounds = test_case["inputs"]["bounds"]
    expected_size = test_case["expected_size"]
    
    result = init_allocate(waterflux_state, bounds)
    
    assert isinstance(result, WaterFluxType), (
        f"Expected WaterFluxType, got {type(result)}"
    )
    assert result.qflx_evap_tot_patch.shape == (expected_size,), (
        f"Expected shape ({expected_size},), got {result.qflx_evap_tot_patch.shape}"
    )


@pytest.mark.parametrize(
    "bounds",
    [
        BoundsType(begp=1, endp=5),
        BoundsType(begp=1, endp=1),
        BoundsType(begp=10, endp=20),
    ],
)
def test_init_allocate_values(bounds):
    """
    Test that init_allocate initializes arrays with NaN values.
    
    Verifies that allocated arrays are filled with NaN to indicate
    uninitialized state before physical values are computed.
    """
    waterflux_state = WaterFluxType(qflx_evap_tot_patch=None)
    result = init_allocate(waterflux_state, bounds)
    
    assert jnp.all(jnp.isnan(result.qflx_evap_tot_patch)), (
        "Expected all NaN values in allocated array"
    )


def test_init_allocate_dtypes():
    """
    Test that init_allocate creates arrays with correct data types.
    
    Verifies float32 dtype for efficient JAX computation.
    """
    waterflux_state = WaterFluxType(qflx_evap_tot_patch=None)
    bounds = BoundsType(begp=1, endp=10)
    result = init_allocate(waterflux_state, bounds)
    
    assert result.qflx_evap_tot_patch.dtype == jnp.float32, (
        f"Expected float32 dtype, got {result.qflx_evap_tot_patch.dtype}"
    )


def test_init_allocate_fortran_indexing():
    """
    Test that init_allocate correctly handles Fortran 1-based indexing.
    
    Verifies that array size calculation accounts for inclusive bounds
    in Fortran convention (size = endp - begp + 1).
    """
    waterflux_state = WaterFluxType(qflx_evap_tot_patch=None)
    bounds = BoundsType(begp=5, endp=14)  # Should create array of size 10
    result = init_allocate(waterflux_state, bounds)
    
    expected_size = bounds.endp - bounds.begp + 1
    assert result.qflx_evap_tot_patch.shape == (expected_size,), (
        f"Fortran indexing error: expected size {expected_size}, "
        f"got {result.qflx_evap_tot_patch.shape[0]}"
    )


# ============================================================================
# Tests for init
# ============================================================================


def test_init_shapes():
    """
    Test that init function returns correct array shapes.
    
    Verifies that the complete initialization process creates arrays
    with shapes matching the bounds specification.
    """
    waterflux_state = WaterFluxType(qflx_evap_tot_patch=None)
    bounds = BoundsType(begp=1, endp=50)
    result = init(waterflux_state, bounds)
    
    expected_size = bounds.endp - bounds.begp + 1
    assert result.qflx_evap_tot_patch.shape == (expected_size,), (
        f"Expected shape ({expected_size},), got {result.qflx_evap_tot_patch.shape}"
    )


def test_init_values():
    """
    Test that init function initializes arrays with NaN values.
    
    Verifies that the complete initialization creates arrays filled
    with NaN values ready for computation.
    """
    waterflux_state = WaterFluxType(qflx_evap_tot_patch=None)
    bounds = BoundsType(begp=1, endp=25)
    result = init(waterflux_state, bounds)
    
    assert jnp.all(jnp.isnan(result.qflx_evap_tot_patch)), (
        "Expected all NaN values after init"
    )


def test_init_dtypes():
    """
    Test that init function creates arrays with correct data types.
    
    Verifies float32 dtype for the complete initialization.
    """
    waterflux_state = WaterFluxType(qflx_evap_tot_patch=None)
    bounds = BoundsType(begp=1, endp=30)
    result = init(waterflux_state, bounds)
    
    assert result.qflx_evap_tot_patch.dtype == jnp.float32, (
        f"Expected float32 dtype, got {result.qflx_evap_tot_patch.dtype}"
    )


# ============================================================================
# Tests for update_qflx_evap_tot_patch
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(tc, id=tc["name"])
        for tc in [
            {
                "name": "test_update_qflx_typical_evaporation",
                "inputs": {
                    "waterflux": WaterFluxType(
                        qflx_evap_tot_patch=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
                    ),
                    "qflx_evap_tot_patch": jnp.array(
                        [0.0001, 0.0025, 0.005, 0.0015, 0.0008]
                    ),
                },
            },
            {
                "name": "test_update_qflx_condensation",
                "inputs": {
                    "waterflux": WaterFluxType(
                        qflx_evap_tot_patch=jnp.array([0.0, 0.0, 0.0])
                    ),
                    "qflx_evap_tot_patch": jnp.array([-0.0005, -0.0012, -0.0003]),
                },
            },
            {
                "name": "test_update_qflx_zero_flux",
                "inputs": {
                    "waterflux": WaterFluxType(
                        qflx_evap_tot_patch=jnp.array([0.0025, 0.003, 0.0015, 0.002])
                    ),
                    "qflx_evap_tot_patch": jnp.array([0.0, 0.0, 0.0, 0.0]),
                },
            },
        ]
    ],
)
def test_update_qflx_evap_tot_patch_shapes(test_case):
    """
    Test that update_qflx_evap_tot_patch preserves array shapes.
    
    Verifies that the updated state maintains the same array dimensions
    as the input state.
    """
    waterflux = test_case["inputs"]["waterflux"]
    qflx_evap_tot_patch = test_case["inputs"]["qflx_evap_tot_patch"]
    
    result = update_qflx_evap_tot_patch(waterflux, qflx_evap_tot_patch)
    
    assert result.qflx_evap_tot_patch.shape == qflx_evap_tot_patch.shape, (
        f"Shape mismatch: expected {qflx_evap_tot_patch.shape}, "
        f"got {result.qflx_evap_tot_patch.shape}"
    )


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(tc, id=tc["name"])
        for tc in [
            {
                "name": "test_update_qflx_typical_evaporation",
                "inputs": {
                    "waterflux": WaterFluxType(
                        qflx_evap_tot_patch=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
                    ),
                    "qflx_evap_tot_patch": jnp.array(
                        [0.0001, 0.0025, 0.005, 0.0015, 0.0008]
                    ),
                },
            },
            {
                "name": "test_update_qflx_condensation",
                "inputs": {
                    "waterflux": WaterFluxType(
                        qflx_evap_tot_patch=jnp.array([0.0, 0.0, 0.0])
                    ),
                    "qflx_evap_tot_patch": jnp.array([-0.0005, -0.0012, -0.0003]),
                },
            },
            {
                "name": "test_update_qflx_zero_flux",
                "inputs": {
                    "waterflux": WaterFluxType(
                        qflx_evap_tot_patch=jnp.array([0.0025, 0.003, 0.0015, 0.002])
                    ),
                    "qflx_evap_tot_patch": jnp.array([0.0, 0.0, 0.0, 0.0]),
                },
            },
            {
                "name": "test_update_qflx_extreme_conditions",
                "inputs": {
                    "waterflux": WaterFluxType(
                        qflx_evap_tot_patch=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    ),
                    "qflx_evap_tot_patch": jnp.array(
                        [0.0095, -0.0098, 0.0, 0.0001, -0.0001, 0.005]
                    ),
                },
            },
            {
                "name": "test_update_qflx_mixed_magnitudes",
                "inputs": {
                    "waterflux": WaterFluxType(
                        qflx_evap_tot_patch=jnp.array(
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        )
                    ),
                    "qflx_evap_tot_patch": jnp.array(
                        [1e-6, 0.001, 0.005, -1e-6, -0.001, 0.0, 0.009, -0.007]
                    ),
                },
            },
        ]
    ],
)
def test_update_qflx_evap_tot_patch_values(test_case):
    """
    Test that update_qflx_evap_tot_patch correctly updates flux values.
    
    Verifies that the new flux values are properly stored in the updated
    state, matching the input values within numerical tolerance.
    """
    waterflux = test_case["inputs"]["waterflux"]
    qflx_evap_tot_patch = test_case["inputs"]["qflx_evap_tot_patch"]
    
    result = update_qflx_evap_tot_patch(waterflux, qflx_evap_tot_patch)
    
    assert jnp.allclose(
        result.qflx_evap_tot_patch, qflx_evap_tot_patch, atol=1e-10, rtol=1e-10
    ), (
        f"Value mismatch: expected {qflx_evap_tot_patch}, "
        f"got {result.qflx_evap_tot_patch}"
    )


def test_update_qflx_evap_tot_patch_dtypes():
    """
    Test that update_qflx_evap_tot_patch preserves data types.
    
    Verifies that the updated array maintains float32 dtype.
    """
    waterflux = WaterFluxType(qflx_evap_tot_patch=jnp.array([0.0, 0.0, 0.0]))
    qflx_evap_tot_patch = jnp.array([0.001, 0.002, 0.003])
    
    result = update_qflx_evap_tot_patch(waterflux, qflx_evap_tot_patch)
    
    assert result.qflx_evap_tot_patch.dtype == jnp.float32, (
        f"Expected float32 dtype, got {result.qflx_evap_tot_patch.dtype}"
    )


def test_update_qflx_evap_tot_patch_positive_evaporation():
    """
    Test update with positive evaporation fluxes (edge case).
    
    Verifies that positive flux values (water leaving surface) are
    correctly handled and stored.
    """
    waterflux = WaterFluxType(qflx_evap_tot_patch=jnp.zeros(3))
    qflx_evap_tot_patch = jnp.array([0.005, 0.008, 0.009])
    
    result = update_qflx_evap_tot_patch(waterflux, qflx_evap_tot_patch)
    
    assert jnp.all(result.qflx_evap_tot_patch > 0), (
        "Expected all positive values for evaporation"
    )
    assert jnp.allclose(
        result.qflx_evap_tot_patch, qflx_evap_tot_patch, atol=1e-10
    ), "Positive flux values not correctly stored"


def test_update_qflx_evap_tot_patch_negative_condensation():
    """
    Test update with negative condensation fluxes (edge case).
    
    Verifies that negative flux values (water arriving at surface via
    condensation/dew) are correctly handled.
    """
    waterflux = WaterFluxType(qflx_evap_tot_patch=jnp.zeros(4))
    qflx_evap_tot_patch = jnp.array([-0.001, -0.005, -0.008, -0.002])
    
    result = update_qflx_evap_tot_patch(waterflux, qflx_evap_tot_patch)
    
    assert jnp.all(result.qflx_evap_tot_patch < 0), (
        "Expected all negative values for condensation"
    )
    assert jnp.allclose(
        result.qflx_evap_tot_patch, qflx_evap_tot_patch, atol=1e-10
    ), "Negative flux values not correctly stored"


def test_update_qflx_evap_tot_patch_zero_flux():
    """
    Test update with zero flux values (edge case).
    
    Verifies that zero flux (equilibrium conditions with no net water
    transfer) is correctly handled.
    """
    waterflux = WaterFluxType(qflx_evap_tot_patch=jnp.array([0.001, 0.002, 0.003]))
    qflx_evap_tot_patch = jnp.zeros(3)
    
    result = update_qflx_evap_tot_patch(waterflux, qflx_evap_tot_patch)
    
    assert jnp.allclose(result.qflx_evap_tot_patch, 0.0, atol=1e-10), (
        "Expected zero flux values"
    )


def test_update_qflx_evap_tot_patch_mixed_signs():
    """
    Test update with mixed positive and negative fluxes (edge case).
    
    Verifies that spatially heterogeneous conditions with both evaporation
    and condensation are correctly handled.
    """
    waterflux = WaterFluxType(qflx_evap_tot_patch=jnp.zeros(6))
    qflx_evap_tot_patch = jnp.array([0.005, -0.003, 0.0, 0.002, -0.001, 0.008])
    
    result = update_qflx_evap_tot_patch(waterflux, qflx_evap_tot_patch)
    
    # Check that signs are preserved
    positive_mask = qflx_evap_tot_patch > 0
    negative_mask = qflx_evap_tot_patch < 0
    zero_mask = qflx_evap_tot_patch == 0
    
    assert jnp.all(result.qflx_evap_tot_patch[positive_mask] > 0), (
        "Positive values not preserved"
    )
    assert jnp.all(result.qflx_evap_tot_patch[negative_mask] < 0), (
        "Negative values not preserved"
    )
    assert jnp.allclose(result.qflx_evap_tot_patch[zero_mask], 0.0, atol=1e-10), (
        "Zero values not preserved"
    )


def test_update_qflx_evap_tot_patch_very_small_values():
    """
    Test update with extremely small flux values near numerical precision.
    
    Verifies that the function handles values near machine epsilon without
    numerical issues or loss of precision.
    """
    waterflux = WaterFluxType(qflx_evap_tot_patch=jnp.zeros(5))
    qflx_evap_tot_patch = jnp.array([1e-10, -1e-10, 5e-11, -3e-11, 0.0])
    
    result = update_qflx_evap_tot_patch(waterflux, qflx_evap_tot_patch)
    
    assert jnp.allclose(
        result.qflx_evap_tot_patch, qflx_evap_tot_patch, atol=1e-12, rtol=1e-12
    ), "Small values not preserved with sufficient precision"


def test_update_qflx_evap_tot_patch_physical_bounds():
    """
    Test update with values at physical bounds (edge case).
    
    Verifies that extreme but physically realistic flux values near the
    typical range boundaries [-0.01, 0.01] kg/m²/s are handled correctly.
    """
    waterflux = WaterFluxType(qflx_evap_tot_patch=jnp.zeros(4))
    qflx_evap_tot_patch = jnp.array([0.01, -0.01, 0.0099, -0.0099])
    
    result = update_qflx_evap_tot_patch(waterflux, qflx_evap_tot_patch)
    
    assert jnp.allclose(
        result.qflx_evap_tot_patch, qflx_evap_tot_patch, atol=1e-10
    ), "Physical boundary values not correctly stored"
    
    # Verify values are within expected physical range
    assert jnp.all(jnp.abs(result.qflx_evap_tot_patch) <= 0.01), (
        "Values exceed typical physical bounds"
    )


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_initialization_workflow():
    """
    Integration test for complete initialization workflow.
    
    Tests the typical sequence: init_waterflux_type -> init_allocate -> update.
    Verifies that the complete workflow produces valid results.
    """
    # Step 1: Initialize with NaN
    n_patches = 10
    waterflux = init_waterflux_type(n_patches)
    assert jnp.all(jnp.isnan(waterflux.qflx_evap_tot_patch))
    
    # Step 2: Allocate with bounds
    bounds = BoundsType(begp=1, endp=n_patches)
    waterflux = init_allocate(waterflux, bounds)
    assert jnp.all(jnp.isnan(waterflux.qflx_evap_tot_patch))
    assert waterflux.qflx_evap_tot_patch.shape == (n_patches,)
    
    # Step 3: Update with actual values
    new_fluxes = jnp.array([0.001, 0.002, -0.001, 0.0, 0.003, 
                            -0.002, 0.004, 0.001, -0.001, 0.002])
    waterflux = update_qflx_evap_tot_patch(waterflux, new_fluxes)
    
    assert jnp.allclose(waterflux.qflx_evap_tot_patch, new_fluxes, atol=1e-10)
    assert not jnp.any(jnp.isnan(waterflux.qflx_evap_tot_patch))


def test_init_function_workflow():
    """
    Integration test for init function workflow.
    
    Tests the init function which combines initialization and allocation.
    """
    waterflux = WaterFluxType(qflx_evap_tot_patch=None)
    bounds = BoundsType(begp=1, endp=20)
    
    # Initialize using init function
    waterflux = init(waterflux, bounds)
    
    expected_size = bounds.endp - bounds.begp + 1
    assert waterflux.qflx_evap_tot_patch.shape == (expected_size,)
    assert jnp.all(jnp.isnan(waterflux.qflx_evap_tot_patch))
    
    # Update with values
    new_fluxes = jnp.linspace(-0.005, 0.005, expected_size)
    waterflux = update_qflx_evap_tot_patch(waterflux, new_fluxes)
    
    assert jnp.allclose(waterflux.qflx_evap_tot_patch, new_fluxes, atol=1e-10)


def test_multiple_updates():
    """
    Integration test for multiple sequential updates.
    
    Verifies that the state can be updated multiple times with different
    values, simulating a time-stepping simulation.
    """
    waterflux = WaterFluxType(qflx_evap_tot_patch=jnp.zeros(5))
    
    # First update
    fluxes_1 = jnp.array([0.001, 0.002, 0.003, 0.004, 0.005])
    waterflux = update_qflx_evap_tot_patch(waterflux, fluxes_1)
    assert jnp.allclose(waterflux.qflx_evap_tot_patch, fluxes_1, atol=1e-10)
    
    # Second update
    fluxes_2 = jnp.array([0.002, 0.003, 0.001, 0.005, 0.004])
    waterflux = update_qflx_evap_tot_patch(waterflux, fluxes_2)
    assert jnp.allclose(waterflux.qflx_evap_tot_patch, fluxes_2, atol=1e-10)
    
    # Third update with different signs
    fluxes_3 = jnp.array([-0.001, 0.002, -0.003, 0.0, 0.001])
    waterflux = update_qflx_evap_tot_patch(waterflux, fluxes_3)
    assert jnp.allclose(waterflux.qflx_evap_tot_patch, fluxes_3, atol=1e-10)


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_bounds_validation():
    """
    Test that bounds are correctly validated (edge case).
    
    Verifies that the Fortran 1-based indexing convention is properly
    handled with various bound configurations.
    """
    waterflux = WaterFluxType(qflx_evap_tot_patch=None)
    
    # Test various valid bounds
    test_bounds = [
        BoundsType(begp=1, endp=1),      # Single patch
        BoundsType(begp=1, endp=100),    # Standard range
        BoundsType(begp=50, endp=150),   # Offset range
        BoundsType(begp=1000, endp=2000), # Large offset
    ]
    
    for bounds in test_bounds:
        result = init_allocate(waterflux, bounds)
        expected_size = bounds.endp - bounds.begp + 1
        assert result.qflx_evap_tot_patch.shape == (expected_size,), (
            f"Failed for bounds {bounds}: expected size {expected_size}, "
            f"got {result.qflx_evap_tot_patch.shape[0]}"
        )


def test_array_immutability():
    """
    Test that updates create new states without modifying originals.
    
    Verifies JAX's functional programming paradigm where updates return
    new states rather than modifying existing ones.
    """
    original_fluxes = jnp.array([0.001, 0.002, 0.003])
    waterflux = WaterFluxType(qflx_evap_tot_patch=original_fluxes)
    
    new_fluxes = jnp.array([0.004, 0.005, 0.006])
    updated_waterflux = update_qflx_evap_tot_patch(waterflux, new_fluxes)
    
    # Original should be unchanged
    assert jnp.allclose(waterflux.qflx_evap_tot_patch, original_fluxes, atol=1e-10), (
        "Original state was modified"
    )
    
    # Updated should have new values
    assert jnp.allclose(updated_waterflux.qflx_evap_tot_patch, new_fluxes, atol=1e-10), (
        "Updated state does not have new values"
    )