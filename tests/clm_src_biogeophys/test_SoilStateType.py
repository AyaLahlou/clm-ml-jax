"""
Comprehensive pytest suite for SoilStateType module.

This module tests the initialization and allocation functions for soil state
variables in the CLM biogeophysics component. Tests cover:
- Bounds creation and validation
- Soil state initialization with various domain sizes
- Array shape and dimension verification
- NaN initialization verification
- Edge cases (minimum layers, large domains, non-contiguous bounds)
- Physical constraint validation
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_biogeophys.SoilStateType import (
    BoundsType,
    SoilStateType,
    create_simple_bounds,
    get_soil_layer_indices,
    init_allocate,
    init_soil_state,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_data():
    """
    Load test data for SoilStateType functions.
    
    Returns:
        dict: Test cases with inputs and metadata for comprehensive testing.
    """
    return {
        "test_cases": [
            {
                "name": "test_nominal_single_patch_column",
                "inputs": {
                    "bounds": {
                        "begp": 1,
                        "endp": 1,
                        "begc": 1,
                        "endc": 1,
                        "begg": 1,
                        "endg": 1,
                    },
                    "nlevsoi": 10,
                    "nlevgrnd": 15,
                    "nlevsno": 5,
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Minimal valid configuration with single patch, column, and gridcell",
                },
            },
            {
                "name": "test_nominal_small_domain",
                "inputs": {
                    "bounds": {
                        "begp": 1,
                        "endp": 5,
                        "begc": 1,
                        "endc": 3,
                        "begg": 1,
                        "endg": 2,
                    },
                    "nlevsoi": 10,
                    "nlevgrnd": 15,
                    "nlevsno": 5,
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Small domain with multiple patches, columns, and gridcells",
                },
            },
            {
                "name": "test_nominal_large_domain",
                "inputs": {
                    "bounds": {
                        "begp": 1,
                        "endp": 100,
                        "begc": 1,
                        "endc": 50,
                        "begg": 1,
                        "endg": 25,
                    },
                    "nlevsoi": 10,
                    "nlevgrnd": 15,
                    "nlevsno": 5,
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Large domain representing regional-scale simulation",
                },
            },
            {
                "name": "test_nominal_custom_layers",
                "inputs": {
                    "bounds": {
                        "begp": 1,
                        "endp": 10,
                        "begc": 1,
                        "endc": 5,
                        "begg": 1,
                        "endg": 3,
                    },
                    "nlevsoi": 8,
                    "nlevgrnd": 12,
                    "nlevsno": 3,
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Custom layer configuration with fewer layers than default",
                },
            },
            {
                "name": "test_nominal_deep_soil_profile",
                "inputs": {
                    "bounds": {
                        "begp": 1,
                        "endp": 20,
                        "begc": 1,
                        "endc": 10,
                        "begg": 1,
                        "endg": 5,
                    },
                    "nlevsoi": 15,
                    "nlevgrnd": 25,
                    "nlevsno": 7,
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Deep soil profile for permafrost studies",
                },
            },
            {
                "name": "test_edge_minimum_layers",
                "inputs": {
                    "bounds": {
                        "begp": 1,
                        "endp": 3,
                        "begc": 1,
                        "endc": 2,
                        "begg": 1,
                        "endg": 1,
                    },
                    "nlevsoi": 1,
                    "nlevgrnd": 1,
                    "nlevsno": 1,
                },
                "metadata": {
                    "type": "edge",
                    "description": "Minimum valid layer configuration",
                    "edge_cases": ["minimum_layers", "boundary"],
                },
            },
            {
                "name": "test_edge_single_layer_soil_multi_ground",
                "inputs": {
                    "bounds": {
                        "begp": 1,
                        "endp": 5,
                        "begc": 1,
                        "endc": 3,
                        "begg": 1,
                        "endg": 2,
                    },
                    "nlevsoi": 1,
                    "nlevgrnd": 20,
                    "nlevsno": 5,
                },
                "metadata": {
                    "type": "edge",
                    "description": "Minimal soil layers with extensive ground layers",
                    "edge_cases": ["minimum_soil_layers", "deep_bedrock"],
                },
            },
            {
                "name": "test_edge_large_snow_layers",
                "inputs": {
                    "bounds": {
                        "begp": 1,
                        "endp": 8,
                        "begc": 1,
                        "endc": 4,
                        "begg": 1,
                        "endg": 2,
                    },
                    "nlevsoi": 10,
                    "nlevgrnd": 15,
                    "nlevsno": 15,
                },
                "metadata": {
                    "type": "edge",
                    "description": "Large number of snow layers for deep snowpack",
                    "edge_cases": ["deep_snowpack", "large_snow_layers"],
                },
            },
            {
                "name": "test_special_non_contiguous_bounds",
                "inputs": {
                    "bounds": {
                        "begp": 10,
                        "endp": 25,
                        "begc": 5,
                        "endc": 15,
                        "begg": 3,
                        "endg": 8,
                    },
                    "nlevsoi": 10,
                    "nlevgrnd": 15,
                    "nlevsno": 5,
                },
                "metadata": {
                    "type": "special",
                    "description": "Non-zero-based bounds for 1-based indexing test",
                    "edge_cases": ["non_zero_start", "offset_indices"],
                },
            },
            {
                "name": "test_special_asymmetric_domain",
                "inputs": {
                    "bounds": {
                        "begp": 1,
                        "endp": 200,
                        "begc": 1,
                        "endc": 20,
                        "begg": 1,
                        "endg": 5,
                    },
                    "nlevsoi": 12,
                    "nlevgrnd": 18,
                    "nlevsno": 6,
                },
                "metadata": {
                    "type": "special",
                    "description": "Highly asymmetric domain with many patches per column",
                    "edge_cases": ["high_patch_density", "asymmetric_domain"],
                },
            },
        ]
    }


@pytest.fixture
def simple_bounds():
    """Create a simple BoundsType for basic testing."""
    return BoundsType(begp=1, endp=5, begc=1, endc=3, begg=1, endg=2)


@pytest.fixture
def default_layers():
    """Default layer configuration for testing."""
    return {"nlevsoi": 10, "nlevgrnd": 15, "nlevsno": 5}


# ============================================================================
# Helper Functions
# ============================================================================


def calculate_dimensions(bounds: BoundsType) -> dict:
    """
    Calculate domain dimensions from bounds.
    
    Args:
        bounds: BoundsType with domain indices
        
    Returns:
        dict: Dictionary with n_patches, n_columns, n_gridcells
    """
    return {
        "n_patches": bounds.endp - bounds.begp + 1,
        "n_columns": bounds.endc - bounds.begc + 1,
        "n_gridcells": bounds.endg - bounds.begg + 1,
    }


def verify_nan_initialization(array: jnp.ndarray, name: str) -> None:
    """
    Verify that an array is initialized with NaN values.
    
    Args:
        array: Array to check
        name: Name of the array for error messages
    """
    assert jnp.all(jnp.isnan(array)), f"{name} should be initialized with NaN values"


# ============================================================================
# Tests for create_simple_bounds
# ============================================================================


@pytest.mark.parametrize(
    "n_patches,n_columns,n_gridcells",
    [
        (1, 1, 1),  # Minimal
        (5, 3, 2),  # Small domain
        (100, 50, 25),  # Large domain
        (10, 10, 10),  # Equal dimensions
        (200, 20, 5),  # Asymmetric
    ],
)
def test_create_simple_bounds_shapes(n_patches, n_columns, n_gridcells):
    """
    Test create_simple_bounds returns correct BoundsType with proper indices.
    
    Verifies:
    - Returns BoundsType instance
    - 1-based indexing (begp=1, begc=1, begg=1)
    - Correct end indices based on counts
    - beg <= end for all dimensions
    """
    bounds = create_simple_bounds(n_patches, n_columns, n_gridcells)
    
    # Verify type
    assert isinstance(bounds, BoundsType), "Should return BoundsType instance"
    
    # Verify 1-based indexing
    assert bounds.begp == 1, "Patch indices should start at 1"
    assert bounds.begc == 1, "Column indices should start at 1"
    assert bounds.begg == 1, "Gridcell indices should start at 1"
    
    # Verify end indices
    assert bounds.endp == n_patches, f"endp should be {n_patches}"
    assert bounds.endc == n_columns, f"endc should be {n_columns}"
    assert bounds.endg == n_gridcells, f"endg should be {n_gridcells}"
    
    # Verify beg <= end
    assert bounds.begp <= bounds.endp, "begp should be <= endp"
    assert bounds.begc <= bounds.endc, "begc should be <= endc"
    assert bounds.begg <= bounds.endg, "begg should be <= endg"


def test_create_simple_bounds_minimum():
    """Test create_simple_bounds with minimum valid inputs (all 1s)."""
    bounds = create_simple_bounds(1, 1, 1)
    
    assert bounds.begp == 1 and bounds.endp == 1
    assert bounds.begc == 1 and bounds.endc == 1
    assert bounds.begg == 1 and bounds.endg == 1


def test_create_simple_bounds_large_domain():
    """Test create_simple_bounds with large domain sizes."""
    bounds = create_simple_bounds(1000, 500, 100)
    
    assert bounds.endp == 1000
    assert bounds.endc == 500
    assert bounds.endg == 100


# ============================================================================
# Tests for get_soil_layer_indices
# ============================================================================


@pytest.mark.parametrize(
    "nlevsno,nlevgrnd,expected",
    [
        (5, 15, (-4, 1, 15)),  # Default configuration
        (1, 1, (0, 1, 1)),  # Minimum layers
        (10, 20, (-9, 1, 20)),  # Large configuration
        (3, 10, (-2, 1, 10)),  # Custom configuration
        (7, 25, (-6, 1, 25)),  # Deep profile
    ],
)
def test_get_soil_layer_indices_values(nlevsno, nlevgrnd, expected):
    """
    Test get_soil_layer_indices returns correct index tuple.
    
    Verifies:
    - Returns tuple of 3 integers
    - snow_start_idx = -(nlevsno - 1)
    - ground_start_idx = 1 (1-based)
    - ground_end_idx = nlevgrnd
    """
    result = get_soil_layer_indices(nlevsno, nlevgrnd)
    
    # Verify type and length
    assert isinstance(result, tuple), "Should return tuple"
    assert len(result) == 3, "Should return 3 values"
    
    # Verify values
    snow_start, ground_start, ground_end = result
    expected_snow, expected_ground_start, expected_ground_end = expected
    
    assert snow_start == expected_snow, f"Snow start index should be {expected_snow}"
    assert ground_start == expected_ground_start, f"Ground start should be {expected_ground_start}"
    assert ground_end == expected_ground_end, f"Ground end should be {expected_ground_end}"


def test_get_soil_layer_indices_default():
    """Test get_soil_layer_indices with default parameters."""
    result = get_soil_layer_indices()
    
    assert result == (-4, 1, 15), "Default should return (-4, 1, 15)"


def test_get_soil_layer_indices_consistency():
    """Test that indices maintain consistent relationships."""
    nlevsno, nlevgrnd = 5, 15
    snow_start, ground_start, ground_end = get_soil_layer_indices(nlevsno, nlevgrnd)
    
    # Snow layers span from snow_start to 0
    assert snow_start < 0, "Snow start should be negative"
    assert abs(snow_start) == nlevsno - 1, "Snow layers should span correctly"
    
    # Ground layers span from 1 to nlevgrnd
    assert ground_start == 1, "Ground should start at 1"
    assert ground_end == nlevgrnd, "Ground should end at nlevgrnd"


# ============================================================================
# Tests for init_soil_state
# ============================================================================


def test_init_soil_state_returns_correct_type(simple_bounds, default_layers):
    """Test that init_soil_state returns SoilStateType instance."""
    result = init_soil_state(simple_bounds, **default_layers)
    
    assert isinstance(result, SoilStateType), "Should return SoilStateType instance"


@pytest.mark.parametrize("test_case_idx", range(10))
def test_init_soil_state_shapes(test_data, test_case_idx):
    """
    Test init_soil_state creates arrays with correct shapes.
    
    Verifies all array dimensions match expected sizes based on:
    - n_columns = endc - begc + 1
    - n_patches = endp - begp + 1
    - Layer counts (nlevsoi, nlevgrnd, nlevsno)
    """
    test_case = test_data["test_cases"][test_case_idx]
    inputs = test_case["inputs"]
    
    # Create bounds
    bounds_dict = inputs["bounds"]
    bounds = BoundsType(**bounds_dict)
    
    # Get layer parameters
    nlevsoi = inputs["nlevsoi"]
    nlevgrnd = inputs["nlevgrnd"]
    nlevsno = inputs["nlevsno"]
    
    # Calculate expected dimensions
    dims = calculate_dimensions(bounds)
    n_columns = dims["n_columns"]
    n_patches = dims["n_patches"]
    
    # Initialize soil state
    soil_state = init_soil_state(bounds, nlevsoi, nlevgrnd, nlevsno)
    
    # Verify column arrays with nlevsoi
    assert soil_state.cellorg_col.shape == (n_columns, nlevsoi), \
        f"cellorg_col shape mismatch for {test_case['name']}"
    assert soil_state.cellsand_col.shape == (n_columns, nlevsoi), \
        f"cellsand_col shape mismatch for {test_case['name']}"
    assert soil_state.cellclay_col.shape == (n_columns, nlevsoi), \
        f"cellclay_col shape mismatch for {test_case['name']}"
    
    # Verify column arrays with nlevgrnd
    assert soil_state.hksat_col.shape == (n_columns, nlevgrnd), \
        f"hksat_col shape mismatch for {test_case['name']}"
    assert soil_state.hk_l_col.shape == (n_columns, nlevgrnd), \
        f"hk_l_col shape mismatch for {test_case['name']}"
    assert soil_state.smp_l_col.shape == (n_columns, nlevgrnd), \
        f"smp_l_col shape mismatch for {test_case['name']}"
    assert soil_state.bsw_col.shape == (n_columns, nlevgrnd), \
        f"bsw_col shape mismatch for {test_case['name']}"
    assert soil_state.watsat_col.shape == (n_columns, nlevgrnd), \
        f"watsat_col shape mismatch for {test_case['name']}"
    assert soil_state.sucsat_col.shape == (n_columns, nlevgrnd), \
        f"sucsat_col shape mismatch for {test_case['name']}"
    assert soil_state.tkmg_col.shape == (n_columns, nlevgrnd), \
        f"tkmg_col shape mismatch for {test_case['name']}"
    assert soil_state.tkdry_col.shape == (n_columns, nlevgrnd), \
        f"tkdry_col shape mismatch for {test_case['name']}"
    assert soil_state.csol_col.shape == (n_columns, nlevgrnd), \
        f"csol_col shape mismatch for {test_case['name']}"
    
    # Verify scalar column arrays
    assert soil_state.dsl_col.shape == (n_columns,), \
        f"dsl_col shape mismatch for {test_case['name']}"
    assert soil_state.soilresis_col.shape == (n_columns,), \
        f"soilresis_col shape mismatch for {test_case['name']}"
    
    # Verify thk_col with combined layers
    assert soil_state.thk_col.shape == (n_columns, nlevgrnd + nlevsno), \
        f"thk_col shape mismatch for {test_case['name']}"
    
    # Verify patch arrays
    assert soil_state.rootfr_patch.shape == (n_patches, nlevgrnd), \
        f"rootfr_patch shape mismatch for {test_case['name']}"


@pytest.mark.parametrize("test_case_idx", [0, 1, 5, 9])
def test_init_soil_state_nan_initialization(test_data, test_case_idx):
    """
    Test that init_soil_state initializes all arrays with NaN values.
    
    Verifies that all fields in SoilStateType are filled with NaN,
    indicating uninitialized state ready for data population.
    """
    test_case = test_data["test_cases"][test_case_idx]
    inputs = test_case["inputs"]
    
    bounds = BoundsType(**inputs["bounds"])
    soil_state = init_soil_state(
        bounds, inputs["nlevsoi"], inputs["nlevgrnd"], inputs["nlevsno"]
    )
    
    # Verify NaN initialization for all fields
    verify_nan_initialization(soil_state.cellorg_col, "cellorg_col")
    verify_nan_initialization(soil_state.cellsand_col, "cellsand_col")
    verify_nan_initialization(soil_state.cellclay_col, "cellclay_col")
    verify_nan_initialization(soil_state.hksat_col, "hksat_col")
    verify_nan_initialization(soil_state.hk_l_col, "hk_l_col")
    verify_nan_initialization(soil_state.smp_l_col, "smp_l_col")
    verify_nan_initialization(soil_state.bsw_col, "bsw_col")
    verify_nan_initialization(soil_state.watsat_col, "watsat_col")
    verify_nan_initialization(soil_state.sucsat_col, "sucsat_col")
    verify_nan_initialization(soil_state.dsl_col, "dsl_col")
    verify_nan_initialization(soil_state.soilresis_col, "soilresis_col")
    verify_nan_initialization(soil_state.thk_col, "thk_col")
    verify_nan_initialization(soil_state.tkmg_col, "tkmg_col")
    verify_nan_initialization(soil_state.tkdry_col, "tkdry_col")
    verify_nan_initialization(soil_state.csol_col, "csol_col")
    verify_nan_initialization(soil_state.rootfr_patch, "rootfr_patch")


def test_init_soil_state_default_parameters():
    """Test init_soil_state with default layer parameters."""
    bounds = BoundsType(begp=1, endp=3, begc=1, endc=2, begg=1, endg=1)
    
    # Call with defaults
    soil_state = init_soil_state(bounds)
    
    # Verify shapes with default values (10, 15, 5)
    assert soil_state.cellorg_col.shape == (2, 10)
    assert soil_state.hksat_col.shape == (2, 15)
    assert soil_state.thk_col.shape == (2, 20)  # 15 + 5
    assert soil_state.rootfr_patch.shape == (3, 15)


def test_init_soil_state_dtypes():
    """Test that init_soil_state creates arrays with correct dtypes."""
    bounds = BoundsType(begp=1, endp=2, begc=1, endc=1, begg=1, endg=1)
    soil_state = init_soil_state(bounds, 5, 10, 3)
    
    # All arrays should be float (JAX default float32 or float64)
    assert jnp.issubdtype(soil_state.cellorg_col.dtype, jnp.floating)
    assert jnp.issubdtype(soil_state.hksat_col.dtype, jnp.floating)
    assert jnp.issubdtype(soil_state.rootfr_patch.dtype, jnp.floating)


# ============================================================================
# Tests for init_allocate
# ============================================================================


def test_init_allocate_returns_correct_type(simple_bounds, default_layers):
    """Test that init_allocate returns SoilStateType instance."""
    result = init_allocate(simple_bounds, **default_layers)
    
    assert isinstance(result, SoilStateType), "Should return SoilStateType instance"


@pytest.mark.parametrize("test_case_idx", range(10))
def test_init_allocate_shapes(test_data, test_case_idx):
    """
    Test init_allocate creates arrays with correct shapes.
    
    Should produce identical results to init_soil_state.
    """
    test_case = test_data["test_cases"][test_case_idx]
    inputs = test_case["inputs"]
    
    bounds = BoundsType(**inputs["bounds"])
    nlevsoi = inputs["nlevsoi"]
    nlevgrnd = inputs["nlevgrnd"]
    nlevsno = inputs["nlevsno"]
    
    dims = calculate_dimensions(bounds)
    n_columns = dims["n_columns"]
    n_patches = dims["n_patches"]
    
    soil_state = init_allocate(bounds, nlevsoi, nlevgrnd, nlevsno)
    
    # Verify all shapes (same as init_soil_state)
    assert soil_state.cellorg_col.shape == (n_columns, nlevsoi)
    assert soil_state.cellsand_col.shape == (n_columns, nlevsoi)
    assert soil_state.cellclay_col.shape == (n_columns, nlevsoi)
    assert soil_state.hksat_col.shape == (n_columns, nlevgrnd)
    assert soil_state.hk_l_col.shape == (n_columns, nlevgrnd)
    assert soil_state.smp_l_col.shape == (n_columns, nlevgrnd)
    assert soil_state.bsw_col.shape == (n_columns, nlevgrnd)
    assert soil_state.watsat_col.shape == (n_columns, nlevgrnd)
    assert soil_state.sucsat_col.shape == (n_columns, nlevgrnd)
    assert soil_state.dsl_col.shape == (n_columns,)
    assert soil_state.soilresis_col.shape == (n_columns,)
    assert soil_state.thk_col.shape == (n_columns, nlevgrnd + nlevsno)
    assert soil_state.tkmg_col.shape == (n_columns, nlevgrnd)
    assert soil_state.tkdry_col.shape == (n_columns, nlevgrnd)
    assert soil_state.csol_col.shape == (n_columns, nlevgrnd)
    assert soil_state.rootfr_patch.shape == (n_patches, nlevgrnd)


def test_init_allocate_equivalence_to_init_soil_state():
    """
    Test that init_allocate produces identical results to init_soil_state.
    
    Both functions should initialize the same structure with NaN values.
    """
    bounds = BoundsType(begp=1, endp=10, begc=1, endc=5, begg=1, endg=3)
    nlevsoi, nlevgrnd, nlevsno = 8, 12, 4
    
    state1 = init_soil_state(bounds, nlevsoi, nlevgrnd, nlevsno)
    state2 = init_allocate(bounds, nlevsoi, nlevgrnd, nlevsno)
    
    # Compare shapes
    assert state1.cellorg_col.shape == state2.cellorg_col.shape
    assert state1.hksat_col.shape == state2.hksat_col.shape
    assert state1.thk_col.shape == state2.thk_col.shape
    assert state1.rootfr_patch.shape == state2.rootfr_patch.shape
    
    # Both should be NaN-initialized
    assert jnp.all(jnp.isnan(state1.cellorg_col))
    assert jnp.all(jnp.isnan(state2.cellorg_col))


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_edge_case_minimum_domain():
    """Test with absolute minimum domain size (1x1x1)."""
    bounds = create_simple_bounds(1, 1, 1)
    soil_state = init_soil_state(bounds, 1, 1, 1)
    
    # Verify minimal shapes
    assert soil_state.cellorg_col.shape == (1, 1)
    assert soil_state.hksat_col.shape == (1, 1)
    assert soil_state.thk_col.shape == (1, 2)  # 1 + 1
    assert soil_state.rootfr_patch.shape == (1, 1)
    assert soil_state.dsl_col.shape == (1,)


def test_edge_case_large_domain():
    """Test with large domain to verify scalability."""
    bounds = create_simple_bounds(500, 250, 100)
    soil_state = init_soil_state(bounds, 10, 15, 5)
    
    # Verify large shapes
    assert soil_state.cellorg_col.shape == (250, 10)
    assert soil_state.rootfr_patch.shape == (500, 15)
    
    # Verify still NaN-initialized
    assert jnp.all(jnp.isnan(soil_state.cellorg_col))


def test_edge_case_non_contiguous_bounds():
    """Test with non-zero starting indices."""
    bounds = BoundsType(begp=10, endp=20, begc=5, endc=10, begg=2, endg=5)
    soil_state = init_soil_state(bounds, 10, 15, 5)
    
    # Dimensions should be based on range, not absolute values
    assert soil_state.cellorg_col.shape == (6, 10)  # 10-5+1 = 6 columns
    assert soil_state.rootfr_patch.shape == (11, 15)  # 20-10+1 = 11 patches


def test_edge_case_asymmetric_layers():
    """Test with very different layer counts."""
    bounds = create_simple_bounds(5, 3, 2)
    
    # Many soil layers, few ground layers
    soil_state1 = init_soil_state(bounds, 20, 5, 2)
    assert soil_state1.cellorg_col.shape == (3, 20)
    assert soil_state1.hksat_col.shape == (3, 5)
    assert soil_state1.thk_col.shape == (3, 7)  # 5 + 2
    
    # Few soil layers, many ground layers
    soil_state2 = init_soil_state(bounds, 2, 30, 10)
    assert soil_state2.cellorg_col.shape == (3, 2)
    assert soil_state2.hksat_col.shape == (3, 30)
    assert soil_state2.thk_col.shape == (3, 40)  # 30 + 10


# ============================================================================
# Integration Tests
# ============================================================================


def test_integration_full_workflow():
    """
    Test complete workflow: create bounds -> get indices -> initialize state.
    
    Simulates typical usage pattern in CLM model.
    """
    # Step 1: Create bounds
    n_patches, n_columns, n_gridcells = 20, 10, 5
    bounds = create_simple_bounds(n_patches, n_columns, n_gridcells)
    
    # Step 2: Get layer indices
    nlevsno, nlevgrnd = 5, 15
    snow_start, ground_start, ground_end = get_soil_layer_indices(nlevsno, nlevgrnd)
    
    # Step 3: Initialize soil state
    nlevsoi = 10
    soil_state = init_soil_state(bounds, nlevsoi, nlevgrnd, nlevsno)
    
    # Verify consistency
    assert bounds.endp - bounds.begp + 1 == n_patches
    assert bounds.endc - bounds.begc + 1 == n_columns
    assert ground_end == nlevgrnd
    assert soil_state.cellorg_col.shape == (n_columns, nlevsoi)
    assert soil_state.rootfr_patch.shape == (n_patches, nlevgrnd)


def test_integration_multiple_initializations():
    """Test that multiple initializations produce independent results."""
    bounds = create_simple_bounds(5, 3, 2)
    
    state1 = init_soil_state(bounds, 10, 15, 5)
    state2 = init_soil_state(bounds, 10, 15, 5)
    
    # Should have same shapes
    assert state1.cellorg_col.shape == state2.cellorg_col.shape
    
    # Both should be NaN (can't test for different values since both are NaN)
    assert jnp.all(jnp.isnan(state1.cellorg_col))
    assert jnp.all(jnp.isnan(state2.cellorg_col))


# ============================================================================
# Validation Tests
# ============================================================================


def test_validation_bounds_consistency():
    """Test that bounds maintain beg <= end relationships."""
    bounds = create_simple_bounds(10, 5, 3)
    
    assert bounds.begp <= bounds.endp, "Patch bounds inconsistent"
    assert bounds.begc <= bounds.endc, "Column bounds inconsistent"
    assert bounds.begg <= bounds.endg, "Gridcell bounds inconsistent"


def test_validation_layer_counts_positive():
    """Test that layer counts are positive as required."""
    bounds = create_simple_bounds(3, 2, 1)
    
    # All layer counts must be >= 1
    for nlevsoi in [1, 5, 10]:
        for nlevgrnd in [1, 10, 20]:
            for nlevsno in [1, 3, 7]:
                soil_state = init_soil_state(bounds, nlevsoi, nlevgrnd, nlevsno)
                assert soil_state.cellorg_col.shape[1] == nlevsoi
                assert soil_state.hksat_col.shape[1] == nlevgrnd


def test_validation_thk_col_combined_layers():
    """Test that thk_col correctly combines ground and snow layers."""
    bounds = create_simple_bounds(2, 1, 1)
    
    test_configs = [
        (10, 15, 5, 20),  # Default: 15 + 5 = 20
        (5, 10, 3, 13),   # Custom: 10 + 3 = 13
        (1, 1, 1, 2),     # Minimum: 1 + 1 = 2
        (20, 30, 10, 40), # Large: 30 + 10 = 40
    ]
    
    for nlevsoi, nlevgrnd, nlevsno, expected_thk_layers in test_configs:
        soil_state = init_soil_state(bounds, nlevsoi, nlevgrnd, nlevsno)
        assert soil_state.thk_col.shape[1] == expected_thk_layers, \
            f"thk_col should have {expected_thk_layers} layers for nlevgrnd={nlevgrnd}, nlevsno={nlevsno}"


# ============================================================================
# Documentation Tests
# ============================================================================


def test_documentation_bounds_type_fields():
    """Test that BoundsType has all required fields."""
    bounds = create_simple_bounds(5, 3, 2)
    
    required_fields = ["begp", "endp", "begc", "endc", "begg", "endg"]
    for field in required_fields:
        assert hasattr(bounds, field), f"BoundsType missing field: {field}"


def test_documentation_soil_state_type_fields():
    """Test that SoilStateType has all required fields."""
    bounds = create_simple_bounds(3, 2, 1)
    soil_state = init_soil_state(bounds, 5, 10, 3)
    
    required_fields = [
        "cellorg_col", "cellsand_col", "cellclay_col",
        "hksat_col", "hk_l_col", "smp_l_col",
        "bsw_col", "watsat_col", "sucsat_col",
        "dsl_col", "soilresis_col",
        "thk_col", "tkmg_col", "tkdry_col", "csol_col",
        "rootfr_patch"
    ]
    
    for field in required_fields:
        assert hasattr(soil_state, field), f"SoilStateType missing field: {field}"
        assert isinstance(getattr(soil_state, field), jnp.ndarray), \
            f"{field} should be a JAX array"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])