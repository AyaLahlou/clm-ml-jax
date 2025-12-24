"""
Comprehensive pytest suite for SurfaceAlbedoType module.

This test suite validates the initialization and update functions for surface albedo
state management in the CLM biogeophysics module. Tests cover:
- State initialization with various grid dimensions
- Bounds-based allocation
- State updates (full and partial)
- Edge cases (boundary values, extreme conditions)
- Physical constraints (albedo in [0,1], coszen in [-1,1])
- Dimensional consistency
"""

import sys
from pathlib import Path
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_biogeophys.SurfaceAlbedoType import (
    BoundsType,
    SurfaceAlbedoState,
    init,
    init_allocate,
    init_surface_albedo_state,
    update_surface_albedo_state,
)


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load test data for SurfaceAlbedoType functions.
    
    Returns:
        Dictionary containing test cases with inputs and metadata.
    """
    return {
        "test_cases": [
            {
                "name": "test_init_surface_albedo_state_nominal",
                "function": "init_surface_albedo_state",
                "inputs": {"ncol": 10, "npatch": 25, "numrad": 2},
                "metadata": {
                    "type": "nominal",
                    "description": "Standard initialization with typical grid dimensions",
                },
            },
            {
                "name": "test_init_surface_albedo_state_single_column",
                "function": "init_surface_albedo_state",
                "inputs": {"ncol": 1, "npatch": 1, "numrad": 2},
                "metadata": {
                    "type": "edge",
                    "description": "Minimum valid grid size",
                    "edge_cases": ["minimum_dimensions"],
                },
            },
            {
                "name": "test_init_surface_albedo_state_large_grid",
                "function": "init_surface_albedo_state",
                "inputs": {"ncol": 1000, "npatch": 5000, "numrad": 2},
                "metadata": {
                    "type": "special",
                    "description": "Large-scale simulation",
                },
            },
            {
                "name": "test_init_allocate_nominal_bounds",
                "function": "init_allocate",
                "inputs": {
                    "bounds": {
                        "begp": 0,
                        "endp": 49,
                        "begc": 0,
                        "endc": 19,
                        "begg": 0,
                        "endg": 9,
                        "begl": 0,
                        "endl": 14,
                    },
                    "numrad": 2,
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Standard bounds initialization",
                },
            },
            {
                "name": "test_init_allocate_single_element",
                "function": "init_allocate",
                "inputs": {
                    "bounds": {
                        "begp": 0,
                        "endp": 0,
                        "begc": 0,
                        "endc": 0,
                        "begg": 0,
                        "endg": 0,
                        "begl": 0,
                        "endl": 0,
                    },
                    "numrad": 2,
                },
                "metadata": {
                    "type": "edge",
                    "description": "Minimum bounds with single element",
                    "edge_cases": ["minimum_bounds"],
                },
            },
            {
                "name": "test_init_allocate_custom_numrad",
                "function": "init_allocate",
                "inputs": {
                    "bounds": {
                        "begp": 0,
                        "endp": 29,
                        "begc": 0,
                        "endc": 14,
                        "begg": 0,
                        "endg": 4,
                        "begl": 0,
                        "endl": 9,
                    },
                    "numrad": 4,
                },
                "metadata": {
                    "type": "special",
                    "description": "Non-default radiation bands (4 instead of 2)",
                },
            },
        ]
    }


@pytest.fixture
def update_test_data() -> Dict[str, Any]:
    """
    Load test data for update_surface_albedo_state function.
    
    Returns:
        Dictionary containing update test cases with state and update values.
    """
    return {
        "test_cases": [
            {
                "name": "test_update_surface_albedo_state_full_update",
                "inputs": {
                    "state": {
                        "coszen_col": [0.5, 0.6, 0.7, 0.8, 0.9],
                        "albd_patch": [
                            [0.15, 0.25],
                            [0.2, 0.3],
                            [0.18, 0.28],
                            [0.22, 0.32],
                            [0.16, 0.26],
                            [0.19, 0.29],
                            [0.21, 0.31],
                            [0.17, 0.27],
                        ],
                        "albi_patch": [
                            [0.25, 0.35],
                            [0.3, 0.4],
                            [0.28, 0.38],
                            [0.32, 0.42],
                            [0.26, 0.36],
                            [0.29, 0.39],
                            [0.31, 0.41],
                            [0.27, 0.37],
                        ],
                        "albgrd_col": [
                            [0.2, 0.3],
                            [0.22, 0.32],
                            [0.24, 0.34],
                            [0.26, 0.36],
                            [0.28, 0.38],
                        ],
                        "albgri_col": [
                            [0.3, 0.4],
                            [0.32, 0.42],
                            [0.34, 0.44],
                            [0.36, 0.46],
                            [0.38, 0.48],
                        ],
                    },
                    "coszen_col": [0.55, 0.65, 0.75, 0.85, 0.95],
                    "albd_patch": [
                        [0.1, 0.2],
                        [0.12, 0.22],
                        [0.14, 0.24],
                        [0.16, 0.26],
                        [0.11, 0.21],
                        [0.13, 0.23],
                        [0.15, 0.25],
                        [0.17, 0.27],
                    ],
                    "albi_patch": [
                        [0.2, 0.3],
                        [0.22, 0.32],
                        [0.24, 0.34],
                        [0.26, 0.36],
                        [0.21, 0.31],
                        [0.23, 0.33],
                        [0.25, 0.35],
                        [0.27, 0.37],
                    ],
                    "albgrd_col": [
                        [0.18, 0.28],
                        [0.19, 0.29],
                        [0.2, 0.3],
                        [0.21, 0.31],
                        [0.22, 0.32],
                    ],
                    "albgri_col": [
                        [0.28, 0.38],
                        [0.29, 0.39],
                        [0.3, 0.4],
                        [0.31, 0.41],
                        [0.32, 0.42],
                    ],
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Update all fields with typical values",
                },
            },
            {
                "name": "test_update_surface_albedo_state_partial_update",
                "inputs": {
                    "state": {
                        "coszen_col": [0.4, 0.5, 0.6],
                        "albd_patch": [
                            [0.15, 0.25],
                            [0.2, 0.3],
                            [0.18, 0.28],
                            [0.22, 0.32],
                        ],
                        "albi_patch": [
                            [0.25, 0.35],
                            [0.3, 0.4],
                            [0.28, 0.38],
                            [0.32, 0.42],
                        ],
                        "albgrd_col": [[0.2, 0.3], [0.22, 0.32], [0.24, 0.34]],
                        "albgri_col": [[0.3, 0.4], [0.32, 0.42], [0.34, 0.44]],
                    },
                    "coszen_col": [0.7, 0.8, 0.9],
                    "albd_patch": None,
                    "albi_patch": None,
                    "albgrd_col": [[0.19, 0.29], [0.21, 0.31], [0.23, 0.33]],
                    "albgri_col": None,
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Partial update with only coszen and ground direct albedo",
                },
            },
            {
                "name": "test_update_surface_albedo_state_extreme_albedo",
                "inputs": {
                    "state": {
                        "coszen_col": [0.5, 0.6],
                        "albd_patch": [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
                        "albi_patch": [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
                        "albgrd_col": [[0.5, 0.5], [0.5, 0.5]],
                        "albgri_col": [[0.5, 0.5], [0.5, 0.5]],
                    },
                    "coszen_col": [0.0, 1.0],
                    "albd_patch": [[0.0, 0.0], [1.0, 1.0], [0.05, 0.95]],
                    "albi_patch": [[0.0, 0.0], [1.0, 1.0], [0.05, 0.95]],
                    "albgrd_col": [[0.0, 1.0], [1.0, 0.0]],
                    "albgri_col": [[0.0, 1.0], [1.0, 0.0]],
                },
                "metadata": {
                    "type": "edge",
                    "description": "Boundary albedo values and extreme solar angles",
                    "edge_cases": [
                        "zero_albedo",
                        "unity_albedo",
                        "zero_coszen",
                        "unity_coszen",
                    ],
                },
            },
            {
                "name": "test_update_surface_albedo_state_nighttime",
                "inputs": {
                    "state": {
                        "coszen_col": [0.5, 0.6, 0.7, 0.8],
                        "albd_patch": [
                            [0.15, 0.25],
                            [0.2, 0.3],
                            [0.18, 0.28],
                            [0.22, 0.32],
                            [0.16, 0.26],
                        ],
                        "albi_patch": [
                            [0.25, 0.35],
                            [0.3, 0.4],
                            [0.28, 0.38],
                            [0.32, 0.42],
                            [0.26, 0.36],
                        ],
                        "albgrd_col": [
                            [0.2, 0.3],
                            [0.22, 0.32],
                            [0.24, 0.34],
                            [0.26, 0.36],
                        ],
                        "albgri_col": [
                            [0.3, 0.4],
                            [0.32, 0.42],
                            [0.34, 0.44],
                            [0.36, 0.46],
                        ],
                    },
                    "coszen_col": [-0.1, -0.3, -0.5, -0.8],
                    "albd_patch": [
                        [0.12, 0.22],
                        [0.14, 0.24],
                        [0.16, 0.26],
                        [0.18, 0.28],
                        [0.13, 0.23],
                    ],
                    "albi_patch": [
                        [0.22, 0.32],
                        [0.24, 0.34],
                        [0.26, 0.36],
                        [0.28, 0.38],
                        [0.23, 0.33],
                    ],
                    "albgrd_col": [
                        [0.18, 0.28],
                        [0.19, 0.29],
                        [0.2, 0.3],
                        [0.21, 0.31],
                    ],
                    "albgri_col": [
                        [0.28, 0.38],
                        [0.29, 0.39],
                        [0.3, 0.4],
                        [0.31, 0.41],
                    ],
                },
                "metadata": {
                    "type": "edge",
                    "description": "Nighttime conditions with negative coszen",
                    "edge_cases": ["negative_coszen", "nighttime"],
                },
            },
        ]
    }


# ============================================================================
# Tests for init_surface_albedo_state
# ============================================================================


@pytest.mark.parametrize(
    "ncol,npatch,numrad",
    [
        (10, 25, 2),  # Nominal case
        (1, 1, 2),  # Minimum dimensions
        (1000, 5000, 2),  # Large grid
        (5, 10, 4),  # Custom numrad
    ],
)
def test_init_surface_albedo_state_shapes(ncol: int, npatch: int, numrad: int):
    """
    Test that init_surface_albedo_state returns correct array shapes.
    
    Verifies that all arrays in the returned state have dimensions consistent
    with the input parameters (ncol, npatch, numrad).
    """
    state = init_surface_albedo_state(ncol=ncol, npatch=npatch, numrad=numrad)
    
    assert isinstance(state, SurfaceAlbedoState), "Return type should be SurfaceAlbedoState"
    assert state.coszen_col.shape == (ncol,), f"coszen_col shape mismatch: expected ({ncol},), got {state.coszen_col.shape}"
    assert state.albd_patch.shape == (npatch, numrad), f"albd_patch shape mismatch: expected ({npatch}, {numrad}), got {state.albd_patch.shape}"
    assert state.albi_patch.shape == (npatch, numrad), f"albi_patch shape mismatch: expected ({npatch}, {numrad}), got {state.albi_patch.shape}"
    assert state.albgrd_col.shape == (ncol, numrad), f"albgrd_col shape mismatch: expected ({ncol}, {numrad}), got {state.albgrd_col.shape}"
    assert state.albgri_col.shape == (ncol, numrad), f"albgri_col shape mismatch: expected ({ncol}, {numrad}), got {state.albgri_col.shape}"


def test_init_surface_albedo_state_initialization_values():
    """
    Test that init_surface_albedo_state initializes arrays with NaN values.
    
    All arrays should be initialized to NaN to indicate uninitialized state.
    """
    state = init_surface_albedo_state(ncol=5, npatch=10, numrad=2)
    
    assert jnp.all(jnp.isnan(state.coszen_col)), "coszen_col should be initialized with NaN"
    assert jnp.all(jnp.isnan(state.albd_patch)), "albd_patch should be initialized with NaN"
    assert jnp.all(jnp.isnan(state.albi_patch)), "albi_patch should be initialized with NaN"
    assert jnp.all(jnp.isnan(state.albgrd_col)), "albgrd_col should be initialized with NaN"
    assert jnp.all(jnp.isnan(state.albgri_col)), "albgri_col should be initialized with NaN"


def test_init_surface_albedo_state_dtypes():
    """
    Test that init_surface_albedo_state returns arrays with correct dtypes.
    
    All arrays should be float32 for memory efficiency in large simulations.
    """
    state = init_surface_albedo_state(ncol=5, npatch=10, numrad=2)
    
    assert state.coszen_col.dtype == jnp.float32, f"coszen_col dtype should be float32, got {state.coszen_col.dtype}"
    assert state.albd_patch.dtype == jnp.float32, f"albd_patch dtype should be float32, got {state.albd_patch.dtype}"
    assert state.albi_patch.dtype == jnp.float32, f"albi_patch dtype should be float32, got {state.albi_patch.dtype}"
    assert state.albgrd_col.dtype == jnp.float32, f"albgrd_col dtype should be float32, got {state.albgrd_col.dtype}"
    assert state.albgri_col.dtype == jnp.float32, f"albgri_col dtype should be float32, got {state.albgri_col.dtype}"


# ============================================================================
# Tests for init_allocate
# ============================================================================


@pytest.mark.parametrize(
    "bounds_dict,numrad",
    [
        (
            {"begp": 0, "endp": 49, "begc": 0, "endc": 19, "begg": 0, "endg": 9, "begl": 0, "endl": 14},
            2,
        ),  # Nominal
        (
            {"begp": 0, "endp": 0, "begc": 0, "endc": 0, "begg": 0, "endg": 0, "begl": 0, "endl": 0},
            2,
        ),  # Single element
        (
            {"begp": 0, "endp": 29, "begc": 0, "endc": 14, "begg": 0, "endg": 4, "begl": 0, "endl": 9},
            4,
        ),  # Custom numrad
        (
            {"begp": 10, "endp": 59, "begc": 5, "endc": 24, "begg": 0, "endg": 9, "begl": 0, "endl": 14},
            2,
        ),  # Non-zero start
    ],
)
def test_init_allocate_shapes(bounds_dict: Dict[str, int], numrad: int):
    """
    Test that init_allocate returns correct array shapes based on bounds.
    
    Verifies that array dimensions match the bounds specification:
    - npatch = endp - begp + 1
    - ncol = endc - begc + 1
    """
    bounds = BoundsType(**bounds_dict)
    state = init_allocate(bounds=bounds, numrad=numrad)
    
    expected_npatch = bounds.endp - bounds.begp + 1
    expected_ncol = bounds.endc - bounds.begc + 1
    
    assert isinstance(state, SurfaceAlbedoState), "Return type should be SurfaceAlbedoState"
    assert state.coszen_col.shape == (expected_ncol,), f"coszen_col shape mismatch"
    assert state.albd_patch.shape == (expected_npatch, numrad), f"albd_patch shape mismatch"
    assert state.albi_patch.shape == (expected_npatch, numrad), f"albi_patch shape mismatch"
    assert state.albgrd_col.shape == (expected_ncol, numrad), f"albgrd_col shape mismatch"
    assert state.albgri_col.shape == (expected_ncol, numrad), f"albgri_col shape mismatch"


def test_init_allocate_initialization_values():
    """
    Test that init_allocate initializes arrays with NaN values.
    
    All arrays should be initialized to NaN to indicate uninitialized state.
    """
    bounds = BoundsType(begp=0, endp=9, begc=0, endc=4)
    state = init_allocate(bounds=bounds, numrad=2)
    
    assert jnp.all(jnp.isnan(state.coszen_col)), "coszen_col should be initialized with NaN"
    assert jnp.all(jnp.isnan(state.albd_patch)), "albd_patch should be initialized with NaN"
    assert jnp.all(jnp.isnan(state.albi_patch)), "albi_patch should be initialized with NaN"
    assert jnp.all(jnp.isnan(state.albgrd_col)), "albgrd_col should be initialized with NaN"
    assert jnp.all(jnp.isnan(state.albgri_col)), "albgri_col should be initialized with NaN"


def test_init_allocate_dtypes():
    """
    Test that init_allocate returns arrays with correct dtypes.
    
    All arrays should be float32 for memory efficiency.
    """
    bounds = BoundsType(begp=0, endp=9, begc=0, endc=4)
    state = init_allocate(bounds=bounds, numrad=2)
    
    assert state.coszen_col.dtype == jnp.float32, f"coszen_col dtype should be float32"
    assert state.albd_patch.dtype == jnp.float32, f"albd_patch dtype should be float32"
    assert state.albi_patch.dtype == jnp.float32, f"albi_patch dtype should be float32"
    assert state.albgrd_col.dtype == jnp.float32, f"albgrd_col dtype should be float32"
    assert state.albgri_col.dtype == jnp.float32, f"albgri_col dtype should be float32"


# ============================================================================
# Tests for init (wrapper function)
# ============================================================================


def test_init_wrapper_equivalence():
    """
    Test that init function is equivalent to init_allocate.
    
    The init function should be a wrapper that calls init_allocate.
    """
    bounds = BoundsType(begp=0, endp=19, begc=0, endc=9)
    numrad = 2
    
    state_init = init(bounds=bounds, numrad=numrad)
    state_allocate = init_allocate(bounds=bounds, numrad=numrad)
    
    assert state_init.coszen_col.shape == state_allocate.coszen_col.shape
    assert state_init.albd_patch.shape == state_allocate.albd_patch.shape
    assert state_init.albi_patch.shape == state_allocate.albi_patch.shape
    assert state_init.albgrd_col.shape == state_allocate.albgrd_col.shape
    assert state_init.albgri_col.shape == state_allocate.albgri_col.shape


# ============================================================================
# Tests for update_surface_albedo_state
# ============================================================================


def test_update_surface_albedo_state_full_update():
    """
    Test full update of all state fields.
    
    Verifies that when all optional parameters are provided, the state
    is updated correctly with the new values.
    """
    # Create initial state
    initial_state = SurfaceAlbedoState(
        coszen_col=jnp.array([0.5, 0.6, 0.7], dtype=jnp.float32),
        albd_patch=jnp.array([[0.15, 0.25], [0.2, 0.3]], dtype=jnp.float32),
        albi_patch=jnp.array([[0.25, 0.35], [0.3, 0.4]], dtype=jnp.float32),
        albgrd_col=jnp.array([[0.2, 0.3], [0.22, 0.32], [0.24, 0.34]], dtype=jnp.float32),
        albgri_col=jnp.array([[0.3, 0.4], [0.32, 0.42], [0.34, 0.44]], dtype=jnp.float32),
    )
    
    # New values
    new_coszen = jnp.array([0.8, 0.85, 0.9], dtype=jnp.float32)
    new_albd = jnp.array([[0.1, 0.2], [0.12, 0.22]], dtype=jnp.float32)
    new_albi = jnp.array([[0.2, 0.3], [0.22, 0.32]], dtype=jnp.float32)
    new_albgrd = jnp.array([[0.18, 0.28], [0.19, 0.29], [0.2, 0.3]], dtype=jnp.float32)
    new_albgri = jnp.array([[0.28, 0.38], [0.29, 0.39], [0.3, 0.4]], dtype=jnp.float32)
    
    # Update state
    updated_state = update_surface_albedo_state(
        state=initial_state,
        coszen_col=new_coszen,
        albd_patch=new_albd,
        albi_patch=new_albi,
        albgrd_col=new_albgrd,
        albgri_col=new_albgri,
    )
    
    # Verify updates
    assert jnp.allclose(updated_state.coszen_col, new_coszen, atol=1e-6, rtol=1e-6), "coszen_col not updated correctly"
    assert jnp.allclose(updated_state.albd_patch, new_albd, atol=1e-6, rtol=1e-6), "albd_patch not updated correctly"
    assert jnp.allclose(updated_state.albi_patch, new_albi, atol=1e-6, rtol=1e-6), "albi_patch not updated correctly"
    assert jnp.allclose(updated_state.albgrd_col, new_albgrd, atol=1e-6, rtol=1e-6), "albgrd_col not updated correctly"
    assert jnp.allclose(updated_state.albgri_col, new_albgri, atol=1e-6, rtol=1e-6), "albgri_col not updated correctly"


def test_update_surface_albedo_state_partial_update():
    """
    Test partial update where only some fields are updated.
    
    Verifies that when some parameters are None, those fields retain
    their original values while others are updated.
    """
    # Create initial state
    initial_state = SurfaceAlbedoState(
        coszen_col=jnp.array([0.4, 0.5, 0.6], dtype=jnp.float32),
        albd_patch=jnp.array([[0.15, 0.25], [0.2, 0.3]], dtype=jnp.float32),
        albi_patch=jnp.array([[0.25, 0.35], [0.3, 0.4]], dtype=jnp.float32),
        albgrd_col=jnp.array([[0.2, 0.3], [0.22, 0.32], [0.24, 0.34]], dtype=jnp.float32),
        albgri_col=jnp.array([[0.3, 0.4], [0.32, 0.42], [0.34, 0.44]], dtype=jnp.float32),
    )
    
    # Update only coszen and albgrd
    new_coszen = jnp.array([0.7, 0.8, 0.9], dtype=jnp.float32)
    new_albgrd = jnp.array([[0.19, 0.29], [0.21, 0.31], [0.23, 0.33]], dtype=jnp.float32)
    
    updated_state = update_surface_albedo_state(
        state=initial_state,
        coszen_col=new_coszen,
        albd_patch=None,
        albi_patch=None,
        albgrd_col=new_albgrd,
        albgri_col=None,
    )
    
    # Verify updated fields
    assert jnp.allclose(updated_state.coszen_col, new_coszen, atol=1e-6, rtol=1e-6), "coszen_col not updated correctly"
    assert jnp.allclose(updated_state.albgrd_col, new_albgrd, atol=1e-6, rtol=1e-6), "albgrd_col not updated correctly"
    
    # Verify unchanged fields
    assert jnp.allclose(updated_state.albd_patch, initial_state.albd_patch, atol=1e-6, rtol=1e-6), "albd_patch should remain unchanged"
    assert jnp.allclose(updated_state.albi_patch, initial_state.albi_patch, atol=1e-6, rtol=1e-6), "albi_patch should remain unchanged"
    assert jnp.allclose(updated_state.albgri_col, initial_state.albgri_col, atol=1e-6, rtol=1e-6), "albgri_col should remain unchanged"


def test_update_surface_albedo_state_extreme_values():
    """
    Test update with extreme but valid values.
    
    Tests boundary conditions:
    - Albedo values at 0.0 (perfect absorption) and 1.0 (perfect reflection)
    - Coszen at 0.0 (horizon) and 1.0 (zenith)
    """
    initial_state = SurfaceAlbedoState(
        coszen_col=jnp.array([0.5, 0.6], dtype=jnp.float32),
        albd_patch=jnp.array([[0.5, 0.5], [0.5, 0.5]], dtype=jnp.float32),
        albi_patch=jnp.array([[0.5, 0.5], [0.5, 0.5]], dtype=jnp.float32),
        albgrd_col=jnp.array([[0.5, 0.5], [0.5, 0.5]], dtype=jnp.float32),
        albgri_col=jnp.array([[0.5, 0.5], [0.5, 0.5]], dtype=jnp.float32),
    )
    
    # Extreme values
    new_coszen = jnp.array([0.0, 1.0], dtype=jnp.float32)
    new_albd = jnp.array([[0.0, 0.0], [1.0, 1.0]], dtype=jnp.float32)
    new_albi = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    new_albgrd = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    new_albgri = jnp.array([[0.0, 0.0], [1.0, 1.0]], dtype=jnp.float32)
    
    updated_state = update_surface_albedo_state(
        state=initial_state,
        coszen_col=new_coszen,
        albd_patch=new_albd,
        albi_patch=new_albi,
        albgrd_col=new_albgrd,
        albgri_col=new_albgri,
    )
    
    # Verify extreme values are preserved
    assert jnp.allclose(updated_state.coszen_col, new_coszen, atol=1e-6, rtol=1e-6), "Extreme coszen values not preserved"
    assert jnp.allclose(updated_state.albd_patch, new_albd, atol=1e-6, rtol=1e-6), "Extreme albd values not preserved"
    assert jnp.allclose(updated_state.albi_patch, new_albi, atol=1e-6, rtol=1e-6), "Extreme albi values not preserved"
    assert jnp.allclose(updated_state.albgrd_col, new_albgrd, atol=1e-6, rtol=1e-6), "Extreme albgrd values not preserved"
    assert jnp.allclose(updated_state.albgri_col, new_albgri, atol=1e-6, rtol=1e-6), "Extreme albgri values not preserved"
    
    # Verify physical constraints
    assert jnp.all(updated_state.coszen_col >= 0.0) and jnp.all(updated_state.coszen_col <= 1.0), "coszen outside [0, 1]"
    assert jnp.all(updated_state.albd_patch >= 0.0) and jnp.all(updated_state.albd_patch <= 1.0), "albd outside [0, 1]"
    assert jnp.all(updated_state.albi_patch >= 0.0) and jnp.all(updated_state.albi_patch <= 1.0), "albi outside [0, 1]"
    assert jnp.all(updated_state.albgrd_col >= 0.0) and jnp.all(updated_state.albgrd_col <= 1.0), "albgrd outside [0, 1]"
    assert jnp.all(updated_state.albgri_col >= 0.0) and jnp.all(updated_state.albgri_col <= 1.0), "albgri outside [0, 1]"


def test_update_surface_albedo_state_nighttime():
    """
    Test update with nighttime conditions (negative coszen).
    
    Verifies that negative cosine values (sun below horizon) are handled
    correctly, which is physically valid for nighttime.
    """
    initial_state = SurfaceAlbedoState(
        coszen_col=jnp.array([0.5, 0.6, 0.7], dtype=jnp.float32),
        albd_patch=jnp.array([[0.15, 0.25], [0.2, 0.3]], dtype=jnp.float32),
        albi_patch=jnp.array([[0.25, 0.35], [0.3, 0.4]], dtype=jnp.float32),
        albgrd_col=jnp.array([[0.2, 0.3], [0.22, 0.32], [0.24, 0.34]], dtype=jnp.float32),
        albgri_col=jnp.array([[0.3, 0.4], [0.32, 0.42], [0.34, 0.44]], dtype=jnp.float32),
    )
    
    # Nighttime coszen values (negative)
    new_coszen = jnp.array([-0.1, -0.3, -0.5], dtype=jnp.float32)
    
    updated_state = update_surface_albedo_state(
        state=initial_state,
        coszen_col=new_coszen,
        albd_patch=None,
        albi_patch=None,
        albgrd_col=None,
        albgri_col=None,
    )
    
    # Verify nighttime values
    assert jnp.allclose(updated_state.coszen_col, new_coszen, atol=1e-6, rtol=1e-6), "Nighttime coszen not updated correctly"
    assert jnp.all(updated_state.coszen_col < 0.0), "Expected negative coszen for nighttime"
    assert jnp.all(updated_state.coszen_col >= -1.0), "coszen should be >= -1.0"


def test_update_surface_albedo_state_shape_consistency():
    """
    Test that update maintains shape consistency.
    
    Verifies that the updated state has the same shapes as the initial state.
    """
    initial_state = init_surface_albedo_state(ncol=8, npatch=15, numrad=2)
    
    # Create update arrays with matching shapes
    new_coszen = jnp.ones((8,), dtype=jnp.float32) * 0.7
    new_albd = jnp.ones((15, 2), dtype=jnp.float32) * 0.2
    
    updated_state = update_surface_albedo_state(
        state=initial_state,
        coszen_col=new_coszen,
        albd_patch=new_albd,
        albi_patch=None,
        albgrd_col=None,
        albgri_col=None,
    )
    
    # Verify shapes are preserved
    assert updated_state.coszen_col.shape == initial_state.coszen_col.shape, "coszen_col shape changed"
    assert updated_state.albd_patch.shape == initial_state.albd_patch.shape, "albd_patch shape changed"
    assert updated_state.albi_patch.shape == initial_state.albi_patch.shape, "albi_patch shape changed"
    assert updated_state.albgrd_col.shape == initial_state.albgrd_col.shape, "albgrd_col shape changed"
    assert updated_state.albgri_col.shape == initial_state.albgri_col.shape, "albgri_col shape changed"


def test_update_surface_albedo_state_dtypes():
    """
    Test that update preserves float32 dtypes.
    
    Verifies that all arrays maintain float32 dtype after update.
    """
    initial_state = init_surface_albedo_state(ncol=5, npatch=10, numrad=2)
    
    new_coszen = jnp.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=jnp.float32)
    
    updated_state = update_surface_albedo_state(
        state=initial_state,
        coszen_col=new_coszen,
        albd_patch=None,
        albi_patch=None,
        albgrd_col=None,
        albgri_col=None,
    )
    
    assert updated_state.coszen_col.dtype == jnp.float32, "coszen_col dtype changed"
    assert updated_state.albd_patch.dtype == jnp.float32, "albd_patch dtype changed"
    assert updated_state.albi_patch.dtype == jnp.float32, "albi_patch dtype changed"
    assert updated_state.albgrd_col.dtype == jnp.float32, "albgrd_col dtype changed"
    assert updated_state.albgri_col.dtype == jnp.float32, "albgri_col dtype changed"


# ============================================================================
# Edge case tests
# ============================================================================


def test_physical_constraints_albedo_range():
    """
    Test that albedo values are constrained to [0, 1].
    
    Verifies physical constraint that albedo (fraction of reflected radiation)
    must be between 0 (perfect absorption) and 1 (perfect reflection).
    """
    state = init_surface_albedo_state(ncol=10, npatch=20, numrad=2)
    
    # Update with valid albedo values
    valid_albd = jnp.linspace(0.0, 1.0, 40, dtype=jnp.float32).reshape(20, 2)
    valid_albi = jnp.linspace(0.0, 1.0, 40, dtype=jnp.float32).reshape(20, 2)
    valid_albgrd = jnp.linspace(0.0, 1.0, 20, dtype=jnp.float32).reshape(10, 2)
    valid_albgri = jnp.linspace(0.0, 1.0, 20, dtype=jnp.float32).reshape(10, 2)
    
    updated_state = update_surface_albedo_state(
        state=state,
        coszen_col=None,
        albd_patch=valid_albd,
        albi_patch=valid_albi,
        albgrd_col=valid_albgrd,
        albgri_col=valid_albgri,
    )
    
    # Verify all albedo values are in [0, 1]
    assert jnp.all(updated_state.albd_patch >= 0.0) and jnp.all(updated_state.albd_patch <= 1.0), "albd_patch outside [0, 1]"
    assert jnp.all(updated_state.albi_patch >= 0.0) and jnp.all(updated_state.albi_patch <= 1.0), "albi_patch outside [0, 1]"
    assert jnp.all(updated_state.albgrd_col >= 0.0) and jnp.all(updated_state.albgrd_col <= 1.0), "albgrd_col outside [0, 1]"
    assert jnp.all(updated_state.albgri_col >= 0.0) and jnp.all(updated_state.albgri_col <= 1.0), "albgri_col outside [0, 1]"


def test_physical_constraints_coszen_range():
    """
    Test that coszen values are constrained to [-1, 1].
    
    Verifies physical constraint that cosine of solar zenith angle must be
    in [-1, 1], with negative values indicating sun below horizon (nighttime).
    """
    state = init_surface_albedo_state(ncol=10, npatch=20, numrad=2)
    
    # Update with valid coszen values spanning full range
    valid_coszen = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float32)
    
    updated_state = update_surface_albedo_state(
        state=state,
        coszen_col=valid_coszen,
        albd_patch=None,
        albi_patch=None,
        albgrd_col=None,
        albgri_col=None,
    )
    
    # Verify coszen is in [-1, 1]
    assert jnp.all(updated_state.coszen_col >= -1.0) and jnp.all(updated_state.coszen_col <= 1.0), "coszen_col outside [-1, 1]"


def test_multiband_radiation():
    """
    Test with multiple radiation bands (more than default 2).
    
    Verifies that the system correctly handles non-standard numbers of
    radiation bands (e.g., for multi-spectral simulations).
    """
    numrad = 6  # More bands than default
    state = init_surface_albedo_state(ncol=5, npatch=10, numrad=numrad)
    
    # Verify shapes with custom numrad
    assert state.albd_patch.shape == (10, numrad), f"albd_patch shape incorrect for numrad={numrad}"
    assert state.albi_patch.shape == (10, numrad), f"albi_patch shape incorrect for numrad={numrad}"
    assert state.albgrd_col.shape == (5, numrad), f"albgrd_col shape incorrect for numrad={numrad}"
    assert state.albgri_col.shape == (5, numrad), f"albgri_col shape incorrect for numrad={numrad}"
    
    # Update with multi-band data
    new_albd = jnp.ones((10, numrad), dtype=jnp.float32) * 0.3
    updated_state = update_surface_albedo_state(
        state=state,
        coszen_col=None,
        albd_patch=new_albd,
        albi_patch=None,
        albgrd_col=None,
        albgri_col=None,
    )
    
    assert updated_state.albd_patch.shape == (10, numrad), "Shape changed after update"
    assert jnp.allclose(updated_state.albd_patch, new_albd, atol=1e-6, rtol=1e-6), "Multi-band update failed"


def test_realistic_albedo_scenarios():
    """
    Test with realistic albedo values for different surface types.
    
    Tests typical albedo ranges for:
    - Snow/ice: high albedo (~0.8-0.95)
    - Vegetation: low visible, moderate NIR (~0.1-0.2 vis, ~0.4-0.5 NIR)
    - Soil: moderate albedo (~0.15-0.35)
    - Water: low albedo (~0.05-0.15)
    """
    state = init_surface_albedo_state(ncol=4, npatch=4, numrad=2)
    
    # Realistic albedo values: [visible, NIR]
    snow_albedo = jnp.array([[0.85, 0.90]], dtype=jnp.float32)
    vegetation_albedo = jnp.array([[0.15, 0.45]], dtype=jnp.float32)
    soil_albedo = jnp.array([[0.20, 0.30]], dtype=jnp.float32)
    water_albedo = jnp.array([[0.08, 0.10]], dtype=jnp.float32)
    
    realistic_albd = jnp.vstack([snow_albedo, vegetation_albedo, soil_albedo, water_albedo])
    realistic_albi = realistic_albd * 1.1  # Diffuse typically slightly higher
    realistic_albi = jnp.clip(realistic_albi, 0.0, 1.0)  # Ensure within bounds
    
    updated_state = update_surface_albedo_state(
        state=state,
        coszen_col=jnp.array([0.7, 0.75, 0.8, 0.85], dtype=jnp.float32),
        albd_patch=realistic_albd,
        albi_patch=realistic_albi,
        albgrd_col=realistic_albd,
        albgri_col=realistic_albi,
    )
    
    # Verify realistic values are preserved
    assert jnp.allclose(updated_state.albd_patch, realistic_albd, atol=1e-6, rtol=1e-6), "Realistic albedo values not preserved"
    
    # Verify physical constraints
    assert jnp.all(updated_state.albd_patch >= 0.0) and jnp.all(updated_state.albd_patch <= 1.0), "Albedo outside physical range"
    assert jnp.all(updated_state.albi_patch >= 0.0) and jnp.all(updated_state.albi_patch <= 1.0), "Albedo outside physical range"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])