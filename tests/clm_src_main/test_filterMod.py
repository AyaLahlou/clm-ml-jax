"""
Comprehensive pytest suite for filterMod module.

This test suite covers the CLM filter data structure and associated functions
for processing subsets of CLM patches and columns based on different criteria.

Test Coverage:
- clumpfilter dataclass initialization and field validation
- allocFilters: Filter allocation with various grid sizes
- setFilters: Default filter initialization
- setExposedvegpFilter: Exposed vegetation filter setting
- setExposedvegpFilter_jax: JAX-compiled filter setting
- create_filter_instance: Factory function for filter creation
- reset_global_filter: Global filter reset
- get_filter_indices: Filter index retrieval
- apply_patch_filter: Patch filter application
- apply_column_filter: Column filter application

Edge Cases:
- Minimum grid sizes (1x1)
- Empty filters (all zeros)
- Full filters (all non-zero)
- Sparse/non-contiguous indices
- Boundary values (0.0, 1.0, near-zero, near-one)
- Large realistic grids (10k patches)
- Multidimensional data arrays
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import pytest
import jax.numpy as jnp
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from clm_src_main.filterMod import (
    clumpfilter,
    allocFilters,
    setFilters,
    setExposedvegpFilter,
    setExposedvegpFilter_jax,
    create_filter_instance,
    reset_global_filter,
    get_filter_indices,
    apply_patch_filter,
    apply_column_filter,
)


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load and provide test data for filterMod tests.
    
    Returns:
        Dictionary containing test cases with inputs and metadata
    """
    return {
        "small_grid": {
            "begp": 1,
            "endp": 10,
            "begc": 1,
            "endc": 5,
            "description": "Small grid (10 patches, 5 columns)"
        },
        "single_element": {
            "begp": 1,
            "endp": 1,
            "begc": 1,
            "endc": 1,
            "description": "Minimum valid grid size"
        },
        "large_grid": {
            "begp": 1,
            "endp": 10000,
            "begc": 1,
            "endc": 5000,
            "description": "Large realistic grid"
        },
        "mixed_coverage": {
            "nolakeurban_indices": jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=jnp.int32),
            "frac_veg_nosno": jnp.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 0.0, 0.001, 0.999]),
            "max_patches": 10,
            "expected_count": 8,  # All non-zero values
            "description": "Mixed vegetation coverage"
        },
        "all_zero": {
            "nolakeurban_indices": jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32),
            "frac_veg_nosno": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "max_patches": 5,
            "expected_count": 0,
            "description": "All zero vegetation coverage"
        },
        "all_exposed": {
            "nolakeurban_indices": jnp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.int32),
            "frac_veg_nosno": jnp.array([1.0, 0.95, 0.88, 0.76, 0.65, 0.54, 0.43, 0.32]),
            "max_patches": 8,
            "expected_count": 8,
            "description": "All patches exposed"
        },
        "sparse_coverage": {
            "nolakeurban_indices": jnp.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=jnp.int32),
            "frac_veg_nosno": jnp.concatenate([
                jnp.zeros(14),
                jnp.array([0.3]),
                jnp.zeros(4),
                jnp.array([0.6]),
                jnp.zeros(30)
            ]),
            "max_patches": 50,
            "expected_count": 2,
            "description": "Sparse non-contiguous indices"
        },
    }


@pytest.fixture
def sample_filter() -> clumpfilter:
    """
    Create a sample filter instance for testing.
    
    Returns:
        Initialized clumpfilter instance
    """
    return create_filter_instance(begp=1, endp=10, begc=1, endc=5)


# ============================================================================
# Tests for clumpfilter dataclass
# ============================================================================

def test_clumpfilter_initialization():
    """Test clumpfilter dataclass initialization with default values."""
    filter_inst = clumpfilter()
    
    assert filter_inst.num_exposedvegp == 0
    assert filter_inst.exposedvegp is None
    assert filter_inst.num_nolakeurbanp == 0
    assert filter_inst.nolakeurbanp is None
    assert filter_inst.num_nolakec == 0
    assert filter_inst.nolakec is None
    assert filter_inst.num_nourbanc == 0
    assert filter_inst.nourbanc is None
    assert filter_inst.num_hydrologyc == 0
    assert filter_inst.hydrologyc is None
    assert filter_inst.begp is None
    assert filter_inst.endp is None
    assert filter_inst.begc is None
    assert filter_inst.endc is None


def test_clumpfilter_custom_initialization():
    """Test clumpfilter dataclass initialization with custom values."""
    exposedvegp_arr = jnp.array([1, 2, 3], dtype=jnp.int32)
    
    filter_inst = clumpfilter(
        num_exposedvegp=3,
        exposedvegp=exposedvegp_arr,
        begp=1,
        endp=10,
        begc=1,
        endc=5
    )
    
    assert filter_inst.num_exposedvegp == 3
    assert jnp.array_equal(filter_inst.exposedvegp, exposedvegp_arr)
    assert filter_inst.begp == 1
    assert filter_inst.endp == 10
    assert filter_inst.begc == 1
    assert filter_inst.endc == 5


# ============================================================================
# Tests for allocFilters
# ============================================================================

@pytest.mark.parametrize("grid_config", [
    {"begp": 1, "endp": 10, "begc": 1, "endc": 5, "name": "small_grid"},
    {"begp": 1, "endp": 1, "begc": 1, "endc": 1, "name": "single_element"},
    {"begp": 1, "endp": 100, "begc": 1, "endc": 50, "name": "medium_grid"},
])
def test_allocFilters_shapes(grid_config: Dict[str, Any]):
    """
    Test allocFilters creates arrays with correct shapes.
    
    Verifies that filter arrays are allocated with dimensions matching
    the patch and column ranges.
    """
    filter_inst = clumpfilter()
    begp, endp = grid_config["begp"], grid_config["endp"]
    begc, endc = grid_config["begc"], grid_config["endc"]
    
    allocFilters(filter_inst, begp, endp, begc, endc)
    
    num_patches = endp - begp + 1
    num_columns = endc - begc + 1
    
    # Check patch filter shapes
    assert filter_inst.exposedvegp.shape == (num_patches,), \
        f"exposedvegp shape mismatch for {grid_config['name']}"
    assert filter_inst.nolakeurbanp.shape == (num_patches,), \
        f"nolakeurbanp shape mismatch for {grid_config['name']}"
    
    # Check column filter shapes
    assert filter_inst.nolakec.shape == (num_columns,), \
        f"nolakec shape mismatch for {grid_config['name']}"
    assert filter_inst.nourbanc.shape == (num_columns,), \
        f"nourbanc shape mismatch for {grid_config['name']}"
    assert filter_inst.hydrologyc.shape == (num_columns,), \
        f"hydrologyc shape mismatch for {grid_config['name']}"
    
    # Check index storage
    assert filter_inst.begp == begp
    assert filter_inst.endp == endp
    assert filter_inst.begc == begc
    assert filter_inst.endc == endc


def test_allocFilters_dtypes():
    """Test allocFilters creates arrays with correct data types (int32)."""
    filter_inst = clumpfilter()
    allocFilters(filter_inst, begp=1, endp=10, begc=1, endc=5)
    
    assert filter_inst.exposedvegp.dtype == jnp.int32
    assert filter_inst.nolakeurbanp.dtype == jnp.int32
    assert filter_inst.nolakec.dtype == jnp.int32
    assert filter_inst.nourbanc.dtype == jnp.int32
    assert filter_inst.hydrologyc.dtype == jnp.int32


def test_allocFilters_large_grid(test_data: Dict[str, Any]):
    """Test allocFilters with large realistic grid dimensions."""
    filter_inst = clumpfilter()
    large_grid = test_data["large_grid"]
    
    allocFilters(
        filter_inst,
        begp=large_grid["begp"],
        endp=large_grid["endp"],
        begc=large_grid["begc"],
        endc=large_grid["endc"]
    )
    
    assert filter_inst.exposedvegp.shape == (10000,)
    assert filter_inst.nolakec.shape == (5000,)


def test_allocFilters_modifies_in_place():
    """Test that allocFilters modifies the filter instance in place."""
    filter_inst = clumpfilter()
    original_id = id(filter_inst)
    
    allocFilters(filter_inst, begp=1, endp=5, begc=1, endc=3)
    
    assert id(filter_inst) == original_id
    assert filter_inst.exposedvegp is not None


# ============================================================================
# Tests for setFilters
# ============================================================================

def test_setFilters_default_values():
    """
    Test setFilters sets all filters to default values.
    
    All filters should contain a single element with index 1.
    """
    filter_inst = clumpfilter()
    allocFilters(filter_inst, begp=1, endp=10, begc=1, endc=5)
    setFilters(filter_inst)
    
    # Check counts
    assert filter_inst.num_exposedvegp == 1
    assert filter_inst.num_nolakeurbanp == 1
    assert filter_inst.num_nolakec == 1
    assert filter_inst.num_nourbanc == 1
    assert filter_inst.num_hydrologyc == 1
    
    # Check first element is 1 (1-based indexing)
    assert filter_inst.exposedvegp[0] == 1
    assert filter_inst.nolakeurbanp[0] == 1
    assert filter_inst.nolakec[0] == 1
    assert filter_inst.nourbanc[0] == 1
    assert filter_inst.hydrologyc[0] == 1


def test_setFilters_requires_allocation():
    """Test setFilters works correctly on allocated filter."""
    filter_inst = clumpfilter()
    allocFilters(filter_inst, begp=1, endp=5, begc=1, endc=3)
    
    # Should not raise error
    setFilters(filter_inst)
    
    assert filter_inst.num_exposedvegp == 1


# ============================================================================
# Tests for setExposedvegpFilter_jax
# ============================================================================

def test_setExposedvegpFilter_jax_mixed_coverage(test_data: Dict[str, Any]):
    """Test JAX filter setting with mixed vegetation coverage values."""
    mixed = test_data["mixed_coverage"]
    
    exposed_indices, num_exposed = setExposedvegpFilter_jax(
        mixed["nolakeurban_indices"],
        mixed["frac_veg_nosno"],
        mixed["max_patches"]
    )
    
    assert num_exposed == mixed["expected_count"], \
        f"Expected {mixed['expected_count']} exposed patches, got {num_exposed}"
    assert exposed_indices.shape == (mixed["max_patches"],)
    assert exposed_indices.dtype == jnp.int32
    
    # Verify non-zero indices are compacted at beginning
    valid_indices = exposed_indices[:num_exposed]
    assert jnp.all(valid_indices > 0), "All valid indices should be positive (1-based)"


def test_setExposedvegpFilter_jax_all_zero(test_data: Dict[str, Any]):
    """Test JAX filter setting when all patches have zero coverage."""
    all_zero = test_data["all_zero"]
    
    exposed_indices, num_exposed = setExposedvegpFilter_jax(
        all_zero["nolakeurban_indices"],
        all_zero["frac_veg_nosno"],
        all_zero["max_patches"]
    )
    
    assert num_exposed == 0, "Expected no exposed patches with all-zero coverage"
    assert exposed_indices.shape == (all_zero["max_patches"],)


def test_setExposedvegpFilter_jax_all_exposed(test_data: Dict[str, Any]):
    """Test JAX filter setting when all patches are exposed."""
    all_exposed = test_data["all_exposed"]
    
    exposed_indices, num_exposed = setExposedvegpFilter_jax(
        all_exposed["nolakeurban_indices"],
        all_exposed["frac_veg_nosno"],
        all_exposed["max_patches"]
    )
    
    assert num_exposed == all_exposed["expected_count"]
    
    # All nolakeurban indices should be in exposed filter
    valid_indices = exposed_indices[:num_exposed]
    expected_indices = all_exposed["nolakeurban_indices"]
    assert jnp.array_equal(valid_indices, expected_indices)


def test_setExposedvegpFilter_jax_sparse_coverage(test_data: Dict[str, Any]):
    """Test JAX filter setting with sparse non-contiguous indices."""
    sparse = test_data["sparse_coverage"]
    
    exposed_indices, num_exposed = setExposedvegpFilter_jax(
        sparse["nolakeurban_indices"],
        sparse["frac_veg_nosno"],
        sparse["max_patches"]
    )
    
    assert num_exposed == sparse["expected_count"]
    
    # Verify the correct indices are selected (15 and 20 have non-zero values)
    valid_indices = exposed_indices[:num_exposed]
    assert 15 in valid_indices or 20 in valid_indices


def test_setExposedvegpFilter_jax_boundary_values():
    """Test JAX filter setting with boundary values (0.0, 1.0, near-zero, near-one)."""
    nolakeurban_indices = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    frac_veg_nosno = jnp.array([0.0, 1e-10, 0.999999, 1.0])
    max_patches = 4
    
    exposed_indices, num_exposed = setExposedvegpFilter_jax(
        nolakeurban_indices,
        frac_veg_nosno,
        max_patches
    )
    
    # Should include indices 2, 3, 4 (all non-zero)
    assert num_exposed == 3
    valid_indices = exposed_indices[:num_exposed]
    assert 1 not in valid_indices  # Zero should be excluded
    assert 2 in valid_indices  # Near-zero should be included
    assert 3 in valid_indices  # Near-one should be included
    assert 4 in valid_indices  # One should be included


def test_setExposedvegpFilter_jax_output_shapes():
    """Test JAX filter setting returns correct output shapes."""
    nolakeurban_indices = jnp.array([1, 2, 3], dtype=jnp.int32)
    frac_veg_nosno = jnp.array([0.5, 0.0, 0.7])
    max_patches = 10
    
    exposed_indices, num_exposed = setExposedvegpFilter_jax(
        nolakeurban_indices,
        frac_veg_nosno,
        max_patches
    )
    
    assert isinstance(exposed_indices, jnp.ndarray)
    assert isinstance(num_exposed, (int, jnp.integer))
    assert exposed_indices.shape == (max_patches,)


# ============================================================================
# Tests for setExposedvegpFilter
# ============================================================================

def test_setExposedvegpFilter_integration(test_data: Dict[str, Any]):
    """Test setExposedvegpFilter integration with filter instance."""
    mixed = test_data["mixed_coverage"]
    
    filter_inst = clumpfilter()
    allocFilters(filter_inst, begp=1, endp=10, begc=1, endc=5)
    setFilters(filter_inst)
    
    # Set nolakeurbanp filter first
    filter_inst.nolakeurbanp = mixed["nolakeurban_indices"]
    filter_inst.num_nolakeurbanp = len(mixed["nolakeurban_indices"])
    
    setExposedvegpFilter(filter_inst, mixed["frac_veg_nosno"])
    
    assert filter_inst.num_exposedvegp == mixed["expected_count"]
    assert filter_inst.exposedvegp is not None


def test_setExposedvegpFilter_updates_count():
    """Test setExposedvegpFilter correctly updates the count."""
    filter_inst = clumpfilter()
    allocFilters(filter_inst, begp=1, endp=5, begc=1, endc=3)
    setFilters(filter_inst)
    
    # Set nolakeurbanp
    filter_inst.nolakeurbanp = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
    filter_inst.num_nolakeurbanp = 5
    
    frac_veg_nosno = jnp.array([0.5, 0.0, 0.3, 0.0, 0.8])
    
    setExposedvegpFilter(filter_inst, frac_veg_nosno)
    
    # Should have 3 exposed patches (indices 1, 3, 5)
    assert filter_inst.num_exposedvegp == 3


# ============================================================================
# Tests for create_filter_instance
# ============================================================================

def test_create_filter_instance_typical(test_data: Dict[str, Any]):
    """Test factory function with typical grid dimensions."""
    filter_inst = create_filter_instance(begp=1, endp=100, begc=1, endc=50)
    
    assert isinstance(filter_inst, clumpfilter)
    assert filter_inst.begp == 1
    assert filter_inst.endp == 100
    assert filter_inst.begc == 1
    assert filter_inst.endc == 50
    
    # Check allocation
    assert filter_inst.exposedvegp is not None
    assert filter_inst.exposedvegp.shape == (100,)
    
    # Check default values set
    assert filter_inst.num_exposedvegp == 1
    assert filter_inst.exposedvegp[0] == 1


def test_create_filter_instance_small(test_data: Dict[str, Any]):
    """Test factory function with small grid."""
    small = test_data["small_grid"]
    filter_inst = create_filter_instance(
        begp=small["begp"],
        endp=small["endp"],
        begc=small["begc"],
        endc=small["endc"]
    )
    
    assert filter_inst.exposedvegp.shape == (10,)
    assert filter_inst.nolakec.shape == (5,)


def test_create_filter_instance_single_element(test_data: Dict[str, Any]):
    """Test factory function with minimum grid size."""
    single = test_data["single_element"]
    filter_inst = create_filter_instance(
        begp=single["begp"],
        endp=single["endp"],
        begc=single["begc"],
        endc=single["endc"]
    )
    
    assert filter_inst.exposedvegp.shape == (1,)
    assert filter_inst.nolakec.shape == (1,)
    assert filter_inst.num_exposedvegp == 1


# ============================================================================
# Tests for get_filter_indices
# ============================================================================

@pytest.mark.parametrize("filter_name", [
    "exposedvegp",
    "nolakeurbanp",
    "nolakec",
    "nourbanc",
    "hydrologyc"
])
def test_get_filter_indices_all_filters(filter_name: str, sample_filter: clumpfilter):
    """Test get_filter_indices for all filter types."""
    indices, count = get_filter_indices(sample_filter, filter_name)
    
    assert isinstance(indices, jnp.ndarray)
    assert isinstance(count, (int, jnp.integer))
    assert indices.dtype == jnp.int32
    
    # Default filters have count=1 and first index=1
    assert count == 1
    assert indices[0] == 1


def test_get_filter_indices_custom_filter():
    """Test get_filter_indices with custom filter values."""
    filter_inst = create_filter_instance(begp=1, endp=10, begc=1, endc=5)
    
    # Manually set exposedvegp
    filter_inst.exposedvegp = jnp.array([2, 4, 6, 8, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    filter_inst.num_exposedvegp = 4
    
    indices, count = get_filter_indices(filter_inst, "exposedvegp")
    
    assert count == 4
    assert jnp.array_equal(indices[:count], jnp.array([2, 4, 6, 8]))


def test_get_filter_indices_empty_filter():
    """Test get_filter_indices with empty filter (count=0)."""
    filter_inst = create_filter_instance(begp=1, endp=5, begc=1, endc=3)
    
    # Set empty filter
    filter_inst.num_exposedvegp = 0
    
    indices, count = get_filter_indices(filter_inst, "exposedvegp")
    
    assert count == 0


def test_get_filter_indices_boundary():
    """Test get_filter_indices when filter is at maximum capacity."""
    filter_inst = create_filter_instance(begp=1, endp=15, begc=1, endc=10)
    
    # Fill entire filter
    filter_inst.exposedvegp = jnp.arange(1, 16, dtype=jnp.int32)
    filter_inst.num_exposedvegp = 15
    
    indices, count = get_filter_indices(filter_inst, "exposedvegp")
    
    assert count == 15
    assert jnp.array_equal(indices, jnp.arange(1, 16))


# ============================================================================
# Tests for apply_patch_filter
# ============================================================================

def test_apply_patch_filter_1d():
    """Test apply_patch_filter with 1D data array."""
    filter_inst = create_filter_instance(begp=1, endp=8, begc=1, endc=4)
    
    # Set custom filter
    filter_inst.exposedvegp = jnp.array([1, 3, 5, 7, 0, 0, 0, 0], dtype=jnp.int32)
    filter_inst.num_exposedvegp = 4
    
    data = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    
    filtered = apply_patch_filter(data, filter_inst, "exposedvegp")
    
    expected = jnp.array([10.0, 30.0, 50.0, 70.0])
    assert jnp.allclose(filtered, expected, atol=1e-6)


def test_apply_patch_filter_multidimensional():
    """Test apply_patch_filter with 2D data array (patches x features)."""
    filter_inst = create_filter_instance(begp=1, endp=8, begc=1, endc=4)
    
    # Set custom filter
    filter_inst.exposedvegp = jnp.array([1, 3, 5, 7, 0, 0, 0, 0], dtype=jnp.int32)
    filter_inst.num_exposedvegp = 4
    
    data = jnp.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0],
        [19.0, 20.0, 21.0],
        [22.0, 23.0, 24.0]
    ])
    
    filtered = apply_patch_filter(data, filter_inst, "exposedvegp")
    
    expected = jnp.array([
        [1.0, 2.0, 3.0],
        [7.0, 8.0, 9.0],
        [13.0, 14.0, 15.0],
        [19.0, 20.0, 21.0]
    ])
    
    assert filtered.shape == (4, 3)
    assert jnp.allclose(filtered, expected, atol=1e-6)


def test_apply_patch_filter_nolakeurbanp():
    """Test apply_patch_filter with nolakeurbanp filter."""
    filter_inst = create_filter_instance(begp=1, endp=6, begc=1, endc=3)
    
    filter_inst.nolakeurbanp = jnp.array([2, 4, 6, 0, 0, 0], dtype=jnp.int32)
    filter_inst.num_nolakeurbanp = 3
    
    data = jnp.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    
    filtered = apply_patch_filter(data, filter_inst, "nolakeurbanp")
    
    expected = jnp.array([200.0, 400.0, 600.0])
    assert jnp.allclose(filtered, expected, atol=1e-6)


def test_apply_patch_filter_empty():
    """Test apply_patch_filter with empty filter."""
    filter_inst = create_filter_instance(begp=1, endp=5, begc=1, endc=3)
    
    filter_inst.num_exposedvegp = 0
    
    data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    filtered = apply_patch_filter(data, filter_inst, "exposedvegp")
    
    assert filtered.shape[0] == 0


# ============================================================================
# Tests for apply_column_filter
# ============================================================================

def test_apply_column_filter_1d():
    """Test apply_column_filter with 1D data array."""
    filter_inst = create_filter_instance(begp=1, endp=10, begc=1, endc=6)
    
    # Set custom column filter
    filter_inst.nolakec = jnp.array([1, 2, 4, 0, 0, 0], dtype=jnp.int32)
    filter_inst.num_nolakec = 3
    
    data = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    
    filtered = apply_column_filter(data, filter_inst, "nolakec")
    
    expected = jnp.array([10.0, 20.0, 40.0])
    assert jnp.allclose(filtered, expected, atol=1e-6)


def test_apply_column_filter_multidimensional():
    """Test apply_column_filter with 2D data array (columns x features)."""
    filter_inst = create_filter_instance(begp=1, endp=10, begc=1, endc=5)
    
    filter_inst.hydrologyc = jnp.array([1, 3, 5, 0, 0], dtype=jnp.int32)
    filter_inst.num_hydrologyc = 3
    
    data = jnp.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0]
    ])
    
    filtered = apply_column_filter(data, filter_inst, "hydrologyc")
    
    expected = jnp.array([
        [1.0, 2.0],
        [5.0, 6.0],
        [9.0, 10.0]
    ])
    
    assert filtered.shape == (3, 2)
    assert jnp.allclose(filtered, expected, atol=1e-6)


@pytest.mark.parametrize("filter_name", ["nolakec", "nourbanc", "hydrologyc"])
def test_apply_column_filter_all_types(filter_name: str):
    """Test apply_column_filter with all column filter types."""
    filter_inst = create_filter_instance(begp=1, endp=10, begc=1, endc=4)
    
    data = jnp.array([100.0, 200.0, 300.0, 400.0])
    
    # Default filter has count=1, index=1
    filtered = apply_column_filter(data, filter_inst, filter_name)
    
    assert filtered.shape == (1,)
    assert jnp.allclose(filtered, jnp.array([100.0]), atol=1e-6)


# ============================================================================
# Tests for reset_global_filter
# ============================================================================

def test_reset_global_filter():
    """Test reset_global_filter resets the global filter variable."""
    # This test verifies the function executes without error
    # The actual global state management depends on implementation
    reset_global_filter()
    # No assertion needed - just verify no exception raised


# ============================================================================
# Edge case and integration tests
# ============================================================================

def test_filter_workflow_integration():
    """
    Integration test for complete filter workflow.
    
    Tests: create -> allocate -> set defaults -> set exposed veg -> apply filter
    """
    # Create and initialize
    filter_inst = create_filter_instance(begp=1, endp=20, begc=1, endc=10)
    
    # Set nolakeurbanp filter
    nolakeurban_indices = jnp.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], dtype=jnp.int32)
    filter_inst.nolakeurbanp = jnp.pad(
        nolakeurban_indices,
        (0, 10),
        constant_values=0
    )
    filter_inst.num_nolakeurbanp = 10
    
    # Set exposed vegetation
    frac_veg_nosno = jnp.zeros(20)
    frac_veg_nosno = frac_veg_nosno.at[jnp.array([0, 2, 4, 6, 8])].set(
        jnp.array([0.5, 0.7, 0.3, 0.9, 0.6])
    )
    
    setExposedvegpFilter(filter_inst, frac_veg_nosno)
    
    # Apply filter to data
    data = jnp.arange(1, 21, dtype=jnp.float32)
    filtered = apply_patch_filter(data, filter_inst, "exposedvegp")
    
    # Verify results
    assert filter_inst.num_exposedvegp == 5
    assert filtered.shape[0] == 5


def test_filter_with_zero_based_indexing_conversion():
    """
    Test that 1-based filter indices correctly access 0-based array data.
    
    This is critical for Fortran-to-Python translation.
    """
    filter_inst = create_filter_instance(begp=1, endp=5, begc=1, endc=3)
    
    # Set 1-based indices
    filter_inst.exposedvegp = jnp.array([1, 3, 5, 0, 0], dtype=jnp.int32)
    filter_inst.num_exposedvegp = 3
    
    # 0-based data array
    data = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    
    filtered = apply_patch_filter(data, filter_inst, "exposedvegp")
    
    # Should get elements at 0-based indices 0, 2, 4 (1-based 1, 3, 5)
    expected = jnp.array([10.0, 30.0, 50.0])
    assert jnp.allclose(filtered, expected, atol=1e-6)


def test_filter_dtypes_consistency():
    """Test that all filter operations maintain consistent dtypes."""
    filter_inst = create_filter_instance(begp=1, endp=10, begc=1, endc=5)
    
    # Check all filter arrays are int32
    assert filter_inst.exposedvegp.dtype == jnp.int32
    assert filter_inst.nolakeurbanp.dtype == jnp.int32
    assert filter_inst.nolakec.dtype == jnp.int32
    assert filter_inst.nourbanc.dtype == jnp.int32
    assert filter_inst.hydrologyc.dtype == jnp.int32
    
    # Check counts are integers
    assert isinstance(filter_inst.num_exposedvegp, (int, jnp.integer))
    assert isinstance(filter_inst.num_nolakeurbanp, (int, jnp.integer))


def test_filter_physical_constraints():
    """Test that filters respect physical constraints (frac_veg_nosno in [0,1])."""
    filter_inst = create_filter_instance(begp=1, endp=5, begc=1, endc=3)
    
    filter_inst.nolakeurbanp = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
    filter_inst.num_nolakeurbanp = 5
    
    # Valid fractions in [0, 1]
    frac_veg_nosno = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    setExposedvegpFilter(filter_inst, frac_veg_nosno)
    
    # Should include all non-zero values (4 patches)
    assert filter_inst.num_exposedvegp == 4


def test_filter_compaction():
    """Test that filter indices are compacted at the beginning of arrays."""
    nolakeurban_indices = jnp.array([2, 5, 8, 11, 14], dtype=jnp.int32)
    frac_veg_nosno = jnp.zeros(15)
    frac_veg_nosno = frac_veg_nosno.at[jnp.array([1, 4, 7])].set(
        jnp.array([0.5, 0.6, 0.7])
    )
    max_patches = 15
    
    exposed_indices, num_exposed = setExposedvegpFilter_jax(
        nolakeurban_indices,
        frac_veg_nosno,
        max_patches
    )
    
    # Valid indices should be at the beginning
    valid_indices = exposed_indices[:num_exposed]
    assert jnp.all(valid_indices > 0)
    
    # Remaining indices should be zero or padding
    padding = exposed_indices[num_exposed:]
    assert jnp.all(padding == 0) or len(padding) == 0


def test_filter_shapes_with_different_dimensions():
    """Test filter operations with various array dimensions."""
    filter_inst = create_filter_instance(begp=1, endp=6, begc=1, endc=3)
    
    filter_inst.exposedvegp = jnp.array([1, 2, 3, 0, 0, 0], dtype=jnp.int32)
    filter_inst.num_exposedvegp = 3
    
    # Test 1D
    data_1d = jnp.arange(6, dtype=jnp.float32)
    filtered_1d = apply_patch_filter(data_1d, filter_inst, "exposedvegp")
    assert filtered_1d.shape == (3,)
    
    # Test 2D
    data_2d = jnp.arange(12, dtype=jnp.float32).reshape(6, 2)
    filtered_2d = apply_patch_filter(data_2d, filter_inst, "exposedvegp")
    assert filtered_2d.shape == (3, 2)
    
    # Test 3D
    data_3d = jnp.arange(24, dtype=jnp.float32).reshape(6, 2, 2)
    filtered_3d = apply_patch_filter(data_3d, filter_inst, "exposedvegp")
    assert filtered_3d.shape == (3, 2, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])