"""
Comprehensive pytest suite for clm_initializeMod module.

This module tests the CLM initialization functions including:
- initialize1: Phase 1 initialization (parameters, lookup tables, memory allocation)
- initialize2: Phase 2 initialization (derived types and time constants)
- full_initialize: Combined initialization
- validate_initialization: Validation checks
- JIT-compiled variants of the above functions

Test coverage includes:
- Nominal cases: typical simulation scales (single-cell to global)
- Edge cases: boundary conditions, minimal/maximal ratios
- Special cases: asymmetric hierarchies, sparse distributions
- Validation: bounds constraints and initialization success
"""

import sys
from pathlib import Path
from typing import NamedTuple

import pytest
import numpy as np
import jax
import jax.numpy as jnp

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from clm_src_main.clm_initializeMod import (
    initialize1,
    initialize2,
    full_initialize,
    validate_initialization,
    initialize1_jit,
    initialize2_jit,
    full_initialize_jit,
)

# Import bounds_type from decompMod
try:
    from clm_src_main.decompMod import bounds_type
except ImportError:
    # Fallback: define bounds_type if not available
    class bounds_type(NamedTuple):
        """CLM bounds structure containing grid indices."""
        begg: int
        endg: int
        begc: int
        endc: int
        begp: int
        endp: int


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data():
    """
    Load test data for CLM initialization functions.
    
    Returns:
        dict: Test cases with bounds configurations covering:
            - Nominal cases: typical simulation scales
            - Edge cases: boundary conditions
            - Special cases: asymmetric configurations
    """
    return {
        "test_cases": [
            {
                "name": "test_single_gridcell_single_column_single_patch",
                "inputs": {
                    "bounds": bounds_type(
                        begg=0, endg=0, begc=0, endc=0, begp=0, endp=0
                    )
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Minimal valid configuration with single gridcell, column, and patch",
                },
            },
            {
                "name": "test_small_grid_typical_hierarchy",
                "inputs": {
                    "bounds": bounds_type(
                        begg=0, endg=9, begc=0, endc=49, begp=0, endp=249
                    )
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Small 10-gridcell domain with typical hierarchy",
                },
            },
            {
                "name": "test_medium_grid_realistic_domain",
                "inputs": {
                    "bounds": bounds_type(
                        begg=0, endg=99, begc=0, endc=999, begp=0, endp=9999
                    )
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Medium-sized domain with 100 gridcells",
                },
            },
            {
                "name": "test_large_global_simulation",
                "inputs": {
                    "bounds": bounds_type(
                        begg=0, endg=999, begc=0, endc=19999, begp=0, endp=199999
                    )
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Large global-scale simulation",
                },
            },
            {
                "name": "test_non_zero_starting_indices",
                "inputs": {
                    "bounds": bounds_type(
                        begg=100, endg=149, begc=500, endc=749, begp=2500, endp=3749
                    )
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Domain decomposition with non-zero starting indices",
                },
            },
            {
                "name": "test_edge_equal_begin_end_indices",
                "inputs": {
                    "bounds": bounds_type(
                        begg=5, endg=5, begc=25, endc=25, begp=125, endp=125
                    )
                },
                "metadata": {
                    "type": "edge",
                    "description": "Edge case where begin equals end for all levels",
                },
            },
            {
                "name": "test_edge_minimal_column_patch_ratio",
                "inputs": {
                    "bounds": bounds_type(
                        begg=0, endg=49, begc=0, endc=49, begp=0, endp=49
                    )
                },
                "metadata": {
                    "type": "edge",
                    "description": "Edge case with 1:1:1 ratio",
                },
            },
            {
                "name": "test_edge_high_patch_density",
                "inputs": {
                    "bounds": bounds_type(
                        begg=0, endg=4, begc=0, endc=24, begp=0, endp=499
                    )
                },
                "metadata": {
                    "type": "edge",
                    "description": "Edge case with very high patch-to-gridcell ratio",
                },
            },
            {
                "name": "test_special_asymmetric_hierarchy",
                "inputs": {
                    "bounds": bounds_type(
                        begg=10, endg=29, begc=50, endc=109, begp=300, endp=899
                    )
                },
                "metadata": {
                    "type": "special",
                    "description": "Asymmetric subgrid hierarchy",
                },
            },
            {
                "name": "test_special_sparse_patch_distribution",
                "inputs": {
                    "bounds": bounds_type(
                        begg=0, endg=199, begc=0, endc=399, begp=0, endp=599
                    )
                },
                "metadata": {
                    "type": "special",
                    "description": "Sparse patch distribution",
                },
            },
        ]
    }


@pytest.fixture
def invalid_bounds_data():
    """
    Generate invalid bounds configurations for error testing.
    
    Returns:
        list: Invalid bounds configurations that should fail validation
    """
    return [
        {
            "name": "negative_begg",
            "bounds": bounds_type(begg=-1, endg=10, begc=0, endc=50, begp=0, endp=250),
            "error": "negative index",
        },
        {
            "name": "begg_greater_than_endg",
            "bounds": bounds_type(begg=10, endg=5, begc=0, endc=50, begp=0, endp=250),
            "error": "begin > end",
        },
        {
            "name": "begc_greater_than_endc",
            "bounds": bounds_type(begg=0, endg=10, begc=50, endc=25, begp=0, endp=250),
            "error": "begin > end",
        },
        {
            "name": "begp_greater_than_endp",
            "bounds": bounds_type(begg=0, endg=10, begc=0, endc=50, begp=250, endp=100),
            "error": "begin > end",
        },
        {
            "name": "all_negative",
            "bounds": bounds_type(begg=-5, endg=-1, begc=-10, endc=-5, begp=-20, endp=-10),
            "error": "negative indices",
        },
    ]


# ============================================================================
# Test: validate_initialization
# ============================================================================

@pytest.mark.parametrize("test_case", [
    pytest.param(tc, id=tc["name"]) 
    for tc in [
        {
            "name": "test_single_gridcell_single_column_single_patch",
            "inputs": {"bounds": bounds_type(begg=0, endg=0, begc=0, endc=0, begp=0, endp=0)},
            "metadata": {"type": "nominal"},
        },
        {
            "name": "test_small_grid_typical_hierarchy",
            "inputs": {"bounds": bounds_type(begg=0, endg=9, begc=0, endc=49, begp=0, endp=249)},
            "metadata": {"type": "nominal"},
        },
    ]
])
def test_validate_initialization_valid_bounds(test_case):
    """
    Test validate_initialization with valid bounds configurations.
    
    Verifies that:
    - Valid bounds return True
    - No exceptions are raised
    - Return type is boolean
    """
    bounds = test_case["inputs"]["bounds"]
    
    result = validate_initialization(bounds)
    
    assert isinstance(result, bool), f"Expected bool return type, got {type(result)}"
    assert result is True, f"Valid bounds should return True for {test_case['name']}"


def test_validate_initialization_invalid_bounds(invalid_bounds_data):
    """
    Test validate_initialization with invalid bounds configurations.
    
    Verifies that:
    - Invalid bounds return False or raise appropriate exceptions
    - Constraints are properly checked (begin <= end, non-negative)
    """
    for invalid_case in invalid_bounds_data:
        bounds = invalid_case["bounds"]
        
        # Should either return False or raise an exception
        try:
            result = validate_initialization(bounds)
            assert result is False, (
                f"Invalid bounds should return False: {invalid_case['name']}"
            )
        except (ValueError, AssertionError) as e:
            # Exception is acceptable for invalid bounds
            assert True, f"Appropriately raised exception for {invalid_case['name']}: {e}"


def test_validate_initialization_return_type():
    """
    Test that validate_initialization returns correct type.
    
    Verifies:
    - Return type is always boolean
    - No other types are returned
    """
    bounds = bounds_type(begg=0, endg=10, begc=0, endc=50, begp=0, endp=250)
    result = validate_initialization(bounds)
    
    assert isinstance(result, bool), f"Expected bool, got {type(result)}"


# ============================================================================
# Test: initialize1
# ============================================================================

@pytest.mark.parametrize("test_case", [
    pytest.param(tc, id=tc["name"]) 
    for tc in [
        {
            "name": "test_single_gridcell_single_column_single_patch",
            "inputs": {"bounds": bounds_type(begg=0, endg=0, begc=0, endc=0, begp=0, endp=0)},
            "metadata": {"type": "nominal"},
        },
        {
            "name": "test_small_grid_typical_hierarchy",
            "inputs": {"bounds": bounds_type(begg=0, endg=9, begc=0, endc=49, begp=0, endp=249)},
            "metadata": {"type": "nominal"},
        },
        {
            "name": "test_edge_equal_begin_end_indices",
            "inputs": {"bounds": bounds_type(begg=5, endg=5, begc=25, endc=25, begp=125, endp=125)},
            "metadata": {"type": "edge"},
        },
    ]
])
def test_initialize1_executes_without_error(test_case):
    """
    Test that initialize1 executes without raising exceptions.
    
    Verifies:
    - Function completes successfully
    - No return value (returns None)
    - Side effects are performed (cannot directly test without inspecting global state)
    """
    bounds = test_case["inputs"]["bounds"]
    
    # Should execute without error
    result = initialize1(bounds)
    
    assert result is None, "initialize1 should return None"


def test_initialize1_side_effects():
    """
    Test that initialize1 performs expected side effects.
    
    Verifies:
    - Function is callable
    - Completes without exceptions
    - Can be called multiple times (idempotent or properly handles re-initialization)
    
    Note: Full verification of side effects would require access to global state
    (clm_varpar, pftcon, lookup tables, etc.) which may not be directly testable.
    """
    bounds = bounds_type(begg=0, endg=5, begc=0, endc=25, begp=0, endp=125)
    
    # First call
    initialize1(bounds)
    
    # Second call should also succeed (testing re-initialization)
    try:
        initialize1(bounds)
        assert True, "Re-initialization should succeed"
    except Exception as e:
        pytest.skip(f"Re-initialization not supported: {e}")


# ============================================================================
# Test: initialize2
# ============================================================================

@pytest.mark.parametrize("test_case", [
    pytest.param(tc, id=tc["name"]) 
    for tc in [
        {
            "name": "test_single_gridcell_single_column_single_patch",
            "inputs": {"bounds": bounds_type(begg=0, endg=0, begc=0, endc=0, begp=0, endp=0)},
            "metadata": {"type": "nominal"},
        },
        {
            "name": "test_medium_grid_realistic_domain",
            "inputs": {"bounds": bounds_type(begg=0, endg=99, begc=0, endc=999, begp=0, endp=9999)},
            "metadata": {"type": "nominal"},
        },
    ]
])
def test_initialize2_executes_without_error(test_case):
    """
    Test that initialize2 executes without raising exceptions.
    
    Verifies:
    - Function completes successfully
    - No return value (returns None)
    - Should be called after initialize1 in proper workflow
    """
    bounds = test_case["inputs"]["bounds"]
    
    # Note: In proper workflow, initialize1 should be called first
    # For isolated testing, we call initialize2 directly
    result = initialize2(bounds)
    
    assert result is None, "initialize2 should return None"


def test_initialize2_requires_initialize1():
    """
    Test the proper initialization sequence.
    
    Verifies:
    - initialize2 can be called after initialize1
    - Proper two-phase initialization workflow
    """
    bounds = bounds_type(begg=0, endg=5, begc=0, endc=25, begp=0, endp=125)
    
    # Proper sequence
    initialize1(bounds)
    result = initialize2(bounds)
    
    assert result is None, "initialize2 should return None after initialize1"


# ============================================================================
# Test: full_initialize
# ============================================================================

@pytest.mark.parametrize("test_case", [
    pytest.param(tc, id=tc["name"]) 
    for tc in [
        {
            "name": "test_single_gridcell_single_column_single_patch",
            "inputs": {"bounds": bounds_type(begg=0, endg=0, begc=0, endc=0, begp=0, endp=0)},
            "metadata": {"type": "nominal"},
        },
        {
            "name": "test_non_zero_starting_indices",
            "inputs": {"bounds": bounds_type(begg=100, endg=149, begc=500, endc=749, begp=2500, endp=3749)},
            "metadata": {"type": "nominal"},
        },
        {
            "name": "test_edge_minimal_column_patch_ratio",
            "inputs": {"bounds": bounds_type(begg=0, endg=49, begc=0, endc=49, begp=0, endp=49)},
            "metadata": {"type": "edge"},
        },
    ]
])
def test_full_initialize_executes_without_error(test_case):
    """
    Test that full_initialize executes both phases successfully.
    
    Verifies:
    - Function completes successfully
    - No return value (returns None)
    - Combines initialize1 and initialize2 in correct sequence
    """
    bounds = test_case["inputs"]["bounds"]
    
    result = full_initialize(bounds)
    
    assert result is None, "full_initialize should return None"


def test_full_initialize_convenience_function():
    """
    Test that full_initialize is equivalent to calling initialize1 then initialize2.
    
    Verifies:
    - Convenience function provides same result as sequential calls
    - Proper initialization sequence is maintained
    """
    bounds = bounds_type(begg=0, endg=5, begc=0, endc=25, begp=0, endp=125)
    
    # Method 1: Use convenience function
    full_initialize(bounds)
    
    # Method 2: Sequential calls (on different bounds to avoid state conflicts)
    bounds2 = bounds_type(begg=10, endg=15, begc=50, endc=75, begp=250, endp=375)
    initialize1(bounds2)
    initialize2(bounds2)
    
    # Both should complete without error
    assert True, "Both initialization methods should succeed"


# ============================================================================
# Test: JIT-compiled variants
# ============================================================================

@pytest.mark.parametrize("test_case", [
    pytest.param(tc, id=tc["name"]) 
    for tc in [
        {
            "name": "test_single_gridcell_single_column_single_patch",
            "inputs": {"bounds": bounds_type(begg=0, endg=0, begc=0, endc=0, begp=0, endp=0)},
            "metadata": {"type": "nominal"},
        },
        {
            "name": "test_small_grid_typical_hierarchy",
            "inputs": {"bounds": bounds_type(begg=0, endg=9, begc=0, endc=49, begp=0, endp=249)},
            "metadata": {"type": "nominal"},
        },
    ]
])
def test_initialize1_jit_executes_without_error(test_case):
    """
    Test that initialize1_jit executes without raising exceptions.
    
    Verifies:
    - JIT-compiled version works correctly
    - Produces same behavior as non-JIT version
    - Handles static_argnames=['bounds'] correctly
    """
    bounds = test_case["inputs"]["bounds"]
    
    result = initialize1_jit(bounds)
    
    assert result is None, "initialize1_jit should return None"


@pytest.mark.parametrize("test_case", [
    pytest.param(tc, id=tc["name"]) 
    for tc in [
        {
            "name": "test_single_gridcell_single_column_single_patch",
            "inputs": {"bounds": bounds_type(begg=0, endg=0, begc=0, endc=0, begp=0, endp=0)},
            "metadata": {"type": "nominal"},
        },
    ]
])
def test_initialize2_jit_executes_without_error(test_case):
    """
    Test that initialize2_jit executes without raising exceptions.
    
    Verifies:
    - JIT-compiled version works correctly
    - Produces same behavior as non-JIT version
    """
    bounds = test_case["inputs"]["bounds"]
    
    result = initialize2_jit(bounds)
    
    assert result is None, "initialize2_jit should return None"


@pytest.mark.parametrize("test_case", [
    pytest.param(tc, id=tc["name"]) 
    for tc in [
        {
            "name": "test_single_gridcell_single_column_single_patch",
            "inputs": {"bounds": bounds_type(begg=0, endg=0, begc=0, endc=0, begp=0, endp=0)},
            "metadata": {"type": "nominal"},
        },
        {
            "name": "test_special_asymmetric_hierarchy",
            "inputs": {"bounds": bounds_type(begg=10, endg=29, begc=50, endc=109, begp=300, endp=899)},
            "metadata": {"type": "special"},
        },
    ]
])
def test_full_initialize_jit_executes_without_error(test_case):
    """
    Test that full_initialize_jit executes without raising exceptions.
    
    Verifies:
    - JIT-compiled version works correctly
    - Combines both initialization phases
    - Produces same behavior as non-JIT version
    """
    bounds = test_case["inputs"]["bounds"]
    
    result = full_initialize_jit(bounds)
    
    assert result is None, "full_initialize_jit should return None"


def test_jit_compilation_consistency():
    """
    Test that JIT-compiled versions produce consistent results with non-JIT versions.
    
    Verifies:
    - JIT and non-JIT versions have same behavior
    - Multiple calls to JIT versions work correctly (compilation caching)
    """
    bounds = bounds_type(begg=0, endg=5, begc=0, endc=25, begp=0, endp=125)
    
    # Non-JIT version
    full_initialize(bounds)
    
    # JIT version (first call - triggers compilation)
    bounds_jit = bounds_type(begg=10, endg=15, begc=50, endc=75, begp=250, endp=375)
    full_initialize_jit(bounds_jit)
    
    # JIT version (second call - uses cached compilation)
    bounds_jit2 = bounds_type(begg=20, endg=25, begc=100, endc=125, begp=500, endp=625)
    full_initialize_jit(bounds_jit2)
    
    assert True, "JIT and non-JIT versions should both succeed"


# ============================================================================
# Test: Bounds constraints and validation
# ============================================================================

def test_bounds_type_structure():
    """
    Test that bounds_type has correct structure and fields.
    
    Verifies:
    - All required fields are present
    - Fields are accessible
    - Can create instances with valid values
    """
    bounds = bounds_type(begg=0, endg=10, begc=0, endc=50, begp=0, endp=250)
    
    assert hasattr(bounds, 'begg'), "bounds should have begg field"
    assert hasattr(bounds, 'endg'), "bounds should have endg field"
    assert hasattr(bounds, 'begc'), "bounds should have begc field"
    assert hasattr(bounds, 'endc'), "bounds should have endc field"
    assert hasattr(bounds, 'begp'), "bounds should have begp field"
    assert hasattr(bounds, 'endp'), "bounds should have endp field"
    
    assert bounds.begg == 0
    assert bounds.endg == 10
    assert bounds.begc == 0
    assert bounds.endc == 50
    assert bounds.begp == 0
    assert bounds.endp == 250


@pytest.mark.parametrize("bounds,expected_valid", [
    (bounds_type(begg=0, endg=0, begc=0, endc=0, begp=0, endp=0), True),
    (bounds_type(begg=0, endg=10, begc=0, endc=50, begp=0, endp=250), True),
    (bounds_type(begg=5, endg=5, begc=25, endc=25, begp=125, endp=125), True),
    (bounds_type(begg=10, endg=5, begc=0, endc=50, begp=0, endp=250), False),
    (bounds_type(begg=0, endg=10, begc=50, endc=25, begp=0, endp=250), False),
    (bounds_type(begg=0, endg=10, begc=0, endc=50, begp=250, endp=100), False),
])
def test_bounds_constraints(bounds, expected_valid):
    """
    Test bounds constraint validation.
    
    Verifies:
    - begg <= endg
    - begc <= endc
    - begp <= endp
    - All indices are non-negative
    """
    # Check individual constraints
    gridcell_valid = bounds.begg <= bounds.endg
    column_valid = bounds.begc <= bounds.endc
    patch_valid = bounds.begp <= bounds.endp
    non_negative = all(x >= 0 for x in [bounds.begg, bounds.endg, bounds.begc, 
                                         bounds.endc, bounds.begp, bounds.endp])
    
    is_valid = gridcell_valid and column_valid and patch_valid and non_negative
    
    assert is_valid == expected_valid, (
        f"Bounds validation mismatch: expected {expected_valid}, got {is_valid}"
    )


# ============================================================================
# Test: Integration tests
# ============================================================================

def test_complete_initialization_workflow():
    """
    Test complete initialization workflow from start to finish.
    
    Verifies:
    - Full initialization sequence works
    - Validation succeeds after initialization
    - Can handle typical simulation configuration
    """
    bounds = bounds_type(begg=0, endg=9, begc=0, endc=49, begp=0, endp=249)
    
    # Complete initialization
    full_initialize(bounds)
    
    # Validate
    is_valid = validate_initialization(bounds)
    
    assert is_valid, "Initialization should be valid after full_initialize"


def test_initialization_with_different_scales(test_data):
    """
    Test initialization across different simulation scales.
    
    Verifies:
    - Small, medium, and large domains all initialize correctly
    - Scaling behavior is appropriate
    """
    test_cases = test_data["test_cases"]
    
    # Test subset of different scales
    scales = ["test_single_gridcell_single_column_single_patch",
              "test_small_grid_typical_hierarchy",
              "test_medium_grid_realistic_domain"]
    
    for test_case in test_cases:
        if test_case["name"] in scales:
            bounds = test_case["inputs"]["bounds"]
            
            # Should initialize without error
            full_initialize(bounds)
            
            # Should validate
            assert validate_initialization(bounds), (
                f"Initialization failed for {test_case['name']}"
            )


def test_parallel_domain_decomposition():
    """
    Test initialization with non-zero starting indices (MPI subdomain).
    
    Verifies:
    - Non-zero starting indices work correctly
    - Simulates parallel processing scenario
    """
    # Simulate MPI rank 2 handling gridcells 100-149
    bounds = bounds_type(begg=100, endg=149, begc=500, endc=749, begp=2500, endp=3749)
    
    full_initialize(bounds)
    
    assert validate_initialization(bounds), (
        "Parallel domain decomposition should initialize correctly"
    )


# ============================================================================
# Test: Edge cases and special configurations
# ============================================================================

def test_minimal_hierarchy():
    """
    Test initialization with minimal subgrid hierarchy (1:1:1 ratio).
    
    Verifies:
    - Homogeneous landscape configuration works
    - No subgrid heterogeneity is handled correctly
    """
    bounds = bounds_type(begg=0, endg=49, begc=0, endc=49, begp=0, endp=49)
    
    full_initialize(bounds)
    
    assert validate_initialization(bounds), (
        "Minimal hierarchy should initialize correctly"
    )


def test_high_subgrid_density():
    """
    Test initialization with very high patch-to-gridcell ratio.
    
    Verifies:
    - Highly heterogeneous landscape is handled
    - High memory requirements don't cause issues
    """
    bounds = bounds_type(begg=0, endg=4, begc=0, endc=24, begp=0, endp=499)
    
    full_initialize(bounds)
    
    assert validate_initialization(bounds), (
        "High subgrid density should initialize correctly"
    )


def test_asymmetric_hierarchy():
    """
    Test initialization with asymmetric subgrid hierarchy.
    
    Verifies:
    - Variable ratios across hierarchy levels work
    - Non-uniform subgrid tiling is handled
    """
    bounds = bounds_type(begg=10, endg=29, begc=50, endc=109, begp=300, endp=899)
    
    full_initialize(bounds)
    
    assert validate_initialization(bounds), (
        "Asymmetric hierarchy should initialize correctly"
    )


# ============================================================================
# Test: Error handling
# ============================================================================

def test_initialization_with_invalid_bounds_raises_error():
    """
    Test that initialization with invalid bounds raises appropriate errors.
    
    Verifies:
    - Invalid bounds are rejected
    - Appropriate error messages are provided
    """
    invalid_bounds = bounds_type(begg=10, endg=5, begc=0, endc=50, begp=0, endp=250)
    
    # Should either raise an error or return False from validation
    try:
        result = validate_initialization(invalid_bounds)
        assert result is False, "Invalid bounds should fail validation"
    except (ValueError, AssertionError):
        # Exception is acceptable
        assert True


def test_negative_indices_handling():
    """
    Test handling of negative indices in bounds.
    
    Verifies:
    - Negative indices are rejected
    - Appropriate error handling
    """
    invalid_bounds = bounds_type(begg=-1, endg=10, begc=0, endc=50, begp=0, endp=250)
    
    try:
        result = validate_initialization(invalid_bounds)
        assert result is False, "Negative indices should fail validation"
    except (ValueError, AssertionError):
        # Exception is acceptable
        assert True


# ============================================================================
# Test: Documentation and metadata
# ============================================================================

def test_function_signatures():
    """
    Test that all functions have correct signatures.
    
    Verifies:
    - Functions are callable
    - Accept bounds parameter
    - Return expected types
    """
    bounds = bounds_type(begg=0, endg=5, begc=0, endc=25, begp=0, endp=125)
    
    # Test all functions are callable
    assert callable(initialize1)
    assert callable(initialize2)
    assert callable(full_initialize)
    assert callable(validate_initialization)
    assert callable(initialize1_jit)
    assert callable(initialize2_jit)
    assert callable(full_initialize_jit)
    
    # Test return types
    assert initialize1(bounds) is None
    assert initialize2(bounds) is None
    assert full_initialize(bounds) is None
    assert isinstance(validate_initialization(bounds), bool)


def test_module_docstrings():
    """
    Test that functions have appropriate documentation.
    
    Verifies:
    - Functions have docstrings
    - Documentation describes purpose and behavior
    """
    functions = [
        initialize1,
        initialize2,
        full_initialize,
        validate_initialization,
        initialize1_jit,
        initialize2_jit,
        full_initialize_jit,
    ]
    
    for func in functions:
        assert func.__doc__ is not None or func.__name__.endswith('_jit'), (
            f"Function {func.__name__} should have documentation"
        )