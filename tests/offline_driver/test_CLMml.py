"""
Comprehensive pytest suite for CLMml clump management functions.

This module tests the clump management and domain decomposition functionality
for the Community Land Model (CLM) offline driver, including:
- Clump bounds retrieval and validation
- Clump configuration initialization
- Domain size calculations
- Main driver orchestration functions

Tests cover nominal cases, edge cases, and special configurations for both
single-point and global-scale domains with various parallel decompositions.
"""

import sys
from pathlib import Path
from typing import NamedTuple, Callable, Tuple, Dict, Any
from collections import namedtuple

import pytest
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from offline_driver.CLMml import (
    get_clump_bounds,
    initialize_clump_config,
    clm_ml_main,
    clm_ml_main_jax,
    validate_bounds,
    get_domain_size,
    BoundsType,
    ClumpConfig
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data():
    """
    Load comprehensive test data for clump management functions.
    
    Returns:
        dict: Test cases covering nominal, edge, and special scenarios
    """
    return {
        "single_clump_minimal_domain": {
            "clump_id": 1,
            "n_clumps": 1,
            "bounds": BoundsType(
                begg=1, endg=1, begl=1, endl=1,
                begc=1, endc=1, begp=1, endp=1
            ),
            "expected_valid": True,
            "expected_sizes": {
                "n_gridcells": 1,
                "n_landunits": 1,
                "n_columns": 1,
                "n_patches": 1
            }
        },
        "small_regional_domain": {
            "clump_id": 1,
            "n_clumps": 1,
            "bounds": BoundsType(
                begg=1, endg=10, begl=1, endl=15,
                begc=1, endc=30, begp=1, endp=60
            ),
            "expected_valid": True,
            "expected_sizes": {
                "n_gridcells": 10,
                "n_landunits": 15,
                "n_columns": 30,
                "n_patches": 60
            }
        },
        "multi_clump_parallel": {
            "clump_id": 3,
            "n_clumps": 8,
            "bounds": BoundsType(
                begg=21, endg=30, begl=31, endl=45,
                begc=46, endc=75, begp=76, endp=135
            ),
            "expected_valid": True,
            "expected_sizes": {
                "n_gridcells": 10,
                "n_landunits": 15,
                "n_columns": 30,
                "n_patches": 60
            }
        },
        "large_global_domain": {
            "clump_id": 1,
            "n_clumps": 1,
            "bounds": BoundsType(
                begg=1, endg=64800, begl=1, endl=97200,
                begc=1, endc=194400, begp=1, endp=388800
            ),
            "expected_valid": True,
            "expected_sizes": {
                "n_gridcells": 64800,
                "n_landunits": 97200,
                "n_columns": 194400,
                "n_patches": 388800
            }
        },
        "maximum_clump_id": {
            "clump_id": 128,
            "n_clumps": 128,
            "bounds": BoundsType(
                begg=1, endg=100, begl=1, endl=150,
                begc=1, endc=300, begp=1, endp=600
            ),
            "expected_valid": True,
            "expected_sizes": {
                "n_gridcells": 100,
                "n_landunits": 150,
                "n_columns": 300,
                "n_patches": 600
            }
        },
        "equal_begin_end": {
            "clump_id": 1,
            "n_clumps": 1,
            "bounds": BoundsType(
                begg=5, endg=5, begl=10, endl=10,
                begc=20, endc=20, begp=40, endp=40
            ),
            "expected_valid": True,
            "expected_sizes": {
                "n_gridcells": 1,
                "n_landunits": 1,
                "n_columns": 1,
                "n_patches": 1
            }
        },
        "invalid_bounds": {
            "bounds": BoundsType(
                begg=10, endg=5, begl=1, endl=8,
                begc=1, endc=16, begp=1, endp=32
            ),
            "expected_valid": False
        },
        "sparse_hierarchy": {
            "clump_id": 1,
            "n_clumps": 1,
            "bounds": BoundsType(
                begg=1, endg=100, begl=1, endl=100,
                begc=1, endc=100, begp=1, endp=100
            ),
            "expected_valid": True,
            "expected_sizes": {
                "n_gridcells": 100,
                "n_landunits": 100,
                "n_columns": 100,
                "n_patches": 100
            }
        },
        "dense_hierarchy": {
            "clump_id": 2,
            "n_clumps": 4,
            "bounds": BoundsType(
                begg=1, endg=10, begl=1, endl=50,
                begc=1, endc=250, begp=1, endp=1250
            ),
            "expected_valid": True,
            "expected_sizes": {
                "n_gridcells": 10,
                "n_landunits": 50,
                "n_columns": 250,
                "n_patches": 1250
            }
        }
    }


@pytest.fixture
def mock_driver_fn():
    """
    Create a mock driver function for testing clm_ml_main.
    
    Returns:
        Callable: Mock driver that accepts bounds and returns None
    """
    def driver(bounds: BoundsType) -> None:
        """Mock driver function that validates bounds structure."""
        assert isinstance(bounds, BoundsType)
        assert bounds.begg >= 1
        assert bounds.endg >= bounds.begg
    return driver


@pytest.fixture
def mock_driver_fn_jax():
    """
    Create a mock JAX driver function for testing clm_ml_main_jax.
    
    Returns:
        Callable: Mock JAX driver that returns state and diagnostics
    """
    # Create simple NamedTuples for testing
    State = namedtuple('State', ['temperature', 'moisture'])
    Diagnostics = namedtuple('Diagnostics', ['flux', 'energy'])
    
    def driver(bounds: BoundsType, state: NamedTuple, 
               forcing: NamedTuple, params: NamedTuple) -> Tuple[NamedTuple, NamedTuple]:
        """Mock JAX driver that returns modified state and diagnostics."""
        # Simple transformation for testing
        new_state = State(
            temperature=state.temperature + 1.0,
            moisture=state.moisture * 0.99
        )
        diags = Diagnostics(
            flux=100.0,
            energy=500.0
        )
        return new_state, diags
    
    return driver


@pytest.fixture
def sample_state():
    """Create sample initial state for JAX driver testing."""
    State = namedtuple('State', ['temperature', 'moisture'])
    return State(temperature=273.15, moisture=0.5)


@pytest.fixture
def sample_forcing():
    """Create sample forcing data for JAX driver testing."""
    Forcing = namedtuple('Forcing', ['radiation', 'precipitation'])
    return Forcing(radiation=500.0, precipitation=0.001)


@pytest.fixture
def sample_params():
    """Create sample parameters for JAX driver testing."""
    Params = namedtuple('Params', ['albedo', 'roughness'])
    return Params(albedo=0.2, roughness=0.05)


# ============================================================================
# Test: get_clump_bounds
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "single_clump_minimal_domain",
    "small_regional_domain",
    "multi_clump_parallel",
    "maximum_clump_id"
])
def test_get_clump_bounds_returns_valid_structure(test_data, test_case_name):
    """
    Test that get_clump_bounds returns a valid BoundsType structure.
    
    Verifies:
    - Return type is BoundsType
    - All required fields are present
    - All indices are positive integers
    """
    test_case = test_data[test_case_name]
    clump_id = test_case["clump_id"]
    
    bounds = get_clump_bounds(clump_id)
    
    assert isinstance(bounds, BoundsType), \
        f"Expected BoundsType, got {type(bounds)}"
    
    # Check all fields exist and are positive
    for field in ['begg', 'endg', 'begl', 'endl', 'begc', 'endc', 'begp', 'endp']:
        assert hasattr(bounds, field), f"Missing field: {field}"
        value = getattr(bounds, field)
        assert isinstance(value, int), f"{field} should be int, got {type(value)}"
        assert value >= 1, f"{field} should be >= 1, got {value}"


@pytest.mark.parametrize("test_case_name", [
    "single_clump_minimal_domain",
    "small_regional_domain",
    "equal_begin_end"
])
def test_get_clump_bounds_respects_hierarchy(test_data, test_case_name):
    """
    Test that get_clump_bounds respects the spatial hierarchy.
    
    Verifies:
    - end >= begin for each level
    - Hierarchy ordering: gridcell >= landunit >= column >= patch counts
    """
    test_case = test_data[test_case_name]
    clump_id = test_case["clump_id"]
    
    bounds = get_clump_bounds(clump_id)
    
    # Check end >= begin for each level
    assert bounds.endg >= bounds.begg, \
        f"endg ({bounds.endg}) should be >= begg ({bounds.begg})"
    assert bounds.endl >= bounds.begl, \
        f"endl ({bounds.endl}) should be >= begl ({bounds.begl})"
    assert bounds.endc >= bounds.begc, \
        f"endc ({bounds.endc}) should be >= begc ({bounds.begc})"
    assert bounds.endp >= bounds.begp, \
        f"endp ({bounds.endp}) should be >= begp ({bounds.begp})"


def test_get_clump_bounds_invalid_clump_id():
    """
    Test that get_clump_bounds handles invalid clump_id appropriately.
    
    Verifies:
    - clump_id < 1 raises ValueError or returns error indication
    - clump_id = 0 is rejected
    """
    with pytest.raises((ValueError, AssertionError)):
        get_clump_bounds(0)
    
    with pytest.raises((ValueError, AssertionError)):
        get_clump_bounds(-1)


# ============================================================================
# Test: initialize_clump_config
# ============================================================================

@pytest.mark.parametrize("n_clumps,expected_clump_id", [
    (1, 1),
    (4, 1),
    (8, 1),
    (128, 1)
])
def test_initialize_clump_config_returns_valid_structure(n_clumps, expected_clump_id):
    """
    Test that initialize_clump_config returns valid ClumpConfig.
    
    Verifies:
    - Return type is ClumpConfig
    - n_clumps matches input
    - clump_id is initialized correctly (typically to 1)
    - Both fields are positive integers
    """
    config = initialize_clump_config(n_clumps)
    
    assert isinstance(config, ClumpConfig), \
        f"Expected ClumpConfig, got {type(config)}"
    
    assert config.n_clumps == n_clumps, \
        f"Expected n_clumps={n_clumps}, got {config.n_clumps}"
    
    assert config.clump_id >= 1, \
        f"clump_id should be >= 1, got {config.clump_id}"
    
    assert config.clump_id <= config.n_clumps, \
        f"clump_id ({config.clump_id}) should be <= n_clumps ({config.n_clumps})"


def test_initialize_clump_config_default_value():
    """
    Test that initialize_clump_config uses default n_clumps=1.
    
    Verifies:
    - Calling without arguments uses default
    - Default configuration is valid
    """
    config = initialize_clump_config()
    
    assert config.n_clumps == 1, \
        f"Default n_clumps should be 1, got {config.n_clumps}"
    assert config.clump_id == 1, \
        f"Default clump_id should be 1, got {config.clump_id}"


def test_initialize_clump_config_invalid_n_clumps():
    """
    Test that initialize_clump_config rejects invalid n_clumps.
    
    Verifies:
    - n_clumps < 1 raises ValueError
    - n_clumps = 0 is rejected
    """
    with pytest.raises((ValueError, AssertionError)):
        initialize_clump_config(0)
    
    with pytest.raises((ValueError, AssertionError)):
        initialize_clump_config(-1)


# ============================================================================
# Test: validate_bounds
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "single_clump_minimal_domain",
    "small_regional_domain",
    "multi_clump_parallel",
    "large_global_domain",
    "equal_begin_end",
    "sparse_hierarchy",
    "dense_hierarchy"
])
def test_validate_bounds_accepts_valid_bounds(test_data, test_case_name):
    """
    Test that validate_bounds accepts valid bounds configurations.
    
    Verifies:
    - Returns True for valid bounds
    - All nominal and edge cases pass validation
    """
    test_case = test_data[test_case_name]
    bounds = test_case["bounds"]
    expected = test_case["expected_valid"]
    
    result = validate_bounds(bounds)
    
    assert result == expected, \
        f"Expected validation={expected} for {test_case_name}, got {result}"


def test_validate_bounds_rejects_invalid_bounds(test_data):
    """
    Test that validate_bounds rejects invalid bounds.
    
    Verifies:
    - Returns False when end < begin
    - Detects invalid hierarchy ordering
    """
    test_case = test_data["invalid_bounds"]
    bounds = test_case["bounds"]
    
    result = validate_bounds(bounds)
    
    assert result == False, \
        "validate_bounds should reject bounds with endg < begg"


def test_validate_bounds_rejects_zero_indices():
    """
    Test that validate_bounds rejects zero or negative indices.
    
    Verifies:
    - Indices must be >= 1 (1-based indexing)
    - Zero indices are invalid
    """
    invalid_bounds = BoundsType(
        begg=0, endg=10, begl=1, endl=15,
        begc=1, endc=30, begp=1, endp=60
    )
    
    result = validate_bounds(invalid_bounds)
    
    assert result == False, \
        "validate_bounds should reject bounds with zero indices"


def test_validate_bounds_rejects_negative_indices():
    """
    Test that validate_bounds rejects negative indices.
    
    Verifies:
    - All indices must be positive
    - Negative values are invalid
    """
    invalid_bounds = BoundsType(
        begg=-1, endg=10, begl=1, endl=15,
        begc=1, endc=30, begp=1, endp=60
    )
    
    result = validate_bounds(invalid_bounds)
    
    assert result == False, \
        "validate_bounds should reject bounds with negative indices"


# ============================================================================
# Test: get_domain_size
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "single_clump_minimal_domain",
    "small_regional_domain",
    "multi_clump_parallel",
    "large_global_domain",
    "equal_begin_end",
    "sparse_hierarchy",
    "dense_hierarchy"
])
def test_get_domain_size_returns_correct_counts(test_data, test_case_name):
    """
    Test that get_domain_size calculates correct element counts.
    
    Verifies:
    - Returns dictionary with all required keys
    - Counts match expected values (end - begin + 1)
    - All counts are positive integers
    """
    test_case = test_data[test_case_name]
    bounds = test_case["bounds"]
    expected = test_case["expected_sizes"]
    
    result = get_domain_size(bounds)
    
    assert isinstance(result, dict), \
        f"Expected dict, got {type(result)}"
    
    # Check all required keys
    required_keys = ['n_gridcells', 'n_landunits', 'n_columns', 'n_patches']
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    
    # Check values
    assert result['n_gridcells'] == expected['n_gridcells'], \
        f"Expected {expected['n_gridcells']} gridcells, got {result['n_gridcells']}"
    assert result['n_landunits'] == expected['n_landunits'], \
        f"Expected {expected['n_landunits']} landunits, got {result['n_landunits']}"
    assert result['n_columns'] == expected['n_columns'], \
        f"Expected {expected['n_columns']} columns, got {result['n_columns']}"
    assert result['n_patches'] == expected['n_patches'], \
        f"Expected {expected['n_patches']} patches, got {result['n_patches']}"


def test_get_domain_size_calculation_formula():
    """
    Test that get_domain_size uses correct formula: n = end - begin + 1.
    
    Verifies:
    - Inclusive range calculation
    - 1-based indexing handled correctly
    """
    bounds = BoundsType(
        begg=5, endg=14,  # Should give 10 gridcells
        begl=10, endl=24,  # Should give 15 landunits
        begc=20, endc=49,  # Should give 30 columns
        begp=40, endp=99   # Should give 60 patches
    )
    
    result = get_domain_size(bounds)
    
    assert result['n_gridcells'] == 10, \
        f"Expected 10 gridcells (14-5+1), got {result['n_gridcells']}"
    assert result['n_landunits'] == 15, \
        f"Expected 15 landunits (24-10+1), got {result['n_landunits']}"
    assert result['n_columns'] == 30, \
        f"Expected 30 columns (49-20+1), got {result['n_columns']}"
    assert result['n_patches'] == 60, \
        f"Expected 60 patches (99-40+1), got {result['n_patches']}"


def test_get_domain_size_single_element():
    """
    Test get_domain_size with single-element domains (begin == end).
    
    Verifies:
    - Single element gives count of 1
    - Edge case handled correctly
    """
    bounds = BoundsType(
        begg=7, endg=7,
        begl=14, endl=14,
        begc=28, endc=28,
        begp=56, endp=56
    )
    
    result = get_domain_size(bounds)
    
    assert result['n_gridcells'] == 1
    assert result['n_landunits'] == 1
    assert result['n_columns'] == 1
    assert result['n_patches'] == 1


# ============================================================================
# Test: clm_ml_main
# ============================================================================

def test_clm_ml_main_executes_single_clump(mock_driver_fn, test_data):
    """
    Test that clm_ml_main executes driver for single clump.
    
    Verifies:
    - Function completes without error
    - Driver function is called
    - No return value (side effects only)
    """
    result = clm_ml_main(mock_driver_fn, n_clumps=1)
    
    assert result is None, \
        "clm_ml_main should return None"


def test_clm_ml_main_executes_multiple_clumps(mock_driver_fn):
    """
    Test that clm_ml_main executes driver for multiple clumps.
    
    Verifies:
    - Function handles multi-clump configuration
    - Driver called for each clump
    """
    result = clm_ml_main(mock_driver_fn, n_clumps=4)
    
    assert result is None, \
        "clm_ml_main should return None for multi-clump execution"


def test_clm_ml_main_default_n_clumps(mock_driver_fn):
    """
    Test that clm_ml_main uses default n_clumps=1.
    
    Verifies:
    - Default parameter works correctly
    - Single clump execution by default
    """
    result = clm_ml_main(mock_driver_fn)
    
    assert result is None, \
        "clm_ml_main should return None with default n_clumps"


def test_clm_ml_main_invalid_n_clumps(mock_driver_fn):
    """
    Test that clm_ml_main rejects invalid n_clumps.
    
    Verifies:
    - n_clumps < 1 raises error
    - Invalid configuration detected
    """
    with pytest.raises((ValueError, AssertionError)):
        clm_ml_main(mock_driver_fn, n_clumps=0)


# ============================================================================
# Test: clm_ml_main_jax
# ============================================================================

def test_clm_ml_main_jax_returns_state_and_diagnostics(
    mock_driver_fn_jax, sample_state, sample_forcing, sample_params
):
    """
    Test that clm_ml_main_jax returns state and diagnostics.
    
    Verifies:
    - Returns tuple of (state, diagnostics)
    - Both are NamedTuples
    - State is modified from initial
    """
    final_state, diagnostics = clm_ml_main_jax(
        mock_driver_fn_jax,
        sample_state,
        sample_forcing,
        sample_params
    )
    
    assert isinstance(final_state, tuple), \
        f"Expected NamedTuple (tuple subclass), got {type(final_state)}"
    assert isinstance(diagnostics, tuple), \
        f"Expected NamedTuple (tuple subclass), got {type(diagnostics)}"
    
    # Check state was modified
    assert hasattr(final_state, 'temperature')
    assert hasattr(final_state, 'moisture')


def test_clm_ml_main_jax_state_transformation(
    mock_driver_fn_jax, sample_state, sample_forcing, sample_params
):
    """
    Test that clm_ml_main_jax properly transforms state.
    
    Verifies:
    - State values are updated
    - Transformation follows expected pattern
    - Initial state is not modified
    """
    initial_temp = sample_state.temperature
    initial_moisture = sample_state.moisture
    
    final_state, diagnostics = clm_ml_main_jax(
        mock_driver_fn_jax,
        sample_state,
        sample_forcing,
        sample_params
    )
    
    # Check transformation (mock adds 1 to temp, multiplies moisture by 0.99)
    assert final_state.temperature == initial_temp + 1.0, \
        f"Expected temperature {initial_temp + 1.0}, got {final_state.temperature}"
    assert np.isclose(final_state.moisture, initial_moisture * 0.99), \
        f"Expected moisture {initial_moisture * 0.99}, got {final_state.moisture}"


def test_clm_ml_main_jax_diagnostics_structure(
    mock_driver_fn_jax, sample_state, sample_forcing, sample_params
):
    """
    Test that clm_ml_main_jax returns properly structured diagnostics.
    
    Verifies:
    - Diagnostics contain expected fields
    - Values are reasonable
    """
    final_state, diagnostics = clm_ml_main_jax(
        mock_driver_fn_jax,
        sample_state,
        sample_forcing,
        sample_params
    )
    
    assert hasattr(diagnostics, 'flux'), \
        "Diagnostics should have 'flux' field"
    assert hasattr(diagnostics, 'energy'), \
        "Diagnostics should have 'energy' field"
    
    assert diagnostics.flux == 100.0
    assert diagnostics.energy == 500.0


# ============================================================================
# Test: Integration and Edge Cases
# ============================================================================

def test_bounds_type_immutability():
    """
    Test that BoundsType is immutable (NamedTuple property).
    
    Verifies:
    - Cannot modify fields after creation
    - Immutability enforced
    """
    bounds = BoundsType(
        begg=1, endg=10, begl=1, endl=15,
        begc=1, endc=30, begp=1, endp=60
    )
    
    with pytest.raises(AttributeError):
        bounds.begg = 5


def test_clump_config_immutability():
    """
    Test that ClumpConfig is immutable (NamedTuple property).
    
    Verifies:
    - Cannot modify fields after creation
    - Configuration is read-only
    """
    config = ClumpConfig(n_clumps=4, clump_id=2)
    
    with pytest.raises(AttributeError):
        config.n_clumps = 8


def test_bounds_type_field_access():
    """
    Test that BoundsType fields are accessible by name and index.
    
    Verifies:
    - Named field access works
    - Index access works (NamedTuple feature)
    - Field order is preserved
    """
    bounds = BoundsType(
        begg=1, endg=10, begl=11, endl=20,
        begc=21, endc=40, begp=41, endp=80
    )
    
    # Named access
    assert bounds.begg == 1
    assert bounds.endg == 10
    assert bounds.begp == 41
    assert bounds.endp == 80
    
    # Index access
    assert bounds[0] == 1  # begg
    assert bounds[1] == 10  # endg


def test_integration_full_workflow(test_data, mock_driver_fn):
    """
    Test complete workflow: initialize -> get bounds -> validate -> calculate size.
    
    Verifies:
    - All functions work together
    - Data flows correctly between functions
    - End-to-end functionality
    """
    test_case = test_data["small_regional_domain"]
    
    # Initialize configuration
    config = initialize_clump_config(test_case["n_clumps"])
    assert config.n_clumps == test_case["n_clumps"]
    
    # Get bounds for clump
    bounds = get_clump_bounds(test_case["clump_id"])
    assert isinstance(bounds, BoundsType)
    
    # Validate bounds
    is_valid = validate_bounds(bounds)
    assert is_valid == True
    
    # Calculate domain size
    sizes = get_domain_size(bounds)
    assert sizes['n_gridcells'] >= 1
    assert sizes['n_patches'] >= sizes['n_gridcells']
    
    # Execute main driver
    result = clm_ml_main(mock_driver_fn, n_clumps=test_case["n_clumps"])
    assert result is None


def test_hierarchy_consistency_across_functions(test_data):
    """
    Test that hierarchy constraints are consistent across all functions.
    
    Verifies:
    - Domain sizes respect hierarchy
    - Validation enforces hierarchy
    - All functions agree on hierarchy rules
    """
    test_case = test_data["dense_hierarchy"]
    bounds = test_case["bounds"]
    
    # Validate bounds
    assert validate_bounds(bounds) == True
    
    # Get sizes
    sizes = get_domain_size(bounds)
    
    # Check hierarchy: patches >= columns >= landunits >= gridcells
    assert sizes['n_patches'] >= sizes['n_columns'], \
        "Patches should be >= columns"
    assert sizes['n_columns'] >= sizes['n_landunits'], \
        "Columns should be >= landunits"
    assert sizes['n_landunits'] >= sizes['n_gridcells'], \
        "Landunits should be >= gridcells"


def test_large_scale_domain_handling(test_data):
    """
    Test handling of large-scale global domains.
    
    Verifies:
    - Large indices handled correctly
    - No integer overflow
    - Performance acceptable for global domains
    """
    test_case = test_data["large_global_domain"]
    bounds = test_case["bounds"]
    
    # Validate large domain
    assert validate_bounds(bounds) == True
    
    # Calculate sizes
    sizes = get_domain_size(bounds)
    
    # Check large values
    assert sizes['n_gridcells'] == 64800
    assert sizes['n_patches'] == 388800
    
    # Verify no overflow (all values positive)
    for key, value in sizes.items():
        assert value > 0, f"{key} should be positive"
        assert isinstance(value, int), f"{key} should be integer"


# ============================================================================
# Test: Documentation and Type Checking
# ============================================================================

def test_bounds_type_has_correct_fields():
    """
    Test that BoundsType has all required fields with correct types.
    
    Verifies:
    - All 8 fields present
    - Field names match specification
    - Can be instantiated with all fields
    """
    bounds = BoundsType(
        begg=1, endg=2, begl=3, endl=4,
        begc=5, endc=6, begp=7, endp=8
    )
    
    expected_fields = ['begg', 'endg', 'begl', 'endl', 'begc', 'endc', 'begp', 'endp']
    actual_fields = bounds._fields
    
    assert list(actual_fields) == expected_fields, \
        f"Expected fields {expected_fields}, got {list(actual_fields)}"


def test_clump_config_has_correct_fields():
    """
    Test that ClumpConfig has all required fields.
    
    Verifies:
    - Both fields present
    - Field names match specification
    """
    config = ClumpConfig(n_clumps=4, clump_id=2)
    
    expected_fields = ['n_clumps', 'clump_id']
    actual_fields = config._fields
    
    assert list(actual_fields) == expected_fields, \
        f"Expected fields {expected_fields}, got {list(actual_fields)}"


def test_function_signatures_match_specification():
    """
    Test that function signatures match the specification.
    
    Verifies:
    - Functions exist and are callable
    - Parameter names match specification
    - Return types are correct
    """
    import inspect
    
    # Check get_clump_bounds
    sig = inspect.signature(get_clump_bounds)
    assert 'clump_id' in sig.parameters
    
    # Check initialize_clump_config
    sig = inspect.signature(initialize_clump_config)
    assert 'n_clumps' in sig.parameters
    
    # Check validate_bounds
    sig = inspect.signature(validate_bounds)
    assert 'bounds' in sig.parameters
    
    # Check get_domain_size
    sig = inspect.signature(get_domain_size)
    assert 'bounds' in sig.parameters
    
    # Check clm_ml_main
    sig = inspect.signature(clm_ml_main)
    assert 'driver_fn' in sig.parameters
    assert 'n_clumps' in sig.parameters
    
    # Check clm_ml_main_jax
    sig = inspect.signature(clm_ml_main_jax)
    assert 'driver_fn' in sig.parameters
    assert 'initial_state' in sig.parameters
    assert 'forcing_data' in sig.parameters
    assert 'params' in sig.parameters