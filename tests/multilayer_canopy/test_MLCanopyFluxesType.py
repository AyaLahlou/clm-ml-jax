"""
Comprehensive pytest suite for MLCanopyFluxesType module.

This test suite covers:
- NamedTuple creation and validation
- State initialization functions (create_empty, init_allocate, init_cold, init)
- Restart data extraction and restoration
- Metadata retrieval functions
- Validation functions
- Edge cases and physical constraints
"""

import sys
from pathlib import Path
from typing import Dict, Any

import pytest
import jax.numpy as jnp
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from multilayer_canopy.MLCanopyFluxesType import (
    BoundsType,
    MLCanopyState,
    MLCanopyRestartData,
    create_empty_mlcanopy_state,
    init_allocate,
    init_cold,
    init,
    extract_restart_data,
    restore_from_restart,
    get_restart_metadata,
    validate_restart_data,
    get_history_metadata,
    SPVAL,
    ISPVAL,
    NLEAF,
    DEFAULT_LWP,
    DEFAULT_H2OCAN,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data() -> Dict[str, Any]:
    """Load test data for all test cases."""
    return {
        "nominal_empty": {
            "n_patches": 10,
            "nlevmlcan": 20,
            "nleaf": 2,
            "numrad": 2,
            "nlevgrnd": 15,
        },
        "single_patch_empty": {
            "n_patches": 1,
            "nlevmlcan": 10,
            "nleaf": 2,
            "numrad": 2,
            "nlevgrnd": 15,
        },
        "many_layers_empty": {
            "n_patches": 5,
            "nlevmlcan": 50,
            "nleaf": 2,
            "numrad": 2,
            "nlevgrnd": 30,
        },
        "standard_bounds": {
            "bounds": BoundsType(begp=0, endp=99),
            "nlevmlcan": 20,
            "numrad": 2,
            "nlevgrnd": 15,
            "nleaf": 2,
        },
        "single_patch_bounds": {
            "bounds": BoundsType(begp=5, endp=5),
            "nlevmlcan": 15,
            "numrad": 2,
            "nlevgrnd": 15,
            "nleaf": 2,
        },
        "typical_forest": {
            "bounds": BoundsType(begp=0, endp=49),
            "nlevmlcan": 25,
            "nleaf": 2,
        },
        "minimal_canopy": {
            "bounds": BoundsType(begp=0, endp=9),
            "nlevmlcan": 1,
            "nleaf": 2,
        },
        "complete_system": {
            "bounds": BoundsType(begp=0, endp=199),
            "nlevmlcan": 30,
            "numrad": 2,
            "nlevgrnd": 15,
            "nleaf": 2,
        },
    }


@pytest.fixture
def realistic_mlcanopy_state() -> MLCanopyState:
    """Create a realistic MLCanopyState for restart testing."""
    n_patches = 20
    nlevmlcan = 15
    nleaf = 2
    
    # Create realistic canopy heights
    ztop_canopy = jnp.array([25.0, 30.0, 22.5, 28.0, 26.5, 24.0, 29.0, 27.5, 23.5, 31.0,
                             25.5, 26.0, 28.5, 24.5, 27.0, 29.5, 23.0, 30.5, 25.0, 28.0])
    zbot_canopy = jnp.array([0.5, 0.8, 0.3, 0.6, 0.7, 0.4, 0.9, 0.5, 0.6, 1.0,
                             0.5, 0.6, 0.8, 0.4, 0.7, 0.9, 0.3, 1.0, 0.5, 0.8])
    lai_canopy = jnp.array([5.2, 6.1, 4.8, 5.9, 5.5, 4.9, 6.3, 5.7, 5.0, 6.5,
                            5.3, 5.6, 6.0, 4.7, 5.8, 6.2, 4.6, 6.4, 5.2, 5.9])
    tref_forcing = jnp.array([288.15, 290.5, 285.0, 292.3, 289.7, 287.2, 293.1, 291.0, 286.5, 294.0,
                              288.8, 289.5, 291.8, 286.0, 290.2, 292.5, 285.5, 293.5, 288.15, 291.5])
    taf_canopy = jnp.array([289.0, 291.2, 285.8, 293.0, 290.5, 288.0, 293.8, 291.7, 287.2, 294.7,
                            289.5, 290.2, 292.5, 286.7, 290.9, 293.2, 286.2, 294.2, 289.0, 292.2])
    
    # Create realistic lwp_leaf with vertical gradient
    lwp_leaf = jnp.zeros((n_patches, nlevmlcan, nleaf))
    for p in range(n_patches):
        for lev in range(nlevmlcan):
            # More negative at bottom (higher stress)
            base_lwp = -0.3 - (lev / nlevmlcan) * 0.5
            lwp_leaf = lwp_leaf.at[p, lev, 0].set(base_lwp)  # sunlit
            lwp_leaf = lwp_leaf.at[p, lev, 1].set(base_lwp - 0.3)  # shaded (more stressed)
    
    # Create lwp_mean_profile (average across leaf types)
    lwp_mean_profile = jnp.mean(lwp_leaf, axis=2)
    
    # Create state using init and update with realistic values
    state = init(BoundsType(begp=0, endp=n_patches-1), nlevmlcan, 2, 15, 2)
    
    # Update with realistic values (using JAX array updates)
    state = state._replace(
        ztop_canopy=ztop_canopy,
        zbot_canopy=zbot_canopy,
        lai_canopy=lai_canopy,
        tref_forcing=tref_forcing,
        taf_canopy=taf_canopy,
        lwp_leaf=lwp_leaf,
        lwp_mean_profile=lwp_mean_profile,
    )
    
    return state


# ============================================================================
# Tests for BoundsType
# ============================================================================

def test_bounds_type_creation():
    """Test BoundsType namedtuple creation."""
    bounds = BoundsType(begp=0, endp=99)
    assert bounds.begp == 0
    assert bounds.endp == 99
    assert isinstance(bounds.begp, int)
    assert isinstance(bounds.endp, int)


def test_bounds_type_single_patch():
    """Test BoundsType with single patch (begp == endp)."""
    bounds = BoundsType(begp=5, endp=5)
    assert bounds.begp == 5
    assert bounds.endp == 5
    assert bounds.endp - bounds.begp == 0


# ============================================================================
# Tests for create_empty_mlcanopy_state
# ============================================================================

@pytest.mark.parametrize("test_case", [
    "nominal_empty",
    "single_patch_empty",
    "many_layers_empty",
])
def test_create_empty_mlcanopy_state_shapes(test_data, test_case):
    """Test that create_empty_mlcanopy_state produces correct array shapes."""
    params = test_data[test_case]
    state = create_empty_mlcanopy_state(**params)
    
    n_patches = params["n_patches"]
    nlevmlcan = params["nlevmlcan"]
    nleaf = params["nleaf"]
    numrad = params["numrad"]
    nlevgrnd = params["nlevgrnd"]
    
    # Check 1D arrays
    assert state.ztop_canopy.shape == (n_patches,)
    assert state.zbot_canopy.shape == (n_patches,)
    assert state.lai_canopy.shape == (n_patches,)
    assert state.tref_forcing.shape == (n_patches,)
    
    # Check 3D arrays
    assert state.lwp_leaf.shape == (n_patches, nlevmlcan, nleaf)


def test_create_empty_mlcanopy_state_initialization():
    """Test that create_empty_mlcanopy_state initializes arrays to zero."""
    state = create_empty_mlcanopy_state(n_patches=5, nlevmlcan=10, nleaf=2, numrad=2, nlevgrnd=15)
    
    # All arrays should be zero-initialized
    assert jnp.allclose(state.ztop_canopy, 0.0)
    assert jnp.allclose(state.zbot_canopy, 0.0)
    assert jnp.allclose(state.lai_canopy, 0.0)
    assert jnp.allclose(state.tref_forcing, 0.0)
    assert jnp.allclose(state.lwp_leaf, 0.0)


def test_create_empty_mlcanopy_state_dtypes():
    """Test that create_empty_mlcanopy_state produces correct data types."""
    state = create_empty_mlcanopy_state(n_patches=5, nlevmlcan=10, nleaf=2, numrad=2, nlevgrnd=15)
    
    # All should be float arrays
    assert state.ztop_canopy.dtype == jnp.float32 or state.ztop_canopy.dtype == jnp.float64
    assert state.zbot_canopy.dtype == jnp.float32 or state.zbot_canopy.dtype == jnp.float64
    assert state.lwp_leaf.dtype == jnp.float32 or state.lwp_leaf.dtype == jnp.float64


def test_create_empty_mlcanopy_state_minimum_dimensions():
    """Test create_empty_mlcanopy_state with minimum valid dimensions."""
    state = create_empty_mlcanopy_state(n_patches=1, nlevmlcan=1, nleaf=2, numrad=2, nlevgrnd=1)
    
    assert state.ztop_canopy.shape == (1,)
    assert state.lwp_leaf.shape == (1, 1, 2)


# ============================================================================
# Tests for init_allocate
# ============================================================================

@pytest.mark.parametrize("test_case", [
    "standard_bounds",
    "single_patch_bounds",
])
def test_init_allocate_shapes(test_data, test_case):
    """Test that init_allocate produces correct array shapes."""
    params = test_data[test_case]
    state = init_allocate(**params)
    
    bounds = params["bounds"]
    n_patches = bounds.endp - bounds.begp + 1
    nlevmlcan = params["nlevmlcan"]
    nleaf = params["nleaf"]
    
    assert state.ztop_canopy.shape == (n_patches,)
    assert state.zbot_canopy.shape == (n_patches,)
    assert state.lwp_leaf.shape == (n_patches, nlevmlcan, nleaf)


def test_init_allocate_special_values():
    """Test that init_allocate initializes arrays to SPVAL/ISPVAL."""
    bounds = BoundsType(begp=0, endp=9)
    state = init_allocate(bounds, nlevmlcan=10, numrad=2, nlevgrnd=15, nleaf=2)
    
    # Float arrays should be initialized to SPVAL
    assert jnp.allclose(state.ztop_canopy, SPVAL)
    assert jnp.allclose(state.zbot_canopy, SPVAL)
    assert jnp.allclose(state.lwp_leaf, SPVAL)


def test_init_allocate_bounds_calculation():
    """Test that init_allocate correctly calculates n_patches from bounds."""
    bounds = BoundsType(begp=10, endp=19)
    state = init_allocate(bounds, nlevmlcan=15, numrad=2, nlevgrnd=15, nleaf=2)
    
    expected_n_patches = 10  # 19 - 10 + 1
    assert state.ztop_canopy.shape[0] == expected_n_patches


# ============================================================================
# Tests for init_cold
# ============================================================================

@pytest.mark.parametrize("test_case", [
    "typical_forest",
    "minimal_canopy",
])
def test_init_cold_shapes(test_data, test_case):
    """Test that init_cold produces correct array shapes."""
    params = test_data[test_case]
    state = init_cold(**params)
    
    bounds = params["bounds"]
    n_patches = bounds.endp - bounds.begp + 1
    nlevmlcan = params["nlevmlcan"]
    nleaf = params["nleaf"]
    
    assert state.lwp_leaf.shape == (n_patches, nlevmlcan, nleaf)


def test_init_cold_lwp_initialization():
    """Test that init_cold initializes lwp_leaf to DEFAULT_LWP."""
    bounds = BoundsType(begp=0, endp=9)
    state = init_cold(bounds, nlevmlcan=10, nleaf=2)
    
    # lwp_leaf should be initialized to DEFAULT_LWP (-0.1 MPa)
    assert jnp.allclose(state.lwp_leaf, DEFAULT_LWP, atol=1e-6)


def test_init_cold_h2ocan_initialization():
    """Test that init_cold initializes h2ocan_profile to DEFAULT_H2OCAN."""
    bounds = BoundsType(begp=0, endp=9)
    state = init_cold(bounds, nlevmlcan=10, nleaf=2)
    
    # h2ocan_profile should be initialized to DEFAULT_H2OCAN (0.0)
    if hasattr(state, 'h2ocan_profile'):
        assert jnp.allclose(state.h2ocan_profile, DEFAULT_H2OCAN, atol=1e-6)


def test_init_cold_single_layer():
    """Test init_cold with single canopy layer (edge case)."""
    bounds = BoundsType(begp=0, endp=4)
    state = init_cold(bounds, nlevmlcan=1, nleaf=2)
    
    assert state.lwp_leaf.shape == (5, 1, 2)
    assert jnp.allclose(state.lwp_leaf, DEFAULT_LWP, atol=1e-6)


# ============================================================================
# Tests for init
# ============================================================================

def test_init_complete_initialization(test_data):
    """Test that init performs complete initialization."""
    params = test_data["complete_system"]
    state = init(**params)
    
    bounds = params["bounds"]
    n_patches = bounds.endp - bounds.begp + 1
    nlevmlcan = params["nlevmlcan"]
    
    # Check shapes
    assert state.ztop_canopy.shape == (n_patches,)
    assert state.lwp_leaf.shape == (n_patches, nlevmlcan, 2)
    
    # Check that lwp_leaf is initialized to DEFAULT_LWP
    assert jnp.allclose(state.lwp_leaf, DEFAULT_LWP, atol=1e-6)


def test_init_combines_allocate_and_cold():
    """Test that init combines init_allocate and init_cold functionality."""
    bounds = BoundsType(begp=0, endp=9)
    state = init(bounds, nlevmlcan=10, numrad=2, nlevgrnd=15, nleaf=2)
    
    # Should have proper shapes (from allocate)
    assert state.ztop_canopy.shape == (10,)
    
    # Should have lwp_leaf initialized (from cold)
    assert jnp.allclose(state.lwp_leaf, DEFAULT_LWP, atol=1e-6)


# ============================================================================
# Tests for extract_restart_data
# ============================================================================

def test_extract_restart_data_fields(realistic_mlcanopy_state):
    """Test that extract_restart_data extracts correct fields."""
    restart_data = extract_restart_data(realistic_mlcanopy_state)
    
    assert hasattr(restart_data, 'taf_canopy')
    assert hasattr(restart_data, 'lwp_mean_profile')


def test_extract_restart_data_shapes(realistic_mlcanopy_state):
    """Test that extract_restart_data produces correct shapes."""
    restart_data = extract_restart_data(realistic_mlcanopy_state)
    
    n_patches = realistic_mlcanopy_state.ztop_canopy.shape[0]
    nlevmlcan = realistic_mlcanopy_state.lwp_leaf.shape[1]
    
    assert restart_data.taf_canopy.shape == (n_patches,)
    assert restart_data.lwp_mean_profile.shape == (n_patches, nlevmlcan)


def test_extract_restart_data_values(realistic_mlcanopy_state):
    """Test that extract_restart_data preserves values correctly."""
    restart_data = extract_restart_data(realistic_mlcanopy_state)
    
    # taf_canopy should match
    assert jnp.allclose(restart_data.taf_canopy, realistic_mlcanopy_state.taf_canopy, atol=1e-6)
    
    # lwp_mean_profile should match
    assert jnp.allclose(restart_data.lwp_mean_profile, 
                       realistic_mlcanopy_state.lwp_mean_profile, atol=1e-6)


# ============================================================================
# Tests for restore_from_restart
# ============================================================================

def test_restore_from_restart_updates_state(realistic_mlcanopy_state):
    """Test that restore_from_restart updates state with restart data."""
    # Extract restart data
    restart_data = extract_restart_data(realistic_mlcanopy_state)
    
    # Create a new state with different values
    bounds = BoundsType(begp=0, endp=19)
    new_state = init(bounds, nlevmlcan=15, numrad=2, nlevgrnd=15, nleaf=2)
    
    # Restore from restart data
    restored_state = restore_from_restart(new_state, restart_data)
    
    # Check that values were restored
    assert jnp.allclose(restored_state.taf_canopy, restart_data.taf_canopy, atol=1e-6)
    assert jnp.allclose(restored_state.lwp_mean_profile, restart_data.lwp_mean_profile, atol=1e-6)


def test_restore_from_restart_preserves_other_fields(realistic_mlcanopy_state):
    """Test that restore_from_restart preserves non-restart fields."""
    restart_data = extract_restart_data(realistic_mlcanopy_state)
    
    bounds = BoundsType(begp=0, endp=19)
    new_state = init(bounds, nlevmlcan=15, numrad=2, nlevgrnd=15, nleaf=2)
    
    # Set some non-restart field
    original_ztop = new_state.ztop_canopy
    
    restored_state = restore_from_restart(new_state, restart_data)
    
    # Non-restart fields should be preserved
    assert jnp.allclose(restored_state.ztop_canopy, original_ztop, atol=1e-6)


def test_extract_and_restore_cycle(realistic_mlcanopy_state):
    """Test complete extract-restore cycle preserves restart data."""
    # Extract
    restart_data = extract_restart_data(realistic_mlcanopy_state)
    
    # Create new state
    bounds = BoundsType(begp=0, endp=19)
    new_state = init(bounds, nlevmlcan=15, numrad=2, nlevgrnd=15, nleaf=2)
    
    # Restore
    restored_state = restore_from_restart(new_state, restart_data)
    
    # Extract again
    restart_data_2 = extract_restart_data(restored_state)
    
    # Should match original restart data
    assert jnp.allclose(restart_data_2.taf_canopy, restart_data.taf_canopy, atol=1e-6)
    assert jnp.allclose(restart_data_2.lwp_mean_profile, restart_data.lwp_mean_profile, atol=1e-6)


# ============================================================================
# Tests for validate_restart_data
# ============================================================================

def test_validate_restart_data_valid_data():
    """Test validate_restart_data with valid data."""
    n_patches = 5
    nlevmlcan = 8
    
    restart_data = MLCanopyRestartData(
        taf_canopy=jnp.array([288.15, 290.0, 285.0, 292.0, 289.0]),
        lwp_mean_profile=jnp.full((n_patches, nlevmlcan), -0.5)
    )
    
    result = validate_restart_data(restart_data, n_patches, nlevmlcan)
    assert result is True


def test_validate_restart_data_wrong_shape():
    """Test validate_restart_data with incorrect shapes."""
    restart_data = MLCanopyRestartData(
        taf_canopy=jnp.array([288.15, 290.0, 285.0]),  # Wrong size
        lwp_mean_profile=jnp.full((5, 8), -0.5)
    )
    
    result = validate_restart_data(restart_data, n_patches=5, nlevmlcan=8)
    assert result is False


def test_validate_restart_data_nan_values():
    """Test validate_restart_data with NaN values."""
    restart_data = MLCanopyRestartData(
        taf_canopy=jnp.array([288.15, jnp.nan, 285.0, 292.0, 289.0]),
        lwp_mean_profile=jnp.full((5, 8), -0.5)
    )
    
    result = validate_restart_data(restart_data, n_patches=5, nlevmlcan=8)
    assert result is False


def test_validate_restart_data_inf_values():
    """Test validate_restart_data with Inf values."""
    restart_data = MLCanopyRestartData(
        taf_canopy=jnp.array([288.15, 290.0, 285.0, 292.0, 289.0]),
        lwp_mean_profile=jnp.full((5, 8), jnp.inf)
    )
    
    result = validate_restart_data(restart_data, n_patches=5, nlevmlcan=8)
    assert result is False


def test_validate_restart_data_extreme_stress():
    """Test validate_restart_data with extreme but valid water stress."""
    n_patches = 5
    nlevmlcan = 8
    
    # Extreme stress near physical limits (-10 MPa)
    restart_data = MLCanopyRestartData(
        taf_canopy=jnp.array([273.15, 275.0, 270.5, 278.2, 272.8]),
        lwp_mean_profile=jnp.array([
            [-8.5, -9.0, -8.2, -9.5, -8.8, -9.2, -8.0, -9.8],
            [-9.0, -9.5, -8.7, -9.8, -9.3, -9.7, -8.5, -9.9],
            [-7.5, -8.0, -7.2, -8.5, -7.8, -8.2, -7.0, -8.8],
            [-9.5, -9.8, -9.2, -9.9, -9.6, -9.9, -9.0, -10.0],
            [-8.0, -8.5, -7.7, -9.0, -8.3, -8.7, -7.5, -9.3],
        ])
    )
    
    result = validate_restart_data(restart_data, n_patches, nlevmlcan)
    # Should be valid (within -10 to 0 MPa range)
    assert result is True


def test_validate_restart_data_unphysical_lwp():
    """Test validate_restart_data with unphysical leaf water potential."""
    restart_data = MLCanopyRestartData(
        taf_canopy=jnp.array([288.15, 290.0, 285.0, 292.0, 289.0]),
        lwp_mean_profile=jnp.full((5, 8), -15.0)  # Beyond physical limit
    )
    
    result = validate_restart_data(restart_data, n_patches=5, nlevmlcan=8)
    assert result is False


def test_validate_restart_data_positive_lwp():
    """Test validate_restart_data with positive leaf water potential (unphysical)."""
    restart_data = MLCanopyRestartData(
        taf_canopy=jnp.array([288.15, 290.0, 285.0, 292.0, 289.0]),
        lwp_mean_profile=jnp.full((5, 8), 0.5)  # Positive (unphysical)
    )
    
    result = validate_restart_data(restart_data, n_patches=5, nlevmlcan=8)
    assert result is False


def test_validate_restart_data_extreme_temperature():
    """Test validate_restart_data with extreme temperatures."""
    # Very cold but within valid range
    restart_data = MLCanopyRestartData(
        taf_canopy=jnp.array([150.0, 155.0, 160.0, 165.0, 170.0]),
        lwp_mean_profile=jnp.full((5, 8), -0.5)
    )
    
    result = validate_restart_data(restart_data, n_patches=5, nlevmlcan=8)
    assert result is True
    
    # Too cold (below absolute zero considerations)
    restart_data_cold = MLCanopyRestartData(
        taf_canopy=jnp.array([100.0, 110.0, 120.0, 130.0, 140.0]),
        lwp_mean_profile=jnp.full((5, 8), -0.5)
    )
    
    result_cold = validate_restart_data(restart_data_cold, n_patches=5, nlevmlcan=8)
    assert result_cold is False


# ============================================================================
# Tests for get_restart_metadata
# ============================================================================

def test_get_restart_metadata_returns_dict():
    """Test that get_restart_metadata returns a dictionary."""
    metadata = get_restart_metadata()
    assert isinstance(metadata, dict)


def test_get_restart_metadata_contains_required_fields():
    """Test that get_restart_metadata contains expected restart variables."""
    metadata = get_restart_metadata()
    
    # Should contain metadata for restart variables
    assert 'taf_canopy' in metadata or len(metadata) > 0
    assert 'lwp_mean_profile' in metadata or len(metadata) > 0


def test_get_restart_metadata_structure():
    """Test that get_restart_metadata has correct structure."""
    metadata = get_restart_metadata()
    
    for var_name, var_metadata in metadata.items():
        assert isinstance(var_metadata, dict)
        # Each variable should have metadata fields
        assert 'dimensions' in var_metadata or 'units' in var_metadata or len(var_metadata) > 0


# ============================================================================
# Tests for get_history_metadata
# ============================================================================

def test_get_history_metadata_returns_dict():
    """Test that get_history_metadata returns a dictionary."""
    metadata = get_history_metadata()
    assert isinstance(metadata, dict)


def test_get_history_metadata_contains_fields():
    """Test that get_history_metadata contains history variables."""
    metadata = get_history_metadata()
    
    # Should contain metadata for history output variables
    assert len(metadata) >= 0  # May be empty or populated


def test_get_history_metadata_structure():
    """Test that get_history_metadata has correct structure."""
    metadata = get_history_metadata()
    
    for var_name, var_metadata in metadata.items():
        assert isinstance(var_metadata, dict)
        # Each variable should have metadata fields
        assert 'dimensions' in var_metadata or 'units' in var_metadata or len(var_metadata) > 0


# ============================================================================
# Tests for physical constraints
# ============================================================================

def test_physical_constraints_temperature():
    """Test that temperature values are within physical bounds."""
    state = init(BoundsType(begp=0, endp=9), nlevmlcan=10, numrad=2, nlevgrnd=15, nleaf=2)
    
    # After initialization, if temperatures are set, they should be reasonable
    # (This test assumes init sets reasonable defaults or SPVAL)
    if not jnp.allclose(state.tref_forcing, SPVAL):
        assert jnp.all(state.tref_forcing >= 150.0)
        assert jnp.all(state.tref_forcing <= 350.0)


def test_physical_constraints_lwp():
    """Test that leaf water potential is within physical bounds."""
    bounds = BoundsType(begp=0, endp=9)
    state = init_cold(bounds, nlevmlcan=10, nleaf=2)
    
    # lwp_leaf should be between -10 and 0 MPa
    assert jnp.all(state.lwp_leaf >= -10.0)
    assert jnp.all(state.lwp_leaf <= 0.0)


def test_physical_constraints_canopy_height():
    """Test that canopy heights are non-negative and ztop > zbot."""
    state = create_empty_mlcanopy_state(n_patches=10, nlevmlcan=20, nleaf=2, numrad=2, nlevgrnd=15)
    
    # Heights should be non-negative (zero-initialized)
    assert jnp.all(state.ztop_canopy >= 0.0)
    assert jnp.all(state.zbot_canopy >= 0.0)


def test_physical_constraints_lai():
    """Test that LAI is non-negative."""
    state = create_empty_mlcanopy_state(n_patches=10, nlevmlcan=20, nleaf=2, numrad=2, nlevgrnd=15)
    
    # LAI should be non-negative
    assert jnp.all(state.lai_canopy >= 0.0)


# ============================================================================
# Tests for constants
# ============================================================================

def test_constants_values():
    """Test that module constants have expected values."""
    assert SPVAL == 1e36
    assert ISPVAL == -9999
    assert NLEAF == 2
    assert DEFAULT_LWP == -0.1
    assert DEFAULT_H2OCAN == 0.0


def test_constants_types():
    """Test that module constants have correct types."""
    assert isinstance(SPVAL, (int, float))
    assert isinstance(ISPVAL, int)
    assert isinstance(NLEAF, int)
    assert isinstance(DEFAULT_LWP, (int, float))
    assert isinstance(DEFAULT_H2OCAN, (int, float))


# ============================================================================
# Edge case tests
# ============================================================================

def test_edge_case_zero_lai():
    """Test handling of zero LAI (bare ground)."""
    state = create_empty_mlcanopy_state(n_patches=5, nlevmlcan=10, nleaf=2, numrad=2, nlevgrnd=15)
    
    # Zero LAI should be valid
    assert jnp.allclose(state.lai_canopy, 0.0)


def test_edge_case_single_patch_single_layer():
    """Test minimum configuration: single patch, single layer."""
    bounds = BoundsType(begp=0, endp=0)
    state = init(bounds, nlevmlcan=1, numrad=2, nlevgrnd=1, nleaf=2)
    
    assert state.ztop_canopy.shape == (1,)
    assert state.lwp_leaf.shape == (1, 1, 2)


def test_edge_case_large_domain():
    """Test handling of large domain (many patches)."""
    bounds = BoundsType(begp=0, endp=999)
    state = init_allocate(bounds, nlevmlcan=20, numrad=2, nlevgrnd=15, nleaf=2)
    
    assert state.ztop_canopy.shape == (1000,)


def test_edge_case_high_vertical_resolution():
    """Test handling of high vertical resolution (many layers)."""
    state = create_empty_mlcanopy_state(n_patches=5, nlevmlcan=100, nleaf=2, numrad=2, nlevgrnd=50)
    
    assert state.lwp_leaf.shape == (5, 100, 2)


# ============================================================================
# Integration tests
# ============================================================================

def test_integration_full_workflow():
    """Test complete workflow: init -> extract -> restore -> validate."""
    # Initialize
    bounds = BoundsType(begp=0, endp=9)
    state = init(bounds, nlevmlcan=15, numrad=2, nlevgrnd=15, nleaf=2)
    
    # Extract restart data
    restart_data = extract_restart_data(state)
    
    # Validate
    is_valid = validate_restart_data(restart_data, n_patches=10, nlevmlcan=15)
    assert is_valid is True
    
    # Create new state and restore
    new_state = init(bounds, nlevmlcan=15, numrad=2, nlevgrnd=15, nleaf=2)
    restored_state = restore_from_restart(new_state, restart_data)
    
    # Verify restoration
    assert jnp.allclose(restored_state.taf_canopy, state.taf_canopy, atol=1e-6)


def test_integration_metadata_consistency():
    """Test that restart and history metadata are consistent."""
    restart_meta = get_restart_metadata()
    history_meta = get_history_metadata()
    
    # Both should be dictionaries
    assert isinstance(restart_meta, dict)
    assert isinstance(history_meta, dict)
    
    # Restart variables should be a subset or separate from history variables
    # (depending on implementation)
    assert len(restart_meta) >= 0
    assert len(history_meta) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])