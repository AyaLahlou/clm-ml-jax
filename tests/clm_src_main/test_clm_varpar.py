"""
Comprehensive pytest suite for clm_varpar module.

Tests CLM variable parameters including layer definitions, radiation parameters,
and plant functional type settings. Covers initialization, validation, state
management, and array generation for CLM4.5 and CLM5.0 configurations.
"""

import sys
from pathlib import Path
from typing import Dict, Any

import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_main.clm_varpar import (
    CLMParameters,
    clm_varpar_init,
    get_clm_parameters,
    set_custom_parameters,
    get_layer_info,
    get_radiation_info,
    get_pft_info,
    create_layer_arrays,
    reset_parameters,
    load_preset,
    snow_layer_indices,
    soil_layer_indices,
    ground_layer_indices,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_state():
    """Reset parameters before and after each test to ensure clean state."""
    reset_parameters()
    yield
    reset_parameters()


@pytest.fixture
def default_clm5_params():
    """Fixture providing initialized CLM5.0 parameters."""
    load_preset("CLM5_0")
    return get_clm_parameters()


@pytest.fixture
def default_clm4_5_params():
    """Fixture providing initialized CLM4.5 parameters."""
    load_preset("CLM4_5")
    return get_clm_parameters()


@pytest.fixture
def uninitialized_params():
    """Fixture providing reset/uninitialized parameters."""
    reset_parameters()
    return get_clm_parameters()


@pytest.fixture
def custom_small_config():
    """Fixture for small custom configuration."""
    set_custom_parameters(3, 5, 8)
    return get_clm_parameters()


@pytest.fixture
def test_data():
    """Load test data for parametrized tests."""
    return {
        "valid_layer_combinations": [
            {"nlevsno": 3, "nlevsoi": 5, "nlevgrnd": 8},
            {"nlevsno": 5, "nlevsoi": 10, "nlevgrnd": 15},
            {"nlevsno": 12, "nlevsoi": 20, "nlevgrnd": 25},
            {"nlevsno": 10, "nlevsoi": 30, "nlevgrnd": 50},
            {"nlevsno": 20, "nlevsoi": 50, "nlevgrnd": 75},
        ],
        "invalid_layer_combinations": [
            {"nlevsno": 0, "nlevsoi": 10, "nlevgrnd": 15, "error": "nlevsno < 1"},
            {"nlevsno": 5, "nlevsoi": 0, "nlevgrnd": 15, "error": "nlevsoi < 1"},
            {"nlevsno": 5, "nlevsoi": 10, "nlevgrnd": 5, "error": "nlevgrnd < nlevsoi"},
            {"nlevsno": 51, "nlevsoi": 10, "nlevgrnd": 15, "error": "nlevsno > 50"},
            {"nlevsno": 5, "nlevsoi": 101, "nlevgrnd": 101, "error": "nlevsoi > 100"},
            {"nlevsno": -5, "nlevsoi": 10, "nlevgrnd": 15, "error": "negative nlevsno"},
        ],
        "preset_names": [
            {"name": "CLM5_0", "valid": True},
            {"name": "CLM4_5", "valid": True},
            {"name": "CLM3_0", "valid": False, "error": "unknown preset"},
            {"name": "invalid", "valid": False, "error": "unknown preset"},
            {"name": "", "valid": False, "error": "empty string"},
        ],
    }


# ============================================================================
# CLMParameters Dataclass Tests
# ============================================================================


def test_clm_parameters_initialization():
    """Test CLMParameters dataclass initialization with default values."""
    params = CLMParameters()
    
    assert params.nlevsno == -1, "Default nlevsno should be -1"
    assert params.nlevsoi == -1, "Default nlevsoi should be -1"
    assert params.nlevgrnd == -1, "Default nlevgrnd should be -1"
    assert params.numrad == 2, "numrad should be 2"
    assert params.ivis == 1, "ivis should be 1"
    assert params.inir == 2, "inir should be 2"
    assert params.mxpft == 78, "mxpft should be 78"


def test_clm_parameters_is_initialized():
    """Test is_initialized method for various states."""
    # Uninitialized state
    params = CLMParameters()
    assert not params.is_initialized(), "Should not be initialized with default values"
    
    # Partially initialized
    params = CLMParameters(nlevsno=5, nlevsoi=-1, nlevgrnd=-1)
    assert not params.is_initialized(), "Should not be initialized with partial values"
    
    # Fully initialized
    params = CLMParameters(nlevsno=5, nlevsoi=10, nlevgrnd=15)
    assert params.is_initialized(), "Should be initialized with all positive values"


def test_clm_parameters_validate():
    """Test validate method for constraint checking."""
    # Valid configuration
    params = CLMParameters(nlevsno=5, nlevsoi=10, nlevgrnd=15)
    assert params.validate(), "Valid configuration should pass validation"
    
    # Invalid: nlevgrnd < nlevsoi
    params = CLMParameters(nlevsno=5, nlevsoi=20, nlevgrnd=15)
    assert not params.validate(), "Should fail when nlevgrnd < nlevsoi"
    
    # Invalid: out of bounds
    params = CLMParameters(nlevsno=51, nlevsoi=10, nlevgrnd=15)
    assert not params.validate(), "Should fail when nlevsno > 50"
    
    params = CLMParameters(nlevsno=5, nlevsoi=101, nlevgrnd=101)
    assert not params.validate(), "Should fail when nlevsoi > 100"


def test_clm_parameters_get_config_dict():
    """Test get_config_dict returns all parameter values."""
    params = CLMParameters(nlevsno=12, nlevsoi=20, nlevgrnd=25)
    config = params.get_config_dict()
    
    expected_keys = ["nlevsno", "nlevsoi", "nlevgrnd", "numrad", "ivis", "inir", "mxpft"]
    assert all(key in config for key in expected_keys), "Config dict should contain all parameters"
    
    assert config["nlevsno"] == 12, "nlevsno should match"
    assert config["nlevsoi"] == 20, "nlevsoi should match"
    assert config["nlevgrnd"] == 25, "nlevgrnd should match"
    assert config["numrad"] == 2, "numrad should be 2"
    assert config["ivis"] == 1, "ivis should be 1"
    assert config["inir"] == 2, "inir should be 2"
    assert config["mxpft"] == 78, "mxpft should be 78"


# ============================================================================
# Preset Loading Tests
# ============================================================================


def test_load_preset_clm5_0():
    """Test loading CLM5.0 preset configuration."""
    load_preset("CLM5_0")
    params = get_clm_parameters()
    
    assert params.nlevsno == 12, "CLM5.0 should have 12 snow layers"
    assert params.nlevsoi == 20, "CLM5.0 should have 20 soil layers"
    assert params.nlevgrnd == 25, "CLM5.0 should have 25 ground layers"
    assert params.is_initialized(), "Should be initialized after loading preset"
    assert params.validate(), "CLM5.0 preset should be valid"


def test_load_preset_clm4_5():
    """Test loading CLM4.5 preset configuration."""
    load_preset("CLM4_5")
    params = get_clm_parameters()
    
    assert params.nlevsno == 5, "CLM4.5 should have 5 snow layers"
    assert params.nlevsoi == 10, "CLM4.5 should have 10 soil layers"
    assert params.nlevgrnd == 15, "CLM4.5 should have 15 ground layers"
    assert params.is_initialized(), "Should be initialized after loading preset"
    assert params.validate(), "CLM4.5 preset should be valid"


@pytest.mark.parametrize("preset_info", [
    {"name": "CLM3_0", "valid": False},
    {"name": "invalid", "valid": False},
    {"name": "", "valid": False},
])
def test_load_preset_invalid(preset_info):
    """Test loading invalid preset names raises ValueError."""
    with pytest.raises(ValueError, match=".*preset.*"):
        load_preset(preset_info["name"])


# ============================================================================
# Custom Parameter Setting Tests
# ============================================================================


@pytest.mark.parametrize("config", [
    {"nlevsno": 3, "nlevsoi": 5, "nlevgrnd": 8},
    {"nlevsno": 5, "nlevsoi": 10, "nlevgrnd": 15},
    {"nlevsno": 12, "nlevsoi": 20, "nlevgrnd": 25},
    {"nlevsno": 10, "nlevsoi": 30, "nlevgrnd": 50},
    {"nlevsno": 20, "nlevsoi": 50, "nlevgrnd": 75},
])
def test_set_custom_parameters_valid(config):
    """Test setting valid custom parameter configurations."""
    set_custom_parameters(config["nlevsno"], config["nlevsoi"], config["nlevgrnd"])
    params = get_clm_parameters()
    
    assert params.nlevsno == config["nlevsno"], f"nlevsno should be {config['nlevsno']}"
    assert params.nlevsoi == config["nlevsoi"], f"nlevsoi should be {config['nlevsoi']}"
    assert params.nlevgrnd == config["nlevgrnd"], f"nlevgrnd should be {config['nlevgrnd']}"
    assert params.is_initialized(), "Should be initialized after setting custom parameters"
    assert params.validate(), "Custom parameters should be valid"


@pytest.mark.parametrize("config", [
    {"nlevsno": 0, "nlevsoi": 10, "nlevgrnd": 15},
    {"nlevsno": 5, "nlevsoi": 0, "nlevgrnd": 15},
    {"nlevsno": 5, "nlevsoi": 10, "nlevgrnd": 5},
    {"nlevsno": 51, "nlevsoi": 10, "nlevgrnd": 15},
    {"nlevsno": 5, "nlevsoi": 101, "nlevgrnd": 101},
    {"nlevsno": -5, "nlevsoi": 10, "nlevgrnd": 15},
])
def test_set_custom_parameters_invalid(config):
    """Test setting invalid custom parameters raises ValueError."""
    with pytest.raises(ValueError):
        set_custom_parameters(config["nlevsno"], config["nlevsoi"], config["nlevgrnd"])


def test_set_custom_parameters_minimal():
    """Test minimal valid configuration (1 layer each)."""
    set_custom_parameters(1, 1, 1)
    params = get_clm_parameters()
    
    assert params.nlevsno == 1, "Should accept minimum of 1 snow layer"
    assert params.nlevsoi == 1, "Should accept minimum of 1 soil layer"
    assert params.nlevgrnd == 1, "Should accept minimum of 1 ground layer"
    assert params.validate(), "Minimal configuration should be valid"


def test_set_custom_parameters_maximal():
    """Test maximum valid configuration at constraint boundaries."""
    set_custom_parameters(50, 100, 100)
    params = get_clm_parameters()
    
    assert params.nlevsno == 50, "Should accept maximum of 50 snow layers"
    assert params.nlevsoi == 100, "Should accept maximum of 100 soil layers"
    assert params.nlevgrnd == 100, "Should accept maximum of 100 ground layers"
    assert params.validate(), "Maximal configuration should be valid"


def test_set_custom_parameters_equal_soil_ground():
    """Test boundary case where nlevgrnd equals nlevsoi (no inactive layers)."""
    set_custom_parameters(10, 25, 25)
    params = get_clm_parameters()
    
    assert params.nlevsoi == params.nlevgrnd, "nlevsoi should equal nlevgrnd"
    assert params.validate(), "Equal soil/ground layers should be valid"
    
    layer_info = get_layer_info()
    assert layer_info["inactive_soil_layers"] == 0, "Should have no inactive layers"


# ============================================================================
# Layer Info Tests
# ============================================================================


def test_get_layer_info_clm5(default_clm5_params):
    """Test get_layer_info for CLM5.0 configuration."""
    layer_info = get_layer_info()
    
    assert layer_info["max_snow_layers"] == 12, "Should have 12 snow layers"
    assert layer_info["active_soil_layers"] == 20, "Should have 20 active soil layers"
    assert layer_info["total_ground_layers"] == 25, "Should have 25 total ground layers"
    assert layer_info["inactive_soil_layers"] == 5, "Should have 5 inactive layers"
    assert layer_info["total_subsurface_layers"] == 37, "Should have 37 total subsurface layers (12+25)"


def test_get_layer_info_clm4_5(default_clm4_5_params):
    """Test get_layer_info for CLM4.5 configuration."""
    layer_info = get_layer_info()
    
    assert layer_info["max_snow_layers"] == 5, "Should have 5 snow layers"
    assert layer_info["active_soil_layers"] == 10, "Should have 10 active soil layers"
    assert layer_info["total_ground_layers"] == 15, "Should have 15 total ground layers"
    assert layer_info["inactive_soil_layers"] == 5, "Should have 5 inactive layers"
    assert layer_info["total_subsurface_layers"] == 20, "Should have 20 total subsurface layers (5+15)"


def test_get_layer_info_custom_with_inactive(custom_small_config):
    """Test get_layer_info with custom configuration having inactive layers."""
    layer_info = get_layer_info()
    
    assert layer_info["max_snow_layers"] == 3, "Should have 3 snow layers"
    assert layer_info["active_soil_layers"] == 5, "Should have 5 active soil layers"
    assert layer_info["total_ground_layers"] == 8, "Should have 8 total ground layers"
    assert layer_info["inactive_soil_layers"] == 3, "Should have 3 inactive layers"
    assert layer_info["total_subsurface_layers"] == 11, "Should have 11 total subsurface layers (3+8)"


# ============================================================================
# Radiation and PFT Info Tests
# ============================================================================


def test_get_radiation_info():
    """Test radiation band information is correct and immutable."""
    rad_info = get_radiation_info()
    
    assert rad_info["total_bands"] == 2, "Should have 2 radiation bands"
    assert rad_info["visible_band_index"] == 1, "Visible band index should be 1"
    assert rad_info["near_infrared_band_index"] == 2, "NIR band index should be 2"


def test_get_pft_info():
    """Test PFT information is correct and immutable."""
    pft_info = get_pft_info()
    
    assert pft_info["max_pft_types"] == 78, "Should have 78 PFT types"


def test_radiation_constants_immutable(default_clm5_params):
    """Test that radiation constants remain fixed across different configurations."""
    rad_info_clm5 = get_radiation_info()
    
    load_preset("CLM4_5")
    rad_info_clm4_5 = get_radiation_info()
    
    assert rad_info_clm5 == rad_info_clm4_5, "Radiation info should be identical across configs"


def test_pft_constants_immutable(default_clm5_params):
    """Test that PFT constants remain fixed across different configurations."""
    pft_info_clm5 = get_pft_info()
    
    load_preset("CLM4_5")
    pft_info_clm4_5 = get_pft_info()
    
    assert pft_info_clm5 == pft_info_clm4_5, "PFT info should be identical across configs"


# ============================================================================
# Layer Array Creation Tests
# ============================================================================


def test_create_layer_arrays_shapes(default_clm5_params):
    """Test that create_layer_arrays returns arrays with correct shapes."""
    arrays = create_layer_arrays()
    
    assert arrays["snow_layers"].shape == (12,), "Snow layers should have shape (12,)"
    assert arrays["soil_layers"].shape == (20,), "Soil layers should have shape (20,)"
    assert arrays["ground_layers"].shape == (25,), "Ground layers should have shape (25,)"
    assert arrays["radiation_bands"].shape == (2,), "Radiation bands should have shape (2,)"
    assert arrays["pft_indices"].shape == (78,), "PFT indices should have shape (78,)"


def test_create_layer_arrays_dtypes(default_clm5_params):
    """Test that create_layer_arrays returns arrays with correct dtypes."""
    arrays = create_layer_arrays()
    
    assert arrays["snow_layers"].dtype == jnp.int32, "Snow layers should be int32"
    assert arrays["soil_layers"].dtype == jnp.int32, "Soil layers should be int32"
    assert arrays["ground_layers"].dtype == jnp.int32, "Ground layers should be int32"
    assert arrays["radiation_bands"].dtype == jnp.int32, "Radiation bands should be int32"
    assert arrays["pft_indices"].dtype == jnp.int32, "PFT indices should be int32"


def test_create_layer_arrays_values_clm5(default_clm5_params):
    """Test that create_layer_arrays returns correct values for CLM5.0."""
    arrays = create_layer_arrays()
    
    # Snow layers: -11 to 0
    expected_snow = jnp.arange(-11, 1, dtype=jnp.int32)
    assert jnp.array_equal(arrays["snow_layers"], expected_snow), "Snow layer indices incorrect"
    
    # Soil layers: 1 to 20
    expected_soil = jnp.arange(1, 21, dtype=jnp.int32)
    assert jnp.array_equal(arrays["soil_layers"], expected_soil), "Soil layer indices incorrect"
    
    # Ground layers: 1 to 25
    expected_ground = jnp.arange(1, 26, dtype=jnp.int32)
    assert jnp.array_equal(arrays["ground_layers"], expected_ground), "Ground layer indices incorrect"
    
    # Radiation bands: 1 to 2
    expected_rad = jnp.array([1, 2], dtype=jnp.int32)
    assert jnp.array_equal(arrays["radiation_bands"], expected_rad), "Radiation band indices incorrect"
    
    # PFT indices: 0 to 77
    expected_pft = jnp.arange(0, 78, dtype=jnp.int32)
    assert jnp.array_equal(arrays["pft_indices"], expected_pft), "PFT indices incorrect"


def test_create_layer_arrays_uninitialized(uninitialized_params):
    """Test that create_layer_arrays raises RuntimeError when uninitialized."""
    with pytest.raises(RuntimeError, match=".*not initialized.*"):
        create_layer_arrays()


# ============================================================================
# Individual Layer Index Function Tests
# ============================================================================


def test_snow_layer_indices_values():
    """Test snow_layer_indices generates correct negative indices."""
    indices = snow_layer_indices(7)
    expected = jnp.array([-6, -5, -4, -3, -2, -1, 0], dtype=jnp.int32)
    
    assert indices.shape == (7,), "Should have shape (7,)"
    assert indices.dtype == jnp.int32, "Should be int32"
    assert jnp.array_equal(indices, expected), "Snow indices should range from -6 to 0"


def test_snow_layer_indices_single():
    """Test snow_layer_indices with single layer."""
    indices = snow_layer_indices(1)
    expected = jnp.array([0], dtype=jnp.int32)
    
    assert indices.shape == (1,), "Should have shape (1,)"
    assert jnp.array_equal(indices, expected), "Single snow layer should be index 0"


def test_soil_layer_indices_values():
    """Test soil_layer_indices generates correct positive indices."""
    indices = soil_layer_indices(10)
    expected = jnp.arange(1, 11, dtype=jnp.int32)
    
    assert indices.shape == (10,), "Should have shape (10,)"
    assert indices.dtype == jnp.int32, "Should be int32"
    assert jnp.array_equal(indices, expected), "Soil indices should range from 1 to 10"


def test_ground_layer_indices_values():
    """Test ground_layer_indices generates correct positive indices."""
    indices = ground_layer_indices(15)
    expected = jnp.arange(1, 16, dtype=jnp.int32)
    
    assert indices.shape == (15,), "Should have shape (15,)"
    assert indices.dtype == jnp.int32, "Should be int32"
    assert jnp.array_equal(indices, expected), "Ground indices should range from 1 to 15"


# ============================================================================
# JAX JIT Compilation Tests
# ============================================================================


def test_snow_layer_indices_jit_compatible():
    """Test that snow_layer_indices works with JAX JIT compilation."""
    jitted_fn = jax.jit(snow_layer_indices)
    
    result_jit = jitted_fn(5)
    result_no_jit = snow_layer_indices(5)
    
    assert jnp.array_equal(result_jit, result_no_jit), "JIT and non-JIT results should match"


def test_soil_layer_indices_jit_compatible():
    """Test that soil_layer_indices works with JAX JIT compilation."""
    jitted_fn = jax.jit(soil_layer_indices)
    
    result_jit = jitted_fn(10)
    result_no_jit = soil_layer_indices(10)
    
    assert jnp.array_equal(result_jit, result_no_jit), "JIT and non-JIT results should match"


def test_ground_layer_indices_jit_compatible():
    """Test that ground_layer_indices works with JAX JIT compilation."""
    jitted_fn = jax.jit(ground_layer_indices)
    
    result_jit = jitted_fn(15)
    result_no_jit = ground_layer_indices(15)
    
    assert jnp.array_equal(result_jit, result_no_jit), "JIT and non-JIT results should match"


# ============================================================================
# State Management Tests
# ============================================================================


def test_reset_parameters():
    """Test that reset_parameters returns to uninitialized state."""
    # Initialize with preset
    load_preset("CLM5_0")
    params = get_clm_parameters()
    assert params.is_initialized(), "Should be initialized after loading preset"
    
    # Reset
    reset_parameters()
    params = get_clm_parameters()
    
    assert not params.is_initialized(), "Should not be initialized after reset"
    assert params.nlevsno == -1, "nlevsno should be -1 after reset"
    assert params.nlevsoi == -1, "nlevsoi should be -1 after reset"
    assert params.nlevgrnd == -1, "nlevgrnd should be -1 after reset"


def test_reset_and_reinitialize_sequence():
    """Test sequence: initialize -> reset -> reinitialize to verify state management."""
    # Step 1: Initialize with CLM5.0
    load_preset("CLM5_0")
    params = get_clm_parameters()
    assert params.is_initialized(), "Should be initialized after CLM5.0 preset"
    assert params.nlevsno == 12, "Should have CLM5.0 snow layers"
    
    # Step 2: Reset
    reset_parameters()
    params = get_clm_parameters()
    assert not params.is_initialized(), "Should not be initialized after reset"
    
    # Step 3: Reinitialize with CLM4.5
    load_preset("CLM4_5")
    params = get_clm_parameters()
    assert params.is_initialized(), "Should be initialized after CLM4.5 preset"
    assert params.nlevsno == 5, "Should have CLM4.5 snow layers"


def test_multiple_preset_switches():
    """Test switching between presets multiple times."""
    # Load CLM5.0
    load_preset("CLM5_0")
    params = get_clm_parameters()
    assert params.nlevsno == 12, "Should have CLM5.0 configuration"
    
    # Switch to CLM4.5
    load_preset("CLM4_5")
    params = get_clm_parameters()
    assert params.nlevsno == 5, "Should have CLM4.5 configuration"
    
    # Switch back to CLM5.0
    load_preset("CLM5_0")
    params = get_clm_parameters()
    assert params.nlevsno == 12, "Should have CLM5.0 configuration again"


# ============================================================================
# Array Consistency Tests
# ============================================================================


def test_layer_array_consistency():
    """Verify all layer arrays have correct shapes and consistent indexing."""
    set_custom_parameters(6, 12, 18)
    arrays = create_layer_arrays()
    
    # Check shapes
    assert arrays["snow_layers"].shape == (6,), "Snow layers shape incorrect"
    assert arrays["soil_layers"].shape == (12,), "Soil layers shape incorrect"
    assert arrays["ground_layers"].shape == (18,), "Ground layers shape incorrect"
    
    # Check snow index range: -5 to 0
    assert arrays["snow_layers"][0] == -5, "First snow index should be -5"
    assert arrays["snow_layers"][-1] == 0, "Last snow index should be 0"
    
    # Check soil index range: 1 to 12
    assert arrays["soil_layers"][0] == 1, "First soil index should be 1"
    assert arrays["soil_layers"][-1] == 12, "Last soil index should be 12"
    
    # Check ground index range: 1 to 18
    assert arrays["ground_layers"][0] == 1, "First ground index should be 1"
    assert arrays["ground_layers"][-1] == 18, "Last ground index should be 18"


def test_layer_indices_monotonic():
    """Test that all layer indices are monotonically increasing."""
    set_custom_parameters(8, 15, 20)
    arrays = create_layer_arrays()
    
    # Snow layers should be monotonically increasing (from negative to 0)
    snow_diffs = jnp.diff(arrays["snow_layers"])
    assert jnp.all(snow_diffs == 1), "Snow layer indices should increase by 1"
    
    # Soil layers should be monotonically increasing
    soil_diffs = jnp.diff(arrays["soil_layers"])
    assert jnp.all(soil_diffs == 1), "Soil layer indices should increase by 1"
    
    # Ground layers should be monotonically increasing
    ground_diffs = jnp.diff(arrays["ground_layers"])
    assert jnp.all(ground_diffs == 1), "Ground layer indices should increase by 1"


# ============================================================================
# Configuration Dictionary Tests
# ============================================================================


def test_config_dict_completeness(default_clm5_params):
    """Verify get_config_dict returns all expected parameters."""
    config = default_clm5_params.get_config_dict()
    
    expected_keys = ["nlevsno", "nlevsoi", "nlevgrnd", "numrad", "ivis", "inir", "mxpft"]
    
    for key in expected_keys:
        assert key in config, f"Config dict should contain {key}"
    
    # Verify values match
    assert config["nlevsno"] == 12, "nlevsno should be 12"
    assert config["nlevsoi"] == 20, "nlevsoi should be 20"
    assert config["nlevgrnd"] == 25, "nlevgrnd should be 25"


def test_config_dict_types(default_clm5_params):
    """Test that config dict values have correct types."""
    config = default_clm5_params.get_config_dict()
    
    assert isinstance(config["nlevsno"], int), "nlevsno should be int"
    assert isinstance(config["nlevsoi"], int), "nlevsoi should be int"
    assert isinstance(config["nlevgrnd"], int), "nlevgrnd should be int"
    assert isinstance(config["numrad"], int), "numrad should be int"
    assert isinstance(config["ivis"], int), "ivis should be int"
    assert isinstance(config["inir"], int), "inir should be int"
    assert isinstance(config["mxpft"], int), "mxpft should be int"


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_uninitialized_state_behavior(uninitialized_params):
    """Test behavior when parameters are in uninitialized state."""
    assert uninitialized_params.nlevsno == -1, "nlevsno should be -1"
    assert uninitialized_params.nlevsoi == -1, "nlevsoi should be -1"
    assert uninitialized_params.nlevgrnd == -1, "nlevgrnd should be -1"
    assert not uninitialized_params.is_initialized(), "Should not be initialized"
    
    # Should raise error when trying to create arrays
    with pytest.raises(RuntimeError):
        create_layer_arrays()


def test_boundary_nlevgrnd_equals_nlevsoi():
    """Test boundary case where nlevgrnd equals nlevsoi (no inactive layers)."""
    set_custom_parameters(10, 25, 25)
    params = get_clm_parameters()
    layer_info = get_layer_info()
    
    assert params.nlevsoi == params.nlevgrnd, "nlevsoi should equal nlevgrnd"
    assert layer_info["inactive_soil_layers"] == 0, "Should have zero inactive layers"
    assert params.validate(), "Configuration should be valid"


def test_large_inactive_layer_count():
    """Test configuration with large number of inactive layers."""
    set_custom_parameters(8, 15, 50)
    params = get_clm_parameters()
    layer_info = get_layer_info()
    
    assert layer_info["inactive_soil_layers"] == 35, "Should have 35 inactive layers"
    assert params.validate(), "Configuration should be valid"


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_workflow_clm5():
    """Test complete workflow with CLM5.0 configuration."""
    # Load preset
    load_preset("CLM5_0")
    
    # Get parameters
    params = get_clm_parameters()
    assert params.is_initialized(), "Should be initialized"
    assert params.validate(), "Should be valid"
    
    # Get layer info
    layer_info = get_layer_info()
    assert layer_info["max_snow_layers"] == 12, "Should have 12 snow layers"
    
    # Get radiation info
    rad_info = get_radiation_info()
    assert rad_info["total_bands"] == 2, "Should have 2 radiation bands"
    
    # Get PFT info
    pft_info = get_pft_info()
    assert pft_info["max_pft_types"] == 78, "Should have 78 PFT types"
    
    # Create arrays
    arrays = create_layer_arrays()
    assert len(arrays) == 5, "Should have 5 array types"
    
    # Get config dict
    config = params.get_config_dict()
    assert len(config) == 7, "Should have 7 config parameters"


def test_full_workflow_custom():
    """Test complete workflow with custom configuration."""
    # Set custom parameters
    set_custom_parameters(7, 14, 21)
    
    # Verify initialization
    params = get_clm_parameters()
    assert params.is_initialized(), "Should be initialized"
    assert params.validate(), "Should be valid"
    
    # Create and verify arrays
    arrays = create_layer_arrays()
    assert arrays["snow_layers"].shape == (7,), "Snow layers shape incorrect"
    assert arrays["soil_layers"].shape == (14,), "Soil layers shape incorrect"
    assert arrays["ground_layers"].shape == (21,), "Ground layers shape incorrect"
    
    # Verify layer info
    layer_info = get_layer_info()
    assert layer_info["inactive_soil_layers"] == 7, "Should have 7 inactive layers"


# ============================================================================
# Parametrized Test Suites
# ============================================================================


@pytest.mark.parametrize("nlevsno", [1, 5, 10, 20, 50])
def test_snow_layer_indices_parametrized(nlevsno):
    """Parametrized test for snow layer index generation."""
    indices = snow_layer_indices(nlevsno)
    
    assert indices.shape == (nlevsno,), f"Shape should be ({nlevsno},)"
    assert indices[0] == -(nlevsno - 1), f"First index should be {-(nlevsno - 1)}"
    assert indices[-1] == 0, "Last index should be 0"
    assert jnp.all(jnp.diff(indices) == 1), "Indices should increase by 1"


@pytest.mark.parametrize("nlevsoi", [1, 10, 25, 50, 100])
def test_soil_layer_indices_parametrized(nlevsoi):
    """Parametrized test for soil layer index generation."""
    indices = soil_layer_indices(nlevsoi)
    
    assert indices.shape == (nlevsoi,), f"Shape should be ({nlevsoi},)"
    assert indices[0] == 1, "First index should be 1"
    assert indices[-1] == nlevsoi, f"Last index should be {nlevsoi}"
    assert jnp.all(jnp.diff(indices) == 1), "Indices should increase by 1"


@pytest.mark.parametrize("nlevgrnd", [1, 15, 30, 60, 100])
def test_ground_layer_indices_parametrized(nlevgrnd):
    """Parametrized test for ground layer index generation."""
    indices = ground_layer_indices(nlevgrnd)
    
    assert indices.shape == (nlevgrnd,), f"Shape should be ({nlevgrnd},)"
    assert indices[0] == 1, "First index should be 1"
    assert indices[-1] == nlevgrnd, f"Last index should be {nlevgrnd}"
    assert jnp.all(jnp.diff(indices) == 1), "Indices should increase by 1"


@pytest.mark.parametrize("config", [
    {"nlevsno": 3, "nlevsoi": 5, "nlevgrnd": 8},
    {"nlevsno": 5, "nlevsoi": 10, "nlevgrnd": 15},
    {"nlevsno": 12, "nlevsoi": 20, "nlevgrnd": 25},
    {"nlevsno": 10, "nlevsoi": 30, "nlevgrnd": 50},
    {"nlevsno": 20, "nlevsoi": 50, "nlevgrnd": 75},
])
def test_valid_configurations_parametrized(config):
    """Parametrized test for various valid configurations."""
    set_custom_parameters(config["nlevsno"], config["nlevsoi"], config["nlevgrnd"])
    params = get_clm_parameters()
    
    assert params.is_initialized(), "Should be initialized"
    assert params.validate(), "Should be valid"
    assert params.nlevsno == config["nlevsno"], "nlevsno mismatch"
    assert params.nlevsoi == config["nlevsoi"], "nlevsoi mismatch"
    assert params.nlevgrnd == config["nlevgrnd"], "nlevgrnd mismatch"


@pytest.mark.parametrize("config", [
    {"nlevsno": 0, "nlevsoi": 10, "nlevgrnd": 15},
    {"nlevsno": 5, "nlevsoi": 0, "nlevgrnd": 15},
    {"nlevsno": 5, "nlevsoi": 10, "nlevgrnd": 5},
    {"nlevsno": 51, "nlevsoi": 10, "nlevgrnd": 15},
    {"nlevsno": 5, "nlevsoi": 101, "nlevgrnd": 101},
    {"nlevsno": -5, "nlevsoi": 10, "nlevgrnd": 15},
])
def test_invalid_configurations_parametrized(config):
    """Parametrized test for various invalid configurations."""
    with pytest.raises(ValueError):
        set_custom_parameters(config["nlevsno"], config["nlevsoi"], config["nlevgrnd"])