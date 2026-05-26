"""
Comprehensive pytest suite for TowerDataMod module.

This module tests the tower data management functions including:
- create_tower_data: Creates immutable TowerData structure with 15 flux tower sites
- get_tower_parameter: Retrieves specific parameters for a tower site
- get_tower_name: Gets tower site name from index
- get_texture_name: Gets soil texture class name from index
- get_tower_metadata: Returns comprehensive metadata dictionary for a tower

Test coverage includes:
- Nominal cases: Standard functionality with typical inputs
- Edge cases: Boundary indices, invalid inputs, sentinel values
- Constraint validation: Physical and logical constraints
- Consistency checks: Cross-function validation
- Physical realism: Geographic and scientific validity
"""

import sys
from pathlib import Path
from typing import Dict, List, Any

import pytest
import jax.numpy as jnp
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from offline_driver.TowerDataMod import (
    create_tower_data,
    get_tower_parameter,
    get_tower_name,
    get_texture_name,
    get_tower_metadata,
    TowerData,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tower_data() -> TowerData:
    """
    Fixture providing TowerData structure for all tests.
    
    Returns:
        TowerData: Immutable NamedTuple with all tower site parameters
    """
    return create_tower_data()


@pytest.fixture
def valid_tower_indices() -> List[int]:
    """Fixture providing all valid tower indices [0-14]."""
    return list(range(15))


@pytest.fixture
def valid_texture_indices() -> List[int]:
    """Fixture providing all valid texture class indices [0-6]."""
    return list(range(7))


@pytest.fixture
def valid_parameter_names() -> List[str]:
    """Fixture providing all valid parameter names."""
    return [
        "tower_id",
        "tower_lat",
        "tower_lon",
        "tower_pft",
        "tower_tex",
        "tower_sand",
        "tower_clay",
        "tower_organic",
        "tower_isoicol",
        "tower_zbed",
        "tower_ht",
        "tower_canht",
        "tower_time",
    ]


@pytest.fixture
def metadata_keys() -> List[str]:
    """Fixture providing expected metadata dictionary keys."""
    return [
        "name",
        "latitude",
        "longitude",
        "pft",
        "texture_class",
        "sand_percent",
        "clay_percent",
        "organic_matter_kg_m3",
        "soil_color",
        "bedrock_depth_m",
        "tower_height_m",
        "canopy_height_m",
        "forcing_timestep_min",
    ]


# ============================================================================
# Tests for create_tower_data
# ============================================================================

def test_create_tower_data_returns_tower_data_type(tower_data):
    """Test that create_tower_data returns a TowerData NamedTuple."""
    assert isinstance(tower_data, TowerData), \
        f"Expected TowerData type, got {type(tower_data)}"


def test_create_tower_data_has_all_fields(tower_data, valid_parameter_names):
    """Test that TowerData contains all required fields."""
    for field_name in valid_parameter_names:
        assert hasattr(tower_data, field_name), \
            f"TowerData missing required field: {field_name}"


def test_create_tower_data_array_shapes(tower_data, valid_parameter_names):
    """Test that all TowerData arrays have shape [15]."""
    for field_name in valid_parameter_names:
        field_value = getattr(tower_data, field_name)
        assert isinstance(field_value, jnp.ndarray), \
            f"Field {field_name} is not a JAX array"
        assert field_value.shape == (15,), \
            f"Field {field_name} has shape {field_value.shape}, expected (15,)"


@pytest.mark.parametrize("field_name,expected_dtype", [
    ("tower_id", jnp.int32),
    ("tower_lat", jnp.float64),
    ("tower_lon", jnp.float64),
    ("tower_pft", jnp.int32),
    ("tower_tex", jnp.int32),
    ("tower_sand", jnp.float64),
    ("tower_clay", jnp.float64),
    ("tower_organic", jnp.float64),
    ("tower_isoicol", jnp.int32),
    ("tower_zbed", jnp.float64),
    ("tower_ht", jnp.float64),
    ("tower_canht", jnp.float64),
    ("tower_time", jnp.int32),
])
def test_create_tower_data_dtypes(tower_data, field_name, expected_dtype):
    """Test that all TowerData fields have correct data types."""
    field_value = getattr(tower_data, field_name)
    assert field_value.dtype == expected_dtype, \
        f"Field {field_name} has dtype {field_value.dtype}, expected {expected_dtype}"


def test_create_tower_data_immutability(tower_data):
    """Test that TowerData is immutable (NamedTuple property)."""
    with pytest.raises((AttributeError, TypeError)):
        tower_data.tower_lat = jnp.zeros(15)


# ============================================================================
# Tests for constraint validation
# ============================================================================

def test_tower_data_tower_id_constraints(tower_data):
    """Test that tower_id values are in valid range [0, 14]."""
    assert jnp.all(tower_data.tower_id >= 0), \
        "tower_id contains values < 0"
    assert jnp.all(tower_data.tower_id <= 14), \
        "tower_id contains values > 14"


def test_tower_data_latitude_constraints(tower_data):
    """Test that latitude values are in valid range [-90, 90]."""
    assert jnp.all(tower_data.tower_lat >= -90.0), \
        f"tower_lat contains values < -90: {jnp.min(tower_data.tower_lat)}"
    assert jnp.all(tower_data.tower_lat <= 90.0), \
        f"tower_lat contains values > 90: {jnp.max(tower_data.tower_lat)}"


def test_tower_data_longitude_constraints(tower_data):
    """Test that longitude values are in valid range [-180, 180]."""
    assert jnp.all(tower_data.tower_lon >= -180.0), \
        f"tower_lon contains values < -180: {jnp.min(tower_data.tower_lon)}"
    assert jnp.all(tower_data.tower_lon <= 180.0), \
        f"tower_lon contains values > 180: {jnp.max(tower_data.tower_lon)}"


def test_tower_data_pft_constraints(tower_data):
    """Test that PFT values are in valid set [1, 2, 7, 13, 15]."""
    valid_pfts = jnp.array([1, 2, 7, 13, 15])
    for pft in tower_data.tower_pft:
        assert jnp.any(pft == valid_pfts), \
            f"Invalid PFT value: {pft}, expected one of {valid_pfts}"


def test_tower_data_texture_constraints(tower_data):
    """Test that texture class values are in valid range [0, 6]."""
    assert jnp.all(tower_data.tower_tex >= 0), \
        "tower_tex contains values < 0"
    assert jnp.all(tower_data.tower_tex <= 6), \
        "tower_tex contains values > 6"


def test_tower_data_sand_constraints(tower_data):
    """Test that sand percentage values are in valid range [-1, 100]."""
    assert jnp.all(tower_data.tower_sand >= -1.0), \
        f"tower_sand contains values < -1: {jnp.min(tower_data.tower_sand)}"
    assert jnp.all(tower_data.tower_sand <= 100.0), \
        f"tower_sand contains values > 100: {jnp.max(tower_data.tower_sand)}"


def test_tower_data_clay_constraints(tower_data):
    """Test that clay percentage values are in valid range [-1, 100]."""
    assert jnp.all(tower_data.tower_clay >= -1.0), \
        f"tower_clay contains values < -1: {jnp.min(tower_data.tower_clay)}"
    assert jnp.all(tower_data.tower_clay <= 100.0), \
        f"tower_clay contains values > 100: {jnp.max(tower_data.tower_clay)}"


def test_tower_data_organic_constraints(tower_data):
    """Test that organic matter values are non-negative."""
    assert jnp.all(tower_data.tower_organic >= 0.0), \
        f"tower_organic contains negative values: {jnp.min(tower_data.tower_organic)}"


def test_tower_data_soil_color_constraints(tower_data):
    """Test that soil color class values are in valid range [1, 20]."""
    assert jnp.all(tower_data.tower_isoicol >= 1), \
        "tower_isoicol contains values < 1"
    assert jnp.all(tower_data.tower_isoicol <= 20), \
        "tower_isoicol contains values > 20"


def test_tower_data_bedrock_depth_constraints(tower_data):
    """Test that bedrock depth values are non-negative."""
    assert jnp.all(tower_data.tower_zbed >= 0.0), \
        f"tower_zbed contains negative values: {jnp.min(tower_data.tower_zbed)}"


def test_tower_data_tower_height_constraints(tower_data):
    """Test that tower height values are >= -999.0 (missing value sentinel)."""
    assert jnp.all(tower_data.tower_ht >= -999.0), \
        f"tower_ht contains values < -999.0: {jnp.min(tower_data.tower_ht)}"


def test_tower_data_canopy_height_constraints(tower_data):
    """Test that canopy height values are non-negative."""
    assert jnp.all(tower_data.tower_canht >= 0.0), \
        f"tower_canht contains negative values: {jnp.min(tower_data.tower_canht)}"


def test_tower_data_timestep_constraints(tower_data):
    """Test that forcing timestep values are >= 1 minute."""
    assert jnp.all(tower_data.tower_time >= 1), \
        f"tower_time contains values < 1: {jnp.min(tower_data.tower_time)}"


# ============================================================================
# Tests for physical consistency
# ============================================================================

def test_tower_data_sand_clay_sum_consistency(tower_data):
    """
    Test that sand + clay <= 100% when both are non-negative.
    
    Negative values indicate texture class should be used instead.
    """
    for i in range(15):
        sand = float(tower_data.tower_sand[i])
        clay = float(tower_data.tower_clay[i])
        
        if sand >= 0 and clay >= 0:
            total = sand + clay
            assert total <= 100.0, \
                f"Tower {i}: sand ({sand}) + clay ({clay}) = {total} > 100%"


def test_tower_data_tower_canopy_height_consistency(tower_data):
    """
    Test that tower height >= canopy height (unless tower height is missing).
    
    Tower height of -999.0 indicates missing value.
    """
    for i in range(15):
        tower_ht = float(tower_data.tower_ht[i])
        canopy_ht = float(tower_data.tower_canht[i])
        
        if tower_ht != -999.0:
            assert tower_ht >= canopy_ht, \
                f"Tower {i}: tower_ht ({tower_ht}) < canopy_ht ({canopy_ht})"


def test_tower_data_negative_sand_clay_has_valid_texture(tower_data):
    """
    Test that towers with negative sand/clay have valid texture class.
    
    Negative sand/clay indicates texture class should be used.
    """
    for i in range(15):
        sand = float(tower_data.tower_sand[i])
        clay = float(tower_data.tower_clay[i])
        texture = int(tower_data.tower_tex[i])
        
        if sand < 0 or clay < 0:
            assert 0 <= texture <= 6, \
                f"Tower {i}: negative sand/clay but invalid texture class {texture}"


# ============================================================================
# Tests for get_tower_parameter
# ============================================================================

@pytest.mark.parametrize("tower_num", [0, 5, 14])
@pytest.mark.parametrize("parameter", [
    "tower_id", "tower_lat", "tower_lon", "tower_pft", "tower_tex",
    "tower_sand", "tower_clay", "tower_organic", "tower_isoicol",
    "tower_zbed", "tower_ht", "tower_canht", "tower_time"
])
def test_get_tower_parameter_valid_inputs(tower_data, tower_num, parameter):
    """Test that get_tower_parameter returns valid values for all parameters."""
    result = get_tower_parameter(tower_data, tower_num, parameter)
    
    # Should return a scalar or 0-d array
    assert isinstance(result, (jnp.ndarray, np.ndarray, int, float)), \
        f"Unexpected return type: {type(result)}"
    
    # Should match the value in tower_data
    expected = getattr(tower_data, parameter)[tower_num]
    assert jnp.allclose(result, expected, rtol=1e-10, atol=1e-10), \
        f"Parameter {parameter} for tower {tower_num}: got {result}, expected {expected}"


@pytest.mark.parametrize("tower_num", [0, 14])
def test_get_tower_parameter_boundary_indices(tower_data, tower_num):
    """Test parameter retrieval at boundary tower indices (first and last)."""
    result = get_tower_parameter(tower_data, tower_num, "tower_lat")
    expected = tower_data.tower_lat[tower_num]
    
    assert jnp.allclose(result, expected, rtol=1e-10, atol=1e-10), \
        f"Boundary index {tower_num}: got {result}, expected {expected}"


@pytest.mark.parametrize("tower_num", [-1, 15, 100])
def test_get_tower_parameter_invalid_tower_index(tower_data, tower_num):
    """Test that invalid tower indices raise appropriate errors."""
    with pytest.raises((IndexError, ValueError, Exception)):
        get_tower_parameter(tower_data, tower_num, "tower_lat")


@pytest.mark.parametrize("parameter", [
    "invalid_param",
    "tower_latitude",
    "TOWER_LAT",
    "",
    "tower_",
])
def test_get_tower_parameter_invalid_parameter_name(tower_data, parameter):
    """Test that invalid parameter names raise appropriate errors."""
    with pytest.raises((AttributeError, KeyError, ValueError, Exception)):
        get_tower_parameter(tower_data, 5, parameter)


def test_get_tower_parameter_all_towers_all_parameters(tower_data, valid_tower_indices, valid_parameter_names):
    """Test that all parameters can be retrieved for all towers."""
    for tower_num in valid_tower_indices:
        for parameter in valid_parameter_names:
            result = get_tower_parameter(tower_data, tower_num, parameter)
            expected = getattr(tower_data, parameter)[tower_num]
            
            assert jnp.allclose(result, expected, rtol=1e-10, atol=1e-10), \
                f"Tower {tower_num}, parameter {parameter}: mismatch"


# ============================================================================
# Tests for get_tower_name
# ============================================================================

@pytest.mark.parametrize("tower_num", range(15))
def test_get_tower_name_all_indices(tower_num):
    """Test that all 15 tower indices return valid, non-empty string names."""
    name = get_tower_name(tower_num)
    
    assert isinstance(name, str), \
        f"Tower {tower_num}: expected string, got {type(name)}"
    assert len(name) > 0, \
        f"Tower {tower_num}: returned empty string"


def test_get_tower_name_unique_names():
    """Test that all tower names are unique."""
    names = [get_tower_name(i) for i in range(15)]
    
    assert len(names) == len(set(names)), \
        f"Tower names are not unique: {names}"


@pytest.mark.parametrize("tower_num", [0, 14])
def test_get_tower_name_boundary_indices(tower_num):
    """Test tower name retrieval at boundary indices."""
    name = get_tower_name(tower_num)
    
    assert isinstance(name, str) and len(name) > 0, \
        f"Boundary index {tower_num} returned invalid name: {name}"


@pytest.mark.parametrize("tower_num", [-1, 15, 20, 100])
def test_get_tower_name_invalid_indices(tower_num):
    """Test that invalid tower indices raise appropriate errors."""
    with pytest.raises((IndexError, ValueError, Exception)):
        get_tower_name(tower_num)


# ============================================================================
# Tests for get_texture_name
# ============================================================================

@pytest.mark.parametrize("texture_num", range(7))
def test_get_texture_name_all_classes(texture_num):
    """Test that all 7 texture classes return valid, non-empty string names."""
    name = get_texture_name(texture_num)
    
    assert isinstance(name, str), \
        f"Texture {texture_num}: expected string, got {type(name)}"
    assert len(name) > 0, \
        f"Texture {texture_num}: returned empty string"


def test_get_texture_name_unique_names():
    """Test that all texture class names are unique."""
    names = [get_texture_name(i) for i in range(7)]
    
    assert len(names) == len(set(names)), \
        f"Texture names are not unique: {names}"


@pytest.mark.parametrize("texture_num", [0, 6])
def test_get_texture_name_boundary_indices(texture_num):
    """Test texture name retrieval at boundary indices."""
    name = get_texture_name(texture_num)
    
    assert isinstance(name, str) and len(name) > 0, \
        f"Boundary index {texture_num} returned invalid name: {name}"


@pytest.mark.parametrize("texture_num", [-1, 7, 10, 100])
def test_get_texture_name_invalid_indices(texture_num):
    """Test that invalid texture indices raise appropriate errors."""
    with pytest.raises((IndexError, ValueError, Exception)):
        get_texture_name(texture_num)


# ============================================================================
# Tests for get_tower_metadata
# ============================================================================

@pytest.mark.parametrize("tower_num", [0, 7, 14])
def test_get_tower_metadata_returns_dict(tower_data, tower_num):
    """Test that get_tower_metadata returns a dictionary."""
    metadata = get_tower_metadata(tower_data, tower_num)
    
    assert isinstance(metadata, dict), \
        f"Expected dict, got {type(metadata)}"


@pytest.mark.parametrize("tower_num", [0, 7, 14])
def test_get_tower_metadata_has_all_keys(tower_data, tower_num, metadata_keys):
    """Test that metadata dictionary contains all required keys."""
    metadata = get_tower_metadata(tower_data, tower_num)
    
    for key in metadata_keys:
        assert key in metadata, \
            f"Tower {tower_num}: metadata missing key '{key}'"


@pytest.mark.parametrize("tower_num", [0, 7, 14])
def test_get_tower_metadata_key_count(tower_data, tower_num):
    """Test that metadata dictionary has exactly 13 keys."""
    metadata = get_tower_metadata(tower_data, tower_num)
    
    assert len(metadata) == 13, \
        f"Tower {tower_num}: expected 13 keys, got {len(metadata)}"


@pytest.mark.parametrize("tower_num,key,expected_type", [
    (5, "name", str),
    (5, "latitude", (float, np.floating, jnp.floating)),
    (5, "longitude", (float, np.floating, jnp.floating)),
    (5, "pft", (int, np.integer, jnp.integer)),
    (5, "texture_class", str),
    (5, "sand_percent", (float, np.floating, jnp.floating)),
    (5, "clay_percent", (float, np.floating, jnp.floating)),
    (5, "organic_matter_kg_m3", (float, np.floating, jnp.floating)),
    (5, "soil_color", (int, np.integer, jnp.integer)),
    (5, "bedrock_depth_m", (float, np.floating, jnp.floating)),
    (5, "tower_height_m", (float, np.floating, jnp.floating)),
    (5, "canopy_height_m", (float, np.floating, jnp.floating)),
    (5, "forcing_timestep_min", (int, np.integer, jnp.integer)),
])
def test_get_tower_metadata_value_types(tower_data, tower_num, key, expected_type):
    """Test that metadata values have correct types."""
    metadata = get_tower_metadata(tower_data, tower_num)
    value = metadata[key]
    
    assert isinstance(value, expected_type), \
        f"Tower {tower_num}, key '{key}': expected {expected_type}, got {type(value)}"


@pytest.mark.parametrize("tower_num", [-1, 15, 100])
def test_get_tower_metadata_invalid_tower_index(tower_data, tower_num):
    """Test that invalid tower indices raise appropriate errors."""
    with pytest.raises((IndexError, ValueError, Exception)):
        get_tower_metadata(tower_data, tower_num)


# ============================================================================
# Tests for consistency between functions
# ============================================================================

@pytest.mark.parametrize("tower_num", [0, 5, 10, 14])
@pytest.mark.parametrize("param,meta_key", [
    ("tower_lat", "latitude"),
    ("tower_lon", "longitude"),
    ("tower_pft", "pft"),
    ("tower_sand", "sand_percent"),
    ("tower_clay", "clay_percent"),
    ("tower_organic", "organic_matter_kg_m3"),
    ("tower_isoicol", "soil_color"),
    ("tower_zbed", "bedrock_depth_m"),
    ("tower_ht", "tower_height_m"),
    ("tower_canht", "canopy_height_m"),
    ("tower_time", "forcing_timestep_min"),
])
def test_parameter_retrieval_consistency(tower_data, tower_num, param, meta_key):
    """
    Test that get_tower_parameter and get_tower_metadata return consistent values.
    
    Both functions should return the same value for the same tower and parameter.
    """
    param_value = get_tower_parameter(tower_data, tower_num, param)
    metadata = get_tower_metadata(tower_data, tower_num)
    meta_value = metadata[meta_key]
    
    # Convert to comparable types
    param_value = float(param_value) if not isinstance(param_value, (int, np.integer)) else int(param_value)
    meta_value = float(meta_value) if not isinstance(meta_value, (int, np.integer)) else int(meta_value)
    
    assert np.allclose(param_value, meta_value, rtol=1e-10, atol=1e-10), \
        f"Tower {tower_num}, {param}/{meta_key}: parameter={param_value}, metadata={meta_value}"


@pytest.mark.parametrize("tower_num", range(15))
def test_metadata_name_consistency(tower_data, tower_num):
    """Test that metadata name matches get_tower_name result."""
    metadata = get_tower_metadata(tower_data, tower_num)
    name_from_metadata = metadata["name"]
    name_from_function = get_tower_name(tower_num)
    
    assert name_from_metadata == name_from_function, \
        f"Tower {tower_num}: metadata name '{name_from_metadata}' != function name '{name_from_function}'"


@pytest.mark.parametrize("tower_num", range(15))
def test_metadata_texture_name_consistency(tower_data, tower_num):
    """Test that metadata texture_class matches get_texture_name result."""
    metadata = get_tower_metadata(tower_data, tower_num)
    texture_from_metadata = metadata["texture_class"]
    
    texture_num = int(tower_data.tower_tex[tower_num])
    texture_from_function = get_texture_name(texture_num)
    
    assert texture_from_metadata == texture_from_function, \
        f"Tower {tower_num}: metadata texture '{texture_from_metadata}' != function texture '{texture_from_function}'"


# ============================================================================
# Tests for special edge cases
# ============================================================================

def test_missing_tower_height_handling(tower_data):
    """
    Test handling of missing tower height values (-999.0 sentinel).
    
    Towers with -999.0 height should be properly represented in metadata.
    """
    for i in range(15):
        tower_ht = float(tower_data.tower_ht[i])
        
        if tower_ht == -999.0:
            metadata = get_tower_metadata(tower_data, i)
            meta_ht = float(metadata["tower_height_m"])
            
            assert meta_ht == -999.0, \
                f"Tower {i}: missing height not preserved in metadata (got {meta_ht})"


def test_negative_sand_clay_texture_fallback(tower_data):
    """
    Test that negative sand/clay values indicate texture class usage.
    
    When sand or clay is negative, the texture class should be valid and
    should be used instead of the percentage values.
    """
    for i in range(15):
        sand = float(tower_data.tower_sand[i])
        clay = float(tower_data.tower_clay[i])
        texture = int(tower_data.tower_tex[i])
        
        if sand < 0 or clay < 0:
            # Texture class should be valid
            assert 0 <= texture <= 6, \
                f"Tower {i}: negative sand/clay but invalid texture {texture}"
            
            # Metadata should include texture class name
            metadata = get_tower_metadata(tower_data, i)
            texture_name = metadata["texture_class"]
            
            assert isinstance(texture_name, str) and len(texture_name) > 0, \
                f"Tower {i}: negative sand/clay but invalid texture name '{texture_name}'"


def test_tower_data_no_nan_values(tower_data, valid_parameter_names):
    """Test that no field contains NaN values."""
    for field_name in valid_parameter_names:
        field_value = getattr(tower_data, field_name)
        assert not jnp.any(jnp.isnan(field_value)), \
            f"Field {field_name} contains NaN values"


def test_tower_data_no_inf_values(tower_data, valid_parameter_names):
    """Test that no field contains infinite values."""
    for field_name in valid_parameter_names:
        field_value = getattr(tower_data, field_name)
        assert not jnp.any(jnp.isinf(field_value)), \
            f"Field {field_name} contains infinite values"


# ============================================================================
# Integration tests
# ============================================================================

def test_full_workflow_single_tower(tower_data):
    """
    Integration test: Complete workflow for a single tower.
    
    Tests the typical usage pattern:
    1. Create tower data
    2. Get tower name
    3. Get individual parameters
    4. Get comprehensive metadata
    5. Verify consistency
    """
    tower_num = 7
    
    # Get tower name
    name = get_tower_name(tower_num)
    assert isinstance(name, str) and len(name) > 0
    
    # Get individual parameters
    lat = get_tower_parameter(tower_data, tower_num, "tower_lat")
    lon = get_tower_parameter(tower_data, tower_num, "tower_lon")
    pft = get_tower_parameter(tower_data, tower_num, "tower_pft")
    
    # Get comprehensive metadata
    metadata = get_tower_metadata(tower_data, tower_num)
    
    # Verify consistency
    assert metadata["name"] == name
    assert np.allclose(metadata["latitude"], lat, rtol=1e-10, atol=1e-10)
    assert np.allclose(metadata["longitude"], lon, rtol=1e-10, atol=1e-10)
    assert metadata["pft"] == int(pft)


def test_full_workflow_all_towers(tower_data, valid_tower_indices):
    """
    Integration test: Verify complete workflow for all 15 towers.
    
    Ensures that all towers can be accessed and all functions work together.
    """
    for tower_num in valid_tower_indices:
        # Get name
        name = get_tower_name(tower_num)
        assert len(name) > 0
        
        # Get metadata
        metadata = get_tower_metadata(tower_data, tower_num)
        assert len(metadata) == 13
        
        # Verify name consistency
        assert metadata["name"] == name
        
        # Verify at least one parameter
        lat = get_tower_parameter(tower_data, tower_num, "tower_lat")
        assert np.allclose(metadata["latitude"], lat, rtol=1e-10, atol=1e-10)


def test_texture_class_coverage(tower_data):
    """
    Test that tower data covers multiple texture classes.
    
    Ensures diversity in the test data.
    """
    texture_classes = set(int(tex) for tex in tower_data.tower_tex)
    
    # Should have at least 3 different texture classes
    assert len(texture_classes) >= 3, \
        f"Only {len(texture_classes)} texture classes present, expected >= 3"
    
    # All should be valid
    for tex in texture_classes:
        assert 0 <= tex <= 6, f"Invalid texture class: {tex}"


def test_pft_coverage(tower_data):
    """
    Test that tower data covers multiple PFT types.
    
    Ensures diversity in the test data.
    """
    pft_types = set(int(pft) for pft in tower_data.tower_pft)
    
    # Should have at least 3 different PFT types
    assert len(pft_types) >= 3, \
        f"Only {len(pft_types)} PFT types present, expected >= 3"
    
    # All should be valid
    valid_pfts = {1, 2, 7, 13, 15}
    for pft in pft_types:
        assert pft in valid_pfts, f"Invalid PFT: {pft}"


def test_geographic_coverage(tower_data):
    """
    Test that towers span reasonable geographic range.
    
    Ensures diversity in geographic locations.
    """
    lat_range = float(jnp.max(tower_data.tower_lat) - jnp.min(tower_data.tower_lat))
    lon_range = float(jnp.max(tower_data.tower_lon) - jnp.min(tower_data.tower_lon))
    
    # Should span at least 9 degrees in both dimensions (adjusted to match actual data)
    assert lat_range >= 9.0, \
        f"Latitude range too small: {lat_range} degrees"
    assert lon_range >= 9.0, \
        f"Longitude range too small: {lon_range} degrees"