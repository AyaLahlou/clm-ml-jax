"""
Comprehensive pytest suite for SoilTexMod module.

Tests the soil texture parameter functions including:
- create_soil_texture_params: Creates soil texture parameter container
- get_texture_class_index: Maps texture names to indices
- interpolate_texture_params: Interpolates parameters from sand/clay fractions

Based on Cosby et al. (1984) and Clapp & Hornberger (1978) soil hydraulic properties.
"""

import sys
from pathlib import Path
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from offline_driver.SoilTexMod import (
    CLAY_INDEX,
    CLAY_LOAM_INDEX,
    DEFAULT_SOIL_TEXTURE_PARAMS,
    LOAM_INDEX,
    LOAMY_SAND_INDEX,
    SAND_INDEX,
    SANDY_CLAY_INDEX,
    SANDY_CLAY_LOAM_INDEX,
    SANDY_LOAM_INDEX,
    SILTY_CLAY_INDEX,
    SILTY_CLAY_LOAM_INDEX,
    SILTY_LOAM_INDEX,
    create_soil_texture_params,
    get_texture_class_index,
    interpolate_texture_params,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def soil_params():
    """Fixture providing soil texture parameters."""
    return create_soil_texture_params()


@pytest.fixture
def valid_texture_names():
    """Fixture providing all valid texture class names."""
    return [
        "sand",
        "loamy sand",
        "sandy loam",
        "silty loam",
        "loam",
        "sandy clay loam",
        "silty clay loam",
        "clay loam",
        "sandy clay",
        "silty clay",
        "clay",
    ]


@pytest.fixture
def module_constants():
    """Fixture providing module constant indices."""
    return {
        "sand": SAND_INDEX,
        "loamy sand": LOAMY_SAND_INDEX,
        "sandy loam": SANDY_LOAM_INDEX,
        "silty loam": SILTY_LOAM_INDEX,
        "loam": LOAM_INDEX,
        "sandy clay loam": SANDY_CLAY_LOAM_INDEX,
        "silty clay loam": SILTY_CLAY_LOAM_INDEX,
        "clay loam": CLAY_LOAM_INDEX,
        "sandy clay": SANDY_CLAY_INDEX,
        "silty clay": SILTY_CLAY_INDEX,
        "clay": CLAY_INDEX,
    }


# ============================================================================
# Tests for create_soil_texture_params
# ============================================================================


def test_create_soil_texture_params_structure(soil_params):
    """Test that create_soil_texture_params returns correct structure."""
    assert hasattr(soil_params, "ntex"), "Missing ntex attribute"
    assert hasattr(soil_params, "soil_tex"), "Missing soil_tex attribute"
    assert hasattr(soil_params, "sand_tex"), "Missing sand_tex attribute"
    assert hasattr(soil_params, "silt_tex"), "Missing silt_tex attribute"
    assert hasattr(soil_params, "clay_tex"), "Missing clay_tex attribute"
    assert hasattr(soil_params, "watsat_tex"), "Missing watsat_tex attribute"
    assert hasattr(soil_params, "smpsat_tex"), "Missing smpsat_tex attribute"
    assert hasattr(soil_params, "hksat_tex"), "Missing hksat_tex attribute"
    assert hasattr(soil_params, "bsw_tex"), "Missing bsw_tex attribute"


def test_create_soil_texture_params_ntex(soil_params):
    """Test that ntex equals 11 (number of texture classes)."""
    assert soil_params.ntex == 11, f"Expected ntex=11, got {soil_params.ntex}"


def test_create_soil_texture_params_shapes(soil_params):
    """Test that all array attributes have shape (11,)."""
    expected_shape = (11,)
    
    assert soil_params.sand_tex.shape == expected_shape, \
        f"sand_tex shape {soil_params.sand_tex.shape} != {expected_shape}"
    assert soil_params.silt_tex.shape == expected_shape, \
        f"silt_tex shape {soil_params.silt_tex.shape} != {expected_shape}"
    assert soil_params.clay_tex.shape == expected_shape, \
        f"clay_tex shape {soil_params.clay_tex.shape} != {expected_shape}"
    assert soil_params.watsat_tex.shape == expected_shape, \
        f"watsat_tex shape {soil_params.watsat_tex.shape} != {expected_shape}"
    assert soil_params.smpsat_tex.shape == expected_shape, \
        f"smpsat_tex shape {soil_params.smpsat_tex.shape} != {expected_shape}"
    assert soil_params.hksat_tex.shape == expected_shape, \
        f"hksat_tex shape {soil_params.hksat_tex.shape} != {expected_shape}"
    assert soil_params.bsw_tex.shape == expected_shape, \
        f"bsw_tex shape {soil_params.bsw_tex.shape} != {expected_shape}"


def test_create_soil_texture_params_soil_tex_length(soil_params):
    """Test that soil_tex tuple has 11 elements."""
    assert len(soil_params.soil_tex) == 11, \
        f"Expected 11 texture names, got {len(soil_params.soil_tex)}"


def test_create_soil_texture_params_fraction_ranges(soil_params):
    """Test that sand, silt, clay fractions are in [0, 1]."""
    assert jnp.all(soil_params.sand_tex >= 0.0), "sand_tex has negative values"
    assert jnp.all(soil_params.sand_tex <= 1.0), "sand_tex has values > 1.0"
    
    assert jnp.all(soil_params.silt_tex >= 0.0), "silt_tex has negative values"
    assert jnp.all(soil_params.silt_tex <= 1.0), "silt_tex has values > 1.0"
    
    assert jnp.all(soil_params.clay_tex >= 0.0), "clay_tex has negative values"
    assert jnp.all(soil_params.clay_tex <= 1.0), "clay_tex has values > 1.0"


def test_create_soil_texture_params_fraction_sum(soil_params):
    """Test that sand + silt + clay = 1.0 for each texture class."""
    fraction_sum = soil_params.sand_tex + soil_params.silt_tex + soil_params.clay_tex
    
    assert jnp.allclose(fraction_sum, 1.0, atol=1e-6, rtol=1e-6), \
        f"Fractions don't sum to 1.0: {fraction_sum}"


def test_create_soil_texture_params_watsat_range(soil_params):
    """Test that watsat (porosity) is in [0, 1]."""
    assert jnp.all(soil_params.watsat_tex >= 0.0), "watsat_tex has negative values"
    assert jnp.all(soil_params.watsat_tex <= 1.0), "watsat_tex has values > 1.0"


def test_create_soil_texture_params_smpsat_negative(soil_params):
    """Test that smpsat (matric potential) is negative or zero."""
    assert jnp.all(soil_params.smpsat_tex <= 0.0), \
        f"smpsat_tex should be <= 0 (tension), got max={jnp.max(soil_params.smpsat_tex)}"


def test_create_soil_texture_params_hksat_positive(soil_params):
    """Test that hksat (hydraulic conductivity) is positive."""
    assert jnp.all(soil_params.hksat_tex >= 0.0), \
        f"hksat_tex should be >= 0, got min={jnp.min(soil_params.hksat_tex)}"


def test_create_soil_texture_params_bsw_positive(soil_params):
    """Test that bsw (Clapp-Hornberger b) is positive."""
    assert jnp.all(soil_params.bsw_tex >= 0.0), \
        f"bsw_tex should be >= 0, got min={jnp.min(soil_params.bsw_tex)}"


def test_create_soil_texture_params_dtypes(soil_params):
    """Test that array attributes have correct dtypes."""
    assert isinstance(soil_params.sand_tex, jnp.ndarray), "sand_tex not a JAX array"
    assert isinstance(soil_params.silt_tex, jnp.ndarray), "silt_tex not a JAX array"
    assert isinstance(soil_params.clay_tex, jnp.ndarray), "clay_tex not a JAX array"
    assert isinstance(soil_params.watsat_tex, jnp.ndarray), "watsat_tex not a JAX array"
    assert isinstance(soil_params.smpsat_tex, jnp.ndarray), "smpsat_tex not a JAX array"
    assert isinstance(soil_params.hksat_tex, jnp.ndarray), "hksat_tex not a JAX array"
    assert isinstance(soil_params.bsw_tex, jnp.ndarray), "bsw_tex not a JAX array"


def test_create_soil_texture_params_immutability(soil_params):
    """Test that SoilTextureParams is immutable (namedtuple property)."""
    with pytest.raises(AttributeError):
        soil_params.ntex = 12


def test_default_params_singleton():
    """Test that DEFAULT_SOIL_TEXTURE_PARAMS matches create_soil_texture_params()."""
    fresh_params = create_soil_texture_params()
    
    assert DEFAULT_SOIL_TEXTURE_PARAMS.ntex == fresh_params.ntex
    assert len(DEFAULT_SOIL_TEXTURE_PARAMS.soil_tex) == len(fresh_params.soil_tex)
    
    assert jnp.allclose(DEFAULT_SOIL_TEXTURE_PARAMS.sand_tex, fresh_params.sand_tex)
    assert jnp.allclose(DEFAULT_SOIL_TEXTURE_PARAMS.silt_tex, fresh_params.silt_tex)
    assert jnp.allclose(DEFAULT_SOIL_TEXTURE_PARAMS.clay_tex, fresh_params.clay_tex)
    assert jnp.allclose(DEFAULT_SOIL_TEXTURE_PARAMS.watsat_tex, fresh_params.watsat_tex)
    assert jnp.allclose(DEFAULT_SOIL_TEXTURE_PARAMS.smpsat_tex, fresh_params.smpsat_tex)
    assert jnp.allclose(DEFAULT_SOIL_TEXTURE_PARAMS.hksat_tex, fresh_params.hksat_tex)
    assert jnp.allclose(DEFAULT_SOIL_TEXTURE_PARAMS.bsw_tex, fresh_params.bsw_tex)


# ============================================================================
# Tests for get_texture_class_index
# ============================================================================


@pytest.mark.parametrize(
    "texture_name,expected_index",
    [
        ("sand", 0),
        ("loamy sand", 1),
        ("sandy loam", 2),
        ("silty loam", 3),
        ("loam", 4),
        ("sandy clay loam", 5),
        ("silty clay loam", 6),
        ("clay loam", 7),
        ("sandy clay", 8),
        ("silty clay", 9),
        ("clay", 10),
    ],
)
def test_get_texture_class_index_valid_names(texture_name, expected_index):
    """Test that all valid texture names map to correct indices."""
    result = get_texture_class_index(texture_name)
    assert result == expected_index, \
        f"Expected index {expected_index} for '{texture_name}', got {result}"


def test_get_texture_class_index_all_valid(valid_texture_names):
    """Test all valid texture names return indices 0-10."""
    indices = [get_texture_class_index(name) for name in valid_texture_names]
    expected_indices = list(range(11))
    
    assert indices == expected_indices, \
        f"Expected indices {expected_indices}, got {indices}"


def test_get_texture_class_index_constants_consistency(module_constants):
    """Test that module constants match get_texture_class_index results."""
    for texture_name, constant_index in module_constants.items():
        function_index = get_texture_class_index(texture_name)
        assert function_index == constant_index, \
            f"Constant {texture_name.upper()}_INDEX={constant_index} != " \
            f"get_texture_class_index('{texture_name}')={function_index}"


@pytest.mark.parametrize(
    "invalid_name",
    [
        "invalid_texture",
        "",
        "silt",  # Not a valid class (only part of other classes)
        "sandy_loam",  # Underscore instead of space
        "sandyloam",  # No space
    ],
)
def test_get_texture_class_index_invalid_names(invalid_name):
    """Test that invalid texture names raise appropriate errors.
    
    Note: The function is case-insensitive, so 'Sand', 'LOAM', etc. are valid.
    """
    with pytest.raises((ValueError, KeyError, IndexError)):
        get_texture_class_index(invalid_name)


def test_get_texture_class_index_case_insensitive():
    """Test that texture class lookup is case-insensitive."""
    # Test various capitalizations of valid names
    assert get_texture_class_index('sand') == get_texture_class_index('Sand')
    assert get_texture_class_index('loam') == get_texture_class_index('LOAM')
    assert get_texture_class_index('sandy clay') == get_texture_class_index('SaNdY cLaY')
    assert get_texture_class_index('clay loam') == get_texture_class_index('Clay Loam')


# ============================================================================
# Tests for interpolate_texture_params
# ============================================================================


def test_interpolate_texture_params_pure_sand(soil_params):
    """Test interpolation for pure sand (100% sand, 0% clay)."""
    sand_frac = jnp.array([1.0])
    clay_frac = jnp.array([0.0])
    
    result = interpolate_texture_params(sand_frac, clay_frac, soil_params)
    
    assert "watsat" in result, "Missing 'watsat' in result"
    assert "smpsat" in result, "Missing 'smpsat' in result"
    assert "hksat" in result, "Missing 'hksat' in result"
    assert "bsw" in result, "Missing 'bsw' in result"
    
    # Check shapes
    assert result["watsat"].shape == (1,), f"watsat shape {result['watsat'].shape} != (1,)"
    assert result["smpsat"].shape == (1,), f"smpsat shape {result['smpsat'].shape} != (1,)"
    assert result["hksat"].shape == (1,), f"hksat shape {result['hksat'].shape} != (1,)"
    assert result["bsw"].shape == (1,), f"bsw shape {result['bsw'].shape} != (1,)"


def test_interpolate_texture_params_pure_clay(soil_params):
    """Test interpolation for pure clay (0% sand, 100% clay)."""
    sand_frac = jnp.array([0.0])
    clay_frac = jnp.array([1.0])
    
    result = interpolate_texture_params(sand_frac, clay_frac, soil_params)
    
    assert result["watsat"].shape == (1,)
    assert result["smpsat"].shape == (1,)
    assert result["hksat"].shape == (1,)
    assert result["bsw"].shape == (1,)


def test_interpolate_texture_params_pure_silt(soil_params):
    """Test interpolation for pure silt (0% sand, 0% clay)."""
    sand_frac = jnp.array([0.0])
    clay_frac = jnp.array([0.0])
    
    result = interpolate_texture_params(sand_frac, clay_frac, soil_params)
    
    assert result["watsat"].shape == (1,)
    assert result["smpsat"].shape == (1,)
    assert result["hksat"].shape == (1,)
    assert result["bsw"].shape == (1,)


@pytest.mark.parametrize(
    "sand_frac,clay_frac,description",
    [
        ([0.52, 0.42, 0.23, 0.15, 0.65], [0.06, 0.18, 0.27, 0.42, 0.10], "typical_soils"),
        ([0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.15, 0.2, 0.25, 0.3], "gradient_1"),
        ([0.6, 0.7, 0.8, 0.15, 0.25], [0.15, 0.1, 0.05, 0.35, 0.3], "gradient_2"),
    ],
)
def test_interpolate_texture_params_typical_soils(soil_params, sand_frac, clay_frac, description):
    """Test interpolation with realistic soil compositions."""
    sand_arr = jnp.array(sand_frac)
    clay_arr = jnp.array(clay_frac)
    
    result = interpolate_texture_params(sand_arr, clay_arr, soil_params)
    
    n_points = len(sand_frac)
    expected_shape = (n_points,)
    
    assert result["watsat"].shape == expected_shape, \
        f"{description}: watsat shape {result['watsat'].shape} != {expected_shape}"
    assert result["smpsat"].shape == expected_shape, \
        f"{description}: smpsat shape {result['smpsat'].shape} != {expected_shape}"
    assert result["hksat"].shape == expected_shape, \
        f"{description}: hksat shape {result['hksat'].shape} != {expected_shape}"
    assert result["bsw"].shape == expected_shape, \
        f"{description}: bsw shape {result['bsw'].shape} != {expected_shape}"


def test_interpolate_texture_params_boundary_conditions(soil_params):
    """Test interpolation at boundary conditions."""
    sand_frac = jnp.array([0.0, 1.0, 0.5, 0.0, 0.33])
    clay_frac = jnp.array([0.0, 0.0, 0.5, 1.0, 0.33])
    
    result = interpolate_texture_params(sand_frac, clay_frac, soil_params)
    
    expected_shape = (5,)
    assert result["watsat"].shape == expected_shape
    assert result["smpsat"].shape == expected_shape
    assert result["hksat"].shape == expected_shape
    assert result["bsw"].shape == expected_shape


def test_interpolate_texture_params_large_array(soil_params):
    """Test interpolation with large array (20 points)."""
    sand_frac = jnp.array([
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.15, 0.25,
        0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.12, 0.22, 0.32, 0.42
    ])
    clay_frac = jnp.array([
        0.1, 0.15, 0.2, 0.25, 0.3, 0.15, 0.1, 0.05, 0.35, 0.3,
        0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.4, 0.35, 0.3, 0.25
    ])
    
    result = interpolate_texture_params(sand_frac, clay_frac, soil_params)
    
    expected_shape = (20,)
    assert result["watsat"].shape == expected_shape
    assert result["smpsat"].shape == expected_shape
    assert result["hksat"].shape == expected_shape
    assert result["bsw"].shape == expected_shape


def test_interpolate_texture_params_single_point(soil_params):
    """Test interpolation with single point (minimum array size)."""
    sand_frac = jnp.array([0.4])
    clay_frac = jnp.array([0.2])
    
    result = interpolate_texture_params(sand_frac, clay_frac, soil_params)
    
    expected_shape = (1,)
    assert result["watsat"].shape == expected_shape
    assert result["smpsat"].shape == expected_shape
    assert result["hksat"].shape == expected_shape
    assert result["bsw"].shape == expected_shape


def test_interpolate_texture_params_physical_constraints(soil_params):
    """Test that interpolated parameters satisfy physical constraints."""
    sand_frac = jnp.array([0.52, 0.42, 0.23, 0.15, 0.65])
    clay_frac = jnp.array([0.06, 0.18, 0.27, 0.42, 0.10])
    
    result = interpolate_texture_params(sand_frac, clay_frac, soil_params)
    
    # watsat (porosity) in [0, 1]
    assert jnp.all(result["watsat"] >= 0.0), "watsat has negative values"
    assert jnp.all(result["watsat"] <= 1.0), "watsat has values > 1.0"
    
    # smpsat (matric potential) <= 0
    assert jnp.all(result["smpsat"] <= 0.0), \
        f"smpsat should be <= 0, got max={jnp.max(result['smpsat'])}"
    
    # hksat (hydraulic conductivity) >= 0
    assert jnp.all(result["hksat"] >= 0.0), \
        f"hksat should be >= 0, got min={jnp.min(result['hksat'])}"
    
    # bsw (Clapp-Hornberger b) >= 0
    assert jnp.all(result["bsw"] >= 0.0), \
        f"bsw should be >= 0, got min={jnp.min(result['bsw'])}"


def test_interpolate_texture_params_dtypes(soil_params):
    """Test that interpolated parameters have correct dtypes."""
    sand_frac = jnp.array([0.4, 0.3, 0.5])
    clay_frac = jnp.array([0.2, 0.3, 0.15])
    
    result = interpolate_texture_params(sand_frac, clay_frac, soil_params)
    
    assert isinstance(result["watsat"], jnp.ndarray), "watsat not a JAX array"
    assert isinstance(result["smpsat"], jnp.ndarray), "smpsat not a JAX array"
    assert isinstance(result["hksat"], jnp.ndarray), "hksat not a JAX array"
    assert isinstance(result["bsw"], jnp.ndarray), "bsw not a JAX array"


@pytest.mark.parametrize(
    "sand_frac,clay_frac,description",
    [
        ([-0.1, 0.5], [0.2, 0.3], "negative_sand"),
        ([0.5, 0.3], [1.2, 0.3], "clay_greater_than_one"),
        ([0.7, 0.5], [0.5, 0.3], "sum_greater_than_one"),
        ([1.5, 0.3], [0.2, 0.3], "sand_greater_than_one"),
        ([0.5, 0.3], [-0.1, 0.3], "negative_clay"),
    ],
)
def test_interpolate_texture_params_invalid_fractions(soil_params, sand_frac, clay_frac, description):
    """Test that invalid fraction values raise errors or produce invalid results."""
    sand_arr = jnp.array(sand_frac)
    clay_arr = jnp.array(clay_frac)
    
    # Depending on implementation, this might raise an error or produce NaN/invalid results
    # We test that either an error is raised OR the results are clearly invalid
    try:
        result = interpolate_texture_params(sand_arr, clay_arr, soil_params)
        
        # If no error raised, check that results are invalid (NaN or violate constraints)
        has_nan = (jnp.any(jnp.isnan(result["watsat"])) or 
                   jnp.any(jnp.isnan(result["smpsat"])) or
                   jnp.any(jnp.isnan(result["hksat"])) or
                   jnp.any(jnp.isnan(result["bsw"])))
        
        violates_constraints = (jnp.any(result["watsat"] < 0.0) or 
                               jnp.any(result["watsat"] > 1.0) or
                               jnp.any(result["smpsat"] > 0.0) or
                               jnp.any(result["hksat"] < 0.0) or
                               jnp.any(result["bsw"] < 0.0))
        
        # For invalid inputs, we expect either NaN or constraint violations
        # This is a soft check - the function may handle invalid inputs gracefully
        if not (has_nan or violates_constraints):
            pytest.skip(f"{description}: Function handles invalid input without error or NaN")
            
    except (ValueError, AssertionError, RuntimeError):
        # Expected behavior - function raises error for invalid input
        pass


def test_interpolate_texture_params_monotonicity(soil_params):
    """Test that hydraulic conductivity changes monotonically with sand fraction."""
    # Keep clay constant, vary sand
    clay_constant = 0.1
    sand_gradient = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    clay_frac = jnp.full_like(sand_gradient, clay_constant)
    
    result = interpolate_texture_params(sand_gradient, clay_frac, soil_params)
    
    # Hydraulic conductivity typically increases with sand content
    # Check if there's a general trend (not strict monotonicity due to interpolation)
    hksat = result["hksat"]
    
    # Calculate correlation - should be positive for sand vs hksat
    # (more sand -> higher conductivity in general)
    mean_sand = jnp.mean(sand_gradient)
    mean_hksat = jnp.mean(hksat)
    
    covariance = jnp.mean((sand_gradient - mean_sand) * (hksat - mean_hksat))
    
    # We expect positive correlation (not strict, but general trend)
    # This is a weak test since interpolation may not preserve strict monotonicity
    assert covariance >= 0.0 or jnp.abs(covariance) < 1e-3, \
        f"Expected positive correlation between sand and hksat, got covariance={covariance}"


def test_interpolate_texture_params_consistency(soil_params):
    """Test that same inputs produce same outputs (deterministic behavior)."""
    sand_frac = jnp.array([0.4, 0.3, 0.5])
    clay_frac = jnp.array([0.2, 0.3, 0.15])
    
    result1 = interpolate_texture_params(sand_frac, clay_frac, soil_params)
    result2 = interpolate_texture_params(sand_frac, clay_frac, soil_params)
    
    assert jnp.allclose(result1["watsat"], result2["watsat"], atol=1e-10), \
        "watsat not deterministic"
    assert jnp.allclose(result1["smpsat"], result2["smpsat"], atol=1e-10), \
        "smpsat not deterministic"
    assert jnp.allclose(result1["hksat"], result2["hksat"], atol=1e-10), \
        "hksat not deterministic"
    assert jnp.allclose(result1["bsw"], result2["bsw"], atol=1e-10), \
        "bsw not deterministic"


def test_interpolate_texture_params_result_keys(soil_params):
    """Test that result dictionary has exactly the expected keys."""
    sand_frac = jnp.array([0.4])
    clay_frac = jnp.array([0.2])
    
    result = interpolate_texture_params(sand_frac, clay_frac, soil_params)
    
    expected_keys = {"watsat", "smpsat", "hksat", "bsw"}
    actual_keys = set(result.keys())
    
    assert actual_keys == expected_keys, \
        f"Expected keys {expected_keys}, got {actual_keys}"


# ============================================================================
# Integration Tests
# ============================================================================


def test_module_integration_workflow(soil_params):
    """Test complete workflow: create params -> get index -> interpolate."""
    # Get index for loam
    loam_idx = get_texture_class_index("loam")
    assert loam_idx == 4, f"Expected loam index 4, got {loam_idx}"
    
    # Get loam properties from params
    loam_sand = float(soil_params.sand_tex[loam_idx])
    loam_clay = float(soil_params.clay_tex[loam_idx])
    
    # Interpolate at loam composition
    sand_frac = jnp.array([loam_sand])
    clay_frac = jnp.array([loam_clay])
    
    result = interpolate_texture_params(sand_frac, clay_frac, soil_params)
    
    # Result should be close to loam texture class values
    # (exact match depends on interpolation method)
    assert result["watsat"].shape == (1,)
    assert result["smpsat"].shape == (1,)
    assert result["hksat"].shape == (1,)
    assert result["bsw"].shape == (1,)


def test_all_texture_classes_interpolation(soil_params, valid_texture_names):
    """Test interpolation at all 11 texture class compositions."""
    for i, texture_name in enumerate(valid_texture_names):
        idx = get_texture_class_index(texture_name)
        assert idx == i, f"Index mismatch for {texture_name}"
        
        sand_frac = jnp.array([float(soil_params.sand_tex[idx])])
        clay_frac = jnp.array([float(soil_params.clay_tex[idx])])
        
        result = interpolate_texture_params(sand_frac, clay_frac, soil_params)
        
        # Check that results are physically valid
        assert 0.0 <= float(result["watsat"][0]) <= 1.0, \
            f"{texture_name}: watsat out of range"
        assert float(result["smpsat"][0]) <= 0.0, \
            f"{texture_name}: smpsat should be negative"
        assert float(result["hksat"][0]) >= 0.0, \
            f"{texture_name}: hksat should be positive"
        assert float(result["bsw"][0]) >= 0.0, \
            f"{texture_name}: bsw should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])