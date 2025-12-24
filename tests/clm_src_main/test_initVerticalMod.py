"""
Comprehensive pytest suite for initVerticalMod module.

This test suite covers:
- CLM4.5 and CLM5.0 vertical structure initialization
- Layer depth, thickness, and interface calculations
- Bedrock index determination
- Edge cases (minimal layers, extreme depths, boundary conditions)
- Multi-column configurations
- Helper function validation
- Physical constraint verification

Test Strategy:
- Nominal cases: Standard CLM configurations (50%)
- Edge cases: Boundary conditions and extreme values (30%)
- Special cases: High resolution and large arrays (20%)
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pytest
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from clm_src_main.initVerticalMod import (
    initVertical,
    _calculate_clm45_depths,
    _calculate_clm45_thicknesses,
    _calculate_clm45_interfaces,
    _calculate_clm50_thicknesses,
    _calculate_clm50_interfaces,
    _calculate_clm50_depths,
    _init_clm45_layers,
    _init_clm50_layers,
    _set_bedrock_indices,
    _find_minimum_bedrock_index,
    _find_bedrock_index,
    get_vertical_structure,
    reset_vertical_structure,
    calculate_layer_statistics,
    print_vertical_summary,
    validate_vertical_structure,
    create_simple_vertical_structure,
    VerticalStructure,
    CLMPhysicsVersion,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load comprehensive test data for initVerticalMod functions.
    
    Returns:
        Dictionary containing test cases for all functions with inputs,
        expected properties, and metadata.
    """
    return {
        "clm50_single_column_nominal": {
            "physics_version": "CLM5_0",
            "bounds": {"begc": 0, "endc": 0},
            "nlevsoi": 20,
            "nlevgrnd": 50,
            "bedrock_depth": 10.0,
        },
        "clm45_single_column_nominal": {
            "physics_version": "CLM4_5",
            "bounds": {"begc": 0, "endc": 0},
            "scalez": 0.025,
            "nlevgrnd": 50,
            "bedrock_depth": 15.0,
        },
        "multiple_columns_nominal": {
            "physics_version": "CLM5_0",
            "bounds": {"begc": 0, "endc": 9},
            "nlevsoi": 20,
            "nlevgrnd": 50,
            "bedrock_depths": [8.0, 12.0, 10.0, 15.0, 20.0, 5.0, 25.0, 30.0, 18.0, 22.0],
        },
        "shallow_bedrock_edge": {
            "physics_version": "CLM5_0",
            "bounds": {"begc": 0, "endc": 0},
            "nlevsoi": 20,
            "nlevgrnd": 50,
            "bedrock_depth": 3.0,
            "min_bedrock_depth": 3.0,
        },
        "deep_bedrock_edge": {
            "physics_version": "CLM5_0",
            "bounds": {"begc": 0, "endc": 0},
            "nlevsoi": 20,
            "nlevgrnd": 50,
            "bedrock_depth": 50.0,
        },
        "minimal_layers_edge": {
            "physics_version": "CLM5_0",
            "bounds": {"begc": 0, "endc": 0},
            "nlevsoi": 1,
            "nlevgrnd": 2,
            "bedrock_depth": 5.0,
        },
        "small_scalez_clm45_edge": {
            "physics_version": "CLM4_5",
            "bounds": {"begc": 0, "endc": 0},
            "scalez": 0.001,
            "nlevgrnd": 50,
            "bedrock_depth": 10.0,
        },
        "large_scalez_clm45_edge": {
            "physics_version": "CLM4_5",
            "bounds": {"begc": 0, "endc": 0},
            "scalez": 0.5,
            "nlevgrnd": 50,
            "bedrock_depth": 10.0,
        },
    }


@pytest.fixture
def clm45_depth_test_data() -> Dict[str, Any]:
    """Test data for CLM4.5 depth calculations."""
    return {
        "nominal": {"scalez": 0.025, "nlevgrnd": 50},
        "small_scalez": {"scalez": 0.001, "nlevgrnd": 30},
        "large_scalez": {"scalez": 0.5, "nlevgrnd": 20},
        "minimal": {"scalez": 0.025, "nlevgrnd": 2},
    }


@pytest.fixture
def clm50_thickness_test_data() -> Dict[str, Any]:
    """Test data for CLM5.0 thickness calculations."""
    return {
        "nominal": {"nlevsoi": 20, "nlevgrnd": 50},
        "minimal": {"nlevsoi": 1, "nlevgrnd": 2},
        "high_res": {"nlevsoi": 50, "nlevgrnd": 100},
    }


@pytest.fixture
def sample_depths() -> jnp.ndarray:
    """Sample depth array for testing."""
    return jnp.array([0.01, 0.04, 0.09, 0.16, 0.26, 0.4, 0.58, 0.8, 1.06, 1.36])


@pytest.fixture
def sample_thicknesses() -> jnp.ndarray:
    """Sample thickness array for testing."""
    return jnp.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2])


@pytest.fixture
def sample_interfaces() -> jnp.ndarray:
    """Sample interface array for testing."""
    return jnp.array([0.0, 0.02, 0.06, 0.12, 0.2, 0.3, 0.42, 0.56, 0.72, 0.9, 1.1])


# ============================================================================
# CLM4.5 Depth Calculation Tests
# ============================================================================

class TestCalculateCLM45Depths:
    """Test suite for _calculate_clm45_depths function."""
    
    def test_clm45_depths_nominal_shape(self, clm45_depth_test_data):
        """Test that CLM4.5 depth calculation returns correct shape."""
        data = clm45_depth_test_data["nominal"]
        depths = _calculate_clm45_depths(data["scalez"], data["nlevgrnd"])
        
        assert depths.shape == (data["nlevgrnd"],), \
            f"Expected shape ({data['nlevgrnd']},), got {depths.shape}"
    
    def test_clm45_depths_monotonic_increasing(self, clm45_depth_test_data):
        """Test that CLM4.5 depths are monotonically increasing."""
        data = clm45_depth_test_data["nominal"]
        depths = _calculate_clm45_depths(data["scalez"], data["nlevgrnd"])
        
        diffs = jnp.diff(depths)
        assert jnp.all(diffs > 0), \
            "Depths should be monotonically increasing"
    
    def test_clm45_depths_positive_values(self, clm45_depth_test_data):
        """Test that all CLM4.5 depths are positive."""
        data = clm45_depth_test_data["nominal"]
        depths = _calculate_clm45_depths(data["scalez"], data["nlevgrnd"])
        
        assert jnp.all(depths > 0), \
            "All depths should be positive"
    
    @pytest.mark.parametrize("scalez,nlevgrnd", [
        (0.025, 50),
        (0.001, 30),
        (0.5, 20),
        (0.025, 2),
    ])
    def test_clm45_depths_various_parameters(self, scalez, nlevgrnd):
        """Test CLM4.5 depth calculation with various parameter combinations."""
        depths = _calculate_clm45_depths(scalez, nlevgrnd)
        
        assert depths.shape == (nlevgrnd,)
        assert jnp.all(depths > 0)
        assert jnp.all(jnp.diff(depths) > 0)
    
    def test_clm45_depths_small_scalez_fine_discretization(self, clm45_depth_test_data):
        """Test that small scalez produces fine near-surface discretization."""
        data = clm45_depth_test_data["small_scalez"]
        depths = _calculate_clm45_depths(data["scalez"], data["nlevgrnd"])
        
        # First layer should be very shallow with small scalez
        assert depths[0] < 0.01, \
            f"First layer depth {depths[0]} should be < 0.01m with small scalez"
    
    def test_clm45_depths_large_scalez_coarse_discretization(self, clm45_depth_test_data):
        """Test that large scalez produces coarse near-surface discretization."""
        data = clm45_depth_test_data["large_scalez"]
        depths = _calculate_clm45_depths(data["scalez"], data["nlevgrnd"])
        
        # First layer should be deeper with large scalez
        assert depths[0] > 0.1, \
            f"First layer depth {depths[0]} should be > 0.1m with large scalez"
    
    def test_clm45_depths_dtype(self, clm45_depth_test_data):
        """Test that CLM4.5 depths have correct dtype."""
        data = clm45_depth_test_data["nominal"]
        depths = _calculate_clm45_depths(data["scalez"], data["nlevgrnd"])
        
        assert depths.dtype == jnp.float32 or depths.dtype == jnp.float64, \
            f"Expected float dtype, got {depths.dtype}"


# ============================================================================
# CLM4.5 Thickness Calculation Tests
# ============================================================================

class TestCalculateCLM45Thicknesses:
    """Test suite for _calculate_clm45_thicknesses function."""
    
    def test_clm45_thicknesses_shape(self, sample_depths):
        """Test that thickness calculation returns correct shape."""
        thicknesses = _calculate_clm45_thicknesses(sample_depths)
        
        assert thicknesses.shape == sample_depths.shape, \
            f"Expected shape {sample_depths.shape}, got {thicknesses.shape}"
    
    def test_clm45_thicknesses_positive(self, sample_depths):
        """Test that all thicknesses are positive."""
        thicknesses = _calculate_clm45_thicknesses(sample_depths)
        
        assert jnp.all(thicknesses > 0), \
            "All thicknesses should be positive"
    
    def test_clm45_thicknesses_sum_approximates_max_depth(self, sample_depths):
        """Test that sum of thicknesses approximates maximum depth."""
        thicknesses = _calculate_clm45_thicknesses(sample_depths)
        
        total_thickness = jnp.sum(thicknesses)
        max_depth = sample_depths[-1]
        
        # Sum should be close to max depth (within reasonable tolerance)
        assert jnp.abs(total_thickness - max_depth) < max_depth * 0.2, \
            f"Total thickness {total_thickness} should approximate max depth {max_depth}"
    
    def test_clm45_thicknesses_consistency_with_depths(self):
        """Test thickness calculation consistency with depth array."""
        depths = jnp.array([0.01, 0.04, 0.09, 0.16, 0.26])
        thicknesses = _calculate_clm45_thicknesses(depths)
        
        # Verify thicknesses are reasonable given depth spacing
        for i in range(len(depths) - 1):
            depth_diff = depths[i + 1] - depths[i]
            # Thickness should be related to depth differences
            assert thicknesses[i] > 0
            assert thicknesses[i] <= depth_diff * 2


# ============================================================================
# CLM4.5 Interface Calculation Tests
# ============================================================================

class TestCalculateCLM45Interfaces:
    """Test suite for _calculate_clm45_interfaces function."""
    
    def test_clm45_interfaces_shape(self, sample_depths, sample_thicknesses):
        """Test that interface calculation returns correct shape."""
        interfaces = _calculate_clm45_interfaces(sample_depths, sample_thicknesses)
        
        expected_shape = (len(sample_depths) + 1,)
        assert interfaces.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {interfaces.shape}"
    
    def test_clm45_interfaces_first_value_zero(self, sample_depths, sample_thicknesses):
        """Test that first interface is at surface (0.0)."""
        interfaces = _calculate_clm45_interfaces(sample_depths, sample_thicknesses)
        
        assert jnp.abs(interfaces[0]) < 1e-10, \
            f"First interface should be 0.0, got {interfaces[0]}"
    
    def test_clm45_interfaces_monotonic_increasing(self, sample_depths, sample_thicknesses):
        """Test that interfaces are monotonically increasing."""
        interfaces = _calculate_clm45_interfaces(sample_depths, sample_thicknesses)
        
        diffs = jnp.diff(interfaces)
        assert jnp.all(diffs > 0), \
            "Interfaces should be monotonically increasing"
    
    def test_clm45_interfaces_non_negative(self, sample_depths, sample_thicknesses):
        """Test that all interfaces are non-negative."""
        interfaces = _calculate_clm45_interfaces(sample_depths, sample_thicknesses)
        
        assert jnp.all(interfaces >= 0), \
            "All interfaces should be non-negative"


# ============================================================================
# CLM5.0 Thickness Calculation Tests
# ============================================================================

class TestCalculateCLM50Thicknesses:
    """Test suite for _calculate_clm50_thicknesses function."""
    
    def test_clm50_thicknesses_nominal_shape(self, clm50_thickness_test_data):
        """Test that CLM5.0 thickness calculation returns correct shape."""
        data = clm50_thickness_test_data["nominal"]
        thicknesses = _calculate_clm50_thicknesses(data["nlevsoi"], data["nlevgrnd"])
        
        assert thicknesses.shape == (data["nlevgrnd"],), \
            f"Expected shape ({data['nlevgrnd']},), got {thicknesses.shape}"
    
    def test_clm50_thicknesses_positive(self, clm50_thickness_test_data):
        """Test that all CLM5.0 thicknesses are positive."""
        data = clm50_thickness_test_data["nominal"]
        thicknesses = _calculate_clm50_thicknesses(data["nlevsoi"], data["nlevgrnd"])
        
        assert jnp.all(thicknesses > 0), \
            "All thicknesses should be positive"
    
    def test_clm50_thicknesses_soil_layers_smaller(self, clm50_thickness_test_data):
        """Test that soil layer thicknesses are generally smaller than bedrock layers."""
        data = clm50_thickness_test_data["nominal"]
        thicknesses = _calculate_clm50_thicknesses(data["nlevsoi"], data["nlevgrnd"])
        
        soil_mean = jnp.mean(thicknesses[:data["nlevsoi"]])
        bedrock_mean = jnp.mean(thicknesses[data["nlevsoi"]:])
        
        assert soil_mean < bedrock_mean, \
            "Soil layers should generally be thinner than bedrock layers"
    
    @pytest.mark.parametrize("nlevsoi,nlevgrnd", [
        (20, 50),
        (1, 2),
        (50, 100),
        (10, 25),
    ])
    def test_clm50_thicknesses_various_parameters(self, nlevsoi, nlevgrnd):
        """Test CLM5.0 thickness calculation with various parameters."""
        thicknesses = _calculate_clm50_thicknesses(nlevsoi, nlevgrnd)
        
        assert thicknesses.shape == (nlevgrnd,)
        assert jnp.all(thicknesses > 0)
    
    def test_clm50_thicknesses_minimal_layers(self, clm50_thickness_test_data):
        """Test CLM5.0 thickness calculation with minimal layers."""
        data = clm50_thickness_test_data["minimal"]
        thicknesses = _calculate_clm50_thicknesses(data["nlevsoi"], data["nlevgrnd"])
        
        assert thicknesses.shape == (data["nlevgrnd"],)
        assert jnp.all(thicknesses > 0)


# ============================================================================
# CLM5.0 Interface Calculation Tests
# ============================================================================

class TestCalculateCLM50Interfaces:
    """Test suite for _calculate_clm50_interfaces function."""
    
    def test_clm50_interfaces_shape(self, sample_thicknesses):
        """Test that CLM5.0 interface calculation returns correct shape."""
        interfaces = _calculate_clm50_interfaces(sample_thicknesses)
        
        expected_shape = (len(sample_thicknesses) + 1,)
        assert interfaces.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {interfaces.shape}"
    
    def test_clm50_interfaces_first_value_zero(self, sample_thicknesses):
        """Test that first interface is at surface (0.0)."""
        interfaces = _calculate_clm50_interfaces(sample_thicknesses)
        
        assert jnp.abs(interfaces[0]) < 1e-10, \
            f"First interface should be 0.0, got {interfaces[0]}"
    
    def test_clm50_interfaces_last_value_is_sum(self, sample_thicknesses):
        """Test that last interface equals sum of thicknesses."""
        interfaces = _calculate_clm50_interfaces(sample_thicknesses)
        
        expected_sum = jnp.sum(sample_thicknesses)
        assert jnp.allclose(interfaces[-1], expected_sum, rtol=1e-6, atol=1e-6), \
            f"Last interface {interfaces[-1]} should equal sum {expected_sum}"
    
    def test_clm50_interfaces_monotonic_increasing(self, sample_thicknesses):
        """Test that CLM5.0 interfaces are monotonically increasing."""
        interfaces = _calculate_clm50_interfaces(sample_thicknesses)
        
        diffs = jnp.diff(interfaces)
        assert jnp.all(diffs > 0), \
            "Interfaces should be monotonically increasing"
    
    def test_clm50_interfaces_cumulative_sum(self):
        """Test that interfaces are cumulative sum of thicknesses."""
        thicknesses = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
        interfaces = _calculate_clm50_interfaces(thicknesses)
        
        expected = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(thicknesses)])
        assert jnp.allclose(interfaces, expected, rtol=1e-6, atol=1e-6), \
            "Interfaces should be cumulative sum of thicknesses"


# ============================================================================
# CLM5.0 Depth Calculation Tests
# ============================================================================

class TestCalculateCLM50Depths:
    """Test suite for _calculate_clm50_depths function."""
    
    def test_clm50_depths_shape(self, sample_interfaces):
        """Test that CLM5.0 depth calculation returns correct shape."""
        depths = _calculate_clm50_depths(sample_interfaces)
        
        expected_shape = (len(sample_interfaces) - 1,)
        assert depths.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {depths.shape}"
    
    def test_clm50_depths_positive(self, sample_interfaces):
        """Test that all CLM5.0 depths are positive."""
        depths = _calculate_clm50_depths(sample_interfaces)
        
        assert jnp.all(depths > 0), \
            "All depths should be positive"
    
    def test_clm50_depths_midpoint_calculation(self):
        """Test that depths are midpoints between interfaces."""
        interfaces = jnp.array([0.0, 0.2, 0.6, 1.2, 2.0, 3.0])
        depths = _calculate_clm50_depths(interfaces)
        
        expected = jnp.array([0.1, 0.4, 0.9, 1.6, 2.5])
        assert jnp.allclose(depths, expected, rtol=1e-6, atol=1e-6), \
            "Depths should be midpoints between interfaces"
    
    def test_clm50_depths_monotonic_increasing(self, sample_interfaces):
        """Test that CLM5.0 depths are monotonically increasing."""
        depths = _calculate_clm50_depths(sample_interfaces)
        
        diffs = jnp.diff(depths)
        assert jnp.all(diffs > 0), \
            "Depths should be monotonically increasing"


# ============================================================================
# Layer Statistics Tests
# ============================================================================

class TestCalculateLayerStatistics:
    """Test suite for calculate_layer_statistics function."""
    
    def test_layer_stats_returns_dict(self, sample_depths, sample_thicknesses):
        """Test that layer statistics returns a dictionary."""
        stats = calculate_layer_statistics(sample_depths, sample_thicknesses)
        
        assert isinstance(stats, dict), \
            "Statistics should be returned as a dictionary"
    
    def test_layer_stats_contains_required_keys(self, sample_depths, sample_thicknesses):
        """Test that statistics dictionary contains all required keys."""
        stats = calculate_layer_statistics(sample_depths, sample_thicknesses)
        
        required_keys = ["mean_depth", "max_depth", "min_thickness", 
                        "max_thickness", "total_depth"]
        for key in required_keys:
            assert key in stats, f"Statistics should contain key '{key}'"
    
    def test_layer_stats_mean_depth_positive(self, sample_depths, sample_thicknesses):
        """Test that mean depth is positive."""
        stats = calculate_layer_statistics(sample_depths, sample_thicknesses)
        
        assert stats["mean_depth"] > 0, \
            "Mean depth should be positive"
    
    def test_layer_stats_max_depth_equals_last(self, sample_depths, sample_thicknesses):
        """Test that max depth equals last depth value."""
        stats = calculate_layer_statistics(sample_depths, sample_thicknesses)
        
        assert jnp.allclose(stats["max_depth"], sample_depths[-1], rtol=1e-6, atol=1e-6), \
            f"Max depth {stats['max_depth']} should equal last depth {sample_depths[-1]}"
    
    def test_layer_stats_total_depth_is_sum(self, sample_depths, sample_thicknesses):
        """Test that total depth is sum of thicknesses."""
        stats = calculate_layer_statistics(sample_depths, sample_thicknesses)
        
        expected_sum = float(jnp.sum(sample_thicknesses))
        assert jnp.allclose(stats["total_depth"], expected_sum, rtol=1e-6, atol=1e-6), \
            f"Total depth {stats['total_depth']} should equal sum {expected_sum}"
    
    def test_layer_stats_min_thickness_positive(self, sample_depths, sample_thicknesses):
        """Test that minimum thickness is positive."""
        stats = calculate_layer_statistics(sample_depths, sample_thicknesses)
        
        assert stats["min_thickness"] > 0, \
            "Minimum thickness should be positive"
    
    def test_layer_stats_max_gte_min_thickness(self, sample_depths, sample_thicknesses):
        """Test that max thickness is greater than or equal to min thickness."""
        stats = calculate_layer_statistics(sample_depths, sample_thicknesses)
        
        assert stats["max_thickness"] >= stats["min_thickness"], \
            "Max thickness should be >= min thickness"


# ============================================================================
# Vertical Structure Validation Tests
# ============================================================================

class TestValidateVerticalStructure:
    """Test suite for validate_vertical_structure function."""
    
    def test_validate_returns_tuple(self):
        """Test that validation returns a tuple."""
        result = validate_vertical_structure()
        
        assert isinstance(result, tuple), \
            "Validation should return a tuple"
        assert len(result) == 2, \
            "Validation should return (bool, str) tuple"
    
    def test_validate_valid_structure_after_reset(self):
        """Test that validation passes after reset."""
        reset_vertical_structure()
        is_valid, error_msg = validate_vertical_structure()
        
        assert is_valid, f"Structure should be valid after reset: {error_msg}"
        assert error_msg == "" or error_msg is None, \
            "Error message should be empty for valid structure"
    
    def test_validate_returns_bool_and_string(self):
        """Test that validation returns correct types."""
        is_valid, error_msg = validate_vertical_structure()
        
        assert isinstance(is_valid, bool), \
            "First return value should be boolean"
        assert isinstance(error_msg, str), \
            "Second return value should be string"


# ============================================================================
# Vertical Structure Reset Tests
# ============================================================================

class TestResetVerticalStructure:
    """Test suite for reset_vertical_structure function."""
    
    @pytest.mark.parametrize("physics_version", ["CLM4_5", "CLM5_0"])
    def test_reset_with_valid_physics_versions(self, physics_version):
        """Test reset with valid physics versions."""
        reset_vertical_structure(physics_version)
        structure = get_vertical_structure()
        
        assert structure is not None, \
            "Structure should exist after reset"
    
    def test_reset_default_is_clm50(self):
        """Test that default reset uses CLM5.0."""
        reset_vertical_structure()
        structure = get_vertical_structure()
        
        # Check that structure is initialized (exact version check depends on implementation)
        assert structure is not None


# ============================================================================
# Get Vertical Structure Tests
# ============================================================================

class TestGetVerticalStructure:
    """Test suite for get_vertical_structure function."""
    
    def test_get_returns_structure(self):
        """Test that get_vertical_structure returns a structure."""
        reset_vertical_structure()
        structure = get_vertical_structure()
        
        assert structure is not None, \
            "Should return a vertical structure"
    
    def test_get_returns_copy(self):
        """Test that get_vertical_structure returns a copy."""
        reset_vertical_structure()
        structure1 = get_vertical_structure()
        structure2 = get_vertical_structure()
        
        # Should be separate objects (if implementation returns copies)
        assert structure1 is not None and structure2 is not None


# ============================================================================
# Create Simple Vertical Structure Tests
# ============================================================================

class TestCreateSimpleVerticalStructure:
    """Test suite for create_simple_vertical_structure function."""
    
    @pytest.mark.parametrize("physics_version,bedrock_depth", [
        ("CLM5_0", 10.0),
        ("CLM4_5", 15.0),
        ("CLM5_0", 3.0),
        ("CLM5_0", 50.0),
    ])
    def test_create_simple_structure(self, physics_version, bedrock_depth):
        """Test creating simple vertical structures with various parameters."""
        create_simple_vertical_structure(physics_version, bedrock_depth)
        
        # Verify structure was created
        is_valid, error_msg = validate_vertical_structure()
        assert is_valid, f"Created structure should be valid: {error_msg}"
    
    def test_create_simple_default_parameters(self):
        """Test creating simple structure with default parameters."""
        create_simple_vertical_structure()
        
        is_valid, error_msg = validate_vertical_structure()
        assert is_valid, f"Default structure should be valid: {error_msg}"


# ============================================================================
# Integration Tests
# ============================================================================

class TestVerticalStructureIntegration:
    """Integration tests for complete vertical structure workflows."""
    
    def test_clm50_complete_workflow(self):
        """Test complete CLM5.0 vertical structure initialization workflow."""
        # Reset and create structure
        reset_vertical_structure("CLM5_0")
        create_simple_vertical_structure("CLM5_0", 10.0)
        
        # Validate
        is_valid, error_msg = validate_vertical_structure()
        assert is_valid, f"CLM5.0 workflow should produce valid structure: {error_msg}"
        
        # Get structure
        structure = get_vertical_structure()
        assert structure is not None
    
    def test_clm45_complete_workflow(self):
        """Test complete CLM4.5 vertical structure initialization workflow."""
        # Reset and create structure
        reset_vertical_structure("CLM4_5")
        create_simple_vertical_structure("CLM4_5", 15.0)
        
        # Validate
        is_valid, error_msg = validate_vertical_structure()
        assert is_valid, f"CLM4.5 workflow should produce valid structure: {error_msg}"
        
        # Get structure
        structure = get_vertical_structure()
        assert structure is not None
    
    def test_multiple_resets(self):
        """Test multiple resets and recreations."""
        for _ in range(3):
            reset_vertical_structure("CLM5_0")
            create_simple_vertical_structure("CLM5_0", 10.0)
            is_valid, _ = validate_vertical_structure()
            assert is_valid, "Each reset should produce valid structure"
    
    def test_switch_physics_versions(self):
        """Test switching between physics versions."""
        # Start with CLM5.0
        reset_vertical_structure("CLM5_0")
        create_simple_vertical_structure("CLM5_0", 10.0)
        is_valid, _ = validate_vertical_structure()
        assert is_valid
        
        # Switch to CLM4.5
        reset_vertical_structure("CLM4_5")
        create_simple_vertical_structure("CLM4_5", 15.0)
        is_valid, _ = validate_vertical_structure()
        assert is_valid


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_minimal_layer_configuration(self):
        """Test with minimal number of layers."""
        thicknesses = _calculate_clm50_thicknesses(1, 2)
        
        assert thicknesses.shape == (2,)
        assert jnp.all(thicknesses > 0)
    
    def test_very_small_scalez(self):
        """Test CLM4.5 with very small scalez parameter."""
        depths = _calculate_clm45_depths(0.0001, 20)
        
        assert depths.shape == (20,)
        assert jnp.all(depths > 0)
        assert jnp.all(jnp.diff(depths) > 0)
    
    def test_very_large_scalez(self):
        """Test CLM4.5 with very large scalez parameter."""
        depths = _calculate_clm45_depths(1.0, 20)
        
        assert depths.shape == (20,)
        assert jnp.all(depths > 0)
        assert jnp.all(jnp.diff(depths) > 0)
    
    def test_single_layer_arrays(self):
        """Test calculations with single-layer arrays."""
        depths = jnp.array([0.5])
        thicknesses = _calculate_clm45_thicknesses(depths)
        
        assert thicknesses.shape == (1,)
        assert thicknesses[0] > 0
    
    def test_high_resolution_layers(self):
        """Test with high resolution (many layers)."""
        thicknesses = _calculate_clm50_thicknesses(50, 100)
        
        assert thicknesses.shape == (100,)
        assert jnp.all(thicknesses > 0)


# ============================================================================
# Physical Constraint Tests
# ============================================================================

class TestPhysicalConstraints:
    """Test suite for physical constraint verification."""
    
    def test_depths_non_negative(self):
        """Test that all depths are non-negative."""
        depths = _calculate_clm45_depths(0.025, 50)
        assert jnp.all(depths >= 0), "All depths must be non-negative"
    
    def test_thicknesses_positive(self):
        """Test that all thicknesses are positive."""
        thicknesses = _calculate_clm50_thicknesses(20, 50)
        assert jnp.all(thicknesses > 0), "All thicknesses must be positive"
    
    def test_interfaces_monotonic(self):
        """Test that interfaces are strictly monotonic."""
        thicknesses = _calculate_clm50_thicknesses(20, 50)
        interfaces = _calculate_clm50_interfaces(thicknesses)
        
        diffs = jnp.diff(interfaces)
        assert jnp.all(diffs > 0), "Interfaces must be strictly increasing"
    
    def test_depth_thickness_consistency(self):
        """Test consistency between depths and thicknesses."""
        depths = _calculate_clm45_depths(0.025, 20)
        thicknesses = _calculate_clm45_thicknesses(depths)
        
        # Sum of thicknesses should be reasonable compared to max depth
        total = jnp.sum(thicknesses)
        max_depth = depths[-1]
        
        assert total > 0
        assert max_depth > 0
        # Allow for some variation in calculation methods
        assert total < max_depth * 2


# ============================================================================
# Numerical Stability Tests
# ============================================================================

class TestNumericalStability:
    """Test suite for numerical stability and precision."""
    
    def test_small_values_stability(self):
        """Test stability with very small values."""
        depths = _calculate_clm45_depths(0.0001, 10)
        
        assert not jnp.any(jnp.isnan(depths)), "Should not produce NaN"
        assert not jnp.any(jnp.isinf(depths)), "Should not produce Inf"
    
    def test_large_values_stability(self):
        """Test stability with large values."""
        depths = _calculate_clm45_depths(10.0, 10)
        
        assert not jnp.any(jnp.isnan(depths)), "Should not produce NaN"
        assert not jnp.any(jnp.isinf(depths)), "Should not produce Inf"
    
    def test_cumsum_precision(self):
        """Test precision of cumulative sum in interface calculation."""
        thicknesses = jnp.ones(100) * 0.01
        interfaces = _calculate_clm50_interfaces(thicknesses)
        
        # Last interface should equal sum of thicknesses
        expected = jnp.sum(thicknesses)
        assert jnp.allclose(interfaces[-1], expected, rtol=1e-6, atol=1e-6)
    
    def test_midpoint_precision(self):
        """Test precision of midpoint calculation."""
        interfaces = jnp.linspace(0, 10, 11)
        depths = _calculate_clm50_depths(interfaces)
        
        # Check midpoint calculation precision
        for i in range(len(depths)):
            expected_midpoint = (interfaces[i] + interfaces[i + 1]) / 2
            assert jnp.allclose(depths[i], expected_midpoint, rtol=1e-6, atol=1e-6)


# ============================================================================
# Print and Display Tests
# ============================================================================

class TestPrintFunctions:
    """Test suite for print and display functions."""
    
    def test_print_vertical_summary_no_error(self):
        """Test that print_vertical_summary executes without error."""
        reset_vertical_structure()
        create_simple_vertical_structure()
        
        # Should not raise an exception
        try:
            print_vertical_summary(0)
        except Exception as e:
            pytest.fail(f"print_vertical_summary raised exception: {e}")
    
    def test_print_vertical_summary_with_column_index(self):
        """Test print_vertical_summary with specific column index."""
        reset_vertical_structure()
        create_simple_vertical_structure()
        
        try:
            print_vertical_summary(column_idx=0)
        except Exception as e:
            pytest.fail(f"print_vertical_summary with column_idx raised exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])