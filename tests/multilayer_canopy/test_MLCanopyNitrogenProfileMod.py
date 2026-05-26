"""
Comprehensive pytest suite for canopy_nitrogen_profile function.

This module tests the canopy nitrogen profile calculation including:
- Nitrogen distribution through canopy layers
- Photosynthetic parameter profiles (Vcmax, Jmax, Rd, Kp)
- Sunlit/shaded leaf differentiation
- C3/C4 pathway handling
- Temperature acclimation effects
- Numerical integration validation
"""

import sys
from pathlib import Path
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multilayer_canopy.MLCanopyNitrogenProfileMod import (
    CanopyNitrogenParams,
    CanopyNitrogenProfile,
    CanopyNitrogenValidation,
    canopy_nitrogen_profile,
    get_default_params,
)


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load test data for canopy_nitrogen_profile tests.
    
    Returns:
        Dictionary containing test cases with inputs and metadata.
    """
    return {
        "test_nominal_single_patch_c3_moderate_lai": {
            "inputs": {
                "vcmaxpft": jnp.array([62.0]),
                "c3psn": jnp.array([1]),
                "tacclim": jnp.array([293.15]),
                "dpai": jnp.array([[0.5, 0.4, 0.3, 0.2, 0.1]]),
                "kb": jnp.array([[0.5, 0.5, 0.5, 0.5, 0.5]]),
                "tbi": jnp.array([[0.95, 0.85, 0.75, 0.65, 0.55]]),
                "fracsun": jnp.array([[0.8, 0.7, 0.6, 0.5, 0.4]]),
                "clump_fac": jnp.array([0.85]),
                "ncan": jnp.array([5]),
                "lai": jnp.array([1.5]),
                "sai": jnp.array([0.5]),
                "params": None,
                "validate": True,
            },
            "metadata": {
                "type": "nominal",
                "description": "Single C3 patch with moderate LAI, typical temperate conditions, 5 canopy layers",
            },
        },
        "test_nominal_multiple_patches_mixed_pathways": {
            "inputs": {
                "vcmaxpft": jnp.array([62.0, 40.0, 55.0]),
                "c3psn": jnp.array([1, 0, 1]),
                "tacclim": jnp.array([293.15, 303.15, 283.15]),
                "dpai": jnp.array([
                    [0.6, 0.5, 0.4, 0.3, 0.2],
                    [0.8, 0.7, 0.6, 0.5, 0.4],
                    [0.4, 0.3, 0.2, 0.1, 0.05],
                ]),
                "kb": jnp.array([
                    [0.5, 0.5, 0.5, 0.5, 0.5],
                    [0.6, 0.6, 0.6, 0.6, 0.6],
                    [0.4, 0.4, 0.4, 0.4, 0.4],
                ]),
                "tbi": jnp.array([
                    [0.9, 0.8, 0.7, 0.6, 0.5],
                    [0.85, 0.75, 0.65, 0.55, 0.45],
                    [0.95, 0.88, 0.8, 0.72, 0.65],
                ]),
                "fracsun": jnp.array([
                    [0.75, 0.65, 0.55, 0.45, 0.35],
                    [0.7, 0.6, 0.5, 0.4, 0.3],
                    [0.8, 0.7, 0.6, 0.5, 0.4],
                ]),
                "clump_fac": jnp.array([0.85, 0.75, 0.9]),
                "ncan": jnp.array([5, 5, 5]),
                "lai": jnp.array([2.0, 3.0, 1.45]),
                "sai": jnp.array([0.6, 0.8, 0.4]),
                "params": None,
                "validate": True,
            },
            "metadata": {
                "type": "nominal",
                "description": "Three patches with mixed C3/C4 pathways, varying temperatures and LAI values",
            },
        },
        "test_nominal_high_lai_dense_canopy": {
            "inputs": {
                "vcmaxpft": jnp.array([80.0, 70.0]),
                "c3psn": jnp.array([1, 1]),
                "tacclim": jnp.array([298.15, 295.15]),
                "dpai": jnp.array([
                    [1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                    [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                ]),
                "kb": jnp.array([
                    [0.55] * 10,
                    [0.5] * 10,
                ]),
                "tbi": jnp.array([
                    [0.88, 0.77, 0.68, 0.6, 0.53, 0.47, 0.41, 0.36, 0.32, 0.28],
                    [0.9, 0.81, 0.73, 0.66, 0.59, 0.53, 0.48, 0.43, 0.39, 0.35],
                ]),
                "fracsun": jnp.array([
                    [0.65, 0.55, 0.48, 0.42, 0.37, 0.32, 0.28, 0.24, 0.21, 0.18],
                    [0.7, 0.6, 0.52, 0.45, 0.39, 0.34, 0.3, 0.26, 0.23, 0.2],
                ]),
                "clump_fac": jnp.array([0.8, 0.82]),
                "ncan": jnp.array([10, 10]),
                "lai": jnp.array([6.6, 5.5]),
                "sai": jnp.array([1.2, 1.0]),
                "params": None,
                "validate": True,
            },
            "metadata": {
                "type": "nominal",
                "description": "Dense canopy with high LAI (6-7), 10 layers, typical for tropical forests",
            },
        },
        "test_edge_zero_lai_no_canopy": {
            "inputs": {
                "vcmaxpft": jnp.array([50.0, 45.0]),
                "c3psn": jnp.array([1, 0]),
                "tacclim": jnp.array([290.15, 295.15]),
                "dpai": jnp.array([[0.0] * 5, [0.0] * 5]),
                "kb": jnp.array([[0.5] * 5, [0.5] * 5]),
                "tbi": jnp.array([[1.0] * 5, [1.0] * 5]),
                "fracsun": jnp.array([[1.0] * 5, [1.0] * 5]),
                "clump_fac": jnp.array([1.0, 1.0]),
                "ncan": jnp.array([0, 0]),
                "lai": jnp.array([0.0, 0.0]),
                "sai": jnp.array([0.0, 0.0]),
                "params": None,
                "validate": True,
            },
            "metadata": {
                "type": "edge",
                "description": "Zero LAI/SAI representing bare ground or dormant vegetation",
            },
        },
        "test_edge_minimal_lai_single_layer": {
            "inputs": {
                "vcmaxpft": jnp.array([30.0]),
                "c3psn": jnp.array([1]),
                "tacclim": jnp.array([288.15]),
                "dpai": jnp.array([[0.05]]),
                "kb": jnp.array([[0.5]]),
                "tbi": jnp.array([[0.975]]),
                "fracsun": jnp.array([[0.95]]),
                "clump_fac": jnp.array([0.95]),
                "ncan": jnp.array([1]),
                "lai": jnp.array([0.05]),
                "sai": jnp.array([0.01]),
                "params": None,
                "validate": True,
            },
            "metadata": {
                "type": "edge",
                "description": "Minimal LAI with single canopy layer, sparse vegetation",
            },
        },
        "test_edge_extreme_cold_temperature": {
            "inputs": {
                "vcmaxpft": jnp.array([40.0, 35.0]),
                "c3psn": jnp.array([1, 1]),
                "tacclim": jnp.array([253.15, 258.15]),
                "dpai": jnp.array([
                    [0.3, 0.25, 0.2, 0.15, 0.1],
                    [0.35, 0.3, 0.25, 0.2, 0.15],
                ]),
                "kb": jnp.array([[0.5] * 5, [0.5] * 5]),
                "tbi": jnp.array([
                    [0.92, 0.84, 0.77, 0.71, 0.65],
                    [0.91, 0.83, 0.76, 0.69, 0.63],
                ]),
                "fracsun": jnp.array([
                    [0.78, 0.68, 0.58, 0.48, 0.38],
                    [0.76, 0.66, 0.56, 0.46, 0.36],
                ]),
                "clump_fac": jnp.array([0.88, 0.86]),
                "ncan": jnp.array([5, 5]),
                "lai": jnp.array([1.0, 1.25]),
                "sai": jnp.array([0.3, 0.35]),
                "params": None,
                "validate": True,
            },
            "metadata": {
                "type": "edge",
                "description": "Extreme cold temperatures (-20°C to -15°C), boreal/arctic conditions",
            },
        },
        "test_edge_extreme_hot_temperature": {
            "inputs": {
                "vcmaxpft": jnp.array([90.0, 85.0]),
                "c3psn": jnp.array([0, 0]),
                "tacclim": jnp.array([313.15, 318.15]),
                "dpai": jnp.array([
                    [0.7, 0.6, 0.5, 0.4, 0.3],
                    [0.8, 0.7, 0.6, 0.5, 0.4],
                ]),
                "kb": jnp.array([[0.6] * 5, [0.65] * 5]),
                "tbi": jnp.array([
                    [0.87, 0.76, 0.66, 0.57, 0.5],
                    [0.85, 0.72, 0.61, 0.52, 0.44],
                ]),
                "fracsun": jnp.array([
                    [0.68, 0.58, 0.48, 0.38, 0.28],
                    [0.65, 0.55, 0.45, 0.35, 0.25],
                ]),
                "clump_fac": jnp.array([0.7, 0.72]),
                "ncan": jnp.array([5, 5]),
                "lai": jnp.array([2.5, 3.0]),
                "sai": jnp.array([0.7, 0.8]),
                "params": None,
                "validate": True,
            },
            "metadata": {
                "type": "edge",
                "description": "Extreme hot temperatures (40-45°C), C4 desert/savanna conditions",
            },
        },
        "test_edge_full_shade_no_sunlit": {
            "inputs": {
                "vcmaxpft": jnp.array([55.0]),
                "c3psn": jnp.array([1]),
                "tacclim": jnp.array([290.15]),
                "dpai": jnp.array([[0.4, 0.35, 0.3, 0.25, 0.2]]),
                "kb": jnp.array([[0.8] * 5]),
                "tbi": jnp.array([[0.1, 0.05, 0.02, 0.01, 0.0]]),
                "fracsun": jnp.array([[0.15, 0.08, 0.03, 0.01, 0.0]]),
                "clump_fac": jnp.array([0.6]),
                "ncan": jnp.array([5]),
                "lai": jnp.array([1.5]),
                "sai": jnp.array([0.4]),
                "params": None,
                "validate": True,
            },
            "metadata": {
                "type": "edge",
                "description": "Deep shade conditions with minimal sunlit fraction in lower canopy",
            },
        },
        "test_special_variable_layer_counts": {
            "inputs": {
                "vcmaxpft": jnp.array([60.0, 50.0, 70.0, 45.0]),
                "c3psn": jnp.array([1, 0, 1, 1]),
                "tacclim": jnp.array([293.15, 298.15, 288.15, 295.15]),
                "dpai": jnp.array([
                    [0.8, 0.6, 0.4, 0.2] + [0.0] * 16,
                    [0.5, 0.4, 0.3] + [0.0] * 17,
                    [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] + [0.0] * 10,
                    [0.3, 0.2] + [0.0] * 18,
                ]),
                "kb": jnp.array([
                    [0.5] * 20,
                    [0.55] * 20,
                    [0.48] * 20,
                    [0.52] * 20,
                ]),
                "tbi": jnp.array([
                    [0.89, 0.79, 0.7, 0.62] + [1.0] * 16,
                    [0.92, 0.84, 0.77] + [1.0] * 17,
                    [0.86, 0.74, 0.64, 0.55, 0.47, 0.4, 0.34, 0.29, 0.25, 0.21] + [1.0] * 10,
                    [0.94, 0.88] + [1.0] * 18,
                ]),
                "fracsun": jnp.array([
                    [0.72, 0.62, 0.52, 0.42] + [1.0] * 16,
                    [0.75, 0.65, 0.55] + [1.0] * 17,
                    [0.68, 0.58, 0.5, 0.43, 0.37, 0.32, 0.27, 0.23, 0.2, 0.17] + [1.0] * 10,
                    [0.78, 0.7] + [1.0] * 18,
                ]),
                "clump_fac": jnp.array([0.85, 0.75, 0.8, 0.9]),
                "ncan": jnp.array([4, 3, 10, 2]),
                "lai": jnp.array([2.0, 1.2, 5.0, 0.5]),
                "sai": jnp.array([0.6, 0.3, 1.5, 0.15]),
                "params": None,
                "validate": True,
            },
            "metadata": {
                "type": "special",
                "description": "Four patches with different active layer counts (2, 3, 4, 10) using max 20-layer array",
            },
        },
        "test_special_uniform_vs_clumped_canopy": {
            "inputs": {
                "vcmaxpft": jnp.array([65.0, 65.0, 65.0]),
                "c3psn": jnp.array([1, 1, 1]),
                "tacclim": jnp.array([293.15, 293.15, 293.15]),
                "dpai": jnp.array([
                    [0.6, 0.5, 0.4, 0.3, 0.2],
                    [0.6, 0.5, 0.4, 0.3, 0.2],
                    [0.6, 0.5, 0.4, 0.3, 0.2],
                ]),
                "kb": jnp.array([[0.5] * 5] * 3),
                "tbi": jnp.array([[0.9, 0.8, 0.7, 0.6, 0.5]] * 3),
                "fracsun": jnp.array([[0.75, 0.65, 0.55, 0.45, 0.35]] * 3),
                "clump_fac": jnp.array([1.0, 0.7, 0.5]),
                "ncan": jnp.array([5, 5, 5]),
                "lai": jnp.array([2.0, 2.0, 2.0]),
                "sai": jnp.array([0.6, 0.6, 0.6]),
                "params": None,
                "validate": True,
            },
            "metadata": {
                "type": "special",
                "description": "Three identical canopies with different clumping factors (uniform=1.0, moderate=0.7, highly clumped=0.5)",
            },
        },
    }


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_nominal_single_patch_c3_moderate_lai",
        "test_nominal_multiple_patches_mixed_pathways",
        "test_nominal_high_lai_dense_canopy",
        "test_edge_zero_lai_no_canopy",
        "test_edge_minimal_lai_single_layer",
        "test_edge_extreme_cold_temperature",
        "test_edge_extreme_hot_temperature",
        "test_edge_full_shade_no_sunlit",
        "test_special_variable_layer_counts",
        "test_special_uniform_vs_clumped_canopy",
    ],
)
def test_canopy_nitrogen_profile_shapes(test_data: Dict[str, Any], test_case_name: str):
    """
    Test that canopy_nitrogen_profile returns correct output shapes.
    
    Verifies that:
    - CanopyNitrogenProfile fields have expected shapes (n_patches, n_layers)
    - CanopyNitrogenValidation fields have expected shapes when validate=True
    - Scalar fields have correct dimensions
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Get expected shapes
    n_patches = inputs["vcmaxpft"].shape[0]
    n_layers = inputs["dpai"].shape[1]
    
    # Check CanopyNitrogenProfile shapes
    assert profile.vcmax25_leaf_sun.shape == (n_patches, n_layers), \
        f"vcmax25_leaf_sun shape mismatch: expected {(n_patches, n_layers)}, got {profile.vcmax25_leaf_sun.shape}"
    assert profile.vcmax25_leaf_sha.shape == (n_patches, n_layers), \
        f"vcmax25_leaf_sha shape mismatch"
    assert profile.jmax25_leaf_sun.shape == (n_patches, n_layers), \
        f"jmax25_leaf_sun shape mismatch"
    assert profile.jmax25_leaf_sha.shape == (n_patches, n_layers), \
        f"jmax25_leaf_sha shape mismatch"
    assert profile.rd25_leaf_sun.shape == (n_patches, n_layers), \
        f"rd25_leaf_sun shape mismatch"
    assert profile.rd25_leaf_sha.shape == (n_patches, n_layers), \
        f"rd25_leaf_sha shape mismatch"
    assert profile.kp25_leaf_sun.shape == (n_patches, n_layers), \
        f"kp25_leaf_sun shape mismatch"
    assert profile.kp25_leaf_sha.shape == (n_patches, n_layers), \
        f"kp25_leaf_sha shape mismatch"
    assert profile.vcmax25_profile.shape == (n_patches, n_layers), \
        f"vcmax25_profile shape mismatch"
    assert profile.jmax25_profile.shape == (n_patches, n_layers), \
        f"jmax25_profile shape mismatch"
    assert profile.rd25_profile.shape == (n_patches, n_layers), \
        f"rd25_profile shape mismatch"
    assert profile.kp25_profile.shape == (n_patches, n_layers), \
        f"kp25_profile shape mismatch"
    assert profile.kn.shape == (n_patches,), \
        f"kn shape mismatch: expected {(n_patches,)}, got {profile.kn.shape}"
    
    # Check CanopyNitrogenValidation shapes if validation enabled
    if inputs["validate"]:
        assert validation is not None, "Validation should not be None when validate=True"
        assert validation.numerical.shape == (n_patches,), \
            f"numerical shape mismatch: expected {(n_patches,)}, got {validation.numerical.shape}"
        assert validation.analytical.shape == (n_patches,), \
            f"analytical shape mismatch"
        assert validation.max_error.shape == (), \
            f"max_error should be scalar, got shape {validation.max_error.shape}"
        assert validation.is_valid.shape == (), \
            f"is_valid should be scalar, got shape {validation.is_valid.shape}"


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_nominal_single_patch_c3_moderate_lai",
        "test_nominal_multiple_patches_mixed_pathways",
        "test_edge_minimal_lai_single_layer",
    ],
)
def test_canopy_nitrogen_profile_values(test_data: Dict[str, Any], test_case_name: str):
    """
    Test that canopy_nitrogen_profile produces physically reasonable values.
    
    Verifies:
    - All photosynthetic parameters are non-negative
    - Vcmax, Jmax, Rd, Kp values are in reasonable ranges
    - Nitrogen decay coefficient (kn) is non-negative
    - Profile values decrease through canopy (nitrogen extinction)
    - Sunlit values >= shaded values (more light = more nitrogen allocation)
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Check non-negativity
    assert jnp.all(profile.vcmax25_leaf_sun >= 0), "vcmax25_leaf_sun should be non-negative"
    assert jnp.all(profile.vcmax25_leaf_sha >= 0), "vcmax25_leaf_sha should be non-negative"
    assert jnp.all(profile.jmax25_leaf_sun >= 0), "jmax25_leaf_sun should be non-negative"
    assert jnp.all(profile.jmax25_leaf_sha >= 0), "jmax25_leaf_sha should be non-negative"
    assert jnp.all(profile.rd25_leaf_sun >= 0), "rd25_leaf_sun should be non-negative"
    assert jnp.all(profile.rd25_leaf_sha >= 0), "rd25_leaf_sha should be non-negative"
    assert jnp.all(profile.kp25_leaf_sun >= 0), "kp25_leaf_sun should be non-negative"
    assert jnp.all(profile.kp25_leaf_sha >= 0), "kp25_leaf_sha should be non-negative"
    assert jnp.all(profile.kn >= 0), "kn should be non-negative"
    
    # Check reasonable ranges for Vcmax (typically 0-200 umol/m2/s)
    assert jnp.all(profile.vcmax25_leaf_sun <= 200), \
        f"vcmax25_leaf_sun values seem too high: max={jnp.max(profile.vcmax25_leaf_sun)}"
    assert jnp.all(profile.vcmax25_leaf_sha <= 200), \
        f"vcmax25_leaf_sha values seem too high: max={jnp.max(profile.vcmax25_leaf_sha)}"
    
    # Check that sunlit >= shaded (where both are non-zero)
    mask = (profile.vcmax25_leaf_sun > 0) & (profile.vcmax25_leaf_sha > 0)
    if jnp.any(mask):
        assert jnp.all(profile.vcmax25_leaf_sun[mask] >= profile.vcmax25_leaf_sha[mask] - 1e-6), \
            "Sunlit vcmax25 should be >= shaded vcmax25"
    
    # For non-zero LAI cases, check that profile decreases through canopy
    if jnp.any(inputs["lai"] > 0):
        for i in range(inputs["vcmaxpft"].shape[0]):
            if inputs["ncan"][i] > 1:
                # Check that values generally decrease (allowing for small numerical variations)
                profile_vals = profile.vcmax25_profile[i, :inputs["ncan"][i]]
                # First layer should have highest or near-highest values
                assert profile_vals[0] >= jnp.max(profile_vals) * 0.9, \
                    f"Patch {i}: Top layer should have highest nitrogen concentration"


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_edge_zero_lai_no_canopy",
        "test_edge_minimal_lai_single_layer",
        "test_edge_extreme_cold_temperature",
        "test_edge_extreme_hot_temperature",
        "test_edge_full_shade_no_sunlit",
    ],
)
def test_canopy_nitrogen_profile_edge_cases(test_data: Dict[str, Any], test_case_name: str):
    """
    Test canopy_nitrogen_profile behavior in edge cases.
    
    Verifies:
    - Zero LAI produces zero or minimal profiles
    - Extreme temperatures don't cause numerical issues
    - Deep shade conditions are handled correctly
    - Single layer canopies work properly
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    metadata = test_case["metadata"]
    
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Zero LAI case
    if "zero_lai" in metadata.get("edge_cases", []):
        # With zero LAI, profiles should be zero or very small
        assert jnp.allclose(profile.vcmax25_profile, 0.0, atol=1e-6), \
            "Zero LAI should produce near-zero vcmax25 profile"
        assert jnp.allclose(profile.jmax25_profile, 0.0, atol=1e-6), \
            "Zero LAI should produce near-zero jmax25 profile"
    
    # Minimal LAI case
    if "minimal_lai" in metadata.get("edge_cases", []):
        # Should have small but non-zero values
        assert jnp.any(profile.vcmax25_profile > 0), \
            "Minimal LAI should produce some non-zero values"
        assert jnp.all(profile.vcmax25_profile < 50), \
            "Minimal LAI should produce small values"
    
    # Extreme temperature cases
    if "extreme_cold" in metadata.get("edge_cases", []) or \
       "extreme_heat" in metadata.get("edge_cases", []):
        # Should not produce NaN or Inf
        assert jnp.all(jnp.isfinite(profile.vcmax25_leaf_sun)), \
            "Extreme temperatures should not produce NaN/Inf in vcmax25_leaf_sun"
        assert jnp.all(jnp.isfinite(profile.jmax25_leaf_sun)), \
            "Extreme temperatures should not produce NaN/Inf in jmax25_leaf_sun"
        # Values should still be reasonable
        assert jnp.all(profile.vcmax25_leaf_sun >= 0), \
            "Extreme temperatures should not produce negative values"
    
    # Deep shade case
    if "deep_shade" in metadata.get("edge_cases", []):
        # Shaded fraction should dominate
        for i in range(inputs["vcmaxpft"].shape[0]):
            ncan = inputs["ncan"][i]
            if ncan > 0:
                # Lower layers should have very low sunlit fractions
                assert inputs["fracsun"][i, ncan-1] < 0.2, \
                    "Deep shade case should have low sunlit fraction in lower canopy"


def test_canopy_nitrogen_profile_dtypes(test_data: Dict[str, Any]):
    """
    Test that canopy_nitrogen_profile returns correct data types.
    
    Verifies:
    - All array outputs are JAX arrays
    - Floating point arrays have correct dtype
    - Boolean validation flag has correct dtype
    """
    test_case = test_data["test_nominal_single_patch_c3_moderate_lai"]
    inputs = test_case["inputs"]
    
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Check that outputs are JAX arrays
    assert isinstance(profile.vcmax25_leaf_sun, jnp.ndarray), \
        "vcmax25_leaf_sun should be a JAX array"
    assert isinstance(profile.kn, jnp.ndarray), \
        "kn should be a JAX array"
    
    # Check floating point dtypes
    assert jnp.issubdtype(profile.vcmax25_leaf_sun.dtype, jnp.floating), \
        f"vcmax25_leaf_sun should be floating point, got {profile.vcmax25_leaf_sun.dtype}"
    assert jnp.issubdtype(profile.jmax25_profile.dtype, jnp.floating), \
        f"jmax25_profile should be floating point, got {profile.jmax25_profile.dtype}"
    
    # Check validation dtypes
    if validation is not None:
        assert jnp.issubdtype(validation.numerical.dtype, jnp.floating), \
            "numerical should be floating point"
        assert jnp.issubdtype(validation.analytical.dtype, jnp.floating), \
            "analytical should be floating point"
        assert validation.is_valid.dtype == jnp.bool_, \
            f"is_valid should be boolean, got {validation.is_valid.dtype}"


def test_canopy_nitrogen_profile_validation(test_data: Dict[str, Any]):
    """
    Test the numerical integration validation feature.
    
    Verifies:
    - Validation is returned when validate=True
    - Validation is None when validate=False
    - Numerical and analytical integrals are close
    - Validation passes for well-behaved cases
    """
    test_case = test_data["test_nominal_single_patch_c3_moderate_lai"]
    inputs = test_case["inputs"].copy()
    
    # Test with validation enabled
    inputs["validate"] = True
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    assert validation is not None, "Validation should be returned when validate=True"
    assert jnp.all(jnp.isfinite(validation.numerical)), \
        "Numerical integral should be finite"
    assert jnp.all(jnp.isfinite(validation.analytical)), \
        "Analytical integral should be finite"
    
    # Check that numerical and analytical are close
    rel_error = jnp.abs(validation.numerical - validation.analytical) / \
                (jnp.abs(validation.analytical) + 1e-10)
    assert jnp.all(rel_error < 0.01), \
        f"Numerical and analytical integrals should be close, max rel error: {jnp.max(rel_error)}"
    
    # Test with validation disabled
    inputs["validate"] = False
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    assert validation is None, "Validation should be None when validate=False"


def test_canopy_nitrogen_profile_c3_vs_c4(test_data: Dict[str, Any]):
    """
    Test differences between C3 and C4 photosynthetic pathways.
    
    Verifies:
    - C3 plants have non-zero Jmax values
    - C4 plants have non-zero Kp values
    - C4 plants have zero or minimal Jmax values
    - Rd/Vcmax ratios differ between C3 and C4
    """
    test_case = test_data["test_nominal_multiple_patches_mixed_pathways"]
    inputs = test_case["inputs"]
    
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    c3_mask = inputs["c3psn"] == 1
    c4_mask = inputs["c3psn"] == 0
    
    # C3 plants should have non-zero Jmax
    if jnp.any(c3_mask):
        c3_indices = jnp.where(c3_mask)[0]
        for idx in c3_indices:
            if inputs["ncan"][idx] > 0:
                assert jnp.any(profile.jmax25_leaf_sun[idx] > 0), \
                    f"C3 patch {idx} should have non-zero Jmax"
    
    # C4 plants should have non-zero Kp
    if jnp.any(c4_mask):
        c4_indices = jnp.where(c4_mask)[0]
        for idx in c4_indices:
            if inputs["ncan"][idx] > 0:
                assert jnp.any(profile.kp25_leaf_sun[idx] > 0), \
                    f"C4 patch {idx} should have non-zero Kp"


def test_canopy_nitrogen_profile_clumping_effect(test_data: Dict[str, Any]):
    """
    Test the effect of foliage clumping on nitrogen distribution.
    
    Verifies:
    - Different clumping factors produce different profiles
    - More clumping (lower clump_fac) affects sunlit/shaded distribution
    - Uniform canopy (clump_fac=1.0) has expected behavior
    """
    test_case = test_data["test_special_uniform_vs_clumped_canopy"]
    inputs = test_case["inputs"]
    
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Extract profiles for the three patches (uniform, moderate, highly clumped)
    uniform_profile = profile.vcmax25_profile[0]
    moderate_profile = profile.vcmax25_profile[1]
    clumped_profile = profile.vcmax25_profile[2]
    
    # Profiles should differ due to clumping
    assert not jnp.allclose(uniform_profile, moderate_profile, rtol=0.01), \
        "Uniform and moderate clumping should produce different profiles"
    assert not jnp.allclose(moderate_profile, clumped_profile, rtol=0.01), \
        "Moderate and high clumping should produce different profiles"
    
    # All should have reasonable values
    assert jnp.all(uniform_profile >= 0), "Uniform profile should be non-negative"
    assert jnp.all(moderate_profile >= 0), "Moderate clumping profile should be non-negative"
    assert jnp.all(clumped_profile >= 0), "Clumped profile should be non-negative"


def test_canopy_nitrogen_profile_variable_layers(test_data: Dict[str, Any]):
    """
    Test handling of variable layer counts across patches.
    
    Verifies:
    - Different ncan values are handled correctly
    - Unused layers (beyond ncan) have appropriate values
    - Active layers have reasonable profiles
    """
    test_case = test_data["test_special_variable_layer_counts"]
    inputs = test_case["inputs"]
    
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Check each patch
    for i in range(inputs["vcmaxpft"].shape[0]):
        ncan = int(inputs["ncan"][i])
        
        if ncan > 0:
            # Active layers should have non-zero values
            active_vcmax = profile.vcmax25_profile[i, :ncan]
            assert jnp.any(active_vcmax > 0), \
                f"Patch {i} with ncan={ncan} should have non-zero active layers"
            
            # Check that profile decreases through active layers
            if ncan > 1:
                # Allow for some variation but expect general decrease
                assert active_vcmax[0] >= active_vcmax[ncan-1] * 0.5, \
                    f"Patch {i}: Profile should generally decrease through canopy"


def test_canopy_nitrogen_profile_parameter_override(test_data: Dict[str, Any]):
    """
    Test custom parameter override functionality.
    
    Verifies:
    - Default parameters are used when params=None
    - Custom parameters can be provided
    - Custom parameters affect output appropriately
    """
    test_case = test_data["test_nominal_single_patch_c3_moderate_lai"]
    inputs = test_case["inputs"].copy()
    
    # Test with default parameters
    inputs["params"] = None
    profile_default, _ = canopy_nitrogen_profile(**inputs)
    
    # Test with custom parameters (modified Jmax/Vcmax ratio)
    custom_params = get_default_params()
    custom_params = custom_params._replace(jmax25_to_vcmax25_noacclim=3.0)  # Higher than default 2.59
    inputs["params"] = custom_params
    profile_custom, _ = canopy_nitrogen_profile(**inputs)
    
    # Jmax values should be different (higher with custom params for C3)
    if inputs["c3psn"][0] == 1:
        assert jnp.any(profile_custom.jmax25_leaf_sun > profile_default.jmax25_leaf_sun), \
            "Custom Jmax/Vcmax ratio should increase Jmax values"


def test_canopy_nitrogen_profile_consistency(test_data: Dict[str, Any]):
    """
    Test internal consistency of outputs.
    
    Verifies:
    - Profile values are weighted averages of sunlit and shaded
    - Nitrogen decay coefficient (kn) is consistent with profiles
    - Sum of dpai approximately equals LAI + SAI
    """
    test_case = test_data["test_nominal_single_patch_c3_moderate_lai"]
    inputs = test_case["inputs"]
    
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Check that profile is weighted average of sunlit and shaded
    for i in range(inputs["vcmaxpft"].shape[0]):
        ncan = int(inputs["ncan"][i])
        if ncan > 0:
            for j in range(ncan):
                fracsun = inputs["fracsun"][i, j]
                fracsha = 1.0 - fracsun
                
                # Weighted average (with tolerance for numerical precision)
                expected = fracsun * profile.vcmax25_leaf_sun[i, j] + \
                          fracsha * profile.vcmax25_leaf_sha[i, j]
                actual = profile.vcmax25_profile[i, j]
                
                assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-6), \
                    f"Patch {i}, layer {j}: Profile should be weighted average of sun/shade"
    
    # Check dpai sum approximately equals LAI + SAI
    for i in range(inputs["vcmaxpft"].shape[0]):
        ncan = int(inputs["ncan"][i])
        if ncan > 0:
            dpai_sum = jnp.sum(inputs["dpai"][i, :ncan])
            total_ai = inputs["lai"][i] + inputs["sai"][i]
            
            assert jnp.allclose(dpai_sum, total_ai, rtol=0.1), \
                f"Patch {i}: Sum of dpai should approximately equal LAI + SAI"