"""
Comprehensive pytest suite for leaf_boundary_layer function.

This module tests the leaf boundary layer conductance calculations for
multilayer canopy models, including:
- Nominal conditions across various biomes (temperate, tropical, arctic, desert)
- Edge cases (near-zero wind, high wind, extreme temperature gradients)
- Special cases (tiny leaves, high altitude)
- Output shape and dtype validation
- Physical constraint verification
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multilayer_canopy.MLLeafBoundaryLayerMod import (
    BoundaryLayerParams,
    LeafBoundaryLayerOutputs,
    leaf_boundary_layer,
    leaf_boundary_layer_sunlit_shaded,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_data():
    """
    Load and provide test data for leaf_boundary_layer function.
    
    Returns:
        dict: Test cases with inputs and metadata
    """
    return {
        "nominal_single_patch_single_layer": {
            "inputs": {
                "dleaf": jnp.array([0.05]),
                "tref": jnp.array([298.15]),
                "pref": jnp.array([101325.0]),
                "wind": jnp.array([[2.5]]),
                "tair": jnp.array([[295.15]]),
                "tleaf": jnp.array([[298.15]]),
                "rhomol": jnp.array([[41.0]]),
                "dpai": jnp.array([[1.5]]),
                "params": None,
            },
            "metadata": {
                "type": "nominal",
                "description": "Typical temperate forest conditions: 5cm leaf, 25°C air, 2.5 m/s wind, moderate LAI",
                "expected_shape": (1, 1),
            },
        },
        "nominal_multi_patch_multi_layer": {
            "inputs": {
                "dleaf": jnp.array([0.03, 0.08, 0.05]),
                "tref": jnp.array([293.15, 303.15, 288.15]),
                "pref": jnp.array([101325.0, 95000.0, 101325.0]),
                "wind": jnp.array([
                    [1.2, 0.8, 0.5, 0.3],
                    [3.5, 2.1, 1.4, 0.9],
                    [0.5, 0.3, 0.2, 0.1],
                ]),
                "tair": jnp.array([
                    [291.15, 292.15, 293.15, 294.15],
                    [301.15, 302.15, 303.15, 304.15],
                    [286.15, 287.15, 288.15, 289.15],
                ]),
                "tleaf": jnp.array([
                    [293.15, 294.15, 295.15, 296.15],
                    [303.15, 304.15, 305.15, 306.15],
                    [288.15, 289.15, 290.15, 291.15],
                ]),
                "rhomol": jnp.array([
                    [42.0, 41.8, 41.5, 41.2],
                    [39.5, 39.3, 39.0, 38.8],
                    [43.5, 43.2, 43.0, 42.8],
                ]),
                "dpai": jnp.array([
                    [2.0, 1.5, 1.0, 0.5],
                    [3.5, 2.8, 2.0, 1.2],
                    [1.2, 0.9, 0.6, 0.3],
                ]),
                "params": None,
            },
            "metadata": {
                "type": "nominal",
                "description": "Three patches with varying conditions: cool/calm, hot/windy, cold/still",
                "expected_shape": (3, 4),
            },
        },
        "nominal_tropical_conditions": {
            "inputs": {
                "dleaf": jnp.array([0.12, 0.15]),
                "tref": jnp.array([303.15, 305.15]),
                "pref": jnp.array([101325.0, 101325.0]),
                "wind": jnp.array([
                    [4.0, 2.5, 1.5],
                    [3.5, 2.0, 1.0],
                ]),
                "tair": jnp.array([
                    [301.15, 302.15, 303.15],
                    [303.15, 304.15, 305.15],
                ]),
                "tleaf": jnp.array([
                    [304.15, 305.15, 306.15],
                    [306.15, 307.15, 308.15],
                ]),
                "rhomol": jnp.array([
                    [39.8, 39.5, 39.2],
                    [39.3, 39.0, 38.7],
                ]),
                "dpai": jnp.array([
                    [4.5, 3.2, 1.8],
                    [5.0, 3.8, 2.2],
                ]),
                "params": None,
            },
            "metadata": {
                "type": "nominal",
                "description": "Tropical rainforest: large leaves, high temperatures, high LAI",
                "expected_shape": (2, 3),
            },
        },
        "nominal_arctic_conditions": {
            "inputs": {
                "dleaf": jnp.array([0.01, 0.02]),
                "tref": jnp.array([273.15, 275.15]),
                "pref": jnp.array([101325.0, 101325.0]),
                "wind": jnp.array([
                    [5.0, 3.5],
                    [6.0, 4.0],
                ]),
                "tair": jnp.array([
                    [271.15, 272.15],
                    [273.15, 274.15],
                ]),
                "tleaf": jnp.array([
                    [272.15, 273.15],
                    [274.15, 275.15],
                ]),
                "rhomol": jnp.array([
                    [44.5, 44.2],
                    [44.0, 43.7],
                ]),
                "dpai": jnp.array([
                    [0.5, 0.3],
                    [0.8, 0.5],
                ]),
                "params": None,
            },
            "metadata": {
                "type": "nominal",
                "description": "Arctic tundra: tiny leaves, near-freezing temps, low LAI",
                "expected_shape": (2, 2),
            },
        },
        "nominal_desert_conditions": {
            "inputs": {
                "dleaf": jnp.array([0.02, 0.03, 0.025]),
                "tref": jnp.array([313.15, 315.15, 310.15]),
                "pref": jnp.array([95000.0, 94000.0, 96000.0]),
                "wind": jnp.array([
                    [8.0, 6.0, 4.0],
                    [10.0, 7.5, 5.0],
                    [6.0, 4.5, 3.0],
                ]),
                "tair": jnp.array([
                    [311.15, 312.15, 313.15],
                    [313.15, 314.15, 315.15],
                    [308.15, 309.15, 310.15],
                ]),
                "tleaf": jnp.array([
                    [315.15, 316.15, 317.15],
                    [318.15, 319.15, 320.15],
                    [312.15, 313.15, 314.15],
                ]),
                "rhomol": jnp.array([
                    [37.5, 37.3, 37.0],
                    [36.8, 36.5, 36.2],
                    [38.2, 38.0, 37.7],
                ]),
                "dpai": jnp.array([
                    [0.3, 0.2, 0.1],
                    [0.4, 0.3, 0.15],
                    [0.25, 0.18, 0.08],
                ]),
                "params": None,
            },
            "metadata": {
                "type": "nominal",
                "description": "Hot desert: small leaves, very high temps, very low LAI",
                "expected_shape": (3, 3),
            },
        },
        "edge_near_zero_wind": {
            "inputs": {
                "dleaf": jnp.array([0.05, 0.06]),
                "tref": jnp.array([298.15, 300.15]),
                "pref": jnp.array([101325.0, 101325.0]),
                "wind": jnp.array([
                    [0.01, 0.005, 0.001],
                    [0.02, 0.01, 0.005],
                ]),
                "tair": jnp.array([
                    [296.15, 297.15, 298.15],
                    [298.15, 299.15, 300.15],
                ]),
                "tleaf": jnp.array([
                    [298.15, 299.15, 300.15],
                    [300.15, 301.15, 302.15],
                ]),
                "rhomol": jnp.array([
                    [41.2, 41.0, 40.8],
                    [40.5, 40.3, 40.0],
                ]),
                "dpai": jnp.array([
                    [1.5, 1.0, 0.5],
                    [2.0, 1.5, 1.0],
                ]),
                "params": None,
            },
            "metadata": {
                "type": "edge",
                "description": "Near-zero wind speeds to test free convection dominance",
                "expected_shape": (2, 3),
                "edge_cases": ["near_zero_wind", "free_convection_regime"],
            },
        },
        "edge_very_high_wind": {
            "inputs": {
                "dleaf": jnp.array([0.04, 0.05]),
                "tref": jnp.array([295.15, 298.15]),
                "pref": jnp.array([101325.0, 101325.0]),
                "wind": jnp.array([
                    [25.0, 20.0, 15.0],
                    [30.0, 25.0, 20.0],
                ]),
                "tair": jnp.array([
                    [293.15, 294.15, 295.15],
                    [296.15, 297.15, 298.15],
                ]),
                "tleaf": jnp.array([
                    [294.15, 295.15, 296.15],
                    [297.15, 298.15, 299.15],
                ]),
                "rhomol": jnp.array([
                    [41.8, 41.5, 41.2],
                    [41.0, 40.8, 40.5],
                ]),
                "dpai": jnp.array([
                    [2.5, 2.0, 1.5],
                    [3.0, 2.5, 2.0],
                ]),
                "params": None,
            },
            "metadata": {
                "type": "edge",
                "description": "Very high wind speeds (storm conditions)",
                "expected_shape": (2, 3),
                "edge_cases": ["high_wind", "turbulent_regime"],
            },
        },
        "edge_extreme_temperature_gradient": {
            "inputs": {
                "dleaf": jnp.array([0.05]),
                "tref": jnp.array([298.15]),
                "pref": jnp.array([101325.0]),
                "wind": jnp.array([[1.0, 0.8, 0.6]]),
                "tair": jnp.array([[280.15, 285.15, 290.15]]),
                "tleaf": jnp.array([[310.15, 308.15, 305.15]]),
                "rhomol": jnp.array([[43.0, 42.0, 41.5]]),
                "dpai": jnp.array([[1.5, 1.2, 0.8]]),
                "params": None,
            },
            "metadata": {
                "type": "edge",
                "description": "Large temperature difference (30K) to test free convection",
                "expected_shape": (1, 3),
                "edge_cases": ["extreme_temperature_gradient", "strong_free_convection"],
            },
        },
        "special_tiny_leaves": {
            "inputs": {
                "dleaf": jnp.array([0.001, 0.002, 0.0005]),
                "tref": jnp.array([295.15, 298.15, 293.15]),
                "pref": jnp.array([101325.0, 101325.0, 101325.0]),
                "wind": jnp.array([
                    [2.0, 1.5],
                    [3.0, 2.0],
                    [1.5, 1.0],
                ]),
                "tair": jnp.array([
                    [293.15, 294.15],
                    [296.15, 297.15],
                    [291.15, 292.15],
                ]),
                "tleaf": jnp.array([
                    [295.15, 296.15],
                    [298.15, 299.15],
                    [293.15, 294.15],
                ]),
                "rhomol": jnp.array([
                    [41.5, 41.3],
                    [41.0, 40.8],
                    [42.0, 41.8],
                ]),
                "dpai": jnp.array([
                    [0.5, 0.3],
                    [0.8, 0.5],
                    [0.4, 0.2],
                ]),
                "params": None,
            },
            "metadata": {
                "type": "special",
                "description": "Extremely small leaf dimensions (0.5-2mm)",
                "expected_shape": (3, 2),
                "edge_cases": ["tiny_leaves", "low_reynolds_number"],
            },
        },
        "special_high_altitude": {
            "inputs": {
                "dleaf": jnp.array([0.04, 0.05]),
                "tref": jnp.array([278.15, 280.15]),
                "pref": jnp.array([65000.0, 63000.0]),
                "wind": jnp.array([
                    [4.0, 3.0, 2.0],
                    [5.0, 4.0, 3.0],
                ]),
                "tair": jnp.array([
                    [276.15, 277.15, 278.15],
                    [278.15, 279.15, 280.15],
                ]),
                "tleaf": jnp.array([
                    [279.15, 280.15, 281.15],
                    [281.15, 282.15, 283.15],
                ]),
                "rhomol": jnp.array([
                    [28.5, 28.3, 28.0],
                    [27.8, 27.5, 27.2],
                ]),
                "dpai": jnp.array([
                    [1.0, 0.7, 0.4],
                    [1.2, 0.9, 0.5],
                ]),
                "params": None,
            },
            "metadata": {
                "type": "special",
                "description": "High altitude conditions (~4000m): low pressure, low density",
                "expected_shape": (2, 3),
                "edge_cases": ["low_pressure", "low_density", "high_altitude"],
            },
        },
    }


@pytest.fixture
def default_params():
    """
    Provide default boundary layer parameters for testing.
    
    Returns:
        BoundaryLayerParams: Default parameter values
    """
    return BoundaryLayerParams(
        gb_type=0,
        gb_factor=1.0,
        visc0=1.326e-5,
        dh0=1.895e-5,
        dv0=2.178e-5,
        dc0=1.381e-5,
        tfrz=273.15,
        grav=9.80665,
    )


# ============================================================================
# Shape Tests
# ============================================================================


@pytest.mark.parametrize(
    "test_case_name",
    [
        "nominal_single_patch_single_layer",
        "nominal_multi_patch_multi_layer",
        "nominal_tropical_conditions",
        "nominal_arctic_conditions",
        "nominal_desert_conditions",
        "edge_near_zero_wind",
        "edge_very_high_wind",
        "edge_extreme_temperature_gradient",
        "special_tiny_leaves",
        "special_high_altitude",
    ],
)
def test_leaf_boundary_layer_output_shapes(test_data, test_case_name):
    """
    Test that leaf_boundary_layer returns outputs with correct shapes.
    
    Verifies that gbh, gbv, and gbc all have shape (n_patches, n_canopy_layers)
    matching the input wind/tair/tleaf arrays.
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    expected_shape = test_case["metadata"]["expected_shape"]
    
    result = leaf_boundary_layer(**inputs)
    
    assert isinstance(result, LeafBoundaryLayerOutputs), (
        f"Expected LeafBoundaryLayerOutputs, got {type(result)}"
    )
    
    assert result.gbh.shape == expected_shape, (
        f"gbh shape mismatch: expected {expected_shape}, got {result.gbh.shape}"
    )
    assert result.gbv.shape == expected_shape, (
        f"gbv shape mismatch: expected {expected_shape}, got {result.gbv.shape}"
    )
    assert result.gbc.shape == expected_shape, (
        f"gbc shape mismatch: expected {expected_shape}, got {result.gbc.shape}"
    )


# ============================================================================
# Data Type Tests
# ============================================================================


@pytest.mark.parametrize(
    "test_case_name",
    [
        "nominal_single_patch_single_layer",
        "nominal_multi_patch_multi_layer",
    ],
)
def test_leaf_boundary_layer_output_dtypes(test_data, test_case_name):
    """
    Test that leaf_boundary_layer returns outputs with correct data types.
    
    Verifies that all outputs are JAX arrays with float dtype.
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    result = leaf_boundary_layer(**inputs)
    
    assert isinstance(result.gbh, jnp.ndarray), (
        f"gbh should be jnp.ndarray, got {type(result.gbh)}"
    )
    assert isinstance(result.gbv, jnp.ndarray), (
        f"gbv should be jnp.ndarray, got {type(result.gbv)}"
    )
    assert isinstance(result.gbc, jnp.ndarray), (
        f"gbc should be jnp.ndarray, got {type(result.gbc)}"
    )
    
    assert jnp.issubdtype(result.gbh.dtype, jnp.floating), (
        f"gbh should be float dtype, got {result.gbh.dtype}"
    )
    assert jnp.issubdtype(result.gbv.dtype, jnp.floating), (
        f"gbv should be float dtype, got {result.gbv.dtype}"
    )
    assert jnp.issubdtype(result.gbc.dtype, jnp.floating), (
        f"gbc should be float dtype, got {result.gbc.dtype}"
    )


# ============================================================================
# Physical Constraint Tests
# ============================================================================


@pytest.mark.parametrize(
    "test_case_name",
    [
        "nominal_single_patch_single_layer",
        "nominal_multi_patch_multi_layer",
        "nominal_tropical_conditions",
        "nominal_arctic_conditions",
        "nominal_desert_conditions",
        "edge_near_zero_wind",
        "edge_very_high_wind",
        "edge_extreme_temperature_gradient",
        "special_tiny_leaves",
        "special_high_altitude",
    ],
)
def test_leaf_boundary_layer_physical_constraints(test_data, test_case_name):
    """
    Test that leaf_boundary_layer outputs satisfy physical constraints.
    
    Verifies that:
    - All conductances are non-negative
    - No NaN or Inf values in outputs
    - Conductances are finite and reasonable (< 100 mol/m2/s)
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    result = leaf_boundary_layer(**inputs)
    
    # Check non-negativity
    assert jnp.all(result.gbh >= 0), (
        f"gbh contains negative values: min={jnp.min(result.gbh)}"
    )
    assert jnp.all(result.gbv >= 0), (
        f"gbv contains negative values: min={jnp.min(result.gbv)}"
    )
    assert jnp.all(result.gbc >= 0), (
        f"gbc contains negative values: min={jnp.min(result.gbc)}"
    )
    
    # Check for NaN/Inf
    assert jnp.all(jnp.isfinite(result.gbh)), (
        "gbh contains NaN or Inf values"
    )
    assert jnp.all(jnp.isfinite(result.gbv)), (
        "gbv contains NaN or Inf values"
    )
    assert jnp.all(jnp.isfinite(result.gbc)), (
        "gbc contains NaN or Inf values"
    )
    
    # Check reasonable upper bounds (conductances rarely exceed 100 mol/m2/s)
    assert jnp.all(result.gbh < 100.0), (
        f"gbh contains unreasonably large values: max={jnp.max(result.gbh)}"
    )
    assert jnp.all(result.gbv < 100.0), (
        f"gbv contains unreasonably large values: max={jnp.max(result.gbv)}"
    )
    assert jnp.all(result.gbc < 100.0), (
        f"gbc contains unreasonably large values: max={jnp.max(result.gbc)}"
    )


# ============================================================================
# Value Tests
# ============================================================================


def test_leaf_boundary_layer_nominal_values(test_data):
    """
    Test that leaf_boundary_layer produces reasonable values for nominal conditions.
    
    For typical temperate forest conditions, conductances should be in the range
    of 0.1-10 mol/m2/s, with heat conductance typically similar to vapor conductance.
    """
    test_case = test_data["nominal_single_patch_single_layer"]
    inputs = test_case["inputs"]
    
    result = leaf_boundary_layer(**inputs)
    
    # Typical boundary layer conductances are 0.1-10 mol/m2/s
    assert 0.1 < result.gbh[0, 0] < 10.0, (
        f"gbh value {result.gbh[0, 0]} outside expected range [0.1, 10.0]"
    )
    assert 0.1 < result.gbv[0, 0] < 10.0, (
        f"gbv value {result.gbv[0, 0]} outside expected range [0.1, 10.0]"
    )
    assert 0.1 < result.gbc[0, 0] < 10.0, (
        f"gbc value {result.gbc[0, 0]} outside expected range [0.1, 10.0]"
    )


def test_leaf_boundary_layer_wind_scaling(test_data):
    """
    Test that conductances increase with wind speed.
    
    Higher wind speeds should lead to higher boundary layer conductances
    due to increased turbulent mixing.
    """
    test_case = test_data["nominal_multi_patch_multi_layer"]
    inputs = test_case["inputs"]
    
    result = leaf_boundary_layer(**inputs)
    
    # For patch 1 (hot/windy), conductances should generally decrease with depth
    # as wind decreases (layers 0->1->2->3 have decreasing wind)
    gbh_patch1 = result.gbh[1, :]
    
    # Check that at least the first layer has higher conductance than the last
    assert gbh_patch1[0] > gbh_patch1[-1], (
        f"Expected gbh to decrease with wind speed, but got "
        f"gbh[0]={gbh_patch1[0]} <= gbh[-1]={gbh_patch1[-1]}"
    )


def test_leaf_boundary_layer_leaf_size_scaling(test_data):
    """
    Test that conductances scale appropriately with leaf size.
    
    Smaller leaves should have higher boundary layer conductances
    (lower resistance) due to thinner boundary layers.
    """
    test_case = test_data["special_tiny_leaves"]
    inputs = test_case["inputs"]
    
    result = leaf_boundary_layer(**inputs)
    
    # Patch 2 has smallest leaves (0.0005m), patch 1 has largest (0.002m)
    # At similar conditions, smaller leaves should have higher conductance
    gbh_tiny = result.gbh[2, 0]
    gbh_small = result.gbh[1, 0]
    
    assert gbh_tiny > gbh_small, (
        f"Expected smaller leaves to have higher conductance, but got "
        f"gbh_tiny={gbh_tiny} <= gbh_small={gbh_small}"
    )


def test_leaf_boundary_layer_temperature_gradient_effect(test_data):
    """
    Test that large temperature gradients enhance conductance via free convection.
    
    When leaf temperature is much higher than air temperature, free convection
    should contribute to boundary layer conductance.
    """
    test_case = test_data["edge_extreme_temperature_gradient"]
    inputs = test_case["inputs"]
    
    result = leaf_boundary_layer(**inputs)
    
    # With 30K temperature difference and low wind, free convection should be significant
    # Conductances should still be reasonable (not zero, not excessive)
    assert jnp.all(result.gbh > 0.05), (
        f"Expected significant conductance with large temp gradient, "
        f"but got min gbh={jnp.min(result.gbh)}"
    )


def test_leaf_boundary_layer_near_zero_wind(test_data):
    """
    Test that near-zero wind conditions don't cause numerical issues.
    
    With very low wind speeds, free convection should dominate and
    conductances should remain positive and finite.
    """
    test_case = test_data["edge_near_zero_wind"]
    inputs = test_case["inputs"]
    
    result = leaf_boundary_layer(**inputs)
    
    # Even with near-zero wind, conductances should be positive
    assert jnp.all(result.gbh > 0), (
        "Expected positive conductances even with near-zero wind"
    )
    assert jnp.all(jnp.isfinite(result.gbh)), (
        "Near-zero wind caused non-finite conductances"
    )


def test_leaf_boundary_layer_high_altitude(test_data):
    """
    Test that high altitude conditions (low pressure/density) are handled correctly.
    
    Lower air density should affect diffusivities and conductances, but
    results should remain physically reasonable.
    """
    test_case = test_data["special_high_altitude"]
    inputs = test_case["inputs"]
    
    result = leaf_boundary_layer(**inputs)
    
    # At high altitude, conductances may be different but should be reasonable
    assert jnp.all(result.gbh > 0.05), (
        f"High altitude conductances too low: min gbh={jnp.min(result.gbh)}"
    )
    assert jnp.all(result.gbh < 20.0), (
        f"High altitude conductances too high: max gbh={jnp.max(result.gbh)}"
    )


# ============================================================================
# Relative Conductance Tests
# ============================================================================


def test_leaf_boundary_layer_conductance_ratios():
    """
    Test that relative magnitudes of gbh, gbv, and gbc are physically reasonable.
    
    Based on molecular diffusivities:
    - dv (water vapor) > dh (heat) > dc (CO2)
    - Therefore: gbv should be slightly > gbh, and gbh should be > gbc
    """
    # Simple test case
    dleaf = jnp.array([0.05])
    tref = jnp.array([298.15])
    pref = jnp.array([101325.0])
    wind = jnp.array([[2.0]])
    tair = jnp.array([[295.15]])
    tleaf = jnp.array([[298.15]])
    rhomol = jnp.array([[41.0]])
    dpai = jnp.array([[1.5]])
    
    result = leaf_boundary_layer(
        dleaf=dleaf,
        tref=tref,
        pref=pref,
        wind=wind,
        tair=tair,
        tleaf=tleaf,
        rhomol=rhomol,
        dpai=dpai,
        params=None,
    )
    
    gbh = result.gbh[0, 0]
    gbv = result.gbv[0, 0]
    gbc = result.gbc[0, 0]
    
    # Water vapor diffusivity > heat diffusivity
    assert gbv >= gbh * 0.95, (
        f"Expected gbv >= gbh, but got gbv={gbv}, gbh={gbh}"
    )
    
    # Heat diffusivity > CO2 diffusivity
    assert gbh >= gbc * 0.95, (
        f"Expected gbh >= gbc, but got gbh={gbh}, gbc={gbc}"
    )


# ============================================================================
# Custom Parameters Tests
# ============================================================================


def test_leaf_boundary_layer_custom_params(default_params):
    """
    Test that custom BoundaryLayerParams are properly used.
    
    Modifying gb_factor should scale the conductances proportionally.
    """
    # Simple test case
    dleaf = jnp.array([0.05])
    tref = jnp.array([298.15])
    pref = jnp.array([101325.0])
    wind = jnp.array([[2.0]])
    tair = jnp.array([[295.15]])
    tleaf = jnp.array([[298.15]])
    rhomol = jnp.array([[41.0]])
    dpai = jnp.array([[1.5]])
    
    # Run with default params
    result_default = leaf_boundary_layer(
        dleaf=dleaf,
        tref=tref,
        pref=pref,
        wind=wind,
        tair=tair,
        tleaf=tleaf,
        rhomol=rhomol,
        dpai=dpai,
        params=None,
    )
    
    # Run with doubled gb_factor
    custom_params = BoundaryLayerParams(
        gb_type=default_params.gb_type,
        gb_factor=2.0,  # Double the factor
        visc0=default_params.visc0,
        dh0=default_params.dh0,
        dv0=default_params.dv0,
        dc0=default_params.dc0,
        tfrz=default_params.tfrz,
        grav=default_params.grav,
    )
    
    result_custom = leaf_boundary_layer(
        dleaf=dleaf,
        tref=tref,
        pref=pref,
        wind=wind,
        tair=tair,
        tleaf=tleaf,
        rhomol=rhomol,
        dpai=dpai,
        params=custom_params,
    )
    
    # With doubled gb_factor, conductances should increase
    # (exact relationship depends on gb_type and convection regime)
    assert result_custom.gbh[0, 0] > result_default.gbh[0, 0], (
        "Increased gb_factor should increase conductances"
    )


def test_leaf_boundary_layer_gb_type_parameter(default_params):
    """
    Test that different gb_type values produce different results.
    
    Different calculation methods (gb_type 0-3) should yield different
    conductance values.
    """
    # Simple test case
    dleaf = jnp.array([0.05])
    tref = jnp.array([298.15])
    pref = jnp.array([101325.0])
    wind = jnp.array([[2.0]])
    tair = jnp.array([[295.15]])
    tleaf = jnp.array([[298.15]])
    rhomol = jnp.array([[41.0]])
    dpai = jnp.array([[1.5]])
    
    results = []
    for gb_type in [0, 1, 2, 3]:
        params = BoundaryLayerParams(
            gb_type=gb_type,
            gb_factor=default_params.gb_factor,
            visc0=default_params.visc0,
            dh0=default_params.dh0,
            dv0=default_params.dv0,
            dc0=default_params.dc0,
            tfrz=default_params.tfrz,
            grav=default_params.grav,
        )
        
        result = leaf_boundary_layer(
            dleaf=dleaf,
            tref=tref,
            pref=pref,
            wind=wind,
            tair=tair,
            tleaf=tleaf,
            rhomol=rhomol,
            dpai=dpai,
            params=params,
        )
        results.append(result.gbh[0, 0])
    
    # At least some gb_types should produce different results
    assert not jnp.allclose(results[0], results[1], rtol=0.01) or \
           not jnp.allclose(results[1], results[2], rtol=0.01) or \
           not jnp.allclose(results[2], results[3], rtol=0.01), (
        f"Different gb_type values should produce different results, "
        f"but got similar values: {results}"
    )


# ============================================================================
# Sunlit/Shaded Function Tests
# ============================================================================


def test_leaf_boundary_layer_sunlit_shaded_shapes():
    """
    Test that leaf_boundary_layer_sunlit_shaded returns correct shapes.
    
    Should return two LeafBoundaryLayerOutputs tuples, one for sunlit
    and one for shaded leaves, both with correct shapes.
    """
    dleaf = jnp.array([0.05, 0.06])
    tref = jnp.array([298.15, 300.15])
    pref = jnp.array([101325.0, 101325.0])
    wind = jnp.array([[2.0, 1.5], [2.5, 2.0]])
    tair = jnp.array([[295.15, 296.15], [298.15, 299.15]])
    tleaf_sun = jnp.array([[300.15, 301.15], [302.15, 303.15]])
    tleaf_sha = jnp.array([[296.15, 297.15], [299.15, 300.15]])
    rhomol = jnp.array([[41.0, 40.8], [40.5, 40.3]])
    dpai = jnp.array([[1.5, 1.0], [2.0, 1.5]])
    
    result_sun, result_sha = leaf_boundary_layer_sunlit_shaded(
        dleaf=dleaf,
        tref=tref,
        pref=pref,
        wind=wind,
        tair=tair,
        tleaf_sun=tleaf_sun,
        tleaf_sha=tleaf_sha,
        rhomol=rhomol,
        dpai=dpai,
        params=None,
    )
    
    expected_shape = (2, 2)
    
    assert result_sun.gbh.shape == expected_shape
    assert result_sun.gbv.shape == expected_shape
    assert result_sun.gbc.shape == expected_shape
    
    assert result_sha.gbh.shape == expected_shape
    assert result_sha.gbv.shape == expected_shape
    assert result_sha.gbc.shape == expected_shape


def test_leaf_boundary_layer_sunlit_shaded_temperature_effect():
    """
    Test that sunlit leaves (higher temp) have different conductances than shaded.
    
    Sunlit leaves are typically warmer, which affects free convection and
    should result in different boundary layer conductances.
    """
    dleaf = jnp.array([0.05])
    tref = jnp.array([298.15])
    pref = jnp.array([101325.0])
    wind = jnp.array([[1.0]])
    tair = jnp.array([[295.15]])
    tleaf_sun = jnp.array([[305.15]])  # 10K warmer than air
    tleaf_sha = jnp.array([[297.15]])  # 2K warmer than air
    rhomol = jnp.array([[41.0]])
    dpai = jnp.array([[1.5]])
    
    result_sun, result_sha = leaf_boundary_layer_sunlit_shaded(
        dleaf=dleaf,
        tref=tref,
        pref=pref,
        wind=wind,
        tair=tair,
        tleaf_sun=tleaf_sun,
        tleaf_sha=tleaf_sha,
        rhomol=rhomol,
        dpai=dpai,
        params=None,
    )
    
    # Sunlit leaves with larger temp gradient should have different conductance
    # (likely higher due to free convection, but depends on gb_type)
    assert not jnp.allclose(result_sun.gbh, result_sha.gbh, rtol=0.01), (
        f"Expected different conductances for sunlit vs shaded, but got "
        f"sun={result_sun.gbh[0,0]}, sha={result_sha.gbh[0,0]}"
    )


# ============================================================================
# Consistency Tests
# ============================================================================


def test_leaf_boundary_layer_consistency_across_patches():
    """
    Test that identical patches produce identical results.
    
    When all inputs are the same for multiple patches, outputs should be identical.
    """
    # Create identical patches
    dleaf = jnp.array([0.05, 0.05, 0.05])
    tref = jnp.array([298.15, 298.15, 298.15])
    pref = jnp.array([101325.0, 101325.0, 101325.0])
    wind = jnp.array([[2.0, 1.5], [2.0, 1.5], [2.0, 1.5]])
    tair = jnp.array([[295.15, 296.15], [295.15, 296.15], [295.15, 296.15]])
    tleaf = jnp.array([[298.15, 299.15], [298.15, 299.15], [298.15, 299.15]])
    rhomol = jnp.array([[41.0, 40.8], [41.0, 40.8], [41.0, 40.8]])
    dpai = jnp.array([[1.5, 1.0], [1.5, 1.0], [1.5, 1.0]])
    
    result = leaf_boundary_layer(
        dleaf=dleaf,
        tref=tref,
        pref=pref,
        wind=wind,
        tair=tair,
        tleaf=tleaf,
        rhomol=rhomol,
        dpai=dpai,
        params=None,
    )
    
    # All patches should have identical results
    assert jnp.allclose(result.gbh[0, :], result.gbh[1, :], rtol=1e-6), (
        "Identical patches should produce identical gbh"
    )
    assert jnp.allclose(result.gbh[1, :], result.gbh[2, :], rtol=1e-6), (
        "Identical patches should produce identical gbh"
    )


def test_leaf_boundary_layer_deterministic():
    """
    Test that the function is deterministic (same inputs -> same outputs).
    
    Running the function twice with identical inputs should produce
    identical results.
    """
    dleaf = jnp.array([0.05])
    tref = jnp.array([298.15])
    pref = jnp.array([101325.0])
    wind = jnp.array([[2.0]])
    tair = jnp.array([[295.15]])
    tleaf = jnp.array([[298.15]])
    rhomol = jnp.array([[41.0]])
    dpai = jnp.array([[1.5]])
    
    result1 = leaf_boundary_layer(
        dleaf=dleaf,
        tref=tref,
        pref=pref,
        wind=wind,
        tair=tair,
        tleaf=tleaf,
        rhomol=rhomol,
        dpai=dpai,
        params=None,
    )
    
    result2 = leaf_boundary_layer(
        dleaf=dleaf,
        tref=tref,
        pref=pref,
        wind=wind,
        tair=tair,
        tleaf=tleaf,
        rhomol=rhomol,
        dpai=dpai,
        params=None,
    )
    
    assert jnp.allclose(result1.gbh, result2.gbh, rtol=1e-10), (
        "Function should be deterministic"
    )
    assert jnp.allclose(result1.gbv, result2.gbv, rtol=1e-10), (
        "Function should be deterministic"
    )
    assert jnp.allclose(result1.gbc, result2.gbc, rtol=1e-10), (
        "Function should be deterministic"
    )


# ============================================================================
# Edge Case Robustness Tests
# ============================================================================


def test_leaf_boundary_layer_zero_dpai():
    """
    Test that zero plant area index doesn't cause issues.
    
    With zero LAI, there are no leaves, but the function should still
    return valid (likely zero or very small) conductances.
    """
    dleaf = jnp.array([0.05])
    tref = jnp.array([298.15])
    pref = jnp.array([101325.0])
    wind = jnp.array([[2.0]])
    tair = jnp.array([[295.15]])
    tleaf = jnp.array([[298.15]])
    rhomol = jnp.array([[41.0]])
    dpai = jnp.array([[0.0]])  # Zero LAI
    
    result = leaf_boundary_layer(
        dleaf=dleaf,
        tref=tref,
        pref=pref,
        wind=wind,
        tair=tair,
        tleaf=tleaf,
        rhomol=rhomol,
        dpai=dpai,
        params=None,
    )
    
    # Should not crash and should return finite values
    assert jnp.all(jnp.isfinite(result.gbh)), (
        "Zero dpai caused non-finite conductances"
    )


def test_leaf_boundary_layer_equal_temperatures():
    """
    Test that equal leaf and air temperatures don't cause issues.
    
    When tleaf == tair, there's no temperature gradient for free convection,
    but forced convection should still work.
    """
    dleaf = jnp.array([0.05])
    tref = jnp.array([298.15])
    pref = jnp.array([101325.0])
    wind = jnp.array([[2.0]])
    tair = jnp.array([[298.15]])
    tleaf = jnp.array([[298.15]])  # Same as tair
    rhomol = jnp.array([[41.0]])
    dpai = jnp.array([[1.5]])
    
    result = leaf_boundary_layer(
        dleaf=dleaf,
        tref=tref,
        pref=pref,
        wind=wind,
        tair=tair,
        tleaf=tleaf,
        rhomol=rhomol,
        dpai=dpai,
        params=None,
    )
    
    # Should still have positive conductance from forced convection
    assert result.gbh[0, 0] > 0, (
        "Equal temperatures should still produce positive conductance"
    )
    assert jnp.isfinite(result.gbh[0, 0]), (
        "Equal temperatures caused non-finite conductance"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])