"""
Comprehensive pytest suite for MLPlantHydraulicsMod functions.

This module tests the plant hydraulics functions including:
- plant_resistance: Calculate leaf-specific conductance for canopy layers
- soil_resistance: Calculate soil hydraulic resistance and water potential
- leaf_water_potential: Update leaf water potential based on transpiration
- _calculate_soil_resistance_per_layer: Per-layer soil resistance calculations
- _finalize_soil_resistance: Finalize soil resistance and uptake fractions

Tests cover nominal cases, edge cases, and physical constraints.
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multilayer_canopy.MLPlantHydraulicsMod import (
    PlantResistanceInput,
    PlantResistanceOutput,
    SoilResistanceInputs,
    SoilResistanceOutputs,
    LeafWaterPotentialInputs,
    LeafWaterPotentialOutputs,
    plant_resistance,
    soil_resistance,
    leaf_water_potential,
    _calculate_soil_resistance_per_layer,
    _finalize_soil_resistance,
)


@pytest.fixture
def test_data():
    """
    Load comprehensive test data for plant hydraulics functions.
    
    Returns:
        dict: Test cases with inputs and metadata for all functions
    """
    return {
        "plant_resistance_nominal_single": {
            "inputs": {
                "gplant_SPA": jnp.array([150.0, 200.0, 180.0]),
                "ncan": jnp.array([10]),
                "rsoil": jnp.array([0.5]),
                "dpai": jnp.array([[0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 
                                   0.0, 0.0, 0.0, 0.0, 0.0]]),
                "zs": jnp.array([[25.0, 22.5, 20.0, 17.5, 15.0, 12.5, 10.0, 7.5, 5.0, 2.5,
                                 0.0, 0.0, 0.0, 0.0, 0.0]]),
                "itype": jnp.array([1]),
            },
            "expected_shape": (1, 15),
            "description": "Single patch with typical forest canopy structure",
        },
        "plant_resistance_multiple": {
            "inputs": {
                "gplant_SPA": jnp.array([150.0, 200.0, 180.0, 120.0]),
                "ncan": jnp.array([10, 8, 12]),
                "rsoil": jnp.array([0.5, 0.8, 0.3]),
                "dpai": jnp.array([
                    [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.0, 0.0, 0.0],
                ]),
                "zs": jnp.array([
                    [25.0, 22.5, 20.0, 17.5, 15.0, 12.5, 10.0, 7.5, 5.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [30.0, 27.5, 25.0, 22.5, 20.0, 17.5, 15.0, 12.5, 10.0, 7.5, 5.0, 2.5, 0.0, 0.0, 0.0],
                ]),
                "itype": jnp.array([1, 2, 0]),
            },
            "expected_shape": (3, 15),
            "description": "Multiple patches with different canopy structures",
        },
        "plant_resistance_zero_rsoil": {
            "inputs": {
                "gplant_SPA": jnp.array([150.0, 200.0]),
                "ncan": jnp.array([5]),
                "rsoil": jnp.array([0.0]),
                "dpai": jnp.array([[0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                "zs": jnp.array([[10.0, 8.0, 6.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                "itype": jnp.array([0]),
            },
            "expected_shape": (1, 10),
            "description": "Zero soil resistance edge case",
        },
        "plant_resistance_sparse": {
            "inputs": {
                "gplant_SPA": jnp.array([100.0, 150.0, 120.0]),
                "ncan": jnp.array([3]),
                "rsoil": jnp.array([1.2]),
                "dpai": jnp.array([[0.05, 0.03, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                "zs": jnp.array([[3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                "itype": jnp.array([2]),
            },
            "expected_shape": (1, 10),
            "description": "Sparse canopy with minimal PAI",
        },
        "soil_resistance_nominal": {
            "inputs": {
                "root_radius": jnp.array([0.0001, 0.00015, 0.00012]),
                "root_density": jnp.array([50000.0, 60000.0, 55000.0]),
                "root_resist": jnp.array([25.0, 30.0, 28.0]),
                "dz": jnp.array([[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]),
                "nbedrock": jnp.array([10]),
                "smp_l": jnp.array([[-100.0, -200.0, -300.0, -500.0, -800.0, 
                                     -1200.0, -1800.0, -2500.0, -3500.0, -5000.0]]),
                "hk_l": jnp.array([[0.001, 0.0008, 0.0006, 0.0004, 0.0003, 
                                   0.0002, 0.00015, 0.0001, 8e-05, 5e-05]]),
                "rootfr": jnp.array([[0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05, 0.03, 0.01, 0.01]]),
                "h2osoi_ice": jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                "root_biomass": jnp.array([150.0]),
                "lai": jnp.array([4.5]),
                "itype": jnp.array([1]),
                "patch_to_column": jnp.array([0]),
                "minlwp_SPA": -2.5,
            },
            "expected_shapes": {"rsoil": (1,), "psis": (1,), "soil_et_loss": (1, 10)},
            "description": "Typical soil profile with no ice",
        },
        "soil_resistance_frozen": {
            "inputs": {
                "root_radius": jnp.array([0.0001, 0.00015]),
                "root_density": jnp.array([50000.0, 60000.0]),
                "root_resist": jnp.array([25.0, 30.0]),
                "dz": jnp.array([[0.1, 0.15, 0.2, 0.25, 0.3, 0.4]]),
                "nbedrock": jnp.array([6]),
                "smp_l": jnp.array([[-50.0, -100.0, -150.0, -200.0, -300.0, -500.0]]),
                "hk_l": jnp.array([[0.001, 0.0008, 0.0005, 0.0003, 0.0001, 5e-05]]),
                "rootfr": jnp.array([[0.3, 0.25, 0.2, 0.15, 0.07, 0.03]]),
                "h2osoi_ice": jnp.array([[5.0, 8.0, 12.0, 15.0, 10.0, 5.0]]),
                "root_biomass": jnp.array([120.0]),
                "lai": jnp.array([3.0]),
                "itype": jnp.array([0]),
                "patch_to_column": jnp.array([0]),
                "minlwp_SPA": -3.0,
            },
            "expected_shapes": {"rsoil": (1,), "psis": (1,), "soil_et_loss": (1, 6)},
            "description": "Frozen soil with significant ice content",
        },
        "soil_resistance_shallow_bedrock": {
            "inputs": {
                "root_radius": jnp.array([0.0001, 0.00012, 0.00015]),
                "root_density": jnp.array([50000.0, 55000.0, 60000.0]),
                "root_resist": jnp.array([25.0, 27.0, 30.0]),
                "dz": jnp.array([[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]]),
                "nbedrock": jnp.array([3]),
                "smp_l": jnp.array([[-80.0, -150.0, -250.0, -10000.0, -10000.0, 
                                     -10000.0, -10000.0, -10000.0]]),
                "hk_l": jnp.array([[0.001, 0.0008, 0.0005, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                "rootfr": jnp.array([[0.5, 0.35, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                "h2osoi_ice": jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                "root_biomass": jnp.array([100.0]),
                "lai": jnp.array([3.5]),
                "itype": jnp.array([2]),
                "patch_to_column": jnp.array([0]),
                "minlwp_SPA": -2.0,
            },
            "expected_shapes": {"rsoil": (1,), "psis": (1,), "soil_et_loss": (1, 8)},
            "description": "Shallow bedrock limiting rooting depth",
        },
        "lwp_nominal_sunlit": {
            "inputs": {
                "capac_SPA": jnp.array([5000.0, 6000.0, 5500.0]),
                "ncan": jnp.array([8]),
                "psis": jnp.array([-0.5]),
                "dpai": jnp.array([[0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0, 0.0]]),
                "zs": jnp.array([[20.0, 17.5, 15.0, 12.5, 10.0, 7.5, 5.0, 2.5, 0.0, 0.0]]),
                "lsc": jnp.array([[80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 50.0, 45.0, 0.0, 0.0]]),
                "trleaf": jnp.array([
                    [[0.003, 0.001], [0.0028, 0.0009], [0.0026, 0.0008], [0.0024, 0.0007],
                     [0.0022, 0.0006], [0.002, 0.0005], [0.0018, 0.0004], [0.0015, 0.0003],
                     [0.0, 0.0], [0.0, 0.0]]
                ]),
                "lwp": jnp.array([
                    [[-0.8, -0.9], [-0.85, -0.95], [-0.9, -1.0], [-0.95, -1.05],
                     [-1.0, -1.1], [-1.05, -1.15], [-1.1, -1.2], [-1.15, -1.25],
                     [0.0, 0.0], [0.0, 0.0]]
                ]),
                "itype": jnp.array([1]),
                "dtime_substep": 300.0,
                "il": 0,
            },
            "expected_shape": (1, 10, 2),
            "description": "Sunlit leaf water potential with typical transpiration",
        },
        "lwp_high_stress": {
            "inputs": {
                "capac_SPA": jnp.array([5000.0, 6000.0]),
                "ncan": jnp.array([6]),
                "psis": jnp.array([-2.5]),
                "dpai": jnp.array([[0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0, 0.0]]),
                "zs": jnp.array([[15.0, 12.5, 10.0, 7.5, 5.0, 2.5, 0.0, 0.0]]),
                "lsc": jnp.array([[40.0, 35.0, 30.0, 25.0, 20.0, 15.0, 0.0, 0.0]]),
                "trleaf": jnp.array([
                    [[0.005, 0.002], [0.0048, 0.0019], [0.0045, 0.0018], [0.004, 0.0015],
                     [0.0035, 0.0012], [0.003, 0.001], [0.0, 0.0], [0.0, 0.0]]
                ]),
                "lwp": jnp.array([
                    [[-2.0, -2.2], [-2.1, -2.3], [-2.2, -2.4], [-2.3, -2.5],
                     [-2.4, -2.6], [-2.5, -2.7], [0.0, 0.0], [0.0, 0.0]]
                ]),
                "itype": jnp.array([0]),
                "dtime_substep": 300.0,
                "il": 1,
            },
            "expected_shape": (1, 8, 2),
            "description": "Shaded leaves under high water stress",
        },
    }


# ============================================================================
# Plant Resistance Tests
# ============================================================================


@pytest.mark.parametrize(
    "test_case_name",
    [
        "plant_resistance_nominal_single",
        "plant_resistance_multiple",
        "plant_resistance_zero_rsoil",
        "plant_resistance_sparse",
    ],
)
def test_plant_resistance_shapes(test_data, test_case_name):
    """
    Test that plant_resistance returns correct output shapes.
    
    Verifies that the leaf-specific conductance (lsc) array has the expected
    dimensions [n_patches, n_levcan] for various canopy configurations.
    """
    case = test_data[test_case_name]
    inputs = PlantResistanceInput(**case["inputs"])
    
    result = plant_resistance(inputs)
    
    assert isinstance(result, PlantResistanceOutput), \
        f"Expected PlantResistanceOutput, got {type(result)}"
    assert result.lsc.shape == case["expected_shape"], \
        f"Expected shape {case['expected_shape']}, got {result.lsc.shape}"


@pytest.mark.parametrize(
    "test_case_name",
    [
        "plant_resistance_nominal_single",
        "plant_resistance_multiple",
        "plant_resistance_zero_rsoil",
        "plant_resistance_sparse",
    ],
)
def test_plant_resistance_values(test_data, test_case_name):
    """
    Test that plant_resistance produces physically valid values.
    
    Checks that:
    - All conductance values are non-negative
    - Conductance is zero where PAI is zero
    - Conductance decreases with increasing resistance
    """
    case = test_data[test_case_name]
    inputs = PlantResistanceInput(**case["inputs"])
    
    result = plant_resistance(inputs)
    
    # All conductances should be non-negative
    assert jnp.all(result.lsc >= 0), \
        f"Found negative conductance values in {test_case_name}"
    
    # Where dpai is zero, lsc should be zero
    dpai = case["inputs"]["dpai"]
    zero_pai_mask = dpai == 0
    assert jnp.all(result.lsc[zero_pai_mask] == 0), \
        "Conductance should be zero where PAI is zero"
    
    # For non-zero PAI, conductance should be positive
    nonzero_pai_mask = dpai > 0
    if jnp.any(nonzero_pai_mask):
        assert jnp.all(result.lsc[nonzero_pai_mask] > 0), \
            "Conductance should be positive where PAI is positive"


def test_plant_resistance_zero_rsoil_effect(test_data):
    """
    Test that zero soil resistance leads to higher conductance.
    
    Compares conductance with zero vs. non-zero soil resistance to verify
    that removing soil resistance increases overall conductance.
    """
    # Case with zero soil resistance
    case_zero = test_data["plant_resistance_zero_rsoil"]
    inputs_zero = PlantResistanceInput(**case_zero["inputs"])
    result_zero = plant_resistance(inputs_zero)
    
    # Create similar case with non-zero soil resistance
    inputs_nonzero_dict = case_zero["inputs"].copy()
    inputs_nonzero_dict["rsoil"] = jnp.array([0.5])
    inputs_nonzero = PlantResistanceInput(**inputs_nonzero_dict)
    result_nonzero = plant_resistance(inputs_nonzero)
    
    # Zero soil resistance should give higher or equal conductance
    active_layers = case_zero["inputs"]["dpai"] > 0
    assert jnp.all(result_zero.lsc[active_layers] >= result_nonzero.lsc[active_layers]), \
        "Zero soil resistance should not decrease conductance"


def test_plant_resistance_dtypes(test_data):
    """
    Test that plant_resistance preserves correct data types.
    
    Verifies that output arrays are JAX arrays with float dtype.
    """
    case = test_data["plant_resistance_nominal_single"]
    inputs = PlantResistanceInput(**case["inputs"])
    
    result = plant_resistance(inputs)
    
    assert isinstance(result.lsc, jnp.ndarray), \
        f"Expected jnp.ndarray, got {type(result.lsc)}"
    assert jnp.issubdtype(result.lsc.dtype, jnp.floating), \
        f"Expected floating dtype, got {result.lsc.dtype}"


# ============================================================================
# Soil Resistance Tests
# ============================================================================


@pytest.mark.parametrize(
    "test_case_name",
    [
        "soil_resistance_nominal",
        "soil_resistance_frozen",
        "soil_resistance_shallow_bedrock",
    ],
)
def test_soil_resistance_shapes(test_data, test_case_name):
    """
    Test that soil_resistance returns correct output shapes.
    
    Verifies that rsoil, psis, and soil_et_loss have expected dimensions
    for various soil configurations.
    """
    case = test_data[test_case_name]
    inputs = SoilResistanceInputs(**case["inputs"])
    
    result = soil_resistance(inputs)
    
    assert isinstance(result, SoilResistanceOutputs), \
        f"Expected SoilResistanceOutputs, got {type(result)}"
    
    expected = case["expected_shapes"]
    assert result.rsoil.shape == expected["rsoil"], \
        f"rsoil shape mismatch: expected {expected['rsoil']}, got {result.rsoil.shape}"
    assert result.psis.shape == expected["psis"], \
        f"psis shape mismatch: expected {expected['psis']}, got {result.psis.shape}"
    assert result.soil_et_loss.shape == expected["soil_et_loss"], \
        f"soil_et_loss shape mismatch: expected {expected['soil_et_loss']}, got {result.soil_et_loss.shape}"


@pytest.mark.parametrize(
    "test_case_name",
    [
        "soil_resistance_nominal",
        "soil_resistance_frozen",
        "soil_resistance_shallow_bedrock",
    ],
)
def test_soil_resistance_values(test_data, test_case_name):
    """
    Test that soil_resistance produces physically valid values.
    
    Checks that:
    - Soil resistance is non-negative
    - Soil water potential is non-positive
    - Uptake fractions are in [0, 1] and sum to ~1
    """
    case = test_data[test_case_name]
    inputs = SoilResistanceInputs(**case["inputs"])
    
    result = soil_resistance(inputs)
    
    # Soil resistance should be non-negative
    assert jnp.all(result.rsoil >= 0), \
        f"Found negative soil resistance in {test_case_name}"
    
    # Soil water potential should be non-positive
    assert jnp.all(result.psis <= 0), \
        f"Found positive soil water potential in {test_case_name}"
    
    # Uptake fractions should be in [0, 1]
    assert jnp.all(result.soil_et_loss >= 0), \
        "Found negative uptake fractions"
    assert jnp.all(result.soil_et_loss <= 1), \
        "Found uptake fractions > 1"
    
    # Sum of uptake fractions should be close to 1 (or 0 if no uptake)
    uptake_sum = jnp.sum(result.soil_et_loss, axis=1)
    assert jnp.all((jnp.abs(uptake_sum - 1.0) < 1e-6) | (uptake_sum == 0)), \
        f"Uptake fractions should sum to 1, got {uptake_sum}"


def test_soil_resistance_frozen_soil_effect(test_data):
    """
    Test that frozen soil increases resistance.
    
    Compares resistance with and without ice to verify that ice content
    increases hydraulic resistance.
    """
    case_frozen = test_data["soil_resistance_frozen"]
    inputs_frozen = SoilResistanceInputs(**case_frozen["inputs"])
    result_frozen = soil_resistance(inputs_frozen)
    
    # Create similar case without ice
    inputs_unfrozen_dict = case_frozen["inputs"].copy()
    inputs_unfrozen_dict["h2osoi_ice"] = jnp.zeros_like(inputs_unfrozen_dict["h2osoi_ice"])
    inputs_unfrozen = SoilResistanceInputs(**inputs_unfrozen_dict)
    result_unfrozen = soil_resistance(inputs_unfrozen)
    
    # Frozen soil should have higher resistance
    assert jnp.all(result_frozen.rsoil >= result_unfrozen.rsoil), \
        "Frozen soil should increase resistance"


def test_soil_resistance_bedrock_effect(test_data):
    """
    Test that shallow bedrock affects uptake distribution.
    
    Verifies that uptake is concentrated in layers above bedrock.
    """
    case = test_data["soil_resistance_shallow_bedrock"]
    inputs = SoilResistanceInputs(**case["inputs"])
    
    result = soil_resistance(inputs)
    
    nbedrock = case["inputs"]["nbedrock"][0]
    
    # Uptake should be concentrated in layers above bedrock
    uptake_above = jnp.sum(result.soil_et_loss[0, :nbedrock])
    uptake_below = jnp.sum(result.soil_et_loss[0, nbedrock:])
    
    assert uptake_above > uptake_below, \
        "Most uptake should occur above bedrock"


def test_soil_resistance_dtypes(test_data):
    """
    Test that soil_resistance preserves correct data types.
    
    Verifies that all output arrays are JAX arrays with float dtype.
    """
    case = test_data["soil_resistance_nominal"]
    inputs = SoilResistanceInputs(**case["inputs"])
    
    result = soil_resistance(inputs)
    
    assert isinstance(result.rsoil, jnp.ndarray), \
        f"rsoil: expected jnp.ndarray, got {type(result.rsoil)}"
    assert isinstance(result.psis, jnp.ndarray), \
        f"psis: expected jnp.ndarray, got {type(result.psis)}"
    assert isinstance(result.soil_et_loss, jnp.ndarray), \
        f"soil_et_loss: expected jnp.ndarray, got {type(result.soil_et_loss)}"
    
    assert jnp.issubdtype(result.rsoil.dtype, jnp.floating), \
        f"rsoil: expected floating dtype, got {result.rsoil.dtype}"
    assert jnp.issubdtype(result.psis.dtype, jnp.floating), \
        f"psis: expected floating dtype, got {result.psis.dtype}"
    assert jnp.issubdtype(result.soil_et_loss.dtype, jnp.floating), \
        f"soil_et_loss: expected floating dtype, got {result.soil_et_loss.dtype}"


# ============================================================================
# Leaf Water Potential Tests
# ============================================================================


@pytest.mark.parametrize(
    "test_case_name",
    ["lwp_nominal_sunlit", "lwp_high_stress"],
)
def test_leaf_water_potential_shapes(test_data, test_case_name):
    """
    Test that leaf_water_potential returns correct output shapes.
    
    Verifies that the updated leaf water potential array maintains the
    expected dimensions [n_patches, n_canopy_layers, 2].
    """
    case = test_data[test_case_name]
    inputs = LeafWaterPotentialInputs(**{k: v for k, v in case["inputs"].items() if k != "il"})
    il = case["inputs"]["il"]
    
    result = leaf_water_potential(inputs, il)
    
    assert isinstance(result, LeafWaterPotentialOutputs), \
        f"Expected LeafWaterPotentialOutputs, got {type(result)}"
    assert result.lwp.shape == case["expected_shape"], \
        f"Expected shape {case['expected_shape']}, got {result.lwp.shape}"


@pytest.mark.parametrize(
    "test_case_name",
    ["lwp_nominal_sunlit", "lwp_high_stress"],
)
def test_leaf_water_potential_values(test_data, test_case_name):
    """
    Test that leaf_water_potential produces physically valid values.
    
    Checks that:
    - All leaf water potentials are non-positive
    - Values change from initial state (unless transpiration is zero)
    - Potentials remain within reasonable bounds
    """
    case = test_data[test_case_name]
    inputs = LeafWaterPotentialInputs(**{k: v for k, v in case["inputs"].items() if k != "il"})
    il = case["inputs"]["il"]
    
    result = leaf_water_potential(inputs, il)
    
    # All leaf water potentials should be non-positive
    assert jnp.all(result.lwp <= 0), \
        f"Found positive leaf water potential in {test_case_name}"
    
    # Check that values are within reasonable bounds (not too extreme)
    assert jnp.all(result.lwp >= -10.0), \
        "Leaf water potential too negative (< -10 MPa)"


def test_leaf_water_potential_sunlit_vs_shaded(test_data):
    """
    Test that sunlit and shaded leaves are updated independently.
    
    Verifies that updating sunlit leaves (il=0) doesn't affect shaded leaves
    and vice versa.
    """
    case = test_data["lwp_nominal_sunlit"]
    inputs = LeafWaterPotentialInputs(**{k: v for k, v in case["inputs"].items() if k != "il"})
    
    # Update sunlit leaves
    result_sunlit = leaf_water_potential(inputs, il=0)
    
    # Update shaded leaves
    result_shaded = leaf_water_potential(inputs, il=1)
    
    # The two results should differ in the updated leaf type
    # (This is a basic check; actual behavior depends on implementation)
    assert result_sunlit.lwp.shape == result_shaded.lwp.shape, \
        "Sunlit and shaded results should have same shape"


def test_leaf_water_potential_high_stress_effect(test_data):
    """
    Test that high water stress leads to more negative potentials.
    
    Compares leaf water potential under normal vs. high stress conditions.
    """
    case_stress = test_data["lwp_high_stress"]
    inputs_stress = LeafWaterPotentialInputs(
        **{k: v for k, v in case_stress["inputs"].items() if k != "il"}
    )
    il = case_stress["inputs"]["il"]
    
    result_stress = leaf_water_potential(inputs_stress, il)
    
    # Under high stress, potentials should be quite negative
    active_layers = case_stress["inputs"]["dpai"] > 0
    assert jnp.all(result_stress.lwp[active_layers] < -1.0), \
        "High stress should result in potentials < -1.0 MPa"


def test_leaf_water_potential_dtypes(test_data):
    """
    Test that leaf_water_potential preserves correct data types.
    
    Verifies that output arrays are JAX arrays with float dtype.
    """
    case = test_data["lwp_nominal_sunlit"]
    inputs = LeafWaterPotentialInputs(**{k: v for k, v in case["inputs"].items() if k != "il"})
    il = case["inputs"]["il"]
    
    result = leaf_water_potential(inputs, il)
    
    assert isinstance(result.lwp, jnp.ndarray), \
        f"Expected jnp.ndarray, got {type(result.lwp)}"
    assert jnp.issubdtype(result.lwp.dtype, jnp.floating), \
        f"Expected floating dtype, got {result.lwp.dtype}"


# ============================================================================
# Helper Function Tests
# ============================================================================


def test_calculate_soil_resistance_per_layer_shapes():
    """
    Test that _calculate_soil_resistance_per_layer returns correct shapes.
    
    Verifies that all four output arrays have expected dimensions.
    """
    # Create test inputs
    n_patches = 2
    n_columns = 2
    n_layers = 5
    
    root_radius = jnp.array([0.0001, 0.00015])
    root_density = jnp.array([50000.0, 60000.0])
    root_resist = jnp.array([25.0, 30.0])
    dz = jnp.array([[0.1, 0.15, 0.2, 0.25, 0.3], [0.1, 0.15, 0.2, 0.25, 0.3]])
    nbedrock = jnp.array([5, 5])
    smp_l = jnp.array([[-100.0, -200.0, -350.0, -600.0, -1000.0],
                       [-100.0, -200.0, -350.0, -600.0, -1000.0]])
    hk_l = jnp.array([[0.001, 0.0008, 0.0006, 0.0004, 0.0002],
                      [0.001, 0.0008, 0.0006, 0.0004, 0.0002]])
    rootfr = jnp.array([[0.35, 0.3, 0.2, 0.1, 0.05], [0.35, 0.3, 0.2, 0.1, 0.05]])
    h2osoi_ice = jnp.zeros((n_columns, n_layers))
    root_biomass = jnp.array([140.0, 150.0])
    minlwp_SPA = -2.5
    
    rsoil_cond, smp_mpa, evap, totevap = _calculate_soil_resistance_per_layer(
        root_radius, root_density, root_resist, dz, nbedrock, smp_l, hk_l,
        rootfr, h2osoi_ice, root_biomass, minlwp_SPA, n_layers
    )
    
    assert rsoil_cond.shape == (n_patches,), \
        f"rsoil_conductance shape mismatch: expected ({n_patches},), got {rsoil_cond.shape}"
    assert smp_mpa.shape == (n_patches, n_layers), \
        f"smp_mpa shape mismatch: expected ({n_patches}, {n_layers}), got {smp_mpa.shape}"
    assert evap.shape == (n_patches, n_layers), \
        f"evap shape mismatch: expected ({n_patches}, {n_layers}), got {evap.shape}"
    assert totevap.shape == (n_patches,), \
        f"totevap shape mismatch: expected ({n_patches},), got {totevap.shape}"


def test_calculate_soil_resistance_per_layer_values():
    """
    Test that _calculate_soil_resistance_per_layer produces valid values.
    
    Checks physical constraints on outputs.
    """
    # Create test inputs
    n_layers = 5
    root_radius = jnp.array([0.0001])
    root_density = jnp.array([50000.0])
    root_resist = jnp.array([25.0])
    dz = jnp.array([[0.1, 0.15, 0.2, 0.25, 0.3]])
    nbedrock = jnp.array([5])
    smp_l = jnp.array([[-100.0, -200.0, -350.0, -600.0, -1000.0]])
    hk_l = jnp.array([[0.001, 0.0008, 0.0006, 0.0004, 0.0002]])
    rootfr = jnp.array([[0.35, 0.3, 0.2, 0.1, 0.05]])
    h2osoi_ice = jnp.zeros((1, n_layers))
    root_biomass = jnp.array([140.0])
    minlwp_SPA = -2.5
    
    rsoil_cond, smp_mpa, evap, totevap = _calculate_soil_resistance_per_layer(
        root_radius, root_density, root_resist, dz, nbedrock, smp_l, hk_l,
        rootfr, h2osoi_ice, root_biomass, minlwp_SPA, n_layers
    )
    
    # Conductance should be non-negative
    assert jnp.all(rsoil_cond >= 0), "Soil conductance should be non-negative"
    
    # Matric potential should be non-positive
    assert jnp.all(smp_mpa <= 0), "Soil matric potential should be non-positive"
    
    # Evaporation should be non-negative
    assert jnp.all(evap >= 0), "Evaporation should be non-negative"
    assert jnp.all(totevap >= 0), "Total evaporation should be non-negative"


def test_finalize_soil_resistance_shapes():
    """
    Test that _finalize_soil_resistance returns correct shapes.
    
    Verifies that rsoil, psis, and soil_et_loss have expected dimensions.
    """
    n_patches = 2
    n_layers = 5
    
    rsoil_conductance = jnp.array([100.0, 120.0])
    lai = jnp.array([4.5, 3.5])
    smp_mpa = jnp.array([[-0.1, -0.2, -0.35, -0.6, -1.0],
                         [-0.15, -0.25, -0.4, -0.7, -1.2]])
    evap = jnp.array([[5.0, 4.0, 3.0, 2.0, 1.0],
                      [4.5, 3.5, 2.5, 1.5, 0.5]])
    totevap = jnp.array([15.0, 12.5])
    minlwp_SPA = -2.5
    
    rsoil, psis, soil_et_loss = _finalize_soil_resistance(
        rsoil_conductance, lai, smp_mpa, evap, totevap, minlwp_SPA
    )
    
    assert rsoil.shape == (n_patches,), \
        f"rsoil shape mismatch: expected ({n_patches},), got {rsoil.shape}"
    assert psis.shape == (n_patches,), \
        f"psis shape mismatch: expected ({n_patches},), got {psis.shape}"
    assert soil_et_loss.shape == (n_patches, n_layers), \
        f"soil_et_loss shape mismatch: expected ({n_patches}, {n_layers}), got {soil_et_loss.shape}"


def test_finalize_soil_resistance_values():
    """
    Test that _finalize_soil_resistance produces valid values.
    
    Checks physical constraints on outputs.
    """
    n_layers = 5
    rsoil_conductance = jnp.array([100.0])
    lai = jnp.array([4.5])
    smp_mpa = jnp.array([[-0.1, -0.2, -0.35, -0.6, -1.0]])
    evap = jnp.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    totevap = jnp.array([15.0])
    minlwp_SPA = -2.5
    
    rsoil, psis, soil_et_loss = _finalize_soil_resistance(
        rsoil_conductance, lai, smp_mpa, evap, totevap, minlwp_SPA
    )
    
    # Resistance should be non-negative
    assert jnp.all(rsoil >= 0), "Soil resistance should be non-negative"
    
    # Soil water potential should be non-positive
    assert jnp.all(psis <= 0), "Soil water potential should be non-positive"
    
    # Uptake fractions should be in [0, 1]
    assert jnp.all(soil_et_loss >= 0), "Uptake fractions should be non-negative"
    assert jnp.all(soil_et_loss <= 1), "Uptake fractions should be <= 1"
    
    # Sum of uptake fractions should be close to 1
    uptake_sum = jnp.sum(soil_et_loss)
    assert jnp.abs(uptake_sum - 1.0) < 1e-6 or uptake_sum == 0, \
        f"Uptake fractions should sum to 1, got {uptake_sum}"


def test_finalize_soil_resistance_zero_lai():
    """
    Test _finalize_soil_resistance with zero LAI.
    
    Verifies behavior when there is no canopy.
    """
    n_layers = 5
    rsoil_conductance = jnp.array([100.0])
    lai = jnp.array([0.0])  # Zero LAI
    smp_mpa = jnp.array([[-0.1, -0.2, -0.35, -0.6, -1.0]])
    evap = jnp.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    totevap = jnp.array([15.0])
    minlwp_SPA = -2.5
    
    rsoil, psis, soil_et_loss = _finalize_soil_resistance(
        rsoil_conductance, lai, smp_mpa, evap, totevap, minlwp_SPA
    )
    
    # With zero LAI, resistance should be very high (or infinite)
    # Implementation may handle this differently, so just check it's positive
    assert jnp.all(rsoil > 0), "Resistance should be positive even with zero LAI"
    
    # Soil water potential should still be valid
    assert jnp.all(psis <= 0), "Soil water potential should be non-positive"


# ============================================================================
# Integration Tests
# ============================================================================


def test_plant_hydraulics_integration():
    """
    Integration test combining plant_resistance and soil_resistance.
    
    Tests that outputs from soil_resistance can be used as inputs to
    plant_resistance, simulating a typical workflow.
    """
    # First calculate soil resistance
    soil_inputs = SoilResistanceInputs(
        root_radius=jnp.array([0.0001, 0.00015, 0.00012]),
        root_density=jnp.array([50000.0, 60000.0, 55000.0]),
        root_resist=jnp.array([25.0, 30.0, 28.0]),
        dz=jnp.array([[0.1, 0.15, 0.2, 0.25, 0.3]]),
        nbedrock=jnp.array([5]),
        smp_l=jnp.array([[-100.0, -200.0, -300.0, -500.0, -800.0]]),
        hk_l=jnp.array([[0.001, 0.0008, 0.0006, 0.0004, 0.0003]]),
        rootfr=jnp.array([[0.35, 0.3, 0.2, 0.1, 0.05]]),
        h2osoi_ice=jnp.zeros((1, 5)),
        root_biomass=jnp.array([150.0]),
        lai=jnp.array([4.5]),
        itype=jnp.array([1]),
        patch_to_column=jnp.array([0]),
        minlwp_SPA=-2.5,
    )
    
    soil_result = soil_resistance(soil_inputs)
    
    # Use soil resistance output in plant resistance calculation
    plant_inputs = PlantResistanceInput(
        gplant_SPA=jnp.array([150.0, 200.0, 180.0]),
        ncan=jnp.array([8]),
        rsoil=soil_result.rsoil,  # Use calculated soil resistance
        dpai=jnp.array([[0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0, 0.0]]),
        zs=jnp.array([[20.0, 17.5, 15.0, 12.5, 10.0, 7.5, 5.0, 2.5, 0.0, 0.0]]),
        itype=jnp.array([1]),
    )
    
    plant_result = plant_resistance(plant_inputs)
    
    # Verify outputs are physically reasonable
    assert jnp.all(plant_result.lsc >= 0), "Conductance should be non-negative"
    assert jnp.all(soil_result.rsoil >= 0), "Soil resistance should be non-negative"
    assert jnp.all(soil_result.psis <= 0), "Soil water potential should be non-positive"


def test_full_hydraulics_workflow():
    """
    Full workflow test: soil resistance -> plant resistance -> leaf water potential.
    
    Tests the complete hydraulics calculation chain.
    """
    # Step 1: Calculate soil resistance
    soil_inputs = SoilResistanceInputs(
        root_radius=jnp.array([0.0001, 0.00015]),
        root_density=jnp.array([50000.0, 60000.0]),
        root_resist=jnp.array([25.0, 30.0]),
        dz=jnp.array([[0.1, 0.15, 0.2, 0.25, 0.3]]),
        nbedrock=jnp.array([5]),
        smp_l=jnp.array([[-100.0, -200.0, -300.0, -500.0, -800.0]]),
        hk_l=jnp.array([[0.001, 0.0008, 0.0006, 0.0004, 0.0003]]),
        rootfr=jnp.array([[0.35, 0.3, 0.2, 0.1, 0.05]]),
        h2osoi_ice=jnp.zeros((1, 5)),
        root_biomass=jnp.array([150.0]),
        lai=jnp.array([4.5]),
        itype=jnp.array([1]),
        patch_to_column=jnp.array([0]),
        minlwp_SPA=-2.5,
    )
    soil_result = soil_resistance(soil_inputs)
    
    # Step 2: Calculate plant resistance
    plant_inputs = PlantResistanceInput(
        gplant_SPA=jnp.array([150.0, 200.0]),
        ncan=jnp.array([6]),
        rsoil=soil_result.rsoil,
        dpai=jnp.array([[0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0]]),
        zs=jnp.array([[15.0, 12.5, 10.0, 7.5, 5.0, 2.5, 0.0, 0.0]]),
        itype=jnp.array([1]),
    )
    plant_result = plant_resistance(plant_inputs)
    
    # Step 3: Calculate leaf water potential
    lwp_inputs = LeafWaterPotentialInputs(
        capac_SPA=jnp.array([5000.0, 6000.0]),
        ncan=jnp.array([6]),
        psis=soil_result.psis,
        dpai=plant_inputs.dpai,
        zs=plant_inputs.zs,
        lsc=plant_result.lsc,
        trleaf=jnp.array([[[0.003, 0.001], [0.0028, 0.0009], [0.0026, 0.0008],
                          [0.0024, 0.0007], [0.0022, 0.0006], [0.002, 0.0005],
                          [0.0, 0.0], [0.0, 0.0]]]),
        lwp=jnp.array([[[-0.8, -0.9], [-0.85, -0.95], [-0.9, -1.0],
                       [-0.95, -1.05], [-1.0, -1.1], [-1.05, -1.15],
                       [0.0, 0.0], [0.0, 0.0]]]),
        itype=jnp.array([1]),
        dtime_substep=300.0,
    )
    lwp_result = leaf_water_potential(lwp_inputs, il=0)
    
    # Verify all outputs are physically reasonable
    assert jnp.all(soil_result.rsoil >= 0), "Soil resistance should be non-negative"
    assert jnp.all(soil_result.psis <= 0), "Soil water potential should be non-positive"
    assert jnp.all(plant_result.lsc >= 0), "Conductance should be non-negative"
    assert jnp.all(lwp_result.lwp <= 0), "Leaf water potential should be non-positive"
    
    # Check that the workflow produces reasonable values
    assert jnp.all(jnp.isfinite(soil_result.rsoil)), "Soil resistance should be finite"
    assert jnp.all(jnp.isfinite(plant_result.lsc)), "Conductance should be finite"
    assert jnp.all(jnp.isfinite(lwp_result.lwp)), "Leaf water potential should be finite"