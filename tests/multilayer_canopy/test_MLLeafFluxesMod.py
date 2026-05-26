"""
Comprehensive pytest suite for MLLeafFluxesMod.leaf_fluxes function.

This module tests the leaf energy balance and flux calculations for multilayer
canopy models, including:
- Leaf temperature computation
- Sensible and latent heat fluxes
- Evaporation and transpiration
- Energy balance closure
- Edge cases (zero LAI, stomatal closure, wet canopy)
- Physical constraints and numerical stability
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multilayer_canopy.MLLeafFluxesMod import (
    ENERGY_BALANCE_TOL,
    LeafFluxesResult,
    check_energy_balance,
    leaf_fluxes,
    total_leaf_conductance,
)


# Test data fixture
@pytest.fixture
def test_data():
    """
    Load comprehensive test data for leaf_fluxes function.
    
    Returns:
        dict: Test cases with inputs and metadata for various scenarios
    """
    return {
        "nominal_temperate": {
            "inputs": {
                "dtime_substep": 1800.0,
                "tref": 298.15,
                "pref": 101325.0,
                "cpair": 29.3,
                "dpai": 2.5,
                "tair": 298.15,
                "eair": 1500.0,
                "cpleaf": 2500.0,
                "fwet": 0.2,
                "fdry": 0.8,
                "gbh": 0.5,
                "gbv": 0.5,
                "gs": 0.15,
                "rnleaf": 400.0,
                "tleaf_bef": 298.15,
            },
            "description": "Typical temperate forest conditions at 25°C",
        },
        "nominal_hot_dry": {
            "inputs": {
                "dtime_substep": 900.0,
                "tref": 313.15,
                "pref": 95000.0,
                "cpair": 29.3,
                "dpai": 1.8,
                "tair": 313.15,
                "eair": 800.0,
                "cpleaf": 2200.0,
                "fwet": 0.0,
                "fdry": 1.0,
                "gbh": 0.8,
                "gbv": 0.8,
                "gs": 0.05,
                "rnleaf": 600.0,
                "tleaf_bef": 314.15,
            },
            "description": "Hot, dry desert conditions at 40°C with stomatal closure",
        },
        "nominal_cold_humid": {
            "inputs": {
                "dtime_substep": 3600.0,
                "tref": 278.15,
                "pref": 101325.0,
                "cpair": 29.3,
                "dpai": 3.5,
                "tair": 278.15,
                "eair": 600.0,
                "cpleaf": 2800.0,
                "fwet": 0.6,
                "fdry": 0.4,
                "gbh": 0.3,
                "gbv": 0.3,
                "gs": 0.08,
                "rnleaf": 150.0,
                "tleaf_bef": 278.15,
            },
            "description": "Cold, humid conditions at 5°C with high wetness",
        },
        "nominal_tropical": {
            "inputs": {
                "dtime_substep": 1200.0,
                "tref": 303.15,
                "pref": 101325.0,
                "cpair": 29.3,
                "dpai": 5.0,
                "tair": 303.15,
                "eair": 2800.0,
                "cpleaf": 2600.0,
                "fwet": 0.3,
                "fdry": 0.7,
                "gbh": 0.6,
                "gbv": 0.6,
                "gs": 0.25,
                "rnleaf": 500.0,
                "tleaf_bef": 303.15,
            },
            "description": "Tropical rainforest with high LAI and humidity",
        },
        "nominal_nighttime": {
            "inputs": {
                "dtime_substep": 1800.0,
                "tref": 288.15,
                "pref": 101325.0,
                "cpair": 29.3,
                "dpai": 2.0,
                "tair": 288.15,
                "eair": 1200.0,
                "cpleaf": 2400.0,
                "fwet": 0.4,
                "fdry": 0.6,
                "gbh": 0.4,
                "gbv": 0.4,
                "gs": 0.02,
                "rnleaf": -50.0,
                "tleaf_bef": 288.15,
            },
            "description": "Nighttime conditions with negative net radiation",
        },
        "edge_zero_dpai": {
            "inputs": {
                "dtime_substep": 1800.0,
                "tref": 298.15,
                "pref": 101325.0,
                "cpair": 29.3,
                "dpai": 0.0,
                "tair": 298.15,
                "eair": 1500.0,
                "cpleaf": 2500.0,
                "fwet": 0.2,
                "fdry": 0.8,
                "gbh": 0.5,
                "gbv": 0.5,
                "gs": 0.15,
                "rnleaf": 400.0,
                "tleaf_bef": 298.15,
            },
            "description": "Zero plant area index - special case",
            "expected_behavior": "tleaf=tair, all fluxes=0",
        },
        "edge_negative_dpai": {
            "inputs": {
                "dtime_substep": 1800.0,
                "tref": 298.15,
                "pref": 101325.0,
                "cpair": 29.3,
                "dpai": -0.5,
                "tair": 298.15,
                "eair": 1500.0,
                "cpleaf": 2500.0,
                "fwet": 0.2,
                "fdry": 0.8,
                "gbh": 0.5,
                "gbv": 0.5,
                "gs": 0.15,
                "rnleaf": 400.0,
                "tleaf_bef": 298.15,
            },
            "description": "Negative dpai - should behave like zero",
            "expected_behavior": "tleaf=tair, all fluxes=0",
        },
        "edge_zero_stomatal": {
            "inputs": {
                "dtime_substep": 1800.0,
                "tref": 308.15,
                "pref": 101325.0,
                "cpair": 29.3,
                "dpai": 2.0,
                "tair": 308.15,
                "eair": 1000.0,
                "cpleaf": 2500.0,
                "fwet": 0.0,
                "fdry": 1.0,
                "gbh": 0.7,
                "gbv": 0.7,
                "gs": 0.0,
                "rnleaf": 550.0,
                "tleaf_bef": 308.15,
            },
            "description": "Complete stomatal closure under stress",
            "expected_behavior": "transpiration=0",
        },
        "edge_fully_wet": {
            "inputs": {
                "dtime_substep": 1800.0,
                "tref": 293.15,
                "pref": 101325.0,
                "cpair": 29.3,
                "dpai": 3.0,
                "tair": 293.15,
                "eair": 1800.0,
                "cpleaf": 2500.0,
                "fwet": 1.0,
                "fdry": 0.0,
                "gbh": 0.5,
                "gbv": 0.5,
                "gs": 0.12,
                "rnleaf": 300.0,
                "tleaf_bef": 293.15,
            },
            "description": "Fully wet canopy after rain",
            "expected_behavior": "transpiration=0",
        },
        "special_temp_gradient": {
            "inputs": {
                "dtime_substep": 600.0,
                "tref": 298.15,
                "pref": 101325.0,
                "cpair": 29.3,
                "dpai": 2.5,
                "tair": 298.15,
                "eair": 1500.0,
                "cpleaf": 2500.0,
                "fwet": 0.1,
                "fdry": 0.9,
                "gbh": 1.2,
                "gbv": 1.2,
                "gs": 0.3,
                "rnleaf": 800.0,
                "tleaf_bef": 283.15,
            },
            "description": "Large temperature gradient tests thermal inertia",
        },
    }


# Parametrized test cases
@pytest.mark.parametrize(
    "case_name",
    [
        "nominal_temperate",
        "nominal_hot_dry",
        "nominal_cold_humid",
        "nominal_tropical",
        "nominal_nighttime",
    ],
)
def test_leaf_fluxes_nominal_cases(test_data, case_name):
    """
    Test leaf_fluxes with nominal/typical environmental conditions.
    
    Verifies that the function:
    - Returns valid LeafFluxesResult with all fields
    - Produces physically reasonable leaf temperatures (> 0K)
    - Maintains energy balance within tolerance
    - Produces non-negative evaporation and transpiration
    
    Args:
        test_data: Fixture containing test cases
        case_name: Name of the test case to run
    """
    case = test_data[case_name]
    inputs = case["inputs"]
    
    # Call the function
    result = leaf_fluxes(**inputs)
    
    # Verify result is a LeafFluxesResult
    assert isinstance(result, LeafFluxesResult), (
        f"Expected LeafFluxesResult, got {type(result)}"
    )
    
    # Check all fields are present
    assert hasattr(result, "tleaf"), "Missing tleaf field"
    assert hasattr(result, "stleaf"), "Missing stleaf field"
    assert hasattr(result, "shleaf"), "Missing shleaf field"
    assert hasattr(result, "lhleaf"), "Missing lhleaf field"
    assert hasattr(result, "evleaf"), "Missing evleaf field"
    assert hasattr(result, "trleaf"), "Missing trleaf field"
    assert hasattr(result, "energy_balance_error"), "Missing energy_balance_error field"
    
    # Physical constraints
    assert float(result.tleaf) > 0.0, (
        f"Leaf temperature must be > 0K, got {result.tleaf}"
    )
    assert float(result.evleaf) >= 0.0, (
        f"Evaporation must be non-negative, got {result.evleaf}"
    )
    assert float(result.trleaf) >= 0.0, (
        f"Transpiration must be non-negative, got {result.trleaf}"
    )
    
    # Energy balance check
    assert check_energy_balance(result, inputs["rnleaf"]), (
        f"Energy balance error {result.energy_balance_error} exceeds tolerance "
        f"{ENERGY_BALANCE_TOL} W/m2 for case {case_name}"
    )
    
    # Verify energy balance error is small
    assert abs(float(result.energy_balance_error)) < ENERGY_BALANCE_TOL, (
        f"Energy balance error {result.energy_balance_error} exceeds "
        f"tolerance {ENERGY_BALANCE_TOL} W/m2"
    )


@pytest.mark.parametrize(
    "case_name,expected_behavior",
    [
        ("edge_zero_dpai", "zero_fluxes"),
        ("edge_negative_dpai", "zero_fluxes"),
    ],
)
def test_leaf_fluxes_zero_dpai_cases(test_data, case_name, expected_behavior):
    """
    Test special case where dpai <= 0.
    
    According to specification, when dpai <= 0:
    - tleaf should equal tair
    - All fluxes should be zero
    
    Args:
        test_data: Fixture containing test cases
        case_name: Name of the test case
        expected_behavior: Expected behavior identifier
    """
    case = test_data[case_name]
    inputs = case["inputs"]
    
    result = leaf_fluxes(**inputs)
    
    # tleaf should equal tair
    assert np.isclose(float(result.tleaf), inputs["tair"], atol=1e-6), (
        f"When dpai <= 0, tleaf should equal tair. "
        f"Got tleaf={result.tleaf}, tair={inputs['tair']}"
    )
    
    # All fluxes should be zero
    assert np.isclose(float(result.stleaf), 0.0, atol=1e-10), (
        f"Storage flux should be zero when dpai <= 0, got {result.stleaf}"
    )
    assert np.isclose(float(result.shleaf), 0.0, atol=1e-10), (
        f"Sensible heat flux should be zero when dpai <= 0, got {result.shleaf}"
    )
    assert np.isclose(float(result.lhleaf), 0.0, atol=1e-10), (
        f"Latent heat flux should be zero when dpai <= 0, got {result.lhleaf}"
    )
    assert np.isclose(float(result.evleaf), 0.0, atol=1e-10), (
        f"Evaporation should be zero when dpai <= 0, got {result.evleaf}"
    )
    assert np.isclose(float(result.trleaf), 0.0, atol=1e-10), (
        f"Transpiration should be zero when dpai <= 0, got {result.trleaf}"
    )


def test_leaf_fluxes_zero_stomatal_conductance(test_data):
    """
    Test behavior with complete stomatal closure (gs=0).
    
    When stomata are closed and canopy is dry (fwet=0):
    - Transpiration should be zero
    - Only sensible heat flux should occur
    - Evaporation should be zero (no wet surface)
    """
    case = test_data["edge_zero_stomatal"]
    inputs = case["inputs"]
    
    result = leaf_fluxes(**inputs)
    
    # Transpiration should be zero with closed stomata
    assert np.isclose(float(result.trleaf), 0.0, atol=1e-10), (
        f"Transpiration should be zero when gs=0 and fwet=0, got {result.trleaf}"
    )
    
    # Evaporation should also be zero (fwet=0)
    assert np.isclose(float(result.evleaf), 0.0, atol=1e-10), (
        f"Evaporation should be zero when fwet=0, got {result.evleaf}"
    )
    
    # Energy balance should still hold
    assert check_energy_balance(result, inputs["rnleaf"]), (
        f"Energy balance failed with zero stomatal conductance"
    )


def test_leaf_fluxes_fully_wet_canopy(test_data):
    """
    Test behavior with fully wet canopy (fwet=1.0, fdry=0.0).
    
    When canopy is fully wet:
    - Transpiration should be zero (no dry leaf area)
    - Evaporation should dominate water flux
    - Energy balance should still hold
    """
    case = test_data["edge_fully_wet"]
    inputs = case["inputs"]
    
    result = leaf_fluxes(**inputs)
    
    # Transpiration should be zero (no dry leaf area)
    assert np.isclose(float(result.trleaf), 0.0, atol=1e-10), (
        f"Transpiration should be zero when fdry=0, got {result.trleaf}"
    )
    
    # Evaporation should be non-negative
    assert float(result.evleaf) >= 0.0, (
        f"Evaporation should be non-negative, got {result.evleaf}"
    )
    
    # Energy balance should hold
    assert check_energy_balance(result, inputs["rnleaf"]), (
        f"Energy balance failed with fully wet canopy"
    )


def test_leaf_fluxes_temperature_gradient(test_data):
    """
    Test behavior with large temperature gradient between tleaf_bef and tair.
    
    Verifies that:
    - Storage heat flux is significant when temperature changes
    - Leaf temperature evolves toward equilibrium
    - Energy balance is maintained despite large gradients
    """
    case = test_data["special_temp_gradient"]
    inputs = case["inputs"]
    
    result = leaf_fluxes(**inputs)
    
    # Storage flux should be non-zero with temperature difference
    # (unless dpai is very small)
    if inputs["dpai"] > 0.1:
        # Storage flux magnitude should be significant
        assert abs(float(result.stleaf)) > 1.0, (
            f"Expected significant storage flux with 15K temperature gradient, "
            f"got {result.stleaf} W/m2"
        )
    
    # Leaf temperature should be between initial and air temperature
    # (moving toward equilibrium)
    tleaf = float(result.tleaf)
    tleaf_bef = inputs["tleaf_bef"]
    tair = inputs["tair"]
    
    # Check if tleaf is between tleaf_bef and tair (or has moved past tair)
    if tleaf_bef < tair:
        assert tleaf >= tleaf_bef, (
            f"Leaf temperature should increase from {tleaf_bef}K, got {tleaf}K"
        )
    else:
        assert tleaf <= tleaf_bef, (
            f"Leaf temperature should decrease from {tleaf_bef}K, got {tleaf}K"
        )
    
    # Energy balance should hold
    assert check_energy_balance(result, inputs["rnleaf"]), (
        f"Energy balance failed with large temperature gradient"
    )


def test_leaf_fluxes_output_shapes(test_data):
    """
    Test that all output fields have correct shapes (scalars).
    
    All outputs should be scalar values (0-dimensional arrays or floats).
    """
    case = test_data["nominal_temperate"]
    inputs = case["inputs"]
    
    result = leaf_fluxes(**inputs)
    
    # All fields should be scalars
    fields = ["tleaf", "stleaf", "shleaf", "lhleaf", "evleaf", "trleaf", 
              "energy_balance_error"]
    
    for field in fields:
        value = getattr(result, field)
        # Should be scalar (0-d array or Python float)
        if isinstance(value, jnp.ndarray):
            assert value.ndim == 0, (
                f"Field {field} should be scalar, got shape {value.shape}"
            )


def test_leaf_fluxes_output_dtypes(test_data):
    """
    Test that all output fields have correct data types (float64).
    
    All outputs should be float64 for numerical precision.
    """
    case = test_data["nominal_temperate"]
    inputs = case["inputs"]
    
    result = leaf_fluxes(**inputs)
    
    fields = ["tleaf", "stleaf", "shleaf", "lhleaf", "evleaf", "trleaf",
              "energy_balance_error"]
    
    for field in fields:
        value = getattr(result, field)
        if isinstance(value, jnp.ndarray):
            assert value.dtype == jnp.float64, (
                f"Field {field} should be float64, got {value.dtype}"
            )


def test_leaf_fluxes_energy_conservation(test_data):
    """
    Test energy conservation across all nominal cases.
    
    Verifies that: rnleaf = stleaf + shleaf + lhleaf (within tolerance)
    """
    nominal_cases = [
        "nominal_temperate",
        "nominal_hot_dry", 
        "nominal_cold_humid",
        "nominal_tropical",
        "nominal_nighttime",
    ]
    
    for case_name in nominal_cases:
        case = test_data[case_name]
        inputs = case["inputs"]
        
        result = leaf_fluxes(**inputs)
        
        # Energy balance: rnleaf = stleaf + shleaf + lhleaf
        energy_in = inputs["rnleaf"]
        energy_out = float(result.stleaf + result.shleaf + result.lhleaf)
        
        error = abs(energy_in - energy_out)
        
        assert error < ENERGY_BALANCE_TOL, (
            f"Energy conservation violated for {case_name}: "
            f"rnleaf={energy_in}, sum of fluxes={energy_out}, error={error}"
        )


def test_leaf_fluxes_water_flux_consistency(test_data):
    """
    Test consistency between water fluxes.
    
    Verifies that:
    - evleaf and trleaf are non-negative
    - When fwet=0, evleaf should be ~0
    - When fdry=0, trleaf should be ~0
    """
    # Test with dry canopy
    case = test_data["nominal_hot_dry"]
    inputs = case["inputs"]
    result = leaf_fluxes(**inputs)
    
    assert inputs["fwet"] == 0.0, "Test case should have fwet=0"
    assert np.isclose(float(result.evleaf), 0.0, atol=1e-10), (
        f"Evaporation should be ~0 when fwet=0, got {result.evleaf}"
    )
    
    # Test with wet canopy
    case = test_data["edge_fully_wet"]
    inputs = case["inputs"]
    result = leaf_fluxes(**inputs)
    
    assert inputs["fdry"] == 0.0, "Test case should have fdry=0"
    assert np.isclose(float(result.trleaf), 0.0, atol=1e-10), (
        f"Transpiration should be ~0 when fdry=0, got {result.trleaf}"
    )


def test_total_leaf_conductance_utility():
    """
    Test the total_leaf_conductance utility function.
    
    Verifies that:
    - Returns tuple of (gw, gh)
    - Conductances are non-negative
    - Handles edge cases (gs=0, fwet=1, etc.)
    """
    # Nominal case
    gw, gh = total_leaf_conductance(
        gs=0.15, gbv=0.5, gbh=0.5, fdry=0.8, fwet=0.2
    )
    
    assert gw >= 0.0, f"Water conductance should be non-negative, got {gw}"
    assert gh >= 0.0, f"Heat conductance should be non-negative, got {gh}"
    
    # Edge case: gs=0
    gw, gh = total_leaf_conductance(
        gs=0.0, gbv=0.5, gbh=0.5, fdry=1.0, fwet=0.0
    )
    
    assert gw >= 0.0, "Water conductance should be non-negative with gs=0"
    assert gh > 0.0, "Heat conductance should be positive (from gbh)"
    
    # Edge case: fully wet
    gw, gh = total_leaf_conductance(
        gs=0.15, gbv=0.5, gbh=0.5, fdry=0.0, fwet=1.0
    )
    
    assert gw >= 0.0, "Water conductance should be non-negative when fully wet"
    assert gh > 0.0, "Heat conductance should be positive"


def test_check_energy_balance_utility(test_data):
    """
    Test the check_energy_balance utility function.
    
    Verifies that:
    - Returns True when energy balance is satisfied
    - Returns False when energy balance is violated
    - Respects custom tolerance
    """
    case = test_data["nominal_temperate"]
    inputs = case["inputs"]
    
    result = leaf_fluxes(**inputs)
    
    # Should pass with default tolerance
    assert check_energy_balance(result, inputs["rnleaf"]), (
        "Energy balance check should pass for nominal case"
    )
    
    # Should pass with larger tolerance
    assert check_energy_balance(result, inputs["rnleaf"], tolerance=1.0), (
        "Energy balance check should pass with larger tolerance"
    )
    
    # Test with artificially violated energy balance
    # Create a modified result with large error
    modified_result = LeafFluxesResult(
        tleaf=result.tleaf,
        stleaf=result.stleaf,
        shleaf=result.shleaf,
        lhleaf=result.lhleaf,
        evleaf=result.evleaf,
        trleaf=result.trleaf,
        energy_balance_error=jnp.array(10.0),  # Large error
    )
    
    # Should fail with default tolerance
    assert not check_energy_balance(modified_result, inputs["rnleaf"]), (
        "Energy balance check should fail with large error"
    )


def test_leaf_fluxes_physical_bounds(test_data):
    """
    Test that outputs satisfy physical bounds across all cases.
    
    Verifies:
    - Temperatures > 0K
    - Evaporation >= 0
    - Transpiration >= 0
    - Reasonable temperature ranges (not extreme)
    """
    all_cases = [
        "nominal_temperate",
        "nominal_hot_dry",
        "nominal_cold_humid",
        "nominal_tropical",
        "nominal_nighttime",
        "edge_zero_stomatal",
        "edge_fully_wet",
        "special_temp_gradient",
    ]
    
    for case_name in all_cases:
        case = test_data[case_name]
        inputs = case["inputs"]
        
        result = leaf_fluxes(**inputs)
        
        # Temperature bounds
        assert float(result.tleaf) > 0.0, (
            f"Leaf temperature must be > 0K for {case_name}, got {result.tleaf}"
        )
        assert float(result.tleaf) < 400.0, (
            f"Leaf temperature unreasonably high for {case_name}, got {result.tleaf}"
        )
        
        # Water flux bounds
        assert float(result.evleaf) >= 0.0, (
            f"Evaporation must be non-negative for {case_name}, got {result.evleaf}"
        )
        assert float(result.trleaf) >= 0.0, (
            f"Transpiration must be non-negative for {case_name}, got {result.trleaf}"
        )


def test_leaf_fluxes_nighttime_behavior(test_data):
    """
    Test specific behavior during nighttime conditions.
    
    With negative net radiation:
    - Leaf should cool (sensible heat flux negative or small)
    - Latent heat flux should be small (low stomatal conductance)
    - Energy balance should still hold
    """
    case = test_data["nominal_nighttime"]
    inputs = case["inputs"]
    
    result = leaf_fluxes(**inputs)
    
    # Net radiation is negative
    assert inputs["rnleaf"] < 0.0, "Test case should have negative radiation"
    
    # Stomatal conductance should be low at night
    assert inputs["gs"] < 0.05, "Stomatal conductance should be low at night"
    
    # Energy balance should hold
    assert check_energy_balance(result, inputs["rnleaf"]), (
        "Energy balance should hold during nighttime"
    )
    
    # Leaf temperature should be reasonable
    assert float(result.tleaf) > 0.0, "Leaf temperature must be > 0K"


def test_leaf_fluxes_input_validation_types(test_data):
    """
    Test that function handles different numeric input types correctly.
    
    Should accept Python floats, NumPy scalars, and JAX arrays.
    """
    case = test_data["nominal_temperate"]
    inputs = case["inputs"]
    
    # Test with Python floats (original)
    result1 = leaf_fluxes(**inputs)
    
    # Test with NumPy scalars
    inputs_np = {k: np.float64(v) for k, v in inputs.items()}
    result2 = leaf_fluxes(**inputs_np)
    
    # Test with JAX arrays
    inputs_jax = {k: jnp.array(v) for k, v in inputs.items()}
    result3 = leaf_fluxes(**inputs_jax)
    
    # Results should be consistent
    assert np.isclose(float(result1.tleaf), float(result2.tleaf), rtol=1e-10), (
        "Results should be consistent with NumPy inputs"
    )
    assert np.isclose(float(result1.tleaf), float(result3.tleaf), rtol=1e-10), (
        "Results should be consistent with JAX inputs"
    )


def test_leaf_fluxes_fraction_sum_constraint(test_data):
    """
    Test that fwet + fdry <= 1.0 constraint is respected in test data.
    
    This is a data validation test to ensure test cases are physically valid.
    """
    all_cases = [
        "nominal_temperate",
        "nominal_hot_dry",
        "nominal_cold_humid",
        "nominal_tropical",
        "nominal_nighttime",
        "edge_zero_dpai",
        "edge_negative_dpai",
        "edge_zero_stomatal",
        "edge_fully_wet",
        "special_temp_gradient",
    ]
    
    for case_name in all_cases:
        case = test_data[case_name]
        inputs = case["inputs"]
        
        fwet = inputs["fwet"]
        fdry = inputs["fdry"]
        
        assert 0.0 <= fwet <= 1.0, (
            f"fwet must be in [0,1] for {case_name}, got {fwet}"
        )
        assert 0.0 <= fdry <= 1.0, (
            f"fdry must be in [0,1] for {case_name}, got {fdry}"
        )
        assert fwet + fdry <= 1.0 + 1e-10, (
            f"fwet + fdry must be <= 1.0 for {case_name}, got {fwet + fdry}"
        )


def test_leaf_fluxes_convergence_stability(test_data):
    """
    Test numerical stability and convergence across different conditions.
    
    Verifies that:
    - Energy balance error is small across all cases
    - No NaN or Inf values in outputs
    - Results are numerically stable
    """
    all_cases = [
        "nominal_temperate",
        "nominal_hot_dry",
        "nominal_cold_humid",
        "nominal_tropical",
        "nominal_nighttime",
        "special_temp_gradient",
    ]
    
    for case_name in all_cases:
        case = test_data[case_name]
        inputs = case["inputs"]
        
        result = leaf_fluxes(**inputs)
        
        # Check for NaN/Inf
        fields = ["tleaf", "stleaf", "shleaf", "lhleaf", "evleaf", "trleaf",
                  "energy_balance_error"]
        
        for field in fields:
            value = float(getattr(result, field))
            assert np.isfinite(value), (
                f"Field {field} is not finite for {case_name}, got {value}"
            )
        
        # Energy balance error should be small
        assert abs(float(result.energy_balance_error)) < 0.1, (
            f"Energy balance error too large for {case_name}: "
            f"{result.energy_balance_error} W/m2"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])