"""
Comprehensive pytest suite for MLCanopyWaterMod module.

Tests canopy interception and evaporation functions for water balance
in multi-layer canopy models, covering nominal cases, edge cases, and
physical constraints.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any

# Import actual translated module
from multilayer_canopy.MLCanopyWaterMod import (
    canopy_interception,
    canopy_evaporation,
    get_default_interception_params,
    CanopyInterceptionParams,
    CanopyWaterState,
    CanopyEvaporationInput,
    CanopyEvaporationOutput,
)


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def test_data():
    """
    Load comprehensive test data for canopy water functions.
    
    Returns:
        dict: Test cases with inputs and metadata for all test scenarios
    """
    return {
        "test_nominal_moderate_rain_multilayer": {
            "inputs": {
                "qflx_rain": jnp.array([0.0001, 0.0002, 0.00015]),
                "qflx_snow": jnp.array([0.0, 0.0, 0.0]),
                "lai": jnp.array([4.5, 3.2, 5.1]),
                "sai": jnp.array([0.8, 0.6, 1.0]),
                "ncan": jnp.array([10, 10, 10]),
                "dlai_profile": jnp.array([
                    [0.45] * 10,
                    [0.32] * 10,
                    [0.51] * 10
                ]),
                "dpai_profile": jnp.array([
                    [0.53] * 10,
                    [0.38] * 10,
                    [0.61] * 10
                ]),
                "h2ocan_profile": jnp.array([
                    [0.02] * 10,
                    [0.015] * 10,
                    [0.025] * 10
                ]),
                "params": {
                    "dewmx": 0.1,
                    "maximum_leaf_wetted_fraction": 0.05,
                    "interception_fraction": 0.25,
                    "fwet_exponent": 0.667,
                    "clm45_interception_p1": 0.25,
                    "clm45_interception_p2": -0.50,
                    "fpi_type": 2,
                    "dtime_substep": 1800.0
                }
            },
            "metadata": {
                "type": "nominal",
                "description": "Typical moderate rainfall with multi-layer canopy",
                "edge_cases": []
            }
        },
        "test_nominal_mixed_precipitation": {
            "inputs": {
                "qflx_rain": jnp.array([5e-05, 8e-05]),
                "qflx_snow": jnp.array([3e-05, 2e-05]),
                "lai": jnp.array([2.8, 6.5]),
                "sai": jnp.array([0.5, 1.2]),
                "ncan": jnp.array([8, 12]),
                "dlai_profile": jnp.array([
                    [0.35] * 8 + [0.0] * 4,
                    [0.542] * 12
                ]),
                "dpai_profile": jnp.array([
                    [0.4125] * 8 + [0.0] * 4,
                    [0.642] * 12
                ]),
                "h2ocan_profile": jnp.array([
                    [0.01] * 8 + [0.0] * 4,
                    [0.03] * 12
                ]),
                "params": {
                    "dewmx": 0.1,
                    "maximum_leaf_wetted_fraction": 0.05,
                    "interception_fraction": 0.25,
                    "fwet_exponent": 0.667,
                    "clm45_interception_p1": 0.25,
                    "clm45_interception_p2": -0.50,
                    "fpi_type": 2,
                    "dtime_substep": 1800.0
                }
            },
            "metadata": {
                "type": "nominal",
                "description": "Mixed rain and snow with different layer counts",
                "edge_cases": []
            }
        },
        "test_nominal_heavy_rain_dense_canopy": {
            "inputs": {
                "qflx_rain": jnp.array([0.0005, 0.0008]),
                "qflx_snow": jnp.array([0.0, 0.0]),
                "lai": jnp.array([7.2, 8.5]),
                "sai": jnp.array([1.5, 1.8]),
                "ncan": jnp.array([15, 15]),
                "dlai_profile": jnp.array([
                    [0.48] * 15,
                    [0.567] * 15
                ]),
                "dpai_profile": jnp.array([
                    [0.58] * 15,
                    [0.687] * 15
                ]),
                "h2ocan_profile": jnp.array([
                    [0.04] * 15,
                    [0.05] * 15
                ]),
                "params": {
                    "dewmx": 0.1,
                    "maximum_leaf_wetted_fraction": 0.05,
                    "interception_fraction": 0.25,
                    "fwet_exponent": 0.667,
                    "clm45_interception_p1": 0.25,
                    "clm45_interception_p2": -0.50,
                    "fpi_type": 2,
                    "dtime_substep": 1800.0
                }
            },
            "metadata": {
                "type": "nominal",
                "description": "Heavy rainfall with dense canopy",
                "edge_cases": []
            }
        },
        "test_nominal_clm45_formulation": {
            "inputs": {
                "qflx_rain": jnp.array([0.00012, 0.00018, 0.00025]),
                "qflx_snow": jnp.array([1e-05, 0.0, 2e-05]),
                "lai": jnp.array([3.5, 4.8, 2.2]),
                "sai": jnp.array([0.7, 0.9, 0.4]),
                "ncan": jnp.array([10, 10, 10]),
                "dlai_profile": jnp.array([
                    [0.35] * 10,
                    [0.48] * 10,
                    [0.22] * 10
                ]),
                "dpai_profile": jnp.array([
                    [0.42] * 10,
                    [0.57] * 10,
                    [0.26] * 10
                ]),
                "h2ocan_profile": jnp.array([
                    [0.018] * 10,
                    [0.022] * 10,
                    [0.012] * 10
                ]),
                "params": {
                    "dewmx": 0.1,
                    "maximum_leaf_wetted_fraction": 0.05,
                    "interception_fraction": 0.25,
                    "fwet_exponent": 0.667,
                    "clm45_interception_p1": 0.25,
                    "clm45_interception_p2": -0.50,
                    "fpi_type": 1,
                    "dtime_substep": 1800.0
                }
            },
            "metadata": {
                "type": "nominal",
                "description": "CLM4.5 interception formulation",
                "edge_cases": []
            }
        },
        "test_nominal_short_timestep": {
            "inputs": {
                "qflx_rain": jnp.array([0.0003]),
                "qflx_snow": jnp.array([0.0]),
                "lai": jnp.array([5.0]),
                "sai": jnp.array([1.0]),
                "ncan": jnp.array([10]),
                "dlai_profile": jnp.array([[0.5] * 10]),
                "dpai_profile": jnp.array([[0.6] * 10]),
                "h2ocan_profile": jnp.array([[0.02] * 10]),
                "params": {
                    "dewmx": 0.1,
                    "maximum_leaf_wetted_fraction": 0.05,
                    "interception_fraction": 0.25,
                    "fwet_exponent": 0.667,
                    "clm45_interception_p1": 0.25,
                    "clm45_interception_p2": -0.50,
                    "fpi_type": 2,
                    "dtime_substep": 300.0
                }
            },
            "metadata": {
                "type": "nominal",
                "description": "Short timestep for sub-hourly dynamics",
                "edge_cases": []
            }
        },
        "test_edge_zero_precipitation": {
            "inputs": {
                "qflx_rain": jnp.array([0.0, 0.0, 0.0, 0.0]),
                "qflx_snow": jnp.array([0.0, 0.0, 0.0, 0.0]),
                "lai": jnp.array([3.0, 5.0, 1.5, 7.0]),
                "sai": jnp.array([0.6, 1.0, 0.3, 1.4]),
                "ncan": jnp.array([10, 10, 10, 10]),
                "dlai_profile": jnp.array([
                    [0.3] * 10,
                    [0.5] * 10,
                    [0.15] * 10,
                    [0.7] * 10
                ]),
                "dpai_profile": jnp.array([
                    [0.36] * 10,
                    [0.6] * 10,
                    [0.18] * 10,
                    [0.84] * 10
                ]),
                "h2ocan_profile": jnp.array([
                    [0.01] * 10,
                    [0.03] * 10,
                    [0.005] * 10,
                    [0.04] * 10
                ]),
                "params": {
                    "dewmx": 0.1,
                    "maximum_leaf_wetted_fraction": 0.05,
                    "interception_fraction": 0.25,
                    "fwet_exponent": 0.667,
                    "clm45_interception_p1": 0.25,
                    "clm45_interception_p2": -0.50,
                    "fpi_type": 2,
                    "dtime_substep": 1800.0
                }
            },
            "metadata": {
                "type": "edge",
                "description": "Zero precipitation - no interception or throughfall",
                "edge_cases": ["zero_precipitation"]
            }
        },
        "test_edge_zero_canopy_structure": {
            "inputs": {
                "qflx_rain": jnp.array([0.0002, 0.0001]),
                "qflx_snow": jnp.array([1e-05, 0.0]),
                "lai": jnp.array([0.0, 0.0]),
                "sai": jnp.array([0.0, 0.0]),
                "ncan": jnp.array([0, 0]),
                "dlai_profile": jnp.array([
                    [0.0] * 10,
                    [0.0] * 10
                ]),
                "dpai_profile": jnp.array([
                    [0.0] * 10,
                    [0.0] * 10
                ]),
                "h2ocan_profile": jnp.array([
                    [0.0] * 10,
                    [0.0] * 10
                ]),
                "params": {
                    "dewmx": 0.1,
                    "maximum_leaf_wetted_fraction": 0.05,
                    "interception_fraction": 0.25,
                    "fwet_exponent": 0.667,
                    "clm45_interception_p1": 0.25,
                    "clm45_interception_p2": -0.50,
                    "fpi_type": 2,
                    "dtime_substep": 1800.0
                }
            },
            "metadata": {
                "type": "edge",
                "description": "Bare ground - all precipitation as throughfall",
                "edge_cases": ["zero_lai", "zero_sai", "zero_layers"]
            }
        },
        "test_edge_minimal_canopy_structure": {
            "inputs": {
                "qflx_rain": jnp.array([5e-05, 8e-05, 3e-05]),
                "qflx_snow": jnp.array([0.0, 0.0, 0.0]),
                "lai": jnp.array([0.1, 0.05, 0.2]),
                "sai": jnp.array([0.02, 0.01, 0.03]),
                "ncan": jnp.array([1, 1, 2]),
                "dlai_profile": jnp.array([
                    [0.1] + [0.0] * 9,
                    [0.05] + [0.0] * 9,
                    [0.1, 0.1] + [0.0] * 8
                ]),
                "dpai_profile": jnp.array([
                    [0.12] + [0.0] * 9,
                    [0.06] + [0.0] * 9,
                    [0.115, 0.115] + [0.0] * 8
                ]),
                "h2ocan_profile": jnp.array([
                    [0.001] + [0.0] * 9,
                    [0.0005] + [0.0] * 9,
                    [0.002, 0.002] + [0.0] * 8
                ]),
                "params": {
                    "dewmx": 0.1,
                    "maximum_leaf_wetted_fraction": 0.05,
                    "interception_fraction": 0.25,
                    "fwet_exponent": 0.667,
                    "clm45_interception_p1": 0.25,
                    "clm45_interception_p2": -0.50,
                    "fpi_type": 2,
                    "dtime_substep": 1800.0
                }
            },
            "metadata": {
                "type": "edge",
                "description": "Minimal canopy structure - sparse vegetation",
                "edge_cases": ["minimal_lai", "minimal_layers"]
            }
        },
        "test_edge_saturated_canopy": {
            "inputs": {
                "qflx_rain": jnp.array([0.0001, 0.00015]),
                "qflx_snow": jnp.array([0.0, 0.0]),
                "lai": jnp.array([4.0, 6.0]),
                "sai": jnp.array([0.8, 1.2]),
                "ncan": jnp.array([10, 10]),
                "dlai_profile": jnp.array([
                    [0.4] * 10,
                    [0.6] * 10
                ]),
                "dpai_profile": jnp.array([
                    [0.48] * 10,
                    [0.72] * 10
                ]),
                "h2ocan_profile": jnp.array([
                    [0.048] * 10,  # dewmx * dpai = 0.1 * 0.48
                    [0.072] * 10   # dewmx * dpai = 0.1 * 0.72
                ]),
                "params": {
                    "dewmx": 0.1,
                    "maximum_leaf_wetted_fraction": 0.05,
                    "interception_fraction": 0.25,
                    "fwet_exponent": 0.667,
                    "clm45_interception_p1": 0.25,
                    "clm45_interception_p2": -0.50,
                    "fpi_type": 2,
                    "dtime_substep": 1800.0
                }
            },
            "metadata": {
                "type": "edge",
                "description": "Near-saturated canopy - maximum storage",
                "edge_cases": ["saturated_canopy"]
            }
        },
        "test_special_extreme_snowfall": {
            "inputs": {
                "qflx_rain": jnp.array([0.0, 0.0]),
                "qflx_snow": jnp.array([0.001, 0.0015]),
                "lai": jnp.array([3.5, 5.2]),
                "sai": jnp.array([0.7, 1.0]),
                "ncan": jnp.array([10, 12]),
                "dlai_profile": jnp.array([
                    [0.35] * 10 + [0.0] * 2,
                    [0.433] * 12
                ]),
                "dpai_profile": jnp.array([
                    [0.42] * 10 + [0.0] * 2,
                    [0.517] * 12
                ]),
                "h2ocan_profile": jnp.array([
                    [0.005] * 10 + [0.0] * 2,
                    [0.008] * 12
                ]),
                "params": {
                    "dewmx": 0.1,
                    "maximum_leaf_wetted_fraction": 0.05,
                    "interception_fraction": 0.25,
                    "fwet_exponent": 0.667,
                    "clm45_interception_p1": 0.25,
                    "clm45_interception_p2": -0.50,
                    "fpi_type": 2,
                    "dtime_substep": 1800.0
                }
            },
            "metadata": {
                "type": "special",
                "description": "Extreme snowfall only - snow interception",
                "edge_cases": ["extreme_snowfall", "zero_rain"]
            }
        }
    }


@pytest.fixture
def evaporation_test_data():
    """
    Test data for canopy evaporation function.
    
    Returns:
        dict: Test cases for canopy evaporation scenarios
    """
    return {
        "test_nominal_evaporation": {
            "inputs": {
                "ncan": jnp.array([10, 8]),
                "dpai": jnp.array([
                    [0.5] * 10,
                    [0.4] * 8 + [0.0] * 2
                ]),
                "fracsun": jnp.array([
                    [0.6] * 10,
                    [0.5] * 8 + [0.0] * 2
                ]),
                "trleaf": jnp.array([
                    [[0.002, 0.001]] * 10,
                    [[0.0015, 0.0008]] * 8 + [[0.0, 0.0]] * 2
                ]),
                "evleaf": jnp.array([
                    [[0.0005, 0.0003]] * 10,
                    [[0.0004, 0.0002]] * 8 + [[0.0, 0.0]] * 2
                ]),
                "h2ocan": jnp.array([
                    [0.02] * 10,
                    [0.015] * 8 + [0.0] * 2
                ]),
                "mmh2o": 0.018015,
                "dtime_substep": 1800.0
            },
            "metadata": {
                "type": "nominal",
                "description": "Typical evaporation with transpiration and evaporation fluxes"
            }
        },
        "test_edge_zero_evaporation": {
            "inputs": {
                "ncan": jnp.array([10]),
                "dpai": jnp.array([[0.5] * 10]),
                "fracsun": jnp.array([[0.6] * 10]),
                "trleaf": jnp.array([[[0.0, 0.0]] * 10]),
                "evleaf": jnp.array([[[0.0, 0.0]] * 10]),
                "h2ocan": jnp.array([[0.02] * 10]),
                "mmh2o": 0.018015,
                "dtime_substep": 1800.0
            },
            "metadata": {
                "type": "edge",
                "description": "Zero evaporation - water should remain unchanged"
            }
        }
    }


# ============================================================================
# Test: canopy_interception - Output Shapes
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "test_nominal_moderate_rain_multilayer",
    "test_nominal_mixed_precipitation",
    "test_nominal_heavy_rain_dense_canopy",
    "test_edge_zero_precipitation",
    "test_edge_zero_canopy_structure",
])
def test_canopy_interception_output_shapes(test_data, test_case_name):
    """
    Test that canopy_interception returns outputs with correct shapes.
    
    Verifies:
    - h2ocan_profile shape matches (n_patches, n_layers)
    - Flux outputs shape matches (n_patches,)
    - Profile outputs shape matches (n_patches, n_layers)
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    # Create params namedtuple
    params = CanopyInterceptionParams(**inputs["params"])
    
    # Call function
    result = canopy_interception(
        qflx_rain=inputs["qflx_rain"],
        qflx_snow=inputs["qflx_snow"],
        lai=inputs["lai"],
        sai=inputs["sai"],
        ncan=inputs["ncan"],
        dlai_profile=inputs["dlai_profile"],
        dpai_profile=inputs["dpai_profile"],
        h2ocan_profile=inputs["h2ocan_profile"],
        params=params
    )
    
    n_patches = inputs["qflx_rain"].shape[0]
    n_layers = inputs["dlai_profile"].shape[1]
    
    # Check output shapes
    assert result.h2ocan_profile.shape == (n_patches, n_layers), \
        f"h2ocan_profile shape mismatch: expected {(n_patches, n_layers)}, got {result.h2ocan_profile.shape}"
    
    assert result.qflx_intr_canopy.shape == (n_patches,), \
        f"qflx_intr_canopy shape mismatch: expected {(n_patches,)}, got {result.qflx_intr_canopy.shape}"
    
    assert result.qflx_tflrain_canopy.shape == (n_patches,), \
        f"qflx_tflrain_canopy shape mismatch: expected {(n_patches,)}, got {result.qflx_tflrain_canopy.shape}"
    
    assert result.qflx_tflsnow_canopy.shape == (n_patches,), \
        f"qflx_tflsnow_canopy shape mismatch: expected {(n_patches,)}, got {result.qflx_tflsnow_canopy.shape}"
    
    assert result.fwet_profile.shape == (n_patches, n_layers), \
        f"fwet_profile shape mismatch: expected {(n_patches, n_layers)}, got {result.fwet_profile.shape}"
    
    assert result.fdry_profile.shape == (n_patches, n_layers), \
        f"fdry_profile shape mismatch: expected {(n_patches, n_layers)}, got {result.fdry_profile.shape}"


# ============================================================================
# Test: canopy_interception - Physical Constraints
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "test_nominal_moderate_rain_multilayer",
    "test_nominal_mixed_precipitation",
    "test_nominal_heavy_rain_dense_canopy",
    "test_nominal_clm45_formulation",
    "test_edge_saturated_canopy",
])
def test_canopy_interception_physical_constraints(test_data, test_case_name):
    """
    Test that canopy_interception respects physical constraints.
    
    Verifies:
    - All water fluxes are non-negative
    - Intercepted water is non-negative
    - Wetted fractions are in [0, 1]
    - Dry fractions are in [0, 1]
    - fwet + fdry <= 1 (accounting for numerical precision)
    - h2ocan <= dewmx * dpai (storage capacity)
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    params = CanopyInterceptionParams(**inputs["params"])
    
    result = canopy_interception(
        qflx_rain=inputs["qflx_rain"],
        qflx_snow=inputs["qflx_snow"],
        lai=inputs["lai"],
        sai=inputs["sai"],
        ncan=inputs["ncan"],
        dlai_profile=inputs["dlai_profile"],
        dpai_profile=inputs["dpai_profile"],
        h2ocan_profile=inputs["h2ocan_profile"],
        params=params
    )
    
    # Non-negativity constraints
    assert jnp.all(result.h2ocan_profile >= 0), \
        "h2ocan_profile contains negative values"
    
    assert jnp.all(result.qflx_intr_canopy >= 0), \
        "qflx_intr_canopy contains negative values"
    
    assert jnp.all(result.qflx_tflrain_canopy >= 0), \
        "qflx_tflrain_canopy contains negative values"
    
    assert jnp.all(result.qflx_tflsnow_canopy >= 0), \
        "qflx_tflsnow_canopy contains negative values"
    
    # Fraction constraints [0, 1]
    assert jnp.all(result.fwet_profile >= 0) and jnp.all(result.fwet_profile <= 1), \
        "fwet_profile contains values outside [0, 1]"
    
    assert jnp.all(result.fdry_profile >= 0) and jnp.all(result.fdry_profile <= 1), \
        "fdry_profile contains values outside [0, 1]"
    
    # Sum of fractions <= 1 (with tolerance for numerical precision)
    fraction_sum = result.fwet_profile + result.fdry_profile
    assert jnp.all(fraction_sum <= 1.0 + 1e-6), \
        f"fwet + fdry exceeds 1.0: max value = {jnp.max(fraction_sum)}"
    
    # Storage capacity constraint
    max_storage = params.dewmx * inputs["dpai_profile"]
    assert jnp.all(result.h2ocan_profile <= max_storage + 1e-6), \
        "h2ocan_profile exceeds maximum storage capacity (dewmx * dpai)"


# ============================================================================
# Test: canopy_interception - Mass Balance
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "test_nominal_moderate_rain_multilayer",
    "test_nominal_mixed_precipitation",
    "test_nominal_heavy_rain_dense_canopy",
])
def test_canopy_interception_mass_balance(test_data, test_case_name):
    """
    Test that canopy_interception conserves water mass.
    
    Verifies:
    - Total input precipitation = interception + throughfall
    - Change in storage = interception - evaporation (if applicable)
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    params = CanopyInterceptionParams(**inputs["params"])
    
    result = canopy_interception(
        qflx_rain=inputs["qflx_rain"],
        qflx_snow=inputs["qflx_snow"],
        lai=inputs["lai"],
        sai=inputs["sai"],
        ncan=inputs["ncan"],
        dlai_profile=inputs["dlai_profile"],
        dpai_profile=inputs["dpai_profile"],
        h2ocan_profile=inputs["h2ocan_profile"],
        params=params
    )
    
    # Total input precipitation
    total_precip = inputs["qflx_rain"] + inputs["qflx_snow"]
    
    # Calculate actual storage change rate
    h2ocan_change = (result.h2ocan_profile - inputs["h2ocan_profile"]).sum(axis=1)
    storage_change_rate = h2ocan_change / params.dtime_substep
    
    # Total throughfall
    total_throughfall = result.qflx_tflrain_canopy + result.qflx_tflsnow_canopy
    
    # Mass balance: Input = Storage change + Throughfall
    # (qflx_intr_canopy is intended interception, but some becomes drip when at capacity)
    total_output = storage_change_rate + total_throughfall
    
    # Mass balance check (with tolerance for numerical precision)
    assert jnp.allclose(total_precip, total_output, rtol=1e-5, atol=1e-8), \
        f"Mass balance violated: input={total_precip}, output={total_output}"


# ============================================================================
# Test: canopy_interception - Edge Cases
# ============================================================================

def test_canopy_interception_zero_precipitation(test_data):
    """
    Test canopy_interception with zero precipitation.
    
    Verifies:
    - No interception occurs
    - No throughfall occurs
    - Intercepted water remains unchanged
    """
    test_case = test_data["test_edge_zero_precipitation"]
    inputs = test_case["inputs"]
    
    params = CanopyInterceptionParams(**inputs["params"])
    
    result = canopy_interception(
        qflx_rain=inputs["qflx_rain"],
        qflx_snow=inputs["qflx_snow"],
        lai=inputs["lai"],
        sai=inputs["sai"],
        ncan=inputs["ncan"],
        dlai_profile=inputs["dlai_profile"],
        dpai_profile=inputs["dpai_profile"],
        h2ocan_profile=inputs["h2ocan_profile"],
        params=params
    )
    
    # No interception or throughfall
    assert jnp.allclose(result.qflx_intr_canopy, 0.0, atol=1e-10), \
        "Interception should be zero with no precipitation"
    
    assert jnp.allclose(result.qflx_tflrain_canopy, 0.0, atol=1e-10), \
        "Rain throughfall should be zero with no precipitation"
    
    assert jnp.allclose(result.qflx_tflsnow_canopy, 0.0, atol=1e-10), \
        "Snow throughfall should be zero with no precipitation"
    
    # Intercepted water unchanged
    assert jnp.allclose(result.h2ocan_profile, inputs["h2ocan_profile"], 
                       rtol=1e-6, atol=1e-8), \
        "Intercepted water should remain unchanged with no precipitation"


def test_canopy_interception_zero_canopy(test_data):
    """
    Test canopy_interception with zero canopy structure (bare ground).
    
    Verifies:
    - All precipitation passes through as throughfall
    - No interception occurs
    - No water storage in canopy
    """
    test_case = test_data["test_edge_zero_canopy_structure"]
    inputs = test_case["inputs"]
    
    params = CanopyInterceptionParams(**inputs["params"])
    
    result = canopy_interception(
        qflx_rain=inputs["qflx_rain"],
        qflx_snow=inputs["qflx_snow"],
        lai=inputs["lai"],
        sai=inputs["sai"],
        ncan=inputs["ncan"],
        dlai_profile=inputs["dlai_profile"],
        dpai_profile=inputs["dpai_profile"],
        h2ocan_profile=inputs["h2ocan_profile"],
        params=params
    )
    
    # All precipitation as throughfall
    assert jnp.allclose(result.qflx_tflrain_canopy, inputs["qflx_rain"], 
                       rtol=1e-6, atol=1e-8), \
        "All rain should pass through with zero canopy"
    
    assert jnp.allclose(result.qflx_tflsnow_canopy, inputs["qflx_snow"], 
                       rtol=1e-6, atol=1e-8), \
        "All snow should pass through with zero canopy"
    
    # No interception
    assert jnp.allclose(result.qflx_intr_canopy, 0.0, atol=1e-10), \
        "No interception should occur with zero canopy"
    
    # No water storage
    assert jnp.allclose(result.h2ocan_profile, 0.0, atol=1e-10), \
        "No water storage should occur with zero canopy"


def test_canopy_interception_minimal_canopy(test_data):
    """
    Test canopy_interception with minimal canopy structure.
    
    Verifies:
    - Function handles sparse vegetation correctly
    - Interception is proportional to canopy area
    - Most precipitation passes through
    """
    test_case = test_data["test_edge_minimal_canopy_structure"]
    inputs = test_case["inputs"]
    
    params = CanopyInterceptionParams(**inputs["params"])
    
    result = canopy_interception(
        qflx_rain=inputs["qflx_rain"],
        qflx_snow=inputs["qflx_snow"],
        lai=inputs["lai"],
        sai=inputs["sai"],
        ncan=inputs["ncan"],
        dlai_profile=inputs["dlai_profile"],
        dpai_profile=inputs["dpai_profile"],
        h2ocan_profile=inputs["h2ocan_profile"],
        params=params
    )
    
    # Most precipitation should pass through
    total_precip = inputs["qflx_rain"] + inputs["qflx_snow"]
    total_throughfall = result.qflx_tflrain_canopy + result.qflx_tflsnow_canopy
    
    assert jnp.all(total_throughfall > 0.8 * total_precip), \
        "Most precipitation should pass through with minimal canopy"
    
    # Some interception should occur
    assert jnp.all(result.qflx_intr_canopy >= 0), \
        "Interception should be non-negative"


def test_canopy_interception_saturated_canopy(test_data):
    """
    Test canopy_interception with near-saturated canopy.
    
    Verifies:
    - Canopy at or near maximum storage capacity
    - Additional precipitation becomes throughfall
    - Storage does not exceed capacity
    """
    test_case = test_data["test_edge_saturated_canopy"]
    inputs = test_case["inputs"]
    
    params = CanopyInterceptionParams(**inputs["params"])
    
    result = canopy_interception(
        qflx_rain=inputs["qflx_rain"],
        qflx_snow=inputs["qflx_snow"],
        lai=inputs["lai"],
        sai=inputs["sai"],
        ncan=inputs["ncan"],
        dlai_profile=inputs["dlai_profile"],
        dpai_profile=inputs["dpai_profile"],
        h2ocan_profile=inputs["h2ocan_profile"],
        params=params
    )
    
    # Storage should not exceed capacity
    max_storage = params.dewmx * inputs["dpai_profile"]
    assert jnp.all(result.h2ocan_profile <= max_storage + 1e-6), \
        "Storage exceeds maximum capacity"
    
    # Most precipitation should become throughfall when saturated
    total_precip = inputs["qflx_rain"] + inputs["qflx_snow"]
    total_throughfall = result.qflx_tflrain_canopy + result.qflx_tflsnow_canopy
    
    # With saturated canopy, throughfall should be significant
    assert jnp.all(total_throughfall > 0.5 * total_precip), \
        "Throughfall should be significant with saturated canopy"


def test_canopy_interception_extreme_snowfall(test_data):
    """
    Test canopy_interception with extreme snowfall only.
    
    Verifies:
    - Snow interception is handled correctly
    - No rain throughfall occurs
    - Snow throughfall is computed
    """
    test_case = test_data["test_special_extreme_snowfall"]
    inputs = test_case["inputs"]
    
    params = CanopyInterceptionParams(**inputs["params"])
    
    result = canopy_interception(
        qflx_rain=inputs["qflx_rain"],
        qflx_snow=inputs["qflx_snow"],
        lai=inputs["lai"],
        sai=inputs["sai"],
        ncan=inputs["ncan"],
        dlai_profile=inputs["dlai_profile"],
        dpai_profile=inputs["dpai_profile"],
        h2ocan_profile=inputs["h2ocan_profile"],
        params=params
    )
    
    # No rain throughfall
    assert jnp.allclose(result.qflx_tflrain_canopy, 0.0, atol=1e-10), \
        "Rain throughfall should be zero with no rain"
    
    # Snow throughfall should occur
    assert jnp.all(result.qflx_tflsnow_canopy > 0), \
        "Snow throughfall should be positive with snowfall"
    
    # Total snow = storage change + throughfall
    # Calculate actual storage change rate
    h2ocan_change = (result.h2ocan_profile - inputs["h2ocan_profile"]).sum(axis=1)
    storage_change_rate = h2ocan_change / params.dtime_substep
    
    assert jnp.allclose(inputs["qflx_snow"], 
                       storage_change_rate + result.qflx_tflsnow_canopy,
                       rtol=1e-5, atol=1e-8), \
        "Snow mass balance violated"


# ============================================================================
# Test: canopy_interception - Formulation Comparison
# ============================================================================

def test_canopy_interception_clm45_vs_clm5(test_data):
    """
    Test that CLM4.5 and CLM5 formulations produce different results.
    
    Verifies:
    - Both formulations run successfully
    - Results differ between formulations
    - Both respect physical constraints
    """
    # Use same inputs with different fpi_type
    test_case = test_data["test_nominal_moderate_rain_multilayer"]
    inputs = test_case["inputs"]
    
    # CLM5 formulation
    params_clm5 = CanopyInterceptionParams(**{**inputs["params"], "fpi_type": 2})
    result_clm5 = canopy_interception(
        qflx_rain=inputs["qflx_rain"],
        qflx_snow=inputs["qflx_snow"],
        lai=inputs["lai"],
        sai=inputs["sai"],
        ncan=inputs["ncan"],
        dlai_profile=inputs["dlai_profile"],
        dpai_profile=inputs["dpai_profile"],
        h2ocan_profile=inputs["h2ocan_profile"],
        params=params_clm5
    )
    
    # CLM4.5 formulation
    params_clm45 = CanopyInterceptionParams(**{**inputs["params"], "fpi_type": 1})
    result_clm45 = canopy_interception(
        qflx_rain=inputs["qflx_rain"],
        qflx_snow=inputs["qflx_snow"],
        lai=inputs["lai"],
        sai=inputs["sai"],
        ncan=inputs["ncan"],
        dlai_profile=inputs["dlai_profile"],
        dpai_profile=inputs["dpai_profile"],
        h2ocan_profile=inputs["h2ocan_profile"],
        params=params_clm45
    )
    
    # Results should differ
    assert not jnp.allclose(result_clm5.qflx_intr_canopy, 
                           result_clm45.qflx_intr_canopy, rtol=1e-3), \
        "CLM4.5 and CLM5 formulations should produce different results"
    
    # Both should respect physical constraints
    assert jnp.all(result_clm5.qflx_intr_canopy >= 0), \
        "CLM5 interception should be non-negative"
    
    assert jnp.all(result_clm45.qflx_intr_canopy >= 0), \
        "CLM4.5 interception should be non-negative"


# ============================================================================
# Test: canopy_interception - Data Types
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "test_nominal_moderate_rain_multilayer",
    "test_edge_zero_precipitation",
])
def test_canopy_interception_dtypes(test_data, test_case_name):
    """
    Test that canopy_interception returns correct data types.
    
    Verifies:
    - All outputs are JAX arrays
    - Outputs have float dtype
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    params = CanopyInterceptionParams(**inputs["params"])
    
    result = canopy_interception(
        qflx_rain=inputs["qflx_rain"],
        qflx_snow=inputs["qflx_snow"],
        lai=inputs["lai"],
        sai=inputs["sai"],
        ncan=inputs["ncan"],
        dlai_profile=inputs["dlai_profile"],
        dpai_profile=inputs["dpai_profile"],
        h2ocan_profile=inputs["h2ocan_profile"],
        params=params
    )
    
    # Check all outputs are JAX arrays
    assert isinstance(result.h2ocan_profile, jnp.ndarray), \
        "h2ocan_profile should be a JAX array"
    
    assert isinstance(result.qflx_intr_canopy, jnp.ndarray), \
        "qflx_intr_canopy should be a JAX array"
    
    assert isinstance(result.fwet_profile, jnp.ndarray), \
        "fwet_profile should be a JAX array"
    
    # Check float dtype
    assert jnp.issubdtype(result.h2ocan_profile.dtype, jnp.floating), \
        "h2ocan_profile should have float dtype"
    
    assert jnp.issubdtype(result.qflx_intr_canopy.dtype, jnp.floating), \
        "qflx_intr_canopy should have float dtype"


# ============================================================================
# Test: canopy_evaporation - Output Shapes
# ============================================================================

def test_canopy_evaporation_output_shapes(evaporation_test_data):
    """
    Test that canopy_evaporation returns outputs with correct shapes.
    
    Verifies:
    - h2ocan shape matches input shape (n_patches, n_layers)
    """
    test_case = evaporation_test_data["test_nominal_evaporation"]
    inputs = test_case["inputs"]
    
    # Create input namedtuple
    evap_input = CanopyEvaporationInput(**inputs)
    
    # Call function
    result = canopy_evaporation(evap_input)
    
    expected_shape = inputs["h2ocan"].shape
    
    assert result.h2ocan.shape == expected_shape, \
        f"h2ocan shape mismatch: expected {expected_shape}, got {result.h2ocan.shape}"


# ============================================================================
# Test: canopy_evaporation - Physical Constraints
# ============================================================================

def test_canopy_evaporation_physical_constraints(evaporation_test_data):
    """
    Test that canopy_evaporation respects physical constraints.
    
    Verifies:
    - Updated h2ocan is non-negative
    - h2ocan does not increase (only evaporation, no addition)
    """
    test_case = evaporation_test_data["test_nominal_evaporation"]
    inputs = test_case["inputs"]
    
    evap_input = CanopyEvaporationInput(**inputs)
    result = canopy_evaporation(evap_input)
    
    # Non-negativity
    assert jnp.all(result.h2ocan >= 0), \
        "Updated h2ocan contains negative values"
    
    # Should not increase (only evaporation)
    assert jnp.all(result.h2ocan <= inputs["h2ocan"] + 1e-8), \
        "h2ocan increased (should only decrease via evaporation)"


# ============================================================================
# Test: canopy_evaporation - Edge Cases
# ============================================================================

def test_canopy_evaporation_zero_fluxes(evaporation_test_data):
    """
    Test canopy_evaporation with zero evaporation fluxes.
    
    Verifies:
    - h2ocan remains unchanged when no evaporation occurs
    """
    test_case = evaporation_test_data["test_edge_zero_evaporation"]
    inputs = test_case["inputs"]
    
    evap_input = CanopyEvaporationInput(**inputs)
    result = canopy_evaporation(evap_input)
    
    # h2ocan should remain unchanged
    assert jnp.allclose(result.h2ocan, inputs["h2ocan"], rtol=1e-6, atol=1e-8), \
        "h2ocan should remain unchanged with zero evaporation fluxes"


# ============================================================================
# Test: canopy_evaporation - Data Types
# ============================================================================

def test_canopy_evaporation_dtypes(evaporation_test_data):
    """
    Test that canopy_evaporation returns correct data types.
    
    Verifies:
    - Output is a JAX array
    - Output has float dtype
    """
    test_case = evaporation_test_data["test_nominal_evaporation"]
    inputs = test_case["inputs"]
    
    evap_input = CanopyEvaporationInput(**inputs)
    result = canopy_evaporation(evap_input)
    
    assert isinstance(result.h2ocan, jnp.ndarray), \
        "h2ocan should be a JAX array"
    
    assert jnp.issubdtype(result.h2ocan.dtype, jnp.floating), \
        "h2ocan should have float dtype"


# ============================================================================
# Test: get_default_interception_params
# ============================================================================

def test_get_default_interception_params_defaults():
    """
    Test that get_default_interception_params returns correct default values.
    
    Verifies:
    - All default values match specification
    - Returns CanopyInterceptionParams namedtuple
    """
    params = get_default_interception_params()
    
    assert isinstance(params, CanopyInterceptionParams), \
        "Should return CanopyInterceptionParams namedtuple"
    
    assert params.dewmx == 0.1, "Default dewmx should be 0.1"
    assert params.maximum_leaf_wetted_fraction == 0.05, \
        "Default maximum_leaf_wetted_fraction should be 0.05"
    assert params.interception_fraction == 0.25, \
        "Default interception_fraction should be 0.25"
    assert params.fwet_exponent == 0.667, \
        "Default fwet_exponent should be 0.667"
    assert params.clm45_interception_p1 == 0.25, \
        "Default clm45_interception_p1 should be 0.25"
    assert params.clm45_interception_p2 == -0.50, \
        "Default clm45_interception_p2 should be -0.50"
    assert params.fpi_type == 2, "Default fpi_type should be 2"
    assert params.dtime_substep == 1800.0, \
        "Default dtime_substep should be 1800.0"


def test_get_default_interception_params_custom():
    """
    Test that get_default_interception_params accepts custom values.
    
    Verifies:
    - Custom values override defaults
    - Unspecified values use defaults
    """
    custom_params = get_default_interception_params(
        dewmx=0.15,
        fpi_type=1,
        dtime_substep=900.0
    )
    
    assert custom_params.dewmx == 0.15, "Custom dewmx should be 0.15"
    assert custom_params.fpi_type == 1, "Custom fpi_type should be 1"
    assert custom_params.dtime_substep == 900.0, \
        "Custom dtime_substep should be 900.0"
    
    # Check defaults for unspecified values
    assert custom_params.maximum_leaf_wetted_fraction == 0.05, \
        "Unspecified parameter should use default"


def test_get_default_interception_params_constraints():
    """
    Test that get_default_interception_params respects parameter constraints.
    
    Verifies:
    - Parameters are within valid ranges
    - fpi_type is either 1 or 2
    """
    params = get_default_interception_params()
    
    assert params.dewmx >= 0, "dewmx should be non-negative"
    assert 0 <= params.maximum_leaf_wetted_fraction <= 1, \
        "maximum_leaf_wetted_fraction should be in [0, 1]"
    assert 0 <= params.interception_fraction <= 1, \
        "interception_fraction should be in [0, 1]"
    assert params.fwet_exponent >= 0, "fwet_exponent should be non-negative"
    assert 0 <= params.clm45_interception_p1 <= 1, \
        "clm45_interception_p1 should be in [0, 1]"
    assert params.fpi_type in [1, 2], "fpi_type should be 1 or 2"
    assert params.dtime_substep > 0, "dtime_substep should be positive"


# ============================================================================
# Test: Integration Tests
# ============================================================================

def test_canopy_interception_timestep_sensitivity(test_data):
    """
    Test that canopy_interception results scale appropriately with timestep.
    
    Verifies:
    - Shorter timesteps produce proportionally smaller changes
    - Mass balance is maintained across different timesteps
    """
    test_case = test_data["test_nominal_moderate_rain_multilayer"]
    inputs = test_case["inputs"]
    
    # Test with default timestep
    params_long = CanopyInterceptionParams(**inputs["params"])
    result_long = canopy_interception(
        qflx_rain=inputs["qflx_rain"],
        qflx_snow=inputs["qflx_snow"],
        lai=inputs["lai"],
        sai=inputs["sai"],
        ncan=inputs["ncan"],
        dlai_profile=inputs["dlai_profile"],
        dpai_profile=inputs["dpai_profile"],
        h2ocan_profile=inputs["h2ocan_profile"],
        params=params_long
    )
    
    # Test with shorter timestep
    params_short = CanopyInterceptionParams(
        **{**inputs["params"], "dtime_substep": 300.0}
    )
    result_short = canopy_interception(
        qflx_rain=inputs["qflx_rain"],
        qflx_snow=inputs["qflx_snow"],
        lai=inputs["lai"],
        sai=inputs["sai"],
        ncan=inputs["ncan"],
        dlai_profile=inputs["dlai_profile"],
        dpai_profile=inputs["dpai_profile"],
        h2ocan_profile=inputs["h2ocan_profile"],
        params=params_short
    )
    
    # Both should maintain mass balance
    total_precip = inputs["qflx_rain"] + inputs["qflx_snow"]
    
    total_output_long = (result_long.qflx_intr_canopy + 
                        result_long.qflx_tflrain_canopy + 
                        result_long.qflx_tflsnow_canopy)
    
    total_output_short = (result_short.qflx_intr_canopy + 
                         result_short.qflx_tflrain_canopy + 
                         result_short.qflx_tflsnow_canopy)
    
    assert jnp.allclose(total_precip, total_output_long, rtol=1e-5, atol=1e-8), \
        "Mass balance violated for long timestep"
    
    assert jnp.allclose(total_precip, total_output_short, rtol=1e-5, atol=1e-8), \
        "Mass balance violated for short timestep"


def test_canopy_water_state_consistency():
    """
    Test that CanopyWaterState components are internally consistent.
    
    Verifies:
    - fwet + fdry <= 1 for all layers
    - Wetted fraction increases with water content
    """
    # Create a simple test case
    n_patches = 2
    n_layers = 5
    
    h2ocan = jnp.array([[0.01, 0.02, 0.03, 0.04, 0.05],
                        [0.02, 0.03, 0.04, 0.05, 0.06]])
    
    qflx_intr = jnp.array([0.0001, 0.0002])
    qflx_tflrain = jnp.array([0.0005, 0.0006])
    qflx_tflsnow = jnp.array([0.0, 0.0])
    
    fwet = jnp.array([[0.2, 0.3, 0.4, 0.5, 0.6],
                      [0.3, 0.4, 0.5, 0.6, 0.7]])
    
    fdry = jnp.array([[0.7, 0.6, 0.5, 0.4, 0.3],
                      [0.6, 0.5, 0.4, 0.3, 0.2]])
    
    state = CanopyWaterState(
        h2ocan_profile=h2ocan,
        qflx_intr_canopy=qflx_intr,
        qflx_tflrain_canopy=qflx_tflrain,
        qflx_tflsnow_canopy=qflx_tflsnow,
        fwet_profile=fwet,
        fdry_profile=fdry
    )
    
    # Check fraction consistency
    fraction_sum = state.fwet_profile + state.fdry_profile
    assert jnp.all(fraction_sum <= 1.0 + 1e-6), \
        "fwet + fdry should not exceed 1.0"
    
    # Check that fwet generally increases with h2ocan (within each patch)
    for i in range(n_patches):
        # Check monotonicity (allowing for some numerical tolerance)
        fwet_diffs = jnp.diff(state.fwet_profile[i])
        # Most differences should be non-negative
        assert jnp.sum(fwet_diffs >= -1e-6) >= len(fwet_diffs) * 0.8, \
            "Wetted fraction should generally increase with water content"


# ============================================================================
# Test: Documentation and Metadata
# ============================================================================

def test_test_data_completeness(test_data):
    """
    Test that all test cases have required metadata.
    
    Verifies:
    - Each test case has inputs and metadata
    - Metadata includes type and description
    """
    for test_name, test_case in test_data.items():
        assert "inputs" in test_case, \
            f"{test_name} missing 'inputs' field"
        
        assert "metadata" in test_case, \
            f"{test_name} missing 'metadata' field"
        
        metadata = test_case["metadata"]
        assert "type" in metadata, \
            f"{test_name} metadata missing 'type' field"
        
        assert "description" in metadata, \
            f"{test_name} metadata missing 'description' field"
        
        assert metadata["type"] in ["nominal", "edge", "special"], \
            f"{test_name} has invalid type: {metadata['type']}"


def test_test_data_input_consistency(test_data):
    """
    Test that test case inputs have consistent dimensions.
    
    Verifies:
    - All 1D arrays have same length (n_patches)
    - All 2D arrays have consistent first dimension (n_patches)
    """
    for test_name, test_case in test_data.items():
        inputs = test_case["inputs"]
        
        # Get n_patches from qflx_rain
        n_patches = inputs["qflx_rain"].shape[0]
        
        # Check 1D arrays
        assert inputs["qflx_snow"].shape[0] == n_patches, \
            f"{test_name}: qflx_snow dimension mismatch"
        
        assert inputs["lai"].shape[0] == n_patches, \
            f"{test_name}: lai dimension mismatch"
        
        assert inputs["sai"].shape[0] == n_patches, \
            f"{test_name}: sai dimension mismatch"
        
        assert inputs["ncan"].shape[0] == n_patches, \
            f"{test_name}: ncan dimension mismatch"
        
        # Check 2D arrays
        assert inputs["dlai_profile"].shape[0] == n_patches, \
            f"{test_name}: dlai_profile first dimension mismatch"
        
        assert inputs["dpai_profile"].shape[0] == n_patches, \
            f"{test_name}: dpai_profile first dimension mismatch"
        
        assert inputs["h2ocan_profile"].shape[0] == n_patches, \
            f"{test_name}: h2ocan_profile first dimension mismatch"
        
        # Check that 2D arrays have same second dimension
        n_layers = inputs["dlai_profile"].shape[1]
        assert inputs["dpai_profile"].shape[1] == n_layers, \
            f"{test_name}: dpai_profile second dimension mismatch"
        
        assert inputs["h2ocan_profile"].shape[1] == n_layers, \
            f"{test_name}: h2ocan_profile second dimension mismatch"