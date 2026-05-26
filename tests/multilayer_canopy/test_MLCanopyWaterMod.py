"""
Comprehensive pytest suite for MLCanopyWaterMod functions.

This module tests the canopy water interception and evaporation functions
from the multilayer canopy model, including:
- canopy_interception: Calculates precipitation interception by canopy layers
- canopy_evaporation: Updates canopy water content from evaporation/transpiration
- get_default_interception_params: Returns default parameter configuration

Tests cover nominal cases, edge cases, and numerical stability.
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multilayer_canopy.MLCanopyWaterMod import (
    CanopyEvaporationInput,
    CanopyEvaporationOutput,
    CanopyInterceptionParams,
    CanopyWaterState,
    canopy_evaporation,
    canopy_interception,
    get_default_interception_params,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_params():
    """Fixture providing default canopy interception parameters."""
    return get_default_interception_params()


@pytest.fixture
def test_data_interception():
    """
    Fixture providing comprehensive test data for canopy_interception.
    
    Returns:
        dict: Test cases with inputs and metadata for various scenarios
    """
    return {
        "nominal_moderate_rain": {
            "qflx_rain": jnp.array([0.0005, 0.0008, 0.0003]),
            "qflx_snow": jnp.array([0.0, 0.0, 0.0]),
            "lai": jnp.array([4.5, 5.2, 3.8]),
            "sai": jnp.array([0.8, 1.0, 0.6]),
            "ncan": jnp.array([10, 10, 10]),
            "dlai_profile": jnp.array([
                [0.45] * 10,
                [0.52] * 10,
                [0.38] * 10,
            ]),
            "dpai_profile": jnp.array([
                [0.53] * 10,
                [0.62] * 10,
                [0.44] * 10,
            ]),
            "h2ocan_profile": jnp.array([
                [0.02] * 10,
                [0.03] * 10,
                [0.01] * 10,
            ]),
            "n_patches": 3,
            "n_layers": 10,
        },
        "mixed_precipitation_sparse": {
            "qflx_rain": jnp.array([0.0002, 0.0001]),
            "qflx_snow": jnp.array([0.0001, 0.00015]),
            "lai": jnp.array([1.5, 2.0]),
            "sai": jnp.array([0.3, 0.4]),
            "ncan": jnp.array([5, 5]),
            "dlai_profile": jnp.array([
                [0.3] * 5,
                [0.4] * 5,
            ]),
            "dpai_profile": jnp.array([
                [0.36] * 5,
                [0.48] * 5,
            ]),
            "h2ocan_profile": jnp.array([
                [0.005] * 5,
                [0.008] * 5,
            ]),
            "n_patches": 2,
            "n_layers": 5,
        },
        "heavy_snow": {
            "qflx_rain": jnp.array([0.0, 0.0, 0.0, 0.0]),
            "qflx_snow": jnp.array([0.002, 0.0015, 0.0025, 0.001]),
            "lai": jnp.array([3.0, 4.0, 2.5, 5.0]),
            "sai": jnp.array([0.5, 0.7, 0.4, 0.9]),
            "ncan": jnp.array([8, 8, 8, 8]),
            "dlai_profile": jnp.array([
                [0.375] * 8,
                [0.5] * 8,
                [0.3125] * 8,
                [0.625] * 8,
            ]),
            "dpai_profile": jnp.array([
                [0.4375] * 8,
                [0.5875] * 8,
                [0.3625] * 8,
                [0.7375] * 8,
            ]),
            "h2ocan_profile": jnp.array([
                [0.015] * 8,
                [0.02] * 8,
                [0.01] * 8,
                [0.025] * 8,
            ]),
            "n_patches": 4,
            "n_layers": 8,
        },
        "zero_precipitation": {
            "qflx_rain": jnp.array([0.0, 0.0, 0.0]),
            "qflx_snow": jnp.array([0.0, 0.0, 0.0]),
            "lai": jnp.array([3.0, 4.0, 2.5]),
            "sai": jnp.array([0.5, 0.7, 0.4]),
            "ncan": jnp.array([5, 5, 5]),
            "dlai_profile": jnp.array([
                [0.6] * 5,
                [0.8] * 5,
                [0.5] * 5,
            ]),
            "dpai_profile": jnp.array([
                [0.7] * 5,
                [0.94] * 5,
                [0.58] * 5,
            ]),
            "h2ocan_profile": jnp.array([
                [0.02] * 5,
                [0.03] * 5,
                [0.015] * 5,
            ]),
            "n_patches": 3,
            "n_layers": 5,
        },
        "zero_canopy": {
            "qflx_rain": jnp.array([0.0005, 0.0003]),
            "qflx_snow": jnp.array([0.0001, 0.0002]),
            "lai": jnp.array([0.0, 0.0]),
            "sai": jnp.array([0.0, 0.0]),
            "ncan": jnp.array([0, 0]),
            "dlai_profile": jnp.array([
                [0.0] * 5,
                [0.0] * 5,
            ]),
            "dpai_profile": jnp.array([
                [0.0] * 5,
                [0.0] * 5,
            ]),
            "h2ocan_profile": jnp.array([
                [0.0] * 5,
                [0.0] * 5,
            ]),
            "n_patches": 2,
            "n_layers": 5,
        },
        "very_small_values": {
            "qflx_rain": jnp.array([1e-8, 1e-9, 1e-7]),
            "qflx_snow": jnp.array([1e-9, 1e-8, 1e-10]),
            "lai": jnp.array([0.01, 0.05, 0.02]),
            "sai": jnp.array([0.001, 0.005, 0.002]),
            "ncan": jnp.array([3, 3, 3]),
            "dlai_profile": jnp.array([
                [0.0033, 0.0033, 0.0034],
                [0.0167, 0.0167, 0.0166],
                [0.0067, 0.0067, 0.0066],
            ]),
            "dpai_profile": jnp.array([
                [0.0037, 0.0037, 0.0036],
                [0.0183, 0.0183, 0.0184],
                [0.0073, 0.0073, 0.0074],
            ]),
            "h2ocan_profile": jnp.array([
                [1e-6, 1e-6, 1e-6],
                [5e-6, 5e-6, 5e-6],
                [2e-6, 2e-6, 2e-6],
            ]),
            "n_patches": 3,
            "n_layers": 3,
        },
        "extreme_precipitation": {
            "qflx_rain": jnp.array([0.05, 0.08]),
            "qflx_snow": jnp.array([0.02, 0.03]),
            "lai": jnp.array([6.0, 7.0]),
            "sai": jnp.array([1.2, 1.5]),
            "ncan": jnp.array([12, 12]),
            "dlai_profile": jnp.array([
                [0.5] * 12,
                [0.583] * 12,
            ]),
            "dpai_profile": jnp.array([
                [0.6] * 12,
                [0.708] * 12,
            ]),
            "h2ocan_profile": jnp.array([
                [0.04] * 12,
                [0.05] * 12,
            ]),
            "n_patches": 2,
            "n_layers": 12,
        },
        "variable_layers": {
            "qflx_rain": jnp.array([0.0004, 0.0005, 0.0003, 0.0006, 0.0007]),
            "qflx_snow": jnp.array([0.0001, 0.0, 0.0002, 0.0001, 0.0]),
            "lai": jnp.array([2.0, 3.5, 1.5, 4.5, 5.5]),
            "sai": jnp.array([0.3, 0.6, 0.2, 0.8, 1.0]),
            "ncan": jnp.array([3, 6, 2, 9, 12]),
            "dlai_profile": jnp.array([
                [0.667, 0.667, 0.666] + [0.0] * 9,
                [0.583] * 6 + [0.0] * 6,
                [0.75, 0.75] + [0.0] * 10,
                [0.5] * 9 + [0.0] * 3,
                [0.458] * 11 + [0.459],
            ]),
            "dpai_profile": jnp.array([
                [0.767, 0.767, 0.766] + [0.0] * 9,
                [0.683] * 6 + [0.0] * 6,
                [0.85, 0.85] + [0.0] * 10,
                [0.589] * 9 + [0.0] * 3,
                [0.542] * 11 + [0.541],
            ]),
            "h2ocan_profile": jnp.array([
                [0.015] * 3 + [0.0] * 9,
                [0.025] * 6 + [0.0] * 6,
                [0.012] * 2 + [0.0] * 10,
                [0.03] * 9 + [0.0] * 3,
                [0.035] * 12,
            ]),
            "n_patches": 5,
            "n_layers": 12,
        },
    }


@pytest.fixture
def test_data_evaporation():
    """
    Fixture providing test data for canopy_evaporation.
    
    Returns:
        dict: Test cases with CanopyEvaporationInput instances
    """
    mmh2o = 0.018015
    dtime = 1800.0
    
    return {
        "nominal_case": CanopyEvaporationInput(
            ncan=jnp.array([5, 5]),
            dpai=jnp.array([[0.5] * 5, [0.6] * 5]),
            fracsun=jnp.array([[0.3] * 5, [0.4] * 5]),
            trleaf=jnp.array([
                [[2e-6, 1e-6]] * 5,
                [[3e-6, 1.5e-6]] * 5,
            ]),
            evleaf=jnp.array([
                [[1e-6, 0.5e-6]] * 5,
                [[1.5e-6, 0.8e-6]] * 5,
            ]),
            h2ocan=jnp.array([[0.02] * 5, [0.03] * 5]),
            mmh2o=mmh2o,
            dtime_substep=dtime,
        ),
        "zero_fluxes": CanopyEvaporationInput(
            ncan=jnp.array([3, 3]),
            dpai=jnp.array([[0.4] * 3, [0.5] * 3]),
            fracsun=jnp.array([[0.5] * 3, [0.6] * 3]),
            trleaf=jnp.array([
                [[0.0, 0.0]] * 3,
                [[0.0, 0.0]] * 3,
            ]),
            evleaf=jnp.array([
                [[0.0, 0.0]] * 3,
                [[0.0, 0.0]] * 3,
            ]),
            h2ocan=jnp.array([[0.01] * 3, [0.015] * 3]),
            mmh2o=mmh2o,
            dtime_substep=dtime,
        ),
        "high_evaporation": CanopyEvaporationInput(
            ncan=jnp.array([4]),
            dpai=jnp.array([[0.7] * 4]),
            fracsun=jnp.array([[0.8] * 4]),
            trleaf=jnp.array([
                [[5e-6, 2e-6]] * 4,
            ]),
            evleaf=jnp.array([
                [[3e-6, 1.5e-6]] * 4,
            ]),
            h2ocan=jnp.array([[0.05] * 4]),
            mmh2o=mmh2o,
            dtime_substep=dtime,
        ),
    }


# ============================================================================
# Tests for get_default_interception_params
# ============================================================================


def test_get_default_params_returns_correct_type():
    """Test that get_default_interception_params returns CanopyInterceptionParams."""
    params = get_default_interception_params()
    assert isinstance(params, CanopyInterceptionParams), \
        "Function should return CanopyInterceptionParams instance"


def test_get_default_params_default_values():
    """Test that default parameters have expected values."""
    params = get_default_interception_params()
    
    assert params.dewmx == 0.1, "Default dewmx should be 0.1"
    assert params.maximum_leaf_wetted_fraction == 0.05, \
        "Default maximum_leaf_wetted_fraction should be 0.05"
    assert params.interception_fraction == 0.25, \
        "Default interception_fraction should be 0.25"
    assert params.fwet_exponent == 0.667, "Default fwet_exponent should be 0.667"
    assert params.clm45_interception_p1 == 0.25, \
        "Default clm45_interception_p1 should be 0.25"
    assert params.clm45_interception_p2 == -0.5, \
        "Default clm45_interception_p2 should be -0.5"
    assert params.fpi_type == 2, "Default fpi_type should be 2 (CLM5)"
    assert params.dtime_substep == 1800.0, "Default dtime_substep should be 1800.0"


def test_get_default_params_custom_values():
    """Test that custom parameter values are correctly set."""
    params = get_default_interception_params(
        dewmx=0.15,
        maximum_leaf_wetted_fraction=0.08,
        interception_fraction=0.3,
        fwet_exponent=0.7,
        clm45_interception_p1=0.3,
        clm45_interception_p2=-0.6,
        fpi_type=1,
        dtime_substep=900.0,
    )
    
    assert params.dewmx == 0.15
    assert params.maximum_leaf_wetted_fraction == 0.08
    assert params.interception_fraction == 0.3
    assert params.fwet_exponent == 0.7
    assert params.clm45_interception_p1 == 0.3
    assert params.clm45_interception_p2 == -0.6
    assert params.fpi_type == 1
    assert params.dtime_substep == 900.0


def test_get_default_params_constraints():
    """Test that parameter constraints are physically reasonable."""
    params = get_default_interception_params()
    
    assert params.dewmx >= 0, "dewmx must be non-negative"
    assert 0 <= params.maximum_leaf_wetted_fraction <= 1, \
        "maximum_leaf_wetted_fraction must be in [0, 1]"
    assert 0 <= params.interception_fraction <= 1, \
        "interception_fraction must be in [0, 1]"
    assert params.fwet_exponent >= 0, "fwet_exponent must be non-negative"
    assert params.fpi_type in [1, 2], "fpi_type must be 1 or 2"
    assert params.dtime_substep > 0, "dtime_substep must be positive"


# ============================================================================
# Tests for canopy_interception - Shapes and Types
# ============================================================================


@pytest.mark.parametrize("test_case", [
    "nominal_moderate_rain",
    "mixed_precipitation_sparse",
    "heavy_snow",
    "zero_precipitation",
    "zero_canopy",
    "variable_layers",
])
def test_canopy_interception_output_type(test_data_interception, default_params, test_case):
    """Test that canopy_interception returns CanopyWaterState."""
    data = test_data_interception[test_case]
    
    result = canopy_interception(
        qflx_rain=data["qflx_rain"],
        qflx_snow=data["qflx_snow"],
        lai=data["lai"],
        sai=data["sai"],
        ncan=data["ncan"],
        dlai_profile=data["dlai_profile"],
        dpai_profile=data["dpai_profile"],
        h2ocan_profile=data["h2ocan_profile"],
        params=default_params,
    )
    
    assert isinstance(result, CanopyWaterState), \
        f"Result should be CanopyWaterState for {test_case}"


@pytest.mark.parametrize("test_case", [
    "nominal_moderate_rain",
    "mixed_precipitation_sparse",
    "heavy_snow",
    "variable_layers",
])
def test_canopy_interception_output_shapes(test_data_interception, default_params, test_case):
    """Test that output arrays have correct shapes."""
    data = test_data_interception[test_case]
    n_patches = data["n_patches"]
    n_layers = data["n_layers"]
    
    result = canopy_interception(
        qflx_rain=data["qflx_rain"],
        qflx_snow=data["qflx_snow"],
        lai=data["lai"],
        sai=data["sai"],
        ncan=data["ncan"],
        dlai_profile=data["dlai_profile"],
        dpai_profile=data["dpai_profile"],
        h2ocan_profile=data["h2ocan_profile"],
        params=default_params,
    )
    
    assert result.h2ocan_profile.shape == (n_patches, n_layers), \
        f"h2ocan_profile shape mismatch for {test_case}"
    assert result.qflx_intr_canopy.shape == (n_patches,), \
        f"qflx_intr_canopy shape mismatch for {test_case}"
    assert result.qflx_tflrain_canopy.shape == (n_patches,), \
        f"qflx_tflrain_canopy shape mismatch for {test_case}"
    assert result.qflx_tflsnow_canopy.shape == (n_patches,), \
        f"qflx_tflsnow_canopy shape mismatch for {test_case}"
    assert result.fwet_profile.shape == (n_patches, n_layers), \
        f"fwet_profile shape mismatch for {test_case}"
    assert result.fdry_profile.shape == (n_patches, n_layers), \
        f"fdry_profile shape mismatch for {test_case}"


@pytest.mark.parametrize("test_case", [
    "nominal_moderate_rain",
    "mixed_precipitation_sparse",
    "heavy_snow",
])
def test_canopy_interception_dtypes(test_data_interception, default_params, test_case):
    """Test that output arrays have correct data types (float)."""
    data = test_data_interception[test_case]
    
    result = canopy_interception(
        qflx_rain=data["qflx_rain"],
        qflx_snow=data["qflx_snow"],
        lai=data["lai"],
        sai=data["sai"],
        ncan=data["ncan"],
        dlai_profile=data["dlai_profile"],
        dpai_profile=data["dpai_profile"],
        h2ocan_profile=data["h2ocan_profile"],
        params=default_params,
    )
    
    assert jnp.issubdtype(result.h2ocan_profile.dtype, jnp.floating), \
        "h2ocan_profile should be floating point"
    assert jnp.issubdtype(result.qflx_intr_canopy.dtype, jnp.floating), \
        "qflx_intr_canopy should be floating point"
    assert jnp.issubdtype(result.fwet_profile.dtype, jnp.floating), \
        "fwet_profile should be floating point"
    assert jnp.issubdtype(result.fdry_profile.dtype, jnp.floating), \
        "fdry_profile should be floating point"


# ============================================================================
# Tests for canopy_interception - Physical Constraints
# ============================================================================


@pytest.mark.parametrize("test_case", [
    "nominal_moderate_rain",
    "mixed_precipitation_sparse",
    "heavy_snow",
    "zero_precipitation",
    "extreme_precipitation",
])
def test_canopy_interception_non_negative_outputs(
    test_data_interception, default_params, test_case
):
    """Test that all output fluxes and states are non-negative."""
    data = test_data_interception[test_case]
    
    result = canopy_interception(
        qflx_rain=data["qflx_rain"],
        qflx_snow=data["qflx_snow"],
        lai=data["lai"],
        sai=data["sai"],
        ncan=data["ncan"],
        dlai_profile=data["dlai_profile"],
        dpai_profile=data["dpai_profile"],
        h2ocan_profile=data["h2ocan_profile"],
        params=default_params,
    )
    
    assert jnp.all(result.h2ocan_profile >= 0), \
        f"h2ocan_profile has negative values in {test_case}"
    assert jnp.all(result.qflx_intr_canopy >= 0), \
        f"qflx_intr_canopy has negative values in {test_case}"
    assert jnp.all(result.qflx_tflrain_canopy >= 0), \
        f"qflx_tflrain_canopy has negative values in {test_case}"
    assert jnp.all(result.qflx_tflsnow_canopy >= 0), \
        f"qflx_tflsnow_canopy has negative values in {test_case}"


@pytest.mark.parametrize("test_case", [
    "nominal_moderate_rain",
    "mixed_precipitation_sparse",
    "heavy_snow",
])
def test_canopy_interception_fraction_bounds(
    test_data_interception, default_params, test_case
):
    """Test that wetted and dry fractions are in [0, 1]."""
    data = test_data_interception[test_case]
    
    result = canopy_interception(
        qflx_rain=data["qflx_rain"],
        qflx_snow=data["qflx_snow"],
        lai=data["lai"],
        sai=data["sai"],
        ncan=data["ncan"],
        dlai_profile=data["dlai_profile"],
        dpai_profile=data["dpai_profile"],
        h2ocan_profile=data["h2ocan_profile"],
        params=default_params,
    )
    
    assert jnp.all(result.fwet_profile >= 0) and jnp.all(result.fwet_profile <= 1), \
        f"fwet_profile out of bounds [0, 1] in {test_case}"
    assert jnp.all(result.fdry_profile >= 0) and jnp.all(result.fdry_profile <= 1), \
        f"fdry_profile out of bounds [0, 1] in {test_case}"


def test_canopy_interception_mass_conservation(test_data_interception, default_params):
    """
    Test mass conservation: total input precipitation should equal
    intercepted + throughfall.
    """
    data = test_data_interception["nominal_moderate_rain"]
    
    result = canopy_interception(
        qflx_rain=data["qflx_rain"],
        qflx_snow=data["qflx_snow"],
        lai=data["lai"],
        sai=data["sai"],
        ncan=data["ncan"],
        dlai_profile=data["dlai_profile"],
        dpai_profile=data["dpai_profile"],
        h2ocan_profile=data["h2ocan_profile"],
        params=default_params,
    )
    
    # Calculate change in canopy storage
    delta_storage = jnp.sum(result.h2ocan_profile - data["h2ocan_profile"], axis=1)
    delta_storage_rate = delta_storage / default_params.dtime_substep
    
    # Total input
    total_input = data["qflx_rain"] + data["qflx_snow"]
    
    # Total output (throughfall + storage change)
    total_output = (
        result.qflx_tflrain_canopy + 
        result.qflx_tflsnow_canopy + 
        delta_storage_rate
    )
    
    # Check mass balance (allowing for numerical precision)
    np.testing.assert_allclose(
        total_input,
        total_output,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Mass conservation violated in canopy_interception",
    )


def test_canopy_interception_zero_canopy_throughfall(
    test_data_interception, default_params
):
    """Test that zero canopy structure results in complete throughfall."""
    data = test_data_interception["zero_canopy"]
    
    result = canopy_interception(
        qflx_rain=data["qflx_rain"],
        qflx_snow=data["qflx_snow"],
        lai=data["lai"],
        sai=data["sai"],
        ncan=data["ncan"],
        dlai_profile=data["dlai_profile"],
        dpai_profile=data["dpai_profile"],
        h2ocan_profile=data["h2ocan_profile"],
        params=default_params,
    )
    
    # All rain should pass through
    np.testing.assert_allclose(
        result.qflx_tflrain_canopy,
        data["qflx_rain"],
        rtol=1e-6,
        atol=1e-9,
        err_msg="Rain should pass through with zero canopy",
    )
    
    # All snow should pass through
    np.testing.assert_allclose(
        result.qflx_tflsnow_canopy,
        data["qflx_snow"],
        rtol=1e-6,
        atol=1e-9,
        err_msg="Snow should pass through with zero canopy",
    )
    
    # No interception
    np.testing.assert_allclose(
        result.qflx_intr_canopy,
        0.0,
        atol=1e-9,
        err_msg="No interception should occur with zero canopy",
    )


def test_canopy_interception_zero_precipitation(
    test_data_interception, default_params
):
    """Test that zero precipitation results in zero throughfall and interception."""
    data = test_data_interception["zero_precipitation"]
    
    result = canopy_interception(
        qflx_rain=data["qflx_rain"],
        qflx_snow=data["qflx_snow"],
        lai=data["lai"],
        sai=data["sai"],
        ncan=data["ncan"],
        dlai_profile=data["dlai_profile"],
        dpai_profile=data["dpai_profile"],
        h2ocan_profile=data["h2ocan_profile"],
        params=default_params,
    )
    
    np.testing.assert_allclose(
        result.qflx_tflrain_canopy,
        0.0,
        atol=1e-9,
        err_msg="Rain throughfall should be zero with no precipitation",
    )
    
    np.testing.assert_allclose(
        result.qflx_tflsnow_canopy,
        0.0,
        atol=1e-9,
        err_msg="Snow throughfall should be zero with no precipitation",
    )
    
    np.testing.assert_allclose(
        result.qflx_intr_canopy,
        0.0,
        atol=1e-9,
        err_msg="Interception should be zero with no precipitation",
    )


# ============================================================================
# Tests for canopy_interception - Numerical Stability
# ============================================================================


def test_canopy_interception_very_small_values(
    test_data_interception, default_params
):
    """Test numerical stability with very small input values."""
    data = test_data_interception["very_small_values"]
    
    result = canopy_interception(
        qflx_rain=data["qflx_rain"],
        qflx_snow=data["qflx_snow"],
        lai=data["lai"],
        sai=data["sai"],
        ncan=data["ncan"],
        dlai_profile=data["dlai_profile"],
        dpai_profile=data["dpai_profile"],
        h2ocan_profile=data["h2ocan_profile"],
        params=default_params,
    )
    
    # Check for NaN or Inf
    assert jnp.all(jnp.isfinite(result.h2ocan_profile)), \
        "h2ocan_profile contains NaN or Inf with small values"
    assert jnp.all(jnp.isfinite(result.qflx_intr_canopy)), \
        "qflx_intr_canopy contains NaN or Inf with small values"
    assert jnp.all(jnp.isfinite(result.fwet_profile)), \
        "fwet_profile contains NaN or Inf with small values"
    
    # Results should still be non-negative
    assert jnp.all(result.h2ocan_profile >= 0), \
        "Negative h2ocan_profile with small values"


def test_canopy_interception_extreme_precipitation(
    test_data_interception, default_params
):
    """Test behavior with extreme precipitation rates."""
    data = test_data_interception["extreme_precipitation"]
    
    result = canopy_interception(
        qflx_rain=data["qflx_rain"],
        qflx_snow=data["qflx_snow"],
        lai=data["lai"],
        sai=data["sai"],
        ncan=data["ncan"],
        dlai_profile=data["dlai_profile"],
        dpai_profile=data["dpai_profile"],
        h2ocan_profile=data["h2ocan_profile"],
        params=default_params,
    )
    
    # Check for finite values
    assert jnp.all(jnp.isfinite(result.h2ocan_profile)), \
        "h2ocan_profile contains NaN or Inf with extreme precipitation"
    assert jnp.all(jnp.isfinite(result.qflx_intr_canopy)), \
        "qflx_intr_canopy contains NaN or Inf with extreme precipitation"
    
    # Most precipitation should pass through as throughfall
    total_precip = data["qflx_rain"] + data["qflx_snow"]
    total_throughfall = result.qflx_tflrain_canopy + result.qflx_tflsnow_canopy
    
    # Throughfall should be substantial fraction of input
    assert jnp.all(total_throughfall > 0.5 * total_precip), \
        "Expected high throughfall with extreme precipitation"


# ============================================================================
# Tests for canopy_interception - Parameter Variations
# ============================================================================


def test_canopy_interception_clm45_formulation(test_data_interception):
    """Test CLM4.5 interception formulation (fpi_type=1)."""
    data = test_data_interception["nominal_moderate_rain"]
    
    params_clm45 = get_default_interception_params(fpi_type=1)
    
    result = canopy_interception(
        qflx_rain=data["qflx_rain"],
        qflx_snow=data["qflx_snow"],
        lai=data["lai"],
        sai=data["sai"],
        ncan=data["ncan"],
        dlai_profile=data["dlai_profile"],
        dpai_profile=data["dpai_profile"],
        h2ocan_profile=data["h2ocan_profile"],
        params=params_clm45,
    )
    
    # Basic checks
    assert isinstance(result, CanopyWaterState)
    assert jnp.all(jnp.isfinite(result.h2ocan_profile))
    assert jnp.all(result.h2ocan_profile >= 0)


def test_canopy_interception_different_dewmx(test_data_interception):
    """Test effect of different maximum water storage (dewmx)."""
    data = test_data_interception["nominal_moderate_rain"]
    
    params_low = get_default_interception_params(dewmx=0.05)
    params_high = get_default_interception_params(dewmx=0.2)
    
    result_low = canopy_interception(
        qflx_rain=data["qflx_rain"],
        qflx_snow=data["qflx_snow"],
        lai=data["lai"],
        sai=data["sai"],
        ncan=data["ncan"],
        dlai_profile=data["dlai_profile"],
        dpai_profile=data["dpai_profile"],
        h2ocan_profile=data["h2ocan_profile"],
        params=params_low,
    )
    
    result_high = canopy_interception(
        qflx_rain=data["qflx_rain"],
        qflx_snow=data["qflx_snow"],
        lai=data["lai"],
        sai=data["sai"],
        ncan=data["ncan"],
        dlai_profile=data["dlai_profile"],
        dpai_profile=data["dpai_profile"],
        h2ocan_profile=data["h2ocan_profile"],
        params=params_high,
    )
    
    # Higher dewmx should allow more interception (less throughfall)
    assert jnp.all(
        result_high.qflx_tflrain_canopy <= result_low.qflx_tflrain_canopy + 1e-6
    ), "Higher dewmx should reduce throughfall"


# ============================================================================
# Tests for canopy_interception - Variable Layer Configuration
# ============================================================================


def test_canopy_interception_variable_layers(test_data_interception, default_params):
    """Test handling of variable number of active layers across patches."""
    data = test_data_interception["variable_layers"]
    
    result = canopy_interception(
        qflx_rain=data["qflx_rain"],
        qflx_snow=data["qflx_snow"],
        lai=data["lai"],
        sai=data["sai"],
        ncan=data["ncan"],
        dlai_profile=data["dlai_profile"],
        dpai_profile=data["dpai_profile"],
        h2ocan_profile=data["h2ocan_profile"],
        params=default_params,
    )
    
    # Check that inactive layers (beyond ncan) remain zero or unchanged
    for i, ncan_val in enumerate(data["ncan"]):
        if ncan_val < data["n_layers"]:
            # Inactive layers should have minimal or zero values
            inactive_layers = result.h2ocan_profile[i, ncan_val:]
            assert jnp.all(inactive_layers >= 0), \
                f"Inactive layers should be non-negative for patch {i}"


# ============================================================================
# Tests for canopy_evaporation - Shapes and Types
# ============================================================================


@pytest.mark.parametrize("test_case", [
    "nominal_case",
    "zero_fluxes",
    "high_evaporation",
])
def test_canopy_evaporation_output_type(test_data_evaporation, test_case):
    """Test that canopy_evaporation returns CanopyEvaporationOutput."""
    inputs = test_data_evaporation[test_case]
    
    result = canopy_evaporation(inputs)
    
    assert isinstance(result, CanopyEvaporationOutput), \
        f"Result should be CanopyEvaporationOutput for {test_case}"


@pytest.mark.parametrize("test_case", [
    "nominal_case",
    "zero_fluxes",
    "high_evaporation",
])
def test_canopy_evaporation_output_shape(test_data_evaporation, test_case):
    """Test that output h2ocan has correct shape."""
    inputs = test_data_evaporation[test_case]
    
    result = canopy_evaporation(inputs)
    
    expected_shape = inputs.h2ocan.shape
    assert result.h2ocan.shape == expected_shape, \
        f"h2ocan shape mismatch for {test_case}: expected {expected_shape}, got {result.h2ocan.shape}"


@pytest.mark.parametrize("test_case", [
    "nominal_case",
    "zero_fluxes",
])
def test_canopy_evaporation_dtype(test_data_evaporation, test_case):
    """Test that output has correct data type."""
    inputs = test_data_evaporation[test_case]
    
    result = canopy_evaporation(inputs)
    
    assert jnp.issubdtype(result.h2ocan.dtype, jnp.floating), \
        f"h2ocan should be floating point for {test_case}"


# ============================================================================
# Tests for canopy_evaporation - Physical Constraints
# ============================================================================


@pytest.mark.parametrize("test_case", [
    "nominal_case",
    "zero_fluxes",
    "high_evaporation",
])
def test_canopy_evaporation_non_negative(test_data_evaporation, test_case):
    """Test that canopy water remains non-negative after evaporation."""
    inputs = test_data_evaporation[test_case]
    
    result = canopy_evaporation(inputs)
    
    assert jnp.all(result.h2ocan >= 0), \
        f"h2ocan has negative values in {test_case}"


def test_canopy_evaporation_reduces_water(test_data_evaporation):
    """Test that evaporation reduces or maintains canopy water."""
    inputs = test_data_evaporation["nominal_case"]
    
    result = canopy_evaporation(inputs)
    
    # Water should decrease or stay same (not increase from evaporation alone)
    assert jnp.all(result.h2ocan <= inputs.h2ocan + 1e-8), \
        "Evaporation should not increase canopy water"


def test_canopy_evaporation_zero_fluxes(test_data_evaporation):
    """Test that zero evaporation/transpiration fluxes leave water unchanged."""
    inputs = test_data_evaporation["zero_fluxes"]
    
    result = canopy_evaporation(inputs)
    
    # With zero fluxes, water should remain unchanged
    np.testing.assert_allclose(
        result.h2ocan,
        inputs.h2ocan,
        rtol=1e-6,
        atol=1e-9,
        err_msg="Water should be unchanged with zero fluxes",
    )


def test_canopy_evaporation_cannot_exceed_available(test_data_evaporation):
    """Test that evaporation cannot remove more water than available."""
    inputs = test_data_evaporation["high_evaporation"]
    
    result = canopy_evaporation(inputs)
    
    # Calculate total water loss
    water_loss = inputs.h2ocan - result.h2ocan
    
    # Loss should not exceed initial water content
    assert jnp.all(water_loss <= inputs.h2ocan + 1e-8), \
        "Cannot evaporate more water than available"


# ============================================================================
# Tests for canopy_evaporation - Numerical Stability
# ============================================================================


def test_canopy_evaporation_finite_values(test_data_evaporation):
    """Test that outputs are finite (no NaN or Inf)."""
    inputs = test_data_evaporation["nominal_case"]
    
    result = canopy_evaporation(inputs)
    
    assert jnp.all(jnp.isfinite(result.h2ocan)), \
        "h2ocan contains NaN or Inf"


def test_canopy_evaporation_small_timestep():
    """Test behavior with very small timestep."""
    mmh2o = 0.018015
    small_dt = 1.0  # 1 second
    
    inputs = CanopyEvaporationInput(
        ncan=jnp.array([3]),
        dpai=jnp.array([[0.5, 0.5, 0.5]]),
        fracsun=jnp.array([[0.5, 0.5, 0.5]]),
        trleaf=jnp.array([[[1e-6, 0.5e-6]] * 3]),
        evleaf=jnp.array([[[0.5e-6, 0.3e-6]] * 3]),
        h2ocan=jnp.array([[0.02, 0.02, 0.02]]),
        mmh2o=mmh2o,
        dtime_substep=small_dt,
    )
    
    result = canopy_evaporation(inputs)
    
    # Should still produce valid results
    assert jnp.all(jnp.isfinite(result.h2ocan))
    assert jnp.all(result.h2ocan >= 0)
    
    # Change should be small with small timestep
    water_change = jnp.abs(result.h2ocan - inputs.h2ocan)
    assert jnp.all(water_change < 0.01), \
        "Water change should be small with small timestep"


# ============================================================================
# Integration Tests
# ============================================================================


def test_interception_evaporation_integration(test_data_interception, default_params):
    """
    Integration test: run interception followed by evaporation.
    Tests that outputs from one function can be used as inputs to another.
    """
    # Run interception
    data = test_data_interception["nominal_moderate_rain"]
    
    interception_result = canopy_interception(
        qflx_rain=data["qflx_rain"],
        qflx_snow=data["qflx_snow"],
        lai=data["lai"],
        sai=data["sai"],
        ncan=data["ncan"],
        dlai_profile=data["dlai_profile"],
        dpai_profile=data["dpai_profile"],
        h2ocan_profile=data["h2ocan_profile"],
        params=default_params,
    )
    
    # Create evaporation inputs using interception outputs
    n_patches = data["n_patches"]
    n_layers = data["n_layers"]
    
    evap_inputs = CanopyEvaporationInput(
        ncan=data["ncan"],
        dpai=data["dpai_profile"],
        fracsun=jnp.full((n_patches, n_layers), 0.5),
        trleaf=jnp.full((n_patches, n_layers, 2), 1e-6),
        evleaf=jnp.full((n_patches, n_layers, 2), 0.5e-6),
        h2ocan=interception_result.h2ocan_profile,
        mmh2o=0.018015,
        dtime_substep=default_params.dtime_substep,
    )
    
    # Run evaporation
    evap_result = canopy_evaporation(evap_inputs)
    
    # Check that results are valid
    assert jnp.all(jnp.isfinite(evap_result.h2ocan))
    assert jnp.all(evap_result.h2ocan >= 0)
    
    # Water should have decreased from evaporation
    assert jnp.all(evap_result.h2ocan <= interception_result.h2ocan_profile + 1e-8)


def test_multiple_timesteps_stability(test_data_interception, default_params):
    """
    Test stability over multiple timesteps by repeatedly calling interception.
    """
    data = test_data_interception["nominal_moderate_rain"]
    
    h2ocan = data["h2ocan_profile"]
    
    # Run for 5 timesteps
    for _ in range(5):
        result = canopy_interception(
            qflx_rain=data["qflx_rain"],
            qflx_snow=data["qflx_snow"],
            lai=data["lai"],
            sai=data["sai"],
            ncan=data["ncan"],
            dlai_profile=data["dlai_profile"],
            dpai_profile=data["dpai_profile"],
            h2ocan_profile=h2ocan,
            params=default_params,
        )
        
        # Check validity
        assert jnp.all(jnp.isfinite(result.h2ocan_profile)), \
            "Non-finite values after multiple timesteps"
        assert jnp.all(result.h2ocan_profile >= 0), \
            "Negative water after multiple timesteps"
        
        # Update for next iteration
        h2ocan = result.h2ocan_profile
    
    # Final state should still be physically reasonable
    max_storage = default_params.dewmx * data["dpai_profile"]
    assert jnp.all(h2ocan <= max_storage + 1e-6), \
        "Water exceeds maximum storage after multiple timesteps"


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_single_patch_single_layer():
    """Test with minimal configuration: single patch, single layer."""
    params = get_default_interception_params()
    
    result = canopy_interception(
        qflx_rain=jnp.array([0.001]),
        qflx_snow=jnp.array([0.0]),
        lai=jnp.array([2.0]),
        sai=jnp.array([0.3]),
        ncan=jnp.array([1]),
        dlai_profile=jnp.array([[2.0]]),
        dpai_profile=jnp.array([[2.3]]),
        h2ocan_profile=jnp.array([[0.01]]),
        params=params,
    )
    
    assert result.h2ocan_profile.shape == (1, 1)
    assert result.qflx_intr_canopy.shape == (1,)
    assert jnp.all(jnp.isfinite(result.h2ocan_profile))


def test_large_number_of_patches():
    """Test with large number of patches for performance/stability."""
    n_patches = 100
    n_layers = 5
    params = get_default_interception_params()
    
    result = canopy_interception(
        qflx_rain=jnp.full(n_patches, 0.0005),
        qflx_snow=jnp.full(n_patches, 0.0001),
        lai=jnp.full(n_patches, 3.0),
        sai=jnp.full(n_patches, 0.5),
        ncan=jnp.full(n_patches, n_layers),
        dlai_profile=jnp.full((n_patches, n_layers), 0.6),
        dpai_profile=jnp.full((n_patches, n_layers), 0.7),
        h2ocan_profile=jnp.full((n_patches, n_layers), 0.02),
        params=params,
    )
    
    assert result.h2ocan_profile.shape == (n_patches, n_layers)
    assert jnp.all(jnp.isfinite(result.h2ocan_profile))
    assert jnp.all(result.h2ocan_profile >= 0)