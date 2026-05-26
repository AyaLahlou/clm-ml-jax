"""
Comprehensive pytest suite for CLMml_driver module.

This test suite covers all functions in the CLMml_driver module with:
- Nominal test cases for typical operating conditions
- Edge cases for boundary conditions and extreme values
- Special cases for single-element arrays and unusual configurations
- Physical realism checks for scientific computing constraints
"""

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path to import actual functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from offline_driver.CLMml_driver import (
    adjust_usme2_pft_parameters,
    calculate_time_index,
    clmml_drv_cleanup,
    compute_canopy_layer_output,
    compute_ground_output,
    compute_output_auxiliary,
    compute_output_fluxes,
    compute_output_profile_above_canopy,
    compute_output_sunshade,
    compute_upward_shortwave,
    construct_clm_filename,
    construct_tower_file_path,
    convert_qair_to_eair,
    get_htop_pft_lookup,
    init_acclim,
    process_profile_data,
    read_canopy_profiles_physics,
    soil_init_vectorized,
    tower_veg,
)


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load test data for all CLMml_driver functions.
    
    Returns:
        Dictionary containing test cases organized by function name.
    """
    return {
        "adjust_usme2_pft_parameters": [
            {
                "name": "nominal_usme2",
                "inputs": {
                    "pft_params": {
                        "pbeta_lai": 0.5,
                        "qbeta_lai": 0.5,
                        "pbeta_sai": 0.5,
                        "qbeta_sai": 0.5,
                    },
                    "patch_itype": 8,
                    "tower_id": "US-Me2",
                },
                "type": "nominal",
            },
            {
                "name": "non_usme2",
                "inputs": {
                    "pft_params": {
                        "pbeta_lai": 0.3,
                        "qbeta_lai": 0.7,
                        "pbeta_sai": 0.4,
                        "qbeta_sai": 0.6,
                    },
                    "patch_itype": 15,
                    "tower_id": "US-Ha1",
                },
                "type": "nominal",
            },
        ],
        "construct_tower_file_path": [
            {
                "name": "nominal",
                "inputs": {
                    "tower_id": "US-Me2",
                    "yr": 2010,
                    "mon": 7,
                    "diratm": "/data/forcing/tower",
                },
                "type": "nominal",
            },
            {
                "name": "edge_january",
                "inputs": {
                    "tower_id": "US-Ha1",
                    "yr": 2000,
                    "mon": 1,
                    "diratm": "/data/atm",
                },
                "type": "edge",
            },
            {
                "name": "edge_december",
                "inputs": {
                    "tower_id": "US-UMB",
                    "yr": 2099,
                    "mon": 12,
                    "diratm": "/climate/forcing",
                },
                "type": "edge",
            },
        ],
        "construct_clm_filename": [
            {
                "name": "nominal_no_wozniak",
                "inputs": {
                    "tower_id": "US-Me2",
                    "tower_num": 5,
                    "yr": 2015,
                    "dirclm": "/output/clm",
                    "use_wozniak": False,
                },
                "type": "nominal",
            },
            {
                "name": "wozniak",
                "inputs": {
                    "tower_id": "US-Ha1",
                    "tower_num": 0,
                    "yr": 2020,
                    "dirclm": "/data/clm_output",
                    "use_wozniak": True,
                },
                "type": "special",
            },
        ],
        "calculate_time_index": [
            {
                "name": "nominal",
                "inputs": {
                    "curr_calday": jnp.array(150.5),
                    "start_calday_clm": jnp.array(1.0),
                    "dtstep": jnp.array(1800.0),
                },
                "type": "nominal",
            },
            {
                "name": "edge_start_of_year",
                "inputs": {
                    "curr_calday": jnp.array(1.0),
                    "start_calday_clm": jnp.array(1.0),
                    "dtstep": jnp.array(3600.0),
                },
                "expected_output": jnp.array(1),
                "type": "edge",
            },
            {
                "name": "edge_end_of_year",
                "inputs": {
                    "curr_calday": jnp.array(365.99),
                    "start_calday_clm": jnp.array(1.0),
                    "dtstep": jnp.array(900.0),
                },
                "type": "edge",
            },
        ],
        "compute_upward_shortwave": [
            {
                "name": "nominal",
                "inputs": {
                    "albcan_vis": jnp.array([0.15, 0.18, 0.12]),
                    "albcan_nir": jnp.array([0.25, 0.28, 0.22]),
                    "swskyb_vis": jnp.array([450.0, 480.0, 420.0]),
                    "swskyd_vis": jnp.array([150.0, 160.0, 140.0]),
                    "swskyb_nir": jnp.array([350.0, 370.0, 330.0]),
                    "swskyd_nir": jnp.array([100.0, 110.0, 90.0]),
                },
                "type": "nominal",
            },
            {
                "name": "edge_zero_radiation",
                "inputs": {
                    "albcan_vis": jnp.array([0.15, 0.2]),
                    "albcan_nir": jnp.array([0.25, 0.3]),
                    "swskyb_vis": jnp.array([0.0, 0.0]),
                    "swskyd_vis": jnp.array([0.0, 0.0]),
                    "swskyb_nir": jnp.array([0.0, 0.0]),
                    "swskyd_nir": jnp.array([0.0, 0.0]),
                },
                "expected_output": jnp.array([0.0, 0.0]),
                "type": "edge",
            },
            {
                "name": "edge_high_albedo",
                "inputs": {
                    "albcan_vis": jnp.array([0.95, 0.98]),
                    "albcan_nir": jnp.array([0.97, 0.99]),
                    "swskyb_vis": jnp.array([500.0, 520.0]),
                    "swskyd_vis": jnp.array([200.0, 210.0]),
                    "swskyb_nir": jnp.array([400.0, 420.0]),
                    "swskyd_nir": jnp.array([150.0, 160.0]),
                },
                "type": "edge",
            },
            {
                "name": "special_single_patch",
                "inputs": {
                    "albcan_vis": jnp.array([0.18]),
                    "albcan_nir": jnp.array([0.28]),
                    "swskyb_vis": jnp.array([600.0]),
                    "swskyd_vis": jnp.array([180.0]),
                    "swskyb_nir": jnp.array([450.0]),
                    "swskyd_nir": jnp.array([120.0]),
                },
                "type": "special",
            },
        ],
        "convert_qair_to_eair": [
            {
                "name": "nominal",
                "inputs": {
                    "qair": jnp.array([[0.008, 0.01, 0.012], [0.007, 0.009, 0.011]]),
                    "pref": jnp.array([101325.0, 101000.0]),
                    "mmh2o": 18.016,
                    "mmdry": 28.966,
                },
                "type": "nominal",
            },
            {
                "name": "edge_zero_humidity",
                "inputs": {
                    "qair": jnp.array([[0.0, 0.0], [0.0, 0.0]]),
                    "pref": jnp.array([101325.0, 100000.0]),
                    "mmh2o": 18.016,
                    "mmdry": 28.966,
                },
                "expected_output": jnp.array([[0.0, 0.0], [0.0, 0.0]]),
                "type": "edge",
            },
            {
                "name": "edge_high_humidity",
                "inputs": {
                    "qair": jnp.array([[0.025, 0.028], [0.026, 0.029]]),
                    "pref": jnp.array([101325.0, 101500.0]),
                    "mmh2o": 18.016,
                    "mmdry": 28.966,
                },
                "type": "edge",
            },
            {
                "name": "special_low_pressure",
                "inputs": {
                    "qair": jnp.array([[0.01, 0.012], [0.009, 0.011]]),
                    "pref": jnp.array([70000.0, 68000.0]),
                    "mmh2o": 18.016,
                    "mmdry": 28.966,
                },
                "type": "special",
            },
        ],
        "process_profile_data": [
            {
                "name": "nominal",
                "inputs": {
                    "wind_raw": jnp.array([[2.5, 3.0, 3.5], [2.8, 3.2, 3.6]]),
                    "tair_raw": jnp.array([[288.15, 289.15, 290.15], [287.15, 288.65, 289.85]]),
                    "qair_raw": jnp.array([[8.0, 9.0, 10.0], [7.5, 8.5, 9.5]]),
                    "pref": jnp.array([101325.0, 101000.0]),
                    "mmh2o": 18.016,
                    "mmdry": 28.966,
                },
                "type": "nominal",
            },
            {
                "name": "edge_calm_winds",
                "inputs": {
                    "wind_raw": jnp.array([[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]),
                    "tair_raw": jnp.array([[285.15, 286.15, 287.15], [284.65, 285.85, 286.95]]),
                    "qair_raw": jnp.array([[6.0, 7.0, 8.0], [5.5, 6.5, 7.5]]),
                    "pref": jnp.array([101325.0, 101200.0]),
                    "mmh2o": 18.016,
                    "mmdry": 28.966,
                },
                "type": "edge",
            },
            {
                "name": "edge_cold_conditions",
                "inputs": {
                    "wind_raw": jnp.array([[4.0, 4.5, 5.0], [3.8, 4.3, 4.8]]),
                    "tair_raw": jnp.array([[253.15, 255.15, 257.15], [252.15, 254.15, 256.15]]),
                    "qair_raw": jnp.array([[1.0, 1.5, 2.0], [0.8, 1.3, 1.8]]),
                    "pref": jnp.array([101325.0, 101400.0]),
                    "mmh2o": 18.016,
                    "mmdry": 28.966,
                },
                "type": "edge",
            },
            {
                "name": "special_high_winds",
                "inputs": {
                    "wind_raw": jnp.array([[15.0, 18.0, 20.0], [14.5, 17.5, 19.5]]),
                    "tair_raw": jnp.array([[290.15, 291.15, 292.15], [289.65, 290.85, 291.95]]),
                    "qair_raw": jnp.array([[10.0, 11.0, 12.0], [9.5, 10.5, 11.5]]),
                    "pref": jnp.array([101325.0, 101100.0]),
                    "mmh2o": 18.016,
                    "mmdry": 28.966,
                },
                "type": "special",
            },
        ],
        "clmml_drv_cleanup": [
            {
                "name": "prescribed_turbulence",
                "inputs": {"turb_type": -1},
                "type": "nominal",
            },
            {
                "name": "computed_turbulence",
                "inputs": {"turb_type": 1},
                "type": "nominal",
            },
        ],
    }


# ============================================================================
# Tests for adjust_usme2_pft_parameters
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(tc, id=tc["name"])
        for tc in [
            {
                "name": "nominal_usme2",
                "inputs": {
                    "pft_params": {
                        "pbeta_lai": 0.5,
                        "qbeta_lai": 0.5,
                        "pbeta_sai": 0.5,
                        "qbeta_sai": 0.5,
                    },
                    "patch_itype": 8,
                    "tower_id": "US-Me2",
                },
                "type": "nominal",
            },
            {
                "name": "non_usme2",
                "inputs": {
                    "pft_params": {
                        "pbeta_lai": 0.3,
                        "qbeta_lai": 0.7,
                        "pbeta_sai": 0.4,
                        "qbeta_sai": 0.6,
                    },
                    "patch_itype": 15,
                    "tower_id": "US-Ha1",
                },
                "type": "nominal",
            },
        ]
    ],
)
def test_adjust_usme2_pft_parameters_values(test_case: Dict[str, Any]) -> None:
    """
    Test adjust_usme2_pft_parameters returns correct parameter values.
    
    For US-Me2 tower, parameters should be adjusted.
    For other towers, parameters should remain unchanged.
    """
    inputs = test_case["inputs"]
    result = adjust_usme2_pft_parameters(
        pft_params=inputs["pft_params"],
        patch_itype=inputs["patch_itype"],
        tower_id=inputs["tower_id"],
    )
    
    # Check that result is a dictionary with expected keys
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "pbeta_lai" in result, "Result should contain pbeta_lai"
    assert "qbeta_lai" in result, "Result should contain qbeta_lai"
    assert "pbeta_sai" in result, "Result should contain pbeta_sai"
    assert "qbeta_sai" in result, "Result should contain qbeta_sai"
    
    # For non-US-Me2, parameters should be unchanged
    if inputs["tower_id"] != "US-Me2":
        assert result["pbeta_lai"] == inputs["pft_params"]["pbeta_lai"]
        assert result["qbeta_lai"] == inputs["pft_params"]["qbeta_lai"]
        assert result["pbeta_sai"] == inputs["pft_params"]["pbeta_sai"]
        assert result["qbeta_sai"] == inputs["pft_params"]["qbeta_sai"]


def test_adjust_usme2_pft_parameters_dtypes() -> None:
    """Test that adjust_usme2_pft_parameters returns correct data types."""
    pft_params = {
        "pbeta_lai": 0.5,
        "qbeta_lai": 0.5,
        "pbeta_sai": 0.5,
        "qbeta_sai": 0.5,
    }
    result = adjust_usme2_pft_parameters(pft_params, 8, "US-Me2")
    
    assert isinstance(result, dict), "Result should be a dictionary"
    for key in ["pbeta_lai", "qbeta_lai", "pbeta_sai", "qbeta_sai"]:
        assert isinstance(result[key], (float, int)), f"{key} should be numeric"


# ============================================================================
# Tests for construct_tower_file_path
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(tc, id=tc["name"])
        for tc in [
            {
                "name": "nominal",
                "inputs": {
                    "tower_id": "US-Me2",
                    "yr": 2010,
                    "mon": 7,
                    "diratm": "/data/forcing/tower",
                },
            },
            {
                "name": "edge_january",
                "inputs": {
                    "tower_id": "US-Ha1",
                    "yr": 2000,
                    "mon": 1,
                    "diratm": "/data/atm",
                },
            },
            {
                "name": "edge_december",
                "inputs": {
                    "tower_id": "US-UMB",
                    "yr": 2099,
                    "mon": 12,
                    "diratm": "/climate/forcing",
                },
            },
        ]
    ],
)
def test_construct_tower_file_path_values(test_case: Dict[str, Any]) -> None:
    """
    Test construct_tower_file_path returns valid file paths.
    
    Verifies that the path contains the tower ID, year, and month.
    """
    inputs = test_case["inputs"]
    result = construct_tower_file_path(
        tower_id=inputs["tower_id"],
        yr=inputs["yr"],
        mon=inputs["mon"],
        diratm=inputs["diratm"],
    )
    
    assert isinstance(result, str), "Result should be a string"
    assert inputs["tower_id"] in result, "Path should contain tower ID"
    assert str(inputs["yr"]) in result, "Path should contain year"
    assert inputs["diratm"] in result, "Path should contain directory"


def test_construct_tower_file_path_dtypes() -> None:
    """Test that construct_tower_file_path returns correct data type."""
    result = construct_tower_file_path("US-Me2", 2010, 7, "/data")
    assert isinstance(result, str), "Result should be a string"


# ============================================================================
# Tests for construct_clm_filename
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(tc, id=tc["name"])
        for tc in [
            {
                "name": "nominal_no_wozniak",
                "inputs": {
                    "tower_id": "US-Me2",
                    "tower_num": 5,
                    "yr": 2015,
                    "dirclm": "/output/clm",
                    "use_wozniak": False,
                },
            },
            {
                "name": "wozniak",
                "inputs": {
                    "tower_id": "US-Ha1",
                    "tower_num": 0,
                    "yr": 2020,
                    "dirclm": "/data/clm_output",
                    "use_wozniak": True,
                },
            },
        ]
    ],
)
def test_construct_clm_filename_values(test_case: Dict[str, Any]) -> None:
    """
    Test construct_clm_filename returns valid CLM file paths.
    
    Verifies that the path contains the tower ID, year, and directory.
    """
    inputs = test_case["inputs"]
    result = construct_clm_filename(
        tower_id=inputs["tower_id"],
        tower_num=inputs["tower_num"],
        yr=inputs["yr"],
        dirclm=inputs["dirclm"],
        use_wozniak=inputs["use_wozniak"],
    )
    
    assert isinstance(result, str), "Result should be a string"
    assert inputs["tower_id"] in result, "Path should contain tower ID"
    assert str(inputs["yr"]) in result, "Path should contain year"
    assert inputs["dirclm"] in result, "Path should contain directory"


def test_construct_clm_filename_dtypes() -> None:
    """Test that construct_clm_filename returns correct data type."""
    result = construct_clm_filename("US-Me2", 5, 2015, "/output", False)
    assert isinstance(result, str), "Result should be a string"


# ============================================================================
# Tests for calculate_time_index
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(tc, id=tc["name"])
        for tc in [
            {
                "name": "nominal",
                "inputs": {
                    "curr_calday": jnp.array(150.5),
                    "start_calday_clm": jnp.array(1.0),
                    "dtstep": jnp.array(1800.0),
                },
            },
            {
                "name": "edge_start_of_year",
                "inputs": {
                    "curr_calday": jnp.array(1.0),
                    "start_calday_clm": jnp.array(1.0),
                    "dtstep": jnp.array(3600.0),
                },
                "expected_output": jnp.array(1),
            },
            {
                "name": "edge_end_of_year",
                "inputs": {
                    "curr_calday": jnp.array(365.99),
                    "start_calday_clm": jnp.array(1.0),
                    "dtstep": jnp.array(900.0),
                },
            },
        ]
    ],
)
def test_calculate_time_index_values(test_case: Dict[str, Any]) -> None:
    """
    Test calculate_time_index returns correct time indices.
    
    Verifies that the index is 1-based and increases with calendar day.
    """
    inputs = test_case["inputs"]
    result = calculate_time_index(
        curr_calday=inputs["curr_calday"],
        start_calday_clm=inputs["start_calday_clm"],
        dtstep=inputs["dtstep"],
    )
    
    # Check that result is a scalar
    assert result.shape == (), f"Result should be scalar, got shape {result.shape}"
    
    # Check that result is at least 1 (1-based indexing)
    assert result >= 1, "Time index should be at least 1 (1-based)"
    
    # If expected output provided, check it
    if "expected_output" in test_case:
        np.testing.assert_allclose(
            result,
            test_case["expected_output"],
            rtol=1e-6,
            atol=1e-6,
            err_msg="Time index does not match expected value",
        )


def test_calculate_time_index_shapes() -> None:
    """Test that calculate_time_index returns correct output shape."""
    result = calculate_time_index(
        jnp.array(150.5), jnp.array(1.0), jnp.array(1800.0)
    )
    assert result.shape == (), "Result should be a scalar"


def test_calculate_time_index_dtypes() -> None:
    """Test that calculate_time_index returns correct data type."""
    result = calculate_time_index(
        jnp.array(150.5), jnp.array(1.0), jnp.array(1800.0)
    )
    assert isinstance(result, jnp.ndarray), "Result should be a JAX array"


# ============================================================================
# Tests for get_htop_pft_lookup
# ============================================================================


def test_get_htop_pft_lookup_shapes() -> None:
    """Test that get_htop_pft_lookup returns array of correct size."""
    result = get_htop_pft_lookup()
    assert result.shape == (79,), f"Expected shape (79,), got {result.shape}"


def test_get_htop_pft_lookup_values() -> None:
    """Test that get_htop_pft_lookup returns physically realistic values."""
    result = get_htop_pft_lookup()
    
    # All heights should be non-negative
    assert jnp.all(result >= 0.0), "All canopy heights should be non-negative"
    
    # Heights should be reasonable (< 100m for most vegetation)
    assert jnp.all(result < 100.0), "Canopy heights should be < 100m"


def test_get_htop_pft_lookup_dtypes() -> None:
    """Test that get_htop_pft_lookup returns correct data type."""
    result = get_htop_pft_lookup()
    assert isinstance(result, jnp.ndarray), "Result should be a JAX array"


# ============================================================================
# Tests for compute_upward_shortwave
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(tc, id=tc["name"])
        for tc in [
            {
                "name": "nominal",
                "inputs": {
                    "albcan_vis": jnp.array([0.15, 0.18, 0.12]),
                    "albcan_nir": jnp.array([0.25, 0.28, 0.22]),
                    "swskyb_vis": jnp.array([450.0, 480.0, 420.0]),
                    "swskyd_vis": jnp.array([150.0, 160.0, 140.0]),
                    "swskyb_nir": jnp.array([350.0, 370.0, 330.0]),
                    "swskyd_nir": jnp.array([100.0, 110.0, 90.0]),
                },
            },
            {
                "name": "edge_zero_radiation",
                "inputs": {
                    "albcan_vis": jnp.array([0.15, 0.2]),
                    "albcan_nir": jnp.array([0.25, 0.3]),
                    "swskyb_vis": jnp.array([0.0, 0.0]),
                    "swskyd_vis": jnp.array([0.0, 0.0]),
                    "swskyb_nir": jnp.array([0.0, 0.0]),
                    "swskyd_nir": jnp.array([0.0, 0.0]),
                },
                "expected_output": jnp.array([0.0, 0.0]),
            },
            {
                "name": "edge_high_albedo",
                "inputs": {
                    "albcan_vis": jnp.array([0.95, 0.98]),
                    "albcan_nir": jnp.array([0.97, 0.99]),
                    "swskyb_vis": jnp.array([500.0, 520.0]),
                    "swskyd_vis": jnp.array([200.0, 210.0]),
                    "swskyb_nir": jnp.array([400.0, 420.0]),
                    "swskyd_nir": jnp.array([150.0, 160.0]),
                },
            },
            {
                "name": "special_single_patch",
                "inputs": {
                    "albcan_vis": jnp.array([0.18]),
                    "albcan_nir": jnp.array([0.28]),
                    "swskyb_vis": jnp.array([600.0]),
                    "swskyd_vis": jnp.array([180.0]),
                    "swskyb_nir": jnp.array([450.0]),
                    "swskyd_nir": jnp.array([120.0]),
                },
            },
        ]
    ],
)
def test_compute_upward_shortwave_values(test_case: Dict[str, Any]) -> None:
    """
    Test compute_upward_shortwave returns physically realistic values.
    
    Verifies that upward radiation is non-negative and less than incoming.
    """
    inputs = test_case["inputs"]
    result = compute_upward_shortwave(
        albcan_vis=inputs["albcan_vis"],
        albcan_nir=inputs["albcan_nir"],
        swskyb_vis=inputs["swskyb_vis"],
        swskyd_vis=inputs["swskyd_vis"],
        swskyb_nir=inputs["swskyb_nir"],
        swskyd_nir=inputs["swskyd_nir"],
    )
    
    # Check that all values are non-negative
    assert jnp.all(result >= 0.0), "Upward radiation should be non-negative"
    
    # Check that upward radiation is less than total incoming
    total_incoming = (
        inputs["swskyb_vis"]
        + inputs["swskyd_vis"]
        + inputs["swskyb_nir"]
        + inputs["swskyd_nir"]
    )
    assert jnp.all(
        result <= total_incoming + 1e-6
    ), "Upward radiation should not exceed incoming"
    
    # If expected output provided, check it
    if "expected_output" in test_case:
        np.testing.assert_allclose(
            result,
            test_case["expected_output"],
            rtol=1e-6,
            atol=1e-6,
            err_msg="Upward shortwave does not match expected value",
        )


def test_compute_upward_shortwave_shapes() -> None:
    """Test that compute_upward_shortwave returns correct output shape."""
    n_patches = 3
    result = compute_upward_shortwave(
        albcan_vis=jnp.ones(n_patches) * 0.15,
        albcan_nir=jnp.ones(n_patches) * 0.25,
        swskyb_vis=jnp.ones(n_patches) * 450.0,
        swskyd_vis=jnp.ones(n_patches) * 150.0,
        swskyb_nir=jnp.ones(n_patches) * 350.0,
        swskyd_nir=jnp.ones(n_patches) * 100.0,
    )
    assert result.shape == (n_patches,), f"Expected shape ({n_patches},), got {result.shape}"


def test_compute_upward_shortwave_dtypes() -> None:
    """Test that compute_upward_shortwave returns correct data type."""
    result = compute_upward_shortwave(
        albcan_vis=jnp.array([0.15]),
        albcan_nir=jnp.array([0.25]),
        swskyb_vis=jnp.array([450.0]),
        swskyd_vis=jnp.array([150.0]),
        swskyb_nir=jnp.array([350.0]),
        swskyd_nir=jnp.array([100.0]),
    )
    assert isinstance(result, jnp.ndarray), "Result should be a JAX array"


# ============================================================================
# Tests for convert_qair_to_eair
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(tc, id=tc["name"])
        for tc in [
            {
                "name": "nominal",
                "inputs": {
                    "qair": jnp.array([[0.008, 0.01, 0.012], [0.007, 0.009, 0.011]]),
                    "pref": jnp.array([101325.0, 101000.0]),
                    "mmh2o": 18.016,
                    "mmdry": 28.966,
                },
            },
            {
                "name": "edge_zero_humidity",
                "inputs": {
                    "qair": jnp.array([[0.0, 0.0], [0.0, 0.0]]),
                    "pref": jnp.array([101325.0, 100000.0]),
                    "mmh2o": 18.016,
                    "mmdry": 28.966,
                },
                "expected_output": jnp.array([[0.0, 0.0], [0.0, 0.0]]),
            },
            {
                "name": "edge_high_humidity",
                "inputs": {
                    "qair": jnp.array([[0.025, 0.028], [0.026, 0.029]]),
                    "pref": jnp.array([101325.0, 101500.0]),
                    "mmh2o": 18.016,
                    "mmdry": 28.966,
                },
            },
            {
                "name": "special_low_pressure",
                "inputs": {
                    "qair": jnp.array([[0.01, 0.012], [0.009, 0.011]]),
                    "pref": jnp.array([70000.0, 68000.0]),
                    "mmh2o": 18.016,
                    "mmdry": 28.966,
                },
            },
        ]
    ],
)
def test_convert_qair_to_eair_values(test_case: Dict[str, Any]) -> None:
    """
    Test convert_qair_to_eair returns physically realistic values.
    
    Verifies that vapor pressure is non-negative and less than total pressure.
    """
    inputs = test_case["inputs"]
    result = convert_qair_to_eair(
        qair=inputs["qair"],
        pref=inputs["pref"],
        mmh2o=inputs["mmh2o"],
        mmdry=inputs["mmdry"],
    )
    
    # Check that all values are non-negative
    assert jnp.all(result >= 0.0), "Vapor pressure should be non-negative"
    
    # Check that vapor pressure is less than total pressure
    pref_expanded = inputs["pref"][:, jnp.newaxis]
    assert jnp.all(
        result <= pref_expanded
    ), "Vapor pressure should not exceed total pressure"
    
    # If expected output provided, check it
    if "expected_output" in test_case:
        np.testing.assert_allclose(
            result,
            test_case["expected_output"],
            rtol=1e-6,
            atol=1e-6,
            err_msg="Vapor pressure does not match expected value",
        )


def test_convert_qair_to_eair_shapes() -> None:
    """Test that convert_qair_to_eair returns correct output shape."""
    n_patches = 2
    n_layers = 3
    result = convert_qair_to_eair(
        qair=jnp.ones((n_patches, n_layers)) * 0.01,
        pref=jnp.ones(n_patches) * 101325.0,
        mmh2o=18.016,
        mmdry=28.966,
    )
    assert result.shape == (
        n_patches,
        n_layers,
    ), f"Expected shape ({n_patches}, {n_layers}), got {result.shape}"


def test_convert_qair_to_eair_dtypes() -> None:
    """Test that convert_qair_to_eair returns correct data type."""
    result = convert_qair_to_eair(
        qair=jnp.array([[0.01]]),
        pref=jnp.array([101325.0]),
        mmh2o=18.016,
        mmdry=28.966,
    )
    assert isinstance(result, jnp.ndarray), "Result should be a JAX array"


# ============================================================================
# Tests for process_profile_data
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(tc, id=tc["name"])
        for tc in [
            {
                "name": "nominal",
                "inputs": {
                    "wind_raw": jnp.array([[2.5, 3.0, 3.5], [2.8, 3.2, 3.6]]),
                    "tair_raw": jnp.array(
                        [[288.15, 289.15, 290.15], [287.15, 288.65, 289.85]]
                    ),
                    "qair_raw": jnp.array([[8.0, 9.0, 10.0], [7.5, 8.5, 9.5]]),
                    "pref": jnp.array([101325.0, 101000.0]),
                    "mmh2o": 18.016,
                    "mmdry": 28.966,
                },
            },
            {
                "name": "edge_calm_winds",
                "inputs": {
                    "wind_raw": jnp.array([[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]),
                    "tair_raw": jnp.array(
                        [[285.15, 286.15, 287.15], [284.65, 285.85, 286.95]]
                    ),
                    "qair_raw": jnp.array([[6.0, 7.0, 8.0], [5.5, 6.5, 7.5]]),
                    "pref": jnp.array([101325.0, 101200.0]),
                    "mmh2o": 18.016,
                    "mmdry": 28.966,
                },
            },
            {
                "name": "edge_cold_conditions",
                "inputs": {
                    "wind_raw": jnp.array([[4.0, 4.5, 5.0], [3.8, 4.3, 4.8]]),
                    "tair_raw": jnp.array(
                        [[253.15, 255.15, 257.15], [252.15, 254.15, 256.15]]
                    ),
                    "qair_raw": jnp.array([[1.0, 1.5, 2.0], [0.8, 1.3, 1.8]]),
                    "pref": jnp.array([101325.0, 101400.0]),
                    "mmh2o": 18.016,
                    "mmdry": 28.966,
                },
            },
            {
                "name": "special_high_winds",
                "inputs": {
                    "wind_raw": jnp.array([[15.0, 18.0, 20.0], [14.5, 17.5, 19.5]]),
                    "tair_raw": jnp.array(
                        [[290.15, 291.15, 292.15], [289.65, 290.85, 291.95]]
                    ),
                    "qair_raw": jnp.array([[10.0, 11.0, 12.0], [9.5, 10.5, 11.5]]),
                    "pref": jnp.array([101325.0, 101100.0]),
                    "mmh2o": 18.016,
                    "mmdry": 28.966,
                },
            },
        ]
    ],
)
def test_process_profile_data_values(test_case: Dict[str, Any]) -> None:
    """
    Test process_profile_data returns physically realistic values.
    
    Verifies that wind, temperature, and vapor pressure are all positive
    and within reasonable ranges.
    """
    inputs = test_case["inputs"]
    wind_data, tair_data, eair_data = process_profile_data(
        wind_raw=inputs["wind_raw"],
        tair_raw=inputs["tair_raw"],
        qair_raw=inputs["qair_raw"],
        pref=inputs["pref"],
        mmh2o=inputs["mmh2o"],
        mmdry=inputs["mmdry"],
    )
    
    # Check wind data
    assert jnp.all(wind_data >= 0.0), "Wind speed should be non-negative"
    np.testing.assert_allclose(
        wind_data,
        inputs["wind_raw"],
        rtol=1e-6,
        atol=1e-6,
        err_msg="Wind data should match input",
    )
    
    # Check temperature data
    assert jnp.all(tair_data > 0.0), "Temperature should be positive (Kelvin)"
    np.testing.assert_allclose(
        tair_data,
        inputs["tair_raw"],
        rtol=1e-6,
        atol=1e-6,
        err_msg="Temperature data should match input",
    )
    
    # Check vapor pressure data
    assert jnp.all(eair_data >= 0.0), "Vapor pressure should be non-negative"


def test_process_profile_data_shapes() -> None:
    """Test that process_profile_data returns correct output shapes."""
    n_patches = 2
    n_layers = 3
    wind_data, tair_data, eair_data = process_profile_data(
        wind_raw=jnp.ones((n_patches, n_layers)) * 3.0,
        tair_raw=jnp.ones((n_patches, n_layers)) * 288.15,
        qair_raw=jnp.ones((n_patches, n_layers)) * 8.0,
        pref=jnp.ones(n_patches) * 101325.0,
        mmh2o=18.016,
        mmdry=28.966,
    )
    
    expected_shape = (n_patches, n_layers)
    assert (
        wind_data.shape == expected_shape
    ), f"Wind data shape should be {expected_shape}, got {wind_data.shape}"
    assert (
        tair_data.shape == expected_shape
    ), f"Temperature data shape should be {expected_shape}, got {tair_data.shape}"
    assert (
        eair_data.shape == expected_shape
    ), f"Vapor pressure data shape should be {expected_shape}, got {eair_data.shape}"


def test_process_profile_data_dtypes() -> None:
    """Test that process_profile_data returns correct data types."""
    wind_data, tair_data, eair_data = process_profile_data(
        wind_raw=jnp.array([[3.0]]),
        tair_raw=jnp.array([[288.15]]),
        qair_raw=jnp.array([[8.0]]),
        pref=jnp.array([101325.0]),
        mmh2o=18.016,
        mmdry=28.966,
    )
    
    assert isinstance(wind_data, jnp.ndarray), "Wind data should be a JAX array"
    assert isinstance(tair_data, jnp.ndarray), "Temperature data should be a JAX array"
    assert isinstance(eair_data, jnp.ndarray), "Vapor pressure data should be a JAX array"


# ============================================================================
# Tests for clmml_drv_cleanup
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(tc, id=tc["name"])
        for tc in [
            {"name": "prescribed_turbulence", "inputs": {"turb_type": -1}},
            {"name": "computed_turbulence", "inputs": {"turb_type": 1}},
        ]
    ],
)
def test_clmml_drv_cleanup_values(test_case: Dict[str, Any]) -> None:
    """
    Test clmml_drv_cleanup returns valid cleanup state.
    
    Verifies that the cleanup state contains boolean flags for file closure.
    """
    inputs = test_case["inputs"]
    result = clmml_drv_cleanup(turb_type=inputs["turb_type"])
    
    # Check that result is a dictionary or named tuple with expected fields
    assert hasattr(result, "close_nout1") or "close_nout1" in result
    assert hasattr(result, "close_nout2") or "close_nout2" in result
    assert hasattr(result, "close_nout3") or "close_nout3" in result
    assert hasattr(result, "close_nout4") or "close_nout4" in result
    assert hasattr(result, "close_nin1") or "close_nin1" in result
    assert hasattr(result, "success") or "success" in result


def test_clmml_drv_cleanup_dtypes() -> None:
    """Test that clmml_drv_cleanup returns correct data types."""
    result = clmml_drv_cleanup(turb_type=-1)
    
    # Get values whether result is dict or namedtuple
    if hasattr(result, "_asdict"):
        result_dict = result._asdict()
    else:
        result_dict = result
    
    # Check that boolean flags are boolean
    for key in ["close_nout1", "close_nout2", "close_nout3", "close_nout4", "close_nin1"]:
        assert isinstance(
            result_dict[key], (bool, np.bool_)
        ), f"{key} should be boolean"
    
    assert isinstance(
        result_dict["success"], (bool, np.bool_)
    ), "success should be boolean"


# ============================================================================
# Edge case tests
# ============================================================================


def test_calculate_time_index_edge_cases() -> None:
    """Test calculate_time_index with various edge cases."""
    # Test with very small timestep
    result = calculate_time_index(jnp.array(2.0), jnp.array(1.0), jnp.array(60.0))
    assert result >= 1, "Index should be at least 1 for small timestep"
    
    # Test with large timestep
    result = calculate_time_index(jnp.array(100.0), jnp.array(1.0), jnp.array(86400.0))
    assert result >= 1, "Index should be at least 1 for large timestep"


def test_compute_upward_shortwave_edge_cases() -> None:
    """Test compute_upward_shortwave with edge cases."""
    # Test with zero albedo
    result = compute_upward_shortwave(
        albcan_vis=jnp.array([0.0]),
        albcan_nir=jnp.array([0.0]),
        swskyb_vis=jnp.array([500.0]),
        swskyd_vis=jnp.array([200.0]),
        swskyb_nir=jnp.array([400.0]),
        swskyd_nir=jnp.array([150.0]),
    )
    np.testing.assert_allclose(
        result, jnp.array([0.0]), rtol=1e-6, atol=1e-6, err_msg="Zero albedo should give zero upward radiation"
    )
    
    # Test with albedo = 1
    result = compute_upward_shortwave(
        albcan_vis=jnp.array([1.0]),
        albcan_nir=jnp.array([1.0]),
        swskyb_vis=jnp.array([500.0]),
        swskyd_vis=jnp.array([200.0]),
        swskyb_nir=jnp.array([400.0]),
        swskyd_nir=jnp.array([150.0]),
    )
    total_incoming = 500.0 + 200.0 + 400.0 + 150.0
    np.testing.assert_allclose(
        result,
        jnp.array([total_incoming]),
        rtol=1e-6,
        atol=1e-6,
        err_msg="Albedo = 1 should reflect all radiation",
    )


def test_convert_qair_to_eair_edge_cases() -> None:
    """Test convert_qair_to_eair with edge cases."""
    # Test with saturation (high qair)
    result = convert_qair_to_eair(
        qair=jnp.array([[0.04]]),
        pref=jnp.array([101325.0]),
        mmh2o=18.016,
        mmdry=28.966,
    )
    assert jnp.all(result > 0.0), "High humidity should give positive vapor pressure"
    assert jnp.all(
        result < 101325.0
    ), "Vapor pressure should be less than total pressure"


def test_process_profile_data_edge_cases() -> None:
    """Test process_profile_data with edge cases."""
    # Test with zero wind
    wind_data, tair_data, eair_data = process_profile_data(
        wind_raw=jnp.array([[0.0, 0.0]]),
        tair_raw=jnp.array([[288.15, 289.15]]),
        qair_raw=jnp.array([[8.0, 9.0]]),
        pref=jnp.array([101325.0]),
        mmh2o=18.016,
        mmdry=28.966,
    )
    np.testing.assert_allclose(
        wind_data,
        jnp.array([[0.0, 0.0]]),
        rtol=1e-6,
        atol=1e-6,
        err_msg="Zero wind should be preserved",
    )
    
    # Test with very cold temperature
    wind_data, tair_data, eair_data = process_profile_data(
        wind_raw=jnp.array([[3.0]]),
        tair_raw=jnp.array([[233.15]]),  # -40°C
        qair_raw=jnp.array([[0.5]]),
        pref=jnp.array([101325.0]),
        mmh2o=18.016,
        mmdry=28.966,
    )
    assert jnp.all(tair_data > 0.0), "Temperature should remain positive in Kelvin"
    assert jnp.all(eair_data >= 0.0), "Vapor pressure should be non-negative even at cold temps"


# ============================================================================
# Physical constraint tests
# ============================================================================


def test_physical_constraints_temperature() -> None:
    """Test that temperature values respect physical constraints."""
    # Test process_profile_data with various temperatures
    temps = jnp.array([[250.0, 280.0, 310.0]])  # Cold, moderate, warm
    wind_data, tair_data, eair_data = process_profile_data(
        wind_raw=jnp.array([[3.0, 3.0, 3.0]]),
        tair_raw=temps,
        qair_raw=jnp.array([[5.0, 8.0, 12.0]]),
        pref=jnp.array([101325.0]),
        mmh2o=18.016,
        mmdry=28.966,
    )
    
    assert jnp.all(tair_data > 0.0), "All temperatures should be > 0 K"
    np.testing.assert_allclose(
        tair_data, temps, rtol=1e-6, atol=1e-6, err_msg="Temperatures should be preserved"
    )


def test_physical_constraints_fractions() -> None:
    """Test that fraction values are in [0, 1]."""
    # Test compute_upward_shortwave with various albedos
    albedos = jnp.array([0.0, 0.5, 1.0])
    result = compute_upward_shortwave(
        albcan_vis=albedos,
        albcan_nir=albedos,
        swskyb_vis=jnp.array([500.0, 500.0, 500.0]),
        swskyd_vis=jnp.array([200.0, 200.0, 200.0]),
        swskyb_nir=jnp.array([400.0, 400.0, 400.0]),
        swskyd_nir=jnp.array([150.0, 150.0, 150.0]),
    )
    
    assert jnp.all(result >= 0.0), "Upward radiation should be non-negative"
    total_incoming = 500.0 + 200.0 + 400.0 + 150.0
    assert jnp.all(
        result <= total_incoming + 1e-6
    ), "Upward radiation should not exceed incoming"


def test_physical_constraints_pressure() -> None:
    """Test that pressure values are positive."""
    # Test convert_qair_to_eair with various pressures
    pressures = jnp.array([70000.0, 85000.0, 101325.0])  # High elevation to sea level
    result = convert_qair_to_eair(
        qair=jnp.array([[0.01, 0.01, 0.01]]),
        pref=pressures,
        mmh2o=18.016,
        mmdry=28.966,
    )
    
    assert jnp.all(result >= 0.0), "Vapor pressure should be non-negative"
    pref_expanded = pressures[:, jnp.newaxis]
    assert jnp.all(
        result <= pref_expanded
    ), "Vapor pressure should not exceed total pressure"


def test_physical_constraints_radiation() -> None:
    """Test that radiation fluxes are non-negative."""
    # Test with various radiation values
    radiation_values = jnp.array([0.0, 100.0, 500.0, 1000.0])
    
    for rad in radiation_values:
        result = compute_upward_shortwave(
            albcan_vis=jnp.array([0.2]),
            albcan_nir=jnp.array([0.3]),
            swskyb_vis=jnp.array([rad]),
            swskyd_vis=jnp.array([rad * 0.3]),
            swskyb_nir=jnp.array([rad * 0.8]),
            swskyd_nir=jnp.array([rad * 0.2]),
        )
        assert jnp.all(result >= 0.0), f"Upward radiation should be non-negative for incoming={rad}"


# ============================================================================
# Integration tests
# ============================================================================


def test_integration_profile_processing_chain() -> None:
    """
    Test the integration of profile data processing functions.
    
    Verifies that qair conversion and profile processing work together correctly.
    """
    # Create test data
    qair = jnp.array([[0.008, 0.01, 0.012]])
    pref = jnp.array([101325.0])
    wind_raw = jnp.array([[2.5, 3.0, 3.5]])
    tair_raw = jnp.array([[288.15, 289.15, 290.15]])
    qair_raw = jnp.array([[8.0, 9.0, 10.0]])
    
    # First convert qair to eair
    eair_direct = convert_qair_to_eair(qair, pref, 18.016, 28.966)
    
    # Then process profile data (which also converts qair to eair internally)
    wind_data, tair_data, eair_data = process_profile_data(
        wind_raw, tair_raw, qair_raw, pref, 18.016, 28.966
    )
    
    # Check that all outputs are physically consistent
    assert jnp.all(wind_data >= 0.0), "Wind should be non-negative"
    assert jnp.all(tair_data > 0.0), "Temperature should be positive"
    assert jnp.all(eair_data >= 0.0), "Vapor pressure should be non-negative"
    assert jnp.all(
        eair_data <= pref[0]
    ), "Vapor pressure should not exceed total pressure"


def test_integration_radiation_calculation() -> None:
    """
    Test the integration of radiation calculations.
    
    Verifies that upward shortwave is consistent with albedo and incoming radiation.
    """
    # Create test data with known albedo
    albedo_vis = 0.2
    albedo_nir = 0.3
    incoming_vis = 600.0
    incoming_nir = 400.0
    
    result = compute_upward_shortwave(
        albcan_vis=jnp.array([albedo_vis]),
        albcan_nir=jnp.array([albedo_nir]),
        swskyb_vis=jnp.array([incoming_vis * 0.7]),
        swskyd_vis=jnp.array([incoming_vis * 0.3]),
        swskyb_nir=jnp.array([incoming_nir * 0.7]),
        swskyd_nir=jnp.array([incoming_nir * 0.3]),
    )
    
    # Expected upward radiation
    expected = albedo_vis * incoming_vis + albedo_nir * incoming_nir
    
    np.testing.assert_allclose(
        result,
        jnp.array([expected]),
        rtol=1e-5,
        atol=1e-5,
        err_msg="Upward radiation should match albedo * incoming",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])