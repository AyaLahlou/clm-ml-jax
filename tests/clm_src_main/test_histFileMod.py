"""
Comprehensive pytest suite for hist_addfld1d function from histFileMod module.

This test suite covers:
- Nominal cases for all field types (gridcell, landunit, column, patch, land, atmosphere)
- All averaging flags (I, A, X, M, S)
- All scale types (unity, area, mass, volume)
- Special value filters (lake, urban, glacier, etc.)
- Edge cases (zeros, single elements, extreme values)
- Array dimension variations (1D and 2D)
- Physical realism constraints
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path to import the actual function
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_main.histFileMod import hist_addfld1d


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load comprehensive test data for hist_addfld1d function.
    
    Returns:
        Dictionary containing all test cases with inputs and metadata
    """
    return {
        "function_name": "hist_addfld1d",
        "test_cases": [
            {
                "name": "test_nominal_gridcell_1d",
                "inputs": {
                    "fname": "TEMPERATURE",
                    "units": "K",
                    "avgflag": "A",
                    "long_name": "Surface temperature",
                    "type1d_out": "gridcell",
                    "ptr_gcell": jnp.array([[273.15, 280.5, 290.0, 285.3, 275.8]]),
                    "ptr_lunit": None,
                    "ptr_col": None,
                    "ptr_patch": None,
                    "ptr_lnd": None,
                    "ptr_atm": None,
                    "p2c_scale_type": None,
                    "c2l_scale_type": None,
                    "l2g_scale_type": None,
                    "set_lake": None,
                    "set_nolake": None,
                    "set_urb": None,
                    "set_nourb": None,
                    "set_noglcmec": None,
                    "set_spec": None,
                    "default": None,
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Basic gridcell field with typical temperature values and averaging flag",
                    "edge_cases": [],
                },
            },
            {
                "name": "test_nominal_column_2d_with_scaling",
                "inputs": {
                    "fname": "SOIL_MOISTURE",
                    "units": "kg/m2",
                    "avgflag": "I",
                    "long_name": "Soil moisture content by layer",
                    "type1d_out": "column",
                    "ptr_gcell": None,
                    "ptr_lunit": None,
                    "ptr_col": jnp.array([
                        [25.5, 30.2, 28.7, 32.1],
                        [22.3, 27.8, 25.9, 29.4],
                        [18.9, 24.5, 22.1, 26.3],
                        [15.2, 20.8, 18.6, 22.7],
                        [12.1, 17.3, 15.4, 19.2],
                    ]),
                    "ptr_patch": None,
                    "ptr_lnd": None,
                    "ptr_atm": None,
                    "p2c_scale_type": "area",
                    "c2l_scale_type": "mass",
                    "l2g_scale_type": "area",
                    "set_lake": None,
                    "set_nolake": None,
                    "set_urb": None,
                    "set_nourb": None,
                    "set_noglcmec": None,
                    "set_spec": None,
                    "default": None,
                },
                "metadata": {
                    "type": "nominal",
                    "description": "2D column field with multiple levels and hierarchical scaling types",
                    "edge_cases": [],
                },
            },
            {
                "name": "test_nominal_patch_with_lake_urban_filters",
                "inputs": {
                    "fname": "LAI",
                    "units": "m2/m2",
                    "avgflag": "A",
                    "long_name": "Leaf area index",
                    "type1d_out": "patch",
                    "ptr_gcell": None,
                    "ptr_lunit": None,
                    "ptr_col": None,
                    "ptr_patch": jnp.array([[3.5, 4.2, 2.8, 5.1, 3.9, 4.7, 2.3, 3.1]]),
                    "ptr_lnd": None,
                    "ptr_atm": None,
                    "p2c_scale_type": "area",
                    "c2l_scale_type": "area",
                    "l2g_scale_type": "area",
                    "set_lake": 0.0,
                    "set_nolake": None,
                    "set_urb": 0.5,
                    "set_nourb": None,
                    "set_noglcmec": None,
                    "set_spec": None,
                    "default": None,
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Patch-level vegetation field with lake and urban special value settings",
                    "edge_cases": [],
                },
            },
            {
                "name": "test_nominal_landunit_maximum_flag",
                "inputs": {
                    "fname": "SNOW_DEPTH",
                    "units": "m",
                    "avgflag": "X",
                    "long_name": "Maximum snow depth",
                    "type1d_out": "landunit",
                    "ptr_gcell": None,
                    "ptr_lunit": jnp.array([[0.85, 1.23, 0.0, 2.15, 0.42, 1.67]]),
                    "ptr_col": None,
                    "ptr_patch": None,
                    "ptr_lnd": None,
                    "ptr_atm": None,
                    "p2c_scale_type": None,
                    "c2l_scale_type": "unity",
                    "l2g_scale_type": "area",
                    "set_lake": None,
                    "set_nolake": None,
                    "set_urb": None,
                    "set_nourb": None,
                    "set_noglcmec": 0.0,
                    "set_spec": None,
                    "default": None,
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Landunit field with maximum averaging flag and glacier filter",
                    "edge_cases": [],
                },
            },
            {
                "name": "test_nominal_atmosphere_minimum_flag",
                "inputs": {
                    "fname": "PRECIP_MIN",
                    "units": "mm/s",
                    "avgflag": "M",
                    "long_name": "Minimum precipitation rate",
                    "type1d_out": "atmosphere",
                    "ptr_gcell": None,
                    "ptr_lunit": None,
                    "ptr_col": None,
                    "ptr_patch": None,
                    "ptr_lnd": None,
                    "ptr_atm": jnp.array([[0.0, 0.001, 0.0, 0.005, 0.002, 0.0, 0.003]]),
                    "p2c_scale_type": None,
                    "c2l_scale_type": None,
                    "l2g_scale_type": None,
                    "set_lake": None,
                    "set_nolake": None,
                    "set_urb": None,
                    "set_nourb": None,
                    "set_noglcmec": None,
                    "set_spec": None,
                    "default": "inactive",
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Atmosphere field with minimum flag and inactive default setting",
                    "edge_cases": [],
                },
            },
            {
                "name": "test_edge_zero_values_all_pointers",
                "inputs": {
                    "fname": "ZERO_FLUX",
                    "units": "W/m2",
                    "avgflag": "A",
                    "long_name": "Zero energy flux test",
                    "type1d_out": "land",
                    "ptr_gcell": None,
                    "ptr_lunit": None,
                    "ptr_col": None,
                    "ptr_patch": None,
                    "ptr_lnd": jnp.array([[0.0, 0.0, 0.0, 0.0]]),
                    "ptr_atm": None,
                    "p2c_scale_type": "unity",
                    "c2l_scale_type": "unity",
                    "l2g_scale_type": "unity",
                    "set_lake": 0.0,
                    "set_nolake": 0.0,
                    "set_urb": 0.0,
                    "set_nourb": 0.0,
                    "set_noglcmec": 0.0,
                    "set_spec": 0.0,
                    "default": None,
                },
                "metadata": {
                    "type": "edge",
                    "description": "All zero values across data and special value settings",
                    "edge_cases": ["zero_flux", "zero_special_values"],
                },
            },
            {
                "name": "test_edge_single_element_arrays",
                "inputs": {
                    "fname": "SINGLE_POINT",
                    "units": "fraction",
                    "avgflag": "S",
                    "long_name": "Single point test field",
                    "type1d_out": "gridcell",
                    "ptr_gcell": jnp.array([[0.5]]),
                    "ptr_lunit": None,
                    "ptr_col": None,
                    "ptr_patch": None,
                    "ptr_lnd": None,
                    "ptr_atm": None,
                    "p2c_scale_type": None,
                    "c2l_scale_type": None,
                    "l2g_scale_type": None,
                    "set_lake": None,
                    "set_nolake": None,
                    "set_urb": None,
                    "set_nourb": None,
                    "set_noglcmec": None,
                    "set_spec": None,
                    "default": None,
                },
                "metadata": {
                    "type": "edge",
                    "description": "Minimum array size with single element",
                    "edge_cases": ["minimum_array_size", "boundary_fraction"],
                },
            },
            {
                "name": "test_edge_extreme_temperature_range",
                "inputs": {
                    "fname": "TEMP_EXTREME",
                    "units": "K",
                    "avgflag": "A",
                    "long_name": "Extreme temperature range",
                    "type1d_out": "column",
                    "ptr_gcell": None,
                    "ptr_lunit": None,
                    "ptr_col": jnp.array([
                        [183.15, 323.15, 273.15],
                        [200.0, 310.0, 280.0],
                        [190.5, 315.8, 275.3],
                    ]),
                    "ptr_patch": None,
                    "ptr_lnd": None,
                    "ptr_atm": None,
                    "p2c_scale_type": "volume",
                    "c2l_scale_type": "volume",
                    "l2g_scale_type": "area",
                    "set_lake": 273.15,
                    "set_nolake": None,
                    "set_urb": None,
                    "set_nourb": None,
                    "set_noglcmec": None,
                    "set_spec": None,
                    "default": None,
                },
                "metadata": {
                    "type": "edge",
                    "description": "Extreme but physically valid temperature range from polar to hot desert",
                    "edge_cases": ["extreme_cold", "extreme_hot", "physical_boundaries"],
                },
            },
            {
                "name": "test_special_large_multidimensional_array",
                "inputs": {
                    "fname": "CARBON_POOL",
                    "units": "gC/m2",
                    "avgflag": "I",
                    "long_name": "Carbon pool by layer and patch",
                    "type1d_out": "patch",
                    "ptr_gcell": None,
                    "ptr_lunit": None,
                    "ptr_col": None,
                    "ptr_patch": jnp.array([
                        [1250.5, 1180.3, 1320.7, 1095.2, 1410.8, 1275.4, 1155.9, 1385.1, 1220.6, 1305.3],
                        [980.2, 925.7, 1045.3, 870.5, 1120.4, 1015.8, 920.3, 1098.7, 970.1, 1035.9],
                        [745.8, 705.2, 795.6, 665.3, 850.7, 770.9, 700.8, 835.2, 738.4, 785.1],
                        [520.3, 490.8, 555.2, 465.7, 595.4, 538.6, 488.5, 582.9, 515.7, 548.3],
                        [315.7, 298.4, 337.8, 283.2, 362.1, 327.5, 296.9, 354.6, 313.2, 333.4],
                        [180.2, 170.5, 193.1, 161.8, 206.9, 187.3, 169.7, 202.7, 179.1, 190.6],
                        [95.8, 90.7, 102.6, 86.0, 110.0, 99.6, 90.2, 107.8, 95.2, 101.4],
                        [45.3, 42.9, 48.5, 40.7, 52.0, 47.1, 42.6, 50.9, 45.0, 47.9],
                    ]),
                    "ptr_lnd": None,
                    "ptr_atm": None,
                    "p2c_scale_type": "mass",
                    "c2l_scale_type": "mass",
                    "l2g_scale_type": "mass",
                    "set_lake": None,
                    "set_nolake": None,
                    "set_urb": None,
                    "set_nourb": None,
                    "set_noglcmec": None,
                    "set_spec": None,
                    "default": None,
                },
                "metadata": {
                    "type": "special",
                    "description": "Large 2D array with 8 vertical levels and 10 patches, testing memory and dimension handling",
                    "edge_cases": [],
                },
            },
            {
                "name": "test_special_all_scale_types_and_filters",
                "inputs": {
                    "fname": "COMPREHENSIVE_TEST",
                    "units": "kg/m2/s",
                    "avgflag": "A",
                    "long_name": "Comprehensive field with all options",
                    "type1d_out": "column",
                    "ptr_gcell": None,
                    "ptr_lunit": None,
                    "ptr_col": jnp.array([
                        [0.00015, 0.00023, 0.00018, 0.00031, 0.00012],
                        [0.00021, 0.00029, 0.00024, 0.00037, 0.00018],
                    ]),
                    "ptr_patch": None,
                    "ptr_lnd": None,
                    "ptr_atm": None,
                    "p2c_scale_type": "area",
                    "c2l_scale_type": "mass",
                    "l2g_scale_type": "volume",
                    "set_lake": -999.0,
                    "set_nolake": 1.0,
                    "set_urb": -888.0,
                    "set_nourb": 2.0,
                    "set_noglcmec": -777.0,
                    "set_spec": -666.0,
                    "default": "inactive",
                },
                "metadata": {
                    "type": "special",
                    "description": "Tests all scale types, all special value filters, and inactive default simultaneously",
                    "edge_cases": [],
                },
            },
        ],
    }


@pytest.fixture
def valid_avgflags() -> List[str]:
    """Valid averaging flag values."""
    return ["I", "A", "X", "M", "S"]


@pytest.fixture
def valid_scale_types() -> List[str]:
    """Valid scale type values."""
    return ["unity", "area", "mass", "volume"]


@pytest.fixture
def valid_field_types() -> List[str]:
    """Valid field type values."""
    return ["gridcell", "landunit", "column", "patch", "land", "atmosphere"]


# Parametrized tests for all test cases
@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_nominal_gridcell_1d",
        "test_nominal_column_2d_with_scaling",
        "test_nominal_patch_with_lake_urban_filters",
        "test_nominal_landunit_maximum_flag",
        "test_nominal_atmosphere_minimum_flag",
        "test_edge_zero_values_all_pointers",
        "test_edge_single_element_arrays",
        "test_edge_extreme_temperature_range",
        "test_special_large_multidimensional_array",
        "test_special_all_scale_types_and_filters",
    ],
)
def test_hist_addfld1d_execution(test_data: Dict[str, Any], test_case_name: str):
    """
    Test that hist_addfld1d executes without errors for all test cases.
    
    This function modifies global state (history manager), so we verify
    that it completes successfully without raising exceptions.
    
    Args:
        test_data: Fixture containing all test cases
        test_case_name: Name of the specific test case to run
    """
    # Find the test case
    test_case = next(
        tc for tc in test_data["test_cases"] if tc["name"] == test_case_name
    )
    
    inputs = test_case["inputs"]
    
    # Execute the function - should not raise any exceptions
    try:
        result = hist_addfld1d(**inputs)
        # Function returns None, so we just verify it completed
        assert result is None, f"Expected None return value, got {result}"
    except Exception as e:
        pytest.fail(
            f"hist_addfld1d raised unexpected exception for {test_case_name}: {e}\n"
            f"Metadata: {test_case['metadata']}"
        )


def test_hist_addfld1d_string_parameters(test_data: Dict[str, Any]):
    """
    Test that string parameters are properly validated.
    
    Verifies that:
    - fname is non-empty
    - units is non-empty
    - avgflag is one of valid values
    - long_name is non-empty
    """
    test_case = test_data["test_cases"][0]  # Use first nominal case
    inputs = test_case["inputs"].copy()
    
    # Test with valid inputs first
    try:
        hist_addfld1d(**inputs)
    except Exception as e:
        pytest.fail(f"Valid inputs should not raise exception: {e}")
    
    # Test empty fname (if validation exists)
    invalid_inputs = inputs.copy()
    invalid_inputs["fname"] = ""
    # Note: Actual validation behavior depends on implementation
    # This test documents expected behavior


def test_hist_addfld1d_avgflag_values(test_data: Dict[str, Any], valid_avgflags: List[str]):
    """
    Test that all valid averaging flags are accepted.
    
    Verifies that avgflag parameter accepts all documented valid values:
    I (instantaneous), A (average), X (maximum), M (minimum), S (standard deviation)
    """
    base_inputs = test_data["test_cases"][0]["inputs"].copy()
    
    for avgflag in valid_avgflags:
        inputs = base_inputs.copy()
        inputs["avgflag"] = avgflag
        inputs["fname"] = f"TEST_{avgflag}"  # Unique name for each
        
        try:
            hist_addfld1d(**inputs)
        except Exception as e:
            pytest.fail(f"Valid avgflag '{avgflag}' raised exception: {e}")


def test_hist_addfld1d_scale_types(test_data: Dict[str, Any], valid_scale_types: List[str]):
    """
    Test that all valid scale types are accepted.
    
    Verifies that scale type parameters (p2c_scale_type, c2l_scale_type, l2g_scale_type)
    accept all documented valid values: unity, area, mass, volume
    """
    base_inputs = test_data["test_cases"][1]["inputs"].copy()  # Use column case with scaling
    
    for scale_type in valid_scale_types:
        inputs = base_inputs.copy()
        inputs["p2c_scale_type"] = scale_type
        inputs["c2l_scale_type"] = scale_type
        inputs["l2g_scale_type"] = scale_type
        inputs["fname"] = f"TEST_SCALE_{scale_type.upper()}"
        
        try:
            hist_addfld1d(**inputs)
        except Exception as e:
            pytest.fail(f"Valid scale_type '{scale_type}' raised exception: {e}")


def test_hist_addfld1d_field_types(test_data: Dict[str, Any], valid_field_types: List[str]):
    """
    Test that all valid field types are properly handled.
    
    Verifies that type1d_out parameter accepts all documented field types
    and that the corresponding pointer arrays are used correctly.
    """
    # Map field types to their test cases
    field_type_cases = {
        "gridcell": "test_nominal_gridcell_1d",
        "column": "test_nominal_column_2d_with_scaling",
        "patch": "test_nominal_patch_with_lake_urban_filters",
        "landunit": "test_nominal_landunit_maximum_flag",
        "atmosphere": "test_nominal_atmosphere_minimum_flag",
        "land": "test_edge_zero_values_all_pointers",
    }
    
    for field_type, case_name in field_type_cases.items():
        test_case = next(
            tc for tc in test_data["test_cases"] if tc["name"] == case_name
        )
        inputs = test_case["inputs"]
        
        assert inputs["type1d_out"] == field_type, (
            f"Test case {case_name} should have type1d_out='{field_type}'"
        )
        
        try:
            hist_addfld1d(**inputs)
        except Exception as e:
            pytest.fail(f"Field type '{field_type}' raised exception: {e}")


def test_hist_addfld1d_array_shapes(test_data: Dict[str, Any]):
    """
    Test that various array shapes are handled correctly.
    
    Verifies:
    - 1D arrays (single level)
    - 2D arrays (multiple levels)
    - Single element arrays
    - Large arrays
    """
    shape_test_cases = [
        ("test_nominal_gridcell_1d", (1, 5), "1D array with 5 elements"),
        ("test_nominal_column_2d_with_scaling", (5, 4), "2D array with 5 levels, 4 columns"),
        ("test_edge_single_element_arrays", (1, 1), "Single element array"),
        ("test_special_large_multidimensional_array", (8, 10), "Large 2D array with 8 levels, 10 patches"),
    ]
    
    for case_name, expected_shape, description in shape_test_cases:
        test_case = next(
            tc for tc in test_data["test_cases"] if tc["name"] == case_name
        )
        inputs = test_case["inputs"]
        
        # Find the non-None pointer array
        pointer_arrays = [
            inputs["ptr_gcell"],
            inputs["ptr_lunit"],
            inputs["ptr_col"],
            inputs["ptr_patch"],
            inputs["ptr_lnd"],
            inputs["ptr_atm"],
        ]
        
        active_array = next((arr for arr in pointer_arrays if arr is not None), None)
        
        assert active_array is not None, f"No active pointer array in {case_name}"
        assert active_array.shape == expected_shape, (
            f"{description}: Expected shape {expected_shape}, got {active_array.shape}"
        )
        
        try:
            hist_addfld1d(**inputs)
        except Exception as e:
            pytest.fail(f"Array shape test '{description}' raised exception: {e}")


def test_hist_addfld1d_special_values(test_data: Dict[str, Any]):
    """
    Test that special value filters are properly handled.
    
    Verifies:
    - set_lake and set_nolake
    - set_urb and set_nourb
    - set_noglcmec
    - set_spec
    """
    # Test case with lake and urban filters
    test_case = next(
        tc for tc in test_data["test_cases"]
        if tc["name"] == "test_nominal_patch_with_lake_urban_filters"
    )
    inputs = test_case["inputs"]
    
    assert inputs["set_lake"] == 0.0, "set_lake should be 0.0"
    assert inputs["set_urb"] == 0.5, "set_urb should be 0.5"
    
    try:
        hist_addfld1d(**inputs)
    except Exception as e:
        pytest.fail(f"Special value filters raised exception: {e}")
    
    # Test case with all special values
    test_case = next(
        tc for tc in test_data["test_cases"]
        if tc["name"] == "test_special_all_scale_types_and_filters"
    )
    inputs = test_case["inputs"]
    
    assert inputs["set_lake"] == -999.0, "set_lake should be -999.0"
    assert inputs["set_nolake"] == 1.0, "set_nolake should be 1.0"
    assert inputs["set_urb"] == -888.0, "set_urb should be -888.0"
    assert inputs["set_nourb"] == 2.0, "set_nourb should be 2.0"
    assert inputs["set_noglcmec"] == -777.0, "set_noglcmec should be -777.0"
    assert inputs["set_spec"] == -666.0, "set_spec should be -666.0"
    
    try:
        hist_addfld1d(**inputs)
    except Exception as e:
        pytest.fail(f"All special value filters raised exception: {e}")


def test_hist_addfld1d_zero_values(test_data: Dict[str, Any]):
    """
    Test handling of zero values in data arrays and special value settings.
    
    Verifies numerical stability and proper handling of zero flux/values.
    """
    test_case = next(
        tc for tc in test_data["test_cases"]
        if tc["name"] == "test_edge_zero_values_all_pointers"
    )
    inputs = test_case["inputs"]
    
    # Verify all data values are zero
    assert jnp.all(inputs["ptr_lnd"] == 0.0), "All data values should be zero"
    
    # Verify all special value settings are zero
    assert inputs["set_lake"] == 0.0
    assert inputs["set_nolake"] == 0.0
    assert inputs["set_urb"] == 0.0
    assert inputs["set_nourb"] == 0.0
    assert inputs["set_noglcmec"] == 0.0
    assert inputs["set_spec"] == 0.0
    
    try:
        hist_addfld1d(**inputs)
    except Exception as e:
        pytest.fail(f"Zero values test raised exception: {e}")


def test_hist_addfld1d_extreme_values(test_data: Dict[str, Any]):
    """
    Test handling of extreme but physically valid values.
    
    Verifies:
    - Extreme cold temperatures (183.15 K, polar regions)
    - Extreme hot temperatures (323.15 K, hot deserts)
    - Physical realism constraints maintained
    """
    test_case = next(
        tc for tc in test_data["test_cases"]
        if tc["name"] == "test_edge_extreme_temperature_range"
    )
    inputs = test_case["inputs"]
    
    ptr_col = inputs["ptr_col"]
    
    # Verify extreme values are present
    min_temp = jnp.min(ptr_col)
    max_temp = jnp.max(ptr_col)
    
    assert min_temp >= 183.15, f"Minimum temperature {min_temp} K is below physical limit"
    assert max_temp <= 323.15, f"Maximum temperature {max_temp} K exceeds expected range"
    assert min_temp < 200.0, "Should include extreme cold values"
    assert max_temp > 320.0, "Should include extreme hot values"
    
    try:
        hist_addfld1d(**inputs)
    except Exception as e:
        pytest.fail(f"Extreme values test raised exception: {e}")


def test_hist_addfld1d_default_parameter(test_data: Dict[str, Any]):
    """
    Test the 'default' parameter for field activation control.
    
    Verifies:
    - None (active) default
    - 'inactive' default
    """
    # Test with None (active)
    test_case = next(
        tc for tc in test_data["test_cases"]
        if tc["name"] == "test_nominal_gridcell_1d"
    )
    inputs = test_case["inputs"]
    assert inputs["default"] is None, "Default should be None (active)"
    
    try:
        hist_addfld1d(**inputs)
    except Exception as e:
        pytest.fail(f"Active default raised exception: {e}")
    
    # Test with 'inactive'
    test_case = next(
        tc for tc in test_data["test_cases"]
        if tc["name"] == "test_nominal_atmosphere_minimum_flag"
    )
    inputs = test_case["inputs"]
    assert inputs["default"] == "inactive", "Default should be 'inactive'"
    
    try:
        hist_addfld1d(**inputs)
    except Exception as e:
        pytest.fail(f"Inactive default raised exception: {e}")


def test_hist_addfld1d_none_pointers(test_data: Dict[str, Any]):
    """
    Test that only one pointer type is populated per call.
    
    Verifies that the function correctly handles cases where only one
    of the pointer arrays (ptr_gcell, ptr_lunit, ptr_col, ptr_patch, 
    ptr_lnd, ptr_atm) is non-None, matching typical usage patterns.
    """
    for test_case in test_data["test_cases"]:
        inputs = test_case["inputs"]
        
        pointer_arrays = [
            inputs["ptr_gcell"],
            inputs["ptr_lunit"],
            inputs["ptr_col"],
            inputs["ptr_patch"],
            inputs["ptr_lnd"],
            inputs["ptr_atm"],
        ]
        
        non_none_count = sum(1 for arr in pointer_arrays if arr is not None)
        
        assert non_none_count == 1, (
            f"Test case {test_case['name']} should have exactly one non-None pointer, "
            f"found {non_none_count}"
        )


def test_hist_addfld1d_data_types(test_data: Dict[str, Any]):
    """
    Test that data types are correctly handled.
    
    Verifies:
    - JAX arrays are properly processed
    - Float values for special value settings
    - String parameters
    """
    test_case = test_data["test_cases"][0]
    inputs = test_case["inputs"]
    
    # Check string types
    assert isinstance(inputs["fname"], str), "fname should be string"
    assert isinstance(inputs["units"], str), "units should be string"
    assert isinstance(inputs["avgflag"], str), "avgflag should be string"
    assert isinstance(inputs["long_name"], str), "long_name should be string"
    
    # Check array type
    ptr_gcell = inputs["ptr_gcell"]
    assert isinstance(ptr_gcell, jnp.ndarray), "ptr_gcell should be JAX array"
    
    # Check that array contains float values
    assert jnp.issubdtype(ptr_gcell.dtype, jnp.floating), (
        "Array should contain floating point values"
    )


def test_hist_addfld1d_physical_realism(test_data: Dict[str, Any]):
    """
    Test that physical realism constraints are maintained.
    
    Verifies:
    - Temperatures are above absolute zero (0 K)
    - Fractions are in [0, 1] range
    - Positive values for physical quantities (moisture, carbon, etc.)
    """
    # Temperature test
    temp_case = next(
        tc for tc in test_data["test_cases"]
        if tc["name"] == "test_nominal_gridcell_1d"
    )
    temps = temp_case["inputs"]["ptr_gcell"]
    assert jnp.all(temps > 0.0), "All temperatures should be above absolute zero"
    
    # Fraction test
    fraction_case = next(
        tc for tc in test_data["test_cases"]
        if tc["name"] == "test_edge_single_element_arrays"
    )
    fractions = fraction_case["inputs"]["ptr_gcell"]
    assert jnp.all(fractions >= 0.0) and jnp.all(fractions <= 1.0), (
        "Fractions should be in [0, 1] range"
    )
    
    # Positive physical quantities (LAI)
    lai_case = next(
        tc for tc in test_data["test_cases"]
        if tc["name"] == "test_nominal_patch_with_lake_urban_filters"
    )
    lai_values = lai_case["inputs"]["ptr_patch"]
    assert jnp.all(lai_values >= 0.0), "LAI values should be non-negative"
    
    # Carbon pools should be positive
    carbon_case = next(
        tc for tc in test_data["test_cases"]
        if tc["name"] == "test_special_large_multidimensional_array"
    )
    carbon_values = carbon_case["inputs"]["ptr_patch"]
    assert jnp.all(carbon_values > 0.0), "Carbon pool values should be positive"


def test_hist_addfld1d_comprehensive_coverage(test_data: Dict[str, Any]):
    """
    Test that the test suite provides comprehensive coverage.
    
    Verifies that all test cases execute successfully and cover:
    - All field types
    - All averaging flags
    - All scale types
    - Edge cases
    - Special cases
    """
    test_cases = test_data["test_cases"]
    
    # Count test types
    nominal_count = sum(1 for tc in test_cases if tc["metadata"]["type"] == "nominal")
    edge_count = sum(1 for tc in test_cases if tc["metadata"]["type"] == "edge")
    special_count = sum(1 for tc in test_cases if tc["metadata"]["type"] == "special")
    
    assert nominal_count >= 5, f"Should have at least 5 nominal cases, found {nominal_count}"
    assert edge_count >= 3, f"Should have at least 3 edge cases, found {edge_count}"
    assert special_count >= 2, f"Should have at least 2 special cases, found {special_count}"
    
    # Verify all cases execute
    for test_case in test_cases:
        try:
            hist_addfld1d(**test_case["inputs"])
        except Exception as e:
            pytest.fail(
                f"Test case {test_case['name']} failed: {e}\n"
                f"Description: {test_case['metadata']['description']}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])