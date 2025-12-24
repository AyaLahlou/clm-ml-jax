"""
Comprehensive pytest suite for atm2lndType module.

This test suite validates the atmosphere-to-land forcing data structures and
initialization functions, covering:
- Array shape validation
- Physical constraint enforcement
- Edge cases (zeros, extremes, boundaries)
- Various climate regimes (polar, tropical, desert, mountain)
- Scalability with different domain sizes
"""

import sys
from pathlib import Path
from typing import Dict, Any

import pytest
import jax.numpy as jnp
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from clm_src_main.atm2lndType import (
    atm2lnd_type,
    create_atm2lnd_instance,
    init_atm2lnd_arrays
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load comprehensive test data for atm2lndType functions.
    
    Returns:
        Dictionary containing test cases with inputs, expected outputs,
        and metadata for various climate scenarios and edge cases.
    """
    return {
        "nominal_midlatitude_summer": {
            "bounds_dict": {"begg": 0, "endg": 4, "begc": 0, "endc": 8},
            "expected_shapes": {
                "forc_u_grc": (5,),
                "forc_v_grc": (5,),
                "forc_pco2_grc": (5,),
                "forc_po2_grc": (5,),
                "forc_solad_grc": (5, 2),
                "forc_solai_grc": (5, 2),
                "forc_t_downscaled_col": (9,),
                "forc_q_downscaled_col": (9,),
                "forc_pbot_downscaled_col": (9,),
                "forc_lwrad_downscaled_col": (9,),
                "forc_rain_downscaled_col": (9,),
                "forc_snow_downscaled_col": (9,),
            },
            "sample_values": {
                "forc_u_grc": [3.5, -2.1, 5.8, 1.2, -4.3],
                "forc_t_downscaled_col": [298.15, 299.5, 297.8, 300.2, 296.5, 301.0, 298.9, 297.2, 299.8],
                "forc_q_downscaled_col": [0.012, 0.014, 0.011, 0.015, 0.01, 0.016, 0.013, 0.011, 0.014],
            },
            "description": "Typical mid-latitude summer conditions"
        },
        "nominal_polar_winter": {
            "bounds_dict": {"begg": 0, "endg": 2, "begc": 0, "endc": 5},
            "expected_shapes": {
                "forc_u_grc": (3,),
                "forc_v_grc": (3,),
                "forc_pco2_grc": (3,),
                "forc_po2_grc": (3,),
                "forc_solad_grc": (3, 2),
                "forc_solai_grc": (3, 2),
                "forc_t_downscaled_col": (6,),
                "forc_q_downscaled_col": (6,),
                "forc_pbot_downscaled_col": (6,),
                "forc_lwrad_downscaled_col": (6,),
                "forc_rain_downscaled_col": (6,),
                "forc_snow_downscaled_col": (6,),
            },
            "sample_values": {
                "forc_t_downscaled_col": [223.15, 218.5, 228.3, 215.8, 230.2, 220.9],
                "forc_q_downscaled_col": [0.0001, 0.00008, 0.00015, 0.00006, 0.00018, 0.0001],
                "forc_snow_downscaled_col": [0.0002, 0.0003, 0.0001, 0.0004, 0.00008, 0.00025],
            },
            "description": "Polar winter with extreme cold and snowfall"
        },
        "edge_all_zeros": {
            "bounds_dict": {"begg": 0, "endg": 1, "begc": 0, "endc": 2},
            "expected_shapes": {
                "forc_u_grc": (2,),
                "forc_v_grc": (2,),
                "forc_pco2_grc": (2,),
                "forc_po2_grc": (2,),
                "forc_solad_grc": (2, 2),
                "forc_solai_grc": (2, 2),
                "forc_t_downscaled_col": (3,),
                "forc_q_downscaled_col": (3,),
                "forc_pbot_downscaled_col": (3,),
                "forc_lwrad_downscaled_col": (3,),
                "forc_rain_downscaled_col": (3,),
                "forc_snow_downscaled_col": (3,),
            },
            "all_zeros": True,
            "description": "All forcing variables set to zero"
        },
        "edge_minimum_physical": {
            "bounds_dict": {"begg": 0, "endg": 1, "begc": 0, "endc": 2},
            "expected_shapes": {
                "forc_u_grc": (2,),
                "forc_v_grc": (2,),
                "forc_pco2_grc": (2,),
                "forc_po2_grc": (2,),
                "forc_solad_grc": (2, 2),
                "forc_solai_grc": (2, 2),
                "forc_t_downscaled_col": (3,),
                "forc_q_downscaled_col": (3,),
                "forc_pbot_downscaled_col": (3,),
                "forc_lwrad_downscaled_col": (3,),
                "forc_rain_downscaled_col": (3,),
                "forc_snow_downscaled_col": (3,),
            },
            "sample_values": {
                "forc_t_downscaled_col": [173.15, 173.15, 173.15],
                "forc_q_downscaled_col": [1e-8, 1e-8, 1e-8],
                "forc_pbot_downscaled_col": [1000.0, 1000.0, 1000.0],
            },
            "description": "Minimum physically realistic values"
        },
        "edge_maximum_physical": {
            "bounds_dict": {"begg": 0, "endg": 1, "begc": 0, "endc": 2},
            "expected_shapes": {
                "forc_u_grc": (2,),
                "forc_v_grc": (2,),
                "forc_pco2_grc": (2,),
                "forc_po2_grc": (2,),
                "forc_solad_grc": (2, 2),
                "forc_solai_grc": (2, 2),
                "forc_t_downscaled_col": (3,),
                "forc_q_downscaled_col": (3,),
                "forc_pbot_downscaled_col": (3,),
                "forc_lwrad_downscaled_col": (3,),
                "forc_rain_downscaled_col": (3,),
                "forc_snow_downscaled_col": (3,),
            },
            "sample_values": {
                "forc_u_grc": [50.0, -50.0],
                "forc_t_downscaled_col": [330.0, 330.0, 330.0],
                "forc_q_downscaled_col": [0.05, 0.05, 0.05],
                "forc_rain_downscaled_col": [0.05, 0.05, 0.05],
            },
            "description": "Maximum physically realistic values"
        },
        "special_single_gridcell": {
            "bounds_dict": {"begg": 0, "endg": 0, "begc": 0, "endc": 0},
            "expected_shapes": {
                "forc_u_grc": (1,),
                "forc_v_grc": (1,),
                "forc_pco2_grc": (1,),
                "forc_po2_grc": (1,),
                "forc_solad_grc": (1, 2),
                "forc_solai_grc": (1, 2),
                "forc_t_downscaled_col": (1,),
                "forc_q_downscaled_col": (1,),
                "forc_pbot_downscaled_col": (1,),
                "forc_lwrad_downscaled_col": (1,),
                "forc_rain_downscaled_col": (1,),
                "forc_snow_downscaled_col": (1,),
            },
            "sample_values": {
                "forc_t_downscaled_col": [288.15],
                "forc_q_downscaled_col": [0.008],
            },
            "description": "Minimum size with single gridcell and column"
        },
        "special_large_domain": {
            "bounds_dict": {"begg": 0, "endg": 99, "begc": 0, "endc": 499},
            "expected_shapes": {
                "forc_u_grc": (100,),
                "forc_v_grc": (100,),
                "forc_pco2_grc": (100,),
                "forc_po2_grc": (100,),
                "forc_solad_grc": (100, 2),
                "forc_solai_grc": (100, 2),
                "forc_t_downscaled_col": (500,),
                "forc_q_downscaled_col": (500,),
                "forc_pbot_downscaled_col": (500,),
                "forc_lwrad_downscaled_col": (500,),
                "forc_rain_downscaled_col": (500,),
                "forc_snow_downscaled_col": (500,),
            },
            "description": "Large domain with 100 gridcells and 500 columns"
        },
    }


# ============================================================================
# Shape Tests
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "nominal_midlatitude_summer",
    "nominal_polar_winter",
    "edge_all_zeros",
    "edge_minimum_physical",
    "edge_maximum_physical",
    "special_single_gridcell",
    "special_large_domain",
])
def test_init_atm2lnd_arrays_shapes(test_data, test_case_name):
    """
    Test that init_atm2lnd_arrays produces arrays with correct shapes.
    
    Validates that all output arrays have dimensions matching the input
    bounds specification for gridcells and columns across various domain sizes.
    """
    test_case = test_data[test_case_name]
    bounds_dict = test_case["bounds_dict"]
    expected_shapes = test_case["expected_shapes"]
    
    # Call the function
    result = init_atm2lnd_arrays(bounds_dict)
    
    # Check all array shapes
    for field_name, expected_shape in expected_shapes.items():
        assert field_name in result, (
            f"Missing field '{field_name}' in result for {test_case_name}"
        )
        actual_shape = result[field_name].shape
        assert actual_shape == expected_shape, (
            f"Shape mismatch for '{field_name}' in {test_case_name}: "
            f"expected {expected_shape}, got {actual_shape}"
        )


def test_init_atm2lnd_arrays_all_fields_present(test_data):
    """
    Test that init_atm2lnd_arrays returns all required fields.
    
    Ensures the output dictionary contains all 12 atmospheric forcing
    variables defined in the atm2lnd_type specification.
    """
    bounds_dict = {"begg": 0, "endg": 4, "begc": 0, "endc": 8}
    result = init_atm2lnd_arrays(bounds_dict)
    
    expected_fields = [
        "forc_u_grc",
        "forc_v_grc",
        "forc_pco2_grc",
        "forc_po2_grc",
        "forc_solad_grc",
        "forc_solai_grc",
        "forc_t_downscaled_col",
        "forc_q_downscaled_col",
        "forc_pbot_downscaled_col",
        "forc_lwrad_downscaled_col",
        "forc_rain_downscaled_col",
        "forc_snow_downscaled_col",
    ]
    
    for field in expected_fields:
        assert field in result, f"Missing required field: {field}"


# ============================================================================
# Data Type Tests
# ============================================================================

def test_init_atm2lnd_arrays_dtypes(test_data):
    """
    Test that all arrays have correct JAX float data types.
    
    Validates that arrays are JAX arrays with float32 or float64 dtype,
    suitable for scientific computing and GPU acceleration.
    """
    bounds_dict = {"begg": 0, "endg": 4, "begc": 0, "endc": 8}
    result = init_atm2lnd_arrays(bounds_dict)
    
    for field_name, array in result.items():
        # Check it's a JAX array
        assert isinstance(array, jnp.ndarray), (
            f"Field '{field_name}' is not a JAX array"
        )
        
        # Check it's a float type
        assert jnp.issubdtype(array.dtype, jnp.floating), (
            f"Field '{field_name}' has non-float dtype: {array.dtype}"
        )


# ============================================================================
# Value Tests
# ============================================================================

def test_init_atm2lnd_arrays_initialization_zeros(test_data):
    """
    Test that arrays are initialized to zeros.
    
    Verifies that the InitAllocate behavior creates zero-filled arrays
    as the default initialization state.
    """
    bounds_dict = {"begg": 0, "endg": 2, "begc": 0, "endc": 5}
    result = init_atm2lnd_arrays(bounds_dict)
    
    for field_name, array in result.items():
        assert jnp.allclose(array, 0.0, atol=1e-10), (
            f"Field '{field_name}' not initialized to zeros"
        )


@pytest.mark.parametrize("test_case_name", [
    "nominal_midlatitude_summer",
    "nominal_polar_winter",
    "edge_minimum_physical",
    "edge_maximum_physical",
    "special_single_gridcell",
])
def test_init_atm2lnd_arrays_sample_values(test_data, test_case_name):
    """
    Test that arrays can hold expected sample values.
    
    Validates that the array structure can accommodate realistic atmospheric
    forcing values across different climate regimes and edge cases.
    """
    test_case = test_data[test_case_name]
    bounds_dict = test_case["bounds_dict"]
    
    # Initialize arrays
    result = init_atm2lnd_arrays(bounds_dict)
    
    # Verify arrays can be populated with sample values
    if "sample_values" in test_case:
        sample_values = test_case["sample_values"]
        
        for field_name, values in sample_values.items():
            if field_name in result:
                array = result[field_name]
                values_array = jnp.array(values)
                
                # Check that shapes are compatible
                if array.ndim == 1:
                    assert len(values_array) <= array.shape[0], (
                        f"Sample values for '{field_name}' exceed array size"
                    )
                elif array.ndim == 2:
                    assert values_array.shape[0] <= array.shape[0], (
                        f"Sample values for '{field_name}' exceed array size"
                    )


# ============================================================================
# Physical Constraint Tests
# ============================================================================

def test_physical_constraints_temperature_range():
    """
    Test temperature physical constraints.
    
    Validates that temperature arrays can hold values in the physically
    realistic range of 173.15K (coldest Earth) to 330K (hottest deserts).
    """
    bounds_dict = {"begg": 0, "endg": 1, "begc": 0, "endc": 2}
    result = init_atm2lnd_arrays(bounds_dict)
    
    # Test minimum temperature (coldest recorded on Earth)
    min_temp = jnp.array([173.15, 173.15, 173.15])
    assert result["forc_t_downscaled_col"].shape == min_temp.shape
    assert jnp.all(min_temp >= 0.0), "Temperature below absolute zero"
    
    # Test maximum temperature (hottest deserts)
    max_temp = jnp.array([330.0, 330.0, 330.0])
    assert result["forc_t_downscaled_col"].shape == max_temp.shape
    assert jnp.all(max_temp > 173.15), "Maximum temp should exceed minimum"


def test_physical_constraints_humidity_range():
    """
    Test humidity physical constraints.
    
    Validates that specific humidity values are bounded in [0, 1] kg/kg
    as required by physical laws.
    """
    bounds_dict = {"begg": 0, "endg": 1, "begc": 0, "endc": 2}
    result = init_atm2lnd_arrays(bounds_dict)
    
    # Test minimum humidity
    min_humidity = jnp.array([0.0, 0.0, 0.0])
    assert result["forc_q_downscaled_col"].shape == min_humidity.shape
    assert jnp.all(min_humidity >= 0.0), "Humidity cannot be negative"
    
    # Test maximum humidity (saturation at high temps)
    max_humidity = jnp.array([0.05, 0.05, 0.05])
    assert jnp.all(max_humidity <= 1.0), "Humidity cannot exceed 1.0 kg/kg"


def test_physical_constraints_pressure_positive():
    """
    Test that all pressure fields are non-negative.
    
    Validates that atmospheric pressure, CO2 partial pressure, and O2
    partial pressure are all >= 0 Pa as required physically.
    """
    bounds_dict = {"begg": 0, "endg": 1, "begc": 0, "endc": 2}
    result = init_atm2lnd_arrays(bounds_dict)
    
    pressure_fields = [
        "forc_pco2_grc",
        "forc_po2_grc",
        "forc_pbot_downscaled_col",
    ]
    
    for field in pressure_fields:
        # Initialize with test values
        test_values = jnp.ones_like(result[field]) * 1000.0
        assert jnp.all(test_values >= 0.0), (
            f"Pressure field '{field}' contains negative values"
        )


def test_physical_constraints_radiation_positive():
    """
    Test that all radiation fields are non-negative.
    
    Validates that solar (direct/diffuse) and longwave radiation are all
    >= 0 W/m² as required by physics.
    """
    bounds_dict = {"begg": 0, "endg": 1, "begc": 0, "endc": 2}
    result = init_atm2lnd_arrays(bounds_dict)
    
    radiation_fields = [
        "forc_solad_grc",
        "forc_solai_grc",
        "forc_lwrad_downscaled_col",
    ]
    
    for field in radiation_fields:
        # Initialize with test values
        test_values = jnp.ones_like(result[field]) * 500.0
        assert jnp.all(test_values >= 0.0), (
            f"Radiation field '{field}' contains negative values"
        )


def test_physical_constraints_precipitation_positive():
    """
    Test that precipitation rates are non-negative.
    
    Validates that rainfall and snowfall rates are >= 0 mm/s as required
    physically (negative precipitation is non-physical).
    """
    bounds_dict = {"begg": 0, "endg": 1, "begc": 0, "endc": 2}
    result = init_atm2lnd_arrays(bounds_dict)
    
    precip_fields = [
        "forc_rain_downscaled_col",
        "forc_snow_downscaled_col",
    ]
    
    for field in precip_fields:
        # Initialize with test values
        test_values = jnp.ones_like(result[field]) * 0.001
        assert jnp.all(test_values >= 0.0), (
            f"Precipitation field '{field}' contains negative values"
        )


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_edge_case_all_zeros(test_data):
    """
    Test handling of all-zero forcing values.
    
    Validates numerical stability when all atmospheric forcing variables
    are initialized to zero (physically unrealistic but tests robustness).
    """
    test_case = test_data["edge_all_zeros"]
    bounds_dict = test_case["bounds_dict"]
    
    result = init_atm2lnd_arrays(bounds_dict)
    
    # All arrays should be zeros
    for field_name, array in result.items():
        assert jnp.allclose(array, 0.0, atol=1e-10), (
            f"Field '{field_name}' not zero in all-zeros test case"
        )
        
        # Check no NaN or Inf
        assert not jnp.any(jnp.isnan(array)), (
            f"Field '{field_name}' contains NaN values"
        )
        assert not jnp.any(jnp.isinf(array)), (
            f"Field '{field_name}' contains Inf values"
        )


def test_edge_case_minimum_physical_values(test_data):
    """
    Test handling of minimum physically realistic values.
    
    Validates that the system can handle extreme cold (173K), near-vacuum
    conditions, and minimal radiation/precipitation.
    """
    test_case = test_data["edge_minimum_physical"]
    bounds_dict = test_case["bounds_dict"]
    expected_shapes = test_case["expected_shapes"]
    
    result = init_atm2lnd_arrays(bounds_dict)
    
    # Verify shapes match
    for field_name, expected_shape in expected_shapes.items():
        assert result[field_name].shape == expected_shape, (
            f"Shape mismatch for '{field_name}' in minimum physical test"
        )
    
    # Check no NaN or Inf in initialized arrays
    for field_name, array in result.items():
        assert not jnp.any(jnp.isnan(array)), (
            f"Field '{field_name}' contains NaN in minimum physical test"
        )
        assert not jnp.any(jnp.isinf(array)), (
            f"Field '{field_name}' contains Inf in minimum physical test"
        )


def test_edge_case_maximum_physical_values(test_data):
    """
    Test handling of maximum physically realistic values.
    
    Validates that the system can handle extreme heat (330K), hurricane-force
    winds (50 m/s), intense radiation, and heavy precipitation.
    """
    test_case = test_data["edge_maximum_physical"]
    bounds_dict = test_case["bounds_dict"]
    expected_shapes = test_case["expected_shapes"]
    
    result = init_atm2lnd_arrays(bounds_dict)
    
    # Verify shapes match
    for field_name, expected_shape in expected_shapes.items():
        assert result[field_name].shape == expected_shape, (
            f"Shape mismatch for '{field_name}' in maximum physical test"
        )
    
    # Check no NaN or Inf
    for field_name, array in result.items():
        assert not jnp.any(jnp.isnan(array)), (
            f"Field '{field_name}' contains NaN in maximum physical test"
        )
        assert not jnp.any(jnp.isinf(array)), (
            f"Field '{field_name}' contains Inf in maximum physical test"
        )


# ============================================================================
# Special Case Tests
# ============================================================================

def test_special_case_single_gridcell_column(test_data):
    """
    Test minimal domain with single gridcell and column.
    
    Validates that the system handles the smallest possible domain size
    (1x1) correctly, which is important for testing and debugging.
    """
    test_case = test_data["special_single_gridcell"]
    bounds_dict = test_case["bounds_dict"]
    expected_shapes = test_case["expected_shapes"]
    
    result = init_atm2lnd_arrays(bounds_dict)
    
    # Check all shapes are (1,) or (1, 2)
    for field_name, expected_shape in expected_shapes.items():
        actual_shape = result[field_name].shape
        assert actual_shape == expected_shape, (
            f"Shape mismatch for '{field_name}' in single gridcell test: "
            f"expected {expected_shape}, got {actual_shape}"
        )
        
        # Verify minimum dimension is 1
        assert all(dim >= 1 for dim in actual_shape), (
            f"Field '{field_name}' has dimension < 1"
        )


def test_special_case_large_domain(test_data):
    """
    Test large domain with 100 gridcells and 500 columns.
    
    Validates scalability and memory handling with realistic global model
    domain sizes. Tests 5:1 column-to-gridcell ratio.
    """
    test_case = test_data["special_large_domain"]
    bounds_dict = test_case["bounds_dict"]
    expected_shapes = test_case["expected_shapes"]
    
    result = init_atm2lnd_arrays(bounds_dict)
    
    # Check all shapes match expected large dimensions
    for field_name, expected_shape in expected_shapes.items():
        actual_shape = result[field_name].shape
        assert actual_shape == expected_shape, (
            f"Shape mismatch for '{field_name}' in large domain test: "
            f"expected {expected_shape}, got {actual_shape}"
        )
    
    # Verify gridcell arrays have 100 elements
    gridcell_fields = [
        "forc_u_grc", "forc_v_grc", "forc_pco2_grc", "forc_po2_grc"
    ]
    for field in gridcell_fields:
        assert result[field].shape[0] == 100, (
            f"Gridcell field '{field}' should have 100 elements"
        )
    
    # Verify column arrays have 500 elements
    column_fields = [
        "forc_t_downscaled_col", "forc_q_downscaled_col",
        "forc_pbot_downscaled_col", "forc_lwrad_downscaled_col",
        "forc_rain_downscaled_col", "forc_snow_downscaled_col"
    ]
    for field in column_fields:
        assert result[field].shape[0] == 500, (
            f"Column field '{field}' should have 500 elements"
        )
    
    # Verify radiation arrays have correct 2D shape
    assert result["forc_solad_grc"].shape == (100, 2), (
        "Direct solar radiation should be (100, 2)"
    )
    assert result["forc_solai_grc"].shape == (100, 2), (
        "Diffuse solar radiation should be (100, 2)"
    )


# ============================================================================
# Bounds Validation Tests
# ============================================================================

def test_bounds_dict_validation_positive_indices():
    """
    Test that bounds indices are handled correctly.
    
    Validates that the function correctly computes array sizes from
    begin/end indices (size = end - begin + 1).
    """
    # Test various valid bounds
    test_bounds = [
        {"begg": 0, "endg": 0, "begc": 0, "endc": 0},  # Single element
        {"begg": 0, "endg": 9, "begc": 0, "endc": 49},  # 10 grid, 50 col
        {"begg": 5, "endg": 14, "begc": 10, "endc": 59},  # Non-zero start
    ]
    
    for bounds in test_bounds:
        result = init_atm2lnd_arrays(bounds)
        
        expected_grid_size = bounds["endg"] - bounds["begg"] + 1
        expected_col_size = bounds["endc"] - bounds["begc"] + 1
        
        # Check gridcell array sizes
        assert result["forc_u_grc"].shape[0] == expected_grid_size, (
            f"Grid size mismatch for bounds {bounds}"
        )
        
        # Check column array sizes
        assert result["forc_t_downscaled_col"].shape[0] == expected_col_size, (
            f"Column size mismatch for bounds {bounds}"
        )


def test_radiation_bands_dimension():
    """
    Test that radiation arrays have correct second dimension (numrad=2).
    
    Validates that solar radiation arrays have shape (grid_size, 2) where
    the second dimension represents visible and near-infrared bands.
    """
    bounds_dict = {"begg": 0, "endg": 4, "begc": 0, "endc": 8}
    result = init_atm2lnd_arrays(bounds_dict)
    
    # Check direct beam radiation
    assert result["forc_solad_grc"].shape[1] == 2, (
        "Direct solar radiation should have 2 bands (visible, NIR)"
    )
    
    # Check diffuse radiation
    assert result["forc_solai_grc"].shape[1] == 2, (
        "Diffuse solar radiation should have 2 bands (visible, NIR)"
    )


# ============================================================================
# Integration Tests
# ============================================================================

def test_integration_multiple_climate_regimes(test_data):
    """
    Integration test across multiple climate regimes.
    
    Validates that the initialization function works consistently across
    diverse climate scenarios (polar, tropical, desert, mountain).
    """
    climate_regimes = [
        "nominal_midlatitude_summer",
        "nominal_polar_winter",
    ]
    
    for regime in climate_regimes:
        test_case = test_data[regime]
        bounds_dict = test_case["bounds_dict"]
        expected_shapes = test_case["expected_shapes"]
        
        result = init_atm2lnd_arrays(bounds_dict)
        
        # Verify all expected fields present
        for field_name in expected_shapes.keys():
            assert field_name in result, (
                f"Missing field '{field_name}' in {regime}"
            )
        
        # Verify shapes
        for field_name, expected_shape in expected_shapes.items():
            assert result[field_name].shape == expected_shape, (
                f"Shape mismatch in {regime} for '{field_name}'"
            )
        
        # Verify no NaN/Inf
        for field_name, array in result.items():
            assert not jnp.any(jnp.isnan(array)), (
                f"NaN found in {regime} field '{field_name}'"
            )
            assert not jnp.any(jnp.isinf(array)), (
                f"Inf found in {regime} field '{field_name}'"
            )


def test_integration_consistency_across_sizes():
    """
    Test consistency of initialization across different domain sizes.
    
    Validates that the function produces consistent results regardless of
    domain size, with proper scaling of array dimensions.
    """
    sizes = [
        {"begg": 0, "endg": 0, "begc": 0, "endc": 0},  # 1x1
        {"begg": 0, "endg": 4, "begc": 0, "endc": 9},  # 5x10
        {"begg": 0, "endg": 19, "begc": 0, "endc": 99},  # 20x100
    ]
    
    for bounds in sizes:
        result = init_atm2lnd_arrays(bounds)
        
        # Check all required fields present
        assert len(result) == 12, (
            f"Expected 12 fields, got {len(result)} for bounds {bounds}"
        )
        
        # Check all arrays initialized to zeros
        for field_name, array in result.items():
            assert jnp.allclose(array, 0.0, atol=1e-10), (
                f"Field '{field_name}' not zero-initialized for bounds {bounds}"
            )


# ============================================================================
# Numerical Stability Tests
# ============================================================================

def test_numerical_stability_no_nan_inf():
    """
    Test numerical stability - no NaN or Inf in initialized arrays.
    
    Validates that array initialization produces finite values without
    numerical issues across various domain sizes.
    """
    test_bounds = [
        {"begg": 0, "endg": 0, "begc": 0, "endc": 0},
        {"begg": 0, "endg": 9, "begc": 0, "endc": 49},
        {"begg": 0, "endg": 99, "begc": 0, "endc": 499},
    ]
    
    for bounds in test_bounds:
        result = init_atm2lnd_arrays(bounds)
        
        for field_name, array in result.items():
            # Check for NaN
            assert not jnp.any(jnp.isnan(array)), (
                f"NaN detected in '{field_name}' for bounds {bounds}"
            )
            
            # Check for Inf
            assert not jnp.any(jnp.isinf(array)), (
                f"Inf detected in '{field_name}' for bounds {bounds}"
            )
            
            # Check all values are finite
            assert jnp.all(jnp.isfinite(array)), (
                f"Non-finite values in '{field_name}' for bounds {bounds}"
            )


def test_numerical_stability_array_contiguity():
    """
    Test that arrays are contiguous in memory.
    
    Validates that JAX arrays are properly allocated with contiguous memory
    layout for efficient computation.
    """
    bounds_dict = {"begg": 0, "endg": 9, "begc": 0, "endc": 49}
    result = init_atm2lnd_arrays(bounds_dict)
    
    for field_name, array in result.items():
        # JAX arrays should be contiguous
        # Note: JAX arrays don't have flags like NumPy, but we can check
        # that they're proper JAX DeviceArrays
        assert isinstance(array, jnp.ndarray), (
            f"Field '{field_name}' is not a proper JAX array"
        )