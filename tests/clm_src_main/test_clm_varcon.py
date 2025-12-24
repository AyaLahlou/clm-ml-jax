"""
Comprehensive pytest suite for clm_varcon module.

This module tests the CLM variable constants and conversion functions including:
- Physical constants retrieval (get_jax_constants, get_constant)
- Special value detection (is_special_value, is_special_int_value)
- Temperature conversions (celsius_to_kelvin, kelvin_to_celsius)
- Stefan-Boltzmann radiation flux calculations

Tests cover:
- Nominal/typical cases
- Edge cases (absolute zero, extreme temperatures, tolerance boundaries)
- Physical realism and constant relationships
- Array dimension variations
- Numerical precision and roundtrip conversions
- Error handling for invalid inputs
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_main.clm_varcon import (
    celsius_to_kelvin,
    get_constant,
    get_jax_constants,
    is_special_int_value,
    is_special_value,
    kelvin_to_celsius,
    stefan_boltzmann_flux,
)


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load test data for clm_varcon module tests.
    
    Returns:
        Dictionary containing test cases with inputs and expected outputs
    """
    return {
        "constants": {
            "rpi": 3.141592654,
            "tfrz": 273.15,
            "sb": 5.67e-08,
            "grav": 9.80665,
            "vkc": 0.4,
            "denh2o": 1000.0,
            "denice": 917.0,
            "tkwat": 0.57,
            "tkice": 2.29,
            "tkair": 0.023,
            "hfus": 333700.0,
            "hvap": 2501000.0,
            "hsub": 2834700.0,
            "cpice": 2117.27,
            "cpliq": 4188.0,
            "thk_bedrock": 3.0,
            "csol_bedrock": 2000000.0,
            "zmin_bedrock": 0.4,
            "spval": 1e36,
            "ispval": -9999,
        },
        "expected_keys": [
            "rpi", "tfrz", "sb", "grav", "vkc", "denh2o", "denice",
            "tkwat", "tkice", "tkair", "hfus", "hvap", "hsub",
            "cpice", "cpliq", "thk_bedrock", "csol_bedrock",
            "zmin_bedrock", "spval", "ispval"
        ],
    }


class TestGetJaxConstants:
    """Test suite for get_jax_constants function."""
    
    def test_get_jax_constants_structure(self, test_data):
        """
        Verify get_jax_constants returns all expected constants with correct structure.
        
        Tests that:
        - All expected constant keys are present
        - No extra keys are present
        - Dictionary structure is correct
        """
        constants = get_jax_constants()
        
        # Check all expected keys are present
        expected_keys = set(test_data["expected_keys"])
        actual_keys = set(constants.keys())
        
        assert actual_keys == expected_keys, (
            f"Constant keys mismatch. Missing: {expected_keys - actual_keys}, "
            f"Extra: {actual_keys - expected_keys}"
        )
    
    def test_get_jax_constants_dtypes(self, test_data):
        """
        Verify constants have appropriate JAX array dtypes.
        
        Tests that:
        - Float constants are float32
        - Integer constants (ispval) are int32
        - All values are JAX arrays
        """
        constants = get_jax_constants()
        
        # Check float constants
        float_constants = [k for k in constants.keys() if k != "ispval"]
        for key in float_constants:
            assert isinstance(constants[key], jnp.ndarray), (
                f"Constant '{key}' should be JAX array"
            )
            assert constants[key].dtype == jnp.float32, (
                f"Float constant '{key}' should be float32, got {constants[key].dtype}"
            )
        
        # Check integer constant
        assert constants["ispval"].dtype == jnp.int32, (
            f"Integer constant 'ispval' should be int32, got {constants['ispval'].dtype}"
        )
    
    def test_get_jax_constants_values(self, test_data):
        """
        Verify constants have correct numerical values.
        
        Tests that constant values match expected physical constants.
        """
        constants = get_jax_constants()
        expected = test_data["constants"]
        
        for key, expected_value in expected.items():
            actual_value = float(constants[key])
            assert np.isclose(actual_value, expected_value, rtol=1e-6, atol=1e-10), (
                f"Constant '{key}' value mismatch: expected {expected_value}, "
                f"got {actual_value}"
            )


class TestGetConstant:
    """Test suite for get_constant function."""
    
    @pytest.mark.parametrize("const_name,expected_value", [
        ("tfrz", 273.15),
        ("grav", 9.80665),
        ("sb", 5.67e-08),
        ("denh2o", 1000.0),
        ("ispval", -9999),
    ])
    def test_get_constant_as_python_scalar(self, const_name, expected_value):
        """
        Test retrieving constants as Python scalars.
        
        Verifies that constants can be retrieved as native Python types
        (float or int) when as_jax=False.
        """
        value = get_constant(const_name, as_jax=False)
        
        # Check type
        if const_name == "ispval":
            assert isinstance(value, (int, np.integer)), (
                f"Constant '{const_name}' should be integer type"
            )
        else:
            assert isinstance(value, (float, np.floating)), (
                f"Constant '{const_name}' should be float type"
            )
        
        # Check value
        assert np.isclose(value, expected_value, rtol=1e-6, atol=1e-10), (
            f"Constant '{const_name}' value mismatch: expected {expected_value}, "
            f"got {value}"
        )
    
    @pytest.mark.parametrize("const_name,expected_value", [
        ("tfrz", 273.15),
        ("grav", 9.80665),
        ("sb", 5.67e-08),
        ("denh2o", 1000.0),
        ("ispval", -9999),
    ])
    def test_get_constant_as_jax_array(self, const_name, expected_value):
        """
        Test retrieving constants as JAX arrays.
        
        Verifies that constants can be retrieved as JAX arrays with
        appropriate dtypes when as_jax=True.
        """
        value = get_constant(const_name, as_jax=True)
        
        # Check it's a JAX array
        assert isinstance(value, jnp.ndarray), (
            f"Constant '{const_name}' should be JAX array when as_jax=True"
        )
        
        # Check dtype
        if const_name == "ispval":
            assert value.dtype == jnp.int32, (
                f"Integer constant '{const_name}' should be int32"
            )
        else:
            assert value.dtype == jnp.float32, (
                f"Float constant '{const_name}' should be float32"
            )
        
        # Check value
        assert np.isclose(float(value), expected_value, rtol=1e-6, atol=1e-10), (
            f"Constant '{const_name}' value mismatch: expected {expected_value}, "
            f"got {float(value)}"
        )
    
    def test_get_constant_invalid_name(self):
        """
        Test error handling for invalid constant name.
        
        Verifies that requesting a non-existent constant raises KeyError.
        """
        with pytest.raises(KeyError):
            get_constant("nonexistent_constant", as_jax=False)


class TestIsSpecialValue:
    """Test suite for is_special_value function."""
    
    def test_is_special_value_exact_match(self):
        """
        Test exact match with special value (spval = 1e36).
        
        Verifies that the exact special value is correctly identified.
        """
        result = is_special_value(1e36, tolerance=1e-10)
        assert result is True, "Exact special value should be detected"
    
    @pytest.mark.parametrize("value,tolerance,expected", [
        (1.0000000001e36, 1e-5, True),
        (9.9999999999e35, 1e-5, True),
        (1.001e36, 1e-5, False),
    ])
    def test_is_special_value_within_tolerance(self, value, tolerance, expected):
        """
        Test special value detection within tolerance.
        
        Verifies that values within tolerance of spval are correctly identified.
        """
        result = is_special_value(value, tolerance=tolerance)
        assert result == expected, (
            f"Value {value} with tolerance {tolerance} should be {expected}"
        )
    
    @pytest.mark.parametrize("value,expected", [
        (-1e36, False),
        (0.0, False),
        (100.5, False),
        (1e35, False),
        (1e37, False),
    ])
    def test_is_special_value_non_special(self, value, expected):
        """
        Test non-special values including negative, zero, and normal values.
        
        Verifies that regular values are not incorrectly identified as special.
        """
        result = is_special_value(value, tolerance=1e-10)
        assert result == expected, (
            f"Value {value} should not be detected as special"
        )
    
    @pytest.mark.parametrize("value,tolerance,expected", [
        (1e36, 0.0, True),
        (1e36, 1e30, True),
        (5e35, 6e35, True),
    ])
    def test_is_special_value_extreme_tolerance(self, value, tolerance, expected):
        """
        Test special value detection with extreme tolerance values.
        
        Tests edge cases with zero tolerance and very large tolerance.
        """
        result = is_special_value(value, tolerance=tolerance)
        assert result == expected, (
            f"Value {value} with extreme tolerance {tolerance} should be {expected}"
        )


class TestIsSpecialIntValue:
    """Test suite for is_special_int_value function."""
    
    @pytest.mark.parametrize("value,expected", [
        (-9999, True),
        (0, False),
        (100, False),
        (-10000, False),
        (-9998, False),
        (9999, False),
    ])
    def test_is_special_int_value(self, value, expected):
        """
        Test special integer value detection (ispval = -9999).
        
        Verifies that only the exact special integer value is detected.
        """
        result = is_special_int_value(value)
        assert result == expected, (
            f"Integer value {value} should {'be' if expected else 'not be'} "
            f"detected as special"
        )


class TestCelsiusToKelvin:
    """Test suite for celsius_to_kelvin function."""
    
    def test_celsius_to_kelvin_nominal(self):
        """
        Test temperature conversion from Celsius to Kelvin with typical values.
        
        Tests common temperature ranges including freezing point, room temperature,
        and boiling point of water.
        """
        celsius = jnp.array([[0.0, 25.0, 100.0], [-40.0, 15.5, 37.0]])
        expected = jnp.array([[273.15, 298.15, 373.15], [233.15, 288.65, 310.15]])
        
        result = celsius_to_kelvin(celsius)
        
        assert result.shape == expected.shape, (
            f"Shape mismatch: expected {expected.shape}, got {result.shape}"
        )
        assert jnp.allclose(result, expected, rtol=1e-6, atol=1e-6), (
            f"Value mismatch in celsius_to_kelvin conversion"
        )
    
    def test_celsius_to_kelvin_edge_cases(self):
        """
        Test absolute zero boundary and near-zero Celsius values.
        
        Tests the physical lower limit (absolute zero at -273.15°C) and
        values very close to 0°C.
        """
        celsius = jnp.array([[-273.15, -273.14, -200.0], [0.0, 1e-10, -1e-10]])
        expected = jnp.array([[0.0, 0.01, 73.15], [273.15, 273.15, 273.15]])
        
        result = celsius_to_kelvin(celsius)
        
        assert result.shape == expected.shape, (
            f"Shape mismatch: expected {expected.shape}, got {result.shape}"
        )
        assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-5), (
            f"Value mismatch in celsius_to_kelvin edge cases"
        )
    
    def test_celsius_to_kelvin_shapes(self):
        """
        Test that celsius_to_kelvin preserves array shapes.
        
        Verifies shape preservation for various input dimensions.
        """
        test_shapes = [(5,), (3, 4), (2, 3, 4), (2, 2, 2, 2)]
        
        for shape in test_shapes:
            celsius = jnp.ones(shape) * 25.0
            result = celsius_to_kelvin(celsius)
            
            assert result.shape == shape, (
                f"Shape not preserved: input {shape}, output {result.shape}"
            )
    
    def test_celsius_to_kelvin_dtypes(self):
        """
        Test that celsius_to_kelvin maintains appropriate dtypes.
        
        Verifies that output dtype matches input dtype.
        """
        celsius_f32 = jnp.array([0.0, 25.0], dtype=jnp.float32)
        result_f32 = celsius_to_kelvin(celsius_f32)
        
        assert result_f32.dtype == jnp.float32, (
            f"Expected float32 output, got {result_f32.dtype}"
        )


class TestKelvinToCelsius:
    """Test suite for kelvin_to_celsius function."""
    
    def test_kelvin_to_celsius_nominal(self):
        """
        Test temperature conversion from Kelvin to Celsius with typical values.
        
        Tests common temperature ranges in Kelvin scale.
        """
        kelvin = jnp.array([[273.15, 298.15, 373.15], [233.15, 288.65, 310.15]])
        expected = jnp.array([[0.0, 25.0, 100.0], [-40.0, 15.5, 37.0]])
        
        result = kelvin_to_celsius(kelvin)
        
        assert result.shape == expected.shape, (
            f"Shape mismatch: expected {expected.shape}, got {result.shape}"
        )
        assert jnp.allclose(result, expected, rtol=1e-6, atol=1e-6), (
            f"Value mismatch in kelvin_to_celsius conversion"
        )
    
    def test_kelvin_to_celsius_edge_cases(self):
        """
        Test absolute zero, near-zero Kelvin, and extreme temperatures.
        
        Tests physical limits including absolute zero and solar surface temperature.
        """
        kelvin = jnp.array([[0.0, 1e-10, 1e-5], [273.15, 1000.0, 5778.0]])
        expected = jnp.array([[-273.15, -273.15, -273.15], [0.0, 726.85, 5504.85]])
        
        result = kelvin_to_celsius(kelvin)
        
        assert result.shape == expected.shape, (
            f"Shape mismatch: expected {expected.shape}, got {result.shape}"
        )
        assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-5), (
            f"Value mismatch in kelvin_to_celsius edge cases"
        )
    
    def test_kelvin_to_celsius_shapes(self):
        """
        Test that kelvin_to_celsius preserves array shapes.
        
        Verifies shape preservation for various input dimensions.
        """
        test_shapes = [(5,), (3, 4), (2, 3, 4), (2, 2, 2, 2)]
        
        for shape in test_shapes:
            kelvin = jnp.ones(shape) * 298.15
            result = kelvin_to_celsius(kelvin)
            
            assert result.shape == shape, (
                f"Shape not preserved: input {shape}, output {result.shape}"
            )
    
    def test_kelvin_to_celsius_dtypes(self):
        """
        Test that kelvin_to_celsius maintains appropriate dtypes.
        
        Verifies that output dtype matches input dtype.
        """
        kelvin_f32 = jnp.array([273.15, 298.15], dtype=jnp.float32)
        result_f32 = kelvin_to_celsius(kelvin_f32)
        
        assert result_f32.dtype == jnp.float32, (
            f"Expected float32 output, got {result_f32.dtype}"
        )


class TestTemperatureConversionRoundtrip:
    """Test suite for temperature conversion roundtrip accuracy."""
    
    @pytest.mark.parametrize("celsius_original", [
        -50.0, 0.0, 25.0, 100.0, 500.0, -273.15, 15.5, -40.0
    ])
    def test_temperature_conversion_roundtrip(self, celsius_original):
        """
        Test roundtrip conversion: Celsius -> Kelvin -> Celsius maintains precision.
        
        Verifies that converting from Celsius to Kelvin and back preserves
        the original value within numerical precision limits.
        """
        celsius_input = jnp.array([celsius_original])
        kelvin = celsius_to_kelvin(celsius_input)
        celsius_output = kelvin_to_celsius(kelvin)
        
        assert jnp.allclose(celsius_output, celsius_input, rtol=1e-6, atol=1e-6), (
            f"Roundtrip conversion failed: {celsius_original} -> "
            f"{float(kelvin)} -> {float(celsius_output)}"
        )
    
    def test_temperature_conversion_roundtrip_array(self):
        """
        Test roundtrip conversion with arrays of various temperatures.
        
        Verifies numerical precision is maintained for array operations.
        """
        celsius_original = jnp.array([[-50.0, 0.0, 25.0, 100.0, 500.0]])
        kelvin = celsius_to_kelvin(celsius_original)
        celsius_roundtrip = kelvin_to_celsius(kelvin)
        
        max_error = jnp.max(jnp.abs(celsius_roundtrip - celsius_original))
        
        assert max_error < 1e-5, (
            f"Roundtrip conversion error too large: {float(max_error)}"
        )
        assert jnp.allclose(celsius_roundtrip, celsius_original, rtol=1e-6, atol=1e-6), (
            f"Roundtrip conversion failed for array"
        )


class TestStefanBoltzmannFlux:
    """Test suite for stefan_boltzmann_flux function."""
    
    def test_stefan_boltzmann_flux_nominal(self):
        """
        Test Stefan-Boltzmann radiation flux for typical Earth temperatures.
        
        Tests flux calculations for common atmospheric and surface temperatures.
        """
        temperature = jnp.array([[273.15, 288.15, 300.0], [250.0, 273.15, 310.0]])
        
        # Calculate expected flux: σT^4 where σ = 5.67e-8
        sb_constant = 5.67e-8
        expected = sb_constant * temperature ** 4
        
        result = stefan_boltzmann_flux(temperature)
        
        assert result.shape == expected.shape, (
            f"Shape mismatch: expected {expected.shape}, got {result.shape}"
        )
        assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-3), (
            f"Stefan-Boltzmann flux calculation mismatch"
        )
    
    def test_stefan_boltzmann_flux_edge_cases(self):
        """
        Test Stefan-Boltzmann flux at extreme temperatures.
        
        Tests near-zero temperatures, solar surface temperature (5778K),
        and intermediate values.
        """
        temperature = jnp.array([[1e-10, 1.0, 10.0], [5778.0, 1000.0, 100.0]])
        
        sb_constant = 5.67e-8
        expected = sb_constant * temperature ** 4
        
        result = stefan_boltzmann_flux(temperature)
        
        assert result.shape == expected.shape, (
            f"Shape mismatch: expected {expected.shape}, got {result.shape}"
        )
        
        # For extreme values, use relative tolerance
        assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-10), (
            f"Stefan-Boltzmann flux calculation mismatch for edge cases"
        )
    
    def test_stefan_boltzmann_flux_physical_limits(self):
        """
        Test Stefan-Boltzmann flux across cosmic microwave background to stellar temperatures.
        
        Tests physically meaningful temperature ranges from CMB (2.725K) to
        hot stellar temperatures (10000K).
        """
        # CMB temperature, room temperature, solar surface, hot star
        temperature = jnp.array([[2.725, 3.0, 10.0], [273.15, 5778.0, 10000.0]])
        
        sb_constant = 5.67e-8
        expected = sb_constant * temperature ** 4
        
        result = stefan_boltzmann_flux(temperature)
        
        # Verify physical reasonableness
        assert jnp.all(result >= 0), "Flux should be non-negative"
        assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-10), (
            f"Stefan-Boltzmann flux calculation mismatch for physical limits"
        )
        
        # Check order of magnitude for CMB
        cmb_flux = result[0, 0]
        assert 1e-7 < cmb_flux < 1e-5, (
            f"CMB flux out of expected range: {float(cmb_flux)}"
        )
        
        # Check order of magnitude for solar surface
        solar_flux = result[1, 1]
        assert 1e7 < solar_flux < 1e8, (
            f"Solar surface flux out of expected range: {float(solar_flux)}"
        )
    
    def test_stefan_boltzmann_flux_shapes(self):
        """
        Test that stefan_boltzmann_flux preserves array shapes.
        
        Verifies shape preservation for various input dimensions.
        """
        test_shapes = [(5,), (3, 4), (2, 3, 4), (2, 2, 2, 2)]
        
        for shape in test_shapes:
            temperature = jnp.ones(shape) * 300.0
            result = stefan_boltzmann_flux(temperature)
            
            assert result.shape == shape, (
                f"Shape not preserved: input {shape}, output {result.shape}"
            )
    
    def test_stefan_boltzmann_flux_dtypes(self):
        """
        Test that stefan_boltzmann_flux maintains appropriate dtypes.
        
        Verifies that output dtype matches input dtype.
        """
        temperature_f32 = jnp.array([273.15, 300.0], dtype=jnp.float32)
        result_f32 = stefan_boltzmann_flux(temperature_f32)
        
        assert result_f32.dtype == jnp.float32, (
            f"Expected float32 output, got {result_f32.dtype}"
        )


class TestMultidimensionalArrays:
    """Test suite for multidimensional array handling."""
    
    def test_multidimensional_celsius_to_kelvin(self):
        """
        Test temperature conversions with 3D arrays (e.g., spatial + temporal data).
        
        Simulates real-world scenarios with spatial grids over time.
        """
        celsius_3d = jnp.array([[[[0.0, 10.0], [20.0, 30.0]], 
                                  [[5.0, 15.0], [25.0, 35.0]]]])
        
        expected_shape = (1, 2, 2, 2)
        
        result = celsius_to_kelvin(celsius_3d)
        
        assert result.shape == expected_shape, (
            f"3D array shape not preserved: expected {expected_shape}, "
            f"got {result.shape}"
        )
        
        # Verify conversion is correct
        expected_kelvin = celsius_3d + 273.15
        assert jnp.allclose(result, expected_kelvin, rtol=1e-6, atol=1e-6), (
            f"3D celsius_to_kelvin conversion incorrect"
        )
    
    def test_multidimensional_kelvin_to_celsius(self):
        """
        Test Kelvin to Celsius conversion with 3D arrays.
        
        Verifies conversion works correctly for multidimensional data.
        """
        kelvin_3d = jnp.array([[[[273.15, 283.15], [293.15, 303.15]], 
                                 [[278.15, 288.15], [298.15, 308.15]]]])
        
        expected_shape = (1, 2, 2, 2)
        
        result = kelvin_to_celsius(kelvin_3d)
        
        assert result.shape == expected_shape, (
            f"3D array shape not preserved: expected {expected_shape}, "
            f"got {result.shape}"
        )
        
        # Verify conversion is correct
        expected_celsius = kelvin_3d - 273.15
        assert jnp.allclose(result, expected_celsius, rtol=1e-6, atol=1e-6), (
            f"3D kelvin_to_celsius conversion incorrect"
        )
    
    def test_multidimensional_stefan_boltzmann(self):
        """
        Test Stefan-Boltzmann flux with 3D temperature arrays.
        
        Verifies flux calculations work for multidimensional spatial data.
        """
        temperature_3d = jnp.array([[[[273.15, 283.15], [293.15, 303.15]], 
                                      [[278.15, 288.15], [298.15, 308.15]]]])
        
        expected_shape = (1, 2, 2, 2)
        
        result = stefan_boltzmann_flux(temperature_3d)
        
        assert result.shape == expected_shape, (
            f"3D array shape not preserved: expected {expected_shape}, "
            f"got {result.shape}"
        )
        
        # Verify calculation
        sb_constant = 5.67e-8
        expected_flux = sb_constant * temperature_3d ** 4
        assert jnp.allclose(result, expected_flux, rtol=1e-5, atol=1e-3), (
            f"3D Stefan-Boltzmann flux calculation incorrect"
        )


class TestScalarArrayConsistency:
    """Test suite for scalar vs array input consistency."""
    
    def test_scalar_vs_array_celsius_to_kelvin(self):
        """
        Test that scalar and single-element array inputs produce consistent results.
        
        Verifies that the function handles both scalar-like and array inputs
        consistently.
        """
        celsius_scalar = 25.0
        celsius_array = jnp.array([[25.0]])
        
        result_scalar = celsius_to_kelvin(jnp.array([celsius_scalar]))
        result_array = celsius_to_kelvin(celsius_array)
        
        expected = 298.15
        
        assert jnp.allclose(result_scalar, expected, rtol=1e-6, atol=1e-6), (
            f"Scalar-like input conversion incorrect"
        )
        assert jnp.allclose(result_array, expected, rtol=1e-6, atol=1e-6), (
            f"Array input conversion incorrect"
        )
        assert jnp.allclose(result_scalar[0], result_array[0, 0], rtol=1e-10, atol=1e-10), (
            f"Scalar and array results should match"
        )


class TestPhysicalConstantsRelationships:
    """Test suite for physical relationships between constants."""
    
    def test_sublimation_enthalpy_relationship(self):
        """
        Verify physical relationship: h_sublimation = h_fusion + h_vaporization.
        
        Tests thermodynamic consistency of enthalpy constants.
        """
        hfus = get_constant("hfus", as_jax=False)
        hvap = get_constant("hvap", as_jax=False)
        hsub = get_constant("hsub", as_jax=False)
        
        expected_hsub = hfus + hvap
        
        assert np.isclose(hsub, expected_hsub, rtol=1e-6, atol=1.0), (
            f"Sublimation enthalpy relationship violated: "
            f"hsub={hsub}, hfus+hvap={expected_hsub}"
        )
    
    def test_density_relationship(self):
        """
        Verify physical relationship: density of water > density of ice.
        
        Tests that water is denser than ice (anomalous property of water).
        """
        denh2o = get_constant("denh2o", as_jax=False)
        denice = get_constant("denice", as_jax=False)
        
        assert denh2o > denice, (
            f"Water should be denser than ice: "
            f"denh2o={denh2o}, denice={denice}"
        )
    
    def test_thermal_conductivity_relationship(self):
        """
        Verify physical relationship: thermal conductivity of ice > water.
        
        Tests that ice conducts heat better than liquid water.
        """
        tkice = get_constant("tkice", as_jax=False)
        tkwat = get_constant("tkwat", as_jax=False)
        
        assert tkice > tkwat, (
            f"Ice should have higher thermal conductivity than water: "
            f"tkice={tkice}, tkwat={tkwat}"
        )
    
    def test_heat_capacity_relationship(self):
        """
        Verify physical relationship: heat capacity of liquid water > ice.
        
        Tests that liquid water has higher specific heat than ice.
        """
        cpliq = get_constant("cpliq", as_jax=False)
        cpice = get_constant("cpice", as_jax=False)
        
        assert cpliq > cpice, (
            f"Liquid water should have higher heat capacity than ice: "
            f"cpliq={cpliq}, cpice={cpice}"
        )
    
    def test_physical_constants_positive(self):
        """
        Verify all physical constants (except ispval) are positive.
        
        Tests that physical quantities have physically meaningful signs.
        """
        constants = get_jax_constants()
        
        for key, value in constants.items():
            if key != "ispval":  # ispval is intentionally negative
                assert float(value) > 0, (
                    f"Physical constant '{key}' should be positive, got {float(value)}"
                )


class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error conditions."""
    
    def test_temperature_conversion_preserves_nan(self):
        """
        Test that NaN values are preserved through temperature conversions.
        
        Verifies that invalid/missing data markers are maintained.
        """
        celsius_with_nan = jnp.array([0.0, jnp.nan, 25.0])
        kelvin_result = celsius_to_kelvin(celsius_with_nan)
        
        assert jnp.isnan(kelvin_result[1]), (
            "NaN should be preserved in celsius_to_kelvin"
        )
        assert jnp.isfinite(kelvin_result[0]) and jnp.isfinite(kelvin_result[2]), (
            "Valid values should remain finite"
        )
    
    def test_stefan_boltzmann_with_zero_temperature(self):
        """
        Test Stefan-Boltzmann flux at absolute zero.
        
        Verifies that flux is zero at 0K (no thermal radiation).
        """
        temperature = jnp.array([0.0, 1e-10, 1e-5])
        result = stefan_boltzmann_flux(temperature)
        
        # At absolute zero, flux should be zero
        assert jnp.allclose(result[0], 0.0, atol=1e-20), (
            f"Flux at absolute zero should be zero, got {float(result[0])}"
        )
        
        # Near-zero temperatures should have very small flux
        assert result[1] < 1e-30, (
            f"Flux at near-zero temperature should be negligible, got {float(result[1])}"
        )