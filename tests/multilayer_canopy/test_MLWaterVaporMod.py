"""
Comprehensive pytest suite for MLWaterVaporMod water vapor functions.

This module tests the following functions:
- sat_vap: Saturation vapor pressure and its temperature derivative
- lat_vap: Molar latent heat of vaporization
- sat_vap_with_constants: Saturation vapor pressure with custom constants
- vapor_pressure_deficit: Vapor pressure deficit calculation

Test coverage includes:
- Nominal atmospheric conditions
- Edge cases (freezing point, temperature bounds, saturation)
- Multi-dimensional array inputs
- Physical realism constraints
- Shape preservation and dtype consistency
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from multilayer_canopy.MLWaterVaporMod import (
    WaterVaporConstants,
    DEFAULT_CONSTANTS,
    sat_vap,
    lat_vap,
    sat_vap_with_constants,
    vapor_pressure_deficit,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def default_constants():
    """Provide default water vapor constants for testing."""
    return WaterVaporConstants(
        tfrz=273.15,
        hvap=2501000.0,
        hsub=2834000.0,
        mmh2o=0.018015
    )


@pytest.fixture
def test_data():
    """
    Provide comprehensive test data for all water vapor functions.
    
    Returns:
        dict: Test cases organized by function and test type
    """
    return {
        "sat_vap": {
            "nominal_room_temp": {
                "t": jnp.array([293.15, 298.15, 303.15]),
                "tfrz": 273.15,
                "expected_es_range": (2000.0, 5000.0),
                "expected_desdt_range": (100.0, 400.0),
            },
            "edge_freezing": {
                "t": jnp.array([273.15]),
                "tfrz": 273.15,
                "expected_es": 611.0,
                "expected_desdt": 44.0,
                "tolerance": 50.0,
            },
            "edge_bounds": {
                "t": jnp.array([198.15, 373.15]),
                "tfrz": 273.15,
                "expected_es_min": 1.0,
                "expected_es_max": 101325.0,
            },
            "multidim": {
                "t": jnp.array([[273.15, 283.15, 293.15],
                               [303.15, 313.15, 323.15]]),
                "tfrz": 273.15,
                "expected_shape": (2, 3),
            },
        },
        "lat_vap": {
            "nominal_atmospheric": {
                "t": jnp.array([250.0, 273.15, 300.0, 320.0]),
                "expected_range": (44000.0, 52000.0),
            },
            "edge_near_zero": {
                "t": jnp.array([1.0, 10.0, 50.0]),
                "expected_min": 40000.0,
                "expected_max": 60000.0,
            },
        },
        "sat_vap_with_constants": {
            "nominal": {
                "t": jnp.array([273.15, 288.15, 303.15]),
                "expected_es_range": (600.0, 5000.0),
            },
        },
        "vapor_pressure_deficit": {
            "nominal_humidity_range": {
                "t": jnp.array([293.15, 293.15, 293.15, 293.15]),
                "rh": jnp.array([0.0, 25.0, 50.0, 100.0]),
                "expected_vpd_decreasing": True,
            },
            "edge_saturated": {
                "t": jnp.array([273.15, 298.15, 323.15]),
                "rh": jnp.array([100.0, 100.0, 100.0]),
                "expected_vpd": 0.0,
                "tolerance": 1e-3,
            },
            "special_3d": {
                "t": jnp.array([[[273.15, 283.15], [293.15, 303.15]],
                               [[313.15, 323.15], [333.15, 343.15]]]),
                "rh": jnp.array([[[90.0, 80.0], [70.0, 60.0]],
                                [[50.0, 40.0], [30.0, 20.0]]]),
                "expected_shape": (2, 2, 2),
            },
        },
    }


# ============================================================================
# Tests for sat_vap
# ============================================================================

class TestSatVap:
    """Test suite for sat_vap function."""
    
    def test_sat_vap_shapes_nominal(self, test_data):
        """
        Test that sat_vap returns correct output shapes for various input shapes.
        
        Verifies:
        - Output tuple contains two arrays
        - Both arrays have same shape as input temperature array
        """
        data = test_data["sat_vap"]["nominal_room_temp"]
        t = data["t"]
        
        es, desdt = sat_vap(t, data["tfrz"])
        
        assert isinstance(es, jnp.ndarray), "es should be a JAX array"
        assert isinstance(desdt, jnp.ndarray), "desdt should be a JAX array"
        assert es.shape == t.shape, f"es shape {es.shape} should match input shape {t.shape}"
        assert desdt.shape == t.shape, f"desdt shape {desdt.shape} should match input shape {t.shape}"
    
    def test_sat_vap_shapes_multidimensional(self, test_data):
        """
        Test sat_vap with multi-dimensional array inputs.
        
        Verifies shape preservation for 2D temperature grids.
        """
        data = test_data["sat_vap"]["multidim"]
        t = data["t"]
        
        es, desdt = sat_vap(t, data["tfrz"])
        
        assert es.shape == data["expected_shape"], \
            f"Expected shape {data['expected_shape']}, got {es.shape}"
        assert desdt.shape == data["expected_shape"], \
            f"Expected shape {data['expected_shape']}, got {desdt.shape}"
    
    def test_sat_vap_values_room_temperature(self, test_data):
        """
        Test sat_vap produces physically reasonable values at room temperature.
        
        Verifies:
        - Saturation vapor pressure increases with temperature
        - Values are in expected range for 20-30°C
        - Temperature derivative is positive
        """
        data = test_data["sat_vap"]["nominal_room_temp"]
        t = data["t"]
        
        es, desdt = sat_vap(t, data["tfrz"])
        
        # Check es is in expected range
        assert jnp.all(es >= data["expected_es_range"][0]), \
            f"es values {es} below expected minimum {data['expected_es_range'][0]}"
        assert jnp.all(es <= data["expected_es_range"][1]), \
            f"es values {es} above expected maximum {data['expected_es_range'][1]}"
        
        # Check es increases with temperature
        assert jnp.all(jnp.diff(es) > 0), \
            "Saturation vapor pressure should increase with temperature"
        
        # Check desdt is positive and in expected range
        assert jnp.all(desdt > 0), "Temperature derivative should be positive"
        assert jnp.all(desdt >= data["expected_desdt_range"][0]), \
            f"desdt values {desdt} below expected minimum"
        assert jnp.all(desdt <= data["expected_desdt_range"][1]), \
            f"desdt values {desdt} above expected maximum"
    
    def test_sat_vap_edge_freezing_point(self, test_data):
        """
        Test sat_vap at the freezing point of water.
        
        Verifies values near the triple point of water (~611 Pa).
        """
        data = test_data["sat_vap"]["edge_freezing"]
        t = data["t"]
        
        es, desdt = sat_vap(t, data["tfrz"])
        
        # Check es is close to triple point pressure
        assert jnp.allclose(es, data["expected_es"], atol=data["tolerance"]), \
            f"At freezing point, es should be ~{data['expected_es']} Pa, got {es[0]}"
        
        # Check desdt is reasonable
        assert jnp.allclose(desdt, data["expected_desdt"], atol=data["tolerance"]), \
            f"At freezing point, desdt should be ~{data['expected_desdt']} Pa/K, got {desdt[0]}"
    
    def test_sat_vap_edge_temperature_bounds(self, test_data):
        """
        Test sat_vap at extreme temperature boundaries.
        
        Verifies:
        - Function handles minimum temperature (-75°C)
        - Function handles maximum temperature (100°C, boiling point)
        - Values are physically reasonable at boundaries
        """
        data = test_data["sat_vap"]["edge_bounds"]
        t = data["t"]
        
        es, desdt = sat_vap(t, data["tfrz"])
        
        # Check minimum temperature produces small but positive es
        assert es[0] > data["expected_es_min"], \
            f"At minimum temperature, es should be > {data['expected_es_min']} Pa"
        
        # Check maximum temperature produces es near atmospheric pressure
        assert jnp.allclose(es[1], data["expected_es_max"], rtol=0.1), \
            f"At boiling point, es should be ~{data['expected_es_max']} Pa, got {es[1]}"
        
        # Both derivatives should be positive
        assert jnp.all(desdt > 0), "Temperature derivatives should be positive"
    
    def test_sat_vap_dtypes(self, test_data):
        """
        Test that sat_vap preserves appropriate data types.
        
        Verifies outputs are JAX arrays with float dtype.
        """
        data = test_data["sat_vap"]["nominal_room_temp"]
        t = data["t"]
        
        es, desdt = sat_vap(t, data["tfrz"])
        
        assert jnp.issubdtype(es.dtype, jnp.floating), \
            f"es should have float dtype, got {es.dtype}"
        assert jnp.issubdtype(desdt.dtype, jnp.floating), \
            f"desdt should have float dtype, got {desdt.dtype}"
    
    @pytest.mark.parametrize("t_val,tfrz_val", [
        (273.15, 273.15),  # Freezing point
        (293.15, 273.15),  # Room temperature
        (373.15, 273.15),  # Boiling point
    ])
    def test_sat_vap_parametrized_temperatures(self, t_val, tfrz_val):
        """
        Parametrized test for sat_vap at key temperature points.
        
        Tests freezing, room, and boiling temperatures.
        """
        t = jnp.array([t_val])
        es, desdt = sat_vap(t, tfrz_val)
        
        assert es.shape == (1,), "Output shape should match input"
        assert es[0] > 0, "Saturation vapor pressure should be positive"
        assert desdt[0] > 0, "Temperature derivative should be positive"


# ============================================================================
# Tests for lat_vap
# ============================================================================

class TestLatVap:
    """Test suite for lat_vap function."""
    
    def test_lat_vap_shapes(self, test_data, default_constants):
        """
        Test that lat_vap returns correct output shape.
        
        Verifies output shape matches input temperature array shape.
        """
        data = test_data["lat_vap"]["nominal_atmospheric"]
        t = data["t"]
        
        lv = lat_vap(t, default_constants)
        
        assert isinstance(lv, jnp.ndarray), "Output should be a JAX array"
        assert lv.shape == t.shape, \
            f"Output shape {lv.shape} should match input shape {t.shape}"
    
    def test_lat_vap_values_atmospheric_range(self, test_data, default_constants):
        """
        Test lat_vap produces physically reasonable values across atmospheric temperatures.
        
        Verifies:
        - Latent heat is in expected range (44-52 kJ/mol)
        - Values decrease slightly with increasing temperature
        - All values are positive
        """
        data = test_data["lat_vap"]["nominal_atmospheric"]
        t = data["t"]
        
        lv = lat_vap(t, default_constants)
        
        # Check values are in expected range
        assert jnp.all(lv >= data["expected_range"][0]), \
            f"Latent heat values {lv} below expected minimum {data['expected_range'][0]}"
        assert jnp.all(lv <= data["expected_range"][1]), \
            f"Latent heat values {lv} above expected maximum {data['expected_range'][1]}"
        
        # All values should be positive
        assert jnp.all(lv > 0), "Latent heat should be positive"
    
    def test_lat_vap_edge_near_zero_kelvin(self, test_data, default_constants):
        """
        Test lat_vap at very low temperatures approaching absolute zero.
        
        Verifies function handles extreme cold conditions and uses sublimation heat.
        """
        data = test_data["lat_vap"]["edge_near_zero"]
        t = data["t"]
        
        lv = lat_vap(t, default_constants)
        
        # Check values are physically reasonable
        assert jnp.all(lv >= data["expected_min"]), \
            f"Latent heat at low T should be >= {data['expected_min']}"
        assert jnp.all(lv <= data["expected_max"]), \
            f"Latent heat at low T should be <= {data['expected_max']}"
        
        # Should be positive
        assert jnp.all(lv > 0), "Latent heat should be positive even at low temperatures"
    
    def test_lat_vap_temperature_dependence(self, default_constants):
        """
        Test that latent heat decreases with increasing temperature.
        
        This is a fundamental thermodynamic property.
        """
        t = jnp.array([250.0, 273.15, 300.0, 320.0])
        lv = lat_vap(t, default_constants)
        
        # For temperatures above freezing, latent heat should decrease
        above_freezing = t > default_constants.tfrz
        if jnp.sum(above_freezing) > 1:
            lv_above = lv[above_freezing]
            # Check general decreasing trend (allowing for small numerical variations)
            assert lv_above[0] >= lv_above[-1], \
                "Latent heat should generally decrease with temperature above freezing"
    
    def test_lat_vap_dtypes(self, test_data, default_constants):
        """
        Test that lat_vap preserves appropriate data types.
        """
        data = test_data["lat_vap"]["nominal_atmospheric"]
        t = data["t"]
        
        lv = lat_vap(t, default_constants)
        
        assert jnp.issubdtype(lv.dtype, jnp.floating), \
            f"Output should have float dtype, got {lv.dtype}"


# ============================================================================
# Tests for sat_vap_with_constants
# ============================================================================

class TestSatVapWithConstants:
    """Test suite for sat_vap_with_constants function."""
    
    def test_sat_vap_with_constants_shapes(self, test_data, default_constants):
        """
        Test that sat_vap_with_constants returns correct output shapes.
        """
        data = test_data["sat_vap_with_constants"]["nominal"]
        t = data["t"]
        
        es, desdt = sat_vap_with_constants(t, default_constants)
        
        assert es.shape == t.shape, \
            f"es shape {es.shape} should match input shape {t.shape}"
        assert desdt.shape == t.shape, \
            f"desdt shape {desdt.shape} should match input shape {t.shape}"
    
    def test_sat_vap_with_constants_values(self, test_data, default_constants):
        """
        Test that sat_vap_with_constants produces physically reasonable values.
        
        Verifies values are in expected range and increase with temperature.
        """
        data = test_data["sat_vap_with_constants"]["nominal"]
        t = data["t"]
        
        es, desdt = sat_vap_with_constants(t, default_constants)
        
        # Check es is in expected range
        assert jnp.all(es >= data["expected_es_range"][0]), \
            f"es values below expected minimum"
        assert jnp.all(es <= data["expected_es_range"][1]), \
            f"es values above expected maximum"
        
        # Check es increases with temperature
        assert jnp.all(jnp.diff(es) > 0), \
            "Saturation vapor pressure should increase with temperature"
        
        # Check desdt is positive
        assert jnp.all(desdt > 0), "Temperature derivative should be positive"
    
    def test_sat_vap_with_constants_consistency_with_sat_vap(self, default_constants):
        """
        Test that sat_vap_with_constants produces same results as sat_vap.
        
        When using default constants, both functions should give identical results.
        """
        t = jnp.array([273.15, 288.15, 303.15])
        
        # Call both functions
        es1, desdt1 = sat_vap(t, default_constants.tfrz)
        es2, desdt2 = sat_vap_with_constants(t, default_constants)
        
        # Results should be very close (allowing for numerical precision)
        assert jnp.allclose(es1, es2, rtol=1e-6, atol=1e-6), \
            "sat_vap and sat_vap_with_constants should produce same es values"
        assert jnp.allclose(desdt1, desdt2, rtol=1e-6, atol=1e-6), \
            "sat_vap and sat_vap_with_constants should produce same desdt values"
    
    def test_sat_vap_with_constants_custom_constants(self):
        """
        Test sat_vap_with_constants with custom constants.
        
        Verifies function accepts and uses custom WaterVaporConstants.
        """
        t = jnp.array([273.15, 298.15])
        custom_constants = WaterVaporConstants(
            tfrz=273.15,
            hvap=2500000.0,  # Slightly different
            hsub=2835000.0,  # Slightly different
            mmh2o=0.018015
        )
        
        es, desdt = sat_vap_with_constants(t, custom_constants)
        
        assert es.shape == t.shape, "Output shape should match input"
        assert jnp.all(es > 0), "Saturation vapor pressure should be positive"
        assert jnp.all(desdt > 0), "Temperature derivative should be positive"
    
    def test_sat_vap_with_constants_dtypes(self, test_data, default_constants):
        """
        Test that sat_vap_with_constants preserves appropriate data types.
        """
        data = test_data["sat_vap_with_constants"]["nominal"]
        t = data["t"]
        
        es, desdt = sat_vap_with_constants(t, default_constants)
        
        assert jnp.issubdtype(es.dtype, jnp.floating), \
            f"es should have float dtype, got {es.dtype}"
        assert jnp.issubdtype(desdt.dtype, jnp.floating), \
            f"desdt should have float dtype, got {desdt.dtype}"


# ============================================================================
# Tests for vapor_pressure_deficit
# ============================================================================

class TestVaporPressureDeficit:
    """Test suite for vapor_pressure_deficit function."""
    
    def test_vpd_shapes_1d(self, test_data, default_constants):
        """
        Test that vapor_pressure_deficit returns correct output shape for 1D inputs.
        """
        data = test_data["vapor_pressure_deficit"]["nominal_humidity_range"]
        t = data["t"]
        rh = data["rh"]
        
        vpd = vapor_pressure_deficit(t, rh, default_constants)
        
        assert isinstance(vpd, jnp.ndarray), "Output should be a JAX array"
        assert vpd.shape == t.shape, \
            f"Output shape {vpd.shape} should match input shape {t.shape}"
    
    def test_vpd_shapes_3d(self, test_data, default_constants):
        """
        Test that vapor_pressure_deficit preserves shape for 3D inputs.
        """
        data = test_data["vapor_pressure_deficit"]["special_3d"]
        t = data["t"]
        rh = data["rh"]
        
        vpd = vapor_pressure_deficit(t, rh, default_constants)
        
        assert vpd.shape == data["expected_shape"], \
            f"Expected shape {data['expected_shape']}, got {vpd.shape}"
    
    def test_vpd_values_humidity_range(self, test_data, default_constants):
        """
        Test VPD values across humidity range at constant temperature.
        
        Verifies:
        - VPD decreases as relative humidity increases
        - VPD is zero at 100% RH
        - VPD is maximum at 0% RH
        """
        data = test_data["vapor_pressure_deficit"]["nominal_humidity_range"]
        t = data["t"]
        rh = data["rh"]
        
        vpd = vapor_pressure_deficit(t, rh, default_constants)
        
        # VPD should decrease with increasing RH
        assert jnp.all(jnp.diff(vpd) <= 0), \
            "VPD should decrease as relative humidity increases"
        
        # VPD should be zero at 100% RH
        assert jnp.allclose(vpd[-1], 0.0, atol=1e-3), \
            f"VPD should be ~0 at 100% RH, got {vpd[-1]}"
        
        # VPD should be positive for RH < 100%
        assert jnp.all(vpd[:-1] > 0), \
            "VPD should be positive for RH < 100%"
    
    def test_vpd_edge_saturated_conditions(self, test_data, default_constants):
        """
        Test VPD at saturated conditions (100% RH).
        
        VPD should be exactly zero at saturation regardless of temperature.
        """
        data = test_data["vapor_pressure_deficit"]["edge_saturated"]
        t = data["t"]
        rh = data["rh"]
        
        vpd = vapor_pressure_deficit(t, rh, default_constants)
        
        # All VPD values should be zero
        assert jnp.allclose(vpd, data["expected_vpd"], atol=data["tolerance"]), \
            f"VPD should be {data['expected_vpd']} at 100% RH, got {vpd}"
    
    def test_vpd_edge_dry_conditions(self, default_constants):
        """
        Test VPD at completely dry conditions (0% RH).
        
        VPD should equal saturation vapor pressure at 0% RH.
        """
        t = jnp.array([293.15])
        rh = jnp.array([0.0])
        
        vpd = vapor_pressure_deficit(t, rh, default_constants)
        es, _ = sat_vap_with_constants(t, default_constants)
        
        # VPD should equal es at 0% RH
        assert jnp.allclose(vpd, es, rtol=1e-6), \
            f"VPD should equal es at 0% RH: vpd={vpd[0]}, es={es[0]}"
    
    def test_vpd_physical_constraints(self, default_constants):
        """
        Test that VPD satisfies physical constraints.
        
        Verifies:
        - VPD is always non-negative
        - VPD increases with temperature at constant RH
        - VPD is bounded by saturation vapor pressure
        """
        t = jnp.array([273.15, 283.15, 293.15, 303.15])
        rh = jnp.array([50.0, 50.0, 50.0, 50.0])
        
        vpd = vapor_pressure_deficit(t, rh, default_constants)
        
        # VPD should be non-negative
        assert jnp.all(vpd >= 0), "VPD should be non-negative"
        
        # VPD should increase with temperature at constant RH
        assert jnp.all(jnp.diff(vpd) > 0), \
            "VPD should increase with temperature at constant RH"
        
        # VPD should be less than saturation vapor pressure
        es, _ = sat_vap_with_constants(t, default_constants)
        assert jnp.all(vpd <= es), \
            "VPD should be less than or equal to saturation vapor pressure"
    
    def test_vpd_dtypes(self, test_data, default_constants):
        """
        Test that vapor_pressure_deficit preserves appropriate data types.
        """
        data = test_data["vapor_pressure_deficit"]["nominal_humidity_range"]
        t = data["t"]
        rh = data["rh"]
        
        vpd = vapor_pressure_deficit(t, rh, default_constants)
        
        assert jnp.issubdtype(vpd.dtype, jnp.floating), \
            f"Output should have float dtype, got {vpd.dtype}"
    
    @pytest.mark.parametrize("rh_val,expected_fraction", [
        (0.0, 1.0),    # 0% RH: VPD = es
        (25.0, 0.75),  # 25% RH: VPD = 0.75 * es
        (50.0, 0.5),   # 50% RH: VPD = 0.5 * es
        (75.0, 0.25),  # 75% RH: VPD = 0.25 * es
        (100.0, 0.0),  # 100% RH: VPD = 0
    ])
    def test_vpd_parametrized_humidity_levels(self, rh_val, expected_fraction, default_constants):
        """
        Parametrized test for VPD at different humidity levels.
        
        Verifies VPD = es * (1 - RH/100) relationship.
        """
        t = jnp.array([293.15])
        rh = jnp.array([rh_val])
        
        vpd = vapor_pressure_deficit(t, rh, default_constants)
        es, _ = sat_vap_with_constants(t, default_constants)
        
        expected_vpd = es * expected_fraction
        
        assert jnp.allclose(vpd, expected_vpd, rtol=1e-5), \
            f"At {rh_val}% RH, VPD should be {expected_fraction} * es"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests verifying interactions between functions."""
    
    def test_consistency_across_functions(self, default_constants):
        """
        Test consistency of saturation vapor pressure across different functions.
        
        Verifies that sat_vap and sat_vap_with_constants produce consistent
        results when used in vapor_pressure_deficit calculations.
        """
        t = jnp.array([273.15, 293.15, 313.15])
        rh = jnp.array([50.0, 50.0, 50.0])
        
        # Calculate VPD
        vpd = vapor_pressure_deficit(t, rh, default_constants)
        
        # Calculate es independently
        es, _ = sat_vap_with_constants(t, default_constants)
        
        # VPD should equal es * (1 - rh/100)
        expected_vpd = es * (1.0 - rh / 100.0)
        
        assert jnp.allclose(vpd, expected_vpd, rtol=1e-5), \
            "VPD calculation should be consistent with es calculation"
    
    def test_thermodynamic_consistency(self, default_constants):
        """
        Test thermodynamic consistency between latent heat and vapor pressure.
        
        Verifies that the Clausius-Clapeyron relation is approximately satisfied.
        """
        t = jnp.array([273.15, 283.15, 293.15])
        
        # Get saturation vapor pressure and derivative
        es, desdt = sat_vap_with_constants(t, default_constants)
        
        # Get latent heat
        lv = lat_vap(t, default_constants)
        
        # Clausius-Clapeyron: d(ln(es))/dT ≈ Lv / (R * T^2)
        # Or: desdt/es ≈ Lv / (R * T^2)
        # R = 8.314 J/(mol·K)
        R = 8.314
        
        # Calculate left side: desdt/es
        left_side = desdt / es
        
        # Calculate right side: Lv / (R * T^2)
        right_side = lv / (R * t**2)
        
        # These should be approximately equal (within ~20% due to approximations)
        ratio = left_side / right_side
        assert jnp.all((ratio > 0.8) & (ratio < 1.2)), \
            f"Clausius-Clapeyron relation not satisfied: ratio = {ratio}"
    
    def test_array_broadcasting(self, default_constants):
        """
        Test that functions handle array broadcasting correctly.
        
        Verifies functions work with different but compatible array shapes.
        """
        # 1D temperature array
        t = jnp.array([273.15, 293.15, 313.15])
        
        # Scalar RH (should broadcast)
        rh_scalar = jnp.array([50.0])
        
        vpd = vapor_pressure_deficit(t, jnp.broadcast_to(rh_scalar, t.shape), default_constants)
        
        assert vpd.shape == t.shape, \
            "Broadcasting should produce output matching temperature shape"
        assert jnp.all(vpd > 0), "All VPD values should be positive"


# ============================================================================
# Edge Case and Error Handling Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_value_inputs(self, default_constants):
        """
        Test functions with single-value (scalar-like) inputs.
        """
        t = jnp.array([293.15])
        rh = jnp.array([50.0])
        
        # Test all functions
        es, desdt = sat_vap(t, default_constants.tfrz)
        assert es.shape == (1,) and desdt.shape == (1,)
        
        lv = lat_vap(t, default_constants)
        assert lv.shape == (1,)
        
        es2, desdt2 = sat_vap_with_constants(t, default_constants)
        assert es2.shape == (1,) and desdt2.shape == (1,)
        
        vpd = vapor_pressure_deficit(t, rh, default_constants)
        assert vpd.shape == (1,)
    
    def test_extreme_temperature_gradients(self, default_constants):
        """
        Test functions with large temperature gradients.
        
        Verifies numerical stability across wide temperature ranges.
        """
        t = jnp.array([200.0, 250.0, 300.0, 350.0, 373.15])
        
        es, desdt = sat_vap_with_constants(t, default_constants)
        
        # All values should be finite
        assert jnp.all(jnp.isfinite(es)), "es should be finite for all temperatures"
        assert jnp.all(jnp.isfinite(desdt)), "desdt should be finite for all temperatures"
        
        # Values should increase monotonically
        assert jnp.all(jnp.diff(es) > 0), "es should increase monotonically"
        assert jnp.all(jnp.diff(desdt) > 0), "desdt should increase monotonically"
    
    def test_boundary_relative_humidity(self, default_constants):
        """
        Test VPD at boundary RH values (0% and 100%).
        """
        t = jnp.array([293.15])
        
        # Test 0% RH
        vpd_0 = vapor_pressure_deficit(t, jnp.array([0.0]), default_constants)
        es, _ = sat_vap_with_constants(t, default_constants)
        assert jnp.allclose(vpd_0, es, rtol=1e-6), "VPD at 0% RH should equal es"
        
        # Test 100% RH
        vpd_100 = vapor_pressure_deficit(t, jnp.array([100.0]), default_constants)
        assert jnp.allclose(vpd_100, 0.0, atol=1e-3), "VPD at 100% RH should be zero"
    
    def test_numerical_precision(self, default_constants):
        """
        Test numerical precision and stability.
        
        Verifies functions maintain precision across multiple operations.
        """
        t = jnp.array([273.15, 273.16, 273.17])  # Very close temperatures
        
        es, desdt = sat_vap_with_constants(t, default_constants)
        
        # Small temperature differences should produce small es differences
        es_diff = jnp.diff(es)
        assert jnp.all(es_diff > 0), "Small T increases should produce small es increases"
        assert jnp.all(es_diff < 10.0), "es differences should be small for small T differences"
        
        # desdt should be relatively constant for small T range
        desdt_variation = jnp.std(desdt) / jnp.mean(desdt)
        assert desdt_variation < 0.01, "desdt should be nearly constant over small T range"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])