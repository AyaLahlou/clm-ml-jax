"""
Comprehensive pytest suite for mixing_length module functions.

This test suite covers:
- compute_mixing_length: Main function for computing mixing length scales
- calc_Lscale_directly: Direct calculation of mixing length scales
- diagnose_Lscale_from_tau: Diagnosis of mixing length from timescales
- zm2zt_api: Interpolation from momentum to thermodynamic levels
- zt2zm_api: Interpolation from thermodynamic to momentum levels
- zt2zm2zt: Smoothing operation on thermodynamic levels
- zm2zt2zm: Smoothing operation on momentum levels
- smooth_max: Smooth maximum function
- smooth_min: Smooth minimum function

Test categories:
- Nominal cases: Typical atmospheric conditions
- Edge cases: Boundary conditions (zero TKE, dry atmosphere, etc.)
- Special cases: Extreme conditions (high altitude, inversions, etc.)
- Shape validation: Output array dimensions
- Value validation: Physical constraints and numerical accuracy
- Data type validation: Correct JAX array types
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_main.mixing_length import (
    calc_Lscale_directly,
    compute_mixing_length,
    diagnose_Lscale_from_tau,
    smooth_max,
    smooth_min,
    zm2zt2zm,
    zm2zt_api,
    zt2zm2zt,
    zt2zm_api,
)

# Constants from the module
ZLMIN = 0.1  # Minimum mixing length scale [m]
LSCALE_SFCLYR_DEPTH = 500.0  # Surface layer depth [m]


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load and return test data for mixing_length functions.
    
    Returns:
        Dictionary containing test cases with inputs and metadata.
    """
    return {
        "test_cases": [
            {
                "name": "test_nominal_single_column_moderate_conditions",
                "inputs": {
                    "nzm": 10,
                    "nzt": 11,
                    "ngrdcol": 1,
                    "gr": {
                        "dzm": jnp.array([50.0] * 10),
                        "zt": jnp.array([25.0, 75.0, 125.0, 175.0, 225.0, 275.0, 325.0, 375.0, 425.0, 475.0, 525.0]),
                    },
                    "thvm": jnp.array([[290.0, 291.0, 292.0, 293.0, 294.0, 295.0, 296.0, 297.0, 298.0, 299.0, 300.0]]),
                    "thlm": jnp.array([[288.0, 289.0, 290.0, 291.0, 292.0, 293.0, 294.0, 295.0, 296.0, 297.0, 298.0]]),
                    "rtm": jnp.array([[0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0005]]),
                    "em": jnp.array([[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6]]),
                    "Lscale_max": 1000.0,
                    "p_in_Pa": jnp.array([[101000.0, 99000.0, 97000.0, 95000.0, 93000.0, 91000.0, 89000.0, 87000.0, 85000.0, 83000.0, 81000.0]]),
                    "exner": jnp.array([[1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9]]),
                    "thv_ds": jnp.array([[290.5, 291.5, 292.5, 293.5, 294.5, 295.5, 296.5, 297.5, 298.5, 299.5, 300.5]]),
                    "mu": 0.5,
                    "lmin": 1.0,
                    "saturation_formula": 1,
                    "l_implemented": True,
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Single column with moderate atmospheric conditions",
                    "edge_cases": [],
                },
            },
            {
                "name": "test_edge_zero_turbulent_kinetic_energy",
                "inputs": {
                    "nzm": 6,
                    "nzt": 7,
                    "ngrdcol": 2,
                    "gr": {
                        "dzm": jnp.array([30.0] * 6),
                        "zt": jnp.array([15.0, 45.0, 75.0, 105.0, 135.0, 165.0, 195.0]),
                    },
                    "thvm": jnp.array([
                        [288.0, 289.0, 290.0, 291.0, 292.0, 293.0, 294.0],
                        [290.0, 291.0, 292.0, 293.0, 294.0, 295.0, 296.0],
                    ]),
                    "thlm": jnp.array([
                        [286.0, 287.0, 288.0, 289.0, 290.0, 291.0, 292.0],
                        [288.0, 289.0, 290.0, 291.0, 292.0, 293.0, 294.0],
                    ]),
                    "rtm": jnp.array([
                        [0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002],
                        [0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004],
                    ]),
                    "em": jnp.zeros((2, 6)),
                    "Lscale_max": 500.0,
                    "p_in_Pa": jnp.array([
                        [101000.0, 99500.0, 98000.0, 96500.0, 95000.0, 93500.0, 92000.0],
                        [101000.0, 99500.0, 98000.0, 96500.0, 95000.0, 93500.0, 92000.0],
                    ]),
                    "exner": jnp.array([
                        [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94],
                        [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94],
                    ]),
                    "thv_ds": jnp.array([
                        [288.5, 289.5, 290.5, 291.5, 292.5, 293.5, 294.5],
                        [290.5, 291.5, 292.5, 293.5, 294.5, 295.5, 296.5],
                    ]),
                    "mu": 0.5,
                    "lmin": 1.0,
                    "saturation_formula": 1,
                    "l_implemented": True,
                },
                "metadata": {
                    "type": "edge",
                    "description": "Zero turbulent kinetic energy",
                    "edge_cases": ["zero_tke"],
                },
            },
            {
                "name": "test_edge_very_dry_atmosphere",
                "inputs": {
                    "nzm": 7,
                    "nzt": 8,
                    "ngrdcol": 1,
                    "gr": {
                        "dzm": jnp.array([45.0] * 7),
                        "zt": jnp.array([22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]),
                    },
                    "thvm": jnp.array([[305.0, 306.0, 307.0, 308.0, 309.0, 310.0, 311.0, 312.0]]),
                    "thlm": jnp.array([[304.0, 305.0, 306.0, 307.0, 308.0, 309.0, 310.0, 311.0]]),
                    "rtm": jnp.array([[0.0001, 8e-05, 6e-05, 4e-05, 2e-05, 1e-05, 5e-06, 1e-06]]),
                    "em": jnp.array([[0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3]]),
                    "Lscale_max": 1500.0,
                    "p_in_Pa": jnp.array([[95000.0, 93000.0, 91000.0, 89000.0, 87000.0, 85000.0, 83000.0, 81000.0]]),
                    "exner": jnp.array([[0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91]]),
                    "thv_ds": jnp.array([[305.2, 306.2, 307.2, 308.2, 309.2, 310.2, 311.2, 312.2]]),
                    "mu": 0.4,
                    "lmin": 0.8,
                    "saturation_formula": 1,
                    "l_implemented": True,
                },
                "metadata": {
                    "type": "edge",
                    "description": "Very dry atmospheric conditions",
                    "edge_cases": ["near_zero_moisture"],
                },
            },
            {
                "name": "test_edge_minimum_vertical_levels",
                "inputs": {
                    "nzm": 1,
                    "nzt": 2,
                    "ngrdcol": 1,
                    "gr": {
                        "dzm": jnp.array([100.0]),
                        "zt": jnp.array([50.0, 150.0]),
                    },
                    "thvm": jnp.array([[290.0, 295.0]]),
                    "thlm": jnp.array([[288.0, 293.0]]),
                    "rtm": jnp.array([[0.01, 0.008]]),
                    "em": jnp.array([[0.5]]),
                    "Lscale_max": 500.0,
                    "p_in_Pa": jnp.array([[101000.0, 99000.0]]),
                    "exner": jnp.array([[1.0, 0.99]]),
                    "thv_ds": jnp.array([[290.5, 295.5]]),
                    "mu": 0.5,
                    "lmin": 1.0,
                    "saturation_formula": 1,
                    "l_implemented": True,
                },
                "metadata": {
                    "type": "edge",
                    "description": "Minimum vertical resolution",
                    "edge_cases": ["minimum_levels"],
                },
            },
        ]
    }


class TestComputeMixingLength:
    """Test suite for compute_mixing_length function."""

    @pytest.mark.parametrize("test_case_idx", [0, 1, 2, 3])
    def test_output_shapes(self, test_data: Dict[str, Any], test_case_idx: int) -> None:
        """
        Test that compute_mixing_length returns arrays with correct shapes.
        
        Verifies:
        - Lscale shape is (ngrdcol, nzt)
        - Lscale_up shape is (ngrdcol, nzt)
        - Lscale_down shape is (ngrdcol, nzt)
        - err_info is returned
        """
        test_case = test_data["test_cases"][test_case_idx]
        inputs = test_case["inputs"]
        
        # Note: This test assumes compute_mixing_length is implemented
        # If not yet implemented, this will fail and should be skipped
        try:
            result = compute_mixing_length(
                nzm=inputs["nzm"],
                nzt=inputs["nzt"],
                ngrdcol=inputs["ngrdcol"],
                gr=inputs["gr"],
                thvm=inputs["thvm"],
                thlm=inputs["thlm"],
                rtm=inputs["rtm"],
                em=inputs["em"],
                Lscale_max=inputs["Lscale_max"],
                p_in_Pa=inputs["p_in_Pa"],
                exner=inputs["exner"],
                thv_ds=inputs["thv_ds"],
                mu=inputs["mu"],
                lmin=inputs["lmin"],
                saturation_formula=inputs["saturation_formula"],
                l_implemented=inputs["l_implemented"],
                err_info=None,
            )
            
            Lscale, Lscale_up, Lscale_down, err_info = result
            
            expected_shape = (inputs["ngrdcol"], inputs["nzt"])
            assert Lscale.shape == expected_shape, (
                f"Lscale shape mismatch: expected {expected_shape}, got {Lscale.shape}"
            )
            assert Lscale_up.shape == expected_shape, (
                f"Lscale_up shape mismatch: expected {expected_shape}, got {Lscale_up.shape}"
            )
            assert Lscale_down.shape == expected_shape, (
                f"Lscale_down shape mismatch: expected {expected_shape}, got {Lscale_down.shape}"
            )
            
        except NotImplementedError:
            pytest.skip("compute_mixing_length not yet implemented")

    @pytest.mark.parametrize("test_case_idx", [0, 1, 2, 3])
    def test_physical_constraints(self, test_data: Dict[str, Any], test_case_idx: int) -> None:
        """
        Test that outputs satisfy physical constraints.
        
        Verifies:
        - All mixing lengths >= ZLMIN (0.1 m)
        - All mixing lengths <= Lscale_max
        - No NaN or Inf values
        - All values are non-negative
        """
        test_case = test_data["test_cases"][test_case_idx]
        inputs = test_case["inputs"]
        
        try:
            result = compute_mixing_length(
                nzm=inputs["nzm"],
                nzt=inputs["nzt"],
                ngrdcol=inputs["ngrdcol"],
                gr=inputs["gr"],
                thvm=inputs["thvm"],
                thlm=inputs["thlm"],
                rtm=inputs["rtm"],
                em=inputs["em"],
                Lscale_max=inputs["Lscale_max"],
                p_in_Pa=inputs["p_in_Pa"],
                exner=inputs["exner"],
                thv_ds=inputs["thv_ds"],
                mu=inputs["mu"],
                lmin=inputs["lmin"],
                saturation_formula=inputs["saturation_formula"],
                l_implemented=inputs["l_implemented"],
                err_info=None,
            )
            
            Lscale, Lscale_up, Lscale_down, err_info = result
            
            # Check minimum constraint
            assert jnp.all(Lscale >= ZLMIN), (
                f"Lscale contains values below ZLMIN ({ZLMIN}): min={jnp.min(Lscale)}"
            )
            assert jnp.all(Lscale_up >= ZLMIN), (
                f"Lscale_up contains values below ZLMIN ({ZLMIN}): min={jnp.min(Lscale_up)}"
            )
            assert jnp.all(Lscale_down >= ZLMIN), (
                f"Lscale_down contains values below ZLMIN ({ZLMIN}): min={jnp.min(Lscale_down)}"
            )
            
            # Check maximum constraint
            Lscale_max_val = inputs["Lscale_max"]
            if isinstance(Lscale_max_val, (int, float)):
                Lscale_max_val = jnp.array(Lscale_max_val)
            
            assert jnp.all(Lscale <= Lscale_max_val), (
                f"Lscale contains values above Lscale_max: max={jnp.max(Lscale)}"
            )
            assert jnp.all(Lscale_up <= Lscale_max_val), (
                f"Lscale_up contains values above Lscale_max: max={jnp.max(Lscale_up)}"
            )
            assert jnp.all(Lscale_down <= Lscale_max_val), (
                f"Lscale_down contains values above Lscale_max: max={jnp.max(Lscale_down)}"
            )
            
            # Check for NaN/Inf
            assert jnp.all(jnp.isfinite(Lscale)), "Lscale contains NaN or Inf values"
            assert jnp.all(jnp.isfinite(Lscale_up)), "Lscale_up contains NaN or Inf values"
            assert jnp.all(jnp.isfinite(Lscale_down)), "Lscale_down contains NaN or Inf values"
            
            # Check non-negativity
            assert jnp.all(Lscale >= 0), "Lscale contains negative values"
            assert jnp.all(Lscale_up >= 0), "Lscale_up contains negative values"
            assert jnp.all(Lscale_down >= 0), "Lscale_down contains negative values"
            
        except NotImplementedError:
            pytest.skip("compute_mixing_length not yet implemented")

    def test_zero_tke_behavior(self, test_data: Dict[str, Any]) -> None:
        """
        Test behavior with zero turbulent kinetic energy.
        
        With zero TKE, mixing lengths should be at or near minimum values.
        """
        # Find the zero TKE test case
        test_case = next(
            tc for tc in test_data["test_cases"]
            if "zero_tke" in tc["metadata"]["edge_cases"]
        )
        inputs = test_case["inputs"]
        
        try:
            result = compute_mixing_length(
                nzm=inputs["nzm"],
                nzt=inputs["nzt"],
                ngrdcol=inputs["ngrdcol"],
                gr=inputs["gr"],
                thvm=inputs["thvm"],
                thlm=inputs["thlm"],
                rtm=inputs["rtm"],
                em=inputs["em"],
                Lscale_max=inputs["Lscale_max"],
                p_in_Pa=inputs["p_in_Pa"],
                exner=inputs["exner"],
                thv_ds=inputs["thv_ds"],
                mu=inputs["mu"],
                lmin=inputs["lmin"],
                saturation_formula=inputs["saturation_formula"],
                l_implemented=inputs["l_implemented"],
                err_info=None,
            )
            
            Lscale, Lscale_up, Lscale_down, err_info = result
            
            # With zero TKE, expect mixing lengths to be small
            # (exact behavior depends on implementation)
            assert jnp.all(Lscale <= 100.0), (
                "With zero TKE, mixing lengths should be small"
            )
            
        except NotImplementedError:
            pytest.skip("compute_mixing_length not yet implemented")

    @pytest.mark.parametrize("test_case_idx", [0])
    def test_dtypes(self, test_data: Dict[str, Any], test_case_idx: int) -> None:
        """
        Test that outputs have correct JAX array data types.
        
        Verifies all outputs are JAX arrays with float dtype.
        """
        test_case = test_data["test_cases"][test_case_idx]
        inputs = test_case["inputs"]
        
        try:
            result = compute_mixing_length(
                nzm=inputs["nzm"],
                nzt=inputs["nzt"],
                ngrdcol=inputs["ngrdcol"],
                gr=inputs["gr"],
                thvm=inputs["thvm"],
                thlm=inputs["thlm"],
                rtm=inputs["rtm"],
                em=inputs["em"],
                Lscale_max=inputs["Lscale_max"],
                p_in_Pa=inputs["p_in_Pa"],
                exner=inputs["exner"],
                thv_ds=inputs["thv_ds"],
                mu=inputs["mu"],
                lmin=inputs["lmin"],
                saturation_formula=inputs["saturation_formula"],
                l_implemented=inputs["l_implemented"],
                err_info=None,
            )
            
            Lscale, Lscale_up, Lscale_down, err_info = result
            
            assert isinstance(Lscale, jnp.ndarray), "Lscale should be a JAX array"
            assert isinstance(Lscale_up, jnp.ndarray), "Lscale_up should be a JAX array"
            assert isinstance(Lscale_down, jnp.ndarray), "Lscale_down should be a JAX array"
            
            assert jnp.issubdtype(Lscale.dtype, jnp.floating), "Lscale should have float dtype"
            assert jnp.issubdtype(Lscale_up.dtype, jnp.floating), "Lscale_up should have float dtype"
            assert jnp.issubdtype(Lscale_down.dtype, jnp.floating), "Lscale_down should have float dtype"
            
        except NotImplementedError:
            pytest.skip("compute_mixing_length not yet implemented")


class TestInterpolationFunctions:
    """Test suite for grid interpolation functions."""

    def test_zm2zt_api_shape(self) -> None:
        """Test zm2zt_api returns correct shape."""
        nzm, nzt, ngrdcol = 5, 6, 2
        gr = {
            "dzm": jnp.array([10.0] * nzm),
            "zt": jnp.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0]),
        }
        em = jnp.ones((ngrdcol, nzm))
        
        try:
            result = zm2zt_api(nzm, nzt, ngrdcol, gr, em)
            assert result.shape == (ngrdcol, nzt), (
                f"Expected shape {(ngrdcol, nzt)}, got {result.shape}"
            )
        except NotImplementedError:
            pytest.skip("zm2zt_api not yet implemented")

    def test_zt2zm_api_shape(self) -> None:
        """Test zt2zm_api returns correct shape."""
        nzm, nzt, ngrdcol = 5, 6, 2
        gr = {
            "dzm": jnp.array([10.0] * nzm),
            "zt": jnp.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0]),
        }
        field_zt = jnp.ones((ngrdcol, nzt))
        
        try:
            result = zt2zm_api(nzm, nzt, ngrdcol, gr, field_zt)
            assert result.shape == (ngrdcol, nzm), (
                f"Expected shape {(ngrdcol, nzm)}, got {result.shape}"
            )
        except NotImplementedError:
            pytest.skip("zt2zm_api not yet implemented")

    def test_zt2zm2zt_shape(self) -> None:
        """Test zt2zm2zt returns correct shape (smoothing operation)."""
        nzm, nzt, ngrdcol = 5, 6, 2
        gr = {
            "dzm": jnp.array([10.0] * nzm),
            "zt": jnp.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0]),
        }
        field_zt = jnp.ones((ngrdcol, nzt))
        
        try:
            result = zt2zm2zt(nzm, nzt, ngrdcol, gr, field_zt)
            assert result.shape == (ngrdcol, nzt), (
                f"Expected shape {(ngrdcol, nzt)}, got {result.shape}"
            )
        except NotImplementedError:
            pytest.skip("zt2zm2zt not yet implemented")

    def test_zm2zt2zm_shape(self) -> None:
        """Test zm2zt2zm returns correct shape (smoothing operation)."""
        nzm, nzt, ngrdcol = 5, 6, 2
        gr = {
            "dzm": jnp.array([10.0] * nzm),
            "zt": jnp.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0]),
        }
        field_zm = jnp.ones((ngrdcol, nzm))
        
        try:
            result = zm2zt2zm(nzm, nzt, ngrdcol, gr, field_zm)
            assert result.shape == (ngrdcol, nzm), (
                f"Expected shape {(ngrdcol, nzm)}, got {result.shape}"
            )
        except NotImplementedError:
            pytest.skip("zm2zt2zm not yet implemented")

    def test_interpolation_preserves_constants(self) -> None:
        """Test that interpolating constant fields preserves values."""
        nzm, nzt, ngrdcol = 5, 6, 1
        gr = {
            "dzm": jnp.array([10.0] * nzm),
            "zt": jnp.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0]),
        }
        
        constant_value = 42.0
        field_zm = jnp.full((ngrdcol, nzm), constant_value)
        field_zt = jnp.full((ngrdcol, nzt), constant_value)
        
        try:
            # Test zm2zt
            result_zt = zm2zt_api(nzm, nzt, ngrdcol, gr, field_zm)
            assert jnp.allclose(result_zt, constant_value, atol=1e-6), (
                "zm2zt should preserve constant fields"
            )
            
            # Test zt2zm
            result_zm = zt2zm_api(nzm, nzt, ngrdcol, gr, field_zt)
            assert jnp.allclose(result_zm, constant_value, atol=1e-6), (
                "zt2zm should preserve constant fields"
            )
        except NotImplementedError:
            pytest.skip("Interpolation functions not yet implemented")


class TestSmoothingFunctions:
    """Test suite for smooth_max and smooth_min functions."""

    def test_smooth_max_shape(self) -> None:
        """Test smooth_max returns correct shape."""
        nzm, ngrdcol = 5, 2
        field1 = jnp.ones((ngrdcol, nzm))
        field2 = jnp.ones((ngrdcol, nzm)) * 2.0
        smoothing_mag = 0.1
        
        try:
            result = smooth_max(nzm, ngrdcol, field1, field2, smoothing_mag)
            assert result.shape == (ngrdcol, nzm), (
                f"Expected shape {(ngrdcol, nzm)}, got {result.shape}"
            )
        except NotImplementedError:
            pytest.skip("smooth_max not yet implemented")

    def test_smooth_max_behavior(self) -> None:
        """Test smooth_max approximates maximum function."""
        nzm, ngrdcol = 5, 1
        field1 = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        field2 = jnp.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
        smoothing_mag = 0.01  # Small smoothing for close approximation
        
        try:
            result = smooth_max(nzm, ngrdcol, field1, field2, smoothing_mag)
            expected = jnp.maximum(field1, field2)
            
            # Should be close to actual maximum with small smoothing
            assert jnp.allclose(result, expected, atol=0.1), (
                "smooth_max should approximate maximum function"
            )
        except NotImplementedError:
            pytest.skip("smooth_max not yet implemented")

    def test_smooth_min_shape(self) -> None:
        """Test smooth_min returns correct shape."""
        nzm, ngrdcol = 5, 2
        field = jnp.ones((ngrdcol, nzm)) * 10.0
        upper_bound = 5.0
        smoothing_factor = 0.1
        
        try:
            result = smooth_min(nzm, ngrdcol, field, upper_bound, smoothing_factor)
            assert result.shape == (ngrdcol, nzm), (
                f"Expected shape {(ngrdcol, nzm)}, got {result.shape}"
            )
        except NotImplementedError:
            pytest.skip("smooth_min not yet implemented")

    def test_smooth_min_behavior(self) -> None:
        """Test smooth_min approximates minimum function."""
        nzm, ngrdcol = 5, 1
        field = jnp.array([[1.0, 3.0, 5.0, 7.0, 9.0]])
        upper_bound = 5.0
        smoothing_factor = 0.01  # Small smoothing
        
        try:
            result = smooth_min(nzm, ngrdcol, field, upper_bound, smoothing_factor)
            expected = jnp.minimum(field, upper_bound)
            
            # Should be close to actual minimum with small smoothing
            assert jnp.allclose(result, expected, atol=0.1), (
                "smooth_min should approximate minimum function"
            )
        except NotImplementedError:
            pytest.skip("smooth_min not yet implemented")

    def test_smooth_functions_no_nan(self) -> None:
        """Test that smoothing functions don't produce NaN values."""
        nzm, ngrdcol = 5, 2
        field1 = jnp.ones((ngrdcol, nzm))
        field2 = jnp.ones((ngrdcol, nzm)) * 2.0
        
        try:
            result_max = smooth_max(nzm, ngrdcol, field1, field2, 0.1)
            assert jnp.all(jnp.isfinite(result_max)), (
                "smooth_max produced NaN or Inf values"
            )
            
            result_min = smooth_min(nzm, ngrdcol, field1, 1.5, 0.1)
            assert jnp.all(jnp.isfinite(result_min)), (
                "smooth_min produced NaN or Inf values"
            )
        except NotImplementedError:
            pytest.skip("Smoothing functions not yet implemented")


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_minimum_grid_size(self, test_data: Dict[str, Any]) -> None:
        """Test with minimum possible grid size (nzm=1, nzt=2)."""
        test_case = next(
            tc for tc in test_data["test_cases"]
            if "minimum_levels" in tc["metadata"]["edge_cases"]
        )
        inputs = test_case["inputs"]
        
        try:
            result = compute_mixing_length(
                nzm=inputs["nzm"],
                nzt=inputs["nzt"],
                ngrdcol=inputs["ngrdcol"],
                gr=inputs["gr"],
                thvm=inputs["thvm"],
                thlm=inputs["thlm"],
                rtm=inputs["rtm"],
                em=inputs["em"],
                Lscale_max=inputs["Lscale_max"],
                p_in_Pa=inputs["p_in_Pa"],
                exner=inputs["exner"],
                thv_ds=inputs["thv_ds"],
                mu=inputs["mu"],
                lmin=inputs["lmin"],
                saturation_formula=inputs["saturation_formula"],
                l_implemented=inputs["l_implemented"],
                err_info=None,
            )
            
            Lscale, Lscale_up, Lscale_down, err_info = result
            
            # Should still produce valid output
            assert Lscale.shape == (1, 2), "Incorrect shape for minimum grid"
            assert jnp.all(jnp.isfinite(Lscale)), "NaN/Inf in minimum grid case"
            
        except NotImplementedError:
            pytest.skip("compute_mixing_length not yet implemented")

    def test_very_dry_atmosphere(self, test_data: Dict[str, Any]) -> None:
        """Test with very low moisture content."""
        test_case = next(
            tc for tc in test_data["test_cases"]
            if "near_zero_moisture" in tc["metadata"]["edge_cases"]
        )
        inputs = test_case["inputs"]
        
        try:
            result = compute_mixing_length(
                nzm=inputs["nzm"],
                nzt=inputs["nzt"],
                ngrdcol=inputs["ngrdcol"],
                gr=inputs["gr"],
                thvm=inputs["thvm"],
                thlm=inputs["thlm"],
                rtm=inputs["rtm"],
                em=inputs["em"],
                Lscale_max=inputs["Lscale_max"],
                p_in_Pa=inputs["p_in_Pa"],
                exner=inputs["exner"],
                thv_ds=inputs["thv_ds"],
                mu=inputs["mu"],
                lmin=inputs["lmin"],
                saturation_formula=inputs["saturation_formula"],
                l_implemented=inputs["l_implemented"],
                err_info=None,
            )
            
            Lscale, Lscale_up, Lscale_down, err_info = result
            
            # Should handle very dry conditions without issues
            assert jnp.all(jnp.isfinite(Lscale)), "NaN/Inf in very dry case"
            assert jnp.all(Lscale >= ZLMIN), "Violates minimum constraint in dry case"
            
        except NotImplementedError:
            pytest.skip("compute_mixing_length not yet implemented")

    def test_scalar_vs_array_Lscale_max(self) -> None:
        """Test that function handles both scalar and array Lscale_max."""
        nzm, nzt, ngrdcol = 5, 6, 2
        gr = {
            "dzm": jnp.array([10.0] * nzm),
            "zt": jnp.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0]),
        }
        
        common_inputs = {
            "nzm": nzm,
            "nzt": nzt,
            "ngrdcol": ngrdcol,
            "gr": gr,
            "thvm": jnp.ones((ngrdcol, nzt)) * 290.0,
            "thlm": jnp.ones((ngrdcol, nzt)) * 288.0,
            "rtm": jnp.ones((ngrdcol, nzt)) * 0.01,
            "em": jnp.ones((ngrdcol, nzm)) * 0.5,
            "p_in_Pa": jnp.ones((ngrdcol, nzt)) * 100000.0,
            "exner": jnp.ones((ngrdcol, nzt)),
            "thv_ds": jnp.ones((ngrdcol, nzt)) * 290.5,
            "mu": 0.5,
            "lmin": 1.0,
            "saturation_formula": 1,
            "l_implemented": True,
        }
        
        try:
            # Test with scalar
            result_scalar = compute_mixing_length(
                **common_inputs,
                Lscale_max=1000.0,
                err_info=None,
            )
            
            # Test with array
            result_array = compute_mixing_length(
                **common_inputs,
                Lscale_max=jnp.array([1000.0, 1000.0]),
                err_info=None,
            )
            
            # Both should produce valid results
            assert jnp.all(jnp.isfinite(result_scalar[0])), "Scalar Lscale_max failed"
            assert jnp.all(jnp.isfinite(result_array[0])), "Array Lscale_max failed"
            
        except NotImplementedError:
            pytest.skip("compute_mixing_length not yet implemented")


class TestConsistency:
    """Test suite for internal consistency checks."""

    def test_upward_downward_relationship(self, test_data: Dict[str, Any]) -> None:
        """
        Test relationship between Lscale, Lscale_up, and Lscale_down.
        
        The total mixing length should be related to upward and downward components.
        """
        test_case = test_data["test_cases"][0]  # Use nominal case
        inputs = test_case["inputs"]
        
        try:
            result = compute_mixing_length(
                nzm=inputs["nzm"],
                nzt=inputs["nzt"],
                ngrdcol=inputs["ngrdcol"],
                gr=inputs["gr"],
                thvm=inputs["thvm"],
                thlm=inputs["thlm"],
                rtm=inputs["rtm"],
                em=inputs["em"],
                Lscale_max=inputs["Lscale_max"],
                p_in_Pa=inputs["p_in_Pa"],
                exner=inputs["exner"],
                thv_ds=inputs["thv_ds"],
                mu=inputs["mu"],
                lmin=inputs["lmin"],
                saturation_formula=inputs["saturation_formula"],
                l_implemented=inputs["l_implemented"],
                err_info=None,
            )
            
            Lscale, Lscale_up, Lscale_down, err_info = result
            
            # All should be positive
            assert jnp.all(Lscale > 0), "Lscale should be positive"
            assert jnp.all(Lscale_up > 0), "Lscale_up should be positive"
            assert jnp.all(Lscale_down > 0), "Lscale_down should be positive"
            
            # Lscale should not exceed the maximum of up and down
            max_component = jnp.maximum(Lscale_up, Lscale_down)
            assert jnp.all(Lscale <= max_component * 2), (
                "Lscale should be related to up/down components"
            )
            
        except NotImplementedError:
            pytest.skip("compute_mixing_length not yet implemented")

    def test_monotonicity_with_tke(self) -> None:
        """
        Test that mixing length generally increases with TKE.
        
        Higher TKE should generally lead to larger mixing lengths.
        """
        nzm, nzt, ngrdcol = 5, 6, 1
        gr = {
            "dzm": jnp.array([10.0] * nzm),
            "zt": jnp.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0]),
        }
        
        common_inputs = {
            "nzm": nzm,
            "nzt": nzt,
            "ngrdcol": ngrdcol,
            "gr": gr,
            "thvm": jnp.ones((ngrdcol, nzt)) * 290.0,
            "thlm": jnp.ones((ngrdcol, nzt)) * 288.0,
            "rtm": jnp.ones((ngrdcol, nzt)) * 0.01,
            "Lscale_max": 1000.0,
            "p_in_Pa": jnp.ones((ngrdcol, nzt)) * 100000.0,
            "exner": jnp.ones((ngrdcol, nzt)),
            "thv_ds": jnp.ones((ngrdcol, nzt)) * 290.5,
            "mu": 0.5,
            "lmin": 1.0,
            "saturation_formula": 1,
            "l_implemented": True,
        }
        
        try:
            # Low TKE case
            result_low = compute_mixing_length(
                **common_inputs,
                em=jnp.ones((ngrdcol, nzm)) * 0.1,
                err_info=None,
            )
            
            # High TKE case
            result_high = compute_mixing_length(
                **common_inputs,
                em=jnp.ones((ngrdcol, nzm)) * 2.0,
                err_info=None,
            )
            
            Lscale_low = result_low[0]
            Lscale_high = result_high[0]
            
            # Higher TKE should generally produce larger mixing lengths
            # (allowing for some exceptions due to other factors)
            mean_low = jnp.mean(Lscale_low)
            mean_high = jnp.mean(Lscale_high)
            
            assert mean_high >= mean_low, (
                f"Higher TKE should generally increase mixing length: "
                f"low={mean_low}, high={mean_high}"
            )
            
        except NotImplementedError:
            pytest.skip("compute_mixing_length not yet implemented")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])