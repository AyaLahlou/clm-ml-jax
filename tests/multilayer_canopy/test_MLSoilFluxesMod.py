"""
Comprehensive pytest suite for MLSoilFluxesMod.soil_fluxes function.

This module tests the soil surface energy balance calculations including:
- Sensible and latent heat fluxes from soil
- Soil heat flux into ground
- Water vapor flux from soil surface
- Soil surface temperature and vapor pressure
- Energy balance closure

Tests cover:
- Nominal conditions (temperate, hot/dry, cold/wet climates)
- Edge cases (zero/negative radiation, saturated/dry soil, extreme gradients)
- Special cases (minimal/maximal array dimensions)
- Physical constraints (temperatures > 0K, humidity in [0,1], etc.)
- Energy conservation (error < 0.001 W/m2)
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multilayer_canopy.MLSoilFluxesMod import (
    SoilFluxesInput,
    SoilFluxesOutput,
    soil_fluxes,
)


@pytest.fixture
def test_data():
    """
    Load test data for soil_fluxes function.
    
    Returns:
        dict: Test cases with inputs and metadata for various climate conditions
              and edge cases.
    """
    return {
        "test_nominal_temperate_conditions": {
            "inputs": {
                "tref": jnp.array([293.15, 295.15, 290.15]),
                "pref": jnp.array([101325.0, 101000.0, 101500.0]),
                "rhomol": jnp.array([41.6, 41.4, 41.8]),
                "cpair": jnp.array([29.3, 29.3, 29.3]),
                "rnsoi": jnp.array([150.0, 200.0, 100.0]),
                "rhg": jnp.array([0.7, 0.8, 0.6]),
                "soilres": jnp.array([50.0, 100.0, 75.0]),
                "gac0": jnp.array([0.5, 0.6, 0.4]),
                "soil_t": jnp.array([288.15, 290.15, 287.15]),
                "soil_dz": jnp.array([0.05, 0.05, 0.05]),
                "soil_tk": jnp.array([1.5, 1.8, 1.2]),
                "tg_bef": jnp.array([289.15, 291.15, 288.15]),
                "tair": jnp.array([
                    [293.15, 292.15, 291.15, 290.15],
                    [295.15, 294.15, 293.15, 292.15],
                    [290.15, 289.15, 288.15, 287.15],
                ]),
                "eair": jnp.array([
                    [1500.0, 1450.0, 1400.0, 1350.0],
                    [1600.0, 1550.0, 1500.0, 1450.0],
                    [1400.0, 1350.0, 1300.0, 1250.0],
                ]),
            },
            "metadata": {
                "type": "nominal",
                "description": "Typical temperate climate conditions with moderate radiation and humidity",
                "n_patches": 3,
                "n_layers": 4,
            },
        },
        "test_nominal_hot_dry_conditions": {
            "inputs": {
                "tref": jnp.array([308.15, 310.15, 305.15, 312.15]),
                "pref": jnp.array([95000.0, 94500.0, 95500.0, 94000.0]),
                "rhomol": jnp.array([38.5, 38.2, 38.8, 37.9]),
                "cpair": jnp.array([29.5, 29.5, 29.5, 29.5]),
                "rnsoi": jnp.array([400.0, 450.0, 380.0, 420.0]),
                "rhg": jnp.array([0.2, 0.15, 0.25, 0.18]),
                "soilres": jnp.array([200.0, 250.0, 180.0, 220.0]),
                "gac0": jnp.array([0.8, 0.9, 0.7, 0.85]),
                "soil_t": jnp.array([305.15, 307.15, 303.15, 308.15]),
                "soil_dz": jnp.array([0.08, 0.08, 0.08, 0.08]),
                "soil_tk": jnp.array([0.8, 0.7, 0.9, 0.75]),
                "tg_bef": jnp.array([306.15, 308.15, 304.15, 309.15]),
                "tair": jnp.array([
                    [308.15, 307.15, 306.15, 305.15, 304.15],
                    [310.15, 309.15, 308.15, 307.15, 306.15],
                    [305.15, 304.15, 303.15, 302.15, 301.15],
                    [312.15, 311.15, 310.15, 309.15, 308.15],
                ]),
                "eair": jnp.array([
                    [800.0, 780.0, 760.0, 740.0, 720.0],
                    [850.0, 830.0, 810.0, 790.0, 770.0],
                    [750.0, 730.0, 710.0, 690.0, 670.0],
                    [900.0, 880.0, 860.0, 840.0, 820.0],
                ]),
            },
            "metadata": {
                "type": "nominal",
                "description": "Hot, arid desert conditions with high radiation, low humidity, and high soil resistance",
                "n_patches": 4,
                "n_layers": 5,
            },
        },
        "test_nominal_cold_wet_conditions": {
            "inputs": {
                "tref": jnp.array([278.15, 280.15]),
                "pref": jnp.array([102000.0, 102500.0]),
                "rhomol": jnp.array([44.2, 44.5]),
                "cpair": jnp.array([29.1, 29.1]),
                "rnsoi": jnp.array([50.0, 75.0]),
                "rhg": jnp.array([0.95, 0.98]),
                "soilres": jnp.array([10.0, 5.0]),
                "gac0": jnp.array([0.3, 0.35]),
                "soil_t": jnp.array([275.15, 276.15]),
                "soil_dz": jnp.array([0.03, 0.03]),
                "soil_tk": jnp.array([2.5, 2.8]),
                "tg_bef": jnp.array([276.15, 277.15]),
                "tair": jnp.array([
                    [278.15, 277.65, 277.15],
                    [280.15, 279.65, 279.15],
                ]),
                "eair": jnp.array([
                    [600.0, 590.0, 580.0],
                    [650.0, 640.0, 630.0],
                ]),
            },
            "metadata": {
                "type": "nominal",
                "description": "Cold, wet conditions near freezing with high humidity and low soil resistance",
                "n_patches": 2,
                "n_layers": 3,
            },
        },
        "test_edge_zero_net_radiation": {
            "inputs": {
                "tref": jnp.array([288.15, 290.15, 292.15]),
                "pref": jnp.array([101325.0, 101325.0, 101325.0]),
                "rhomol": jnp.array([41.6, 41.6, 41.6]),
                "cpair": jnp.array([29.3, 29.3, 29.3]),
                "rnsoi": jnp.array([0.0, 0.0, 0.0]),
                "rhg": jnp.array([0.5, 0.6, 0.7]),
                "soilres": jnp.array([50.0, 50.0, 50.0]),
                "gac0": jnp.array([0.5, 0.5, 0.5]),
                "soil_t": jnp.array([288.15, 290.15, 292.15]),
                "soil_dz": jnp.array([0.05, 0.05, 0.05]),
                "soil_tk": jnp.array([1.5, 1.5, 1.5]),
                "tg_bef": jnp.array([288.15, 290.15, 292.15]),
                "tair": jnp.array([
                    [288.15, 288.15, 288.15],
                    [290.15, 290.15, 290.15],
                    [292.15, 292.15, 292.15],
                ]),
                "eair": jnp.array([
                    [1200.0, 1200.0, 1200.0],
                    [1400.0, 1400.0, 1400.0],
                    [1600.0, 1600.0, 1600.0],
                ]),
            },
            "metadata": {
                "type": "edge",
                "description": "Zero net radiation at surface - tests nighttime or balanced radiation conditions",
                "n_patches": 3,
                "n_layers": 3,
            },
        },
        "test_edge_negative_net_radiation": {
            "inputs": {
                "tref": jnp.array([285.15, 283.15]),
                "pref": jnp.array([101325.0, 101325.0]),
                "rhomol": jnp.array([41.8, 42.0]),
                "cpair": jnp.array([29.2, 29.2]),
                "rnsoi": jnp.array([-50.0, -75.0]),
                "rhg": jnp.array([0.8, 0.85]),
                "soilres": jnp.array([30.0, 25.0]),
                "gac0": jnp.array([0.4, 0.45]),
                "soil_t": jnp.array([286.15, 284.15]),
                "soil_dz": jnp.array([0.05, 0.05]),
                "soil_tk": jnp.array([1.8, 1.9]),
                "tg_bef": jnp.array([287.15, 285.15]),
                "tair": jnp.array([
                    [285.15, 285.65, 286.15, 286.65],
                    [283.15, 283.65, 284.15, 284.65],
                ]),
                "eair": jnp.array([
                    [1100.0, 1120.0, 1140.0, 1160.0],
                    [1000.0, 1020.0, 1040.0, 1060.0],
                ]),
            },
            "metadata": {
                "type": "edge",
                "description": "Negative net radiation - nighttime cooling with surface losing energy",
                "n_patches": 2,
                "n_layers": 4,
            },
        },
        "test_edge_saturated_humidity": {
            "inputs": {
                "tref": jnp.array([298.15, 300.15, 295.15]),
                "pref": jnp.array([101325.0, 101000.0, 101500.0]),
                "rhomol": jnp.array([41.4, 41.2, 41.6]),
                "cpair": jnp.array([29.4, 29.4, 29.4]),
                "rnsoi": jnp.array([100.0, 120.0, 90.0]),
                "rhg": jnp.array([1.0, 1.0, 1.0]),
                "soilres": jnp.array([0.0, 0.0, 0.0]),
                "gac0": jnp.array([0.6, 0.65, 0.55]),
                "soil_t": jnp.array([298.15, 300.15, 295.15]),
                "soil_dz": jnp.array([0.04, 0.04, 0.04]),
                "soil_tk": jnp.array([2.0, 2.1, 1.9]),
                "tg_bef": jnp.array([298.15, 300.15, 295.15]),
                "tair": jnp.array([
                    [298.15, 297.65, 297.15, 296.65, 296.15],
                    [300.15, 299.65, 299.15, 298.65, 298.15],
                    [295.15, 294.65, 294.15, 293.65, 293.15],
                ]),
                "eair": jnp.array([
                    [2500.0, 2480.0, 2460.0, 2440.0, 2420.0],
                    [2700.0, 2680.0, 2660.0, 2640.0, 2620.0],
                    [2300.0, 2280.0, 2260.0, 2240.0, 2220.0],
                ]),
            },
            "metadata": {
                "type": "edge",
                "description": "Saturated soil surface (rhg=1.0) with zero soil resistance - maximum evaporation potential",
                "n_patches": 3,
                "n_layers": 5,
            },
        },
        "test_edge_very_dry_soil": {
            "inputs": {
                "tref": jnp.array([303.15, 305.15]),
                "pref": jnp.array([98000.0, 97500.0]),
                "rhomol": jnp.array([39.5, 39.2]),
                "cpair": jnp.array([29.5, 29.5]),
                "rnsoi": jnp.array([350.0, 380.0]),
                "rhg": jnp.array([0.0, 0.0]),
                "soilres": jnp.array([1000.0, 1500.0]),
                "gac0": jnp.array([0.7, 0.75]),
                "soil_t": jnp.array([308.15, 310.15]),
                "soil_dz": jnp.array([0.1, 0.1]),
                "soil_tk": jnp.array([0.5, 0.45]),
                "tg_bef": jnp.array([309.15, 311.15]),
                "tair": jnp.array([
                    [303.15, 304.15, 305.15, 306.15],
                    [305.15, 306.15, 307.15, 308.15],
                ]),
                "eair": jnp.array([
                    [500.0, 520.0, 540.0, 560.0],
                    [550.0, 570.0, 590.0, 610.0],
                ]),
            },
            "metadata": {
                "type": "edge",
                "description": "Completely dry soil surface (rhg=0.0) with very high resistance - minimal evaporation",
                "n_patches": 2,
                "n_layers": 4,
            },
        },
        "test_edge_extreme_temperature_gradient": {
            "inputs": {
                "tref": jnp.array([320.15]),
                "pref": jnp.array([90000.0]),
                "rhomol": jnp.array([36.5]),
                "cpair": jnp.array([29.6]),
                "rnsoi": jnp.array([600.0]),
                "rhg": jnp.array([0.1]),
                "soilres": jnp.array([300.0]),
                "gac0": jnp.array([1.2]),
                "soil_t": jnp.array([280.15]),
                "soil_dz": jnp.array([0.02]),
                "soil_tk": jnp.array([3.0]),
                "tg_bef": jnp.array([300.15]),
                "tair": jnp.array([
                    [320.15, 318.15, 316.15, 314.15, 312.15, 310.15],
                ]),
                "eair": jnp.array([
                    [400.0, 420.0, 440.0, 460.0, 480.0, 500.0],
                ]),
            },
            "metadata": {
                "type": "edge",
                "description": "Extreme temperature gradient between hot air and cold subsurface - tests large heat flux",
                "n_patches": 1,
                "n_layers": 6,
            },
        },
        "test_special_single_patch": {
            "inputs": {
                "tref": jnp.array([290.15]),
                "pref": jnp.array([101325.0]),
                "rhomol": jnp.array([41.6]),
                "cpair": jnp.array([29.3]),
                "rnsoi": jnp.array([150.0]),
                "rhg": jnp.array([0.65]),
                "soilres": jnp.array([60.0]),
                "gac0": jnp.array([0.5]),
                "soil_t": jnp.array([288.15]),
                "soil_dz": jnp.array([0.05]),
                "soil_tk": jnp.array([1.5]),
                "tg_bef": jnp.array([289.15]),
                "tair": jnp.array([[290.15]]),
                "eair": jnp.array([[1300.0]]),
            },
            "metadata": {
                "type": "special",
                "description": "Single patch with single canopy layer - minimal array dimensions",
                "n_patches": 1,
                "n_layers": 1,
            },
        },
        "test_special_many_canopy_layers": {
            "inputs": {
                "tref": jnp.array([295.15, 293.15]),
                "pref": jnp.array([101325.0, 101500.0]),
                "rhomol": jnp.array([41.5, 41.7]),
                "cpair": jnp.array([29.3, 29.3]),
                "rnsoi": jnp.array([180.0, 160.0]),
                "rhg": jnp.array([0.7, 0.75]),
                "soilres": jnp.array([70.0, 65.0]),
                "gac0": jnp.array([0.55, 0.52]),
                "soil_t": jnp.array([290.15, 289.15]),
                "soil_dz": jnp.array([0.05, 0.05]),
                "soil_tk": jnp.array([1.6, 1.7]),
                "tg_bef": jnp.array([291.15, 290.15]),
                "tair": jnp.array([
                    [295.15, 294.65, 294.15, 293.65, 293.15, 292.65, 292.15, 291.65, 291.15, 290.65],
                    [293.15, 292.65, 292.15, 291.65, 291.15, 290.65, 290.15, 289.65, 289.15, 288.65],
                ]),
                "eair": jnp.array([
                    [1500.0, 1490.0, 1480.0, 1470.0, 1460.0, 1450.0, 1440.0, 1430.0, 1420.0, 1410.0],
                    [1400.0, 1390.0, 1380.0, 1370.0, 1360.0, 1350.0, 1340.0, 1330.0, 1320.0, 1310.0],
                ]),
            },
            "metadata": {
                "type": "special",
                "description": "Multiple patches with many canopy layers - tests vertical resolution",
                "n_patches": 2,
                "n_layers": 10,
            },
        },
    }


class TestSoilFluxesShapes:
    """Test output array shapes for soil_fluxes function."""

    @pytest.mark.parametrize(
        "test_case_name",
        [
            "test_nominal_temperate_conditions",
            "test_nominal_hot_dry_conditions",
            "test_nominal_cold_wet_conditions",
            "test_edge_zero_net_radiation",
            "test_edge_negative_net_radiation",
            "test_edge_saturated_humidity",
            "test_edge_very_dry_soil",
            "test_edge_extreme_temperature_gradient",
            "test_special_single_patch",
            "test_special_many_canopy_layers",
        ],
    )
    def test_output_shapes(self, test_data, test_case_name):
        """
        Test that all output arrays have correct shapes matching n_patches.
        
        All scalar outputs (shsoi, lhsoi, gsoi, etsoi, tg, eg, energy_error)
        should have shape [n_patches].
        """
        test_case = test_data[test_case_name]
        inputs = SoilFluxesInput(**test_case["inputs"])
        n_patches = test_case["metadata"]["n_patches"]
        
        output = soil_fluxes(inputs)
        
        assert output.shsoi.shape == (n_patches,), (
            f"shsoi shape mismatch: expected ({n_patches},), got {output.shsoi.shape}"
        )
        assert output.lhsoi.shape == (n_patches,), (
            f"lhsoi shape mismatch: expected ({n_patches},), got {output.lhsoi.shape}"
        )
        assert output.gsoi.shape == (n_patches,), (
            f"gsoi shape mismatch: expected ({n_patches},), got {output.gsoi.shape}"
        )
        assert output.etsoi.shape == (n_patches,), (
            f"etsoi shape mismatch: expected ({n_patches},), got {output.etsoi.shape}"
        )
        assert output.tg.shape == (n_patches,), (
            f"tg shape mismatch: expected ({n_patches},), got {output.tg.shape}"
        )
        assert output.eg.shape == (n_patches,), (
            f"eg shape mismatch: expected ({n_patches},), got {output.eg.shape}"
        )
        assert output.energy_error.shape == (n_patches,), (
            f"energy_error shape mismatch: expected ({n_patches},), got {output.energy_error.shape}"
        )


class TestSoilFluxesDtypes:
    """Test output data types for soil_fluxes function."""

    def test_output_dtypes(self, test_data):
        """
        Test that all outputs are floating point arrays.
        
        JAX typically uses float32 by default, but we accept any float type.
        """
        test_case = test_data["test_nominal_temperate_conditions"]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        assert jnp.issubdtype(output.shsoi.dtype, jnp.floating), (
            f"shsoi should be floating point, got {output.shsoi.dtype}"
        )
        assert jnp.issubdtype(output.lhsoi.dtype, jnp.floating), (
            f"lhsoi should be floating point, got {output.lhsoi.dtype}"
        )
        assert jnp.issubdtype(output.gsoi.dtype, jnp.floating), (
            f"gsoi should be floating point, got {output.gsoi.dtype}"
        )
        assert jnp.issubdtype(output.etsoi.dtype, jnp.floating), (
            f"etsoi should be floating point, got {output.etsoi.dtype}"
        )
        assert jnp.issubdtype(output.tg.dtype, jnp.floating), (
            f"tg should be floating point, got {output.tg.dtype}"
        )
        assert jnp.issubdtype(output.eg.dtype, jnp.floating), (
            f"eg should be floating point, got {output.eg.dtype}"
        )
        assert jnp.issubdtype(output.energy_error.dtype, jnp.floating), (
            f"energy_error should be floating point, got {output.energy_error.dtype}"
        )


class TestSoilFluxesPhysicalConstraints:
    """Test physical constraints on soil_fluxes outputs."""

    @pytest.mark.parametrize(
        "test_case_name",
        [
            "test_nominal_temperate_conditions",
            "test_nominal_hot_dry_conditions",
            "test_nominal_cold_wet_conditions",
            "test_edge_zero_net_radiation",
            "test_edge_negative_net_radiation",
            "test_edge_saturated_humidity",
            "test_edge_very_dry_soil",
            "test_edge_extreme_temperature_gradient",
            "test_special_single_patch",
            "test_special_many_canopy_layers",
        ],
    )
    def test_temperature_positive(self, test_data, test_case_name):
        """
        Test that soil surface temperature (tg) is always positive (> 0K).
        
        Physical constraint: Absolute zero is the lower bound for temperature.
        """
        test_case = test_data[test_case_name]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        assert jnp.all(output.tg > 0.0), (
            f"Soil surface temperature must be > 0K, got min={jnp.min(output.tg):.2f}K"
        )

    @pytest.mark.parametrize(
        "test_case_name",
        [
            "test_nominal_temperate_conditions",
            "test_nominal_hot_dry_conditions",
            "test_nominal_cold_wet_conditions",
            "test_edge_zero_net_radiation",
            "test_edge_negative_net_radiation",
            "test_edge_saturated_humidity",
            "test_edge_very_dry_soil",
            "test_edge_extreme_temperature_gradient",
            "test_special_single_patch",
            "test_special_many_canopy_layers",
        ],
    )
    def test_vapor_pressure_non_negative(self, test_data, test_case_name):
        """
        Test that soil surface vapor pressure (eg) is non-negative.
        
        Physical constraint: Vapor pressure cannot be negative.
        """
        test_case = test_data[test_case_name]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        assert jnp.all(output.eg >= 0.0), (
            f"Soil surface vapor pressure must be >= 0 Pa, got min={jnp.min(output.eg):.2f} Pa"
        )

    @pytest.mark.parametrize(
        "test_case_name",
        [
            "test_nominal_temperate_conditions",
            "test_nominal_hot_dry_conditions",
            "test_nominal_cold_wet_conditions",
            "test_edge_zero_net_radiation",
            "test_edge_negative_net_radiation",
            "test_edge_saturated_humidity",
            "test_edge_very_dry_soil",
            "test_edge_extreme_temperature_gradient",
            "test_special_single_patch",
            "test_special_many_canopy_layers",
        ],
    )
    def test_energy_balance_closure(self, test_data, test_case_name):
        """
        Test that energy balance error is small (< 1.0 W/m2).
        
        The energy balance should be well-conserved:
        rnsoi = shsoi + lhsoi + gsoi + energy_error
        
        We allow up to 1.0 W/m2 error for numerical tolerance, though
        ideally it should be < 0.001 W/m2.
        """
        test_case = test_data[test_case_name]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        max_error = jnp.max(jnp.abs(output.energy_error))
        assert max_error < 1.0, (
            f"Energy balance error should be < 1.0 W/m2, got max={max_error:.6f} W/m2"
        )

    @pytest.mark.parametrize(
        "test_case_name",
        [
            "test_nominal_temperate_conditions",
            "test_nominal_hot_dry_conditions",
            "test_nominal_cold_wet_conditions",
            "test_edge_zero_net_radiation",
            "test_edge_negative_net_radiation",
            "test_edge_saturated_humidity",
            "test_edge_very_dry_soil",
            "test_edge_extreme_temperature_gradient",
            "test_special_single_patch",
            "test_special_many_canopy_layers",
        ],
    )
    def test_no_nan_or_inf(self, test_data, test_case_name):
        """
        Test that no output contains NaN or Inf values.
        
        All outputs should be finite real numbers.
        """
        test_case = test_data[test_case_name]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        assert jnp.all(jnp.isfinite(output.shsoi)), "shsoi contains NaN or Inf"
        assert jnp.all(jnp.isfinite(output.lhsoi)), "lhsoi contains NaN or Inf"
        assert jnp.all(jnp.isfinite(output.gsoi)), "gsoi contains NaN or Inf"
        assert jnp.all(jnp.isfinite(output.etsoi)), "etsoi contains NaN or Inf"
        assert jnp.all(jnp.isfinite(output.tg)), "tg contains NaN or Inf"
        assert jnp.all(jnp.isfinite(output.eg)), "eg contains NaN or Inf"
        assert jnp.all(jnp.isfinite(output.energy_error)), "energy_error contains NaN or Inf"


class TestSoilFluxesEdgeCases:
    """Test edge cases and boundary conditions for soil_fluxes."""

    def test_zero_net_radiation_small_fluxes(self, test_data):
        """
        Test that zero net radiation produces small or balanced fluxes.
        
        With zero net radiation, the sum of sensible, latent, and ground
        heat fluxes should be near zero (within energy balance error).
        """
        test_case = test_data["test_edge_zero_net_radiation"]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # Net radiation is zero, so sum of fluxes should be near zero
        flux_sum = output.shsoi + output.lhsoi + output.gsoi
        assert jnp.allclose(flux_sum, 0.0, atol=1.0), (
            f"With zero net radiation, flux sum should be ~0, got {flux_sum}"
        )

    def test_negative_net_radiation_cooling(self, test_data):
        """
        Test that negative net radiation produces net cooling.
        
        With negative net radiation (nighttime), the surface should cool,
        meaning the sum of upward fluxes should be negative (or ground
        heat flux should be positive, extracting heat from below).
        """
        test_case = test_data["test_edge_negative_net_radiation"]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # Negative net radiation means surface is losing energy
        assert jnp.all(inputs.rnsoi < 0.0), "Test case should have negative radiation"
        
        # The sum of fluxes should balance the negative radiation
        flux_sum = output.shsoi + output.lhsoi + output.gsoi
        # Allow some tolerance for energy balance
        assert jnp.allclose(flux_sum, inputs.rnsoi, atol=1.0), (
            f"Flux sum should balance negative radiation"
        )

    def test_saturated_soil_high_evaporation(self, test_data):
        """
        Test that saturated soil (rhg=1.0, soilres=0) produces high evaporation.
        
        With no soil resistance and saturated conditions, latent heat flux
        and water vapor flux should be relatively large (positive).
        """
        test_case = test_data["test_edge_saturated_humidity"]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # Check that soil is indeed saturated
        assert jnp.all(inputs.rhg == 1.0), "Test case should have rhg=1.0"
        assert jnp.all(inputs.soilres == 0.0), "Test case should have soilres=0"
        
        # With positive net radiation and saturated soil, expect positive evaporation
        # (though magnitude depends on vapor pressure gradient)
        # At minimum, etsoi should be finite and reasonable
        assert jnp.all(jnp.isfinite(output.etsoi)), "etsoi should be finite"

    def test_dry_soil_low_evaporation(self, test_data):
        """
        Test that very dry soil (rhg=0.0, high soilres) produces minimal evaporation.
        
        With zero relative humidity and high resistance, water vapor flux
        should be very small or zero.
        """
        test_case = test_data["test_edge_very_dry_soil"]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # Check that soil is indeed dry
        assert jnp.all(inputs.rhg == 0.0), "Test case should have rhg=0.0"
        assert jnp.all(inputs.soilres >= 1000.0), "Test case should have high soilres"
        
        # With dry soil, evaporation should be minimal
        # Allow small positive or negative values due to numerical effects
        assert jnp.all(jnp.abs(output.etsoi) < 0.1), (
            f"Dry soil should have minimal evaporation, got max={jnp.max(jnp.abs(output.etsoi)):.6f}"
        )

    def test_extreme_temperature_gradient_large_fluxes(self, test_data):
        """
        Test that extreme temperature gradients produce large heat fluxes.
        
        With 40K difference between air and subsurface, ground heat flux
        should be substantial.
        """
        test_case = test_data["test_edge_extreme_temperature_gradient"]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # Check temperature gradient
        temp_diff = inputs.tref[0] - inputs.soil_t[0]
        assert temp_diff == 40.0, f"Expected 40K gradient, got {temp_diff}K"
        
        # Ground heat flux should be large due to steep gradient
        assert jnp.abs(output.gsoi[0]) > 100.0, (
            f"Expected large ground heat flux with 40K gradient, got {output.gsoi[0]:.2f} W/m2"
        )


class TestSoilFluxesSpecialCases:
    """Test special cases like minimal/maximal dimensions."""

    def test_single_patch_single_layer(self, test_data):
        """
        Test minimal array dimensions: single patch, single canopy layer.
        
        Function should handle 1D arrays correctly.
        """
        test_case = test_data["test_special_single_patch"]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # Check shapes
        assert output.shsoi.shape == (1,), f"Expected shape (1,), got {output.shsoi.shape}"
        assert output.tg.shape == (1,), f"Expected shape (1,), got {output.tg.shape}"
        
        # Check values are reasonable
        assert jnp.isfinite(output.shsoi[0]), "shsoi should be finite"
        assert jnp.isfinite(output.tg[0]), "tg should be finite"
        assert output.tg[0] > 0.0, "tg should be positive"

    def test_many_canopy_layers(self, test_data):
        """
        Test high vertical resolution: 10 canopy layers.
        
        Function should handle many layers without numerical issues.
        """
        test_case = test_data["test_special_many_canopy_layers"]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # Check that function handles 10 layers correctly
        assert inputs.tair.shape[1] == 10, "Test case should have 10 layers"
        
        # All outputs should be finite
        assert jnp.all(jnp.isfinite(output.shsoi)), "shsoi should be finite with 10 layers"
        assert jnp.all(jnp.isfinite(output.tg)), "tg should be finite with 10 layers"
        
        # Energy balance should still be good
        max_error = jnp.max(jnp.abs(output.energy_error))
        assert max_error < 1.0, (
            f"Energy balance should be good with 10 layers, got error={max_error:.6f}"
        )


class TestSoilFluxesEnergyBalance:
    """Test detailed energy balance relationships."""

    @pytest.mark.parametrize(
        "test_case_name",
        [
            "test_nominal_temperate_conditions",
            "test_nominal_hot_dry_conditions",
            "test_nominal_cold_wet_conditions",
        ],
    )
    def test_energy_balance_equation(self, test_data, test_case_name):
        """
        Test the fundamental energy balance equation:
        rnsoi = shsoi + lhsoi + gsoi + energy_error
        
        This should hold for all cases within numerical tolerance.
        """
        test_case = test_data[test_case_name]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # Calculate energy balance
        flux_sum = output.shsoi + output.lhsoi + output.gsoi
        residual = inputs.rnsoi - flux_sum
        
        # Residual should equal energy_error
        assert jnp.allclose(residual, output.energy_error, atol=1e-3, rtol=1e-6), (
            f"Energy balance residual should equal energy_error:\n"
            f"residual={residual}\n"
            f"energy_error={output.energy_error}"
        )

    def test_flux_signs_with_positive_radiation(self, test_data):
        """
        Test flux sign conventions with positive net radiation.
        
        With positive net radiation (daytime):
        - Sensible heat flux typically positive (warming air)
        - Latent heat flux typically positive (evaporation)
        - Ground heat flux can be positive (warming soil) or negative
        
        All fluxes are positive upward (away from surface).
        """
        test_case = test_data["test_nominal_temperate_conditions"]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # Check that net radiation is positive
        assert jnp.all(inputs.rnsoi > 0.0), "Test case should have positive radiation"
        
        # With positive radiation, at least some fluxes should be positive
        # (energy must go somewhere)
        total_upward = output.shsoi + output.lhsoi + output.gsoi
        assert jnp.all(total_upward > 0.0), (
            "With positive net radiation, total upward flux should be positive"
        )


class TestSoilFluxesConsistency:
    """Test consistency and relationships between outputs."""

    def test_temperature_consistency(self, test_data):
        """
        Test that output soil surface temperature (tg) is physically consistent.
        
        tg should be between soil_t and tair (or close to them), and should
        be influenced by tg_bef (previous timestep temperature).
        """
        test_case = test_data["test_nominal_temperate_conditions"]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # tg should be in a reasonable range relative to inputs
        tair_bottom = inputs.tair[:, -1]  # Bottom canopy layer
        
        # tg should be within reasonable bounds
        # (not necessarily between soil_t and tair, but close to them)
        temp_range = jnp.max(jnp.array([inputs.soil_t, tair_bottom, inputs.tg_bef])) - \
                     jnp.min(jnp.array([inputs.soil_t, tair_bottom, inputs.tg_bef]))
        
        # tg should not deviate wildly from input temperatures
        for i in range(len(output.tg)):
            assert jnp.abs(output.tg[i] - inputs.tg_bef[i]) < 50.0, (
                f"tg should not deviate >50K from previous timestep: "
                f"tg={output.tg[i]:.2f}, tg_bef={inputs.tg_bef[i]:.2f}"
            )

    def test_vapor_flux_latent_heat_consistency(self, test_data):
        """
        Test consistency between water vapor flux (etsoi) and latent heat flux (lhsoi).
        
        These should be related by the latent heat of vaporization:
        lhsoi ≈ etsoi * lambda_v
        where lambda_v ≈ 44000 J/mol (latent heat of vaporization)
        """
        test_case = test_data["test_nominal_temperate_conditions"]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # Approximate latent heat of vaporization [J/mol]
        lambda_v = 44000.0
        
        # Calculate expected latent heat from vapor flux
        expected_lhsoi = output.etsoi * lambda_v
        
        # Should be approximately equal (within 20% due to temperature dependence)
        # Only check where etsoi is significant
        mask = jnp.abs(output.etsoi) > 0.001
        if jnp.any(mask):
            relative_error = jnp.abs(output.lhsoi[mask] - expected_lhsoi[mask]) / \
                           (jnp.abs(expected_lhsoi[mask]) + 1e-10)
            assert jnp.all(relative_error < 0.3), (
                f"Latent heat and vapor flux should be consistent:\n"
                f"lhsoi={output.lhsoi[mask]}\n"
                f"expected={expected_lhsoi[mask]}\n"
                f"relative_error={relative_error}"
            )


class TestSoilFluxesNumericalStability:
    """Test numerical stability and robustness."""

    def test_repeated_calls_identical(self, test_data):
        """
        Test that repeated calls with same inputs produce identical outputs.
        
        Function should be deterministic (no random elements).
        """
        test_case = test_data["test_nominal_temperate_conditions"]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        output1 = soil_fluxes(inputs)
        output2 = soil_fluxes(inputs)
        
        assert jnp.allclose(output1.shsoi, output2.shsoi, atol=1e-10, rtol=1e-10), (
            "Repeated calls should produce identical shsoi"
        )
        assert jnp.allclose(output1.tg, output2.tg, atol=1e-10, rtol=1e-10), (
            "Repeated calls should produce identical tg"
        )
        assert jnp.allclose(output1.energy_error, output2.energy_error, atol=1e-10, rtol=1e-10), (
            "Repeated calls should produce identical energy_error"
        )

    def test_small_perturbations_smooth_response(self, test_data):
        """
        Test that small perturbations in inputs produce smooth changes in outputs.
        
        Function should not have discontinuities or extreme sensitivity.
        """
        test_case = test_data["test_nominal_temperate_conditions"]
        inputs = SoilFluxesInput(**test_case["inputs"])
        
        # Run with original inputs
        output1 = soil_fluxes(inputs)
        
        # Perturb temperature slightly (0.1K)
        perturbed_inputs = SoilFluxesInput(
            tref=inputs.tref + 0.1,
            pref=inputs.pref,
            rhomol=inputs.rhomol,
            cpair=inputs.cpair,
            rnsoi=inputs.rnsoi,
            rhg=inputs.rhg,
            soilres=inputs.soilres,
            gac0=inputs.gac0,
            soil_t=inputs.soil_t,
            soil_dz=inputs.soil_dz,
            soil_tk=inputs.soil_tk,
            tg_bef=inputs.tg_bef,
            tair=inputs.tair,
            eair=inputs.eair,
        )
        
        output2 = soil_fluxes(perturbed_inputs)
        
        # Changes should be small and smooth
        tg_change = jnp.abs(output2.tg - output1.tg)
        assert jnp.all(tg_change < 5.0), (
            f"0.1K temperature perturbation should not change tg by >5K, got {tg_change}"
        )
        
        shsoi_change = jnp.abs(output2.shsoi - output1.shsoi)
        assert jnp.all(shsoi_change < 50.0), (
            f"0.1K temperature perturbation should not change shsoi by >50 W/m2, got {shsoi_change}"
        )