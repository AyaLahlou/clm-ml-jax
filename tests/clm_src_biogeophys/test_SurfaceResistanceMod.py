"""
Comprehensive pytest suite for calc_soilevap_resis function from SurfaceResistanceMod.

This module tests the calculation of soil evaporative resistance and dry surface layer
thickness for land surface modeling. Tests cover nominal conditions, edge cases, and
special configurations.

Test Coverage:
- Nominal cases: Typical field conditions with varied moisture and temperature
- Edge cases: Dry soil, saturated soil, frozen soil, extreme soil properties
- Special cases: Empty filters, single layers, extreme gradients
- Shape validation: Output array dimensions
- Data type validation: JAX array types
- Value validation: Physical realism and numerical accuracy
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_biogeophys.SurfaceResistanceMod import calc_soilevap_resis


# Define NamedTuples matching the function signature
class BoundsType(NamedTuple):
    """Column bounds containing begc and endc indices."""
    begc: int
    endc: int
    begg: int
    endg: int
    begp: int
    endp: int


class SoilStateType(NamedTuple):
    """Soil state containing hydraulic properties and resistance variables."""
    dsl_col: jnp.ndarray  # (ncols,)
    soilresis_col: jnp.ndarray  # (ncols,)
    watsat: jnp.ndarray  # (ncols, nlevs)
    sucsat: jnp.ndarray  # (ncols, nlevs)
    bsw: jnp.ndarray  # (ncols, nlevs)


class WaterStateType(NamedTuple):
    """Water state variables (ice and liquid water content)."""
    h2osoi_ice: jnp.ndarray  # (ncols, nlevs)
    h2osoi_liq: jnp.ndarray  # (ncols, nlevs)


class TemperatureType(NamedTuple):
    """Temperature variables (soil temperature)."""
    t_soisno: jnp.ndarray  # (ncols, nlevs)


class ColumnType(NamedTuple):
    """Column geometry (layer thickness)."""
    dz: jnp.ndarray  # (ncols, nlevs)


@pytest.fixture
def test_data():
    """
    Load and prepare test data for calc_soilevap_resis tests.
    
    Returns:
        dict: Test cases with inputs and metadata
    """
    return {
        "test_nominal_single_column_moderate_moisture": {
            "inputs": {
                "bounds": BoundsType(begc=0, endc=1, begg=0, endg=1, begp=0, endp=1),
                "num_nolakec": 1,
                "filter_nolakec": jnp.array([0]),
                "soilstate_inst": SoilStateType(
                    dsl_col=jnp.array([50.0]),
                    soilresis_col=jnp.array([5000.0]),
                    watsat=jnp.array([[0.45, 0.43, 0.41, 0.39, 0.37]]),
                    sucsat=jnp.array([[200.0, 220.0, 240.0, 260.0, 280.0]]),
                    bsw=jnp.array([[4.5, 4.8, 5.1, 5.4, 5.7]]),
                ),
                "waterstate_inst": WaterStateType(
                    h2osoi_ice=jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
                    h2osoi_liq=jnp.array([[25.0, 30.0, 35.0, 40.0, 45.0]]),
                ),
                "temperature_inst": TemperatureType(
                    t_soisno=jnp.array([[285.0, 284.0, 283.0, 282.0, 281.0]]),
                ),
                "col": ColumnType(
                    dz=jnp.array([[0.02, 0.04, 0.06, 0.08, 0.1]]),
                ),
            },
            "metadata": {
                "type": "nominal",
                "description": "Single column with moderate soil moisture",
            },
        },
        "test_nominal_multiple_columns_varied_conditions": {
            "inputs": {
                "bounds": BoundsType(begc=0, endc=3, begg=0, endg=1, begp=0, endp=3),
                "num_nolakec": 3,
                "filter_nolakec": jnp.array([0, 1, 2]),
                "soilstate_inst": SoilStateType(
                    dsl_col=jnp.array([30.0, 75.0, 120.0]),
                    soilresis_col=jnp.array([3000.0, 8000.0, 15000.0]),
                    watsat=jnp.array([
                        [0.5, 0.48, 0.46, 0.44],
                        [0.4, 0.38, 0.36, 0.34],
                        [0.35, 0.33, 0.31, 0.29],
                    ]),
                    sucsat=jnp.array([
                        [150.0, 180.0, 210.0, 240.0],
                        [250.0, 280.0, 310.0, 340.0],
                        [300.0, 330.0, 360.0, 390.0],
                    ]),
                    bsw=jnp.array([
                        [3.5, 4.0, 4.5, 5.0],
                        [5.5, 6.0, 6.5, 7.0],
                        [7.5, 8.0, 8.5, 9.0],
                    ]),
                ),
                "waterstate_inst": WaterStateType(
                    h2osoi_ice=jnp.array([
                        [0.0, 0.0, 0.0, 0.0],
                        [5.0, 8.0, 10.0, 12.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]),
                    h2osoi_liq=jnp.array([
                        [40.0, 45.0, 50.0, 55.0],
                        [20.0, 25.0, 30.0, 35.0],
                        [10.0, 15.0, 20.0, 25.0],
                    ]),
                ),
                "temperature_inst": TemperatureType(
                    t_soisno=jnp.array([
                        [290.0, 289.0, 288.0, 287.0],
                        [275.0, 274.0, 273.5, 273.0],
                        [295.0, 294.0, 293.0, 292.0],
                    ]),
                ),
                "col": ColumnType(
                    dz=jnp.array([
                        [0.025, 0.05, 0.075, 0.1],
                        [0.03, 0.06, 0.09, 0.12],
                        [0.02, 0.04, 0.06, 0.08],
                    ]),
                ),
            },
            "metadata": {
                "type": "nominal",
                "description": "Three columns with varied conditions",
            },
        },
        "test_edge_dry_soil_zero_liquid_water": {
            "inputs": {
                "bounds": BoundsType(begc=0, endc=2, begg=0, endg=1, begp=0, endp=2),
                "num_nolakec": 2,
                "filter_nolakec": jnp.array([0, 1]),
                "soilstate_inst": SoilStateType(
                    dsl_col=jnp.array([150.0, 180.0]),
                    soilresis_col=jnp.array([50000.0, 80000.0]),
                    watsat=jnp.array([
                        [0.45, 0.43, 0.41],
                        [0.4, 0.38, 0.36],
                    ]),
                    sucsat=jnp.array([
                        [200.0, 220.0, 240.0],
                        [250.0, 270.0, 290.0],
                    ]),
                    bsw=jnp.array([
                        [4.5, 5.0, 5.5],
                        [5.5, 6.0, 6.5],
                    ]),
                ),
                "waterstate_inst": WaterStateType(
                    h2osoi_ice=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                    h2osoi_liq=jnp.array([[0.0, 0.0, 0.0], [0.1, 0.05, 0.01]]),
                ),
                "temperature_inst": TemperatureType(
                    t_soisno=jnp.array([
                        [305.0, 303.0, 301.0],
                        [310.0, 308.0, 306.0],
                    ]),
                ),
                "col": ColumnType(
                    dz=jnp.array([
                        [0.03, 0.06, 0.09],
                        [0.025, 0.05, 0.075],
                    ]),
                ),
            },
            "metadata": {
                "type": "edge",
                "description": "Extremely dry soil with zero or near-zero liquid water",
            },
        },
        "test_edge_saturated_soil_maximum_moisture": {
            "inputs": {
                "bounds": BoundsType(begc=0, endc=1, begg=0, endg=1, begp=0, endp=1),
                "num_nolakec": 1,
                "filter_nolakec": jnp.array([0]),
                "soilstate_inst": SoilStateType(
                    dsl_col=jnp.array([0.0]),
                    soilresis_col=jnp.array([0.0]),
                    watsat=jnp.array([[0.55, 0.53, 0.51, 0.49]]),
                    sucsat=jnp.array([[100.0, 120.0, 140.0, 160.0]]),
                    bsw=jnp.array([[3.0, 3.5, 4.0, 4.5]]),
                ),
                "waterstate_inst": WaterStateType(
                    h2osoi_ice=jnp.array([[0.0, 0.0, 0.0, 0.0]]),
                    h2osoi_liq=jnp.array([[110.0, 106.0, 102.0, 98.0]]),
                ),
                "temperature_inst": TemperatureType(
                    t_soisno=jnp.array([[285.0, 284.5, 284.0, 283.5]]),
                ),
                "col": ColumnType(
                    dz=jnp.array([[0.02, 0.04, 0.06, 0.08]]),
                ),
            },
            "metadata": {
                "type": "edge",
                "description": "Saturated soil with maximum liquid water content",
            },
        },
        "test_edge_frozen_soil_all_ice": {
            "inputs": {
                "bounds": BoundsType(begc=0, endc=2, begg=0, endg=1, begp=0, endp=2),
                "num_nolakec": 2,
                "filter_nolakec": jnp.array([0, 1]),
                "soilstate_inst": SoilStateType(
                    dsl_col=jnp.array([100.0, 120.0]),
                    soilresis_col=jnp.array([25000.0, 35000.0]),
                    watsat=jnp.array([
                        [0.48, 0.46, 0.44],
                        [0.42, 0.4, 0.38],
                    ]),
                    sucsat=jnp.array([
                        [180.0, 200.0, 220.0],
                        [220.0, 240.0, 260.0],
                    ]),
                    bsw=jnp.array([
                        [4.2, 4.7, 5.2],
                        [5.2, 5.7, 6.2],
                    ]),
                ),
                "waterstate_inst": WaterStateType(
                    h2osoi_ice=jnp.array([
                        [50.0, 55.0, 60.0],
                        [45.0, 50.0, 55.0],
                    ]),
                    h2osoi_liq=jnp.array([
                        [0.0, 0.0, 0.0],
                        [0.5, 0.2, 0.1],
                    ]),
                ),
                "temperature_inst": TemperatureType(
                    t_soisno=jnp.array([
                        [268.0, 267.0, 266.0],
                        [270.0, 269.0, 268.0],
                    ]),
                ),
                "col": ColumnType(
                    dz=jnp.array([
                        [0.03, 0.06, 0.09],
                        [0.025, 0.05, 0.075],
                    ]),
                ),
            },
            "metadata": {
                "type": "edge",
                "description": "Frozen soil with all or nearly all water as ice",
            },
        },
        "test_special_empty_filter_no_processing": {
            "inputs": {
                "bounds": BoundsType(begc=0, endc=3, begg=0, endg=1, begp=0, endp=3),
                "num_nolakec": 0,
                "filter_nolakec": jnp.array([], dtype=jnp.int32),
                "soilstate_inst": SoilStateType(
                    dsl_col=jnp.array([50.0, 60.0, 70.0]),
                    soilresis_col=jnp.array([5000.0, 6000.0, 7000.0]),
                    watsat=jnp.array([
                        [0.45, 0.43],
                        [0.42, 0.4],
                        [0.4, 0.38],
                    ]),
                    sucsat=jnp.array([
                        [200.0, 220.0],
                        [220.0, 240.0],
                        [240.0, 260.0],
                    ]),
                    bsw=jnp.array([
                        [4.5, 5.0],
                        [5.0, 5.5],
                        [5.5, 6.0],
                    ]),
                ),
                "waterstate_inst": WaterStateType(
                    h2osoi_ice=jnp.array([
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ]),
                    h2osoi_liq=jnp.array([
                        [25.0, 30.0],
                        [28.0, 33.0],
                        [30.0, 35.0],
                    ]),
                ),
                "temperature_inst": TemperatureType(
                    t_soisno=jnp.array([
                        [285.0, 284.0],
                        [286.0, 285.0],
                        [287.0, 286.0],
                    ]),
                ),
                "col": ColumnType(
                    dz=jnp.array([
                        [0.03, 0.06],
                        [0.03, 0.06],
                        [0.03, 0.06],
                    ]),
                ),
            },
            "metadata": {
                "type": "special",
                "description": "Empty filter (all lake points)",
            },
        },
        "test_special_single_layer_soil": {
            "inputs": {
                "bounds": BoundsType(begc=0, endc=2, begg=0, endg=1, begp=0, endp=2),
                "num_nolakec": 2,
                "filter_nolakec": jnp.array([0, 1]),
                "soilstate_inst": SoilStateType(
                    dsl_col=jnp.array([40.0, 55.0]),
                    soilresis_col=jnp.array([4000.0, 5500.0]),
                    watsat=jnp.array([[0.47], [0.41]]),
                    sucsat=jnp.array([[190.0], [230.0]]),
                    bsw=jnp.array([[4.3], [5.8]]),
                ),
                "waterstate_inst": WaterStateType(
                    h2osoi_ice=jnp.array([[0.0], [3.0]]),
                    h2osoi_liq=jnp.array([[32.0], [24.0]]),
                ),
                "temperature_inst": TemperatureType(
                    t_soisno=jnp.array([[288.0], [276.0]]),
                ),
                "col": ColumnType(
                    dz=jnp.array([[0.05], [0.04]]),
                ),
            },
            "metadata": {
                "type": "special",
                "description": "Single soil layer configuration",
            },
        },
    }


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_nominal_single_column_moderate_moisture",
        "test_nominal_multiple_columns_varied_conditions",
        "test_edge_dry_soil_zero_liquid_water",
        "test_edge_saturated_soil_maximum_moisture",
        "test_edge_frozen_soil_all_ice",
        "test_special_empty_filter_no_processing",
        "test_special_single_layer_soil",
    ],
)
def test_calc_soilevap_resis_shapes(test_data, test_case_name):
    """
    Test that calc_soilevap_resis returns outputs with correct shapes.
    
    Verifies that:
    - Output is a SoilStateType namedtuple
    - dsl_col has shape (ncols,)
    - soilresis_col has shape (ncols,)
    - Other fields maintain input shapes
    
    Args:
        test_data: Fixture providing test cases
        test_case_name: Name of the test case to run
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    result = calc_soilevap_resis(**inputs)
    
    # Check that result is a SoilStateType
    assert isinstance(result, SoilStateType), (
        f"Expected SoilStateType, got {type(result)}"
    )
    
    # Check shapes of modified fields
    ncols = inputs["bounds"].endc - inputs["bounds"].begc
    assert result.dsl_col.shape == (ncols,), (
        f"Expected dsl_col shape ({ncols},), got {result.dsl_col.shape}"
    )
    assert result.soilresis_col.shape == (ncols,), (
        f"Expected soilresis_col shape ({ncols},), got {result.soilresis_col.shape}"
    )
    
    # Check shapes of unchanged fields
    assert result.watsat.shape == inputs["soilstate_inst"].watsat.shape, (
        f"watsat shape changed unexpectedly"
    )
    assert result.sucsat.shape == inputs["soilstate_inst"].sucsat.shape, (
        f"sucsat shape changed unexpectedly"
    )
    assert result.bsw.shape == inputs["soilstate_inst"].bsw.shape, (
        f"bsw shape changed unexpectedly"
    )


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_nominal_single_column_moderate_moisture",
        "test_nominal_multiple_columns_varied_conditions",
        "test_edge_dry_soil_zero_liquid_water",
        "test_edge_saturated_soil_maximum_moisture",
        "test_edge_frozen_soil_all_ice",
        "test_special_single_layer_soil",
    ],
)
def test_calc_soilevap_resis_dtypes(test_data, test_case_name):
    """
    Test that calc_soilevap_resis returns outputs with correct data types.
    
    Verifies that all output arrays are JAX arrays with appropriate dtypes.
    
    Args:
        test_data: Fixture providing test cases
        test_case_name: Name of the test case to run
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    result = calc_soilevap_resis(**inputs)
    
    # Check that all fields are JAX arrays
    assert isinstance(result.dsl_col, jnp.ndarray), (
        f"dsl_col should be jnp.ndarray, got {type(result.dsl_col)}"
    )
    assert isinstance(result.soilresis_col, jnp.ndarray), (
        f"soilresis_col should be jnp.ndarray, got {type(result.soilresis_col)}"
    )
    assert isinstance(result.watsat, jnp.ndarray), (
        f"watsat should be jnp.ndarray, got {type(result.watsat)}"
    )
    assert isinstance(result.sucsat, jnp.ndarray), (
        f"sucsat should be jnp.ndarray, got {type(result.sucsat)}"
    )
    assert isinstance(result.bsw, jnp.ndarray), (
        f"bsw should be jnp.ndarray, got {type(result.bsw)}"
    )
    
    # Check that dtypes are floating point
    assert jnp.issubdtype(result.dsl_col.dtype, jnp.floating), (
        f"dsl_col should be floating point, got {result.dsl_col.dtype}"
    )
    assert jnp.issubdtype(result.soilresis_col.dtype, jnp.floating), (
        f"soilresis_col should be floating point, got {result.soilresis_col.dtype}"
    )


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_nominal_single_column_moderate_moisture",
        "test_nominal_multiple_columns_varied_conditions",
        "test_edge_dry_soil_zero_liquid_water",
        "test_edge_saturated_soil_maximum_moisture",
        "test_edge_frozen_soil_all_ice",
        "test_special_single_layer_soil",
    ],
)
def test_calc_soilevap_resis_physical_constraints(test_data, test_case_name):
    """
    Test that calc_soilevap_resis outputs satisfy physical constraints.
    
    Verifies that:
    - dsl_col is non-negative and within [0, 200] mm
    - soilresis_col is non-negative
    - Unchanged fields remain identical to inputs
    
    Args:
        test_data: Fixture providing test cases
        test_case_name: Name of the test case to run
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    result = calc_soilevap_resis(**inputs)
    
    # Check dsl_col constraints
    assert jnp.all(result.dsl_col >= 0.0), (
        f"dsl_col should be non-negative, got min={jnp.min(result.dsl_col)}"
    )
    assert jnp.all(result.dsl_col <= 200.0), (
        f"dsl_col should be <= 200 mm, got max={jnp.max(result.dsl_col)}"
    )
    
    # Check soilresis_col constraints
    assert jnp.all(result.soilresis_col >= 0.0), (
        f"soilresis_col should be non-negative, got min={jnp.min(result.soilresis_col)}"
    )
    
    # Check that unchanged fields are identical
    assert jnp.allclose(result.watsat, inputs["soilstate_inst"].watsat, atol=1e-10), (
        "watsat should remain unchanged"
    )
    assert jnp.allclose(result.sucsat, inputs["soilstate_inst"].sucsat, atol=1e-10), (
        "sucsat should remain unchanged"
    )
    assert jnp.allclose(result.bsw, inputs["soilstate_inst"].bsw, atol=1e-10), (
        "bsw should remain unchanged"
    )


def test_calc_soilevap_resis_empty_filter(test_data):
    """
    Test that calc_soilevap_resis handles empty filter correctly.
    
    When num_nolakec=0 (all lake points), the function should return
    the input soilstate unchanged.
    
    Args:
        test_data: Fixture providing test cases
    """
    test_case = test_data["test_special_empty_filter_no_processing"]
    inputs = test_case["inputs"]
    
    result = calc_soilevap_resis(**inputs)
    
    # With empty filter, outputs should match inputs exactly
    assert jnp.allclose(result.dsl_col, inputs["soilstate_inst"].dsl_col, atol=1e-10), (
        "dsl_col should be unchanged with empty filter"
    )
    assert jnp.allclose(
        result.soilresis_col, inputs["soilstate_inst"].soilresis_col, atol=1e-10
    ), (
        "soilresis_col should be unchanged with empty filter"
    )


def test_calc_soilevap_resis_dry_soil_high_resistance(test_data):
    """
    Test that dry soil produces high evaporative resistance.
    
    Extremely dry soil (zero or near-zero liquid water) should result in
    high soil resistance values and potentially large dry surface layer thickness.
    
    Args:
        test_data: Fixture providing test cases
    """
    test_case = test_data["test_edge_dry_soil_zero_liquid_water"]
    inputs = test_case["inputs"]
    
    result = calc_soilevap_resis(**inputs)
    
    # Dry soil should have high resistance
    # Note: Exact values depend on implementation, but should be significantly
    # higher than the nominal case
    assert jnp.all(result.soilresis_col > 1000.0), (
        f"Dry soil should have high resistance, got min={jnp.min(result.soilresis_col)}"
    )
    
    # DSL should be positive for dry soil
    assert jnp.all(result.dsl_col > 0.0), (
        f"Dry soil should have positive DSL, got min={jnp.min(result.dsl_col)}"
    )


def test_calc_soilevap_resis_saturated_soil_low_resistance(test_data):
    """
    Test that saturated soil produces low evaporative resistance.
    
    Saturated soil (high liquid water content) should result in low soil
    resistance values and small or zero dry surface layer thickness.
    
    Args:
        test_data: Fixture providing test cases
    """
    test_case = test_data["test_edge_saturated_soil_maximum_moisture"]
    inputs = test_case["inputs"]
    
    result = calc_soilevap_resis(**inputs)
    
    # Saturated soil should have low resistance compared to dry soil
    # The exact threshold depends on implementation
    assert jnp.all(result.soilresis_col < 50000.0), (
        f"Saturated soil should have relatively low resistance, "
        f"got max={jnp.max(result.soilresis_col)}"
    )
    
    # DSL should be small for saturated soil
    assert jnp.all(result.dsl_col < 100.0), (
        f"Saturated soil should have small DSL, got max={jnp.max(result.dsl_col)}"
    )


def test_calc_soilevap_resis_frozen_soil_high_resistance(test_data):
    """
    Test that frozen soil produces high evaporative resistance.
    
    Frozen soil (high ice content, low liquid water) should result in high
    soil resistance values due to limited liquid water availability.
    
    Args:
        test_data: Fixture providing test cases
    """
    test_case = test_data["test_edge_frozen_soil_all_ice"]
    inputs = test_case["inputs"]
    
    result = calc_soilevap_resis(**inputs)
    
    # Frozen soil should have high resistance
    assert jnp.all(result.soilresis_col > 5000.0), (
        f"Frozen soil should have high resistance, got min={jnp.min(result.soilresis_col)}"
    )
    
    # DSL should be positive for frozen soil
    assert jnp.all(result.dsl_col > 0.0), (
        f"Frozen soil should have positive DSL, got min={jnp.min(result.dsl_col)}"
    )


def test_calc_soilevap_resis_single_layer(test_data):
    """
    Test that calc_soilevap_resis handles single-layer soil correctly.
    
    Single-layer soil profiles should be processed without errors and
    produce physically reasonable results.
    
    Args:
        test_data: Fixture providing test cases
    """
    test_case = test_data["test_special_single_layer_soil"]
    inputs = test_case["inputs"]
    
    result = calc_soilevap_resis(**inputs)
    
    # Check that computation completes successfully
    assert result.dsl_col.shape == (2,), (
        f"Expected dsl_col shape (2,), got {result.dsl_col.shape}"
    )
    assert result.soilresis_col.shape == (2,), (
        f"Expected soilresis_col shape (2,), got {result.soilresis_col.shape}"
    )
    
    # Check physical constraints
    assert jnp.all(result.dsl_col >= 0.0), (
        "dsl_col should be non-negative for single layer"
    )
    assert jnp.all(result.soilresis_col >= 0.0), (
        "soilresis_col should be non-negative for single layer"
    )


def test_calc_soilevap_resis_multiple_columns_independence(test_data):
    """
    Test that columns are processed independently.
    
    Results for each column should depend only on that column's inputs,
    not on other columns' data.
    
    Args:
        test_data: Fixture providing test cases
    """
    test_case = test_data["test_nominal_multiple_columns_varied_conditions"]
    inputs = test_case["inputs"]
    
    # Run with all columns
    result_all = calc_soilevap_resis(**inputs)
    
    # Run with only first column
    inputs_col0 = {
        "bounds": BoundsType(begc=0, endc=1, begg=0, endg=1, begp=0, endp=1),
        "num_nolakec": 1,
        "filter_nolakec": jnp.array([0]),
        "soilstate_inst": SoilStateType(
            dsl_col=inputs["soilstate_inst"].dsl_col[:1],
            soilresis_col=inputs["soilstate_inst"].soilresis_col[:1],
            watsat=inputs["soilstate_inst"].watsat[:1],
            sucsat=inputs["soilstate_inst"].sucsat[:1],
            bsw=inputs["soilstate_inst"].bsw[:1],
        ),
        "waterstate_inst": WaterStateType(
            h2osoi_ice=inputs["waterstate_inst"].h2osoi_ice[:1],
            h2osoi_liq=inputs["waterstate_inst"].h2osoi_liq[:1],
        ),
        "temperature_inst": TemperatureType(
            t_soisno=inputs["temperature_inst"].t_soisno[:1],
        ),
        "col": ColumnType(
            dz=inputs["col"].dz[:1],
        ),
    }
    
    result_col0 = calc_soilevap_resis(**inputs_col0)
    
    # Results for column 0 should match
    assert jnp.allclose(result_all.dsl_col[0], result_col0.dsl_col[0], atol=1e-6), (
        "Column 0 results should be independent of other columns"
    )
    assert jnp.allclose(
        result_all.soilresis_col[0], result_col0.soilresis_col[0], atol=1e-6
    ), (
        "Column 0 resistance should be independent of other columns"
    )


def test_calc_soilevap_resis_numerical_stability(test_data):
    """
    Test numerical stability with extreme but valid inputs.
    
    The function should handle extreme values without producing NaN or Inf.
    
    Args:
        test_data: Fixture providing test cases
    """
    # Test with dry soil
    test_case_dry = test_data["test_edge_dry_soil_zero_liquid_water"]
    result_dry = calc_soilevap_resis(**test_case_dry["inputs"])
    
    assert not jnp.any(jnp.isnan(result_dry.dsl_col)), (
        "dsl_col should not contain NaN for dry soil"
    )
    assert not jnp.any(jnp.isnan(result_dry.soilresis_col)), (
        "soilresis_col should not contain NaN for dry soil"
    )
    assert not jnp.any(jnp.isinf(result_dry.dsl_col)), (
        "dsl_col should not contain Inf for dry soil"
    )
    assert not jnp.any(jnp.isinf(result_dry.soilresis_col)), (
        "soilresis_col should not contain Inf for dry soil"
    )
    
    # Test with saturated soil
    test_case_sat = test_data["test_edge_saturated_soil_maximum_moisture"]
    result_sat = calc_soilevap_resis(**test_case_sat["inputs"])
    
    assert not jnp.any(jnp.isnan(result_sat.dsl_col)), (
        "dsl_col should not contain NaN for saturated soil"
    )
    assert not jnp.any(jnp.isnan(result_sat.soilresis_col)), (
        "soilresis_col should not contain NaN for saturated soil"
    )
    assert not jnp.any(jnp.isinf(result_sat.dsl_col)), (
        "dsl_col should not contain Inf for saturated soil"
    )
    assert not jnp.any(jnp.isinf(result_sat.soilresis_col)), (
        "soilresis_col should not contain Inf for saturated soil"
    )


def test_calc_soilevap_resis_filter_indices_validity(test_data):
    """
    Test that filter indices are properly used.
    
    Only columns specified in filter_nolakec should be modified.
    
    Args:
        test_data: Fixture providing test cases
    """
    test_case = test_data["test_nominal_multiple_columns_varied_conditions"]
    inputs = test_case["inputs"]
    
    # Modify to only process first column
    inputs_partial = {
        **inputs,
        "num_nolakec": 1,
        "filter_nolakec": jnp.array([0]),
    }
    
    result = calc_soilevap_resis(**inputs_partial)
    
    # First column should be processed (values may change)
    # Other columns should remain unchanged
    assert jnp.allclose(
        result.dsl_col[1], inputs["soilstate_inst"].dsl_col[1], atol=1e-10
    ), (
        "Unfiltered column 1 dsl_col should remain unchanged"
    )
    assert jnp.allclose(
        result.dsl_col[2], inputs["soilstate_inst"].dsl_col[2], atol=1e-10
    ), (
        "Unfiltered column 2 dsl_col should remain unchanged"
    )
    assert jnp.allclose(
        result.soilresis_col[1], inputs["soilstate_inst"].soilresis_col[1], atol=1e-10
    ), (
        "Unfiltered column 1 soilresis_col should remain unchanged"
    )
    assert jnp.allclose(
        result.soilresis_col[2], inputs["soilstate_inst"].soilresis_col[2], atol=1e-10
    ), (
        "Unfiltered column 2 soilresis_col should remain unchanged"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])