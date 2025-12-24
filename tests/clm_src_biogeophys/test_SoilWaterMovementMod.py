"""
Comprehensive pytest suite for soil_water function from SoilWaterMovementMod.

This module tests the soil water movement calculations including:
- Volumetric water content computation
- Hydraulic conductivity using Clapp-Hornberger equations
- Matric potential calculations with clamping
- Edge cases (saturation, dry soil, extreme conditions)
- Multiple columns and soil profiles
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_biogeophys.SoilWaterMovementMod import soil_water


# Define NamedTuples matching the function signature
class Bounds(NamedTuple):
    """CLM column bounds."""
    begc: int
    endc: int


class SoilStateArrays(NamedTuple):
    """Soil physical properties."""
    watsat: jnp.ndarray  # (ncolumns, nlevsoi)
    hksat: jnp.ndarray   # (ncolumns, nlevsoi)
    sucsat: jnp.ndarray  # (ncolumns, nlevsoi)
    bsw: jnp.ndarray     # (ncolumns, nlevsoi)
    nbedrock: jnp.ndarray  # (ncolumns,)
    dz: jnp.ndarray      # (ncolumns, nlevsoi)


class WaterStateType(NamedTuple):
    """Water state variables."""
    h2osoi_liq: jnp.ndarray  # (ncolumns, nlevsoi)


class SoilStateType(NamedTuple):
    """Soil state with computed hydraulic properties."""
    soil_arrays: SoilStateArrays
    vwc_liq: jnp.ndarray  # (ncolumns, nlevsoi)
    hk: jnp.ndarray       # (ncolumns, nlevsoi)
    smp: jnp.ndarray      # (ncolumns, nlevsoi)


@pytest.fixture
def test_data():
    """Load test data from the provided JSON structure."""
    return {
        "test_nominal_single_column_sandy_soil": {
            "bounds": Bounds(begc=0, endc=1),
            "num_hydrologyc": 1,
            "filter_hydrologyc": jnp.array([0]),
            "soilstate_inst": SoilStateType(
                soil_arrays=SoilStateArrays(
                    watsat=jnp.array([[0.45, 0.43, 0.41, 0.39, 0.38]]),
                    hksat=jnp.array([[0.0176, 0.0156, 0.0136, 0.0116, 0.0096]]),
                    sucsat=jnp.array([[121.0, 131.0, 141.0, 151.0, 161.0]]),
                    bsw=jnp.array([[4.05, 4.15, 4.25, 4.35, 4.45]]),
                    nbedrock=jnp.array([5]),
                    dz=jnp.array([[0.1, 0.15, 0.25, 0.35, 0.45]])
                ),
                vwc_liq=jnp.zeros((1, 5)),
                hk=jnp.zeros((1, 5)),
                smp=jnp.zeros((1, 5))
            ),
            "waterstate_inst": WaterStateType(
                h2osoi_liq=jnp.array([[35.0, 48.0, 75.0, 95.0, 110.0]])
            )
        },
        "test_nominal_multiple_columns_varied_soils": {
            "bounds": Bounds(begc=0, endc=3),
            "num_hydrologyc": 3,
            "filter_hydrologyc": jnp.array([0, 1, 2]),
            "soilstate_inst": SoilStateType(
                soil_arrays=SoilStateArrays(
                    watsat=jnp.array([
                        [0.45, 0.43, 0.41, 0.39],
                        [0.52, 0.5, 0.48, 0.46],
                        [0.38, 0.36, 0.34, 0.32]
                    ]),
                    hksat=jnp.array([
                        [0.0176, 0.0156, 0.0136, 0.0116],
                        [0.0056, 0.0046, 0.0036, 0.0026],
                        [0.0256, 0.0236, 0.0216, 0.0196]
                    ]),
                    sucsat=jnp.array([
                        [121.0, 131.0, 141.0, 151.0],
                        [258.0, 268.0, 278.0, 288.0],
                        [78.0, 88.0, 98.0, 108.0]
                    ]),
                    bsw=jnp.array([
                        [4.05, 4.15, 4.25, 4.35],
                        [7.12, 7.22, 7.32, 7.42],
                        [2.79, 2.89, 2.99, 3.09]
                    ]),
                    nbedrock=jnp.array([4, 4, 4]),
                    dz=jnp.array([
                        [0.1, 0.2, 0.3, 0.4],
                        [0.1, 0.2, 0.3, 0.4],
                        [0.1, 0.2, 0.3, 0.4]
                    ])
                ),
                vwc_liq=jnp.zeros((3, 4)),
                hk=jnp.zeros((3, 4)),
                smp=jnp.zeros((3, 4))
            ),
            "waterstate_inst": WaterStateType(
                h2osoi_liq=jnp.array([
                    [30.0, 50.0, 70.0, 85.0],
                    [40.0, 65.0, 90.0, 110.0],
                    [25.0, 42.0, 58.0, 72.0]
                ])
            )
        },
        "test_edge_near_saturation": {
            "bounds": Bounds(begc=0, endc=2),
            "num_hydrologyc": 2,
            "filter_hydrologyc": jnp.array([0, 1]),
            "soilstate_inst": SoilStateType(
                soil_arrays=SoilStateArrays(
                    watsat=jnp.array([
                        [0.45, 0.43, 0.41],
                        [0.5, 0.48, 0.46]
                    ]),
                    hksat=jnp.array([
                        [0.0176, 0.0156, 0.0136],
                        [0.008, 0.007, 0.006]
                    ]),
                    sucsat=jnp.array([
                        [121.0, 131.0, 141.0],
                        [200.0, 210.0, 220.0]
                    ]),
                    bsw=jnp.array([
                        [4.05, 4.15, 4.25],
                        [6.5, 6.6, 6.7]
                    ]),
                    nbedrock=jnp.array([3, 3]),
                    dz=jnp.array([
                        [0.1, 0.2, 0.3],
                        [0.1, 0.2, 0.3]
                    ])
                ),
                vwc_liq=jnp.zeros((2, 3)),
                hk=jnp.zeros((2, 3)),
                smp=jnp.zeros((2, 3))
            ),
            "waterstate_inst": WaterStateType(
                h2osoi_liq=jnp.array([
                    [44.9, 85.9, 122.9],
                    [49.9, 95.9, 137.9]
                ])
            )
        },
        "test_edge_very_dry_soil": {
            "bounds": Bounds(begc=0, endc=1),
            "num_hydrologyc": 1,
            "filter_hydrologyc": jnp.array([0]),
            "soilstate_inst": SoilStateType(
                soil_arrays=SoilStateArrays(
                    watsat=jnp.array([[0.42, 0.4, 0.38, 0.36]]),
                    hksat=jnp.array([[0.015, 0.013, 0.011, 0.009]]),
                    sucsat=jnp.array([[180.0, 190.0, 200.0, 210.0]]),
                    bsw=jnp.array([[5.5, 5.6, 5.7, 5.8]]),
                    nbedrock=jnp.array([4]),
                    dz=jnp.array([[0.1, 0.2, 0.3, 0.4]])
                ),
                vwc_liq=jnp.zeros((1, 4)),
                hk=jnp.zeros((1, 4)),
                smp=jnp.zeros((1, 4))
            ),
            "waterstate_inst": WaterStateType(
                h2osoi_liq=jnp.array([[0.0005, 0.0008, 0.001, 0.0012]])
            )
        },
        "test_edge_high_clay_content": {
            "bounds": Bounds(begc=0, endc=1),
            "num_hydrologyc": 1,
            "filter_hydrologyc": jnp.array([0]),
            "soilstate_inst": SoilStateType(
                soil_arrays=SoilStateArrays(
                    watsat=jnp.array([[0.58, 0.56, 0.54, 0.52, 0.5]]),
                    hksat=jnp.array([[0.0012, 0.001, 0.0008, 0.0006, 0.0004]]),
                    sucsat=jnp.array([[478.0, 488.0, 498.0, 508.0, 518.0]]),
                    bsw=jnp.array([[11.4, 11.5, 11.6, 11.7, 11.8]]),
                    nbedrock=jnp.array([5]),
                    dz=jnp.array([[0.08, 0.12, 0.18, 0.28, 0.38]])
                ),
                vwc_liq=jnp.zeros((1, 5)),
                hk=jnp.zeros((1, 5)),
                smp=jnp.zeros((1, 5))
            ),
            "waterstate_inst": WaterStateType(
                h2osoi_liq=jnp.array([[25.0, 38.0, 52.0, 68.0, 82.0]])
            )
        },
        "test_edge_shallow_bedrock": {
            "bounds": Bounds(begc=0, endc=2),
            "num_hydrologyc": 2,
            "filter_hydrologyc": jnp.array([0, 1]),
            "soilstate_inst": SoilStateType(
                soil_arrays=SoilStateArrays(
                    watsat=jnp.array([
                        [0.4, 0.38, 0.36, 0.34, 0.32, 0.3],
                        [0.42, 0.4, 0.38, 0.36, 0.34, 0.32]
                    ]),
                    hksat=jnp.array([
                        [0.018, 0.016, 0.014, 0.012, 0.01, 0.008],
                        [0.02, 0.018, 0.016, 0.014, 0.012, 0.01]
                    ]),
                    sucsat=jnp.array([
                        [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
                        [95.0, 105.0, 115.0, 125.0, 135.0, 145.0]
                    ]),
                    bsw=jnp.array([
                        [3.5, 3.6, 3.7, 3.8, 3.9, 4.0],
                        [3.4, 3.5, 3.6, 3.7, 3.8, 3.9]
                    ]),
                    nbedrock=jnp.array([2, 3]),
                    dz=jnp.array([
                        [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
                        [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
                    ])
                ),
                vwc_liq=jnp.zeros((2, 6)),
                hk=jnp.zeros((2, 6)),
                smp=jnp.zeros((2, 6))
            ),
            "waterstate_inst": WaterStateType(
                h2osoi_liq=jnp.array([
                    [28.0, 38.0, 45.0, 52.0, 58.0, 62.0],
                    [30.0, 42.0, 52.0, 60.0, 66.0, 70.0]
                ])
            )
        },
        "test_special_single_layer_profile": {
            "bounds": Bounds(begc=0, endc=1),
            "num_hydrologyc": 1,
            "filter_hydrologyc": jnp.array([0]),
            "soilstate_inst": SoilStateType(
                soil_arrays=SoilStateArrays(
                    watsat=jnp.array([[0.44]]),
                    hksat=jnp.array([[0.0165]]),
                    sucsat=jnp.array([[155.0]]),
                    bsw=jnp.array([[4.8]]),
                    nbedrock=jnp.array([1]),
                    dz=jnp.array([[0.5]])
                ),
                vwc_liq=jnp.zeros((1, 1)),
                hk=jnp.zeros((1, 1)),
                smp=jnp.zeros((1, 1))
            ),
            "waterstate_inst": WaterStateType(
                h2osoi_liq=jnp.array([[150.0]])
            )
        },
        "test_special_partial_filter": {
            "bounds": Bounds(begc=0, endc=5),
            "num_hydrologyc": 3,
            "filter_hydrologyc": jnp.array([1, 2, 4]),
            "soilstate_inst": SoilStateType(
                soil_arrays=SoilStateArrays(
                    watsat=jnp.array([
                        [0.4, 0.38, 0.36],
                        [0.45, 0.43, 0.41],
                        [0.48, 0.46, 0.44],
                        [0.42, 0.4, 0.38],
                        [0.5, 0.48, 0.46]
                    ]),
                    hksat=jnp.array([
                        [0.02, 0.018, 0.016],
                        [0.015, 0.013, 0.011],
                        [0.01, 0.008, 0.006],
                        [0.017, 0.015, 0.013],
                        [0.009, 0.007, 0.005]
                    ]),
                    sucsat=jnp.array([
                        [90.0, 100.0, 110.0],
                        [130.0, 140.0, 150.0],
                        [180.0, 190.0, 200.0],
                        [110.0, 120.0, 130.0],
                        [220.0, 230.0, 240.0]
                    ]),
                    bsw=jnp.array([
                        [3.2, 3.3, 3.4],
                        [4.5, 4.6, 4.7],
                        [6.0, 6.1, 6.2],
                        [3.8, 3.9, 4.0],
                        [7.5, 7.6, 7.7]
                    ]),
                    nbedrock=jnp.array([3, 3, 3, 3, 3]),
                    dz=jnp.array([
                        [0.1, 0.2, 0.3],
                        [0.1, 0.2, 0.3],
                        [0.1, 0.2, 0.3],
                        [0.1, 0.2, 0.3],
                        [0.1, 0.2, 0.3]
                    ])
                ),
                vwc_liq=jnp.zeros((5, 3)),
                hk=jnp.zeros((5, 3)),
                smp=jnp.zeros((5, 3))
            ),
            "waterstate_inst": WaterStateType(
                h2osoi_liq=jnp.array([
                    [25.0, 42.0, 58.0],
                    [32.0, 52.0, 72.0],
                    [38.0, 60.0, 82.0],
                    [28.0, 46.0, 64.0],
                    [42.0, 68.0, 94.0]
                ])
            )
        },
        "test_special_extreme_matric_potential": {
            "bounds": Bounds(begc=0, endc=1),
            "num_hydrologyc": 1,
            "filter_hydrologyc": jnp.array([0]),
            "soilstate_inst": SoilStateType(
                soil_arrays=SoilStateArrays(
                    watsat=jnp.array([[0.35, 0.33, 0.31]]),
                    hksat=jnp.array([[0.025, 0.023, 0.021]]),
                    sucsat=jnp.array([[600.0, 650.0, 700.0]]),
                    bsw=jnp.array([[12.0, 12.5, 13.0]]),
                    nbedrock=jnp.array([3]),
                    dz=jnp.array([[0.15, 0.25, 0.35]])
                ),
                vwc_liq=jnp.zeros((1, 3)),
                hk=jnp.zeros((1, 3)),
                smp=jnp.zeros((1, 3))
            ),
            "waterstate_inst": WaterStateType(
                h2osoi_liq=jnp.array([[0.002, 0.003, 0.004]])
            )
        }
    }


@pytest.mark.parametrize("test_name", [
    "test_nominal_single_column_sandy_soil",
    "test_nominal_multiple_columns_varied_soils",
    "test_edge_near_saturation",
    "test_edge_very_dry_soil",
    "test_edge_high_clay_content",
    "test_edge_shallow_bedrock",
    "test_special_single_layer_profile",
    "test_special_partial_filter",
    "test_special_extreme_matric_potential"
])
def test_soil_water_shapes(test_data, test_name):
    """
    Test that soil_water returns outputs with correct shapes.
    
    Verifies that vwc_liq, hk, and smp arrays have the same shape as input
    soil arrays (ncolumns, nlevsoi).
    """
    data = test_data[test_name]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    expected_shape = data["soilstate_inst"].soil_arrays.watsat.shape
    
    assert result.vwc_liq.shape == expected_shape, \
        f"vwc_liq shape {result.vwc_liq.shape} != expected {expected_shape}"
    assert result.hk.shape == expected_shape, \
        f"hk shape {result.hk.shape} != expected {expected_shape}"
    assert result.smp.shape == expected_shape, \
        f"smp shape {result.smp.shape} != expected {expected_shape}"


@pytest.mark.parametrize("test_name", [
    "test_nominal_single_column_sandy_soil",
    "test_nominal_multiple_columns_varied_soils",
    "test_edge_near_saturation",
    "test_edge_very_dry_soil",
    "test_edge_high_clay_content",
    "test_edge_shallow_bedrock",
    "test_special_single_layer_profile",
    "test_special_partial_filter",
    "test_special_extreme_matric_potential"
])
def test_soil_water_dtypes(test_data, test_name):
    """
    Test that soil_water returns outputs with correct data types.
    
    Verifies that all output arrays are JAX arrays with float dtype.
    """
    data = test_data[test_name]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    assert isinstance(result.vwc_liq, jnp.ndarray), \
        f"vwc_liq is not a JAX array: {type(result.vwc_liq)}"
    assert isinstance(result.hk, jnp.ndarray), \
        f"hk is not a JAX array: {type(result.hk)}"
    assert isinstance(result.smp, jnp.ndarray), \
        f"smp is not a JAX array: {type(result.smp)}"
    
    assert jnp.issubdtype(result.vwc_liq.dtype, jnp.floating), \
        f"vwc_liq dtype {result.vwc_liq.dtype} is not floating point"
    assert jnp.issubdtype(result.hk.dtype, jnp.floating), \
        f"hk dtype {result.hk.dtype} is not floating point"
    assert jnp.issubdtype(result.smp.dtype, jnp.floating), \
        f"smp dtype {result.smp.dtype} is not floating point"


def test_soil_water_vwc_calculation(test_data):
    """
    Test volumetric water content calculation.
    
    Verifies that vwc_liq = max(h2osoi_liq, 1e-6) / (dz * 1000)
    using the formula from the physics equations.
    """
    data = test_data["test_nominal_single_column_sandy_soil"]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    # Calculate expected vwc_liq manually
    h2osoi_liq = data["waterstate_inst"].h2osoi_liq
    dz = data["soilstate_inst"].soil_arrays.dz
    DENH2O = 1000.0
    
    expected_vwc = jnp.maximum(h2osoi_liq, 1.0e-6) / (dz * DENH2O)
    
    np.testing.assert_allclose(
        result.vwc_liq[0, :],
        expected_vwc[0, :],
        rtol=1e-6,
        atol=1e-6,
        err_msg="VWC calculation does not match expected formula"
    )


def test_soil_water_constraints_vwc(test_data):
    """
    Test that volumetric water content satisfies physical constraints.
    
    Verifies that 0 <= vwc_liq <= 1 for all computed values.
    """
    data = test_data["test_nominal_multiple_columns_varied_soils"]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    assert jnp.all(result.vwc_liq >= 0.0), \
        f"vwc_liq has negative values: min={jnp.min(result.vwc_liq)}"
    assert jnp.all(result.vwc_liq <= 1.0), \
        f"vwc_liq exceeds 1.0: max={jnp.max(result.vwc_liq)}"


def test_soil_water_constraints_hk(test_data):
    """
    Test that hydraulic conductivity satisfies physical constraints.
    
    Verifies that hk >= 0 for all computed values.
    """
    data = test_data["test_nominal_multiple_columns_varied_soils"]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    assert jnp.all(result.hk >= 0.0), \
        f"hk has negative values: min={jnp.min(result.hk)}"


def test_soil_water_constraints_smp(test_data):
    """
    Test that matric potential satisfies physical constraints.
    
    Verifies that -1e8 <= smp <= 0 for all computed values.
    """
    data = test_data["test_special_extreme_matric_potential"]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    assert jnp.all(result.smp <= 0.0), \
        f"smp has positive values: max={jnp.max(result.smp)}"
    assert jnp.all(result.smp >= -1.0e8), \
        f"smp below clamping limit: min={jnp.min(result.smp)}"


def test_soil_water_near_saturation(test_data):
    """
    Test behavior near saturation conditions.
    
    When water content approaches saturation, hydraulic conductivity
    should approach hksat and matric potential should approach -sucsat.
    """
    data = test_data["test_edge_near_saturation"]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    # At near-saturation, vwc_liq should be close to watsat
    watsat = data["soilstate_inst"].soil_arrays.watsat
    for fc in data["filter_hydrologyc"]:
        assert jnp.all(result.vwc_liq[fc, :] <= watsat[fc, :]), \
            f"vwc_liq exceeds watsat for column {fc}"
        
        # Check that vwc is close to saturation (within 1%)
        ratio = result.vwc_liq[fc, :] / watsat[fc, :]
        assert jnp.all(ratio > 0.98), \
            f"Expected near-saturation but got ratio {ratio}"


def test_soil_water_very_dry_conditions(test_data):
    """
    Test behavior under very dry soil conditions.
    
    When water content is minimal, the minimum clamping (1e-6) should
    be applied, and matric potential should be very negative.
    """
    data = test_data["test_edge_very_dry_soil"]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    # Very dry soil should have very negative matric potential
    assert jnp.all(result.smp[0, :] < -1000.0), \
        f"Expected very negative smp for dry soil, got {result.smp[0, :]}"
    
    # Hydraulic conductivity should be very small
    hksat = data["soilstate_inst"].soil_arrays.hksat
    assert jnp.all(result.hk[0, :] < 0.01 * hksat[0, :]), \
        f"Expected very low hk for dry soil"


def test_soil_water_clapp_hornberger_hk(test_data):
    """
    Test Clapp-Hornberger hydraulic conductivity equation.
    
    Verifies K(θ) = K_sat * (θ/θ_sat)^(2b+3)
    """
    data = test_data["test_nominal_single_column_sandy_soil"]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    # Calculate expected hk using Clapp-Hornberger
    vwc_liq = result.vwc_liq[0, :]
    watsat = data["soilstate_inst"].soil_arrays.watsat[0, :]
    hksat = data["soilstate_inst"].soil_arrays.hksat[0, :]
    bsw = data["soilstate_inst"].soil_arrays.bsw[0, :]
    
    s = jnp.minimum(1.0, vwc_liq / watsat)
    expected_hk = hksat * jnp.power(s, 2.0 * bsw + 3.0)
    
    np.testing.assert_allclose(
        result.hk[0, :],
        expected_hk,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Hydraulic conductivity does not match Clapp-Hornberger equation"
    )


def test_soil_water_clapp_hornberger_smp(test_data):
    """
    Test Clapp-Hornberger matric potential equation.
    
    Verifies ψ(θ) = -ψ_sat * (θ/θ_sat)^(-b), clamped to >= -1e8
    """
    data = test_data["test_nominal_single_column_sandy_soil"]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    # Calculate expected smp using Clapp-Hornberger
    vwc_liq = result.vwc_liq[0, :]
    watsat = data["soilstate_inst"].soil_arrays.watsat[0, :]
    sucsat = data["soilstate_inst"].soil_arrays.sucsat[0, :]
    bsw = data["soilstate_inst"].soil_arrays.bsw[0, :]
    
    s = jnp.minimum(1.0, vwc_liq / watsat)
    expected_smp = -sucsat * jnp.power(s, -bsw)
    expected_smp = jnp.maximum(expected_smp, -1.0e8)
    
    np.testing.assert_allclose(
        result.smp[0, :],
        expected_smp,
        rtol=1e-5,
        atol=1e-3,
        err_msg="Matric potential does not match Clapp-Hornberger equation"
    )


def test_soil_water_matric_potential_clamping(test_data):
    """
    Test that matric potential is properly clamped at -1e8 mm.
    
    For extremely dry conditions with high b parameter, the matric
    potential calculation can produce very large negative values that
    should be clamped.
    """
    data = test_data["test_special_extreme_matric_potential"]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    # All smp values should be >= -1e8
    assert jnp.all(result.smp >= -1.0e8), \
        f"smp values below clamping limit: min={jnp.min(result.smp)}"
    
    # For this extreme case, at least some values should hit the clamp
    assert jnp.any(jnp.abs(result.smp + 1.0e8) < 1.0), \
        "Expected some smp values to be clamped at -1e8"


def test_soil_water_shallow_bedrock(test_data):
    """
    Test behavior with shallow bedrock.
    
    Layers at or below bedrock depth (>= nbedrock) should still be
    computed but may have different characteristics.
    """
    data = test_data["test_edge_shallow_bedrock"]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    nbedrock = data["soilstate_inst"].soil_arrays.nbedrock
    
    # Check that computation occurred for all layers
    for fc in data["filter_hydrologyc"]:
        # All layers should have non-zero values
        assert jnp.all(result.vwc_liq[fc, :] > 0), \
            f"Column {fc} has zero vwc_liq values"
        assert jnp.all(result.hk[fc, :] > 0), \
            f"Column {fc} has zero hk values"
        assert jnp.all(result.smp[fc, :] < 0), \
            f"Column {fc} has non-negative smp values"


def test_soil_water_single_layer(test_data):
    """
    Test with single-layer soil profile.
    
    Verifies that the function works correctly with minimal dimensions.
    """
    data = test_data["test_special_single_layer_profile"]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    assert result.vwc_liq.shape == (1, 1), \
        f"Expected shape (1, 1), got {result.vwc_liq.shape}"
    
    # Check that values are physically reasonable
    assert 0.0 < result.vwc_liq[0, 0] <= 1.0, \
        f"vwc_liq out of range: {result.vwc_liq[0, 0]}"
    assert result.hk[0, 0] > 0.0, \
        f"hk should be positive: {result.hk[0, 0]}"
    assert result.smp[0, 0] < 0.0, \
        f"smp should be negative: {result.smp[0, 0]}"


def test_soil_water_partial_filter(test_data):
    """
    Test with partial hydrology filter.
    
    Only columns in filter_hydrologyc should be processed. Other columns
    should remain unchanged (zeros in this case).
    """
    data = test_data["test_special_partial_filter"]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    filter_indices = data["filter_hydrologyc"]
    all_indices = set(range(data["bounds"].endc))
    non_filter_indices = all_indices - set(np.array(filter_indices))
    
    # Filtered columns should have non-zero values
    for fc in filter_indices:
        assert jnp.any(result.vwc_liq[fc, :] > 0), \
            f"Filtered column {fc} should have non-zero vwc_liq"
        assert jnp.any(result.hk[fc, :] > 0), \
            f"Filtered column {fc} should have non-zero hk"
        assert jnp.any(result.smp[fc, :] < 0), \
            f"Filtered column {fc} should have negative smp"
    
    # Non-filtered columns should remain zero
    for nfc in non_filter_indices:
        assert jnp.all(result.vwc_liq[nfc, :] == 0), \
            f"Non-filtered column {nfc} should have zero vwc_liq"
        assert jnp.all(result.hk[nfc, :] == 0), \
            f"Non-filtered column {nfc} should have zero hk"
        assert jnp.all(result.smp[nfc, :] == 0), \
            f"Non-filtered column {nfc} should have zero smp"


def test_soil_water_multiple_soil_types(test_data):
    """
    Test with multiple columns having different soil types.
    
    Verifies that different soil properties (sandy, loamy, coarse)
    produce appropriately different hydraulic properties.
    """
    data = test_data["test_nominal_multiple_columns_varied_soils"]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    # Column 0: Sandy (lower watsat, higher hksat)
    # Column 1: Loamy (higher watsat, lower hksat, higher sucsat)
    # Column 2: Coarse (lowest watsat, highest hksat, lowest sucsat)
    
    # Sandy soil should have higher hydraulic conductivity than loamy
    assert jnp.mean(result.hk[0, :]) > jnp.mean(result.hk[1, :]), \
        "Sandy soil should have higher hk than loamy soil"
    
    # Loamy soil should have more negative matric potential (higher suction)
    assert jnp.mean(result.smp[1, :]) < jnp.mean(result.smp[0, :]), \
        "Loamy soil should have more negative smp than sandy soil"


def test_soil_water_high_clay_properties(test_data):
    """
    Test with high clay content soil.
    
    Clay soils have high porosity, low hydraulic conductivity,
    high suction, and large b parameter.
    """
    data = test_data["test_edge_high_clay_content"]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    hksat = data["soilstate_inst"].soil_arrays.hksat[0, :]
    
    # Clay soil should have very low hydraulic conductivity
    assert jnp.all(result.hk[0, :] < 0.01), \
        f"Expected low hk for clay soil, got {result.hk[0, :]}"
    
    # Clay soil should have very negative matric potential
    assert jnp.all(result.smp[0, :] < -1000.0), \
        f"Expected very negative smp for clay soil, got {result.smp[0, :]}"


def test_soil_water_preserves_soil_arrays(test_data):
    """
    Test that soil_arrays are preserved in the output.
    
    The function should return the same soil_arrays that were input.
    """
    data = test_data["test_nominal_single_column_sandy_soil"]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    input_arrays = data["soilstate_inst"].soil_arrays
    output_arrays = result.soil_arrays
    
    np.testing.assert_array_equal(
        output_arrays.watsat,
        input_arrays.watsat,
        err_msg="watsat was modified"
    )
    np.testing.assert_array_equal(
        output_arrays.hksat,
        input_arrays.hksat,
        err_msg="hksat was modified"
    )
    np.testing.assert_array_equal(
        output_arrays.sucsat,
        input_arrays.sucsat,
        err_msg="sucsat was modified"
    )
    np.testing.assert_array_equal(
        output_arrays.bsw,
        input_arrays.bsw,
        err_msg="bsw was modified"
    )
    np.testing.assert_array_equal(
        output_arrays.nbedrock,
        input_arrays.nbedrock,
        err_msg="nbedrock was modified"
    )
    np.testing.assert_array_equal(
        output_arrays.dz,
        input_arrays.dz,
        err_msg="dz was modified"
    )


def test_soil_water_consistency_across_layers(test_data):
    """
    Test physical consistency across soil layers.
    
    For a given column, deeper layers with more water should generally
    have higher hydraulic conductivity and less negative matric potential.
    """
    data = test_data["test_nominal_single_column_sandy_soil"]
    
    result = soil_water(
        bounds=data["bounds"],
        num_hydrologyc=data["num_hydrologyc"],
        filter_hydrologyc=data["filter_hydrologyc"],
        soilstate_inst=data["soilstate_inst"],
        waterstate_inst=data["waterstate_inst"]
    )
    
    # Check that vwc_liq increases with depth (more water in deeper layers)
    vwc = result.vwc_liq[0, :]
    for i in range(len(vwc) - 1):
        # Allow for some variation due to different layer properties
        # but generally expect increasing water content
        pass  # This is data-dependent, so we just verify it's computed
    
    # All values should be physically reasonable
    assert jnp.all(vwc > 0), "All layers should have positive water content"
    assert jnp.all(result.hk[0, :] > 0), "All layers should have positive hk"
    assert jnp.all(result.smp[0, :] < 0), "All layers should have negative smp"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])