"""
Comprehensive pytest suite for SurfaceAlbedoMod functions.

This module tests the surface albedo calculation functions including:
- create_surface_albedo_state: Initialize albedo state with soil color classes
- surface_albedo_init_time_const: Initialize time-constant albedo parameters
- soil_albedo: Calculate soil albedo from moisture and color classes
- soil_albedo_wrapper: Wrapper function for CLM integration

Tests cover:
- Nominal cases with typical soil moisture and albedo values
- Edge cases (saturated, dry, boundary albedo values)
- Physical constraints (albdry >= albsat, albedo in [0,1])
- Array shapes and data types
- Numerical accuracy
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_biogeophys.SurfaceAlbedoMod import (
    BoundsType,
    SoilAlbedoInputs,
    SoilAlbedoOutputs,
    SurfaceAlbedoConstants,
    SurfaceAlbedoState,
    SurfAlbType,
    WaterStateType,
    create_surface_albedo_state,
    soil_albedo,
    soil_albedo_wrapper,
    surface_albedo_init_time_const,
)


@pytest.fixture
def test_data():
    """Load test data for all surface albedo functions."""
    return {
        "create_surface_albedo_state_nominal_8color": {
            "ncolor": 8,
            "ncols": 5,
            "albsat_init": jnp.array([
                [0.12, 0.24], [0.11, 0.22], [0.1, 0.2], [0.09, 0.18],
                [0.08, 0.16], [0.07, 0.14], [0.06, 0.12], [0.05, 0.1]
            ]),
            "albdry_init": jnp.array([
                [0.24, 0.48], [0.22, 0.44], [0.2, 0.4], [0.18, 0.36],
                [0.16, 0.32], [0.14, 0.28], [0.12, 0.24], [0.1, 0.2]
            ]),
            "isoicol_init": jnp.array([0, 2, 4, 6, 7], dtype=jnp.int32),
        },
        "create_surface_albedo_state_nominal_20color": {
            "ncolor": 20,
            "ncols": 10,
            "albsat_init": jnp.array([
                [0.12, 0.24], [0.11, 0.22], [0.1, 0.2], [0.09, 0.18],
                [0.08, 0.16], [0.07, 0.14], [0.06, 0.12], [0.05, 0.1],
                [0.13, 0.26], [0.14, 0.28], [0.15, 0.3], [0.16, 0.32],
                [0.17, 0.34], [0.18, 0.36], [0.19, 0.38], [0.2, 0.4],
                [0.11, 0.23], [0.12, 0.25], [0.13, 0.27], [0.14, 0.29]
            ]),
            "albdry_init": jnp.array([
                [0.24, 0.48], [0.22, 0.44], [0.2, 0.4], [0.18, 0.36],
                [0.16, 0.32], [0.14, 0.28], [0.12, 0.24], [0.1, 0.2],
                [0.26, 0.52], [0.28, 0.56], [0.3, 0.6], [0.32, 0.64],
                [0.34, 0.68], [0.36, 0.72], [0.38, 0.76], [0.4, 0.8],
                [0.22, 0.46], [0.24, 0.5], [0.26, 0.54], [0.28, 0.58]
            ]),
            "isoicol_init": jnp.array([0, 1, 5, 9, 10, 14, 15, 18, 19, 12], dtype=jnp.int32),
        },
        "create_surface_albedo_state_edge_minimum_albedo": {
            "ncolor": 8,
            "ncols": 3,
            "albsat_init": jnp.array([
                [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
            ]),
            "albdry_init": jnp.array([
                [0.0, 0.0], [0.01, 0.02], [0.02, 0.04], [0.03, 0.06],
                [0.04, 0.08], [0.05, 0.1], [0.06, 0.12], [0.07, 0.14]
            ]),
            "isoicol_init": jnp.array([0, 0, 7], dtype=jnp.int32),
        },
        "create_surface_albedo_state_edge_maximum_albedo": {
            "ncolor": 8,
            "ncols": 4,
            "albsat_init": jnp.array([
                [0.9, 0.95], [0.85, 0.92], [0.8, 0.9], [0.75, 0.88],
                [0.7, 0.85], [0.65, 0.82], [0.6, 0.8], [0.55, 0.78]
            ]),
            "albdry_init": jnp.array([
                [0.95, 1.0], [0.92, 0.98], [0.9, 0.96], [0.88, 0.94],
                [0.85, 0.92], [0.82, 0.9], [0.8, 0.88], [0.78, 0.86]
            ]),
            "isoicol_init": jnp.array([0, 1, 2, 3], dtype=jnp.int32),
        },
        "surface_albedo_init_time_const_nominal": {
            "tower_isoicol": 5,
            "n_columns": 8,
            "numrad": 2,
            "ivis": 0,
            "inir": 1,
            "mxsoil_color": 20,
        },
        "surface_albedo_init_time_const_edge_single_column": {
            "tower_isoicol": 0,
            "n_columns": 1,
            "numrad": 2,
            "ivis": 0,
            "inir": 1,
            "mxsoil_color": 8,
        },
        "soil_albedo_nominal_moderate_moisture": {
            "h2osoi_vol": jnp.array([
                [0.25, 0.3, 0.28, 0.26, 0.24, 0.22, 0.2, 0.18, 0.16, 0.14],
                [0.2, 0.25, 0.23, 0.21, 0.19, 0.17, 0.15, 0.13, 0.11, 0.09],
                [0.3, 0.35, 0.33, 0.31, 0.29, 0.27, 0.25, 0.23, 0.21, 0.19],
                [0.15, 0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08, 0.06, 0.04]
            ]),
            "isoicol": jnp.array([3, 7, 12, 15], dtype=jnp.int32),
            "filter_nourbanc": jnp.array([0, 1, 2, 3], dtype=jnp.int32),
        },
        "soil_albedo_edge_saturated_soil": {
            "h2osoi_vol": jnp.array([
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.95, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.9, 0.95, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            ]),
            "isoicol": jnp.array([0, 5, 10], dtype=jnp.int32),
            "filter_nourbanc": jnp.array([0, 1, 2], dtype=jnp.int32),
        },
        "soil_albedo_edge_completely_dry": {
            "h2osoi_vol": jnp.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.01, 0.005, 0.002, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.02, 0.01, 0.005, 0.002, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0]
            ]),
            "isoicol": jnp.array([2, 8, 19], dtype=jnp.int32),
            "filter_nourbanc": jnp.array([0, 1, 2], dtype=jnp.int32),
        },
        "soil_albedo_wrapper_nominal_mixed_conditions": {
            "bounds": {"begc": 0, "endc": 6},
            "num_nourbanc": 5,
            "filter_nourbanc": jnp.array([0, 2, 3, 5, 6], dtype=jnp.int32),
            "h2osoi_vol_col": jnp.array([
                [0.22, 0.25, 0.28, 0.3, 0.32, 0.34, 0.35, 0.36, 0.37, 0.38],
                [0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36],
                [0.35, 0.38, 0.4, 0.42, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49],
                [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28],
                [0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46],
                [0.15, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34],
                [0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58]
            ]),
            "isoicol": jnp.array([4, 1, 8, 12, 6, 15, 18], dtype=jnp.int32),
        },
    }


@pytest.fixture
def standard_albedo_tables():
    """Standard albedo lookup tables for 20-color soil classification."""
    return {
        "albsat": jnp.array([
            [0.12, 0.24], [0.11, 0.22], [0.1, 0.2], [0.09, 0.18],
            [0.08, 0.16], [0.07, 0.14], [0.06, 0.12], [0.05, 0.1],
            [0.13, 0.26], [0.14, 0.28], [0.15, 0.3], [0.16, 0.32],
            [0.17, 0.34], [0.18, 0.36], [0.19, 0.38], [0.2, 0.4],
            [0.11, 0.23], [0.12, 0.25], [0.13, 0.27], [0.14, 0.29]
        ]),
        "albdry": jnp.array([
            [0.24, 0.48], [0.22, 0.44], [0.2, 0.4], [0.18, 0.36],
            [0.16, 0.32], [0.14, 0.28], [0.12, 0.24], [0.1, 0.2],
            [0.26, 0.52], [0.28, 0.56], [0.3, 0.6], [0.32, 0.64],
            [0.34, 0.68], [0.36, 0.72], [0.38, 0.76], [0.4, 0.8],
            [0.22, 0.46], [0.24, 0.5], [0.26, 0.54], [0.28, 0.58]
        ]),
    }


# ============================================================================
# Tests for create_surface_albedo_state
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        "create_surface_albedo_state_nominal_8color",
        "create_surface_albedo_state_nominal_20color",
        "create_surface_albedo_state_edge_minimum_albedo",
        "create_surface_albedo_state_edge_maximum_albedo",
    ],
)
def test_create_surface_albedo_state_shapes(test_data, test_case):
    """
    Test that create_surface_albedo_state returns correct output shapes.
    
    Verifies:
    - albsat shape is (ncolor, 2)
    - albdry shape is (ncolor, 2)
    - isoicol shape is (ncols,)
    """
    data = test_data[test_case]
    
    result = create_surface_albedo_state(
        ncolor=data["ncolor"],
        ncols=data["ncols"],
        albsat_init=data["albsat_init"],
        albdry_init=data["albdry_init"],
        isoicol_init=data["isoicol_init"],
    )
    
    assert isinstance(result, SurfaceAlbedoState), "Result should be SurfaceAlbedoState"
    assert result.albsat.shape == (data["ncolor"], 2), f"albsat shape mismatch"
    assert result.albdry.shape == (data["ncolor"], 2), f"albdry shape mismatch"
    assert result.isoicol.shape == (data["ncols"],), f"isoicol shape mismatch"


@pytest.mark.parametrize(
    "test_case",
    [
        "create_surface_albedo_state_nominal_8color",
        "create_surface_albedo_state_nominal_20color",
    ],
)
def test_create_surface_albedo_state_values(test_data, test_case):
    """
    Test that create_surface_albedo_state preserves input values correctly.
    
    Verifies:
    - Output albsat matches input albsat_init
    - Output albdry matches input albdry_init
    - Output isoicol matches input isoicol_init
    """
    data = test_data[test_case]
    
    result = create_surface_albedo_state(
        ncolor=data["ncolor"],
        ncols=data["ncols"],
        albsat_init=data["albsat_init"],
        albdry_init=data["albdry_init"],
        isoicol_init=data["isoicol_init"],
    )
    
    np.testing.assert_allclose(
        result.albsat, data["albsat_init"], atol=1e-6, rtol=1e-6,
        err_msg="albsat values should match input"
    )
    np.testing.assert_allclose(
        result.albdry, data["albdry_init"], atol=1e-6, rtol=1e-6,
        err_msg="albdry values should match input"
    )
    np.testing.assert_array_equal(
        result.isoicol, data["isoicol_init"],
        err_msg="isoicol values should match input"
    )


def test_create_surface_albedo_state_physical_constraints(test_data):
    """
    Test that create_surface_albedo_state enforces physical constraints.
    
    Verifies:
    - All albedo values are in [0, 1]
    - albdry >= albsat for all color classes
    - NIR albedo >= VIS albedo (typical for soils)
    - isoicol indices are valid (>= 0, < ncolor)
    """
    test_case = "create_surface_albedo_state_nominal_20color"
    data = test_data[test_case]
    
    result = create_surface_albedo_state(
        ncolor=data["ncolor"],
        ncols=data["ncols"],
        albsat_init=data["albsat_init"],
        albdry_init=data["albdry_init"],
        isoicol_init=data["isoicol_init"],
    )
    
    # Check albedo bounds
    assert jnp.all(result.albsat >= 0.0) and jnp.all(result.albsat <= 1.0), \
        "albsat values must be in [0, 1]"
    assert jnp.all(result.albdry >= 0.0) and jnp.all(result.albdry <= 1.0), \
        "albdry values must be in [0, 1]"
    
    # Check albdry >= albsat
    assert jnp.all(result.albdry >= result.albsat), \
        "Dry soil albedo must be >= saturated soil albedo"
    
    # Check NIR >= VIS (column 1 >= column 0)
    assert jnp.all(result.albsat[:, 1] >= result.albsat[:, 0]), \
        "NIR albsat should be >= VIS albsat"
    assert jnp.all(result.albdry[:, 1] >= result.albdry[:, 0]), \
        "NIR albdry should be >= VIS albdry"
    
    # Check isoicol indices
    assert jnp.all(result.isoicol >= 0) and jnp.all(result.isoicol < data["ncolor"]), \
        f"isoicol indices must be in [0, {data['ncolor']})"


def test_create_surface_albedo_state_dtypes(test_data):
    """
    Test that create_surface_albedo_state returns correct data types.
    
    Verifies:
    - albsat and albdry are float arrays
    - isoicol is int32 array
    """
    test_case = "create_surface_albedo_state_nominal_8color"
    data = test_data[test_case]
    
    result = create_surface_albedo_state(
        ncolor=data["ncolor"],
        ncols=data["ncols"],
        albsat_init=data["albsat_init"],
        albdry_init=data["albdry_init"],
        isoicol_init=data["isoicol_init"],
    )
    
    assert jnp.issubdtype(result.albsat.dtype, jnp.floating), \
        "albsat should be floating point"
    assert jnp.issubdtype(result.albdry.dtype, jnp.floating), \
        "albdry should be floating point"
    assert result.isoicol.dtype == jnp.int32, \
        "isoicol should be int32"


def test_create_surface_albedo_state_edge_cases(test_data):
    """
    Test create_surface_albedo_state with edge case values.
    
    Tests:
    - Minimum albedo (0.0) - perfectly absorbing
    - Maximum albedo (1.0) - perfectly reflecting
    - Boundary soil color indices
    """
    # Test minimum albedo
    min_data = test_data["create_surface_albedo_state_edge_minimum_albedo"]
    result_min = create_surface_albedo_state(
        ncolor=min_data["ncolor"],
        ncols=min_data["ncols"],
        albsat_init=min_data["albsat_init"],
        albdry_init=min_data["albdry_init"],
        isoicol_init=min_data["isoicol_init"],
    )
    assert jnp.all(result_min.albsat >= 0.0), "Minimum albedo test failed"
    
    # Test maximum albedo
    max_data = test_data["create_surface_albedo_state_edge_maximum_albedo"]
    result_max = create_surface_albedo_state(
        ncolor=max_data["ncolor"],
        ncols=max_data["ncols"],
        albsat_init=max_data["albsat_init"],
        albdry_init=max_data["albdry_init"],
        isoicol_init=max_data["isoicol_init"],
    )
    assert jnp.all(result_max.albdry <= 1.0), "Maximum albedo test failed"
    
    # Test boundary indices
    assert jnp.all(result_min.isoicol >= 0), "Minimum index test failed"
    assert jnp.all(result_max.isoicol < max_data["ncolor"]), "Maximum index test failed"


# ============================================================================
# Tests for surface_albedo_init_time_const
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        "surface_albedo_init_time_const_nominal",
        "surface_albedo_init_time_const_edge_single_column",
    ],
)
def test_surface_albedo_init_time_const_shapes(test_data, test_case):
    """
    Test that surface_albedo_init_time_const returns correct output shapes.
    
    Verifies:
    - isoicol shape is (n_columns,)
    - albsat shape is (mxsoil_color, numrad)
    - albdry shape is (mxsoil_color, numrad)
    - mxsoil_color is scalar int
    """
    data = test_data[test_case]
    
    result = surface_albedo_init_time_const(
        tower_isoicol=data["tower_isoicol"],
        n_columns=data["n_columns"],
        numrad=data["numrad"],
        ivis=data["ivis"],
        inir=data["inir"],
        mxsoil_color=data["mxsoil_color"],
    )
    
    assert isinstance(result, SurfaceAlbedoConstants), \
        "Result should be SurfaceAlbedoConstants"
    assert result.isoicol.shape == (data["n_columns"],), \
        f"isoicol shape should be ({data['n_columns']},)"
    assert result.albsat.shape == (data["mxsoil_color"], data["numrad"]), \
        f"albsat shape should be ({data['mxsoil_color']}, {data['numrad']})"
    assert result.albdry.shape == (data["mxsoil_color"], data["numrad"]), \
        f"albdry shape should be ({data['mxsoil_color']}, {data['numrad']})"
    assert isinstance(result.mxsoil_color, int), \
        "mxsoil_color should be int"


def test_surface_albedo_init_time_const_values(test_data):
    """
    Test that surface_albedo_init_time_const produces valid values.
    
    Verifies:
    - All isoicol values equal tower_isoicol
    - mxsoil_color matches input
    - Albedo tables are populated with valid values
    """
    test_case = "surface_albedo_init_time_const_nominal"
    data = test_data[test_case]
    
    result = surface_albedo_init_time_const(
        tower_isoicol=data["tower_isoicol"],
        n_columns=data["n_columns"],
        numrad=data["numrad"],
        ivis=data["ivis"],
        inir=data["inir"],
        mxsoil_color=data["mxsoil_color"],
    )
    
    # Check isoicol is broadcast correctly
    assert jnp.all(result.isoicol == data["tower_isoicol"]), \
        "All columns should have tower_isoicol value"
    
    # Check mxsoil_color
    assert result.mxsoil_color == data["mxsoil_color"], \
        "mxsoil_color should match input"
    
    # Check albedo tables have valid values
    assert jnp.all(result.albsat >= 0.0) and jnp.all(result.albsat <= 1.0), \
        "albsat values must be in [0, 1]"
    assert jnp.all(result.albdry >= 0.0) and jnp.all(result.albdry <= 1.0), \
        "albdry values must be in [0, 1]"


def test_surface_albedo_init_time_const_physical_constraints(test_data):
    """
    Test that surface_albedo_init_time_const enforces physical constraints.
    
    Verifies:
    - albdry >= albsat for all soil colors
    - NIR albedo >= VIS albedo
    - tower_isoicol is valid index
    """
    test_case = "surface_albedo_init_time_const_nominal"
    data = test_data[test_case]
    
    result = surface_albedo_init_time_const(
        tower_isoicol=data["tower_isoicol"],
        n_columns=data["n_columns"],
        numrad=data["numrad"],
        ivis=data["ivis"],
        inir=data["inir"],
        mxsoil_color=data["mxsoil_color"],
    )
    
    # Check albdry >= albsat
    assert jnp.all(result.albdry >= result.albsat), \
        "Dry soil albedo must be >= saturated soil albedo"
    
    # Check NIR >= VIS
    assert jnp.all(result.albsat[:, data["inir"]] >= result.albsat[:, data["ivis"]]), \
        "NIR albsat should be >= VIS albsat"
    assert jnp.all(result.albdry[:, data["inir"]] >= result.albdry[:, data["ivis"]]), \
        "NIR albdry should be >= VIS albdry"
    
    # Check tower_isoicol is valid
    assert 0 <= data["tower_isoicol"] < data["mxsoil_color"], \
        f"tower_isoicol must be in [0, {data['mxsoil_color']})"


def test_surface_albedo_init_time_const_dtypes(test_data):
    """
    Test that surface_albedo_init_time_const returns correct data types.
    
    Verifies:
    - isoicol is int32
    - albsat and albdry are float arrays
    - mxsoil_color is Python int
    """
    test_case = "surface_albedo_init_time_const_nominal"
    data = test_data[test_case]
    
    result = surface_albedo_init_time_const(
        tower_isoicol=data["tower_isoicol"],
        n_columns=data["n_columns"],
        numrad=data["numrad"],
        ivis=data["ivis"],
        inir=data["inir"],
        mxsoil_color=data["mxsoil_color"],
    )
    
    assert result.isoicol.dtype == jnp.int32, "isoicol should be int32"
    assert jnp.issubdtype(result.albsat.dtype, jnp.floating), \
        "albsat should be floating point"
    assert jnp.issubdtype(result.albdry.dtype, jnp.floating), \
        "albdry should be floating point"
    assert isinstance(result.mxsoil_color, int), \
        "mxsoil_color should be Python int"


# ============================================================================
# Tests for soil_albedo
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        "soil_albedo_nominal_moderate_moisture",
        "soil_albedo_edge_saturated_soil",
        "soil_albedo_edge_completely_dry",
    ],
)
def test_soil_albedo_shapes(test_data, standard_albedo_tables, test_case):
    """
    Test that soil_albedo returns correct output shapes.
    
    Verifies:
    - albsoib shape is (ncols, numrad)
    - albsoid shape is (ncols, numrad)
    """
    data = test_data[test_case]
    ncols = data["h2osoi_vol"].shape[0]
    numrad = 2
    
    inputs = SoilAlbedoInputs(
        h2osoi_vol=data["h2osoi_vol"],
        isoicol=data["isoicol"],
        albsat=standard_albedo_tables["albsat"],
        albdry=standard_albedo_tables["albdry"],
        filter_nourbanc=data["filter_nourbanc"],
    )
    
    result = soil_albedo(inputs=inputs, numrad=numrad)
    
    assert isinstance(result, SoilAlbedoOutputs), \
        "Result should be SoilAlbedoOutputs"
    assert result.albsoib.shape == (ncols, numrad), \
        f"albsoib shape should be ({ncols}, {numrad})"
    assert result.albsoid.shape == (ncols, numrad), \
        f"albsoid shape should be ({ncols}, {numrad})"


def test_soil_albedo_values_moderate_moisture(test_data, standard_albedo_tables):
    """
    Test soil_albedo with moderate moisture conditions.
    
    Verifies:
    - Albedo values are between albsat and albdry
    - Direct beam and diffuse albedo are equal (for soil)
    - Values are physically reasonable
    """
    test_case = "soil_albedo_nominal_moderate_moisture"
    data = test_data[test_case]
    numrad = 2
    
    inputs = SoilAlbedoInputs(
        h2osoi_vol=data["h2osoi_vol"],
        isoicol=data["isoicol"],
        albsat=standard_albedo_tables["albsat"],
        albdry=standard_albedo_tables["albdry"],
        filter_nourbanc=data["filter_nourbanc"],
    )
    
    result = soil_albedo(inputs=inputs, numrad=numrad)
    
    # Check albedo is in valid range
    assert jnp.all(result.albsoib >= 0.0) and jnp.all(result.albsoib <= 1.0), \
        "albsoib must be in [0, 1]"
    assert jnp.all(result.albsoid >= 0.0) and jnp.all(result.albsoid <= 1.0), \
        "albsoid must be in [0, 1]"
    
    # Check direct beam equals diffuse for soil
    np.testing.assert_allclose(
        result.albsoib, result.albsoid, atol=1e-6, rtol=1e-6,
        err_msg="Direct beam and diffuse albedo should be equal for soil"
    )
    
    # For moderate moisture, albedo should be between albsat and albdry
    for i, col_idx in enumerate(data["filter_nourbanc"]):
        soil_color = data["isoicol"][col_idx]
        for ib in range(numrad):
            albsat_val = standard_albedo_tables["albsat"][soil_color, ib]
            albdry_val = standard_albedo_tables["albdry"][soil_color, ib]
            alb_val = result.albsoib[col_idx, ib]
            
            assert albsat_val <= alb_val <= albdry_val + 0.11, \
                f"Albedo should be between albsat and albdry+0.11 for col {col_idx}, band {ib}"


def test_soil_albedo_saturated_conditions(test_data, standard_albedo_tables):
    """
    Test soil_albedo with fully saturated soil.
    
    Verifies:
    - Albedo approaches albsat values for saturated soil
    - Moisture correction is minimal
    """
    test_case = "soil_albedo_edge_saturated_soil"
    data = test_data[test_case]
    numrad = 2
    
    inputs = SoilAlbedoInputs(
        h2osoi_vol=data["h2osoi_vol"],
        isoicol=data["isoicol"],
        albsat=standard_albedo_tables["albsat"],
        albdry=standard_albedo_tables["albdry"],
        filter_nourbanc=data["filter_nourbanc"],
    )
    
    result = soil_albedo(inputs=inputs, numrad=numrad)
    
    # For saturated soil, albedo should be close to albsat
    for i, col_idx in enumerate(data["filter_nourbanc"]):
        soil_color = data["isoicol"][col_idx]
        for ib in range(numrad):
            albsat_val = standard_albedo_tables["albsat"][soil_color, ib]
            alb_val = result.albsoib[col_idx, ib]
            
            # Allow small tolerance for numerical precision
            np.testing.assert_allclose(
                alb_val, albsat_val, atol=0.02, rtol=0.1,
                err_msg=f"Saturated albedo should be close to albsat for col {col_idx}, band {ib}"
            )


def test_soil_albedo_dry_conditions(test_data, standard_albedo_tables):
    """
    Test soil_albedo with completely dry soil.
    
    Verifies:
    - Albedo approaches albdry + moisture correction
    - Maximum moisture correction (0.11) is applied
    """
    test_case = "soil_albedo_edge_completely_dry"
    data = test_data[test_case]
    numrad = 2
    
    inputs = SoilAlbedoInputs(
        h2osoi_vol=data["h2osoi_vol"],
        isoicol=data["isoicol"],
        albsat=standard_albedo_tables["albsat"],
        albdry=standard_albedo_tables["albdry"],
        filter_nourbanc=data["filter_nourbanc"],
    )
    
    result = soil_albedo(inputs=inputs, numrad=numrad)
    
    # For dry soil, albedo should be close to albdry + 0.11
    for i, col_idx in enumerate(data["filter_nourbanc"]):
        soil_color = data["isoicol"][col_idx]
        for ib in range(numrad):
            albdry_val = standard_albedo_tables["albdry"][soil_color, ib]
            alb_val = result.albsoib[col_idx, ib]
            
            # Albedo should be greater than albdry (due to moisture correction)
            assert alb_val >= albdry_val, \
                f"Dry albedo should be >= albdry for col {col_idx}, band {ib}"
            
            # Should not exceed albdry + 0.11 (max correction)
            assert alb_val <= min(albdry_val + 0.11, 1.0), \
                f"Dry albedo should not exceed albdry + 0.11 for col {col_idx}, band {ib}"


def test_soil_albedo_physical_constraints(test_data, standard_albedo_tables):
    """
    Test that soil_albedo enforces physical constraints.
    
    Verifies:
    - All albedo values in [0, 1]
    - NIR albedo >= VIS albedo
    - Direct beam equals diffuse for soil
    """
    test_case = "soil_albedo_nominal_moderate_moisture"
    data = test_data[test_case]
    numrad = 2
    
    inputs = SoilAlbedoInputs(
        h2osoi_vol=data["h2osoi_vol"],
        isoicol=data["isoicol"],
        albsat=standard_albedo_tables["albsat"],
        albdry=standard_albedo_tables["albdry"],
        filter_nourbanc=data["filter_nourbanc"],
    )
    
    result = soil_albedo(inputs=inputs, numrad=numrad)
    
    # Check bounds
    assert jnp.all(result.albsoib >= 0.0) and jnp.all(result.albsoib <= 1.0), \
        "albsoib must be in [0, 1]"
    assert jnp.all(result.albsoid >= 0.0) and jnp.all(result.albsoid <= 1.0), \
        "albsoid must be in [0, 1]"
    
    # Check NIR >= VIS
    assert jnp.all(result.albsoib[:, 1] >= result.albsoib[:, 0]), \
        "NIR albedo should be >= VIS albedo"
    
    # Check direct beam equals diffuse
    np.testing.assert_allclose(
        result.albsoib, result.albsoid, atol=1e-6, rtol=1e-6,
        err_msg="Direct beam and diffuse should be equal"
    )


def test_soil_albedo_dtypes(test_data, standard_albedo_tables):
    """
    Test that soil_albedo returns correct data types.
    
    Verifies:
    - albsoib and albsoid are float arrays
    """
    test_case = "soil_albedo_nominal_moderate_moisture"
    data = test_data[test_case]
    
    inputs = SoilAlbedoInputs(
        h2osoi_vol=data["h2osoi_vol"],
        isoicol=data["isoicol"],
        albsat=standard_albedo_tables["albsat"],
        albdry=standard_albedo_tables["albdry"],
        filter_nourbanc=data["filter_nourbanc"],
    )
    
    result = soil_albedo(inputs=inputs, numrad=2)
    
    assert jnp.issubdtype(result.albsoib.dtype, jnp.floating), \
        "albsoib should be floating point"
    assert jnp.issubdtype(result.albsoid.dtype, jnp.floating), \
        "albsoid should be floating point"


# ============================================================================
# Tests for soil_albedo_wrapper
# ============================================================================


def test_soil_albedo_wrapper_shapes(test_data, standard_albedo_tables):
    """
    Test that soil_albedo_wrapper returns correct output shapes.
    
    Verifies:
    - albgrd_col shape is (ncols, numrad)
    - albgri_col shape is (ncols, numrad)
    """
    test_case = "soil_albedo_wrapper_nominal_mixed_conditions"
    data = test_data[test_case]
    ncols = data["bounds"]["endc"] - data["bounds"]["begc"] + 1
    numrad = 2
    
    bounds = BoundsType(begc=data["bounds"]["begc"], endc=data["bounds"]["endc"])
    waterstate = WaterStateType(h2osoi_vol_col=data["h2osoi_vol_col"])
    
    result = soil_albedo_wrapper(
        bounds=bounds,
        num_nourbanc=data["num_nourbanc"],
        filter_nourbanc=data["filter_nourbanc"],
        waterstate_inst=waterstate,
        isoicol=data["isoicol"],
        albsat=standard_albedo_tables["albsat"],
        albdry=standard_albedo_tables["albdry"],
        numrad=numrad,
    )
    
    assert isinstance(result, SurfAlbType), "Result should be SurfAlbType"
    assert result.albgrd_col.shape == (ncols, numrad), \
        f"albgrd_col shape should be ({ncols}, {numrad})"
    assert result.albgri_col.shape == (ncols, numrad), \
        f"albgri_col shape should be ({ncols}, {numrad})"


def test_soil_albedo_wrapper_values(test_data, standard_albedo_tables):
    """
    Test soil_albedo_wrapper produces valid albedo values.
    
    Verifies:
    - Albedo values are in valid range
    - Direct beam equals diffuse
    - Non-urban points have computed albedo
    """
    test_case = "soil_albedo_wrapper_nominal_mixed_conditions"
    data = test_data[test_case]
    numrad = 2
    
    bounds = BoundsType(begc=data["bounds"]["begc"], endc=data["bounds"]["endc"])
    waterstate = WaterStateType(h2osoi_vol_col=data["h2osoi_vol_col"])
    
    result = soil_albedo_wrapper(
        bounds=bounds,
        num_nourbanc=data["num_nourbanc"],
        filter_nourbanc=data["filter_nourbanc"],
        waterstate_inst=waterstate,
        isoicol=data["isoicol"],
        albsat=standard_albedo_tables["albsat"],
        albdry=standard_albedo_tables["albdry"],
        numrad=numrad,
    )
    
    # Check valid range
    assert jnp.all(result.albgrd_col >= 0.0) and jnp.all(result.albgrd_col <= 1.0), \
        "albgrd_col must be in [0, 1]"
    assert jnp.all(result.albgri_col >= 0.0) and jnp.all(result.albgri_col <= 1.0), \
        "albgri_col must be in [0, 1]"
    
    # Check direct beam equals diffuse
    np.testing.assert_allclose(
        result.albgrd_col, result.albgri_col, atol=1e-6, rtol=1e-6,
        err_msg="Direct beam and diffuse albedo should be equal"
    )
    
    # Check non-urban points have non-zero albedo
    for col_idx in data["filter_nourbanc"]:
        assert jnp.any(result.albgrd_col[col_idx] > 0.0), \
            f"Non-urban column {col_idx} should have non-zero albedo"


def test_soil_albedo_wrapper_physical_constraints(test_data, standard_albedo_tables):
    """
    Test that soil_albedo_wrapper enforces physical constraints.
    
    Verifies:
    - NIR albedo >= VIS albedo
    - Albedo values consistent with moisture levels
    """
    test_case = "soil_albedo_wrapper_nominal_mixed_conditions"
    data = test_data[test_case]
    numrad = 2
    
    bounds = BoundsType(begc=data["bounds"]["begc"], endc=data["bounds"]["endc"])
    waterstate = WaterStateType(h2osoi_vol_col=data["h2osoi_vol_col"])
    
    result = soil_albedo_wrapper(
        bounds=bounds,
        num_nourbanc=data["num_nourbanc"],
        filter_nourbanc=data["filter_nourbanc"],
        waterstate_inst=waterstate,
        isoicol=data["isoicol"],
        albsat=standard_albedo_tables["albsat"],
        albdry=standard_albedo_tables["albdry"],
        numrad=numrad,
    )
    
    # Check NIR >= VIS
    for col_idx in data["filter_nourbanc"]:
        assert result.albgrd_col[col_idx, 1] >= result.albgrd_col[col_idx, 0], \
            f"NIR albedo should be >= VIS albedo for column {col_idx}"


def test_soil_albedo_wrapper_dtypes(test_data, standard_albedo_tables):
    """
    Test that soil_albedo_wrapper returns correct data types.
    
    Verifies:
    - albgrd_col and albgri_col are float arrays
    """
    test_case = "soil_albedo_wrapper_nominal_mixed_conditions"
    data = test_data[test_case]
    
    bounds = BoundsType(begc=data["bounds"]["begc"], endc=data["bounds"]["endc"])
    waterstate = WaterStateType(h2osoi_vol_col=data["h2osoi_vol_col"])
    
    result = soil_albedo_wrapper(
        bounds=bounds,
        num_nourbanc=data["num_nourbanc"],
        filter_nourbanc=data["filter_nourbanc"],
        waterstate_inst=waterstate,
        isoicol=data["isoicol"],
        albsat=standard_albedo_tables["albsat"],
        albdry=standard_albedo_tables["albdry"],
        numrad=2,
    )
    
    assert jnp.issubdtype(result.albgrd_col.dtype, jnp.floating), \
        "albgrd_col should be floating point"
    assert jnp.issubdtype(result.albgri_col.dtype, jnp.floating), \
        "albgri_col should be floating point"


def test_soil_albedo_wrapper_filter_application(test_data, standard_albedo_tables):
    """
    Test that soil_albedo_wrapper correctly applies the non-urban filter.
    
    Verifies:
    - Only filtered columns have computed albedo
    - Unfiltered columns may have default/zero values
    """
    test_case = "soil_albedo_wrapper_nominal_mixed_conditions"
    data = test_data[test_case]
    
    bounds = BoundsType(begc=data["bounds"]["begc"], endc=data["bounds"]["endc"])
    waterstate = WaterStateType(h2osoi_vol_col=data["h2osoi_vol_col"])
    
    result = soil_albedo_wrapper(
        bounds=bounds,
        num_nourbanc=data["num_nourbanc"],
        filter_nourbanc=data["filter_nourbanc"],
        waterstate_inst=waterstate,
        isoicol=data["isoicol"],
        albsat=standard_albedo_tables["albsat"],
        albdry=standard_albedo_tables["albdry"],
        numrad=2,
    )
    
    # Check that filtered columns have reasonable albedo values
    for col_idx in data["filter_nourbanc"]:
        soil_color = data["isoicol"][col_idx]
        for ib in range(2):
            albsat_val = standard_albedo_tables["albsat"][soil_color, ib]
            albdry_val = standard_albedo_tables["albdry"][soil_color, ib]
            alb_val = result.albgrd_col[col_idx, ib]
            
            # Should be in reasonable range relative to lookup tables
            assert albsat_val <= alb_val <= albdry_val + 0.11, \
                f"Filtered column {col_idx} albedo should be in valid range"


# ============================================================================
# Integration Tests
# ============================================================================


def test_integration_full_workflow(test_data, standard_albedo_tables):
    """
    Integration test for complete workflow from initialization to albedo calculation.
    
    Tests:
    1. Create surface albedo state
    2. Initialize time constants
    3. Calculate soil albedo
    4. Wrapper function integration
    
    Verifies end-to-end consistency.
    """
    # Step 1: Create state
    state = create_surface_albedo_state(
        ncolor=20,
        ncols=7,
        albsat_init=standard_albedo_tables["albsat"],
        albdry_init=standard_albedo_tables["albdry"],
        isoicol_init=jnp.array([4, 1, 8, 12, 6, 15, 18], dtype=jnp.int32),
    )
    
    # Step 2: Initialize constants
    constants = surface_albedo_init_time_const(
        tower_isoicol=5,
        n_columns=7,
        numrad=2,
        ivis=0,
        inir=1,
        mxsoil_color=20,
    )
    
    # Step 3: Calculate soil albedo
    test_case = "soil_albedo_wrapper_nominal_mixed_conditions"
    data = test_data[test_case]
    
    inputs = SoilAlbedoInputs(
        h2osoi_vol=data["h2osoi_vol_col"],
        isoicol=state.isoicol,
        albsat=state.albsat,
        albdry=state.albdry,
        filter_nourbanc=data["filter_nourbanc"],
    )
    
    result_direct = soil_albedo(inputs=inputs, numrad=2)
    
    # Step 4: Use wrapper
    bounds = BoundsType(begc=0, endc=6)
    waterstate = WaterStateType(h2osoi_vol_col=data["h2osoi_vol_col"])
    
    result_wrapper = soil_albedo_wrapper(
        bounds=bounds,
        num_nourbanc=data["num_nourbanc"],
        filter_nourbanc=data["filter_nourbanc"],
        waterstate_inst=waterstate,
        isoicol=state.isoicol,
        albsat=state.albsat,
        albdry=state.albdry,
        numrad=2,
    )
    
    # Verify consistency between direct and wrapper results
    for col_idx in data["filter_nourbanc"]:
        np.testing.assert_allclose(
            result_wrapper.albgrd_col[col_idx],
            result_direct.albsoib[col_idx],
            atol=1e-6,
            rtol=1e-6,
            err_msg=f"Wrapper and direct results should match for column {col_idx}",
        )


def test_integration_moisture_gradient_response():
    """
    Integration test verifying albedo response to moisture gradient.
    
    Tests that albedo decreases monotonically with increasing moisture
    (from dry to saturated conditions).
    """
    # Create moisture gradient from dry to saturated
    moisture_levels = jnp.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
    ncols = len(moisture_levels)
    nlevgrnd = 10
    
    # Create h2osoi_vol with uniform moisture per column
    h2osoi_vol = jnp.repeat(moisture_levels[:, jnp.newaxis], nlevgrnd, axis=1)
    
    # Use same soil color for all columns
    isoicol = jnp.full(ncols, 10, dtype=jnp.int32)
    
    # Standard albedo tables
    albsat = jnp.array([
        [0.12, 0.24], [0.11, 0.22], [0.1, 0.2], [0.09, 0.18],
        [0.08, 0.16], [0.07, 0.14], [0.06, 0.12], [0.05, 0.1],
        [0.13, 0.26], [0.14, 0.28], [0.15, 0.3], [0.16, 0.32],
        [0.17, 0.34], [0.18, 0.36], [0.19, 0.38], [0.2, 0.4],
        [0.11, 0.23], [0.12, 0.25], [0.13, 0.27], [0.14, 0.29]
    ])
    albdry = jnp.array([
        [0.24, 0.48], [0.22, 0.44], [0.2, 0.4], [0.18, 0.36],
        [0.16, 0.32], [0.14, 0.28], [0.12, 0.24], [0.1, 0.2],
        [0.26, 0.52], [0.28, 0.56], [0.3, 0.6], [0.32, 0.64],
        [0.34, 0.68], [0.36, 0.72], [0.38, 0.76], [0.4, 0.8],
        [0.22, 0.46], [0.24, 0.5], [0.26, 0.54], [0.28, 0.58]
    ])
    
    inputs = SoilAlbedoInputs(
        h2osoi_vol=h2osoi_vol,
        isoicol=isoicol,
        albsat=albsat,
        albdry=albdry,
        filter_nourbanc=jnp.arange(ncols, dtype=jnp.int32),
    )
    
    result = soil_albedo(inputs=inputs, numrad=2)
    
    # Check that albedo generally decreases with increasing moisture
    # (allowing for small numerical variations)
    for ib in range(2):
        albedo_values = result.albsoib[:, ib]
        
        # First value (dry) should be highest
        assert albedo_values[0] >= albedo_values[-1], \
            f"Dry soil albedo should be >= saturated for band {ib}"
        
        # Check general decreasing trend (with tolerance for moisture correction)
        for i in range(len(albedo_values) - 1):
            # Allow small increases due to moisture correction formula
            assert albedo_values[i] >= albedo_values[i + 1] - 0.05, \
                f"Albedo should generally decrease with moisture for band {ib}"