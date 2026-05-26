"""
Comprehensive pytest suite for clmDataMod module.

This test suite validates the CLM data processing functions including:
- clm_data: Main function for processing CLM data inputs
- read_clm_data_slice: Extract time slices from NetCDF data
- validate_clm_data: Validate and constrain CLM data
- read_clm_soil: Read soil moisture from NetCDF files
- create_clm_data_inputs: Create input data structures
- get_default_soil_properties: Generate default soil properties

Test coverage includes:
- Nominal cases: Typical growing season conditions
- Edge cases: Zero LAI/SAI, night-time, drought, saturation
- Special cases: Shallow bedrock, dense canopy, heterogeneous soils
- Physical constraints: All values respect physical limits
- Both CLM4.5 and CLM5.0 physics versions
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from offline_driver.clmDataMod import (
    CLMDataInputs,
    CLMDataOutputs,
    CLMDataSlice,
    SoilMoistureData,
    clm_data,
    create_clm_data_inputs,
    get_default_soil_properties,
    read_clm_data_slice,
    validate_clm_data,
)


@pytest.fixture
def test_data():
    """
    Load comprehensive test data for clmDataMod functions.
    
    Returns:
        dict: Test cases organized by function and scenario type
    """
    return {
        "test_nominal_single_patch_column_clm45": {
            "function": "clm_data",
            "n_patches": 1,
            "n_columns": 1,
            "nlevgrnd": 15,
            "nlevsoi": 20,
            "clm_phys": "CLM4_5",
            "dz": [[0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64]],
            "nbedrock": [15],
            "watsat": [[0.45, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.32]],
            "elai_loc": 3.5,
            "esai_loc": 0.8,
            "coszen_loc": 0.707,
            "h2osoi_clm45": [0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39],
            "h2osoi_clm50": [0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.4, 0.4, 0.4, 0.4],
            "denh2o": 1000.0,
            "type": "nominal",
            "description": "Typical growing season conditions with moderate LAI, single patch/column, CLM4.5 physics"
        },
        "test_nominal_multiple_patches_clm50": {
            "function": "clm_data",
            "n_patches": 5,
            "n_columns": 3,
            "nlevgrnd": 15,
            "nlevsoi": 20,
            "clm_phys": "CLM5_0",
            "dz": [
                [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64],
                [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64],
                [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64]
            ],
            "nbedrock": [15, 12, 10],
            "watsat": [
                [0.5, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38, 0.37, 0.36],
                [0.42, 0.42, 0.41, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31, 0.3, 0.29],
                [0.55, 0.54, 0.53, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41]
            ],
            "elai_loc": 4.2,
            "esai_loc": 1.1,
            "coszen_loc": 0.866,
            "h2osoi_clm45": [0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44],
            "h2osoi_clm50": [0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47],
            "denh2o": 1000.0,
            "type": "nominal",
            "description": "Multiple patches and columns with varying bedrock depths, CLM5.0 physics"
        },
        "test_edge_zero_lai_sai": {
            "function": "clm_data",
            "n_patches": 2,
            "n_columns": 2,
            "nlevgrnd": 15,
            "nlevsoi": 20,
            "clm_phys": "CLM4_5",
            "dz": [
                [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64],
                [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64]
            ],
            "nbedrock": [15, 15],
            "watsat": [
                [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
            ],
            "elai_loc": 0.0,
            "esai_loc": 0.0,
            "coszen_loc": 0.5,
            "h2osoi_clm45": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "h2osoi_clm50": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "denh2o": 1000.0,
            "type": "edge",
            "description": "Bare soil conditions with zero LAI and SAI (winter/desert scenario)"
        },
        "test_edge_night_time_negative_coszen": {
            "function": "clm_data",
            "n_patches": 1,
            "n_columns": 1,
            "nlevgrnd": 15,
            "nlevsoi": 20,
            "clm_phys": "CLM5_0",
            "dz": [[0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64]],
            "nbedrock": [15],
            "watsat": [[0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34]],
            "elai_loc": 2.5,
            "esai_loc": 0.5,
            "coszen_loc": -0.3,
            "h2osoi_clm45": [0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29],
            "h2osoi_clm50": [0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34],
            "denh2o": 1000.0,
            "type": "edge",
            "description": "Night-time conditions with negative cosine of solar zenith angle"
        },
        "test_edge_dry_soil_minimal_moisture": {
            "function": "clm_data",
            "n_patches": 3,
            "n_columns": 2,
            "nlevgrnd": 15,
            "nlevsoi": 20,
            "clm_phys": "CLM4_5",
            "dz": [
                [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64],
                [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64]
            ],
            "nbedrock": [15, 15],
            "watsat": [
                [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35],
                [0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38]
            ],
            "elai_loc": 1.2,
            "esai_loc": 0.3,
            "coszen_loc": 0.95,
            "h2osoi_clm45": [0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.06, 0.07, 0.07, 0.08],
            "h2osoi_clm50": [0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.06, 0.07, 0.07, 0.08, 0.09, 0.09, 0.1, 0.1, 0.11],
            "denh2o": 1000.0,
            "type": "edge",
            "description": "Drought conditions with very low soil moisture near wilting point"
        },
        "test_edge_saturated_soil": {
            "function": "clm_data",
            "n_patches": 2,
            "n_columns": 2,
            "nlevgrnd": 15,
            "nlevsoi": 20,
            "clm_phys": "CLM5_0",
            "dz": [
                [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64],
                [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64]
            ],
            "nbedrock": [15, 15],
            "watsat": [
                [0.52, 0.51, 0.5, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38],
                [0.6, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47, 0.46]
            ],
            "elai_loc": 5.8,
            "esai_loc": 1.5,
            "coszen_loc": 0.342,
            "h2osoi_clm45": [0.52, 0.51, 0.5, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38],
            "h2osoi_clm50": [0.6, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41],
            "denh2o": 1000.0,
            "type": "edge",
            "description": "Fully saturated soil conditions (post-rainfall or flooding)"
        },
        "test_special_shallow_bedrock": {
            "function": "clm_data",
            "n_patches": 4,
            "n_columns": 3,
            "nlevgrnd": 15,
            "nlevsoi": 20,
            "clm_phys": "CLM4_5",
            "dz": [
                [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64],
                [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64],
                [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64]
            ],
            "nbedrock": [3, 5, 8],
            "watsat": [
                [0.45, 0.44, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31],
                [0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34],
                [0.5, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38, 0.37, 0.36]
            ],
            "elai_loc": 2.8,
            "esai_loc": 0.7,
            "coszen_loc": 0.643,
            "h2osoi_clm45": [0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36],
            "h2osoi_clm50": [0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41],
            "denh2o": 1000.0,
            "type": "special",
            "description": "Shallow bedrock at different depths across columns (rocky terrain)"
        },
        "test_special_high_lai_dense_canopy": {
            "function": "clm_data",
            "n_patches": 6,
            "n_columns": 4,
            "nlevgrnd": 15,
            "nlevsoi": 20,
            "clm_phys": "CLM5_0",
            "dz": [
                [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64],
                [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64],
                [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64],
                [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64]
            ],
            "nbedrock": [15, 15, 15, 15],
            "watsat": [
                [0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.32],
                [0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33],
                [0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34],
                [0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35]
            ],
            "elai_loc": 8.5,
            "esai_loc": 2.3,
            "coszen_loc": 0.574,
            "h2osoi_clm45": [0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49],
            "h2osoi_clm50": [0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54],
            "denh2o": 1000.0,
            "type": "special",
            "description": "Dense tropical forest canopy with very high LAI and SAI values"
        },
    }


def create_clm_inputs_from_test_case(test_case):
    """
    Helper function to create CLMDataInputs from test case dictionary.
    
    Args:
        test_case: Dictionary containing test case parameters
        
    Returns:
        CLMDataInputs: Properly formatted input structure
    """
    # Convert lists to JAX arrays
    dz = jnp.array(test_case["dz"])
    nbedrock = jnp.array(test_case["nbedrock"])
    watsat = jnp.array(test_case["watsat"])
    h2osoi_clm45 = jnp.array(test_case["h2osoi_clm45"])
    h2osoi_clm50 = jnp.array(test_case["h2osoi_clm50"])
    
    return CLMDataInputs(
        dz=dz,
        nbedrock=nbedrock,
        watsat=watsat,
        elai_loc=test_case["elai_loc"],
        esai_loc=test_case["esai_loc"],
        coszen_loc=test_case["coszen_loc"],
        h2osoi_clm45=h2osoi_clm45,
        h2osoi_clm50=h2osoi_clm50,
        clm_phys=test_case["clm_phys"],
        nlevgrnd=test_case["nlevgrnd"],
        nlevsoi=test_case["nlevsoi"],
        denh2o=test_case["denh2o"]
    )


# ============================================================================
# Tests for clm_data function
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "test_nominal_single_patch_column_clm45",
    "test_nominal_multiple_patches_clm50",
    "test_edge_zero_lai_sai",
    "test_edge_night_time_negative_coszen",
    "test_edge_dry_soil_minimal_moisture",
    "test_edge_saturated_soil",
    "test_special_shallow_bedrock",
    "test_special_high_lai_dense_canopy",
])
def test_clm_data_shapes(test_data, test_case_name):
    """
    Test that clm_data returns outputs with correct shapes.
    
    Validates:
    - elai shape: (n_patches,)
    - esai shape: (n_patches,)
    - coszen shape: (n_columns,)
    - h2osoi_vol shape: (n_columns, nlevgrnd)
    - h2osoi_liq shape: (n_columns, nlevgrnd)
    - h2osoi_ice shape: (n_columns, nlevgrnd)
    """
    test_case = test_data[test_case_name]
    inputs = create_clm_inputs_from_test_case(test_case)
    
    outputs = clm_data(
        inputs=inputs,
        n_patches=test_case["n_patches"],
        n_columns=test_case["n_columns"]
    )
    
    # Verify output shapes
    assert outputs.elai.shape == (test_case["n_patches"],), \
        f"elai shape mismatch: expected {(test_case['n_patches'],)}, got {outputs.elai.shape}"
    
    assert outputs.esai.shape == (test_case["n_patches"],), \
        f"esai shape mismatch: expected {(test_case['n_patches'],)}, got {outputs.esai.shape}"
    
    assert outputs.coszen.shape == (test_case["n_columns"],), \
        f"coszen shape mismatch: expected {(test_case['n_columns'],)}, got {outputs.coszen.shape}"
    
    assert outputs.h2osoi_vol.shape == (test_case["n_columns"], test_case["nlevgrnd"]), \
        f"h2osoi_vol shape mismatch: expected {(test_case['n_columns'], test_case['nlevgrnd'])}, got {outputs.h2osoi_vol.shape}"
    
    assert outputs.h2osoi_liq.shape == (test_case["n_columns"], test_case["nlevgrnd"]), \
        f"h2osoi_liq shape mismatch: expected {(test_case['n_columns'], test_case['nlevgrnd'])}, got {outputs.h2osoi_liq.shape}"
    
    assert outputs.h2osoi_ice.shape == (test_case["n_columns"], test_case["nlevgrnd"]), \
        f"h2osoi_ice shape mismatch: expected {(test_case['n_columns'], test_case['nlevgrnd'])}, got {outputs.h2osoi_ice.shape}"


@pytest.mark.parametrize("test_case_name", [
    "test_nominal_single_patch_column_clm45",
    "test_nominal_multiple_patches_clm50",
])
def test_clm_data_values_nominal(test_data, test_case_name):
    """
    Test that clm_data produces physically reasonable values for nominal cases.
    
    Validates:
    - LAI/SAI values are broadcast correctly to all patches
    - coszen values are broadcast correctly to all columns
    - Soil moisture values are within physical bounds [0, 1]
    - Liquid water and ice are non-negative
    """
    test_case = test_data[test_case_name]
    inputs = create_clm_inputs_from_test_case(test_case)
    
    outputs = clm_data(
        inputs=inputs,
        n_patches=test_case["n_patches"],
        n_columns=test_case["n_columns"]
    )
    
    # Check LAI/SAI broadcast to all patches
    assert jnp.allclose(outputs.elai, test_case["elai_loc"], atol=1e-6), \
        f"elai values not correctly broadcast: expected {test_case['elai_loc']}, got {outputs.elai}"
    
    assert jnp.allclose(outputs.esai, test_case["esai_loc"], atol=1e-6), \
        f"esai values not correctly broadcast: expected {test_case['esai_loc']}, got {outputs.esai}"
    
    # Check coszen broadcast to all columns
    assert jnp.allclose(outputs.coszen, test_case["coszen_loc"], atol=1e-6), \
        f"coszen values not correctly broadcast: expected {test_case['coszen_loc']}, got {outputs.coszen}"
    
    # Check soil moisture is within physical bounds
    assert jnp.all(outputs.h2osoi_vol >= 0.0), \
        f"h2osoi_vol has negative values: min={jnp.min(outputs.h2osoi_vol)}"
    
    assert jnp.all(outputs.h2osoi_vol <= 1.0), \
        f"h2osoi_vol exceeds 1.0: max={jnp.max(outputs.h2osoi_vol)}"
    
    # Check liquid water and ice are non-negative
    assert jnp.all(outputs.h2osoi_liq >= 0.0), \
        f"h2osoi_liq has negative values: min={jnp.min(outputs.h2osoi_liq)}"
    
    assert jnp.all(outputs.h2osoi_ice >= 0.0), \
        f"h2osoi_ice has negative values: min={jnp.min(outputs.h2osoi_ice)}"


@pytest.mark.parametrize("test_case_name", [
    "test_edge_zero_lai_sai",
    "test_edge_night_time_negative_coszen",
    "test_edge_dry_soil_minimal_moisture",
    "test_edge_saturated_soil",
])
def test_clm_data_edge_cases(test_data, test_case_name):
    """
    Test that clm_data handles edge cases correctly.
    
    Edge cases tested:
    - Zero LAI/SAI (bare soil)
    - Negative coszen (night-time)
    - Very low soil moisture (drought)
    - Saturated soil (flooding)
    """
    test_case = test_data[test_case_name]
    inputs = create_clm_inputs_from_test_case(test_case)
    
    outputs = clm_data(
        inputs=inputs,
        n_patches=test_case["n_patches"],
        n_columns=test_case["n_columns"]
    )
    
    # Verify function completes without errors
    assert outputs is not None, "clm_data returned None for edge case"
    
    # Check specific edge case conditions
    if "zero_lai" in test_case.get("description", "").lower():
        assert jnp.all(outputs.elai == 0.0), \
            f"Expected zero LAI, got {outputs.elai}"
        assert jnp.all(outputs.esai == 0.0), \
            f"Expected zero SAI, got {outputs.esai}"
    
    if "night" in test_case.get("description", "").lower():
        assert jnp.all(outputs.coszen < 0.0), \
            f"Expected negative coszen for night-time, got {outputs.coszen}"
    
    if "drought" in test_case.get("description", "").lower():
        assert jnp.all(outputs.h2osoi_vol < 0.15), \
            f"Expected low soil moisture for drought, got max={jnp.max(outputs.h2osoi_vol)}"
    
    if "saturated" in test_case.get("description", "").lower():
        # Soil moisture should be close to saturation (watsat)
        watsat = jnp.array(test_case["watsat"])
        assert jnp.allclose(outputs.h2osoi_vol, watsat, atol=0.05), \
            f"Expected saturated soil moisture close to watsat"


def test_clm_data_dtypes(test_data):
    """
    Test that clm_data returns outputs with correct data types.
    
    All outputs should be JAX arrays with float32 or float64 dtype.
    """
    test_case = test_data["test_nominal_single_patch_column_clm45"]
    inputs = create_clm_inputs_from_test_case(test_case)
    
    outputs = clm_data(
        inputs=inputs,
        n_patches=test_case["n_patches"],
        n_columns=test_case["n_columns"]
    )
    
    # Check all outputs are JAX arrays
    assert isinstance(outputs.elai, jnp.ndarray), \
        f"elai is not a JAX array: {type(outputs.elai)}"
    assert isinstance(outputs.esai, jnp.ndarray), \
        f"esai is not a JAX array: {type(outputs.esai)}"
    assert isinstance(outputs.coszen, jnp.ndarray), \
        f"coszen is not a JAX array: {type(outputs.coszen)}"
    assert isinstance(outputs.h2osoi_vol, jnp.ndarray), \
        f"h2osoi_vol is not a JAX array: {type(outputs.h2osoi_vol)}"
    assert isinstance(outputs.h2osoi_liq, jnp.ndarray), \
        f"h2osoi_liq is not a JAX array: {type(outputs.h2osoi_liq)}"
    assert isinstance(outputs.h2osoi_ice, jnp.ndarray), \
        f"h2osoi_ice is not a JAX array: {type(outputs.h2osoi_ice)}"
    
    # Check dtypes are floating point
    assert jnp.issubdtype(outputs.elai.dtype, jnp.floating), \
        f"elai dtype is not floating point: {outputs.elai.dtype}"
    assert jnp.issubdtype(outputs.h2osoi_vol.dtype, jnp.floating), \
        f"h2osoi_vol dtype is not floating point: {outputs.h2osoi_vol.dtype}"


def test_clm_data_clm45_vs_clm50(test_data):
    """
    Test that clm_data handles both CLM4.5 and CLM5.0 physics versions.
    
    Validates:
    - CLM4.5 uses nlevgrnd=15 layers
    - CLM5.0 uses nlevsoi=20 layers
    - Both versions produce valid outputs
    """
    # Test CLM4.5
    test_case_45 = test_data["test_nominal_single_patch_column_clm45"]
    inputs_45 = create_clm_inputs_from_test_case(test_case_45)
    outputs_45 = clm_data(
        inputs=inputs_45,
        n_patches=test_case_45["n_patches"],
        n_columns=test_case_45["n_columns"]
    )
    
    assert outputs_45.h2osoi_vol.shape[1] == 15, \
        f"CLM4.5 should use 15 ground layers, got {outputs_45.h2osoi_vol.shape[1]}"
    
    # Test CLM5.0
    test_case_50 = test_data["test_nominal_multiple_patches_clm50"]
    inputs_50 = create_clm_inputs_from_test_case(test_case_50)
    outputs_50 = clm_data(
        inputs=inputs_50,
        n_patches=test_case_50["n_patches"],
        n_columns=test_case_50["n_columns"]
    )
    
    # Note: Output still uses nlevgrnd layers, but input uses nlevsoi
    assert outputs_50.h2osoi_vol.shape[1] == 15, \
        f"Output should use nlevgrnd=15 layers, got {outputs_50.h2osoi_vol.shape[1]}"


# ============================================================================
# Tests for read_clm_data_slice function
# ============================================================================

def test_read_clm_data_slice_shapes():
    """
    Test that read_clm_data_slice extracts correct time slice shapes.
    
    Validates:
    - Output is scalar (0-dimensional) for each variable
    - Time indexing works correctly (1-based to 0-based conversion)
    """
    nlndgrid = 10
    ntime = 24
    
    # Create synthetic data
    elai_data = jnp.arange(nlndgrid * ntime, dtype=jnp.float32).reshape(nlndgrid, ntime)
    esai_data = jnp.arange(nlndgrid * ntime, dtype=jnp.float32).reshape(nlndgrid, ntime) * 0.2
    coszen_data = jnp.sin(jnp.arange(nlndgrid * ntime, dtype=jnp.float32).reshape(nlndgrid, ntime) * 0.1)
    
    # Extract time slice (1-based index)
    time_index = 5
    data_slice = read_clm_data_slice(elai_data, esai_data, coszen_data, time_index)
    
    # Check shapes (should be scalars or 0-d arrays)
    assert data_slice.elai.ndim == 0 or data_slice.elai.shape == (), \
        f"elai should be scalar, got shape {data_slice.elai.shape}"
    assert data_slice.esai.ndim == 0 or data_slice.esai.shape == (), \
        f"esai should be scalar, got shape {data_slice.esai.shape}"
    assert data_slice.coszen.ndim == 0 or data_slice.coszen.shape == (), \
        f"coszen should be scalar, got shape {data_slice.coszen.shape}"


def test_read_clm_data_slice_values():
    """
    Test that read_clm_data_slice extracts correct values.
    
    Validates:
    - Correct time slice is extracted
    - Values match expected data at that time index
    """
    nlndgrid = 5
    ntime = 10
    
    # Create known data pattern
    elai_data = jnp.ones((nlndgrid, ntime)) * jnp.arange(ntime)
    esai_data = jnp.ones((nlndgrid, ntime)) * jnp.arange(ntime) * 0.5
    coszen_data = jnp.ones((nlndgrid, ntime)) * jnp.arange(ntime) * 0.1
    
    # Extract time slice 3 (1-based, so index 2 in 0-based)
    time_index = 3
    data_slice = read_clm_data_slice(elai_data, esai_data, coszen_data, time_index)
    
    # Check values (time_index-1 because of 1-based to 0-based conversion)
    expected_elai = 2.0  # time_index - 1 = 3 - 1 = 2
    expected_esai = 1.0  # 2 * 0.5
    expected_coszen = 0.2  # 2 * 0.1
    
    assert jnp.allclose(data_slice.elai, expected_elai, atol=1e-6), \
        f"elai value mismatch: expected {expected_elai}, got {data_slice.elai}"
    assert jnp.allclose(data_slice.esai, expected_esai, atol=1e-6), \
        f"esai value mismatch: expected {expected_esai}, got {data_slice.esai}"
    assert jnp.allclose(data_slice.coszen, expected_coszen, atol=1e-6), \
        f"coszen value mismatch: expected {expected_coszen}, got {data_slice.coszen}"


def test_read_clm_data_slice_edge_first_last():
    """
    Test read_clm_data_slice with first and last time indices.
    
    Validates:
    - First time index (1) works correctly
    - Last time index works correctly
    """
    nlndgrid = 3
    ntime = 8
    
    elai_data = jnp.arange(nlndgrid * ntime, dtype=jnp.float32).reshape(nlndgrid, ntime)
    esai_data = elai_data * 0.3
    coszen_data = jnp.sin(elai_data * 0.2)
    
    # Test first time index
    data_slice_first = read_clm_data_slice(elai_data, esai_data, coszen_data, 1)
    assert data_slice_first is not None, "Failed to extract first time slice"
    
    # Test last time index
    data_slice_last = read_clm_data_slice(elai_data, esai_data, coszen_data, ntime)
    assert data_slice_last is not None, "Failed to extract last time slice"
    
    # Values should be different
    assert not jnp.allclose(data_slice_first.elai, data_slice_last.elai), \
        "First and last time slices should have different values"


# ============================================================================
# Tests for validate_clm_data function
# ============================================================================

def test_validate_clm_data_constraints():
    """
    Test that validate_clm_data enforces physical constraints.
    
    Validates:
    - elai >= 0
    - esai >= 0
    - coszen in [-1, 1]
    """
    # Test with valid data
    valid_data = CLMDataSlice(
        elai=jnp.array(3.5),
        esai=jnp.array(0.8),
        coszen=jnp.array(0.707)
    )
    validated = validate_clm_data(valid_data)
    
    assert jnp.allclose(validated.elai, 3.5, atol=1e-6), \
        "Valid elai should not be modified"
    assert jnp.allclose(validated.esai, 0.8, atol=1e-6), \
        "Valid esai should not be modified"
    assert jnp.allclose(validated.coszen, 0.707, atol=1e-6), \
        "Valid coszen should not be modified"


def test_validate_clm_data_negative_lai():
    """
    Test that validate_clm_data handles negative LAI/SAI values.
    
    Negative values should be clipped to 0.
    """
    invalid_data = CLMDataSlice(
        elai=jnp.array(-1.5),
        esai=jnp.array(-0.3),
        coszen=jnp.array(0.5)
    )
    validated = validate_clm_data(invalid_data)
    
    assert validated.elai >= 0.0, \
        f"Negative elai should be clipped to 0, got {validated.elai}"
    assert validated.esai >= 0.0, \
        f"Negative esai should be clipped to 0, got {validated.esai}"


def test_validate_clm_data_coszen_bounds():
    """
    Test that validate_clm_data enforces coszen bounds [-1, 1].
    
    Values outside [-1, 1] should be clipped.
    """
    # Test coszen > 1
    data_high = CLMDataSlice(
        elai=jnp.array(2.0),
        esai=jnp.array(0.5),
        coszen=jnp.array(1.5)
    )
    validated_high = validate_clm_data(data_high)
    assert validated_high.coszen <= 1.0, \
        f"coszen > 1 should be clipped, got {validated_high.coszen}"
    
    # Test coszen < -1
    data_low = CLMDataSlice(
        elai=jnp.array(2.0),
        esai=jnp.array(0.5),
        coszen=jnp.array(-1.5)
    )
    validated_low = validate_clm_data(data_low)
    assert validated_low.coszen >= -1.0, \
        f"coszen < -1 should be clipped, got {validated_low.coszen}"


# ============================================================================
# Tests for get_default_soil_properties function
# ============================================================================

def test_get_default_soil_properties_shapes():
    """
    Test that get_default_soil_properties returns correct shapes.
    
    Validates:
    - dz shape: (n_columns, nlevgrnd)
    - nbedrock shape: (n_columns,)
    - watsat shape: (n_columns, nlevgrnd)
    """
    n_columns = 5
    nlevgrnd = 15
    
    dz, nbedrock, watsat = get_default_soil_properties(n_columns, nlevgrnd)
    
    assert dz.shape == (n_columns, nlevgrnd), \
        f"dz shape mismatch: expected {(n_columns, nlevgrnd)}, got {dz.shape}"
    assert nbedrock.shape == (n_columns,), \
        f"nbedrock shape mismatch: expected {(n_columns,)}, got {nbedrock.shape}"
    assert watsat.shape == (n_columns, nlevgrnd), \
        f"watsat shape mismatch: expected {(n_columns, nlevgrnd)}, got {watsat.shape}"


def test_get_default_soil_properties_values():
    """
    Test that get_default_soil_properties returns physically reasonable values.
    
    Validates:
    - dz > 0 (positive layer thickness)
    - nbedrock >= 0 (valid layer index)
    - watsat in [0, 1] (valid volumetric fraction)
    - Layer thickness increases with depth
    """
    n_columns = 3
    nlevgrnd = 15
    
    dz, nbedrock, watsat = get_default_soil_properties(n_columns, nlevgrnd)
    
    # Check dz is positive
    assert jnp.all(dz > 0.0), \
        f"dz should be positive, got min={jnp.min(dz)}"
    
    # Check nbedrock is valid
    assert jnp.all(nbedrock >= 0), \
        f"nbedrock should be non-negative, got min={jnp.min(nbedrock)}"
    assert jnp.all(nbedrock <= nlevgrnd), \
        f"nbedrock should be <= nlevgrnd, got max={jnp.max(nbedrock)}"
    
    # Check watsat is in valid range
    assert jnp.all(watsat >= 0.0), \
        f"watsat should be >= 0, got min={jnp.min(watsat)}"
    assert jnp.all(watsat <= 1.0), \
        f"watsat should be <= 1, got max={jnp.max(watsat)}"
    
    # Check layer thickness increases with depth (typical CLM profile)
    for col in range(n_columns):
        for lev in range(nlevgrnd - 1):
            assert dz[col, lev] <= dz[col, lev + 1] or jnp.isclose(dz[col, lev], dz[col, lev + 1], atol=1e-6), \
                f"Layer thickness should increase with depth at column {col}, layer {lev}"


def test_get_default_soil_properties_different_nlevgrnd():
    """
    Test get_default_soil_properties with different nlevgrnd values.
    
    Validates:
    - Function works with various layer counts
    - Shapes are correct for different nlevgrnd
    """
    n_columns = 2
    
    for nlevgrnd in [10, 15, 20, 25]:
        dz, nbedrock, watsat = get_default_soil_properties(n_columns, nlevgrnd)
        
        assert dz.shape == (n_columns, nlevgrnd), \
            f"dz shape incorrect for nlevgrnd={nlevgrnd}"
        assert watsat.shape == (n_columns, nlevgrnd), \
            f"watsat shape incorrect for nlevgrnd={nlevgrnd}"


# ============================================================================
# Tests for create_clm_data_inputs function
# ============================================================================

def test_create_clm_data_inputs_structure():
    """
    Test that create_clm_data_inputs creates proper CLMDataInputs structure.
    
    Validates:
    - Returns CLMDataInputs namedtuple
    - All required fields are present
    - Field types are correct
    """
    n_columns = 2
    nlevgrnd = 15
    nlevsoi = 20
    
    # Create test data
    dz = jnp.ones((n_columns, nlevgrnd)) * 0.1
    nbedrock = jnp.array([15, 12])
    watsat = jnp.ones((n_columns, nlevgrnd)) * 0.45
    
    data_slice = CLMDataSlice(
        elai=jnp.array(3.5),
        esai=jnp.array(0.8),
        coszen=jnp.array(0.707)
    )
    
    soil_moisture = SoilMoistureData(
        h2osoi_clm45=jnp.ones((1, 1, nlevgrnd)) * 0.3,
        h2osoi_clm50=jnp.ones((1, 1, nlevsoi)) * 0.3
    )
    
    inputs = create_clm_data_inputs(
        dz=dz,
        nbedrock=nbedrock,
        watsat=watsat,
        data_slice=data_slice,
        soil_moisture=soil_moisture,
        clm_phys="CLM4_5",
        nlevgrnd=nlevgrnd,
        nlevsoi=nlevsoi
    )
    
    # Check type
    assert isinstance(inputs, CLMDataInputs), \
        f"Should return CLMDataInputs, got {type(inputs)}"
    
    # Check all fields exist
    assert hasattr(inputs, "dz"), "Missing dz field"
    assert hasattr(inputs, "nbedrock"), "Missing nbedrock field"
    assert hasattr(inputs, "watsat"), "Missing watsat field"
    assert hasattr(inputs, "elai_loc"), "Missing elai_loc field"
    assert hasattr(inputs, "esai_loc"), "Missing esai_loc field"
    assert hasattr(inputs, "coszen_loc"), "Missing coszen_loc field"
    assert hasattr(inputs, "h2osoi_clm45"), "Missing h2osoi_clm45 field"
    assert hasattr(inputs, "h2osoi_clm50"), "Missing h2osoi_clm50 field"
    assert hasattr(inputs, "clm_phys"), "Missing clm_phys field"
    assert hasattr(inputs, "nlevgrnd"), "Missing nlevgrnd field"
    assert hasattr(inputs, "nlevsoi"), "Missing nlevsoi field"
    assert hasattr(inputs, "denh2o"), "Missing denh2o field"


def test_create_clm_data_inputs_values():
    """
    Test that create_clm_data_inputs correctly transfers values.
    
    Validates:
    - Scalar values from data_slice are extracted correctly
    - Array values maintain correct shapes
    - Default values are applied correctly
    """
    n_columns = 3
    nlevgrnd = 15
    nlevsoi = 20
    
    dz = jnp.array([[0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.54, 0.64]] * n_columns)
    nbedrock = jnp.array([15, 12, 10])
    watsat = jnp.ones((n_columns, nlevgrnd)) * 0.45
    
    data_slice = CLMDataSlice(
        elai=jnp.array(4.2),
        esai=jnp.array(1.1),
        coszen=jnp.array(0.866)
    )
    
    soil_moisture = SoilMoistureData(
        h2osoi_clm45=jnp.ones((1, 1, nlevgrnd)) * 0.35,
        h2osoi_clm50=jnp.ones((1, 1, nlevsoi)) * 0.32
    )
    
    inputs = create_clm_data_inputs(
        dz=dz,
        nbedrock=nbedrock,
        watsat=watsat,
        data_slice=data_slice,
        soil_moisture=soil_moisture,
        clm_phys="CLM5_0",
        nlevgrnd=nlevgrnd,
        nlevsoi=nlevsoi,
        denh2o=1000.0
    )
    
    # Check scalar values
    assert jnp.allclose(inputs.elai_loc, 4.2, atol=1e-6), \
        f"elai_loc mismatch: expected 4.2, got {inputs.elai_loc}"
    assert jnp.allclose(inputs.esai_loc, 1.1, atol=1e-6), \
        f"esai_loc mismatch: expected 1.1, got {inputs.esai_loc}"
    assert jnp.allclose(inputs.coszen_loc, 0.866, atol=1e-6), \
        f"coszen_loc mismatch: expected 0.866, got {inputs.coszen_loc}"
    
    # Check string value
    assert inputs.clm_phys == "CLM5_0", \
        f"clm_phys mismatch: expected 'CLM5_0', got {inputs.clm_phys}"
    
    # Check integer values
    assert inputs.nlevgrnd == nlevgrnd, \
        f"nlevgrnd mismatch: expected {nlevgrnd}, got {inputs.nlevgrnd}"
    assert inputs.nlevsoi == nlevsoi, \
        f"nlevsoi mismatch: expected {nlevsoi}, got {inputs.nlevsoi}"
    
    # Check default denh2o
    assert jnp.allclose(inputs.denh2o, 1000.0, atol=1e-6), \
        f"denh2o mismatch: expected 1000.0, got {inputs.denh2o}"


def test_create_clm_data_inputs_default_values():
    """
    Test that create_clm_data_inputs applies default values correctly.
    
    Validates:
    - Default nlevgrnd = 15
    - Default nlevsoi = 20
    - Default denh2o = 1000.0
    """
    n_columns = 1
    nlevgrnd = 15
    nlevsoi = 20
    
    dz = jnp.ones((n_columns, nlevgrnd)) * 0.1
    nbedrock = jnp.array([15])
    watsat = jnp.ones((n_columns, nlevgrnd)) * 0.45
    
    data_slice = CLMDataSlice(
        elai=jnp.array(3.0),
        esai=jnp.array(0.7),
        coszen=jnp.array(0.5)
    )
    
    soil_moisture = SoilMoistureData(
        h2osoi_clm45=jnp.ones((1, 1, nlevgrnd)) * 0.3,
        h2osoi_clm50=jnp.ones((1, 1, nlevsoi)) * 0.3
    )
    
    # Create without specifying defaults
    inputs = create_clm_data_inputs(
        dz=dz,
        nbedrock=nbedrock,
        watsat=watsat,
        data_slice=data_slice,
        soil_moisture=soil_moisture,
        clm_phys="CLM4_5"
    )
    
    # Check defaults
    assert inputs.nlevgrnd == 15, \
        f"Default nlevgrnd should be 15, got {inputs.nlevgrnd}"
    assert inputs.nlevsoi == 20, \
        f"Default nlevsoi should be 20, got {inputs.nlevsoi}"
    assert jnp.allclose(inputs.denh2o, 1000.0, atol=1e-6), \
        f"Default denh2o should be 1000.0, got {inputs.denh2o}"


# ============================================================================
# Integration tests
# ============================================================================

def test_full_workflow_clm45(test_data):
    """
    Integration test for complete CLM4.5 workflow.
    
    Tests the full pipeline:
    1. Get default soil properties
    2. Create data slice
    3. Create soil moisture data
    4. Create CLM inputs
    5. Run clm_data
    6. Validate outputs
    """
    n_patches = 2
    n_columns = 2
    nlevgrnd = 15
    nlevsoi = 20
    
    # Step 1: Get default soil properties
    dz, nbedrock, watsat = get_default_soil_properties(n_columns, nlevgrnd)
    
    # Step 2: Create data slice
    data_slice = CLMDataSlice(
        elai=jnp.array(3.5),
        esai=jnp.array(0.8),
        coszen=jnp.array(0.707)
    )
    
    # Step 3: Validate data slice
    validated_slice = validate_clm_data(data_slice)
    
    # Step 4: Create soil moisture data
    soil_moisture = SoilMoistureData(
        h2osoi_clm45=jnp.ones((1, 1, nlevgrnd)) * 0.3,
        h2osoi_clm50=jnp.ones((1, 1, nlevsoi)) * 0.3
    )
    
    # Step 5: Create CLM inputs
    inputs = create_clm_data_inputs(
        dz=dz,
        nbedrock=nbedrock,
        watsat=watsat,
        data_slice=validated_slice,
        soil_moisture=soil_moisture,
        clm_phys="CLM4_5",
        nlevgrnd=nlevgrnd,
        nlevsoi=nlevsoi
    )
    
    # Step 6: Run clm_data
    outputs = clm_data(inputs=inputs, n_patches=n_patches, n_columns=n_columns)
    
    # Step 7: Validate outputs
    assert outputs is not None, "clm_data returned None"
    assert outputs.elai.shape == (n_patches,), "Incorrect elai shape"
    assert outputs.h2osoi_vol.shape == (n_columns, nlevgrnd), "Incorrect h2osoi_vol shape"
    assert jnp.all(outputs.h2osoi_vol >= 0.0), "Negative soil moisture"
    assert jnp.all(outputs.h2osoi_vol <= 1.0), "Soil moisture exceeds 1.0"


def test_full_workflow_clm50(test_data):
    """
    Integration test for complete CLM5.0 workflow.
    
    Similar to CLM4.5 test but with CLM5.0 physics.
    """
    n_patches = 3
    n_columns = 2
    nlevgrnd = 15
    nlevsoi = 20
    
    # Get default properties
    dz, nbedrock, watsat = get_default_soil_properties(n_columns, nlevgrnd)
    
    # Create and validate data slice
    data_slice = CLMDataSlice(
        elai=jnp.array(4.2),
        esai=jnp.array(1.1),
        coszen=jnp.array(0.866)
    )
    validated_slice = validate_clm_data(data_slice)
    
    # Create soil moisture
    soil_moisture = SoilMoistureData(
        h2osoi_clm45=jnp.ones((1, 1, nlevgrnd)) * 0.35,
        h2osoi_clm50=jnp.ones((1, 1, nlevsoi)) * 0.32
    )
    
    # Create inputs
    inputs = create_clm_data_inputs(
        dz=dz,
        nbedrock=nbedrock,
        watsat=watsat,
        data_slice=validated_slice,
        soil_moisture=soil_moisture,
        clm_phys="CLM5_0",
        nlevgrnd=nlevgrnd,
        nlevsoi=nlevsoi
    )
    
    # Run clm_data
    outputs = clm_data(inputs=inputs, n_patches=n_patches, n_columns=n_columns)
    
    # Validate
    assert outputs is not None, "clm_data returned None"
    assert outputs.elai.shape == (n_patches,), "Incorrect elai shape"
    assert outputs.h2osoi_vol.shape == (n_columns, nlevgrnd), "Incorrect h2osoi_vol shape"
    assert jnp.all(outputs.h2osoi_liq >= 0.0), "Negative liquid water"
    assert jnp.all(outputs.h2osoi_ice >= 0.0), "Negative ice"


def test_consistency_across_multiple_calls(test_data):
    """
    Test that clm_data produces consistent results across multiple calls.
    
    Validates:
    - Same inputs produce same outputs
    - Function is deterministic
    """
    test_case = test_data["test_nominal_single_patch_column_clm45"]
    inputs = create_clm_inputs_from_test_case(test_case)
    
    # Run multiple times
    outputs1 = clm_data(
        inputs=inputs,
        n_patches=test_case["n_patches"],
        n_columns=test_case["n_columns"]
    )
    
    outputs2 = clm_data(
        inputs=inputs,
        n_patches=test_case["n_patches"],
        n_columns=test_case["n_columns"]
    )
    
    outputs3 = clm_data(
        inputs=inputs,
        n_patches=test_case["n_patches"],
        n_columns=test_case["n_columns"]
    )
    
    # Check consistency
    assert jnp.allclose(outputs1.elai, outputs2.elai, atol=1e-10), \
        "elai values differ between calls"
    assert jnp.allclose(outputs1.elai, outputs3.elai, atol=1e-10), \
        "elai values differ between calls"
    
    assert jnp.allclose(outputs1.h2osoi_vol, outputs2.h2osoi_vol, atol=1e-10), \
        "h2osoi_vol values differ between calls"
    assert jnp.allclose(outputs1.h2osoi_vol, outputs3.h2osoi_vol, atol=1e-10), \
        "h2osoi_vol values differ between calls"
    
    assert jnp.allclose(outputs1.h2osoi_liq, outputs2.h2osoi_liq, atol=1e-10), \
        "h2osoi_liq values differ between calls"
    assert jnp.allclose(outputs1.h2osoi_liq, outputs3.h2osoi_liq, atol=1e-10), \
        "h2osoi_liq values differ between calls"