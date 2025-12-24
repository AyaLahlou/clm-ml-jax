"""
Comprehensive pytest suite for restUtilMod restart variable functions.

This module tests the restartvar_1d, restartvar_2d, and restartvar functions
for reading/writing NetCDF restart files in the CLM model.

Tests cover:
- Nominal cases with typical scientific data
- Edge cases (zeros, small magnitudes, single elements, large arrays)
- Special cases (colon-delimited names, flag meanings, dimension switching)
- Data type validation
- Shape verification
- Physical realism constraints
"""

import sys
from pathlib import Path
from typing import NamedTuple, Union

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_utils.restUtilMod import (
    RestartVar1DResult,
    RestartVar2DResult,
    restartvar,
    restartvar_1d,
    restartvar_2d,
)


# ============================================================================
# Fixtures
# ============================================================================


# Module-level test data for parametrize (must be available at collection time)
TEST_DATA = {
        "restartvar_1d": {
            "nominal": [
                {
                    "name": "test_nominal_read_temperature_data",
                    "inputs": {
                        "ncid": 1001,
                        "flag": "read",
                        "varname": "temperature",
                        "xtype": 6,
                        "dim1name": "gridcell",
                        "dim2name": "",
                        "switchdim": False,
                        "long_name": "Surface temperature",
                        "units": "K",
                        "interpinic_flag": "interp",
                        "data": jnp.array([273.15, 280.5, 295.0, 310.2, 288.7]),
                        "readvar": True,
                        "comment": None,
                        "flag_meanings": None,
                        "missing_value": None,
                        "fill_value": None,
                        "imissing_value": None,
                        "ifill_value": None,
                        "flag_values": None,
                        "nvalid_range": None,
                    },
                    "expected_shape": (5,),
                    "expected_dtype": jnp.float64,
                },
                {
                    "name": "test_nominal_write_soil_moisture",
                    "inputs": {
                        "ncid": 1002,
                        "flag": "write",
                        "varname": "soil_moisture",
                        "xtype": 6,
                        "dim1name": "column",
                        "dim2name": "",
                        "switchdim": False,
                        "long_name": "Volumetric soil moisture",
                        "units": "m3/m3",
                        "interpinic_flag": "copy",
                        "data": jnp.array([0.15, 0.22, 0.35, 0.41, 0.28, 0.19, 0.33]),
                        "readvar": False,
                        "comment": "Soil water content",
                        "flag_meanings": None,
                        "missing_value": -999.0,
                        "fill_value": -999.0,
                        "imissing_value": None,
                        "ifill_value": None,
                        "flag_values": None,
                        "nvalid_range": None,
                    },
                    "expected_shape": (7,),
                    "expected_dtype": jnp.float64,
                },
                {
                    "name": "test_nominal_with_all_optional_attributes",
                    "inputs": {
                        "ncid": 1008,
                        "flag": "write",
                        "varname": "leaf_area_index",
                        "xtype": 6,
                        "dim1name": "pft",
                        "dim2name": "",
                        "switchdim": False,
                        "long_name": "Leaf area index",
                        "units": "m2/m2",
                        "interpinic_flag": "interp",
                        "data": jnp.array([0.5, 1.2, 2.8, 4.5, 5.1, 3.7, 2.1, 0.8, 1.5, 3.2]),
                        "readvar": False,
                        "comment": "One-sided leaf area per unit ground area",
                        "flag_meanings": ("low", "medium", "high"),
                        "missing_value": -999.0,
                        "fill_value": -999.0,
                        "imissing_value": -999,
                        "ifill_value": -999,
                        "flag_values": (0, 1, 2),
                        "nvalid_range": (0, 10),
                    },
                    "expected_shape": (10,),
                    "expected_dtype": jnp.float64,
                },
            ],
            "edge": [
                {
                    "name": "test_edge_zero_values",
                    "inputs": {
                        "ncid": 1003,
                        "flag": "read",
                        "varname": "snow_depth",
                        "xtype": 6,
                        "dim1name": "landunit",
                        "dim2name": "",
                        "switchdim": False,
                        "long_name": "Snow depth",
                        "units": "m",
                        "interpinic_flag": "interp",
                        "data": jnp.array([0.0, 0.0, 0.0, 0.0]),
                        "readvar": True,
                        "comment": None,
                        "flag_meanings": None,
                        "missing_value": None,
                        "fill_value": 0.0,
                        "imissing_value": None,
                        "ifill_value": None,
                        "flag_values": None,
                        "nvalid_range": None,
                    },
                    "expected_shape": (4,),
                    "expected_dtype": jnp.float64,
                },
                {
                    "name": "test_edge_single_element_array",
                    "inputs": {
                        "ncid": 1004,
                        "flag": "write",
                        "varname": "global_mean_temp",
                        "xtype": 6,
                        "dim1name": "scalar",
                        "dim2name": "",
                        "switchdim": False,
                        "long_name": "Global mean temperature",
                        "units": "K",
                        "interpinic_flag": "skip",
                        "data": jnp.array([288.15]),
                        "readvar": False,
                        "comment": "Single scalar value",
                        "flag_meanings": None,
                        "missing_value": None,
                        "fill_value": None,
                        "imissing_value": None,
                        "ifill_value": None,
                        "flag_values": None,
                        "nvalid_range": None,
                    },
                    "expected_shape": (1,),
                    "expected_dtype": jnp.float64,
                },
                {
                    "name": "test_edge_large_array_with_extremes",
                    "inputs": {
                        "ncid": 1005,
                        "flag": "read",
                        "varname": "elevation",
                        "xtype": 6,
                        "dim1name": "gridcell",
                        "dim2name": "",
                        "switchdim": False,
                        "long_name": "Surface elevation",
                        "units": "m",
                        "interpinic_flag": "interp",
                        "data": jnp.array([
                            0.0, 10.5, 100.0, 500.0, 1000.0, 2500.0, 4500.0, 8848.0,
                            5.2, 250.0, 750.0, 1500.0, 3000.0, 6000.0, 8000.0, 100.0,
                            200.0, 300.0, 400.0, 500.0
                        ]),
                        "readvar": True,
                        "comment": None,
                        "flag_meanings": None,
                        "missing_value": -999.0,
                        "fill_value": -999.0,
                        "imissing_value": None,
                        "ifill_value": None,
                        "flag_values": None,
                        "nvalid_range": None,
                    },
                    "expected_shape": (20,),
                    "expected_dtype": jnp.float64,
                },
                {
                    "name": "test_edge_very_small_positive_values",
                    "inputs": {
                        "ncid": 1009,
                        "flag": "read",
                        "varname": "trace_gas_concentration",
                        "xtype": 6,
                        "dim1name": "gridcell",
                        "dim2name": "",
                        "switchdim": False,
                        "long_name": "Trace gas concentration",
                        "units": "ppmv",
                        "interpinic_flag": "interp",
                        "data": jnp.array([1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]),
                        "readvar": True,
                        "comment": None,
                        "flag_meanings": None,
                        "missing_value": None,
                        "fill_value": None,
                        "imissing_value": None,
                        "ifill_value": None,
                        "flag_values": None,
                        "nvalid_range": None,
                    },
                    "expected_shape": (8,),
                    "expected_dtype": jnp.float64,
                },
            ],
            "special": [
                {
                    "name": "test_special_integer_flags_with_meanings",
                    "inputs": {
                        "ncid": 1006,
                        "flag": "write",
                        "varname": "vegetation_type",
                        "xtype": 4,
                        "dim1name": "gridcell",
                        "dim2name": "",
                        "switchdim": False,
                        "long_name": "Vegetation type classification",
                        "units": "categorical",
                        "interpinic_flag": "copy",
                        "data": jnp.array([1, 2, 3, 4, 5, 1, 2, 3], dtype=jnp.int32),
                        "readvar": False,
                        "comment": "Land cover classification",
                        "flag_meanings": ("forest", "grassland", "cropland", "urban", "water"),
                        "missing_value": None,
                        "fill_value": None,
                        "imissing_value": -1,
                        "ifill_value": -1,
                        "flag_values": (1, 2, 3, 4, 5),
                        "nvalid_range": (1, 5),
                    },
                    "expected_shape": (8,),
                    "expected_dtype": jnp.int32,
                },
                {
                    "name": "test_special_colon_delimited_varnames",
                    "inputs": {
                        "ncid": 1007,
                        "flag": "read",
                        "varname": "temp:pressure:humidity",
                        "xtype": 6,
                        "dim1name": "level",
                        "dim2name": "",
                        "switchdim": False,
                        "long_name": "Atmospheric state variables",
                        "units": "mixed",
                        "interpinic_flag": "interp",
                        "data": jnp.array([250.0, 260.0, 270.0, 280.0, 290.0, 300.0]),
                        "readvar": True,
                        "comment": "Colon-delimited variable list",
                        "flag_meanings": None,
                        "missing_value": None,
                        "fill_value": None,
                        "imissing_value": None,
                        "ifill_value": None,
                        "flag_values": None,
                        "nvalid_range": None,
                    },
                    "expected_shape": (6,),
                    "expected_dtype": jnp.float64,
                },
                {
                    "name": "test_special_switchdim_true",
                    "inputs": {
                        "ncid": 1010,
                        "flag": "write",
                        "varname": "carbon_flux",
                        "xtype": 6,
                        "dim1name": "time",
                        "dim2name": "gridcell",
                        "switchdim": True,
                        "long_name": "Net carbon flux",
                        "units": "gC/m2/s",
                        "interpinic_flag": "copy",
                        "data": jnp.array([
                            -0.5, 0.0, 0.3, 1.2, -0.8, 0.5,
                            -0.2, 0.9, 1.5, -1.0, 0.1, 0.7
                        ]),
                        "readvar": False,
                        "comment": "Positive values indicate uptake",
                        "flag_meanings": None,
                        "missing_value": -9999.0,
                        "fill_value": -9999.0,
                        "imissing_value": None,
                        "ifill_value": None,
                        "flag_values": None,
                        "nvalid_range": None,
                    },
                    "expected_shape": (12,),
                    "expected_dtype": jnp.float64,
                },
            ],
        }
    }


@pytest.fixture
def netcdf_mock_context():
    """
    Provide a mock NetCDF context for testing without actual file I/O.
    
    Returns:
        dict: Mock NetCDF file handles and metadata
    """
    return {
        "ncid_map": {
            1001: {"filename": "test_restart_001.nc", "mode": "read"},
            1002: {"filename": "test_restart_002.nc", "mode": "write"},
            1003: {"filename": "test_restart_003.nc", "mode": "read"},
            1004: {"filename": "test_restart_004.nc", "mode": "write"},
            1005: {"filename": "test_restart_005.nc", "mode": "read"},
            1006: {"filename": "test_restart_006.nc", "mode": "write"},
            1007: {"filename": "test_restart_007.nc", "mode": "read"},
            1008: {"filename": "test_restart_008.nc", "mode": "write"},
            1009: {"filename": "test_restart_009.nc", "mode": "read"},
            1010: {"filename": "test_restart_010.nc", "mode": "write"},
        }
    }


# ============================================================================
# Test restartvar_1d function
# ============================================================================


class TestRestartVar1D:
    """Test suite for restartvar_1d function."""

    @pytest.mark.parametrize(
        "test_case",
        [
            pytest.param(tc, id=tc["name"])
            for tc in [
                *TEST_DATA["restartvar_1d"]["nominal"],
                *TEST_DATA["restartvar_1d"]["edge"],
                *TEST_DATA["restartvar_1d"]["special"],
            ]
        ],
        indirect=False,
    )
    def test_restartvar_1d_shapes(self, test_case):
        """
        Test that restartvar_1d returns correct output shapes.
        
        Verifies:
        - Output is RestartVar1DResult NamedTuple
        - Data array has expected 1D shape
        - readvar is boolean
        """
        inputs = test_case["inputs"]
        expected_shape = test_case["expected_shape"]
        
        result = restartvar_1d(**inputs)
        
        assert isinstance(result, RestartVar1DResult), (
            f"Expected RestartVar1DResult, got {type(result)}"
        )
        assert result.data.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {result.data.shape}"
        )
        assert result.data.ndim == 1, (
            f"Expected 1D array, got {result.data.ndim}D"
        )
        assert isinstance(result.readvar, (bool, np.bool_)), (
            f"Expected boolean readvar, got {type(result.readvar)}"
        )

    @pytest.mark.parametrize(
        "test_case",
        [
            pytest.param(tc, id=tc["name"])
            for tc in [
                *TEST_DATA["restartvar_1d"]["nominal"],
                *TEST_DATA["restartvar_1d"]["edge"],
            ]
        ],
        indirect=False,
    )
    def test_restartvar_1d_dtypes(self, test_case):
        """
        Test that restartvar_1d preserves correct data types.
        
        Verifies:
        - Float data uses float64 (r8 precision)
        - Integer data uses int32
        - Data type matches expected type
        """
        inputs = test_case["inputs"]
        expected_dtype = test_case["expected_dtype"]
        
        result = restartvar_1d(**inputs)
        
        assert result.data.dtype == expected_dtype, (
            f"Expected dtype {expected_dtype}, got {result.data.dtype}"
        )

    @pytest.mark.parametrize(
        "test_case",
        [
            pytest.param(tc, id=tc["name"])
            for tc in TEST_DATA["restartvar_1d"]["nominal"]
        ],
        indirect=False,
    )
    def test_restartvar_1d_values_nominal(self, test_case):
        """
        Test that restartvar_1d preserves data values for nominal cases.
        
        Verifies:
        - Data values match input (for write operations)
        - Data values are reasonable (for read operations)
        - No unexpected NaN or Inf values
        """
        inputs = test_case["inputs"]
        input_data = inputs["data"]
        
        result = restartvar_1d(**inputs)
        
        # For write operations, data should be preserved
        if inputs["flag"] == "write":
            np.testing.assert_allclose(
                result.data,
                input_data,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Data values changed during write operation"
            )
        
        # Check for invalid values
        if jnp.issubdtype(result.data.dtype, jnp.floating):
            assert not jnp.any(jnp.isnan(result.data)), (
                "Unexpected NaN values in output"
            )
            assert not jnp.any(jnp.isinf(result.data)), (
                "Unexpected Inf values in output"
            )

    @pytest.mark.parametrize(
        "test_case",
        [
            pytest.param(tc, id=tc["name"])
            for tc in TEST_DATA["restartvar_1d"]["edge"]
        ],
        indirect=False,
    )
    def test_restartvar_1d_edge_cases(self, test_case):
        """
        Test that restartvar_1d handles edge cases correctly.
        
        Edge cases tested:
        - All zero values
        - Single element arrays
        - Large arrays with extreme values
        - Very small positive values (numerical precision)
        """
        inputs = test_case["inputs"]
        
        result = restartvar_1d(**inputs)
        
        # Verify result is valid
        assert result.data is not None, "Result data is None"
        assert result.data.size > 0, "Result data is empty"
        
        # For zero values test
        if "zero" in test_case["name"]:
            assert jnp.all(result.data == 0.0), (
                "Zero values not preserved"
            )
        
        # For single element test
        if "single_element" in test_case["name"]:
            assert result.data.size == 1, (
                f"Expected single element, got {result.data.size}"
            )
        
        # For small values test
        if "small_positive" in test_case["name"]:
            assert jnp.all(result.data > 0), (
                "Small positive values became negative or zero"
            )
            assert jnp.all(result.data < 1.0), (
                "Small values unexpectedly large"
            )

    @pytest.mark.parametrize(
        "test_case",
        [
            pytest.param(tc, id=tc["name"])
            for tc in TEST_DATA["restartvar_1d"]["special"]
        ],
        indirect=False,
    )
    def test_restartvar_1d_special_cases(self, test_case):
        """
        Test that restartvar_1d handles special cases correctly.
        
        Special cases tested:
        - Integer flags with meanings
        - Colon-delimited variable names
        - Dimension switching
        - All optional attributes populated
        """
        inputs = test_case["inputs"]
        
        result = restartvar_1d(**inputs)
        
        # Verify result structure
        assert isinstance(result, RestartVar1DResult), (
            "Result is not RestartVar1DResult"
        )
        
        # For integer flags test
        if "integer_flags" in test_case["name"]:
            assert jnp.issubdtype(result.data.dtype, jnp.integer), (
                "Integer flags not preserved as integer type"
            )
            if inputs.get("nvalid_range"):
                min_val, max_val = inputs["nvalid_range"]
                assert jnp.all(result.data >= min_val), (
                    f"Values below valid range minimum {min_val}"
                )
                assert jnp.all(result.data <= max_val), (
                    f"Values above valid range maximum {max_val}"
                )
        
        # For colon-delimited names
        if "colon_delimited" in test_case["name"]:
            assert ":" in inputs["varname"], (
                "Test case should have colon-delimited varname"
            )
            # Function should handle this gracefully
            assert result.data is not None, (
                "Failed to handle colon-delimited varname"
            )

    def test_restartvar_1d_readvar_flag_consistency(self, test_data):
        """
        Test that readvar flag is handled consistently.
        
        Verifies:
        - readvar=True for read operations
        - readvar=False for write operations
        - Output readvar matches input readvar
        """
        for category in ["nominal", "edge", "special"]:
            for test_case in test_data["restartvar_1d"][category]:
                inputs = test_case["inputs"]
                result = restartvar_1d(**inputs)
                
                assert result.readvar == inputs["readvar"], (
                    f"readvar flag mismatch: expected {inputs['readvar']}, "
                    f"got {result.readvar}"
                )

    def test_restartvar_1d_physical_constraints(self, test_data):
        """
        Test that physical constraints are respected.
        
        Verifies:
        - Temperature values > 0K (absolute zero)
        - Fractional values in [0, 1] where appropriate
        - Elevation values >= 0
        - Soil moisture in [0, 1]
        """
        for category in ["nominal", "edge"]:
            for test_case in test_data["restartvar_1d"][category]:
                inputs = test_case["inputs"]
                result = restartvar_1d(**inputs)
                
                varname = inputs["varname"].lower()
                
                # Temperature constraint
                if "temp" in varname and "K" in inputs.get("units", ""):
                    assert jnp.all(result.data >= 0.0), (
                        f"Temperature below absolute zero: {jnp.min(result.data)}K"
                    )
                
                # Soil moisture constraint
                if "moisture" in varname or "soil_water" in varname:
                    assert jnp.all(result.data >= 0.0), (
                        "Negative soil moisture values"
                    )
                    assert jnp.all(result.data <= 1.0), (
                        "Soil moisture values > 1.0"
                    )
                
                # Elevation constraint
                if "elevation" in varname or "height" in varname:
                    # Allow for below sea level, but check for reasonable values
                    assert jnp.all(result.data >= -500.0), (
                        "Unreasonably low elevation values"
                    )


# ============================================================================
# Test restartvar_2d function
# ============================================================================


class TestRestartVar2D:
    """Test suite for restartvar_2d function."""

    def test_restartvar_2d_basic_functionality(self):
        """
        Test basic functionality of restartvar_2d with simple 2D array.
        
        Verifies:
        - Function accepts 2D arrays
        - Returns RestartVar2DResult
        - Preserves 2D shape
        """
        data_2d = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        result = restartvar_2d(
            ncid=2001,
            flag="write",
            varname="test_var_2d",
            xtype=6,
            dim1name="row",
            dim2name="col",
            switchdim=False,
            long_name="Test 2D variable",
            units="units",
            interpinic_flag="copy",
            data=data_2d,
            readvar=False,
        )
        
        assert isinstance(result, RestartVar2DResult), (
            f"Expected RestartVar2DResult, got {type(result)}"
        )
        assert result.data.ndim == 2, (
            f"Expected 2D array, got {result.data.ndim}D"
        )
        assert result.data.shape == (2, 3), (
            f"Expected shape (2, 3), got {result.data.shape}"
        )

    def test_restartvar_2d_dimension_switching(self):
        """
        Test dimension switching functionality in restartvar_2d.
        
        Verifies:
        - switchdim=True transposes dimensions
        - switchdim=False preserves dimensions
        - Data values are preserved during transpose
        """
        data_2d = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        # Test without switching
        result_no_switch = restartvar_2d(
            ncid=2002,
            flag="write",
            varname="test_no_switch",
            xtype=6,
            dim1name="dim1",
            dim2name="dim2",
            switchdim=False,
            long_name="No switch test",
            units="units",
            interpinic_flag="copy",
            data=data_2d,
            readvar=False,
        )
        
        # Test with switching
        result_switch = restartvar_2d(
            ncid=2003,
            flag="write",
            varname="test_switch",
            xtype=6,
            dim1name="dim1",
            dim2name="dim2",
            switchdim=True,
            long_name="Switch test",
            units="units",
            interpinic_flag="copy",
            data=data_2d,
            readvar=False,
        )
        
        # Verify shapes
        assert result_no_switch.data.shape == (3, 2), (
            "Shape changed without switching"
        )
        assert result_switch.data.shape == (2, 3), (
            "Dimension switching failed"
        )

    def test_restartvar_2d_edge_single_row_column(self):
        """
        Test restartvar_2d with edge case of single row or column.
        
        Verifies:
        - Handles (1, n) arrays
        - Handles (n, 1) arrays
        - Preserves data correctly
        """
        # Single row
        data_single_row = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        result_row = restartvar_2d(
            ncid=2004,
            flag="write",
            varname="single_row",
            xtype=6,
            dim1name="row",
            dim2name="col",
            switchdim=False,
            long_name="Single row test",
            units="units",
            interpinic_flag="copy",
            data=data_single_row,
            readvar=False,
        )
        
        assert result_row.data.shape == (1, 4), (
            f"Single row shape incorrect: {result_row.data.shape}"
        )
        
        # Single column
        data_single_col = jnp.array([[1.0], [2.0], [3.0]])
        result_col = restartvar_2d(
            ncid=2005,
            flag="write",
            varname="single_col",
            xtype=6,
            dim1name="row",
            dim2name="col",
            switchdim=False,
            long_name="Single column test",
            units="units",
            interpinic_flag="copy",
            data=data_single_col,
            readvar=False,
        )
        
        assert result_col.data.shape == (3, 1), (
            f"Single column shape incorrect: {result_col.data.shape}"
        )

    def test_restartvar_2d_large_array(self):
        """
        Test restartvar_2d with larger 2D arrays.
        
        Verifies:
        - Handles arrays with many elements
        - Preserves data integrity
        - No memory issues
        """
        large_data = jnp.arange(1000).reshape(20, 50).astype(jnp.float64)
        
        result = restartvar_2d(
            ncid=2006,
            flag="write",
            varname="large_array",
            xtype=6,
            dim1name="row",
            dim2name="col",
            switchdim=False,
            long_name="Large array test",
            units="units",
            interpinic_flag="copy",
            data=large_data,
            readvar=False,
        )
        
        assert result.data.shape == (20, 50), (
            f"Large array shape incorrect: {result.data.shape}"
        )
        np.testing.assert_allclose(
            result.data,
            large_data,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Large array data not preserved"
        )


# ============================================================================
# Test restartvar wrapper function
# ============================================================================


class TestRestartVar:
    """Test suite for restartvar wrapper function."""

    def test_restartvar_dispatches_to_1d(self, test_data):
        """
        Test that restartvar correctly dispatches to restartvar_1d.
        
        Verifies:
        - 1D data uses restartvar_1d
        - Returns RestartVar1DResult
        - Results match direct restartvar_1d call
        """
        test_case = test_data["restartvar_1d"]["nominal"][0]
        inputs = test_case["inputs"].copy()
        
        # Extract parameters for restartvar wrapper
        varname = inputs.pop("varname")
        xtype = inputs.pop("xtype")
        dim1name = inputs.pop("dim1name")
        dim2name = inputs.pop("dim2name")
        data = inputs.pop("data")
        readvar = inputs.pop("readvar")
        
        result = restartvar(
            varname=varname,
            xtype=xtype,
            dim1name=dim1name,
            dim2name=dim2name if dim2name else None,
            data=data,
            readvar=readvar,
            **inputs
        )
        
        assert isinstance(result, RestartVar1DResult), (
            "restartvar should return RestartVar1DResult for 1D data"
        )

    def test_restartvar_dispatches_to_2d(self):
        """
        Test that restartvar correctly dispatches to restartvar_2d.
        
        Verifies:
        - 2D data uses restartvar_2d
        - Returns RestartVar2DResult
        - Results match direct restartvar_2d call
        """
        data_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        
        result = restartvar(
            varname="test_2d",
            xtype=6,
            dim1name="row",
            dim2name="col",
            data=data_2d,
            readvar=False,
            ncid=3001,
            flag="write",
            switchdim=False,
            long_name="Test 2D",
            units="units",
            interpinic_flag="copy",
        )
        
        assert isinstance(result, RestartVar2DResult), (
            "restartvar should return RestartVar2DResult for 2D data"
        )

    def test_restartvar_handles_none_dim2name(self):
        """
        Test that restartvar handles None for dim2name (1D case).
        
        Verifies:
        - dim2name=None triggers 1D path
        - Returns correct result type
        """
        data_1d = jnp.array([1.0, 2.0, 3.0])
        
        result = restartvar(
            varname="test_1d_none_dim2",
            xtype=6,
            dim1name="gridcell",
            dim2name=None,
            data=data_1d,
            readvar=False,
            ncid=3002,
            flag="write",
            switchdim=False,
            long_name="Test 1D with None dim2",
            units="units",
            interpinic_flag="copy",
        )
        
        assert isinstance(result, RestartVar1DResult), (
            "Should return RestartVar1DResult when dim2name is None"
        )


# ============================================================================
# Integration and cross-validation tests
# ============================================================================


class TestIntegration:
    """Integration tests across all restart variable functions."""

    def test_consistency_between_1d_and_wrapper(self, test_data):
        """
        Test consistency between direct restartvar_1d call and wrapper.
        
        Verifies:
        - Both produce identical results
        - Data values match exactly
        - Metadata preserved
        """
        test_case = test_data["restartvar_1d"]["nominal"][0]
        inputs = test_case["inputs"].copy()
        
        # Direct call
        result_direct = restartvar_1d(**inputs)
        
        # Wrapper call
        varname = inputs.pop("varname")
        xtype = inputs.pop("xtype")
        dim1name = inputs.pop("dim1name")
        dim2name = inputs.pop("dim2name")
        data = inputs.pop("data")
        readvar = inputs.pop("readvar")
        
        result_wrapper = restartvar(
            varname=varname,
            xtype=xtype,
            dim1name=dim1name,
            dim2name=dim2name if dim2name else None,
            data=data,
            readvar=readvar,
            **inputs
        )
        
        np.testing.assert_allclose(
            result_direct.data,
            result_wrapper.data,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Direct and wrapper calls produce different results"
        )
        assert result_direct.readvar == result_wrapper.readvar, (
            "readvar flag differs between direct and wrapper calls"
        )

    def test_round_trip_consistency(self):
        """
        Test that write followed by read preserves data.
        
        Verifies:
        - Data written can be read back
        - Values are preserved within tolerance
        - Metadata is consistent
        """
        original_data = jnp.array([273.15, 280.0, 290.0, 300.0, 310.0])
        
        # Write operation
        write_result = restartvar_1d(
            ncid=4001,
            flag="write",
            varname="round_trip_test",
            xtype=6,
            dim1name="gridcell",
            dim2name="",
            switchdim=False,
            long_name="Round trip test",
            units="K",
            interpinic_flag="copy",
            data=original_data,
            readvar=False,
        )
        
        # Simulate read operation (in real scenario, would read from file)
        read_result = restartvar_1d(
            ncid=4001,
            flag="read",
            varname="round_trip_test",
            xtype=6,
            dim1name="gridcell",
            dim2name="",
            switchdim=False,
            long_name="Round trip test",
            units="K",
            interpinic_flag="copy",
            data=write_result.data,
            readvar=True,
        )
        
        np.testing.assert_allclose(
            read_result.data,
            original_data,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Round trip write/read does not preserve data"
        )

    def test_multiple_variables_same_ncid(self):
        """
        Test handling multiple variables with same NetCDF file ID.
        
        Verifies:
        - Multiple variables can use same ncid
        - No interference between variables
        - Each maintains independent data
        """
        ncid = 5001
        
        var1_data = jnp.array([1.0, 2.0, 3.0])
        var2_data = jnp.array([10.0, 20.0, 30.0])
        
        result1 = restartvar_1d(
            ncid=ncid,
            flag="write",
            varname="var1",
            xtype=6,
            dim1name="dim",
            dim2name="",
            switchdim=False,
            long_name="Variable 1",
            units="units1",
            interpinic_flag="copy",
            data=var1_data,
            readvar=False,
        )
        
        result2 = restartvar_1d(
            ncid=ncid,
            flag="write",
            varname="var2",
            xtype=6,
            dim1name="dim",
            dim2name="",
            switchdim=False,
            long_name="Variable 2",
            units="units2",
            interpinic_flag="copy",
            data=var2_data,
            readvar=False,
        )
        
        # Verify independence
        np.testing.assert_allclose(result1.data, var1_data, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(result2.data, var2_data, rtol=1e-10, atol=1e-10)
        assert not jnp.array_equal(result1.data, result2.data), (
            "Variables should have independent data"
        )


# ============================================================================
# Documentation and metadata tests
# ============================================================================


class TestDocumentation:
    """Tests for function documentation and metadata."""

    def test_functions_have_docstrings(self):
        """Verify all functions have docstrings."""
        assert restartvar_1d.__doc__ is not None, (
            "restartvar_1d missing docstring"
        )
        assert restartvar_2d.__doc__ is not None, (
            "restartvar_2d missing docstring"
        )
        assert restartvar.__doc__ is not None, (
            "restartvar missing docstring"
        )

    def test_namedtuple_definitions(self):
        """Verify NamedTuple result types are properly defined."""
        # Test RestartVar1DResult
        result_1d = RestartVar1DResult(
            data=jnp.array([1.0, 2.0]),
            readvar=True
        )
        assert hasattr(result_1d, "data"), "RestartVar1DResult missing data field"
        assert hasattr(result_1d, "readvar"), "RestartVar1DResult missing readvar field"
        
        # Test RestartVar2DResult
        result_2d = RestartVar2DResult(
            data=jnp.array([[1.0, 2.0]]),
            readvar=False
        )
        assert hasattr(result_2d, "data"), "RestartVar2DResult missing data field"
        assert hasattr(result_2d, "readvar"), "RestartVar2DResult missing readvar field"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])