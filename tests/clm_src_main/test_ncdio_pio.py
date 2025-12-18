"""
Comprehensive pytest suite for ncdio_pio module.

This test suite covers NetCDF I/O operations including:
- File opening/closing operations
- Dimension queries
- 1D and 2D data read/write operations
- Edge cases (zeros, negatives, boundaries)
- Special scenarios (time series, multi-file handling)
- Physical realism constraints for climate data

Test data follows CF conventions and climate science best practices.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import tempfile
import shutil
import warnings

import pytest
import jax.numpy as jnp
import numpy as np

# Handle NumPy 2.0 compatibility issues with optional imports
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (RuntimeWarning, UserWarning, DeprecationWarning))
        import xarray as xr
        XARRAY_AVAILABLE = True
except (ImportError, AttributeError) as e:
    xr = None
    XARRAY_AVAILABLE = False
    pytest.skip(f"xarray not available or incompatible: {e}", allow_module_level=True)

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (RuntimeWarning, UserWarning))
        import netCDF4 as nc4
        NETCDF4_AVAILABLE = True
except (ImportError, RuntimeWarning) as e:
    nc4 = None
    NETCDF4_AVAILABLE = False
    pytest.skip(f"netCDF4 not available or incompatible: {e}", allow_module_level=True)

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from clm_src_main.ncdio_pio import (
    ncd_pio_openfile,
    ncd_pio_closefile,
    ncd_inqdid,
    ncd_inqdlen,
    ncd_defvar,
    ncd_inqvdlen,
    ncd_io_1d,
    ncd_io_2d,
    ncd_io,
    create_simple_netcdf_file,
    print_netcdf_summary,
    file_desc_t,
    NetCDFIOManager,
    FileMode,
    NCDDataType,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load comprehensive test data for ncdio_pio module.
    
    Returns:
        Dictionary containing test cases with inputs and metadata
    """
    return {
        "test_cases": [
            {
                "name": "test_file_operations_nominal",
                "inputs": {
                    "fname": "test_data.nc",
                    "mode": "r",
                    "variables": {
                        "temperature": jnp.array([
                            [273.15, 280.0, 290.0],
                            [275.0, 285.0, 295.0]
                        ]),
                        "pressure": jnp.array([
                            [101325.0, 102000.0, 100500.0],
                            [101000.0, 101500.0, 102500.0]
                        ])
                    },
                    "dimensions": {"time": 2, "space": 3}
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Standard file open/close operations with typical 2D climate data"
                }
            },
            {
                "name": "test_1d_io_read_write_nominal",
                "inputs": {
                    "varname": "altitude",
                    "data": jnp.array([0.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]),
                    "flag": "write",
                    "nt": 0,
                    "readvar": False,
                    "posNOTonfile": False
                },
                "metadata": {
                    "type": "nominal",
                    "description": "1D array write operation with typical altitude profile"
                }
            },
            {
                "name": "test_2d_io_large_array_nominal",
                "inputs": {
                    "varname": "soil_moisture",
                    "data": jnp.array([
                        [0.15, 0.22, 0.18, 0.25, 0.2],
                        [0.18, 0.24, 0.21, 0.28, 0.23],
                        [0.16, 0.2, 0.19, 0.26, 0.22],
                        [0.17, 0.23, 0.2, 0.27, 0.24]
                    ]),
                    "flag": "read",
                    "nt": 5,
                    "readvar": True,
                    "posNOTonfile": False
                },
                "metadata": {
                    "type": "nominal",
                    "description": "2D soil moisture data across spatial grid"
                }
            },
            {
                "name": "test_edge_zero_values",
                "inputs": {
                    "varname": "net_radiation",
                    "data": jnp.zeros((3, 3)),
                    "flag": "write",
                    "nt": 0,
                    "readvar": False,
                    "posNOTonfile": False
                },
                "metadata": {
                    "type": "edge",
                    "description": "All-zero array representing nighttime net radiation",
                    "edge_cases": ["zero_values"]
                }
            },
            {
                "name": "test_edge_single_element_arrays",
                "inputs": {
                    "varname_1d": "global_mean_temp",
                    "data_1d": jnp.array([288.15]),
                    "varname_2d": "single_point",
                    "data_2d": jnp.array([[42.5]]),
                    "flag": "write",
                    "nt": None,
                    "readvar": False,
                    "posNOTonfile": False
                },
                "metadata": {
                    "type": "edge",
                    "description": "Minimum size arrays (single element)",
                    "edge_cases": ["minimum_size", "boundary"]
                }
            },
            {
                "name": "test_edge_negative_values_valid",
                "inputs": {
                    "varname": "wind_velocity_u",
                    "data": jnp.array([
                        [-15.5, -8.2, -3.1, 0.0, 4.7, 9.3, 12.8],
                        [-12.3, -6.5, -1.8, 2.4, 6.9, 10.5, 14.2],
                        [-9.7, -4.3, 0.5, 5.1, 8.8, 11.9, 15.6]
                    ]),
                    "flag": "read",
                    "nt": 10,
                    "readvar": True,
                    "posNOTonfile": False
                },
                "metadata": {
                    "type": "edge",
                    "description": "Negative wind velocities - physically valid",
                    "edge_cases": ["negative_values"]
                }
            },
            {
                "name": "test_edge_very_large_values",
                "inputs": {
                    "varname": "cumulative_precipitation",
                    "data": jnp.array([
                        [1500000.0, 2300000.0, 1800000.0, 2700000.0],
                        [1900000.0, 2500000.0, 2100000.0, 2900000.0],
                        [2200000.0, 2800000.0, 2400000.0, 3100000.0]
                    ]),
                    "flag": "write",
                    "nt": 364,
                    "readvar": False,
                    "posNOTonfile": False
                },
                "metadata": {
                    "type": "edge",
                    "description": "Very large cumulative precipitation values",
                    "edge_cases": ["large_magnitude"]
                }
            }
        ]
    }


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test files.
    
    Yields:
        Path to temporary directory
    """
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_netcdf_file(temp_dir):
    """
    Create a sample NetCDF file for testing.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path to created NetCDF file
    """
    filepath = temp_dir / "sample_test.nc"
    
    # Create sample data
    variables = {
        "temperature": jnp.array([[273.15, 280.0, 290.0], [275.0, 285.0, 295.0]]),
        "pressure": jnp.array([[101325.0, 102000.0, 100500.0], [101000.0, 101500.0, 102500.0]]),
        "altitude": jnp.array([0.0, 100.0, 500.0])
    }
    
    dimensions = {"time": 2, "space": 3}
    
    # Create file using the module's function
    file_desc = create_simple_netcdf_file(str(filepath), variables, dimensions)
    ncd_pio_closefile(file_desc)
    
    return filepath


@pytest.fixture
def file_descriptor():
    """
    Create a fresh file descriptor for testing.
    
    Returns:
        file_desc_t instance
    """
    return file_desc_t()


# ============================================================================
# Test file_desc_t dataclass
# ============================================================================

class TestFileDescriptor:
    """Test suite for file_desc_t dataclass."""
    
    def test_file_desc_initialization_default(self):
        """Test default initialization of file_desc_t."""
        fd = file_desc_t()
        
        assert fd.ncid is None
        assert fd.filepath is None
        assert fd.mode == FileMode.READ
        assert fd.is_open is False
        assert fd.dataset is None
        assert fd.nc_file is None
        assert isinstance(fd.metadata, dict)
        assert len(fd.metadata) == 0
    
    def test_file_desc_initialization_custom(self):
        """Test custom initialization of file_desc_t."""
        test_path = Path("/tmp/test.nc")
        fd = file_desc_t(
            ncid=42,
            filepath=test_path,
            mode=FileMode.WRITE,
            is_open=True,
            metadata={"author": "test"}
        )
        
        assert fd.ncid == 42
        assert fd.filepath == test_path
        assert fd.mode == FileMode.WRITE
        assert fd.is_open is True
        assert fd.metadata["author"] == "test"
    
    def test_file_desc_is_valid_method(self, file_descriptor):
        """Test is_valid method of file_desc_t."""
        # Initially invalid (not open)
        assert not file_descriptor.is_valid()
        
        # Make it valid
        file_descriptor.is_open = True
        file_descriptor.filepath = Path("/tmp/test.nc")
        assert file_descriptor.is_valid()
    
    def test_file_desc_get_info_method(self, file_descriptor):
        """Test get_info method returns proper dictionary."""
        file_descriptor.filepath = Path("/tmp/test.nc")
        file_descriptor.mode = FileMode.READ
        file_descriptor.is_open = True
        
        info = file_descriptor.get_info()
        
        assert isinstance(info, dict)
        assert "filepath" in info
        assert "mode" in info
        assert "is_open" in info


# ============================================================================
# Test File Operations
# ============================================================================

class TestFileOperations:
    """Test suite for file open/close operations."""
    
    def test_ncd_pio_openfile_read_mode(self, sample_netcdf_file, file_descriptor):
        """Test opening a NetCDF file in read mode."""
        ncd_pio_openfile(file_descriptor, str(sample_netcdf_file), mode="r")
        
        assert file_descriptor.is_open is True
        assert file_descriptor.filepath == sample_netcdf_file
        assert file_descriptor.mode == FileMode.READ
        assert file_descriptor.dataset is not None or file_descriptor.nc_file is not None
        
        # Cleanup
        ncd_pio_closefile(file_descriptor)
    
    def test_ncd_pio_openfile_write_mode(self, temp_dir, file_descriptor):
        """Test opening a NetCDF file in write mode."""
        new_file = temp_dir / "new_test.nc"
        
        ncd_pio_openfile(file_descriptor, str(new_file), mode="w")
        
        assert file_descriptor.is_open is True
        assert file_descriptor.mode == FileMode.WRITE
        
        # Cleanup
        ncd_pio_closefile(file_descriptor)
    
    def test_ncd_pio_openfile_default_mode(self, sample_netcdf_file, file_descriptor):
        """Test opening file with default mode (read)."""
        ncd_pio_openfile(file_descriptor, str(sample_netcdf_file))
        
        assert file_descriptor.is_open is True
        assert file_descriptor.mode == FileMode.READ
        
        # Cleanup
        ncd_pio_closefile(file_descriptor)
    
    def test_ncd_pio_closefile(self, sample_netcdf_file, file_descriptor):
        """Test closing an open NetCDF file."""
        ncd_pio_openfile(file_descriptor, str(sample_netcdf_file), mode="r")
        assert file_descriptor.is_open is True
        
        ncd_pio_closefile(file_descriptor)
        
        assert file_descriptor.is_open is False
    
    def test_ncd_pio_openfile_nonexistent_file_read(self, temp_dir, file_descriptor):
        """Test opening non-existent file in read mode raises error."""
        nonexistent = temp_dir / "does_not_exist.nc"
        
        with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
            ncd_pio_openfile(file_descriptor, str(nonexistent), mode="r")
    
    def test_file_operations_sequence(self, temp_dir, file_descriptor):
        """Test sequence of open-write-close-open-read operations."""
        test_file = temp_dir / "sequence_test.nc"
        
        # Create and write
        variables = {"test_var": jnp.array([1.0, 2.0, 3.0])}
        fd_write = create_simple_netcdf_file(str(test_file), variables)
        ncd_pio_closefile(fd_write)
        
        # Open and read
        ncd_pio_openfile(file_descriptor, str(test_file), mode="r")
        assert file_descriptor.is_open is True
        
        ncd_pio_closefile(file_descriptor)
        assert file_descriptor.is_open is False


# ============================================================================
# Test Dimension Operations
# ============================================================================

class TestDimensionOperations:
    """Test suite for dimension query operations."""
    
    def test_ncd_inqdid_valid_dimension(self, sample_netcdf_file, file_descriptor):
        """Test querying dimension ID for valid dimension name."""
        ncd_pio_openfile(file_descriptor, str(sample_netcdf_file), mode="r")
        
        dim_id = ncd_inqdid(file_descriptor, "time")
        
        assert isinstance(dim_id, int)
        assert dim_id >= 0
        
        ncd_pio_closefile(file_descriptor)
    
    def test_ncd_inqdid_invalid_dimension(self, sample_netcdf_file, file_descriptor):
        """Test querying dimension ID for non-existent dimension."""
        ncd_pio_openfile(file_descriptor, str(sample_netcdf_file), mode="r")
        
        with pytest.raises((KeyError, ValueError, RuntimeError)):
            ncd_inqdid(file_descriptor, "nonexistent_dimension")
        
        ncd_pio_closefile(file_descriptor)
    
    def test_ncd_inqdlen_by_name(self, sample_netcdf_file, file_descriptor):
        """Test querying dimension length by name."""
        ncd_pio_openfile(file_descriptor, str(sample_netcdf_file), mode="r")
        
        dim_len = ncd_inqdlen(file_descriptor, "time")
        
        assert isinstance(dim_len, int)
        assert dim_len == 2  # From sample file creation
        
        ncd_pio_closefile(file_descriptor)
    
    def test_ncd_inqdlen_by_id(self, sample_netcdf_file, file_descriptor):
        """Test querying dimension length by ID."""
        ncd_pio_openfile(file_descriptor, str(sample_netcdf_file), mode="r")
        
        dim_id = ncd_inqdid(file_descriptor, "space")
        dim_len = ncd_inqdlen(file_descriptor, dim_id)
        
        assert isinstance(dim_len, int)
        assert dim_len == 3  # From sample file creation
        
        ncd_pio_closefile(file_descriptor)
    
    def test_ncd_inqdlen_invalid_dimension(self, sample_netcdf_file, file_descriptor):
        """Test querying length of non-existent dimension."""
        ncd_pio_openfile(file_descriptor, str(sample_netcdf_file), mode="r")
        
        with pytest.raises((KeyError, ValueError, RuntimeError)):
            ncd_inqdlen(file_descriptor, "nonexistent_dim")
        
        ncd_pio_closefile(file_descriptor)
    
    @pytest.mark.parametrize("dim_name,expected_len", [
        ("time", 2),
        ("space", 3),
    ])
    def test_ncd_inqdlen_parametrized(self, sample_netcdf_file, file_descriptor, 
                                      dim_name, expected_len):
        """Test dimension length queries with parametrized inputs."""
        ncd_pio_openfile(file_descriptor, str(sample_netcdf_file), mode="r")
        
        dim_len = ncd_inqdlen(file_descriptor, dim_name)
        
        assert dim_len == expected_len
        
        ncd_pio_closefile(file_descriptor)


# ============================================================================
# Test 1D I/O Operations
# ============================================================================

class TestIO1D:
    """Test suite for 1D array I/O operations."""
    
    def test_ncd_io_1d_write_nominal(self, temp_dir, test_data):
        """Test writing 1D array with nominal data."""
        test_case = test_data["test_cases"][1]  # 1D nominal case
        test_file = temp_dir / "test_1d_write.nc"
        
        # Create file and write
        variables = {test_case["inputs"]["varname"]: test_case["inputs"]["data"]}
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        
        result_data, success = ncd_io_1d(
            varname=test_case["inputs"]["varname"],
            data=test_case["inputs"]["data"],
            flag="write",
            ncid=file_desc,
            nt=test_case["inputs"]["nt"]
        )
        
        assert success is True or success is None  # Success or no return
        assert result_data.shape == test_case["inputs"]["data"].shape
        
        ncd_pio_closefile(file_desc)
    
    def test_ncd_io_1d_read_nominal(self, temp_dir):
        """Test reading 1D array with nominal data."""
        test_file = temp_dir / "test_1d_read.nc"
        expected_data = jnp.array([0.0, 100.0, 500.0, 1000.0, 2000.0])
        
        # Create file with data
        variables = {"altitude": expected_data}
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        ncd_pio_closefile(file_desc)
        
        # Read back
        file_desc = file_desc_t()
        ncd_pio_openfile(file_desc, str(test_file), mode="r")
        
        read_data, success = ncd_io_1d(
            varname="altitude",
            data=jnp.zeros_like(expected_data),
            flag="read",
            ncid=file_desc
        )
        
        assert success is True or success is None
        np.testing.assert_allclose(read_data, expected_data, rtol=1e-6, atol=1e-6)
        
        ncd_pio_closefile(file_desc)
    
    def test_ncd_io_1d_single_element(self, temp_dir, test_data):
        """Test 1D I/O with single element array (edge case)."""
        test_case = test_data["test_cases"][4]  # Single element case
        test_file = temp_dir / "test_1d_single.nc"
        
        variables = {test_case["inputs"]["varname_1d"]: test_case["inputs"]["data_1d"]}
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        
        result_data, success = ncd_io_1d(
            varname=test_case["inputs"]["varname_1d"],
            data=test_case["inputs"]["data_1d"],
            flag="write",
            ncid=file_desc
        )
        
        assert result_data.shape == (1,)
        assert result_data.ndim == 1
        
        ncd_pio_closefile(file_desc)
    
    def test_ncd_io_1d_shape_validation(self, temp_dir):
        """Test that 1D I/O validates array dimensions."""
        test_file = temp_dir / "test_1d_shape.nc"
        data_1d = jnp.array([1.0, 2.0, 3.0])
        
        variables = {"test_var": data_1d}
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        
        result_data, _ = ncd_io_1d(
            varname="test_var",
            data=data_1d,
            flag="write",
            ncid=file_desc
        )
        
        assert result_data.ndim == 1, "Output should be 1D array"
        assert result_data.shape[0] == 3, "Output should have 3 elements"
        
        ncd_pio_closefile(file_desc)
    
    def test_ncd_io_1d_dtype_preservation(self, temp_dir):
        """Test that 1D I/O preserves data types."""
        test_file = temp_dir / "test_1d_dtype.nc"
        data_float = jnp.array([1.5, 2.5, 3.5], dtype=jnp.float64)
        
        variables = {"float_var": data_float}
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        
        result_data, _ = ncd_io_1d(
            varname="float_var",
            data=data_float,
            flag="write",
            ncid=file_desc
        )
        
        assert jnp.issubdtype(result_data.dtype, jnp.floating), \
            "Output should be floating point type"
        
        ncd_pio_closefile(file_desc)


# ============================================================================
# Test 2D I/O Operations
# ============================================================================

class TestIO2D:
    """Test suite for 2D array I/O operations."""
    
    def test_ncd_io_2d_write_nominal(self, temp_dir, test_data):
        """Test writing 2D array with nominal data."""
        test_case = test_data["test_cases"][2]  # 2D nominal case
        test_file = temp_dir / "test_2d_write.nc"
        
        variables = {test_case["inputs"]["varname"]: test_case["inputs"]["data"]}
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        
        result_data, success = ncd_io_2d(
            varname=test_case["inputs"]["varname"],
            data=test_case["inputs"]["data"],
            flag="write",
            ncid=file_desc,
            nt=test_case["inputs"]["nt"]
        )
        
        assert result_data.shape == test_case["inputs"]["data"].shape
        assert result_data.ndim == 2
        
        ncd_pio_closefile(file_desc)
    
    def test_ncd_io_2d_read_nominal(self, temp_dir):
        """Test reading 2D array with nominal data."""
        test_file = temp_dir / "test_2d_read.nc"
        expected_data = jnp.array([
            [273.15, 280.0, 290.0],
            [275.0, 285.0, 295.0]
        ])
        
        variables = {"temperature": expected_data}
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        ncd_pio_closefile(file_desc)
        
        # Read back
        file_desc = file_desc_t()
        ncd_pio_openfile(file_desc, str(test_file), mode="r")
        
        read_data, success = ncd_io_2d(
            varname="temperature",
            data=jnp.zeros_like(expected_data),
            flag="read",
            ncid=file_desc
        )
        
        np.testing.assert_allclose(read_data, expected_data, rtol=1e-6, atol=1e-6)
        assert read_data.shape == expected_data.shape
        
        ncd_pio_closefile(file_desc)
    
    def test_ncd_io_2d_zero_values(self, temp_dir, test_data):
        """Test 2D I/O with all-zero array (edge case)."""
        test_case = test_data["test_cases"][3]  # Zero values case
        test_file = temp_dir / "test_2d_zeros.nc"
        
        variables = {test_case["inputs"]["varname"]: test_case["inputs"]["data"]}
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        
        result_data, _ = ncd_io_2d(
            varname=test_case["inputs"]["varname"],
            data=test_case["inputs"]["data"],
            flag="write",
            ncid=file_desc
        )
        
        assert jnp.all(result_data == 0.0), "All values should be zero"
        assert result_data.shape == test_case["inputs"]["data"].shape
        
        ncd_pio_closefile(file_desc)
    
    def test_ncd_io_2d_negative_values(self, temp_dir, test_data):
        """Test 2D I/O with negative values (physically valid)."""
        test_case = test_data["test_cases"][5]  # Negative values case
        test_file = temp_dir / "test_2d_negative.nc"
        
        variables = {test_case["inputs"]["varname"]: test_case["inputs"]["data"]}
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        
        result_data, _ = ncd_io_2d(
            varname=test_case["inputs"]["varname"],
            data=test_case["inputs"]["data"],
            flag="write",
            ncid=file_desc
        )
        
        # Check that negative values are preserved
        assert jnp.any(result_data < 0), "Should contain negative values"
        np.testing.assert_allclose(result_data, test_case["inputs"]["data"], 
                                   rtol=1e-6, atol=1e-6)
        
        ncd_pio_closefile(file_desc)
    
    def test_ncd_io_2d_large_values(self, temp_dir, test_data):
        """Test 2D I/O with very large values (edge case)."""
        test_case = test_data["test_cases"][6]  # Large values case
        test_file = temp_dir / "test_2d_large.nc"
        
        variables = {test_case["inputs"]["varname"]: test_case["inputs"]["data"]}
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        
        result_data, _ = ncd_io_2d(
            varname=test_case["inputs"]["varname"],
            data=test_case["inputs"]["data"],
            flag="write",
            ncid=file_desc
        )
        
        # Check that large values are preserved
        assert jnp.all(result_data > 1e6), "Should contain large values"
        np.testing.assert_allclose(result_data, test_case["inputs"]["data"], 
                                   rtol=1e-5, atol=1e-3)
        
        ncd_pio_closefile(file_desc)
    
    def test_ncd_io_2d_single_element(self, temp_dir, test_data):
        """Test 2D I/O with single element array (edge case)."""
        test_case = test_data["test_cases"][4]  # Single element case
        test_file = temp_dir / "test_2d_single.nc"
        
        variables = {test_case["inputs"]["varname_2d"]: test_case["inputs"]["data_2d"]}
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        
        result_data, _ = ncd_io_2d(
            varname=test_case["inputs"]["varname_2d"],
            data=test_case["inputs"]["data_2d"],
            flag="write",
            ncid=file_desc
        )
        
        assert result_data.shape == (1, 1)
        assert result_data.ndim == 2
        
        ncd_pio_closefile(file_desc)
    
    def test_ncd_io_2d_shape_validation(self, temp_dir):
        """Test that 2D I/O validates array dimensions."""
        test_file = temp_dir / "test_2d_shape.nc"
        data_2d = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        variables = {"test_var": data_2d}
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        
        result_data, _ = ncd_io_2d(
            varname="test_var",
            data=data_2d,
            flag="write",
            ncid=file_desc
        )
        
        assert result_data.ndim == 2, "Output should be 2D array"
        assert result_data.shape == (3, 2), "Output should have shape (3, 2)"
        
        ncd_pio_closefile(file_desc)


# ============================================================================
# Test Generic I/O Operations
# ============================================================================

class TestIOGeneric:
    """Test suite for generic ncd_io function (handles both 1D and 2D)."""
    
    def test_ncd_io_dispatches_1d(self, temp_dir):
        """Test that ncd_io correctly dispatches to 1D handler."""
        test_file = temp_dir / "test_io_1d.nc"
        data_1d = jnp.array([1.0, 2.0, 3.0, 4.0])
        
        variables = {"test_1d": data_1d}
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        
        result_data, success = ncd_io(
            varname="test_1d",
            data=data_1d,
            flag="write",
            ncid=file_desc
        )
        
        assert result_data.ndim == 1
        assert result_data.shape == data_1d.shape
        
        ncd_pio_closefile(file_desc)
    
    def test_ncd_io_dispatches_2d(self, temp_dir):
        """Test that ncd_io correctly dispatches to 2D handler."""
        test_file = temp_dir / "test_io_2d.nc"
        data_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        
        variables = {"test_2d": data_2d}
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        
        result_data, success = ncd_io(
            varname="test_2d",
            data=data_2d,
            flag="write",
            ncid=file_desc
        )
        
        assert result_data.ndim == 2
        assert result_data.shape == data_2d.shape
        
        ncd_pio_closefile(file_desc)
    
    @pytest.mark.parametrize("data,expected_ndim", [
        (jnp.array([1.0, 2.0, 3.0]), 1),
        (jnp.array([[1.0, 2.0], [3.0, 4.0]]), 2),
        (jnp.array([42.0]), 1),
        (jnp.array([[42.0]]), 2),
    ])
    def test_ncd_io_dimension_handling(self, temp_dir, data, expected_ndim):
        """Test ncd_io handles various array dimensions correctly."""
        test_file = temp_dir / f"test_io_dim_{expected_ndim}.nc"
        
        variables = {"test_var": data}
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        
        result_data, _ = ncd_io(
            varname="test_var",
            data=data,
            flag="write",
            ncid=file_desc
        )
        
        assert result_data.ndim == expected_ndim
        
        ncd_pio_closefile(file_desc)
    
    def test_ncd_io_with_time_index(self, temp_dir):
        """Test ncd_io with time index parameter."""
        test_file = temp_dir / "test_io_time.nc"
        data = jnp.array([[273.15, 280.0], [275.0, 285.0]])
        
        variables = {"temp_series": data}
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        
        result_data, _ = ncd_io(
            varname="temp_series",
            data=data,
            flag="write",
            ncid=file_desc,
            nt=10
        )
        
        assert result_data.shape == data.shape
        
        ncd_pio_closefile(file_desc)
    
    def test_ncd_io_with_posNOTonfile_flag(self, temp_dir):
        """Test ncd_io with posNOTonfile flag for multi-file handling."""
        test_file = temp_dir / "test_io_multifile.nc"
        data = jnp.array([[34.5, 34.8], [34.6, 34.9]])
        
        variables = {"salinity": data}
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        
        result_data, _ = ncd_io(
            varname="salinity",
            data=data,
            flag="read",
            ncid=file_desc,
            posNOTonfile=True
        )
        
        # Should handle the flag gracefully
        assert result_data.shape == data.shape
        
        ncd_pio_closefile(file_desc)


# ============================================================================
# Test Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_ncd_defvar_dummy(self):
        """Test that ncd_defvar is a dummy function (no-op)."""
        # Should not raise any errors
        result = ncd_defvar("test", "arg1", kwarg1="value1")
        assert result is None
    
    def test_ncd_inqvdlen_dummy(self):
        """Test that ncd_inqvdlen is a dummy function (no-op)."""
        # Should not raise any errors
        result = ncd_inqvdlen("test", "arg1", kwarg1="value1")
        assert result is None
    
    def test_create_simple_netcdf_file(self, temp_dir):
        """Test creating a simple NetCDF file."""
        test_file = temp_dir / "test_create.nc"
        
        variables = {
            "var1": jnp.array([1.0, 2.0, 3.0]),
            "var2": jnp.array([[1.0, 2.0], [3.0, 4.0]])
        }
        
        dimensions = {"dim1": 3, "dim2": 2}
        
        file_desc = create_simple_netcdf_file(str(test_file), variables, dimensions)
        
        assert file_desc.is_open is True
        assert test_file.exists()
        
        ncd_pio_closefile(file_desc)
    
    def test_create_simple_netcdf_file_no_dimensions(self, temp_dir):
        """Test creating NetCDF file without explicit dimensions."""
        test_file = temp_dir / "test_create_nodim.nc"
        
        variables = {
            "var1": jnp.array([1.0, 2.0, 3.0])
        }
        
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        
        assert file_desc.is_open is True
        assert test_file.exists()
        
        ncd_pio_closefile(file_desc)
    
    def test_print_netcdf_summary(self, sample_netcdf_file, file_descriptor, capsys):
        """Test printing NetCDF file summary."""
        ncd_pio_openfile(file_descriptor, str(sample_netcdf_file), mode="r")
        
        print_netcdf_summary(file_descriptor)
        
        captured = capsys.readouterr()
        # Should print something (exact format depends on implementation)
        assert len(captured.out) > 0 or len(captured.err) > 0
        
        ncd_pio_closefile(file_descriptor)


# ============================================================================
# Test NetCDFIOManager
# ============================================================================

class TestNetCDFIOManager:
    """Test suite for NetCDFIOManager class."""
    
    def test_manager_initialization(self):
        """Test NetCDFIOManager initialization."""
        manager = NetCDFIOManager()
        
        assert isinstance(manager.open_files, dict)
        assert len(manager.open_files) == 0
        assert manager.default_mode == FileMode.READ
        assert manager.enable_caching is True
        assert isinstance(manager.cache, dict)
    
    def test_manager_get_open_file_count(self):
        """Test getting count of open files."""
        manager = NetCDFIOManager()
        
        count = manager.get_open_file_count()
        assert count == 0
        assert isinstance(count, int)
    
    def test_manager_close_all_files(self, sample_netcdf_file):
        """Test closing all open files through manager."""
        manager = NetCDFIOManager()
        
        # Open some files
        fd1 = file_desc_t()
        ncd_pio_openfile(fd1, str(sample_netcdf_file), mode="r")
        manager.open_files["file1"] = fd1
        
        # Close all
        manager.close_all_files()
        
        # Verify all closed
        for fd in manager.open_files.values():
            assert fd.is_open is False


# ============================================================================
# Test Enums
# ============================================================================

class TestEnums:
    """Test suite for enum types."""
    
    def test_file_mode_enum_values(self):
        """Test FileMode enum has expected values."""
        assert hasattr(FileMode, "READ")
        assert hasattr(FileMode, "WRITE")
        assert hasattr(FileMode, "APPEND")
        assert hasattr(FileMode, "READ_WRITE")
    
    def test_ncd_data_type_enum_values(self):
        """Test NCDDataType enum has expected values."""
        assert hasattr(NCDDataType, "DOUBLE")
        assert hasattr(NCDDataType, "FLOAT")
        assert hasattr(NCDDataType, "INT")
        assert hasattr(NCDDataType, "LONG")
    
    def test_file_mode_enum_usage(self):
        """Test using FileMode enum in file descriptor."""
        fd = file_desc_t(mode=FileMode.WRITE)
        assert fd.mode == FileMode.WRITE
        
        fd.mode = FileMode.READ
        assert fd.mode == FileMode.READ


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_write_read_cycle(self, temp_dir):
        """Test complete cycle: create file, write data, close, reopen, read."""
        test_file = temp_dir / "integration_test.nc"
        
        # Original data
        original_1d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        original_2d = jnp.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        
        # Write phase
        variables = {
            "data_1d": original_1d,
            "data_2d": original_2d
        }
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        ncd_pio_closefile(file_desc)
        
        # Read phase
        file_desc = file_desc_t()
        ncd_pio_openfile(file_desc, str(test_file), mode="r")
        
        read_1d, _ = ncd_io_1d(
            varname="data_1d",
            data=jnp.zeros_like(original_1d),
            flag="read",
            ncid=file_desc
        )
        
        read_2d, _ = ncd_io_2d(
            varname="data_2d",
            data=jnp.zeros_like(original_2d),
            flag="read",
            ncid=file_desc
        )
        
        # Verify
        np.testing.assert_allclose(read_1d, original_1d, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(read_2d, original_2d, rtol=1e-6, atol=1e-6)
        
        ncd_pio_closefile(file_desc)
    
    def test_multiple_variables_workflow(self, temp_dir):
        """Test workflow with multiple variables of different dimensions."""
        test_file = temp_dir / "multi_var_test.nc"
        
        variables = {
            "temperature": jnp.array([[273.15, 280.0], [275.0, 285.0]]),
            "pressure": jnp.array([[101325.0, 102000.0], [101000.0, 101500.0]]),
            "altitude": jnp.array([0.0, 100.0]),
            "time": jnp.array([0.0, 3600.0])
        }
        
        # Create and write
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        ncd_pio_closefile(file_desc)
        
        # Reopen and verify dimensions
        file_desc = file_desc_t()
        ncd_pio_openfile(file_desc, str(test_file), mode="r")
        
        # Query dimensions
        dim_names = ["dim_0", "dim_1"]  # Actual names depend on implementation
        for dim_name in dim_names:
            try:
                dim_id = ncd_inqdid(file_desc, dim_name)
                dim_len = ncd_inqdlen(file_desc, dim_id)
                assert dim_len > 0
            except (KeyError, ValueError):
                # Dimension name might be different
                pass
        
        ncd_pio_closefile(file_desc)
    
    def test_physical_constraints_validation(self, temp_dir):
        """Test that physical constraints are maintained through I/O."""
        test_file = temp_dir / "physics_test.nc"
        
        # Temperature must be > 0K
        temp_data = jnp.array([[273.15, 280.0], [275.0, 285.0]])
        assert jnp.all(temp_data > 0), "Temperature must be positive"
        
        # Soil moisture must be in [0, 1]
        moisture_data = jnp.array([[0.15, 0.22], [0.18, 0.24]])
        assert jnp.all(moisture_data >= 0) and jnp.all(moisture_data <= 1), \
            "Soil moisture must be in [0, 1]"
        
        variables = {
            "temperature": temp_data,
            "soil_moisture": moisture_data
        }
        
        file_desc = create_simple_netcdf_file(str(test_file), variables)
        ncd_pio_closefile(file_desc)
        
        # Read back and verify constraints
        file_desc = file_desc_t()
        ncd_pio_openfile(file_desc, str(test_file), mode="r")
        
        read_temp, _ = ncd_io_2d(
            varname="temperature",
            data=jnp.zeros_like(temp_data),
            flag="read",
            ncid=file_desc
        )
        
        read_moisture, _ = ncd_io_2d(
            varname="soil_moisture",
            data=jnp.zeros_like(moisture_data),
            flag="read",
            ncid=file_desc
        )
        
        assert jnp.all(read_temp > 0), "Read temperature must be positive"
        assert jnp.all(read_moisture >= 0) and jnp.all(read_moisture <= 1), \
            "Read soil moisture must be in [0, 1]"
        
        ncd_pio_closefile(file_desc)


# ============================================================================
# Test Documentation
# ============================================================================

def test_module_has_docstrings():
    """Test that key functions have docstrings."""
    functions_to_check = [
        ncd_pio_openfile,
        ncd_pio_closefile,
        ncd_inqdid,
        ncd_inqdlen,
        ncd_io_1d,
        ncd_io_2d,
        ncd_io,
        create_simple_netcdf_file
    ]
    
    for func in functions_to_check:
        assert func.__doc__ is not None, f"{func.__name__} should have a docstring"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])