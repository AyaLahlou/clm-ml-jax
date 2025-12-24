"""
Comprehensive pytest suite for GridcellType module.

This test suite covers:
- Dataclass initialization and field validation
- Coordinate setting and retrieval methods
- Index conversion (Python/Fortran)
- Grid information and validation methods
- Utility functions (distance calculation, coordinate conversion)
- Edge cases (polar regions, dateline crossing, boundary values)
- Array dimension validation
- Physical realism constraints
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import pytest
import jax.numpy as jnp
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from clm_src_main.GridcellType import (
    gridcell_type,
    create_gridcell_instance,
    reset_global_gridcell,
    create_regular_grid,
    calculate_distance_haversine,
    degrees_to_radians,
    radians_to_degrees,
    normalize_longitude,
    validate_coordinates,
    create_coordinate_mesh,
    grc
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load comprehensive test data for GridcellType testing.
    
    Returns:
        Dictionary containing test cases with inputs and metadata
    """
    return {
        "test_nominal_single_gridcell": {
            "begg": 1,
            "endg": 1,
            "latitudes": [45.0],
            "longitudes": [-122.5],
            "metadata": {
                "type": "nominal",
                "description": "Single gridcell at mid-latitude location (Portland, OR area)"
            }
        },
        "test_nominal_small_regional_grid": {
            "begg": 0,
            "endg": 9,
            "latitudes": [40.0, 40.5, 41.0, 41.5, 42.0, 42.5, 43.0, 43.5, 44.0, 44.5],
            "longitudes": [-105.0, -104.5, -104.0, -103.5, -103.0, -102.5, -102.0, -101.5, -101.0, -100.5],
            "metadata": {
                "type": "nominal",
                "description": "Small regional grid covering Colorado/Nebraska area with 10 gridcells"
            }
        },
        "test_nominal_global_coarse_grid": {
            "begg": 100,
            "endg": 119,
            "latitudes": [-75.0, -60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0,
                         75.0, -80.0, -65.0, -50.0, -35.0, -20.0, -5.0, 10.0, 25.0, 40.0],
            "longitudes": [0.0, 18.0, 36.0, 54.0, 72.0, 90.0, 108.0, 126.0, 144.0, 162.0,
                          180.0, 198.0, 216.0, 234.0, 252.0, 270.0, 288.0, 306.0, 324.0, 342.0],
            "metadata": {
                "type": "nominal",
                "description": "Coarse global grid with 20 gridcells spanning full latitude/longitude ranges"
            }
        },
        "test_edge_equator_prime_meridian": {
            "begg": 0,
            "endg": 4,
            "latitudes": [0.0, 0.0, 0.0, 0.0, 0.0],
            "longitudes": [0.0, 0.0, 0.0, 0.0, 0.0],
            "metadata": {
                "type": "edge",
                "description": "All gridcells at equator and prime meridian (zero coordinates)",
                "edge_cases": ["zero_latitude", "zero_longitude"]
            }
        },
        "test_edge_polar_regions": {
            "begg": 1,
            "endg": 8,
            "latitudes": [90.0, 89.5, -90.0, -89.5, 85.0, 87.5, -85.0, -87.5],
            "longitudes": [0.0, 45.0, 180.0, -180.0, 90.0, 270.0, -90.0, 135.0],
            "metadata": {
                "type": "edge",
                "description": "Gridcells at and near poles (boundary latitudes)",
                "edge_cases": ["max_latitude", "min_latitude", "polar_convergence"]
            }
        },
        "test_edge_longitude_boundaries": {
            "begg": 50,
            "endg": 56,
            "latitudes": [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            "longitudes": [-180.0, -179.9, 0.0, 179.9, 180.0, 359.9, 360.0],
            "metadata": {
                "type": "edge",
                "description": "Longitude boundary conditions including dateline and wraparound",
                "edge_cases": ["min_longitude", "max_longitude", "dateline_crossing"]
            }
        },
        "test_edge_single_index_begg_equals_endg": {
            "begg": 42,
            "endg": 42,
            "latitudes": [51.5],
            "longitudes": [-0.1],
            "metadata": {
                "type": "edge",
                "description": "Minimum grid size where begg equals endg (London coordinates)",
                "edge_cases": ["minimum_grid_size"]
            }
        },
    }


@pytest.fixture
def clean_global_gridcell():
    """Reset global gridcell instance before each test."""
    reset_global_gridcell()
    yield
    reset_global_gridcell()


# ============================================================================
# Dataclass Initialization Tests
# ============================================================================

class TestGridcellTypeInitialization:
    """Test gridcell_type dataclass initialization and basic properties."""
    
    def test_create_uninitialized_instance(self):
        """Test creating an uninitialized gridcell_type instance."""
        gc = gridcell_type()
        
        assert gc.latdeg is None, "latdeg should be None for uninitialized instance"
        assert gc.londeg is None, "londeg should be None for uninitialized instance"
        assert gc.begg is None, "begg should be None for uninitialized instance"
        assert gc.endg is None, "endg should be None for uninitialized instance"
        assert not gc.is_initialized(), "Uninitialized instance should return False"
    
    @pytest.mark.parametrize("begg,endg", [
        (0, 0),
        (1, 1),
        (0, 10),
        (100, 200),
        (1000, 1999),
    ])
    def test_init_method_valid_indices(self, begg: int, endg: int):
        """Test Init method with valid index ranges."""
        gc = gridcell_type()
        gc.Init(begg, endg)
        
        assert gc.is_initialized(), f"Instance should be initialized for begg={begg}, endg={endg}"
        assert gc.begg == begg, f"begg should be {begg}"
        assert gc.endg == endg, f"endg should be {endg}"
        
        expected_count = endg - begg + 1
        assert gc.get_gridcell_count() == expected_count, \
            f"Gridcell count should be {expected_count}"
        
        # Check array shapes
        assert gc.latdeg.shape == (expected_count,), \
            f"latdeg shape should be ({expected_count},)"
        assert gc.londeg.shape == (expected_count,), \
            f"londeg shape should be ({expected_count},)"
    
    def test_init_method_invalid_indices(self):
        """Test Init method with invalid index ranges (endg < begg)."""
        gc = gridcell_type()
        
        with pytest.raises((ValueError, AssertionError)):
            gc.Init(10, 5)  # endg < begg should raise error


# ============================================================================
# Coordinate Setting and Retrieval Tests
# ============================================================================

class TestCoordinateOperations:
    """Test coordinate setting and retrieval methods."""
    
    @pytest.mark.parametrize("test_case_name", [
        "test_nominal_single_gridcell",
        "test_nominal_small_regional_grid",
        "test_edge_equator_prime_meridian",
        "test_edge_polar_regions",
    ])
    def test_set_coordinates_full_array(self, test_data: Dict, test_case_name: str):
        """Test setting coordinates for all gridcells at once."""
        case = test_data[test_case_name]
        
        gc = gridcell_type()
        gc.Init(case["begg"], case["endg"])
        
        lat_array = jnp.array(case["latitudes"])
        lon_array = jnp.array(case["longitudes"])
        
        gc.set_coordinates(lat_array, lon_array)
        
        # Retrieve and verify
        retrieved_lat, retrieved_lon = gc.get_coordinates()
        
        np.testing.assert_allclose(retrieved_lat, lat_array, rtol=1e-6, atol=1e-6,
                                   err_msg="Retrieved latitudes don't match set values")
        np.testing.assert_allclose(retrieved_lon, lon_array, rtol=1e-6, atol=1e-6,
                                   err_msg="Retrieved longitudes don't match set values")
    
    def test_set_single_coordinate(self):
        """Test setting coordinates for individual gridcells."""
        gc = gridcell_type()
        gc.Init(0, 4)
        
        test_coords = [
            (0, 45.0, -122.5),
            (1, 40.7, -74.0),
            (2, 51.5, -0.1),
            (3, -33.9, 151.2),
            (4, 35.7, 139.7),
        ]
        
        for idx, lat, lon in test_coords:
            gc.set_single_coordinate(idx, lat, lon)
        
        # Verify each coordinate
        for idx, expected_lat, expected_lon in test_coords:
            retrieved_lat, retrieved_lon = gc.get_coordinates(idx)
            
            assert np.isclose(retrieved_lat, expected_lat, rtol=1e-6, atol=1e-6), \
                f"Latitude at index {idx} should be {expected_lat}"
            assert np.isclose(retrieved_lon, expected_lon, rtol=1e-6, atol=1e-6), \
                f"Longitude at index {idx} should be {expected_lon}"
    
    def test_get_coordinates_invalid_index(self):
        """Test retrieving coordinates with invalid index."""
        gc = gridcell_type()
        gc.Init(0, 5)
        
        with pytest.raises((IndexError, ValueError)):
            gc.get_coordinates(10)  # Index out of range
    
    def test_coordinate_array_dimension_mismatch(self):
        """Test setting coordinates with mismatched array dimensions."""
        gc = gridcell_type()
        gc.Init(0, 9)
        
        lat_array = jnp.array([45.0, 46.0])  # Only 2 elements
        lon_array = jnp.array([-122.0, -121.0, -120.0])  # 3 elements
        
        with pytest.raises((ValueError, AssertionError)):
            gc.set_coordinates(lat_array, lon_array)


# ============================================================================
# Index Conversion Tests
# ============================================================================

class TestIndexConversion:
    """Test Python/Fortran index conversion methods."""
    
    @pytest.mark.parametrize("begg,endg,python_idx,expected_fortran", [
        (0, 10, 0, 0),
        (0, 10, 5, 5),
        (0, 10, 10, 10),
        (1, 10, 0, 1),
        (1, 10, 5, 6),
        (100, 200, 0, 100),
        (100, 200, 50, 150),
    ])
    def test_get_fortran_index(self, begg: int, endg: int, 
                               python_idx: int, expected_fortran: int):
        """Test conversion from Python (0-based) to Fortran (offset by begg) index."""
        gc = gridcell_type()
        gc.Init(begg, endg)
        
        fortran_idx = gc.get_fortran_index(python_idx)
        
        assert fortran_idx == expected_fortran, \
            f"Fortran index should be {expected_fortran} for Python index {python_idx}"
    
    @pytest.mark.parametrize("begg,endg,fortran_idx,expected_python", [
        (0, 10, 0, 0),
        (0, 10, 5, 5),
        (0, 10, 10, 10),
        (1, 10, 1, 0),
        (1, 10, 6, 5),
        (100, 200, 100, 0),
        (100, 200, 150, 50),
    ])
    def test_get_python_index(self, begg: int, endg: int,
                             fortran_idx: int, expected_python: int):
        """Test conversion from Fortran (offset by begg) to Python (0-based) index."""
        gc = gridcell_type()
        gc.Init(begg, endg)
        
        python_idx = gc.get_python_index(fortran_idx)
        
        assert python_idx == expected_python, \
            f"Python index should be {expected_python} for Fortran index {fortran_idx}"
    
    def test_index_conversion_roundtrip(self):
        """Test that index conversion is reversible."""
        gc = gridcell_type()
        gc.Init(50, 150)
        
        for python_idx in range(0, 101, 10):
            fortran_idx = gc.get_fortran_index(python_idx)
            recovered_python = gc.get_python_index(fortran_idx)
            
            assert recovered_python == python_idx, \
                f"Roundtrip conversion failed for Python index {python_idx}"


# ============================================================================
# Grid Information and Validation Tests
# ============================================================================

class TestGridInformation:
    """Test grid information retrieval and validation methods."""
    
    def test_get_coordinate_bounds(self, test_data: Dict):
        """Test retrieval of coordinate bounds and statistics."""
        case = test_data["test_nominal_small_regional_grid"]
        
        gc = gridcell_type()
        gc.Init(case["begg"], case["endg"])
        gc.set_coordinates(jnp.array(case["latitudes"]), 
                          jnp.array(case["longitudes"]))
        
        bounds = gc.get_coordinate_bounds()
        
        # Verify all expected keys are present
        expected_keys = ["lat_min", "lat_max", "lat_mean", 
                        "lon_min", "lon_max", "lon_mean"]
        for key in expected_keys:
            assert key in bounds, f"Bounds dictionary should contain '{key}'"
        
        # Verify bounds are correct
        lat_array = jnp.array(case["latitudes"])
        lon_array = jnp.array(case["longitudes"])
        
        assert np.isclose(bounds["lat_min"], float(jnp.min(lat_array)), rtol=1e-6), \
            "lat_min should match minimum latitude"
        assert np.isclose(bounds["lat_max"], float(jnp.max(lat_array)), rtol=1e-6), \
            "lat_max should match maximum latitude"
        assert np.isclose(bounds["lat_mean"], float(jnp.mean(lat_array)), rtol=1e-6), \
            "lat_mean should match mean latitude"
        
        assert np.isclose(bounds["lon_min"], float(jnp.min(lon_array)), rtol=1e-6), \
            "lon_min should match minimum longitude"
        assert np.isclose(bounds["lon_max"], float(jnp.max(lon_array)), rtol=1e-6), \
            "lon_max should match maximum longitude"
        assert np.isclose(bounds["lon_mean"], float(jnp.mean(lon_array)), rtol=1e-6), \
            "lon_mean should match mean longitude"
    
    def test_validate_arrays_valid_coordinates(self, test_data: Dict):
        """Test array validation with valid coordinates."""
        case = test_data["test_nominal_small_regional_grid"]
        
        gc = gridcell_type()
        gc.Init(case["begg"], case["endg"])
        gc.set_coordinates(jnp.array(case["latitudes"]), 
                          jnp.array(case["longitudes"]))
        
        assert gc.validate_arrays(), "Valid coordinates should pass validation"
    
    def test_validate_arrays_invalid_latitude(self):
        """Test array validation with out-of-range latitude."""
        gc = gridcell_type()
        gc.Init(0, 2)
        
        # Set invalid latitude (> 90)
        invalid_lat = jnp.array([45.0, 95.0, 50.0])
        valid_lon = jnp.array([-120.0, -110.0, -100.0])
        
        gc.set_coordinates(invalid_lat, valid_lon)
        
        assert not gc.validate_arrays(), \
            "Validation should fail for latitude > 90"
    
    def test_validate_arrays_invalid_longitude(self):
        """Test array validation with out-of-range longitude."""
        gc = gridcell_type()
        gc.Init(0, 2)
        
        # Set invalid longitude (> 360)
        valid_lat = jnp.array([45.0, 50.0, 55.0])
        invalid_lon = jnp.array([-120.0, 370.0, -100.0])
        
        gc.set_coordinates(valid_lat, invalid_lon)
        
        assert not gc.validate_arrays(), \
            "Validation should fail for longitude > 360"
    
    def test_get_grid_info_comprehensive(self, test_data: Dict):
        """Test comprehensive grid information retrieval."""
        case = test_data["test_nominal_small_regional_grid"]
        
        gc = gridcell_type()
        gc.Init(case["begg"], case["endg"])
        gc.set_coordinates(jnp.array(case["latitudes"]), 
                          jnp.array(case["longitudes"]))
        
        info = gc.get_grid_info()
        
        # Verify all expected keys
        expected_keys = [
            "total_gridcells", "begg_index", "endg_index", "initialized",
            "lat_min", "lat_max", "lat_mean",
            "lon_min", "lon_max", "lon_mean",
            "valid_coordinates", "missing_coordinates"
        ]
        
        for key in expected_keys:
            assert key in info, f"Grid info should contain '{key}'"
        
        # Verify specific values
        assert info["total_gridcells"] == case["endg"] - case["begg"] + 1, \
            "total_gridcells should match expected count"
        assert info["begg_index"] == case["begg"], "begg_index should match"
        assert info["endg_index"] == case["endg"], "endg_index should match"
        assert info["initialized"], "initialized should be True"
        assert info["valid_coordinates"], "valid_coordinates should be True"
        assert not info["missing_coordinates"], "missing_coordinates should be False"


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestFactoryFunctions:
    """Test factory functions for creating gridcell instances."""
    
    def test_create_gridcell_instance(self):
        """Test create_gridcell_instance factory function."""
        gc = create_gridcell_instance(10, 20)
        
        assert gc.is_initialized(), "Created instance should be initialized"
        assert gc.begg == 10, "begg should be 10"
        assert gc.endg == 20, "endg should be 20"
        assert gc.get_gridcell_count() == 11, "Should have 11 gridcells"
    
    @pytest.mark.parametrize("begg,endg,lat_range,lon_range", [
        (0, 9, (-45.0, 45.0), (-90.0, 90.0)),
        (1, 15, (30.0, 60.0), (-120.0, -80.0)),
        (100, 199, (-90.0, 90.0), (0.0, 360.0)),
    ])
    def test_create_regular_grid(self, begg: int, endg: int,
                                lat_range: Tuple[float, float],
                                lon_range: Tuple[float, float]):
        """Test create_regular_grid factory function."""
        gc = create_regular_grid(begg, endg, lat_range, lon_range)
        
        assert gc.is_initialized(), "Created grid should be initialized"
        assert gc.begg == begg, f"begg should be {begg}"
        assert gc.endg == endg, f"endg should be {endg}"
        
        # Verify coordinates are within specified ranges
        lat, lon = gc.get_coordinates()
        
        assert float(jnp.min(lat)) >= lat_range[0], \
            f"Minimum latitude should be >= {lat_range[0]}"
        assert float(jnp.max(lat)) <= lat_range[1], \
            f"Maximum latitude should be <= {lat_range[1]}"
        assert float(jnp.min(lon)) >= lon_range[0], \
            f"Minimum longitude should be >= {lon_range[0]}"
        assert float(jnp.max(lon)) <= lon_range[1], \
            f"Maximum longitude should be <= {lon_range[1]}"
    
    def test_global_gridcell_instance(self, clean_global_gridcell):
        """Test global gridcell instance (grc)."""
        # Global instance should start uninitialized
        assert not grc.is_initialized(), "Global instance should start uninitialized"
        
        # Initialize it
        grc.Init(0, 10)
        
        assert grc.is_initialized(), "Global instance should be initialized after Init"
        assert grc.get_gridcell_count() == 11, "Global instance should have 11 gridcells"


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions for coordinate operations."""
    
    @pytest.mark.parametrize("degrees,expected_radians", [
        (0.0, 0.0),
        (90.0, np.pi / 2),
        (180.0, np.pi),
        (360.0, 2 * np.pi),
        (-90.0, -np.pi / 2),
    ])
    def test_degrees_to_radians(self, degrees: float, expected_radians: float):
        """Test degree to radian conversion."""
        deg_array = jnp.array([degrees])
        rad_array = degrees_to_radians(deg_array)
        
        np.testing.assert_allclose(rad_array, jnp.array([expected_radians]),
                                   rtol=1e-6, atol=1e-6,
                                   err_msg=f"Conversion of {degrees}° to radians failed")
    
    @pytest.mark.parametrize("radians,expected_degrees", [
        (0.0, 0.0),
        (np.pi / 2, 90.0),
        (np.pi, 180.0),
        (2 * np.pi, 360.0),
        (-np.pi / 2, -90.0),
    ])
    def test_radians_to_degrees(self, radians: float, expected_degrees: float):
        """Test radian to degree conversion."""
        rad_array = jnp.array([radians])
        deg_array = radians_to_degrees(rad_array)
        
        np.testing.assert_allclose(deg_array, jnp.array([expected_degrees]),
                                   rtol=1e-6, atol=1e-6,
                                   err_msg=f"Conversion of {radians} rad to degrees failed")
    
    def test_angle_conversion_roundtrip(self):
        """Test that angle conversion is reversible."""
        original_degrees = jnp.array([0.0, 45.0, 90.0, 135.0, 180.0, 270.0, 360.0])
        
        radians = degrees_to_radians(original_degrees)
        recovered_degrees = radians_to_degrees(radians)
        
        np.testing.assert_allclose(recovered_degrees, original_degrees,
                                   rtol=1e-6, atol=1e-6,
                                   err_msg="Roundtrip angle conversion failed")
    
    @pytest.mark.parametrize("longitude,range_type,expected", [
        (0.0, "180", 0.0),
        (180.0, "180", 180.0),
        (270.0, "180", -90.0),
        (360.0, "180", 0.0),
        (-90.0, "180", -90.0),
        (-180.0, "360", 180.0),
        (0.0, "360", 0.0),
        (180.0, "360", 180.0),
        (270.0, "360", 270.0),
        (360.0, "360", 0.0),
    ])
    def test_normalize_longitude(self, longitude: float, range_type: str, expected: float):
        """Test longitude normalization to different ranges."""
        lon_array = jnp.array([longitude])
        normalized = normalize_longitude(lon_array, range_type)
        
        np.testing.assert_allclose(normalized, jnp.array([expected]),
                                   rtol=1e-6, atol=1e-6,
                                   err_msg=f"Normalization of {longitude}° to {range_type} range failed")
    
    @pytest.mark.parametrize("latitudes,longitudes,expected_valid", [
        ([45.0, 50.0], [-120.0, -110.0], True),
        ([0.0, 90.0, -90.0], [0.0, 180.0, -180.0], True),
        ([95.0, 50.0], [-120.0, -110.0], False),  # Invalid latitude
        ([45.0, 50.0], [-120.0, 370.0], False),  # Invalid longitude
        ([45.0, -95.0], [-120.0, -110.0], False),  # Invalid latitude
    ])
    def test_validate_coordinates_function(self, latitudes: list, longitudes: list,
                                          expected_valid: bool):
        """Test standalone coordinate validation function."""
        lat_array = jnp.array(latitudes)
        lon_array = jnp.array(longitudes)
        
        is_valid = validate_coordinates(lat_array, lon_array)
        
        assert is_valid == expected_valid, \
            f"Validation should return {expected_valid} for lat={latitudes}, lon={longitudes}"
    
    def test_calculate_distance_haversine_known_distances(self):
        """Test haversine distance calculation with known distances."""
        # Test case: New York to London (approximately 5570 km)
        lat1 = jnp.array([40.7128])
        lon1 = jnp.array([-74.0060])
        lat2 = jnp.array([51.5074])
        lon2 = jnp.array([-0.1278])
        
        distance = calculate_distance_haversine(lat1, lon1, lat2, lon2)
        
        # Allow 1% tolerance for spherical Earth approximation
        expected_distance = 5570.0  # km
        np.testing.assert_allclose(distance, jnp.array([expected_distance]),
                                   rtol=0.01, atol=50.0,
                                   err_msg="NYC to London distance calculation failed")
    
    def test_calculate_distance_haversine_same_point(self):
        """Test haversine distance for same point (should be zero)."""
        lat = jnp.array([45.0, 50.0, 55.0])
        lon = jnp.array([-120.0, -110.0, -100.0])
        
        distance = calculate_distance_haversine(lat, lon, lat, lon)
        
        np.testing.assert_allclose(distance, jnp.zeros_like(distance),
                                   rtol=1e-6, atol=1e-6,
                                   err_msg="Distance between same points should be zero")
    
    def test_calculate_distance_haversine_antipodal_points(self):
        """Test haversine distance for antipodal points (opposite sides of Earth)."""
        # Points at opposite sides of Earth (approximately 20,000 km)
        lat1 = jnp.array([0.0])
        lon1 = jnp.array([0.0])
        lat2 = jnp.array([0.0])
        lon2 = jnp.array([180.0])
        
        distance = calculate_distance_haversine(lat1, lon1, lat2, lon2)
        
        # Earth's circumference / 2 ≈ 20,037 km
        expected_distance = 20037.0
        np.testing.assert_allclose(distance, jnp.array([expected_distance]),
                                   rtol=0.01, atol=100.0,
                                   err_msg="Antipodal distance calculation failed")
    
    def test_create_coordinate_mesh(self):
        """Test coordinate mesh creation."""
        lat_values = jnp.array([40.0, 45.0, 50.0])
        lon_values = jnp.array([-120.0, -110.0, -100.0, -90.0])
        
        lat_mesh, lon_mesh = create_coordinate_mesh(lat_values, lon_values)
        
        # Verify shapes
        expected_size = len(lat_values) * len(lon_values)
        assert lat_mesh.shape == (expected_size,), \
            f"lat_mesh should have shape ({expected_size},)"
        assert lon_mesh.shape == (expected_size,), \
            f"lon_mesh should have shape ({expected_size},)"
        
        # Verify all combinations are present
        for lat in lat_values:
            for lon in lon_values:
                assert jnp.any((lat_mesh == lat) & (lon_mesh == lon)), \
                    f"Mesh should contain combination ({lat}, {lon})"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_polar_coordinates(self, test_data: Dict):
        """Test handling of polar coordinates (±90° latitude)."""
        case = test_data["test_edge_polar_regions"]
        
        gc = gridcell_type()
        gc.Init(case["begg"], case["endg"])
        gc.set_coordinates(jnp.array(case["latitudes"]), 
                          jnp.array(case["longitudes"]))
        
        # Should handle polar coordinates without error
        assert gc.validate_arrays(), "Polar coordinates should be valid"
        
        lat, lon = gc.get_coordinates()
        assert float(jnp.max(jnp.abs(lat))) <= 90.0, \
            "All latitudes should be within ±90°"
    
    def test_dateline_crossing(self, test_data: Dict):
        """Test handling of dateline crossing (±180° longitude)."""
        case = test_data["test_edge_longitude_boundaries"]
        
        gc = gridcell_type()
        gc.Init(case["begg"], case["endg"])
        gc.set_coordinates(jnp.array(case["latitudes"]), 
                          jnp.array(case["longitudes"]))
        
        # Should handle dateline coordinates without error
        lat, lon = gc.get_coordinates()
        
        # Verify all longitudes are within valid range
        assert float(jnp.min(lon)) >= -180.0, "Minimum longitude should be >= -180°"
        assert float(jnp.max(lon)) <= 360.0, "Maximum longitude should be <= 360°"
    
    def test_zero_coordinates(self, test_data: Dict):
        """Test handling of zero coordinates (equator/prime meridian)."""
        case = test_data["test_edge_equator_prime_meridian"]
        
        gc = gridcell_type()
        gc.Init(case["begg"], case["endg"])
        gc.set_coordinates(jnp.array(case["latitudes"]), 
                          jnp.array(case["longitudes"]))
        
        lat, lon = gc.get_coordinates()
        
        # All coordinates should be zero
        np.testing.assert_allclose(lat, jnp.zeros_like(lat), rtol=1e-6, atol=1e-6,
                                   err_msg="All latitudes should be zero")
        np.testing.assert_allclose(lon, jnp.zeros_like(lon), rtol=1e-6, atol=1e-6,
                                   err_msg="All longitudes should be zero")
    
    def test_minimum_grid_size(self, test_data: Dict):
        """Test minimum grid size (begg == endg)."""
        case = test_data["test_edge_single_index_begg_equals_endg"]
        
        gc = gridcell_type()
        gc.Init(case["begg"], case["endg"])
        
        assert gc.get_gridcell_count() == 1, "Grid should have exactly 1 gridcell"
        
        gc.set_coordinates(jnp.array(case["latitudes"]), 
                          jnp.array(case["longitudes"]))
        
        lat, lon = gc.get_coordinates(0)
        
        assert np.isclose(lat, case["latitudes"][0], rtol=1e-6), \
            "Single gridcell latitude should match"
        assert np.isclose(lon, case["longitudes"][0], rtol=1e-6), \
            "Single gridcell longitude should match"
    
    def test_large_grid_scalability(self):
        """Test handling of large grids (performance and correctness)."""
        begg = 0
        endg = 999  # 1000 gridcells
        
        gc = gridcell_type()
        gc.Init(begg, endg)
        
        # Create regular grid
        lat_values = jnp.linspace(-90.0, 90.0, 1000)
        lon_values = jnp.linspace(-180.0, 180.0, 1000)
        
        gc.set_coordinates(lat_values, lon_values)
        
        assert gc.get_gridcell_count() == 1000, "Should have 1000 gridcells"
        assert gc.validate_arrays(), "Large grid should pass validation"
        
        # Test random access
        for idx in [0, 250, 500, 750, 999]:
            lat, lon = gc.get_coordinates(idx)
            assert -90.0 <= lat <= 90.0, f"Latitude at index {idx} should be valid"
            assert -180.0 <= lon <= 180.0, f"Longitude at index {idx} should be valid"


# ============================================================================
# Data Type Tests
# ============================================================================

class TestDataTypes:
    """Test data type handling and conversions."""
    
    def test_coordinate_dtypes(self):
        """Test that coordinates maintain proper JAX array dtypes."""
        gc = gridcell_type()
        gc.Init(0, 5)
        
        lat = jnp.array([40.0, 41.0, 42.0, 43.0, 44.0, 45.0], dtype=jnp.float32)
        lon = jnp.array([-120.0, -119.0, -118.0, -117.0, -116.0, -115.0], dtype=jnp.float32)
        
        gc.set_coordinates(lat, lon)
        
        retrieved_lat, retrieved_lon = gc.get_coordinates()
        
        # Should maintain JAX array type
        assert isinstance(retrieved_lat, jnp.ndarray), "Retrieved latitude should be JAX array"
        assert isinstance(retrieved_lon, jnp.ndarray), "Retrieved longitude should be JAX array"
    
    def test_index_dtypes(self):
        """Test that indices are proper Python integers."""
        gc = gridcell_type()
        gc.Init(10, 20)
        
        assert isinstance(gc.begg, int), "begg should be Python int"
        assert isinstance(gc.endg, int), "endg should be Python int"
        assert isinstance(gc.get_gridcell_count(), int), \
            "get_gridcell_count should return Python int"
    
    def test_mixed_numeric_types(self):
        """Test handling of mixed numeric types (int, float, JAX arrays)."""
        gc = gridcell_type()
        gc.Init(0, 2)
        
        # Mix of Python float and JAX arrays
        gc.set_single_coordinate(0, 45.0, -120.0)  # Python floats
        gc.set_single_coordinate(1, float(jnp.array(46.0)), 
                                float(jnp.array(-119.0)))  # JAX scalars
        
        lat, lon = gc.get_coordinates()
        
        # Should work without errors
        assert lat.shape == (3,), "Should have 3 latitudes"
        assert lon.shape == (3,), "Should have 3 longitudes"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple operations."""
    
    def test_complete_workflow(self, test_data: Dict):
        """Test complete workflow from creation to information retrieval."""
        case = test_data["test_nominal_small_regional_grid"]
        
        # Create instance
        gc = create_gridcell_instance(case["begg"], case["endg"])
        
        # Set coordinates
        gc.set_coordinates(jnp.array(case["latitudes"]), 
                          jnp.array(case["longitudes"]))
        
        # Validate
        assert gc.validate_arrays(), "Grid should be valid"
        
        # Get information
        info = gc.get_grid_info()
        assert info["initialized"], "Should be initialized"
        assert info["valid_coordinates"], "Coordinates should be valid"
        
        # Get bounds
        bounds = gc.get_coordinate_bounds()
        assert bounds["lat_min"] <= bounds["lat_max"], \
            "Latitude bounds should be ordered"
        assert bounds["lon_min"] <= bounds["lon_max"], \
            "Longitude bounds should be ordered"
        
        # Test index conversion
        for py_idx in range(gc.get_gridcell_count()):
            fort_idx = gc.get_fortran_index(py_idx)
            recovered_py = gc.get_python_index(fort_idx)
            assert recovered_py == py_idx, "Index conversion should be reversible"
    
    def test_multiple_instances_independence(self):
        """Test that multiple gridcell instances are independent."""
        gc1 = create_gridcell_instance(0, 5)
        gc2 = create_gridcell_instance(10, 15)
        
        lat1 = jnp.array([40.0, 41.0, 42.0, 43.0, 44.0, 45.0])
        lon1 = jnp.array([-120.0, -119.0, -118.0, -117.0, -116.0, -115.0])
        
        lat2 = jnp.array([50.0, 51.0, 52.0, 53.0, 54.0, 55.0])
        lon2 = jnp.array([-100.0, -99.0, -98.0, -97.0, -96.0, -95.0])
        
        gc1.set_coordinates(lat1, lon1)
        gc2.set_coordinates(lat2, lon2)
        
        # Verify independence
        retrieved_lat1, retrieved_lon1 = gc1.get_coordinates()
        retrieved_lat2, retrieved_lon2 = gc2.get_coordinates()
        
        np.testing.assert_allclose(retrieved_lat1, lat1, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(retrieved_lat2, lat2, rtol=1e-6, atol=1e-6)
        
        assert not jnp.allclose(retrieved_lat1, retrieved_lat2), \
            "Different instances should have different coordinates"
    
    def test_regular_grid_properties(self):
        """Test properties of regular grids created by factory function."""
        gc = create_regular_grid(0, 99, (-45.0, 45.0), (-90.0, 90.0))
        
        lat, lon = gc.get_coordinates()
        
        # Check that coordinates span the specified ranges
        assert float(jnp.min(lat)) >= -45.0, "Latitudes should be >= -45°"
        assert float(jnp.max(lat)) <= 45.0, "Latitudes should be <= 45°"
        assert float(jnp.min(lon)) >= -90.0, "Longitudes should be >= -90°"
        assert float(jnp.max(lon)) <= 90.0, "Longitudes should be <= 90°"
        
        # Check that grid is reasonably regular (spacing should be consistent)
        lat_diffs = jnp.diff(jnp.sort(lat))
        lon_diffs = jnp.diff(jnp.sort(lon))
        
        # Standard deviation of differences should be small for regular grid
        assert float(jnp.std(lat_diffs)) < 1.0, \
            "Latitude spacing should be relatively uniform"
        assert float(jnp.std(lon_diffs)) < 1.0, \
            "Longitude spacing should be relatively uniform"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])