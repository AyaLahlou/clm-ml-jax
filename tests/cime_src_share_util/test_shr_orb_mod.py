"""
Comprehensive pytest suite for shr_orb_mod orbital calculation functions.

This module tests the JAX implementations of:
- shr_orb_cosz: Calculate cosine of solar zenith angle
- shr_orb_decl: Calculate solar declination and Earth-sun distance factor
- shr_orb_params: Calculate Earth's orbital parameters for a given year

Tests cover:
- Nominal cases (typical seasonal and geographic conditions)
- Edge cases (poles, boundaries, extreme values)
- Array broadcasting and multi-dimensional inputs
- Physical constraints and astronomical accuracy
- Numerical stability
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

import pytest
import jax.numpy as jnp
import numpy as np
from jax import config

# Enable 64-bit precision for numerical accuracy
config.update("jax_enable_x64", True)

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from cime_src_share_util.shr_orb_mod import (
    shr_orb_cosz,
    shr_orb_decl,
    shr_orb_params,
    OrbitalParams
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load comprehensive test data for orbital functions.
    
    Returns:
        Dictionary containing test cases for all functions with inputs,
        expected behaviors, and metadata.
    """
    return {
        "shr_orb_cosz": {
            "equinox_equator": {
                "jday": 80.0,
                "lat": 0.0,
                "lon": 0.0,
                "declin": 0.0,
                "expected_range": (0.9, 1.0),
                "description": "Spring equinox at equator, noon - sun near zenith"
            },
            "summer_solstice_north": {
                "jday": 172.0,
                "lat": 0.4101,
                "lon": 0.0,
                "declin": 0.4091,
                "expected_range": (0.95, 1.0),
                "description": "Summer solstice at Tropic of Cancer"
            },
            "winter_pole": {
                "jday": 355.0,
                "lat": 1.5708,
                "lon": 0.0,
                "declin": -0.4091,
                "expected_range": (-1.0, 0.0),
                "description": "Winter at North Pole - polar night"
            },
            "midnight_negative": {
                "jday": 200.0,
                "lat": 0.0,
                "lon": 3.1416,
                "declin": 0.2,
                "expected_range": (-1.0, 0.0),
                "description": "Midnight at equator - sun below horizon"
            },
            "south_pole_summer": {
                "jday": 355.0,
                "lat": -1.5708,
                "lon": 0.0,
                "declin": -0.4091,
                "expected_range": (0.0, 1.0),
                "description": "South Pole during southern summer"
            }
        },
        "shr_orb_decl": {
            "vernal_equinox": {
                "calday": 80.5,
                "eccen": 0.0167,
                "mvelpp": 4.9368,
                "lambm0": 4.8951,
                "obliqr": 0.4091,
                "expected_delta_range": (-0.1, 0.1),
                "expected_eccf_range": (0.95, 1.05),
                "description": "Vernal equinox with current orbital parameters"
            },
            "summer_solstice": {
                "calday": 172.0,
                "eccen": 0.0167,
                "mvelpp": 4.9368,
                "lambm0": 4.8951,
                "obliqr": 0.4091,
                "expected_delta_range": (0.35, 0.45),
                "expected_eccf_range": (0.95, 1.0),
                "description": "Summer solstice - maximum declination"
            },
            "zero_eccentricity": {
                "calday": 1.0,
                "eccen": 0.0,
                "mvelpp": 3.1416,
                "lambm0": 0.0,
                "obliqr": 0.4091,
                "expected_delta_range": (-0.45, -0.35),
                "expected_eccf_range": (0.99, 1.01),
                "description": "Circular orbit - eccf should be near 1.0"
            },
            "perihelion": {
                "calday": 3.0,
                "eccen": 0.0167,
                "mvelpp": 4.9368,
                "lambm0": 4.8951,
                "obliqr": 0.4091,
                "expected_delta_range": (-0.45, -0.35),
                "expected_eccf_range": (1.03, 1.08),
                "description": "Near perihelion - maximum eccf"
            },
            "aphelion": {
                "calday": 186.0,
                "eccen": 0.0167,
                "mvelpp": 4.9368,
                "lambm0": 4.8951,
                "obliqr": 0.4091,
                "expected_delta_range": (0.35, 0.45),
                "expected_eccf_range": (0.93, 0.98),
                "description": "Near aphelion - minimum eccf"
            }
        },
        "shr_orb_params": {
            "current_epoch": {
                "iyear_AD": 2000,
                "expected_eccen_range": (0.015, 0.018),
                "expected_obliq_range": (23.4, 23.5),
                "description": "Year 2000 - current epoch"
            },
            "reference_year": {
                "iyear_AD": 1950,
                "expected_eccen_range": (0.015, 0.018),
                "expected_obliq_range": (23.4, 23.5),
                "description": "Reference year 1950"
            },
            "ice_age": {
                "iyear_AD": -20000,
                "expected_eccen_range": (0.0, 0.06),
                "expected_obliq_range": (22.0, 24.5),
                "description": "Last glacial maximum"
            }
        }
    }


@pytest.fixture
def orbital_constants() -> Dict[str, float]:
    """
    Physical constants and constraints for orbital calculations.
    
    Returns:
        Dictionary of physical constants and valid ranges.
    """
    return {
        "PI": np.pi,
        "MAX_LAT": np.pi / 2,
        "MIN_LAT": -np.pi / 2,
        "MAX_LON": np.pi,
        "MIN_LON": -np.pi,
        "MAX_DECLIN": 0.4091,  # ~23.44 degrees
        "MIN_DECLIN": -0.4091,
        "MAX_ECCEN": 0.06,
        "MIN_ECCEN": 0.0,
        "MAX_OBLIQ_DEG": 24.5,
        "MIN_OBLIQ_DEG": 22.0,
        "MAX_OBLIQ_RAD": 0.4276,
        "MIN_OBLIQ_RAD": 0.384,
        "MIN_CALDAY": 1.0,
        "MAX_CALDAY": 365.99,
        "MIN_YEAR": -998050,
        "MAX_YEAR": 1001950
    }


# ============================================================================
# Tests for shr_orb_cosz
# ============================================================================

class TestShrOrbCosz:
    """Test suite for shr_orb_cosz function."""
    
    def test_cosz_scalar_inputs(self, test_data):
        """
        Test shr_orb_cosz with scalar inputs.
        
        Verifies that the function correctly handles scalar inputs and returns
        a scalar output within the valid cosine range [-1, 1].
        """
        case = test_data["shr_orb_cosz"]["equinox_equator"]
        
        result = shr_orb_cosz(
            jday=case["jday"],
            lat=case["lat"],
            lon=case["lon"],
            declin=case["declin"]
        )
        
        # Check output is scalar
        assert jnp.ndim(result) == 0, "Output should be scalar for scalar inputs"
        
        # Check value is in valid cosine range
        assert -1.0 <= float(result) <= 1.0, "Cosine must be in [-1, 1]"
        
        # Check expected range for this case
        assert case["expected_range"][0] <= float(result) <= case["expected_range"][1], \
            f"Expected cosz in {case['expected_range']}, got {float(result)}"
    
    @pytest.mark.parametrize("case_name", [
        "equinox_equator",
        "summer_solstice_north",
        "winter_pole",
        "midnight_negative",
        "south_pole_summer"
    ])
    def test_cosz_physical_cases(self, test_data, case_name):
        """
        Test shr_orb_cosz for various physical scenarios.
        
        Parametrized test covering equinoxes, solstices, polar conditions,
        and day/night transitions.
        """
        case = test_data["shr_orb_cosz"][case_name]
        
        result = shr_orb_cosz(
            jday=case["jday"],
            lat=case["lat"],
            lon=case["lon"],
            declin=case["declin"]
        )
        
        # Check valid cosine range
        assert -1.0 <= float(result) <= 1.0, \
            f"{case['description']}: cosine must be in [-1, 1]"
        
        # Check expected range
        assert case["expected_range"][0] <= float(result) <= case["expected_range"][1], \
            f"{case['description']}: expected {case['expected_range']}, got {float(result)}"
    
    def test_cosz_array_broadcasting(self):
        """
        Test shr_orb_cosz with 1D array inputs.
        
        Verifies that the function correctly broadcasts over array inputs
        and returns an array of the same shape.
        """
        jday = jnp.array([1.0, 91.0, 182.0, 273.0, 365.0])
        lat = jnp.array([0.0, 0.7854, 1.0472, -0.7854, -1.5708])
        lon = jnp.array([0.0, 1.5708, 3.1416, -1.5708, 0.0])
        declin = jnp.array([-0.4091, 0.0, 0.4091, 0.0, -0.4091])
        
        result = shr_orb_cosz(jday=jday, lat=lat, lon=lon, declin=declin)
        
        # Check output shape
        assert result.shape == jday.shape, \
            f"Output shape {result.shape} should match input shape {jday.shape}"
        
        # Check all values in valid range
        assert jnp.all((result >= -1.0) & (result <= 1.0)), \
            "All cosine values must be in [-1, 1]"
    
    def test_cosz_multidim_array(self):
        """
        Test shr_orb_cosz with 2D array inputs.
        
        Verifies that the function handles multi-dimensional arrays correctly
        and maintains shape consistency.
        """
        jday = jnp.array([[1.0, 91.0], [182.0, 273.0]])
        lat = jnp.array([[0.0, 0.7854], [-0.7854, 0.0]])
        lon = jnp.array([[0.0, 1.5708], [-1.5708, 3.1416]])
        declin = jnp.array([[-0.4091, 0.0], [0.4091, 0.0]])
        
        result = shr_orb_cosz(jday=jday, lat=lat, lon=lon, declin=declin)
        
        # Check output shape
        assert result.shape == (2, 2), \
            f"Output shape {result.shape} should be (2, 2)"
        
        # Check all values in valid range
        assert jnp.all((result >= -1.0) & (result <= 1.0)), \
            "All cosine values must be in [-1, 1]"
    
    def test_cosz_boundary_latitudes(self, orbital_constants):
        """
        Test shr_orb_cosz at boundary latitudes (poles).
        
        Verifies correct behavior at the North and South poles.
        """
        # North Pole
        result_north = shr_orb_cosz(
            jday=172.0,
            lat=orbital_constants["MAX_LAT"],
            lon=0.0,
            declin=0.4091
        )
        assert -1.0 <= float(result_north) <= 1.0, "North pole result must be valid"
        
        # South Pole
        result_south = shr_orb_cosz(
            jday=172.0,
            lat=orbital_constants["MIN_LAT"],
            lon=0.0,
            declin=0.4091
        )
        assert -1.0 <= float(result_south) <= 1.0, "South pole result must be valid"
    
    def test_cosz_boundary_longitudes(self, orbital_constants):
        """
        Test shr_orb_cosz at boundary longitudes.
        
        Verifies correct behavior at longitude boundaries (±π).
        """
        # Maximum longitude
        result_max = shr_orb_cosz(
            jday=180.0,
            lat=0.0,
            lon=orbital_constants["MAX_LON"],
            declin=0.0
        )
        assert -1.0 <= float(result_max) <= 1.0, "Max longitude result must be valid"
        
        # Minimum longitude
        result_min = shr_orb_cosz(
            jday=180.0,
            lat=0.0,
            lon=orbital_constants["MIN_LON"],
            declin=0.0
        )
        assert -1.0 <= float(result_min) <= 1.0, "Min longitude result must be valid"
    
    def test_cosz_dtype_preservation(self):
        """
        Test that shr_orb_cosz preserves float64 dtype.
        
        Verifies that the function maintains numerical precision.
        """
        result = shr_orb_cosz(
            jday=jnp.float64(180.0),
            lat=jnp.float64(0.0),
            lon=jnp.float64(0.0),
            declin=jnp.float64(0.0)
        )
        
        assert result.dtype == jnp.float64, \
            f"Expected float64, got {result.dtype}"
    
    def test_cosz_symmetry(self):
        """
        Test symmetry properties of shr_orb_cosz.
        
        Verifies that cosz is symmetric about the equator for equinox conditions.
        """
        lat_north = 0.5
        lat_south = -0.5
        
        result_north = shr_orb_cosz(jday=80.0, lat=lat_north, lon=0.0, declin=0.0)
        result_south = shr_orb_cosz(jday=80.0, lat=lat_south, lon=0.0, declin=0.0)
        
        # At equinox with zero declination, should be symmetric
        assert jnp.allclose(result_north, result_south, rtol=1e-6), \
            "Equinox should be symmetric about equator"


# ============================================================================
# Tests for shr_orb_decl
# ============================================================================

class TestShrOrbDecl:
    """Test suite for shr_orb_decl function."""
    
    def test_decl_scalar_inputs(self, test_data):
        """
        Test shr_orb_decl with scalar inputs.
        
        Verifies that the function returns a tuple of (delta, eccf) with
        correct shapes and value ranges.
        """
        case = test_data["shr_orb_decl"]["vernal_equinox"]
        
        delta, eccf = shr_orb_decl(
            calday=case["calday"],
            eccen=case["eccen"],
            mvelpp=case["mvelpp"],
            lambm0=case["lambm0"],
            obliqr=case["obliqr"]
        )
        
        # Check outputs are scalars
        assert jnp.ndim(delta) == 0, "delta should be scalar for scalar inputs"
        assert jnp.ndim(eccf) == 0, "eccf should be scalar for scalar inputs"
        
        # Check delta in valid range
        assert -0.4091 <= float(delta) <= 0.4091, \
            f"delta must be in [-0.4091, 0.4091], got {float(delta)}"
        
        # Check eccf in valid range
        assert 0.9 <= float(eccf) <= 1.1, \
            f"eccf must be in [0.9, 1.1], got {float(eccf)}"
    
    @pytest.mark.parametrize("case_name", [
        "vernal_equinox",
        "summer_solstice",
        "zero_eccentricity",
        "perihelion",
        "aphelion"
    ])
    def test_decl_physical_cases(self, test_data, case_name):
        """
        Test shr_orb_decl for various orbital configurations.
        
        Parametrized test covering equinoxes, solstices, perihelion,
        aphelion, and special orbital conditions.
        """
        case = test_data["shr_orb_decl"][case_name]
        
        delta, eccf = shr_orb_decl(
            calday=case["calday"],
            eccen=case["eccen"],
            mvelpp=case["mvelpp"],
            lambm0=case["lambm0"],
            obliqr=case["obliqr"]
        )
        
        # Check delta range
        assert case["expected_delta_range"][0] <= float(delta) <= case["expected_delta_range"][1], \
            f"{case['description']}: delta expected {case['expected_delta_range']}, got {float(delta)}"
        
        # Check eccf range
        assert case["expected_eccf_range"][0] <= float(eccf) <= case["expected_eccf_range"][1], \
            f"{case['description']}: eccf expected {case['expected_eccf_range']}, got {float(eccf)}"
    
    def test_decl_array_calday(self):
        """
        Test shr_orb_decl with array of calendar days.
        
        Verifies that the function correctly handles array inputs for calday
        and returns arrays of the same shape.
        """
        calday = jnp.array([1.0, 80.5, 172.0, 266.0, 355.0, 365.99])
        eccen = 0.0167
        mvelpp = 4.9368
        lambm0 = 4.8951
        obliqr = 0.4091
        
        delta, eccf = shr_orb_decl(
            calday=calday,
            eccen=eccen,
            mvelpp=mvelpp,
            lambm0=lambm0,
            obliqr=obliqr
        )
        
        # Check output shapes
        assert delta.shape == calday.shape, \
            f"delta shape {delta.shape} should match calday shape {calday.shape}"
        assert eccf.shape == calday.shape, \
            f"eccf shape {eccf.shape} should match calday shape {calday.shape}"
        
        # Check all values in valid ranges
        assert jnp.all((delta >= -0.5) & (delta <= 0.5)), \
            "All delta values must be in valid range"
        assert jnp.all((eccf >= 0.9) & (eccf <= 1.1)), \
            "All eccf values must be in valid range"
    
    def test_decl_zero_eccentricity(self):
        """
        Test shr_orb_decl with zero eccentricity (circular orbit).
        
        Verifies that eccf is exactly 1.0 for a circular orbit.
        """
        calday = jnp.array([1.0, 100.0, 200.0, 300.0])
        
        delta, eccf = shr_orb_decl(
            calday=calday,
            eccen=0.0,
            mvelpp=3.1416,
            lambm0=0.0,
            obliqr=0.4091
        )
        
        # For zero eccentricity, eccf should be very close to 1.0
        assert jnp.allclose(eccf, 1.0, atol=1e-6), \
            f"Zero eccentricity should give eccf ≈ 1.0, got {eccf}"
    
    def test_decl_high_eccentricity(self, orbital_constants):
        """
        Test shr_orb_decl with maximum valid eccentricity.
        
        Verifies that the function handles high eccentricity correctly
        and eccf varies more from 1.0.
        """
        delta, eccf = shr_orb_decl(
            calday=1.0,
            eccen=orbital_constants["MAX_ECCEN"],
            mvelpp=0.0,
            lambm0=-3.1416,
            obliqr=0.4276
        )
        
        # Check valid ranges
        assert -0.5 <= float(delta) <= 0.5, "delta must be in valid range"
        assert 0.85 <= float(eccf) <= 1.15, \
            f"High eccentricity eccf should vary more, got {float(eccf)}"
    
    def test_decl_boundary_obliquity(self):
        """
        Test shr_orb_decl at boundary obliquity values.
        
        Verifies correct behavior at minimum and maximum obliquity.
        """
        calday = 172.0  # Summer solstice
        eccen = 0.0167
        mvelpp = 4.9368
        lambm0 = 4.8951
        
        # Minimum obliquity
        delta_min, _ = shr_orb_decl(
            calday=calday,
            eccen=eccen,
            mvelpp=mvelpp,
            lambm0=lambm0,
            obliqr=0.384
        )
        
        # Maximum obliquity
        delta_max, _ = shr_orb_decl(
            calday=calday,
            eccen=eccen,
            mvelpp=mvelpp,
            lambm0=lambm0,
            obliqr=0.4276
        )
        
        # Higher obliquity should give larger declination magnitude
        assert abs(float(delta_max)) > abs(float(delta_min)), \
            "Higher obliquity should increase declination magnitude"
    
    def test_decl_seasonal_variation(self):
        """
        Test that shr_orb_decl shows correct seasonal variation.
        
        Verifies that declination varies sinusoidally through the year.
        """
        calday = jnp.linspace(1.0, 365.0, 365)
        
        delta, eccf = shr_orb_decl(
            calday=calday,
            eccen=0.0167,
            mvelpp=4.9368,
            lambm0=4.8951,
            obliqr=0.4091
        )
        
        # Check that delta reaches both positive and negative extremes
        assert jnp.max(delta) > 0.35, "Should reach positive declination"
        assert jnp.min(delta) < -0.35, "Should reach negative declination"
        
        # Check that eccf varies through the year
        assert jnp.max(eccf) > 1.0, "eccf should exceed 1.0 at perihelion"
        assert jnp.min(eccf) < 1.0, "eccf should be less than 1.0 at aphelion"
    
    def test_decl_dtype_preservation(self):
        """
        Test that shr_orb_decl preserves float64 dtype.
        
        Verifies numerical precision is maintained.
        """
        delta, eccf = shr_orb_decl(
            calday=jnp.float64(180.0),
            eccen=0.0167,
            mvelpp=4.9368,
            lambm0=4.8951,
            obliqr=0.4091
        )
        
        assert delta.dtype == jnp.float64, f"Expected float64, got {delta.dtype}"
        assert eccf.dtype == jnp.float64, f"Expected float64, got {eccf.dtype}"


# ============================================================================
# Tests for shr_orb_params
# ============================================================================

class TestShrOrbParams:
    """Test suite for shr_orb_params function."""
    
    def test_params_current_epoch(self, test_data):
        """
        Test shr_orb_params for year 2000.
        
        Verifies that orbital parameters for the current epoch are
        within expected ranges.
        """
        case = test_data["shr_orb_params"]["current_epoch"]
        
        params = shr_orb_params(iyear_AD=case["iyear_AD"])
        
        # Check return type
        assert isinstance(params, OrbitalParams), \
            "Should return OrbitalParams namedtuple"
        
        # Check eccentricity
        assert case["expected_eccen_range"][0] <= float(params.eccen) <= case["expected_eccen_range"][1], \
            f"eccen expected {case['expected_eccen_range']}, got {float(params.eccen)}"
        
        # Check obliquity
        assert case["expected_obliq_range"][0] <= float(params.obliq) <= case["expected_obliq_range"][1], \
            f"obliq expected {case['expected_obliq_range']}, got {float(params.obliq)}"
    
    @pytest.mark.parametrize("case_name", [
        "current_epoch",
        "reference_year",
        "ice_age"
    ])
    def test_params_various_years(self, test_data, case_name):
        """
        Test shr_orb_params for various historical and future years.
        
        Parametrized test covering current epoch, reference year,
        and ice age conditions.
        """
        case = test_data["shr_orb_params"][case_name]
        
        params = shr_orb_params(iyear_AD=case["iyear_AD"])
        
        # Check all parameters are in valid ranges
        assert 0.0 <= float(params.eccen) <= 0.06, \
            f"{case['description']}: eccen must be in [0, 0.06]"
        assert 22.0 <= float(params.obliq) <= 24.5, \
            f"{case['description']}: obliq must be in [22, 24.5] degrees"
        assert 0.0 <= float(params.mvelp) < 360.0, \
            f"{case['description']}: mvelp must be in [0, 360) degrees"
    
    def test_params_boundary_years(self, orbital_constants):
        """
        Test shr_orb_params at boundary years.
        
        Verifies that the function handles extreme past and future years
        within the valid algorithm range.
        """
        # Near minimum year
        params_past = shr_orb_params(iyear_AD=-998000)
        assert 0.0 <= float(params_past.eccen) <= 0.06, \
            "Past extreme: eccen must be valid"
        assert 22.0 <= float(params_past.obliq) <= 24.5, \
            "Past extreme: obliq must be valid"
        
        # Near maximum year
        params_future = shr_orb_params(iyear_AD=1001900)
        assert 0.0 <= float(params_future.eccen) <= 0.06, \
            "Future extreme: eccen must be valid"
        assert 22.0 <= float(params_future.obliq) <= 24.5, \
            "Future extreme: obliq must be valid"
    
    def test_params_namedtuple_fields(self):
        """
        Test that shr_orb_params returns all expected fields.
        
        Verifies that the OrbitalParams namedtuple contains all
        required fields with correct types.
        """
        params = shr_orb_params(iyear_AD=2000)
        
        # Check all fields exist
        assert hasattr(params, 'eccen'), "Missing eccen field"
        assert hasattr(params, 'obliq'), "Missing obliq field"
        assert hasattr(params, 'mvelp'), "Missing mvelp field"
        assert hasattr(params, 'obliqr'), "Missing obliqr field"
        assert hasattr(params, 'lambm0'), "Missing lambm0 field"
        assert hasattr(params, 'mvelpp'), "Missing mvelpp field"
        
        # Check types
        assert isinstance(params.eccen, jnp.ndarray), "eccen should be ndarray"
        assert isinstance(params.obliq, jnp.ndarray), "obliq should be ndarray"
        assert isinstance(params.mvelp, jnp.ndarray), "mvelp should be ndarray"
        assert isinstance(params.obliqr, jnp.ndarray), "obliqr should be ndarray"
        assert isinstance(params.lambm0, jnp.ndarray), "lambm0 should be ndarray"
        assert isinstance(params.mvelpp, jnp.ndarray), "mvelpp should be ndarray"
    
    def test_params_angle_conversions(self):
        """
        Test that angle conversions are consistent.
        
        Verifies that obliq and obliqr are consistent (degrees vs radians),
        and that mvelp and mvelpp are related correctly.
        """
        params = shr_orb_params(iyear_AD=2000)
        
        # Check obliq to obliqr conversion
        obliqr_from_obliq = float(params.obliq) * np.pi / 180.0
        assert jnp.allclose(params.obliqr, obliqr_from_obliq, rtol=1e-6), \
            "obliqr should be obliq converted to radians"
        
        # Check mvelpp is mvelp + pi (in radians)
        mvelp_rad = float(params.mvelp) * np.pi / 180.0
        expected_mvelpp = mvelp_rad + np.pi
        # Normalize to [0, 2π)
        expected_mvelpp = expected_mvelpp % (2 * np.pi)
        assert jnp.allclose(params.mvelpp, expected_mvelpp, rtol=1e-6), \
            "mvelpp should be mvelp (in radians) + π"
    
    def test_params_temporal_continuity(self):
        """
        Test that orbital parameters change smoothly over time.
        
        Verifies that parameters don't have discontinuous jumps
        between consecutive years.
        """
        years = [1990, 2000, 2010, 2020]
        params_list = [shr_orb_params(iyear_AD=year) for year in years]
        
        # Check that consecutive values don't differ too much
        for i in range(len(params_list) - 1):
            eccen_diff = abs(float(params_list[i+1].eccen) - float(params_list[i].eccen))
            obliq_diff = abs(float(params_list[i+1].obliq) - float(params_list[i].obliq))
            
            assert eccen_diff < 0.001, \
                f"Eccentricity should change smoothly, diff={eccen_diff}"
            assert obliq_diff < 0.01, \
                f"Obliquity should change smoothly, diff={obliq_diff}"
    
    def test_params_dtype_consistency(self):
        """
        Test that all returned parameters have consistent dtypes.
        
        Verifies that all fields use float64 for numerical precision.
        """
        params = shr_orb_params(iyear_AD=2000)
        
        assert params.eccen.dtype == jnp.float64, "eccen should be float64"
        assert params.obliq.dtype == jnp.float64, "obliq should be float64"
        assert params.mvelp.dtype == jnp.float64, "mvelp should be float64"
        assert params.obliqr.dtype == jnp.float64, "obliqr should be float64"
        assert params.lambm0.dtype == jnp.float64, "lambm0 should be float64"
        assert params.mvelpp.dtype == jnp.float64, "mvelpp should be float64"


# ============================================================================
# Integration Tests
# ============================================================================

class TestOrbitalIntegration:
    """Integration tests combining multiple orbital functions."""
    
    def test_params_to_decl_integration(self):
        """
        Test integration of shr_orb_params output into shr_orb_decl.
        
        Verifies that orbital parameters from shr_orb_params can be
        used directly in shr_orb_decl calculations.
        """
        # Get orbital parameters for year 2000
        params = shr_orb_params(iyear_AD=2000)
        
        # Use them in declination calculation
        calday = jnp.array([80.5, 172.0, 266.0, 355.0])
        delta, eccf = shr_orb_decl(
            calday=calday,
            eccen=float(params.eccen),
            mvelpp=float(params.mvelpp),
            lambm0=float(params.lambm0),
            obliqr=float(params.obliqr)
        )
        
        # Check outputs are valid
        assert jnp.all((delta >= -0.5) & (delta <= 0.5)), \
            "Declination must be in valid range"
        assert jnp.all((eccf >= 0.9) & (eccf <= 1.1)), \
            "Distance factor must be in valid range"
    
    def test_decl_to_cosz_integration(self):
        """
        Test integration of shr_orb_decl output into shr_orb_cosz.
        
        Verifies that declination from shr_orb_decl can be used
        in shr_orb_cosz calculations.
        """
        # Calculate declination for summer solstice
        delta, eccf = shr_orb_decl(
            calday=172.0,
            eccen=0.0167,
            mvelpp=4.9368,
            lambm0=4.8951,
            obliqr=0.4091
        )
        
        # Use declination in cosz calculation
        lat = jnp.array([0.0, 0.4091, 0.7854, 1.0472])
        lon = jnp.zeros_like(lat)
        jday = jnp.full_like(lat, 172.0)
        
        cosz = shr_orb_cosz(
            jday=jday,
            lat=lat,
            lon=lon,
            declin=delta
        )
        
        # Check outputs are valid
        assert jnp.all((cosz >= -1.0) & (cosz <= 1.0)), \
            "Cosine must be in valid range"
    
    def test_full_pipeline_annual_cycle(self):
        """
        Test complete pipeline for annual solar cycle.
        
        Verifies that the full calculation chain (params -> decl -> cosz)
        produces physically reasonable results for a complete year.
        """
        # Get orbital parameters
        params = shr_orb_params(iyear_AD=2000)
        
        # Calculate for full year
        calday = jnp.linspace(1.0, 365.0, 365)
        delta, eccf = shr_orb_decl(
            calday=calday,
            eccen=float(params.eccen),
            mvelpp=float(params.mvelpp),
            lambm0=float(params.lambm0),
            obliqr=float(params.obliqr)
        )
        
        # Calculate cosz at equator, noon
        lat = jnp.zeros_like(calday)
        lon = jnp.zeros_like(calday)
        cosz = shr_orb_cosz(
            jday=calday,
            lat=lat,
            lon=lon,
            declin=delta
        )
        
        # Check annual variation
        assert jnp.max(cosz) > 0.9, "Should reach high sun at equinoxes"
        assert jnp.min(cosz) < 0.7, "Should have lower sun at solstices"
        
        # Check that maximum cosz occurs near equinoxes (day 80 and 266)
        equinox_indices = [79, 265]  # 0-indexed
        equinox_cosz = cosz[equinox_indices]
        assert jnp.all(equinox_cosz > 0.95), \
            "Equinox should have high sun at equator"


# ============================================================================
# Edge Case and Robustness Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and numerical robustness."""
    
    def test_cosz_near_zero_declination(self):
        """
        Test shr_orb_cosz with very small declination values.
        
        Verifies numerical stability near zero.
        """
        small_declin = jnp.array([1e-10, -1e-10, 0.0])
        
        result = shr_orb_cosz(
            jday=jnp.full_like(small_declin, 180.0),
            lat=jnp.zeros_like(small_declin),
            lon=jnp.zeros_like(small_declin),
            declin=small_declin
        )
        
        # Should all be close to each other and near 1.0
        assert jnp.allclose(result, 1.0, atol=1e-6), \
            "Small declination at equator noon should give cosz ≈ 1"
    
    def test_decl_boundary_calday(self, orbital_constants):
        """
        Test shr_orb_decl at calendar day boundaries.
        
        Verifies correct behavior at day 1 and day 365.99.
        """
        calday_boundary = jnp.array([
            orbital_constants["MIN_CALDAY"],
            orbital_constants["MAX_CALDAY"]
        ])
        
        delta, eccf = shr_orb_decl(
            calday=calday_boundary,
            eccen=0.0167,
            mvelpp=4.9368,
            lambm0=4.8951,
            obliqr=0.4091
        )
        
        # Both should give valid results
        assert jnp.all((delta >= -0.5) & (delta <= 0.5)), \
            "Boundary calday should give valid delta"
        assert jnp.all((eccf >= 0.9) & (eccf <= 1.1)), \
            "Boundary calday should give valid eccf"
        
        # Day 1 and day 365.99 should be very similar
        assert jnp.allclose(delta[0], delta[1], atol=0.01), \
            "Day 1 and 365.99 should have similar declination"
    
    def test_params_year_zero_handling(self):
        """
        Test shr_orb_params near year 0 (BC/AD transition).
        
        Verifies correct handling of the calendar transition.
        """
        # Test years around 0 AD
        years = [-100, -1, 1, 100]
        
        for year in years:
            params = shr_orb_params(iyear_AD=year)
            
            # All should return valid parameters
            assert 0.0 <= float(params.eccen) <= 0.06, \
                f"Year {year}: eccen must be valid"
            assert 22.0 <= float(params.obliq) <= 24.5, \
                f"Year {year}: obliq must be valid"
    
    def test_numerical_stability_extreme_latitudes(self):
        """
        Test numerical stability at extreme latitudes.
        
        Verifies that calculations remain stable near the poles.
        """
        # Test very close to poles
        lat_near_pole = jnp.array([
            1.5707963267948966,  # Exactly π/2
            1.5707963267948964,  # Slightly less
            -1.5707963267948966,  # Exactly -π/2
            -1.5707963267948964   # Slightly more
        ])
        
        result = shr_orb_cosz(
            jday=jnp.full_like(lat_near_pole, 172.0),
            lat=lat_near_pole,
            lon=jnp.zeros_like(lat_near_pole),
            declin=jnp.full_like(lat_near_pole, 0.4091)
        )
        
        # All should be valid
        assert jnp.all((result >= -1.0) & (result <= 1.0)), \
            "Extreme latitudes should give valid cosz"
        
        # Should not have NaN or Inf
        assert jnp.all(jnp.isfinite(result)), \
            "Results should be finite at extreme latitudes"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])