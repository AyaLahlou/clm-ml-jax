"""
Comprehensive pytest suite for create_empty_rsl_lookup_tables function.

This module tests the creation of empty RSL (Roughness Sublayer) lookup tables
used in multilayer canopy physics simulations. Tests cover:
- Nominal cases with various grid dimensions
- Edge cases (minimum dimensions, boundary values)
- Special cases (large grids, extreme asymmetry)
- Shape verification, dtype checking, and initialization validation
"""

import pytest
import jax.numpy as jnp
import numpy as np
from collections import namedtuple
from typing import Any, Dict


# Define namedtuples matching the function signature
MLCanopyConstants = namedtuple('MLCanopyConstants', [
    'rgas', 'mmdry', 'mmh2o', 'cpd', 'cpw', 'visc0', 'dh0', 'dv0', 'dc0',
    'lapse_rate', 'kc25', 'kcha', 'ko25', 'koha', 'cp25', 'cpha',
    'vcmaxha_noacclim', 'vcmaxha_acclim', 'vcmaxhd_noacclim', 'vcmaxhd_acclim',
    'vcmaxse_noacclim', 'vcmaxse_acclim', 'jmaxha_noacclim', 'jmaxha_acclim',
    'jmaxhd_noacclim', 'jmaxhd_acclim', 'jmaxse_noacclim', 'jmaxse_acclim',
    'rdha', 'rdhd', 'rdse', 'jmax25_to_vcmax25_noacclim', 'jmax25_to_vcmax25_acclim',
    'rd25_to_vcmax25_c3', 'rd25_to_vcmax25_c4', 'kp25_to_vcmax25_c4',
    'phi_psii', 'theta_j', 'qe_c4', 'colim_c3a', 'colim_c3b', 'colim_c4a', 'colim_c4b',
    'dh2o_to_dco2', 'rh_min_bb', 'vpd_min_med', 'cpbio', 'fcarbon', 'fwater',
    'gb_factor', 'dewmx', 'maximum_leaf_wetted_fraction', 'interception_fraction',
    'fwet_exponent', 'clm45_interception_p1', 'clm45_interception_p2',
    'chil_min', 'chil_max', 'kb_max', 'j_to_umol', 'emg', 'cd', 'beta_neutral_max',
    'cr', 'c2', 'pr0', 'pr1', 'pr2', 'z0mg', 'wind_forc_min', 'eta_max',
    'zeta_min', 'zeta_max', 'beta_min', 'beta_max', 'wind_min', 'ra_max',
    'n_z', 'n_l'
])

RSLPsihatLookupTables = namedtuple('RSLPsihatLookupTables', [
    'zdtgrid_m', 'dtlgrid_m', 'psigrid_m',
    'zdtgrid_h', 'dtlgrid_h', 'psigrid_h'
])


# Mock implementation for testing (replace with actual import)
def create_empty_rsl_lookup_tables(constants):
    """
    Create empty RSL lookup tables with correct shapes initialized to zeros.
    
    Args:
        constants: MLCanopyConstants containing n_z and n_l dimensions
        
    Returns:
        RSLPsihatLookupTables with all arrays as JAX arrays initialized to zeros
    """
    n_z = constants.n_z
    n_l = constants.n_l
    
    return RSLPsihatLookupTables(
        zdtgrid_m=jnp.zeros((n_z, 1)),
        dtlgrid_m=jnp.zeros((1, n_l)),
        psigrid_m=jnp.zeros((n_z, n_l)),
        zdtgrid_h=jnp.zeros((n_z, 1)),
        dtlgrid_h=jnp.zeros((1, n_l)),
        psigrid_h=jnp.zeros((n_z, n_l))
    )


# Default constants for testing
DEFAULT_CONSTANTS = MLCanopyConstants(
    rgas=8.314, mmdry=28.97, mmh2o=18.016, cpd=1005.0, cpw=1846.0,
    visc0=1.5e-05, dh0=2.12e-05, dv0=2.4e-05, dc0=1.47e-05, lapse_rate=0.0065,
    kc25=404.9, kcha=79430.0, ko25=278.4, koha=36380.0, cp25=42.75, cpha=37830.0,
    vcmaxha_noacclim=72000.0, vcmaxha_acclim=65330.0, vcmaxhd_noacclim=200000.0,
    vcmaxhd_acclim=200000.0, vcmaxse_noacclim=668.39, vcmaxse_acclim=668.39,
    jmaxha_noacclim=50000.0, jmaxha_acclim=43540.0, jmaxhd_noacclim=200000.0,
    jmaxhd_acclim=152040.0, jmaxse_noacclim=659.7, jmaxse_acclim=495.0,
    rdha=46390.0, rdhd=150650.0, rdse=490.0, jmax25_to_vcmax25_noacclim=1.67,
    jmax25_to_vcmax25_acclim=1.67, rd25_to_vcmax25_c3=0.015, rd25_to_vcmax25_c4=0.025,
    kp25_to_vcmax25_c4=0.02, phi_psii=0.7, theta_j=0.9, qe_c4=0.05,
    colim_c3a=0.98, colim_c3b=0.95, colim_c4a=0.8, colim_c4b=0.95,
    dh2o_to_dco2=1.6, rh_min_bb=0.3, vpd_min_med=50.0, cpbio=2000000.0,
    fcarbon=0.5, fwater=0.5, gb_factor=1.0, dewmx=0.1,
    maximum_leaf_wetted_fraction=0.05, interception_fraction=0.25,
    fwet_exponent=0.667, clm45_interception_p1=0.25, clm45_interception_p2=0.5,
    chil_min=-0.4, chil_max=0.6, kb_max=0.9, j_to_umol=4.6, emg=0.97,
    cd=0.3, beta_neutral_max=1.0, cr=0.3, c2=0.75, pr0=0.5, pr1=0.3, pr2=0.3,
    z0mg=0.01, wind_forc_min=0.1, eta_max=10.0, zeta_min=-2.0, zeta_max=1.0,
    beta_min=0.0, beta_max=2.0, wind_min=0.1, ra_max=999.0,
    n_z=10, n_l=10
)


@pytest.fixture
def test_data():
    """
    Fixture providing test data for create_empty_rsl_lookup_tables tests.
    
    Returns:
        Dict containing test cases with inputs and expected outputs
    """
    return {
        'nominal_default': {
            'constants': DEFAULT_CONSTANTS,
            'expected_shapes': {
                'zdtgrid_m': (10, 1),
                'dtlgrid_m': (1, 10),
                'psigrid_m': (10, 10),
                'zdtgrid_h': (10, 1),
                'dtlgrid_h': (1, 10),
                'psigrid_h': (10, 10)
            }
        },
        'nominal_small': {
            'constants': DEFAULT_CONSTANTS._replace(n_z=5, n_l=5),
            'expected_shapes': {
                'zdtgrid_m': (5, 1),
                'dtlgrid_m': (1, 5),
                'psigrid_m': (5, 5),
                'zdtgrid_h': (5, 1),
                'dtlgrid_h': (1, 5),
                'psigrid_h': (5, 5)
            }
        },
        'nominal_large': {
            'constants': DEFAULT_CONSTANTS._replace(n_z=50, n_l=50),
            'expected_shapes': {
                'zdtgrid_m': (50, 1),
                'dtlgrid_m': (1, 50),
                'psigrid_m': (50, 50),
                'zdtgrid_h': (50, 1),
                'dtlgrid_h': (1, 50),
                'psigrid_h': (50, 50)
            }
        },
        'nominal_rectangular': {
            'constants': DEFAULT_CONSTANTS._replace(n_z=20, n_l=15),
            'expected_shapes': {
                'zdtgrid_m': (20, 1),
                'dtlgrid_m': (1, 15),
                'psigrid_m': (20, 15),
                'zdtgrid_h': (20, 1),
                'dtlgrid_h': (1, 15),
                'psigrid_h': (20, 15)
            }
        },
        'edge_minimum': {
            'constants': DEFAULT_CONSTANTS._replace(n_z=1, n_l=1),
            'expected_shapes': {
                'zdtgrid_m': (1, 1),
                'dtlgrid_m': (1, 1),
                'psigrid_m': (1, 1),
                'zdtgrid_h': (1, 1),
                'dtlgrid_h': (1, 1),
                'psigrid_h': (1, 1)
            }
        },
        'special_very_large': {
            'constants': DEFAULT_CONSTANTS._replace(n_z=100, n_l=100),
            'expected_shapes': {
                'zdtgrid_m': (100, 1),
                'dtlgrid_m': (1, 100),
                'psigrid_m': (100, 100),
                'zdtgrid_h': (100, 1),
                'dtlgrid_h': (1, 100),
                'psigrid_h': (100, 100)
            }
        },
        'special_extreme_asymmetry': {
            'constants': DEFAULT_CONSTANTS._replace(n_z=2, n_l=100),
            'expected_shapes': {
                'zdtgrid_m': (2, 1),
                'dtlgrid_m': (1, 100),
                'psigrid_m': (2, 100),
                'zdtgrid_h': (2, 1),
                'dtlgrid_h': (1, 100),
                'psigrid_h': (2, 100)
            }
        }
    }


# Parametrized test cases
test_cases = [
    ('nominal_default', 10, 10, 'Default dimensions (10x10)'),
    ('nominal_small', 5, 5, 'Small dimensions (5x5)'),
    ('nominal_large', 50, 50, 'Large dimensions (50x50)'),
    ('nominal_rectangular', 20, 15, 'Rectangular dimensions (20x15)'),
    ('edge_minimum', 1, 1, 'Minimum dimensions (1x1)'),
    ('special_very_large', 100, 100, 'Very large dimensions (100x100)'),
    ('special_extreme_asymmetry', 2, 100, 'Extreme asymmetry (2x100)'),
]


@pytest.mark.parametrize('test_name,n_z,n_l,description', test_cases)
def test_create_empty_rsl_lookup_tables_shapes(test_data, test_name, n_z, n_l, description):
    """
    Test that create_empty_rsl_lookup_tables returns arrays with correct shapes.
    
    Verifies that all six arrays in the returned RSLPsihatLookupTables have
    the expected shapes based on n_z and n_l dimensions:
    - zdtgrid_m, zdtgrid_h: (n_z, 1)
    - dtlgrid_m, dtlgrid_h: (1, n_l)
    - psigrid_m, psigrid_h: (n_z, n_l)
    
    Args:
        test_data: Fixture providing test cases
        test_name: Name of the test case
        n_z: Expected vertical dimension
        n_l: Expected stability dimension
        description: Human-readable test description
    """
    test_case = test_data[test_name]
    constants = test_case['constants']
    expected_shapes = test_case['expected_shapes']
    
    result = create_empty_rsl_lookup_tables(constants)
    
    # Verify all shapes
    assert result.zdtgrid_m.shape == expected_shapes['zdtgrid_m'], \
        f"{description}: zdtgrid_m shape mismatch. Expected {expected_shapes['zdtgrid_m']}, got {result.zdtgrid_m.shape}"
    
    assert result.dtlgrid_m.shape == expected_shapes['dtlgrid_m'], \
        f"{description}: dtlgrid_m shape mismatch. Expected {expected_shapes['dtlgrid_m']}, got {result.dtlgrid_m.shape}"
    
    assert result.psigrid_m.shape == expected_shapes['psigrid_m'], \
        f"{description}: psigrid_m shape mismatch. Expected {expected_shapes['psigrid_m']}, got {result.psigrid_m.shape}"
    
    assert result.zdtgrid_h.shape == expected_shapes['zdtgrid_h'], \
        f"{description}: zdtgrid_h shape mismatch. Expected {expected_shapes['zdtgrid_h']}, got {result.zdtgrid_h.shape}"
    
    assert result.dtlgrid_h.shape == expected_shapes['dtlgrid_h'], \
        f"{description}: dtlgrid_h shape mismatch. Expected {expected_shapes['dtlgrid_h']}, got {result.dtlgrid_h.shape}"
    
    assert result.psigrid_h.shape == expected_shapes['psigrid_h'], \
        f"{description}: psigrid_h shape mismatch. Expected {expected_shapes['psigrid_h']}, got {result.psigrid_h.shape}"


@pytest.mark.parametrize('test_name,n_z,n_l,description', test_cases)
def test_create_empty_rsl_lookup_tables_values(test_data, test_name, n_z, n_l, description):
    """
    Test that all arrays are initialized to zeros.
    
    Verifies that create_empty_rsl_lookup_tables initializes all arrays
    in the lookup tables to zero values, as expected for empty tables
    that will be populated later.
    
    Args:
        test_data: Fixture providing test cases
        test_name: Name of the test case
        n_z: Expected vertical dimension
        n_l: Expected stability dimension
        description: Human-readable test description
    """
    test_case = test_data[test_name]
    constants = test_case['constants']
    
    result = create_empty_rsl_lookup_tables(constants)
    
    # Verify all arrays are zeros
    assert jnp.allclose(result.zdtgrid_m, 0.0, atol=1e-10), \
        f"{description}: zdtgrid_m not initialized to zeros"
    
    assert jnp.allclose(result.dtlgrid_m, 0.0, atol=1e-10), \
        f"{description}: dtlgrid_m not initialized to zeros"
    
    assert jnp.allclose(result.psigrid_m, 0.0, atol=1e-10), \
        f"{description}: psigrid_m not initialized to zeros"
    
    assert jnp.allclose(result.zdtgrid_h, 0.0, atol=1e-10), \
        f"{description}: zdtgrid_h not initialized to zeros"
    
    assert jnp.allclose(result.dtlgrid_h, 0.0, atol=1e-10), \
        f"{description}: dtlgrid_h not initialized to zeros"
    
    assert jnp.allclose(result.psigrid_h, 0.0, atol=1e-10), \
        f"{description}: psigrid_h not initialized to zeros"


@pytest.mark.parametrize('test_name,n_z,n_l,description', test_cases)
def test_create_empty_rsl_lookup_tables_dtypes(test_data, test_name, n_z, n_l, description):
    """
    Test that all arrays are JAX arrays with correct dtype.
    
    Verifies that create_empty_rsl_lookup_tables returns JAX arrays
    (jnp.ndarray) for GPU compatibility, not NumPy arrays.
    
    Args:
        test_data: Fixture providing test cases
        test_name: Name of the test case
        n_z: Expected vertical dimension
        n_l: Expected stability dimension
        description: Human-readable test description
    """
    test_case = test_data[test_name]
    constants = test_case['constants']
    
    result = create_empty_rsl_lookup_tables(constants)
    
    # Verify all arrays are JAX arrays
    assert isinstance(result.zdtgrid_m, jnp.ndarray), \
        f"{description}: zdtgrid_m is not a JAX array"
    
    assert isinstance(result.dtlgrid_m, jnp.ndarray), \
        f"{description}: dtlgrid_m is not a JAX array"
    
    assert isinstance(result.psigrid_m, jnp.ndarray), \
        f"{description}: psigrid_m is not a JAX array"
    
    assert isinstance(result.zdtgrid_h, jnp.ndarray), \
        f"{description}: zdtgrid_h is not a JAX array"
    
    assert isinstance(result.dtlgrid_h, jnp.ndarray), \
        f"{description}: dtlgrid_h is not a JAX array"
    
    assert isinstance(result.psigrid_h, jnp.ndarray), \
        f"{description}: psigrid_h is not a JAX array"


def test_create_empty_rsl_lookup_tables_return_type(test_data):
    """
    Test that the function returns the correct namedtuple type.
    
    Verifies that create_empty_rsl_lookup_tables returns an instance
    of RSLPsihatLookupTables namedtuple with all expected fields.
    """
    constants = test_data['nominal_default']['constants']
    result = create_empty_rsl_lookup_tables(constants)
    
    # Verify return type
    assert isinstance(result, RSLPsihatLookupTables), \
        f"Return type should be RSLPsihatLookupTables, got {type(result)}"
    
    # Verify all fields are present
    expected_fields = ['zdtgrid_m', 'dtlgrid_m', 'psigrid_m', 
                      'zdtgrid_h', 'dtlgrid_h', 'psigrid_h']
    
    for field in expected_fields:
        assert hasattr(result, field), \
            f"Missing field '{field}' in returned namedtuple"


def test_create_empty_rsl_lookup_tables_edge_zero_fractions():
    """
    Test with fraction parameters at zero boundary.
    
    Verifies that the function works correctly when fraction parameters
    (rh_min_bb, fcarbon, fwater, emg, wind speeds) are at their minimum
    valid values (0). This tests boundary conditions for physical constraints.
    """
    constants = DEFAULT_CONSTANTS._replace(
        rh_min_bb=0.0,
        vpd_min_med=0.0,
        fcarbon=0.0,
        fwater=0.0,
        maximum_leaf_wetted_fraction=0.0,
        interception_fraction=0.0,
        emg=0.0,
        wind_forc_min=0.0,
        wind_min=0.0
    )
    
    result = create_empty_rsl_lookup_tables(constants)
    
    # Verify shapes are still correct
    assert result.zdtgrid_m.shape == (10, 1)
    assert result.dtlgrid_m.shape == (1, 10)
    assert result.psigrid_m.shape == (10, 10)
    
    # Verify initialization to zeros
    assert jnp.allclose(result.psigrid_m, 0.0, atol=1e-10), \
        "Arrays should still be initialized to zeros with zero fraction parameters"


def test_create_empty_rsl_lookup_tables_edge_max_fractions():
    """
    Test with fraction parameters at maximum boundary.
    
    Verifies that the function works correctly when fraction parameters
    (rh_min_bb, fcarbon, fwater, emg) are at their maximum valid values (1).
    This tests upper boundary conditions for physical constraints.
    """
    constants = DEFAULT_CONSTANTS._replace(
        rh_min_bb=1.0,
        fcarbon=1.0,
        fwater=1.0,
        maximum_leaf_wetted_fraction=1.0,
        interception_fraction=1.0,
        emg=1.0
    )
    
    result = create_empty_rsl_lookup_tables(constants)
    
    # Verify shapes are still correct
    assert result.zdtgrid_h.shape == (10, 1)
    assert result.dtlgrid_h.shape == (1, 10)
    assert result.psigrid_h.shape == (10, 10)
    
    # Verify initialization to zeros
    assert jnp.allclose(result.psigrid_h, 0.0, atol=1e-10), \
        "Arrays should still be initialized to zeros with maximum fraction parameters"


def test_create_empty_rsl_lookup_tables_consistency():
    """
    Test that momentum and heat grids have consistent dimensions.
    
    Verifies that the momentum (_m) and heat (_h) grids have identical
    shapes, as they should represent the same physical grid structure.
    """
    constants = DEFAULT_CONSTANTS._replace(n_z=15, n_l=20)
    result = create_empty_rsl_lookup_tables(constants)
    
    # Verify momentum and heat grids have same shapes
    assert result.zdtgrid_m.shape == result.zdtgrid_h.shape, \
        "Momentum and heat zdtgrid should have same shape"
    
    assert result.dtlgrid_m.shape == result.dtlgrid_h.shape, \
        "Momentum and heat dtlgrid should have same shape"
    
    assert result.psigrid_m.shape == result.psigrid_h.shape, \
        "Momentum and heat psigrid should have same shape"


def test_create_empty_rsl_lookup_tables_memory_layout():
    """
    Test that arrays have expected memory layout for broadcasting.
    
    Verifies that zdtgrid arrays are column vectors (n_z, 1) and
    dtlgrid arrays are row vectors (1, n_l), which is the correct
    layout for broadcasting operations in lookup table interpolation.
    """
    constants = DEFAULT_CONSTANTS._replace(n_z=7, n_l=13)
    result = create_empty_rsl_lookup_tables(constants)
    
    # Verify column vector layout for zdtgrid
    assert result.zdtgrid_m.shape[1] == 1, \
        "zdtgrid_m should be a column vector with second dimension = 1"
    assert result.zdtgrid_h.shape[1] == 1, \
        "zdtgrid_h should be a column vector with second dimension = 1"
    
    # Verify row vector layout for dtlgrid
    assert result.dtlgrid_m.shape[0] == 1, \
        "dtlgrid_m should be a row vector with first dimension = 1"
    assert result.dtlgrid_h.shape[0] == 1, \
        "dtlgrid_h should be a row vector with first dimension = 1"
    
    # Verify 2D grid layout for psigrid
    assert len(result.psigrid_m.shape) == 2, \
        "psigrid_m should be a 2D array"
    assert len(result.psigrid_h.shape) == 2, \
        "psigrid_h should be a 2D array"


def test_create_empty_rsl_lookup_tables_independence():
    """
    Test that multiple calls produce independent arrays.
    
    Verifies that calling the function multiple times produces
    independent arrays that don't share memory, preventing
    unintended side effects from modifications.
    """
    constants = DEFAULT_CONSTANTS._replace(n_z=5, n_l=5)
    
    result1 = create_empty_rsl_lookup_tables(constants)
    result2 = create_empty_rsl_lookup_tables(constants)
    
    # Verify arrays are not the same object
    assert result1.psigrid_m is not result2.psigrid_m, \
        "Multiple calls should produce independent arrays"
    
    # Verify shapes are identical
    assert result1.psigrid_m.shape == result2.psigrid_m.shape, \
        "Multiple calls should produce arrays with same shape"


def test_create_empty_rsl_lookup_tables_no_nan_inf():
    """
    Test that arrays contain no NaN or Inf values.
    
    Verifies that all arrays are properly initialized with finite
    values (zeros), with no NaN or Inf values that could cause
    numerical issues in downstream calculations.
    """
    constants = DEFAULT_CONSTANTS
    result = create_empty_rsl_lookup_tables(constants)
    
    # Check all arrays for NaN/Inf
    arrays_to_check = [
        ('zdtgrid_m', result.zdtgrid_m),
        ('dtlgrid_m', result.dtlgrid_m),
        ('psigrid_m', result.psigrid_m),
        ('zdtgrid_h', result.zdtgrid_h),
        ('dtlgrid_h', result.dtlgrid_h),
        ('psigrid_h', result.psigrid_h)
    ]
    
    for name, array in arrays_to_check:
        assert jnp.all(jnp.isfinite(array)), \
            f"{name} contains NaN or Inf values"
        assert not jnp.any(jnp.isnan(array)), \
            f"{name} contains NaN values"
        assert not jnp.any(jnp.isinf(array)), \
            f"{name} contains Inf values"


@pytest.mark.parametrize('n_z,n_l', [
    (3, 7),
    (11, 13),
    (17, 19),
])
def test_create_empty_rsl_lookup_tables_prime_dimensions(n_z, n_l):
    """
    Test with prime number dimensions to catch indexing errors.
    
    Prime numbers are good test cases because they don't have
    common factors, which can help catch off-by-one errors or
    incorrect dimension handling.
    
    Args:
        n_z: Prime number for vertical dimension
        n_l: Prime number for stability dimension
    """
    constants = DEFAULT_CONSTANTS._replace(n_z=n_z, n_l=n_l)
    result = create_empty_rsl_lookup_tables(constants)
    
    # Verify correct shapes
    assert result.zdtgrid_m.shape == (n_z, 1)
    assert result.dtlgrid_m.shape == (1, n_l)
    assert result.psigrid_m.shape == (n_z, n_l)
    
    # Verify total elements
    assert result.psigrid_m.size == n_z * n_l, \
        f"psigrid_m should have {n_z * n_l} elements"


def test_create_empty_rsl_lookup_tables_constants_immutability():
    """
    Test that the function doesn't modify the input constants.
    
    Verifies that the input MLCanopyConstants namedtuple is not
    modified by the function, ensuring no side effects on the
    input parameters.
    """
    constants = DEFAULT_CONSTANTS._replace(n_z=8, n_l=12)
    original_n_z = constants.n_z
    original_n_l = constants.n_l
    
    _ = create_empty_rsl_lookup_tables(constants)
    
    # Verify constants unchanged
    assert constants.n_z == original_n_z, \
        "Function should not modify input constants.n_z"
    assert constants.n_l == original_n_l, \
        "Function should not modify input constants.n_l"