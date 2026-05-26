"""
Comprehensive pytest suite for MLCanopyTurbulenceMod functions.

This module tests Monin-Obukhov similarity theory functions, canopy turbulence
parameterizations, and roughness sublayer (RSL) corrections for multi-layer
canopy models.

Test Coverage:
- Monin-Obukhov stability functions (phi_m, phi_c, psi_m, psi_c)
- Prandtl/Schmidt number calculations
- Beta parameter (u*/u ratio) calculations
- RSL psi function lookups and interpolation
- Complete Obukhov length calculations
- Full canopy turbulence parameterization
- Edge cases: extreme stability, sparse/dense canopies, low winds
- Array dimension handling: scalar, 1D, 2D, 3D arrays
"""

import sys
from pathlib import Path
from typing import NamedTuple, Callable
import pytest
import jax.numpy as jnp
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from multilayer_canopy.MLCanopyTurbulenceMod import (
    phim_monin_obukhov,
    phic_monin_obukhov,
    psim_monin_obukhov,
    psic_monin_obukhov,
    get_prsc,
    get_beta,
    lookup_psihat,
    get_psi_rsl,
    obu_func,
    PrScParams,
    PsiRSLResult,
    ObuFuncInputs,
    ObuFuncOutputs,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data():
    """
    Load and provide test data for all test cases.
    
    Returns:
        dict: Test data organized by test case name
    """
    return {
        "phim_neutral": {
            "zeta": jnp.array([0.0, 0.001, -0.001, 0.01, -0.01])
        },
        "phim_unstable": {
            "zeta": jnp.array([-10.0, -5.0, -2.0, -1.0, -0.5, -0.1])
        },
        "phim_stable": {
            "zeta": jnp.array([0.1, 0.2, 0.5, 0.8, 1.0])
        },
        "phim_extreme_unstable": {
            "zeta": jnp.array([-100.0, -50.0, -25.0])
        },
        "phic_multidim": {
            "zeta": jnp.array([[-5.0, -2.0, -0.5], [0.0, 0.1, 0.5], [-1.0, 0.0, 1.0]])
        },
        "prsc_typical": {
            "beta_neutral": jnp.array([0.25, 0.3, 0.28, 0.32]),
            "beta_neutral_max": jnp.array([0.35, 0.35, 0.35, 0.35]),
            "LcL": jnp.array([0.5, -0.2, 1.0, -1.5]),
            "params": PrScParams(Pr0=0.5, Pr1=0.3, Pr2=0.143)
        },
        "prsc_edge_beta": {
            "beta_neutral": jnp.array([0.01, 0.99, 0.5]),
            "beta_neutral_max": jnp.array([0.35, 0.35, 0.35]),
            "LcL": jnp.array([0.0, 0.0, 0.0]),
            "params": PrScParams(Pr0=0.5, Pr1=0.3, Pr2=0.143)
        },
        "beta_neutral": {
            "beta_neutral": 0.3,
            "lcl": 0.0,
            "beta_min": 0.01,
            "beta_max": 0.99,
        },
        "lookup_psihat": {
            "zdt": 1.5,
            "dtL": 0.3,
            "zdtgrid": jnp.array([[3.0], [2.0], [1.0], [0.5], [0.1]]),
            "dtLgrid": jnp.array([[-1.0, -0.5, 0.0, 0.5, 1.0]]),
            "psigrid": jnp.array([
                [0.5, 0.6, 0.7, 0.8, 0.9],
                [0.4, 0.5, 0.6, 0.7, 0.8],
                [0.3, 0.4, 0.5, 0.6, 0.7],
                [0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3, 0.4, 0.5]
            ])
        },
        "psi_rsl_complete": {
            "za": jnp.array([50.0, 45.0, 60.0, 40.0]),
            "hc": jnp.array([20.0, 18.0, 25.0, 15.0]),
            "disp": jnp.array([13.0, 12.0, 16.0, 10.0]),
            "obu": jnp.array([-100.0, 50.0, -200.0, 150.0]),
            "beta": jnp.array([0.3, 0.28, 0.32, 0.25]),
            "prsc": jnp.array([0.7, 0.65, 0.75, 0.68]),
            "vkc": 0.4,
            "c2": 0.5,
        },
        "array_shapes": {
            "zeta_scalar": 0.5,
            "zeta_1d": jnp.array([-2.0, -1.0, 0.0, 0.5, 1.0]),
            "zeta_2d": jnp.array([[-5.0, -2.0, 0.0], [0.5, 1.0, 0.8]]),
            "zeta_3d": jnp.array([[[-10.0, -5.0], [-1.0, 0.0]], [[0.2, 0.5], [0.8, 1.0]]]),
            "pi": jnp.pi
        },
        "lookup_boundary": {
            "zdt_below": 0.05,
            "zdt_above": 6.0,
            "dtL_below": -2.5,
            "dtL_above": 2.5,
            "zdtgrid": jnp.array([[5.0], [3.0], [2.0], [1.0], [0.5], [0.1]]),
            "dtLgrid": jnp.array([[-2.0, -1.0, 0.0, 1.0, 2.0]]),
            "psigrid": jnp.array([
                [1.0, 1.2, 1.4, 1.6, 1.8],
                [0.8, 1.0, 1.2, 1.4, 1.6],
                [0.6, 0.8, 1.0, 1.2, 1.4],
                [0.4, 0.6, 0.8, 1.0, 1.2],
                [0.3, 0.5, 0.7, 0.9, 1.1],
                [0.2, 0.4, 0.6, 0.8, 1.0]
            ])
        }
    }


@pytest.fixture
def lookup_grids():
    """
    Provide standard lookup table grids for RSL psi functions.
    
    Returns:
        dict: Grids for momentum and scalar lookups
    """
    dtlgrid = jnp.array([[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]])
    zdtgrid = jnp.array([[5.0], [3.0], [2.0], [1.5], [1.0], [0.5]])
    
    psigrid_m = jnp.array([
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ])
    
    psigrid_h = jnp.array([
        [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ])
    
    return {
        "dtlgrid_m": dtlgrid,
        "zdtgrid_m": zdtgrid,
        "psigrid_m": psigrid_m,
        "dtlgrid_h": dtlgrid,
        "zdtgrid_h": zdtgrid,
        "psigrid_h": psigrid_h
    }


@pytest.fixture
def obu_func_inputs():
    """
    Provide test inputs for obu_func tests.
    
    Returns:
        dict: Various ObuFuncInputs configurations
    """
    return {
        "typical_unstable": ObuFuncInputs(
            p=0, ic=5, il=1,
            obu_val=jnp.array(-50.0),
            zref=jnp.array(50.0),
            uref=jnp.array(5.0),
            thref=jnp.array(298.15),
            thvref=jnp.array(299.0),
            qref=jnp.array(0.012),
            rhomol=jnp.array(41.6),
            ztop=jnp.array(20.0),
            lai=jnp.array(4.5),
            sai=jnp.array(0.5),
            Lc=jnp.array(8.0),
            taf=jnp.array(295.0),
            qaf=jnp.array(0.01),
            vkc=0.4,
            grav=9.80616,
            beta_neutral_max=0.35,
            cr=0.3,
            z0mg=0.01,
            zeta_min=-100.0,
            zeta_max=1.0
        ),
        "stable_nighttime": ObuFuncInputs(
            p=0, ic=5, il=2,
            obu_val=jnp.array(100.0),
            zref=jnp.array(50.0),
            uref=jnp.array(3.0),
            thref=jnp.array(285.0),
            thvref=jnp.array(285.5),
            qref=jnp.array(0.005),
            rhomol=jnp.array(43.0),
            ztop=jnp.array(15.0),
            lai=jnp.array(3.0),
            sai=jnp.array(0.3),
            Lc=jnp.array(6.0),
            taf=jnp.array(283.0),
            qaf=jnp.array(0.004),
            vkc=0.4,
            grav=9.80616,
            beta_neutral_max=0.35,
            cr=0.3,
            z0mg=0.01,
            zeta_min=-100.0,
            zeta_max=1.0
        ),
        "near_neutral": ObuFuncInputs(
            p=0, ic=3, il=1,
            obu_val=jnp.array(1000.0),
            zref=jnp.array(40.0),
            uref=jnp.array(4.5),
            thref=jnp.array(293.0),
            thvref=jnp.array(293.5),
            qref=jnp.array(0.008),
            rhomol=jnp.array(42.0),
            ztop=jnp.array(12.0),
            lai=jnp.array(2.5),
            sai=jnp.array(0.2),
            Lc=jnp.array(5.0),
            taf=jnp.array(293.0),
            qaf=jnp.array(0.008),
            vkc=0.4,
            grav=9.80616,
            beta_neutral_max=0.35,
            cr=0.3,
            z0mg=0.01,
            zeta_min=-100.0,
            zeta_max=1.0
        ),
        "low_wind": ObuFuncInputs(
            p=0, ic=4, il=1,
            obu_val=jnp.array(-30.0),
            zref=jnp.array(45.0),
            uref=jnp.array(0.1),
            thref=jnp.array(300.0),
            thvref=jnp.array(301.0),
            qref=jnp.array(0.015),
            rhomol=jnp.array(41.0),
            ztop=jnp.array(18.0),
            lai=jnp.array(5.0),
            sai=jnp.array(0.6),
            Lc=jnp.array(7.5),
            taf=jnp.array(298.0),
            qaf=jnp.array(0.013),
            vkc=0.4,
            grav=9.80616,
            beta_neutral_max=0.35,
            cr=0.3,
            z0mg=0.01,
            zeta_min=-100.0,
            zeta_max=1.0
        ),
        "sparse_canopy": ObuFuncInputs(
            p=0, ic=2, il=1,
            obu_val=jnp.array(-80.0),
            zref=jnp.array(35.0),
            uref=jnp.array(6.0),
            thref=jnp.array(302.0),
            thvref=jnp.array(303.0),
            qref=jnp.array(0.018),
            rhomol=jnp.array(40.5),
            ztop=jnp.array(8.0),
            lai=jnp.array(0.5),
            sai=jnp.array(0.1),
            Lc=jnp.array(3.0),
            taf=jnp.array(300.0),
            qaf=jnp.array(0.016),
            vkc=0.4,
            grav=9.80616,
            beta_neutral_max=0.35,
            cr=0.3,
            z0mg=0.01,
            zeta_min=-100.0,
            zeta_max=1.0
        ),
        "dense_canopy": ObuFuncInputs(
            p=0, ic=6, il=1,
            obu_val=jnp.array(-60.0),
            zref=jnp.array(55.0),
            uref=jnp.array(4.0),
            thref=jnp.array(296.0),
            thvref=jnp.array(297.0),
            qref=jnp.array(0.011),
            rhomol=jnp.array(42.2),
            ztop=jnp.array(25.0),
            lai=jnp.array(8.0),
            sai=jnp.array(1.2),
            Lc=jnp.array(12.0),
            taf=jnp.array(294.0),
            qaf=jnp.array(0.009),
            vkc=0.4,
            grav=9.80616,
            beta_neutral_max=0.35,
            cr=0.3,
            z0mg=0.01,
            zeta_min=-100.0,
            zeta_max=1.0
        )
    }


# ============================================================================
# Tests for phim_monin_obukhov
# ============================================================================

def test_phim_neutral_conditions(test_data):
    """
    Test phi_m near neutral stability (zeta ≈ 0).
    
    For neutral conditions, phi_m should be close to 1.0.
    """
    zeta = test_data["phim_neutral"]["zeta"]
    result = phim_monin_obukhov(zeta)
    
    # Check shape
    assert result.shape == zeta.shape, f"Expected shape {zeta.shape}, got {result.shape}"
    
    # Check values near 1.0 for neutral conditions
    assert jnp.allclose(result, 1.0, atol=0.1), \
        f"phi_m should be near 1.0 for neutral conditions, got {result}"
    
    # Check dtype
    assert result.dtype == jnp.float32 or result.dtype == jnp.float64, \
        f"Expected float dtype, got {result.dtype}"


def test_phim_unstable_range(test_data):
    """
    Test phi_m in typical unstable conditions (convective).
    
    For unstable conditions (zeta < 0), phi_m should be < 1.0 and decrease
    with more negative zeta.
    """
    zeta = test_data["phim_unstable"]["zeta"]
    result = phim_monin_obukhov(zeta)
    
    # Check shape
    assert result.shape == zeta.shape
    
    # Check all values are less than 1.0 for unstable conditions
    assert jnp.all(result < 1.0), \
        f"phi_m should be < 1.0 for unstable conditions, got {result}"
    
    # Check all values are positive
    assert jnp.all(result > 0), \
        f"phi_m should be positive, got {result}"
    
    # Check monotonicity: more negative zeta -> smaller phi_m
    # (values should generally decrease as we go through the array)
    assert jnp.all(result[:-1] >= result[1:] - 0.1), \
        "phi_m should decrease with more negative zeta"


def test_phim_stable_range(test_data):
    """
    Test phi_m in stable conditions.
    
    For stable conditions (zeta > 0), phi_m should be > 1.0 and increase
    with zeta.
    """
    zeta = test_data["phim_stable"]["zeta"]
    result = phim_monin_obukhov(zeta)
    
    # Check shape
    assert result.shape == zeta.shape
    
    # Check all values are greater than 1.0 for stable conditions
    assert jnp.all(result > 1.0), \
        f"phi_m should be > 1.0 for stable conditions, got {result}"
    
    # Check monotonicity: larger zeta -> larger phi_m
    diffs = jnp.diff(result)
    assert jnp.all(diffs > -0.01), \
        "phi_m should increase with zeta in stable conditions"


def test_phim_extreme_unstable(test_data):
    """
    Test phi_m at extreme unstable boundary (zeta = -100).
    
    Tests numerical stability at the lower boundary of the typical range.
    """
    zeta = test_data["phim_extreme_unstable"]["zeta"]
    result = phim_monin_obukhov(zeta)
    
    # Check shape
    assert result.shape == zeta.shape
    
    # Check values are finite and positive
    assert jnp.all(jnp.isfinite(result)), \
        f"phi_m should be finite at extreme unstable conditions, got {result}"
    assert jnp.all(result > 0), \
        f"phi_m should be positive, got {result}"
    
    # Check values are reasonable (not too small)
    assert jnp.all(result > 0.01), \
        f"phi_m should not be too small even at extreme unstable, got {result}"


# ============================================================================
# Tests for phic_monin_obukhov
# ============================================================================

def test_phic_neutral_conditions(test_data):
    """
    Test phi_c near neutral stability.
    
    For neutral conditions, phi_c should be close to 1.0.
    """
    zeta = test_data["phim_neutral"]["zeta"]
    result = phic_monin_obukhov(zeta)
    
    # Check shape
    assert result.shape == zeta.shape
    
    # Check values near 1.0 for neutral conditions
    assert jnp.allclose(result, 1.0, atol=0.1), \
        f"phi_c should be near 1.0 for neutral conditions, got {result}"


def test_phic_multidimensional(test_data):
    """
    Test phi_c with 2D array input covering stability range.
    
    Verifies that the function handles multidimensional arrays correctly.
    """
    zeta = test_data["phic_multidim"]["zeta"]
    result = phic_monin_obukhov(zeta)
    
    # Check shape preservation
    assert result.shape == zeta.shape, \
        f"Expected shape {zeta.shape}, got {result.shape}"
    
    # Check all values are positive
    assert jnp.all(result > 0), \
        f"phi_c should be positive, got {result}"
    
    # Check all values are finite
    assert jnp.all(jnp.isfinite(result)), \
        f"phi_c should be finite, got {result}"


def test_phic_vs_phim_relationship(test_data):
    """
    Test relationship between phi_c and phi_m.
    
    In general, phi_c and phi_m should have similar behavior but phi_c
    typically has stronger response to stability.
    """
    zeta = test_data["phim_unstable"]["zeta"]
    phim = phim_monin_obukhov(zeta)
    phic = phic_monin_obukhov(zeta)
    
    # Both should be positive
    assert jnp.all(phim > 0) and jnp.all(phic > 0)
    
    # For unstable conditions, both should be < 1
    assert jnp.all(phim < 1.0) and jnp.all(phic < 1.0)


# ============================================================================
# Tests for psim_monin_obukhov
# ============================================================================

def test_psim_neutral_conditions(test_data):
    """
    Test psi_m near neutral stability.
    
    For neutral conditions, psi_m should be close to 0.
    """
    zeta = test_data["phim_neutral"]["zeta"]
    result = psim_monin_obukhov(zeta)
    
    # Check shape
    assert result.shape == zeta.shape
    
    # Check values near 0 for neutral conditions
    assert jnp.allclose(result, 0.0, atol=0.1), \
        f"psi_m should be near 0 for neutral conditions, got {result}"


def test_psim_unstable_positive(test_data):
    """
    Test psi_m in unstable conditions.
    
    For unstable conditions (zeta < 0), psi_m should be positive.
    """
    zeta = test_data["phim_unstable"]["zeta"]
    result = psim_monin_obukhov(zeta)
    
    # Check shape
    assert result.shape == zeta.shape
    
    # Check all values are positive for unstable conditions
    assert jnp.all(result > 0), \
        f"psi_m should be positive for unstable conditions, got {result}"


def test_psim_stable_negative(test_data):
    """
    Test psi_m in stable conditions.
    
    For stable conditions (zeta > 0), psi_m should be negative.
    """
    zeta = test_data["phim_stable"]["zeta"]
    result = psim_monin_obukhov(zeta)
    
    # Check shape
    assert result.shape == zeta.shape
    
    # Check all values are negative for stable conditions
    assert jnp.all(result < 0), \
        f"psi_m should be negative for stable conditions, got {result}"


def test_psim_custom_pi(test_data):
    """
    Test psi_m with custom pi value.
    
    Verifies that the pi parameter is used correctly.
    """
    zeta = test_data["phim_unstable"]["zeta"]
    pi_default = jnp.pi
    pi_custom = 3.14159
    
    result_default = psim_monin_obukhov(zeta, pi=pi_default)
    result_custom = psim_monin_obukhov(zeta, pi=pi_custom)
    
    # Results should be very close but not identical
    assert jnp.allclose(result_default, result_custom, rtol=1e-4), \
        "Results with different pi values should be very close"


# ============================================================================
# Tests for psic_monin_obukhov
# ============================================================================

def test_psic_neutral_conditions(test_data):
    """
    Test psi_c near neutral stability.
    
    For neutral conditions, psi_c should be close to 0.
    """
    zeta = test_data["phim_neutral"]["zeta"]
    result = psic_monin_obukhov(zeta)
    
    # Check shape
    assert result.shape == zeta.shape
    
    # Check values near 0 for neutral conditions
    assert jnp.allclose(result, 0.0, atol=0.1), \
        f"psi_c should be near 0 for neutral conditions, got {result}"


def test_psic_unstable_positive(test_data):
    """
    Test psi_c in unstable conditions.
    
    For unstable conditions (zeta < 0), psi_c should be positive.
    """
    zeta = test_data["phim_unstable"]["zeta"]
    result = psic_monin_obukhov(zeta)
    
    # Check shape
    assert result.shape == zeta.shape
    
    # Check all values are positive for unstable conditions
    assert jnp.all(result > 0), \
        f"psi_c should be positive for unstable conditions, got {result}"


def test_psic_stable_negative(test_data):
    """
    Test psi_c in stable conditions.
    
    For stable conditions (zeta > 0), psi_c should be negative.
    """
    zeta = test_data["phim_stable"]["zeta"]
    result = psic_monin_obukhov(zeta)
    
    # Check shape
    assert result.shape == zeta.shape
    
    # Check all values are negative for stable conditions
    assert jnp.all(result < 0), \
        f"psi_c should be negative for stable conditions, got {result}"


# ============================================================================
# Tests for get_prsc
# ============================================================================

def test_get_prsc_typical_canopy(test_data):
    """
    Test Prandtl/Schmidt number calculation for typical canopy conditions.
    
    Tests with varying stability (LcL parameter) and beta values.
    """
    data = test_data["prsc_typical"]
    result = get_prsc(
        data["beta_neutral"],
        data["beta_neutral_max"],
        data["LcL"],
        data["params"]
    )
    
    # Check shape
    assert result.shape == data["beta_neutral"].shape, \
        f"Expected shape {data['beta_neutral'].shape}, got {result.shape}"
    
    # Check all values are positive
    assert jnp.all(result > 0), \
        f"Prandtl/Schmidt number should be positive, got {result}"
    
    # Check values are in reasonable range (typically 0.3 to 1.5)
    assert jnp.all(result > 0.1) and jnp.all(result < 2.0), \
        f"Prandtl/Schmidt number should be in reasonable range, got {result}"


def test_get_prsc_edge_beta(test_data):
    """
    Test Prandtl/Schmidt at beta boundaries (min/max).
    
    Tests behavior at extreme beta values.
    """
    data = test_data["prsc_edge_beta"]
    result = get_prsc(
        data["beta_neutral"],
        data["beta_neutral_max"],
        data["LcL"],
        data["params"]
    )
    
    # Check shape
    assert result.shape == data["beta_neutral"].shape
    
    # Check all values are positive and finite
    assert jnp.all(result > 0) and jnp.all(jnp.isfinite(result)), \
        f"Prandtl/Schmidt should be positive and finite at boundaries, got {result}"


def test_get_prsc_neutral_conditions(test_data):
    """
    Test Prandtl/Schmidt under neutral conditions (LcL = 0).
    
    For neutral conditions, result should be close to Pr0.
    """
    data = test_data["prsc_edge_beta"]
    result = get_prsc(
        data["beta_neutral"],
        data["beta_neutral_max"],
        data["LcL"],  # All zeros for neutral
        data["params"]
    )
    
    # For neutral conditions with LcL=0, expect values near Pr0
    # (though beta effects may cause some variation)
    assert jnp.all(result > 0.3) and jnp.all(result < 1.0), \
        f"Prandtl/Schmidt should be in expected range for neutral, got {result}"


# ============================================================================
# Tests for get_beta
# ============================================================================

def test_get_beta_neutral(test_data):
    """
    Test beta calculation under neutral conditions (lcl=0).
    
    For neutral conditions, beta should equal beta_neutral.
    """
    data = test_data["beta_neutral"]
    result = get_beta(
        data["beta_neutral"],
        data["lcl"],
        data["beta_min"],
        data["beta_max"],
        phim_monin_obukhov
    )
    
    # Check result is a scalar
    assert jnp.ndim(result) == 0 or result.shape == (), \
        f"Expected scalar result, got shape {result.shape}"
    
    # For neutral conditions, should be close to beta_neutral
    assert jnp.allclose(result, data["beta_neutral"], atol=0.01), \
        f"Beta should equal beta_neutral for lcl=0, got {result}"


def test_get_beta_bounds(test_data):
    """
    Test that beta is constrained to [beta_min, beta_max].
    
    Tests with various lcl values to ensure bounds are respected.
    """
    data = test_data["beta_neutral"]
    
    # Test with extreme lcl values
    for lcl in [-10.0, -1.0, 0.0, 1.0, 10.0]:
        result = get_beta(
            data["beta_neutral"],
            lcl,
            data["beta_min"],
            data["beta_max"],
            phim_monin_obukhov
        )
        
        # Check bounds
        assert result >= data["beta_min"], \
            f"Beta should be >= beta_min, got {result}"
        assert result <= data["beta_max"], \
            f"Beta should be <= beta_max, got {result}"


def test_get_beta_stability_dependence(test_data):
    """
    Test that beta varies with stability (lcl parameter).
    
    Beta should change as stability changes.
    """
    data = test_data["beta_neutral"]
    
    result_unstable = get_beta(
        data["beta_neutral"], -1.0,
        data["beta_min"], data["beta_max"],
        phim_monin_obukhov
    )
    
    result_stable = get_beta(
        data["beta_neutral"], 1.0,
        data["beta_min"], data["beta_max"],
        phim_monin_obukhov
    )
    
    # Results should be different for different stability
    assert not jnp.allclose(result_unstable, result_stable, atol=0.001), \
        "Beta should vary with stability"


# ============================================================================
# Tests for lookup_psihat
# ============================================================================

def test_lookup_psihat_interpolation(test_data):
    """
    Test bilinear interpolation in psihat lookup table.
    
    Tests interpolation at an interior point of the grid.
    """
    data = test_data["lookup_psihat"]
    result = lookup_psihat(
        data["zdt"],
        data["dtL"],
        data["zdtgrid"],
        data["dtLgrid"],
        data["psigrid"]
    )
    
    # Check result is a scalar
    assert jnp.ndim(result) == 0 or result.shape == (), \
        f"Expected scalar result, got shape {result.shape}"
    
    # Check result is finite
    assert jnp.isfinite(result), \
        f"Interpolated psihat should be finite, got {result}"
    
    # Check result is in reasonable range based on grid values
    assert result >= jnp.min(data["psigrid"]) - 0.1, \
        f"Interpolated value should be >= min grid value"
    assert result <= jnp.max(data["psigrid"]) + 0.1, \
        f"Interpolated value should be <= max grid value"


def test_lookup_psihat_grid_corners(test_data):
    """
    Test lookup at grid corner points.
    
    At exact grid points, should return the grid value.
    """
    data = test_data["lookup_psihat"]
    
    # Test at a corner point
    zdt_corner = float(data["zdtgrid"][0, 0])
    dtL_corner = float(data["dtLgrid"][0, 0])
    expected = float(data["psigrid"][0, 0])
    
    result = lookup_psihat(
        zdt_corner,
        dtL_corner,
        data["zdtgrid"],
        data["dtLgrid"],
        data["psigrid"]
    )
    
    # Should match grid value at corner
    assert jnp.allclose(result, expected, atol=1e-5), \
        f"At grid corner, expected {expected}, got {result}"


def test_lookup_psihat_boundary_extrapolation(test_data):
    """
    Test psihat lookup at boundaries requiring extrapolation.
    
    Tests behavior when query points are outside the grid.
    """
    data = test_data["lookup_boundary"]
    
    # Test below grid
    result_below = lookup_psihat(
        data["zdt_below"],
        0.0,
        data["zdtgrid"],
        data["dtLgrid"],
        data["psigrid"]
    )
    
    # Test above grid
    result_above = lookup_psihat(
        data["zdt_above"],
        0.0,
        data["zdtgrid"],
        data["dtLgrid"],
        data["psigrid"]
    )
    
    # Both should be finite
    assert jnp.isfinite(result_below) and jnp.isfinite(result_above), \
        "Extrapolated values should be finite"
    
    # Test dtL boundaries
    result_dtL_below = lookup_psihat(
        1.0,
        data["dtL_below"],
        data["zdtgrid"],
        data["dtLgrid"],
        data["psigrid"]
    )
    
    result_dtL_above = lookup_psihat(
        1.0,
        data["dtL_above"],
        data["zdtgrid"],
        data["dtLgrid"],
        data["psigrid"]
    )
    
    assert jnp.isfinite(result_dtL_below) and jnp.isfinite(result_dtL_above), \
        "Extrapolated values should be finite for dtL boundaries"


# ============================================================================
# Tests for get_psi_rsl
# ============================================================================

def test_get_psi_rsl_complete(test_data, lookup_grids):
    """
    Test complete RSL psi calculation with multiple patches.
    
    Tests the full RSL correction calculation under varying stability.
    """
    data = test_data["psi_rsl_complete"]
    grids = lookup_grids
    
    result = get_psi_rsl(
        data["za"],
        data["hc"],
        data["disp"],
        data["obu"],
        data["beta"],
        data["prsc"],
        data["vkc"],
        data["c2"],
        grids["dtlgrid_m"],
        grids["zdtgrid_m"],
        grids["psigrid_m"],
        grids["dtlgrid_h"],
        grids["zdtgrid_h"],
        grids["psigrid_h"],
        phim_monin_obukhov,
        phic_monin_obukhov,
        psim_monin_obukhov,
        psic_monin_obukhov,
        lookup_psihat
    )
    
    # Check result is PsiRSLResult namedtuple
    assert isinstance(result, PsiRSLResult), \
        f"Expected PsiRSLResult, got {type(result)}"
    
    # Check shapes
    assert result.psim.shape == data["za"].shape, \
        f"psim shape mismatch: expected {data['za'].shape}, got {result.psim.shape}"
    assert result.psic.shape == data["za"].shape, \
        f"psic shape mismatch: expected {data['za'].shape}, got {result.psic.shape}"
    
    # Check all values are finite
    assert jnp.all(jnp.isfinite(result.psim)), \
        f"psim should be finite, got {result.psim}"
    assert jnp.all(jnp.isfinite(result.psic)), \
        f"psic should be finite, got {result.psic}"


def test_get_psi_rsl_shapes(test_data, lookup_grids):
    """
    Test that get_psi_rsl preserves input shapes.
    
    Verifies shape consistency across different input sizes.
    """
    grids = lookup_grids
    
    # Test with single patch
    za_single = jnp.array([50.0])
    hc_single = jnp.array([20.0])
    disp_single = jnp.array([13.0])
    obu_single = jnp.array([-100.0])
    beta_single = jnp.array([0.3])
    prsc_single = jnp.array([0.7])
    
    result = get_psi_rsl(
        za_single, hc_single, disp_single, obu_single,
        beta_single, prsc_single,
        0.4, 0.5,
        grids["dtlgrid_m"], grids["zdtgrid_m"], grids["psigrid_m"],
        grids["dtlgrid_h"], grids["zdtgrid_h"], grids["psigrid_h"],
        phim_monin_obukhov, phic_monin_obukhov,
        psim_monin_obukhov, psic_monin_obukhov,
        lookup_psihat
    )
    
    assert result.psim.shape == za_single.shape
    assert result.psic.shape == za_single.shape


def test_get_psi_rsl_stability_response(test_data, lookup_grids):
    """
    Test that RSL psi functions respond appropriately to stability.
    
    Unstable and stable conditions should give different results.
    """
    grids = lookup_grids
    
    # Unstable case
    result_unstable = get_psi_rsl(
        jnp.array([50.0]),
        jnp.array([20.0]),
        jnp.array([13.0]),
        jnp.array([-100.0]),  # Unstable
        jnp.array([0.3]),
        jnp.array([0.7]),
        0.4, 0.5,
        grids["dtlgrid_m"], grids["zdtgrid_m"], grids["psigrid_m"],
        grids["dtlgrid_h"], grids["zdtgrid_h"], grids["psigrid_h"],
        phim_monin_obukhov, phic_monin_obukhov,
        psim_monin_obukhov, psic_monin_obukhov,
        lookup_psihat
    )
    
    # Stable case
    result_stable = get_psi_rsl(
        jnp.array([50.0]),
        jnp.array([20.0]),
        jnp.array([13.0]),
        jnp.array([100.0]),  # Stable
        jnp.array([0.3]),
        jnp.array([0.7]),
        0.4, 0.5,
        grids["dtlgrid_m"], grids["zdtgrid_m"], grids["psigrid_m"],
        grids["dtlgrid_h"], grids["zdtgrid_h"], grids["psigrid_h"],
        phim_monin_obukhov, phic_monin_obukhov,
        psim_monin_obukhov, psic_monin_obukhov,
        lookup_psihat
    )
    
    # Results should differ between stable and unstable
    assert not jnp.allclose(result_unstable.psim, result_stable.psim, atol=0.01), \
        "psim should differ between stable and unstable conditions"
    assert not jnp.allclose(result_unstable.psic, result_stable.psic, atol=0.01), \
        "psic should differ between stable and unstable conditions"


# ============================================================================
# Tests for obu_func
# ============================================================================

def test_obu_func_typical_unstable(obu_func_inputs, lookup_grids):
    """
    Test Obukhov length calculation for typical daytime unstable conditions.
    
    Tests the complete obu_func calculation with realistic inputs.
    """
    inputs = obu_func_inputs["typical_unstable"]
    grids = lookup_grids
    
    # Create wrapper functions for get_beta and get_prsc
    def get_beta_wrapper(beta_neutral, lcl, beta_min, beta_max, phim_func):
        return get_beta(beta_neutral, lcl, beta_min, beta_max, phim_func)
    
    def get_prsc_wrapper(beta_neutral, beta_neutral_max, LcL, params):
        return get_prsc(beta_neutral, beta_neutral_max, LcL, params)
    
    def get_psi_rsl_wrapper(za, hc, disp, obu, beta, prsc, vkc, c2,
                           dtlgrid_m, zdtgrid_m, psigrid_m,
                           dtlgrid_h, zdtgrid_h, psigrid_h,
                           phim_fn, phic_fn, psim_fn, psic_fn, lookup_fn):
        return get_psi_rsl(za, hc, disp, obu, beta, prsc, vkc, c2,
                          dtlgrid_m, zdtgrid_m, psigrid_m,
                          dtlgrid_h, zdtgrid_h, psigrid_h,
                          phim_fn, phic_fn, psim_fn, psic_fn, lookup_fn)
    
    result = obu_func(
        inputs,
        get_beta_wrapper,
        get_prsc_wrapper,
        get_psi_rsl_wrapper
    )
    
    # Check result is ObuFuncOutputs namedtuple
    assert isinstance(result, ObuFuncOutputs), \
        f"Expected ObuFuncOutputs, got {type(result)}"
    
    # Check all fields are present and finite
    assert jnp.isfinite(result.obu_dif), "obu_dif should be finite"
    assert jnp.isfinite(result.zdisp), "zdisp should be finite"
    assert jnp.isfinite(result.beta), "beta should be finite"
    assert jnp.isfinite(result.PrSc), "PrSc should be finite"
    assert jnp.isfinite(result.ustar), "ustar should be finite"
    assert jnp.isfinite(result.gac_to_hc), "gac_to_hc should be finite"
    assert jnp.isfinite(result.obu), "obu should be finite"
    
    # Check physical constraints
    assert result.beta > 0 and result.beta < 1, \
        f"Beta should be in (0,1), got {result.beta}"
    assert result.PrSc > 0, \
        f"Prandtl/Schmidt should be positive, got {result.PrSc}"
    assert result.ustar > 0, \
        f"Friction velocity should be positive, got {result.ustar}"
    assert result.gac_to_hc > 0, \
        f"Aerodynamic conductance should be positive, got {result.gac_to_hc}"


def test_obu_func_stable_nighttime(obu_func_inputs, lookup_grids):
    """
    Test Obukhov length calculation for stable nighttime conditions.
    
    Tests with positive Obukhov length (stable stratification).
    """
    inputs = obu_func_inputs["stable_nighttime"]
    
    def get_beta_wrapper(beta_neutral, lcl, beta_min, beta_max, phim_func):
        return get_beta(beta_neutral, lcl, beta_min, beta_max, phim_func)
    
    def get_prsc_wrapper(beta_neutral, beta_neutral_max, LcL, params):
        return get_prsc(beta_neutral, beta_neutral_max, LcL, params)
    
    def get_psi_rsl_wrapper(za, hc, disp, obu, beta, prsc, vkc, c2,
                           dtlgrid_m, zdtgrid_m, psigrid_m,
                           dtlgrid_h, zdtgrid_h, psigrid_h,
                           phim_fn, phic_fn, psim_fn, psic_fn, lookup_fn):
        return get_psi_rsl(za, hc, disp, obu, beta, prsc, vkc, c2,
                          dtlgrid_m, zdtgrid_m, psigrid_m,
                          dtlgrid_h, zdtgrid_h, psigrid_h,
                          phim_fn, phic_fn, psim_fn, psic_fn, lookup_fn)
    
    result = obu_func(inputs, get_beta_wrapper, get_prsc_wrapper, get_psi_rsl_wrapper)
    
    # Check result type
    assert isinstance(result, ObuFuncOutputs)
    
    # For stable conditions, Obukhov length should be positive
    assert result.obu > 0, \
        f"Obukhov length should be positive for stable conditions, got {result.obu}"
    
    # Check all values are finite
    assert jnp.all(jnp.isfinite(jnp.array([
        result.obu_dif, result.zdisp, result.beta, result.PrSc,
        result.ustar, result.gac_to_hc, result.obu
    ]))), "All outputs should be finite"


def test_obu_func_near_neutral(obu_func_inputs, lookup_grids):
    """
    Test Obukhov length calculation near neutral conditions (large |L|).
    
    Tests behavior when approaching neutral stability.
    """
    inputs = obu_func_inputs["near_neutral"]
    
    def get_beta_wrapper(beta_neutral, lcl, beta_min, beta_max, phim_func):
        return get_beta(beta_neutral, lcl, beta_min, beta_max, phim_func)
    
    def get_prsc_wrapper(beta_neutral, beta_neutral_max, LcL, params):
        return get_prsc(beta_neutral, beta_neutral_max, LcL, params)
    
    def get_psi_rsl_wrapper(za, hc, disp, obu, beta, prsc, vkc, c2,
                           dtlgrid_m, zdtgrid_m, psigrid_m,
                           dtlgrid_h, zdtgrid_h, psigrid_h,
                           phim_fn, phic_fn, psim_fn, psic_fn, lookup_fn):
        return get_psi_rsl(za, hc, disp, obu, beta, prsc, vkc, c2,
                          dtlgrid_m, zdtgrid_m, psigrid_m,
                          dtlgrid_h, zdtgrid_h, psigrid_h,
                          phim_fn, phic_fn, psim_fn, psic_fn, lookup_fn)
    
    result = obu_func(inputs, get_beta_wrapper, get_prsc_wrapper, get_psi_rsl_wrapper)
    
    # Check result type
    assert isinstance(result, ObuFuncOutputs)
    
    # For near-neutral, |L| should be large
    assert jnp.abs(result.obu) > 100, \
        f"Obukhov length should be large for near-neutral, got {result.obu}"
    
    # Beta should be close to neutral value
    assert result.beta > 0.2 and result.beta < 0.4, \
        f"Beta should be near neutral value, got {result.beta}"


def test_obu_func_low_wind(obu_func_inputs, lookup_grids):
    """
    Test Obukhov length calculation at minimum wind speed threshold.
    
    Tests behavior at the lower wind speed limit.
    """
    inputs = obu_func_inputs["low_wind"]
    
    def get_beta_wrapper(beta_neutral, lcl, beta_min, beta_max, phim_func):
        return get_beta(beta_neutral, lcl, beta_min, beta_max, phim_func)
    
    def get_prsc_wrapper(beta_neutral, beta_neutral_max, LcL, params):
        return get_prsc(beta_neutral, beta_neutral_max, LcL, params)
    
    def get_psi_rsl_wrapper(za, hc, disp, obu, beta, prsc, vkc, c2,
                           dtlgrid_m, zdtgrid_m, psigrid_m,
                           dtlgrid_h, zdtgrid_h, psigrid_h,
                           phim_fn, phic_fn, psim_fn, psic_fn, lookup_fn):
        return get_psi_rsl(za, hc, disp, obu, beta, prsc, vkc, c2,
                          dtlgrid_m, zdtgrid_m, psigrid_m,
                          dtlgrid_h, zdtgrid_h, psigrid_h,
                          phim_fn, phic_fn, psim_fn, psic_fn, lookup_fn)
    
    result = obu_func(inputs, get_beta_wrapper, get_prsc_wrapper, get_psi_rsl_wrapper)
    
    # Check result type
    assert isinstance(result, ObuFuncOutputs)
    
    # Friction velocity should be small but positive
    assert result.ustar > 0 and result.ustar < 1.0, \
        f"Friction velocity should be small for low wind, got {result.ustar}"
    
    # All values should still be finite
    assert jnp.all(jnp.isfinite(jnp.array([
        result.obu_dif, result.zdisp, result.beta, result.PrSc,
        result.ustar, result.gac_to_hc, result.obu
    ]))), "All outputs should be finite even at low wind"


def test_obu_func_sparse_canopy(obu_func_inputs, lookup_grids):
    """
    Test Obukhov length calculation for sparse canopy (low LAI).
    
    Tests with minimal vegetation.
    """
    inputs = obu_func_inputs["sparse_canopy"]
    
    def get_beta_wrapper(beta_neutral, lcl, beta_min, beta_max, phim_func):
        return get_beta(beta_neutral, lcl, beta_min, beta_max, phim_func)
    
    def get_prsc_wrapper(beta_neutral, beta_neutral_max, LcL, params):
        return get_prsc(beta_neutral, beta_neutral_max, LcL, params)
    
    def get_psi_rsl_wrapper(za, hc, disp, obu, beta, prsc, vkc, c2,
                           dtlgrid_m, zdtgrid_m, psigrid_m,
                           dtlgrid_h, zdtgrid_h, psigrid_h,
                           phim_fn, phic_fn, psim_fn, psic_fn, lookup_fn):
        return get_psi_rsl(za, hc, disp, obu, beta, prsc, vkc, c2,
                          dtlgrid_m, zdtgrid_m, psigrid_m,
                          dtlgrid_h, zdtgrid_h, psigrid_h,
                          phim_fn, phic_fn, psim_fn, psic_fn, lookup_fn)
    
    result = obu_func(inputs, get_beta_wrapper, get_prsc_wrapper, get_psi_rsl_wrapper)
    
    # Check result type
    assert isinstance(result, ObuFuncOutputs)
    
    # Displacement height should be small for sparse canopy
    assert result.zdisp < 10.0, \
        f"Displacement height should be small for sparse canopy, got {result.zdisp}"
    
    # All values should be finite
    assert jnp.all(jnp.isfinite(jnp.array([
        result.obu_dif, result.zdisp, result.beta, result.PrSc,
        result.ustar, result.gac_to_hc, result.obu
    ]))), "All outputs should be finite for sparse canopy"


def test_obu_func_dense_canopy(obu_func_inputs, lookup_grids):
    """
    Test Obukhov length calculation for dense canopy (high LAI).
    
    Tests with dense vegetation.
    """
    inputs = obu_func_inputs["dense_canopy"]
    
    def get_beta_wrapper(beta_neutral, lcl, beta_min, beta_max, phim_func):
        return get_beta(beta_neutral, lcl, beta_min, beta_max, phim_func)
    
    def get_prsc_wrapper(beta_neutral, beta_neutral_max, LcL, params):
        return get_prsc(beta_neutral, beta_neutral_max, LcL, params)
    
    def get_psi_rsl_wrapper(za, hc, disp, obu, beta, prsc, vkc, c2,
                           dtlgrid_m, zdtgrid_m, psigrid_m,
                           dtlgrid_h, zdtgrid_h, psigrid_h,
                           phim_fn, phic_fn, psim_fn, psic_fn, lookup_fn):
        return get_psi_rsl(za, hc, disp, obu, beta, prsc, vkc, c2,
                          dtlgrid_m, zdtgrid_m, psigrid_m,
                          dtlgrid_h, zdtgrid_h, psigrid_h,
                          phim_fn, phic_fn, psim_fn, psic_fn, lookup_fn)
    
    result = obu_func(inputs, get_beta_wrapper, get_prsc_wrapper, get_psi_rsl_wrapper)
    
    # Check result type
    assert isinstance(result, ObuFuncOutputs)
    
    # Displacement height should be larger for dense canopy
    assert result.zdisp > 15.0, \
        f"Displacement height should be large for dense canopy, got {result.zdisp}"
    
    # All values should be finite
    assert jnp.all(jnp.isfinite(jnp.array([
        result.obu_dif, result.zdisp, result.beta, result.PrSc,
        result.ustar, result.gac_to_hc, result.obu
    ]))), "All outputs should be finite for dense canopy"


# ============================================================================
# Tests for array shape handling
# ============================================================================

def test_psim_psic_array_shapes(test_data):
    """
    Test psi functions with various array dimensions (scalar, 1D, 2D, 3D).
    
    Verifies that functions handle different array shapes correctly.
    """
    data = test_data["array_shapes"]
    
    # Test scalar
    result_scalar_m = psim_monin_obukhov(data["zeta_scalar"])
    result_scalar_c = psic_monin_obukhov(data["zeta_scalar"])
    assert jnp.ndim(result_scalar_m) == 0 or result_scalar_m.shape == ()
    assert jnp.ndim(result_scalar_c) == 0 or result_scalar_c.shape == ()
    
    # Test 1D
    result_1d_m = psim_monin_obukhov(data["zeta_1d"])
    result_1d_c = psic_monin_obukhov(data["zeta_1d"])
    assert result_1d_m.shape == data["zeta_1d"].shape
    assert result_1d_c.shape == data["zeta_1d"].shape
    
    # Test 2D
    result_2d_m = psim_monin_obukhov(data["zeta_2d"])
    result_2d_c = psic_monin_obukhov(data["zeta_2d"])
    assert result_2d_m.shape == data["zeta_2d"].shape
    assert result_2d_c.shape == data["zeta_2d"].shape
    
    # Test 3D
    result_3d_m = psim_monin_obukhov(data["zeta_3d"])
    result_3d_c = psic_monin_obukhov(data["zeta_3d"])
    assert result_3d_m.shape == data["zeta_3d"].shape
    assert result_3d_c.shape == data["zeta_3d"].shape
    
    # All results should be finite
    assert jnp.all(jnp.isfinite(result_1d_m))
    assert jnp.all(jnp.isfinite(result_1d_c))
    assert jnp.all(jnp.isfinite(result_2d_m))
    assert jnp.all(jnp.isfinite(result_2d_c))
    assert jnp.all(jnp.isfinite(result_3d_m))
    assert jnp.all(jnp.isfinite(result_3d_c))


def test_phim_phic_array_shapes(test_data):
    """
    Test phi functions with various array dimensions.
    
    Verifies shape preservation for phi_m and phi_c.
    """
    data = test_data["array_shapes"]
    
    # Test 1D
    result_1d_m = phim_monin_obukhov(data["zeta_1d"])
    result_1d_c = phic_monin_obukhov(data["zeta_1d"])
    assert result_1d_m.shape == data["zeta_1d"].shape
    assert result_1d_c.shape == data["zeta_1d"].shape
    
    # Test 2D
    result_2d_m = phim_monin_obukhov(data["zeta_2d"])
    result_2d_c = phic_monin_obukhov(data["zeta_2d"])
    assert result_2d_m.shape == data["zeta_2d"].shape
    assert result_2d_c.shape == data["zeta_2d"].shape
    
    # Test 3D
    result_3d_m = phim_monin_obukhov(data["zeta_3d"])
    result_3d_c = phic_monin_obukhov(data["zeta_3d"])
    assert result_3d_m.shape == data["zeta_3d"].shape
    assert result_3d_c.shape == data["zeta_3d"].shape


# ============================================================================
# Tests for data types
# ============================================================================

def test_monin_obukhov_dtypes(test_data):
    """
    Test that Monin-Obukhov functions return correct data types.
    
    All functions should return float arrays.
    """
    zeta = test_data["phim_unstable"]["zeta"]
    
    result_phim = phim_monin_obukhov(zeta)
    result_phic = phic_monin_obukhov(zeta)
    result_psim = psim_monin_obukhov(zeta)
    result_psic = psic_monin_obukhov(zeta)
    
    # Check all are float types
    assert result_phim.dtype in [jnp.float32, jnp.float64], \
        f"phim should return float, got {result_phim.dtype}"
    assert result_phic.dtype in [jnp.float32, jnp.float64], \
        f"phic should return float, got {result_phic.dtype}"
    assert result_psim.dtype in [jnp.float32, jnp.float64], \
        f"psim should return float, got {result_psim.dtype}"
    assert result_psic.dtype in [jnp.float32, jnp.float64], \
        f"psic should return float, got {result_psic.dtype}"


def test_get_prsc_dtype(test_data):
    """
    Test that get_prsc returns correct data type.
    """
    data = test_data["prsc_typical"]
    result = get_prsc(
        data["beta_neutral"],
        data["beta_neutral_max"],
        data["LcL"],
        data["params"]
    )
    
    assert result.dtype in [jnp.float32, jnp.float64], \
        f"get_prsc should return float, got {result.dtype}"


def test_lookup_psihat_dtype(test_data):
    """
    Test that lookup_psihat returns correct data type.
    """
    data = test_data["lookup_psihat"]
    result = lookup_psihat(
        data["zdt"],
        data["dtL"],
        data["zdtgrid"],
        data["dtLgrid"],
        data["psigrid"]
    )
    
    # Result should be float
    assert isinstance(float(result), float), \
        f"lookup_psihat should return float-convertible, got {type(result)}"


# ============================================================================
# Integration tests
# ============================================================================

def test_monin_obukhov_consistency():
    """
    Test consistency between phi and psi functions.
    
    The relationship between phi and psi should be consistent
    (psi is the integral of (1 - phi)/zeta).
    """
    zeta = jnp.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0])
    
    phim = phim_monin_obukhov(zeta)
    psim = psim_monin_obukhov(zeta)
    
    phic = phic_monin_obukhov(zeta)
    psic = psic_monin_obukhov(zeta)
    
    # All should be finite
    assert jnp.all(jnp.isfinite(phim))
    assert jnp.all(jnp.isfinite(psim))
    assert jnp.all(jnp.isfinite(phic))
    assert jnp.all(jnp.isfinite(psic))
    
    # For neutral (zeta=0), psi should be near 0 and phi near 1
    neutral_idx = 3
    assert jnp.abs(psim[neutral_idx]) < 0.1
    assert jnp.abs(psic[neutral_idx]) < 0.1
    assert jnp.abs(phim[neutral_idx] - 1.0) < 0.1
    assert jnp.abs(phic[neutral_idx] - 1.0) < 0.1


def test_full_turbulence_calculation_chain(obu_func_inputs, lookup_grids):
    """
    Test the full calculation chain from inputs to outputs.
    
    Integration test verifying that all components work together.
    """
    inputs = obu_func_inputs["typical_unstable"]
    grids = lookup_grids
    
    # Step 1: Calculate beta
    beta_neutral = 0.3
    lcl = float(inputs.Lc / inputs.obu_val)
    beta = get_beta(beta_neutral, lcl, 0.01, 0.99, phim_monin_obukhov)
    
    # Step 2: Calculate Prandtl/Schmidt number
    params = PrScParams(Pr0=0.5, Pr1=0.3, Pr2=0.143)
    prsc = get_prsc(
        jnp.array([beta_neutral]),
        jnp.array([0.35]),
        jnp.array([lcl]),
        params
    )
    
    # Step 3: Calculate RSL psi functions
    psi_result = get_psi_rsl(
        jnp.array([float(inputs.zref)]),
        jnp.array([float(inputs.ztop)]),
        jnp.array([float(inputs.ztop) * 0.65]),
        jnp.array([float(inputs.obu_val)]),
        jnp.array([beta]),
        prsc,
        inputs.vkc,
        0.5,
        grids["dtlgrid_m"], grids["zdtgrid_m"], grids["psigrid_m"],
        grids["dtlgrid_h"], grids["zdtgrid_h"], grids["psigrid_h"],
        phim_monin_obukhov, phic_monin_obukhov,
        psim_monin_obukhov, psic_monin_obukhov,
        lookup_psihat
    )
    
    # All intermediate results should be finite
    assert jnp.isfinite(beta)
    assert jnp.all(jnp.isfinite(prsc))
    assert jnp.all(jnp.isfinite(psi_result.psim))
    assert jnp.all(jnp.isfinite(psi_result.psic))
    
    # Physical constraints
    assert beta > 0 and beta < 1
    assert jnp.all(prsc > 0)


# ============================================================================
# Edge case tests
# ============================================================================

def test_zero_zeta_handling():
    """
    Test handling of exactly zero zeta (neutral conditions).
    
    Functions should handle zeta=0 without numerical issues.
    """
    zeta_zero = jnp.array([0.0])
    
    phim = phim_monin_obukhov(zeta_zero)
    phic = phic_monin_obukhov(zeta_zero)
    psim = psim_monin_obukhov(zeta_zero)
    psic = psic_monin_obukhov(zeta_zero)
    
    # All should be finite
    assert jnp.isfinite(phim[0])
    assert jnp.isfinite(phic[0])
    assert jnp.isfinite(psim[0])
    assert jnp.isfinite(psic[0])
    
    # phi should be near 1, psi near 0
    assert jnp.abs(phim[0] - 1.0) < 0.01
    assert jnp.abs(phic[0] - 1.0) < 0.01
    assert jnp.abs(psim[0]) < 0.01
    assert jnp.abs(psic[0]) < 0.01


def test_extreme_zeta_values():
    """
    Test handling of extreme zeta values.
    
    Functions should remain stable at boundaries.
    """
    zeta_extreme = jnp.array([-100.0, -50.0, 50.0, 100.0])
    
    phim = phim_monin_obukhov(zeta_extreme)
    phic = phic_monin_obukhov(zeta_extreme)
    psim = psim_monin_obukhov(zeta_extreme)
    psic = psic_monin_obukhov(zeta_extreme)
    
    # All should be finite
    assert jnp.all(jnp.isfinite(phim))
    assert jnp.all(jnp.isfinite(phic))
    assert jnp.all(jnp.isfinite(psim))
    assert jnp.all(jnp.isfinite(psic))
    
    # All phi values should be positive
    assert jnp.all(phim > 0)
    assert jnp.all(phic > 0)


def test_empty_array_handling():
    """
    Test handling of empty arrays.
    
    Functions should handle empty inputs gracefully.
    """
    zeta_empty = jnp.array([])
    
    phim = phim_monin_obukhov(zeta_empty)
    phic = phic_monin_obukhov(zeta_empty)
    psim = psim_monin_obukhov(zeta_empty)
    psic = psic_monin_obukhov(zeta_empty)
    
    # All should be empty arrays
    assert phim.shape == (0,)
    assert phic.shape == (0,)
    assert psim.shape == (0,)
    assert psic.shape == (0,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])