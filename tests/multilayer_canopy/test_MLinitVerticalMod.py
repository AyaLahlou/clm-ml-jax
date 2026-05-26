"""
Comprehensive pytest suite for MLinitVerticalMod module.

This module tests the vertical structure initialization functions for multilayer
canopy models, including:
- Beta distribution CDF calculations
- Vertical layer structure calculation
- Layer property distribution
- Plant area redistribution
- Vertical structure finalization
- Complete initialization workflows

Tests cover nominal cases, edge cases, and physical realism constraints.
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multilayer_canopy.MLinitVerticalMod import (
    beta_distribution_cdf,
    calculate_layer_properties,
    calculate_vertical_structure,
    finalize_vertical_structure,
    init_vertical_profiles,
    init_vertical_structure,
    redistribute_plant_area,
)


# NamedTuple definitions for type safety
class BoundsType(NamedTuple):
    """Spatial bounds for patches, columns, and gridcells."""
    begp: int
    endp: int
    begc: int
    endc: int
    begg: int
    endg: int


class PatchState(NamedTuple):
    """Patch hierarchy information."""
    column: jnp.ndarray
    gridcell: jnp.ndarray
    itype: jnp.ndarray


class Atm2LndState(NamedTuple):
    """Atmospheric forcing state."""
    forc_u_grc: jnp.ndarray
    forc_v_grc: jnp.ndarray
    forc_pco2_grc: jnp.ndarray
    forc_t_downscaled_col: jnp.ndarray
    forc_q_downscaled_col: jnp.ndarray
    forc_pbot_downscaled_col: jnp.ndarray
    forc_hgt_u: jnp.ndarray


class CanopyStateType(NamedTuple):
    """Canopy state variables."""
    htop: jnp.ndarray
    elai: jnp.ndarray
    esai: jnp.ndarray


class MLCanopyType(NamedTuple):
    """Multilayer canopy state."""
    ncan: jnp.ndarray
    ntop: jnp.ndarray
    nbot: jnp.ndarray
    zref: jnp.ndarray
    ztop: jnp.ndarray
    zbot: jnp.ndarray
    zs: jnp.ndarray
    dz: jnp.ndarray
    zw: jnp.ndarray
    dlai: jnp.ndarray
    dsai: jnp.ndarray
    dlai_frac: jnp.ndarray
    dsai_frac: jnp.ndarray
    wind: jnp.ndarray
    tair: jnp.ndarray
    eair: jnp.ndarray
    cair: jnp.ndarray
    h2ocan: jnp.ndarray
    taf: jnp.ndarray
    qaf: jnp.ndarray
    tg: jnp.ndarray
    tleaf: jnp.ndarray
    lwp: jnp.ndarray


class PFTParams(NamedTuple):
    """PFT-specific parameters."""
    pbeta_lai: jnp.ndarray
    qbeta_lai: jnp.ndarray
    pbeta_sai: jnp.ndarray
    qbeta_sai: jnp.ndarray


class MLCanopyParams(NamedTuple):
    """Multilayer canopy parameters."""
    dz_tall: float = 1.0
    dz_short: float = 0.5
    dz_param: float = 10.0
    nlayer_within: int = 0
    nlayer_above: int = 0
    dpai_min: float = 0.01
    mmh2o: float = 18.016
    mmdry: float = 28.966
    wind_forc_min: float = 0.1
    isun: int = 0
    isha: int = 1
    lai_tol: float = 1e-06
    sai_tol: float = 1e-06


@pytest.fixture
def test_data():
    """
    Load test data for all functions.
    
    Returns:
        dict: Test cases organized by function and test type.
    """
    return {
        "beta_cdf_nominal_symmetric": {
            "pbeta": 2.0,
            "qbeta": 2.0,
            "x": jnp.array([[0.0, 0.25, 0.5, 0.75, 1.0]]),
        },
        "beta_cdf_edge_uniform": {
            "pbeta": 1.0,
            "qbeta": 1.0,
            "x": jnp.array([[0.0, 0.1, 0.5, 0.9, 1.0]]),
        },
        "beta_cdf_edge_skewed": {
            "pbeta": 0.5,
            "qbeta": 5.0,
            "x": jnp.array([[0.0, 0.01, 0.05, 0.1, 0.5, 1.0]]),
        },
        "vertical_structure_nominal_tall": {
            "ztop": jnp.array([25.0, 30.0, 20.0]),
            "zref": jnp.array([40.0, 45.0, 35.0]),
            "nlayer_within": 0,
            "nlayer_above": 0,
            "dz_param": 10.0,
            "dz_tall": 1.0,
            "dz_short": 0.5,
            "nlevmlcan": 50,
        },
        "vertical_structure_edge_short": {
            "ztop": jnp.array([0.5, 1.0, 2.0]),
            "zref": jnp.array([10.0, 10.0, 10.0]),
            "nlayer_within": 0,
            "nlayer_above": 0,
            "dz_param": 10.0,
            "dz_tall": 1.0,
            "dz_short": 0.5,
            "nlevmlcan": 50,
        },
        "vertical_structure_manual_layers": {
            "ztop": jnp.array([15.0, 18.0]),
            "zref": jnp.array([30.0, 35.0]),
            "nlayer_within": 10,
            "nlayer_above": 5,
            "dz_param": 10.0,
            "dz_tall": 1.0,
            "dz_short": 0.5,
            "nlevmlcan": 50,
        },
    }


# ============================================================================
# Beta Distribution CDF Tests
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    ["beta_cdf_nominal_symmetric", "beta_cdf_edge_uniform", "beta_cdf_edge_skewed"],
)
def test_beta_distribution_cdf_shapes(test_data, test_case):
    """
    Test that beta_distribution_cdf returns correct output shapes.
    
    The output should have the same shape as the input x array.
    """
    data = test_data[test_case]
    result = beta_distribution_cdf(data["pbeta"], data["qbeta"], data["x"])
    
    assert result.shape == data["x"].shape, (
        f"Output shape {result.shape} does not match input shape {data['x'].shape}"
    )


def test_beta_distribution_cdf_monotonic(test_data):
    """
    Test that beta CDF is monotonically increasing.
    
    For any valid beta distribution, CDF(x1) <= CDF(x2) when x1 <= x2.
    """
    data = test_data["beta_cdf_nominal_symmetric"]
    result = beta_distribution_cdf(data["pbeta"], data["qbeta"], data["x"])
    
    # Check monotonicity
    result_np = np.array(result[0])
    diffs = np.diff(result_np)
    assert np.all(diffs >= -1e-10), (
        f"CDF is not monotonically increasing: {result_np}"
    )


def test_beta_distribution_cdf_boundaries(test_data):
    """
    Test that beta CDF satisfies boundary conditions.
    
    CDF(0) should be 0 and CDF(1) should be 1 for any valid beta distribution.
    """
    data = test_data["beta_cdf_nominal_symmetric"]
    result = beta_distribution_cdf(data["pbeta"], data["qbeta"], data["x"])
    
    assert np.isclose(result[0, 0], 0.0, atol=1e-6), (
        f"CDF(0) = {result[0, 0]} should be 0"
    )
    assert np.isclose(result[0, -1], 1.0, atol=1e-6), (
        f"CDF(1) = {result[0, -1]} should be 1"
    )


def test_beta_distribution_cdf_uniform_case(test_data):
    """
    Test beta CDF for uniform distribution (p=q=1).
    
    When p=q=1, the beta distribution is uniform, so CDF(x) = x.
    """
    data = test_data["beta_cdf_edge_uniform"]
    result = beta_distribution_cdf(data["pbeta"], data["qbeta"], data["x"])
    
    # For uniform distribution, CDF should equal x
    assert np.allclose(result, data["x"], atol=1e-6, rtol=1e-6), (
        f"Uniform CDF should equal x, got {result} vs {data['x']}"
    )


def test_beta_distribution_cdf_dtypes(test_data):
    """Test that beta_distribution_cdf returns correct data types."""
    data = test_data["beta_cdf_nominal_symmetric"]
    result = beta_distribution_cdf(data["pbeta"], data["qbeta"], data["x"])
    
    assert isinstance(result, jnp.ndarray), (
        f"Result should be jnp.ndarray, got {type(result)}"
    )


# ============================================================================
# Vertical Structure Calculation Tests
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        "vertical_structure_nominal_tall",
        "vertical_structure_edge_short",
        "vertical_structure_manual_layers",
    ],
)
def test_calculate_vertical_structure_shapes(test_data, test_case):
    """
    Test that calculate_vertical_structure returns correct output shapes.
    
    Returns ntop, ncan (n_patches,) and zw (n_patches, nlevmlcan+1).
    """
    data = test_data[test_case]
    ntop, ncan, zw = calculate_vertical_structure(
        data["ztop"],
        data["zref"],
        data["nlayer_within"],
        data["nlayer_above"],
        data["dz_param"],
        data["dz_tall"],
        data["dz_short"],
        data["nlevmlcan"],
    )
    
    n_patches = len(data["ztop"])
    assert ntop.shape == (n_patches,), f"ntop shape {ntop.shape} incorrect"
    assert ncan.shape == (n_patches,), f"ncan shape {ncan.shape} incorrect"
    assert zw.shape == (n_patches, data["nlevmlcan"] + 1), (
        f"zw shape {zw.shape} incorrect"
    )


def test_calculate_vertical_structure_layer_heights(test_data):
    """
    Test that layer heights are monotonically increasing.
    
    Heights at layer interfaces (zw) should increase from 0 to zref.
    """
    data = test_data["vertical_structure_nominal_tall"]
    ntop, ncan, zw = calculate_vertical_structure(
        data["ztop"],
        data["zref"],
        data["nlayer_within"],
        data["nlayer_above"],
        data["dz_param"],
        data["dz_tall"],
        data["dz_short"],
        data["nlevmlcan"],
    )
    
    # Check each patch
    for i in range(len(data["ztop"])):
        active_layers = int(ncan[i]) + 1
        heights = zw[i, :active_layers]
        
        # Check monotonicity
        diffs = np.diff(heights)
        assert np.all(diffs > -1e-10), (
            f"Heights not monotonic for patch {i}: {heights}"
        )
        
        # Check boundaries
        assert np.isclose(heights[0], 0.0, atol=1e-6), (
            f"Bottom height should be 0, got {heights[0]}"
        )
        assert heights[-1] <= data["zref"][i] + 1e-6, (
            f"Top height {heights[-1]} exceeds zref {data['zref'][i]}"
        )


def test_calculate_vertical_structure_tall_vs_short_spacing(test_data):
    """
    Test that tall canopies use dz_tall and short canopies use dz_short.
    
    Canopies above dz_param should use dz_tall spacing, below should use dz_short.
    """
    data_tall = test_data["vertical_structure_nominal_tall"]
    data_short = test_data["vertical_structure_edge_short"]
    
    # Tall canopy
    _, _, zw_tall = calculate_vertical_structure(
        data_tall["ztop"],
        data_tall["zref"],
        data_tall["nlayer_within"],
        data_tall["nlayer_above"],
        data_tall["dz_param"],
        data_tall["dz_tall"],
        data_tall["dz_short"],
        data_tall["nlevmlcan"],
    )
    
    # Short canopy
    _, _, zw_short = calculate_vertical_structure(
        data_short["ztop"],
        data_short["zref"],
        data_short["nlayer_within"],
        data_short["nlayer_above"],
        data_short["dz_param"],
        data_short["dz_tall"],
        data_short["dz_short"],
        data_short["nlevmlcan"],
    )
    
    # Tall canopy should have larger layer spacing
    tall_spacing = np.diff(zw_tall[0, :10])
    short_spacing = np.diff(zw_short[0, :10])
    
    assert np.mean(tall_spacing[tall_spacing > 0]) > np.mean(short_spacing[short_spacing > 0]), (
        "Tall canopy should have larger layer spacing than short canopy"
    )


def test_calculate_vertical_structure_manual_override(test_data):
    """
    Test that manual layer specification overrides automatic calculation.
    
    When nlayer_within and nlayer_above are specified, they should be respected.
    """
    data = test_data["vertical_structure_manual_layers"]
    ntop, ncan, zw = calculate_vertical_structure(
        data["ztop"],
        data["zref"],
        data["nlayer_within"],
        data["nlayer_above"],
        data["dz_param"],
        data["dz_tall"],
        data["dz_short"],
        data["nlevmlcan"],
    )
    
    # With manual specification, ntop should be close to nlayer_within
    # and ncan should be close to nlayer_within + nlayer_above
    expected_total = data["nlayer_within"] + data["nlayer_above"]
    
    for i in range(len(data["ztop"])):
        assert ncan[i] <= expected_total + 2, (
            f"ncan {ncan[i]} exceeds expected {expected_total} by too much"
        )


def test_calculate_vertical_structure_dtypes(test_data):
    """Test that calculate_vertical_structure returns correct data types."""
    data = test_data["vertical_structure_nominal_tall"]
    ntop, ncan, zw = calculate_vertical_structure(
        data["ztop"],
        data["zref"],
        data["nlayer_within"],
        data["nlayer_above"],
        data["dz_param"],
        data["dz_tall"],
        data["dz_short"],
        data["nlevmlcan"],
    )
    
    assert isinstance(ntop, jnp.ndarray), f"ntop should be jnp.ndarray"
    assert isinstance(ncan, jnp.ndarray), f"ncan should be jnp.ndarray"
    assert isinstance(zw, jnp.ndarray), f"zw should be jnp.ndarray"


# ============================================================================
# Layer Properties Tests
# ============================================================================


def test_calculate_layer_properties_shapes():
    """
    Test that calculate_layer_properties returns correct output shapes.
    
    All outputs should have shape (n_patches, n_levels).
    """
    n_patches = 2
    n_levels = 10
    
    zw = jnp.array([
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    ])
    ztop = jnp.array([10.0, 5.0])
    elai = jnp.array([5.0, 3.0])
    esai = jnp.array([1.0, 0.5])
    ncan = jnp.array([10, 10])
    ntop = jnp.array([10, 10])
    itype = jnp.array([1, 2])
    
    pft_params = PFTParams(
        pbeta_lai=jnp.array([2.0, 2.5, 3.0]),
        qbeta_lai=jnp.array([2.0, 2.0, 1.5]),
        pbeta_sai=jnp.array([1.5, 2.0, 2.5]),
        qbeta_sai=jnp.array([1.5, 1.5, 2.0]),
    )
    
    dz, zs, dlai, dsai = calculate_layer_properties(
        zw, ztop, elai, esai, ncan, ntop, itype, pft_params, n_levels
    )
    
    assert dz.shape == (n_patches, n_levels), f"dz shape {dz.shape} incorrect"
    assert zs.shape == (n_patches, n_levels), f"zs shape {zs.shape} incorrect"
    assert dlai.shape == (n_patches, n_levels), f"dlai shape {dlai.shape} incorrect"
    assert dsai.shape == (n_patches, n_levels), f"dsai shape {dsai.shape} incorrect"


def test_calculate_layer_properties_lai_conservation():
    """
    Test that total LAI is conserved across layers.
    
    Sum of dlai over all layers should equal elai for each patch.
    """
    n_levels = 10
    
    zw = jnp.array([
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    ])
    ztop = jnp.array([10.0, 5.0])
    elai = jnp.array([5.0, 3.0])
    esai = jnp.array([1.0, 0.5])
    ncan = jnp.array([10, 10])
    ntop = jnp.array([10, 10])
    itype = jnp.array([1, 2])
    
    pft_params = PFTParams(
        pbeta_lai=jnp.array([2.0, 2.5, 3.0]),
        qbeta_lai=jnp.array([2.0, 2.0, 1.5]),
        pbeta_sai=jnp.array([1.5, 2.0, 2.5]),
        qbeta_sai=jnp.array([1.5, 1.5, 2.0]),
    )
    
    dz, zs, dlai, dsai = calculate_layer_properties(
        zw, ztop, elai, esai, ncan, ntop, itype, pft_params, n_levels
    )
    
    # Check LAI conservation
    for i in range(len(elai)):
        total_lai = jnp.sum(dlai[i])
        assert np.isclose(total_lai, elai[i], atol=1e-4, rtol=1e-4), (
            f"LAI not conserved for patch {i}: {total_lai} vs {elai[i]}"
        )


def test_calculate_layer_properties_zero_lai():
    """
    Test handling of zero LAI (bare ground).
    
    When elai=0, all dlai values should be 0.
    """
    n_levels = 5
    
    zw = jnp.array([
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    ])
    ztop = jnp.array([5.0, 2.5])
    elai = jnp.array([0.0, 0.1])
    esai = jnp.array([0.0, 0.05])
    ncan = jnp.array([5, 5])
    ntop = jnp.array([5, 5])
    itype = jnp.array([0, 1])
    
    pft_params = PFTParams(
        pbeta_lai=jnp.array([2.0, 2.5]),
        qbeta_lai=jnp.array([2.0, 2.0]),
        pbeta_sai=jnp.array([1.5, 2.0]),
        qbeta_sai=jnp.array([1.5, 1.5]),
    )
    
    dz, zs, dlai, dsai = calculate_layer_properties(
        zw, ztop, elai, esai, ncan, ntop, itype, pft_params, n_levels
    )
    
    # First patch should have zero LAI everywhere
    assert np.allclose(dlai[0], 0.0, atol=1e-10), (
        f"Zero LAI patch should have all zero dlai: {dlai[0]}"
    )


def test_calculate_layer_properties_dtypes():
    """Test that calculate_layer_properties returns correct data types."""
    n_levels = 5
    
    zw = jnp.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    ztop = jnp.array([5.0])
    elai = jnp.array([3.0])
    esai = jnp.array([0.5])
    ncan = jnp.array([5])
    ntop = jnp.array([5])
    itype = jnp.array([1])
    
    pft_params = PFTParams(
        pbeta_lai=jnp.array([2.0, 2.5]),
        qbeta_lai=jnp.array([2.0, 2.0]),
        pbeta_sai=jnp.array([1.5, 2.0]),
        qbeta_sai=jnp.array([1.5, 1.5]),
    )
    
    dz, zs, dlai, dsai = calculate_layer_properties(
        zw, ztop, elai, esai, ncan, ntop, itype, pft_params, n_levels
    )
    
    assert isinstance(dz, jnp.ndarray), "dz should be jnp.ndarray"
    assert isinstance(zs, jnp.ndarray), "zs should be jnp.ndarray"
    assert isinstance(dlai, jnp.ndarray), "dlai should be jnp.ndarray"
    assert isinstance(dsai, jnp.ndarray), "dsai should be jnp.ndarray"


# ============================================================================
# Redistribute Plant Area Tests
# ============================================================================


def test_redistribute_plant_area_shapes():
    """
    Test that redistribute_plant_area returns correct output shapes.
    
    Outputs should match input shapes.
    """
    dlai = jnp.array([
        [0.5, 0.8, 1.2, 1.5, 1.0, 0.0, 0.0, 0.0],
        [0.3, 0.6, 0.9, 0.7, 0.5, 0.0, 0.0, 0.0],
    ])
    dsai = jnp.array([
        [0.1, 0.15, 0.2, 0.25, 0.3, 0.0, 0.0, 0.0],
        [0.05, 0.1, 0.15, 0.1, 0.1, 0.0, 0.0, 0.0],
    ])
    ntop = jnp.array([5, 5])
    zw = jnp.array([
        [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ])
    elai = jnp.array([5.0, 3.0])
    esai = jnp.array([1.0, 0.5])
    dpai_min = 0.01
    lai_tol = 1e-06
    
    dlai_out, dsai_out, nbot, zbot = redistribute_plant_area(
        dlai, dsai, ntop, zw, elai, esai, dpai_min, lai_tol
    )
    
    assert dlai_out.shape == dlai.shape, f"dlai shape mismatch"
    assert dsai_out.shape == dsai.shape, f"dsai shape mismatch"
    assert nbot.shape == ntop.shape, f"nbot shape mismatch"
    assert zbot.shape == (len(ntop),), f"zbot shape mismatch"


def test_redistribute_plant_area_conservation():
    """
    Test that total LAI/SAI is conserved during redistribution.
    
    Sum of dlai/dsai should remain equal to elai/esai.
    """
    dlai = jnp.array([
        [0.5, 0.8, 1.2, 1.5, 1.0, 0.0, 0.0, 0.0],
        [0.3, 0.6, 0.9, 0.7, 0.5, 0.0, 0.0, 0.0],
    ])
    dsai = jnp.array([
        [0.1, 0.15, 0.2, 0.25, 0.3, 0.0, 0.0, 0.0],
        [0.05, 0.1, 0.15, 0.1, 0.1, 0.0, 0.0, 0.0],
    ])
    ntop = jnp.array([5, 5])
    zw = jnp.array([
        [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ])
    elai = jnp.array([5.0, 3.0])
    esai = jnp.array([1.0, 0.5])
    dpai_min = 0.01
    lai_tol = 1e-06
    
    dlai_out, dsai_out, nbot, zbot = redistribute_plant_area(
        dlai, dsai, ntop, zw, elai, esai, dpai_min, lai_tol
    )
    
    # Check conservation
    for i in range(len(elai)):
        total_lai = jnp.sum(dlai_out[i])
        total_sai = jnp.sum(dsai_out[i])
        
        assert np.isclose(total_lai, elai[i], atol=lai_tol * 10, rtol=1e-4), (
            f"LAI not conserved for patch {i}: {total_lai} vs {elai[i]}"
        )
        assert np.isclose(total_sai, esai[i], atol=lai_tol * 10, rtol=1e-4), (
            f"SAI not conserved for patch {i}: {total_sai} vs {esai[i]}"
        )


def test_redistribute_plant_area_thin_layers():
    """
    Test redistribution when all layers are below threshold.
    
    Should consolidate thin layers while conserving totals.
    """
    dlai = jnp.array([
        [0.005, 0.008, 0.007, 0.006, 0.004, 0.0, 0.0],
        [0.003, 0.004, 0.005, 0.003, 0.005, 0.0, 0.0],
    ])
    dsai = jnp.array([
        [0.001, 0.002, 0.001, 0.001, 0.001, 0.0, 0.0],
        [0.001, 0.001, 0.001, 0.001, 0.001, 0.0, 0.0],
    ])
    ntop = jnp.array([5, 5])
    zw = jnp.array([
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1],
    ])
    elai = jnp.array([0.03, 0.02])
    esai = jnp.array([0.006, 0.005])
    dpai_min = 0.01
    lai_tol = 1e-06
    
    dlai_out, dsai_out, nbot, zbot = redistribute_plant_area(
        dlai, dsai, ntop, zw, elai, esai, dpai_min, lai_tol
    )
    
    # Should still conserve totals
    for i in range(len(elai)):
        total_lai = jnp.sum(dlai_out[i])
        assert np.isclose(total_lai, elai[i], atol=1e-5, rtol=1e-3), (
            f"LAI not conserved after thin layer redistribution: {total_lai} vs {elai[i]}"
        )


def test_redistribute_plant_area_dtypes():
    """Test that redistribute_plant_area returns correct data types."""
    dlai = jnp.array([[0.5, 0.8, 1.2, 1.5, 1.0]])
    dsai = jnp.array([[0.1, 0.15, 0.2, 0.25, 0.3]])
    ntop = jnp.array([5])
    zw = jnp.array([[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]])
    elai = jnp.array([5.0])
    esai = jnp.array([1.0])
    
    dlai_out, dsai_out, nbot, zbot = redistribute_plant_area(
        dlai, dsai, ntop, zw, elai, esai, 0.01, 1e-06
    )
    
    assert isinstance(dlai_out, jnp.ndarray), "dlai_out should be jnp.ndarray"
    assert isinstance(dsai_out, jnp.ndarray), "dsai_out should be jnp.ndarray"
    assert isinstance(nbot, jnp.ndarray), "nbot should be jnp.ndarray"
    assert isinstance(zbot, jnp.ndarray), "zbot should be jnp.ndarray"


# ============================================================================
# Finalize Vertical Structure Tests
# ============================================================================


def test_finalize_vertical_structure_shapes():
    """
    Test that finalize_vertical_structure returns correct output shapes.
    
    All outputs should match input shapes.
    """
    dlai = jnp.array([
        [0.5, 0.8, 1.2, 1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.6, 0.9, 0.7, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    dsai = jnp.array([
        [0.1, 0.15, 0.2, 0.25, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.05, 0.1, 0.15, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    elai = jnp.array([5.0, 3.0])
    esai = jnp.array([1.0, 0.5])
    ntop = jnp.array([5, 5])
    ncan = jnp.array([10, 10])
    
    dlai_out, dsai_out, dlai_frac, dsai_frac = finalize_vertical_structure(
        dlai, dsai, elai, esai, ntop, ncan, 1e-06
    )
    
    assert dlai_out.shape == dlai.shape, "dlai_out shape mismatch"
    assert dsai_out.shape == dsai.shape, "dsai_out shape mismatch"
    assert dlai_frac.shape == dlai.shape, "dlai_frac shape mismatch"
    assert dsai_frac.shape == dsai.shape, "dsai_frac shape mismatch"


def test_finalize_vertical_structure_fractions_sum():
    """
    Test that fractional profiles sum to 1.0 over active layers.
    
    Sum of dlai_frac and dsai_frac over active layers should be 1.0.
    """
    dlai = jnp.array([
        [0.5, 0.8, 1.2, 1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.6, 0.9, 0.7, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    dsai = jnp.array([
        [0.1, 0.15, 0.2, 0.25, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.05, 0.1, 0.15, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    elai = jnp.array([5.0, 3.0])
    esai = jnp.array([1.0, 0.5])
    ntop = jnp.array([5, 5])
    ncan = jnp.array([10, 10])
    
    dlai_out, dsai_out, dlai_frac, dsai_frac = finalize_vertical_structure(
        dlai, dsai, elai, esai, ntop, ncan, 1e-06
    )
    
    # Check fraction sums
    for i in range(len(elai)):
        if elai[i] > 0:
            lai_frac_sum = jnp.sum(dlai_frac[i, :int(ntop[i])])
            assert np.isclose(lai_frac_sum, 1.0, atol=1e-5, rtol=1e-5), (
                f"LAI fractions don't sum to 1.0 for patch {i}: {lai_frac_sum}"
            )
        
        if esai[i] > 0:
            sai_frac_sum = jnp.sum(dsai_frac[i, :int(ntop[i])])
            assert np.isclose(sai_frac_sum, 1.0, atol=1e-5, rtol=1e-5), (
                f"SAI fractions don't sum to 1.0 for patch {i}: {sai_frac_sum}"
            )


def test_finalize_vertical_structure_zero_totals():
    """
    Test handling of zero LAI/SAI totals.
    
    When elai or esai is zero, fractions should be zero (no division by zero).
    """
    dlai = jnp.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.01, 0.02, 0.01, 0.0, 0.0],
    ])
    dsai = jnp.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.005, 0.005, 0.0, 0.0, 0.0],
    ])
    elai = jnp.array([0.0, 0.04])
    esai = jnp.array([0.0, 0.01])
    ntop = jnp.array([0, 3])
    ncan = jnp.array([5, 5])
    
    dlai_out, dsai_out, dlai_frac, dsai_frac = finalize_vertical_structure(
        dlai, dsai, elai, esai, ntop, ncan, 1e-06
    )
    
    # First patch should have all zero fractions
    assert np.allclose(dlai_frac[0], 0.0, atol=1e-10), (
        "Zero LAI patch should have zero fractions"
    )
    assert np.allclose(dsai_frac[0], 0.0, atol=1e-10), (
        "Zero SAI patch should have zero fractions"
    )


def test_finalize_vertical_structure_dtypes():
    """Test that finalize_vertical_structure returns correct data types."""
    dlai = jnp.array([[0.5, 0.8, 1.2, 1.5, 1.0]])
    dsai = jnp.array([[0.1, 0.15, 0.2, 0.25, 0.3]])
    elai = jnp.array([5.0])
    esai = jnp.array([1.0])
    ntop = jnp.array([5])
    ncan = jnp.array([5])
    
    dlai_out, dsai_out, dlai_frac, dsai_frac = finalize_vertical_structure(
        dlai, dsai, elai, esai, ntop, ncan, 1e-06
    )
    
    assert isinstance(dlai_out, jnp.ndarray), "dlai_out should be jnp.ndarray"
    assert isinstance(dsai_out, jnp.ndarray), "dsai_out should be jnp.ndarray"
    assert isinstance(dlai_frac, jnp.ndarray), "dlai_frac should be jnp.ndarray"
    assert isinstance(dsai_frac, jnp.ndarray), "dsai_frac should be jnp.ndarray"


# ============================================================================
# Integration Tests
# ============================================================================


def test_init_vertical_structure_complete_workflow():
    """
    Test complete vertical structure initialization workflow.
    
    This integration test verifies that init_vertical_structure properly
    initializes all components of MLCanopyType.
    """
    bounds = BoundsType(begp=0, endp=3, begc=0, endc=2, begg=0, endg=1)
    
    canopystate_inst = CanopyStateType(
        htop=jnp.array([15.0, 20.0, 5.0]),
        elai=jnp.array([4.5, 5.5, 2.0]),
        esai=jnp.array([0.8, 1.2, 0.3]),
    )
    
    patch_state = PatchState(
        column=jnp.array([0, 0, 1]),
        gridcell=jnp.array([0, 0, 0]),
        itype=jnp.array([1, 2, 0]),
    )
    
    pft_params = PFTParams(
        pbeta_lai=jnp.array([2.0, 2.5, 3.0]),
        qbeta_lai=jnp.array([2.0, 2.0, 1.5]),
        pbeta_sai=jnp.array([1.5, 2.0, 2.5]),
        qbeta_sai=jnp.array([1.5, 1.5, 2.0]),
    )
    
    params = MLCanopyParams()
    nlevmlcan = 50
    
    mlcanopy = init_vertical_structure(
        bounds, canopystate_inst, patch_state, pft_params, params, nlevmlcan
    )
    
    # Check that all fields are initialized
    assert isinstance(mlcanopy, MLCanopyType), "Should return MLCanopyType"
    assert mlcanopy.ncan.shape == (3,), "ncan shape incorrect"
    assert mlcanopy.ntop.shape == (3,), "ntop shape incorrect"
    assert mlcanopy.zw.shape == (3, nlevmlcan + 1), "zw shape incorrect"
    assert mlcanopy.dlai.shape == (3, nlevmlcan), "dlai shape incorrect"
    
    # Check LAI conservation
    for i in range(3):
        total_lai = jnp.sum(mlcanopy.dlai[i])
        assert np.isclose(total_lai, canopystate_inst.elai[i], atol=1e-3, rtol=1e-3), (
            f"LAI not conserved for patch {i}"
        )


def test_init_vertical_profiles_atmospheric_initialization():
    """
    Test initialization of vertical profiles from atmospheric forcing.
    
    Verifies that wind, temperature, humidity, and CO2 profiles are
    properly initialized from atmospheric boundary conditions.
    """
    atm2lnd_inst = Atm2LndState(
        forc_u_grc=jnp.array([3.5]),
        forc_v_grc=jnp.array([2.0]),
        forc_pco2_grc=jnp.array([40000.0]),
        forc_t_downscaled_col=jnp.array([298.15, 295.15]),
        forc_q_downscaled_col=jnp.array([0.012, 0.01]),
        forc_pbot_downscaled_col=jnp.array([101325.0, 101325.0]),
        forc_hgt_u=jnp.array([40.0, 35.0, 30.0]),
    )
    
    # Create minimal mlcanopy_inst
    n_patches = 3
    n_layers = 10
    
    mlcanopy_inst = MLCanopyType(
        ncan=jnp.array([10, 8, 6]),
        ntop=jnp.array([8, 6, 4]),
        nbot=jnp.array([1, 1, 1]),
        zref=jnp.array([40.0, 35.0, 30.0]),
        ztop=jnp.array([20.0, 15.0, 10.0]),
        zbot=jnp.array([2.0, 2.0, 1.0]),
        zs=jnp.zeros((n_patches, n_layers)),
        dz=jnp.ones((n_patches, n_layers)) * 2.0,
        zw=jnp.zeros((n_patches, n_layers + 1)),
        dlai=jnp.ones((n_patches, n_layers)) * 0.5,
        dsai=jnp.ones((n_patches, n_layers)) * 0.1,
        dlai_frac=jnp.ones((n_patches, n_layers)) * 0.1,
        dsai_frac=jnp.ones((n_patches, n_layers)) * 0.1,
        wind=jnp.zeros((n_patches, n_layers)),
        tair=jnp.zeros((n_patches, n_layers)),
        eair=jnp.zeros((n_patches, n_layers)),
        cair=jnp.zeros((n_patches, n_layers)),
        h2ocan=jnp.zeros((n_patches, n_layers)),
        taf=jnp.zeros(n_patches),
        qaf=jnp.zeros(n_patches),
        tg=jnp.zeros(n_patches),
        tleaf=jnp.zeros((n_patches, n_layers, 2)),
        lwp=jnp.zeros((n_patches, n_layers, 2)),
    )
    
    patch_state = PatchState(
        column=jnp.array([0, 0, 1]),
        gridcell=jnp.array([0, 0, 0]),
        itype=jnp.array([1, 2, 0]),
    )
    
    params = MLCanopyParams()
    
    mlcanopy_out = init_vertical_profiles(
        atm2lnd_inst, mlcanopy_inst, patch_state, params
    )
    
    # Check that profiles are initialized (not all zeros)
    assert not np.allclose(mlcanopy_out.wind, 0.0), "Wind profile should be initialized"
    assert not np.allclose(mlcanopy_out.tair, 0.0), "Temperature profile should be initialized"
    assert not np.allclose(mlcanopy_out.cair, 0.0), "CO2 profile should be initialized"
    
    # Check physical realism
    assert np.all(mlcanopy_out.wind >= 0), "Wind should be non-negative"
    assert np.all(mlcanopy_out.tair > 0), "Temperature should be positive (Kelvin)"


def test_init_vertical_profiles_low_wind_enforcement():
    """
    Test that minimum wind speed is enforced.
    
    When atmospheric wind is very low, it should be clamped to wind_forc_min.
    """
    atm2lnd_inst = Atm2LndState(
        forc_u_grc=jnp.array([0.05]),
        forc_v_grc=jnp.array([0.03]),
        forc_pco2_grc=jnp.array([41000.0]),
        forc_t_downscaled_col=jnp.array([273.15, 275.15]),
        forc_q_downscaled_col=jnp.array([0.001, 0.002]),
        forc_pbot_downscaled_col=jnp.array([95000.0, 96000.0]),
        forc_hgt_u=jnp.array([30.0, 25.0]),
    )
    
    n_patches = 2
    n_layers = 5
    
    mlcanopy_inst = MLCanopyType(
        ncan=jnp.array([5, 4]),
        ntop=jnp.array([4, 3]),
        nbot=jnp.array([1, 1]),
        zref=jnp.array([30.0, 25.0]),
        ztop=jnp.array([8.0, 6.0]),
        zbot=jnp.array([1.0, 1.0]),
        zs=jnp.zeros((n_patches, n_layers)),
        dz=jnp.ones((n_patches, n_layers)) * 1.5,
        zw=jnp.zeros((n_patches, n_layers + 1)),
        dlai=jnp.ones((n_patches, n_layers)) * 0.4,
        dsai=jnp.ones((n_patches, n_layers)) * 0.08,
        dlai_frac=jnp.ones((n_patches, n_layers)) * 0.2,
        dsai_frac=jnp.ones((n_patches, n_layers)) * 0.2,
        wind=jnp.zeros((n_patches, n_layers)),
        tair=jnp.zeros((n_patches, n_layers)),
        eair=jnp.zeros((n_patches, n_layers)),
        cair=jnp.zeros((n_patches, n_layers)),
        h2ocan=jnp.zeros((n_patches, n_layers)),
        taf=jnp.zeros(n_patches),
        qaf=jnp.zeros(n_patches),
        tg=jnp.zeros(n_patches),
        tleaf=jnp.zeros((n_patches, n_layers, 2)),
        lwp=jnp.zeros((n_patches, n_layers, 2)),
    )
    
    patch_state = PatchState(
        column=jnp.array([0, 1]),
        gridcell=jnp.array([0, 0]),
        itype=jnp.array([1, 2]),
    )
    
    params = MLCanopyParams(wind_forc_min=0.1)
    
    mlcanopy_out = init_vertical_profiles(
        atm2lnd_inst, mlcanopy_inst, patch_state, params
    )
    
    # Wind should be at least wind_forc_min
    active_wind = mlcanopy_out.wind[mlcanopy_out.wind > 0]
    if len(active_wind) > 0:
        assert np.all(active_wind >= params.wind_forc_min - 1e-6), (
            f"Wind should be >= {params.wind_forc_min}, got min {np.min(active_wind)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])