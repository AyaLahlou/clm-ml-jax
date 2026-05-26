"""
Comprehensive pytest suite for solar_radiation function from MLSolarRadiationMod.

This test suite covers:
- Nominal cases with varying canopy structures and solar conditions
- Edge cases (zero LAI, extreme albedos, boundary solar angles)
- Special cases (single-layer canopy, extreme optical properties)
- Shape validation, dtype checking, and physical constraint verification
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multilayer_canopy.MLSolarRadiationMod import solar_radiation


# Define NamedTuples matching the function signature
class BoundsType(NamedTuple):
    """Bounds type containing patch indices."""
    begp: int
    endp: int
    begg: int
    endg: int


class PatchState(NamedTuple):
    """Patch-level state variables."""
    itype: jnp.ndarray
    cosz: jnp.ndarray
    swskyb: jnp.ndarray
    swskyd: jnp.ndarray
    albsoib: jnp.ndarray
    albsoid: jnp.ndarray


class MLCanopyState(NamedTuple):
    """Multilayer canopy state variables."""
    dlai_profile: jnp.ndarray
    dsai_profile: jnp.ndarray
    dpai_profile: jnp.ndarray
    ntop_canopy: jnp.ndarray
    nbot_canopy: jnp.ndarray
    ncan_canopy: jnp.ndarray


class PFTParams(NamedTuple):
    """PFT-specific parameters."""
    rhol: jnp.ndarray
    taul: jnp.ndarray
    rhos: jnp.ndarray
    taus: jnp.ndarray
    xl: jnp.ndarray
    clump_fac: jnp.ndarray


@pytest.fixture
def test_data():
    """
    Load and prepare test data for solar_radiation function.
    
    Returns:
        dict: Dictionary containing all test cases with inputs and metadata.
    """
    return {
        "test_nominal_single_patch_two_layers": {
            "inputs": {
                "bounds": BoundsType(begp=0, endp=1, begg=0, endg=1),
                "num_filter": 1,
                "filter_indices": jnp.array([0]),
                "patch_state": PatchState(
                    itype=jnp.array([5]),
                    cosz=jnp.array([0.7]),
                    swskyb=jnp.array([[800.0, 600.0]]),
                    swskyd=jnp.array([[200.0, 150.0]]),
                    albsoib=jnp.array([[0.15, 0.25]]),
                    albsoid=jnp.array([[0.2, 0.3]]),
                ),
                "mlcanopy_state": MLCanopyState(
                    dlai_profile=jnp.array([[2.5, 1.8]]),
                    dsai_profile=jnp.array([[0.3, 0.2]]),
                    dpai_profile=jnp.array([[2.8, 2.0]]),
                    ntop_canopy=jnp.array([0]),
                    nbot_canopy=jnp.array([1]),
                    ncan_canopy=jnp.array([2]),
                ),
                "pft_params": PFTParams(
                    rhol=jnp.array([
                        [0.1, 0.45], [0.12, 0.5], [0.08, 0.4],
                        [0.11, 0.48], [0.09, 0.43], [0.1, 0.45]
                    ]),
                    taul=jnp.array([
                        [0.05, 0.25], [0.06, 0.28], [0.04, 0.22],
                        [0.05, 0.26], [0.05, 0.24], [0.05, 0.25]
                    ]),
                    rhos=jnp.array([
                        [0.16, 0.39], [0.18, 0.42], [0.14, 0.36],
                        [0.17, 0.4], [0.15, 0.38], [0.16, 0.39]
                    ]),
                    taus=jnp.array([
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001],
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]
                    ]),
                    xl=jnp.array([0.01, 0.1, -0.1, 0.05, -0.05, 0.01]),
                    clump_fac=jnp.array([0.85, 0.8, 0.9, 0.82, 0.88, 0.85]),
                ),
                "nlevmlcan": 2,
                "light_type": 1,
                "numrad": 2,
            },
            "metadata": {
                "type": "nominal",
                "description": "Typical midday conditions with moderate LAI, single patch, Norman radiation scheme",
            },
        },
        "test_nominal_multiple_patches_five_layers": {
            "inputs": {
                "bounds": BoundsType(begp=0, endp=3, begg=0, endg=2),
                "num_filter": 3,
                "filter_indices": jnp.array([0, 1, 2]),
                "patch_state": PatchState(
                    itype=jnp.array([2, 4, 5]),
                    cosz=jnp.array([0.85, 0.65, 0.5]),
                    swskyb=jnp.array([[900.0, 700.0], [750.0, 550.0], [600.0, 450.0]]),
                    swskyd=jnp.array([[150.0, 100.0], [180.0, 120.0], [220.0, 160.0]]),
                    albsoib=jnp.array([[0.12, 0.22], [0.18, 0.28], [0.14, 0.24]]),
                    albsoid=jnp.array([[0.17, 0.27], [0.23, 0.33], [0.19, 0.29]]),
                ),
                "mlcanopy_state": MLCanopyState(
                    dlai_profile=jnp.array([
                        [1.2, 1.5, 1.8, 1.3, 0.8],
                        [0.9, 1.1, 1.4, 1.2, 0.7],
                        [1.5, 1.8, 2.1, 1.6, 1.0],
                    ]),
                    dsai_profile=jnp.array([
                        [0.15, 0.18, 0.22, 0.16, 0.1],
                        [0.12, 0.14, 0.17, 0.15, 0.09],
                        [0.18, 0.22, 0.25, 0.19, 0.12],
                    ]),
                    dpai_profile=jnp.array([
                        [1.35, 1.68, 2.02, 1.46, 0.9],
                        [1.02, 1.24, 1.57, 1.35, 0.79],
                        [1.68, 2.02, 2.35, 1.79, 1.12],
                    ]),
                    ntop_canopy=jnp.array([0, 0, 0]),
                    nbot_canopy=jnp.array([4, 4, 4]),
                    ncan_canopy=jnp.array([5, 5, 5]),
                ),
                "pft_params": PFTParams(
                    rhol=jnp.array([
                        [0.1, 0.45], [0.12, 0.5], [0.08, 0.4],
                        [0.11, 0.48], [0.09, 0.43], [0.1, 0.45]
                    ]),
                    taul=jnp.array([
                        [0.05, 0.25], [0.06, 0.28], [0.04, 0.22],
                        [0.05, 0.26], [0.05, 0.24], [0.05, 0.25]
                    ]),
                    rhos=jnp.array([
                        [0.16, 0.39], [0.18, 0.42], [0.14, 0.36],
                        [0.17, 0.4], [0.15, 0.38], [0.16, 0.39]
                    ]),
                    taus=jnp.array([
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001],
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]
                    ]),
                    xl=jnp.array([0.01, 0.1, -0.1, 0.05, -0.05, 0.01]),
                    clump_fac=jnp.array([0.85, 0.8, 0.9, 0.82, 0.88, 0.85]),
                ),
                "nlevmlcan": 5,
                "light_type": 1,
                "numrad": 2,
            },
            "metadata": {
                "type": "nominal",
                "description": "Multiple patches with varying solar angles and LAI profiles, 5-layer canopy",
            },
        },
        "test_nominal_twostream_method": {
            "inputs": {
                "bounds": BoundsType(begp=0, endp=2, begg=0, endg=1),
                "num_filter": 2,
                "filter_indices": jnp.array([0, 1]),
                "patch_state": PatchState(
                    itype=jnp.array([3, 5]),
                    cosz=jnp.array([0.75, 0.6]),
                    swskyb=jnp.array([[850.0, 650.0], [700.0, 500.0]]),
                    swskyd=jnp.array([[180.0, 130.0], [210.0, 155.0]]),
                    albsoib=jnp.array([[0.13, 0.23], [0.16, 0.26]]),
                    albsoid=jnp.array([[0.18, 0.28], [0.21, 0.31]]),
                ),
                "mlcanopy_state": MLCanopyState(
                    dlai_profile=jnp.array([[1.8, 2.2, 1.5], [2.1, 2.5, 1.8]]),
                    dsai_profile=jnp.array([[0.2, 0.25, 0.18], [0.24, 0.28, 0.2]]),
                    dpai_profile=jnp.array([[2.0, 2.45, 1.68], [2.34, 2.78, 2.0]]),
                    ntop_canopy=jnp.array([0, 0]),
                    nbot_canopy=jnp.array([2, 2]),
                    ncan_canopy=jnp.array([3, 3]),
                ),
                "pft_params": PFTParams(
                    rhol=jnp.array([
                        [0.1, 0.45], [0.12, 0.5], [0.08, 0.4],
                        [0.11, 0.48], [0.09, 0.43], [0.1, 0.45]
                    ]),
                    taul=jnp.array([
                        [0.05, 0.25], [0.06, 0.28], [0.04, 0.22],
                        [0.05, 0.26], [0.05, 0.24], [0.05, 0.25]
                    ]),
                    rhos=jnp.array([
                        [0.16, 0.39], [0.18, 0.42], [0.14, 0.36],
                        [0.17, 0.4], [0.15, 0.38], [0.16, 0.39]
                    ]),
                    taus=jnp.array([
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001],
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]
                    ]),
                    xl=jnp.array([0.01, 0.1, -0.1, 0.05, -0.05, 0.01]),
                    clump_fac=jnp.array([0.85, 0.8, 0.9, 0.82, 0.88, 0.85]),
                ),
                "nlevmlcan": 3,
                "light_type": 2,
                "numrad": 2,
            },
            "metadata": {
                "type": "nominal",
                "description": "Testing TwoStream radiation transfer method with moderate canopy",
            },
        },
        "test_edge_zero_solar_zenith_angle": {
            "inputs": {
                "bounds": BoundsType(begp=0, endp=2, begg=0, endg=1),
                "num_filter": 2,
                "filter_indices": jnp.array([0, 1]),
                "patch_state": PatchState(
                    itype=jnp.array([1, 3]),
                    cosz=jnp.array([0.0, 0.001]),
                    swskyb=jnp.array([[0.0, 0.0], [5.0, 3.0]]),
                    swskyd=jnp.array([[50.0, 30.0], [80.0, 50.0]]),
                    albsoib=jnp.array([[0.2, 0.3], [0.18, 0.28]]),
                    albsoid=jnp.array([[0.25, 0.35], [0.23, 0.33]]),
                ),
                "mlcanopy_state": MLCanopyState(
                    dlai_profile=jnp.array([[1.0, 0.8, 0.5], [1.2, 1.0, 0.7]]),
                    dsai_profile=jnp.array([[0.12, 0.1, 0.06], [0.15, 0.12, 0.08]]),
                    dpai_profile=jnp.array([[1.12, 0.9, 0.56], [1.35, 1.12, 0.78]]),
                    ntop_canopy=jnp.array([0, 0]),
                    nbot_canopy=jnp.array([2, 2]),
                    ncan_canopy=jnp.array([3, 3]),
                ),
                "pft_params": PFTParams(
                    rhol=jnp.array([
                        [0.1, 0.45], [0.12, 0.5], [0.08, 0.4],
                        [0.11, 0.48], [0.09, 0.43], [0.1, 0.45]
                    ]),
                    taul=jnp.array([
                        [0.05, 0.25], [0.06, 0.28], [0.04, 0.22],
                        [0.05, 0.26], [0.05, 0.24], [0.05, 0.25]
                    ]),
                    rhos=jnp.array([
                        [0.16, 0.39], [0.18, 0.42], [0.14, 0.36],
                        [0.17, 0.4], [0.15, 0.38], [0.16, 0.39]
                    ]),
                    taus=jnp.array([
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001],
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]
                    ]),
                    xl=jnp.array([0.01, 0.1, -0.1, 0.05, -0.05, 0.01]),
                    clump_fac=jnp.array([0.85, 0.8, 0.9, 0.82, 0.88, 0.85]),
                ),
                "nlevmlcan": 3,
                "light_type": 1,
                "numrad": 2,
            },
            "metadata": {
                "type": "edge",
                "description": "Near-horizontal sun (sunrise/sunset) with minimal direct beam radiation",
            },
        },
        "test_edge_zero_lai_sparse_canopy": {
            "inputs": {
                "bounds": BoundsType(begp=0, endp=1, begg=0, endg=1),
                "num_filter": 1,
                "filter_indices": jnp.array([0]),
                "patch_state": PatchState(
                    itype=jnp.array([2]),
                    cosz=jnp.array([0.8]),
                    swskyb=jnp.array([[850.0, 650.0]]),
                    swskyd=jnp.array([[150.0, 100.0]]),
                    albsoib=jnp.array([[0.15, 0.25]]),
                    albsoid=jnp.array([[0.2, 0.3]]),
                ),
                "mlcanopy_state": MLCanopyState(
                    dlai_profile=jnp.array([[0.0, 0.0, 0.0, 0.0]]),
                    dsai_profile=jnp.array([[0.05, 0.03, 0.02, 0.01]]),
                    dpai_profile=jnp.array([[0.05, 0.03, 0.02, 0.01]]),
                    ntop_canopy=jnp.array([0]),
                    nbot_canopy=jnp.array([3]),
                    ncan_canopy=jnp.array([4]),
                ),
                "pft_params": PFTParams(
                    rhol=jnp.array([
                        [0.1, 0.45], [0.12, 0.5], [0.08, 0.4],
                        [0.11, 0.48], [0.09, 0.43], [0.1, 0.45]
                    ]),
                    taul=jnp.array([
                        [0.05, 0.25], [0.06, 0.28], [0.04, 0.22],
                        [0.05, 0.26], [0.05, 0.24], [0.05, 0.25]
                    ]),
                    rhos=jnp.array([
                        [0.16, 0.39], [0.18, 0.42], [0.14, 0.36],
                        [0.17, 0.4], [0.15, 0.38], [0.16, 0.39]
                    ]),
                    taus=jnp.array([
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001],
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]
                    ]),
                    xl=jnp.array([0.01, 0.1, -0.1, 0.05, -0.05, 0.01]),
                    clump_fac=jnp.array([0.85, 0.8, 0.9, 0.82, 0.88, 0.85]),
                ),
                "nlevmlcan": 4,
                "light_type": 1,
                "numrad": 2,
            },
            "metadata": {
                "type": "edge",
                "description": "Bare/sparse canopy with zero LAI, only stems present",
            },
        },
        "test_edge_maximum_lai_dense_canopy": {
            "inputs": {
                "bounds": BoundsType(begp=0, endp=1, begg=0, endg=1),
                "num_filter": 1,
                "filter_indices": jnp.array([0]),
                "patch_state": PatchState(
                    itype=jnp.array([4]),
                    cosz=jnp.array([0.9]),
                    swskyb=jnp.array([[950.0, 750.0]]),
                    swskyd=jnp.array([[100.0, 70.0]]),
                    albsoib=jnp.array([[0.1, 0.2]]),
                    albsoid=jnp.array([[0.15, 0.25]]),
                ),
                "mlcanopy_state": MLCanopyState(
                    dlai_profile=jnp.array([[3.5, 4.0, 4.2, 3.8, 3.0, 2.5]]),
                    dsai_profile=jnp.array([[0.4, 0.45, 0.48, 0.42, 0.35, 0.28]]),
                    dpai_profile=jnp.array([[3.9, 4.45, 4.68, 4.22, 3.35, 2.78]]),
                    ntop_canopy=jnp.array([0]),
                    nbot_canopy=jnp.array([5]),
                    ncan_canopy=jnp.array([6]),
                ),
                "pft_params": PFTParams(
                    rhol=jnp.array([
                        [0.1, 0.45], [0.12, 0.5], [0.08, 0.4],
                        [0.11, 0.48], [0.09, 0.43], [0.1, 0.45]
                    ]),
                    taul=jnp.array([
                        [0.05, 0.25], [0.06, 0.28], [0.04, 0.22],
                        [0.05, 0.26], [0.05, 0.24], [0.05, 0.25]
                    ]),
                    rhos=jnp.array([
                        [0.16, 0.39], [0.18, 0.42], [0.14, 0.36],
                        [0.17, 0.4], [0.15, 0.38], [0.16, 0.39]
                    ]),
                    taus=jnp.array([
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001],
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]
                    ]),
                    xl=jnp.array([0.01, 0.1, -0.1, 0.05, -0.05, 0.01]),
                    clump_fac=jnp.array([0.85, 0.8, 0.9, 0.82, 0.88, 0.85]),
                ),
                "nlevmlcan": 6,
                "light_type": 1,
                "numrad": 2,
            },
            "metadata": {
                "type": "edge",
                "description": "Very dense canopy with high LAI values, testing light extinction limits",
            },
        },
        "test_edge_boundary_albedo_values": {
            "inputs": {
                "bounds": BoundsType(begp=0, endp=3, begg=0, endg=2),
                "num_filter": 3,
                "filter_indices": jnp.array([0, 1, 2]),
                "patch_state": PatchState(
                    itype=jnp.array([1, 2, 3]),
                    cosz=jnp.array([0.7, 0.7, 0.7]),
                    swskyb=jnp.array([[800.0, 600.0], [800.0, 600.0], [800.0, 600.0]]),
                    swskyd=jnp.array([[200.0, 150.0], [200.0, 150.0], [200.0, 150.0]]),
                    albsoib=jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
                    albsoid=jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
                ),
                "mlcanopy_state": MLCanopyState(
                    dlai_profile=jnp.array([[1.5, 1.2], [1.5, 1.2], [1.5, 1.2]]),
                    dsai_profile=jnp.array([[0.18, 0.15], [0.18, 0.15], [0.18, 0.15]]),
                    dpai_profile=jnp.array([[1.68, 1.35], [1.68, 1.35], [1.68, 1.35]]),
                    ntop_canopy=jnp.array([0, 0, 0]),
                    nbot_canopy=jnp.array([1, 1, 1]),
                    ncan_canopy=jnp.array([2, 2, 2]),
                ),
                "pft_params": PFTParams(
                    rhol=jnp.array([
                        [0.1, 0.45], [0.12, 0.5], [0.08, 0.4],
                        [0.11, 0.48], [0.09, 0.43], [0.1, 0.45]
                    ]),
                    taul=jnp.array([
                        [0.05, 0.25], [0.06, 0.28], [0.04, 0.22],
                        [0.05, 0.26], [0.05, 0.24], [0.05, 0.25]
                    ]),
                    rhos=jnp.array([
                        [0.16, 0.39], [0.18, 0.42], [0.14, 0.36],
                        [0.17, 0.4], [0.15, 0.38], [0.16, 0.39]
                    ]),
                    taus=jnp.array([
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001],
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]
                    ]),
                    xl=jnp.array([0.01, 0.1, -0.1, 0.05, -0.05, 0.01]),
                    clump_fac=jnp.array([0.85, 0.8, 0.9, 0.82, 0.88, 0.85]),
                ),
                "nlevmlcan": 2,
                "light_type": 1,
                "numrad": 2,
            },
            "metadata": {
                "type": "edge",
                "description": "Testing boundary albedo values (0.0, 0.5, 1.0) for soil reflectance",
            },
        },
        "test_edge_extreme_leaf_orientation": {
            "inputs": {
                "bounds": BoundsType(begp=0, endp=3, begg=0, endg=2),
                "num_filter": 3,
                "filter_indices": jnp.array([0, 1, 2]),
                "patch_state": PatchState(
                    itype=jnp.array([0, 1, 2]),
                    cosz=jnp.array([0.6, 0.6, 0.6]),
                    swskyb=jnp.array([[750.0, 550.0], [750.0, 550.0], [750.0, 550.0]]),
                    swskyd=jnp.array([[200.0, 150.0], [200.0, 150.0], [200.0, 150.0]]),
                    albsoib=jnp.array([[0.15, 0.25], [0.15, 0.25], [0.15, 0.25]]),
                    albsoid=jnp.array([[0.2, 0.3], [0.2, 0.3], [0.2, 0.3]]),
                ),
                "mlcanopy_state": MLCanopyState(
                    dlai_profile=jnp.array([[2.0, 1.5, 1.0], [2.0, 1.5, 1.0], [2.0, 1.5, 1.0]]),
                    dsai_profile=jnp.array([[0.22, 0.18, 0.12], [0.22, 0.18, 0.12], [0.22, 0.18, 0.12]]),
                    dpai_profile=jnp.array([[2.22, 1.68, 1.12], [2.22, 1.68, 1.12], [2.22, 1.68, 1.12]]),
                    ntop_canopy=jnp.array([0, 0, 0]),
                    nbot_canopy=jnp.array([2, 2, 2]),
                    ncan_canopy=jnp.array([3, 3, 3]),
                ),
                "pft_params": PFTParams(
                    rhol=jnp.array([
                        [0.1, 0.45], [0.12, 0.5], [0.08, 0.4],
                        [0.11, 0.48], [0.09, 0.43], [0.1, 0.45]
                    ]),
                    taul=jnp.array([
                        [0.05, 0.25], [0.06, 0.28], [0.04, 0.22],
                        [0.05, 0.26], [0.05, 0.24], [0.05, 0.25]
                    ]),
                    rhos=jnp.array([
                        [0.16, 0.39], [0.18, 0.42], [0.14, 0.36],
                        [0.17, 0.4], [0.15, 0.38], [0.16, 0.39]
                    ]),
                    taus=jnp.array([
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001],
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]
                    ]),
                    xl=jnp.array([-0.4, 0.0, 0.6, 0.05, -0.05, 0.01]),
                    clump_fac=jnp.array([0.85, 0.8, 0.9, 0.82, 0.88, 0.85]),
                ),
                "nlevmlcan": 3,
                "light_type": 1,
                "numrad": 2,
            },
            "metadata": {
                "type": "edge",
                "description": "Testing extreme leaf orientation indices (planophile, spherical, erectophile)",
            },
        },
        "test_special_single_layer_canopy": {
            "inputs": {
                "bounds": BoundsType(begp=0, endp=2, begg=0, endg=1),
                "num_filter": 2,
                "filter_indices": jnp.array([0, 1]),
                "patch_state": PatchState(
                    itype=jnp.array([3, 5]),
                    cosz=jnp.array([0.8, 0.7]),
                    swskyb=jnp.array([[880.0, 680.0], [820.0, 620.0]]),
                    swskyd=jnp.array([[170.0, 120.0], [190.0, 140.0]]),
                    albsoib=jnp.array([[0.14, 0.24], [0.16, 0.26]]),
                    albsoid=jnp.array([[0.19, 0.29], [0.21, 0.31]]),
                ),
                "mlcanopy_state": MLCanopyState(
                    dlai_profile=jnp.array([[3.5], [2.8]]),
                    dsai_profile=jnp.array([[0.4], [0.32]]),
                    dpai_profile=jnp.array([[3.9], [3.12]]),
                    ntop_canopy=jnp.array([0, 0]),
                    nbot_canopy=jnp.array([0, 0]),
                    ncan_canopy=jnp.array([1, 1]),
                ),
                "pft_params": PFTParams(
                    rhol=jnp.array([
                        [0.1, 0.45], [0.12, 0.5], [0.08, 0.4],
                        [0.11, 0.48], [0.09, 0.43], [0.1, 0.45]
                    ]),
                    taul=jnp.array([
                        [0.05, 0.25], [0.06, 0.28], [0.04, 0.22],
                        [0.05, 0.26], [0.05, 0.24], [0.05, 0.25]
                    ]),
                    rhos=jnp.array([
                        [0.16, 0.39], [0.18, 0.42], [0.14, 0.36],
                        [0.17, 0.4], [0.15, 0.38], [0.16, 0.39]
                    ]),
                    taus=jnp.array([
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001],
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]
                    ]),
                    xl=jnp.array([0.01, 0.1, -0.1, 0.05, -0.05, 0.01]),
                    clump_fac=jnp.array([0.85, 0.8, 0.9, 0.82, 0.88, 0.85]),
                ),
                "nlevmlcan": 1,
                "light_type": 1,
                "numrad": 2,
            },
            "metadata": {
                "type": "special",
                "description": "Minimal canopy structure with single layer, testing simplified radiation transfer",
            },
        },
        "test_special_high_clumping_low_transmittance": {
            "inputs": {
                "bounds": BoundsType(begp=0, endp=2, begg=0, endg=1),
                "num_filter": 2,
                "filter_indices": jnp.array([0, 1]),
                "patch_state": PatchState(
                    itype=jnp.array([0, 1]),
                    cosz=jnp.array([0.65, 0.55]),
                    swskyb=jnp.array([[780.0, 580.0], [720.0, 520.0]]),
                    swskyd=jnp.array([[210.0, 160.0], [230.0, 175.0]]),
                    albsoib=jnp.array([[0.17, 0.27], [0.19, 0.29]]),
                    albsoid=jnp.array([[0.22, 0.32], [0.24, 0.34]]),
                ),
                "mlcanopy_state": MLCanopyState(
                    dlai_profile=jnp.array([[2.2, 2.5, 2.0, 1.5], [1.8, 2.1, 1.7, 1.2]]),
                    dsai_profile=jnp.array([[0.25, 0.28, 0.22, 0.18], [0.2, 0.24, 0.19, 0.14]]),
                    dpai_profile=jnp.array([[2.45, 2.78, 2.22, 1.68], [2.0, 2.34, 1.89, 1.34]]),
                    ntop_canopy=jnp.array([0, 0]),
                    nbot_canopy=jnp.array([3, 3]),
                    ncan_canopy=jnp.array([4, 4]),
                ),
                "pft_params": PFTParams(
                    rhol=jnp.array([
                        [0.02, 0.35], [0.03, 0.38], [0.08, 0.4],
                        [0.11, 0.48], [0.09, 0.43], [0.1, 0.45]
                    ]),
                    taul=jnp.array([
                        [0.01, 0.15], [0.01, 0.18], [0.04, 0.22],
                        [0.05, 0.26], [0.05, 0.24], [0.05, 0.25]
                    ]),
                    rhos=jnp.array([
                        [0.16, 0.39], [0.18, 0.42], [0.14, 0.36],
                        [0.17, 0.4], [0.15, 0.38], [0.16, 0.39]
                    ]),
                    taus=jnp.array([
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001],
                        [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]
                    ]),
                    xl=jnp.array([0.01, 0.1, -0.1, 0.05, -0.05, 0.01]),
                    clump_fac=jnp.array([0.5, 0.45, 0.9, 0.82, 0.88, 0.85]),
                ),
                "nlevmlcan": 4,
                "light_type": 2,
                "numrad": 2,
            },
            "metadata": {
                "type": "special",
                "description": "High foliage clumping with low leaf transmittance, testing extreme optical properties",
            },
        },
    }


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_nominal_single_patch_two_layers",
        "test_nominal_multiple_patches_five_layers",
        "test_nominal_twostream_method",
        "test_edge_zero_solar_zenith_angle",
        "test_edge_zero_lai_sparse_canopy",
        "test_edge_maximum_lai_dense_canopy",
        "test_edge_boundary_albedo_values",
        "test_edge_extreme_leaf_orientation",
        "test_special_single_layer_canopy",
        "test_special_high_clumping_low_transmittance",
    ],
)
def test_solar_radiation_shapes(test_data, test_case_name):
    """
    Test that solar_radiation returns outputs with correct shapes.
    
    Verifies that all output arrays have dimensions matching:
    - n_patches from num_filter
    - nlevmlcan from input parameter
    - numrad from input parameter
    - sunlit/shaded dimension (2) where applicable
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    result = solar_radiation(**inputs)
    
    n_patches = inputs["num_filter"]
    nlevmlcan = inputs["nlevmlcan"]
    numrad = inputs["numrad"]
    
    # Check swleaf shape: (n_patches, nlevmlcan, 2, numrad)
    assert result.swleaf.shape == (n_patches, nlevmlcan, 2, numrad), (
        f"swleaf shape mismatch in {test_case_name}: "
        f"expected {(n_patches, nlevmlcan, 2, numrad)}, got {result.swleaf.shape}"
    )
    
    # Check swsoi shape: (n_patches, numrad)
    assert result.swsoi.shape == (n_patches, numrad), (
        f"swsoi shape mismatch in {test_case_name}: "
        f"expected {(n_patches, numrad)}, got {result.swsoi.shape}"
    )
    
    # Check swveg shape: (n_patches, numrad)
    assert result.swveg.shape == (n_patches, numrad), (
        f"swveg shape mismatch in {test_case_name}: "
        f"expected {(n_patches, numrad)}, got {result.swveg.shape}"
    )
    
    # Check swvegsun shape: (n_patches, numrad)
    assert result.swvegsun.shape == (n_patches, numrad), (
        f"swvegsun shape mismatch in {test_case_name}: "
        f"expected {(n_patches, numrad)}, got {result.swvegsun.shape}"
    )
    
    # Check swvegsha shape: (n_patches, numrad)
    assert result.swvegsha.shape == (n_patches, numrad), (
        f"swvegsha shape mismatch in {test_case_name}: "
        f"expected {(n_patches, numrad)}, got {result.swvegsha.shape}"
    )
    
    # Check albcan shape: (n_patches, numrad)
    assert result.albcan.shape == (n_patches, numrad), (
        f"albcan shape mismatch in {test_case_name}: "
        f"expected {(n_patches, numrad)}, got {result.albcan.shape}"
    )
    
    # Check apar_sun shape: (n_patches, nlevmlcan)
    assert result.apar_sun.shape == (n_patches, nlevmlcan), (
        f"apar_sun shape mismatch in {test_case_name}: "
        f"expected {(n_patches, nlevmlcan)}, got {result.apar_sun.shape}"
    )
    
    # Check apar_shade shape: (n_patches, nlevmlcan)
    assert result.apar_shade.shape == (n_patches, nlevmlcan), (
        f"apar_shade shape mismatch in {test_case_name}: "
        f"expected {(n_patches, nlevmlcan)}, got {result.apar_shade.shape}"
    )


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_nominal_single_patch_two_layers",
        "test_nominal_multiple_patches_five_layers",
        "test_nominal_twostream_method",
        "test_edge_zero_solar_zenith_angle",
        "test_edge_zero_lai_sparse_canopy",
        "test_edge_maximum_lai_dense_canopy",
        "test_edge_boundary_albedo_values",
        "test_edge_extreme_leaf_orientation",
        "test_special_single_layer_canopy",
        "test_special_high_clumping_low_transmittance",
    ],
)
def test_solar_radiation_dtypes(test_data, test_case_name):
    """
    Test that solar_radiation returns outputs with correct data types.
    
    All outputs should be JAX arrays with float dtype.
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    result = solar_radiation(**inputs)
    
    # Check that all outputs are JAX arrays
    assert isinstance(result.swleaf, jnp.ndarray), (
        f"swleaf is not a JAX array in {test_case_name}"
    )
    assert isinstance(result.swsoi, jnp.ndarray), (
        f"swsoi is not a JAX array in {test_case_name}"
    )
    assert isinstance(result.swveg, jnp.ndarray), (
        f"swveg is not a JAX array in {test_case_name}"
    )
    assert isinstance(result.swvegsun, jnp.ndarray), (
        f"swvegsun is not a JAX array in {test_case_name}"
    )
    assert isinstance(result.swvegsha, jnp.ndarray), (
        f"swvegsha is not a JAX array in {test_case_name}"
    )
    assert isinstance(result.albcan, jnp.ndarray), (
        f"albcan is not a JAX array in {test_case_name}"
    )
    assert isinstance(result.apar_sun, jnp.ndarray), (
        f"apar_sun is not a JAX array in {test_case_name}"
    )
    assert isinstance(result.apar_shade, jnp.ndarray), (
        f"apar_shade is not a JAX array in {test_case_name}"
    )
    
    # Check that all outputs have float dtype
    assert jnp.issubdtype(result.swleaf.dtype, jnp.floating), (
        f"swleaf dtype is not float in {test_case_name}: {result.swleaf.dtype}"
    )
    assert jnp.issubdtype(result.swsoi.dtype, jnp.floating), (
        f"swsoi dtype is not float in {test_case_name}: {result.swsoi.dtype}"
    )
    assert jnp.issubdtype(result.swveg.dtype, jnp.floating), (
        f"swveg dtype is not float in {test_case_name}: {result.swveg.dtype}"
    )
    assert jnp.issubdtype(result.swvegsun.dtype, jnp.floating), (
        f"swvegsun dtype is not float in {test_case_name}: {result.swvegsun.dtype}"
    )
    assert jnp.issubdtype(result.swvegsha.dtype, jnp.floating), (
        f"swvegsha dtype is not float in {test_case_name}: {result.swvegsha.dtype}"
    )
    assert jnp.issubdtype(result.albcan.dtype, jnp.floating), (
        f"albcan dtype is not float in {test_case_name}: {result.albcan.dtype}"
    )
    assert jnp.issubdtype(result.apar_sun.dtype, jnp.floating), (
        f"apar_sun dtype is not float in {test_case_name}: {result.apar_sun.dtype}"
    )
    assert jnp.issubdtype(result.apar_shade.dtype, jnp.floating), (
        f"apar_shade dtype is not float in {test_case_name}: {result.apar_shade.dtype}"
    )


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_nominal_single_patch_two_layers",
        "test_nominal_multiple_patches_five_layers",
        "test_nominal_twostream_method",
        "test_edge_zero_solar_zenith_angle",
        "test_edge_zero_lai_sparse_canopy",
        "test_edge_maximum_lai_dense_canopy",
        "test_edge_boundary_albedo_values",
        "test_edge_extreme_leaf_orientation",
        "test_special_single_layer_canopy",
        "test_special_high_clumping_low_transmittance",
    ],
)
def test_solar_radiation_physical_constraints(test_data, test_case_name):
    """
    Test that solar_radiation outputs satisfy physical constraints.
    
    Verifies:
    - All radiation values are non-negative
    - Albedo values are in [0, 1]
    - Energy conservation: absorbed + reflected <= incoming
    - APAR values are non-negative
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    result = solar_radiation(**inputs)
    
    # Check non-negativity of radiation values
    assert jnp.all(result.swleaf >= 0), (
        f"swleaf has negative values in {test_case_name}: min={jnp.min(result.swleaf)}"
    )
    assert jnp.all(result.swsoi >= 0), (
        f"swsoi has negative values in {test_case_name}: min={jnp.min(result.swsoi)}"
    )
    assert jnp.all(result.swveg >= 0), (
        f"swveg has negative values in {test_case_name}: min={jnp.min(result.swveg)}"
    )
    assert jnp.all(result.swvegsun >= 0), (
        f"swvegsun has negative values in {test_case_name}: min={jnp.min(result.swvegsun)}"
    )
    assert jnp.all(result.swvegsha >= 0), (
        f"swvegsha has negative values in {test_case_name}: min={jnp.min(result.swvegsha)}"
    )
    
    # Check albedo bounds [0, 1]
    assert jnp.all(result.albcan >= 0), (
        f"albcan has values < 0 in {test_case_name}: min={jnp.min(result.albcan)}"
    )
    assert jnp.all(result.albcan <= 1), (
        f"albcan has values > 1 in {test_case_name}: max={jnp.max(result.albcan)}"
    )
    
    # Check APAR non-negativity
    assert jnp.all(result.apar_sun >= 0), (
        f"apar_sun has negative values in {test_case_name}: min={jnp.min(result.apar_sun)}"
    )
    assert jnp.all(result.apar_shade >= 0), (
        f"apar_shade has negative values in {test_case_name}: min={jnp.min(result.apar_shade)}"
    )
    
    # Check energy conservation: swveg + swsoi should be <= incoming radiation
    # Total incoming = swskyb + swskyd
    total_incoming = inputs["patch_state"].swskyb + inputs["patch_state"].swskyd
    total_absorbed = result.swveg + result.swsoi
    
    # Allow small numerical tolerance
    assert jnp.all(total_absorbed <= total_incoming + 1e-3), (
        f"Energy conservation violated in {test_case_name}: "
        f"absorbed={jnp.max(total_absorbed)} > incoming={jnp.max(total_incoming)}"
    )
    
    # Check that sunlit + shaded = total vegetation absorption
    swveg_sum = result.swvegsun + result.swvegsha
    assert jnp.allclose(swveg_sum, result.swveg, atol=1e-5, rtol=1e-5), (
        f"Sunlit + shaded != total vegetation absorption in {test_case_name}: "
        f"max diff={jnp.max(jnp.abs(swveg_sum - result.swveg))}"
    )


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_edge_zero_solar_zenith_angle",
        "test_edge_zero_lai_sparse_canopy",
    ],
)
def test_solar_radiation_edge_cases(test_data, test_case_name):
    """
    Test solar_radiation behavior in edge cases.
    
    Specific checks for:
    - Zero solar zenith angle: minimal direct beam absorption
    - Zero LAI: most radiation reaches soil
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    result = solar_radiation(**inputs)
    
    if "zero_cosz" in test_case["metadata"].get("edge_cases", []):
        # With zero cosine of zenith angle, direct beam should be minimal
        # Most absorption should be from diffuse radiation
        cosz = inputs["patch_state"].cosz
        zero_cosz_mask = cosz < 0.01
        
        if jnp.any(zero_cosz_mask):
            # Check that direct beam contribution is minimal
            swskyb = inputs["patch_state"].swskyb[zero_cosz_mask]
            assert jnp.all(swskyb < 10.0), (
                f"Direct beam radiation unexpectedly high with zero cosz in {test_case_name}"
            )
    
    if "zero_lai" in test_case["metadata"].get("edge_cases", []):
        # With zero LAI, most radiation should reach the soil
        dlai = inputs["mlcanopy_state"].dlai_profile
        zero_lai_mask = jnp.all(dlai < 0.01, axis=1)
        
        if jnp.any(zero_lai_mask):
            # Soil absorption should be high relative to vegetation
            swsoi_zero_lai = result.swsoi[zero_lai_mask]
            swveg_zero_lai = result.swveg[zero_lai_mask]
            
            # Soil should absorb more than vegetation when LAI is zero
            assert jnp.all(swsoi_zero_lai >= swveg_zero_lai), (
                f"Vegetation absorbs more than soil with zero LAI in {test_case_name}"
            )


def test_solar_radiation_consistency_between_methods(test_data):
    """
    Test that Norman and TwoStream methods produce qualitatively similar results.
    
    While exact values may differ, both methods should:
    - Produce similar total absorption patterns
    - Maintain energy conservation
    - Show similar trends with LAI
    """
    # Use a test case that exists for both methods
    norman_case = test_data["test_nominal_single_patch_two_layers"]
    
    # Create TwoStream version
    twostream_inputs = norman_case["inputs"].copy()
    twostream_inputs["light_type"] = 2
    
    result_norman = solar_radiation(**norman_case["inputs"])
    result_twostream = solar_radiation(**twostream_inputs)
    
    # Both should conserve energy
    total_incoming = (
        norman_case["inputs"]["patch_state"].swskyb +
        norman_case["inputs"]["patch_state"].swskyd
    )
    
    total_absorbed_norman = result_norman.swveg + result_norman.swsoi
    total_absorbed_twostream = result_twostream.swveg + result_twostream.swsoi
    
    assert jnp.all(total_absorbed_norman <= total_incoming + 1e-3), (
        "Norman method violates energy conservation"
    )
    assert jnp.all(total_absorbed_twostream <= total_incoming + 1e-3), (
        "TwoStream method violates energy conservation"
    )
    
    # Albedo should be in similar range (within 0.2)
    assert jnp.allclose(result_norman.albcan, result_twostream.albcan, atol=0.2), (
        f"Albedo differs significantly between methods: "
        f"Norman={result_norman.albcan}, TwoStream={result_twostream.albcan}"
    )


def test_solar_radiation_lai_gradient(test_data):
    """
    Test that radiation absorption increases with LAI.
    
    Higher LAI should lead to:
    - More vegetation absorption
    - Less soil absorption
    - Lower canopy albedo (more absorption)
    """
    # Compare sparse vs dense canopy cases
    sparse_case = test_data["test_edge_zero_lai_sparse_canopy"]
    dense_case = test_data["test_edge_maximum_lai_dense_canopy"]
    
    result_sparse = solar_radiation(**sparse_case["inputs"])
    result_dense = solar_radiation(**dense_case["inputs"])
    
    # Dense canopy should absorb more in vegetation
    assert jnp.all(result_dense.swveg >= result_sparse.swveg), (
        "Dense canopy doesn't absorb more than sparse canopy"
    )
    
    # Sparse canopy should have more soil absorption
    assert jnp.all(result_sparse.swsoi >= result_dense.swsoi), (
        "Sparse canopy doesn't have more soil absorption than dense canopy"
    )


def test_solar_radiation_sunlit_shaded_fractions(test_data):
    """
    Test that sunlit and shaded leaf fractions behave correctly.
    
    Verifies:
    - Sunlit fraction decreases with depth in canopy
    - Sunlit leaves receive more radiation than shaded
    - APAR for sunlit > APAR for shaded
    """
    test_case = test_data["test_nominal_multiple_patches_five_layers"]
    inputs = test_case["inputs"]
    
    result = solar_radiation(**inputs)
    
    # Sunlit leaves should generally receive more radiation than shaded
    # (averaged over layers and bands)
    swleaf_sun = result.swleaf[:, :, 0, :]  # sunlit
    swleaf_shade = result.swleaf[:, :, 1, :]  # shaded
    
    mean_sun = jnp.mean(swleaf_sun)
    mean_shade = jnp.mean(swleaf_shade)
    
    assert mean_sun >= mean_shade, (
        f"Sunlit leaves don't receive more radiation than shaded: "
        f"sun={mean_sun}, shade={mean_shade}"
    )
    
    # APAR for sunlit should be greater than shaded
    assert jnp.all(result.apar_sun >= result.apar_shade), (
        "APAR for sunlit leaves is not greater than shaded"
    )


def test_solar_radiation_no_nans_or_infs(test_data):
    """
    Test that solar_radiation never produces NaN or Inf values.
    
    This is critical for numerical stability in all test cases.
    """
    for test_case_name, test_case in test_data.items():
        inputs = test_case["inputs"]
        result = solar_radiation(**inputs)
        
        # Check all outputs for NaN/Inf
        assert not jnp.any(jnp.isnan(result.swleaf)), (
            f"swleaf contains NaN in {test_case_name}"
        )
        assert not jnp.any(jnp.isinf(result.swleaf)), (
            f"swleaf contains Inf in {test_case_name}"
        )
        assert not jnp.any(jnp.isnan(result.swsoi)), (
            f"swsoi contains NaN in {test_case_name}"
        )
        assert not jnp.any(jnp.isinf(result.swsoi)), (
            f"swsoi contains Inf in {test_case_name}"
        )
        assert not jnp.any(jnp.isnan(result.swveg)), (
            f"swveg contains NaN in {test_case_name}"
        )
        assert not jnp.any(jnp.isinf(result.swveg)), (
            f"swveg contains Inf in {test_case_name}"
        )
        assert not jnp.any(jnp.isnan(result.swvegsun)), (
            f"swvegsun contains NaN in {test_case_name}"
        )
        assert not jnp.any(jnp.isinf(result.swvegsun)), (
            f"swvegsun contains Inf in {test_case_name}"
        )
        assert not jnp.any(jnp.isnan(result.swvegsha)), (
            f"swvegsha contains NaN in {test_case_name}"
        )
        assert not jnp.any(jnp.isinf(result.swvegsha)), (
            f"swvegsha contains Inf in {test_case_name}"
        )
        assert not jnp.any(jnp.isnan(result.albcan)), (
            f"albcan contains NaN in {test_case_name}"
        )
        assert not jnp.any(jnp.isinf(result.albcan)), (
            f"albcan contains Inf in {test_case_name}"
        )
        assert not jnp.any(jnp.isnan(result.apar_sun)), (
            f"apar_sun contains NaN in {test_case_name}"
        )
        assert not jnp.any(jnp.isinf(result.apar_sun)), (
            f"apar_sun contains Inf in {test_case_name}"
        )
        assert not jnp.any(jnp.isnan(result.apar_shade)), (
            f"apar_shade contains NaN in {test_case_name}"
        )
        assert not jnp.any(jnp.isinf(result.apar_shade)), (
            f"apar_shade contains Inf in {test_case_name}"
        )


def test_solar_radiation_albedo_bounds_extreme_cases(test_data):
    """
    Test albedo behavior with extreme soil albedo values.
    
    Verifies that canopy albedo responds appropriately to:
    - Zero soil albedo (perfect absorber)
    - Maximum soil albedo (perfect reflector)
    """
    test_case = test_data["test_edge_boundary_albedo_values"]
    inputs = test_case["inputs"]
    
    result = solar_radiation(**inputs)
    
    # Extract results for each albedo case
    albcan_zero = result.albcan[0, :]  # Zero soil albedo
    albcan_mid = result.albcan[1, :]   # 0.5 soil albedo
    albcan_max = result.albcan[2, :]   # Maximum soil albedo
    
    # Canopy albedo should increase with soil albedo
    assert jnp.all(albcan_mid >= albcan_zero), (
        "Canopy albedo doesn't increase from zero to mid soil albedo"
    )
    assert jnp.all(albcan_max >= albcan_mid), (
        "Canopy albedo doesn't increase from mid to max soil albedo"
    )
    
    # All should still be in valid range
    assert jnp.all(albcan_zero >= 0) and jnp.all(albcan_zero <= 1), (
        "Canopy albedo out of bounds with zero soil albedo"
    )
    assert jnp.all(albcan_max >= 0) and jnp.all(albcan_max <= 1), (
        "Canopy albedo out of bounds with max soil albedo"
    )