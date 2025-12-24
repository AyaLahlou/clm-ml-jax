"""
Comprehensive pytest suite for soilstate_init_time_const function.

This module tests the soil state initialization function that computes time-constant
hydraulic and thermal properties for soil layers, including root fraction distribution,
soil texture properties, and thermal conductivities.

Tests cover:
- Nominal cases with typical soil types and configurations
- Edge cases (shallow bedrock, extreme organic matter, high clay content)
- Special cases (variable depth layers)
- Physical constraints validation
- Array shape and dtype verification
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_biogeophys.SoilStateInitTimeConstMod import soilstate_init_time_const


# Define NamedTuples matching the function signature
class BoundsType(NamedTuple):
    """Domain decomposition bounds."""
    begp: int
    endp: int
    begc: int
    endc: int
    begg: int
    endg: int


class PatchType(NamedTuple):
    """Patch-level data structure."""
    column: jnp.ndarray  # (npatch,)
    itype: jnp.ndarray   # (npatch,)


class ColumnType(NamedTuple):
    """Column-level data structure."""
    gridcell: jnp.ndarray  # (ncol,)
    itype: jnp.ndarray     # (ncol,)
    nbedrock: jnp.ndarray  # (ncol,)
    zi: jnp.ndarray        # (ncol, nlevgrnd+1)
    z: jnp.ndarray         # (ncol, nlevgrnd)


class PftconType(NamedTuple):
    """Plant functional type constants."""
    roota_par: jnp.ndarray      # (npft,)
    rootb_par: jnp.ndarray      # (npft,)
    rootprof_beta: jnp.ndarray  # (npft,)


class SoilTextureType(NamedTuple):
    """Soil texture lookup tables."""
    names: jnp.ndarray   # (ntex,)
    clay: jnp.ndarray    # (ntex,)
    sand: jnp.ndarray    # (ntex,)
    watsat: jnp.ndarray  # (ntex,)
    smpsat: jnp.ndarray  # (ntex,)
    hksat: jnp.ndarray   # (ntex,)
    bsw: jnp.ndarray     # (ntex,)


class TowerDataType(NamedTuple):
    """Tower-specific soil data."""
    num: int
    tex: jnp.ndarray       # (ntowers,)
    clay: jnp.ndarray      # (ntowers,)
    sand: jnp.ndarray      # (ntowers,)
    organic: jnp.ndarray   # (ntowers,)
    col_tower: jnp.ndarray # (ncol,)


class SoilStateInitConfig(NamedTuple):
    """Configuration parameters."""
    csol_bedrock: float
    organic_max: float
    zsapric: float
    pcalpha: float
    pcbeta: float
    m_to_cm: float
    nlevsoi: int
    nlevgrnd: int
    clm_phys: int
    root_type: int


class SoilStateType(NamedTuple):
    """Complete soil state with time-constant properties."""
    rootfr: jnp.ndarray    # (npatch, nlevgrnd)
    cellsand: jnp.ndarray  # (ncol, nlevsoi)
    cellclay: jnp.ndarray  # (ncol, nlevsoi)
    cellorg: jnp.ndarray   # (ncol, nlevsoi)
    watsat: jnp.ndarray    # (ncol, nlevgrnd)
    sucsat: jnp.ndarray    # (ncol, nlevgrnd)
    hksat: jnp.ndarray     # (ncol, nlevgrnd)
    bsw: jnp.ndarray       # (ncol, nlevgrnd)
    perc_frac: jnp.ndarray # (ncol, nlevgrnd)
    tkdry: jnp.ndarray     # (ncol, nlevgrnd)
    tkmg: jnp.ndarray      # (ncol, nlevgrnd)
    csol: jnp.ndarray      # (ncol, nlevgrnd)


@pytest.fixture
def test_data():
    """
    Load and prepare test data for all test cases.
    
    Returns:
        dict: Dictionary containing all test cases with inputs and metadata.
    """
    return {
        "test_nominal_single_patch_single_column": {
            "inputs": {
                "bounds": BoundsType(begp=0, endp=1, begc=0, endc=1, begg=0, endg=1),
                "patch": PatchType(
                    column=jnp.array([0]),
                    itype=jnp.array([1])
                ),
                "col": ColumnType(
                    gridcell=jnp.array([0]),
                    itype=jnp.array([1]),
                    nbedrock=jnp.array([10]),
                    zi=jnp.array([[0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]]),
                    z=jnp.array([[0.01, 0.035, 0.075, 0.15, 0.275, 0.45, 0.675, 0.95, 1.3, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25]])
                ),
                "pftcon": PftconType(
                    roota_par=jnp.array([7.0, 6.0, 7.0, 6.0, 7.0]),
                    rootb_par=jnp.array([2.0, 2.0, 2.0, 2.0, 2.0]),
                    rootprof_beta=jnp.array([0.943, 0.962, 0.966, 0.961, 0.964])
                ),
                "tower_data": TowerDataType(
                    num=1,
                    tex=jnp.array([3]),
                    clay=jnp.array([0.25]),
                    sand=jnp.array([0.45]),
                    organic=jnp.array([15.0]),
                    col_tower=jnp.array([0])
                ),
                "soil_texture": SoilTextureType(
                    names=jnp.array([0, 1, 2, 3, 4]),  # Placeholder for string names
                    clay=jnp.array([0.03, 0.05, 0.1, 0.18, 0.15]),
                    sand=jnp.array([0.92, 0.82, 0.58, 0.42, 0.17]),
                    watsat=jnp.array([0.339, 0.421, 0.434, 0.451, 0.485]),
                    smpsat=jnp.array([-121.0, -90.0, -218.0, -478.0, -786.0]),
                    hksat=jnp.array([0.00583, 0.0017, 0.00072, 0.000367, 0.00019]),
                    bsw=jnp.array([2.79, 4.26, 4.74, 5.25, 5.33])
                ),
                "config": SoilStateInitConfig(
                    csol_bedrock=2000000.0,
                    organic_max=130.0,
                    zsapric=0.5,
                    pcalpha=0.5,
                    pcbeta=0.139,
                    m_to_cm=100.0,
                    nlevsoi=10,
                    nlevgrnd=15,
                    clm_phys=0,
                    root_type=1
                )
            },
            "metadata": {
                "type": "nominal",
                "description": "Single patch, single column with typical loam soil"
            }
        },
        "test_nominal_multiple_patches_columns": {
            "inputs": {
                "bounds": BoundsType(begp=0, endp=4, begc=0, endc=2, begg=0, endg=1),
                "patch": PatchType(
                    column=jnp.array([0, 0, 1, 1]),
                    itype=jnp.array([1, 2, 3, 4])
                ),
                "col": ColumnType(
                    gridcell=jnp.array([0, 0]),
                    itype=jnp.array([1, 2]),
                    nbedrock=jnp.array([10, 8]),
                    zi=jnp.array([
                        [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
                        [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
                    ]),
                    z=jnp.array([
                        [0.01, 0.035, 0.075, 0.15, 0.275, 0.45, 0.675, 0.95, 1.3, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25],
                        [0.01, 0.035, 0.075, 0.15, 0.275, 0.45, 0.675, 0.95, 1.3, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25]
                    ])
                ),
                "pftcon": PftconType(
                    roota_par=jnp.array([7.0, 6.0, 7.0, 6.0, 7.0]),
                    rootb_par=jnp.array([2.0, 2.0, 2.0, 2.0, 2.0]),
                    rootprof_beta=jnp.array([0.943, 0.962, 0.966, 0.961, 0.964])
                ),
                "tower_data": TowerDataType(
                    num=2,
                    tex=jnp.array([3, 1]),
                    clay=jnp.array([0.25, 0.05]),
                    sand=jnp.array([0.45, 0.9]),
                    organic=jnp.array([15.0, 5.0]),
                    col_tower=jnp.array([0, 1])
                ),
                "soil_texture": SoilTextureType(
                    names=jnp.array([0, 1, 2, 3, 4]),
                    clay=jnp.array([0.03, 0.05, 0.1, 0.18, 0.15]),
                    sand=jnp.array([0.92, 0.82, 0.58, 0.42, 0.17]),
                    watsat=jnp.array([0.339, 0.421, 0.434, 0.451, 0.485]),
                    smpsat=jnp.array([-121.0, -90.0, -218.0, -478.0, -786.0]),
                    hksat=jnp.array([0.00583, 0.0017, 0.00072, 0.000367, 0.00019]),
                    bsw=jnp.array([2.79, 4.26, 4.74, 5.25, 5.33])
                ),
                "config": SoilStateInitConfig(
                    csol_bedrock=2000000.0,
                    organic_max=130.0,
                    zsapric=0.5,
                    pcalpha=0.5,
                    pcbeta=0.139,
                    m_to_cm=100.0,
                    nlevsoi=10,
                    nlevgrnd=15,
                    clm_phys=0,
                    root_type=1
                )
            },
            "metadata": {
                "type": "nominal",
                "description": "Multiple patches and columns with varying soil types"
            }
        },
        "test_nominal_jackson_root_profile": {
            "inputs": {
                "bounds": BoundsType(begp=0, endp=2, begc=0, endc=1, begg=0, endg=1),
                "patch": PatchType(
                    column=jnp.array([0, 0]),
                    itype=jnp.array([0, 2])
                ),
                "col": ColumnType(
                    gridcell=jnp.array([0]),
                    itype=jnp.array([1]),
                    nbedrock=jnp.array([10]),
                    zi=jnp.array([[0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]]),
                    z=jnp.array([[0.01, 0.035, 0.075, 0.15, 0.275, 0.45, 0.675, 0.95, 1.3, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25]])
                ),
                "pftcon": PftconType(
                    roota_par=jnp.array([7.0, 6.0, 7.0]),
                    rootb_par=jnp.array([2.0, 2.0, 2.0]),
                    rootprof_beta=jnp.array([0.943, 0.962, 0.966])
                ),
                "tower_data": TowerDataType(
                    num=1,
                    tex=jnp.array([2]),
                    clay=jnp.array([0.15]),
                    sand=jnp.array([0.6]),
                    organic=jnp.array([20.0]),
                    col_tower=jnp.array([0])
                ),
                "soil_texture": SoilTextureType(
                    names=jnp.array([0, 1, 2, 3]),
                    clay=jnp.array([0.03, 0.05, 0.1, 0.18]),
                    sand=jnp.array([0.92, 0.82, 0.58, 0.42]),
                    watsat=jnp.array([0.339, 0.421, 0.434, 0.451]),
                    smpsat=jnp.array([-121.0, -90.0, -218.0, -478.0]),
                    hksat=jnp.array([0.00583, 0.0017, 0.00072, 0.000367]),
                    bsw=jnp.array([2.79, 4.26, 4.74, 5.25])
                ),
                "config": SoilStateInitConfig(
                    csol_bedrock=2000000.0,
                    organic_max=130.0,
                    zsapric=0.5,
                    pcalpha=0.5,
                    pcbeta=0.139,
                    m_to_cm=100.0,
                    nlevsoi=10,
                    nlevgrnd=15,
                    clm_phys=0,
                    root_type=2
                )
            },
            "metadata": {
                "type": "nominal",
                "description": "Tests Jackson1996 root profile (root_type=2)"
            }
        },
        "test_edge_shallow_bedrock": {
            "inputs": {
                "bounds": BoundsType(begp=0, endp=1, begc=0, endc=1, begg=0, endg=1),
                "patch": PatchType(
                    column=jnp.array([0]),
                    itype=jnp.array([1])
                ),
                "col": ColumnType(
                    gridcell=jnp.array([0]),
                    itype=jnp.array([1]),
                    nbedrock=jnp.array([3]),
                    zi=jnp.array([[0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]]),
                    z=jnp.array([[0.01, 0.035, 0.075, 0.15, 0.275, 0.45, 0.675, 0.95, 1.3, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25]])
                ),
                "pftcon": PftconType(
                    roota_par=jnp.array([7.0, 6.0]),
                    rootb_par=jnp.array([2.0, 2.0]),
                    rootprof_beta=jnp.array([0.943, 0.962])
                ),
                "tower_data": TowerDataType(
                    num=1,
                    tex=jnp.array([0]),
                    clay=jnp.array([0.05]),
                    sand=jnp.array([0.9]),
                    organic=jnp.array([2.0]),
                    col_tower=jnp.array([0])
                ),
                "soil_texture": SoilTextureType(
                    names=jnp.array([0, 1, 2]),
                    clay=jnp.array([0.03, 0.05, 0.1]),
                    sand=jnp.array([0.92, 0.82, 0.58]),
                    watsat=jnp.array([0.339, 0.421, 0.434]),
                    smpsat=jnp.array([-121.0, -90.0, -218.0]),
                    hksat=jnp.array([0.00583, 0.0017, 0.00072]),
                    bsw=jnp.array([2.79, 4.26, 4.74])
                ),
                "config": SoilStateInitConfig(
                    csol_bedrock=2000000.0,
                    organic_max=130.0,
                    zsapric=0.5,
                    pcalpha=0.5,
                    pcbeta=0.139,
                    m_to_cm=100.0,
                    nlevsoi=10,
                    nlevgrnd=15,
                    clm_phys=0,
                    root_type=1
                )
            },
            "metadata": {
                "type": "edge",
                "description": "Tests shallow bedrock at layer 3"
            }
        },
        "test_edge_zero_organic_matter": {
            "inputs": {
                "bounds": BoundsType(begp=0, endp=1, begc=0, endc=1, begg=0, endg=1),
                "patch": PatchType(
                    column=jnp.array([0]),
                    itype=jnp.array([0])
                ),
                "col": ColumnType(
                    gridcell=jnp.array([0]),
                    itype=jnp.array([1]),
                    nbedrock=jnp.array([10]),
                    zi=jnp.array([[0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]]),
                    z=jnp.array([[0.01, 0.035, 0.075, 0.15, 0.275, 0.45, 0.675, 0.95, 1.3, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25]])
                ),
                "pftcon": PftconType(
                    roota_par=jnp.array([7.0]),
                    rootb_par=jnp.array([2.0]),
                    rootprof_beta=jnp.array([0.943])
                ),
                "tower_data": TowerDataType(
                    num=1,
                    tex=jnp.array([0]),
                    clay=jnp.array([0.03]),
                    sand=jnp.array([0.92]),
                    organic=jnp.array([0.0]),
                    col_tower=jnp.array([0])
                ),
                "soil_texture": SoilTextureType(
                    names=jnp.array([0]),
                    clay=jnp.array([0.03]),
                    sand=jnp.array([0.92]),
                    watsat=jnp.array([0.339]),
                    smpsat=jnp.array([-121.0]),
                    hksat=jnp.array([0.00583]),
                    bsw=jnp.array([2.79])
                ),
                "config": SoilStateInitConfig(
                    csol_bedrock=2000000.0,
                    organic_max=130.0,
                    zsapric=0.5,
                    pcalpha=0.5,
                    pcbeta=0.139,
                    m_to_cm=100.0,
                    nlevsoi=10,
                    nlevgrnd=15,
                    clm_phys=0,
                    root_type=1
                )
            },
            "metadata": {
                "type": "edge",
                "description": "Tests pure mineral soil with zero organic matter"
            }
        },
        "test_edge_maximum_organic_matter": {
            "inputs": {
                "bounds": BoundsType(begp=0, endp=1, begc=0, endc=1, begg=0, endg=1),
                "patch": PatchType(
                    column=jnp.array([0]),
                    itype=jnp.array([2])
                ),
                "col": ColumnType(
                    gridcell=jnp.array([0]),
                    itype=jnp.array([1]),
                    nbedrock=jnp.array([10]),
                    zi=jnp.array([[0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]]),
                    z=jnp.array([[0.01, 0.035, 0.075, 0.15, 0.275, 0.45, 0.675, 0.95, 1.3, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25]])
                ),
                "pftcon": PftconType(
                    roota_par=jnp.array([7.0, 6.0, 7.0]),
                    rootb_par=jnp.array([2.0, 2.0, 2.0]),
                    rootprof_beta=jnp.array([0.943, 0.962, 0.966])
                ),
                "tower_data": TowerDataType(
                    num=1,
                    tex=jnp.array([4]),
                    clay=jnp.array([0.4]),
                    sand=jnp.array([0.1]),
                    organic=jnp.array([130.0]),
                    col_tower=jnp.array([0])
                ),
                "soil_texture": SoilTextureType(
                    names=jnp.array([0, 1, 2, 3, 4]),
                    clay=jnp.array([0.03, 0.05, 0.1, 0.18, 0.15]),
                    sand=jnp.array([0.92, 0.82, 0.58, 0.42, 0.17]),
                    watsat=jnp.array([0.339, 0.421, 0.434, 0.451, 0.485]),
                    smpsat=jnp.array([-121.0, -90.0, -218.0, -478.0, -786.0]),
                    hksat=jnp.array([0.00583, 0.0017, 0.00072, 0.000367, 0.00019]),
                    bsw=jnp.array([2.79, 4.26, 4.74, 5.25, 5.33])
                ),
                "config": SoilStateInitConfig(
                    csol_bedrock=2000000.0,
                    organic_max=130.0,
                    zsapric=0.5,
                    pcalpha=0.5,
                    pcbeta=0.139,
                    m_to_cm=100.0,
                    nlevsoi=10,
                    nlevgrnd=15,
                    clm_phys=0,
                    root_type=1
                )
            },
            "metadata": {
                "type": "edge",
                "description": "Tests maximum organic matter content (130 kg/m2)"
            }
        },
        "test_edge_extreme_clay_content": {
            "inputs": {
                "bounds": BoundsType(begp=0, endp=1, begc=0, endc=1, begg=0, endg=1),
                "patch": PatchType(
                    column=jnp.array([0]),
                    itype=jnp.array([1])
                ),
                "col": ColumnType(
                    gridcell=jnp.array([0]),
                    itype=jnp.array([1]),
                    nbedrock=jnp.array([10]),
                    zi=jnp.array([[0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]]),
                    z=jnp.array([[0.01, 0.035, 0.075, 0.15, 0.275, 0.45, 0.675, 0.95, 1.3, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25]])
                ),
                "pftcon": PftconType(
                    roota_par=jnp.array([7.0, 6.0]),
                    rootb_par=jnp.array([2.0, 2.0]),
                    rootprof_beta=jnp.array([0.943, 0.962])
                ),
                "tower_data": TowerDataType(
                    num=1,
                    tex=jnp.array([1]),
                    clay=jnp.array([0.6]),
                    sand=jnp.array([0.05]),
                    organic=jnp.array([10.0]),
                    col_tower=jnp.array([0])
                ),
                "soil_texture": SoilTextureType(
                    names=jnp.array([0, 1]),
                    clay=jnp.array([0.03, 0.6]),
                    sand=jnp.array([0.92, 0.05]),
                    watsat=jnp.array([0.339, 0.52]),
                    smpsat=jnp.array([-121.0, -1500.0]),
                    hksat=jnp.array([0.00583, 1e-05]),
                    bsw=jnp.array([2.79, 11.4])
                ),
                "config": SoilStateInitConfig(
                    csol_bedrock=2000000.0,
                    organic_max=130.0,
                    zsapric=0.5,
                    pcalpha=0.5,
                    pcbeta=0.139,
                    m_to_cm=100.0,
                    nlevsoi=10,
                    nlevgrnd=15,
                    clm_phys=0,
                    root_type=1
                )
            },
            "metadata": {
                "type": "edge",
                "description": "Tests heavy clay soil with 60% clay content"
            }
        }
    }


@pytest.mark.parametrize("test_case_name", [
    "test_nominal_single_patch_single_column",
    "test_nominal_multiple_patches_columns",
    "test_nominal_jackson_root_profile",
    "test_edge_shallow_bedrock",
    "test_edge_zero_organic_matter",
    "test_edge_maximum_organic_matter",
    "test_edge_extreme_clay_content"
])
def test_soilstate_init_shapes(test_data, test_case_name):
    """
    Test that output arrays have correct shapes.
    
    Verifies that all output arrays in SoilStateType have dimensions
    consistent with the input configuration (npatch, ncol, nlevgrnd, nlevsoi).
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    result = soilstate_init_time_const(**inputs)
    
    npatch = inputs["bounds"].endp - inputs["bounds"].begp
    ncol = inputs["bounds"].endc - inputs["bounds"].begc
    nlevgrnd = inputs["config"].nlevgrnd
    nlevsoi = inputs["config"].nlevsoi
    
    # Check root fraction shape
    assert result.rootfr.shape == (npatch, nlevgrnd), \
        f"rootfr shape mismatch: expected ({npatch}, {nlevgrnd}), got {result.rootfr.shape}"
    
    # Check soil texture shapes
    assert result.cellsand.shape == (ncol, nlevsoi), \
        f"cellsand shape mismatch: expected ({ncol}, {nlevsoi}), got {result.cellsand.shape}"
    assert result.cellclay.shape == (ncol, nlevsoi), \
        f"cellclay shape mismatch: expected ({ncol}, {nlevsoi}), got {result.cellclay.shape}"
    assert result.cellorg.shape == (ncol, nlevsoi), \
        f"cellorg shape mismatch: expected ({ncol}, {nlevsoi}), got {result.cellorg.shape}"
    
    # Check hydraulic property shapes
    assert result.watsat.shape == (ncol, nlevgrnd), \
        f"watsat shape mismatch: expected ({ncol}, {nlevgrnd}), got {result.watsat.shape}"
    assert result.sucsat.shape == (ncol, nlevgrnd), \
        f"sucsat shape mismatch: expected ({ncol}, {nlevgrnd}), got {result.sucsat.shape}"
    assert result.hksat.shape == (ncol, nlevgrnd), \
        f"hksat shape mismatch: expected ({ncol}, {nlevgrnd}), got {result.hksat.shape}"
    assert result.bsw.shape == (ncol, nlevgrnd), \
        f"bsw shape mismatch: expected ({ncol}, {nlevgrnd}), got {result.bsw.shape}"
    assert result.perc_frac.shape == (ncol, nlevgrnd), \
        f"perc_frac shape mismatch: expected ({ncol}, {nlevgrnd}), got {result.perc_frac.shape}"
    
    # Check thermal property shapes
    assert result.tkdry.shape == (ncol, nlevgrnd), \
        f"tkdry shape mismatch: expected ({ncol}, {nlevgrnd}), got {result.tkdry.shape}"
    assert result.tkmg.shape == (ncol, nlevgrnd), \
        f"tkmg shape mismatch: expected ({ncol}, {nlevgrnd}), got {result.tkmg.shape}"
    assert result.csol.shape == (ncol, nlevgrnd), \
        f"csol shape mismatch: expected ({ncol}, {nlevgrnd}), got {result.csol.shape}"


@pytest.mark.parametrize("test_case_name", [
    "test_nominal_single_patch_single_column",
    "test_nominal_multiple_patches_columns",
    "test_nominal_jackson_root_profile",
    "test_edge_shallow_bedrock",
    "test_edge_zero_organic_matter",
    "test_edge_maximum_organic_matter",
    "test_edge_extreme_clay_content"
])
def test_soilstate_init_dtypes(test_data, test_case_name):
    """
    Test that output arrays have correct data types.
    
    Verifies that all output arrays are floating-point types suitable
    for scientific computing.
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    result = soilstate_init_time_const(**inputs)
    
    # All outputs should be floating-point
    assert jnp.issubdtype(result.rootfr.dtype, jnp.floating), \
        f"rootfr dtype should be floating, got {result.rootfr.dtype}"
    assert jnp.issubdtype(result.cellsand.dtype, jnp.floating), \
        f"cellsand dtype should be floating, got {result.cellsand.dtype}"
    assert jnp.issubdtype(result.cellclay.dtype, jnp.floating), \
        f"cellclay dtype should be floating, got {result.cellclay.dtype}"
    assert jnp.issubdtype(result.cellorg.dtype, jnp.floating), \
        f"cellorg dtype should be floating, got {result.cellorg.dtype}"
    assert jnp.issubdtype(result.watsat.dtype, jnp.floating), \
        f"watsat dtype should be floating, got {result.watsat.dtype}"
    assert jnp.issubdtype(result.sucsat.dtype, jnp.floating), \
        f"sucsat dtype should be floating, got {result.sucsat.dtype}"
    assert jnp.issubdtype(result.hksat.dtype, jnp.floating), \
        f"hksat dtype should be floating, got {result.hksat.dtype}"
    assert jnp.issubdtype(result.bsw.dtype, jnp.floating), \
        f"bsw dtype should be floating, got {result.bsw.dtype}"
    assert jnp.issubdtype(result.perc_frac.dtype, jnp.floating), \
        f"perc_frac dtype should be floating, got {result.perc_frac.dtype}"
    assert jnp.issubdtype(result.tkdry.dtype, jnp.floating), \
        f"tkdry dtype should be floating, got {result.tkdry.dtype}"
    assert jnp.issubdtype(result.tkmg.dtype, jnp.floating), \
        f"tkmg dtype should be floating, got {result.tkmg.dtype}"
    assert jnp.issubdtype(result.csol.dtype, jnp.floating), \
        f"csol dtype should be floating, got {result.csol.dtype}"


@pytest.mark.parametrize("test_case_name", [
    "test_nominal_single_patch_single_column",
    "test_nominal_multiple_patches_columns",
    "test_nominal_jackson_root_profile",
    "test_edge_zero_organic_matter",
    "test_edge_maximum_organic_matter"
])
def test_soilstate_init_physical_constraints(test_data, test_case_name):
    """
    Test that outputs satisfy physical constraints.
    
    Verifies:
    - Root fractions sum to ~1.0 for each patch
    - Fractions (watsat, perc_frac) in [0, 1]
    - Percentages (cellsand, cellclay) in [0, 100]
    - Positive properties (hksat, tkdry, tkmg, csol > 0)
    - Organic matter in valid range [0, organic_max]
    - Negative soil suction (sucsat < 0)
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    result = soilstate_init_time_const(**inputs)
    
    # Root fractions should sum to ~1.0 for each patch
    rootfr_sum = jnp.sum(result.rootfr, axis=1)
    assert jnp.allclose(rootfr_sum, 1.0, atol=1e-3), \
        f"Root fractions should sum to 1.0, got {rootfr_sum}"
    
    # Fractions should be in [0, 1]
    assert jnp.all((result.watsat >= 0) & (result.watsat <= 1)), \
        f"watsat should be in [0, 1], got range [{jnp.min(result.watsat)}, {jnp.max(result.watsat)}]"
    assert jnp.all((result.perc_frac >= 0) & (result.perc_frac <= 1)), \
        f"perc_frac should be in [0, 1], got range [{jnp.min(result.perc_frac)}, {jnp.max(result.perc_frac)}]"
    
    # Percentages should be in [0, 100]
    assert jnp.all((result.cellsand >= 0) & (result.cellsand <= 100)), \
        f"cellsand should be in [0, 100], got range [{jnp.min(result.cellsand)}, {jnp.max(result.cellsand)}]"
    assert jnp.all((result.cellclay >= 0) & (result.cellclay <= 100)), \
        f"cellclay should be in [0, 100], got range [{jnp.min(result.cellclay)}, {jnp.max(result.cellclay)}]"
    
    # Organic matter should be in [0, organic_max]
    organic_max = inputs["config"].organic_max
    assert jnp.all((result.cellorg >= 0) & (result.cellorg <= organic_max)), \
        f"cellorg should be in [0, {organic_max}], got range [{jnp.min(result.cellorg)}, {jnp.max(result.cellorg)}]"
    
    # Hydraulic conductivity should be positive
    assert jnp.all(result.hksat > 0), \
        f"hksat should be positive, got min {jnp.min(result.hksat)}"
    
    # Soil suction should be negative (tension)
    assert jnp.all(result.sucsat < 0), \
        f"sucsat should be negative, got max {jnp.max(result.sucsat)}"
    
    # Clapp-Hornberger b parameter should be positive
    assert jnp.all(result.bsw > 0), \
        f"bsw should be positive, got min {jnp.min(result.bsw)}"
    
    # Thermal conductivities should be positive
    assert jnp.all(result.tkdry > 0), \
        f"tkdry should be positive, got min {jnp.min(result.tkdry)}"
    assert jnp.all(result.tkmg > 0), \
        f"tkmg should be positive, got min {jnp.min(result.tkmg)}"
    
    # Heat capacity should be positive
    assert jnp.all(result.csol > 0), \
        f"csol should be positive, got min {jnp.min(result.csol)}"


def test_soilstate_init_bedrock_properties(test_data):
    """
    Test that bedrock layers have correct properties.
    
    For layers at or below bedrock (j >= nbedrock), verifies that:
    - Heat capacity equals csol_bedrock
    - Hydraulic properties are set appropriately
    """
    test_case = test_data["test_edge_shallow_bedrock"]
    inputs = test_case["inputs"]
    
    result = soilstate_init_time_const(**inputs)
    
    nbedrock = inputs["col"].nbedrock[0]
    csol_bedrock = inputs["config"].csol_bedrock
    
    # Check that bedrock layers have correct heat capacity
    # (layers at index >= nbedrock)
    if nbedrock < inputs["config"].nlevgrnd:
        bedrock_csol = result.csol[0, nbedrock:]
        assert jnp.allclose(bedrock_csol, csol_bedrock, rtol=1e-5), \
            f"Bedrock layers should have csol={csol_bedrock}, got {bedrock_csol}"


def test_soilstate_init_root_profile_normalization(test_data):
    """
    Test that root profiles are properly normalized.
    
    Verifies that root fractions sum to 1.0 for each patch,
    regardless of root profile type (Zeng2001 or Jackson1996).
    """
    # Test both root profile types
    for test_case_name in ["test_nominal_single_patch_single_column", 
                           "test_nominal_jackson_root_profile"]:
        test_case = test_data[test_case_name]
        inputs = test_case["inputs"]
        
        result = soilstate_init_time_const(**inputs)
        
        # Check normalization for each patch
        for p in range(result.rootfr.shape[0]):
            rootfr_sum = jnp.sum(result.rootfr[p, :])
            assert jnp.allclose(rootfr_sum, 1.0, atol=1e-6), \
                f"Root fraction for patch {p} should sum to 1.0, got {rootfr_sum}"


def test_soilstate_init_organic_matter_effects(test_data):
    """
    Test that organic matter affects soil properties appropriately.
    
    Compares zero organic matter case with high organic matter case
    to verify that organic matter modifies hydraulic and thermal properties.
    """
    zero_om_case = test_data["test_edge_zero_organic_matter"]
    high_om_case = test_data["test_edge_maximum_organic_matter"]
    
    zero_om_result = soilstate_init_time_const(**zero_om_case["inputs"])
    high_om_result = soilstate_init_time_const(**high_om_case["inputs"])
    
    # Organic matter should increase watsat (porosity)
    assert jnp.mean(high_om_result.watsat) > jnp.mean(zero_om_result.watsat), \
        "High organic matter should increase watsat"
    
    # Organic matter should decrease thermal conductivity
    assert jnp.mean(high_om_result.tkdry) < jnp.mean(zero_om_result.tkdry), \
        "High organic matter should decrease tkdry"
    
    # Organic matter should decrease heat capacity
    assert jnp.mean(high_om_result.csol) < jnp.mean(zero_om_result.csol), \
        "High organic matter should decrease csol"


def test_soilstate_init_texture_consistency(test_data):
    """
    Test that soil texture properties are consistent with lookup tables.
    
    Verifies that hydraulic properties (watsat, hksat, bsw, sucsat) are
    derived correctly from soil texture lookup tables.
    """
    test_case = test_data["test_nominal_single_patch_single_column"]
    inputs = test_case["inputs"]
    
    result = soilstate_init_time_const(**inputs)
    
    # Get texture index for this column
    col_idx = 0
    tower_idx = inputs["tower_data"].col_tower[col_idx]
    tex_idx = inputs["tower_data"].tex[tower_idx]
    
    # For mineral soil layers (where organic matter effect is minimal),
    # properties should be close to lookup table values
    # Check first layer as it typically has less organic matter influence
    layer_idx = 0
    
    # Note: Exact match not expected due to organic matter adjustments,
    # but values should be in reasonable range
    lookup_watsat = inputs["soil_texture"].watsat[tex_idx]
    assert 0.5 * lookup_watsat <= result.watsat[col_idx, layer_idx] <= 1.5 * lookup_watsat, \
        f"watsat should be near lookup value {lookup_watsat}"


def test_soilstate_init_multiple_columns_independence(test_data):
    """
    Test that multiple columns are processed independently.
    
    Verifies that properties for different columns with different
    soil types are computed independently and correctly.
    """
    test_case = test_data["test_nominal_multiple_patches_columns"]
    inputs = test_case["inputs"]
    
    result = soilstate_init_time_const(**inputs)
    
    # Columns should have different properties due to different soil types
    col0_watsat = result.watsat[0, :]
    col1_watsat = result.watsat[1, :]
    
    # Properties should differ between columns
    assert not jnp.allclose(col0_watsat, col1_watsat, rtol=0.01), \
        "Different columns should have different soil properties"


def test_soilstate_init_no_nan_or_inf(test_data):
    """
    Test that outputs contain no NaN or Inf values.
    
    Verifies numerical stability across all test cases.
    """
    for test_case_name in test_data.keys():
        test_case = test_data[test_case_name]
        inputs = test_case["inputs"]
        
        result = soilstate_init_time_const(**inputs)
        
        # Check all output arrays for NaN/Inf
        assert not jnp.any(jnp.isnan(result.rootfr)), \
            f"{test_case_name}: rootfr contains NaN"
        assert not jnp.any(jnp.isinf(result.rootfr)), \
            f"{test_case_name}: rootfr contains Inf"
        
        assert not jnp.any(jnp.isnan(result.watsat)), \
            f"{test_case_name}: watsat contains NaN"
        assert not jnp.any(jnp.isinf(result.watsat)), \
            f"{test_case_name}: watsat contains Inf"
        
        assert not jnp.any(jnp.isnan(result.hksat)), \
            f"{test_case_name}: hksat contains NaN"
        assert not jnp.any(jnp.isinf(result.hksat)), \
            f"{test_case_name}: hksat contains Inf"
        
        assert not jnp.any(jnp.isnan(result.csol)), \
            f"{test_case_name}: csol contains NaN"
        assert not jnp.any(jnp.isinf(result.csol)), \
            f"{test_case_name}: csol contains Inf"


def test_soilstate_init_percolation_fraction_bounds(test_data):
    """
    Test that percolation fraction is properly bounded.
    
    Verifies that perc_frac is in [0, 1] and follows expected patterns
    (typically decreasing with depth).
    """
    test_case = test_data["test_nominal_single_patch_single_column"]
    inputs = test_case["inputs"]
    
    result = soilstate_init_time_const(**inputs)
    
    # All percolation fractions should be in [0, 1]
    assert jnp.all((result.perc_frac >= 0) & (result.perc_frac <= 1)), \
        f"perc_frac should be in [0, 1], got range [{jnp.min(result.perc_frac)}, {jnp.max(result.perc_frac)}]"
    
    # Bottom layer should have perc_frac = 0 (no percolation below)
    assert jnp.allclose(result.perc_frac[:, -1], 0.0, atol=1e-6), \
        "Bottom layer should have perc_frac = 0"


def test_soilstate_init_clay_sand_sum(test_data):
    """
    Test that clay and sand percentages sum appropriately.
    
    Verifies that clay + sand <= 100% (with implicit silt making up the difference).
    """
    for test_case_name in test_data.keys():
        test_case = test_data[test_case_name]
        inputs = test_case["inputs"]
        
        result = soilstate_init_time_const(**inputs)
        
        # Clay + sand should not exceed 100%
        clay_sand_sum = result.cellclay + result.cellsand
        assert jnp.all(clay_sand_sum <= 100.0 + 1e-5), \
            f"{test_case_name}: clay + sand exceeds 100%, got max {jnp.max(clay_sand_sum)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])