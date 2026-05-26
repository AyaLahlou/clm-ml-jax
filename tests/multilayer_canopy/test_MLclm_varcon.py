"""
Comprehensive pytest suite for create_empty_rsl_lookup_tables function.

This module tests the creation of empty RSL (Roughness Sublayer) lookup tables
for the multilayer canopy model. The function initializes zero-filled arrays
with shapes determined by the vertical (n_z) and stability (n_l) dimensions
from the MLCanopyConstants.

Test Coverage:
- Nominal cases: Various grid dimensions and physical parameter regimes
- Edge cases: Minimum dimensions, boundary fraction values
- Special cases: Asymmetric and square dimensions
- Shape verification: All output arrays have correct dimensions
- Value verification: All arrays initialized to zeros
- Type verification: All arrays are JAX arrays
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multilayer_canopy.MLclm_varcon import (
    MLCanopyConstants,
    RSLPsihatLookupTables,
    create_empty_rsl_lookup_tables,
)


@pytest.fixture
def test_data():
    """
    Load test data for create_empty_rsl_lookup_tables function.
    
    Returns:
        dict: Test cases with inputs and metadata for comprehensive testing.
    """
    return {
        "test_nominal_default_dimensions": {
            "constants": MLCanopyConstants(
                rgas=8.314,
                mmdry=28.97,
                mmh2o=18.016,
                cpd=1005.0,
                cpw=1846.0,
                visc0=1.5e-05,
                dh0=2.12e-05,
                dv0=2.42e-05,
                dc0=1.47e-05,
                lapse_rate=0.0065,
                kc25=404.9,
                kcha=79430.0,
                ko25=278.4,
                koha=36380.0,
                cp25=42.75,
                cpha=37830.0,
                vcmaxha_noacclim=72000.0,
                vcmaxha_acclim=65330.0,
                vcmaxhd_noacclim=200000.0,
                vcmaxhd_acclim=200000.0,
                vcmaxse_noacclim=668.39,
                vcmaxse_acclim=668.39,
                jmaxha_noacclim=50000.0,
                jmaxha_acclim=43540.0,
                jmaxhd_noacclim=200000.0,
                jmaxhd_acclim=152040.0,
                jmaxse_noacclim=659.7,
                jmaxse_acclim=495.0,
                rdha=46390.0,
                rdhd=150650.0,
                rdse=490.0,
                jmax25_to_vcmax25_noacclim=1.67,
                jmax25_to_vcmax25_acclim=1.67,
                rd25_to_vcmax25_c3=0.015,
                rd25_to_vcmax25_c4=0.025,
                kp25_to_vcmax25_c4=0.02,
                phi_psii=0.85,
                theta_j=0.9,
                qe_c4=0.05,
                colim_c3a=0.98,
                colim_c3b=0.95,
                colim_c4a=0.8,
                colim_c4b=0.95,
                dh2o_to_dco2=1.6,
                rh_min_bb=0.3,
                vpd_min_med=50.0,
                cpbio=2000000.0,
                fcarbon=0.5,
                fwater=0.5,
                gb_factor=1.0,
                dewmx=0.1,
                maximum_leaf_wetted_fraction=0.05,
                interception_fraction=0.25,
                fwet_exponent=0.667,
                clm45_interception_p1=0.25,
                clm45_interception_p2=0.5,
                chil_min=-0.4,
                chil_max=0.6,
                kb_max=0.9,
                j_to_umol=4.6,
                emg=0.97,
                cd=0.3,
                beta_neutral_max=1.0,
                cr=0.3,
                c2=0.75,
                pr0=0.5,
                pr1=1.0,
                pr2=5.0,
                z0mg=0.01,
                wind_forc_min=0.1,
                eta_max=10.0,
                zeta_min=-2.0,
                zeta_max=1.0,
                beta_min=0.01,
                beta_max=1.0,
                wind_min=0.1,
                ra_max=999.0,
                n_z=50,
                n_l=40,
            ),
            "expected_n_z": 50,
            "expected_n_l": 40,
        },
        "test_nominal_small_dimensions": {
            "constants": MLCanopyConstants(
                rgas=8.314,
                mmdry=28.97,
                mmh2o=18.016,
                cpd=1005.0,
                cpw=1846.0,
                visc0=1.5e-05,
                dh0=2.12e-05,
                dv0=2.42e-05,
                dc0=1.47e-05,
                lapse_rate=0.0065,
                kc25=404.9,
                kcha=79430.0,
                ko25=278.4,
                koha=36380.0,
                cp25=42.75,
                cpha=37830.0,
                vcmaxha_noacclim=72000.0,
                vcmaxha_acclim=65330.0,
                vcmaxhd_noacclim=200000.0,
                vcmaxhd_acclim=200000.0,
                vcmaxse_noacclim=668.39,
                vcmaxse_acclim=668.39,
                jmaxha_noacclim=50000.0,
                jmaxha_acclim=43540.0,
                jmaxhd_noacclim=200000.0,
                jmaxhd_acclim=152040.0,
                jmaxse_noacclim=659.7,
                jmaxse_acclim=495.0,
                rdha=46390.0,
                rdhd=150650.0,
                rdse=490.0,
                jmax25_to_vcmax25_noacclim=1.67,
                jmax25_to_vcmax25_acclim=1.67,
                rd25_to_vcmax25_c3=0.015,
                rd25_to_vcmax25_c4=0.025,
                kp25_to_vcmax25_c4=0.02,
                phi_psii=0.85,
                theta_j=0.9,
                qe_c4=0.05,
                colim_c3a=0.98,
                colim_c3b=0.95,
                colim_c4a=0.8,
                colim_c4b=0.95,
                dh2o_to_dco2=1.6,
                rh_min_bb=0.3,
                vpd_min_med=50.0,
                cpbio=2000000.0,
                fcarbon=0.5,
                fwater=0.5,
                gb_factor=1.0,
                dewmx=0.1,
                maximum_leaf_wetted_fraction=0.05,
                interception_fraction=0.25,
                fwet_exponent=0.667,
                clm45_interception_p1=0.25,
                clm45_interception_p2=0.5,
                chil_min=-0.4,
                chil_max=0.6,
                kb_max=0.9,
                j_to_umol=4.6,
                emg=0.97,
                cd=0.3,
                beta_neutral_max=1.0,
                cr=0.3,
                c2=0.75,
                pr0=0.5,
                pr1=1.0,
                pr2=5.0,
                z0mg=0.01,
                wind_forc_min=0.1,
                eta_max=10.0,
                zeta_min=-2.0,
                zeta_max=1.0,
                beta_min=0.01,
                beta_max=1.0,
                wind_min=0.1,
                ra_max=999.0,
                n_z=10,
                n_l=15,
            ),
            "expected_n_z": 10,
            "expected_n_l": 15,
        },
        "test_nominal_large_dimensions": {
            "constants": MLCanopyConstants(
                rgas=8.314,
                mmdry=28.97,
                mmh2o=18.016,
                cpd=1005.0,
                cpw=1846.0,
                visc0=1.5e-05,
                dh0=2.12e-05,
                dv0=2.42e-05,
                dc0=1.47e-05,
                lapse_rate=0.0065,
                kc25=404.9,
                kcha=79430.0,
                ko25=278.4,
                koha=36380.0,
                cp25=42.75,
                cpha=37830.0,
                vcmaxha_noacclim=72000.0,
                vcmaxha_acclim=65330.0,
                vcmaxhd_noacclim=200000.0,
                vcmaxhd_acclim=200000.0,
                vcmaxse_noacclim=668.39,
                vcmaxse_acclim=668.39,
                jmaxha_noacclim=50000.0,
                jmaxha_acclim=43540.0,
                jmaxhd_noacclim=200000.0,
                jmaxhd_acclim=152040.0,
                jmaxse_noacclim=659.7,
                jmaxse_acclim=495.0,
                rdha=46390.0,
                rdhd=150650.0,
                rdse=490.0,
                jmax25_to_vcmax25_noacclim=1.67,
                jmax25_to_vcmax25_acclim=1.67,
                rd25_to_vcmax25_c3=0.015,
                rd25_to_vcmax25_c4=0.025,
                kp25_to_vcmax25_c4=0.02,
                phi_psii=0.85,
                theta_j=0.9,
                qe_c4=0.05,
                colim_c3a=0.98,
                colim_c3b=0.95,
                colim_c4a=0.8,
                colim_c4b=0.95,
                dh2o_to_dco2=1.6,
                rh_min_bb=0.3,
                vpd_min_med=50.0,
                cpbio=2000000.0,
                fcarbon=0.5,
                fwater=0.5,
                gb_factor=1.0,
                dewmx=0.1,
                maximum_leaf_wetted_fraction=0.05,
                interception_fraction=0.25,
                fwet_exponent=0.667,
                clm45_interception_p1=0.25,
                clm45_interception_p2=0.5,
                chil_min=-0.4,
                chil_max=0.6,
                kb_max=0.9,
                j_to_umol=4.6,
                emg=0.97,
                cd=0.3,
                beta_neutral_max=1.0,
                cr=0.3,
                c2=0.75,
                pr0=0.5,
                pr1=1.0,
                pr2=5.0,
                z0mg=0.01,
                wind_forc_min=0.1,
                eta_max=10.0,
                zeta_min=-2.0,
                zeta_max=1.0,
                beta_min=0.01,
                beta_max=1.0,
                wind_min=0.1,
                ra_max=999.0,
                n_z=100,
                n_l=80,
            ),
            "expected_n_z": 100,
            "expected_n_l": 80,
        },
        "test_edge_minimum_dimensions": {
            "constants": MLCanopyConstants(
                rgas=8.314,
                mmdry=28.97,
                mmh2o=18.016,
                cpd=1005.0,
                cpw=1846.0,
                visc0=1.5e-05,
                dh0=2.12e-05,
                dv0=2.42e-05,
                dc0=1.47e-05,
                lapse_rate=0.0065,
                kc25=404.9,
                kcha=79430.0,
                ko25=278.4,
                koha=36380.0,
                cp25=42.75,
                cpha=37830.0,
                vcmaxha_noacclim=72000.0,
                vcmaxha_acclim=65330.0,
                vcmaxhd_noacclim=200000.0,
                vcmaxhd_acclim=200000.0,
                vcmaxse_noacclim=668.39,
                vcmaxse_acclim=668.39,
                jmaxha_noacclim=50000.0,
                jmaxha_acclim=43540.0,
                jmaxhd_noacclim=200000.0,
                jmaxhd_acclim=152040.0,
                jmaxse_noacclim=659.7,
                jmaxse_acclim=495.0,
                rdha=46390.0,
                rdhd=150650.0,
                rdse=490.0,
                jmax25_to_vcmax25_noacclim=1.67,
                jmax25_to_vcmax25_acclim=1.67,
                rd25_to_vcmax25_c3=0.015,
                rd25_to_vcmax25_c4=0.025,
                kp25_to_vcmax25_c4=0.02,
                phi_psii=0.85,
                theta_j=0.9,
                qe_c4=0.05,
                colim_c3a=0.98,
                colim_c3b=0.95,
                colim_c4a=0.8,
                colim_c4b=0.95,
                dh2o_to_dco2=1.6,
                rh_min_bb=0.3,
                vpd_min_med=50.0,
                cpbio=2000000.0,
                fcarbon=0.5,
                fwater=0.5,
                gb_factor=1.0,
                dewmx=0.1,
                maximum_leaf_wetted_fraction=0.05,
                interception_fraction=0.25,
                fwet_exponent=0.667,
                clm45_interception_p1=0.25,
                clm45_interception_p2=0.5,
                chil_min=-0.4,
                chil_max=0.6,
                kb_max=0.9,
                j_to_umol=4.6,
                emg=0.97,
                cd=0.3,
                beta_neutral_max=1.0,
                cr=0.3,
                c2=0.75,
                pr0=0.5,
                pr1=1.0,
                pr2=5.0,
                z0mg=0.01,
                wind_forc_min=0.1,
                eta_max=10.0,
                zeta_min=-2.0,
                zeta_max=1.0,
                beta_min=0.01,
                beta_max=1.0,
                wind_min=0.1,
                ra_max=999.0,
                n_z=1,
                n_l=1,
            ),
            "expected_n_z": 1,
            "expected_n_l": 1,
        },
        "test_special_asymmetric_dimensions": {
            "constants": MLCanopyConstants(
                rgas=8.314,
                mmdry=28.97,
                mmh2o=18.016,
                cpd=1005.0,
                cpw=1846.0,
                visc0=1.5e-05,
                dh0=2.12e-05,
                dv0=2.42e-05,
                dc0=1.47e-05,
                lapse_rate=0.0065,
                kc25=404.9,
                kcha=79430.0,
                ko25=278.4,
                koha=36380.0,
                cp25=42.75,
                cpha=37830.0,
                vcmaxha_noacclim=72000.0,
                vcmaxha_acclim=65330.0,
                vcmaxhd_noacclim=200000.0,
                vcmaxhd_acclim=200000.0,
                vcmaxse_noacclim=668.39,
                vcmaxse_acclim=668.39,
                jmaxha_noacclim=50000.0,
                jmaxha_acclim=43540.0,
                jmaxhd_noacclim=200000.0,
                jmaxhd_acclim=152040.0,
                jmaxse_noacclim=659.7,
                jmaxse_acclim=495.0,
                rdha=46390.0,
                rdhd=150650.0,
                rdse=490.0,
                jmax25_to_vcmax25_noacclim=1.67,
                jmax25_to_vcmax25_acclim=1.67,
                rd25_to_vcmax25_c3=0.015,
                rd25_to_vcmax25_c4=0.025,
                kp25_to_vcmax25_c4=0.02,
                phi_psii=0.85,
                theta_j=0.9,
                qe_c4=0.05,
                colim_c3a=0.98,
                colim_c3b=0.95,
                colim_c4a=0.8,
                colim_c4b=0.95,
                dh2o_to_dco2=1.6,
                rh_min_bb=0.3,
                vpd_min_med=50.0,
                cpbio=2000000.0,
                fcarbon=0.5,
                fwater=0.5,
                gb_factor=1.0,
                dewmx=0.1,
                maximum_leaf_wetted_fraction=0.05,
                interception_fraction=0.25,
                fwet_exponent=0.667,
                clm45_interception_p1=0.25,
                clm45_interception_p2=0.5,
                chil_min=-0.4,
                chil_max=0.6,
                kb_max=0.9,
                j_to_umol=4.6,
                emg=0.97,
                cd=0.3,
                beta_neutral_max=1.0,
                cr=0.3,
                c2=0.75,
                pr0=0.5,
                pr1=1.0,
                pr2=5.0,
                z0mg=0.01,
                wind_forc_min=0.1,
                eta_max=10.0,
                zeta_min=-2.0,
                zeta_max=1.0,
                beta_min=0.01,
                beta_max=1.0,
                wind_min=0.1,
                ra_max=999.0,
                n_z=5,
                n_l=100,
            ),
            "expected_n_z": 5,
            "expected_n_l": 100,
        },
        "test_special_square_dimensions": {
            "constants": MLCanopyConstants(
                rgas=8.314,
                mmdry=28.97,
                mmh2o=18.016,
                cpd=1005.0,
                cpw=1846.0,
                visc0=1.5e-05,
                dh0=2.12e-05,
                dv0=2.42e-05,
                dc0=1.47e-05,
                lapse_rate=0.0065,
                kc25=404.9,
                kcha=79430.0,
                ko25=278.4,
                koha=36380.0,
                cp25=42.75,
                cpha=37830.0,
                vcmaxha_noacclim=72000.0,
                vcmaxha_acclim=65330.0,
                vcmaxhd_noacclim=200000.0,
                vcmaxhd_acclim=200000.0,
                vcmaxse_noacclim=668.39,
                vcmaxse_acclim=668.39,
                jmaxha_noacclim=50000.0,
                jmaxha_acclim=43540.0,
                jmaxhd_noacclim=200000.0,
                jmaxhd_acclim=152040.0,
                jmaxse_noacclim=659.7,
                jmaxse_acclim=495.0,
                rdha=46390.0,
                rdhd=150650.0,
                rdse=490.0,
                jmax25_to_vcmax25_noacclim=1.67,
                jmax25_to_vcmax25_acclim=1.67,
                rd25_to_vcmax25_c3=0.015,
                rd25_to_vcmax25_c4=0.025,
                kp25_to_vcmax25_c4=0.02,
                phi_psii=0.85,
                theta_j=0.9,
                qe_c4=0.05,
                colim_c3a=0.98,
                colim_c3b=0.95,
                colim_c4a=0.8,
                colim_c4b=0.95,
                dh2o_to_dco2=1.6,
                rh_min_bb=0.3,
                vpd_min_med=50.0,
                cpbio=2000000.0,
                fcarbon=0.5,
                fwater=0.5,
                gb_factor=1.0,
                dewmx=0.1,
                maximum_leaf_wetted_fraction=0.05,
                interception_fraction=0.25,
                fwet_exponent=0.667,
                clm45_interception_p1=0.25,
                clm45_interception_p2=0.5,
                chil_min=-0.4,
                chil_max=0.6,
                kb_max=0.9,
                j_to_umol=4.6,
                emg=0.97,
                cd=0.3,
                beta_neutral_max=1.0,
                cr=0.3,
                c2=0.75,
                pr0=0.5,
                pr1=1.0,
                pr2=5.0,
                z0mg=0.01,
                wind_forc_min=0.1,
                eta_max=10.0,
                zeta_min=-2.0,
                zeta_max=1.0,
                beta_min=0.01,
                beta_max=1.0,
                wind_min=0.1,
                ra_max=999.0,
                n_z=64,
                n_l=64,
            ),
            "expected_n_z": 64,
            "expected_n_l": 64,
        },
    }


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_nominal_default_dimensions",
        "test_nominal_small_dimensions",
        "test_nominal_large_dimensions",
        "test_edge_minimum_dimensions",
        "test_special_asymmetric_dimensions",
        "test_special_square_dimensions",
    ],
)
def test_create_empty_rsl_lookup_tables_shapes(test_data, test_case_name):
    """
    Test that create_empty_rsl_lookup_tables returns arrays with correct shapes.
    
    Verifies that:
    - zdtgrid_m and zdtgrid_h have shape (n_z, 1)
    - dtlgrid_m and dtlgrid_h have shape (1, n_l)
    - psigrid_m and psigrid_h have shape (n_z, n_l)
    
    Args:
        test_data: Fixture providing test cases
        test_case_name: Name of the test case to run
    """
    test_case = test_data[test_case_name]
    constants = test_case["constants"]
    expected_n_z = test_case["expected_n_z"]
    expected_n_l = test_case["expected_n_l"]
    
    # Call the function
    result = create_empty_rsl_lookup_tables(constants)
    
    # Verify result is RSLPsihatLookupTables
    assert isinstance(result, RSLPsihatLookupTables), (
        f"Expected RSLPsihatLookupTables, got {type(result)}"
    )
    
    # Check zdtgrid shapes (n_z, 1)
    assert result.zdtgrid_m.shape == (expected_n_z, 1), (
        f"zdtgrid_m shape mismatch: expected ({expected_n_z}, 1), "
        f"got {result.zdtgrid_m.shape}"
    )
    assert result.zdtgrid_h.shape == (expected_n_z, 1), (
        f"zdtgrid_h shape mismatch: expected ({expected_n_z}, 1), "
        f"got {result.zdtgrid_h.shape}"
    )
    
    # Check dtlgrid shapes (1, n_l)
    assert result.dtlgrid_m.shape == (1, expected_n_l), (
        f"dtlgrid_m shape mismatch: expected (1, {expected_n_l}), "
        f"got {result.dtlgrid_m.shape}"
    )
    assert result.dtlgrid_h.shape == (1, expected_n_l), (
        f"dtlgrid_h shape mismatch: expected (1, {expected_n_l}), "
        f"got {result.dtlgrid_h.shape}"
    )
    
    # Check psigrid shapes (n_z, n_l)
    assert result.psigrid_m.shape == (expected_n_z, expected_n_l), (
        f"psigrid_m shape mismatch: expected ({expected_n_z}, {expected_n_l}), "
        f"got {result.psigrid_m.shape}"
    )
    assert result.psigrid_h.shape == (expected_n_z, expected_n_l), (
        f"psigrid_h shape mismatch: expected ({expected_n_z}, {expected_n_l}), "
        f"got {result.psigrid_h.shape}"
    )


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_nominal_default_dimensions",
        "test_nominal_small_dimensions",
        "test_nominal_large_dimensions",
        "test_edge_minimum_dimensions",
        "test_special_asymmetric_dimensions",
        "test_special_square_dimensions",
    ],
)
def test_create_empty_rsl_lookup_tables_values(test_data, test_case_name):
    """
    Test that create_empty_rsl_lookup_tables initializes all arrays to zeros.
    
    Verifies that all six output arrays (zdtgrid_m, dtlgrid_m, psigrid_m,
    zdtgrid_h, dtlgrid_h, psigrid_h) are filled with zeros.
    
    Args:
        test_data: Fixture providing test cases
        test_case_name: Name of the test case to run
    """
    test_case = test_data[test_case_name]
    constants = test_case["constants"]
    
    # Call the function
    result = create_empty_rsl_lookup_tables(constants)
    
    # Check that all arrays are zeros
    assert jnp.allclose(result.zdtgrid_m, 0.0, atol=1e-10), (
        "zdtgrid_m should be initialized to zeros"
    )
    assert jnp.allclose(result.dtlgrid_m, 0.0, atol=1e-10), (
        "dtlgrid_m should be initialized to zeros"
    )
    assert jnp.allclose(result.psigrid_m, 0.0, atol=1e-10), (
        "psigrid_m should be initialized to zeros"
    )
    assert jnp.allclose(result.zdtgrid_h, 0.0, atol=1e-10), (
        "zdtgrid_h should be initialized to zeros"
    )
    assert jnp.allclose(result.dtlgrid_h, 0.0, atol=1e-10), (
        "dtlgrid_h should be initialized to zeros"
    )
    assert jnp.allclose(result.psigrid_h, 0.0, atol=1e-10), (
        "psigrid_h should be initialized to zeros"
    )
    
    # Verify no NaN or Inf values
    assert not jnp.any(jnp.isnan(result.zdtgrid_m)), "zdtgrid_m contains NaN"
    assert not jnp.any(jnp.isnan(result.dtlgrid_m)), "dtlgrid_m contains NaN"
    assert not jnp.any(jnp.isnan(result.psigrid_m)), "psigrid_m contains NaN"
    assert not jnp.any(jnp.isnan(result.zdtgrid_h)), "zdtgrid_h contains NaN"
    assert not jnp.any(jnp.isnan(result.dtlgrid_h)), "dtlgrid_h contains NaN"
    assert not jnp.any(jnp.isnan(result.psigrid_h)), "psigrid_h contains NaN"
    
    assert not jnp.any(jnp.isinf(result.zdtgrid_m)), "zdtgrid_m contains Inf"
    assert not jnp.any(jnp.isinf(result.dtlgrid_m)), "dtlgrid_m contains Inf"
    assert not jnp.any(jnp.isinf(result.psigrid_m)), "psigrid_m contains Inf"
    assert not jnp.any(jnp.isinf(result.zdtgrid_h)), "zdtgrid_h contains Inf"
    assert not jnp.any(jnp.isinf(result.dtlgrid_h)), "dtlgrid_h contains Inf"
    assert not jnp.any(jnp.isinf(result.psigrid_h)), "psigrid_h contains Inf"


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_nominal_default_dimensions",
        "test_nominal_small_dimensions",
        "test_nominal_large_dimensions",
        "test_edge_minimum_dimensions",
        "test_special_asymmetric_dimensions",
        "test_special_square_dimensions",
    ],
)
def test_create_empty_rsl_lookup_tables_dtypes(test_data, test_case_name):
    """
    Test that create_empty_rsl_lookup_tables returns JAX arrays with correct dtypes.
    
    Verifies that all output arrays are JAX arrays (jnp.ndarray) with
    floating-point dtype for GPU compatibility.
    
    Args:
        test_data: Fixture providing test cases
        test_case_name: Name of the test case to run
    """
    test_case = test_data[test_case_name]
    constants = test_case["constants"]
    
    # Call the function
    result = create_empty_rsl_lookup_tables(constants)
    
    # Check that all arrays are JAX arrays
    assert isinstance(result.zdtgrid_m, jnp.ndarray), (
        f"zdtgrid_m should be JAX array, got {type(result.zdtgrid_m)}"
    )
    assert isinstance(result.dtlgrid_m, jnp.ndarray), (
        f"dtlgrid_m should be JAX array, got {type(result.dtlgrid_m)}"
    )
    assert isinstance(result.psigrid_m, jnp.ndarray), (
        f"psigrid_m should be JAX array, got {type(result.psigrid_m)}"
    )
    assert isinstance(result.zdtgrid_h, jnp.ndarray), (
        f"zdtgrid_h should be JAX array, got {type(result.zdtgrid_h)}"
    )
    assert isinstance(result.dtlgrid_h, jnp.ndarray), (
        f"dtlgrid_h should be JAX array, got {type(result.dtlgrid_h)}"
    )
    assert isinstance(result.psigrid_h, jnp.ndarray), (
        f"psigrid_h should be JAX array, got {type(result.psigrid_h)}"
    )
    
    # Check that all arrays have floating-point dtype
    assert jnp.issubdtype(result.zdtgrid_m.dtype, jnp.floating), (
        f"zdtgrid_m should have floating dtype, got {result.zdtgrid_m.dtype}"
    )
    assert jnp.issubdtype(result.dtlgrid_m.dtype, jnp.floating), (
        f"dtlgrid_m should have floating dtype, got {result.dtlgrid_m.dtype}"
    )
    assert jnp.issubdtype(result.psigrid_m.dtype, jnp.floating), (
        f"psigrid_m should have floating dtype, got {result.psigrid_m.dtype}"
    )
    assert jnp.issubdtype(result.zdtgrid_h.dtype, jnp.floating), (
        f"zdtgrid_h should have floating dtype, got {result.zdtgrid_h.dtype}"
    )
    assert jnp.issubdtype(result.dtlgrid_h.dtype, jnp.floating), (
        f"dtlgrid_h should have floating dtype, got {result.dtlgrid_h.dtype}"
    )
    assert jnp.issubdtype(result.psigrid_h.dtype, jnp.floating), (
        f"psigrid_h should have floating dtype, got {result.psigrid_h.dtype}"
    )


def test_create_empty_rsl_lookup_tables_edge_minimum_dimensions(test_data):
    """
    Test create_empty_rsl_lookup_tables with minimum valid dimensions (1x1).
    
    This edge case tests the boundary condition where both n_z and n_l are 1,
    ensuring the function handles the smallest possible lookup tables correctly.
    """
    test_case = test_data["test_edge_minimum_dimensions"]
    constants = test_case["constants"]
    
    # Call the function
    result = create_empty_rsl_lookup_tables(constants)
    
    # Verify shapes
    assert result.zdtgrid_m.shape == (1, 1)
    assert result.zdtgrid_h.shape == (1, 1)
    assert result.dtlgrid_m.shape == (1, 1)
    assert result.dtlgrid_h.shape == (1, 1)
    assert result.psigrid_m.shape == (1, 1)
    assert result.psigrid_h.shape == (1, 1)
    
    # Verify all values are zero
    assert result.zdtgrid_m[0, 0] == 0.0
    assert result.zdtgrid_h[0, 0] == 0.0
    assert result.dtlgrid_m[0, 0] == 0.0
    assert result.dtlgrid_h[0, 0] == 0.0
    assert result.psigrid_m[0, 0] == 0.0
    assert result.psigrid_h[0, 0] == 0.0


def test_create_empty_rsl_lookup_tables_edge_asymmetric(test_data):
    """
    Test create_empty_rsl_lookup_tables with highly asymmetric dimensions.
    
    This tests the case where n_z << n_l (5x100), ensuring the function
    handles non-square grids correctly, which is important for simulations
    with many stability levels but few vertical levels.
    """
    test_case = test_data["test_special_asymmetric_dimensions"]
    constants = test_case["constants"]
    
    # Call the function
    result = create_empty_rsl_lookup_tables(constants)
    
    # Verify shapes
    assert result.zdtgrid_m.shape == (5, 1)
    assert result.zdtgrid_h.shape == (5, 1)
    assert result.dtlgrid_m.shape == (1, 100)
    assert result.dtlgrid_h.shape == (1, 100)
    assert result.psigrid_m.shape == (5, 100)
    assert result.psigrid_h.shape == (5, 100)
    
    # Verify total number of elements
    assert result.psigrid_m.size == 500
    assert result.psigrid_h.size == 500


def test_create_empty_rsl_lookup_tables_consistency():
    """
    Test that multiple calls with same constants produce identical results.
    
    Verifies that the function is deterministic and produces consistent
    output for the same input constants.
    """
    constants = MLCanopyConstants(
        rgas=8.314,
        mmdry=28.97,
        mmh2o=18.016,
        cpd=1005.0,
        cpw=1846.0,
        visc0=1.5e-05,
        dh0=2.12e-05,
        dv0=2.42e-05,
        dc0=1.47e-05,
        lapse_rate=0.0065,
        kc25=404.9,
        kcha=79430.0,
        ko25=278.4,
        koha=36380.0,
        cp25=42.75,
        cpha=37830.0,
        vcmaxha_noacclim=72000.0,
        vcmaxha_acclim=65330.0,
        vcmaxhd_noacclim=200000.0,
        vcmaxhd_acclim=200000.0,
        vcmaxse_noacclim=668.39,
        vcmaxse_acclim=668.39,
        jmaxha_noacclim=50000.0,
        jmaxha_acclim=43540.0,
        jmaxhd_noacclim=200000.0,
        jmaxhd_acclim=152040.0,
        jmaxse_noacclim=659.7,
        jmaxse_acclim=495.0,
        rdha=46390.0,
        rdhd=150650.0,
        rdse=490.0,
        jmax25_to_vcmax25_noacclim=1.67,
        jmax25_to_vcmax25_acclim=1.67,
        rd25_to_vcmax25_c3=0.015,
        rd25_to_vcmax25_c4=0.025,
        kp25_to_vcmax25_c4=0.02,
        phi_psii=0.85,
        theta_j=0.9,
        qe_c4=0.05,
        colim_c3a=0.98,
        colim_c3b=0.95,
        colim_c4a=0.8,
        colim_c4b=0.95,
        dh2o_to_dco2=1.6,
        rh_min_bb=0.3,
        vpd_min_med=50.0,
        cpbio=2000000.0,
        fcarbon=0.5,
        fwater=0.5,
        gb_factor=1.0,
        dewmx=0.1,
        maximum_leaf_wetted_fraction=0.05,
        interception_fraction=0.25,
        fwet_exponent=0.667,
        clm45_interception_p1=0.25,
        clm45_interception_p2=0.5,
        chil_min=-0.4,
        chil_max=0.6,
        kb_max=0.9,
        j_to_umol=4.6,
        emg=0.97,
        cd=0.3,
        beta_neutral_max=1.0,
        cr=0.3,
        c2=0.75,
        pr0=0.5,
        pr1=1.0,
        pr2=5.0,
        z0mg=0.01,
        wind_forc_min=0.1,
        eta_max=10.0,
        zeta_min=-2.0,
        zeta_max=1.0,
        beta_min=0.01,
        beta_max=1.0,
        wind_min=0.1,
        ra_max=999.0,
        n_z=20,
        n_l=15,
    )
    
    # Call function twice
    result1 = create_empty_rsl_lookup_tables(constants)
    result2 = create_empty_rsl_lookup_tables(constants)
    
    # Verify results are identical
    assert jnp.array_equal(result1.zdtgrid_m, result2.zdtgrid_m)
    assert jnp.array_equal(result1.dtlgrid_m, result2.dtlgrid_m)
    assert jnp.array_equal(result1.psigrid_m, result2.psigrid_m)
    assert jnp.array_equal(result1.zdtgrid_h, result2.zdtgrid_h)
    assert jnp.array_equal(result1.dtlgrid_h, result2.dtlgrid_h)
    assert jnp.array_equal(result1.psigrid_h, result2.psigrid_h)


def test_create_empty_rsl_lookup_tables_independence():
    """
    Test that output arrays are independent (modifying one doesn't affect others).
    
    Verifies that the six output arrays are separate objects and modifications
    to one array don't affect the others.
    """
    constants = MLCanopyConstants(
        rgas=8.314,
        mmdry=28.97,
        mmh2o=18.016,
        cpd=1005.0,
        cpw=1846.0,
        visc0=1.5e-05,
        dh0=2.12e-05,
        dv0=2.42e-05,
        dc0=1.47e-05,
        lapse_rate=0.0065,
        kc25=404.9,
        kcha=79430.0,
        ko25=278.4,
        koha=36380.0,
        cp25=42.75,
        cpha=37830.0,
        vcmaxha_noacclim=72000.0,
        vcmaxha_acclim=65330.0,
        vcmaxhd_noacclim=200000.0,
        vcmaxhd_acclim=200000.0,
        vcmaxse_noacclim=668.39,
        vcmaxse_acclim=668.39,
        jmaxha_noacclim=50000.0,
        jmaxha_acclim=43540.0,
        jmaxhd_noacclim=200000.0,
        jmaxhd_acclim=152040.0,
        jmaxse_noacclim=659.7,
        jmaxse_acclim=495.0,
        rdha=46390.0,
        rdhd=150650.0,
        rdse=490.0,
        jmax25_to_vcmax25_noacclim=1.67,
        jmax25_to_vcmax25_acclim=1.67,
        rd25_to_vcmax25_c3=0.015,
        rd25_to_vcmax25_c4=0.025,
        kp25_to_vcmax25_c4=0.02,
        phi_psii=0.85,
        theta_j=0.9,
        qe_c4=0.05,
        colim_c3a=0.98,
        colim_c3b=0.95,
        colim_c4a=0.8,
        colim_c4b=0.95,
        dh2o_to_dco2=1.6,
        rh_min_bb=0.3,
        vpd_min_med=50.0,
        cpbio=2000000.0,
        fcarbon=0.5,
        fwater=0.5,
        gb_factor=1.0,
        dewmx=0.1,
        maximum_leaf_wetted_fraction=0.05,
        interception_fraction=0.25,
        fwet_exponent=0.667,
        clm45_interception_p1=0.25,
        clm45_interception_p2=0.5,
        chil_min=-0.4,
        chil_max=0.6,
        kb_max=0.9,
        j_to_umol=4.6,
        emg=0.97,
        cd=0.3,
        beta_neutral_max=1.0,
        cr=0.3,
        c2=0.75,
        pr0=0.5,
        pr1=1.0,
        pr2=5.0,
        z0mg=0.01,
        wind_forc_min=0.1,
        eta_max=10.0,
        zeta_min=-2.0,
        zeta_max=1.0,
        beta_min=0.01,
        beta_max=1.0,
        wind_min=0.1,
        ra_max=999.0,
        n_z=10,
        n_l=10,
    )
    
    result = create_empty_rsl_lookup_tables(constants)
    
    # Create modified versions (JAX arrays are immutable, so we create new ones)
    modified_zdtgrid_m = result.zdtgrid_m.at[0, 0].set(1.0)
    
    # Verify original is unchanged
    assert result.zdtgrid_m[0, 0] == 0.0, (
        "Original array should remain unchanged (JAX immutability)"
    )
    
    # Verify other arrays are still zero
    assert jnp.allclose(result.dtlgrid_m, 0.0)
    assert jnp.allclose(result.psigrid_m, 0.0)
    assert jnp.allclose(result.zdtgrid_h, 0.0)
    assert jnp.allclose(result.dtlgrid_h, 0.0)
    assert jnp.allclose(result.psigrid_h, 0.0)


def test_create_empty_rsl_lookup_tables_memory_layout():
    """
    Test that arrays have expected memory layout for efficient computation.
    
    Verifies that the arrays are contiguous in memory, which is important
    for GPU performance and vectorized operations.
    """
    constants = MLCanopyConstants(
        rgas=8.314,
        mmdry=28.97,
        mmh2o=18.016,
        cpd=1005.0,
        cpw=1846.0,
        visc0=1.5e-05,
        dh0=2.12e-05,
        dv0=2.42e-05,
        dc0=1.47e-05,
        lapse_rate=0.0065,
        kc25=404.9,
        kcha=79430.0,
        ko25=278.4,
        koha=36380.0,
        cp25=42.75,
        cpha=37830.0,
        vcmaxha_noacclim=72000.0,
        vcmaxha_acclim=65330.0,
        vcmaxhd_noacclim=200000.0,
        vcmaxhd_acclim=200000.0,
        vcmaxse_noacclim=668.39,
        vcmaxse_acclim=668.39,
        jmaxha_noacclim=50000.0,
        jmaxha_acclim=43540.0,
        jmaxhd_noacclim=200000.0,
        jmaxhd_acclim=152040.0,
        jmaxse_noacclim=659.7,
        jmaxse_acclim=495.0,
        rdha=46390.0,
        rdhd=150650.0,
        rdse=490.0,
        jmax25_to_vcmax25_noacclim=1.67,
        jmax25_to_vcmax25_acclim=1.67,
        rd25_to_vcmax25_c3=0.015,
        rd25_to_vcmax25_c4=0.025,
        kp25_to_vcmax25_c4=0.02,
        phi_psii=0.85,
        theta_j=0.9,
        qe_c4=0.05,
        colim_c3a=0.98,
        colim_c3b=0.95,
        colim_c4a=0.8,
        colim_c4b=0.95,
        dh2o_to_dco2=1.6,
        rh_min_bb=0.3,
        vpd_min_med=50.0,
        cpbio=2000000.0,
        fcarbon=0.5,
        fwater=0.5,
        gb_factor=1.0,
        dewmx=0.1,
        maximum_leaf_wetted_fraction=0.05,
        interception_fraction=0.25,
        fwet_exponent=0.667,
        clm45_interception_p1=0.25,
        clm45_interception_p2=0.5,
        chil_min=-0.4,
        chil_max=0.6,
        kb_max=0.9,
        j_to_umol=4.6,
        emg=0.97,
        cd=0.3,
        beta_neutral_max=1.0,
        cr=0.3,
        c2=0.75,
        pr0=0.5,
        pr1=1.0,
        pr2=5.0,
        z0mg=0.01,
        wind_forc_min=0.1,
        eta_max=10.0,
        zeta_min=-2.0,
        zeta_max=1.0,
        beta_min=0.01,
        beta_max=1.0,
        wind_min=0.1,
        ra_max=999.0,
        n_z=30,
        n_l=25,
    )
    
    result = create_empty_rsl_lookup_tables(constants)
    
    # JAX arrays should be well-formed
    # Check that arrays can be used in computations without errors
    try:
        _ = result.zdtgrid_m + 1.0
        _ = result.dtlgrid_m * 2.0
        _ = result.psigrid_m.sum()
        _ = result.zdtgrid_h.mean()
        _ = result.dtlgrid_h.max()
        _ = result.psigrid_h.min()
    except Exception as e:
        pytest.fail(f"Arrays should support basic operations: {e}")