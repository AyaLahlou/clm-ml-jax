"""
Comprehensive pytest suite for leaf_photosynthesis function from MLLeafPhotosynthesisMod.

This test suite covers:
- Nominal cases for C3, C4, and mixed vegetation
- Edge cases including zero radiation, water stress, temperature extremes
- Special conditions like high altitude and dense canopy
- Output shape, dtype, and value validation
- Physical realism constraints
"""

import sys
from pathlib import Path
from typing import Dict, Any

import pytest
import jax.numpy as jnp
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from multilayer_canopy.MLLeafPhotosynthesisMod import leaf_photosynthesis, PhotosynthesisParams, LeafPhotosynthesisState


@pytest.fixture
def default_params() -> PhotosynthesisParams:
    """
    Fixture providing default PhotosynthesisParams for testing.
    
    Returns:
        PhotosynthesisParams with standard values for physical constants,
        temperature response parameters, and model configuration.
    """
    return PhotosynthesisParams(
        tfrz=273.15,
        rgas=8.314,
        kc25=404.9,
        ko25=278.4,
        cp25=42.75,
        kcha=79430.0,
        koha=36380.0,
        cpha=37830.0,
        vcmaxha_noacclim=65330.0,
        vcmaxha_acclim=65330.0,
        jmaxha_noacclim=43540.0,
        jmaxha_acclim=43540.0,
        vcmaxhd_noacclim=149250.0,
        vcmaxhd_acclim=149250.0,
        jmaxhd_noacclim=152040.0,
        jmaxhd_acclim=152040.0,
        vcmaxse_noacclim=485.0,
        vcmaxse_acclim=485.0,
        jmaxse_noacclim=495.0,
        jmaxse_acclim=495.0,
        rdha=46390.0,
        rdhd=150650.0,
        rdse=490.0,
        phi_psii=0.85,
        theta_j=0.90,
        vpd_min_med=0.1,
        rh_min_bb=0.3,
        dh2o_to_dco2=1.6,
        qe_c4=0.05,
        colim_c3a=0.98,
        colim_c4a=0.80,
        colim_c4b=10000.0,
        gs_type=1,  # Ball-Berry
        acclim_type=0,  # No acclimation
        gspot_type=1,  # Standard optimization
        colim_type=0  # Standard co-limitation
    )


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Fixture providing comprehensive test data for leaf_photosynthesis.
    
    Returns:
        Dictionary containing test cases with inputs and metadata.
    """
    return {
        "test_nominal_c3_single_patch_single_layer": {
            "inputs": {
                "c3psn": jnp.array([1.0]),
                "g0_BB": jnp.array([0.01]),
                "g1_BB": jnp.array([9.0]),
                "g0_MED": jnp.array([0.0]),
                "g1_MED": jnp.array([4.0]),
                "psi50_gs": jnp.array([-2.0]),
                "shape_gs": jnp.array([3.0]),
                "gsmin_SPA": jnp.array([0.001]),
                "iota_SPA": jnp.array([0.0001]),
                "tacclim": jnp.array([298.15]),
                "ncan": jnp.array([1]),
                "dpai": jnp.array([[2.5]]),
                "eair": jnp.array([[1500.0]]),
                "o2ref": jnp.array([209.0]),
                "pref": jnp.array([101325.0]),
                "cair": jnp.array([[[400.0]]]),
                "vcmax25": jnp.array([[[60.0]]]),
                "jmax25": jnp.array([[[120.0]]]),
                "kp25": jnp.array([[[0.0]]]),
                "rd25": jnp.array([[[1.2]]]),
                "tleaf": jnp.array([[[298.15]]]),
                "gbv": jnp.array([[[1.5]]]),
                "gbc": jnp.array([[[1.0]]]),
                "apar": jnp.array([[[1000.0]]]),
                "lwp": jnp.array([[[-0.5]]]),
            },
            "metadata": {
                "type": "nominal",
                "description": "Standard C3 photosynthesis under optimal conditions",
                "expected_shapes": {
                    "n_patches": 1,
                    "n_layers": 1,
                    "n_leaf": 1
                }
            }
        },
        "test_nominal_c4_multiple_patches": {
            "inputs": {
                "c3psn": jnp.array([0.0, 0.0, 0.0]),
                "g0_BB": jnp.array([0.04, 0.04, 0.04]),
                "g1_BB": jnp.array([4.0, 4.0, 4.0]),
                "g0_MED": jnp.array([0.0, 0.0, 0.0]),
                "g1_MED": jnp.array([1.6, 1.6, 1.6]),
                "psi50_gs": jnp.array([-1.5, -1.5, -1.5]),
                "shape_gs": jnp.array([2.5, 2.5, 2.5]),
                "gsmin_SPA": jnp.array([0.001, 0.001, 0.001]),
                "iota_SPA": jnp.array([0.0001, 0.0001, 0.0001]),
                "tacclim": jnp.array([303.15, 303.15, 303.15]),
                "ncan": jnp.array([2, 2, 2]),
                "dpai": jnp.array([[1.5, 1.0], [2.0, 1.5], [1.8, 1.2]]),
                "eair": jnp.array([[1800.0, 1700.0], [1900.0, 1800.0], [1850.0, 1750.0]]),
                "o2ref": jnp.array([209.0, 209.0, 209.0]),
                "pref": jnp.array([101325.0, 101325.0, 101325.0]),
                "cair": jnp.array([[[380.0], [370.0]], [[390.0], [380.0]], [[385.0], [375.0]]]),
                "vcmax25": jnp.array([[[80.0], [70.0]], [[85.0], [75.0]], [[82.0], [72.0]]]),
                "jmax25": jnp.array([[[160.0], [140.0]], [[170.0], [150.0]], [[165.0], [145.0]]]),
                "kp25": jnp.array([[[50.0], [45.0]], [[55.0], [50.0]], [[52.0], [47.0]]]),
                "rd25": jnp.array([[[1.6], [1.4]], [[1.7], [1.5]], [[1.65], [1.45]]]),
                "tleaf": jnp.array([[[303.15], [302.15]], [[304.15], [303.15]], [[303.65], [302.65]]]),
                "gbv": jnp.array([[[2.0], [1.8]], [[2.2], [2.0]], [[2.1], [1.9]]]),
                "gbc": jnp.array([[[1.3], [1.2]], [[1.4], [1.3]], [[1.35], [1.25]]]),
                "apar": jnp.array([[[1500.0], [800.0]], [[1600.0], [900.0]], [[1550.0], [850.0]]]),
                "lwp": jnp.array([[[-0.3], [-0.4]], [[-0.35], [-0.45]], [[-0.32], [-0.42]]]),
            },
            "metadata": {
                "type": "nominal",
                "description": "C4 photosynthesis with multiple patches and canopy layers",
                "expected_shapes": {
                    "n_patches": 3,
                    "n_layers": 2,
                    "n_leaf": 1
                }
            }
        },
        "test_nominal_mixed_c3_c4_multilayer": {
            "inputs": {
                "c3psn": jnp.array([1.0, 0.0]),
                "g0_BB": jnp.array([0.01, 0.04]),
                "g1_BB": jnp.array([9.0, 4.0]),
                "g0_MED": jnp.array([0.0, 0.0]),
                "g1_MED": jnp.array([4.0, 1.6]),
                "psi50_gs": jnp.array([-2.0, -1.5]),
                "shape_gs": jnp.array([3.0, 2.5]),
                "gsmin_SPA": jnp.array([0.001, 0.001]),
                "iota_SPA": jnp.array([0.0001, 0.0001]),
                "tacclim": jnp.array([298.15, 303.15]),
                "ncan": jnp.array([3, 3]),
                "dpai": jnp.array([[2.0, 1.5, 1.0], [1.8, 1.3, 0.8]]),
                "eair": jnp.array([[1500.0, 1450.0, 1400.0], [1800.0, 1750.0, 1700.0]]),
                "o2ref": jnp.array([209.0, 209.0]),
                "pref": jnp.array([101325.0, 101325.0]),
                "cair": jnp.array([[[400.0, 410.0], [395.0, 405.0], [390.0, 400.0]], 
                                   [[380.0, 390.0], [375.0, 385.0], [370.0, 380.0]]]),
                "vcmax25": jnp.array([[[60.0, 55.0], [50.0, 45.0], [40.0, 35.0]], 
                                      [[80.0, 75.0], [70.0, 65.0], [60.0, 55.0]]]),
                "jmax25": jnp.array([[[120.0, 110.0], [100.0, 90.0], [80.0, 70.0]], 
                                     [[160.0, 150.0], [140.0, 130.0], [120.0, 110.0]]]),
                "kp25": jnp.array([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 
                                   [[50.0, 48.0], [45.0, 43.0], [40.0, 38.0]]]),
                "rd25": jnp.array([[[1.2, 1.1], [1.0, 0.9], [0.8, 0.7]], 
                                   [[1.6, 1.5], [1.4, 1.3], [1.2, 1.1]]]),
                "tleaf": jnp.array([[[298.15, 299.15], [297.15, 298.15], [296.15, 297.15]], 
                                    [[303.15, 304.15], [302.15, 303.15], [301.15, 302.15]]]),
                "gbv": jnp.array([[[1.5, 1.6], [1.4, 1.5], [1.3, 1.4]], 
                                  [[2.0, 2.1], [1.9, 2.0], [1.8, 1.9]]]),
                "gbc": jnp.array([[[1.0, 1.05], [0.95, 1.0], [0.9, 0.95]], 
                                  [[1.3, 1.35], [1.25, 1.3], [1.2, 1.25]]]),
                "apar": jnp.array([[[1200.0, 1100.0], [800.0, 700.0], [400.0, 300.0]], 
                                   [[1500.0, 1400.0], [1000.0, 900.0], [500.0, 400.0]]]),
                "lwp": jnp.array([[[-0.5, -0.6], [-0.7, -0.8], [-0.9, -1.0]], 
                                  [[-0.3, -0.4], [-0.5, -0.6], [-0.7, -0.8]]]),
            },
            "metadata": {
                "type": "nominal",
                "description": "Mixed C3 and C4 vegetation with multiple canopy layers and sunlit/shaded leaves",
                "expected_shapes": {
                    "n_patches": 2,
                    "n_layers": 3,
                    "n_leaf": 2
                }
            }
        },
        "test_edge_zero_apar_low_light": {
            "inputs": {
                "c3psn": jnp.array([1.0, 1.0]),
                "g0_BB": jnp.array([0.01, 0.01]),
                "g1_BB": jnp.array([9.0, 9.0]),
                "g0_MED": jnp.array([0.0, 0.0]),
                "g1_MED": jnp.array([4.0, 4.0]),
                "psi50_gs": jnp.array([-2.0, -2.0]),
                "shape_gs": jnp.array([3.0, 3.0]),
                "gsmin_SPA": jnp.array([0.001, 0.001]),
                "iota_SPA": jnp.array([0.0001, 0.0001]),
                "tacclim": jnp.array([298.15, 298.15]),
                "ncan": jnp.array([1, 1]),
                "dpai": jnp.array([[2.0], [2.0]]),
                "eair": jnp.array([[1500.0], [1500.0]]),
                "o2ref": jnp.array([209.0, 209.0]),
                "pref": jnp.array([101325.0, 101325.0]),
                "cair": jnp.array([[[400.0]], [[400.0]]]),
                "vcmax25": jnp.array([[[60.0]], [[60.0]]]),
                "jmax25": jnp.array([[[120.0]], [[120.0]]]),
                "kp25": jnp.array([[[0.0]], [[0.0]]]),
                "rd25": jnp.array([[[1.2]], [[1.2]]]),
                "tleaf": jnp.array([[[298.15]], [[298.15]]]),
                "gbv": jnp.array([[[1.5]], [[1.5]]]),
                "gbc": jnp.array([[[1.0]], [[1.0]]]),
                "apar": jnp.array([[[0.0]], [[10.0]]]),
                "lwp": jnp.array([[[-0.5]], [[-0.5]]]),
            },
            "metadata": {
                "type": "edge",
                "description": "Zero and very low PAR conditions testing dark respiration",
                "edge_cases": ["zero_radiation", "low_light"]
            }
        },
        "test_edge_severe_water_stress": {
            "inputs": {
                "c3psn": jnp.array([1.0, 0.0]),
                "g0_BB": jnp.array([0.01, 0.04]),
                "g1_BB": jnp.array([9.0, 4.0]),
                "g0_MED": jnp.array([0.0, 0.0]),
                "g1_MED": jnp.array([4.0, 1.6]),
                "psi50_gs": jnp.array([-2.0, -1.5]),
                "shape_gs": jnp.array([3.0, 2.5]),
                "gsmin_SPA": jnp.array([0.001, 0.001]),
                "iota_SPA": jnp.array([0.0001, 0.0001]),
                "tacclim": jnp.array([298.15, 303.15]),
                "ncan": jnp.array([1, 1]),
                "dpai": jnp.array([[2.0], [2.0]]),
                "eair": jnp.array([[1500.0], [1800.0]]),
                "o2ref": jnp.array([209.0, 209.0]),
                "pref": jnp.array([101325.0, 101325.0]),
                "cair": jnp.array([[[400.0]], [[380.0]]]),
                "vcmax25": jnp.array([[[60.0]], [[80.0]]]),
                "jmax25": jnp.array([[[120.0]], [[160.0]]]),
                "kp25": jnp.array([[[0.0]], [[50.0]]]),
                "rd25": jnp.array([[[1.2]], [[1.6]]]),
                "tleaf": jnp.array([[[298.15]], [[303.15]]]),
                "gbv": jnp.array([[[1.5]], [[2.0]]]),
                "gbc": jnp.array([[[1.0]], [[1.3]]]),
                "apar": jnp.array([[[1000.0]], [[1500.0]]]),
                "lwp": jnp.array([[[-5.0]], [[-4.5]]]),
            },
            "metadata": {
                "type": "edge",
                "description": "Severe water stress with leaf water potential well below psi50_gs",
                "edge_cases": ["severe_drought", "stomatal_closure"]
            }
        },
        "test_edge_temperature_extremes": {
            "inputs": {
                "c3psn": jnp.array([1.0, 1.0, 0.0]),
                "g0_BB": jnp.array([0.01, 0.01, 0.04]),
                "g1_BB": jnp.array([9.0, 9.0, 4.0]),
                "g0_MED": jnp.array([0.0, 0.0, 0.0]),
                "g1_MED": jnp.array([4.0, 4.0, 1.6]),
                "psi50_gs": jnp.array([-2.0, -2.0, -1.5]),
                "shape_gs": jnp.array([3.0, 3.0, 2.5]),
                "gsmin_SPA": jnp.array([0.001, 0.001, 0.001]),
                "iota_SPA": jnp.array([0.0001, 0.0001, 0.0001]),
                "tacclim": jnp.array([278.15, 313.15, 308.15]),
                "ncan": jnp.array([1, 1, 1]),
                "dpai": jnp.array([[2.0], [2.0], [2.0]]),
                "eair": jnp.array([[800.0], [2500.0], [2200.0]]),
                "o2ref": jnp.array([209.0, 209.0, 209.0]),
                "pref": jnp.array([101325.0, 101325.0, 101325.0]),
                "cair": jnp.array([[[400.0]], [[400.0]], [[380.0]]]),
                "vcmax25": jnp.array([[[60.0]], [[60.0]], [[80.0]]]),
                "jmax25": jnp.array([[[120.0]], [[120.0]], [[160.0]]]),
                "kp25": jnp.array([[[0.0]], [[0.0]], [[50.0]]]),
                "rd25": jnp.array([[[1.2]], [[1.2]], [[1.6]]]),
                "tleaf": jnp.array([[[278.15]], [[318.15]], [[313.15]]]),
                "gbv": jnp.array([[[1.5]], [[1.5]], [[2.0]]]),
                "gbc": jnp.array([[[1.0]], [[1.0]], [[1.3]]]),
                "apar": jnp.array([[[500.0]], [[1200.0]], [[1500.0]]]),
                "lwp": jnp.array([[[-0.5]], [[-1.5]], [[-0.8]]]),
            },
            "metadata": {
                "type": "edge",
                "description": "Temperature extremes: cold stress (5°C), heat stress (45°C), and high temperature (40°C)",
                "edge_cases": ["cold_stress", "heat_stress", "temperature_limits"]
            }
        },
        "test_edge_minimal_conductances": {
            "inputs": {
                "c3psn": jnp.array([1.0]),
                "g0_BB": jnp.array([0.0]),
                "g1_BB": jnp.array([0.1]),
                "g0_MED": jnp.array([0.0]),
                "g1_MED": jnp.array([0.1]),
                "psi50_gs": jnp.array([-2.0]),
                "shape_gs": jnp.array([3.0]),
                "gsmin_SPA": jnp.array([0.0001]),
                "iota_SPA": jnp.array([1e-05]),
                "tacclim": jnp.array([298.15]),
                "ncan": jnp.array([1]),
                "dpai": jnp.array([[2.0]]),
                "eair": jnp.array([[1500.0]]),
                "o2ref": jnp.array([209.0]),
                "pref": jnp.array([101325.0]),
                "cair": jnp.array([[[400.0]]]),
                "vcmax25": jnp.array([[[60.0]]]),
                "jmax25": jnp.array([[[120.0]]]),
                "kp25": jnp.array([[[0.0]]]),
                "rd25": jnp.array([[[1.2]]]),
                "tleaf": jnp.array([[[298.15]]]),
                "gbv": jnp.array([[[0.1]]]),
                "gbc": jnp.array([[[0.05]]]),
                "apar": jnp.array([[[1000.0]]]),
                "lwp": jnp.array([[[-0.5]]]),
            },
            "metadata": {
                "type": "edge",
                "description": "Minimal stomatal and boundary layer conductances testing diffusion limitations",
                "edge_cases": ["minimal_conductance", "diffusion_limited"]
            }
        },
        "test_edge_high_co2_enrichment": {
            "inputs": {
                "c3psn": jnp.array([1.0, 0.0]),
                "g0_BB": jnp.array([0.01, 0.04]),
                "g1_BB": jnp.array([9.0, 4.0]),
                "g0_MED": jnp.array([0.0, 0.0]),
                "g1_MED": jnp.array([4.0, 1.6]),
                "psi50_gs": jnp.array([-2.0, -1.5]),
                "shape_gs": jnp.array([3.0, 2.5]),
                "gsmin_SPA": jnp.array([0.001, 0.001]),
                "iota_SPA": jnp.array([0.0001, 0.0001]),
                "tacclim": jnp.array([298.15, 303.15]),
                "ncan": jnp.array([1, 1]),
                "dpai": jnp.array([[2.0], [2.0]]),
                "eair": jnp.array([[1500.0], [1800.0]]),
                "o2ref": jnp.array([209.0, 209.0]),
                "pref": jnp.array([101325.0, 101325.0]),
                "cair": jnp.array([[[1000.0]], [[1200.0]]]),
                "vcmax25": jnp.array([[[60.0]], [[80.0]]]),
                "jmax25": jnp.array([[[120.0]], [[160.0]]]),
                "kp25": jnp.array([[[0.0]], [[50.0]]]),
                "rd25": jnp.array([[[1.2]], [[1.6]]]),
                "tleaf": jnp.array([[[298.15]], [[303.15]]]),
                "gbv": jnp.array([[[1.5]], [[2.0]]]),
                "gbc": jnp.array([[[1.0]], [[1.3]]]),
                "apar": jnp.array([[[1000.0]], [[1500.0]]]),
                "lwp": jnp.array([[[-0.5]], [[-0.3]]]),
            },
            "metadata": {
                "type": "edge",
                "description": "Elevated CO2 concentrations (1000-1200 ppm) testing CO2 saturation effects",
                "edge_cases": ["high_co2", "co2_saturation"]
            }
        },
        "test_special_high_altitude_low_pressure": {
            "inputs": {
                "c3psn": jnp.array([1.0, 1.0]),
                "g0_BB": jnp.array([0.01, 0.01]),
                "g1_BB": jnp.array([9.0, 9.0]),
                "g0_MED": jnp.array([0.0, 0.0]),
                "g1_MED": jnp.array([4.0, 4.0]),
                "psi50_gs": jnp.array([-2.0, -2.0]),
                "shape_gs": jnp.array([3.0, 3.0]),
                "gsmin_SPA": jnp.array([0.001, 0.001]),
                "iota_SPA": jnp.array([0.0001, 0.0001]),
                "tacclim": jnp.array([288.15, 288.15]),
                "ncan": jnp.array([1, 1]),
                "dpai": jnp.array([[1.5], [1.5]]),
                "eair": jnp.array([[1000.0], [1000.0]]),
                "o2ref": jnp.array([209.0, 209.0]),
                "pref": jnp.array([70000.0, 60000.0]),
                "cair": jnp.array([[[400.0]], [[400.0]]]),
                "vcmax25": jnp.array([[[50.0]], [[50.0]]]),
                "jmax25": jnp.array([[[100.0]], [[100.0]]]),
                "kp25": jnp.array([[[0.0]], [[0.0]]]),
                "rd25": jnp.array([[[1.0]], [[1.0]]]),
                "tleaf": jnp.array([[[288.15]], [[288.15]]]),
                "gbv": jnp.array([[[1.2]], [[1.2]]]),
                "gbc": jnp.array([[[0.8]], [[0.8]]]),
                "apar": jnp.array([[[1200.0]], [[1200.0]]]),
                "lwp": jnp.array([[[-0.8]], [[-0.8]]]),
            },
            "metadata": {
                "type": "special",
                "description": "High altitude conditions with reduced atmospheric pressure (3000m and 4000m elevation)",
                "edge_cases": ["low_pressure", "high_altitude"]
            }
        },
        "test_special_dense_canopy_deep_shade": {
            "inputs": {
                "c3psn": jnp.array([1.0]),
                "g0_BB": jnp.array([0.01]),
                "g1_BB": jnp.array([9.0]),
                "g0_MED": jnp.array([0.0]),
                "g1_MED": jnp.array([4.0]),
                "psi50_gs": jnp.array([-2.0]),
                "shape_gs": jnp.array([3.0]),
                "gsmin_SPA": jnp.array([0.001]),
                "iota_SPA": jnp.array([0.0001]),
                "tacclim": jnp.array([298.15]),
                "ncan": jnp.array([5]),
                "dpai": jnp.array([[1.5, 1.5, 1.5, 1.5, 1.5]]),
                "eair": jnp.array([[1600.0, 1580.0, 1560.0, 1540.0, 1520.0]]),
                "o2ref": jnp.array([209.0]),
                "pref": jnp.array([101325.0]),
                "cair": jnp.array([[[410.0, 400.0], [420.0, 410.0], [430.0, 420.0], [440.0, 430.0], [450.0, 440.0]]]),
                "vcmax25": jnp.array([[[65.0, 60.0], [55.0, 50.0], [45.0, 40.0], [35.0, 30.0], [25.0, 20.0]]]),
                "jmax25": jnp.array([[[130.0, 120.0], [110.0, 100.0], [90.0, 80.0], [70.0, 60.0], [50.0, 40.0]]]),
                "kp25": jnp.array([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
                "rd25": jnp.array([[[1.3, 1.2], [1.1, 1.0], [0.9, 0.8], [0.7, 0.6], [0.5, 0.4]]]),
                "tleaf": jnp.array([[[298.15, 299.15], [297.65, 298.65], [297.15, 298.15], [296.65, 297.65], [296.15, 297.15]]]),
                "gbv": jnp.array([[[1.6, 1.7], [1.5, 1.6], [1.4, 1.5], [1.3, 1.4], [1.2, 1.3]]]),
                "gbc": jnp.array([[[1.05, 1.1], [1.0, 1.05], [0.95, 1.0], [0.9, 0.95], [0.85, 0.9]]]),
                "apar": jnp.array([[[1800.0, 1600.0], [900.0, 700.0], [400.0, 300.0], [150.0, 100.0], [50.0, 30.0]]]),
                "lwp": jnp.array([[[-0.4, -0.5], [-0.6, -0.7], [-0.8, -0.9], [-1.0, -1.1], [-1.2, -1.3]]]),
            },
            "metadata": {
                "type": "special",
                "description": "Dense canopy with 5 layers showing strong light gradient and CO2 depletion",
                "edge_cases": ["deep_shade", "co2_depletion", "dense_canopy"]
            }
        },
    }


class TestLeafPhotosynthesisShapes:
    """Test suite for verifying output shapes of leaf_photosynthesis function."""
    
    @pytest.mark.parametrize("test_case_name", [
        "test_nominal_c3_single_patch_single_layer",
        "test_nominal_c4_multiple_patches",
        "test_nominal_mixed_c3_c4_multilayer",
    ])
    def test_output_shapes_nominal(self, test_data, default_params, test_case_name):
        """
        Test that leaf_photosynthesis returns correctly shaped outputs for nominal cases.
        
        Verifies that all output fields in LeafPhotosynthesisState have shapes
        consistent with input dimensions (n_patches, n_layers, n_leaf).
        """
        test_case = test_data[test_case_name]
        inputs = test_case["inputs"]
        expected = test_case["metadata"]["expected_shapes"]
        
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        # Check that result is a LeafPhotosynthesisState
        assert isinstance(result, LeafPhotosynthesisState), \
            f"Expected LeafPhotosynthesisState, got {type(result)}"
        
        # Expected shapes for different field types
        patch_shape = (expected["n_patches"],)
        layer_shape = (expected["n_patches"], expected["n_layers"], expected["n_leaf"])
        
        # Fields that should have patch-level shape
        patch_fields = ["g0", "g1", "btran"]
        
        # Fields that should have layer-level shape
        layer_fields = ["kc", "ko", "cp", "vcmax", "jmax", "je", "kp", "rd", 
                       "ci", "hs", "vpd", "ceair", "leaf_esat", "gspot",
                       "ac", "aj", "ap", "agross", "anet", "cs", "gs", "alphapsn"]
        
        # Check patch-level fields
        for field in patch_fields:
            field_value = getattr(result, field)
            assert field_value.shape == patch_shape, \
                f"Field {field} has shape {field_value.shape}, expected {patch_shape}"
        
        # Check layer-level fields
        for field in layer_fields:
            field_value = getattr(result, field)
            assert field_value.shape == layer_shape, \
                f"Field {field} has shape {field_value.shape}, expected {layer_shape}"
    
    @pytest.mark.parametrize("test_case_name", [
        "test_edge_zero_apar_low_light",
        "test_edge_severe_water_stress",
        "test_edge_temperature_extremes",
        "test_special_dense_canopy_deep_shade",
    ])
    def test_output_shapes_edge_cases(self, test_data, default_params, test_case_name):
        """
        Test that leaf_photosynthesis returns correctly shaped outputs for edge cases.
        
        Edge cases include extreme conditions that should still produce valid output shapes.
        """
        test_case = test_data[test_case_name]
        inputs = test_case["inputs"]
        
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        # Infer expected shapes from inputs
        n_patches = inputs["c3psn"].shape[0]
        n_layers = inputs["dpai"].shape[1]
        n_leaf = inputs["cair"].shape[2]
        
        patch_shape = (n_patches,)
        layer_shape = (n_patches, n_layers, n_leaf)
        
        # Verify all fields have correct shapes
        patch_fields = ["g0", "g1", "btran"]
        layer_fields = ["kc", "ko", "cp", "vcmax", "jmax", "je", "kp", "rd", 
                       "ci", "hs", "vpd", "ceair", "leaf_esat", "gspot",
                       "ac", "aj", "ap", "agross", "anet", "cs", "gs", "alphapsn"]
        
        for field in patch_fields:
            assert getattr(result, field).shape == patch_shape, \
                f"Edge case {test_case_name}: Field {field} has incorrect shape"
        
        for field in layer_fields:
            assert getattr(result, field).shape == layer_shape, \
                f"Edge case {test_case_name}: Field {field} has incorrect shape"


class TestLeafPhotosynthesisDtypes:
    """Test suite for verifying data types of leaf_photosynthesis outputs."""
    
    def test_output_dtypes(self, test_data, default_params):
        """
        Test that all output fields have float dtype.
        
        All photosynthesis calculations should produce floating-point results.
        """
        test_case = test_data["test_nominal_c3_single_patch_single_layer"]
        inputs = test_case["inputs"]
        
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        # All fields should be float arrays
        for field_name in result._fields:
            field_value = getattr(result, field_name)
            assert jnp.issubdtype(field_value.dtype, jnp.floating), \
                f"Field {field_name} has dtype {field_value.dtype}, expected floating point"


class TestLeafPhotosynthesisValues:
    """Test suite for verifying output values and physical realism."""
    
    def test_nominal_c3_positive_photosynthesis(self, test_data, default_params):
        """
        Test that C3 photosynthesis under optimal conditions produces positive net assimilation.
        
        With adequate light, water, and temperature, net photosynthesis should exceed
        respiration, resulting in positive anet.
        """
        test_case = test_data["test_nominal_c3_single_patch_single_layer"]
        inputs = test_case["inputs"]
        
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        # Net assimilation should be positive under optimal conditions
        assert jnp.all(result.anet > 0), \
            f"Expected positive net assimilation, got {result.anet}"
        
        # Gross assimilation should be greater than net
        assert jnp.all(result.agross >= result.anet), \
            "Gross assimilation should be >= net assimilation"
        
        # Stomatal conductance should be positive
        assert jnp.all(result.gs > 0), \
            f"Expected positive stomatal conductance, got {result.gs}"
    
    def test_zero_apar_negative_anet(self, test_data, default_params):
        """
        Test that zero PAR results in negative net assimilation (dark respiration).
        
        Without light, photosynthesis cannot occur, so net assimilation should equal
        negative respiration.
        """
        test_case = test_data["test_edge_zero_apar_low_light"]
        inputs = test_case["inputs"]
        
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        # First patch has zero APAR
        zero_apar_idx = 0
        assert jnp.all(result.anet[zero_apar_idx, :, :] < 0), \
            "Expected negative net assimilation (respiration) with zero PAR"
        
        # Gross assimilation should be near zero
        assert jnp.all(result.agross[zero_apar_idx, :, :] < 1.0), \
            "Expected near-zero gross assimilation with zero PAR"
    
    def test_severe_water_stress_reduces_conductance(self, test_data, default_params):
        """
        Test that severe water stress significantly reduces stomatal conductance.
        
        When leaf water potential is well below psi50_gs, stomatal conductance
        should be strongly reduced via the btran factor.
        """
        test_case = test_data["test_edge_severe_water_stress"]
        inputs = test_case["inputs"]
        
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        # btran should be very low under severe stress
        assert jnp.all(result.btran < 0.2), \
            f"Expected low btran under severe water stress, got {result.btran}"
        
        # Stomatal conductance should be reduced
        # Compare to minimum conductance
        assert jnp.all(result.gs < 0.1), \
            f"Expected low stomatal conductance under severe stress, got {result.gs}"
    
    def test_temperature_effects_on_kinetics(self, test_data, default_params):
        """
        Test that temperature extremes affect enzyme kinetics appropriately.
        
        Cold temperatures should reduce enzyme activity, while high temperatures
        may cause heat stress and reduced photosynthesis.
        """
        test_case = test_data["test_edge_temperature_extremes"]
        inputs = test_case["inputs"]
        
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        # Cold stress (patch 0): reduced vcmax
        cold_idx = 0
        # Heat stress (patch 1): potentially reduced photosynthesis
        heat_idx = 1
        
        # Vcmax should be temperature-dependent
        assert result.vcmax[cold_idx, 0, 0] < result.vcmax[heat_idx, 0, 0], \
            "Expected lower vcmax at cold temperature"
        
        # All vcmax values should be positive
        assert jnp.all(result.vcmax > 0), \
            "Vcmax should remain positive at all temperatures"
    
    def test_high_co2_increases_ci(self, test_data, default_params):
        """
        Test that elevated CO2 increases intercellular CO2 concentration.
        
        Higher atmospheric CO2 should lead to higher ci values.
        """
        test_case = test_data["test_edge_high_co2_enrichment"]
        inputs = test_case["inputs"]
        
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        # ci should be elevated with high CO2
        # For C3 plants, ci is typically 0.6-0.8 of ca
        # With 1000 ppm CO2, ci should be > 600 ppm
        assert jnp.all(result.ci > 600.0), \
            f"Expected elevated ci with high CO2, got {result.ci}"
    
    def test_c4_vs_c3_differences(self, test_data, default_params):
        """
        Test that C4 plants show expected differences from C3 plants.
        
        C4 plants should have:
        - Non-zero kp (PEP carboxylase activity)
        - Different ci/ca ratios
        - Different response to CO2
        """
        test_case = test_data["test_nominal_mixed_c3_c4_multilayer"]
        inputs = test_case["inputs"]
        
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        c3_idx = 0
        c4_idx = 1
        
        # C4 should have positive kp
        assert jnp.all(result.kp[c4_idx, :, :] > 0), \
            "C4 plants should have positive kp"
        
        # C3 should have zero kp
        assert jnp.all(result.kp[c3_idx, :, :] == 0), \
            "C3 plants should have zero kp"
        
        # C4 typically has lower ci/ca ratio than C3
        c3_ci_ca_ratio = result.ci[c3_idx, 0, 0] / inputs["cair"][c3_idx, 0, 0]
        c4_ci_ca_ratio = result.ci[c4_idx, 0, 0] / inputs["cair"][c4_idx, 0, 0]
        
        assert c4_ci_ca_ratio < c3_ci_ca_ratio, \
            "C4 plants should have lower ci/ca ratio than C3"
    
    def test_physical_constraints(self, test_data, default_params):
        """
        Test that outputs satisfy physical constraints.
        
        Verifies:
        - Non-negative rates (gs, agross, rd)
        - Reasonable ranges for ci, vpd
        - Energy conservation (agross >= anet)
        """
        test_case = test_data["test_nominal_mixed_c3_c4_multilayer"]
        inputs = test_case["inputs"]
        
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        # Stomatal conductance should be non-negative
        assert jnp.all(result.gs >= 0), \
            "Stomatal conductance must be non-negative"
        
        # Gross assimilation should be non-negative
        assert jnp.all(result.agross >= 0), \
            "Gross assimilation must be non-negative"
        
        # Respiration should be non-negative
        assert jnp.all(result.rd >= 0), \
            "Respiration must be non-negative"
        
        # Gross >= Net (accounting for respiration)
        assert jnp.all(result.agross >= result.anet), \
            "Gross assimilation must be >= net assimilation"
        
        # ci should be less than ca (for most conditions)
        for i in range(inputs["cair"].shape[0]):
            for j in range(inputs["cair"].shape[1]):
                for k in range(inputs["cair"].shape[2]):
                    # Allow some tolerance for numerical issues
                    assert result.ci[i, j, k] <= inputs["cair"][i, j, k] * 1.1, \
                        f"ci should not greatly exceed ca at [{i},{j},{k}]"
        
        # VPD should be non-negative
        assert jnp.all(result.vpd >= 0), \
            "VPD must be non-negative"
        
        # btran should be in [0, 1]
        assert jnp.all((result.btran >= 0) & (result.btran <= 1)), \
            "btran must be in range [0, 1]"


class TestLeafPhotosynthesisEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_minimal_conductances(self, test_data, default_params):
        """
        Test behavior with minimal stomatal and boundary layer conductances.
        
        Very low conductances should strongly limit photosynthesis through
        diffusion constraints.
        """
        test_case = test_data["test_edge_minimal_conductances"]
        inputs = test_case["inputs"]
        
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        # With minimal conductances, photosynthesis should be strongly limited
        # gs should be very low
        assert jnp.all(result.gs < 0.2), \
            f"Expected very low gs with minimal conductances, got {result.gs}"
        
        # Net assimilation should be reduced
        assert jnp.all(result.anet < 10.0), \
            "Expected reduced photosynthesis with minimal conductances"
    
    def test_high_altitude_low_pressure(self, test_data, default_params):
        """
        Test photosynthesis at high altitude with reduced atmospheric pressure.
        
        Lower pressure affects partial pressures of gases and may reduce
        photosynthetic rates.
        """
        test_case = test_data["test_special_high_altitude_low_pressure"]
        inputs = test_case["inputs"]
        
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        # Photosynthesis should still occur but may be reduced
        # All outputs should be finite and physically reasonable
        assert jnp.all(jnp.isfinite(result.anet)), \
            "Net assimilation should be finite at high altitude"
        
        assert jnp.all(jnp.isfinite(result.gs)), \
            "Stomatal conductance should be finite at high altitude"
        
        # Compare two altitudes (3000m vs 4000m)
        # Higher altitude (lower pressure) may have slightly reduced rates
        assert result.anet[0, 0, 0] >= result.anet[1, 0, 0] * 0.8, \
            "Photosynthesis at 3000m should not be much less than at 4000m"
    
    def test_dense_canopy_light_gradient(self, test_data, default_params):
        """
        Test photosynthesis in dense canopy with strong light gradient.
        
        Lower canopy layers with very low light should have reduced or negative
        net assimilation.
        """
        test_case = test_data["test_special_dense_canopy_deep_shade"]
        inputs = test_case["inputs"]
        
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        # Top layer should have positive photosynthesis
        assert result.anet[0, 0, 0] > 0, \
            "Top canopy layer should have positive photosynthesis"
        
        # Bottom layer with very low light may have negative net assimilation
        bottom_layer = 4
        # With 30-50 umol/m2/s PAR, photosynthesis should be very low
        assert result.anet[0, bottom_layer, 1] < 2.0, \
            "Deep shade should have very low photosynthesis"
        
        # Photosynthesis should generally decrease with depth
        for layer in range(4):
            assert result.anet[0, layer, 0] >= result.anet[0, layer + 1, 0] * 0.5, \
                f"Photosynthesis should decrease with canopy depth (layer {layer})"
    
    def test_no_nan_or_inf_outputs(self, test_data, default_params):
        """
        Test that no outputs contain NaN or Inf values across all test cases.
        
        Even under extreme conditions, the function should produce finite values.
        """
        for test_case_name, test_case in test_data.items():
            inputs = test_case["inputs"]
            
            result = leaf_photosynthesis(**inputs, params=default_params)
            
            # Check all fields for NaN or Inf
            for field_name in result._fields:
                field_value = getattr(result, field_name)
                assert jnp.all(jnp.isfinite(field_value)), \
                    f"Test {test_case_name}: Field {field_name} contains NaN or Inf"


class TestLeafPhotosynthesisConsistency:
    """Test suite for internal consistency checks."""
    
    def test_co_limitation_consistency(self, test_data, default_params):
        """
        Test that photosynthesis rates satisfy co-limitation relationships.
        
        For C3 plants: agross should be related to min(ac, aj)
        For C4 plants: agross should be related to min(ac, aj, ap)
        """
        test_case = test_data["test_nominal_mixed_c3_c4_multilayer"]
        inputs = test_case["inputs"]
        
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        c3_idx = 0
        c4_idx = 1
        
        # For C3: agross should be close to min(ac, aj) - rd
        c3_min_rate = jnp.minimum(result.ac[c3_idx, 0, 0], result.aj[c3_idx, 0, 0])
        # Allow some tolerance for co-limitation smoothing
        assert result.agross[c3_idx, 0, 0] <= c3_min_rate * 1.1, \
            "C3 gross assimilation should not greatly exceed min(ac, aj)"
        
        # For C4: agross should be related to min(ac, aj, ap)
        c4_min_rate = jnp.minimum(
            jnp.minimum(result.ac[c4_idx, 0, 0], result.aj[c4_idx, 0, 0]),
            result.ap[c4_idx, 0, 0]
        )
        assert result.agross[c4_idx, 0, 0] <= c4_min_rate * 1.1, \
            "C4 gross assimilation should not greatly exceed min(ac, aj, ap)"
    
    def test_respiration_consistency(self, test_data, default_params):
        """
        Test that respiration is consistently applied.
        
        Verifies: anet = agross - rd
        """
        test_case = test_data["test_nominal_c3_single_patch_single_layer"]
        inputs = test_case["inputs"]
        
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        # Check anet = agross - rd relationship
        expected_anet = result.agross - result.rd
        
        assert jnp.allclose(result.anet, expected_anet, rtol=1e-5, atol=1e-6), \
            "Net assimilation should equal gross assimilation minus respiration"
    
    def test_conductance_hierarchy(self, test_data, default_params):
        """
        Test that conductances follow expected hierarchy.
        
        Total conductance should not exceed individual conductances.
        """
        test_case = test_data["test_nominal_c3_single_patch_single_layer"]
        inputs = test_case["inputs"]
        
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        # Stomatal conductance should be positive
        assert jnp.all(result.gs > 0), \
            "Stomatal conductance should be positive"
        
        # gs should generally be less than boundary layer conductance
        # (though not always due to different units and conversions)
        # Just verify gs is in reasonable range
        assert jnp.all(result.gs < 10.0), \
            "Stomatal conductance should be in reasonable range"


class TestLeafPhotosynthesisDocumentation:
    """Test suite for documentation and metadata validation."""
    
    def test_function_has_docstring(self):
        """Test that the leaf_photosynthesis function has documentation."""
        assert leaf_photosynthesis.__doc__ is not None, \
            "Function should have a docstring"
    
    def test_namedtuple_fields_exist(self, test_data, default_params):
        """
        Test that LeafPhotosynthesisState contains all expected fields.
        
        Verifies that the output namedtuple has all documented fields.
        """
        expected_fields = [
            "g0", "g1", "btran", "kc", "ko", "cp", "vcmax", "jmax", "je", "kp", "rd",
            "ci", "hs", "vpd", "ceair", "leaf_esat", "gspot",
            "ac", "aj", "ap", "agross", "anet", "cs", "gs", "alphapsn"
        ]
        
        test_case = test_data["test_nominal_c3_single_patch_single_layer"]
        inputs = test_case["inputs"]
        
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        for field in expected_fields:
            assert hasattr(result, field), \
                f"LeafPhotosynthesisState should have field '{field}'"
    
    def test_params_namedtuple_fields(self, default_params):
        """
        Test that PhotosynthesisParams contains all expected fields.
        
        Verifies that the params namedtuple has all documented configuration fields.
        """
        expected_fields = [
            "tfrz", "rgas", "kc25", "ko25", "cp25", "kcha", "koha", "cpha",
            "vcmaxha_noacclim", "vcmaxha_acclim", "jmaxha_noacclim", "jmaxha_acclim",
            "vcmaxhd_noacclim", "vcmaxhd_acclim", "jmaxhd_noacclim", "jmaxhd_acclim",
            "vcmaxse_noacclim", "vcmaxse_acclim", "jmaxse_noacclim", "jmaxse_acclim",
            "rdha", "rdhd", "rdse", "phi_psii", "theta_j", "vpd_min_med", "rh_min_bb",
            "dh2o_to_dco2", "qe_c4", "colim_c3a", "colim_c4a", "colim_c4b",
            "gs_type", "acclim_type", "gspot_type", "colim_type"
        ]
        
        for field in expected_fields:
            assert hasattr(default_params, field), \
                f"PhotosynthesisParams should have field '{field}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])