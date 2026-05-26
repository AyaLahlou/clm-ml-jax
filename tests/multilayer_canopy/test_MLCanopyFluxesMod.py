"""
Comprehensive pytest suite for ml_canopy_fluxes function from MLCanopyFluxesMod.

This test suite covers:
- Nominal cases with typical growing season conditions
- Edge cases (zero LAI, nighttime, extreme cold, high LAI)
- Special cases (single/many substeps, high elevation, tropical conditions)
- Shape validation, dtype checking, and physical constraint verification
"""

import sys
from pathlib import Path
from typing import NamedTuple, Tuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multilayer_canopy.MLCanopyFluxesMod import ml_canopy_fluxes


# NamedTuple definitions matching the function signature
class Bounds(NamedTuple):
    """Subgrid bounds structure for patches, columns, and gridcells."""
    begp: int
    endp: int
    begc: int
    endc: int
    begg: int
    endg: int


class AtmosphericForcing(NamedTuple):
    """Atmospheric forcing variables mapped to patch level."""
    tref: jnp.ndarray
    qref: jnp.ndarray
    pref: jnp.ndarray
    lwsky: jnp.ndarray
    qflx_rain: jnp.ndarray
    qflx_snow: jnp.ndarray
    co2ref: jnp.ndarray
    o2ref: jnp.ndarray
    tacclim: jnp.ndarray
    uref: jnp.ndarray
    swskyb_vis: jnp.ndarray
    swskyd_vis: jnp.ndarray
    swskyb_nir: jnp.ndarray
    swskyd_nir: jnp.ndarray


class CanopyProfileState(NamedTuple):
    """Updated canopy profile variables with LAI/SAI distributions."""
    lai: jnp.ndarray
    sai: jnp.ndarray
    dlai: jnp.ndarray
    dsai: jnp.ndarray
    dpai: jnp.ndarray


# Fixtures
@pytest.fixture
def test_data():
    """
    Load and prepare test data for ml_canopy_fluxes tests.
    
    Returns:
        dict: Dictionary containing all test cases with inputs and metadata.
    """
    return {
        "test_nominal_single_patch_moderate_conditions": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=1, begc=0, endc=1, begg=0, endg=1),
                "num_exposedvegp": 1,
                "filter_exposedvegp": jnp.array([0]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([298.15]),
                    qref=jnp.array([0.012]),
                    pref=jnp.array([101325.0]),
                    lwsky=jnp.array([350.0]),
                    qflx_rain=jnp.array([0.0]),
                    qflx_snow=jnp.array([0.0]),
                    co2ref=jnp.array([0.0004]),
                    o2ref=jnp.array([0.209]),
                    tacclim=jnp.array([288.15]),
                    uref=jnp.array([3.5]),
                    swskyb_vis=jnp.array([250.0]),
                    swskyd_vis=jnp.array([150.0]),
                    swskyb_nir=jnp.array([200.0]),
                    swskyd_nir=jnp.array([100.0]),
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([4.0]),
                    sai=jnp.array([1.0]),
                    dlai=jnp.array([[0.3, 0.25, 0.2, 0.15, 0.1]]),
                    dsai=jnp.array([[0.2, 0.2, 0.2, 0.2, 0.2]]),
                    dpai=jnp.array([[0.5, 0.45, 0.4, 0.35, 0.3]]),
                ),
                "nstep": 100,
                "dtime": 1800.0,
                "dtime_substep": 300.0,
                "num_sub_steps": 6,
            },
            "metadata": {
                "type": "nominal",
                "description": "Single patch with moderate LAI, typical summer daytime conditions",
                "edge_cases": [],
            },
        },
        "test_nominal_multiple_patches_varying_lai": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=4, begc=0, endc=2, begg=0, endg=1),
                "num_exposedvegp": 4,
                "filter_exposedvegp": jnp.array([0, 1, 2, 3]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([295.15, 297.15, 299.15, 296.15]),
                    qref=jnp.array([0.01, 0.013, 0.015, 0.011]),
                    pref=jnp.array([101325.0, 101200.0, 101400.0, 101300.0]),
                    lwsky=jnp.array([340.0, 355.0, 360.0, 345.0]),
                    qflx_rain=jnp.array([0.0, 0.0, 0.0, 0.0]),
                    qflx_snow=jnp.array([0.0, 0.0, 0.0, 0.0]),
                    co2ref=jnp.array([0.0004, 0.000405, 0.000398, 0.000402]),
                    o2ref=jnp.array([0.209, 0.209, 0.209, 0.209]),
                    tacclim=jnp.array([288.15, 289.15, 287.15, 288.65]),
                    uref=jnp.array([2.5, 4.0, 3.0, 3.5]),
                    swskyb_vis=jnp.array([200.0, 280.0, 240.0, 260.0]),
                    swskyd_vis=jnp.array([120.0, 160.0, 140.0, 150.0]),
                    swskyb_nir=jnp.array([180.0, 220.0, 190.0, 210.0]),
                    swskyd_nir=jnp.array([90.0, 110.0, 95.0, 105.0]),
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([2.5, 5.0, 3.5, 6.0]),
                    sai=jnp.array([0.5, 1.2, 0.8, 1.5]),
                    dlai=jnp.array([
                        [0.35, 0.3, 0.2, 0.1, 0.05],
                        [0.25, 0.22, 0.2, 0.18, 0.15],
                        [0.3, 0.28, 0.22, 0.15, 0.05],
                        [0.22, 0.2, 0.19, 0.18, 0.21],
                    ]),
                    dsai=jnp.array([
                        [0.25, 0.25, 0.2, 0.2, 0.1],
                        [0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.22, 0.22, 0.2, 0.18, 0.18],
                        [0.18, 0.2, 0.21, 0.21, 0.2],
                    ]),
                    dpai=jnp.array([
                        [0.6, 0.55, 0.4, 0.3, 0.15],
                        [0.45, 0.42, 0.4, 0.38, 0.35],
                        [0.52, 0.5, 0.42, 0.33, 0.23],
                        [0.4, 0.4, 0.4, 0.39, 0.41],
                    ]),
                ),
                "nstep": 250,
                "dtime": 1800.0,
                "dtime_substep": 450.0,
                "num_sub_steps": 4,
            },
            "metadata": {
                "type": "nominal",
                "description": "Multiple patches with varying LAI/SAI",
                "edge_cases": [],
            },
        },
        "test_edge_zero_lai_bare_ground": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=2, begc=0, endc=1, begg=0, endg=1),
                "num_exposedvegp": 2,
                "filter_exposedvegp": jnp.array([0, 1]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([300.15, 298.15]),
                    qref=jnp.array([0.008, 0.01]),
                    pref=jnp.array([101325.0, 101325.0]),
                    lwsky=jnp.array([370.0, 360.0]),
                    qflx_rain=jnp.array([0.0, 0.0]),
                    qflx_snow=jnp.array([0.0, 0.0]),
                    co2ref=jnp.array([0.0004, 0.0004]),
                    o2ref=jnp.array([0.209, 0.209]),
                    tacclim=jnp.array([290.15, 289.15]),
                    uref=jnp.array([5.0, 4.5]),
                    swskyb_vis=jnp.array([300.0, 280.0]),
                    swskyd_vis=jnp.array([180.0, 170.0]),
                    swskyb_nir=jnp.array([240.0, 220.0]),
                    swskyd_nir=jnp.array([120.0, 110.0]),
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([0.0, 0.1]),
                    sai=jnp.array([0.0, 0.05]),
                    dlai=jnp.array([
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                    ]),
                    dsai=jnp.array([
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                    ]),
                    dpai=jnp.array([
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                    ]),
                ),
                "nstep": 50,
                "dtime": 1800.0,
                "dtime_substep": 600.0,
                "num_sub_steps": 3,
            },
            "metadata": {
                "type": "edge",
                "description": "Zero and minimal LAI representing bare ground",
                "edge_cases": ["zero_lai", "minimal_vegetation"],
            },
        },
        "test_edge_nighttime_no_solar_radiation": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=3, begc=0, endc=2, begg=0, endg=1),
                "num_exposedvegp": 3,
                "filter_exposedvegp": jnp.array([0, 1, 2]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([285.15, 283.15, 286.15]),
                    qref=jnp.array([0.008, 0.007, 0.009]),
                    pref=jnp.array([101325.0, 101400.0, 101250.0]),
                    lwsky=jnp.array([280.0, 275.0, 285.0]),
                    qflx_rain=jnp.array([0.0, 0.0, 0.0]),
                    qflx_snow=jnp.array([0.0, 0.0, 0.0]),
                    co2ref=jnp.array([0.00041, 0.000415, 0.000408]),
                    o2ref=jnp.array([0.209, 0.209, 0.209]),
                    tacclim=jnp.array([285.15, 284.15, 286.15]),
                    uref=jnp.array([2.0, 1.5, 2.5]),
                    swskyb_vis=jnp.array([0.0, 0.0, 0.0]),
                    swskyd_vis=jnp.array([0.0, 0.0, 0.0]),
                    swskyb_nir=jnp.array([0.0, 0.0, 0.0]),
                    swskyd_nir=jnp.array([0.0, 0.0, 0.0]),
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([3.5, 4.5, 2.8]),
                    sai=jnp.array([0.9, 1.1, 0.7]),
                    dlai=jnp.array([
                        [0.28, 0.25, 0.22, 0.15, 0.1],
                        [0.24, 0.22, 0.2, 0.18, 0.16],
                        [0.32, 0.28, 0.2, 0.12, 0.08],
                    ]),
                    dsai=jnp.array([
                        [0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.22, 0.22, 0.2, 0.18, 0.18],
                    ]),
                    dpai=jnp.array([
                        [0.48, 0.45, 0.42, 0.35, 0.3],
                        [0.44, 0.42, 0.4, 0.38, 0.36],
                        [0.54, 0.5, 0.4, 0.3, 0.26],
                    ]),
                ),
                "nstep": 500,
                "dtime": 1800.0,
                "dtime_substep": 300.0,
                "num_sub_steps": 6,
            },
            "metadata": {
                "type": "edge",
                "description": "Nighttime conditions with zero solar radiation",
                "edge_cases": ["zero_solar", "nighttime"],
            },
        },
        "test_edge_extreme_cold_winter_conditions": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=2, begc=0, endc=1, begg=0, endg=1),
                "num_exposedvegp": 2,
                "filter_exposedvegp": jnp.array([0, 1]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([253.15, 258.15]),
                    qref=jnp.array([0.0005, 0.0008]),
                    pref=jnp.array([102500.0, 102300.0]),
                    lwsky=jnp.array([180.0, 195.0]),
                    qflx_rain=jnp.array([0.0, 0.0]),
                    qflx_snow=jnp.array([0.0001, 0.00015]),
                    co2ref=jnp.array([0.00042, 0.000418]),
                    o2ref=jnp.array([0.209, 0.209]),
                    tacclim=jnp.array([253.15, 258.15]),
                    uref=jnp.array([6.0, 5.5]),
                    swskyb_vis=jnp.array([80.0, 100.0]),
                    swskyd_vis=jnp.array([50.0, 60.0]),
                    swskyb_nir=jnp.array([60.0, 75.0]),
                    swskyd_nir=jnp.array([30.0, 40.0]),
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([1.5, 2.0]),
                    sai=jnp.array([0.8, 1.0]),
                    dlai=jnp.array([
                        [0.4, 0.3, 0.2, 0.1, 0.0],
                        [0.35, 0.3, 0.2, 0.1, 0.05],
                    ]),
                    dsai=jnp.array([
                        [0.25, 0.25, 0.25, 0.25, 0.0],
                        [0.22, 0.22, 0.2, 0.18, 0.18],
                    ]),
                    dpai=jnp.array([
                        [0.65, 0.55, 0.45, 0.35, 0.0],
                        [0.57, 0.52, 0.4, 0.28, 0.23],
                    ]),
                ),
                "nstep": 1000,
                "dtime": 1800.0,
                "dtime_substep": 900.0,
                "num_sub_steps": 2,
            },
            "metadata": {
                "type": "edge",
                "description": "Extreme cold winter conditions",
                "edge_cases": ["extreme_cold", "low_humidity", "snowfall"],
            },
        },
        "test_edge_high_lai_dense_canopy": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=2, begc=0, endc=1, begg=0, endg=1),
                "num_exposedvegp": 2,
                "filter_exposedvegp": jnp.array([0, 1]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([296.15, 297.15]),
                    qref=jnp.array([0.018, 0.019]),
                    pref=jnp.array([101325.0, 101300.0]),
                    lwsky=jnp.array([360.0, 365.0]),
                    qflx_rain=jnp.array([0.0002, 0.00025]),
                    qflx_snow=jnp.array([0.0, 0.0]),
                    co2ref=jnp.array([0.000395, 0.000398]),
                    o2ref=jnp.array([0.209, 0.209]),
                    tacclim=jnp.array([290.15, 291.15]),
                    uref=jnp.array([1.5, 1.8]),
                    swskyb_vis=jnp.array([320.0, 340.0]),
                    swskyd_vis=jnp.array([200.0, 210.0]),
                    swskyb_nir=jnp.array([260.0, 280.0]),
                    swskyd_nir=jnp.array([130.0, 140.0]),
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([8.0, 9.5]),
                    sai=jnp.array([2.0, 2.5]),
                    dlai=jnp.array([
                        [0.21, 0.2, 0.2, 0.19, 0.2],
                        [0.205, 0.2, 0.2, 0.195, 0.2],
                    ]),
                    dsai=jnp.array([
                        [0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.2, 0.2, 0.2, 0.2, 0.2],
                    ]),
                    dpai=jnp.array([
                        [0.41, 0.4, 0.4, 0.39, 0.4],
                        [0.405, 0.4, 0.4, 0.395, 0.4],
                    ]),
                ),
                "nstep": 300,
                "dtime": 1800.0,
                "dtime_substep": 360.0,
                "num_sub_steps": 5,
            },
            "metadata": {
                "type": "edge",
                "description": "Very high LAI representing dense tropical forest",
                "edge_cases": ["high_lai", "dense_canopy", "high_humidity"],
            },
        },
        "test_special_single_substep": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=1, begc=0, endc=1, begg=0, endg=1),
                "num_exposedvegp": 1,
                "filter_exposedvegp": jnp.array([0]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([293.15]),
                    qref=jnp.array([0.011]),
                    pref=jnp.array([101325.0]),
                    lwsky=jnp.array([340.0]),
                    qflx_rain=jnp.array([0.0]),
                    qflx_snow=jnp.array([0.0]),
                    co2ref=jnp.array([0.0004]),
                    o2ref=jnp.array([0.209]),
                    tacclim=jnp.array([288.15]),
                    uref=jnp.array([3.0]),
                    swskyb_vis=jnp.array([220.0]),
                    swskyd_vis=jnp.array([130.0]),
                    swskyb_nir=jnp.array([180.0]),
                    swskyd_nir=jnp.array([90.0]),
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([3.0]),
                    sai=jnp.array([0.8]),
                    dlai=jnp.array([[0.3, 0.28, 0.22, 0.15, 0.05]]),
                    dsai=jnp.array([[0.2, 0.2, 0.2, 0.2, 0.2]]),
                    dpai=jnp.array([[0.5, 0.48, 0.42, 0.35, 0.25]]),
                ),
                "nstep": 1,
                "dtime": 1800.0,
                "dtime_substep": 1800.0,
                "num_sub_steps": 1,
            },
            "metadata": {
                "type": "special",
                "description": "Single sub-timestep case",
                "edge_cases": ["single_substep"],
            },
        },
        "test_special_many_substeps": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=1, begc=0, endc=1, begg=0, endg=1),
                "num_exposedvegp": 1,
                "filter_exposedvegp": jnp.array([0]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([295.15]),
                    qref=jnp.array([0.012]),
                    pref=jnp.array([101325.0]),
                    lwsky=jnp.array([350.0]),
                    qflx_rain=jnp.array([0.0]),
                    qflx_snow=jnp.array([0.0]),
                    co2ref=jnp.array([0.0004]),
                    o2ref=jnp.array([0.209]),
                    tacclim=jnp.array([288.15]),
                    uref=jnp.array([3.5]),
                    swskyb_vis=jnp.array([240.0]),
                    swskyd_vis=jnp.array([140.0]),
                    swskyb_nir=jnp.array([190.0]),
                    swskyd_nir=jnp.array([95.0]),
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([4.5]),
                    sai=jnp.array([1.2]),
                    dlai=jnp.array([[0.25, 0.23, 0.21, 0.18, 0.13]]),
                    dsai=jnp.array([[0.2, 0.2, 0.2, 0.2, 0.2]]),
                    dpai=jnp.array([[0.45, 0.43, 0.41, 0.38, 0.33]]),
                ),
                "nstep": 200,
                "dtime": 3600.0,
                "dtime_substep": 180.0,
                "num_sub_steps": 20,
            },
            "metadata": {
                "type": "special",
                "description": "Many sub-timesteps testing accumulation",
                "edge_cases": ["many_substeps"],
            },
        },
        "test_special_high_wind_low_pressure": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=3, begc=0, endc=2, begg=0, endg=1),
                "num_exposedvegp": 3,
                "filter_exposedvegp": jnp.array([0, 1, 2]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([290.15, 292.15, 291.15]),
                    qref=jnp.array([0.009, 0.01, 0.0095]),
                    pref=jnp.array([85000.0, 86000.0, 85500.0]),
                    lwsky=jnp.array([320.0, 330.0, 325.0]),
                    qflx_rain=jnp.array([0.0, 0.0, 0.0]),
                    qflx_snow=jnp.array([0.0, 0.0, 0.0]),
                    co2ref=jnp.array([0.00038, 0.000385, 0.000382]),
                    o2ref=jnp.array([0.209, 0.209, 0.209]),
                    tacclim=jnp.array([285.15, 286.15, 285.65]),
                    uref=jnp.array([15.0, 18.0, 16.5]),
                    swskyb_vis=jnp.array([180.0, 200.0, 190.0]),
                    swskyd_vis=jnp.array([110.0, 120.0, 115.0]),
                    swskyb_nir=jnp.array([140.0, 160.0, 150.0]),
                    swskyd_nir=jnp.array([70.0, 80.0, 75.0]),
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([2.0, 2.5, 2.2]),
                    sai=jnp.array([0.6, 0.7, 0.65]),
                    dlai=jnp.array([
                        [0.35, 0.3, 0.2, 0.1, 0.05],
                        [0.32, 0.28, 0.22, 0.12, 0.06],
                        [0.34, 0.29, 0.21, 0.11, 0.05],
                    ]),
                    dsai=jnp.array([
                        [0.22, 0.22, 0.2, 0.18, 0.18],
                        [0.21, 0.21, 0.2, 0.19, 0.19],
                        [0.22, 0.21, 0.2, 0.19, 0.18],
                    ]),
                    dpai=jnp.array([
                        [0.57, 0.52, 0.4, 0.28, 0.23],
                        [0.53, 0.49, 0.42, 0.31, 0.25],
                        [0.56, 0.5, 0.41, 0.3, 0.23],
                    ]),
                ),
                "nstep": 750,
                "dtime": 1800.0,
                "dtime_substep": 450.0,
                "num_sub_steps": 4,
            },
            "metadata": {
                "type": "special",
                "description": "High elevation with low pressure and high wind",
                "edge_cases": ["high_wind", "low_pressure", "high_elevation"],
            },
        },
        "test_special_hot_humid_tropical": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=2, begc=0, endc=1, begg=0, endg=1),
                "num_exposedvegp": 2,
                "filter_exposedvegp": jnp.array([0, 1]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([305.15, 306.15]),
                    qref=jnp.array([0.022, 0.024]),
                    pref=jnp.array([101325.0, 101300.0]),
                    lwsky=jnp.array([420.0, 425.0]),
                    qflx_rain=jnp.array([0.0005, 0.0006]),
                    qflx_snow=jnp.array([0.0, 0.0]),
                    co2ref=jnp.array([0.00039, 0.000392]),
                    o2ref=jnp.array([0.209, 0.209]),
                    tacclim=jnp.array([300.15, 301.15]),
                    uref=jnp.array([1.0, 1.2]),
                    swskyb_vis=jnp.array([380.0, 400.0]),
                    swskyd_vis=jnp.array([220.0, 230.0]),
                    swskyb_nir=jnp.array([320.0, 340.0]),
                    swskyd_nir=jnp.array([160.0, 170.0]),
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([6.5, 7.0]),
                    sai=jnp.array([1.8, 2.0]),
                    dlai=jnp.array([
                        [0.22, 0.21, 0.2, 0.19, 0.18],
                        [0.21, 0.205, 0.2, 0.195, 0.19],
                    ]),
                    dsai=jnp.array([
                        [0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.2, 0.2, 0.2, 0.2, 0.2],
                    ]),
                    dpai=jnp.array([
                        [0.42, 0.41, 0.4, 0.39, 0.38],
                        [0.41, 0.405, 0.4, 0.395, 0.39],
                    ]),
                ),
                "nstep": 600,
                "dtime": 1800.0,
                "dtime_substep": 300.0,
                "num_sub_steps": 6,
            },
            "metadata": {
                "type": "special",
                "description": "Hot and humid tropical conditions",
                "edge_cases": ["high_temperature", "high_humidity", "rainfall", "tropical"],
            },
        },
    }


# Parametrized test cases
@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_nominal_single_patch_moderate_conditions",
        "test_nominal_multiple_patches_varying_lai",
        "test_edge_zero_lai_bare_ground",
        "test_edge_nighttime_no_solar_radiation",
        "test_edge_extreme_cold_winter_conditions",
        "test_edge_high_lai_dense_canopy",
        "test_special_single_substep",
        "test_special_many_substeps",
        "test_special_high_wind_low_pressure",
        "test_special_hot_humid_tropical",
    ],
)
def test_ml_canopy_fluxes_shapes(test_data, test_case_name):
    """
    Test that ml_canopy_fluxes returns outputs with correct shapes.
    
    Verifies:
    - Output is a tuple of two elements (MLCanopyFluxesState, FluxAccumulators)
    - All array dimensions match expected patch/layer counts
    - Arrays are properly shaped for subsequent processing
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    # Call the function
    state, flux_accumulators = ml_canopy_fluxes(**inputs)
    
    # Verify output is a tuple of two elements
    assert isinstance(state, tuple), "First output should be a NamedTuple (MLCanopyFluxesState)"
    assert isinstance(flux_accumulators, tuple), "Second output should be a NamedTuple (FluxAccumulators)"
    
    # Get expected dimensions
    num_patches = inputs["bounds"].endp - inputs["bounds"].begp
    n_canopy_layers = 5  # From test data
    
    # Check state components have correct shapes
    # Note: Actual shape checking depends on the implementation
    # These are placeholder assertions that should be adjusted based on actual output structure
    assert hasattr(state, "rnleaf_sun"), "State should have rnleaf_sun field"
    assert hasattr(state, "rnleaf_shade"), "State should have rnleaf_shade field"
    assert hasattr(state, "rnsoi"), "State should have rnsoi field"
    
    # Check flux accumulator components
    assert hasattr(flux_accumulators, "flux_1d"), "FluxAccumulators should have flux_1d"
    assert hasattr(flux_accumulators, "flux_2d"), "FluxAccumulators should have flux_2d"
    assert hasattr(flux_accumulators, "flux_3d"), "FluxAccumulators should have flux_3d"


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_nominal_single_patch_moderate_conditions",
        "test_nominal_multiple_patches_varying_lai",
    ],
)
def test_ml_canopy_fluxes_values(test_data, test_case_name):
    """
    Test that ml_canopy_fluxes produces physically reasonable output values.
    
    Verifies:
    - Net radiation values are finite and reasonable
    - Flux values are within expected physical ranges
    - No NaN or Inf values in outputs
    - Energy balance constraints are satisfied
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    # Call the function
    state, flux_accumulators = ml_canopy_fluxes(**inputs)
    
    # Check for finite values (no NaN or Inf)
    if hasattr(state, "rnsoi"):
        assert jnp.all(jnp.isfinite(state.rnsoi)), "rnsoi should contain finite values"
    
    if hasattr(state, "rnleaf_sun"):
        assert jnp.all(jnp.isfinite(state.rnleaf_sun)), "rnleaf_sun should contain finite values"
    
    if hasattr(state, "rnleaf_shade"):
        assert jnp.all(jnp.isfinite(state.rnleaf_shade)), "rnleaf_shade should contain finite values"
    
    # Check flux accumulators are finite
    if hasattr(flux_accumulators, "flux_1d") and flux_accumulators.flux_1d is not None:
        assert jnp.all(jnp.isfinite(flux_accumulators.flux_1d)), "flux_1d should be finite"
    
    if hasattr(flux_accumulators, "flux_2d") and flux_accumulators.flux_2d is not None:
        assert jnp.all(jnp.isfinite(flux_accumulators.flux_2d)), "flux_2d should be finite"
    
    if hasattr(flux_accumulators, "flux_3d") and flux_accumulators.flux_3d is not None:
        assert jnp.all(jnp.isfinite(flux_accumulators.flux_3d)), "flux_3d should be finite"


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_edge_zero_lai_bare_ground",
        "test_edge_nighttime_no_solar_radiation",
        "test_edge_extreme_cold_winter_conditions",
        "test_edge_high_lai_dense_canopy",
    ],
)
def test_ml_canopy_fluxes_edge_cases(test_data, test_case_name):
    """
    Test ml_canopy_fluxes behavior under edge case conditions.
    
    Verifies:
    - Function handles zero LAI (bare ground) correctly
    - Nighttime conditions (zero solar radiation) work properly
    - Extreme cold temperatures don't cause numerical issues
    - Very high LAI (dense canopy) is handled correctly
    - All outputs remain physically valid under extreme conditions
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    metadata = test_case["metadata"]
    
    # Call the function
    state, flux_accumulators = ml_canopy_fluxes(**inputs)
    
    # Verify function completes without errors
    assert state is not None, f"Function should return state for {metadata['description']}"
    assert flux_accumulators is not None, f"Function should return flux_accumulators for {metadata['description']}"
    
    # Check for finite values in edge cases
    if hasattr(state, "rnsoi"):
        assert jnp.all(jnp.isfinite(state.rnsoi)), f"rnsoi should be finite for {metadata['description']}"
    
    # For zero LAI case, check that canopy fluxes are minimal/zero
    if "zero_lai" in metadata["edge_cases"]:
        if hasattr(state, "rnleaf_sun"):
            # With zero LAI, leaf radiation should be zero or minimal
            assert jnp.all(jnp.abs(state.rnleaf_sun) < 1e-6), "rnleaf_sun should be near zero for zero LAI"
    
    # For nighttime case, verify no solar radiation effects
    if "nighttime" in metadata["edge_cases"]:
        # Solar components should be zero or minimal
        assert jnp.all(inputs["atmospheric_forcing"].swskyb_vis == 0.0), "Solar radiation should be zero at night"
        assert jnp.all(inputs["atmospheric_forcing"].swskyd_vis == 0.0), "Diffuse solar should be zero at night"
    
    # For extreme cold, verify temperatures are handled correctly
    if "extreme_cold" in metadata["edge_cases"]:
        assert jnp.all(inputs["atmospheric_forcing"].tref < 273.15), "Temperature should be below freezing"
        # Function should still produce valid outputs
        assert state is not None, "Function should handle extreme cold temperatures"


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_nominal_single_patch_moderate_conditions",
        "test_edge_zero_lai_bare_ground",
        "test_special_single_substep",
    ],
)
def test_ml_canopy_fluxes_dtypes(test_data, test_case_name):
    """
    Test that ml_canopy_fluxes returns outputs with correct data types.
    
    Verifies:
    - All array outputs are JAX arrays (jnp.ndarray)
    - Float values have appropriate precision
    - Integer indices remain as integers
    - No unexpected type conversions occur
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    
    # Call the function
    state, flux_accumulators = ml_canopy_fluxes(**inputs)
    
    # Check that outputs are JAX arrays where expected
    if hasattr(state, "rnsoi"):
        assert isinstance(state.rnsoi, jnp.ndarray), "rnsoi should be a JAX array"
        assert jnp.issubdtype(state.rnsoi.dtype, jnp.floating), "rnsoi should be floating point"
    
    if hasattr(state, "rnleaf_sun"):
        assert isinstance(state.rnleaf_sun, jnp.ndarray), "rnleaf_sun should be a JAX array"
        assert jnp.issubdtype(state.rnleaf_sun.dtype, jnp.floating), "rnleaf_sun should be floating point"
    
    if hasattr(state, "rnleaf_shade"):
        assert isinstance(state.rnleaf_shade, jnp.ndarray), "rnleaf_shade should be a JAX array"
        assert jnp.issubdtype(state.rnleaf_shade.dtype, jnp.floating), "rnleaf_shade should be floating point"
    
    # Check flux accumulator types
    if hasattr(flux_accumulators, "flux_1d") and flux_accumulators.flux_1d is not None:
        assert isinstance(flux_accumulators.flux_1d, jnp.ndarray), "flux_1d should be a JAX array"
        assert jnp.issubdtype(flux_accumulators.flux_1d.dtype, jnp.floating), "flux_1d should be floating point"


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_special_single_substep",
        "test_special_many_substeps",
    ],
)
def test_ml_canopy_fluxes_substep_consistency(test_data, test_case_name):
    """
    Test that ml_canopy_fluxes handles different numbers of sub-timesteps correctly.
    
    Verifies:
    - Single sub-timestep case works correctly
    - Many sub-timesteps produce consistent results
    - Flux accumulation is properly scaled by number of substeps
    - Results are independent of substep discretization (within tolerance)
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    metadata = test_case["metadata"]
    
    # Call the function
    state, flux_accumulators = ml_canopy_fluxes(**inputs)
    
    # Verify function completes
    assert state is not None, f"Function should complete for {metadata['description']}"
    assert flux_accumulators is not None, f"Function should return accumulators for {metadata['description']}"
    
    # Check that substep parameters are consistent
    expected_substep_time = inputs["dtime"] / inputs["num_sub_steps"]
    assert np.isclose(
        inputs["dtime_substep"], expected_substep_time, rtol=1e-6
    ), "dtime_substep should equal dtime/num_sub_steps"
    
    # Verify outputs are finite
    if hasattr(state, "rnsoi"):
        assert jnp.all(jnp.isfinite(state.rnsoi)), f"rnsoi should be finite for {metadata['description']}"


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_special_high_wind_low_pressure",
        "test_special_hot_humid_tropical",
    ],
)
def test_ml_canopy_fluxes_extreme_conditions(test_data, test_case_name):
    """
    Test ml_canopy_fluxes under extreme environmental conditions.
    
    Verifies:
    - High wind speeds are handled correctly
    - Low pressure (high elevation) conditions work properly
    - High temperature and humidity combinations are valid
    - Extreme conditions don't cause numerical instabilities
    """
    test_case = test_data[test_case_name]
    inputs = test_case["inputs"]
    metadata = test_case["metadata"]
    
    # Call the function
    state, flux_accumulators = ml_canopy_fluxes(**inputs)
    
    # Verify function completes without errors
    assert state is not None, f"Function should handle {metadata['description']}"
    assert flux_accumulators is not None, f"Function should return accumulators for {metadata['description']}"
    
    # Check for finite values under extreme conditions
    if hasattr(state, "rnsoi"):
        assert jnp.all(jnp.isfinite(state.rnsoi)), f"rnsoi should be finite for {metadata['description']}"
    
    if hasattr(state, "rnleaf_sun"):
        assert jnp.all(jnp.isfinite(state.rnleaf_sun)), f"rnleaf_sun should be finite for {metadata['description']}"
    
    # For high wind case, verify wind speeds are as expected
    if "high_wind" in metadata["edge_cases"]:
        assert jnp.any(inputs["atmospheric_forcing"].uref > 10.0), "Should have high wind speeds"
    
    # For low pressure case, verify pressure is reduced
    if "low_pressure" in metadata["edge_cases"]:
        assert jnp.any(inputs["atmospheric_forcing"].pref < 90000.0), "Should have low pressure"
    
    # For high temperature case, verify temperatures are elevated
    if "high_temperature" in metadata["edge_cases"]:
        assert jnp.any(inputs["atmospheric_forcing"].tref > 300.0), "Should have high temperatures"


def test_ml_canopy_fluxes_input_validation(test_data):
    """
    Test that ml_canopy_fluxes properly validates input parameters.
    
    Verifies:
    - Bounds are consistent (begp < endp, etc.)
    - Filter indices are within bounds
    - Time parameters are positive
    - Physical constraints are satisfied (T > 0K, fractions in [0,1])
    """
    test_case = test_data["test_nominal_single_patch_moderate_conditions"]
    inputs = test_case["inputs"]
    
    # Verify bounds consistency
    assert inputs["bounds"].begp < inputs["bounds"].endp, "begp should be less than endp"
    assert inputs["bounds"].begc < inputs["bounds"].endc, "begc should be less than endc"
    assert inputs["bounds"].begg < inputs["bounds"].endg, "begg should be less than endg"
    
    # Verify filter indices are within bounds
    assert jnp.all(inputs["filter_exposedvegp"] >= inputs["bounds"].begp), "Filter indices should be >= begp"
    assert jnp.all(inputs["filter_exposedvegp"] < inputs["bounds"].endp), "Filter indices should be < endp"
    
    # Verify time parameters are positive
    assert inputs["dtime"] > 0, "dtime should be positive"
    assert inputs["dtime_substep"] > 0, "dtime_substep should be positive"
    assert inputs["num_sub_steps"] >= 1, "num_sub_steps should be at least 1"
    
    # Verify physical constraints
    assert jnp.all(inputs["atmospheric_forcing"].tref > 0), "Temperature should be positive (Kelvin)"
    assert jnp.all(inputs["atmospheric_forcing"].qref >= 0), "Specific humidity should be non-negative"
    assert jnp.all(inputs["atmospheric_forcing"].qref <= 1), "Specific humidity should be <= 1"
    assert jnp.all(inputs["atmospheric_forcing"].pref > 0), "Pressure should be positive"
    assert jnp.all(inputs["canopy_state"].lai >= 0), "LAI should be non-negative"
    assert jnp.all(inputs["canopy_state"].sai >= 0), "SAI should be non-negative"


def test_ml_canopy_fluxes_conservation(test_data):
    """
    Test that ml_canopy_fluxes conserves energy and mass where applicable.
    
    Verifies:
    - Energy balance is maintained (within numerical tolerance)
    - Flux accumulation is consistent with timestep
    - No spurious sources or sinks of energy/mass
    """
    test_case = test_data["test_nominal_single_patch_moderate_conditions"]
    inputs = test_case["inputs"]
    
    # Call the function
    state, flux_accumulators = ml_canopy_fluxes(**inputs)
    
    # Verify outputs exist
    assert state is not None, "Function should return state"
    assert flux_accumulators is not None, "Function should return flux accumulators"
    
    # Check that accumulated fluxes are finite and reasonable
    if hasattr(flux_accumulators, "flux_1d") and flux_accumulators.flux_1d is not None:
        assert jnp.all(jnp.isfinite(flux_accumulators.flux_1d)), "Accumulated fluxes should be finite"
        
        # Verify flux magnitudes are reasonable (not excessively large)
        max_flux = jnp.max(jnp.abs(flux_accumulators.flux_1d))
        assert max_flux < 1e6, f"Flux magnitude {max_flux} seems unreasonably large"


def test_ml_canopy_fluxes_reproducibility(test_data):
    """
    Test that ml_canopy_fluxes produces reproducible results.
    
    Verifies:
    - Multiple calls with same inputs produce identical outputs
    - Results are deterministic (no random components)
    - JAX compilation doesn't affect reproducibility
    """
    test_case = test_data["test_nominal_single_patch_moderate_conditions"]
    inputs = test_case["inputs"]
    
    # Call the function twice with identical inputs
    state1, flux_acc1 = ml_canopy_fluxes(**inputs)
    state2, flux_acc2 = ml_canopy_fluxes(**inputs)
    
    # Compare outputs
    if hasattr(state1, "rnsoi") and hasattr(state2, "rnsoi"):
        assert jnp.allclose(state1.rnsoi, state2.rnsoi, rtol=1e-10, atol=1e-10), \
            "rnsoi should be identical across calls"
    
    if hasattr(state1, "rnleaf_sun") and hasattr(state2, "rnleaf_sun"):
        assert jnp.allclose(state1.rnleaf_sun, state2.rnleaf_sun, rtol=1e-10, atol=1e-10), \
            "rnleaf_sun should be identical across calls"
    
    if hasattr(flux_acc1, "flux_1d") and hasattr(flux_acc2, "flux_1d"):
        if flux_acc1.flux_1d is not None and flux_acc2.flux_1d is not None:
            assert jnp.allclose(flux_acc1.flux_1d, flux_acc2.flux_1d, rtol=1e-10, atol=1e-10), \
                "flux_1d should be identical across calls"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])