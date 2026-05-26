"""
Comprehensive pytest suite for MLclm_varctl module.

This module tests the configuration management functionality for the multilayer
canopy model, including:
- Configuration creation (default, CLM5, CLM4.5)
- Configuration validation
- Configuration query functions
- Canopy height-based parameter selection
- Configuration summary generation

Tests cover nominal cases, edge cases, boundary conditions, and constraint
validation for all configuration parameters.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multilayer_canopy.MLclm_varctl import (
    MLCanopyConfig,
    config_summary,
    create_clm45_config,
    create_clm5_config,
    create_default_config,
    get_canopy_dz,
    is_clm5_physics,
    uses_auto_layers,
    uses_rsl_turbulence,
    validate_config,
)


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load test data for MLclm_varctl tests.
    
    Returns:
        Dictionary containing test cases with inputs and expected outputs
    """
    return {
        "default_config": {
            "clm_phys": "CLM4_5",
            "gs_type": 2,
            "gspot_type": 1,
            "colim_type": 1,
            "acclim_type": 1,
            "kn_val": -999.0,
            "turb_type": 1,
            "gb_type": 3,
            "light_type": 2,
            "longwave_type": 1,
            "fpi_type": 2,
            "root_type": 2,
            "fracdir": -999.0,
            "mlcan_to_clm": 0,
            "ml_vert_init": -9999,
            "dz_tall": 0.5,
            "dz_short": 0.1,
            "dz_param": 2.0,
            "dpai_min": 0.01,
            "nlayer_above": 0,
            "nlayer_within": 0,
            "rslfile": "../rsl_lookup_tables/psihat.nc",
            "dtime_substep": 300.0,
        },
        "clm5_config": {
            "clm_phys": "CLM5_0",
            "fpi_type": 2,
            "root_type": 2,
        },
        "clm45_config": {
            "clm_phys": "CLM4_5",
            "fpi_type": 1,
            "root_type": 1,
        },
    }


# ============================================================================
# Configuration Creation Tests
# ============================================================================


def test_create_default_config(test_data: Dict[str, Any]) -> None:
    """
    Test creation of default configuration with all standard values.
    
    Verifies that create_default_config() returns an MLCanopyConfig with
    all expected default values matching the Fortran source.
    """
    config = create_default_config()
    expected = test_data["default_config"]
    
    assert isinstance(config, MLCanopyConfig)
    assert config.clm_phys == expected["clm_phys"]
    assert config.gs_type == expected["gs_type"]
    assert config.gspot_type == expected["gspot_type"]
    assert config.colim_type == expected["colim_type"]
    assert config.acclim_type == expected["acclim_type"]
    assert config.kn_val == expected["kn_val"]
    assert config.turb_type == expected["turb_type"]
    assert config.gb_type == expected["gb_type"]
    assert config.light_type == expected["light_type"]
    assert config.longwave_type == expected["longwave_type"]
    assert config.fpi_type == expected["fpi_type"]
    assert config.root_type == expected["root_type"]
    assert config.fracdir == expected["fracdir"]
    assert config.mlcan_to_clm == expected["mlcan_to_clm"]
    assert config.ml_vert_init == expected["ml_vert_init"]
    assert config.dz_tall == expected["dz_tall"]
    assert config.dz_short == expected["dz_short"]
    assert config.dz_param == expected["dz_param"]
    assert config.dpai_min == expected["dpai_min"]
    assert config.nlayer_above == expected["nlayer_above"]
    assert config.nlayer_within == expected["nlayer_within"]
    assert config.rslfile == expected["rslfile"]
    assert config.dtime_substep == expected["dtime_substep"]


def test_create_clm5_config(test_data: Dict[str, Any]) -> None:
    """
    Test CLM5 physics configuration with appropriate fpi and root types.
    
    Verifies that create_clm5_config() returns a configuration with
    CLM5_0 physics and corresponding CLM5 parameter choices.
    """
    config = create_clm5_config()
    expected = test_data["clm5_config"]
    
    assert isinstance(config, MLCanopyConfig)
    assert config.clm_phys == expected["clm_phys"]
    assert config.fpi_type == expected["fpi_type"]
    assert config.root_type == expected["root_type"]


def test_create_clm45_config(test_data: Dict[str, Any]) -> None:
    """
    Test CLM4.5 physics configuration with legacy fpi and root types.
    
    Verifies that create_clm45_config() returns a configuration with
    CLM4_5 physics and corresponding CLM4.5 parameter choices.
    """
    config = create_clm45_config()
    expected = test_data["clm45_config"]
    
    assert isinstance(config, MLCanopyConfig)
    assert config.clm_phys == expected["clm_phys"]
    assert config.fpi_type == expected["fpi_type"]
    assert config.root_type == expected["root_type"]


# ============================================================================
# Configuration Validation Tests
# ============================================================================


def test_validate_config_default() -> None:
    """
    Test validation of default configuration.
    
    Verifies that the default configuration passes all validation checks.
    """
    config = create_default_config()
    assert validate_config(config) is True


def test_validate_config_all_valid_options() -> None:
    """
    Test validation with all valid but non-default parameter choices.
    
    Creates a configuration with valid alternative values for all parameters
    and verifies it passes validation.
    """
    config = MLCanopyConfig(
        clm_phys="CLM5_0",
        gs_type=0,
        gspot_type=0,
        colim_type=0,
        acclim_type=0,
        kn_val=0.5,
        turb_type=-1,
        gb_type=0,
        light_type=1,
        longwave_type=1,
        fpi_type=1,
        root_type=1,
        fracdir=0.7,
        mlcan_to_clm=1,
        ml_vert_init=1,
        dz_tall=1.0,
        dz_short=0.05,
        dz_param=5.0,
        dpai_min=0.001,
        nlayer_above=5,
        nlayer_within=20,
        rslfile="/custom/path/psihat.nc",
        dtime_substep=60.0,
    )
    assert validate_config(config) is True


def test_validate_config_invalid_clm_phys() -> None:
    """
    Test validation failure with invalid clm_phys value.
    
    Verifies that validation raises ValueError when clm_phys is not
    one of the allowed values ('CLM4_5' or 'CLM5_0').
    """
    config = MLCanopyConfig(
        clm_phys="CLM3_0",  # Invalid value
        gs_type=2,
        gspot_type=1,
        colim_type=1,
        acclim_type=1,
        kn_val=-999.0,
        turb_type=1,
        gb_type=3,
        light_type=2,
        longwave_type=1,
        fpi_type=2,
        root_type=2,
        fracdir=-999.0,
        mlcan_to_clm=0,
        ml_vert_init=-9999,
        dz_tall=0.5,
        dz_short=0.1,
        dz_param=2.0,
        dpai_min=0.01,
        nlayer_above=0,
        nlayer_within=0,
        rslfile="../rsl_lookup_tables/psihat.nc",
        dtime_substep=300.0,
    )
    with pytest.raises(ValueError, match="clm_phys"):
        validate_config(config)


def test_validate_config_invalid_gs_type() -> None:
    """
    Test validation failure with invalid gs_type value.
    
    Verifies that validation raises ValueError when gs_type is not
    one of the allowed values (0, 1, or 2).
    """
    config = MLCanopyConfig(
        clm_phys="CLM4_5",
        gs_type=3,  # Invalid value
        gspot_type=1,
        colim_type=1,
        acclim_type=1,
        kn_val=-999.0,
        turb_type=1,
        gb_type=3,
        light_type=2,
        longwave_type=1,
        fpi_type=2,
        root_type=2,
        fracdir=-999.0,
        mlcan_to_clm=0,
        ml_vert_init=-9999,
        dz_tall=0.5,
        dz_short=0.1,
        dz_param=2.0,
        dpai_min=0.01,
        nlayer_above=0,
        nlayer_within=0,
        rslfile="../rsl_lookup_tables/psihat.nc",
        dtime_substep=300.0,
    )
    with pytest.raises(ValueError, match="gs_type"):
        validate_config(config)


def test_validate_config_boundary_values() -> None:
    """
    Test validation with minimum valid boundary values for continuous parameters.
    
    Verifies that configurations with very small but valid positive values
    for continuous parameters pass validation.
    """
    config = MLCanopyConfig(
        clm_phys="CLM4_5",
        gs_type=2,
        gspot_type=1,
        colim_type=1,
        acclim_type=1,
        kn_val=0.0,
        turb_type=1,
        gb_type=3,
        light_type=2,
        longwave_type=1,
        fpi_type=2,
        root_type=2,
        fracdir=0.0,
        mlcan_to_clm=0,
        ml_vert_init=-9999,
        dz_tall=0.001,
        dz_short=0.001,
        dz_param=0.001,
        dpai_min=0.0001,
        nlayer_above=0,
        nlayer_within=0,
        rslfile="",
        dtime_substep=1.0,
    )
    assert validate_config(config) is True


def test_validate_config_negative_dz_tall() -> None:
    """
    Test validation failure with negative dz_tall (must be > 0).
    
    Verifies that validation raises ValueError when dz_tall violates
    the exclusive minimum constraint (must be > 0).
    """
    config = MLCanopyConfig(
        clm_phys="CLM4_5",
        gs_type=2,
        gspot_type=1,
        colim_type=1,
        acclim_type=1,
        kn_val=-999.0,
        turb_type=1,
        gb_type=3,
        light_type=2,
        longwave_type=1,
        fpi_type=2,
        root_type=2,
        fracdir=-999.0,
        mlcan_to_clm=0,
        ml_vert_init=-9999,
        dz_tall=-0.5,  # Invalid: must be > 0
        dz_short=0.1,
        dz_param=2.0,
        dpai_min=0.01,
        nlayer_above=0,
        nlayer_within=0,
        rslfile="../rsl_lookup_tables/psihat.nc",
        dtime_substep=300.0,
    )
    with pytest.raises(ValueError, match="dz_tall"):
        validate_config(config)


def test_validate_config_zero_dtime_substep() -> None:
    """
    Test validation failure with zero dtime_substep (must be > 0).
    
    Verifies that validation raises ValueError when dtime_substep is zero,
    violating the exclusive minimum constraint.
    """
    config = MLCanopyConfig(
        clm_phys="CLM4_5",
        gs_type=2,
        gspot_type=1,
        colim_type=1,
        acclim_type=1,
        kn_val=-999.0,
        turb_type=1,
        gb_type=3,
        light_type=2,
        longwave_type=1,
        fpi_type=2,
        root_type=2,
        fracdir=-999.0,
        mlcan_to_clm=0,
        ml_vert_init=-9999,
        dz_tall=0.5,
        dz_short=0.1,
        dz_param=2.0,
        dpai_min=0.01,
        nlayer_above=0,
        nlayer_within=0,
        rslfile="../rsl_lookup_tables/psihat.nc",
        dtime_substep=0.0,  # Invalid: must be > 0
    )
    with pytest.raises(ValueError, match="dtime_substep"):
        validate_config(config)


def test_validate_config_negative_nlayer() -> None:
    """
    Test validation failure with negative layer counts.
    
    Verifies that validation raises ValueError when nlayer_above or
    nlayer_within is negative (must be >= 0).
    """
    config = MLCanopyConfig(
        clm_phys="CLM4_5",
        gs_type=2,
        gspot_type=1,
        colim_type=1,
        acclim_type=1,
        kn_val=-999.0,
        turb_type=1,
        gb_type=3,
        light_type=2,
        longwave_type=1,
        fpi_type=2,
        root_type=2,
        fracdir=-999.0,
        mlcan_to_clm=0,
        ml_vert_init=-9999,
        dz_tall=0.5,
        dz_short=0.1,
        dz_param=2.0,
        dpai_min=0.01,
        nlayer_above=-1,  # Invalid: must be >= 0
        nlayer_within=0,
        rslfile="../rsl_lookup_tables/psihat.nc",
        dtime_substep=300.0,
    )
    with pytest.raises(ValueError, match="nlayer_above"):
        validate_config(config)


def test_validate_config_large_layer_counts() -> None:
    """
    Test validation with very large but valid layer counts for high-resolution simulations.
    
    Verifies that configurations with large layer counts (100+) pass validation,
    supporting high-resolution canopy modeling.
    """
    config = MLCanopyConfig(
        clm_phys="CLM5_0",
        gs_type=2,
        gspot_type=1,
        colim_type=1,
        acclim_type=1,
        kn_val=-999.0,
        turb_type=1,
        gb_type=3,
        light_type=2,
        longwave_type=1,
        fpi_type=2,
        root_type=2,
        fracdir=-999.0,
        mlcan_to_clm=0,
        ml_vert_init=-9999,
        dz_tall=0.5,
        dz_short=0.1,
        dz_param=2.0,
        dpai_min=0.01,
        nlayer_above=100,
        nlayer_within=500,
        rslfile="../rsl_lookup_tables/psihat.nc",
        dtime_substep=300.0,
    )
    assert validate_config(config) is True


@pytest.mark.parametrize(
    "gs_type,expected",
    [
        (0, True),  # Medlyn
        (1, True),  # Ball-Berry
        (2, True),  # WUE optimization
    ],
)
def test_validate_config_all_stomatal_schemes(gs_type: int, expected: bool) -> None:
    """
    Test validation of all three stomatal conductance schemes.
    
    Verifies that all valid gs_type values (0=Medlyn, 1=Ball-Berry,
    2=WUE optimization) pass validation.
    """
    config = MLCanopyConfig(
        clm_phys="CLM4_5",
        gs_type=gs_type,
        gspot_type=1,
        colim_type=1,
        acclim_type=1,
        kn_val=-999.0,
        turb_type=1,
        gb_type=3,
        light_type=2,
        longwave_type=1,
        fpi_type=2,
        root_type=2,
        fracdir=-999.0,
        mlcan_to_clm=0,
        ml_vert_init=-9999,
        dz_tall=0.5,
        dz_short=0.1,
        dz_param=2.0,
        dpai_min=0.01,
        nlayer_above=0,
        nlayer_within=0,
        rslfile="../rsl_lookup_tables/psihat.nc",
        dtime_substep=300.0,
    )
    assert validate_config(config) == expected


@pytest.mark.parametrize(
    "fracdir,expected",
    [
        (-999.0, True),  # Auto-compute flag
        (-1.0, True),  # Negative (auto-compute)
        (0.0, True),  # All diffuse
        (0.5, True),  # Mixed
        (1.0, True),  # All direct
    ],
)
def test_validate_config_fracdir_range(fracdir: float, expected: bool) -> None:
    """
    Test validation of fracdir across full valid range.
    
    Verifies that fracdir accepts both the auto-compute flag (-999.0 or negative)
    and physical values in [0, 1] representing the fraction of direct beam radiation.
    """
    config = MLCanopyConfig(
        clm_phys="CLM4_5",
        gs_type=2,
        gspot_type=1,
        colim_type=1,
        acclim_type=1,
        kn_val=-999.0,
        turb_type=1,
        gb_type=3,
        light_type=2,
        longwave_type=1,
        fpi_type=2,
        root_type=2,
        fracdir=fracdir,
        mlcan_to_clm=0,
        ml_vert_init=-9999,
        dz_tall=0.5,
        dz_short=0.1,
        dz_param=2.0,
        dpai_min=0.01,
        nlayer_above=0,
        nlayer_within=0,
        rslfile="../rsl_lookup_tables/psihat.nc",
        dtime_substep=300.0,
    )
    assert validate_config(config) == expected


# ============================================================================
# Configuration Query Tests
# ============================================================================


@pytest.mark.parametrize(
    "clm_phys,expected",
    [
        ("CLM4_5", False),
        ("CLM5_0", True),
    ],
)
def test_is_clm5_physics(clm_phys: str, expected: bool) -> None:
    """
    Test CLM5 physics detection for both CLM versions.
    
    Verifies that is_clm5_physics() correctly identifies CLM5_0 configurations
    and returns False for CLM4_5.
    """
    config = MLCanopyConfig(
        clm_phys=clm_phys,
        gs_type=2,
        gspot_type=1,
        colim_type=1,
        acclim_type=1,
        kn_val=-999.0,
        turb_type=1,
        gb_type=3,
        light_type=2,
        longwave_type=1,
        fpi_type=2,
        root_type=2,
        fracdir=-999.0,
        mlcan_to_clm=0,
        ml_vert_init=-9999,
        dz_tall=0.5,
        dz_short=0.1,
        dz_param=2.0,
        dpai_min=0.01,
        nlayer_above=0,
        nlayer_within=0,
        rslfile="../rsl_lookup_tables/psihat.nc",
        dtime_substep=300.0,
    )
    assert is_clm5_physics(config) == expected


@pytest.mark.parametrize(
    "turb_type,expected",
    [
        (-1, False),  # Dataset
        (0, False),  # Well-mixed
        (1, True),  # Harman & Finnigan RSL
    ],
)
def test_uses_rsl_turbulence(turb_type: int, expected: bool) -> None:
    """
    Test RSL turbulence detection across all valid turb_type values.
    
    Verifies that uses_rsl_turbulence() returns True only when turb_type=1
    (Harman & Finnigan RSL parameterization).
    """
    config = MLCanopyConfig(
        clm_phys="CLM4_5",
        gs_type=2,
        gspot_type=1,
        colim_type=1,
        acclim_type=1,
        kn_val=-999.0,
        turb_type=turb_type,
        gb_type=3,
        light_type=2,
        longwave_type=1,
        fpi_type=2,
        root_type=2,
        fracdir=-999.0,
        mlcan_to_clm=0,
        ml_vert_init=-9999,
        dz_tall=0.5,
        dz_short=0.1,
        dz_param=2.0,
        dpai_min=0.01,
        nlayer_above=0,
        nlayer_within=0,
        rslfile="../rsl_lookup_tables/psihat.nc",
        dtime_substep=300.0,
    )
    assert uses_rsl_turbulence(config) == expected


@pytest.mark.parametrize(
    "nlayer_above,nlayer_within,expected",
    [
        (0, 0, True),  # Both auto
        (5, 0, True),  # Within auto
        (0, 10, True),  # Above auto
        (5, 10, False),  # Both specified
    ],
)
def test_uses_auto_layers(
    nlayer_above: int, nlayer_within: int, expected: bool
) -> None:
    """
    Test auto-layer detection for all combinations of nlayer settings.
    
    Verifies that uses_auto_layers() returns True when either nlayer_above
    or nlayer_within is 0 (auto-determine), and False when both are specified.
    """
    config = MLCanopyConfig(
        clm_phys="CLM4_5",
        gs_type=2,
        gspot_type=1,
        colim_type=1,
        acclim_type=1,
        kn_val=-999.0,
        turb_type=1,
        gb_type=3,
        light_type=2,
        longwave_type=1,
        fpi_type=2,
        root_type=2,
        fracdir=-999.0,
        mlcan_to_clm=0,
        ml_vert_init=-9999,
        dz_tall=0.5,
        dz_short=0.1,
        dz_param=2.0,
        dpai_min=0.01,
        nlayer_above=nlayer_above,
        nlayer_within=nlayer_within,
        rslfile="../rsl_lookup_tables/psihat.nc",
        dtime_substep=300.0,
    )
    assert uses_auto_layers(config) == expected


# ============================================================================
# Canopy Height Parameter Selection Tests
# ============================================================================


def test_get_canopy_dz_tall_canopy() -> None:
    """
    Test dz selection for tall canopy (height > dz_param).
    
    Verifies that get_canopy_dz() returns dz_tall when canopy height
    exceeds the dz_param threshold.
    """
    config = create_default_config()
    canopy_height = 15.0  # > dz_param (2.0)
    result = get_canopy_dz(config, canopy_height)
    assert result == config.dz_tall


def test_get_canopy_dz_short_canopy() -> None:
    """
    Test dz selection for short canopy (height <= dz_param).
    
    Verifies that get_canopy_dz() returns dz_short when canopy height
    is less than or equal to the dz_param threshold.
    """
    config = create_default_config()
    canopy_height = 0.5  # < dz_param (2.0)
    result = get_canopy_dz(config, canopy_height)
    assert result == config.dz_short


def test_get_canopy_dz_boundary_height() -> None:
    """
    Test dz selection at exact boundary (height == dz_param, should use dz_short).
    
    Verifies that get_canopy_dz() returns dz_short when canopy height
    exactly equals dz_param (boundary condition uses <=).
    """
    config = create_default_config()
    canopy_height = config.dz_param  # Exactly at boundary
    result = get_canopy_dz(config, canopy_height)
    assert result == config.dz_short


@pytest.mark.parametrize(
    "canopy_height,expected_dz",
    [
        (0.01, 0.1),  # Very short (moss)
        (0.1, 0.1),  # Short grass
        (1.0, 0.1),  # Tall grass
        (10.0, 0.5),  # Medium tree
        (50.0, 0.5),  # Tall tree
        (100.0, 0.5),  # Very tall forest
    ],
)
def test_get_canopy_dz_extreme_heights(
    canopy_height: float, expected_dz: float
) -> None:
    """
    Test dz selection across wide range of canopy heights.
    
    Verifies that get_canopy_dz() correctly selects dz_short or dz_tall
    for canopy heights ranging from very short grass (0.01m) to tall
    forest (100m).
    """
    config = create_default_config()
    result = get_canopy_dz(config, canopy_height)
    assert result == expected_dz


def test_get_canopy_dz_custom_thresholds() -> None:
    """
    Test dz selection with custom dz parameters.
    
    Verifies that get_canopy_dz() works correctly with non-default
    dz_tall, dz_short, and dz_param values.
    """
    config = MLCanopyConfig(
        clm_phys="CLM4_5",
        gs_type=2,
        gspot_type=1,
        colim_type=1,
        acclim_type=1,
        kn_val=-999.0,
        turb_type=1,
        gb_type=3,
        light_type=2,
        longwave_type=1,
        fpi_type=2,
        root_type=2,
        fracdir=-999.0,
        mlcan_to_clm=0,
        ml_vert_init=-9999,
        dz_tall=1.0,
        dz_short=0.05,
        dz_param=5.0,
        dpai_min=0.01,
        nlayer_above=0,
        nlayer_within=0,
        rslfile="../rsl_lookup_tables/psihat.nc",
        dtime_substep=300.0,
    )
    
    # Test below threshold
    assert get_canopy_dz(config, 3.0) == 0.05
    # Test above threshold
    assert get_canopy_dz(config, 10.0) == 1.0
    # Test at threshold
    assert get_canopy_dz(config, 5.0) == 0.05


# ============================================================================
# Configuration Summary Tests
# ============================================================================


def test_config_summary_format() -> None:
    """
    Test config_summary returns properly formatted multi-line string.
    
    Verifies that config_summary() returns a string containing all
    configuration parameters in a human-readable format.
    """
    config = MLCanopyConfig(
        clm_phys="CLM5_0",
        gs_type=0,
        gspot_type=1,
        colim_type=1,
        acclim_type=1,
        kn_val=0.3,
        turb_type=1,
        gb_type=2,
        light_type=2,
        longwave_type=1,
        fpi_type=2,
        root_type=2,
        fracdir=0.6,
        mlcan_to_clm=1,
        ml_vert_init=1,
        dz_tall=0.8,
        dz_short=0.15,
        dz_param=3.0,
        dpai_min=0.02,
        nlayer_above=3,
        nlayer_within=15,
        rslfile="/data/rsl/psihat.nc",
        dtime_substep=180.0,
    )
    
    summary = config_summary(config)
    
    # Verify it's a string
    assert isinstance(summary, str)
    
    # Verify it contains multiple lines
    assert "\n" in summary
    
    # Verify it contains key parameter names
    assert "clm_phys" in summary
    assert "gs_type" in summary
    assert "dz_tall" in summary
    assert "nlayer_above" in summary
    
    # Verify it contains the actual values
    assert "CLM5_0" in summary
    assert "0.8" in summary
    assert "3" in summary


def test_config_summary_default_config() -> None:
    """
    Test config_summary with default configuration.
    
    Verifies that config_summary() works correctly with the default
    configuration and includes all expected fields.
    """
    config = create_default_config()
    summary = config_summary(config)
    
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "CLM4_5" in summary
    assert "300.0" in summary  # dtime_substep


def test_config_summary_contains_all_fields() -> None:
    """
    Test that config_summary includes all configuration fields.
    
    Verifies that the summary string contains references to all
    23 configuration parameters.
    """
    config = create_default_config()
    summary = config_summary(config)
    
    # Check for all field names
    expected_fields = [
        "clm_phys",
        "gs_type",
        "gspot_type",
        "colim_type",
        "acclim_type",
        "kn_val",
        "turb_type",
        "gb_type",
        "light_type",
        "longwave_type",
        "fpi_type",
        "root_type",
        "fracdir",
        "mlcan_to_clm",
        "ml_vert_init",
        "dz_tall",
        "dz_short",
        "dz_param",
        "dpai_min",
        "nlayer_above",
        "nlayer_within",
        "rslfile",
        "dtime_substep",
    ]
    
    for field in expected_fields:
        assert field in summary, f"Field '{field}' not found in summary"


# ============================================================================
# Integration Tests
# ============================================================================


def test_clm5_config_integration() -> None:
    """
    Integration test for CLM5 configuration workflow.
    
    Tests the complete workflow of creating a CLM5 config, validating it,
    and querying its properties.
    """
    # Create CLM5 config
    config = create_clm5_config()
    
    # Validate it
    assert validate_config(config) is True
    
    # Check CLM5 physics
    assert is_clm5_physics(config) is True
    
    # Check it uses auto layers by default
    assert uses_auto_layers(config) is True
    
    # Get summary
    summary = config_summary(config)
    assert "CLM5_0" in summary


def test_clm45_config_integration() -> None:
    """
    Integration test for CLM4.5 configuration workflow.
    
    Tests the complete workflow of creating a CLM4.5 config, validating it,
    and querying its properties.
    """
    # Create CLM4.5 config
    config = create_clm45_config()
    
    # Validate it
    assert validate_config(config) is True
    
    # Check CLM4.5 physics
    assert is_clm5_physics(config) is False
    
    # Check it uses auto layers by default
    assert uses_auto_layers(config) is True
    
    # Get summary
    summary = config_summary(config)
    assert "CLM4_5" in summary


def test_custom_config_workflow() -> None:
    """
    Integration test for custom configuration workflow.
    
    Tests creating a custom configuration with specific layer counts,
    validating it, and using it with get_canopy_dz().
    """
    # Create custom config with specified layers
    config = MLCanopyConfig(
        clm_phys="CLM5_0",
        gs_type=1,
        gspot_type=1,
        colim_type=1,
        acclim_type=1,
        kn_val=0.4,
        turb_type=1,
        gb_type=3,
        light_type=2,
        longwave_type=1,
        fpi_type=2,
        root_type=2,
        fracdir=0.5,
        mlcan_to_clm=1,
        ml_vert_init=1,
        dz_tall=0.6,
        dz_short=0.12,
        dz_param=3.0,
        dpai_min=0.015,
        nlayer_above=10,
        nlayer_within=30,
        rslfile="/custom/rsl.nc",
        dtime_substep=120.0,
    )
    
    # Validate
    assert validate_config(config) is True
    
    # Check it doesn't use auto layers
    assert uses_auto_layers(config) is False
    
    # Check RSL turbulence
    assert uses_rsl_turbulence(config) is True
    
    # Test canopy dz selection
    assert get_canopy_dz(config, 2.0) == 0.12  # Short
    assert get_canopy_dz(config, 5.0) == 0.6  # Tall
    
    # Get summary
    summary = config_summary(config)
    assert "10" in summary  # nlayer_above
    assert "30" in summary  # nlayer_within


# ============================================================================
# Type and Structure Tests
# ============================================================================


def test_mlcanopy_config_is_namedtuple() -> None:
    """
    Test that MLCanopyConfig is a proper namedtuple.
    
    Verifies that MLCanopyConfig has the expected namedtuple properties
    including immutability and field access.
    """
    config = create_default_config()
    
    # Check it has _fields attribute (namedtuple property)
    assert hasattr(config, "_fields")
    
    # Check field count
    assert len(config._fields) == 23
    
    # Check immutability (should raise AttributeError)
    with pytest.raises(AttributeError):
        config.gs_type = 999  # type: ignore


def test_mlcanopy_config_field_access() -> None:
    """
    Test field access methods for MLCanopyConfig.
    
    Verifies that all fields can be accessed both by attribute and by index.
    """
    config = create_default_config()
    
    # Test attribute access
    assert config.clm_phys == "CLM4_5"
    assert config.gs_type == 2
    
    # Test index access
    assert config[0] == "CLM4_5"  # clm_phys
    assert config[1] == 2  # gs_type
    
    # Test _asdict()
    config_dict = config._asdict()
    assert isinstance(config_dict, dict)
    assert config_dict["clm_phys"] == "CLM4_5"
    assert config_dict["gs_type"] == 2


def test_mlcanopy_config_replace() -> None:
    """
    Test _replace() method for creating modified configs.
    
    Verifies that the namedtuple _replace() method works correctly
    for creating new configurations with modified values.
    """
    config = create_default_config()
    
    # Create modified config
    new_config = config._replace(gs_type=0, dz_tall=1.0)
    
    # Check new values
    assert new_config.gs_type == 0
    assert new_config.dz_tall == 1.0
    
    # Check original unchanged
    assert config.gs_type == 2
    assert config.dz_tall == 0.5
    
    # Check other fields unchanged
    assert new_config.clm_phys == config.clm_phys
    assert new_config.dtime_substep == config.dtime_substep