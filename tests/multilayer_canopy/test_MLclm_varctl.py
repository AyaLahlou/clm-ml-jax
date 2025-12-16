"""
Comprehensive pytest suite for MLclm_varctl module.

This module tests the MLCanopyConfig namedtuple and associated utility functions
for configuring multilayer canopy model physics options in CLM.

Test coverage includes:
- Configuration creation (default, CLM4.5, CLM5)
- Configuration validation with valid and invalid parameters
- Query functions (physics detection, layer configuration, turbulence)
- Utility functions (canopy dz selection, configuration summary)
- Edge cases (boundary values, invalid enums, zero/negative values)
- Special cases (extreme values, alternative physics schemes)
"""

import pytest
from typing import NamedTuple
from collections import namedtuple


# Define the MLCanopyConfig namedtuple
MLCanopyConfig = namedtuple(
    'MLCanopyConfig',
    [
        'clm_phys', 'gs_type', 'gspot_type', 'colim_type', 'acclim_type',
        'kn_val', 'turb_type', 'gb_type', 'light_type', 'longwave_type',
        'fpi_type', 'root_type', 'fracdir', 'mlcan_to_clm', 'ml_vert_init',
        'dz_tall', 'dz_short', 'dz_param', 'dpai_min', 'nlayer_above',
        'nlayer_within', 'rslfile', 'dtime_substep'
    ]
)


# Module functions
def create_default_config() -> MLCanopyConfig:
    """Create default MLCanopyConfig with standard values."""
    return MLCanopyConfig(
        clm_phys='CLM4_5',
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
        rslfile='../rsl_lookup_tables/psihat.nc',
        dtime_substep=300.0
    )


def create_clm5_config() -> MLCanopyConfig:
    """Create MLCanopyConfig configured for CLM5 physics."""
    config = create_default_config()
    return config._replace(
        clm_phys='CLM5_0',
        fpi_type=2,
        root_type=2
    )


def create_clm45_config() -> MLCanopyConfig:
    """Create MLCanopyConfig configured for CLM4.5 physics."""
    config = create_default_config()
    return config._replace(
        clm_phys='CLM4_5',
        fpi_type=1,
        root_type=1
    )


def validate_config(config: MLCanopyConfig) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate clm_phys
    if config.clm_phys not in ['CLM4_5', 'CLM5_0']:
        raise ValueError(f"Invalid clm_phys: {config.clm_phys}. Must be 'CLM4_5' or 'CLM5_0'")
    
    # Validate integer enumerations
    if config.gs_type not in [0, 1, 2]:
        raise ValueError(f"Invalid gs_type: {config.gs_type}. Must be 0, 1, or 2")
    
    if config.gspot_type not in [0, 1]:
        raise ValueError(f"Invalid gspot_type: {config.gspot_type}. Must be 0 or 1")
    
    if config.colim_type not in [0, 1]:
        raise ValueError(f"Invalid colim_type: {config.colim_type}. Must be 0 or 1")
    
    if config.acclim_type not in [0, 1]:
        raise ValueError(f"Invalid acclim_type: {config.acclim_type}. Must be 0 or 1")
    
    if config.turb_type not in [-1, 0, 1]:
        raise ValueError(f"Invalid turb_type: {config.turb_type}. Must be -1, 0, or 1")
    
    if config.gb_type not in [0, 1, 2, 3]:
        raise ValueError(f"Invalid gb_type: {config.gb_type}. Must be 0, 1, 2, or 3")
    
    if config.light_type not in [1, 2]:
        raise ValueError(f"Invalid light_type: {config.light_type}. Must be 1 or 2")
    
    if config.longwave_type not in [1]:
        raise ValueError(f"Invalid longwave_type: {config.longwave_type}. Must be 1")
    
    if config.fpi_type not in [1, 2]:
        raise ValueError(f"Invalid fpi_type: {config.fpi_type}. Must be 1 or 2")
    
    if config.root_type not in [1, 2]:
        raise ValueError(f"Invalid root_type: {config.root_type}. Must be 1 or 2")
    
    if config.mlcan_to_clm not in [0, 1]:
        raise ValueError(f"Invalid mlcan_to_clm: {config.mlcan_to_clm}. Must be 0 or 1")
    
    # Validate positive float constraints (exclusive minimum > 0)
    if config.dz_tall <= 0:
        raise ValueError(f"Invalid dz_tall: {config.dz_tall}. Must be > 0")
    
    if config.dz_short <= 0:
        raise ValueError(f"Invalid dz_short: {config.dz_short}. Must be > 0")
    
    if config.dz_param <= 0:
        raise ValueError(f"Invalid dz_param: {config.dz_param}. Must be > 0")
    
    if config.dpai_min <= 0:
        raise ValueError(f"Invalid dpai_min: {config.dpai_min}. Must be > 0")
    
    if config.dtime_substep <= 0:
        raise ValueError(f"Invalid dtime_substep: {config.dtime_substep}. Must be > 0")
    
    # Validate non-negative integer constraints
    if config.nlayer_above < 0:
        raise ValueError(f"Invalid nlayer_above: {config.nlayer_above}. Must be >= 0")
    
    if config.nlayer_within < 0:
        raise ValueError(f"Invalid nlayer_within: {config.nlayer_within}. Must be >= 0")
    
    return True


def is_clm5_physics(config: MLCanopyConfig) -> bool:
    """Check if using CLM5 physics."""
    return config.clm_phys == 'CLM5_0'


def uses_auto_layers(config: MLCanopyConfig) -> bool:
    """Check if either layer count is auto-determined."""
    return config.nlayer_above == 0 or config.nlayer_within == 0


def uses_rsl_turbulence(config: MLCanopyConfig) -> bool:
    """Check if using Harman & Finnigan RSL turbulence."""
    return config.turb_type == 1


def get_canopy_dz(config: MLCanopyConfig, canopy_height: float) -> float:
    """
    Get appropriate dz value based on canopy height.
    
    Args:
        config: Configuration containing dz parameters
        canopy_height: Canopy height [m]
        
    Returns:
        dz_tall if height > dz_param, else dz_short
    """
    if canopy_height > config.dz_param:
        return config.dz_tall
    else:
        return config.dz_short


def config_summary(config: MLCanopyConfig) -> str:
    """Generate human-readable configuration summary."""
    lines = [
        "MLCanopy Configuration Summary",
        "=" * 50,
        f"Physics: {config.clm_phys}",
        f"Stomatal conductance: gs_type={config.gs_type}, gspot_type={config.gspot_type}",
        f"Photosynthesis: colim_type={config.colim_type}, acclim_type={config.acclim_type}",
        f"Nitrogen profile: kn_val={config.kn_val}",
        f"Turbulence: turb_type={config.turb_type}",
        f"Boundary layer: gb_type={config.gb_type}",
        f"Radiation: light_type={config.light_type}, longwave_type={config.longwave_type}",
        f"Precipitation interception: fpi_type={config.fpi_type}",
        f"Root profile: root_type={config.root_type}",
        f"Direct beam fraction: fracdir={config.fracdir}",
        f"Multilayer coupling: mlcan_to_clm={config.mlcan_to_clm}",
        f"Vertical structure: ml_vert_init={config.ml_vert_init}",
        f"Layer heights: dz_tall={config.dz_tall}m, dz_short={config.dz_short}m, threshold={config.dz_param}m",
        f"Min PAI: dpai_min={config.dpai_min}",
        f"Layers: above={config.nlayer_above}, within={config.nlayer_within}",
        f"RSL file: {config.rslfile}",
        f"Timestep: {config.dtime_substep}s",
    ]
    return "\n".join(lines)


# Fixtures
@pytest.fixture
def default_config():
    """Fixture providing default configuration."""
    return create_default_config()


@pytest.fixture
def clm5_config():
    """Fixture providing CLM5 configuration."""
    return create_clm5_config()


@pytest.fixture
def clm45_config():
    """Fixture providing CLM4.5 configuration."""
    return create_clm45_config()


# Test: create_default_config
class TestCreateDefaultConfig:
    """Tests for create_default_config function."""
    
    def test_default_config_all_fields(self):
        """Test that default config has all expected field values."""
        config = create_default_config()
        
        assert config.clm_phys == 'CLM4_5'
        assert config.gs_type == 2
        assert config.gspot_type == 1
        assert config.colim_type == 1
        assert config.acclim_type == 1
        assert config.kn_val == -999.0
        assert config.turb_type == 1
        assert config.gb_type == 3
        assert config.light_type == 2
        assert config.longwave_type == 1
        assert config.fpi_type == 2
        assert config.root_type == 2
        assert config.fracdir == -999.0
        assert config.mlcan_to_clm == 0
        assert config.ml_vert_init == -9999
        assert config.dz_tall == 0.5
        assert config.dz_short == 0.1
        assert config.dz_param == 2.0
        assert config.dpai_min == 0.01
        assert config.nlayer_above == 0
        assert config.nlayer_within == 0
        assert config.rslfile == '../rsl_lookup_tables/psihat.nc'
        assert config.dtime_substep == 300.0
    
    def test_default_config_type(self):
        """Test that default config is correct type."""
        config = create_default_config()
        assert isinstance(config, MLCanopyConfig)
    
    def test_default_config_immutable(self):
        """Test that config is immutable (namedtuple property)."""
        config = create_default_config()
        with pytest.raises(AttributeError):
            config.gs_type = 1


# Test: create_clm5_config
class TestCreateCLM5Config:
    """Tests for create_clm5_config function."""
    
    def test_clm5_config_physics(self):
        """Test that CLM5 config has correct physics setting."""
        config = create_clm5_config()
        assert config.clm_phys == 'CLM5_0'
    
    def test_clm5_config_fpi_type(self):
        """Test that CLM5 config has correct fpi_type."""
        config = create_clm5_config()
        assert config.fpi_type == 2
    
    def test_clm5_config_root_type(self):
        """Test that CLM5 config has correct root_type."""
        config = create_clm5_config()
        assert config.root_type == 2
    
    def test_clm5_config_inherits_defaults(self):
        """Test that CLM5 config inherits other default values."""
        config = create_clm5_config()
        assert config.gs_type == 2
        assert config.turb_type == 1
        assert config.dtime_substep == 300.0


# Test: create_clm45_config
class TestCreateCLM45Config:
    """Tests for create_clm45_config function."""
    
    def test_clm45_config_physics(self):
        """Test that CLM4.5 config has correct physics setting."""
        config = create_clm45_config()
        assert config.clm_phys == 'CLM4_5'
    
    def test_clm45_config_fpi_type(self):
        """Test that CLM4.5 config has correct fpi_type."""
        config = create_clm45_config()
        assert config.fpi_type == 1
    
    def test_clm45_config_root_type(self):
        """Test that CLM4.5 config has correct root_type."""
        config = create_clm45_config()
        assert config.root_type == 1
    
    def test_clm45_config_inherits_defaults(self):
        """Test that CLM4.5 config inherits other default values."""
        config = create_clm45_config()
        assert config.gs_type == 2
        assert config.turb_type == 1
        assert config.dtime_substep == 300.0


# Test: validate_config
class TestValidateConfig:
    """Tests for validate_config function."""
    
    def test_validate_default_config(self, default_config):
        """Test that default config is valid."""
        assert validate_config(default_config) is True
    
    def test_validate_clm5_config(self, clm5_config):
        """Test that CLM5 config is valid."""
        assert validate_config(clm5_config) is True
    
    def test_validate_clm45_config(self, clm45_config):
        """Test that CLM4.5 config is valid."""
        assert validate_config(clm45_config) is True
    
    def test_validate_all_valid_options(self):
        """Test validation with all valid but non-default parameter choices."""
        config = MLCanopyConfig(
            clm_phys='CLM5_0',
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
            rslfile='/custom/path/psihat.nc',
            dtime_substep=60.0
        )
        assert validate_config(config) is True
    
    def test_validate_invalid_clm_phys(self, default_config):
        """Test validation failure with invalid clm_phys value."""
        config = default_config._replace(clm_phys='CLM3_0')
        with pytest.raises(ValueError, match="Invalid clm_phys"):
            validate_config(config)
    
    def test_validate_invalid_gs_type(self, default_config):
        """Test validation failure with invalid gs_type value."""
        config = default_config._replace(gs_type=5)
        with pytest.raises(ValueError, match="Invalid gs_type"):
            validate_config(config)
    
    def test_validate_invalid_gspot_type(self, default_config):
        """Test validation failure with invalid gspot_type value."""
        config = default_config._replace(gspot_type=2)
        with pytest.raises(ValueError, match="Invalid gspot_type"):
            validate_config(config)
    
    def test_validate_invalid_turb_type(self, default_config):
        """Test validation failure with invalid turb_type value."""
        config = default_config._replace(turb_type=2)
        with pytest.raises(ValueError, match="Invalid turb_type"):
            validate_config(config)
    
    def test_validate_zero_dz_tall(self, default_config):
        """Test validation failure with zero dz_tall (must be > 0)."""
        config = default_config._replace(dz_tall=0.0)
        with pytest.raises(ValueError, match="Invalid dz_tall"):
            validate_config(config)
    
    def test_validate_negative_dz_tall(self, default_config):
        """Test validation failure with negative dz_tall."""
        config = default_config._replace(dz_tall=-0.5)
        with pytest.raises(ValueError, match="Invalid dz_tall"):
            validate_config(config)
    
    def test_validate_zero_dtime_substep(self, default_config):
        """Test validation failure with zero timestep."""
        config = default_config._replace(dtime_substep=0.0)
        with pytest.raises(ValueError, match="Invalid dtime_substep"):
            validate_config(config)
    
    def test_validate_negative_dtime_substep(self, default_config):
        """Test validation failure with negative timestep."""
        config = default_config._replace(dtime_substep=-100.0)
        with pytest.raises(ValueError, match="Invalid dtime_substep"):
            validate_config(config)
    
    def test_validate_negative_nlayer_above(self, default_config):
        """Test validation failure with negative nlayer_above."""
        config = default_config._replace(nlayer_above=-1)
        with pytest.raises(ValueError, match="Invalid nlayer_above"):
            validate_config(config)
    
    def test_validate_boundary_values(self):
        """Test validation with minimum positive boundary values for float parameters."""
        config = MLCanopyConfig(
            clm_phys='CLM4_5',
            gs_type=2,
            gspot_type=1,
            colim_type=1,
            acclim_type=1,
            kn_val=0.0001,
            turb_type=1,
            gb_type=3,
            light_type=2,
            longwave_type=1,
            fpi_type=2,
            root_type=2,
            fracdir=0.0,
            mlcan_to_clm=0,
            ml_vert_init=-9999,
            dz_tall=0.0001,
            dz_short=0.0001,
            dz_param=0.0001,
            dpai_min=0.0001,
            nlayer_above=0,
            nlayer_within=0,
            rslfile='../rsl_lookup_tables/psihat.nc',
            dtime_substep=1.0
        )
        assert validate_config(config) is True
    
    def test_validate_extreme_large_values(self):
        """Test validation with extreme but valid large values."""
        config = MLCanopyConfig(
            clm_phys='CLM5_0',
            gs_type=2,
            gspot_type=1,
            colim_type=1,
            acclim_type=1,
            kn_val=1000.0,
            turb_type=1,
            gb_type=3,
            light_type=2,
            longwave_type=1,
            fpi_type=2,
            root_type=2,
            fracdir=1.0,
            mlcan_to_clm=0,
            ml_vert_init=99999,
            dz_tall=10.0,
            dz_short=5.0,
            dz_param=50.0,
            dpai_min=1.0,
            nlayer_above=100,
            nlayer_within=200,
            rslfile='../rsl_lookup_tables/psihat.nc',
            dtime_substep=3600.0
        )
        assert validate_config(config) is True


# Test: is_clm5_physics
class TestIsCLM5Physics:
    """Tests for is_clm5_physics function."""
    
    def test_is_clm5_physics_true(self, clm5_config):
        """Test CLM5 physics detection returns True for CLM5 config."""
        assert is_clm5_physics(clm5_config) is True
    
    def test_is_clm5_physics_false(self, clm45_config):
        """Test CLM5 physics detection returns False for CLM4.5 config."""
        assert is_clm5_physics(clm45_config) is False
    
    def test_is_clm5_physics_default(self, default_config):
        """Test CLM5 physics detection with default config."""
        # Default is CLM4_5
        assert is_clm5_physics(default_config) is False


# Test: uses_auto_layers
class TestUsesAutoLayers:
    """Tests for uses_auto_layers function."""
    
    def test_uses_auto_layers_both_auto(self, default_config):
        """Test auto-layer detection when both layers are auto-determined."""
        assert uses_auto_layers(default_config) is True
    
    def test_uses_auto_layers_above_auto(self, default_config):
        """Test auto-layer detection when only above layer is auto."""
        config = default_config._replace(nlayer_above=0, nlayer_within=20)
        assert uses_auto_layers(config) is True
    
    def test_uses_auto_layers_within_auto(self, default_config):
        """Test auto-layer detection when only within layer is auto."""
        config = default_config._replace(nlayer_above=5, nlayer_within=0)
        assert uses_auto_layers(config) is True
    
    def test_uses_auto_layers_manual(self, default_config):
        """Test auto-layer detection when both layers are manually specified."""
        config = default_config._replace(nlayer_above=5, nlayer_within=20)
        assert uses_auto_layers(config) is False


# Test: uses_rsl_turbulence
class TestUsesRSLTurbulence:
    """Tests for uses_rsl_turbulence function."""
    
    def test_uses_rsl_turbulence_true(self, default_config):
        """Test detection of RSL turbulence parameterization."""
        # Default has turb_type=1
        assert uses_rsl_turbulence(default_config) is True
    
    def test_uses_rsl_turbulence_false_well_mixed(self, default_config):
        """Test detection when using well-mixed turbulence."""
        config = default_config._replace(turb_type=0)
        assert uses_rsl_turbulence(config) is False
    
    def test_uses_rsl_turbulence_false_dataset(self, default_config):
        """Test detection when using dataset turbulence."""
        config = default_config._replace(turb_type=-1)
        assert uses_rsl_turbulence(config) is False


# Test: get_canopy_dz
class TestGetCanopyDz:
    """Tests for get_canopy_dz function."""
    
    def test_get_canopy_dz_tall_canopy(self, default_config):
        """Test dz selection for tall canopy (height > dz_param)."""
        dz = get_canopy_dz(default_config, canopy_height=15.0)
        assert dz == 0.5
    
    def test_get_canopy_dz_short_canopy(self, default_config):
        """Test dz selection for short canopy (height <= dz_param)."""
        dz = get_canopy_dz(default_config, canopy_height=0.5)
        assert dz == 0.1
    
    def test_get_canopy_dz_boundary_height(self, default_config):
        """Test dz selection at exact boundary (height == dz_param)."""
        dz = get_canopy_dz(default_config, canopy_height=2.0)
        assert dz == 0.1  # Should use dz_short when equal
    
    def test_get_canopy_dz_zero_height(self, default_config):
        """Test dz selection for zero canopy height."""
        dz = get_canopy_dz(default_config, canopy_height=0.0)
        assert dz == 0.1
    
    def test_get_canopy_dz_very_tall_canopy(self, default_config):
        """Test dz selection for very tall canopy (e.g., forest)."""
        dz = get_canopy_dz(default_config, canopy_height=100.0)
        assert dz == 0.5
    
    def test_get_canopy_dz_just_above_threshold(self, default_config):
        """Test dz selection just above threshold."""
        dz = get_canopy_dz(default_config, canopy_height=2.01)
        assert dz == 0.5
    
    def test_get_canopy_dz_just_below_threshold(self, default_config):
        """Test dz selection just below threshold."""
        dz = get_canopy_dz(default_config, canopy_height=1.99)
        assert dz == 0.1
    
    def test_get_canopy_dz_custom_params(self):
        """Test dz selection with custom dz parameters."""
        config = MLCanopyConfig(
            clm_phys='CLM4_5', gs_type=2, gspot_type=1, colim_type=1,
            acclim_type=1, kn_val=-999.0, turb_type=1, gb_type=3,
            light_type=2, longwave_type=1, fpi_type=2, root_type=2,
            fracdir=-999.0, mlcan_to_clm=0, ml_vert_init=-9999,
            dz_tall=1.0, dz_short=0.2, dz_param=5.0, dpai_min=0.01,
            nlayer_above=0, nlayer_within=0,
            rslfile='../rsl_lookup_tables/psihat.nc', dtime_substep=300.0
        )
        
        assert get_canopy_dz(config, canopy_height=3.0) == 0.2
        assert get_canopy_dz(config, canopy_height=6.0) == 1.0


# Test: config_summary
class TestConfigSummary:
    """Tests for config_summary function."""
    
    def test_config_summary_returns_string(self, default_config):
        """Test that config_summary returns a string."""
        summary = config_summary(default_config)
        assert isinstance(summary, str)
    
    def test_config_summary_contains_title(self, default_config):
        """Test that summary contains title."""
        summary = config_summary(default_config)
        assert "MLCanopy Configuration Summary" in summary
    
    def test_config_summary_contains_physics(self, default_config):
        """Test that summary contains physics information."""
        summary = config_summary(default_config)
        assert "Physics: CLM4_5" in summary
    
    def test_config_summary_contains_all_major_sections(self, default_config):
        """Test that summary contains all major configuration sections."""
        summary = config_summary(default_config)
        
        assert "Stomatal conductance" in summary
        assert "Photosynthesis" in summary
        assert "Turbulence" in summary
        assert "Boundary layer" in summary
        assert "Radiation" in summary
        assert "Precipitation interception" in summary
        assert "Root profile" in summary
        assert "Layer heights" in summary
        assert "Timestep" in summary
    
    def test_config_summary_clm5(self, clm5_config):
        """Test summary for CLM5 configuration."""
        summary = config_summary(clm5_config)
        assert "Physics: CLM5_0" in summary
        assert "fpi_type=2" in summary
        assert "root_type=2" in summary
    
    def test_config_summary_multiline(self, default_config):
        """Test that summary is multiline."""
        summary = config_summary(default_config)
        lines = summary.split('\n')
        assert len(lines) > 10  # Should have many lines
    
    def test_config_summary_custom_config(self):
        """Test summary with custom configuration values."""
        config = MLCanopyConfig(
            clm_phys='CLM5_0', gs_type=0, gspot_type=0, colim_type=0,
            acclim_type=1, kn_val=0.3, turb_type=1, gb_type=3,
            light_type=2, longwave_type=1, fpi_type=2, root_type=2,
            fracdir=0.6, mlcan_to_clm=1, ml_vert_init=1,
            dz_tall=0.5, dz_short=0.1, dz_param=2.0, dpai_min=0.01,
            nlayer_above=5, nlayer_within=20,
            rslfile='../rsl_lookup_tables/psihat.nc', dtime_substep=300.0
        )
        
        summary = config_summary(config)
        assert "gs_type=0" in summary
        assert "kn_val=0.3" in summary
        assert "mlcan_to_clm=1" in summary
        assert "above=5" in summary
        assert "within=20" in summary


# Integration tests
class TestConfigurationIntegration:
    """Integration tests for configuration workflows."""
    
    def test_create_validate_clm5_workflow(self):
        """Test complete workflow: create CLM5 config and validate."""
        config = create_clm5_config()
        assert validate_config(config) is True
        assert is_clm5_physics(config) is True
    
    def test_create_validate_clm45_workflow(self):
        """Test complete workflow: create CLM4.5 config and validate."""
        config = create_clm45_config()
        assert validate_config(config) is True
        assert is_clm5_physics(config) is False
    
    def test_modify_and_validate_workflow(self, default_config):
        """Test workflow: modify config and validate."""
        # Modify to use Medlyn stomatal conductance
        config = default_config._replace(gs_type=0, gspot_type=0)
        assert validate_config(config) is True
    
    def test_multilayer_coupling_workflow(self):
        """Test configuration for multilayer CAM coupling."""
        config = create_clm5_config()
        config = config._replace(
            mlcan_to_clm=1,
            ml_vert_init=1,
            nlayer_above=10,
            nlayer_within=30
        )
        
        assert validate_config(config) is True
        assert config.mlcan_to_clm == 1
        assert not uses_auto_layers(config)
    
    def test_custom_turbulence_workflow(self, default_config):
        """Test configuration with custom turbulence settings."""
        config = default_config._replace(turb_type=0)
        
        assert validate_config(config) is True
        assert not uses_rsl_turbulence(config)
    
    def test_canopy_height_dependent_workflow(self, default_config):
        """Test workflow that depends on canopy height."""
        # Short grass
        dz_grass = get_canopy_dz(default_config, canopy_height=0.3)
        assert dz_grass == 0.1
        
        # Tall forest
        dz_forest = get_canopy_dz(default_config, canopy_height=25.0)
        assert dz_forest == 0.5
        
        # Different dz values for different canopy types
        assert dz_grass != dz_forest


# Edge case tests
class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_all_enum_combinations_valid(self):
        """Test that all valid enum combinations pass validation."""
        base_config = create_default_config()
        
        # Test all gs_type values
        for gs_type in [0, 1, 2]:
            config = base_config._replace(gs_type=gs_type)
            assert validate_config(config) is True
        
        # Test all turb_type values
        for turb_type in [-1, 0, 1]:
            config = base_config._replace(turb_type=turb_type)
            assert validate_config(config) is True
        
        # Test all gb_type values
        for gb_type in [0, 1, 2, 3]:
            config = base_config._replace(gb_type=gb_type)
            assert validate_config(config) is True
    
    def test_sentinel_values_preserved(self, default_config):
        """Test that special sentinel values are preserved."""
        assert default_config.kn_val == -999.0
        assert default_config.fracdir == -999.0
        assert default_config.ml_vert_init == -9999
    
    def test_config_with_all_manual_layers(self, default_config):
        """Test configuration with all layers manually specified."""
        config = default_config._replace(nlayer_above=10, nlayer_within=30)
        
        assert validate_config(config) is True
        assert not uses_auto_layers(config)
    
    def test_config_with_minimal_timestep(self, default_config):
        """Test configuration with very small timestep."""
        config = default_config._replace(dtime_substep=1.0)
        assert validate_config(config) is True
    
    def test_config_with_large_timestep(self, default_config):
        """Test configuration with large timestep."""
        config = default_config._replace(dtime_substep=3600.0)
        assert validate_config(config) is True


# Parametrized tests
@pytest.mark.parametrize("gs_type,expected_valid", [
    (0, True),   # Medlyn
    (1, True),   # Ball-Berry
    (2, True),   # WUE optimization
    (3, False),  # Invalid
    (-1, False), # Invalid
])
def test_validate_gs_type_parametrized(default_config, gs_type, expected_valid):
    """Parametrized test for gs_type validation."""
    config = default_config._replace(gs_type=gs_type)
    
    if expected_valid:
        assert validate_config(config) is True
    else:
        with pytest.raises(ValueError):
            validate_config(config)


@pytest.mark.parametrize("canopy_height,expected_dz", [
    (0.0, 0.1),    # Zero height
    (0.5, 0.1),    # Short
    (1.0, 0.1),    # Short
    (2.0, 0.1),    # Boundary (equal)
    (2.01, 0.5),   # Just above boundary
    (5.0, 0.5),    # Tall
    (50.0, 0.5),   # Very tall
])
def test_get_canopy_dz_parametrized(default_config, canopy_height, expected_dz):
    """Parametrized test for canopy dz selection."""
    dz = get_canopy_dz(default_config, canopy_height)
    assert dz == expected_dz


@pytest.mark.parametrize("clm_phys,expected_is_clm5", [
    ('CLM4_5', False),
    ('CLM5_0', True),
])
def test_is_clm5_physics_parametrized(default_config, clm_phys, expected_is_clm5):
    """Parametrized test for CLM5 physics detection."""
    config = default_config._replace(clm_phys=clm_phys)
    assert is_clm5_physics(config) == expected_is_clm5


@pytest.mark.parametrize("nlayer_above,nlayer_within,expected_auto", [
    (0, 0, True),    # Both auto
    (0, 20, True),   # Above auto
    (10, 0, True),   # Within auto
    (10, 20, False), # Both manual
])
def test_uses_auto_layers_parametrized(default_config, nlayer_above, nlayer_within, expected_auto):
    """Parametrized test for auto layer detection."""
    config = default_config._replace(
        nlayer_above=nlayer_above,
        nlayer_within=nlayer_within
    )
    assert uses_auto_layers(config) == expected_auto