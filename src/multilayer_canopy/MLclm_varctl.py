"""
Multilayer Canopy Model Run Control Variables.

Translated from CTSM's MLclm_varctl.F90 (lines 1-52)

This module contains configuration parameters and control variables for the
multilayer canopy model. These parameters control:
- Model physics options (stomatal conductance, photosynthesis, turbulence)
- Canopy layer discretization
- Radiative transfer schemes
- Root and precipitation interception schemes

All parameters are stored in an immutable NamedTuple for JIT compatibility.

Key Configuration Categories:
    1. Physics schemes (gs_type, colim_type, turb_type, etc.)
    2. Canopy discretization (dz_tall, dz_short, dpai_min)
    3. Layer specification (nlayer_above, nlayer_within)
    4. File paths and timesteps (rslfile, dtime_substep)

Usage:
    # Create default configuration
    config = create_default_config()
    
    # Modify specific parameters
    config = config._replace(gs_type=1, turb_type=0)
    
    # Validate configuration
    validate_config(config)
"""

from typing import NamedTuple
import jax.numpy as jnp


# =============================================================================
# Configuration Type
# =============================================================================


class MLCanopyConfig(NamedTuple):
    """Configuration parameters for multilayer canopy model.
    
    This immutable configuration structure contains all control variables
    for the multilayer canopy model, ensuring JIT compatibility and
    reproducible physics.
    
    Attributes:
        clm_phys: Snow/soil layer configuration ('CLM4_5' or 'CLM5_0')
            Controls compatibility with different CLM versions
            
        gs_type: Stomatal conductance scheme (line 17)
            0 = Medlyn conductance model
            1 = Ball-Berry conductance model
            2 = Water use efficiency (WUE) optimization
            
        gspot_type: Stomatal conductance water stress (line 18)
            0 = Potential conductance (no water stress)
            1 = Water-stressed conductance
            
        colim_type: Photosynthesis co-limitation (line 19)
            0 = Minimum rate (Collatz approach)
            1 = Co-limited rate (smoothed transition)
            
        acclim_type: Photosynthesis temperature acclimation (line 20)
            0 = No acclimation
            1 = Temperature acclimation enabled
            
        kn_val: Canopy nitrogen profile coefficient Kn (line 21)
            > 0: User-specified Kn value
            -999.0: Use default calculation
            
        turb_type: Turbulence parameterization (line 22)
            -1 = Use dataset values
            0 = Well-mixed assumption
            1 = Harman & Finnigan roughness sublayer
            
        gb_type: Boundary layer conductance formulation (line 23)
            0 = CLM5 default
            1 = Laminar only
            2 = Laminar + turbulent
            3 = Laminar + turbulent + free convection
            
        light_type: Solar radiative transfer scheme (line 24)
            1 = Norman (1979) scheme
            2 = Two-stream approximation
            
        longwave_type: Longwave radiative transfer scheme (line 25)
            1 = Norman (1979) scheme
            
        fpi_type: Fraction of precipitation intercepted (line 26)
            1 = CLM4.5 formulation
            2 = CLM5 formulation
            
        root_type: Root vertical profile (line 27)
            1 = CLM4.5 (Zeng 2001)
            2 = CLM5 (Jackson 1996)
            
        fracdir: Fraction of solar radiation that is direct beam (line 28)
            >= 0: Use specified value
            < 0: Compute from model
            
        mlcan_to_clm: Pass multilayer fluxes to CLM for CAM coupling (line 29)
            0 = No (use single-layer fluxes)
            1 = Yes (use multilayer fluxes)
            
        ml_vert_init: Multilayer vertical structure initialization flag (line 30)
            -9999 = Not initialized (ispval from clm_varcon)
            Other values indicate initialization state
            
        dz_tall: Height increment for tall canopies [m] (line 33)
            Used when canopy height > dz_param
            
        dz_short: Height increment for short canopies [m] (line 34)
            Used when canopy height <= dz_param
            
        dz_param: Height threshold for tall vs short canopy [m] (line 35)
            Determines which dz value to use
            
        dpai_min: Minimum plant area index for vegetation layer [m2/m2] (line 36)
            Layers with PAI < dpai_min are combined
            
        nlayer_above: Number of above-canopy layers (line 40)
            0 = Auto-determine from canopy height
            > 0 = Use specified number
            
        nlayer_within: Number of within-canopy layers (line 41)
            0 = Auto-determine from PAI and dz
            > 0 = Use specified number
            
        rslfile: Full pathname for RSL psihat lookup tables (line 46)
            Used when turb_type = 1 (Harman & Finnigan)
            
        dtime_substep: Model sub-timestep [s] (line 50)
            Default: 300 s (5 minutes)
            Used for numerical stability in canopy physics
    
    Note:
        All parameters are immutable for JIT compatibility. Use the
        _replace() method to create modified configurations.
        
        Default values match Fortran module initialization (lines 16-50).
    """
    
    # =========================================================================
    # Model Physics Options (lines 16-30)
    # =========================================================================
    
    clm_phys: str = 'CLM4_5'
    gs_type: int = 2
    gspot_type: int = 1
    colim_type: int = 1
    acclim_type: int = 1
    kn_val: float = -999.0
    turb_type: int = 1
    gb_type: int = 3
    light_type: int = 2
    longwave_type: int = 1
    fpi_type: int = 2
    root_type: int = 2
    fracdir: float = -999.0
    mlcan_to_clm: int = 0
    ml_vert_init: int = -9999  # ispval from clm_varcon
    
    # =========================================================================
    # Canopy Layer Discretization (lines 33-36)
    # =========================================================================
    
    dz_tall: float = 0.5
    dz_short: float = 0.1
    dz_param: float = 2.0
    dpai_min: float = 0.01
    
    # =========================================================================
    # Direct Layer Specification (lines 40-42)
    # =========================================================================
    
    nlayer_above: int = 0
    nlayer_within: int = 0
    
    # =========================================================================
    # File Paths and Timestep (lines 46-50)
    # =========================================================================
    
    rslfile: str = '../rsl_lookup_tables/psihat.nc'
    dtime_substep: float = 5.0 * 60.0  # 5 minutes in seconds


# =============================================================================
# Configuration Factory Functions
# =============================================================================


def create_default_config() -> MLCanopyConfig:
    """Create default multilayer canopy configuration.
    
    Returns:
        Default MLCanopyConfig with values from lines 16-50 of MLclm_varctl.F90
        
    Example:
        >>> config = create_default_config()
        >>> config.gs_type
        2
        >>> config.dz_tall
        0.5
        
    Note:
        This function provides a convenient way to get default parameters
        that can then be modified for specific experiments using _replace():
        
        >>> custom_config = config._replace(gs_type=1, turb_type=0)
    """
    return MLCanopyConfig()


def create_clm5_config() -> MLCanopyConfig:
    """Create CLM5-compatible configuration.
    
    Returns:
        MLCanopyConfig configured for CLM5 physics
        
    Note:
        Sets clm_phys='CLM5_0', fpi_type=2, root_type=2
    """
    return MLCanopyConfig(
        clm_phys='CLM5_0',
        fpi_type=2,
        root_type=2,
    )


def create_clm45_config() -> MLCanopyConfig:
    """Create CLM4.5-compatible configuration.
    
    Returns:
        MLCanopyConfig configured for CLM4.5 physics
        
    Note:
        Sets clm_phys='CLM4_5', fpi_type=1, root_type=1
    """
    return MLCanopyConfig(
        clm_phys='CLM4_5',
        fpi_type=1,
        root_type=1,
    )


# =============================================================================
# Configuration Validation
# =============================================================================


def validate_config(config: MLCanopyConfig) -> bool:
    """Validate multilayer canopy configuration parameters.
    
    Checks that all configuration parameters are within valid ranges
    as specified in the Fortran source (lines 16-50).
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If any parameter is out of valid range, with
            descriptive message indicating which parameter failed
            and what the valid range is
            
    Example:
        >>> config = create_default_config()
        >>> validate_config(config)
        True
        
        >>> bad_config = config._replace(gs_type=5)
        >>> validate_config(bad_config)
        ValueError: gs_type must be 0, 1, or 2, got 5
        
    Note:
        Validation checks are based on allowed values documented in
        lines 16-29 of MLclm_varctl.F90.
    """
    # =========================================================================
    # Validate Physics Scheme Options (lines 17-29)
    # =========================================================================
    
    # Stomatal conductance scheme (line 17)
    if config.gs_type not in [0, 1, 2]:
        raise ValueError(
            f"gs_type must be 0 (Medlyn), 1 (Ball-Berry), or 2 (WUE), "
            f"got {config.gs_type}"
        )
    
    # Water stress option (line 18)
    if config.gspot_type not in [0, 1]:
        raise ValueError(
            f"gspot_type must be 0 (potential) or 1 (water-stressed), "
            f"got {config.gspot_type}"
        )
    
    # Photosynthesis co-limitation (line 19)
    if config.colim_type not in [0, 1]:
        raise ValueError(
            f"colim_type must be 0 (minimum) or 1 (co-limited), "
            f"got {config.colim_type}"
        )
    
    # Temperature acclimation (line 20)
    if config.acclim_type not in [0, 1]:
        raise ValueError(
            f"acclim_type must be 0 (off) or 1 (on), "
            f"got {config.acclim_type}"
        )
    
    # Turbulence parameterization (line 22)
    if config.turb_type not in [-1, 0, 1]:
        raise ValueError(
            f"turb_type must be -1 (dataset), 0 (well-mixed), or 1 (H&F RSL), "
            f"got {config.turb_type}"
        )
    
    # Boundary layer conductance (line 23)
    if config.gb_type not in [0, 1, 2, 3]:
        raise ValueError(
            f"gb_type must be 0 (CLM5), 1 (laminar), 2 (lam+turb), "
            f"or 3 (lam+turb+free), got {config.gb_type}"
        )
    
    # Solar radiative transfer (line 24)
    if config.light_type not in [1, 2]:
        raise ValueError(
            f"light_type must be 1 (Norman) or 2 (two-stream), "
            f"got {config.light_type}"
        )
    
    # Longwave radiative transfer (line 25)
    if config.longwave_type != 1:
        raise ValueError(
            f"longwave_type must be 1 (Norman), "
            f"got {config.longwave_type}"
        )
    
    # Precipitation interception (line 26)
    if config.fpi_type not in [1, 2]:
        raise ValueError(
            f"fpi_type must be 1 (CLM4.5) or 2 (CLM5), "
            f"got {config.fpi_type}"
        )
    
    # Root profile (line 27)
    if config.root_type not in [1, 2]:
        raise ValueError(
            f"root_type must be 1 (CLM4.5/Zeng2001) or 2 (CLM5/Jackson1996), "
            f"got {config.root_type}"
        )
    
    # Multilayer to CLM coupling (line 29)
    if config.mlcan_to_clm not in [0, 1]:
        raise ValueError(
            f"mlcan_to_clm must be 0 (no) or 1 (yes), "
            f"got {config.mlcan_to_clm}"
        )
    
    # =========================================================================
    # Validate Discretization Parameters (lines 33-36)
    # =========================================================================
    
    if config.dz_tall <= 0:
        raise ValueError(
            f"dz_tall must be > 0 m, got {config.dz_tall}"
        )
    
    if config.dz_short <= 0:
        raise ValueError(
            f"dz_short must be > 0 m, got {config.dz_short}"
        )
    
    if config.dz_param <= 0:
        raise ValueError(
            f"dz_param must be > 0 m, got {config.dz_param}"
        )
    
    if config.dpai_min <= 0:
        raise ValueError(
            f"dpai_min must be > 0 m2/m2, got {config.dpai_min}"
        )
    
    # =========================================================================
    # Validate Layer Counts (lines 40-42)
    # =========================================================================
    
    if config.nlayer_above < 0:
        raise ValueError(
            f"nlayer_above must be >= 0 (0=auto), got {config.nlayer_above}"
        )
    
    if config.nlayer_within < 0:
        raise ValueError(
            f"nlayer_within must be >= 0 (0=auto), got {config.nlayer_within}"
        )
    
    # =========================================================================
    # Validate Timestep (line 50)
    # =========================================================================
    
    if config.dtime_substep <= 0:
        raise ValueError(
            f"dtime_substep must be > 0 s, got {config.dtime_substep}"
        )
    
    return True


# =============================================================================
# Configuration Query Functions
# =============================================================================


def is_clm5_physics(config: MLCanopyConfig) -> bool:
    """Check if configuration uses CLM5 physics.
    
    Args:
        config: Configuration to check
        
    Returns:
        True if using CLM5 physics options
        
    Note:
        Checks clm_phys string (line 16)
    """
    return config.clm_phys == 'CLM5_0'


def uses_auto_layers(config: MLCanopyConfig) -> bool:
    """Check if configuration uses automatic layer determination.
    
    Args:
        config: Configuration to check
        
    Returns:
        True if either nlayer_above or nlayer_within is 0 (auto)
        
    Note:
        Based on lines 40-42
    """
    return config.nlayer_above == 0 or config.nlayer_within == 0


def uses_rsl_turbulence(config: MLCanopyConfig) -> bool:
    """Check if configuration uses roughness sublayer turbulence.
    
    Args:
        config: Configuration to check
        
    Returns:
        True if using Harman & Finnigan RSL parameterization
        
    Note:
        Based on turb_type (line 22)
    """
    return config.turb_type == 1


def get_canopy_dz(config: MLCanopyConfig, canopy_height: float) -> float:
    """Get appropriate vertical discretization for canopy height.
    
    Args:
        config: Configuration containing dz parameters
        canopy_height: Canopy height [m]
        
    Returns:
        Appropriate dz value [m] based on canopy height
        
    Note:
        Uses dz_param threshold (line 35) to select between
        dz_tall (line 33) and dz_short (line 34)
    """
    return jnp.where(
        canopy_height > config.dz_param,
        config.dz_tall,
        config.dz_short
    )


# =============================================================================
# Configuration Display
# =============================================================================


def config_summary(config: MLCanopyConfig) -> str:
    """Generate human-readable summary of configuration.
    
    Args:
        config: Configuration to summarize
        
    Returns:
        Multi-line string describing configuration
        
    Example:
        >>> config = create_default_config()
        >>> print(config_summary(config))
        Multilayer Canopy Configuration:
          Physics: CLM4_5
          Stomatal conductance: WUE optimization (gs_type=2)
          ...
    """
    gs_names = {0: "Medlyn", 1: "Ball-Berry", 2: "WUE optimization"}
    turb_names = {-1: "Dataset", 0: "Well-mixed", 1: "H&F RSL"}
    
    lines = [
        "Multilayer Canopy Configuration:",
        f"  Physics: {config.clm_phys}",
        f"  Stomatal conductance: {gs_names[config.gs_type]} (gs_type={config.gs_type})",
        f"  Water stress: {'On' if config.gspot_type == 1 else 'Off'}",
        f"  Photosynthesis co-limitation: {'Smoothed' if config.colim_type == 1 else 'Minimum'}",
        f"  Temperature acclimation: {'On' if config.acclim_type == 1 else 'Off'}",
        f"  Turbulence: {turb_names[config.turb_type]} (turb_type={config.turb_type})",
        f"  Boundary layer: gb_type={config.gb_type}",
        f"  Radiative transfer: light_type={config.light_type}, longwave_type={config.longwave_type}",
        "",
        "Discretization:",
        f"  dz_tall={config.dz_tall} m, dz_short={config.dz_short} m",
        f"  dz_param={config.dz_param} m (threshold)",
        f"  dpai_min={config.dpai_min} m2/m2",
        f"  nlayer_above={config.nlayer_above} ({'auto' if config.nlayer_above == 0 else 'fixed'})",
        f"  nlayer_within={config.nlayer_within} ({'auto' if config.nlayer_within == 0 else 'fixed'})",
        "",
        f"Timestep: {config.dtime_substep} s ({config.dtime_substep/60:.1f} min)",
    ]
    
    return "\n".join(lines)