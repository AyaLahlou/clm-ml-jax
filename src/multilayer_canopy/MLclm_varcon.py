"""
Multilayer Canopy Model Constants.

Translated from CTSM's MLclm_varcon.F90 (lines 1-153)

This module contains physical constants and adjustable parameters for the
multilayer canopy model. These include:
- Physical constants (gas constant, molecular masses, specific heats, etc.)
- Photosynthesis parameters (Michaelis-Menten constants, activation energies)
- Stomatal conductance parameters
- Leaf heat capacity parameters
- Boundary layer parameters
- Canopy interception parameters
- Solar and longwave radiation parameters
- Roughness sublayer parameterization parameters
- Numerical limits and constraints

All constants are immutable and stored in NamedTuples for JAX compatibility.

Original Fortran module: MLclm_varcon.F90
Dependencies: clm_varcon (SPVAL), shr_kind_mod (r8)

Key Features:
    - Pure constants module with no computational functions
    - Immutable NamedTuple structures for JIT compilation
    - Comprehensive parameter documentation with physical units
    - Support for both acclimated and non-acclimated photosynthesis
    - Roughness sublayer (RSL) lookup table infrastructure

Usage:
    from jax_ctsm.multilayer_canopy.mlclm_varcon import (
        ML_CANOPY_CONSTANTS,
        MLCanopyConstants,
        RSLPsihatLookupTables,
        create_empty_rsl_lookup_tables,
    )
    
    # Access constants
    rgas = ML_CANOPY_CONSTANTS.rgas
    vcmaxha = ML_CANOPY_CONSTANTS.vcmaxha_noacclim
    
    # Create custom constants (e.g., for sensitivity tests)
    custom_constants = MLCanopyConstants(
        vcmaxha_noacclim=70000.0,  # Modified value
    )
    
    # Initialize lookup tables
    lookup_tables = create_empty_rsl_lookup_tables()
"""

from typing import NamedTuple
import jax.numpy as jnp

# =============================================================================
# Module-Level Constants
# =============================================================================

# Special value for uninitialized/invalid data (from clm_varcon)
# Used to mark parameters that should be set at runtime or are not applicable
SPVAL = 1.0e36


# =============================================================================
# Main Constants Structure
# =============================================================================

class MLCanopyConstants(NamedTuple):
    """Physical constants and parameters for multilayer canopy model.
    
    All values are immutable for JAX compatibility. Parameters are organized
    by functional category matching the original Fortran module structure.
    
    Attributes:
        Physical constants (lines 16-25):
            rgas: Universal gas constant [J/K/mol]
            mmdry: Molecular mass of dry air [kg/mol]
            mmh2o: Molecular mass of water vapor [kg/mol]
            cpd: Specific heat of dry air at constant pressure [J/kg/K]
            cpw: Specific heat of water vapor at constant pressure [J/kg/K]
            visc0: Kinematic viscosity at 0C and 1013.25 hPa [m2/s]
            dh0: Molecular diffusivity (heat) at 0C and 1013.25 hPa [m2/s]
            dv0: Molecular diffusivity (H2O) at 0C and 1013.25 hPa [m2/s]
            dc0: Molecular diffusivity (CO2) at 0C and 1013.25 hPa [m2/s]
            lapse_rate: Temperature lapse rate [K/m]
            
        Leaf photosynthesis parameters (lines 31-68):
            kc25: Michaelis-Menten constant for CO2 at 25C [umol/mol]
            kcha: Activation energy for kc [J/mol]
            ko25: Michaelis-Menten constant for O2 at 25C [mmol/mol]
            koha: Activation energy for ko [J/mol]
            cp25: CO2 compensation point at 25C [umol/mol]
            cpha: Activation energy for cp [J/mol]
            vcmaxha_noacclim: Activation energy for vcmax without acclimation [J/mol]
            vcmaxha_acclim: Activation energy for vcmax with acclimation [J/mol]
            vcmaxhd_noacclim: Deactivation energy for vcmax without acclimation [J/mol]
            vcmaxhd_acclim: Deactivation energy for vcmax with acclimation [J/mol]
            vcmaxse_noacclim: Entropy term for vcmax without acclimation [J/mol/K]
            vcmaxse_acclim: Entropy term for vcmax with acclimation [J/mol/K]
            jmaxha_noacclim: Activation energy for jmax without acclimation [J/mol]
            jmaxha_acclim: Activation energy for jmax with acclimation [J/mol]
            jmaxhd_noacclim: Deactivation energy for jmax without acclimation [J/mol]
            jmaxhd_acclim: Deactivation energy for jmax with acclimation [J/mol]
            jmaxse_noacclim: Entropy term for jmax without acclimation [J/mol/K]
            jmaxse_acclim: Entropy term for jmax with acclimation [J/mol/K]
            rdha: Activation energy for rd [J/mol]
            rdhd: Deactivation energy for rd [J/mol]
            rdse: Entropy term for rd [J/mol/K]
            jmax25_to_vcmax25_noacclim: Ratio of jmax to vcmax at 25C without acclimation [umol/umol]
            jmax25_to_vcmax25_acclim: Ratio of jmax to vcmax at 25C with acclimation [umol/umol]
            rd25_to_vcmax25_c3: Ratio of rd to vcmax at 25C (C3) [umol/umol]
            rd25_to_vcmax25_c4: Ratio of rd to vcmax at 25C (C4) [umol/umol]
            kp25_to_vcmax25_c4: Ratio of kp to vcmax at 25C (C4) [mol/umol]
            phi_psii: C3 quantum yield of PS II [dimensionless]
            theta_j: C3 empirical curvature parameter for electron transport rate [dimensionless]
            qe_c4: C4 quantum yield [mol CO2 / mol photons]
            colim_c3a: Empirical curvature parameter for C3 co-limitation (Ac, Aj) [dimensionless]
            colim_c3b: Empirical curvature parameter for C3 co-limitation (Ap) [dimensionless]
            colim_c4a: Empirical curvature parameter for C4 co-limitation (Ac, Aj) [dimensionless]
            colim_c4b: Empirical curvature parameter for C4 co-limitation (Ap) [dimensionless]
            
        Stomatal conductance parameters (lines 72-74):
            dh2o_to_dco2: Diffusivity H2O / Diffusivity CO2 [dimensionless]
            rh_min_bb: Minimum relative humidity for Ball-Berry stomatal conductance [fraction]
            vpd_min_med: Minimum vapor pressure deficit for Medlyn stomatal conductance [Pa]
            
        Leaf heat capacity parameters (lines 78-80):
            cpbio: Specific heat of dry biomass [J/kg/K]
            fcarbon: Fraction of dry biomass that is carbon [kg C / kg DM]
            fwater: Fraction of fresh biomass that is water [kg H2O / kg FM]
            
        Leaf boundary layer parameters (lines 84-85):
            gb_factor: Empirical correction factor for Nu [dimensionless]
            
        Canopy interception parameters (lines 89-94):
            dewmx: Maximum allowed interception [kg H2O/m2 leaf]
            maximum_leaf_wetted_fraction: Maximum fraction of leaf that can be wet [dimensionless]
            interception_fraction: Fraction of intercepted precipitation [dimensionless]
            fwet_exponent: Exponent for wetted canopy fraction [dimensionless]
            clm45_interception_p1: CLM4.5 interception parameter [dimensionless]
            clm45_interception_p2: CLM4.5 interception parameter [dimensionless]
            
        Solar radiation parameters (lines 98-101):
            chil_min: Minimum value for xl leaf angle orientation parameter [dimensionless]
            chil_max: Maximum value for xl leaf angle orientation parameter [dimensionless]
            kb_max: Maximum value for direct beam extinction coefficient [dimensionless]
            j_to_umol: PAR conversion from W/m2 to umol/m2/s [umol/J]
            
        Longwave radiation parameters (lines 105):
            emg: Ground (soil) emissivity [dimensionless]
            
        Roughness sublayer parameters (lines 109-117):
            cd: RSL leaf drag coefficient [dimensionless]
            beta_neutral_max: RSL maximum value for beta in neutral conditions [dimensionless]
            cr: RSL parameter to calculate beta_neutral [dimensionless]
            c2: RSL depth scale multiplier [dimensionless]
            pr0: RSL neutral value for Pr (Sc) [dimensionless]
            pr1: RSL magnitude of variation of Pr (Sc) with stability [dimensionless]
            pr2: RSL scale of variation of Pr (Sc) with stability [dimensionless]
            z0mg: RSL roughness length of ground [m]
            
        Numerical limits (lines 121-128):
            wind_forc_min: Minimum wind speed at forcing height [m/s]
            eta_max: Maximum value for "eta" parameter [dimensionless]
            zeta_min: Minimum value for Monin-Obukhov zeta parameter [dimensionless]
            zeta_max: Maximum value for Monin-Obukhov zeta parameter [dimensionless]
            beta_min: Minimum value for H&F beta parameter [dimensionless]
            beta_max: Maximum value for H&F beta parameter [dimensionless]
            wind_min: Minimum wind speed in canopy [m/s]
            ra_max: Maximum aerodynamic resistance [s/m]
            
        RSL psihat lookup table dimensions (lines 133):
            n_z: Number of zdt grid points for RSL psihat lookup [dimensionless]
            n_l: Number of dtL grid points for RSL psihat lookup [dimensionless]
    
    Notes:
        - Values marked with SPVAL are uninitialized and should be set at runtime
        - Acclimation parameters (vcmaxse_acclim, jmaxse_acclim, etc.) use SPVAL
          when acclimation is not enabled
        - All floating point values use double precision (equivalent to Fortran r8)
        - Integer values (n_z, n_l) are for array dimensioning
    
    References:
        Original Fortran: MLclm_varcon.F90, lines 1-153
        Physical constants: Lines 16-25
        Photosynthesis: Lines 31-68
        Stomatal conductance: Lines 72-74
        Leaf properties: Lines 78-85
        Interception: Lines 89-94
        Radiation: Lines 98-105
        RSL: Lines 109-117
        Numerical limits: Lines 121-128
        Lookup tables: Lines 133
    """
    
    # =========================================================================
    # Physical constants (lines 16-25)
    # =========================================================================
    rgas: float = 8.31446
    mmdry: float = 28.97e-03
    mmh2o: float = 18.02e-03
    cpd: float = 1005.0
    cpw: float = 1846.0
    visc0: float = 13.3e-06
    dh0: float = 18.9e-06
    dv0: float = 21.8e-06
    dc0: float = 13.8e-06
    lapse_rate: float = 0.0098
    
    # =========================================================================
    # Leaf photosynthesis parameters (lines 31-68)
    # =========================================================================
    
    # Michaelis-Menten constants and their temperature dependencies
    kc25: float = 404.9
    kcha: float = 79430.0
    ko25: float = 278.4
    koha: float = 36380.0
    cp25: float = 42.75
    cpha: float = 37830.0
    
    # Vcmax temperature response (with and without acclimation)
    vcmaxha_noacclim: float = 65330.0
    vcmaxha_acclim: float = 72000.0
    vcmaxhd_noacclim: float = 150000.0
    vcmaxhd_acclim: float = 200000.0
    vcmaxse_noacclim: float = 490.0
    vcmaxse_acclim: float = SPVAL  # Set at runtime if acclimation enabled
    
    # Jmax temperature response (with and without acclimation)
    jmaxha_noacclim: float = 43540.0
    jmaxha_acclim: float = 50000.0
    jmaxhd_noacclim: float = 150000.0
    jmaxhd_acclim: float = 200000.0
    jmaxse_noacclim: float = 490.0
    jmaxse_acclim: float = SPVAL  # Set at runtime if acclimation enabled
    
    # Rd (dark respiration) temperature response
    rdha: float = 46390.0
    rdhd: float = 150000.0
    rdse: float = 490.0
    
    # Ratios between photosynthetic parameters at 25C
    jmax25_to_vcmax25_noacclim: float = 1.67
    jmax25_to_vcmax25_acclim: float = SPVAL  # Set at runtime if acclimation enabled
    rd25_to_vcmax25_c3: float = 0.015
    rd25_to_vcmax25_c4: float = 0.025
    kp25_to_vcmax25_c4: float = 0.02
    
    # Quantum yields and curvature parameters
    phi_psii: float = 0.70  # C3 quantum yield of PS II
    theta_j: float = 0.90   # C3 electron transport curvature
    qe_c4: float = 0.05     # C4 quantum yield
    
    # Co-limitation parameters (smooth transitions between limiting rates)
    colim_c3a: float = 0.98   # C3 co-limitation for Ac and Aj
    colim_c3b: float = SPVAL  # C3 co-limitation for Ap (set at runtime)
    colim_c4a: float = 0.80   # C4 co-limitation for Ac and Aj
    colim_c4b: float = 0.95   # C4 co-limitation for Ap
    
    # =========================================================================
    # Stomatal conductance parameters (lines 72-74)
    # =========================================================================
    dh2o_to_dco2: float = 1.6    # Ratio of H2O to CO2 diffusivity
    rh_min_bb: float = 0.2       # Minimum RH for Ball-Berry model
    vpd_min_med: float = 100.0   # Minimum VPD for Medlyn model [Pa]
    
    # =========================================================================
    # Leaf heat capacity parameters (lines 78-80)
    # =========================================================================
    cpbio: float = 4188.0 / 3.0  # Specific heat of dry biomass [J/kg/K]
    fcarbon: float = 0.5         # Carbon fraction of dry biomass [kg C / kg DM]
    fwater: float = 0.7          # Water fraction of fresh biomass [kg H2O / kg FM]
    
    # =========================================================================
    # Leaf boundary layer parameters (lines 84-85)
    # =========================================================================
    gb_factor: float = 1.5  # Empirical correction factor for Nusselt number
    
    # =========================================================================
    # Canopy interception parameters (lines 89-94)
    # =========================================================================
    dewmx: float = 0.1                        # Maximum interception [kg H2O/m2 leaf]
    maximum_leaf_wetted_fraction: float = 0.05  # Max fraction of leaf that can be wet
    interception_fraction: float = 1.0        # Fraction of precip intercepted
    fwet_exponent: float = 0.67               # Exponent for wetted fraction
    clm45_interception_p1: float = 0.25       # CLM4.5 interception parameter
    clm45_interception_p2: float = -0.50      # CLM4.5 interception parameter
    
    # =========================================================================
    # Solar radiation parameters (lines 98-101)
    # =========================================================================
    chil_min: float = -0.4   # Minimum leaf angle orientation parameter
    chil_max: float = 0.6    # Maximum leaf angle orientation parameter
    kb_max: float = 40.0     # Maximum direct beam extinction coefficient
    j_to_umol: float = 4.6   # PAR conversion from W/m2 to umol/m2/s
    
    # =========================================================================
    # Longwave radiation parameters (lines 105)
    # =========================================================================
    emg: float = 0.96  # Ground (soil) emissivity
    
    # =========================================================================
    # Roughness sublayer (RSL) parameters (lines 109-117)
    # =========================================================================
    cd: float = 0.25              # Leaf drag coefficient
    beta_neutral_max: float = 0.35  # Maximum beta in neutral conditions
    cr: float = 0.3               # Parameter to calculate beta_neutral
    c2: float = 0.5               # Depth scale multiplier
    pr0: float = 0.5              # Neutral Prandtl (Schmidt) number
    pr1: float = 0.3              # Magnitude of Pr variation with stability
    pr2: float = 2.0              # Scale of Pr variation with stability
    z0mg: float = 0.01            # Roughness length of ground [m]
    
    # =========================================================================
    # Numerical limits (lines 121-128)
    # =========================================================================
    wind_forc_min: float = 1.0    # Minimum wind speed at forcing height [m/s]
    eta_max: float = 20.0         # Maximum value for "eta" parameter
    zeta_min: float = -2.0        # Minimum Monin-Obukhov zeta
    zeta_max: float = 1.0         # Maximum Monin-Obukhov zeta
    beta_min: float = 0.2         # Minimum Harman & Finnigan beta
    beta_max: float = 0.5         # Maximum Harman & Finnigan beta
    wind_min: float = 0.1         # Minimum wind speed in canopy [m/s]
    ra_max: float = 500.0         # Maximum aerodynamic resistance [s/m]
    
    # =========================================================================
    # RSL psihat lookup table dimensions (lines 133)
    # =========================================================================
    n_z: int = 276  # Number of zdt grid points for psihat lookup
    n_l: int = 41   # Number of dtL grid points for psihat lookup


# =============================================================================
# Default Instance
# =============================================================================

# Global default instance for convenient access throughout the codebase
ML_CANOPY_CONSTANTS = MLCanopyConstants()


# =============================================================================
# Lookup Table Structures
# =============================================================================

class RSLPsihatLookupTables(NamedTuple):
    """Lookup tables for RSL psihat functions.
    
    These tables are used for efficient interpolation of the roughness sublayer
    psihat functions for momentum and heat. They are initialized by a separate
    initialization routine (LookupPsihatINI in the original Fortran).
    
    The psihat functions represent integrated stability corrections for the
    roughness sublayer, accounting for the effects of canopy drag on turbulent
    transport. Separate tables are maintained for momentum and heat/scalar
    transport due to their different behavior.
    
    Attributes (lines 135-142):
        zdtgrid_m: Grid of zdt on which psihat is given for momentum [nZ, 1]
            zdt = (z - d) / dt where z is height, d is displacement height,
            dt is depth scale
        dtlgrid_m: Grid of dtL on which psihat is given for momentum [1, nL]
            dtL = dt / L where L is Obukhov length (stability parameter)
        psigrid_m: Grid of psihat values for momentum [nZ, nL]
            Interpolated values of the integrated stability correction
        zdtgrid_h: Grid of zdt on which psihat is given for heat [nZ, 1]
        dtlgrid_h: Grid of dtL on which psihat is given for heat [1, nL]
        psigrid_h: Grid of psihat values for heat [nZ, nL]
    
    Notes:
        - Tables are 2D arrays with dimensions (n_z, n_l) from MLCanopyConstants
        - Grid arrays (zdtgrid, dtlgrid) define the interpolation coordinates
        - Psihat values (psigrid) are the function values at grid points
        - Separate tables for momentum (_m) and heat (_h) due to different
          Prandtl number effects
        - These tables are populated by initialization code that either reads
          from file or computes the psihat functions numerically
    
    References:
        Original Fortran: MLclm_varcon.F90, lines 135-142
        Initialization: LookupPsihatINI subroutine (separate module)
        Theory: Harman & Finnigan (2007, 2008) roughness sublayer theory
    """
    zdtgrid_m: jnp.ndarray  # shape: (n_z, 1)
    dtlgrid_m: jnp.ndarray  # shape: (1, n_l)
    psigrid_m: jnp.ndarray  # shape: (n_z, n_l)
    zdtgrid_h: jnp.ndarray  # shape: (n_z, 1)
    dtlgrid_h: jnp.ndarray  # shape: (1, n_l)
    psigrid_h: jnp.ndarray  # shape: (n_z, n_l)


# =============================================================================
# Factory Functions
# =============================================================================

def create_empty_rsl_lookup_tables(
    constants: MLCanopyConstants = ML_CANOPY_CONSTANTS,
) -> RSLPsihatLookupTables:
    """Create empty RSL psihat lookup tables with correct dimensions.
    
    These tables will be populated by the initialization routine that reads
    or computes the psihat values. This function provides a convenient way
    to allocate the arrays with the correct shapes before initialization.
    
    Args:
        constants: Multilayer canopy constants containing table dimensions
            (n_z and n_l). Defaults to global ML_CANOPY_CONSTANTS.
        
    Returns:
        Empty lookup tables with correct shapes, initialized to zeros.
        All arrays are JAX arrays for GPU compatibility.
        
    Example:
        >>> tables = create_empty_rsl_lookup_tables()
        >>> tables.psigrid_m.shape
        (276, 41)
        >>> # Later, populate with actual values:
        >>> tables = tables._replace(
        ...     psigrid_m=computed_psihat_momentum,
        ...     psigrid_h=computed_psihat_heat,
        ... )
        
    Note:
        Corresponds to the array declarations in lines 135-142 of the original
        Fortran module. The actual values are set by LookupPsihatINI.
        
        The grid arrays (zdtgrid, dtlgrid) have shapes (n_z, 1) and (1, n_l)
        respectively to facilitate broadcasting during interpolation.
    
    References:
        Original Fortran: MLclm_varcon.F90, lines 135-142
    """
    n_z = constants.n_z
    n_l = constants.n_l
    
    return RSLPsihatLookupTables(
        zdtgrid_m=jnp.zeros((n_z, 1)),
        dtlgrid_m=jnp.zeros((1, n_l)),
        psigrid_m=jnp.zeros((n_z, n_l)),
        zdtgrid_h=jnp.zeros((n_z, 1)),
        dtlgrid_h=jnp.zeros((1, n_l)),
        psigrid_h=jnp.zeros((n_z, n_l)),
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Constants
    "SPVAL",
    "ML_CANOPY_CONSTANTS",
    
    # Types
    "MLCanopyConstants",
    "RSLPsihatLookupTables",
    
    # Functions
    "create_empty_rsl_lookup_tables",
]