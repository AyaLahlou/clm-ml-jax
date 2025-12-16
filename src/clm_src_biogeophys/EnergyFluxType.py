"""
Energy Flux Type Module.

Translated from CTSM's EnergyFluxType.F90

This module defines the data structure for energy flux variables including
sensible heat, latent heat, longwave radiation, and wind stress components.
These fluxes represent energy and momentum exchange between the land surface
and atmosphere.

Key variables:
    - eflx_sh_tot_patch: Total sensible heat flux [W/m2] (positive to atmosphere)
    - eflx_lh_tot_patch: Total latent heat flux [W/m2] (positive to atmosphere)
    - eflx_lwrad_out_patch: Emitted longwave radiation [W/m2]
    - taux_patch: East-west wind stress [kg/m/s^2]
    - tauy_patch: North-south wind stress [kg/m/s^2]

Sign Convention:
    All fluxes are defined positive upward (from surface to atmosphere).
    
Physics:
    - Sensible heat: Direct thermal energy transfer via conduction/convection
    - Latent heat: Energy transfer via water phase changes (evaporation)
    - Longwave radiation: Thermal infrared emission from surface
    - Wind stress: Momentum transfer from surface drag

Fortran Source: EnergyFluxType.F90 (lines 1-73)
"""

from typing import NamedTuple
import jax.numpy as jnp

# Note: float64 support requires JAX x64 mode enabled:
#   jax.config.update("jax_enable_x64", True)
# This should be done in the calling code or conftest.py for tests


# =============================================================================
# Type Definitions
# =============================================================================

class BoundsType(NamedTuple):
    """Domain bounds for grid indexing.
    
    Attributes:
        begp: Beginning patch index
        endp: Ending patch index
        begc: Beginning column index
        endc: Ending column index
        begg: Beginning gridcell index
        endg: Ending gridcell index
    """
    begp: int
    endp: int
    begc: int
    endc: int
    begg: int
    endg: int


class EnergyFluxType(NamedTuple):
    """Energy flux variables for patches.
    
    All arrays are dimensioned [n_patches] representing per-patch values.
    All fluxes follow the convention: positive = upward (surface to atmosphere).
    
    Attributes:
        eflx_sh_tot_patch: Total sensible heat flux [W/m2], positive to atmosphere
            Represents direct thermal energy transfer between surface and air.
            (Fortran line 21)
            
        eflx_lh_tot_patch: Total latent heat flux [W/m2], positive to atmosphere
            Represents energy transfer via evaporation/transpiration.
            (Fortran line 22)
            
        eflx_lwrad_out_patch: Emitted infrared (longwave) radiation [W/m2]
            Thermal radiation emitted by surface following Stefan-Boltzmann law.
            (Fortran line 23)
            
        taux_patch: Wind (shear) stress in east-west direction [kg/m/s^2]
            Zonal momentum flux from surface drag.
            (Fortran line 26)
            
        tauy_patch: Wind (shear) stress in north-south direction [kg/m/s^2]
            Meridional momentum flux from surface drag.
            (Fortran line 27)
    """
    # Energy fluxes (lines 20-23)
    eflx_sh_tot_patch: jnp.ndarray  # [n_patches]
    eflx_lh_tot_patch: jnp.ndarray  # [n_patches]
    eflx_lwrad_out_patch: jnp.ndarray  # [n_patches]
    
    # Momentum fluxes (lines 25-27)
    taux_patch: jnp.ndarray  # [n_patches]
    tauy_patch: jnp.ndarray  # [n_patches]


# =============================================================================
# Initialization Functions
# =============================================================================

def init_allocate(bounds: BoundsType) -> EnergyFluxType:
    """Initialize and allocate energy flux data structure.
    
    Creates arrays for energy flux variables and initializes them to NaN.
    This follows the Fortran pattern of allocating arrays and setting initial
    values, but uses immutable data structures appropriate for JAX.
    
    In the original Fortran, arrays are allocated from begp:endp and initialized
    to NaN to help detect uninitialized values. In JAX, we create arrays of size
    (endp - begp + 1) with 0-based indexing.
    
    Fortran source: EnergyFluxType.F90, lines 50-71
    
    Args:
        bounds: Bounds type containing patch index ranges
            - begp: Beginning patch index (Fortran 1-based)
            - endp: Ending patch index (Fortran 1-based)
            
    Returns:
        Initialized EnergyFluxType with all fields set to NaN
        
    Note:
        NaN initialization helps detect bugs where variables are used before
        being properly set by physics calculations. Any NaN in output indicates
        a missing calculation step.
        
        Uses float64 to match Fortran's r8 (real*8) double precision type.
        
    Example:
        >>> bounds = BoundsType(begp=1, endp=100, begc=1, endc=50, begg=1, endg=10)
        >>> eflux = init_allocate(bounds)
        >>> eflux.eflx_sh_tot_patch.shape
        (100,)
        >>> jnp.all(jnp.isnan(eflux.eflx_sh_tot_patch))
        True
    """
    # Line 63: begp = bounds%begp ; endp = bounds%endp
    begp = bounds.begp
    endp = bounds.endp
    n_patches = endp - begp + 1
    
    # Lines 65-69: Allocate arrays and initialize to NaN
    # Use float64 to match Fortran r8 (real*8) double precision  
    # Note: JAX requires x64 mode enabled for float64. If not enabled, this will use float32.
    # Tests should enable it via: jax.config.update("jax_enable_x64", True)
    # allocate (this%eflx_sh_tot_patch    (begp:endp))  ; this%eflx_sh_tot_patch    (:) = nan
    eflx_sh_tot_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float64)
    
    # allocate (this%eflx_lh_tot_patch    (begp:endp))  ; this%eflx_lh_tot_patch    (:) = nan
    eflx_lh_tot_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float64)
    
    # allocate (this%eflx_lwrad_out_patch (begp:endp))  ; this%eflx_lwrad_out_patch (:) = nan
    eflx_lwrad_out_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float64)
    
    # allocate (this%taux_patch           (begp:endp))  ; this%taux_patch           (:) = nan
    taux_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float64)
    
    # allocate (this%tauy_patch           (begp:endp))  ; this%tauy_patch           (:) = nan
    tauy_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float64)
    
    return EnergyFluxType(
        eflx_sh_tot_patch=eflx_sh_tot_patch,
        eflx_lh_tot_patch=eflx_lh_tot_patch,
        eflx_lwrad_out_patch=eflx_lwrad_out_patch,
        taux_patch=taux_patch,
        tauy_patch=tauy_patch,
    )


def init_energy_flux(bounds: BoundsType) -> EnergyFluxType:
    """Initialize energy flux data structure.
    
    This function allocates and initializes the energy flux type which holds
    variables for sensible heat, latent heat, longwave radiation, and momentum
    fluxes. This is the main entry point for creating an EnergyFluxType instance.
    
    Translated from EnergyFluxType.F90, lines 40-47.
    
    Args:
        bounds: Domain bounds containing grid dimensions
            - begp: Beginning patch index
            - endp: Ending patch index
            - begc: Beginning column index
            - endc: Ending column index
            - begg: Beginning gridcell index
            - endg: Ending gridcell index
            
    Returns:
        Initialized EnergyFluxType structure with allocated arrays
        
    Note:
        In the original Fortran, this calls InitAllocate as a class method
        (this%InitAllocate(bounds)). In JAX, we directly create and return
        the initialized structure using the init_allocate function.
        
        Uses float64 to match Fortran's r8 (real*8) double precision type.
        
    Example:
        >>> bounds = BoundsType(begp=1, endp=1000, begc=1, endc=500, 
        ...                     begg=1, endg=100)
        >>> eflux = init_energy_flux(bounds)
        >>> isinstance(eflux, EnergyFluxType)
        True
    """
    # Call the allocation routine (equivalent to this%InitAllocate(bounds))
    energy_flux = init_allocate(bounds)
    
    return energy_flux


def init_energyflux_type(n_patches: int) -> EnergyFluxType:
    """Initialize EnergyFluxType with zeros.
    
    Convenience function for creating an EnergyFluxType with all fields
    initialized to zero. This is useful for testing or when bounds information
    is not available.
    
    Args:
        n_patches: Number of patches in the domain
        
    Returns:
        Initialized EnergyFluxType with all arrays set to zero
        
    Note:
        This corresponds to the Init and InitAllocate procedures in the
        Fortran module (lines 30-31), but uses zero initialization instead
        of NaN. For production code, prefer init_energy_flux() which uses
        NaN initialization to catch unset values.
        
        FIXED: Explicitly specify dtype=jnp.float64 to match Fortran's r8 
        (real*8) double precision type. This ensures consistency with 
        init_allocate and the original Fortran implementation.
        
    Example:
        >>> eflux = init_energyflux_type(100)
        >>> eflux.eflx_sh_tot_patch.shape
        (100,)
        >>> jnp.all(eflux.eflx_sh_tot_patch == 0.0)
        True
    """
    # Explicitly specify dtype=jnp.float64 to match Fortran r8 (real*8)
    # This resolves the dtype mismatch where default jnp.zeros creates float32
    return EnergyFluxType(
        eflx_sh_tot_patch=jnp.zeros(n_patches, dtype=jnp.float64),
        eflx_lh_tot_patch=jnp.zeros(n_patches, dtype=jnp.float64),
        eflx_lwrad_out_patch=jnp.zeros(n_patches, dtype=jnp.float64),
        taux_patch=jnp.zeros(n_patches, dtype=jnp.float64),
        tauy_patch=jnp.zeros(n_patches, dtype=jnp.float64),
    )


# =============================================================================
# Utility Functions
# =============================================================================

def update_energy_flux(
    eflux: EnergyFluxType,
    **kwargs
) -> EnergyFluxType:
    """Update energy flux fields.
    
    Convenience function for updating one or more fields in an EnergyFluxType
    instance. Since NamedTuples are immutable, this creates a new instance
    with updated values.
    
    Args:
        eflux: Existing EnergyFluxType instance
        **kwargs: Field names and new values to update
        
    Returns:
        New EnergyFluxType with updated fields
        
    Example:
        >>> eflux = init_energyflux_type(10)
        >>> new_sh = jnp.ones(10) * 50.0  # 50 W/m2 sensible heat
        >>> eflux = update_energy_flux(eflux, eflx_sh_tot_patch=new_sh)
        >>> jnp.all(eflux.eflx_sh_tot_patch == 50.0)
        True
    """
    return eflux._replace(**kwargs)


def validate_energy_flux(eflux: EnergyFluxType) -> bool:
    """Validate energy flux values are physical.
    
    Checks that energy flux values are within reasonable physical bounds.
    This is useful for debugging and quality control.
    
    Args:
        eflux: EnergyFluxType instance to validate
        
    Returns:
        True if all values are valid (not NaN, within reasonable bounds)
        
    Note:
        Reasonable bounds (approximate):
        - Sensible heat: -500 to 1000 W/m2
        - Latent heat: -100 to 1000 W/m2
        - Longwave out: 100 to 700 W/m2
        - Wind stress: -10 to 10 kg/m/s^2
    """
    # Check for NaN
    has_nan = (
        jnp.any(jnp.isnan(eflux.eflx_sh_tot_patch)) |
        jnp.any(jnp.isnan(eflux.eflx_lh_tot_patch)) |
        jnp.any(jnp.isnan(eflux.eflx_lwrad_out_patch)) |
        jnp.any(jnp.isnan(eflux.taux_patch)) |
        jnp.any(jnp.isnan(eflux.tauy_patch))
    )
    
    # Check reasonable bounds
    sh_valid = jnp.all((eflux.eflx_sh_tot_patch >= -500.0) & 
                       (eflux.eflx_sh_tot_patch <= 1000.0))
    lh_valid = jnp.all((eflux.eflx_lh_tot_patch >= -100.0) & 
                       (eflux.eflx_lh_tot_patch <= 1000.0))
    lw_valid = jnp.all((eflux.eflx_lwrad_out_patch >= 100.0) & 
                       (eflux.eflx_lwrad_out_patch <= 700.0))
    tau_valid = jnp.all((jnp.abs(eflux.taux_patch) <= 10.0) & 
                        (jnp.abs(eflux.tauy_patch) <= 10.0))
    
    return ~has_nan & sh_valid & lh_valid & lw_valid & tau_valid