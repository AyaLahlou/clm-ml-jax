"""
Leaf Temperature and Energy Fluxes Module.

Translated from CTSM's MLLeafFluxesMod.F90

This module provides functions for calculating leaf-level energy fluxes
and leaf temperature in a multilayer canopy model. It handles the coupling
between leaf temperature, stomatal conductance, and energy balance.

Key processes:
    - Leaf energy balance (radiation, sensible, latent heat)
    - Leaf temperature calculation
    - Stomatal conductance coupling
    - Sunlit/shaded leaf separation

Physics:
    Energy balance: Rn = H + λE + S
    Where:
        - Rn: Net radiation absorbed by leaf [W/m2]
        - H: Sensible heat flux [W/m2]
        - λE: Latent heat flux [W/m2]
        - S: Storage heat flux [W/m2]

Key equations (lines 95-98):
    T_leaf = (num1 * T_air + num2 * e_air/P + num3) / den
    
Where:
    num1 = 2 * cp * gbh
    num2 = λ * gw
    num3 = Rn - λ * gw * (qsat - dqsat * T_leaf_prev) + C_leaf/dt * T_leaf_prev
    den = C_leaf/dt + num1 + num2 * dqsat

The energy balance is verified to ensure conservation (lines 117-120).

Reference:
    MLLeafFluxesMod.F90 (lines 1-148)
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp

# Import water vapor functions from translated module
# Note: These should be available from the MLWaterVaporMod translation
from multilayer_canopy.MLWaterVaporMod import sat_vap, lat_vap


# =============================================================================
# Type Definitions
# =============================================================================

class LeafFluxesResult(NamedTuple):
    """Results from leaf flux calculations.
    
    All fluxes are per unit leaf area unless otherwise specified.
    
    Attributes:
        tleaf: Leaf temperature [K]
        stleaf: Leaf storage heat flux [W/m2 leaf]
        shleaf: Leaf sensible heat flux [W/m2 leaf]
        lhleaf: Leaf latent heat flux [W/m2 leaf]
        evleaf: Leaf evaporation flux [mol H2O/m2 leaf/s]
        trleaf: Leaf transpiration flux [mol H2O/m2 leaf/s]
        energy_balance_error: Energy balance error for diagnostics [W/m2 leaf]
    
    Note:
        - Positive fluxes indicate energy/water leaving the leaf
        - Energy balance error should be < 1e-3 W/m2 for accurate solutions
        - When dpai <= 0, all fluxes are zero and tleaf = tair
    """
    tleaf: jnp.ndarray
    stleaf: jnp.ndarray
    shleaf: jnp.ndarray
    lhleaf: jnp.ndarray
    evleaf: jnp.ndarray
    trleaf: jnp.ndarray
    energy_balance_error: jnp.ndarray


# =============================================================================
# Constants
# =============================================================================

# Module-level constants matching Fortran's shr_kind_mod
# In Fortran: use shr_kind_mod, only : r8 => shr_kind_r8
# JAX uses float64 by default for r8 precision
R8_DTYPE = jnp.float64

# Energy balance tolerance [W/m2]
# Used for checking energy conservation (line 117)
ENERGY_BALANCE_TOL = 1e-3


# =============================================================================
# Main Functions
# =============================================================================

def leaf_fluxes(
    dtime_substep: float,
    tref: float,
    pref: float,
    cpair: float,
    dpai: float,
    tair: float,
    eair: float,
    cpleaf: float,
    fwet: float,
    fdry: float,
    gbh: float,
    gbv: float,
    gs: float,
    rnleaf: float,
    tleaf_bef: float,
) -> LeafFluxesResult:
    """Calculate leaf temperature and energy fluxes.
    
    Solves the linearized leaf energy balance equation to determine leaf
    temperature, then calculates sensible heat, latent heat, and storage
    heat fluxes. Handles both wet and dry leaf fractions.
    
    The solution uses a linearized approach where the saturation vapor
    pressure is approximated as:
        qsat(T) ≈ qsat(T_prev) + dqsat * (T - T_prev)
    
    This allows the energy balance to be solved analytically for leaf
    temperature, avoiding iterative solution methods.
    
    Args:
        dtime_substep: Model time step [s]
        tref: Air temperature at reference height [K]
        pref: Air pressure at reference height [Pa]
        cpair: Specific heat of air at constant pressure [J/mol/K]
        dpai: Canopy layer plant area index [m2/m2]
        tair: Canopy layer air temperature [K]
        eair: Canopy layer vapor pressure [Pa]
        cpleaf: Canopy layer leaf heat capacity [J/m2 leaf/K]
        fwet: Fraction of plant area index that is wet [0-1]
        fdry: Fraction of plant area index that is green and dry [0-1]
        gbh: Leaf boundary layer conductance for heat [mol/m2 leaf/s]
        gbv: Leaf boundary layer conductance for H2O [mol H2O/m2 leaf/s]
        gs: Leaf stomatal conductance [mol H2O/m2 leaf/s]
        rnleaf: Leaf net radiation [W/m2 leaf]
        tleaf_bef: Leaf temperature from previous timestep [K]
        
    Returns:
        LeafFluxesResult containing leaf temperature and energy fluxes
        
    Note:
        - When dpai <= 0, all fluxes are set to zero and tleaf = tair
        - Energy balance is checked with tolerance of 1e-3 W/m2
        - Wet fraction uses boundary layer conductance only
        - Dry fraction uses coupled stomatal-boundary layer conductance
        - Lines 22-146 from MLLeafFluxesMod.F90
        
    Physics:
        The leaf energy balance is:
            Rn = H + λE + S
        
        Where:
            H = 2 * cp * gbh * (T_leaf - T_air)
            λE = λ * gw * (qsat(T_leaf) - q_air)
            S = C_leaf/dt * (T_leaf - T_leaf_prev)
            
        The factor of 2 in sensible heat accounts for heat transfer
        from both sides of the leaf.
    """
    # Line 67: Latent heat of vaporization at reference temperature
    lambda_vap = lat_vap(tref)
    
    # Line 69: Check if there is plant area
    # This determines whether to calculate fluxes or return zeros
    has_vegetation = dpai > 0.0
    
    # Lines 71-73: Saturation vapor pressure and derivative
    # Convert from Pa to mol/mol by dividing by pressure
    esat, desat = sat_vap(tleaf_bef)
    qsat = esat / pref
    dqsat = desat / pref
    
    # Lines 75-76: Leaf conductance for transpiration
    # Series conductance: 1/gleaf = 1/gs + 1/gbv
    gleaf = gs * gbv / (gs + gbv)
    
    # Lines 78-79: Total conductance for water vapor
    # Dry fraction: coupled stomatal-boundary layer conductance
    # Wet fraction: boundary layer conductance only (no stomatal control)
    gw = gleaf * fdry + gbv * fwet
    
    # Lines 81-86: Linearized leaf temperature calculation
    # Solve energy balance for T_leaf using linearized qsat
    num1 = 2.0 * cpair * gbh
    num2 = lambda_vap * gw
    num3 = (rnleaf - lambda_vap * gw * (qsat - dqsat * tleaf_bef) + 
            cpleaf / dtime_substep * tleaf_bef)
    den = cpleaf / dtime_substep + num1 + num2 * dqsat
    tleaf_calc = (num1 * tair + num2 * eair / pref + num3) / den
    
    # Lines 88-89: Storage heat flux
    # Change in leaf heat content over timestep
    stleaf_calc = (tleaf_calc - tleaf_bef) * cpleaf / dtime_substep
    
    # Lines 91-92: Sensible heat flux
    # Factor of 2 accounts for both sides of leaf
    shleaf_calc = 2.0 * cpair * (tleaf_calc - tair) * gbh
    
    # Lines 94-97: Transpiration and evaporation water fluxes
    # Update saturation vapor pressure using linearization
    num1_flux = qsat + dqsat * (tleaf_calc - tleaf_bef) - eair / pref
    trleaf_calc = gleaf * fdry * num1_flux
    evleaf_calc = gbv * fwet * num1_flux
    
    # Lines 99-100: Latent heat flux
    # Sum of transpiration and evaporation
    lhleaf_calc = (trleaf_calc + evleaf_calc) * lambda_vap
    
    # Lines 102-106: Energy balance error check
    # Should be near zero if solution is accurate
    err = rnleaf - shleaf_calc - lhleaf_calc - stleaf_calc
    
    # Lines 108-115: Handle case with no vegetation
    # Use jnp.where to maintain JIT compatibility (no Python if statements)
    tleaf = jnp.where(has_vegetation, tleaf_calc, tair)
    stleaf = jnp.where(has_vegetation, stleaf_calc, 0.0)
    shleaf = jnp.where(has_vegetation, shleaf_calc, 0.0)
    lhleaf = jnp.where(has_vegetation, lhleaf_calc, 0.0)
    evleaf = jnp.where(has_vegetation, evleaf_calc, 0.0)
    trleaf = jnp.where(has_vegetation, trleaf_calc, 0.0)
    energy_balance_error = jnp.where(has_vegetation, err, 0.0)
    
    return LeafFluxesResult(
        tleaf=tleaf,
        stleaf=stleaf,
        shleaf=shleaf,
        lhleaf=lhleaf,
        evleaf=evleaf,
        trleaf=trleaf,
        energy_balance_error=energy_balance_error,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def check_energy_balance(
    result: LeafFluxesResult,
    rnleaf: float,
    tolerance: float = ENERGY_BALANCE_TOL,
) -> bool:
    """Check if energy balance is satisfied within tolerance.
    
    Verifies that: Rn = H + λE + S
    
    Args:
        result: LeafFluxesResult from leaf_fluxes calculation
        rnleaf: Net radiation [W/m2 leaf]
        tolerance: Maximum allowed error [W/m2]
        
    Returns:
        True if energy balance error is within tolerance
        
    Note:
        This is a diagnostic function for testing and validation.
        The energy balance error is already computed in leaf_fluxes.
    """
    return jnp.abs(result.energy_balance_error) < tolerance


def total_leaf_conductance(
    gs: float,
    gbv: float,
    gbh: float,
    fdry: float,
    fwet: float,
) -> tuple[float, float]:
    """Calculate total leaf conductances for water vapor and heat.
    
    Combines stomatal and boundary layer conductances accounting for
    wet and dry leaf fractions.
    
    Args:
        gs: Stomatal conductance [mol H2O/m2 leaf/s]
        gbv: Boundary layer conductance for H2O [mol H2O/m2 leaf/s]
        gbh: Boundary layer conductance for heat [mol/m2 leaf/s]
        fdry: Fraction of leaf that is dry [0-1]
        fwet: Fraction of leaf that is wet [0-1]
        
    Returns:
        Tuple of (gw, gh) where:
            gw: Total conductance for water vapor [mol H2O/m2 leaf/s]
            gh: Total conductance for heat [mol/m2 leaf/s]
            
    Note:
        - Dry fraction: series conductance of stomata and boundary layer
        - Wet fraction: boundary layer conductance only
        - Heat conductance is always boundary layer only
    """
    # Series conductance for dry fraction
    gleaf = gs * gbv / (gs + gbv)
    
    # Total water vapor conductance
    gw = gleaf * fdry + gbv * fwet
    
    # Heat conductance (boundary layer only)
    gh = gbh
    
    return gw, gh


# =============================================================================
# Module Metadata
# =============================================================================

__module_name__ = "MLLeafFluxesMod"
__fortran_source__ = "MLLeafFluxesMod.F90"
__fortran_lines__ = "1-148"
__version__ = "1.0.0"

# Public interface (matching Fortran's public :: LeafFluxes)
__all__ = [
    "leaf_fluxes",
    "LeafFluxesResult",
    "check_energy_balance",
    "total_leaf_conductance",
    "R8_DTYPE",
    "ENERGY_BALANCE_TOL",
]