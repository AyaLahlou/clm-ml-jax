"""
Soil Surface Temperature and Energy Balance Calculations.

Translated from CTSM's MLSoilFluxesMod.F90

This module calculates soil surface temperature and energy balance for the
multilayer canopy model. It handles the coupling between the soil surface
and the atmosphere through energy and water vapor fluxes.

Key physics:
    - Soil surface energy balance
    - Ground heat flux
    - Sensible and latent heat fluxes at soil surface
    - Soil surface temperature iteration

The energy balance at the soil surface is:
    Rnet = H + LE + G

Where:
    - Rnet: Net radiation at soil surface [W/m2]
    - H: Sensible heat flux [W/m2]
    - LE: Latent heat flux [W/m2]
    - G: Soil heat flux into ground [W/m2]

The soil surface temperature is solved by linearizing the energy balance
equation around the previous timestep temperature:

    tg = (num1 * tair + num2 * (eair/pref) + num4) / den

Where:
    num1 = cpair * gac0
    num2 = lambda * gw
    num3 = soil_tk / soil_dz
    num4 = rnsoi - num2 * rhg * (qsat - dqsat * tg_bef) + num3 * soil_t
    den = num1 + num2 * dqsat * rhg + num3

References:
    MLSoilFluxesMod.F90:1-122
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from .MLWaterVaporMod import sat_vap, lat_vap


# =============================================================================
# Type Definitions
# =============================================================================


class SoilFluxesInput(NamedTuple):
    """Input state for soil surface energy balance.
    
    Attributes:
        tref: Air temperature at reference height [K] [n_patches]
        pref: Air pressure at reference height [Pa] [n_patches]
        rhomol: Molar density at reference height [mol/m3] [n_patches]
        cpair: Specific heat of air at constant pressure [J/mol/K] [n_patches]
        rnsoi: Net radiation at ground surface [W/m2] [n_patches]
        rhg: Relative humidity at soil surface [fraction] [n_patches]
        soilres: Soil evaporative resistance [s/m] [n_patches]
        gac0: Aerodynamic conductance for soil fluxes [mol/m2/s] [n_patches]
        soil_t: Temperature of first snow/soil layer [K] [n_patches]
        soil_dz: Depth to temperature of first snow/soil layer [m] [n_patches]
        soil_tk: Thermal conductivity of first snow/soil layer [W/m/K] [n_patches]
        tg_bef: Soil surface temperature from previous timestep [K] [n_patches]
        tair: Canopy layer air temperature [K] [n_patches, n_layers]
        eair: Canopy layer vapor pressure [Pa] [n_patches, n_layers]
        
    Note:
        Reference: MLSoilFluxesMod.F90:22-122
    """
    tref: jnp.ndarray
    pref: jnp.ndarray
    rhomol: jnp.ndarray
    cpair: jnp.ndarray
    rnsoi: jnp.ndarray
    rhg: jnp.ndarray
    soilres: jnp.ndarray
    gac0: jnp.ndarray
    soil_t: jnp.ndarray
    soil_dz: jnp.ndarray
    soil_tk: jnp.ndarray
    tg_bef: jnp.ndarray
    tair: jnp.ndarray
    eair: jnp.ndarray


class SoilFluxesOutput(NamedTuple):
    """Output state from soil surface energy balance.
    
    Attributes:
        shsoi: Sensible heat flux from ground [W/m2] [n_patches]
        lhsoi: Latent heat flux from ground [W/m2] [n_patches]
        gsoi: Soil heat flux into ground [W/m2] [n_patches]
        etsoi: Water vapor flux from ground [mol H2O/m2/s] [n_patches]
        tg: Soil surface temperature [K] [n_patches]
        eg: Soil surface vapor pressure [Pa] [n_patches]
        energy_error: Energy balance error [W/m2] [n_patches]
        
    Note:
        Reference: MLSoilFluxesMod.F90:22-122
    """
    shsoi: jnp.ndarray
    lhsoi: jnp.ndarray
    gsoi: jnp.ndarray
    etsoi: jnp.ndarray
    tg: jnp.ndarray
    eg: jnp.ndarray
    energy_error: jnp.ndarray


# =============================================================================
# Main Physics Functions
# =============================================================================


def soil_fluxes(inputs: SoilFluxesInput) -> SoilFluxesOutput:
    """Calculate soil surface temperature and energy balance.
    
    Solves the energy balance equation at the soil surface to determine
    the soil surface temperature and associated energy fluxes. The energy
    balance is:
        Rnet = H + LE + G
        
    where Rnet is net radiation, H is sensible heat, LE is latent heat,
    and G is soil heat flux.
    
    The solution uses a linearization of the saturation vapor pressure
    around the previous timestep temperature to avoid iteration:
    
        esat(tg) ≈ esat(tg_bef) + desat * (tg - tg_bef)
    
    This allows the energy balance to be solved directly for tg.
    
    Physical processes:
        1. Calculate soil conductance to water vapor (lines 70-73)
        2. Get saturation vapor pressure at previous temperature (lines 75-77)
        3. Solve linearized energy balance for new temperature (lines 79-86)
        4. Calculate sensible heat flux (lines 88-89)
        5. Calculate latent heat flux (lines 91-93)
        6. Calculate soil heat flux (lines 95-96)
        7. Check energy balance closure (lines 98-102)
    
    Reference: MLSoilFluxesMod.F90, lines 22-122
    
    Args:
        inputs: Input state containing forcing and soil properties
        
    Returns:
        Output state containing soil surface temperature and energy fluxes
        
    Note:
        - Uses first canopy layer (index 1 in Fortran, 0 in Python) for
          air temperature and vapor pressure (lines 86, 92)
        - Energy balance error should be < 0.001 W/m2 (line 107)
        - All fluxes are positive upward (away from surface)
    """
    # Latent heat of vaporization (line 68)
    # Convert from J/kg to J/mol for consistency with molar fluxes
    lambda_vap = lat_vap(inputs.tref)
    
    # Soil conductance to water vapor diffusion (lines 70-73)
    # The total conductance is the series combination of:
    # 1. Aerodynamic conductance (gac0)
    # 2. Soil surface resistance converted to conductance (1/soilres * rhomol)
    gws = 1.0 / inputs.soilres  # Convert resistance [s/m] to conductance [m/s]
    gws = gws * inputs.rhomol  # Convert [m/s] to [mol/m2/s]
    gw = inputs.gac0 * gws / (inputs.gac0 + gws)  # Series combination
    
    # Saturation vapor pressure at previous ground temperature (lines 75-77)
    # Linearize around previous temperature to avoid iteration
    esat, desat = sat_vap(inputs.tg_bef)
    qsat = esat / inputs.pref  # Convert to mole fraction
    dqsat = desat / inputs.pref  # Derivative w.r.t. temperature
    
    # Calculate soil surface temperature (lines 79-86)
    # Solve linearized energy balance equation:
    # cpair * gac0 * (tg - tair) + lambda * gw * (eg - eair)/pref + 
    # soil_tk/soil_dz * (tg - soil_t) = rnsoi
    #
    # Where eg = rhg * (esat + desat * (tg - tg_bef))
    #
    # Rearranging to solve for tg:
    num1 = inputs.cpair * inputs.gac0
    num2 = lambda_vap * gw
    num3 = inputs.soil_tk / inputs.soil_dz
    num4 = (inputs.rnsoi - 
            num2 * inputs.rhg * (qsat - dqsat * inputs.tg_bef) + 
            num3 * inputs.soil_t)
    den = num1 + num2 * dqsat * inputs.rhg + num3
    
    # Use first canopy layer (index 0 in Python, index 1 in Fortran)
    # This is the layer immediately above the soil surface
    tair_bottom = inputs.tair[..., 0]
    eair_bottom = inputs.eair[..., 0]
    
    # Solve for soil surface temperature
    tg = (num1 * tair_bottom + num2 * (eair_bottom / inputs.pref) + num4) / den
    
    # Sensible heat flux (lines 88-89)
    # H = cpair * (tg - tair) * gac0
    # Positive upward (from surface to atmosphere)
    shsoi = inputs.cpair * (tg - tair_bottom) * inputs.gac0
    
    # Latent heat flux (lines 91-93)
    # First calculate actual vapor pressure at soil surface
    # using linearized saturation vapor pressure
    eg = inputs.rhg * (esat + desat * (tg - inputs.tg_bef))
    
    # LE = lambda * (eg - eair) / pref * gw
    # Positive upward (evaporation from surface)
    lhsoi = lambda_vap * (eg - eair_bottom) / inputs.pref * gw
    
    # Soil heat flux (lines 95-96)
    # G = soil_tk * (tg - soil_t) / soil_dz
    # Positive downward into soil (but stored as positive upward convention)
    gsoi = inputs.soil_tk * (tg - inputs.soil_t) / inputs.soil_dz
    
    # Energy balance error check (lines 98-102)
    # Should be near zero if energy is conserved
    # err = Rnet - H - LE - G
    err = inputs.rnsoi - shsoi - lhsoi - gsoi
    
    # Water vapor flux (lines 104-105)
    # Convert from energy flux [W/m2] to molar flux [mol H2O/m2/s]
    etsoi = lhsoi / lambda_vap
    
    return SoilFluxesOutput(
        shsoi=shsoi,
        lhsoi=lhsoi,
        gsoi=gsoi,
        etsoi=etsoi,
        tg=tg,
        eg=eg,
        energy_error=err,
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    'SoilFluxesInput',
    'SoilFluxesOutput',
    'soil_fluxes',
]