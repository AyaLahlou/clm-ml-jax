"""
Canopy Water Module.

Translated from CTSM's MLCanopyWaterMod.F90 (lines 1-249)

This module handles canopy water processes including:
- Interception of precipitation by canopy
- Throughfall to ground surface
- Evaporation of intercepted water
- Dew formation on canopy surfaces

The module provides functions for updating canopy water storage and fluxes
in the multilayer canopy model.

Key processes:
    1. CanopyInterception: Calculates interception and throughfall
       - Distributes intercepted water across canopy layers
       - Calculates wetted and dry fractions
       - Handles canopy drip when storage exceeds capacity
       
    2. CanopyEvaporation: Updates canopy water for evaporation/condensation
       - Processes evaporation from intercepted water
       - Handles dew formation (negative fluxes)
       - Separate calculations for sunlit and shaded leaves

Physics equations:
    Interception fraction:
        CLM4.5: fpi = p1 * (1 - exp(p2*(LAI+SAI)))
        CLM5:   fpi = p * tanh(LAI+SAI)
    
    Wetted fraction:
        fwet = (h2ocan/h2ocanmx)^exponent
    
    Maximum canopy water:
        h2ocanmx = dewmx * plant_area_index
    
    Evaporation/Dew:
        For sunlit:   dew = (evleaf + trleaf) * fracsun * dpai * mmh2o * dt
        For shaded:   dew = (evleaf + trleaf) * (1-fracsun) * dpai * mmh2o * dt

References:
    MLCanopyWaterMod.F90 lines 1-249
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp


# =============================================================================
# Type Definitions
# =============================================================================


class CanopyInterceptionParams(NamedTuple):
    """Parameters for canopy interception.
    
    Attributes:
        dewmx: Maximum water storage per unit PAI [kg H2O/m2 PAI]
        maximum_leaf_wetted_fraction: Maximum wetted fraction [0-1]
        interception_fraction: Interception fraction coefficient (CLM5) [-]
        fwet_exponent: Exponent for wetted fraction calculation [-]
        clm45_interception_p1: CLM4.5 interception parameter 1 [-]
        clm45_interception_p2: CLM4.5 interception parameter 2 [-]
        fpi_type: Interception formulation type (1=CLM4.5, 2=CLM5) [-]
        dtime_substep: Model time step [s]
    
    Reference: MLCanopyWaterMod.F90:23-171, MLclm_varcon.F90, MLclm_varctl.F90
    """
    dewmx: float
    maximum_leaf_wetted_fraction: float
    interception_fraction: float
    fwet_exponent: float
    clm45_interception_p1: float
    clm45_interception_p2: float
    fpi_type: int
    dtime_substep: float


class CanopyWaterState(NamedTuple):
    """State variables for canopy water.
    
    Attributes:
        h2ocan_profile: Canopy layer intercepted water [kg H2O/m2] [n_patches, n_layers]
        qflx_intr_canopy: Intercepted precipitation [kg H2O/m2/s] [n_patches]
        qflx_tflrain_canopy: Total rain throughfall onto ground [kg H2O/m2/s] [n_patches]
        qflx_tflsnow_canopy: Total snow throughfall onto ground [kg H2O/m2/s] [n_patches]
        fwet_profile: Canopy layer fraction of PAI that is wet [0-1] [n_patches, n_layers]
        fdry_profile: Canopy layer fraction of PAI that is green and dry [0-1] [n_patches, n_layers]
    
    Reference: MLCanopyWaterMod.F90:23-171
    """
    h2ocan_profile: jnp.ndarray
    qflx_intr_canopy: jnp.ndarray
    qflx_tflrain_canopy: jnp.ndarray
    qflx_tflsnow_canopy: jnp.ndarray
    fwet_profile: jnp.ndarray
    fdry_profile: jnp.ndarray


class CanopyEvaporationInput(NamedTuple):
    """Input state for canopy evaporation calculations.
    
    Attributes:
        ncan: Number of aboveground layers [n_patches]
        dpai: Canopy layer plant area index [m2/m2] [n_patches, n_layers]
        fracsun: Canopy layer sunlit fraction [-] [n_patches, n_layers]
        trleaf: Leaf transpiration flux [mol H2O/m2 leaf/s] [n_patches, n_layers, 2]
            where last dimension is [isun=0, isha=1]
        evleaf: Leaf evaporation flux [mol H2O/m2 leaf/s] [n_patches, n_layers, 2]
            where last dimension is [isun=0, isha=1]
        h2ocan: Canopy layer intercepted water [kg H2O/m2] [n_patches, n_layers]
        mmh2o: Molecular weight of water [kg/mol] (scalar)
        dtime_substep: Model time step [s] (scalar)
    
    Reference: MLCanopyWaterMod.F90:174-249
    """
    ncan: jnp.ndarray  # [n_patches]
    dpai: jnp.ndarray  # [n_patches, n_layers]
    fracsun: jnp.ndarray  # [n_patches, n_layers]
    trleaf: jnp.ndarray  # [n_patches, n_layers, 2]
    evleaf: jnp.ndarray  # [n_patches, n_layers, 2]
    h2ocan: jnp.ndarray  # [n_patches, n_layers]
    mmh2o: float
    dtime_substep: float


class CanopyEvaporationOutput(NamedTuple):
    """Output state from canopy evaporation calculations.
    
    Attributes:
        h2ocan: Updated canopy layer intercepted water [kg H2O/m2] [n_patches, n_layers]
    
    Reference: MLCanopyWaterMod.F90:174-249
    """
    h2ocan: jnp.ndarray  # [n_patches, n_layers]


# =============================================================================
# Default Parameters
# =============================================================================


def get_default_interception_params(
    dewmx: float = 0.1,
    maximum_leaf_wetted_fraction: float = 0.05,
    interception_fraction: float = 0.25,
    fwet_exponent: float = 0.667,
    clm45_interception_p1: float = 0.25,
    clm45_interception_p2: float = -0.50,
    fpi_type: int = 2,
    dtime_substep: float = 1800.0,
) -> CanopyInterceptionParams:
    """Create default canopy interception parameters.
    
    Args:
        dewmx: Maximum water storage per unit PAI [kg H2O/m2 PAI]
        maximum_leaf_wetted_fraction: Maximum wetted fraction [0-1]
        interception_fraction: Interception fraction coefficient (CLM5) [-]
        fwet_exponent: Exponent for wetted fraction calculation [-]
        clm45_interception_p1: CLM4.5 interception parameter 1 [-]
        clm45_interception_p2: CLM4.5 interception parameter 2 [-]
        fpi_type: Interception formulation type (1=CLM4.5, 2=CLM5) [-]
        dtime_substep: Model time step [s]
        
    Returns:
        CanopyInterceptionParams with default values
        
    Reference: MLclm_varcon.F90, MLclm_varctl.F90
    """
    return CanopyInterceptionParams(
        dewmx=dewmx,
        maximum_leaf_wetted_fraction=maximum_leaf_wetted_fraction,
        interception_fraction=interception_fraction,
        fwet_exponent=fwet_exponent,
        clm45_interception_p1=clm45_interception_p1,
        clm45_interception_p2=clm45_interception_p2,
        fpi_type=fpi_type,
        dtime_substep=dtime_substep,
    )


# =============================================================================
# Constants
# =============================================================================

# Sun/shade indices (MLclm_varpar.F90)
ISUN = 0  # Sunlit leaf index
ISHA = 1  # Shaded leaf index

# Molecular weight of water (MLclm_varcon.F90)
MMH2O_DEFAULT = 0.018015  # [kg/mol]


# =============================================================================
# Main Functions
# =============================================================================


def canopy_interception(
    qflx_rain: jnp.ndarray,
    qflx_snow: jnp.ndarray,
    lai: jnp.ndarray,
    sai: jnp.ndarray,
    ncan: jnp.ndarray,
    dlai_profile: jnp.ndarray,
    dpai_profile: jnp.ndarray,
    h2ocan_profile: jnp.ndarray,
    params: CanopyInterceptionParams,
) -> CanopyWaterState:
    """Calculate canopy interception and throughfall.
    
    Distributes intercepted precipitation across canopy layers, calculates
    wetted fractions, and determines throughfall to the ground.
    
    Algorithm:
        1. Calculate total precipitation and rain/snow fractions
        2. Determine interception fraction (fpi) using CLM4.5 or CLM5 formulation
        3. Calculate direct throughfall: (1-fpi) * precipitation
        4. Distribute intercepted water equally across layers with PAI > 0
        5. Calculate canopy drip when storage exceeds capacity
        6. Update wetted and dry fractions based on water content
        7. Sum total throughfall (direct + drip)
    
    Args:
        qflx_rain: Rainfall [kg H2O/m2/s] [n_patches]
        qflx_snow: Snowfall [kg H2O/m2/s] [n_patches]
        lai: Leaf area index of canopy [m2/m2] [n_patches]
        sai: Stem area index of canopy [m2/m2] [n_patches]
        ncan: Number of aboveground layers [-] [n_patches]
        dlai_profile: Canopy layer leaf area index [m2/m2] [n_patches, n_layers]
        dpai_profile: Canopy layer plant area index [m2/m2] [n_patches, n_layers]
        h2ocan_profile: Canopy layer intercepted water [kg H2O/m2] [n_patches, n_layers]
        params: Canopy interception parameters
        
    Returns:
        CanopyWaterState with updated water states and fluxes
        
    Reference: MLCanopyWaterMod.F90:23-171
    """
    n_patches = qflx_rain.shape[0]
    n_layers = dpai_profile.shape[1]
    
    # Total precipitation (line 77)
    qflx_total = qflx_snow + qflx_rain
    
    # Fraction of precipitation that is rain and snow (lines 79-85)
    has_precip = qflx_total > 0.0
    fracrain = jnp.where(has_precip, qflx_rain / qflx_total, 0.0)
    fracsnow = jnp.where(has_precip, qflx_snow / qflx_total, 0.0)
    
    # Fraction of precipitation that is intercepted (lines 87-94)
    pai_total = lai + sai
    
    # CLM4.5 formulation (line 89)
    fpi_clm45 = params.clm45_interception_p1 * (
        1.0 - jnp.exp(params.clm45_interception_p2 * pai_total)
    )
    
    # CLM5 formulation (line 91)
    fpi_clm5 = params.interception_fraction * jnp.tanh(pai_total)
    
    # Select based on fpi_type (lines 88-93)
    fpi = jnp.where(params.fpi_type == 1, fpi_clm45, fpi_clm5)
    
    # Direct throughfall (lines 96-97)
    qflx_through_rain = qflx_rain * (1.0 - fpi)
    qflx_through_snow = qflx_snow * (1.0 - fpi)
    
    # Intercepted precipitation (line 99)
    qflx_intr = qflx_total * fpi
    
    # Count number of layers with PAI > 0 (lines 101-105)
    has_pai = dpai_profile > 0.0
    n_layers_with_pai = jnp.sum(has_pai, axis=1)  # [n_patches]
    
    # Avoid division by zero
    n_layers_with_pai = jnp.maximum(n_layers_with_pai, 1.0)
    
    # Process each layer (lines 107-149)
    def process_layer(carry: tuple, ic: int) -> tuple:
        """Process a single canopy layer.
        
        Args:
            carry: Tuple of (h2ocan_updated, qflx_candrip_accum)
            ic: Layer index
            
        Returns:
            Updated carry tuple and None for scan output
            
        Reference: MLCanopyWaterMod.F90:107-149
        """
        h2ocan_updated, qflx_candrip_accum = carry
        
        # Only process layer if it has PAI (lines 111, 141-145)
        layer_has_pai = dpai_profile[:, ic] > 0.0
        
        # Maximum external water held in layer (line 113)
        h2ocanmx = params.dewmx * dpai_profile[:, ic]
        
        # Water storage of intercepted precipitation (lines 115-117)
        # Intercepted water is applied equally to all layers with PAI
        h2ocan_new = h2ocan_updated[:, ic] + jnp.where(
            layer_has_pai,
            qflx_intr * params.dtime_substep / n_layers_with_pai,
            0.0
        )
        
        # Excess water that exceeds the maximum capacity (lines 119-124)
        xrun = (h2ocan_new - h2ocanmx) / params.dtime_substep
        has_excess = jnp.logical_and(xrun > 0.0, layer_has_pai)
        qflx_candrip_layer = jnp.where(has_excess, xrun, 0.0)
        h2ocan_new = jnp.where(has_excess, h2ocanmx, h2ocan_new)
        
        # Accumulate canopy drip (only from layers with PAI)
        qflx_candrip_accum = qflx_candrip_accum + qflx_candrip_layer
        
        # Ensure zero for layers without PAI
        h2ocan_new = jnp.where(layer_has_pai, h2ocan_new, 0.0)
        
        # Update h2ocan array
        h2ocan_updated = h2ocan_updated.at[:, ic].set(h2ocan_new)
        
        return (h2ocan_updated, qflx_candrip_accum), None
    
    # Initialize carry
    h2ocan_updated = h2ocan_profile.copy()
    qflx_candrip = jnp.zeros(n_patches)
    
    # Loop through layers (line 108)
    (h2ocan_updated, qflx_candrip), _ = jax.lax.scan(
        process_layer,
        (h2ocan_updated, qflx_candrip),
        jnp.arange(n_layers)
    )
    
    # Calculate wetted and dry fractions (lines 126-139)
    h2ocanmx_all = params.dewmx * dpai_profile
    
    # Wetted fraction of canopy (lines 126-128)
    fwet = jnp.maximum(h2ocan_updated / jnp.maximum(h2ocanmx_all, 1e-10), 0.0) ** params.fwet_exponent
    fwet = jnp.minimum(fwet, params.maximum_leaf_wetted_fraction)
    
    # Fraction of canopy that is green and dry (line 130)
    fdry = (1.0 - fwet) * (dlai_profile / jnp.maximum(dpai_profile, 1e-10))
    
    # Zero out for layers without PAI (lines 141-145)
    layer_has_pai = dpai_profile > 0.0
    fwet = jnp.where(layer_has_pai, fwet, 0.0)
    fdry = jnp.where(layer_has_pai, fdry, 0.0)
    
    # Total throughfall onto ground (lines 151-152)
    qflx_tflrain = qflx_through_rain + qflx_candrip * fracrain
    qflx_tflsnow = qflx_through_snow + qflx_candrip * fracsnow
    
    return CanopyWaterState(
        h2ocan_profile=h2ocan_updated,
        qflx_intr_canopy=qflx_intr,
        qflx_tflrain_canopy=qflx_tflrain,
        qflx_tflsnow_canopy=qflx_tflsnow,
        fwet_profile=fwet,
        fdry_profile=fdry,
    )


def canopy_evaporation(
    inputs: CanopyEvaporationInput,
) -> CanopyEvaporationOutput:
    """Update canopy intercepted water for evaporation and dew.
    
    Processes evaporation from intercepted water and condensation (dew formation)
    when evaporation or transpiration fluxes are negative. Calculations are done
    separately for sunlit and shaded leaf fractions.
    
    Algorithm:
        1. For each layer with PAI > 0:
           a. Calculate dew from sunlit leaves: (evleaf + trleaf) * fracsun * dpai
           b. Add dew if negative (condensation)
           c. Remove water if evleaf > 0 (evaporation)
           d. Repeat for shaded leaves with (1-fracsun)
        2. Convert fluxes from mol/m2 leaf/s to kg/m2 ground using mmh2o
    
    Args:
        inputs: Input state containing canopy structure, fluxes, and water content
        
    Returns:
        Updated canopy intercepted water state
        
    Note:
        - isun index = 0, isha index = 1 in the last dimension of trleaf/evleaf
        - Negative evleaf or trleaf indicates condensation (dew formation)
        - Positive evleaf indicates evaporation of intercepted water
        - Only processes layers where dpai > 0
        
    Reference: MLCanopyWaterMod.F90:174-249
    """
    # Extract constants (lines 218-219)
    dtime = inputs.dtime_substep
    mmh2o = inputs.mmh2o
    
    # Initialize output with input values
    h2ocan = inputs.h2ocan
    
    # Get array dimensions
    n_patches, n_layers = inputs.dpai.shape
    
    # Create mask for valid layers (dpai > 0)
    valid_layer = inputs.dpai > 0.0
    
    # --- Sunlit leaves ---
    
    # Calculate dew from sunlit leaves (lines 224-227)
    # dew = (evleaf + trleaf) * fracsun * dpai * mmh2o * dtime
    dew_sun = (
        (inputs.evleaf[:, :, ISUN] + inputs.trleaf[:, :, ISUN])
        * inputs.fracsun
        * inputs.dpai
        * mmh2o
        * dtime
    )
    
    # Add dew if negative (condensation) (lines 225-227)
    h2ocan = jnp.where(
        valid_layer & (dew_sun < 0.0),
        h2ocan - dew_sun,  # Subtract negative value = add water
        h2ocan
    )
    
    # Evaporate intercepted water if evleaf > 0 (lines 236-238)
    evap_sun = inputs.evleaf[:, :, ISUN] * inputs.fracsun * inputs.dpai * mmh2o * dtime
    h2ocan = jnp.where(
        valid_layer & (inputs.evleaf[:, :, ISUN] > 0.0),
        h2ocan - evap_sun,
        h2ocan
    )
    
    # --- Shaded leaves ---
    
    # Calculate dew from shaded leaves (lines 229-232)
    # dew = (evleaf + trleaf) * (1 - fracsun) * dpai * mmh2o * dtime
    dew_shade = (
        (inputs.evleaf[:, :, ISHA] + inputs.trleaf[:, :, ISHA])
        * (1.0 - inputs.fracsun)
        * inputs.dpai
        * mmh2o
        * dtime
    )
    
    # Add dew if negative (condensation) (lines 230-232)
    h2ocan = jnp.where(
        valid_layer & (dew_shade < 0.0),
        h2ocan - dew_shade,  # Subtract negative value = add water
        h2ocan
    )
    
    # Evaporate intercepted water if evleaf > 0 (lines 240-242)
    evap_shade = inputs.evleaf[:, :, ISHA] * (1.0 - inputs.fracsun) * inputs.dpai * mmh2o * dtime
    h2ocan = jnp.where(
        valid_layer & (inputs.evleaf[:, :, ISHA] > 0.0),
        h2ocan - evap_shade,
        h2ocan
    )
    
    # Note: The commented-out line (line 246) that prevents negative h2ocan
    # is not implemented as noted in the original code
    
    return CanopyEvaporationOutput(h2ocan=h2ocan)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Types
    'CanopyInterceptionParams',
    'CanopyWaterState',
    'CanopyEvaporationInput',
    'CanopyEvaporationOutput',
    # Functions
    'canopy_interception',
    'canopy_evaporation',
    'get_default_interception_params',
    # Constants
    'ISUN',
    'ISHA',
    'MMH2O_DEFAULT',
]