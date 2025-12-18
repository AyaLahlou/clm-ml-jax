"""
SoilWaterMovementMod - Soil and root water interaction coupling

This module contains functions to couple soil and root water interactions,
computing hydraulic properties and soil moisture dynamics for CLM hydrology.

Translated from Fortran source: SoilWaterMovementMod.F90 (lines 1-166)

Core Principles:
- Pure functions with immutable state
- JIT-compatible operations
- Exact physics preservation from Clapp-Hornberger (1978) relationships
- Full type hints and documentation

Public Interface:
    - soil_water: Main entry point for soil water movement calculations
    
Private Functions:
    - soilwater_moisture_form: Compute soil moisture properties
    - compute_hydraulic_properties: Calculate hydraulic conductivity and matric potential
"""

from typing import NamedTuple, Callable, Optional, Tuple
from functools import partial
import jax.numpy as jnp
from jax import jit, Array


# ============================================================================
# Type Definitions
# ============================================================================

class Bounds(NamedTuple):
    """CLM column bounds.
    
    Attributes:
        begc: Beginning column index
        endc: Ending column index
    """
    begc: int
    endc: int


class SoilStateArrays(NamedTuple):
    """Soil state arrays needed for hydraulic property calculations.
    
    Attributes:
        watsat: Soil layer volumetric water content at saturation (porosity) [m3/m3]
        hksat: Soil layer hydraulic conductivity at saturation [mm H2O/s]
        sucsat: Soil layer suction (negative matric potential) at saturation [mm]
        bsw: Soil layer Clapp and Hornberger "b" parameter [-]
        nbedrock: Depth to bedrock index for each column
        dz: Soil layer thickness [m]
    """
    watsat: Array    # (ncolumns, nlevsoi)
    hksat: Array     # (ncolumns, nlevsoi)
    sucsat: Array    # (ncolumns, nlevsoi)
    bsw: Array       # (ncolumns, nlevsoi)
    nbedrock: Array  # (ncolumns,)
    dz: Array        # (ncolumns, nlevsoi)


class WaterStateType(NamedTuple):
    """Water state variables.
    
    Attributes:
        h2osoi_liq: Soil layer liquid water [kg H2O/m2]
    """
    h2osoi_liq: Array  # (ncolumns, nlevsoi)


class SoilStateType(NamedTuple):
    """Soil state variables including hydraulic properties.
    
    Attributes:
        soil_arrays: Soil physical properties
        vwc_liq: Liquid volumetric water content [m3/m3]
        hk: Soil layer hydraulic conductivity [mm H2O/s]
        smp: Soil layer matric potential [mm]
    """
    soil_arrays: SoilStateArrays
    vwc_liq: Array  # (ncolumns, nlevsoi)
    hk: Array       # (ncolumns, nlevsoi)
    smp: Array      # (ncolumns, nlevsoi)


class HydraulicProperties(NamedTuple):
    """Output hydraulic properties for a single column.
    
    Attributes:
        hk: Soil layer hydraulic conductivity [mm H2O/s]
        smp: Soil layer matric potential [mm]
    """
    hk: Array   # (nlayers,)
    smp: Array  # (nlayers,)


# ============================================================================
# Physical Constants
# ============================================================================

# Density of liquid water [kg/m3]
# Fortran line 8: use clm_varcon, only : denh2o
DENH2O = 1000.0


# ============================================================================
# Private Helper Functions
# ============================================================================

@partial(jit, static_argnums=(1,))
def _compute_vwc_liq_single_column(
    h2osoi_liq: Array,
    nlayers: int,
    dz: Array
) -> Array:
    """Compute liquid volumetric water content for a single column.
    
    Fortran reference: lines 93-95
    
    Args:
        h2osoi_liq: Soil layer liquid water [kg H2O/m2], shape (nlevsoi,)
        nlayers: Number of active layers (to bedrock)
        dz: Soil layer thickness [m], shape (nlevsoi,)
        
    Returns:
        vwc_liq: Liquid volumetric water content [m3/m3], shape (nlevsoi,)
        
    Note:
        Formula: vwc_liq = max(h2osoi_liq, 1.0e-6) / (dz * denh2o)
        Applied only to active layers (above bedrock)
    """
    # Fortran line 94-95: vwc_liq(c,j) = max(h2osoi_liq(c,j),1.0e-6_r8)/(dz(c,j)*denh2o)
    layer_mask = jnp.arange(h2osoi_liq.shape[0]) < nlayers
    
    vwc_liq = jnp.where(
        layer_mask,
        jnp.maximum(h2osoi_liq, 1.0e-6) / (dz * DENH2O),
        0.0  # Inactive layers set to zero
    )
    
    return vwc_liq


@jit
def _compute_hydraulic_properties_layer(
    vwc_liq: Array,
    watsat: Array,
    hksat: Array,
    sucsat: Array,
    bsw: Array
) -> HydraulicProperties:
    """Compute hydraulic conductivity and soil matric potential for soil layers.
    
    Implements the Clapp-Hornberger (1978) relationships for hydraulic
    conductivity and matric potential as functions of soil water content.
    
    Physics equations (Fortran lines 147-151):
    - s = vwc_liq / watsat (relative saturation, clamped to [0.01, 1.0])
    - hk = hksat * s^(2*bsw + 3)
    - smp = -sucsat * s^(-bsw) (clamped to >= -1e8)
    
    Fortran reference: lines 112-164
    
    Args:
        vwc_liq: Soil layer liquid volumetric water content [m3 H2O/m3]
        watsat: Volumetric water content at saturation (porosity) [m3/m3]
        hksat: Hydraulic conductivity at saturation [mm H2O/s]
        sucsat: Suction (negative matric potential) at saturation [mm]
        bsw: Clapp and Hornberger "b" parameter [-]
        
    Returns:
        HydraulicProperties containing:
            - hk: Hydraulic conductivity [mm H2O/s]
            - smp: Soil matric potential [mm]
    """
    # Fortran lines 147-149: Compute relative saturation s
    # s = vwc_liq(j) / watsat(c,j)
    # s = min(s, 1._r8)
    # s = max(0.01_r8, s)
    s = vwc_liq / watsat
    s = jnp.minimum(s, 1.0)
    s = jnp.maximum(0.01, s)
    
    # Fortran line 150: Compute hydraulic conductivity
    # hk(j) = hksat(c,j) * s**(2._r8 * bsw(c,j) + 3._r8)
    hk = hksat * jnp.power(s, 2.0 * bsw + 3.0)
    
    # Fortran lines 151-152: Compute soil matric potential
    # smp(j) = -sucsat(c,j) * s**(-bsw(c,j))
    # smp(j) = max(smp(j), -1.e08_r8)
    smp = -sucsat * jnp.power(s, -bsw)
    smp = jnp.maximum(smp, -1.0e8)
    
    return HydraulicProperties(hk=hk, smp=smp)


def _process_single_column(
    c: int,
    soilstate_inst: SoilStateType,
    waterstate_inst: WaterStateType
) -> Tuple[Array, Array, Array]:
    """Process a single column for moisture and hydraulic property calculations.
    
    Fortran reference: lines 87-105
    
    Args:
        c: Column index
        soilstate_inst: Soil state variables
        waterstate_inst: Water state variables
        
    Returns:
        Tuple of (vwc_liq, hk, smp) arrays for this column
    """
    # Fortran line 90: nlayers = nbedrock(c)
    nlayers = soilstate_inst.soil_arrays.nbedrock[c]
    
    # Fortran lines 93-95: Compute liquid volumetric water content
    vwc_col = _compute_vwc_liq_single_column(
        waterstate_inst.h2osoi_liq[c],
        nlayers,
        soilstate_inst.soil_arrays.dz[c]
    )
    
    # Fortran lines 97-99: Call compute_hydraulic_properties
    hydraulic_props = _compute_hydraulic_properties_layer(
        vwc_col,
        soilstate_inst.soil_arrays.watsat[c],
        soilstate_inst.soil_arrays.hksat[c],
        soilstate_inst.soil_arrays.sucsat[c],
        soilstate_inst.soil_arrays.bsw[c]
    )
    
    return vwc_col, hydraulic_props.hk, hydraulic_props.smp


# ============================================================================
# Private Core Functions
# ============================================================================

def _soilwater_moisture_form(
    bounds: Bounds,
    num_hydrologyc: int,
    filter_hydrologyc: Array,
    soilstate_inst: SoilStateType,
    waterstate_inst: WaterStateType
) -> SoilStateType:
    """Compute soil moisture properties for hydrology calculations.
    
    Fortran reference: lines 50-109 (subroutine soilwater_moisture_form)
    
    This function computes liquid volumetric water content and hydraulic
    properties for all columns in the hydrology filter.
    
    Args:
        bounds: CLM column bounds containing begc, endc indices
        num_hydrologyc: Number of columns in CLM hydrology filter
        filter_hydrologyc: CLM column filter indices for hydrology
        soilstate_inst: Soil state variables
        waterstate_inst: Water state variables
        
    Returns:
        Updated SoilStateType with computed vwc_liq, hk, and smp
    """
    ncolumns, nlevsoi = soilstate_inst.soil_arrays.dz.shape
    
    # Initialize output arrays (Fortran lines 78-80)
    vwc_liq = soilstate_inst.vwc_liq
    hk = soilstate_inst.hk
    smp = soilstate_inst.smp
    
    # Process each column in the hydrology filter (Fortran lines 87-105)
    for fc in range(num_hydrologyc):
        c = filter_hydrologyc[fc]
        vwc_col, hk_col, smp_col = _process_single_column(
            c, soilstate_inst, waterstate_inst
        )
        vwc_liq = vwc_liq.at[c].set(vwc_col)
        hk = hk.at[c].set(hk_col)
        smp = smp.at[c].set(smp_col)
    
    # Return updated soil state
    return SoilStateType(
        soil_arrays=soilstate_inst.soil_arrays,
        vwc_liq=vwc_liq,
        hk=hk,
        smp=smp
    )


# ============================================================================
# Public Interface
# ============================================================================

def soil_water(
    bounds: Bounds,
    num_hydrologyc: int,
    filter_hydrologyc: Array,
    soilstate_inst: SoilStateType,
    waterstate_inst: WaterStateType
) -> SoilStateType:
    """Main entry point for soil water movement calculations.
    
    This function serves as the public interface for soil water movement
    calculations. It computes liquid volumetric water content, hydraulic
    conductivity, and soil matric potential for all columns in the hydrology
    filter using the Clapp-Hornberger (1978) relationships.
    
    Fortran reference: lines 25-47 (subroutine SoilWater)
    
    Args:
        bounds: CLM column bounds containing begc, endc indices
        num_hydrologyc: Number of columns in CLM hydrology filter
        filter_hydrologyc: CLM column filter indices for hydrology, shape (num_hydrologyc,)
        soilstate_inst: Soil state variables including:
            - soil_arrays: Physical soil properties (watsat, hksat, sucsat, bsw, nbedrock, dz)
            - vwc_liq: Liquid volumetric water content (will be updated)
            - hk: Hydraulic conductivity (will be updated)
            - smp: Soil matric potential (will be updated)
        waterstate_inst: Water state variables including:
            - h2osoi_liq: Soil layer liquid water [kg H2O/m2]
    
    Returns:
        Updated SoilStateType with computed:
            - vwc_liq: Liquid volumetric water content [m3/m3]
            - hk: Soil layer hydraulic conductivity [mm H2O/s]
            - smp: Soil layer matric potential [mm]
            
    Note:
        This is a thin wrapper around _soilwater_moisture_form that maintains
        the original Fortran interface structure. In JAX, we return the updated
        state rather than modifying in-place.
        
    Physics:
        Uses Clapp-Hornberger (1978) relationships:
        - Hydraulic conductivity: K(θ) = K_sat * (θ/θ_sat)^(2b+3)
        - Matric potential: ψ(θ) = ψ_sat * (θ/θ_sat)^(-b)
        where θ is volumetric water content, b is the pore size distribution index
    """
    # Call the moisture form calculation (Fortran lines 45-46)
    updated_soilstate = _soilwater_moisture_form(
        bounds=bounds,
        num_hydrologyc=num_hydrologyc,
        filter_hydrologyc=filter_hydrologyc,
        soilstate_inst=soilstate_inst,
        waterstate_inst=waterstate_inst
    )
    
    return updated_soilstate


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'soil_water',
    'SoilStateType',
    'WaterStateType',
    'Bounds',
    'SoilStateArrays',
    'HydraulicProperties',
]