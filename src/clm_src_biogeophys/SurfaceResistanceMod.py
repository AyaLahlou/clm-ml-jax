"""Surface resistance calculations for soil evaporation.

This module provides functions to calculate resistance for soil evaporation,
translating the SurfaceResistanceMod Fortran module. It implements the
Sakaguchi and Zeng (2009) parameterization for soil evaporative resistance.

References:
    Sakaguchi, K., and X. Zeng (2009), Effects of soil wetness, plant litter,
    and under-canopy atmospheric stability on ground evaporation in the
    Community Land Model (CLM3.5), J. Geophys. Res., 114, D01107.

Fortran source: SurfaceResistanceMod.F90, lines 1-123
"""

from typing import NamedTuple, Protocol
import jax.numpy as jnp
from jax import Array, jit


# =============================================================================
# Type Definitions
# =============================================================================

class BoundsType(NamedTuple):
    """Domain decomposition bounds.
    
    Attributes:
        begc: Beginning column index
        endc: Ending column index
        begg: Beginning gridcell index
        endg: Ending gridcell index
        begp: Beginning patch index
        endp: Ending patch index
    """
    begc: int
    endc: int
    begg: int
    endg: int
    begp: int
    endp: int


class SoilStateType(NamedTuple):
    """Soil state variables.
    
    Attributes:
        dsl_col: Soil dry surface layer thickness (mm), shape (ncols,)
        soilresis_col: Soil evaporative resistance (s/m), shape (ncols,)
        watsat: Soil layer volumetric water content at saturation (porosity) [-], shape (ncols, nlevs)
        sucsat: Soil layer suction (negative matric potential) at saturation [mm], shape (ncols, nlevs)
        bsw: Soil layer Clapp and Hornberger "b" parameter [-], shape (ncols, nlevs)
    """
    dsl_col: jnp.ndarray
    soilresis_col: jnp.ndarray
    watsat: jnp.ndarray
    sucsat: jnp.ndarray
    bsw: jnp.ndarray


class WaterStateType(NamedTuple):
    """Water state variables.
    
    Attributes:
        h2osoi_ice: Soil layer ice lens [kg H2O/m2], shape (ncols, nlevs)
        h2osoi_liq: Soil layer liquid water [kg H2O/m2], shape (ncols, nlevs)
    """
    h2osoi_ice: jnp.ndarray
    h2osoi_liq: jnp.ndarray


class TemperatureType(NamedTuple):
    """Temperature state variables.
    
    Attributes:
        t_soisno: Soil temperature [K], shape (ncols, nlevs)
    """
    t_soisno: jnp.ndarray


class ColumnType(NamedTuple):
    """Column geometry variables.
    
    Attributes:
        dz: Soil layer thickness [m], shape (ncols, nlevs)
    """
    dz: jnp.ndarray


# =============================================================================
# Physical Constants
# =============================================================================

# Density of liquid water [kg/m3] (Fortran line 8)
DENH2O = 1000.0

# Density of ice [kg/m3] (Fortran line 9)
DENICE = 917.0


# =============================================================================
# Private Functions
# =============================================================================

@jit
def _calc_soil_resistance_sl14(
    soilstate: SoilStateType,
    waterstate: WaterStateType,
    temperature: TemperatureType,
    col: ColumnType,
    filter_indices: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate soil evaporative resistance using Sakaguchi and Zeng (2009) method.
    
    Translates Fortran subroutine calc_soil_resistance_sl14 from SurfaceResistanceMod.F90
    (lines 60-121).
    
    The calculation follows the Sakaguchi and Zeng (2009) parameterization for soil
    evaporative resistance, accounting for:
    - Dry surface layer thickness based on soil moisture
    - Vapor diffusivity through soil pores
    - Tortuosity effects on diffusion
    
    Args:
        soilstate: Soil hydraulic properties (watsat, sucsat, bsw)
        waterstate: Soil water content (ice and liquid)
        temperature: Soil temperature
        col: Column geometry (layer thickness)
        filter_indices: Column indices to process, shape (num_nolakec,)
    
    Returns:
        Tuple of (dsl, soilresis):
            - dsl: Dry surface layer thickness [mm], shape (ncols,)
            - soilresis: Soil evaporative resistance [s/m], shape (ncols,)
    
    References:
        Fortran source: SurfaceResistanceMod.F90, lines 60-121
    """
    ncols = soilstate.watsat.shape[0]
    
    # Initialize output arrays
    dsl = jnp.zeros(ncols)
    soilresis = jnp.zeros(ncols)
    
    # Extract first soil layer (index 0 in Python, index 1 in Fortran)
    # Fortran lines 95-100
    watsat_1 = soilstate.watsat[:, 0]
    sucsat_1 = soilstate.sucsat[:, 0]
    bsw_1 = soilstate.bsw[:, 0]
    t_soisno_1 = temperature.t_soisno[:, 0]
    h2osoi_ice_1 = waterstate.h2osoi_ice[:, 0]
    h2osoi_liq_1 = waterstate.h2osoi_liq[:, 0]
    dz_1 = col.dz[:, 0]
    
    # Calculate volumetric liquid water content (Fortran line 103)
    vwc_liq = jnp.maximum(h2osoi_liq_1, 1.0e-6) / (dz_1 * DENH2O)
    
    # Calculate effective porosity of top layer (Fortran line 104)
    ice_fraction = jnp.minimum(watsat_1, h2osoi_ice_1 / (dz_1 * DENICE))
    eff_por_top = jnp.maximum(0.01, watsat_1 - ice_fraction)
    
    # Calculate air-dry soil moisture (Fortran line 105)
    aird = watsat_1 * (sucsat_1 / 1.0e7) ** (1.0 / bsw_1)
    
    # Calculate water vapor diffusivity (Fortran line 106)
    d0 = 2.12e-5 * (t_soisno_1 / 273.15) ** 1.75
    
    # Calculate air-filled pore space (Fortran line 107)
    eps = watsat_1 - aird
    
    # Calculate tortuosity (Fortran line 108)
    bsw_max = jnp.maximum(3.0, bsw_1)
    tort = eps * eps * (eps / watsat_1) ** (3.0 / bsw_max)
    
    # Calculate dry surface layer thickness (Fortran lines 109-111)
    numerator = jnp.maximum(0.001, 0.8 * eff_por_top - vwc_liq)
    denominator = jnp.maximum(0.001, 0.8 * watsat_1 - aird)
    dsl_calc = 15.0 * numerator / denominator
    dsl_calc = jnp.maximum(dsl_calc, 0.0)
    dsl_calc = jnp.minimum(dsl_calc, 200.0)
    
    # Calculate soil evaporative resistance (Fortran lines 112-113)
    soilresis_calc = dsl_calc / (d0 * tort * 1.0e3) + 20.0
    soilresis_calc = jnp.minimum(1.0e6, soilresis_calc)
    
    # Update only filtered indices
    dsl = dsl.at[filter_indices].set(dsl_calc[filter_indices])
    soilresis = soilresis.at[filter_indices].set(soilresis_calc[filter_indices])
    
    return dsl, soilresis


# =============================================================================
# Public Functions
# =============================================================================

@jit
def calc_soilevap_resis(
    bounds: BoundsType,
    num_nolakec: int,
    filter_nolakec: jnp.ndarray,
    soilstate_inst: SoilStateType,
    waterstate_inst: WaterStateType,
    temperature_inst: TemperatureType,
    col: ColumnType,
) -> SoilStateType:
    """Calculate resistance for soil evaporation.
    
    Translates the calc_soilevap_resis subroutine from SurfaceResistanceMod.F90
    (lines 29-57). This function computes soil evaporative resistance by calling
    the calc_soil_resistance_sl14 function.
    
    Args:
        bounds: Column bounds containing begc and endc indices
        num_nolakec: Number of non-lake points in column filter
        filter_nolakec: Column filter for non-lake points, shape (num_nolakec,)
        soilstate_inst: Soil state containing dsl_col, soilresis_col, and hydraulic properties
        waterstate_inst: Water state variables (ice and liquid water content)
        temperature_inst: Temperature variables (soil temperature)
        col: Column geometry (layer thickness)
        
    Returns:
        Updated SoilStateType with modified dsl_col and soilresis_col
        
    Note:
        Fortran source: SurfaceResistanceMod.F90, lines 29-57
    """
    # Call calc_soil_resistance_sl14 to compute soil resistance
    # Fortran lines 53-55
    dsl_updated, soilresis_updated = _calc_soil_resistance_sl14(
        soilstate=soilstate_inst,
        waterstate=waterstate_inst,
        temperature=temperature_inst,
        col=col,
        filter_indices=filter_nolakec,
    )
    
    # Return updated soil state with new dsl_col and soilresis_col
    return SoilStateType(
        dsl_col=dsl_updated,
        soilresis_col=soilresis_updated,
        watsat=soilstate_inst.watsat,
        sucsat=soilstate_inst.sucsat,
        bsw=soilstate_inst.bsw,
    )