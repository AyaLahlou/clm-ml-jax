"""
SoilStateInitTimeConstMod - Set hydraulic and thermal properties

This module provides functionality for initializing time-constant soil state properties
including hydraulic conductivity, thermal conductivity, and heat capacity.

Translated from Fortran: SoilStateInitTimeConstMod.F90 (lines 1-330)

Key features:
- Pure functional implementation using JAX
- Immutable state management with NamedTuples
- JIT-compatible operations with vectorization
- Support for multiple root distribution methods (Zeng2001, Jackson1996)
- Organic matter adjustments to soil properties
- Tower-specific soil texture data handling

Physics:
- Root distribution profiles (lines 86-135)
- Hydraulic properties: saturation, suction, conductivity (lines 188-267)
- Thermal properties: conductivity, heat capacity (lines 268-330)
- Organic matter effects on soil properties (lines 228-242)
"""

from typing import NamedTuple, Optional, Tuple
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import Array


# ============================================================================
# Type Definitions
# ============================================================================

class BoundsType(NamedTuple):
    """Bounds type for domain decomposition (lines 1-25)"""
    begp: int  # Beginning patch index
    endp: int  # Ending patch index
    begc: int  # Beginning column index
    endc: int  # Ending column index
    begg: int  # Beginning gridcell index
    endg: int  # Ending gridcell index


class PatchType(NamedTuple):
    """Patch-level data structure (lines 1-25)"""
    column: Array  # Column index for each patch (npatch,)
    itype: Array   # Patch type / PFT index (npatch,)


class ColumnType(NamedTuple):
    """Column-level data structure (lines 1-25)"""
    gridcell: Array  # Gridcell index for each column (ncol,)
    itype: Array     # Column type (ncol,)
    nbedrock: Array  # Number of soil layers above bedrock (ncol,)
    zi: Array        # Soil layer interface depth (ncol, nlevgrnd+1) [m]
    z: Array         # Soil layer center depth (ncol, nlevgrnd) [m]


class PftconType(NamedTuple):
    """Plant functional type constants (lines 1-25)"""
    roota_par: Array      # Zeng2001 rooting parameter a (npft,) [1/m]
    rootb_par: Array      # Zeng2001 rooting parameter b (npft,) [1/m]
    rootprof_beta: Array  # Jackson1996 rooting parameter beta (npft,) [-]


class SoilTextureType(NamedTuple):
    """Soil texture lookup tables (lines 188-198)"""
    names: Array      # Texture class names (ntex,)
    clay: Array       # Clay fraction by texture (ntex,) [-]
    sand: Array       # Sand fraction by texture (ntex,) [-]
    watsat: Array     # Saturation by texture (ntex,) [m3/m3]
    smpsat: Array     # Matric potential at saturation (ntex,) [mm]
    hksat: Array      # Hydraulic conductivity (ntex,) [mm/min]
    bsw: Array        # Clapp-Hornberger b parameter (ntex,) [-]


class TowerDataType(NamedTuple):
    """Tower-specific soil data (lines 162-187)"""
    num: int              # Current tower number
    tex: Array            # Texture class per tower (ntowers,)
    clay: Array           # Clay fraction per tower (ntowers,) [-]
    sand: Array           # Sand fraction per tower (ntowers,) [-]
    organic: Array        # Organic matter content per tower (ntowers,) [kg/m2]
    col_tower: Array      # Tower index for each column (ncol,)


@dataclass
class SoilStateType:
    """Complete soil state with hydraulic and thermal properties"""
    # Root distribution (lines 86-135)
    rootfr: Array  # Root fraction (npatch, nlevgrnd) [-]
    
    # Soil texture (lines 162-211)
    cellsand: Array  # Sand percentage (ncol, nlevsoi) [%]
    cellclay: Array  # Clay percentage (ncol, nlevsoi) [%]
    cellorg: Array   # Organic matter (ncol, nlevsoi) [kg/m2]
    
    # Hydraulic properties (lines 213-267)
    watsat: Array    # Volumetric water at saturation (ncol, nlevgrnd) [m3/m3]
    sucsat: Array    # Minimum soil suction (ncol, nlevgrnd) [mm]
    hksat: Array     # Hydraulic conductivity at saturation (ncol, nlevgrnd) [mm/s]
    bsw: Array       # Clapp-Hornberger b parameter (ncol, nlevgrnd) [-]
    perc_frac: Array # Percolation fraction (ncol, nlevgrnd) [-]
    
    # Thermal properties (lines 268-330)
    tkdry: Array     # Thermal conductivity, dry (ncol, nlevgrnd) [W/m/K]
    tkmg: Array      # Thermal conductivity, minerals (ncol, nlevgrnd) [W/m/K]
    csol: Array      # Heat capacity of solids (ncol, nlevgrnd) [J/m3/K]


@dataclass
class SoilStateInitConfig:
    """Configuration for SoilStateInitTimeConst (lines 26-45)"""
    # Physical constants
    csol_bedrock: float = 2.0e6  # Heat capacity of bedrock [J/m3/K]
    organic_max: float = 130.0   # Maximum organic matter [kg/m2]
    zsapric: float = 0.5         # Depth for sapric organic matter [m]
    pcalpha: float = 0.5         # Percolation threshold [-]
    pcbeta: float = 0.139        # Percolation exponent [-]
    m_to_cm: float = 100.0       # Meters to centimeters conversion
    
    # Grid dimensions
    nlevsoi: int = 10            # Number of soil layers
    nlevgrnd: int = 15           # Number of ground layers (soil + bedrock)
    
    # Physics options
    clm_phys: int = 0            # Physics version (0=CLM5_0, 1=other)
    root_type: int = 1           # Root distribution (1=Zeng2001, 2=Jackson1996)


# ============================================================================
# Root Distribution Functions (lines 86-135)
# ============================================================================

def compute_root_profile_zeng2001(
    roota_par: float,
    rootb_par: float,
    zi: Array,
    nlevsoi: int,
) -> Array:
    """Compute root profile using Zeng2001 method.
    
    Implements the Zeng2001 rooting distribution for soil water uptake.
    Fortran lines 111-122.
    
    Args:
        roota_par: Zeng2001 rooting distribution parameter [1/m]
        rootb_par: Zeng2001 rooting distribution parameter [1/m]
        zi: Soil layer depth at layer interface [m], shape (nlevsoi+1,)
        nlevsoi: Number of soil layers
        
    Returns:
        rootfr: Fraction of roots in each soil layer, shape (nlevsoi,)
    """
    # Lines 113-117: Compute for layers 1 to nlevsoi-1
    def compute_layer_j(j):
        return 0.5 * (
            jnp.exp(-roota_par * zi[j-1]) + jnp.exp(-rootb_par * zi[j-1]) -
            jnp.exp(-roota_par * zi[j]) - jnp.exp(-rootb_par * zi[j])
        )
    
    # Vectorize over layers 1 to nlevsoi-1 (indices 0 to nlevsoi-2 in 0-based)
    rootfr_middle = jax.vmap(compute_layer_j)(jnp.arange(1, nlevsoi))
    
    # Lines 119-120: Last layer (nlevsoi)
    j_last = nlevsoi
    rootfr_last = 0.5 * (
        jnp.exp(-roota_par * zi[j_last-1]) +
        jnp.exp(-rootb_par * zi[j_last-1])
    )
    
    # Combine all layers
    rootfr = jnp.concatenate([rootfr_middle, jnp.array([rootfr_last])])
    
    return rootfr


def compute_root_profile_jackson1996(
    beta: float,
    zi: Array,
    nlevsoi: int,
    m_to_cm: float = 100.0,
) -> Array:
    """Compute root profile using Jackson1996 method.
    
    Implements the Jackson1996 rooting distribution for soil water uptake.
    Fortran lines 124-131.
    
    Args:
        beta: Jackson1996 rooting distribution parameter [-]
        zi: Soil layer depth at layer interface [m], shape (nlevsoi+1,)
        nlevsoi: Number of soil layers
        m_to_cm: Conversion factor from meters to centimeters
        
    Returns:
        rootfr: Fraction of roots in each soil layer, shape (nlevsoi,)
    """
    # Lines 128-130: Compute for all soil layers
    def compute_layer_j(j):
        return (
            beta ** (zi[j-1] * m_to_cm) - 
            beta ** (zi[j] * m_to_cm)
        )
    
    # Vectorize over all layers (j from 1 to nlevsoi)
    rootfr = jax.vmap(compute_layer_j)(jnp.arange(1, nlevsoi + 1))
    
    return rootfr


def compute_root_profile_for_patch(
    root_type: int,
    pft_type: int,
    roota_par_pft: Array,
    rootb_par_pft: Array,
    rootprof_beta_pft: Array,
    zi_col: Array,
    nlevsoi: int,
) -> Array:
    """Compute root profile for a single patch based on root_type.
    
    Dispatches to appropriate root profile computation method.
    Fortran lines 107-131.
    
    Args:
        root_type: Root profile type (1=Zeng2001, 2=Jackson1996)
        pft_type: Plant functional type index
        roota_par_pft: Zeng2001 roota parameter for all PFTs, shape (npft,)
        rootb_par_pft: Zeng2001 rootb parameter for all PFTs, shape (npft,)
        rootprof_beta_pft: Jackson1996 beta parameter for all PFTs, shape (npft,)
        zi_col: Soil layer depth at interface for column [m], shape (nlevsoi+1,)
        nlevsoi: Number of soil layers
        
    Returns:
        rootfr: Fraction of roots in each soil layer, shape (nlevsoi,)
    """
    # Lines 107-122: case (1) - Zeng2001 method
    rootfr_zeng = compute_root_profile_zeng2001(
        roota_par_pft[pft_type],
        rootb_par_pft[pft_type],
        zi_col,
        nlevsoi,
    )
    
    # Lines 124-131: case (2) - Jackson1996 method
    rootfr_jackson = compute_root_profile_jackson1996(
        rootprof_beta_pft[pft_type],
        zi_col,
        nlevsoi,
    )
    
    # Select based on root_type using jnp.where for JIT compatibility
    rootfr = jnp.where(
        root_type == 1,
        rootfr_zeng,
        rootfr_jackson,
    )
    
    return rootfr


def adjust_roots_for_bedrock(
    rootfr: Array,
    nbedrock: Array,
    patch_col: Array,
    nlevsoi: int,
    nlevgrnd: int,
) -> Array:
    """Adjust root fractions for bedrock layers and soil depth.
    
    Fortran lines 142-154
    
    This function:
    1. Sets bedrock layers (nlevsoi+1:nlevgrnd) to have no roots
    2. Redistributes roots from below bedrock to layers above bedrock
    
    Args:
        rootfr: Root fraction array (npatch, nlevgrnd)
        nbedrock: Number of soil layers above bedrock per column (ncol,)
        patch_col: Column index for each patch (npatch,)
        nlevsoi: Number of soil layers
        nlevgrnd: Total number of ground layers
        
    Returns:
        Updated root fraction array (npatch, nlevgrnd)
    """
    npatch = rootfr.shape[0]
    
    def adjust_single_patch(p_idx: int, rootfr_p: Array) -> Array:
        """Adjust roots for a single patch."""
        c_idx = patch_col[p_idx]
        nbedrock_c = nbedrock[c_idx]
        
        # Set bedrock layers to zero (lines 144-146)
        rootfr_p = rootfr_p.at[nlevsoi:nlevgrnd].set(0.0)
        
        # Calculate sum of roots below bedrock (lines 150-152)
        below_bedrock_sum = jnp.sum(
            jnp.where(
                (jnp.arange(nlevgrnd) >= nbedrock_c) & (jnp.arange(nlevgrnd) < nlevsoi),
                rootfr_p,
                0.0
            )
        )
        
        # Redistribute to layers above bedrock
        redistribution = jnp.where(
            nbedrock_c > 0,
            below_bedrock_sum / nbedrock_c,
            0.0
        )
        
        # Add redistribution to layers 0:nbedrock(c)
        mask_above = jnp.arange(nlevgrnd) < nbedrock_c
        rootfr_p = jnp.where(
            mask_above,
            rootfr_p + redistribution,
            rootfr_p
        )
        
        # Zero out layers from nbedrock(c) to nlevsoi (line 153)
        mask_zero = (jnp.arange(nlevgrnd) >= nbedrock_c) & (jnp.arange(nlevgrnd) < nlevsoi)
        rootfr_p = jnp.where(mask_zero, 0.0, rootfr_p)
        
        return rootfr_p
    
    # Vectorize over all patches
    rootfr_adjusted = jax.vmap(adjust_single_patch)(
        jnp.arange(npatch),
        rootfr
    )
    
    return rootfr_adjusted


# ============================================================================
# Soil Texture Functions (lines 162-211)
# ============================================================================

def compute_soil_texture_properties(
    tower_data: TowerDataType,
    soil_texture: SoilTextureType,
    organic_max: float,
    ncol: int,
) -> Tuple[Array, Array, Array, Array]:
    """Compute soil texture properties for all columns.
    
    Fortran lines 162-187
    
    Args:
        tower_data: Tower-specific soil data
        soil_texture: Soil texture lookup tables
        organic_max: Maximum organic matter content [kg/m2]
        ncol: Number of columns
        
    Returns:
        Tuple of (om_frac, tex, clay, sand) arrays for all columns
    """
    
    def compute_single_column(c_idx: int) -> Tuple[float, int, float, float]:
        """Compute properties for a single column."""
        tower_idx = tower_data.col_tower[c_idx]
        
        # Organic matter fraction (lines 164-165)
        om_frac = tower_data.organic[tower_idx] / organic_max
        
        # Check if clay and sand are specified (lines 169-173)
        has_clay_sand = (tower_data.clay[tower_idx] >= 0.0) & (tower_data.sand[tower_idx] >= 0.0)
        
        # If specified, use tower clay and sand
        tex_direct = 0
        clay_direct = tower_data.clay[tower_idx]
        sand_direct = tower_data.sand[tower_idx]
        
        # Otherwise, find texture type (lines 177-185)
        tex_matches = jnp.where(
            soil_texture.names == tower_data.tex[tower_idx],
            jnp.arange(len(soil_texture.names)),
            len(soil_texture.names)  # sentinel value
        )
        tex_from_lookup = jnp.min(tex_matches)
        
        # Get clay and sand from lookup
        clay_from_lookup = soil_texture.clay[tex_from_lookup]
        sand_from_lookup = soil_texture.sand[tex_from_lookup]
        
        # Select based on whether clay/sand are specified
        tex = jnp.where(has_clay_sand, tex_direct, tex_from_lookup)
        clay = jnp.where(has_clay_sand, clay_direct, clay_from_lookup)
        sand = jnp.where(has_clay_sand, sand_direct, sand_from_lookup)
        
        return om_frac, tex, clay, sand
    
    # Vectorize over all columns
    results = jax.vmap(compute_single_column)(jnp.arange(ncol))
    
    om_frac_arr = results[0]
    tex_arr = results[1]
    clay_arr = results[2]
    sand_arr = results[3]
    
    return om_frac_arr, tex_arr, clay_arr, sand_arr


# ============================================================================
# Hydraulic Properties Functions (lines 188-267)
# ============================================================================

def compute_soil_hydraulic_properties_for_layer(
    j: int,
    tex: int,
    sand: float,
    clay: float,
    om_frac: float,
    z_j: float,
    zsapric: float,
    organic_max: float,
    pcalpha: float,
    pcbeta: float,
    nlevsoi: int,
    soil_texture: SoilTextureType,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Compute hydraulic properties for a single soil layer.
    
    Fortran lines 200-249
    
    Args:
        j: Layer index (0-based)
        tex: Soil texture class index (0 for sand/clay based, >0 for texture class)
        sand: Sand percentage [%]
        clay: Clay percentage [%]
        om_frac: Organic matter fraction [-]
        z_j: Depth at layer j [m]
        zsapric: Depth for sapric organic matter [m]
        organic_max: Maximum organic matter fraction [kg/m2]
        pcalpha: Percolation threshold [-]
        pcbeta: Percolation exponent [-]
        nlevsoi: Number of soil layers
        soil_texture: Soil texture lookup tables
        
    Returns:
        Tuple of (cellsand, cellclay, cellorg, watsat, sucsat, hksat, bsw, perc_frac)
    """
    # Line 202-204: Set organic matter fraction to zero for deep soil
    om_frac_layer = jnp.where(z_j > zsapric, 0.0, om_frac)
    
    # Line 206-211: Sand/clay/organic matter only for soil layers
    is_soil_layer = j < nlevsoi
    cellsand_val = jnp.where(is_soil_layer, sand, 0.0)
    cellclay_val = jnp.where(is_soil_layer, clay, 0.0)
    cellorg_val = jnp.where(is_soil_layer, om_frac_layer * organic_max, 0.0)
    
    # Line 213-218: Hydraulic properties for mineral soil
    use_sand_clay = tex == 0
    
    # Sand/clay based (CLM5 method)
    watsat_sc = 0.489 - 0.00126 * sand
    sucsat_sc = 10.0 * (10.0 ** (1.88 - 0.0131 * sand))
    hksat_sc = 0.0070556 * (10.0 ** (-0.884 + 0.0153 * sand))
    bsw_sc = 2.91 + 0.159 * clay
    
    # Texture class based (Clapp and Hornberger 1978)
    tex_idx = jnp.maximum(tex - 1, 0)
    watsat_tc = soil_texture.watsat[tex_idx]
    sucsat_tc = -soil_texture.smpsat[tex_idx]
    hksat_tc = soil_texture.hksat[tex_idx] / 60.0  # mm/min -> mm/s
    bsw_tc = soil_texture.bsw[tex_idx]
    
    # Select based on tex value
    watsat_mineral = jnp.where(use_sand_clay, watsat_sc, watsat_tc)
    sucsat_mineral = jnp.where(use_sand_clay, sucsat_sc, sucsat_tc)
    hksat_mineral = jnp.where(use_sand_clay, hksat_sc, hksat_tc)
    bsw_mineral = jnp.where(use_sand_clay, bsw_sc, bsw_tc)
    
    # Line 228-231: Adjust hydraulic properties for organic matter
    om_watsat = jnp.maximum(0.93 - 0.1 * (z_j / zsapric), 0.83)
    om_sucsat = jnp.minimum(10.3 - 0.2 * (z_j / zsapric), 10.1)
    om_hksat = jnp.maximum(0.28 - 0.2799 * (z_j / zsapric), hksat_mineral)
    om_b = jnp.minimum(2.7 + 9.3 * (z_j / zsapric), 12.0)
    
    # Line 233-235: Blend mineral and organic properties
    watsat_final = (1.0 - om_frac_layer) * watsat_mineral + om_watsat * om_frac_layer
    sucsat_final = (1.0 - om_frac_layer) * sucsat_mineral + om_sucsat * om_frac_layer
    bsw_final = (1.0 - om_frac_layer) * bsw_mineral + om_frac_layer * om_b
    
    # Line 250-267: Compute unconnected fraction and hydraulic conductivity
    perc_norm = (1.0 - pcalpha) ** (-pcbeta)
    perc_frac_calc = perc_norm * (om_frac_layer - pcalpha) ** pcbeta
    perc_frac_final = jnp.where(om_frac_layer > pcalpha, perc_frac_calc, 0.0)
    
    # Lines 253-267: Hydraulic conductivity with organic matter
    uncon_frac = (1.0 - om_frac_layer) + (1.0 - perc_frac_final) * om_frac_layer
    
    uncon_hksat = jnp.where(
        om_frac_layer < 1.0,
        uncon_frac / (
            (1.0 - om_frac_layer) / hksat_mineral + 
            ((1.0 - perc_frac_final) * om_frac_layer) / om_hksat
        ),
        0.0
    )
    
    hksat_final = uncon_frac * uncon_hksat + (perc_frac_final * om_frac_layer) * om_hksat
    
    return (
        cellsand_val,
        cellclay_val,
        cellorg_val,
        watsat_final,
        sucsat_final,
        hksat_final,
        bsw_final,
        perc_frac_final,
    )


# ============================================================================
# Thermal Properties Functions (lines 268-330)
# ============================================================================

def compute_thermal_conductivity_dry(
    om_frac: float,
    watsat: float,
    j: int,
    nlevsoi: int,
    clm_phys: int
) -> float:
    """Compute thermal conductivity of dry soil.
    
    Fortran lines 268-285
    
    Args:
        om_frac: Organic matter fraction [-]
        watsat: Volumetric soil water at saturation (porosity) [m3/m3]
        j: Current layer index (0-based)
        nlevsoi: Number of soil layers
        clm_phys: Physics version (0 for CLM5_0, 1 for other)
    
    Returns:
        Thermal conductivity of dry soil [W/m/K]
    """
    # Line 268-272: Older simulations used om_frac = 0.02 for soil thermal properties
    om_frac_therm = jnp.where(
        clm_phys != 0,
        0.02,
        om_frac
    )
    
    # Line 274: Thermal conductivity of organic matter, dry
    om_tkdry = 0.05
    
    # Line 275-284: Compute thermal conductivity based on layer
    def compute_soil_layer():
        # Lines 276-278: Bulk density and thermal conductivity of mineral soil
        bulk_dens_min = 2700.0 * (1.0 - watsat)
        tkdry_min = (0.135 * bulk_dens_min + 64.7) / (2700.0 - 0.947 * bulk_dens_min)
        # Line 279: Mix mineral and organic thermal conductivities
        return (1.0 - om_frac_therm) * tkdry_min + om_frac_therm * om_tkdry
    
    def compute_bedrock_layer():
        # Lines 281-283: Bedrock layer (below soil)
        bulk_dens_min = 2700.0
        tkdry_min = (0.135 * bulk_dens_min + 64.7) / (2700.0 - 0.947 * bulk_dens_min)
        return tkdry_min
    
    tkdry = jnp.where(
        j < nlevsoi,
        compute_soil_layer(),
        compute_bedrock_layer()
    )
    
    return tkdry


def compute_soil_thermal_properties(
    tex: int,
    sand: float,
    clay: float,
    quartz: float,
    om_frac_therm: float,
    watsat: float,
    j: int,
    nlevsoi: int,
    csol_bedrock: float = 2.0e6
) -> Tuple[float, float]:
    """Compute thermal conductivity and heat capacity of soil.
    
    Fortran lines 287-330
    
    Args:
        tex: Texture flag (0 for sand/clay based, non-zero for default)
        sand: Sand fraction [-]
        clay: Clay fraction [-]
        quartz: Quartz fraction [-]
        om_frac_therm: Organic matter fraction for thermal properties [-]
        watsat: Volumetric soil water at saturation [m3/m3]
        j: Current layer index (0-based)
        nlevsoi: Number of soil layers
        csol_bedrock: Heat capacity of bedrock [J/m3/K]
    
    Returns:
        Tuple of (tkmg, csol):
            tkmg: Thermal conductivity of minerals [W/m/K]
            csol: Heat capacity of solids [J/m3/K]
    """
    # Constants (lines 287, 300-302, 317)
    om_tksol = 0.25      # Thermal conductivity of organic matter [W/m/K]
    tksol_quartz = 7.7   # Thermal conductivity of quartz [W/m/K]
    tksol_other = 3.0    # Thermal conductivity of other minerals [W/m/K]
    om_cvsol = 2.5e6     # Heat capacity of organic matter [J/m3/K]
    
    # Check if in soil layer
    is_soil = j < nlevsoi
    
    # Lines 300-306: Thermal conductivity of soil minerals
    tksol_min = jnp.power(tksol_quartz, quartz) * jnp.power(tksol_other, 1.0 - quartz)
    tkm = (1.0 - om_frac_therm) * tksol_min + om_frac_therm * om_tksol
    tkmg_soil = jnp.power(tkm, 1.0 - watsat)
    
    # Line 308: Bedrock thermal conductivity
    tkmg = jnp.where(is_soil, tkmg_soil, 3.0)
    
    # Lines 312-323: Heat capacity of soil solids
    if tex == 0:
        # Use sand and clay fractions
        cvsol = ((2.128 * sand + 2.385 * clay) / (sand + clay)) * 1.0e6
    else:
        # Default value
        cvsol = 1.926e6
    
    csol_soil = (1.0 - om_frac_therm) * cvsol + om_frac_therm * om_cvsol
    csol = jnp.where(is_soil, csol_soil, csol_bedrock)
    
    return tkmg, csol


# ============================================================================
# Main Initialization Function (lines 26-330)
# ============================================================================

def soilstate_init_time_const(
    bounds: BoundsType,
    patch: PatchType,
    col: ColumnType,
    pftcon: PftconType,
    tower_data: TowerDataType,
    soil_texture: SoilTextureType,
    config: SoilStateInitConfig,
) -> SoilStateType:
    """Initialize time-constant soil state properties.
    
    This function initializes hydraulic and thermal properties of soil including:
    - Root distribution profiles (Zeng2001 or Jackson1996)
    - Soil texture properties (sand, clay, organic matter)
    - Hydraulic properties (saturation, suction, conductivity)
    - Thermal properties (conductivity, heat capacity)
    
    Translated from Fortran lines 26-330 in SoilStateInitTimeConstMod.F90
    
    Args:
        bounds: Domain decomposition bounds
        patch: Patch-level data (column indices, PFT types)
        col: Column-level data (depths, bedrock)
        pftcon: Plant functional type constants (rooting parameters)
        tower_data: Tower-specific soil data
        soil_texture: Soil texture lookup tables
        config: Configuration parameters
        
    Returns:
        SoilStateType with initialized time-constant properties
        
    Note:
        This is a pure function with no side effects. All state is immutable.
    """
    npatch = bounds.endp - bounds.begp + 1
    ncol = bounds.endc - bounds.begc + 1
    nlevgrnd = config.nlevgrnd
    nlevsoi = config.nlevsoi
    
    # ========================================================================
    # Step 1: Compute root profiles (lines 86-135)
    # ========================================================================
    
    def compute_patch_roots(p_idx: int) -> Array:
        """Compute root profile for a single patch."""
        c_idx = patch.column[p_idx]
        pft_type = patch.itype[p_idx]
        
        # Compute root profile
        rootfr_soil = compute_root_profile_for_patch(
            config.root_type,
            pft_type,
            pftcon.roota_par,
            pftcon.rootb_par,
            pftcon.rootprof_beta,
            col.zi[c_idx],
            nlevsoi,
        )
        
        # Pad to nlevgrnd with zeros
        rootfr_full = jnp.concatenate([
            rootfr_soil,
            jnp.zeros(nlevgrnd - nlevsoi)
        ])
        
        return rootfr_full
    
    # Vectorize over all patches
    rootfr = jax.vmap(compute_patch_roots)(jnp.arange(npatch))
    
    # ========================================================================
    # Step 2: Adjust roots for bedrock (lines 142-154)
    # ========================================================================
    
    rootfr = adjust_roots_for_bedrock(
        rootfr,
        col.nbedrock,
        patch.column,
        nlevsoi,
        nlevgrnd,
    )
    
    # ========================================================================
    # Step 3: Compute soil texture properties (lines 162-211)
    # ========================================================================
    
    om_frac, tex, clay, sand = compute_soil_texture_properties(
        tower_data,
        soil_texture,
        config.organic_max,
        ncol,
    )
    
    # Convert fractions to percentages (line 197-198)
    clay_pct = clay * 100.0
    sand_pct = sand * 100.0
    
    # ========================================================================
    # Step 4: Compute hydraulic and thermal properties (lines 200-330)
    # ========================================================================
    
    def compute_column_properties(c_idx: int) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        """Compute all properties for a single column."""
        
        def compute_layer_properties(j: int) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
            """Compute properties for a single layer."""
            
            # Hydraulic properties (lines 200-267)
            (cellsand_j, cellclay_j, cellorg_j, watsat_j, sucsat_j, 
             hksat_j, bsw_j, perc_frac_j) = compute_soil_hydraulic_properties_for_layer(
                j,
                tex[c_idx],
                sand_pct[c_idx],
                clay_pct[c_idx],
                om_frac[c_idx],
                col.z[c_idx, j],
                config.zsapric,
                config.organic_max,
                config.pcalpha,
                config.pcbeta,
                nlevsoi,
                soil_texture,
            )
            
            # Thermal properties (lines 268-285)
            tkdry_j = compute_thermal_conductivity_dry(
                om_frac[c_idx],
                watsat_j,
                j,
                nlevsoi,
                config.clm_phys,
            )
            
            # Compute quartz fraction for thermal conductivity of solids
            quartz_j = jnp.where(
                j < nlevsoi,
                cellsand_j / 100.0,
                0.0
            )
            
            # Organic matter fraction for thermal properties
            om_frac_therm_j = jnp.where(
                config.clm_phys != 0,
                0.02,
                jnp.where(col.z[c_idx, j] > config.zsapric, 0.0, om_frac[c_idx])
            )
            
            # Thermal conductivity and heat capacity (lines 287-330)
            tkmg_j, csol_j = compute_soil_thermal_properties(
                tex[c_idx],
                sand[c_idx],
                clay[c_idx],
                quartz_j,
                om_frac_therm_j,
                watsat_j,
                j,
                nlevsoi,
                config.csol_bedrock,
            )
            
            return (cellsand_j, cellclay_j, cellorg_j, watsat_j, sucsat_j,
                    hksat_j, bsw_j, perc_frac_j, tkdry_j, tkmg_j, csol_j)
        
        # Vectorize over all layers
        layer_results = jax.vmap(compute_layer_properties)(jnp.arange(nlevgrnd))
        
        return layer_results
    
    # Vectorize over all columns
    col_results = jax.vmap(compute_column_properties)(jnp.arange(ncol))
    
    # Unpack results
    cellsand = col_results[0][:, :nlevsoi]  # Only soil layers
    cellclay = col_results[1][:, :nlevsoi]  # Only soil layers
    cellorg = col_results[2][:, :nlevsoi]   # Only soil layers
    watsat = col_results[3]
    sucsat = col_results[4]
    hksat = col_results[5]
    bsw = col_results[6]
    perc_frac = col_results[7]
    tkdry = col_results[8]
    tkmg = col_results[9]
    csol = col_results[10]
    
    # ========================================================================
    # Return complete soil state
    # ========================================================================
    
    return SoilStateType(
        rootfr=rootfr,
        cellsand=cellsand,
        cellclay=cellclay,
        cellorg=cellorg,
        watsat=watsat,
        sucsat=sucsat,
        hksat=hksat,
        bsw=bsw,
        perc_frac=perc_frac,
        tkdry=tkdry,
        tkmg=tkmg,
        csol=csol,
    )


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    # Types
    'BoundsType',
    'PatchType',
    'ColumnType',
    'PftconType',
    'SoilTextureType',
    'TowerDataType',
    'SoilStateType',
    'SoilStateInitConfig',
    
    # Main function
    'soilstate_init_time_const',
    
    # Root distribution functions
    'compute_root_profile_zeng2001',
    'compute_root_profile_jackson1996',
    'compute_root_profile_for_patch',
    'adjust_roots_for_bedrock',
    
    # Soil texture functions
    'compute_soil_texture_properties',
    
    # Hydraulic properties functions
    'compute_soil_hydraulic_properties_for_layer',
    
    # Thermal properties functions
    'compute_thermal_conductivity_dry',
    'compute_soil_thermal_properties',
]# Backward compatibility alias (capitalize first letters)
SoilStateInitTimeConst = soilstate_init_time_const
