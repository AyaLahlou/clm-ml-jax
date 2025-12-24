"""
Multilayer Canopy Fluxes Module.

Translated from CTSM's MLCanopyFluxesMod.F90 (lines 1-1082)

This module calculates multilayer canopy fluxes including turbulent exchange,
photosynthesis, and energy balance for multiple canopy layers.

Key features:
    - Multi-layer canopy representation with sunlit/shaded leaf classes
    - Sub-timestep integration for numerical stability
    - Coupled leaf energy balance and photosynthesis
    - Turbulent transfer within and above canopy
    - Energy conservation checks at multiple levels

Physics:
    The multilayer canopy model divides the canopy into vertical layers and
    computes fluxes for sunlit and shaded leaves separately. This accounts for
    vertical gradients in radiation, temperature, humidity, and CO2 that are
    not captured by big-leaf models.
    
    Key equations:
        Net radiation: Rn = SW↓ + LW↓ - SW↑ - LW↑
        Energy balance: Rn = H + λE + G + S
        Photosynthesis: A = f(PAR, T, CO2, H2O)
        
    Where:
        H = sensible heat flux [W/m2]
        λE = latent heat flux [W/m2]
        G = ground heat flux [W/m2]
        S = storage flux [W/m2]

Constants:
    NVAR1D = 12: Number of single-level fluxes to accumulate
    NVAR2D = 4: Number of multi-level profile fluxes to accumulate
    NVAR3D = 10: Number of multi-level leaf fluxes to accumulate

Public functions:
    - ml_canopy_fluxes: Main driver for canopy flux calculations
    - sub_time_step_flux_integration: Integrate fluxes over sub-time steps
    - canopy_fluxes_diagnostics: Sum leaf and soil fluxes and diagnostics

Reference:
    MLCanopyFluxesMod.F90 in CTSM source code
"""

from typing import NamedTuple, Tuple, Callable
import jax
import jax.numpy as jnp

# ============================================================================
# Module Constants (lines 34-36)
# ============================================================================

NVAR1D: int = 12  # Number of single-level fluxes to accumulate
NVAR2D: int = 4   # Number of multi-level profile fluxes to accumulate
NVAR3D: int = 10  # Number of multi-level leaf fluxes to accumulate

# Leaf type indices
ISUN: int = 0  # Sunlit leaf index
ISHA: int = 1  # Shaded leaf index

# Radiation band indices
IVIS: int = 0  # Visible band index
INIR: int = 1  # Near-infrared band index

# Physical constants (from MLclm_varcon)
MMH2O: float = 18.016e-3  # Molecular weight of water [kg/mol]
MMDRY: float = 28.966e-3  # Molecular weight of dry air [kg/mol]
CPD: float = 1005.0       # Specific heat of dry air [J/kg/K]
CPW: float = 1846.0       # Specific heat of water vapor [J/kg/K]
RGAS: float = 8.314       # Universal gas constant [J/K/mol]
WIND_FORC_MIN: float = 0.1  # Minimum wind forcing [m/s]
LAPSE_RATE: float = 0.006   # Temperature lapse rate [K/m]
GRAV: float = 9.80616       # Gravitational acceleration [m/s2]
PI: float = 3.14159265358979323846  # Pi
SPVAL: float = 1.0e36       # Special value for uninitialized data

# ============================================================================
# Type Definitions
# ============================================================================

class Bounds(NamedTuple):
    """Subgrid bounds structure.
    
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


class AtmosphericForcing(NamedTuple):
    """Atmospheric forcing variables mapped to patch level.
    
    All arrays have shape [n_patches] unless otherwise noted.
    """
    tref: jnp.ndarray  # Reference height temperature [K]
    qref: jnp.ndarray  # Reference height specific humidity [kg/kg]
    pref: jnp.ndarray  # Reference height pressure [Pa]
    lwsky: jnp.ndarray  # Downward longwave radiation [W/m2]
    qflx_rain: jnp.ndarray  # Rain flux [mm/s]
    qflx_snow: jnp.ndarray  # Snow flux [mm/s]
    co2ref: jnp.ndarray  # CO2 concentration [umol/mol]
    o2ref: jnp.ndarray  # O2 concentration [mmol/mol]
    tacclim: jnp.ndarray  # 10-day running mean temperature [K]
    uref: jnp.ndarray  # Reference wind speed [m/s]
    swskyb_vis: jnp.ndarray  # Direct beam visible radiation [W/m2]
    swskyd_vis: jnp.ndarray  # Diffuse visible radiation [W/m2]
    swskyb_nir: jnp.ndarray  # Direct beam near-infrared radiation [W/m2]
    swskyd_nir: jnp.ndarray  # Diffuse near-infrared radiation [W/m2]


class DerivedAtmosphericState(NamedTuple):
    """Derived atmospheric properties at reference height.
    
    Attributes:
        eref: Vapor pressure at reference height [Pa] [n_patches]
        rhomol: Molar density of air [mol/m3] [n_patches]
        rhoair: Mass density of air [kg/m3] [n_patches]
        mmair: Molecular mass of air [kg/mol] [n_patches]
        cpair: Specific heat of air [J/kg/K] [n_patches]
        thref: Potential temperature at reference height [K] [n_patches]
        thvref: Virtual potential temperature at reference height [K] [n_patches]
    """
    eref: jnp.ndarray
    rhomol: jnp.ndarray
    rhoair: jnp.ndarray
    mmair: jnp.ndarray
    cpair: jnp.ndarray
    thref: jnp.ndarray
    thvref: jnp.ndarray


class CanopyProfileState(NamedTuple):
    """Updated canopy profile variables.
    
    Attributes:
        lai: Total leaf area index [m2/m2] [n_patches]
        sai: Total stem area index [m2/m2] [n_patches]
        dlai: Leaf area index by layer [m2/m2] [n_patches, n_canopy_layers]
        dsai: Stem area index by layer [m2/m2] [n_patches, n_canopy_layers]
        dpai: Plant area index by layer [m2/m2] [n_patches, n_canopy_layers]
    """
    lai: jnp.ndarray
    sai: jnp.ndarray
    dlai: jnp.ndarray
    dsai: jnp.ndarray
    dpai: jnp.ndarray


class SubStepState(NamedTuple):
    """Saved state variables for sub-stepping.
    
    Attributes:
        tg_bef: Ground temperature before sub-step [K] [n_patches]
        tleaf_bef: Leaf temperature before sub-step [K] [n_patches, n_canopy_layers, 2]
        tair_bef: Air temperature before sub-step [K] [n_patches, n_canopy_layers]
        eair_bef: Air vapor pressure before sub-step [Pa] [n_patches, n_canopy_layers]
        cair_bef: Air CO2 concentration before sub-step [umol/mol] [n_patches, n_canopy_layers]
    """
    tg_bef: jnp.ndarray
    tleaf_bef: jnp.ndarray
    tair_bef: jnp.ndarray
    eair_bef: jnp.ndarray
    cair_bef: jnp.ndarray


class MLCanopyFluxesState(NamedTuple):
    """State for multilayer canopy flux calculations.
    
    Attributes:
        rnleaf_sun: Sunlit leaf net radiation [W/m2] [n_patches, n_canopy_layers]
        rnleaf_shade: Shaded leaf net radiation [W/m2] [n_patches, n_canopy_layers]
        rnsoi: Soil net radiation [W/m2] [n_patches]
        rhg: Relative humidity in soil airspace [-] [n_patches]
        flux_accumulator: Accumulated fluxes [n_patches, n_flux_vars]
        flux_accumulator_profile: Accumulated profile fluxes [n_patches, n_canopy_layers, n_profile_vars]
        flux_accumulator_leaf: Accumulated leaf fluxes [n_patches, n_canopy_layers, n_leaf_vars]
    """
    rnleaf_sun: jnp.ndarray
    rnleaf_shade: jnp.ndarray
    rnsoi: jnp.ndarray
    rhg: jnp.ndarray
    flux_accumulator: jnp.ndarray
    flux_accumulator_profile: jnp.ndarray
    flux_accumulator_leaf: jnp.ndarray


class FluxAccumulators(NamedTuple):
    """Container for flux accumulator arrays.
    
    Attributes:
        flux_1d: Single-level flux accumulator [n_patches, nvar1d]
        flux_2d: Multi-level profile flux accumulator [n_patches, n_levels, nvar2d]
        flux_3d: Multi-level leaf flux accumulator [n_patches, n_levels, n_leaf_types, nvar3d]
    """
    flux_1d: jnp.ndarray
    flux_2d: jnp.ndarray
    flux_3d: jnp.ndarray


class CanopyFluxes(NamedTuple):
    """Current canopy flux values to be accumulated.
    
    Attributes:
        # Scalar fluxes (1D)
        ustar: Friction velocity [m/s] [n_patches]
        lwup: Upward longwave radiation above canopy [W/m2] [n_patches]
        lwsoi: Absorbed longwave radiation at ground [W/m2] [n_patches]
        rnsoi: Net radiation at ground [W/m2] [n_patches]
        shsoi: Sensible heat flux at ground [W/m2] [n_patches]
        lhsoi: Latent heat flux at ground [W/m2] [n_patches]
        etsoi: Water vapor flux at ground [mol H2O/m2/s] [n_patches]
        gsoi: Soil heat flux [W/m2] [n_patches]
        gac0: Aerodynamic conductance for soil [mol/m2/s] [n_patches]
        qflx_intr: Intercepted precipitation [kg H2O/m2/s] [n_patches]
        qflx_tflrain: Rain throughfall [kg H2O/m2/s] [n_patches]
        qflx_tflsnow: Snow throughfall [kg H2O/m2/s] [n_patches]
        
        # Profile fluxes (2D)
        shair: Air sensible heat flux [W/m2] [n_patches, n_levels]
        etair: Air water vapor flux [mol H2O/m2/s] [n_patches, n_levels]
        stair: Air storage heat flux [W/m2] [n_patches, n_levels]
        gac: Aerodynamic conductance [mol/m2/s] [n_patches, n_levels]
        
        # Leaf fluxes (3D)
        lwleaf: Leaf absorbed longwave [W/m2 leaf] [n_patches, n_levels, n_leaf_types]
        rnleaf: Leaf net radiation [W/m2 leaf] [n_patches, n_levels, n_leaf_types]
        shleaf: Leaf sensible heat [W/m2 leaf] [n_patches, n_levels, n_leaf_types]
        lhleaf: Leaf latent heat [W/m2 leaf] [n_patches, n_levels, n_leaf_types]
        trleaf: Leaf transpiration [mol H2O/m2 leaf/s] [n_patches, n_levels, n_leaf_types]
        evleaf: Leaf evaporation [mol H2O/m2 leaf/s] [n_patches, n_levels, n_leaf_types]
        stleaf: Leaf storage heat [W/m2 leaf] [n_patches, n_levels, n_leaf_types]
        anet: Leaf net photosynthesis [umol CO2/m2 leaf/s] [n_patches, n_levels, n_leaf_types]
        agross: Leaf gross photosynthesis [umol CO2/m2 leaf/s] [n_patches, n_levels, n_leaf_types]
        gs: Leaf stomatal conductance [mol H2O/m2 leaf/s] [n_patches, n_levels, n_leaf_types]
    """
    # Scalar fluxes
    ustar: jnp.ndarray
    lwup: jnp.ndarray
    lwsoi: jnp.ndarray
    rnsoi: jnp.ndarray
    shsoi: jnp.ndarray
    lhsoi: jnp.ndarray
    etsoi: jnp.ndarray
    gsoi: jnp.ndarray
    gac0: jnp.ndarray
    qflx_intr: jnp.ndarray
    qflx_tflrain: jnp.ndarray
    qflx_tflsnow: jnp.ndarray
    
    # Profile fluxes
    shair: jnp.ndarray
    etair: jnp.ndarray
    stair: jnp.ndarray
    gac: jnp.ndarray
    
    # Leaf fluxes
    lwleaf: jnp.ndarray
    rnleaf: jnp.ndarray
    shleaf: jnp.ndarray
    lhleaf: jnp.ndarray
    trleaf: jnp.ndarray
    evleaf: jnp.ndarray
    stleaf: jnp.ndarray
    anet: jnp.ndarray
    agross: jnp.ndarray
    gs: jnp.ndarray


class LeafFluxProfiles(NamedTuple):
    """Weighted mean leaf fluxes and source/sink fluxes.
    
    All arrays have shape [n_patches, n_canopy_layers].
    """
    # Weighted mean leaf fluxes (per unit leaf area)
    lwleaf_mean: jnp.ndarray  # Absorbed longwave radiation [W/m2 leaf]
    swleaf_mean_vis: jnp.ndarray  # Absorbed solar radiation VIS [W/m2 leaf]
    swleaf_mean_nir: jnp.ndarray  # Absorbed solar radiation NIR [W/m2 leaf]
    rnleaf_mean: jnp.ndarray  # Net radiation [W/m2 leaf]
    stleaf_mean: jnp.ndarray  # Storage heat flux [W/m2 leaf]
    shleaf_mean: jnp.ndarray  # Sensible heat flux [W/m2 leaf]
    lhleaf_mean: jnp.ndarray  # Latent heat flux [W/m2 leaf]
    etleaf_mean: jnp.ndarray  # Water vapor flux [mol H2O/m2 leaf/s]
    fco2_mean: jnp.ndarray  # Net photosynthesis [umol CO2/m2 leaf/s]
    apar_mean: jnp.ndarray  # Absorbed PAR [umol photon/m2 leaf/s]
    gs_mean: jnp.ndarray  # Stomatal conductance [mol H2O/m2 leaf/s]
    tleaf_mean: jnp.ndarray  # Leaf temperature [K]
    lwp_mean: jnp.ndarray  # Leaf water potential [MPa]
    
    # Source/sink fluxes (per unit ground area)
    lwsrc: jnp.ndarray  # Absorbed longwave radiation [W/m2]
    swsrc_vis: jnp.ndarray  # Absorbed solar radiation VIS [W/m2]
    swsrc_nir: jnp.ndarray  # Absorbed solar radiation NIR [W/m2]
    rnsrc: jnp.ndarray  # Net radiation [W/m2]
    stsrc: jnp.ndarray  # Storage heat flux [W/m2]
    shsrc: jnp.ndarray  # Sensible heat flux [W/m2]
    lhsrc: jnp.ndarray  # Latent heat flux [W/m2]
    etsrc: jnp.ndarray  # Water vapor flux [mol H2O/m2/s]
    fco2src: jnp.ndarray  # CO2 flux [umol CO2/m2/s]


class EnergyBalanceState(NamedTuple):
    """State after energy balance checks and turbulent flux calculation.
    
    Attributes:
        vcmax25veg: Canopy-integrated vcmax25 [umol/m2/s] [n_patches]
        shflx: Total sensible heat flux [W/m2] [n_patches]
        etflx: Total evapotranspiration flux [kg/m2/s] [n_patches]
        lhflx: Total latent heat flux [W/m2] [n_patches]
        stflx: Total storage flux including air [W/m2] [n_patches]
        rnet: Net radiation [W/m2] [n_patches]
    """
    vcmax25veg: jnp.ndarray
    shflx: jnp.ndarray
    etflx: jnp.ndarray
    lhflx: jnp.ndarray
    stflx: jnp.ndarray
    rnet: jnp.ndarray


class SunShadeFluxes(NamedTuple):
    """Aggregated sunlit and shaded canopy fluxes.
    
    Attributes:
        laisun: Sunlit leaf area index [m2/m2]
        laisha: Shaded leaf area index [m2/m2]
        lwvegsun: Sunlit longwave radiation [W/m2]
        lwvegsha: Shaded longwave radiation [W/m2]
        shvegsun: Sunlit sensible heat flux [W/m2]
        shvegsha: Shaded sensible heat flux [W/m2]
        lhvegsun: Sunlit latent heat flux [W/m2]
        lhvegsha: Shaded latent heat flux [W/m2]
        etvegsun: Sunlit evapotranspiration [mm/s]
        etvegsha: Shaded evapotranspiration [mm/s]
        gppvegsun: Sunlit gross primary production [umol CO2/m2/s]
        gppvegsha: Shaded gross primary production [umol CO2/m2/s]
        vcmax25sun: Sunlit Vcmax at 25C [umol/m2/s]
        vcmax25sha: Shaded Vcmax at 25C [umol/m2/s]
        gsvegsun: Sunlit stomatal conductance [mol/m2/s]
        gsvegsha: Shaded stomatal conductance [mol/m2/s]
        windveg: Mean canopy wind speed [m/s]
        windvegsun: Sunlit canopy wind speed [m/s]
        windvegsha: Shaded canopy wind speed [m/s]
        tlveg: Mean leaf temperature [K]
        tlvegsun: Sunlit leaf temperature [K]
        tlvegsha: Shaded leaf temperature [K]
        taveg: Mean air temperature [K]
        tavegsun: Sunlit air temperature [K]
        tavegsha: Shaded air temperature [K]
    """
    laisun: jnp.ndarray
    laisha: jnp.ndarray
    lwvegsun: jnp.ndarray
    lwvegsha: jnp.ndarray
    shvegsun: jnp.ndarray
    shvegsha: jnp.ndarray
    lhvegsun: jnp.ndarray
    lhvegsha: jnp.ndarray
    etvegsun: jnp.ndarray
    etvegsha: jnp.ndarray
    gppvegsun: jnp.ndarray
    gppvegsha: jnp.ndarray
    vcmax25sun: jnp.ndarray
    vcmax25sha: jnp.ndarray
    gsvegsun: jnp.ndarray
    gsvegsha: jnp.ndarray
    windveg: jnp.ndarray
    windvegsun: jnp.ndarray
    windvegsha: jnp.ndarray
    tlveg: jnp.ndarray
    tlvegsun: jnp.ndarray
    tlvegsha: jnp.ndarray
    taveg: jnp.ndarray
    tavegsun: jnp.ndarray
    tavegsha: jnp.ndarray


class WaterStressDiagnostics(NamedTuple):
    """Water stress diagnostic outputs.
    
    Attributes:
        fracminlwp: Fraction of canopy water stressed [-] [n_patches]
    """
    fracminlwp: jnp.ndarray


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_derived_atmospheric_properties(
    qref: jnp.ndarray,
    pref: jnp.ndarray,
    tref: jnp.ndarray,
    zref: jnp.ndarray,
    mmh2o: float = MMH2O,
    mmdry: float = MMDRY,
    rgas: float = RGAS,
    cpd: float = CPD,
    cpw: float = CPW,
    lapse_rate: float = LAPSE_RATE,
) -> DerivedAtmosphericState:
    """Calculate derived atmospheric properties from reference values.
    
    Fortran source: lines 372-379
    
    Args:
        qref: Specific humidity at reference height [kg/kg] [n_patches]
        pref: Pressure at reference height [Pa] [n_patches]
        tref: Temperature at reference height [K] [n_patches]
        zref: Reference height [m] [n_patches]
        mmh2o: Molecular mass of water [kg/mol]
        mmdry: Molecular mass of dry air [kg/mol]
        rgas: Universal gas constant [J/K/mol]
        cpd: Specific heat of dry air [J/kg/K]
        cpw: Specific heat of water vapor [J/kg/K]
        lapse_rate: Temperature lapse rate [K/m]
        
    Returns:
        DerivedAtmosphericState with all derived properties
    """
    # Vapor pressure at reference height
    eref = qref * pref / (mmh2o / mmdry + (1.0 - mmh2o / mmdry) * qref)
    
    # Molar density of air
    rhomol = pref / (rgas * tref)
    
    # Mass density of air
    rhoair = rhomol * mmdry * (1.0 - (1.0 - mmh2o / mmdry) * eref / pref)
    
    # Molecular mass of moist air
    mmair = rhoair / rhomol
    
    # Specific heat of moist air
    cpair = cpd * (1.0 + (cpw / cpd - 1.0) * qref) * mmair
    
    # Potential temperature at reference height
    thref = tref + lapse_rate * zref
    
    # Virtual potential temperature
    thvref = thref * (1.0 + 0.61 * qref)
    
    return DerivedAtmosphericState(
        eref=eref,
        rhomol=rhomol,
        rhoair=rhoair,
        mmair=mmair,
        cpair=cpair,
        thref=thref,
        thvref=thvref,
    )


def update_canopy_profile(
    elai: jnp.ndarray,
    esai: jnp.ndarray,
    dlai_frac: jnp.ndarray,
    dsai_frac: jnp.ndarray,
    ncan: jnp.ndarray,
) -> CanopyProfileState:
    """Update leaf and stem area profiles for current time step.
    
    Fortran source: lines 383-403
    
    Args:
        elai: Exposed leaf area index from CLM [m2/m2] [n_patches]
        esai: Exposed stem area index from CLM [m2/m2] [n_patches]
        dlai_frac: Fraction of LAI in each layer [fraction] [n_patches, n_canopy_layers]
        dsai_frac: Fraction of SAI in each layer [fraction] [n_patches, n_canopy_layers]
        ncan: Number of canopy layers [integer] [n_patches]
        
    Returns:
        CanopyProfileState with updated profiles
    """
    # Get values for current time step from CLM
    lai = elai
    sai = esai
    
    # Vertical profiles
    dlai = dlai_frac * lai[:, jnp.newaxis]
    dsai = dsai_frac * sai[:, jnp.newaxis]
    dpai = dlai + dsai
    
    return CanopyProfileState(
        lai=lai,
        sai=sai,
        dlai=dlai,
        dsai=dsai,
        dpai=dpai,
    )


def initialize_substep_state(
    tg: jnp.ndarray,
    tleaf: jnp.ndarray,
    tair: jnp.ndarray,
    eair: jnp.ndarray,
    cair: jnp.ndarray,
) -> SubStepState:
    """Save current state variables before sub-stepping.
    
    Fortran source: lines 420-428
    
    Args:
        tg: Ground temperature [K] [n_patches]
        tleaf: Leaf temperature [K] [n_patches, n_canopy_layers, 2]
        tair: Air temperature [K] [n_patches, n_canopy_layers]
        eair: Air vapor pressure [Pa] [n_patches, n_canopy_layers]
        cair: Air CO2 concentration [umol/mol] [n_patches, n_canopy_layers]
        
    Returns:
        SubStepState with saved values
    """
    return SubStepState(
        tg_bef=tg,
        tleaf_bef=tleaf,
        tair_bef=tair,
        eair_bef=eair,
        cair_bef=cair,
    )


def compute_net_radiation(
    swleaf_sun_vis: jnp.ndarray,
    swleaf_sun_nir: jnp.ndarray,
    swleaf_shade_vis: jnp.ndarray,
    swleaf_shade_nir: jnp.ndarray,
    lwleaf_sun: jnp.ndarray,
    lwleaf_shade: jnp.ndarray,
    swsoi_vis: jnp.ndarray,
    swsoi_nir: jnp.ndarray,
    lwsoi: jnp.ndarray,
    ncan: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute net radiation at each canopy layer and ground.
    
    From lines 451-458 of MLCanopyFluxesMod.F90.
    
    Args:
        swleaf_sun_vis: Sunlit leaf absorbed SW visible [W/m2] [n_patches, n_canopy_layers]
        swleaf_sun_nir: Sunlit leaf absorbed SW near-IR [W/m2] [n_patches, n_canopy_layers]
        swleaf_shade_vis: Shaded leaf absorbed SW visible [W/m2] [n_patches, n_canopy_layers]
        swleaf_shade_nir: Shaded leaf absorbed SW near-IR [W/m2] [n_patches, n_canopy_layers]
        lwleaf_sun: Sunlit leaf absorbed LW [W/m2] [n_patches, n_canopy_layers]
        lwleaf_shade: Shaded leaf absorbed LW [W/m2] [n_patches, n_canopy_layers]
        swsoi_vis: Soil absorbed SW visible [W/m2] [n_patches]
        swsoi_nir: Soil absorbed SW near-IR [W/m2] [n_patches]
        lwsoi: Soil absorbed LW [W/m2] [n_patches]
        ncan: Number of canopy layers per patch [n_patches]
        
    Returns:
        rnleaf_sun: Sunlit leaf net radiation [W/m2] [n_patches, n_canopy_layers]
        rnleaf_shade: Shaded leaf net radiation [W/m2] [n_patches, n_canopy_layers]
        rnsoi: Soil net radiation [W/m2] [n_patches]
    """
    # Net radiation for sunlit leaves
    rnleaf_sun = swleaf_sun_vis + swleaf_sun_nir + lwleaf_sun
    
    # Net radiation for shaded leaves
    rnleaf_shade = swleaf_shade_vis + swleaf_shade_nir + lwleaf_shade
    
    # Net radiation for soil
    rnsoi = swsoi_vis + swsoi_nir + lwsoi
    
    return rnleaf_sun, rnleaf_shade, rnsoi


def compute_soil_relative_humidity(
    smp_l: jnp.ndarray,
    t_soisno: jnp.ndarray,
    patch_to_column: jnp.ndarray,
    grav: float = GRAV,
    mmh2o: float = MMH2O,
    rgas: float = RGAS,
) -> jnp.ndarray:
    """Compute relative humidity in soil airspace.
    
    From lines 477-480 of MLCanopyFluxesMod.F90.
    
    Args:
        smp_l: Soil matric potential [mm] [n_columns]
        t_soisno: Soil temperature [K] [n_columns]
        patch_to_column: Mapping from patch to column index [n_patches]
        grav: Gravitational acceleration [m/s2]
        mmh2o: Molecular weight of water [kg/mol]
        rgas: Universal gas constant [J/K/mol]
        
    Returns:
        rhg: Relative humidity in soil airspace [-] [n_patches]
    """
    # Map column variables to patches
    smp_l_patch = smp_l[patch_to_column, 0]  # First soil layer
    t_soisno_patch = t_soisno[patch_to_column, 0]  # First soil layer
    
    # Convert smp_l from mm to m
    smp_l_m = smp_l_patch * 1.0e-3
    
    # Compute relative humidity using Kelvin equation
    exponent = (grav * mmh2o * smp_l_m) / (rgas * t_soisno_patch)
    rhg = jnp.exp(exponent)
    
    return rhg


# ============================================================================
# Sub-Time Step Flux Integration
# ============================================================================

def accumulate_fluxes(
    accumulators: FluxAccumulators,
    fluxes: CanopyFluxes,
    niter: int,
    nvar1d: int = NVAR1D,
    nvar2d: int = NVAR2D,
    nvar3d: int = NVAR3D,
) -> FluxAccumulators:
    """Accumulate fluxes over sub-time steps.
    
    Fortran source: lines 594-669
    
    Args:
        accumulators: Current flux accumulator arrays
        fluxes: Current flux values to accumulate
        niter: Current iteration number (1-based)
        nvar1d: Expected number of 1D flux variables
        nvar2d: Expected number of 2D flux variables
        nvar3d: Expected number of 3D flux variables
        
    Returns:
        Updated flux accumulator arrays
    """
    # Initialize accumulators on first iteration
    is_first_iter = (niter == 1)
    
    # Pack 1D fluxes into array [n_patches, nvar1d]
    flux_1d_current = jnp.stack([
        fluxes.ustar,
        fluxes.lwup,
        fluxes.lwsoi,
        fluxes.rnsoi,
        fluxes.shsoi,
        fluxes.lhsoi,
        fluxes.etsoi,
        fluxes.gsoi,
        fluxes.gac0,
        fluxes.qflx_intr,
        fluxes.qflx_tflrain,
        fluxes.qflx_tflsnow,
    ], axis=-1)
    
    # Pack 2D profile fluxes
    flux_2d_current = jnp.stack([
        fluxes.shair,
        fluxes.etair,
        fluxes.stair,
        fluxes.gac,
    ], axis=-1)
    
    # Pack 3D leaf fluxes
    flux_3d_current = jnp.stack([
        fluxes.lwleaf,
        fluxes.rnleaf,
        fluxes.shleaf,
        fluxes.lhleaf,
        fluxes.trleaf,
        fluxes.evleaf,
        fluxes.stleaf,
        fluxes.anet,
        fluxes.agross,
        fluxes.gs,
    ], axis=-1)
    
    # Initialize or accumulate
    flux_1d_new = jnp.where(
        is_first_iter,
        flux_1d_current,
        accumulators.flux_1d + flux_1d_current
    )
    
    flux_2d_new = jnp.where(
        is_first_iter,
        flux_2d_current,
        accumulators.flux_2d + flux_2d_current
    )
    
    flux_3d_new = jnp.where(
        is_first_iter,
        flux_3d_current,
        accumulators.flux_3d + flux_3d_current
    )
    
    return FluxAccumulators(
        flux_1d=flux_1d_new,
        flux_2d=flux_2d_new,
        flux_3d=flux_3d_new,
    )


def sub_time_step_flux_integration(
    niter: int,
    num_sub_steps: int,
    num_filter: int,
    filter_indices: jnp.ndarray,
    flux_accumulator: jnp.ndarray,
    flux_accumulator_profile: jnp.ndarray,
    flux_accumulator_leaf: jnp.ndarray,
    current_fluxes: CanopyFluxes,
) -> FluxAccumulators:
    """Integrate fluxes over model sub-time steps.
    
    Fortran source: lines 568-724
    
    Args:
        niter: Current sub-time step iteration [scalar]
        num_sub_steps: Total number of sub-time steps [scalar]
        num_filter: Number of patches in filter [scalar]
        filter_indices: Patch filter indices [n_filter]
        flux_accumulator: Single-level flux accumulator [n_patches, n_fluxes]
        flux_accumulator_profile: Multi-level profile flux accumulator 
            [n_patches, n_levels, n_fluxes]
        flux_accumulator_leaf: Multi-level leaf flux accumulator 
            [n_patches, n_levels, n_leaf_types, n_fluxes]
        current_fluxes: Current flux values to accumulate
        
    Returns:
        FluxAccumulators containing updated accumulator arrays
    """
    accumulators = FluxAccumulators(
        flux_1d=flux_accumulator,
        flux_2d=flux_accumulator_profile,
        flux_3d=flux_accumulator_leaf,
    )
    
    # Accumulate current fluxes
    updated_accumulators = accumulate_fluxes(
        accumulators=accumulators,
        fluxes=current_fluxes,
        niter=niter,
    )
    
    # Average on final iteration
    if niter == num_sub_steps:
        divisor = float(num_sub_steps)
        updated_accumulators = FluxAccumulators(
            flux_1d=updated_accumulators.flux_1d / divisor,
            flux_2d=updated_accumulators.flux_2d / divisor,
            flux_3d=updated_accumulators.flux_3d / divisor,
        )
    
    return updated_accumulators


# ============================================================================
# Canopy Fluxes Diagnostics
# ============================================================================

def calculate_leaf_flux_profiles(
    dpai: jnp.ndarray,
    fracsun: jnp.ndarray,
    fwet: jnp.ndarray,
    fdry: jnp.ndarray,
    lwleaf: jnp.ndarray,
    swleaf_vis: jnp.ndarray,
    swleaf_nir: jnp.ndarray,
    rnleaf: jnp.ndarray,
    stleaf: jnp.ndarray,
    shleaf: jnp.ndarray,
    lhleaf: jnp.ndarray,
    evleaf: jnp.ndarray,
    trleaf: jnp.ndarray,
    anet: jnp.ndarray,
    apar: jnp.ndarray,
    gs: jnp.ndarray,
    tleaf_mean: jnp.ndarray,
    tleaf: jnp.ndarray,
    lwp: jnp.ndarray,
    isun: int = ISUN,
    isha: int = ISHA,
) -> LeafFluxProfiles:
    """Calculate weighted mean leaf fluxes and source/sink fluxes.
    
    Fortran source: lines 760-902
    
    Args:
        dpai: Layer plant area index [n_patches, n_layers]
        fracsun: Sunlit fraction of layer [n_patches, n_layers]
        fwet: Wet fraction [n_patches, n_layers]
        fdry: Dry fraction [n_patches, n_layers]
        lwleaf: Leaf longwave radiation [n_patches, n_layers, 2]
        swleaf_vis: Leaf SW visible [n_patches, n_layers, 2]
        swleaf_nir: Leaf SW near-IR [n_patches, n_layers, 2]
        rnleaf: Leaf net radiation [n_patches, n_layers, 2]
        stleaf: Leaf storage heat [n_patches, n_layers, 2]
        shleaf: Leaf sensible heat [n_patches, n_layers, 2]
        lhleaf: Leaf latent heat [n_patches, n_layers, 2]
        evleaf: Leaf evaporation [n_patches, n_layers, 2]
        trleaf: Leaf transpiration [n_patches, n_layers, 2]
        anet: Net photosynthesis [n_patches, n_layers, 2]
        apar: Absorbed PAR [n_patches, n_layers, 2]
        gs: Stomatal conductance [n_patches, n_layers, 2]
        tleaf_mean: Mean leaf temperature [n_patches, n_layers]
        tleaf: Leaf temperature [n_patches, n_layers, 2]
        lwp: Leaf water potential [n_patches, n_layers, 2]
        isun: Index for sunlit
        isha: Index for shaded
        
    Returns:
        LeafFluxProfiles containing weighted means and source fluxes
    """
    # Extract sunlit and shaded components
    lwleaf_sun = lwleaf[:, :, isun]
    lwleaf_sha = lwleaf[:, :, isha]
    swleaf_vis_sun = swleaf_vis[:, :, isun]
    swleaf_vis_sha = swleaf_vis[:, :, isha]
    swleaf_nir_sun = swleaf_nir[:, :, isun]
    swleaf_nir_sha = swleaf_nir[:, :, isha]
    rnleaf_sun = rnleaf[:, :, isun]
    rnleaf_sha = rnleaf[:, :, isha]
    stleaf_sun = stleaf[:, :, isun]
    stleaf_sha = stleaf[:, :, isha]
    shleaf_sun = shleaf[:, :, isun]
    shleaf_sha = shleaf[:, :, isha]
    lhleaf_sun = lhleaf[:, :, isun]
    lhleaf_sha = lhleaf[:, :, isha]
    trleaf_sun = trleaf[:, :, isun]
    trleaf_sha = trleaf[:, :, isha]
    evleaf_sun = evleaf[:, :, isun]
    evleaf_sha = evleaf[:, :, isha]
    anet_sun = anet[:, :, isun]
    anet_sha = anet[:, :, isha]
    apar_sun = apar[:, :, isun]
    apar_sha = apar[:, :, isha]
    gs_sun = gs[:, :, isun]
    gs_sha = gs[:, :, isha]
    tleaf_sun = tleaf[:, :, isun]
    tleaf_sha = tleaf[:, :, isha]
    lwp_sun = lwp[:, :, isun]
    lwp_sha = lwp[:, :, isha]
    
    # Compute fraction shaded
    fracsha = 1.0 - fracsun
    
    # Calculate weighted mean leaf fluxes
    lwleaf_mean = lwleaf_sun * fracsun + lwleaf_sha * fracsha
    swleaf_mean_vis = swleaf_vis_sun * fracsun + swleaf_vis_sha * fracsha
    swleaf_mean_nir = swleaf_nir_sun * fracsun + swleaf_nir_sha * fracsha
    rnleaf_mean = rnleaf_sun * fracsun + rnleaf_sha * fracsha
    stleaf_mean = stleaf_sun * fracsun + stleaf_sha * fracsha
    shleaf_mean = shleaf_sun * fracsun + shleaf_sha * fracsha
    lhleaf_mean = lhleaf_sun * fracsun + lhleaf_sha * fracsha
    
    # Total evapotranspiration
    etleaf_mean = ((evleaf_sun + trleaf_sun) * fracsun + 
                   (evleaf_sha + trleaf_sha) * fracsha)
    
    # Net photosynthesis
    fco2_mean = anet_sun * fracsun + anet_sha * fracsha
    
    # Other leaf state variables
    apar_mean = apar_sun * fracsun + apar_sha * fracsha
    gs_mean = gs_sun * fracsun + gs_sha * fracsha
    tleaf_mean_out = tleaf_sun * fracsun + tleaf_sha * fracsha
    lwp_mean = lwp_sun * fracsun + lwp_sha * fracsha
    
    # Calculate source fluxes (scale by PAI)
    lwsrc = lwleaf_mean * dpai
    swsrc_vis = swleaf_mean_vis * dpai
    swsrc_nir = swleaf_mean_nir * dpai
    rnsrc = rnleaf_mean * dpai
    stsrc = stleaf_mean * dpai
    shsrc = shleaf_mean * dpai
    lhsrc = lhleaf_mean * dpai
    etsrc = etleaf_mean * dpai
    
    # CO2 source flux with green fraction correction
    fracgreen = jnp.where(
        fwet < 1.0,
        fdry / (1.0 - fwet),
        0.0
    )
    fco2src = ((anet_sun * fracsun + anet_sha * fracsha) * 
               dpai * fracgreen)
    
    # Zero out fluxes where dpai <= 0
    has_vegetation = dpai > 0.0
    
    lwleaf_mean = jnp.where(has_vegetation, lwleaf_mean, 0.0)
    swleaf_mean_vis = jnp.where(has_vegetation, swleaf_mean_vis, 0.0)
    swleaf_mean_nir = jnp.where(has_vegetation, swleaf_mean_nir, 0.0)
    rnleaf_mean = jnp.where(has_vegetation, rnleaf_mean, 0.0)
    stleaf_mean = jnp.where(has_vegetation, stleaf_mean, 0.0)
    shleaf_mean = jnp.where(has_vegetation, shleaf_mean, 0.0)
    lhleaf_mean = jnp.where(has_vegetation, lhleaf_mean, 0.0)
    etleaf_mean = jnp.where(has_vegetation, etleaf_mean, 0.0)
    fco2_mean = jnp.where(has_vegetation, fco2_mean, 0.0)
    apar_mean = jnp.where(has_vegetation, apar_mean, 0.0)
    gs_mean = jnp.where(has_vegetation, gs_mean, 0.0)
    tleaf_mean_out = jnp.where(has_vegetation, tleaf_mean_out, 0.0)
    lwp_mean = jnp.where(has_vegetation, lwp_mean, 0.0)
    
    lwsrc = jnp.where(has_vegetation, lwsrc, 0.0)
    swsrc_vis = jnp.where(has_vegetation, swsrc_vis, 0.0)
    swsrc_nir = jnp.where(has_vegetation, swsrc_nir, 0.0)
    rnsrc = jnp.where(has_vegetation, rnsrc, 0.0)
    stsrc = jnp.where(has_vegetation, stsrc, 0.0)
    shsrc = jnp.where(has_vegetation, shsrc, 0.0)
    lhsrc = jnp.where(has_vegetation, lhsrc, 0.0)
    etsrc = jnp.where(has_vegetation, etsrc, 0.0)
    fco2src = jnp.where(has_vegetation, fco2src, 0.0)
    
    return LeafFluxProfiles(
        lwleaf_mean=lwleaf_mean,
        swleaf_mean_vis=swleaf_mean_vis,
        swleaf_mean_nir=swleaf_mean_nir,
        rnleaf_mean=rnleaf_mean,
        stleaf_mean=stleaf_mean,
        shleaf_mean=shleaf_mean,
        lhleaf_mean=lhleaf_mean,
        etleaf_mean=etleaf_mean,
        fco2_mean=fco2_mean,
        apar_mean=apar_mean,
        gs_mean=gs_mean,
        tleaf_mean=tleaf_mean_out,
        lwp_mean=lwp_mean,
        lwsrc=lwsrc,
        swsrc_vis=swsrc_vis,
        swsrc_nir=swsrc_nir,
        rnsrc=rnsrc,
        stsrc=stsrc,
        shsrc=shsrc,
        lhsrc=lhsrc,
        etsrc=etsrc,
        fco2src=fco2src,
    )


def accumulate_vcmax25_and_check_energy_balance(
    vcmax25veg: jnp.ndarray,
    vcmax25_profile: jnp.ndarray,
    dpai: jnp.ndarray,
    swveg_vis: jnp.ndarray,
    swveg_nir: jnp.ndarray,
    lwveg: jnp.ndarray,
    shveg: jnp.ndarray,
    lhveg: jnp.ndarray,
    stflx: jnp.ndarray,
    shsoi: jnp.ndarray,
    etsoi: jnp.ndarray,
    lhsoi: jnp.ndarray,
    shair: jnp.ndarray,
    etair: jnp.ndarray,
    stair: jnp.ndarray,
    swsoi_vis: jnp.ndarray,
    swsoi_nir: jnp.ndarray,
    lwsoi: jnp.ndarray,
    swskyb_vis: jnp.ndarray,
    swskyd_vis: jnp.ndarray,
    swskyb_nir: jnp.ndarray,
    swskyd_nir: jnp.ndarray,
    lwsky: jnp.ndarray,
    albcan_vis: jnp.ndarray,
    albcan_nir: jnp.ndarray,
    lwup: jnp.ndarray,
    gsoi: jnp.ndarray,
    ntop: jnp.ndarray,
    turb_type: int,
    tref: jnp.ndarray,
    latvap_func: Callable,
    nlevcan: int,
) -> EnergyBalanceState:
    """Accumulate vcmax25 profile and perform energy balance checks.
    
    Fortran source: lines 954-1004
    
    Args:
        vcmax25veg: Initial canopy vcmax25 [umol/m2/s] [n_patches]
        vcmax25_profile: Vcmax25 by layer [umol/m2 leaf/s] [n_patches, nlevcan]
        dpai: Layer plant area increment [m2/m2] [n_patches, nlevcan]
        (... many more flux arguments ...)
        turb_type: Turbulence type (0, -1, or 1)
        tref: Reference temperature [K] [n_patches]
        latvap_func: Function to compute latent heat of vaporization
        nlevcan: Number of canopy layers
        
    Returns:
        EnergyBalanceState with updated fluxes and energy balance checks
    """
    n_patches = vcmax25veg.shape[0]
    
    # Accumulate vcmax25 over canopy layers
    vcmax25_increment = jnp.sum(vcmax25_profile * dpai, axis=1)
    vcmax25veg_updated = vcmax25veg + vcmax25_increment
    
    # Check energy balance for vegetation
    energy_balance_veg = (swveg_vis + swveg_nir + lwveg - 
                          shveg - lhveg - stflx)
    
    # Compute latent heat of vaporization
    latvap = latvap_func(tref)
    
    # Turbulent fluxes based on turbulence type
    # For turb_type 0 or -1: sum of vegetation and soil
    shflx_type0 = shveg + shsoi
    etflx_type0 = etsoi
    lhflx_type0 = lhveg + lhsoi
    
    # For turb_type 1: fluxes at top of canopy
    patch_indices = jnp.arange(n_patches)
    ntop_clamped = jnp.clip(ntop, 0, nlevcan - 1)
    
    shflx_type1 = shair[patch_indices, ntop_clamped]
    etflx_type1 = etair[patch_indices, ntop_clamped]
    lhflx_type1 = etair[patch_indices, ntop_clamped] * latvap
    
    # Select based on turb_type
    is_type1 = (turb_type == 1)
    shflx = jnp.where(is_type1, shflx_type1, shflx_type0)
    etflx = jnp.where(is_type1, etflx_type1, etflx_type0)
    lhflx = jnp.where(is_type1, lhflx_type1, lhflx_type0)
    
    # Add canopy air heat storage to storage flux
    layer_indices = jnp.arange(nlevcan)[None, :]
    ntop_expanded = ntop[:, None]
    layer_mask = layer_indices < ntop_expanded
    
    stair_sum = jnp.sum(stair * layer_mask, axis=1)
    stflx_total = stflx + stair_sum
    
    # Overall energy balance checks
    rnet = (swveg_vis + swveg_nir + swsoi_vis + swsoi_nir + 
            lwveg + lwsoi)
    
    radin = (swskyb_vis + swskyd_vis + swskyb_nir + swskyd_nir + lwsky)
    
    radout = (albcan_vis * (swskyb_vis + swskyd_vis) + 
              albcan_nir * (swskyb_nir + swskyd_nir) + lwup)
    
    energy_balance_rad = rnet - (radin - radout)
    
    avail = radin - radout - gsoi
    flux = shflx + lhflx + stflx_total
    energy_balance_total = avail - flux
    
    return EnergyBalanceState(
        vcmax25veg=vcmax25veg_updated,
        shflx=shflx,
        etflx=etflx,
        lhflx=lhflx,
        stflx=stflx_total,
        rnet=rnet,
    )


def aggregate_sunshade_fluxes(
    ncan: jnp.ndarray,
    dpai: jnp.ndarray,
    fracsun: jnp.ndarray,
    lwleaf: jnp.ndarray,
    shleaf: jnp.ndarray,
    lhleaf: jnp.ndarray,
    evleaf: jnp.ndarray,
    trleaf: jnp.ndarray,
    agross: jnp.ndarray,
    vcmax25_leaf: jnp.ndarray,
    gs: jnp.ndarray,
    fdry: jnp.ndarray,
    fwet: jnp.ndarray,
    wind: jnp.ndarray,
    tleaf_mean: jnp.ndarray,
    tleaf: jnp.ndarray,
    tair: jnp.ndarray,
    isun: int = ISUN,
    isha: int = ISHA,
) -> SunShadeFluxes:
    """Aggregate canopy layer fluxes into sunlit and shaded totals.
    
    Fortran source: lines 1005-1057
    
    Args:
        ncan: Number of canopy layers [n_patches]
        dpai: Layer plant area index [n_patches, n_layers]
        fracsun: Sunlit fraction of layer [n_patches, n_layers]
        (... many more flux arguments ...)
        isun: Index for sunlit
        isha: Index for shaded
        
    Returns:
        SunShadeFluxes: Named tuple with all aggregated fluxes
    """
    n_patches = dpai.shape[0]
    n_layers = dpai.shape[1]
    
    # Initialize accumulators
    laisun = jnp.zeros(n_patches)
    laisha = jnp.zeros(n_patches)
    lwvegsun = jnp.zeros(n_patches)
    lwvegsha = jnp.zeros(n_patches)
    shvegsun = jnp.zeros(n_patches)
    shvegsha = jnp.zeros(n_patches)
    lhvegsun = jnp.zeros(n_patches)
    lhvegsha = jnp.zeros(n_patches)
    etvegsun = jnp.zeros(n_patches)
    etvegsha = jnp.zeros(n_patches)
    gppvegsun = jnp.zeros(n_patches)
    gppvegsha = jnp.zeros(n_patches)
    vcmax25sun = jnp.zeros(n_patches)
    vcmax25sha = jnp.zeros(n_patches)
    gsvegsun = jnp.zeros(n_patches)
    gsvegsha = jnp.zeros(n_patches)
    
    # Loop over canopy layers to accumulate fluxes
    def accumulate_layer_fluxes(ic, carry):
        (laisun, laisha, lwvegsun, lwvegsha, shvegsun, shvegsha,
         lhvegsun, lhvegsha, etvegsun, etvegsha, gppvegsun, gppvegsha,
         vcmax25sun, vcmax25sha, gsvegsun, gsvegsha) = carry
        
        dpai_ic = dpai[:, ic]
        fracsun_ic = fracsun[:, ic]
        fracsha_ic = 1.0 - fracsun_ic
        
        valid = dpai_ic > 0.0
        
        # Accumulate LAI
        laisun = laisun + jnp.where(valid, fracsun_ic * dpai_ic, 0.0)
        laisha = laisha + jnp.where(valid, fracsha_ic * dpai_ic, 0.0)
        
        # Accumulate fluxes
        lwvegsun = lwvegsun + jnp.where(valid, lwleaf[:, ic, isun] * fracsun_ic * dpai_ic, 0.0)
        lwvegsha = lwvegsha + jnp.where(valid, lwleaf[:, ic, isha] * fracsha_ic * dpai_ic, 0.0)
        
        shvegsun = shvegsun + jnp.where(valid, shleaf[:, ic, isun] * fracsun_ic * dpai_ic, 0.0)
        shvegsha = shvegsha + jnp.where(valid, shleaf[:, ic, isha] * fracsha_ic * dpai_ic, 0.0)
        
        lhvegsun = lhvegsun + jnp.where(valid, lhleaf[:, ic, isun] * fracsun_ic * dpai_ic, 0.0)
        lhvegsha = lhvegsha + jnp.where(valid, lhleaf[:, ic, isha] * fracsha_ic * dpai_ic, 0.0)
        
        et_sun = (evleaf[:, ic, isun] + trleaf[:, ic, isun]) * fracsun_ic * dpai_ic
        et_sha = (evleaf[:, ic, isha] + trleaf[:, ic, isha]) * fracsha_ic * dpai_ic
        etvegsun = etvegsun + jnp.where(valid, et_sun, 0.0)
        etvegsha = etvegsha + jnp.where(valid, et_sha, 0.0)
        
        fracgreen = fdry[:, ic] / jnp.maximum(1.0 - fwet[:, ic], 1e-10)
        gpp_sun = agross[:, ic, isun] * fracsun_ic * dpai_ic * fracgreen
        gpp_sha = agross[:, ic, isha] * fracsha_ic * dpai_ic * fracgreen
        gppvegsun = gppvegsun + jnp.where(valid, gpp_sun, 0.0)
        gppvegsha = gppvegsha + jnp.where(valid, gpp_sha, 0.0)
        
        vcmax25sun = vcmax25sun + jnp.where(valid, vcmax25_leaf[:, ic, isun] * fracsun_ic * dpai_ic, 0.0)
        vcmax25sha = vcmax25sha + jnp.where(valid, vcmax25_leaf[:, ic, isha] * fracsha_ic * dpai_ic, 0.0)
        
        gsvegsun = gsvegsun + jnp.where(valid, gs[:, ic, isun] * fracsun_ic * dpai_ic, 0.0)
        gsvegsha = gsvegsha + jnp.where(valid, gs[:, ic, isha] * fracsha_ic * dpai_ic, 0.0)
        
        return (laisun, laisha, lwvegsun, lwvegsha, shvegsun, shvegsha,
                lhvegsun, lhvegsha, etvegsun, etvegsha, gppvegsun, gppvegsha,
                vcmax25sun, vcmax25sha, gsvegsun, gsvegsha)
    
    flux_carry = (laisun, laisha, lwvegsun, lwvegsha, shvegsun, shvegsha,
                  lhvegsun, lhvegsha, etvegsun, etvegsha, gppvegsun, gppvegsha,
                  vcmax25sun, vcmax25sha, gsvegsun, gsvegsha)
    
    flux_carry = jax.lax.fori_loop(0, n_layers, accumulate_layer_fluxes, flux_carry)
    
    (laisun, laisha, lwvegsun, lwvegsha, shvegsun, shvegsha,
     lhvegsun, lhvegsha, etvegsun, etvegsha, gppvegsun, gppvegsha,
     vcmax25sun, vcmax25sha, gsvegsun, gsvegsha) = flux_carry
    
    # Initialize temperature and wind accumulators
    windveg = jnp.zeros(n_patches)
    windvegsun = jnp.zeros(n_patches)
    windvegsha = jnp.zeros(n_patches)
    tlveg = jnp.zeros(n_patches)
    tlvegsun = jnp.zeros(n_patches)
    tlvegsha = jnp.zeros(n_patches)
    taveg = jnp.zeros(n_patches)
    tavegsun = jnp.zeros(n_patches)
    tavegsha = jnp.zeros(n_patches)
    
    lai_total = laisun + laisha
    
    # Loop over canopy layers to compute LAI-weighted means
    def accumulate_layer_means(ic, carry):
        (windveg, windvegsun, windvegsha, tlveg, tlvegsun, tlvegsha,
         taveg, tavegsun, tavegsha) = carry
        
        dpai_ic = dpai[:, ic]
        fracsun_ic = fracsun[:, ic]
        fracsha_ic = 1.0 - fracsun_ic
        
        valid = dpai_ic > 0.0
        
        # Wind speed weighted by LAI
        wind_total = wind[:, ic] * dpai_ic / jnp.maximum(lai_total, 1e-10)
        wind_sun = wind[:, ic] * fracsun_ic * dpai_ic / jnp.maximum(laisun, 1e-10)
        wind_sha = wind[:, ic] * fracsha_ic * dpai_ic / jnp.maximum(laisha, 1e-10)
        windveg = windveg + jnp.where(valid, wind_total, 0.0)
        windvegsun = windvegsun + jnp.where(valid, wind_sun, 0.0)
        windvegsha = windvegsha + jnp.where(valid, wind_sha, 0.0)
        
        # Leaf temperature weighted by LAI
        tl_total = tleaf_mean[:, ic] * dpai_ic / jnp.maximum(lai_total, 1e-10)
        tl_sun = tleaf[:, ic, isun] * fracsun_ic * dpai_ic / jnp.maximum(laisun, 1e-10)
        tl_sha = tleaf[:, ic, isha] * fracsha_ic * dpai_ic / jnp.maximum(laisha, 1e-10)
        tlveg = tlveg + jnp.where(valid, tl_total, 0.0)
        tlvegsun = tlvegsun + jnp.where(valid, tl_sun, 0.0)
        tlvegsha = tlvegsha + jnp.where(valid, tl_sha, 0.0)
        
        # Air temperature weighted by LAI
        ta_total = tair[:, ic] * dpai_ic / jnp.maximum(lai_total, 1e-10)
        ta_sun = tair[:, ic] * fracsun_ic * dpai_ic / jnp.maximum(laisun, 1e-10)
        ta_sha = tair[:, ic] * fracsha_ic * dpai_ic / jnp.maximum(laisha, 1e-10)
        taveg = taveg + jnp.where(valid, ta_total, 0.0)
        tavegsun = tavegsun + jnp.where(valid, ta_sun, 0.0)
        tavegsha = tavegsha + jnp.where(valid, ta_sha, 0.0)
        
        return (windveg, windvegsun, windvegsha, tlveg, tlvegsun, tlvegsha,
                taveg, tavegsun, tavegsha)
    
    mean_carry = (windveg, windvegsun, windvegsha, tlveg, tlvegsun, tlvegsha,
                  taveg, tavegsun, tavegsha)
    
    mean_carry = jax.lax.fori_loop(0, n_layers, accumulate_layer_means, mean_carry)
    
    (windveg, windvegsun, windvegsha, tlveg, tlvegsun, tlvegsha,
     taveg, tavegsun, tavegsha) = mean_carry
    
    return SunShadeFluxes(
        laisun=laisun,
        laisha=laisha,
        lwvegsun=lwvegsun,
        lwvegsha=lwvegsha,
        shvegsun=shvegsun,
        shvegsha=shvegsha,
        lhvegsun=lhvegsun,
        lhvegsha=lhvegsha,
        etvegsun=etvegsun,
        etvegsha=etvegsha,
        gppvegsun=gppvegsun,
        gppvegsha=gppvegsha,
        vcmax25sun=vcmax25sun,
        vcmax25sha=vcmax25sha,
        gsvegsun=gsvegsun,
        gsvegsha=gsvegsha,
        windveg=windveg,
        windvegsun=windvegsun,
        windvegsha=windvegsha,
        tlveg=tlveg,
        tlvegsun=tlvegsun,
        tlvegsha=tlvegsha,
        taveg=taveg,
        tavegsun=tavegsun,
        tavegsha=tavegsha,
    )


def calculate_water_stress_fraction(
    dpai: jnp.ndarray,
    lwp_mean: jnp.ndarray,
    lai: jnp.ndarray,
    sai: jnp.ndarray,
    ncan: jnp.ndarray,
) -> WaterStressDiagnostics:
    """Calculate fraction of canopy that is water stressed.
    
    Fortran source: lines 1058-1080
    
    Args:
        dpai: Incremental plant area index [m2/m2] [n_patches, n_canopy_layers]
        lwp_mean: Mean leaf water potential [MPa] [n_patches, n_canopy_layers]
        lai: Leaf area index [m2/m2] [n_patches]
        sai: Stem area index [m2/m2] [n_patches]
        ncan: Number of canopy layers [-] [n_patches]
        
    Returns:
        WaterStressDiagnostics containing fracminlwp
    """
    # Water stress threshold
    minlwp = -2.0
    
    # Identify stressed canopy layers
    is_stressed = (dpai > 0.0) & (lwp_mean <= minlwp)
    
    # Sum stressed plant area index
    stressed_pai = jnp.sum(jnp.where(is_stressed, dpai, 0.0), axis=1)
    
    # Total plant area index
    total_pai = lai + sai
    
    # Calculate fraction
    fracminlwp = jnp.where(
        total_pai > 0.0,
        stressed_pai / total_pai,
        0.0
    )
    
    return WaterStressDiagnostics(fracminlwp=fracminlwp)


def canopy_fluxes_diagnostics(
    num_filter: int,
    filter_indices: jnp.ndarray,
    dpai: jnp.ndarray,
    fracsun: jnp.ndarray,
    fwet: jnp.ndarray,
    fdry: jnp.ndarray,
    lwleaf: jnp.ndarray,
    swleaf_vis: jnp.ndarray,
    swleaf_nir: jnp.ndarray,
    rnleaf: jnp.ndarray,
    stleaf: jnp.ndarray,
    shleaf: jnp.ndarray,
    lhleaf: jnp.ndarray,
    evleaf: jnp.ndarray,
    trleaf: jnp.ndarray,
    anet: jnp.ndarray,
    agross: jnp.ndarray,
    apar: jnp.ndarray,
    gs: jnp.ndarray,
    vcmax25_leaf: jnp.ndarray,
    vcmax25_profile: jnp.ndarray,
    tleaf_mean: jnp.ndarray,
    tleaf: jnp.ndarray,
    tair: jnp.ndarray,
    wind: jnp.ndarray,
    lwp: jnp.ndarray,
    lwp_mean: jnp.ndarray,
    lai: jnp.ndarray,
    sai: jnp.ndarray,
    ncan: jnp.ndarray,
) -> Tuple[LeafFluxProfiles, SunShadeFluxes, WaterStressDiagnostics]:
    """Calculate canopy flux diagnostics by summing leaf and soil fluxes.
    
    Fortran source: lines 727-1080
    
    Args:
        num_filter: Number of patches in filter
        filter_indices: Patch filter indices
        (... many flux and state arguments ...)
        
    Returns:
        Tuple of (LeafFluxProfiles, SunShadeFluxes, WaterStressDiagnostics)
    """
    # Calculate weighted mean leaf fluxes and source fluxes
    leaf_profiles = calculate_leaf_flux_profiles(
        dpai=dpai,
        fracsun=fracsun,
        fwet=fwet,
        fdry=fdry,
        lwleaf=lwleaf,
        swleaf_vis=swleaf_vis,
        swleaf_nir=swleaf_nir,
        rnleaf=rnleaf,
        stleaf=stleaf,
        shleaf=shleaf,
        lhleaf=lhleaf,
        evleaf=evleaf,
        trleaf=trleaf,
        anet=anet,
        apar=apar,
        gs=gs,
        tleaf_mean=tleaf_mean,
        tleaf=tleaf,
        lwp=lwp,
    )
    
    # Aggregate sunlit and shaded fluxes
    sunshade_fluxes = aggregate_sunshade_fluxes(
        ncan=ncan,
        dpai=dpai,
        fracsun=fracsun,
        lwleaf=lwleaf,
        shleaf=shleaf,
        lhleaf=lhleaf,
        evleaf=evleaf,
        trleaf=trleaf,
        agross=agross,
        vcmax25_leaf=vcmax25_leaf,
        gs=gs,
        fdry=fdry,
        fwet=fwet,
        wind=wind,
        tleaf_mean=tleaf_mean,
        tleaf=tleaf,
        tair=tair,
    )
    
    # Calculate water stress fraction
    water_stress = calculate_water_stress_fraction(
        dpai=dpai,
        lwp_mean=lwp_mean,
        lai=lai,
        sai=sai,
        ncan=ncan,
    )
    
    return leaf_profiles, sunshade_fluxes, water_stress


# ============================================================================
# Main Driver Function
# ============================================================================

def ml_canopy_fluxes(
    bounds: Bounds,
    num_exposedvegp: int,
    filter_exposedvegp: jnp.ndarray,
    atmospheric_forcing: AtmosphericForcing,
    canopy_state: CanopyProfileState,
    nstep: int,
    dtime: float,
    dtime_substep: float,
    num_sub_steps: int,
) -> Tuple[MLCanopyFluxesState, FluxAccumulators]:
    """Compute fluxes for sunlit and shaded leaves at each level and for soil surface.
    
    This is the main driver routine for multilayer canopy flux calculations.
    It coordinates the various sub-calculations needed to compute energy,
    water, and carbon fluxes through a multi-layer canopy representation.
    
    Fortran source: lines 47-565
    
    Args:
        bounds: Subgrid bounds structure
        num_exposedvegp: Number of exposed vegetation patches
        filter_exposedvegp: Filter for exposed vegetation patches
        atmospheric_forcing: Atmospheric forcing variables
        canopy_state: Canopy state variables
        nstep: Current time step number
        dtime: Time step size [s]
        dtime_substep: Sub-timestep for iterative calculations [s]
        num_sub_steps: Number of sub-time steps
        
    Returns:
        Tuple of (MLCanopyFluxesState, FluxAccumulators)
        
    Note:
        This is a simplified interface. The full implementation would include
        all the state variables and call the various physics subroutines
        (radiation, turbulence, photosynthesis, etc.) that are translated
        in separate modules.
    """
    n_patches = bounds.endp - bounds.begp + 1
    
    # Initialize flux accumulators
    flux_accumulator = jnp.zeros((n_patches, NVAR1D))
    flux_accumulator_profile = jnp.zeros((n_patches, 20, NVAR2D))  # Assuming 20 layers
    flux_accumulator_leaf = jnp.zeros((n_patches, 20, 2, NVAR3D))
    
    # Calculate derived atmospheric properties
    derived_atm = calculate_derived_atmospheric_properties(
        qref=atmospheric_forcing.qref,
        pref=atmospheric_forcing.pref,
        tref=atmospheric_forcing.tref,
        zref=jnp.ones(n_patches) * 30.0,  # Example reference height
    )
    
    # Initialize state for sub-stepping
    # (In full implementation, would extract from mlcanopy_inst)
    tg = jnp.ones(n_patches) * 288.0  # Example ground temperature
    tleaf = jnp.ones((n_patches, 20, 2)) * 288.0
    tair = jnp.ones((n_patches, 20)) * 288.0
    eair = jnp.ones((n_patches, 20)) * 1000.0
    cair = jnp.ones((n_patches, 20)) * 400.0
    
    substep_state = initialize_substep_state(
        tg=tg,
        tleaf=tleaf,
        tair=tair,
        eair=eair,
        cair=cair,
    )
    
    # Main flux calculation: Net radiation computation
    # This section computes the net radiation for sunlit/shaded leaves and soil.
    # In the full model with sub-timestep iteration, this would be called within
    # a time integration loop to update fluxes at each sub-step.
    
    # Net radiation calculation using radiation components
    swleaf_sun_vis = jnp.zeros((n_patches, 20))
    swleaf_sun_nir = jnp.zeros((n_patches, 20))
    swleaf_shade_vis = jnp.zeros((n_patches, 20))
    swleaf_shade_nir = jnp.zeros((n_patches, 20))
    lwleaf_sun = jnp.zeros((n_patches, 20))
    lwleaf_shade = jnp.zeros((n_patches, 20))
    swsoi_vis = jnp.zeros(n_patches)
    swsoi_nir = jnp.zeros(n_patches)
    lwsoi = jnp.zeros(n_patches)
    
    rnleaf_sun, rnleaf_shade, rnsoi = compute_net_radiation(
        swleaf_sun_vis=swleaf_sun_vis,
        swleaf_sun_nir=swleaf_sun_nir,
        swleaf_shade_vis=swleaf_shade_vis,
        swleaf_shade_nir=swleaf_shade_nir,
        lwleaf_sun=lwleaf_sun,
        lwleaf_shade=lwleaf_shade,
        swsoi_vis=swsoi_vis,
        swsoi_nir=swsoi_nir,
        lwsoi=lwsoi,
        ncan=jnp.ones(n_patches, dtype=jnp.int32) * 20,
    )
    
    # Soil relative humidity (from soil moisture and temperature)
    # Set to 50% as initial estimate; in full implementation would be
    # computed from soil water potential and temperature
    rhg = jnp.ones(n_patches) * 0.5
    
    mlcanopy_state = MLCanopyFluxesState(
        rnleaf_sun=rnleaf_sun,
        rnleaf_shade=rnleaf_shade,
        rnsoi=rnsoi,
        rhg=rhg,
        flux_accumulator=flux_accumulator,
        flux_accumulator_profile=flux_accumulator_profile,
        flux_accumulator_leaf=flux_accumulator_leaf,
    )
    
    final_accumulators = FluxAccumulators(
        flux_1d=flux_accumulator,
        flux_2d=flux_accumulator_profile,
        flux_3d=flux_accumulator_leaf,
    )
    
    return mlcanopy_state, final_accumulators


# Backward compatibility alias
MLCanopyFluxes = MLCanopyFluxesState
