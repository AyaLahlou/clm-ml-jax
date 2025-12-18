"""
CLM-ML Offline Driver Module.

Translated from CTSM's CLMml_driver.F90 (lines 1-843)

This module provides the main driver for the CLM-ML offline model, coordinating
initialization, time stepping, and output for tower-based simulations. It manages:

- Model initialization and setup
- Time stepping loop with forcing data reading
- Canopy and soil state initialization
- Model execution and flux calculations
- Output writing for fluxes, profiles, and diagnostics

Key Components:
    - CLMml_drv: Main driver routine
    - init_acclim: Acclimation temperature initialization
    - TowerVeg: Tower vegetation initialization
    - SoilInit: Soil state initialization from CLM history
    - output: Model output writing
    - ReadCanopyProfiles: Prescribed profile reading

Physics:
    - Acclimation temperature: Average air temperature over all time slices
    - Soil moisture: h2osoi_liq = h2osoi_vol * dz * denh2o
    - Specific humidity: qair = 1000 * (mmh2o/mmdry) * eair / (pref - (1-mmh2o/mmdry) * eair)

Reference: CLMml_driver.F90:1-843
"""

from typing import NamedTuple, Tuple, Protocol, runtime_checkable, Callable
import jax
import jax.numpy as jnp
from jax import lax
import netCDF4 as nc

# ============================================================================
# Type Definitions
# ============================================================================


class BoundsType(NamedTuple):
    """Domain decomposition bounds.
    
    Attributes:
        begp: First patch index [scalar int]
        endp: Last patch index [scalar int]
        begc: First column index [scalar int]
        endc: Last column index [scalar int]
    """
    begp: int
    endp: int
    begc: int
    endc: int


class Atm2LndState(NamedTuple):
    """Atmospheric forcing state downscaled to land surface.
    
    Attributes:
        forc_t_downscaled_col: Atmospheric temperature [K] [n_columns]
        forc_pbot_downscaled_col: Atmospheric pressure [Pa] [n_columns]
    """
    forc_t_downscaled_col: jnp.ndarray
    forc_pbot_downscaled_col: jnp.ndarray


class TemperatureState(NamedTuple):
    """Temperature state variables.
    
    Attributes:
        t_a10_patch: 10-day average air temperature for acclimation [K] [n_patches]
        t_soisno_col: Soil temperature [K] [n_columns, nlevgrnd]
    """
    t_a10_patch: jnp.ndarray
    t_soisno_col: jnp.ndarray


class FrictionVelState(NamedTuple):
    """Friction velocity state variables."""
    pass


class MLCanopyState(NamedTuple):
    """Multi-layer canopy state variables.
    
    Attributes:
        pref_forcing: Air pressure at reference height [Pa] [n_patches]
        root_biomass_canopy: Fine root biomass [g biomass/m2] [n_patches]
        ncan_canopy: Number of aboveground layers [n_patches]
        ntop_canopy: Index of top canopy layer [n_patches]
        nbot_canopy: Index of bottom canopy layer [n_patches]
        zs_profile: Layer height [m] [n_patches, n_layers]
        wind_profile: Wind speed [m/s] [n_patches, n_layers]
        tair_profile: Air temperature [K] [n_patches, n_layers]
        eair_profile: Vapor pressure [Pa] [n_patches, n_layers]
        dpai_profile: Layer plant area index [m2/m2] [n_patches, n_layers]
        dz_profile: Layer thickness [m] [n_patches, n_layers]
        fracsun_profile: Sunlit fraction [0-1] [n_patches, n_layers]
        rnleaf_leaf: Net radiation [W/m2 leaf] [n_patches, n_layers, 2]
        shleaf_leaf: Sensible heat [W/m2 leaf] [n_patches, n_layers, 2]
        lhleaf_leaf: Latent heat [W/m2 leaf] [n_patches, n_layers, 2]
        anet_leaf: Net photosynthesis [umol/m2/s] [n_patches, n_layers, 2]
        apar_leaf: Absorbed PAR [W/m2 leaf] [n_patches, n_layers, 2]
        gs_leaf: Stomatal conductance [mol/m2/s] [n_patches, n_layers, 2]
        lwp_hist_leaf: Leaf water potential [MPa] [n_patches, n_layers, 2]
        tleaf_hist_leaf: Leaf temperature [K] [n_patches, n_layers, 2]
        vcmax25_leaf: Vcmax at 25C [umol/m2/s] [n_patches, n_layers, 2]
        lsc_profile: Leaf-specific conductance [mmol/m2/s/MPa] [n_patches, n_layers]
        lwp_mean_profile: Mean leaf water potential [MPa] [n_patches, n_layers]
        # Canopy-level aggregates
        rnet_canopy: Net radiation [W/m2] [n_patches]
        stflx_canopy: Sensible heat storage [W/m2] [n_patches]
        shflx_canopy: Sensible heat flux [W/m2] [n_patches]
        lhflx_canopy: Latent heat flux [W/m2] [n_patches]
        gppveg_canopy: Gross primary production [umol CO2/m2/s] [n_patches]
        ustar_canopy: Friction velocity [m/s] [n_patches]
        lwup_canopy: Upward longwave [W/m2] [n_patches]
        taf_canopy: Air temperature at reference [K] [n_patches]
        albcan_canopy: Canopy albedo [0-1] [n_patches, 2]
        lai_canopy: Leaf area index [m2/m2] [n_patches]
        sai_canopy: Stem area index [m2/m2] [n_patches]
        laisun_canopy: Sunlit LAI [m2/m2] [n_patches]
        laisha_canopy: Shaded LAI [m2/m2] [n_patches]
        swveg_canopy: Absorbed SW by vegetation [W/m2] [n_patches, 2]
        swvegsun_canopy: Absorbed SW sunlit [W/m2] [n_patches, 2]
        swvegsha_canopy: Absorbed SW shaded [W/m2] [n_patches, 2]
        gppvegsun_canopy: Sunlit GPP [umol CO2/m2/s] [n_patches]
        gppvegsha_canopy: Shaded GPP [umol CO2/m2/s] [n_patches]
        lhveg_canopy: Total latent heat [W/m2] [n_patches]
        lhvegsun_canopy: Sunlit latent heat [W/m2] [n_patches]
        lhvegsha_canopy: Shaded latent heat [W/m2] [n_patches]
        shveg_canopy: Total sensible heat [W/m2] [n_patches]
        shvegsun_canopy: Sunlit sensible heat [W/m2] [n_patches]
        shvegsha_canopy: Shaded sensible heat [W/m2] [n_patches]
        vcmax25veg_canopy: Total Vcmax at 25C [umol/m2/s] [n_patches]
        vcmax25sun_canopy: Sunlit Vcmax at 25C [umol/m2/s] [n_patches]
        vcmax25sha_canopy: Shaded Vcmax at 25C [umol/m2/s] [n_patches]
        gsveg_canopy: Total stomatal conductance [mol/m2/s] [n_patches]
        gsvegsun_canopy: Sunlit stomatal conductance [mol/m2/s] [n_patches]
        gsvegsha_canopy: Shaded stomatal conductance [mol/m2/s] [n_patches]
        windveg_canopy: Total wind speed [m/s] [n_patches]
        windvegsun_canopy: Sunlit wind speed [m/s] [n_patches]
        windvegsha_canopy: Shaded wind speed [m/s] [n_patches]
        tlveg_canopy: Total leaf temperature [K] [n_patches]
        tlvegsun_canopy: Sunlit leaf temperature [K] [n_patches]
        tlvegsha_canopy: Shaded leaf temperature [K] [n_patches]
        taveg_canopy: Total air temperature [K] [n_patches]
        tavegsun_canopy: Sunlit air temperature [K] [n_patches]
        tavegsha_canopy: Shaded air temperature [K] [n_patches]
        fracminlwp_canopy: Fraction at minimum water potential [0-1] [n_patches]
        # Forcing
        swskyb_forcing: Direct beam SW [W/m2] [n_patches, 2]
        swskyd_forcing: Diffuse SW [W/m2] [n_patches, 2]
        solar_zen_forcing: Solar zenith angle [radians] [n_patches]
        rhomol_forcing: Molar density [mol/m3] [n_patches]
        gac_profile: Aerodynamic conductance [mol/m2/s] [n_patches, n_layers]
        wind_data: Wind from dataset [m/s] [n_patches, n_layers]
        tair_data: Temperature from dataset [K] [n_patches, n_layers]
        eair_data: Vapor pressure from dataset [Pa] [n_patches, n_layers]
    """
    pref_forcing: jnp.ndarray
    root_biomass_canopy: jnp.ndarray
    ncan_canopy: jnp.ndarray
    ntop_canopy: jnp.ndarray
    nbot_canopy: jnp.ndarray
    zs_profile: jnp.ndarray
    wind_profile: jnp.ndarray
    tair_profile: jnp.ndarray
    eair_profile: jnp.ndarray
    dpai_profile: jnp.ndarray
    dz_profile: jnp.ndarray
    fracsun_profile: jnp.ndarray
    rnleaf_leaf: jnp.ndarray
    shleaf_leaf: jnp.ndarray
    lhleaf_leaf: jnp.ndarray
    anet_leaf: jnp.ndarray
    apar_leaf: jnp.ndarray
    gs_leaf: jnp.ndarray
    lwp_hist_leaf: jnp.ndarray
    tleaf_hist_leaf: jnp.ndarray
    vcmax25_leaf: jnp.ndarray
    lsc_profile: jnp.ndarray
    lwp_mean_profile: jnp.ndarray
    rnet_canopy: jnp.ndarray
    stflx_canopy: jnp.ndarray
    shflx_canopy: jnp.ndarray
    lhflx_canopy: jnp.ndarray
    gppveg_canopy: jnp.ndarray
    ustar_canopy: jnp.ndarray
    lwup_canopy: jnp.ndarray
    taf_canopy: jnp.ndarray
    albcan_canopy: jnp.ndarray
    lai_canopy: jnp.ndarray
    sai_canopy: jnp.ndarray
    laisun_canopy: jnp.ndarray
    laisha_canopy: jnp.ndarray
    swveg_canopy: jnp.ndarray
    swvegsun_canopy: jnp.ndarray
    swvegsha_canopy: jnp.ndarray
    gppvegsun_canopy: jnp.ndarray
    gppvegsha_canopy: jnp.ndarray
    lhveg_canopy: jnp.ndarray
    lhvegsun_canopy: jnp.ndarray
    lhvegsha_canopy: jnp.ndarray
    shveg_canopy: jnp.ndarray
    shvegsun_canopy: jnp.ndarray
    shvegsha_canopy: jnp.ndarray
    vcmax25veg_canopy: jnp.ndarray
    vcmax25sun_canopy: jnp.ndarray
    vcmax25sha_canopy: jnp.ndarray
    gsveg_canopy: jnp.ndarray
    gsvegsun_canopy: jnp.ndarray
    gsvegsha_canopy: jnp.ndarray
    windveg_canopy: jnp.ndarray
    windvegsun_canopy: jnp.ndarray
    windvegsha_canopy: jnp.ndarray
    tlveg_canopy: jnp.ndarray
    tlvegsun_canopy: jnp.ndarray
    tlvegsha_canopy: jnp.ndarray
    taveg_canopy: jnp.ndarray
    tavegsun_canopy: jnp.ndarray
    tavegsha_canopy: jnp.ndarray
    fracminlwp_canopy: jnp.ndarray
    swskyb_forcing: jnp.ndarray
    swskyd_forcing: jnp.ndarray
    solar_zen_forcing: jnp.ndarray
    rhomol_forcing: jnp.ndarray
    gac_profile: jnp.ndarray
    wind_data: jnp.ndarray
    tair_data: jnp.ndarray
    eair_data: jnp.ndarray


class CanopyState(NamedTuple):
    """Canopy state variables.
    
    Attributes:
        htop_patch: Canopy top height [m] [n_patches]
    """
    htop_patch: jnp.ndarray


class PatchInfo(NamedTuple):
    """Patch information.
    
    Attributes:
        itype: PFT type index [n_patches]
        column: Column index for each patch [n_patches]
    """
    itype: jnp.ndarray
    column: jnp.ndarray


class ColumnState(NamedTuple):
    """Column-level state variables.
    
    Attributes:
        dz: Soil layer thickness [m] [n_columns, nlevgrnd]
        nbedrock: Depth to bedrock index [n_columns]
    """
    dz: jnp.ndarray
    nbedrock: jnp.ndarray


class SoilState(NamedTuple):
    """Soil state variables.
    
    Attributes:
        watsat_col: Volumetric water content at saturation [m3/m3] [n_columns, nlevgrnd]
        btran_soil: Soil moisture stress factor [0-1] [n_patches]
        psis_soil: Soil water potential [MPa] [n_patches]
        gsoi_soil: Soil heat flux [W/m2] [n_patches]
        rnsoi_soil: Net radiation at soil [W/m2] [n_patches]
        shsoi_soil: Sensible heat from soil [W/m2] [n_patches]
        lhsoi_soil: Latent heat from soil [W/m2] [n_patches]
        tg_soil: Ground temperature [K] [n_patches]
        eg_soil: Ground vapor pressure [Pa] [n_patches]
        gac0_soil: Ground aerodynamic conductance [mol/m2/s] [n_patches]
    """
    watsat_col: jnp.ndarray
    btran_soil: jnp.ndarray
    psis_soil: jnp.ndarray
    gsoi_soil: jnp.ndarray
    rnsoi_soil: jnp.ndarray
    shsoi_soil: jnp.ndarray
    lhsoi_soil: jnp.ndarray
    tg_soil: jnp.ndarray
    eg_soil: jnp.ndarray
    gac0_soil: jnp.ndarray


class WaterState(NamedTuple):
    """Water state variables.
    
    Attributes:
        h2osoi_vol_col: Volumetric water content [m3/m3] [n_columns, nlevgrnd]
        h2osoi_ice_col: Ice lens mass [kg H2O/m2] [n_columns, nlevgrnd]
        h2osoi_liq_col: Liquid water mass [kg H2O/m2] [n_columns, nlevgrnd]
    """
    h2osoi_vol_col: jnp.ndarray
    h2osoi_ice_col: jnp.ndarray
    h2osoi_liq_col: jnp.ndarray


class PhysicalConstants(NamedTuple):
    """Physical constants.
    
    Attributes:
        mwh2o: Molecular weight of water [g/mol]
        mwdry: Molecular weight of dry air [g/mol]
        denh2o: Density of liquid water [kg/m3]
        pi: Pi constant
    """
    mwh2o: float = 18.016
    mwdry: float = 28.966
    denh2o: float = 1000.0
    pi: float = 3.14159265358979323846


class PFTParameters(NamedTuple):
    """PFT-specific parameters.
    
    Attributes:
        pbeta_lai: LAI beta parameter [dimensionless] [n_pfts]
        qbeta_lai: LAI q parameter [dimensionless] [n_pfts]
        pbeta_sai: SAI beta parameter [dimensionless] [n_pfts]
        qbeta_sai: SAI q parameter [dimensionless] [n_pfts]
    """
    pbeta_lai: jnp.ndarray
    qbeta_lai: jnp.ndarray
    pbeta_sai: jnp.ndarray
    qbeta_sai: jnp.ndarray


class InitializationState(NamedTuple):
    """State after initialization phase.
    
    Attributes:
        yr: Current year [integer]
        mon: Current month [1-12]
        day: Current day of month [1-31]
        curr_date_tod: Current time of day [seconds]
        itim: Time step counter
        eccen: Orbital eccentricity [dimensionless]
        obliq: Obliquity [degrees]
        mvelp: Moving vernal equinox longitude of perihelion [degrees]
        obliqr: Obliquity [radians]
        lambm0: Mean longitude of perihelion at vernal equinox [radians]
        mvelpp: Moving vernal equinox longitude of perihelion plus pi [radians]
        fin_tower: Tower meteorology file path
        fin_clm: CLM history file path
        pft_params_adjusted: Whether PFT parameters were adjusted
    """
    yr: int
    mon: int
    day: int
    curr_date_tod: int
    itim: int
    eccen: float
    obliq: float
    mvelp: float
    obliqr: float
    lambm0: float
    mvelpp: float
    fin_tower: str
    fin_clm: str
    pft_params_adjusted: bool


class CLMHistoryFileInfo(NamedTuple):
    """Information for CLM history file initialization.
    
    Attributes:
        fin_clm: Full path to CLM history file
        time_indx: Time slice index into CLM history file [1-based]
        start_calday_clm: Calendar day of first time slice
        curr_calday: Calendar day for start of simulation
    """
    fin_clm: str
    time_indx: int
    start_calday_clm: float
    curr_calday: float


class TimeStepState(NamedTuple):
    """State for a single time step.
    
    Attributes:
        year: Current year
        month: Current month [1-12]
        day: Current day of month [1-31]
        curr_date_tod: Current time of day [seconds]
        curr_time_day: Current time as day fraction [0-1]
        curr_time_sec: Current time in seconds
        curr_calday: Current calendar day [1.0 = 0Z Jan 1]
        time_indx: Time index into CLM history file
    """
    year: jnp.ndarray
    month: jnp.ndarray
    day: jnp.ndarray
    curr_date_tod: jnp.ndarray
    curr_time_day: jnp.ndarray
    curr_time_sec: jnp.ndarray
    curr_calday: jnp.ndarray
    time_indx: jnp.ndarray


class OutputFluxes(NamedTuple):
    """Canopy and soil flux outputs.
    
    Attributes:
        rnet_canopy: Net radiation at canopy [W/m2]
        stflx_canopy: Sensible heat flux storage [W/m2]
        shflx_canopy: Sensible heat flux [W/m2]
        lhflx_canopy: Latent heat flux [W/m2]
        gppveg_canopy: Gross primary production [umol CO2/m2/s]
        ustar_canopy: Friction velocity [m/s]
        swup: Upward shortwave radiation [W/m2]
        lwup_canopy: Upward longwave radiation [W/m2]
        taf_canopy: Air temperature at reference height [K]
        gsoi_soil: Soil heat flux [W/m2]
        rnsoi_soil: Net radiation at soil surface [W/m2]
        shsoi_soil: Sensible heat flux from soil [W/m2]
        lhsoi_soil: Latent heat flux from soil [W/m2]
    """
    rnet_canopy: jnp.ndarray
    stflx_canopy: jnp.ndarray
    shflx_canopy: jnp.ndarray
    lhflx_canopy: jnp.ndarray
    gppveg_canopy: jnp.ndarray
    ustar_canopy: jnp.ndarray
    swup: jnp.ndarray
    lwup_canopy: jnp.ndarray
    taf_canopy: jnp.ndarray
    gsoi_soil: jnp.ndarray
    rnsoi_soil: jnp.ndarray
    shsoi_soil: jnp.ndarray
    lhsoi_soil: jnp.ndarray


class OutputSunShade(NamedTuple):
    """Sunlit/shaded canopy flux outputs.
    
    Attributes:
        solar_zen_deg: Solar zenith angle [degrees]
        sw_vis_total: Total visible shortwave [W/m2]
        lai_sai_total: Total LAI + SAI [m2/m2]
        laisun: Sunlit LAI [m2/m2]
        laisha: Shaded LAI [m2/m2]
        swveg_vis: Absorbed visible SW by vegetation [W/m2]
        swvegsun_vis: Absorbed visible SW by sunlit leaves [W/m2]
        swvegsha_vis: Absorbed visible SW by shaded leaves [W/m2]
        gppveg: Total GPP [umol CO2/m2/s]
        gppvegsun: Sunlit GPP [umol CO2/m2/s]
        gppvegsha: Shaded GPP [umol CO2/m2/s]
        lhveg: Total latent heat [W/m2]
        lhvegsun: Sunlit latent heat [W/m2]
        lhvegsha: Shaded latent heat [W/m2]
        shveg: Total sensible heat [W/m2]
        shvegsun: Sunlit sensible heat [W/m2]
        shvegsha: Shaded sensible heat [W/m2]
        vcmax25veg: Total Vcmax at 25C [umol/m2/s]
        vcmax25sun: Sunlit Vcmax at 25C [umol/m2/s]
        vcmax25sha: Shaded Vcmax at 25C [umol/m2/s]
        gsveg: Total stomatal conductance [mol/m2/s]
        gsvegsun: Sunlit stomatal conductance [mol/m2/s]
        gsvegsha: Shaded stomatal conductance [mol/m2/s]
        windveg: Total wind speed [m/s]
        windvegsun: Sunlit wind speed [m/s]
        windvegsha: Shaded wind speed [m/s]
        tlveg: Total leaf temperature [K]
        tlvegsun: Sunlit leaf temperature [K]
        tlvegsha: Shaded leaf temperature [K]
        taveg: Total air temperature [K]
        tavegsun: Sunlit air temperature [K]
        tavegsha: Shaded air temperature [K]
    """
    solar_zen_deg: jnp.ndarray
    sw_vis_total: jnp.ndarray
    lai_sai_total: jnp.ndarray
    laisun: jnp.ndarray
    laisha: jnp.ndarray
    swveg_vis: jnp.ndarray
    swvegsun_vis: jnp.ndarray
    swvegsha_vis: jnp.ndarray
    gppveg: jnp.ndarray
    gppvegsun: jnp.ndarray
    gppvegsha: jnp.ndarray
    lhveg: jnp.ndarray
    lhvegsun: jnp.ndarray
    lhvegsha: jnp.ndarray
    shveg: jnp.ndarray
    shvegsun: jnp.ndarray
    shvegsha: jnp.ndarray
    vcmax25veg: jnp.ndarray
    vcmax25sun: jnp.ndarray
    vcmax25sha: jnp.ndarray
    gsveg: jnp.ndarray
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


class OutputAuxiliary(NamedTuple):
    """Leaf water potential and soil moisture stress outputs.
    
    Attributes:
        btran: Soil moisture stress factor [0-1]
        lsc_top: Leaf-specific conductance at top layer [mmol/m2/s/MPa]
        psis: Soil water potential [MPa]
        lwp_mean_top: Mean leaf water potential at top [MPa]
        lwp_mean_mid: Mean leaf water potential at mid-canopy [MPa]
        fracminlwp: Fraction of leaves at minimum water potential [0-1]
    """
    btran: jnp.ndarray
    lsc_top: jnp.ndarray
    psis: jnp.ndarray
    lwp_mean_top: jnp.ndarray
    lwp_mean_mid: jnp.ndarray
    fracminlwp: jnp.ndarray


class OutputProfile(NamedTuple):
    """Vertical profile outputs above canopy.
    
    Attributes:
        curr_calday: Current calendar day [days]
        zs: Height above ground [m] [n_patches, n_levels]
        wind: Wind speed [m/s] [n_patches, n_levels]
        tair: Air temperature [K] [n_patches, n_levels]
        qair: Specific humidity [g/kg] [n_patches, n_levels]
    """
    curr_calday: jnp.ndarray
    zs: jnp.ndarray
    wind: jnp.ndarray
    tair: jnp.ndarray
    qair: jnp.ndarray


class CanopyLayerOutput(NamedTuple):
    """Output data for a single canopy layer.
    
    Attributes:
        curr_calday: Current calendar day [days]
        zs: Layer height [m]
        fracsun: Sunlit fraction [0-1]
        lad: Leaf area density [m2/m3]
        lad_sun: Sunlit leaf area density [m2/m3]
        lad_shade: Shaded leaf area density [m2/m3]
        rnleaf_sun: Net radiation sunlit [W/m2 leaf]
        rnleaf_shade: Net radiation shaded [W/m2 leaf]
        shleaf_sun: Sensible heat sunlit [W/m2 leaf]
        shleaf_shade: Sensible heat shaded [W/m2 leaf]
        lhleaf_sun: Latent heat sunlit [W/m2 leaf]
        lhleaf_shade: Latent heat shaded [W/m2 leaf]
        anet_sun: Net photosynthesis sunlit [umol/m2 leaf/s]
        anet_shade: Net photosynthesis shaded [umol/m2 leaf/s]
        apar_sun: Absorbed PAR sunlit [W/m2 leaf]
        apar_shade: Absorbed PAR shaded [W/m2 leaf]
        gs_sun: Stomatal conductance sunlit [mol/m2 leaf/s]
        gs_shade: Stomatal conductance shaded [mol/m2 leaf/s]
        lwp_sun: Leaf water potential sunlit [MPa]
        lwp_shade: Leaf water potential shaded [MPa]
        tleaf_sun: Leaf temperature sunlit [K]
        tleaf_shade: Leaf temperature shaded [K]
        vcmax25_sun: Vcmax at 25C sunlit [umol/m2 leaf/s]
        vcmax25_shade: Vcmax at 25C shaded [umol/m2 leaf/s]
        wind: Wind speed [m/s]
        tair: Air temperature [K]
        qair: Specific humidity [g/kg]
    """
    curr_calday: float
    zs: float
    fracsun: float
    lad: float
    lad_sun: float
    lad_shade: float
    rnleaf_sun: float
    rnleaf_shade: float
    shleaf_sun: float
    shleaf_shade: float
    lhleaf_sun: float
    lhleaf_shade: float
    anet_sun: float
    anet_shade: float
    apar_sun: float
    apar_shade: float
    gs_sun: float
    gs_shade: float
    lwp_sun: float
    lwp_shade: float
    tleaf_sun: float
    tleaf_shade: float
    vcmax25_sun: float
    vcmax25_shade: float
    wind: float
    tair: float
    qair: float


class GroundOutputState(NamedTuple):
    """Ground surface output diagnostics.
    
    Attributes:
        tair: Ground air temperature [K] [n_patches]
        eair: Ground vapor pressure [kPa] [n_patches]
        ra: Aerodynamic resistance [s/m] [n_patches]
    """
    tair: jnp.ndarray
    eair: jnp.ndarray
    ra: jnp.ndarray


class CanopyProfileData(NamedTuple):
    """Data structure for canopy profile information.
    
    Attributes:
        ncan: Number of aboveground layers [n_patches]
        zs: Canopy layer height [m] [n_patches, n_layers]
        wind_data: Wind speed from dataset [m/s] [n_patches, n_layers]
        tair_data: Air temperature from dataset [K] [n_patches, n_layers]
        eair_data: Vapor pressure from dataset [Pa] [n_patches, n_layers]
        pref: Air pressure at reference height [Pa] [n_patches]
    """
    ncan: jnp.ndarray
    zs: jnp.ndarray
    wind_data: jnp.ndarray
    tair_data: jnp.ndarray
    eair_data: jnp.ndarray
    pref: jnp.ndarray


class CleanupState(NamedTuple):
    """State indicating cleanup operations to perform.
    
    Attributes:
        close_nout1: Whether to close output file 1
        close_nout2: Whether to close output file 2
        close_nout3: Whether to close output file 3
        close_nout4: Whether to close output file 4
        close_nin1: Whether to close input file 1
        success: Whether simulation completed successfully
    """
    close_nout1: bool
    close_nout2: bool
    close_nout3: bool
    close_nout4: bool
    close_nin1: bool
    success: bool


class TowerVegInputs(NamedTuple):
    """Inputs for tower vegetation initialization.
    
    Attributes:
        tower_pft: PFT type for this tower site
        tower_canht: Observed canopy height [m] (-999.0 = no observation)
        begp: First patch index
        endp: Last patch index
    """
    tower_pft: int
    tower_canht: float
    begp: int
    endp: int


# ============================================================================
# Protocol Definitions
# ============================================================================


@runtime_checkable
class CLMmlDriverProtocol(Protocol):
    """Protocol defining the CLM-ML driver interface.
    
    Reference: CLMml_driver.F90:1-29
    """
    
    def clmml_drv(self, bounds: BoundsType) -> None:
        """Main model driver routine.
        
        Args:
            bounds: Decomposition bounds for this processor
        """
        ...


# ============================================================================
# Constants and Parameters
# ============================================================================


# Physical constants (lines 560-579)
PHYSICAL_CONSTANTS = PhysicalConstants()

# Band indices (lines 594-666)
IVIS = 0  # Visible band index
INIR = 1  # Near-infrared band index

# Sun/shade indices (lines 667-730)
ISUN = 0  # Sunlit leaf index
ISHA = 1  # Shaded leaf index

# Missing value sentinel (lines 667-730)
MISSING_VALUE = -999.0
ZERO_VALUE = 0.0

# PFT canopy height lookup table (lines 395-399)
# PFT indices: 0=bare, 1-8=trees, 9-16=grasses, 17+=unused
HTOP_PFT_DEFAULT = jnp.array([
    0.0,   # 0: bare ground
    17.0, 17.0, 14.0, 35.0, 35.0, 18.0, 20.0, 20.0,  # 1-8: trees
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,          # 9-16: grasses/crops
] + [0.0] * 62, dtype=jnp.float32)  # 17-78: unused

# Pine parameters for US-Me2 (lines 113-118)
PINE_PBETA_LAI = 11.5
PINE_QBETA_LAI = 3.5
PINE_PBETA_SAI = 11.5
PINE_QBETA_SAI = 3.5

# Fine root biomass default (line 408)
DEFAULT_ROOT_BIOMASS = 500.0  # g biomass/m2


# ============================================================================
# Initialization Functions
# ============================================================================


def adjust_usme2_pft_parameters(
    pft_params: PFTParameters,
    patch_itype: int,
    tower_id: str,
) -> PFTParameters:
    """Adjust PFT parameters for US-Me2 tower site.
    
    For the US-Me2 site, reset leaf/stem area density parameters to pine values.
    
    Reference: CLMml_driver.F90:113-118
    
    Args:
        pft_params: Current PFT parameters
        patch_itype: PFT type index for patch 1
        tower_id: Tower site identifier
        
    Returns:
        Updated PFT parameters (unchanged if not US-Me2)
    """
    if tower_id == 'US-Me2':
        pbeta_lai = pft_params.pbeta_lai.at[patch_itype].set(PINE_PBETA_LAI)
        qbeta_lai = pft_params.qbeta_lai.at[patch_itype].set(PINE_QBETA_LAI)
        pbeta_sai = pft_params.pbeta_sai.at[patch_itype].set(PINE_PBETA_SAI)
        qbeta_sai = pft_params.qbeta_sai.at[patch_itype].set(PINE_QBETA_SAI)
        
        return PFTParameters(
            pbeta_lai=pbeta_lai,
            qbeta_lai=qbeta_lai,
            pbeta_sai=pbeta_sai,
            qbeta_sai=qbeta_sai,
        )
    else:
        return pft_params


def construct_tower_file_path(
    tower_id: str,
    yr: int,
    mon: int,
    diratm: str,
) -> str:
    """Construct path to tower meteorology file.
    
    Reference: CLMml_driver.F90:131-132
    
    Args:
        tower_id: Tower site identifier (e.g., 'US-Me2')
        yr: Year
        mon: Month [1-12]
        diratm: Atmospheric forcing directory path
        
    Returns:
        Full path to tower meteorology netCDF file
    """
    ext = f"{tower_id}/{yr:04d}-{mon:02d}.nc"
    return f"{diratm.rstrip('/')}/{ext}"


def construct_clm_filename(
    tower_id: str,
    tower_num: int,
    yr: int,
    dirclm: str,
    use_wozniak: bool = False,
) -> str:
    """Construct CLM history file path.
    
    Reference: CLMml_driver.F90:140-145
    
    Args:
        tower_id: Tower identifier string
        tower_num: Tower number index
        yr: Year for file
        dirclm: Base directory path for CLM files
        use_wozniak: Whether to use Wozniak-specific naming
        
    Returns:
        Full path to CLM history file
    """
    if use_wozniak:
        ext = f"clm50d30wspinsp_US-UMB_WOZNIAK.clm2.h1.{yr:04d}.nc"
    else:
        ext = f"lp67wspinPTCLM_{tower_id}_I_2000_CLM45.clm2.h1.{yr:04d}.nc"
    
    dirclm_trimmed = dirclm.rstrip()
    fin_clm = f"{dirclm_trimmed}{tower_id}/{ext}"
    
    return fin_clm


def calculate_time_index(
    curr_calday: jnp.ndarray,
    start_calday_clm: jnp.ndarray,
    dtstep: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate time index into CLM history file.
    
    Reference: CLMml_driver.F90:186-188, 245-247
    
    Args:
        curr_calday: Current calendar day [1.0 = 0Z Jan 1]
        start_calday_clm: Starting calendar day from CLM file
        dtstep: Time step size [seconds]
        
    Returns:
        Time index (1-based) into CLM history file
    """
    elapsed_seconds = (curr_calday - start_calday_clm) * 86400.0
    time_steps = elapsed_seconds / dtstep
    time_indx = jnp.round(time_steps).astype(jnp.int32) + 1
    
    return time_indx


def get_htop_pft_lookup() -> jnp.ndarray:
    """Get PFT-specific canopy height lookup table.
    
    Reference: CLMml_driver.F90:395-399
    
    Returns:
        Array of canopy heights [m] indexed by PFT [79 elements]
    """
    return HTOP_PFT_DEFAULT


def init_acclim(
    fin: str,
    tower_num: int,
    ntim: int,
    begp: int,
    endp: int,
    patch_info: PatchInfo,
    atm2lnd_inst: Atm2LndState,
    temperature_inst: TemperatureState,
    frictionvel_inst: FrictionVelState,
    mlcanopy_inst: MLCanopyState,
    tower_met_reader: Callable,
) -> Tuple[TemperatureState, MLCanopyState]:
    """Read tower meteorology data to compute acclimation temperature.
    
    This function processes all time slices from the tower meteorology file
    to compute the average air temperature for physiological acclimation.
    
    Reference: CLMml_driver.F90:296-370
    
    Args:
        fin: Tower meteorology file path
        tower_num: Tower site index
        ntim: Number of time slices to process
        begp: First patch index (0-based in JAX)
        endp: Last patch index (0-based in JAX, inclusive)
        patch_info: Patch hierarchy information
        atm2lnd_inst: Atmospheric forcing state
        temperature_inst: Temperature state
        frictionvel_inst: Friction velocity state
        mlcanopy_inst: Multi-layer canopy state
        tower_met_reader: Function to read tower meteorology data
        
    Returns:
        Updated temperature_inst and mlcanopy_inst with:
            - t_a10_patch: Average air temperature over all time slices [K]
            - pref_forcing: Reference pressure from first timestep [Pa]
    """
    n_patches = endp - begp + 1
    
    # Initialize temperature accumulator (lines 333-335)
    t10_accum = jnp.zeros(n_patches, dtype=jnp.float64)
    pref = jnp.zeros(n_patches, dtype=jnp.float64)
    
    def time_loop_body(itim, carry):
        """Process one time slice."""
        t10_accum, pref, atm2lnd_curr, frictionvel_curr = carry
        
        # Read temperature for this time slice (lines 341-342)
        atm2lnd_updated, frictionvel_updated = tower_met_reader(
            fin, itim, tower_num, begp, endp, atm2lnd_curr, frictionvel_curr
        )
        
        # Extract column indices for patches (lines 344-345)
        patch_indices = jnp.arange(begp, endp + 1)
        col_indices = patch_info.column[patch_indices]
        
        # Sum temperature (lines 347-348)
        forc_t_patches = atm2lnd_updated.forc_t_downscaled_col[col_indices]
        t10_accum = t10_accum + forc_t_patches
        
        # Save pressure for first timestep (lines 350-352)
        forc_pbot_patches = atm2lnd_updated.forc_pbot_downscaled_col[col_indices]
        pref = jnp.where(itim == 0, forc_pbot_patches, pref)
        
        return (t10_accum, pref, atm2lnd_updated, frictionvel_updated)
    
    # Loop over all time slices (lines 337-354)
    init_carry = (t10_accum, pref, atm2lnd_inst, frictionvel_inst)
    t10_accum_final, pref_final, _, _ = lax.fori_loop(
        0, ntim, time_loop_body, init_carry
    )
    
    # Average temperature (lines 356-358)
    t10_avg = t10_accum_final / jnp.float64(ntim)
    
    # Update states
    t_a10_full = temperature_inst.t_a10_patch.at[begp:endp+1].set(t10_avg)
    temperature_inst_updated = temperature_inst._replace(t_a10_patch=t_a10_full)
    
    pref_full = mlcanopy_inst.pref_forcing.at[begp:endp+1].set(pref_final)
    mlcanopy_inst_updated = mlcanopy_inst._replace(pref_forcing=pref_full)
    
    return temperature_inst_updated, mlcanopy_inst_updated


def tower_veg(
    inputs: TowerVegInputs,
    canopystate: CanopyState,
    mlcanopy: MLCanopyState,
    patch: PatchInfo,
) -> Tuple[CanopyState, MLCanopyState, PatchInfo]:
    """Initialize tower vegetation properties.
    
    Assigns PFT type, fine root biomass, and canopy height for all patches
    at a tower site.
    
    Reference: CLMml_driver.F90:373-424
    
    Args:
        inputs: Tower vegetation inputs including PFT and canopy height
        canopystate: Canopy state to update
        mlcanopy: ML canopy state to update
        patch: Patch information to update
        
    Returns:
        Tuple of (updated_canopystate, updated_mlcanopy, updated_patch)
    """
    htop_pft_lookup = get_htop_pft_lookup()
    
    n_patches = inputs.endp - inputs.begp + 1
    
    # Assign PFT type (line 407)
    itype = jnp.full(n_patches, inputs.tower_pft, dtype=jnp.int32)
    
    # Assign fine root biomass (line 408)
    root_biomass = jnp.full(n_patches, DEFAULT_ROOT_BIOMASS, dtype=jnp.float32)
    
    # Get canopy height from PFT lookup (line 409)
    htop = jnp.full(n_patches, htop_pft_lookup[inputs.tower_pft], dtype=jnp.float32)
    
    # Override with tower observation if available (lines 413-415)
    htop = jnp.where(
        inputs.tower_canht != MISSING_VALUE,
        inputs.tower_canht,
        htop
    )
    
    # Update state tuples
    updated_canopystate = canopystate._replace(htop_patch=htop)
    updated_mlcanopy = mlcanopy._replace(root_biomass_canopy=root_biomass)
    updated_patch = patch._replace(itype=itype)
    
    return updated_canopystate, updated_mlcanopy, updated_patch


def soil_init_vectorized(
    ncfilename: str,
    strt: int,
    begc: int,
    endc: int,
    col: ColumnState,
    soilstate_inst: SoilState,
    waterstate_inst: WaterState,
    temperature_inst: TemperatureState,
    clm_phys: str,
    nlevgrnd: int,
    nlevsoi: int,
    denh2o: float,
) -> Tuple[WaterState, TemperatureState]:
    """Initialize soil temperature and moisture from CLM netCDF history file.
    
    Vectorized version for JIT compilation.
    
    Reference: CLMml_driver.F90:427-557
    
    Args:
        ncfilename: Path to CLM netCDF history file
        strt: Current time slice to retrieve (1-indexed)
        begc: First column index (1-indexed)
        endc: Last column index (1-indexed)
        col: Column state with layer thickness and bedrock depth
        soilstate_inst: Soil state with saturation values
        waterstate_inst: Water state to be updated
        temperature_inst: Temperature state to be updated
        clm_phys: CLM physics version ('CLM4_5' or 'CLM5_0')
        nlevgrnd: Number of ground layers
        nlevsoi: Number of soil layers (hydrologically active)
        denh2o: Density of liquid water [kg/m3]
        
    Returns:
        Tuple of (updated_waterstate, updated_temperature)
    """
    # Open netCDF file and read data
    with nc.Dataset(ncfilename, 'r') as ncid:
        tsoi_var = ncid.variables['TSOI']
        tsoi_loc = jnp.array(tsoi_var[strt-1, :, 0, 0])
        
        h2osoi_var = ncid.variables['H2OSOI']
        
        if clm_phys == 'CLM4_5':
            h2osoi_loc = jnp.array(h2osoi_var[strt-1, :nlevgrnd, 0, 0])
        elif clm_phys == 'CLM5_0':
            h2osoi_loc = jnp.array(h2osoi_var[strt-1, :nlevsoi, 0, 0])
        else:
            raise ValueError(f"Unknown clm_phys: {clm_phys}")
    
    n_cols = endc - begc + 1
    
    # Broadcast temperature to all columns (lines 516-518)
    t_soisno_new = jnp.tile(tsoi_loc[jnp.newaxis, :], (n_cols, 1))
    
    # Initialize moisture array
    h2osoi_vol_new = jnp.zeros((n_cols, nlevgrnd))
    
    # Set moisture based on CLM physics version
    if clm_phys == 'CLM4_5':
        h2osoi_vol_new = jnp.tile(h2osoi_loc[jnp.newaxis, :], (n_cols, 1))
    elif clm_phys == 'CLM5_0':
        h2osoi_vol_new = h2osoi_vol_new.at[:, :nlevsoi].set(
            jnp.tile(h2osoi_loc[jnp.newaxis, :], (n_cols, 1))
        )
    
    # Limit to saturation for CLM5.0 (lines 532-537)
    if clm_phys == 'CLM5_0':
        col_indices = jnp.arange(begc - 1, endc)
        layer_indices = jnp.arange(nlevgrnd)
        
        nbedrock_broadcast = col.nbedrock[col_indices, jnp.newaxis]
        layer_broadcast = layer_indices[jnp.newaxis, :]
        
        active_mask = layer_broadcast < nbedrock_broadcast
        
        watsat_subset = soilstate_inst.watsat_col[col_indices, :]
        
        h2osoi_vol_limited = jnp.minimum(h2osoi_vol_new, watsat_subset)
        h2osoi_vol_new = jnp.where(active_mask, h2osoi_vol_limited, h2osoi_vol_new)
    
    # Calculate liquid water (lines 541-544)
    dz_subset = col.dz[begc-1:endc, :]
    h2osoi_liq_new = h2osoi_vol_new * dz_subset * denh2o
    
    # Ice is zero (line 543)
    h2osoi_ice_new = jnp.zeros((n_cols, nlevgrnd))
    
    # Update state tuples
    waterstate_updated = WaterState(
        h2osoi_vol_col=waterstate_inst.h2osoi_vol_col.at[begc-1:endc, :].set(h2osoi_vol_new),
        h2osoi_ice_col=waterstate_inst.h2osoi_ice_col.at[begc-1:endc, :].set(h2osoi_ice_new),
        h2osoi_liq_col=waterstate_inst.h2osoi_liq_col.at[begc-1:endc, :].set(h2osoi_liq_new),
    )
    
    temperature_updated = TemperatureState(
        t_a10_patch=temperature_inst.t_a10_patch,
        t_soisno_col=temperature_inst.t_soisno_col.at[begc-1:endc, :].set(t_soisno_new),
    )
    
    return waterstate_updated, temperature_updated


# ============================================================================
# Output Functions
# ============================================================================


def compute_upward_shortwave(
    albcan_vis: jnp.ndarray,
    albcan_nir: jnp.ndarray,
    swskyb_vis: jnp.ndarray,
    swskyd_vis: jnp.ndarray,
    swskyb_nir: jnp.ndarray,
    swskyd_nir: jnp.ndarray,
) -> jnp.ndarray:
    """Compute upward shortwave radiation.
    
    Reference: CLMml_driver.F90:600-601
    
    Args:
        albcan_vis: Canopy albedo for visible [0-1] [n_patches]
        albcan_nir: Canopy albedo for near-infrared [0-1] [n_patches]
        swskyb_vis: Direct beam visible SW [W/m2] [n_patches]
        swskyd_vis: Diffuse visible SW [W/m2] [n_patches]
        swskyb_nir: Direct beam NIR SW [W/m2] [n_patches]
        swskyd_nir: Diffuse NIR SW [W/m2] [n_patches]
        
    Returns:
        Upward shortwave radiation [W/m2] [n_patches]
    """
    swup = (albcan_vis * (swskyb_vis + swskyd_vis) +
            albcan_nir * (swskyb_nir + swskyd_nir))
    return swup


def compute_output_fluxes(
    mlcan: MLCanopyState,
) -> OutputFluxes:
    """Compute canopy and soil flux outputs.
    
    Reference: CLMml_driver.F90:594-607
    
    Args:
        mlcan: Multi-layer canopy state
        
    Returns:
        OutputFluxes containing all flux variables
    """
    p = 0  # Single patch (0-based indexing)
    
    swup = compute_upward_shortwave(
        mlcan.albcan_canopy[p, IVIS],
        mlcan.albcan_canopy[p, INIR],
        mlcan.swskyb_forcing[p, IVIS],
        mlcan.swskyd_forcing[p, IVIS],
        mlcan.swskyb_forcing[p, INIR],
        mlcan.swskyd_forcing[p, INIR],
    )
    
    return OutputFluxes(
        rnet_canopy=mlcan.rnet_canopy[p],
        stflx_canopy=mlcan.stflx_canopy[p],
        shflx_canopy=mlcan.shflx_canopy[p],
        lhflx_canopy=mlcan.lhflx_canopy[p],
        gppveg_canopy=mlcan.gppveg_canopy[p],
        ustar_canopy=mlcan.ustar_canopy[p],
        swup=swup,
        lwup_canopy=mlcan.lwup_canopy[p],
        taf_canopy=mlcan.taf_canopy[p],
        gsoi_soil=mlcan.gsoi_soil[p],
        rnsoi_soil=mlcan.rnsoi_soil[p],
        shsoi_soil=mlcan.shsoi_soil[p],
        lhsoi_soil=mlcan.lhsoi_soil[p],
    )


def compute_output_sunshade(
    mlcan: MLCanopyState,
) -> OutputSunShade:
    """Compute sunlit/shaded canopy flux outputs.
    
    Reference: CLMml_driver.F90:609-625
    
    Args:
        mlcan: Multi-layer canopy state
        
    Returns:
        OutputSunShade containing all sun/shade variables
    """
    p = 0  # Single patch
    
    # Convert solar zenith angle from radians to degrees (line 612)
    solar_zen_deg = mlcan.solar_zen_forcing[p] * 180.0 / PHYSICAL_CONSTANTS.pi
    
    # Total visible shortwave (line 613)
    sw_vis_total = mlcan.swskyb_forcing[p, IVIS] + mlcan.swskyd_forcing[p, IVIS]
    
    # Total LAI + SAI (line 614)
    lai_sai_total = mlcan.lai_canopy[p] + mlcan.sai_canopy[p]
    
    return OutputSunShade(
        solar_zen_deg=solar_zen_deg,
        sw_vis_total=sw_vis_total,
        lai_sai_total=lai_sai_total,
        laisun=mlcan.laisun_canopy[p],
        laisha=mlcan.laisha_canopy[p],
        swveg_vis=mlcan.swveg_canopy[p, IVIS],
        swvegsun_vis=mlcan.swvegsun_canopy[p, IVIS],
        swvegsha_vis=mlcan.swvegsha_canopy[p, IVIS],
        gppveg=mlcan.gppveg_canopy[p],
        gppvegsun=mlcan.gppvegsun_canopy[p],
        gppvegsha=mlcan.gppvegsha_canopy[p],
        lhveg=mlcan.lhveg_canopy[p],
        lhvegsun=mlcan.lhvegsun_canopy[p],
        lhvegsha=mlcan.lhvegsha_canopy[p],
        shveg=mlcan.shveg_canopy[p],
        shvegsun=mlcan.shvegsun_canopy[p],
        shvegsha=mlcan.shvegsha_canopy[p],
        vcmax25veg=mlcan.vcmax25veg_canopy[p],
        vcmax25sun=mlcan.vcmax25sun_canopy[p],
        vcmax25sha=mlcan.vcmax25sha_canopy[p],
        gsveg=mlcan.gsveg_canopy[p],
        gsvegsun=mlcan.gsvegsun_canopy[p],
        gsvegsha=mlcan.gsvegsha_canopy[p],
        windveg=mlcan.windveg_canopy[p],
        windvegsun=mlcan.windvegsun_canopy[p],
        windvegsha=mlcan.windvegsha_canopy[p],
        tlveg=mlcan.tlveg_canopy[p],
        tlvegsun=mlcan.tlvegsun_canopy[p],
        tlvegsha=mlcan.tlvegsha_canopy[p],
        taveg=mlcan.taveg_canopy[p],
        tavegsun=mlcan.tavegsun_canopy[p],
        tavegsha=mlcan.tavegsha_canopy[p],
    )


def compute_output_auxiliary(
    mlcan: MLCanopyState,
    soilstate: SoilState,
) -> OutputAuxiliary:
    """Compute leaf water potential and soil moisture stress outputs.
    
    Reference: CLMml_driver.F90:627-636
    
    Args:
        mlcan: Multi-layer canopy state
        soilstate: Soil state
        
    Returns:
        OutputAuxiliary containing water stress variables
    """
    p = 0  # Single patch
    
    # Get top and mid-canopy layer indices (lines 633-634)
    top = mlcan.ntop_canopy[p]
    nbot = mlcan.nbot_canopy[p]
    mid = jnp.maximum(1, nbot + (top - nbot + 1) // 2 - 1)
    
    return OutputAuxiliary(
        btran=soilstate.btran_soil[p],
        lsc_top=mlcan.lsc_profile[p, top],
        psis=soilstate.psis_soil[p],
        lwp_mean_top=mlcan.lwp_mean_profile[p, top],
        lwp_mean_mid=mlcan.lwp_mean_profile[p, mid],
        fracminlwp=mlcan.fracminlwp_canopy[p],
    )


def compute_output_profile_above_canopy(
    mlcan: MLCanopyState,
    curr_calday: float,
) -> OutputProfile:
    """Compute vertical profile outputs above canopy.
    
    Reference: CLMml_driver.F90:638-666
    
    Args:
        mlcan: Multi-layer canopy state
        curr_calday: Current calendar day [days]
        
    Returns:
        OutputProfile containing vertical profile data
    """
    p = 0  # Single patch
    
    ncan = mlcan.ncan_canopy[p]
    ntop = mlcan.ntop_canopy[p]
    
    # Extract profile data for layers above canopy
    layer_indices = jnp.arange(ntop + 1, ncan + 1)
    
    tair = mlcan.tair_profile[p, layer_indices]
    
    # Convert vapor pressure to specific humidity (lines 653-655)
    mmh2o = PHYSICAL_CONSTANTS.mwh2o
    mmdry = PHYSICAL_CONSTANTS.mwdry
    eair = mlcan.eair_profile[p, layer_indices]
    pref = mlcan.pref_forcing[p]
    
    qair = (1000.0 * (mmh2o / mmdry) * eair /
            (pref - (1.0 - mmh2o / mmdry) * eair))
    
    wind = mlcan.wind_profile[p, layer_indices]
    zs = mlcan.zs_profile[p, layer_indices]
    
    return OutputProfile(
        curr_calday=curr_calday,
        zs=zs,
        wind=wind,
        tair=tair,
        qair=qair,
    )


def compute_canopy_layer_output(
    ic: int,
    p: int,
    curr_calday: float,
    mlcan: MLCanopyState,
) -> CanopyLayerOutput:
    """Compute output data for a single canopy layer.
    
    Reference: CLMml_driver.F90:667-730
    
    Args:
        ic: Canopy layer index [0-based]
        p: Patch index
        curr_calday: Current calendar day [days]
        mlcan: Multi-layer canopy state
        
    Returns:
        CanopyLayerOutput containing all layer variables
    """
    mmh2o = PHYSICAL_CONSTANTS.mwh2o
    mmdry = PHYSICAL_CONSTANTS.mwdry
    
    # Extract layer values
    tair = mlcan.tair_profile[p, ic]
    
    # Convert vapor pressure to specific humidity
    eair_pa = mlcan.eair_profile[p, ic]
    qair = 1000.0 * (mmh2o / mmdry) * eair_pa / (
        mlcan.pref_forcing[p] - (1.0 - mmh2o / mmdry) * eair_pa
    )
    
    # Leaf area density
    dpai = mlcan.dpai_profile[p, ic]
    dz = mlcan.dz_profile[p, ic]
    lad = jnp.where(dz > 0.0, dpai / dz, 0.0)
    
    # Check if this is a leaf layer
    has_leaves = dpai > 0.0
    
    # Extract layer properties
    zs = mlcan.zs_profile[p, ic]
    fracsun = mlcan.fracsun_profile[p, ic]
    wind = mlcan.wind_profile[p, ic]
    
    # Compute LAD components
    lad_sun = lad * fracsun
    lad_shade = lad * (1.0 - fracsun)
    
    # Extract leaf fluxes (use jnp.where for missing values)
    rnleaf_sun = jnp.where(has_leaves, mlcan.rnleaf_leaf[p, ic, ISUN], MISSING_VALUE)
    rnleaf_shade = jnp.where(has_leaves, mlcan.rnleaf_leaf[p, ic, ISHA], MISSING_VALUE)
    
    shleaf_sun = jnp.where(has_leaves, mlcan.shleaf_leaf[p, ic, ISUN], MISSING_VALUE)
    shleaf_shade = jnp.where(has_leaves, mlcan.shleaf_leaf[p, ic, ISHA], MISSING_VALUE)
    
    lhleaf_sun = jnp.where(has_leaves, mlcan.lhleaf_leaf[p, ic, ISUN], MISSING_VALUE)
    lhleaf_shade = jnp.where(has_leaves, mlcan.lhleaf_leaf[p, ic, ISHA], MISSING_VALUE)
    
    anet_sun = jnp.where(has_leaves, mlcan.anet_leaf[p, ic, ISUN], MISSING_VALUE)
    anet_shade = jnp.where(has_leaves, mlcan.anet_leaf[p, ic, ISHA], MISSING_VALUE)
    
    apar_sun = jnp.where(has_leaves, mlcan.apar_leaf[p, ic, ISUN], MISSING_VALUE)
    apar_shade = jnp.where(has_leaves, mlcan.apar_leaf[p, ic, ISHA], MISSING_VALUE)
    
    gs_sun = jnp.where(has_leaves, mlcan.gs_leaf[p, ic, ISUN], MISSING_VALUE)
    gs_shade = jnp.where(has_leaves, mlcan.gs_leaf[p, ic, ISHA], MISSING_VALUE)
    
    lwp_sun = jnp.where(has_leaves, mlcan.lwp_hist_leaf[p, ic, ISUN], MISSING_VALUE)
    lwp_shade = jnp.where(has_leaves, mlcan.lwp_hist_leaf[p, ic, ISHA], MISSING_VALUE)
    
    tleaf_sun = jnp.where(has_leaves, mlcan.tleaf_hist_leaf[p, ic, ISUN], MISSING_VALUE)
    tleaf_shade = jnp.where(has_leaves, mlcan.tleaf_hist_leaf[p, ic, ISHA], MISSING_VALUE)
    
    vcmax25_sun = jnp.where(has_leaves, mlcan.vcmax25_leaf[p, ic, ISUN], MISSING_VALUE)
    vcmax25_shade = jnp.where(has_leaves, mlcan.vcmax25_leaf[p, ic, ISHA], MISSING_VALUE)
    
    # Handle LAD for non-leaf layers
    lad = jnp.where(has_leaves, lad, ZERO_VALUE)
    lad_sun = jnp.where(has_leaves, lad_sun, ZERO_VALUE)
    lad_shade = jnp.where(has_leaves, lad_shade, ZERO_VALUE)
    
    return CanopyLayerOutput(
        curr_calday=curr_calday,
        zs=zs,
        fracsun=fracsun,
        lad=lad,
        lad_sun=lad_sun,
        lad_shade=lad_shade,
        rnleaf_sun=rnleaf_sun,
        rnleaf_shade=rnleaf_shade,
        shleaf_sun=shleaf_sun,
        shleaf_shade=shleaf_shade,
        lhleaf_sun=lhleaf_sun,
        lhleaf_shade=lhleaf_shade,
        anet_sun=anet_sun,
        anet_shade=anet_shade,
        apar_sun=apar_sun,
        apar_shade=apar_shade,
        gs_sun=gs_sun,
        gs_shade=gs_shade,
        lwp_sun=lwp_sun,
        lwp_shade=lwp_shade,
        tleaf_sun=tleaf_sun,
        tleaf_shade=tleaf_shade,
        vcmax25_sun=vcmax25_sun,
        vcmax25_shade=vcmax25_shade,
        wind=wind,
        tair=tair,
        qair=qair,
    )


def compute_ground_output(
    soilstate: SoilState,
) -> GroundOutputState:
    """Compute ground surface output diagnostics.
    
    Reference: CLMml_driver.F90:731-748
    
    Args:
        soilstate: Soil state
        
    Returns:
        GroundOutputState containing ground diagnostics
    """
    # Ground air temperature (using soil temperature directly)
    tair = soilstate.tg_soil
    
    # Ground vapor pressure: convert from Pa to kPa
    eair = soilstate.eg_soil / 1000.0
    
    # Aerodynamic resistance
    ra = soilstate.rhomol_forcing / soilstate.gac0_soil
    
    return GroundOutputState(
        tair=tair,
        eair=eair,
        ra=ra,
    )


def clmml_drv_cleanup(
    turb_type: int,
) -> CleanupState:
    """Prepare cleanup state for CLM-ML driver.
    
    Reference: CLMml_driver.F90:272-293
    
    Args:
        turb_type: Turbulence type flag [-1 for prescribed, other for computed]
        
    Returns:
        CleanupState indicating which files to close and success status
    """
    # Always close all output files
    close_nout1 = True
    close_nout2 = True
    close_nout3 = True
    close_nout4 = True
    
    # Close input file only if turb_type == -1
    close_nin1 = (turb_type == -1)
    
    # Indicate successful completion
    success = True
    
    return CleanupState(
        close_nout1=close_nout1,
        close_nout2=close_nout2,
        close_nout3=close_nout3,
        close_nout4=close_nout4,
        close_nin1=close_nin1,
        success=success,
    )


# ============================================================================
# Profile Reading Functions
# ============================================================================


def convert_qair_to_eair(
    qair: jnp.ndarray,
    pref: jnp.ndarray,
    mmh2o: float,
    mmdry: float,
) -> jnp.ndarray:
    """Convert specific humidity to vapor pressure.
    
    Reference: CLMml_driver.F90:838
    
    Args:
        qair: Specific humidity [kg/kg] [n_patches, n_layers]
        pref: Reference pressure [Pa] [n_patches]
        mmh2o: Molecular weight of water [g/mol]
        mmdry: Molecular weight of dry air [g/mol]
        
    Returns:
        Vapor pressure [Pa] [n_patches, n_layers]
    """
    pref_expanded = pref[:, jnp.newaxis]
    mm_ratio = mmh2o / mmdry
    
    eair = qair * pref_expanded / (mm_ratio + (1.0 - mm_ratio) * qair)
    
    return eair


def process_profile_data(
    wind_raw: jnp.ndarray,
    tair_raw: jnp.ndarray,
    qair_raw: jnp.ndarray,
    pref: jnp.ndarray,
    mmh2o: float = 18.016,
    mmdry: float = 28.966,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Process raw profile data from dataset.
    
    Reference: CLMml_driver.F90:827-838
    
    Args:
        wind_raw: Wind speed from dataset [m/s] [n_patches, n_layers]
        tair_raw: Air temperature from dataset [K] [n_patches, n_layers]
        qair_raw: Specific humidity from dataset [g/kg] [n_patches, n_layers]
        pref: Reference pressure [Pa] [n_patches]
        mmh2o: Molecular weight of water [g/mol]
        mmdry: Molecular weight of dry air [g/mol]
        
    Returns:
        Tuple of (wind_data, tair_data, eair_data)
    """
    # Convert g/kg to kg/kg (line 827)
    qair_kgkg = qair_raw / 1000.0
    
    # Convert specific humidity to vapor pressure (line 838)
    eair_data = convert_qair_to_eair(qair_kgkg, pref, mmh2o, mmdry)
    
    # Wind and temperature are used directly
    wind_data = wind_raw
    tair_data = tair_raw
    
    return wind_data, tair_data, eair_data


def read_canopy_profiles_physics(
    wind_raw: jnp.ndarray,
    tair_raw: jnp.ndarray,
    qair_raw: jnp.ndarray,
    profile_data: CanopyProfileData,
) -> CanopyProfileData:
    """Apply physics calculations to canopy profile data.
    
    Reference: CLMml_driver.F90:751-841
    
    Args:
        wind_raw: Wind speed from dataset [m/s] [n_patches, n_layers]
        tair_raw: Air temperature from dataset [K] [n_patches, n_layers]
        qair_raw: Specific humidity from dataset [g/kg] [n_patches, n_layers]
        profile_data: Current canopy profile data structure
        
    Returns:
        Updated CanopyProfileData with processed values
    """
    wind_data, tair_data, eair_data = process_profile_data(
        wind_raw,
        tair_raw,
        qair_raw,
        profile_data.pref,
    )
    
    return CanopyProfileData(
        ncan=profile_data.ncan,
        zs=profile_data.zs,
        wind_data=wind_data,
        tair_data=tair_data,
        eair_data=eair_data,
        pref=profile_data.pref,
    )


# ============================================================================
# Module Exports
# ============================================================================


__all__ = [
    # Protocols
    "CLMmlDriverProtocol",
    
    # Types
    "BoundsType",
    "Atm2LndState",
    "TemperatureState",
    "FrictionVelState",
    "MLCanopyState",
    "CanopyState",
    "PatchInfo",
    "ColumnState",
    "SoilState",
    "WaterState",
    "PhysicalConstants",
    "PFTParameters",
    "InitializationState",
    "CLMHistoryFileInfo",
    "TimeStepState",
    "OutputFluxes",
    "OutputSunShade",
    "OutputAuxiliary",
    "OutputProfile",
    "CanopyLayerOutput",
    "GroundOutputState",
    "CanopyProfileData",
    "CleanupState",
    "TowerVegInputs",
    
    # Constants
    "PHYSICAL_CONSTANTS",
    "IVIS",
    "INIR",
    "ISUN",
    "ISHA",
    "MISSING_VALUE",
    "ZERO_VALUE",
    "HTOP_PFT_DEFAULT",
    "PINE_PBETA_LAI",
    "PINE_QBETA_LAI",
    "PINE_PBETA_SAI",
    "PINE_QBETA_SAI",
    "DEFAULT_ROOT_BIOMASS",
    
    # Initialization functions
    "adjust_usme2_pft_parameters",
    "construct_tower_file_path",
    "construct_clm_filename",
    "calculate_time_index",
    "get_htop_pft_lookup",
    "init_acclim",
    "tower_veg",
    "soil_init_vectorized",
    
    # Output functions
    "compute_upward_shortwave",
    "compute_output_fluxes",
    "compute_output_sunshade",
    "compute_output_auxiliary",
    "compute_output_profile_above_canopy",
    "compute_canopy_layer_output",
    "compute_ground_output",
    "clmml_drv_cleanup",
    
    # Profile reading functions
    "convert_qair_to_eair",
    "process_profile_data",
    "read_canopy_profiles_physics",
]