"""
Multilayer Canopy Data Structure and Initialization.

Translated from CTSM's MLCanopyFluxesType.F90

This module defines the data structure for the multilayer canopy model and provides
initialization routines. The multilayer canopy model represents vertical structure
within the canopy, tracking energy, water, and carbon exchange at multiple levels.

Key components:
    - MLCanopyState: Immutable state container for all canopy variables
    - Initialization routines: allocate, history setup, cold start
    - Restart I/O interface

The structure uses NamedTuples for immutability and JIT compatibility.
All arrays are dimensioned as:
    - [n_patches]: Single-level canopy or soil variables
    - [n_patches, nlevmlcan]: Multi-level canopy profiles
    - [n_patches, nlevmlcan, nleaf]: Sunlit/shaded leaf variables (nleaf=2)
    - [n_patches, numrad]: Radiation band variables (numrad=2: visible, near-IR)

Reference:
    MLCanopyFluxesType.F90:1-711
"""

from typing import NamedTuple, Dict, Any
import jax
import jax.numpy as jnp


# =============================================================================
# Constants and Parameters
# =============================================================================

# Special values for uninitialized data
SPVAL = 1.0e36  # Special value for real variables
ISPVAL = -9999  # Special value for integer variables

# Canopy structure constants
NLEAF = 2  # Number of leaf types: 0=sunlit, 1=shaded
ISUN = 0   # Index for sunlit leaves
ISHA = 1   # Index for shaded leaves

# Default initialization values
DEFAULT_LWP = -0.1  # Default leaf water potential [MPa]
DEFAULT_H2OCAN = 0.0  # Default intercepted water [kg/m2]


# =============================================================================
# Type Definitions
# =============================================================================

class BoundsType(NamedTuple):
    """Domain bounds structure.
    
    Attributes:
        begp: Beginning patch index
        endp: Ending patch index
    """
    begp: int
    endp: int


class MLCanopyState(NamedTuple):
    """
    Multilayer canopy state variables.
    
    This is the JAX translation of the mlcanopy_type derived type from
    MLCanopyFluxesType.F90 (lines 1-317).
    
    Variables follow naming conventions:
    - var_canopy: single-level canopy variable
    - var_soil: single-level soil variable  
    - var_forcing: single-level atmospheric forcing variable
    - var_profile: multi-level variable at each canopy layer
    - var_leaf: multi-level variable for sunlit and shaded leaves
    
    All fluxes are per m2 ground area unless noted as "per m2 leaf".
    """
    
    # ============================================================================
    # Vegetation input variables: [n_patches]
    # Lines 38-42
    # ============================================================================
    
    ztop_canopy: jnp.ndarray  # Canopy foliage top height [m]
    zbot_canopy: jnp.ndarray  # Canopy foliage bottom height [m]
    lai_canopy: jnp.ndarray  # Leaf area index of canopy [m2/m2]
    sai_canopy: jnp.ndarray  # Stem area index of canopy [m2/m2]
    root_biomass_canopy: jnp.ndarray  # Fine root biomass [g biomass/m2]
    
    # ============================================================================
    # Atmospheric forcing variables: [n_patches] or [n_patches, numrad]
    # Lines 44-58
    # ============================================================================
    
    zref_forcing: jnp.ndarray  # Atmospheric reference height [m]
    tref_forcing: jnp.ndarray  # Air temperature at reference height [K]
    qref_forcing: jnp.ndarray  # Specific humidity at reference height [kg/kg]
    uref_forcing: jnp.ndarray  # Wind speed at reference height [m/s]
    pref_forcing: jnp.ndarray  # Air pressure at reference height [Pa]
    co2ref_forcing: jnp.ndarray  # Atmospheric CO2 at reference height [umol/mol]
    o2ref_forcing: jnp.ndarray  # Atmospheric O2 at reference height [mmol/mol]
    swskyb_forcing: jnp.ndarray  # Atmospheric direct beam solar radiation [W/m2] [n_patches, numrad]
    swskyd_forcing: jnp.ndarray  # Atmospheric diffuse solar radiation [W/m2] [n_patches, numrad]
    lwsky_forcing: jnp.ndarray  # Atmospheric longwave radiation [W/m2]
    qflx_rain_forcing: jnp.ndarray  # Rainfall [mm H2O/s = kg H2O/m2/s]
    qflx_snow_forcing: jnp.ndarray  # Snowfall [mm H2O/s = kg H2O/m2/s]
    tacclim_forcing: jnp.ndarray  # Average air temperature for acclimation [K]
    
    # ============================================================================
    # Derived atmospheric forcing variables: [n_patches]
    # Lines 60-68
    # ============================================================================
    
    eref_forcing: jnp.ndarray  # Vapor pressure at reference height [Pa]
    thref_forcing: jnp.ndarray  # Atmospheric potential temperature at reference height [K]
    thvref_forcing: jnp.ndarray  # Atmospheric virtual potential temperature at reference height [K]
    rhoair_forcing: jnp.ndarray  # Air density at reference height [kg/m3]
    rhomol_forcing: jnp.ndarray  # Molar density at reference height [mol/m3]
    mmair_forcing: jnp.ndarray  # Molecular mass of air at reference height [kg/mol]
    cpair_forcing: jnp.ndarray  # Specific heat of air (constant pressure) at reference height [J/mol/K]
    solar_zen_forcing: jnp.ndarray  # Solar zenith angle [radians]
    
    # ============================================================================
    # Canopy flux variables: [n_patches] or [n_patches, numrad]
    # Lines 70-130
    # ============================================================================
    
    swveg_canopy: jnp.ndarray  # Absorbed solar radiation: vegetation [W/m2] [n_patches, numrad]
    swvegsun_canopy: jnp.ndarray  # Absorbed solar radiation: sunlit canopy [W/m2] [n_patches, numrad]
    swvegsha_canopy: jnp.ndarray  # Absorbed solar radiation: shaded canopy [W/m2] [n_patches, numrad]
    
    lwveg_canopy: jnp.ndarray  # Absorbed longwave radiation: vegetation [W/m2]
    lwvegsun_canopy: jnp.ndarray  # Absorbed longwave radiation: sunlit canopy [W/m2]
    lwvegsha_canopy: jnp.ndarray  # Absorbed longwave radiation: shaded canopy [W/m2]
    
    shveg_canopy: jnp.ndarray  # Sensible heat flux: vegetation [W/m2]
    shvegsun_canopy: jnp.ndarray  # Sensible heat flux: sunlit canopy [W/m2]
    shvegsha_canopy: jnp.ndarray  # Sensible heat flux: shaded canopy [W/m2]
    
    lhveg_canopy: jnp.ndarray  # Latent heat flux: vegetation [W/m2]
    lhvegsun_canopy: jnp.ndarray  # Latent heat flux: sunlit canopy [W/m2]
    lhvegsha_canopy: jnp.ndarray  # Latent heat flux: shaded canopy [W/m2]
    
    etveg_canopy: jnp.ndarray  # Water vapor flux: vegetation [mol H2O/m2/s]
    etvegsun_canopy: jnp.ndarray  # Water vapor flux: sunlit canopy [mol H2O/m2/s]
    etvegsha_canopy: jnp.ndarray  # Water vapor flux: shaded canopy [mol H2O/m2/s]
    
    gppveg_canopy: jnp.ndarray  # Gross primary production: vegetation [umol CO2/m2/s]
    gppvegsun_canopy: jnp.ndarray  # Gross primary production: sunlit canopy [umol CO2/m2/s]
    gppvegsha_canopy: jnp.ndarray  # Gross primary production: shaded canopy [umol CO2/m2/s]
    
    vcmax25veg_canopy: jnp.ndarray  # Vcmax at 25C: total canopy [umol/m2/s]
    vcmax25sun_canopy: jnp.ndarray  # Vcmax at 25C: sunlit canopy [umol/m2/s]
    vcmax25sha_canopy: jnp.ndarray  # Vcmax at 25C: shaded canopy [umol/m2/s]
    
    gsveg_canopy: jnp.ndarray  # Stomatal conductance: canopy [mol H2O/m2/s]
    gsvegsun_canopy: jnp.ndarray  # Stomatal conductance: sunlit canopy [mol H2O/m2/s]
    gsvegsha_canopy: jnp.ndarray  # Stomatal conductance: shaded canopy [mol H2O/m2/s]
    
    windveg_canopy: jnp.ndarray  # Wind speed: canopy [m/s]
    windvegsun_canopy: jnp.ndarray  # Wind speed: sunlit canopy [m/s]
    windvegsha_canopy: jnp.ndarray  # Wind speed: shaded canopy [m/s]
    
    tlveg_canopy: jnp.ndarray  # Leaf temperature: canopy [K]
    tlvegsun_canopy: jnp.ndarray  # Leaf temperature: sunlit canopy [K]
    tlvegsha_canopy: jnp.ndarray  # Leaf temperature: shaded canopy [K]
    
    taveg_canopy: jnp.ndarray  # Air temperature: canopy [K]
    tavegsun_canopy: jnp.ndarray  # Air temperature: sunlit canopy [K]
    tavegsha_canopy: jnp.ndarray  # Air temperature: shaded canopy [K]
    
    laisun_canopy: jnp.ndarray  # Canopy plant area index (lai+sai): sunlit canopy [m2/m2]
    laisha_canopy: jnp.ndarray  # Canopy plant area index (lai+sai): shaded canopy [m2/m2]
    
    albcan_canopy: jnp.ndarray  # Albedo above canopy [-] [n_patches, numrad]
    lwup_canopy: jnp.ndarray  # Upward longwave radiation above canopy [W/m2]
    rnet_canopy: jnp.ndarray  # Total net radiation, including soil [W/m2]
    shflx_canopy: jnp.ndarray  # Total sensible heat flux, including soil [W/m2]
    lhflx_canopy: jnp.ndarray  # Total latent heat flux, including soil [W/m2]
    etflx_canopy: jnp.ndarray  # Total water vapor flux, including soil [mol H2O/m2/s]
    stflx_canopy: jnp.ndarray  # Canopy storage heat flux [W/m2]
    ustar_canopy: jnp.ndarray  # Friction velocity [m/s]
    gac_to_hc_canopy: jnp.ndarray  # Aerodynamic conductance for a scalar above canopy [mol/m2/s]
    
    qflx_intr_canopy: jnp.ndarray  # Intercepted precipitation [kg H2O/m2/s]
    qflx_tflrain_canopy: jnp.ndarray  # Total rain throughfall onto ground [kg H2O/m2/s]
    qflx_tflsnow_canopy: jnp.ndarray  # Total snow throughfall onto ground [kg H2O/m2/s]
    
    # ============================================================================
    # Canopy diagnostic variables: [n_patches]
    # Lines 132-136
    # ============================================================================
    
    uaf_canopy: jnp.ndarray  # Wind speed at canopy top [m/s]
    taf_canopy: jnp.ndarray  # Air temperature at canopy top [K]
    qaf_canopy: jnp.ndarray  # Specific humidity at canopy top [kg/kg]
    fracminlwp_canopy: jnp.ndarray  # Fraction of canopy that is water-stressed
    
    # ============================================================================
    # Canopy aerodynamic variables: [n_patches]
    # Lines 138-146
    # ============================================================================
    
    obu_canopy: jnp.ndarray  # Obukhov length [m]
    obuold_canopy: jnp.ndarray  # Obukhov length from previous iteration
    nmozsgn_canopy: jnp.ndarray  # Number of times stability changes sign during iteration [int]
    beta_canopy: jnp.ndarray  # Value of u* / u at canopy top [-]
    PrSc_canopy: jnp.ndarray  # Prandtl (Schmidt) number at canopy top [-]
    Lc_canopy: jnp.ndarray  # Canopy density length scale [m]
    zdisp_canopy: jnp.ndarray  # Displacement height [m]
    
    # ============================================================================
    # Canopy stomatal conductance variables: [n_patches]
    # Lines 148-151
    # ============================================================================
    
    g0_canopy: jnp.ndarray  # Ball-Berry or Medlyn minimum leaf conductance [mol H2O/m2/s]
    g1_canopy: jnp.ndarray  # Ball-Berry or Medlyn slope parameter
    
    # ============================================================================
    # Soil energy balance variables: [n_patches] or [n_patches, numrad]
    # Lines 153-172
    # ============================================================================
    
    albsoib_soil: jnp.ndarray  # Direct beam albedo of ground [-] [n_patches, numrad]
    albsoid_soil: jnp.ndarray  # Diffuse albedo of ground [-] [n_patches, numrad]
    swsoi_soil: jnp.ndarray  # Absorbed solar radiation: ground [W/m2] [n_patches, numrad]
    lwsoi_soil: jnp.ndarray  # Absorbed longwave radiation: ground [W/m2]
    rnsoi_soil: jnp.ndarray  # Net radiation: ground [W/m2]
    shsoi_soil: jnp.ndarray  # Sensible heat flux: ground [W/m2]
    lhsoi_soil: jnp.ndarray  # Latent heat flux: ground [W/m2]
    etsoi_soil: jnp.ndarray  # Water vapor flux: ground [mol H2O/m2/s]
    gsoi_soil: jnp.ndarray  # Soil heat flux [W/m2]
    tg_soil: jnp.ndarray  # Soil surface temperature [K]
    tg_bef_soil: jnp.ndarray  # Soil surface temperature for previous timestep [K]
    eg_soil: jnp.ndarray  # Soil surface vapor pressure [Pa]
    rhg_soil: jnp.ndarray  # Relative humidity of airspace at soil surface [fraction]
    gac0_soil: jnp.ndarray  # Aerodynamic conductance for soil fluxes [mol/m2/s]
    soil_t_soil: jnp.ndarray  # Temperature of first snow/soil layer [K]
    soil_dz_soil: jnp.ndarray  # Depth to temperature of first snow/soil layer [m]
    soil_tk_soil: jnp.ndarray  # Thermal conductivity of first snow/soil layer [W/m/K]
    soilres_soil: jnp.ndarray  # Soil evaporative resistance [s/m]
    
    # ============================================================================
    # Soil moisture variables: [n_patches] or [n_patches, nlevgrnd]
    # Lines 174-178
    # ============================================================================
    
    btran_soil: jnp.ndarray  # Soil wetness factor for photosynthesis [-]
    psis_soil: jnp.ndarray  # Weighted soil water potential [MPa]
    rsoil_soil: jnp.ndarray  # Soil hydraulic resistance [MPa.s.m2/mmol H2O]
    soil_et_loss_soil: jnp.ndarray  # Fraction of total transpiration from each soil layer [-] [n_patches, nlevgrnd]
    
    # ============================================================================
    # Canopy layer indices: [n_patches]
    # Lines 180-184
    # ============================================================================
    
    ncan_canopy: jnp.ndarray  # Number of aboveground layers [int]
    ntop_canopy: jnp.ndarray  # Index for top leaf layer [int]
    nbot_canopy: jnp.ndarray  # Index for bottom leaf layer [int]
    
    # ============================================================================
    # Canopy layer variables: [n_patches, nlevmlcan] or [n_patches, nlevmlcan, numrad]
    # Lines 186-207
    # ============================================================================
    
    dlai_frac_profile: jnp.ndarray  # Canopy layer leaf area index (fraction of canopy total)
    dsai_frac_profile: jnp.ndarray  # Canopy layer stem area index (fraction of canopy total)
    dlai_profile: jnp.ndarray  # Canopy layer leaf area index [m2/m2]
    dsai_profile: jnp.ndarray  # Canopy layer stem area index [m2/m2]
    dpai_profile: jnp.ndarray  # Canopy layer plant area index [m2/m2]
    zs_profile: jnp.ndarray  # Canopy layer height for scalar concentration and source [m]
    dz_profile: jnp.ndarray  # Canopy layer thickness [m]
    
    vcmax25_profile: jnp.ndarray  # Canopy layer leaf maximum carboxylation rate at 25C [umol/m2/s]
    jmax25_profile: jnp.ndarray  # Canopy layer C3 maximum electron transport rate at 25C [umol/m2/s]
    kp25_profile: jnp.ndarray  # Canopy layer C4 initial slope of CO2 response curve at 25C [mol/m2/s]
    rd25_profile: jnp.ndarray  # Canopy layer leaf respiration rate at 25C [umol/m2/s]
    cpleaf_profile: jnp.ndarray  # Canopy layer leaf heat capacity [J/m2 leaf/K]
    
    fracsun_profile: jnp.ndarray  # Canopy layer sunlit fraction [-]
    kb_profile: jnp.ndarray  # Direct beam extinction coefficient [-]
    tb_profile: jnp.ndarray  # Canopy layer transmittance of direct beam radiation [-]
    td_profile: jnp.ndarray  # Canopy layer transmittance of diffuse radiation [-]
    tbi_profile: jnp.ndarray  # Cumulative transmittance of direct beam onto canopy layer [-]
    
    # ============================================================================
    # Canopy layer source/sink fluxes: [n_patches, nlevmlcan] or [n_patches, nlevmlcan, numrad]
    # Lines 209-217
    # ============================================================================
    
    swsrc_profile: jnp.ndarray  # Canopy layer source/sink flux: absorbed solar radiation [W/m2] [n_patches, nlevmlcan, numrad]
    lwsrc_profile: jnp.ndarray  # Canopy layer source/sink flux: absorbed longwave radiation [W/m2]
    rnsrc_profile: jnp.ndarray  # Canopy layer source/sink flux: net radiation [W/m2]
    stsrc_profile: jnp.ndarray  # Canopy layer source/sink flux: storage heat flux [W/m2]
    shsrc_profile: jnp.ndarray  # Canopy layer source/sink flux: sensible heat [W/m2]
    lhsrc_profile: jnp.ndarray  # Canopy layer source/sink flux: latent heat [W/m2]
    etsrc_profile: jnp.ndarray  # Canopy layer source/sink flux: water vapor [mol H2O/m2/s]
    fco2src_profile: jnp.ndarray  # Canopy layer source/sink flux: CO2 [umol CO2/m2/s]
    
    # ============================================================================
    # Canopy layer scalar profiles: [n_patches, nlevmlcan]
    # Lines 219-230
    # ============================================================================
    
    wind_profile: jnp.ndarray  # Canopy layer wind speed [m/s]
    tair_profile: jnp.ndarray  # Canopy layer air temperature [K]
    eair_profile: jnp.ndarray  # Canopy layer vapor pressure [Pa]
    cair_profile: jnp.ndarray  # Canopy layer atmospheric CO2 [umol/mol]
    tair_bef_profile: jnp.ndarray  # Canopy layer air temperature for previous timestep [K]
    eair_bef_profile: jnp.ndarray  # Canopy layer vapor pressure for previous timestep [Pa]
    cair_bef_profile: jnp.ndarray  # Canopy layer atmospheric CO2 for previous timestep [umol/mol]
    wind_data_profile: jnp.ndarray  # Canopy layer wind speed FROM DATASET [m/s]
    tair_data_profile: jnp.ndarray  # Canopy layer air temperature FROM DATASET [K]
    eair_data_profile: jnp.ndarray  # Canopy layer vapor pressure FROM DATASET [Pa]
    
    # ============================================================================
    # Canopy layer air fluxes: [n_patches, nlevmlcan]
    # Lines 232-236
    # ============================================================================
    
    shair_profile: jnp.ndarray  # Canopy layer air sensible heat flux [W/m2]
    etair_profile: jnp.ndarray  # Canopy layer air water vapor flux [mol H2O/m2/s]
    stair_profile: jnp.ndarray  # Canopy layer air storage heat flux [W/m2]
    gac_profile: jnp.ndarray  # Canopy layer aerodynamic conductance for scalars [mol/m2/s]
    
    # ============================================================================
    # Canopy layer weighted mean leaf variables: [n_patches, nlevmlcan] or [n_patches, nlevmlcan, numrad]
    # Lines 238-250
    # ============================================================================
    
    swleaf_mean_profile: jnp.ndarray  # Canopy layer weighted mean: leaf absorbed solar radiation [W/m2 leaf] [n_patches, nlevmlcan, numrad]
    lwleaf_mean_profile: jnp.ndarray  # Canopy layer weighted mean: leaf absorbed longwave radiation [W/m2 leaf]
    rnleaf_mean_profile: jnp.ndarray  # Canopy layer weighted mean: leaf net radiation [W/m2 leaf]
    stleaf_mean_profile: jnp.ndarray  # Canopy layer weighted mean: leaf storage heat flux [W/m2 leaf]
    shleaf_mean_profile: jnp.ndarray  # Canopy layer weighted mean: leaf sensible heat flux [W/m2 leaf]
    lhleaf_mean_profile: jnp.ndarray  # Canopy layer weighted mean: leaf latent heat flux [W/m2 leaf]
    etleaf_mean_profile: jnp.ndarray  # Canopy layer weighted mean: leaf water vapor flux [mol H2O/m2 leaf/s]
    fco2_mean_profile: jnp.ndarray  # Canopy layer weighted mean: leaf net photosynthesis [umol CO2/m2 leaf/s]
    apar_mean_profile: jnp.ndarray  # Canopy layer weighted mean: absorbed PAR [umol photon/m2 leaf/s]
    gs_mean_profile: jnp.ndarray  # Canopy layer weighted mean: stomatal conductance [mol H2O/m2 leaf/s]
    tleaf_mean_profile: jnp.ndarray  # Canopy layer weighted mean: leaf temperature [K]
    lwp_mean_profile: jnp.ndarray  # Canopy layer weighted mean: leaf water potential [MPa]
    
    # ============================================================================
    # Canopy layer water variables: [n_patches, nlevmlcan]
    # Lines 252-256
    # ============================================================================
    
    lsc_profile: jnp.ndarray  # Canopy layer leaf-specific conductance [mmol H2O/m2 leaf/s/MPa]
    h2ocan_profile: jnp.ndarray  # Canopy layer intercepted water [kg H2O/m2]
    fwet_profile: jnp.ndarray  # Canopy layer fraction of plant area index that is wet
    fdry_profile: jnp.ndarray  # Canopy layer fraction of plant area index that is green and dry
    
    # ============================================================================
    # Sunlit/shaded leaf variables: [n_patches, nlevmlcan, nleaf] or [n_patches, nlevmlcan, nleaf, numrad]
    # Lines 258-317
    # ============================================================================
    
    tleaf_leaf: jnp.ndarray  # Leaf temperature [K]
    tleaf_bef_leaf: jnp.ndarray  # Leaf temperature for previous timestep [K]
    tleaf_hist_leaf: jnp.ndarray  # Leaf temperature (not sun/shade average) for history files [K]
    swleaf_leaf: jnp.ndarray  # Leaf absorbed solar radiation [W/m2 leaf] [n_patches, nlevmlcan, nleaf, numrad]
    lwleaf_leaf: jnp.ndarray  # Leaf absorbed longwave radiation [W/m2 leaf]
    rnleaf_leaf: jnp.ndarray  # Leaf net radiation [W/m2 leaf]
    stleaf_leaf: jnp.ndarray  # Leaf storage heat flux [W/m2 leaf]
    shleaf_leaf: jnp.ndarray  # Leaf sensible heat flux [W/m2 leaf]
    lhleaf_leaf: jnp.ndarray  # Leaf latent heat flux [W/m2 leaf]
    trleaf_leaf: jnp.ndarray  # Leaf transpiration flux [mol H2O/m2 leaf/s]
    evleaf_leaf: jnp.ndarray  # Leaf evaporation flux [mol H2O/m2 leaf/s]
    
    gbh_leaf: jnp.ndarray  # Leaf boundary layer conductance: heat [mol/m2 leaf/s]
    gbv_leaf: jnp.ndarray  # Leaf boundary layer conductance: H2O [mol H2O/m2 leaf/s]
    gbc_leaf: jnp.ndarray  # Leaf boundary layer conductance: CO2 [mol CO2/m2 leaf/s]
    
    vcmax25_leaf: jnp.ndarray  # Leaf maximum carboxylation rate at 25C [umol/m2/s]
    jmax25_leaf: jnp.ndarray  # Leaf C3 maximum electron transport rate at 25C [umol/m2/s]
    kp25_leaf: jnp.ndarray  # Leaf C4 initial slope of CO2 response curve at 25C [mol/m2/s]
    rd25_leaf: jnp.ndarray  # Leaf respiration rate at 25C [umol/m2/s]
    
    kc_leaf: jnp.ndarray  # Leaf Michaelis-Menten constant for CO2 [umol/mol]
    ko_leaf: jnp.ndarray  # Leaf Michaelis-Menten constant for O2 [mmol/mol]
    cp_leaf: jnp.ndarray  # Leaf CO2 compensation point [umol/mol]
    vcmax_leaf: jnp.ndarray  # Leaf maximum carboxylation rate [umol/m2/s]
    jmax_leaf: jnp.ndarray  # Leaf maximum electron transport rate [umol/m2/s]
    kp_leaf: jnp.ndarray  # Leaf C4 initial slope of CO2 response curve at 25C [mol/m2/s]
    ceair_leaf: jnp.ndarray  # Leaf vapor pressure of air, constrained for stomatal conductance [Pa]
    leaf_esat_leaf: jnp.ndarray  # Leaf saturation vapor pressure [Pa]
    
    apar_leaf: jnp.ndarray  # Leaf absorbed PAR [umol photon/m2 leaf/s]
    je_leaf: jnp.ndarray  # Leaf electron transport rate [umol/m2/s]
    ac_leaf: jnp.ndarray  # Leaf rubisco-limited gross photosynthesis [umol CO2/m2 leaf/s]
    aj_leaf: jnp.ndarray  # Leaf RuBP regeneration-limited gross photosynthesis [umol CO2/m2 leaf/s]
    ap_leaf: jnp.ndarray  # Leaf product-limited (C3) or CO2-limited (C4) gross photosynthesis [umol CO2/m2 leaf/s]
    agross_leaf: jnp.ndarray  # Leaf gross photosynthesis [umol CO2/m2 leaf/s]
    anet_leaf: jnp.ndarray  # Leaf net photosynthesis [umol CO2/m2 leaf/s]
    rd_leaf: jnp.ndarray  # Leaf respiration rate [umol CO2/m2 leaf/s]
    ci_leaf: jnp.ndarray  # Leaf intercellular CO2 [umol/mol]
    cs_leaf: jnp.ndarray  # Leaf surface CO2 [umol/mol]
    
    lwp_leaf: jnp.ndarray  # Leaf water potential [MPa]
    lwp_hist_leaf: jnp.ndarray  # Leaf water potential (not sun/shade average) for history files [MPa]
    hs_leaf: jnp.ndarray  # Leaf fractional humidity at leaf surface [-]
    vpd_leaf: jnp.ndarray  # Leaf vapor pressure deficit [Pa]
    gs_leaf: jnp.ndarray  # Leaf stomatal conductance [mol H2O/m2 leaf/s]
    gspot_leaf: jnp.ndarray  # Leaf stomatal conductance without water stress [mol H2O/m2 leaf/s]
    alphapsn_leaf: jnp.ndarray  # Leaf 13C fractionation factor for photosynthesis [-]


# =============================================================================
# Initialization Functions
# =============================================================================

def create_empty_mlcanopy_state(
    n_patches: int,
    nlevmlcan: int,
    nleaf: int = NLEAF,
    numrad: int = 2,
    nlevgrnd: int = 15,
) -> MLCanopyState:
    """
    Create an empty MLCanopyState with all arrays initialized to zeros.
    
    Args:
        n_patches: Number of patches
        nlevmlcan: Number of canopy layers
        nleaf: Number of leaf types (2: sunlit, shaded)
        numrad: Number of radiation wavebands (2: visible, near-IR)
        nlevgrnd: Number of ground layers
        
    Returns:
        MLCanopyState with zero-initialized arrays
        
    Note:
        This is a helper function for initialization. In practice, arrays
        should be populated with actual values from input data or restart files.
        
    Reference:
        MLCanopyFluxesType.F90:1-317
    """
    return MLCanopyState(
        # Vegetation input variables
        ztop_canopy=jnp.zeros(n_patches),
        zbot_canopy=jnp.zeros(n_patches),
        lai_canopy=jnp.zeros(n_patches),
        sai_canopy=jnp.zeros(n_patches),
        root_biomass_canopy=jnp.zeros(n_patches),
        
        # Atmospheric forcing variables
        zref_forcing=jnp.zeros(n_patches),
        tref_forcing=jnp.zeros(n_patches),
        qref_forcing=jnp.zeros(n_patches),
        uref_forcing=jnp.zeros(n_patches),
        pref_forcing=jnp.zeros(n_patches),
        co2ref_forcing=jnp.zeros(n_patches),
        o2ref_forcing=jnp.zeros(n_patches),
        swskyb_forcing=jnp.zeros((n_patches, numrad)),
        swskyd_forcing=jnp.zeros((n_patches, numrad)),
        lwsky_forcing=jnp.zeros(n_patches),
        qflx_rain_forcing=jnp.zeros(n_patches),
        qflx_snow_forcing=jnp.zeros(n_patches),
        tacclim_forcing=jnp.zeros(n_patches),
        
        # Derived atmospheric forcing variables
        eref_forcing=jnp.zeros(n_patches),
        thref_forcing=jnp.zeros(n_patches),
        thvref_forcing=jnp.zeros(n_patches),
        rhoair_forcing=jnp.zeros(n_patches),
        rhomol_forcing=jnp.zeros(n_patches),
        mmair_forcing=jnp.zeros(n_patches),
        cpair_forcing=jnp.zeros(n_patches),
        solar_zen_forcing=jnp.zeros(n_patches),
        
        # Canopy flux variables
        swveg_canopy=jnp.zeros((n_patches, numrad)),
        swvegsun_canopy=jnp.zeros((n_patches, numrad)),
        swvegsha_canopy=jnp.zeros((n_patches, numrad)),
        lwveg_canopy=jnp.zeros(n_patches),
        lwvegsun_canopy=jnp.zeros(n_patches),
        lwvegsha_canopy=jnp.zeros(n_patches),
        shveg_canopy=jnp.zeros(n_patches),
        shvegsun_canopy=jnp.zeros(n_patches),
        shvegsha_canopy=jnp.zeros(n_patches),
        lhveg_canopy=jnp.zeros(n_patches),
        lhvegsun_canopy=jnp.zeros(n_patches),
        lhvegsha_canopy=jnp.zeros(n_patches),
        etveg_canopy=jnp.zeros(n_patches),
        etvegsun_canopy=jnp.zeros(n_patches),
        etvegsha_canopy=jnp.zeros(n_patches),
        gppveg_canopy=jnp.zeros(n_patches),
        gppvegsun_canopy=jnp.zeros(n_patches),
        gppvegsha_canopy=jnp.zeros(n_patches),
        vcmax25veg_canopy=jnp.zeros(n_patches),
        vcmax25sun_canopy=jnp.zeros(n_patches),
        vcmax25sha_canopy=jnp.zeros(n_patches),
        gsveg_canopy=jnp.zeros(n_patches),
        gsvegsun_canopy=jnp.zeros(n_patches),
        gsvegsha_canopy=jnp.zeros(n_patches),
        windveg_canopy=jnp.zeros(n_patches),
        windvegsun_canopy=jnp.zeros(n_patches),
        windvegsha_canopy=jnp.zeros(n_patches),
        tlveg_canopy=jnp.zeros(n_patches),
        tlvegsun_canopy=jnp.zeros(n_patches),
        tlvegsha_canopy=jnp.zeros(n_patches),
        taveg_canopy=jnp.zeros(n_patches),
        tavegsun_canopy=jnp.zeros(n_patches),
        tavegsha_canopy=jnp.zeros(n_patches),
        laisun_canopy=jnp.zeros(n_patches),
        laisha_canopy=jnp.zeros(n_patches),
        albcan_canopy=jnp.zeros((n_patches, numrad)),
        lwup_canopy=jnp.zeros(n_patches),
        rnet_canopy=jnp.zeros(n_patches),
        shflx_canopy=jnp.zeros(n_patches),
        lhflx_canopy=jnp.zeros(n_patches),
        etflx_canopy=jnp.zeros(n_patches),
        stflx_canopy=jnp.zeros(n_patches),
        ustar_canopy=jnp.zeros(n_patches),
        gac_to_hc_canopy=jnp.zeros(n_patches),
        qflx_intr_canopy=jnp.zeros(n_patches),
        qflx_tflrain_canopy=jnp.zeros(n_patches),
        qflx_tflsnow_canopy=jnp.zeros(n_patches),
        
        # Canopy diagnostic variables
        uaf_canopy=jnp.zeros(n_patches),
        taf_canopy=jnp.zeros(n_patches),
        qaf_canopy=jnp.zeros(n_patches),
        fracminlwp_canopy=jnp.zeros(n_patches),
        
        # Canopy aerodynamic variables
        obu_canopy=jnp.zeros(n_patches),
        obuold_canopy=jnp.zeros(n_patches),
        nmozsgn_canopy=jnp.zeros(n_patches, dtype=jnp.int32),
        beta_canopy=jnp.zeros(n_patches),
        PrSc_canopy=jnp.zeros(n_patches),
        Lc_canopy=jnp.zeros(n_patches),
        zdisp_canopy=jnp.zeros(n_patches),
        
        # Canopy stomatal conductance variables
        g0_canopy=jnp.zeros(n_patches),
        g1_canopy=jnp.zeros(n_patches),
        
        # Soil energy balance variables
        albsoib_soil=jnp.zeros((n_patches, numrad)),
        albsoid_soil=jnp.zeros((n_patches, numrad)),
        swsoi_soil=jnp.zeros((n_patches, numrad)),
        lwsoi_soil=jnp.zeros(n_patches),
        rnsoi_soil=jnp.zeros(n_patches),
        shsoi_soil=jnp.zeros(n_patches),
        lhsoi_soil=jnp.zeros(n_patches),
        etsoi_soil=jnp.zeros(n_patches),
        gsoi_soil=jnp.zeros(n_patches),
        tg_soil=jnp.zeros(n_patches),
        tg_bef_soil=jnp.zeros(n_patches),
        eg_soil=jnp.zeros(n_patches),
        rhg_soil=jnp.zeros(n_patches),
        gac0_soil=jnp.zeros(n_patches),
        soil_t_soil=jnp.zeros(n_patches),
        soil_dz_soil=jnp.zeros(n_patches),
        soil_tk_soil=jnp.zeros(n_patches),
        soilres_soil=jnp.zeros(n_patches),
        
        # Soil moisture variables
        btran_soil=jnp.zeros(n_patches),
        psis_soil=jnp.zeros(n_patches),
        rsoil_soil=jnp.zeros(n_patches),
        soil_et_loss_soil=jnp.zeros((n_patches, nlevgrnd)),
        
        # Canopy layer indices
        ncan_canopy=jnp.zeros(n_patches, dtype=jnp.int32),
        ntop_canopy=jnp.zeros(n_patches, dtype=jnp.int32),
        nbot_canopy=jnp.zeros(n_patches, dtype=jnp.int32),
        
        # Canopy layer variables
        dlai_frac_profile=jnp.zeros((n_patches, nlevmlcan)),
        dsai_frac_profile=jnp.zeros((n_patches, nlevmlcan)),
        dlai_profile=jnp.zeros((n_patches, nlevmlcan)),
        dsai_profile=jnp.zeros((n_patches, nlevmlcan)),
        dpai_profile=jnp.zeros((n_patches, nlevmlcan)),
        zs_profile=jnp.zeros((n_patches, nlevmlcan)),
        dz_profile=jnp.zeros((n_patches, nlevmlcan)),
        vcmax25_profile=jnp.zeros((n_patches, nlevmlcan)),
        jmax25_profile=jnp.zeros((n_patches, nlevmlcan)),
        kp25_profile=jnp.zeros((n_patches, nlevmlcan)),
        rd25_profile=jnp.zeros((n_patches, nlevmlcan)),
        cpleaf_profile=jnp.zeros((n_patches, nlevmlcan)),
        fracsun_profile=jnp.zeros((n_patches, nlevmlcan)),
        kb_profile=jnp.zeros((n_patches, nlevmlcan)),
        tb_profile=jnp.zeros((n_patches, nlevmlcan)),
        td_profile=jnp.zeros((n_patches, nlevmlcan)),
        tbi_profile=jnp.zeros((n_patches, nlevmlcan)),
        
        # Canopy layer source/sink fluxes
        swsrc_profile=jnp.zeros((n_patches, nlevmlcan, numrad)),
        lwsrc_profile=jnp.zeros((n_patches, nlevmlcan)),
        rnsrc_profile=jnp.zeros((n_patches, nlevmlcan)),
        stsrc_profile=jnp.zeros((n_patches, nlevmlcan)),
        shsrc_profile=jnp.zeros((n_patches, nlevmlcan)),
        lhsrc_profile=jnp.zeros((n_patches, nlevmlcan)),
        etsrc_profile=jnp.zeros((n_patches, nlevmlcan)),
        fco2src_profile=jnp.zeros((n_patches, nlevmlcan)),
        
        # Canopy layer scalar profiles
        wind_profile=jnp.zeros((n_patches, nlevmlcan)),
        tair_profile=jnp.zeros((n_patches, nlevmlcan)),
        eair_profile=jnp.zeros((n_patches, nlevmlcan)),
        cair_profile=jnp.zeros((n_patches, nlevmlcan)),
        tair_bef_profile=jnp.zeros((n_patches, nlevmlcan)),
        eair_bef_profile=jnp.zeros((n_patches, nlevmlcan)),
        cair_bef_profile=jnp.zeros((n_patches, nlevmlcan)),
        wind_data_profile=jnp.zeros((n_patches, nlevmlcan)),
        tair_data_profile=jnp.zeros((n_patches, nlevmlcan)),
        eair_data_profile=jnp.zeros((n_patches, nlevmlcan)),
        
        # Canopy layer air fluxes
        shair_profile=jnp.zeros((n_patches, nlevmlcan)),
        etair_profile=jnp.zeros((n_patches, nlevmlcan)),
        stair_profile=jnp.zeros((n_patches, nlevmlcan)),
        gac_profile=jnp.zeros((n_patches, nlevmlcan)),
        
        # Canopy layer weighted mean leaf variables
        swleaf_mean_profile=jnp.zeros((n_patches, nlevmlcan, numrad)),
        lwleaf_mean_profile=jnp.zeros((n_patches, nlevmlcan)),
        rnleaf_mean_profile=jnp.zeros((n_patches, nlevmlcan)),
        stleaf_mean_profile=jnp.zeros((n_patches, nlevmlcan)),
        shleaf_mean_profile=jnp.zeros((n_patches, nlevmlcan)),
        lhleaf_mean_profile=jnp.zeros((n_patches, nlevmlcan)),
        etleaf_mean_profile=jnp.zeros((n_patches, nlevmlcan)),
        fco2_mean_profile=jnp.zeros((n_patches, nlevmlcan)),
        apar_mean_profile=jnp.zeros((n_patches, nlevmlcan)),
        gs_mean_profile=jnp.zeros((n_patches, nlevmlcan)),
        tleaf_mean_profile=jnp.zeros((n_patches, nlevmlcan)),
        lwp_mean_profile=jnp.zeros((n_patches, nlevmlcan)),
        
        # Canopy layer water variables
        lsc_profile=jnp.zeros((n_patches, nlevmlcan)),
        h2ocan_profile=jnp.zeros((n_patches, nlevmlcan)),
        fwet_profile=jnp.zeros((n_patches, nlevmlcan)),
        fdry_profile=jnp.zeros((n_patches, nlevmlcan)),
        
        # Sunlit/shaded leaf variables
        tleaf_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        tleaf_bef_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        tleaf_hist_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        swleaf_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf, numrad)),
        lwleaf_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        rnleaf_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        stleaf_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        shleaf_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        lhleaf_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        trleaf_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        evleaf_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        gbh_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        gbv_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        gbc_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        vcmax25_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        jmax25_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        kp25_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        rd25_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        kc_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        ko_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        cp_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        vcmax_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        jmax_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        kp_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        ceair_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        leaf_esat_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        apar_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        je_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        ac_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        aj_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        ap_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        agross_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        anet_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        rd_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        ci_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        cs_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        lwp_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        lwp_hist_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        hs_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        vpd_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        gs_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        gspot_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        alphapsn_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
    )


def init_allocate(
    bounds: BoundsType,
    nlevmlcan: int,
    numrad: int = 2,
    nlevgrnd: int = 15,
    nleaf: int = NLEAF,
) -> MLCanopyState:
    """
    Allocate and initialize multilayer canopy arrays.
    
    Creates all arrays needed for the multilayer canopy model and initializes
    them to special values (SPVAL for floats, ISPVAL for integers) to mark
    uninitialized data.
    
    Args:
        bounds: Domain bounds containing patch indices
        nlevmlcan: Number of canopy layers
        numrad: Number of radiation bands (typically 2: visible, near-IR)
        nlevgrnd: Number of ground layers
        nleaf: Number of leaf types (2: sunlit, shaded)
        
    Returns:
        MLCanopyState with all arrays allocated and initialized to special values
        
    Reference:
        MLCanopyFluxesType.F90:335-608
    """
    n_patches = bounds.endp - bounds.begp + 1
    
    # Create state with SPVAL initialization
    state = create_empty_mlcanopy_state(n_patches, nlevmlcan, nleaf, numrad, nlevgrnd)
    
    # Replace zeros with SPVAL for float arrays, ISPVAL for integer arrays
    # This marks them as uninitialized
    return state._replace(
        # Vegetation input variables
        ztop_canopy=jnp.full(n_patches, SPVAL),
        zbot_canopy=jnp.full(n_patches, SPVAL),
        lai_canopy=jnp.full(n_patches, SPVAL),
        sai_canopy=jnp.full(n_patches, SPVAL),
        root_biomass_canopy=jnp.full(n_patches, SPVAL),
        
        # Integer arrays use ISPVAL
        ncan_canopy=jnp.full(n_patches, ISPVAL, dtype=jnp.int32),
        ntop_canopy=jnp.full(n_patches, ISPVAL, dtype=jnp.int32),
        nbot_canopy=jnp.full(n_patches, ISPVAL, dtype=jnp.int32),
        nmozsgn_canopy=jnp.full(n_patches, ISPVAL, dtype=jnp.int32),
    )


def init_cold(
    bounds: BoundsType,
    nlevmlcan: int,
    nleaf: int = NLEAF,
) -> MLCanopyState:
    """
    Initialize multilayer canopy variables for cold start.
    
    Sets initial values for leaf water potential and canopy intercepted water.
    These values provide a reasonable starting point but are typically
    overwritten by the initVerticalProfiles routine.
    
    Args:
        bounds: Patch bounds structure containing begp and endp
        nlevmlcan: Number of canopy layers
        nleaf: Number of leaf types (2: sunlit, shaded)
        
    Returns:
        MLCanopyState with initialized arrays:
            - lwp_leaf: [n_patches, nlevmlcan, nleaf] initialized to -0.1 MPa
            - h2ocan_profile: [n_patches, nlevmlcan] initialized to 0.0 kg/m2
            
    Note:
        The constant -0.1 represents a leaf water potential of -0.1 MPa,
        which is a typical value for unstressed vegetation.
        
    Reference:
        MLCanopyFluxesType.F90:642-674
    """
    n_patches = bounds.endp - bounds.begp + 1
    
    # Create base state
    state = create_empty_mlcanopy_state(n_patches, nlevmlcan, nleaf)
    
    # Initialize leaf water potential to -0.1 MPa for both sun and shade
    lwp_leaf = jnp.full((n_patches, nlevmlcan, nleaf), DEFAULT_LWP)
    
    # Initialize intercepted water to zero
    h2ocan_profile = jnp.full((n_patches, nlevmlcan), DEFAULT_H2OCAN)
    
    return state._replace(
        lwp_leaf=lwp_leaf,
        h2ocan_profile=h2ocan_profile,
    )


def init(
    bounds: BoundsType,
    nlevmlcan: int,
    numrad: int = 2,
    nlevgrnd: int = 15,
    nleaf: int = NLEAF,
) -> MLCanopyState:
    """
    Initialize multilayer canopy data structure.
    
    Performs three initialization steps:
    1. Allocate arrays for canopy variables
    2. Setup variables for history output (metadata only)
    3. Initialize values for cold-start
    
    This is a pure functional version of the Fortran Init subroutine.
    
    Args:
        bounds: Domain bounds containing grid dimensions
        nlevmlcan: Number of canopy layers
        numrad: Number of radiation bands
        nlevgrnd: Number of ground layers
        nleaf: Number of leaf types
        
    Returns:
        Initialized multilayer canopy state
        
    Reference:
        MLCanopyFluxesType.F90:318-332
    """
    # Allocate arrays
    state = init_allocate(bounds, nlevmlcan, numrad, nlevgrnd, nleaf)
    
    # Initialize for cold start
    state = init_cold(bounds, nlevmlcan, nleaf)
    
    # History setup is handled externally in JAX (metadata registration)
    
    return state


# =============================================================================
# Restart I/O Functions
# =============================================================================

class MLCanopyRestartData(NamedTuple):
    """Container for multilayer canopy restart variables.
    
    Attributes:
        taf_canopy: Air temperature at canopy top [K] [n_patches]
        lwp_mean_profile: Leaf water potential by layer [MPa] [n_patches, nlevmlcan]
    """
    taf_canopy: jnp.ndarray
    lwp_mean_profile: jnp.ndarray


def extract_restart_data(mlcanopy_state: MLCanopyState) -> MLCanopyRestartData:
    """
    Extract restart variables from multilayer canopy state.
    
    This function extracts the subset of state variables that need to be
    saved to restart files.
    
    Args:
        mlcanopy_state: Complete multilayer canopy state structure
        
    Returns:
        MLCanopyRestartData containing variables for restart file
        
    Reference:
        MLCanopyFluxesType.F90:677-709
    """
    return MLCanopyRestartData(
        taf_canopy=mlcanopy_state.taf_canopy,
        lwp_mean_profile=mlcanopy_state.lwp_mean_profile,
    )


def restore_from_restart(
    mlcanopy_state: MLCanopyState,
    restart_data: MLCanopyRestartData,
) -> MLCanopyState:
    """
    Restore multilayer canopy state from restart data.
    
    This function updates the multilayer canopy state with values read from
    a restart file.
    
    Args:
        mlcanopy_state: Current multilayer canopy state structure
        restart_data: Restart variables read from file
        
    Returns:
        Updated multilayer canopy state with restored variables
        
    Reference:
        MLCanopyFluxesType.F90:677-709
    """
    return mlcanopy_state._replace(
        taf_canopy=restart_data.taf_canopy,
        lwp_mean_profile=restart_data.lwp_mean_profile,
    )


def get_restart_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for restart variables.
    
    Returns metadata describing the restart variables, including dimensions,
    units, and interpolation flags. This information is used by the checkpoint
    system to properly handle the variables.
    
    Returns:
        Dictionary mapping variable names to their metadata
        
    Reference:
        MLCanopyFluxesType.F90:677-709
    """
    return {
        'taf_canopy': {
            'long_name': 'air temperature at canopy top',
            'units': 'K',
            'dimensions': ['pft'],
            'interpinic_flag': 'interp',
        },
        'lwp_mean_profile': {
            'long_name': 'leaf water potential of canopy layer',
            'units': 'MPa',
            'dimensions': ['pft', 'nlevmlcan'],
            'interpinic_flag': 'interp',
            'switchdim': True,
        },
    }


def validate_restart_data(
    restart_data: MLCanopyRestartData,
    n_patches: int,
    nlevmlcan: int,
) -> bool:
    """
    Validate restart data dimensions and values.
    
    Checks that restart data has correct shapes and physically reasonable values.
    
    Args:
        restart_data: Restart data to validate
        n_patches: Expected number of patches
        nlevmlcan: Expected number of canopy layers
        
    Returns:
        True if validation passes, False otherwise
        
    Reference:
        MLCanopyFluxesType.F90:677-709
    """
    # Check shapes
    if restart_data.taf_canopy.shape != (n_patches,):
        return False
    if restart_data.lwp_mean_profile.shape != (n_patches, nlevmlcan):
        return False
    
    # Check for NaN or Inf
    if not jnp.all(jnp.isfinite(restart_data.taf_canopy)):
        return False
    if not jnp.all(jnp.isfinite(restart_data.lwp_mean_profile)):
        return False
    
    # Check physically reasonable ranges
    # Temperature should be between 150K and 350K
    if not jnp.all((restart_data.taf_canopy >= 150.0) & 
                   (restart_data.taf_canopy <= 350.0)):
        return False
    
    # Leaf water potential should be negative (typically -10 to 0 MPa)
    if not jnp.all((restart_data.lwp_mean_profile >= -10.0) & 
                   (restart_data.lwp_mean_profile <= 0.0)):
        return False
    
    return True


# =============================================================================
# History Output Functions
# =============================================================================

def get_history_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for history output fields.
    
    Returns metadata describing the history variables, including dimensions,
    units, and averaging flags.
    
    Returns:
        Dictionary mapping variable names to their metadata
        
    Reference:
        MLCanopyFluxesType.F90:611-639
    """
    return {
        'gppveg_canopy': {
            'long_name': 'Gross primary production',
            'units': 'umol/m2s',
            'dimensions': ['pft'],
            'avg_flag': 'A',
            'set_lake': SPVAL,
            'set_urb': SPVAL,
        },
        'lwp_mean_profile': {
            'long_name': 'Weighted mean leaf water potential of canopy layer',
            'units': 'MPa',
            'dimensions': ['pft', 'nlevmlcan'],
            'avg_flag': 'A',
            'set_lake': SPVAL,
            'set_urb': SPVAL,
        },
    }# Backward compatibility alias
mlcanopy_type = MLCanopyState
