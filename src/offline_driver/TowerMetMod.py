"""
Tower Meteorology Module.

Translated from CTSM's TowerMetMod.F90

This module provides functionality for reading and processing tower meteorology
forcing data for site-level simulations. Tower met data typically comes from eddy
covariance flux tower sites and provides high-frequency atmospheric forcing.

Key functionality:
    - Read tower meteorological forcing from NetCDF files
    - Process atmospheric forcing variables for land surface model
    - Handle time interpolation and unit conversions
    - Partition solar radiation into direct/diffuse and visible/NIR bands
    - Convert relative humidity to specific humidity
    - Calculate longwave radiation when missing

Key physics equations:

Solar radiation partitioning (lines 115-133):
    - Total radiation split 50/50 between visible and NIR bands
    - Direct beam fraction from empirical polynomials:
        r_vis = a0 + fsds_vis*(a1 + fsds_vis*(a2 + fsds_vis*a3))
        r_nir = b0 + fsds_nir*(b1 + fsds_nir*(b2 + fsds_nir*b3))
    - Direct beam fraction clamped to [0.01, 0.99]

Humidity conversion (lines 173-180):
    - Specific humidity from RH:
        e = (RH/100) * esat
        q = (mmh2o/mmdry) * e / (P - (1 - mmh2o/mmdry) * e)
    - Vapor pressure from q:
        eair = q * P / (mmh2o/mmdry + (1 - mmh2o/mmdry) * q)

Longwave radiation (lines 186-189):
    - When missing, calculate from emissivity:
        emiss = 0.7 + 5.95e-5 * 0.01 * eair * exp(1500/T)
        LW = emiss * sigma * T^4

Gas partial pressures (lines 194-195):
    - CO2: 367 ppm -> Pa: P_CO2 = 367e-6 * P_atm
    - O2: 0.209 mol/mol -> Pa: P_O2 = 0.209 * P_atm

Reference: TowerMetMod.F90, lines 1-325
"""

from typing import NamedTuple, Protocol, Tuple
import jax.numpy as jnp


# =============================================================================
# Type Definitions
# =============================================================================

class TowerMetState(NamedTuple):
    """State container for tower meteorology data.
    
    Attributes:
        forc_t: Air temperature [K] [n_patches]
        forc_q: Specific humidity [kg/kg] [n_patches]
        forc_pbot: Atmospheric pressure [Pa] [n_patches]
        forc_u: Wind speed (u component) [m/s] [n_patches]
        forc_v: Wind speed (v component) [m/s] [n_patches]
        forc_lwrad: Downward longwave radiation [W/m2] [n_patches]
        forc_rain: Rain rate [mm/s] [n_patches]
        forc_snow: Snow rate [mm/s] [n_patches]
        forc_solad: Direct beam solar radiation [W/m2] [n_patches, n_bands]
        forc_solai: Diffuse solar radiation [W/m2] [n_patches, n_bands]
        forc_hgt_u: Observational height of wind [m] [n_patches]
        forc_hgt_t: Observational height of temperature [m] [n_patches]
        forc_hgt_q: Observational height of humidity [m] [n_patches]
        forc_pco2: Partial pressure of CO2 [Pa] [n_patches]
        forc_po2: Partial pressure of O2 [Pa] [n_patches]
    """
    forc_t: jnp.ndarray
    forc_q: jnp.ndarray
    forc_pbot: jnp.ndarray
    forc_u: jnp.ndarray
    forc_v: jnp.ndarray
    forc_lwrad: jnp.ndarray
    forc_rain: jnp.ndarray
    forc_snow: jnp.ndarray
    forc_solad: jnp.ndarray
    forc_solai: jnp.ndarray
    forc_hgt_u: jnp.ndarray
    forc_hgt_t: jnp.ndarray
    forc_hgt_q: jnp.ndarray
    forc_pco2: jnp.ndarray
    forc_po2: jnp.ndarray


class TowerMetRawData(NamedTuple):
    """Raw tower meteorology data from NetCDF file.
    
    Attributes:
        zbot: Reference height [m]
        tbot: Air temperature at reference height [K]
        rhbot: Relative humidity at reference height [%]
        qbot: Specific humidity at reference height [kg/kg]
        ubot: Wind speed at reference height [m/s]
        fsdsbot: Solar radiation [W/m2]
        fldsbot: Longwave radiation [W/m2]
        pbot: Air pressure at reference height [Pa]
        prect: Precipitation [mm/s]
    """
    zbot: float
    tbot: float
    rhbot: float
    qbot: float
    ubot: float
    fsdsbot: float
    fldsbot: float
    pbot: float
    prect: float


class SolarRadiationParams(NamedTuple):
    """Parameters for solar radiation partitioning.
    
    Polynomial coefficients for direct beam fraction calculation.
    From lines 88-140 of TowerMetMod.F90.
    
    Attributes:
        a0, a1, a2, a3: Visible band coefficients
        b0, b1, b2, b3: Near-infrared band coefficients
    """
    a0: float
    a1: float
    a2: float
    a3: float
    b0: float
    b1: float
    b2: float
    b3: float


class TowerMetParams(NamedTuple):
    """Parameters for tower meteorology processing.
    
    Attributes:
        mmh2o: Molecular weight of water [kg/kmol]
        mmdry: Molecular weight of dry air [kg/kmol]
        sb: Stefan-Boltzmann constant [W/m2/K4]
        missing_value: Value indicating missing data
        default_pressure: Default atmospheric pressure [Pa]
        default_height: Default observation height [m]
        co2_concentration: CO2 concentration [ppm]
        o2_concentration: O2 concentration [mol/mol]
        solar_params: Solar radiation partitioning parameters
    """
    mmh2o: float = 18.016  # kg/kmol
    mmdry: float = 28.966  # kg/kmol
    sb: float = 5.67e-8  # W/m2/K4
    missing_value: float = -999.0
    default_pressure: float = 101325.0  # Pa
    default_height: float = 30.0  # m
    co2_concentration: float = 367.0  # ppm
    o2_concentration: float = 0.209  # mol/mol
    solar_params: SolarRadiationParams = SolarRadiationParams(
        # Default polynomial coefficients for direct beam fraction
        # These would typically come from empirical fits
        a0=0.17639, a1=0.00380, a2=-9.0039e-6, a3=8.1351e-9,
        b0=0.29548, b1=0.00504, b2=-1.4957e-5, b3=1.4881e-8,
    )


# =============================================================================
# Solar Radiation Processing
# =============================================================================

def partition_solar_radiation(
    fsds: jnp.ndarray,
    params: SolarRadiationParams,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Partition total solar radiation into direct/diffuse and visible/NIR.
    
    From TowerMetMod.F90 lines 115-133.
    
    The total solar radiation is:
    1. Split equally (50/50) between visible and near-infrared wavebands
    2. Each band is split into direct beam and diffuse using empirical polynomials
    3. Direct beam fraction is clamped to [0.01, 0.99]
    
    Args:
        fsds: Total downward shortwave radiation [W/m2] [n_gridcells]
        params: Solar radiation partitioning parameters
        
    Returns:
        Tuple of (forc_solad_vis, forc_solai_vis, forc_solad_nir, forc_solai_nir)
        - forc_solad_vis: Direct beam visible [W/m2] [n_gridcells]
        - forc_solai_vis: Diffuse visible [W/m2] [n_gridcells]
        - forc_solad_nir: Direct beam NIR [W/m2] [n_gridcells]
        - forc_solai_nir: Diffuse NIR [W/m2] [n_gridcells]
    """
    # Line 115: Ensure non-negative radiation
    fsds = jnp.maximum(fsds, 0.0)
    
    # Lines 117-120: Visible band (50% of total)
    fsds_vis = 0.5 * fsds
    # Polynomial fit for direct beam fraction
    rvis = params.a0 + fsds_vis * (
        params.a1 + fsds_vis * (params.a2 + fsds_vis * params.a3)
    )
    # Clamp to [0.01, 0.99]
    rvis = jnp.clip(rvis, 0.01, 0.99)
    
    # Lines 122-125: Near-infrared band (50% of total)
    fsds_nir = 0.5 * fsds
    # Polynomial fit for direct beam fraction
    rnir = params.b0 + fsds_nir * (
        params.b1 + fsds_nir * (params.b2 + fsds_nir * params.b3)
    )
    # Clamp to [0.01, 0.99]
    rnir = jnp.clip(rnir, 0.01, 0.99)
    
    # Lines 127-130: Split into direct beam and diffuse
    forc_solad_vis = fsds_vis * rvis
    forc_solai_vis = fsds_vis * (1.0 - rvis)
    forc_solad_nir = fsds_nir * rnir
    forc_solai_nir = fsds_nir * (1.0 - rnir)
    
    return forc_solad_vis, forc_solai_vis, forc_solad_nir, forc_solai_nir


# =============================================================================
# Humidity and Vapor Pressure Calculations
# =============================================================================

def sat_vap(temperature: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate saturation vapor pressure.
    
    This is a placeholder for the actual SatVap function from MLWaterVaporMod.
    The real implementation would use the Clausius-Clapeyron equation.
    
    Args:
        temperature: Air temperature [K] [n_points]
        
    Returns:
        Tuple of (esat, desat_dT):
        - esat: Saturation vapor pressure [Pa] [n_points]
        - desat_dT: Temperature derivative [Pa/K] [n_points]
    """
    # Simplified saturation vapor pressure (Tetens formula)
    # esat = 611.2 * exp(17.67 * (T - 273.15) / (T - 29.65))
    t_celsius = temperature - 273.15
    esat = 611.2 * jnp.exp(17.67 * t_celsius / (temperature - 29.65))
    
    # Derivative (approximate)
    desat_dT = esat * 17.67 * 243.5 / ((temperature - 29.65) ** 2)
    
    return esat, desat_dT


def rh_to_specific_humidity(
    rh: jnp.ndarray,
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    mmh2o: float,
    mmdry: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert relative humidity to specific humidity.
    
    From TowerMetMod.F90 lines 173-180.
    
    Args:
        rh: Relative humidity [%] [n_points]
        temperature: Air temperature [K] [n_points]
        pressure: Atmospheric pressure [Pa] [n_points]
        mmh2o: Molecular weight of water [kg/kmol]
        mmdry: Molecular weight of dry air [kg/kmol]
        
    Returns:
        Tuple of (q, eair):
        - q: Specific humidity [kg/kg] [n_points]
        - eair: Vapor pressure [Pa] [n_points]
    """
    # Calculate saturation vapor pressure
    esat, _ = sat_vap(temperature)
    
    # Vapor pressure from relative humidity
    eair = (rh / 100.0) * esat
    
    # Specific humidity from vapor pressure
    q = (mmh2o / mmdry) * eair / (
        pressure - (1.0 - mmh2o / mmdry) * eair
    )
    
    return q, eair


def specific_humidity_to_vapor_pressure(
    q: jnp.ndarray,
    pressure: jnp.ndarray,
    mmh2o: float,
    mmdry: float,
) -> jnp.ndarray:
    """Convert specific humidity to vapor pressure.
    
    From TowerMetMod.F90 lines 179-180.
    
    Args:
        q: Specific humidity [kg/kg] [n_points]
        pressure: Atmospheric pressure [Pa] [n_points]
        mmh2o: Molecular weight of water [kg/kmol]
        mmdry: Molecular weight of dry air [kg/kmol]
        
    Returns:
        Vapor pressure [Pa] [n_points]
    """
    eair = q * pressure / (
        mmh2o / mmdry + (1.0 - mmh2o / mmdry) * q
    )
    return eair


# =============================================================================
# Longwave Radiation Calculation
# =============================================================================

def calculate_longwave_radiation(
    temperature: jnp.ndarray,
    eair: jnp.ndarray,
    sb: float,
) -> jnp.ndarray:
    """Calculate atmospheric longwave radiation from temperature and vapor pressure.
    
    From TowerMetMod.F90 lines 186-189.
    
    Uses an empirical emissivity formula:
        emiss = 0.7 + 5.95e-5 * 0.01 * eair * exp(1500/T)
        LW = emiss * sigma * T^4
    
    Args:
        temperature: Air temperature [K] [n_points]
        eair: Vapor pressure [Pa] [n_points]
        sb: Stefan-Boltzmann constant [W/m2/K4]
        
    Returns:
        Downward longwave radiation [W/m2] [n_points]
    """
    # Empirical emissivity formula
    emiss = 0.7 + 5.95e-5 * 0.01 * eair * jnp.exp(1500.0 / temperature)
    
    # Stefan-Boltzmann law
    lwrad = emiss * sb * temperature ** 4
    
    return lwrad


# =============================================================================
# Main Tower Met Processing
# =============================================================================

def process_tower_met(
    raw_data: TowerMetRawData,
    tower_ht: float,
    tower_lat: float,
    tower_lon: float,
    n_patches: int,
    params: TowerMetParams,
) -> TowerMetState:
    """Process raw tower meteorology data into CLM forcing format.
    
    This function combines all processing steps from TowerMetMod.F90:
    1. Assign basic forcing variables (lines 143-149)
    2. Set forcing heights with defaults (lines 154-162)
    3. Set default pressure if missing (line 169)
    4. Convert RH to specific humidity or validate q (lines 172-182)
    5. Calculate longwave if missing (lines 186-189)
    6. Partition solar radiation (lines 115-133)
    7. Calculate wind components (lines 112-113)
    8. Calculate gas partial pressures (lines 194-195)
    
    Args:
        raw_data: Raw tower meteorology data from NetCDF file
        tower_ht: Tower height [m]
        tower_lat: Tower latitude [degrees]
        tower_lon: Tower longitude [degrees]
        n_patches: Number of patches
        params: Tower meteorology parameters
        
    Returns:
        TowerMetState containing all processed forcing variables
        
    Note:
        This function is JIT-compatible and uses jnp.where for conditionals.
    """
    # Extract scalar values
    forc_t = raw_data.tbot
    forc_q = raw_data.qbot
    forc_pbot = raw_data.pbot
    forc_lwrad = raw_data.fldsbot
    forc_rain = raw_data.prect
    forc_snow = 0.0  # Line 149
    forc_u = raw_data.ubot
    fsds = raw_data.fsdsbot
    
    # Forcing height (lines 154, 158, 162)
    forc_hgt = raw_data.zbot
    forc_hgt = tower_ht
    forc_hgt = jnp.where(
        jnp.round(forc_hgt) == params.missing_value,
        params.default_height,
        forc_hgt
    )
    
    # Default pressure if missing (line 169)
    forc_pbot = jnp.where(
        jnp.round(forc_pbot) == params.missing_value,
        params.default_pressure,
        forc_pbot
    )
    
    # Humidity processing (lines 172-182)
    esat, _ = sat_vap(forc_t)
    forc_rh = raw_data.rhbot
    
    rh_valid = jnp.round(forc_rh) != params.missing_value
    q_valid = jnp.round(forc_q) != params.missing_value
    
    # Convert RH to specific humidity if RH is valid
    q_from_rh, eair_from_rh = rh_to_specific_humidity(
        forc_rh, forc_t, forc_pbot, params.mmh2o, params.mmdry
    )
    
    # Use converted q if RH valid, else use original q
    forc_q = jnp.where(rh_valid, q_from_rh, forc_q)
    
    # Calculate vapor pressure from q
    eair_from_q = specific_humidity_to_vapor_pressure(
        forc_q, forc_pbot, params.mmh2o, params.mmdry
    )
    eair = jnp.where(rh_valid, eair_from_rh, eair_from_q)
    
    # Error handling: set to NaN if both RH and q are missing
    both_missing = ~rh_valid & ~q_valid
    forc_q = jnp.where(both_missing, jnp.nan, forc_q)
    eair = jnp.where(both_missing, jnp.nan, eair)
    
    # Calculate longwave if missing (lines 186-189)
    lwrad_missing = jnp.round(forc_lwrad) == params.missing_value
    lwrad_calculated = calculate_longwave_radiation(forc_t, eair, params.sb)
    forc_lwrad = jnp.where(lwrad_missing, lwrad_calculated, forc_lwrad)
    
    # Solar radiation partitioning (lines 115-133)
    solad_vis, solai_vis, solad_nir, solai_nir = partition_solar_radiation(
        fsds, params.solar_params
    )
    
    # Wind components (lines 112-113)
    forc_v = 0.0  # North component set to zero
    
    # Gas partial pressures (lines 194-195)
    forc_pco2 = (params.co2_concentration / 1.0e6) * forc_pbot
    forc_po2 = params.o2_concentration * forc_pbot
    
    # Broadcast scalars to patch arrays
    forc_t_arr = jnp.full(n_patches, forc_t)
    forc_q_arr = jnp.full(n_patches, forc_q)
    forc_pbot_arr = jnp.full(n_patches, forc_pbot)
    forc_u_arr = jnp.full(n_patches, forc_u)
    forc_v_arr = jnp.full(n_patches, forc_v)
    forc_lwrad_arr = jnp.full(n_patches, forc_lwrad)
    forc_rain_arr = jnp.full(n_patches, forc_rain)
    forc_snow_arr = jnp.full(n_patches, forc_snow)
    forc_hgt_arr = jnp.full(n_patches, forc_hgt)
    forc_pco2_arr = jnp.full(n_patches, forc_pco2)
    forc_po2_arr = jnp.full(n_patches, forc_po2)
    
    # Stack solar radiation into [n_patches, 2] arrays (vis, nir)
    solad_vis_arr = jnp.full(n_patches, solad_vis)
    solad_nir_arr = jnp.full(n_patches, solad_nir)
    solai_vis_arr = jnp.full(n_patches, solai_vis)
    solai_nir_arr = jnp.full(n_patches, solai_nir)
    
    forc_solad = jnp.stack([solad_vis_arr, solad_nir_arr], axis=1)
    forc_solai = jnp.stack([solai_vis_arr, solai_nir_arr], axis=1)
    
    return TowerMetState(
        forc_t=forc_t_arr,
        forc_q=forc_q_arr,
        forc_pbot=forc_pbot_arr,
        forc_u=forc_u_arr,
        forc_v=forc_v_arr,
        forc_lwrad=forc_lwrad_arr,
        forc_rain=forc_rain_arr,
        forc_snow=forc_snow_arr,
        forc_solad=forc_solad,
        forc_solai=forc_solai,
        forc_hgt_u=forc_hgt_arr,
        forc_hgt_t=forc_hgt_arr,
        forc_hgt_q=forc_hgt_arr,
        forc_pco2=forc_pco2_arr,
        forc_po2=forc_po2_arr,
    )


# =============================================================================
# I/O Interface (Non-JIT)
# =============================================================================

def read_tower_met_netcdf(
    ncfilename: str,
    strt: int,
) -> TowerMetRawData:
    """Read variables from tower site atmospheric forcing NetCDF files.
    
    This function reads atmospheric forcing data from a NetCDF file at a specific
    time slice. Optional variables are set to -999.0 if not present in the file.
    
    From TowerMetMod.F90 lines 211-325.
    
    Args:
        ncfilename: Path to NetCDF file containing tower meteorology data
        strt: Time slice index to retrieve (0-based)
        
    Returns:
        TowerMetRawData containing all meteorology variables
        
    Note:
        This is an I/O function that should be called outside JIT-compiled code.
        The actual implementation would use netCDF4 or xarray: