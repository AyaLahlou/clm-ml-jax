"""
Tower Meteorology Module.

Translated from CTSM's TowerMetMod.F90

This module provides functionality for reading and processing tower meteorology 
forcing data for site-level simulations. Tower met data typically comes from 
eddy covariance flux tower sites and provides high-frequency atmospheric forcing.

Key functionality:
    - Read tower meteorological forcing from NetCDF files
    - Process atmospheric forcing variables for land surface model
    - Handle time interpolation and unit conversions
    - Partition solar radiation into direct/diffuse and visible/NIR bands
    - Convert relative humidity to specific humidity
    - Calculate longwave radiation from temperature and humidity when missing

Key equations:
    Solar radiation partitioning:
        rvis = a0 + fsds_vis*(a1 + fsds_vis*(a2 + fsds_vis*a3))
        rnir = b0 + fsds_nir*(b1 + fsds_nir*(b2 + fsds_nir*b3))
        
    Specific humidity from relative humidity:
        q = (mmh2o/mmdry) * e / (P - (1 - mmh2o/mmdry) * e)
        where e = (RH/100) * esat
        
    Longwave radiation from emissivity:
        LW = emiss * sigma * T^4
        where emiss = 0.7 + 5.95e-5 * 0.01 * eair * exp(1500/T)

Reference: TowerMetMod.F90, lines 1-327
"""

from typing import NamedTuple, Protocol, Tuple
import jax
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
        forc_pco2: CO2 partial pressure [Pa] [n_patches]
        forc_po2: O2 partial pressure [Pa] [n_patches]
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
    """Raw tower meteorology data read from NetCDF file.
    
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
    
    These empirical coefficients partition total solar radiation into
    direct beam and diffuse components for visible and near-infrared bands.
    
    Attributes:
        a0, a1, a2, a3: Visible band direct beam fraction coefficients
        b0, b1, b2, b3: Near-infrared band direct beam fraction coefficients
    """
    a0: float = 0.17639  # Visible band coefficient 0
    a1: float = 0.00380  # Visible band coefficient 1
    a2: float = -9.0039e-6  # Visible band coefficient 2
    a3: float = 8.1351e-9  # Visible band coefficient 3
    b0: float = 0.29548  # Near-infrared band coefficient 0
    b1: float = 0.00504  # Near-infrared band coefficient 1
    b2: float = -1.4957e-5  # Near-infrared band coefficient 2
    b3: float = 1.4881e-8  # Near-infrared band coefficient 3


class PhysicalConstants(NamedTuple):
    """Physical constants for tower meteorology calculations.
    
    Attributes:
        mmh2o: Molecular weight of water [kg/kmol]
        mmdry: Molecular weight of dry air [kg/kmol]
        sb: Stefan-Boltzmann constant [W/m2/K4]
        co2_ppm: Default CO2 concentration [ppm]
        o2_frac: O2 mole fraction [mol/mol]
        default_pressure: Default surface pressure [Pa]
        default_height: Default forcing height [m]
        missing_value: Missing value indicator
    """
    mmh2o: float = 18.016  # kg/kmol
    mmdry: float = 28.966  # kg/kmol
    sb: float = 5.67e-8  # W/m2/K4
    co2_ppm: float = 367.0  # ppm
    o2_frac: float = 0.209  # mol/mol
    default_pressure: float = 101325.0  # Pa (sea level)
    default_height: float = 30.0  # m
    missing_value: float = -999.0


class GridcellForcing(NamedTuple):
    """Atmospheric forcing at gridcell level.
    
    Attributes:
        forc_u: Eastward wind speed [m/s] [n_gridcells]
        forc_v: Northward wind speed [m/s] [n_gridcells]
        forc_solad: Direct beam radiation [W/m2] [n_gridcells, 2] (vis, nir)
        forc_solai: Diffuse radiation [W/m2] [n_gridcells, 2] (vis, nir)
        forc_pco2: CO2 partial pressure [Pa] [n_gridcells]
        forc_po2: O2 partial pressure [Pa] [n_gridcells]
    """
    forc_u: jnp.ndarray
    forc_v: jnp.ndarray
    forc_solad: jnp.ndarray
    forc_solai: jnp.ndarray
    forc_pco2: jnp.ndarray
    forc_po2: jnp.ndarray


class ColumnForcing(NamedTuple):
    """Atmospheric forcing at column level (downscaled).
    
    Attributes:
        forc_t: Air temperature [K] [n_columns]
        forc_q: Specific humidity [kg/kg] [n_columns]
        forc_pbot: Atmospheric pressure [Pa] [n_columns]
        forc_lwrad: Longwave radiation [W/m2] [n_columns]
        forc_rain: Rainfall rate [mm/s] [n_columns]
        forc_snow: Snowfall rate [mm/s] [n_columns]
    """
    forc_t: jnp.ndarray
    forc_q: jnp.ndarray
    forc_pbot: jnp.ndarray
    forc_lwrad: jnp.ndarray
    forc_rain: jnp.ndarray
    forc_snow: jnp.ndarray


class TowerMetInputs(NamedTuple):
    """Input parameters for TowerMet processing.
    
    Attributes:
        raw_data: Raw meteorology data from file
        tower_ht: Tower height [m]
        tower_lat: Tower latitude [degrees]
        tower_lon: Tower longitude [degrees]
        n_gridcells: Number of gridcells
        n_columns: Number of columns
        n_patches: Number of patches
        col_to_gridcell: Column to gridcell mapping [n_columns]
        col_to_tower: Column to tower index mapping [n_columns]
    """
    raw_data: TowerMetRawData
    tower_ht: float
    tower_lat: float
    tower_lon: float
    n_gridcells: int
    n_columns: int
    n_patches: int
    col_to_gridcell: jnp.ndarray
    col_to_tower: jnp.ndarray


# =============================================================================
# Solar Radiation Partitioning
# =============================================================================


def partition_solar_radiation(
    fsds: float,
    params: SolarRadiationParams,
    ivis: int = 0,
    inir: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Partition total solar radiation into direct beam and diffuse components.
    
    Splits total solar radiation equally between visible and near-infrared bands,
    then calculates direct beam fraction using empirical polynomial fits.
    
    From TowerMetMod.F90 lines 115-130.
    
    Physics:
        For each band (visible, NIR):
            1. Split total radiation: fsds_band = 0.5 * fsds
            2. Calculate direct beam fraction using polynomial:
               r = c0 + fsds_band*(c1 + fsds_band*(c2 + fsds_band*c3))
            3. Clamp fraction to [0.01, 0.99] for numerical stability
            4. Partition: direct = fsds_band * r, diffuse = fsds_band * (1-r)
    
    Args:
        fsds: Total downward shortwave radiation [W/m2] [scalar]
        params: Solar radiation partitioning parameters
        ivis: Index for visible band (default 0)
        inir: Index for near-infrared band (default 1)
        
    Returns:
        forc_solad: Direct beam radiation [W/m2] [2] (vis, nir)
        forc_solai: Diffuse radiation [W/m2] [2] (vis, nir)
        
    Note:
        Direct beam fractions are clamped to [0.01, 0.99] to avoid
        numerical issues in downstream calculations.
    """
    # Ensure non-negative solar radiation (line 115)
    fsds = jnp.maximum(fsds, 0.0)
    
    # Split equally to visible and near-infrared (lines 117, 122)
    fsds_vis = 0.5 * fsds
    fsds_nir = 0.5 * fsds
    
    # Calculate visible band direct beam fraction (lines 118-119)
    rvis = (params.a0 + fsds_vis * (params.a1 + 
            fsds_vis * (params.a2 + fsds_vis * params.a3)))
    rvis = jnp.clip(rvis, 0.01, 0.99)
    
    # Calculate near-infrared band direct beam fraction (lines 123-124)
    rnir = (params.b0 + fsds_nir * (params.b1 + 
            fsds_nir * (params.b2 + fsds_nir * params.b3)))
    rnir = jnp.clip(rnir, 0.01, 0.99)
    
    # Allocate to direct beam and diffuse (lines 126-129)
    forc_solad = jnp.array([
        fsds_vis * rvis,  # visible direct beam
        fsds_nir * rnir,  # near-infrared direct beam
    ])
    
    forc_solai = jnp.array([
        fsds_vis * (1.0 - rvis),  # visible diffuse
        fsds_nir * (1.0 - rnir),  # near-infrared diffuse
    ])
    
    return forc_solad, forc_solai


# =============================================================================
# Humidity Conversions
# =============================================================================


def relative_humidity_to_specific_humidity(
    rh: float,
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    esat: jnp.ndarray,
    constants: PhysicalConstants,
) -> jnp.ndarray:
    """Convert relative humidity to specific humidity.
    
    From TowerMetMod.F90 lines 169-172.
    
    Physics:
        1. Calculate vapor pressure from RH: e = (RH/100) * esat
        2. Convert to specific humidity using molecular weights:
           q = (mmh2o/mmdry) * e / (P - (1 - mmh2o/mmdry) * e)
    
    Args:
        rh: Relative humidity [%] [scalar]
        temperature: Air temperature [K] [n_columns]
        pressure: Atmospheric pressure [Pa] [n_columns]
        esat: Saturation vapor pressure [Pa] [n_columns]
        constants: Physical constants
        
    Returns:
        Specific humidity [kg/kg] [n_columns]
    """
    # Calculate vapor pressure from relative humidity (line 169)
    eair = (rh / 100.0) * esat
    
    # Convert to specific humidity (lines 170-172)
    q = (constants.mmh2o / constants.mmdry) * eair / (
        pressure - (1.0 - constants.mmh2o / constants.mmdry) * eair
    )
    
    return q


def specific_humidity_to_vapor_pressure(
    q: jnp.ndarray,
    pressure: jnp.ndarray,
    constants: PhysicalConstants,
) -> jnp.ndarray:
    """Convert specific humidity to vapor pressure.
    
    From TowerMetMod.F90 lines 174-176.
    
    Physics:
        Invert the specific humidity equation:
        e = q * P / (mmh2o/mmdry + (1 - mmh2o/mmdry) * q)
    
    Args:
        q: Specific humidity [kg/kg] [n_columns]
        pressure: Atmospheric pressure [Pa] [n_columns]
        constants: Physical constants
        
    Returns:
        Vapor pressure [Pa] [n_columns]
    """
    eair = q * pressure / (
        constants.mmh2o / constants.mmdry + 
        (1.0 - constants.mmh2o / constants.mmdry) * q
    )
    
    return eair


# =============================================================================
# Longwave Radiation Calculation
# =============================================================================


def calculate_longwave_radiation(
    temperature: jnp.ndarray,
    vapor_pressure: jnp.ndarray,
    constants: PhysicalConstants,
) -> jnp.ndarray:
    """Calculate atmospheric longwave radiation from temperature and humidity.
    
    From TowerMetMod.F90 lines 182-184.
    
    Physics:
        1. Calculate atmospheric emissivity:
           emiss = 0.7 + 5.95e-5 * 0.01 * eair * exp(1500/T)
        2. Apply Stefan-Boltzmann law:
           LW = emiss * sigma * T^4
    
    Args:
        temperature: Air temperature [K] [n_columns]
        vapor_pressure: Vapor pressure [Pa] [n_columns]
        constants: Physical constants
        
    Returns:
        Downward longwave radiation [W/m2] [n_columns]
    """
    # Calculate atmospheric emissivity (line 182)
    emiss = 0.7 + 5.95e-5 * 0.01 * vapor_pressure * jnp.exp(1500.0 / temperature)
    
    # Apply Stefan-Boltzmann law (line 184)
    lwrad = emiss * constants.sb * temperature**4
    
    return lwrad


# =============================================================================
# Main Processing Functions
# =============================================================================


def assign_tower_met_to_gridcell(
    raw_data: TowerMetRawData,
    gridcell_forcing: GridcellForcing,
    solar_params: SolarRadiationParams,
    gridcell_index: int,
) -> GridcellForcing:
    """Assign tower meteorology to gridcell-level forcing variables.
    
    From TowerMetMod.F90 lines 103-130.
    
    Args:
        raw_data: Tower meteorology data read from file
        gridcell_forcing: Current gridcell forcing state
        solar_params: Parameters for solar radiation partitioning
        gridcell_index: Index of gridcell to update
        
    Returns:
        Updated gridcell forcing with tower met data
        
    Note:
        - Wind v-component is set to zero (line 111)
        - Solar radiation is partitioned into direct/diffuse and vis/nir bands
    """
    # Update wind components (lines 110-111)
    forc_u = gridcell_forcing.forc_u.at[gridcell_index].set(raw_data.ubot)
    forc_v = gridcell_forcing.forc_v.at[gridcell_index].set(0.0)
    
    # Partition solar radiation (lines 113-129)
    forc_solad_new, forc_solai_new = partition_solar_radiation(
        raw_data.fsdsbot, solar_params
    )
    
    # Update solar radiation arrays
    forc_solad = gridcell_forcing.forc_solad.at[gridcell_index, :].set(forc_solad_new)
    forc_solai = gridcell_forcing.forc_solai.at[gridcell_index, :].set(forc_solai_new)
    
    return GridcellForcing(
        forc_u=forc_u,
        forc_v=forc_v,
        forc_solad=forc_solad,
        forc_solai=forc_solai,
        forc_pco2=gridcell_forcing.forc_pco2,
        forc_po2=gridcell_forcing.forc_po2,
    )


def process_tower_met_forcing(
    raw_data: TowerMetRawData,
    tower_ht: float,
    constants: PhysicalConstants,
    n_columns: int,
    n_patches: int,
    sat_vap_func,  # Function to calculate saturation vapor pressure
) -> Tuple[ColumnForcing, jnp.ndarray]:
    """Process tower meteorology data and handle missing values.
    
    Fortran source: TowerMetMod.F90, lines 141-191
    
    This function:
    1. Assigns tower data to CLM forcing variables (lines 141-149)
    2. Sets forcing height with tower data or default (lines 151-159)
    3. Handles missing atmospheric pressure (lines 161-165)
    4. Converts relative humidity to specific humidity (lines 167-179)
    5. Calculates longwave radiation if missing (lines 181-185)
    
    Args:
        raw_data: Tower meteorology data read from file
        tower_ht: Tower height [m]
        constants: Physical constants
        n_columns: Number of columns
        n_patches: Number of patches
        sat_vap_func: Function to calculate saturation vapor pressure
        
    Returns:
        column_forcing: Column-level forcing variables
        forc_hgt_u: Forcing height at patch level [m] [n_patches]
        
    Note:
        Missing values are indicated by -999. Defaults:
        - Pressure: 101325 Pa (sea level)
        - Forcing height: 30 m
        - Longwave: calculated from emissivity formula
    """
    # Lines 141-149: Assign column-level forcing variables
    forc_t = jnp.full(n_columns, raw_data.tbot)
    forc_q_initial = jnp.full(n_columns, raw_data.qbot)
    forc_pbot_initial = jnp.full(n_columns, raw_data.pbot)
    forc_lwrad_initial = jnp.full(n_columns, raw_data.fldsbot)
    forc_rain = jnp.full(n_columns, raw_data.prect)
    forc_snow = jnp.zeros(n_columns)  # Line 149: forc_snow(c) = 0._r8
    
    # Lines 151-159: Set forcing height at patch level
    forc_hgt_u = jnp.full(n_patches, tower_ht)
    
    # Line 159: Set forcing height to 30 m if tower forcing data has no height
    forc_hgt_u = jnp.where(
        jnp.round(forc_hgt_u) == constants.missing_value,
        constants.default_height,
        forc_hgt_u
    )
    
    # Lines 161-165: Set atmospheric pressure to surface value if missing
    forc_pbot = jnp.where(
        jnp.round(forc_pbot_initial) == constants.missing_value,
        constants.default_pressure,
        forc_pbot_initial
    )
    
    # Lines 167-179: Relative humidity -> specific humidity conversion
    # Call SatVap to get saturation vapor pressure
    esat, _ = sat_vap_func(forc_t)
    
    forc_rh = raw_data.rhbot
    
    # Calculate specific humidity from relative humidity
    has_rh = jnp.round(forc_rh) != constants.missing_value
    has_q = jnp.round(forc_q_initial) != constants.missing_value
    
    # Calculate from RH if available
    forc_q_from_rh = relative_humidity_to_specific_humidity(
        forc_rh, forc_t, forc_pbot, esat, constants
    )
    
    # Calculate vapor pressure from Q if RH not available
    eair_from_q = specific_humidity_to_vapor_pressure(
        forc_q_initial, forc_pbot, constants
    )
    
    # Use RH if available, otherwise use Q
    forc_q = jnp.where(has_rh, forc_q_from_rh, forc_q_initial)
    
    # Get vapor pressure for longwave calculation
    eair_from_rh = (forc_rh / 100.0) * esat
    eair = jnp.where(has_rh, eair_from_rh, eair_from_q)
    
    # Lines 181-185: Calculate atmospheric longwave radiation if missing
    forc_lwrad_calculated = calculate_longwave_radiation(
        forc_t, eair, constants
    )
    
    forc_lwrad = jnp.where(
        jnp.round(forc_lwrad_initial) == constants.missing_value,
        forc_lwrad_calculated,
        forc_lwrad_initial
    )
    
    column_forcing = ColumnForcing(
        forc_t=forc_t,
        forc_q=forc_q,
        forc_pbot=forc_pbot,
        forc_lwrad=forc_lwrad,
        forc_rain=forc_rain,
        forc_snow=forc_snow,
    )
    
    return column_forcing, forc_hgt_u


def assign_co2_o2_coordinates(
    forc_pbot: jnp.ndarray,
    tower_lat: float,
    tower_lon: float,
    col_to_gridcell: jnp.ndarray,
    n_gridcells: int,
    constants: PhysicalConstants,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Complete tower meteorology assignments with CO2, O2, and coordinates.
    
    This function performs the final assignments in the TowerMet subroutine:
    - Converts CO2 concentration (367 ppm) to partial pressure
    - Converts O2 concentration (0.209 mol/mol) to partial pressure
    - Assigns tower latitude and longitude to gridcells
    
    Fortran source: TowerMetMod.F90, lines 192-208
    
    Args:
        forc_pbot: Atmospheric pressure [Pa] [n_columns]
        tower_lat: Tower latitude [degrees]
        tower_lon: Tower longitude [degrees]
        col_to_gridcell: Column to gridcell mapping [n_columns]
        n_gridcells: Number of gridcells
        constants: Physical constants
        
    Returns:
        forc_pco2: CO2 partial pressure [Pa] [n_gridcells]
        forc_po2: O2 partial pressure [Pa] [n_gridcells]
        latdeg: Latitude [degrees] [n_gridcells]
        londeg: Longitude [degrees] [n_gridcells]
        
    Note:
        - CO2 conversion: 367 umol/mol / 1e6 * P_atm = P_CO2
        - O2 conversion: 0.209 mol/mol * P_atm = P_O2
    """
    # Initialize output arrays
    forc_pco2 = jnp.zeros(n_gridcells)
    forc_po2 = jnp.zeros(n_gridcells)
    latdeg = jnp.zeros(n_gridcells)
    londeg = jnp.zeros(n_gridcells)
    
    # Line 194-195: CO2: umol/mol -> Pa and O2: mol/mol -> Pa
    pco2_values = (constants.co2_ppm / 1.0e6) * forc_pbot
    po2_values = constants.o2_frac * forc_pbot
    
    # Line 202-203: Latitude and longitude (degrees)
    lat_values = jnp.full_like(forc_pbot, tower_lat)
    lon_values = jnp.full_like(forc_pbot, tower_lon)
    
    # Scatter values to gridcells
    gridcell_indices = col_to_gridcell
    forc_pco2 = forc_pco2.at[gridcell_indices].set(pco2_values)
    forc_po2 = forc_po2.at[gridcell_indices].set(po2_values)
    latdeg = latdeg.at[gridcell_indices].set(lat_values)
    londeg = londeg.at[gridcell_indices].set(lon_values)
    
    return forc_pco2, forc_po2, latdeg, londeg


# =============================================================================
# Main Tower Met Processing
# =============================================================================


def process_tower_met(
    inputs: TowerMetInputs,
    solar_params: SolarRadiationParams,
    constants: PhysicalConstants,
    sat_vap_func,  # Function to calculate saturation vapor pressure
) -> TowerMetState:
    """Process tower meteorology data into CLM forcing variables.
    
    This is the main function that coordinates all tower met processing:
    1. Partition solar radiation into direct/diffuse and vis/nir bands
    2. Process atmospheric variables (T, q, P, LW, precip)
    3. Handle missing values with appropriate defaults
    4. Convert CO2 and O2 to partial pressures
    5. Assign coordinates
    
    Fortran source: TowerMetMod.F90, lines 29-208 (TowerMet subroutine)
    
    Args:
        inputs: Input parameters including raw data and grid dimensions
        solar_params: Parameters for solar radiation partitioning
        constants: Physical constants
        sat_vap_func: Function to calculate saturation vapor pressure
        
    Returns:
        TowerMetState containing all processed forcing variables
        
    Note:
        This function is JIT-compatible and uses pure functional operations.
        NetCDF I/O should be handled separately before calling this function.
    """
    # Initialize gridcell forcing arrays
    gridcell_forcing = GridcellForcing(
        forc_u=jnp.zeros(inputs.n_gridcells),
        forc_v=jnp.zeros(inputs.n_gridcells),
        forc_solad=jnp.zeros((inputs.n_gridcells, 2)),
        forc_solai=jnp.zeros((inputs.n_gridcells, 2)),
        forc_pco2=jnp.zeros(inputs.n_gridcells),
        forc_po2=jnp.zeros(inputs.n_gridcells),
    )
    
    # Process gridcell-level forcing (wind, solar radiation)
    gridcell_forcing = assign_tower_met_to_gridcell(
        inputs.raw_data,
        gridcell_forcing,
        solar_params,
        gridcell_index=0,  # Single tower site
    )
    
    # Process column-level forcing (T, q, P, LW, precip)
    column_forcing, forc_hgt_u = process_tower_met_forcing(
        inputs.raw_data,
        inputs.tower_ht,
        constants,
        inputs.n_columns,
        inputs.n_patches,
        sat_vap_func,
    )
    
    # Assign CO2, O2, and coordinates
    forc_pco2, forc_po2, latdeg, londeg = assign_co2_o2_coordinates(
        column_forcing.forc_pbot,
        inputs.tower_lat,
        inputs.tower_lon,
        inputs.col_to_gridcell,
        inputs.n_gridcells,
        constants,
    )
    
    # Update gridcell forcing with CO2 and O2
    gridcell_forcing = GridcellForcing(
        forc_u=gridcell_forcing.forc_u,
        forc_v=gridcell_forcing.forc_v,
        forc_solad=gridcell_forcing.forc_solad,
        forc_solai=gridcell_forcing.forc_solai,
        forc_pco2=forc_pco2,
        forc_po2=forc_po2,
    )
    
    # Assemble final state
    # Note: In actual use, these would be broadcast/mapped to patches
    return TowerMetState(
        forc_t=column_forcing.forc_t,
        forc_q=column_forcing.forc_q,
        forc_pbot=column_forcing.forc_pbot,
        forc_u=gridcell_forcing.forc_u,
        forc_v=gridcell_forcing.forc_v,
        forc_lwrad=column_forcing.forc_lwrad,
        forc_rain=column_forcing.forc_rain,
        forc_snow=column_forcing.forc_snow,
        forc_solad=gridcell_forcing.forc_solad,
        forc_solai=gridcell_forcing.forc_solai,
        forc_hgt_u=forc_hgt_u,
        forc_hgt_t=forc_hgt_u,  # Same as wind height
        forc_hgt_q=forc_hgt_u,  # Same as wind height
        forc_pco2=gridcell_forcing.forc_pco2,
        forc_po2=gridcell_forcing.forc_po2,
    )


# =============================================================================
# I/O Interface (Non-JIT)
# =============================================================================


def read_tower_met(
    ncfilename: str,
    strt: int,
) -> TowerMetRawData:
    """Read variables from tower site atmospheric forcing NetCDF files.
    
    This function reads atmospheric forcing data from a NetCDF file at a specific
    time slice. Optional variables are set to -999.0 if not present in the file.
    
    The original Fortran code (lines 211-325) uses NetCDF library calls to:
    1. Open the NetCDF file
    2. Query for each variable by name
    3. Read the data at the specified time slice (strt)
    4. Handle missing optional variables by setting to -999.0
    5. Close the file
    
    Fortran source: TowerMetMod.F90, lines 211-325
    
    Args:
        ncfilename: Path to NetCDF file containing tower meteorology data
        strt: Time slice index to retrieve (0-based in Python)
        
    Returns:
        TowerMetRawData containing all meteorology variables
        
    Note:
        This is an I/O function that should be called OUTSIDE of JIT-compiled code.
        In practice, use netCDF4 or xarray to read the data:
        
        Example:
            import xarray as xr
            ds = xr.open_dataset(ncfilename)
            raw_data = TowerMetRawData(
                zbot=float(ds['ZBOT'].isel(time=strt).values),
                tbot=float(ds['TBOT'].isel(time=strt).values),
                rhbot=float(ds['RH'].isel(time=strt).values) if 'RH' in ds else -999.0,
                qbot=float(ds['QBOT'].isel(time=strt).values) if 'QBOT' in ds else -999.0,
                ubot=float(ds['WIND'].isel(time=strt).values),
                fsdsbot=float(ds['FSDS'].isel(time=strt).values),
                fldsbot=float(ds['FLDS'].isel(time=strt).values) if 'FLDS' in ds else -999.0,
                pbot=float(ds['PSRF'].isel(time=strt).values) if 'PSRF' in ds else -999.0,
                prect=float(ds['PRECTmms'].isel(time=strt).values),
            )
    """
    # This is an I/O function that performs file reading and cannot be JIT-compiled.
    # It should be called outside of any @jit decorated functions.
    
    try:
        import xarray as xr
        import os
        
        # Check if file exists
        if not os.path.exists(ncfilename):
            raise FileNotFoundError(f"Tower meteorology file not found: {ncfilename}")
        
        # Open dataset and read variables
        ds = xr.open_dataset(ncfilename)
        
        # Read required variables
        zbot = float(ds['ZBOT'].isel(time=strt).values) if 'ZBOT' in ds else 30.0
        tbot = float(ds['TBOT'].isel(time=strt).values)
        ubot = float(ds['WIND'].isel(time=strt).values)
        fsdsbot = float(ds['FSDS'].isel(time=strt).values)
        prect = float(ds['PRECTmms'].isel(time=strt).values)
        
        # Read optional variables (set to -999.0 if missing)
        rhbot = float(ds['RH'].isel(time=strt).values) if 'RH' in ds else -999.0
        qbot = float(ds['QBOT'].isel(time=strt).values) if 'QBOT' in ds else -999.0
        fldsbot = float(ds['FLDS'].isel(time=strt).values) if 'FLDS' in ds else -999.0
        pbot = float(ds['PSRF'].isel(time=strt).values) if 'PSRF' in ds else -999.0
        
        ds.close()
        
        return TowerMetRawData(
            zbot=zbot,
            tbot=tbot,
            rhbot=rhbot,
            qbot=qbot,
            ubot=ubot,
            fsdsbot=fsdsbot,
            fldsbot=fldsbot,
            pbot=pbot,
            prect=prect,
        )
        
    except ImportError:
        # xarray not available, return default test data
        import warnings
        warnings.warn(
            f"xarray not available. Cannot read {ncfilename}. "
            "Returning default meteorology values for testing. "
            "Install xarray for actual file I/O: pip install xarray netCDF4",
            category=ImportWarning,
            stacklevel=2
        )
        return TowerMetRawData(
            zbot=30.0,  # 30m measurement height
            tbot=293.15,  # 20°C
            rhbot=70.0,  # 70% relative humidity
            qbot=-999.0,  # Not available
            ubot=3.0,  # 3 m/s wind speed
            fsdsbot=400.0,  # 400 W/m² shortwave radiation
            fldsbot=300.0,  # 300 W/m² longwave radiation
            pbot=101325.0,  # Standard atmospheric pressure (Pa)
            prect=0.0,  # No precipitation
        )
    
    except Exception as e:
        # Handle any file reading errors
        raise IOError(f"Error reading tower meteorology file {ncfilename}: {str(e)}")


# =============================================================================
# Default Parameters
# =============================================================================


def get_default_solar_params() -> SolarRadiationParams:
    """Get default solar radiation partitioning parameters.
    
    Returns:
        SolarRadiationParams with default empirical coefficients
    """
    return SolarRadiationParams()


def get_default_constants() -> PhysicalConstants:
    """Get default physical constants.
    
    Returns:
        PhysicalConstants with standard values
    """
    return PhysicalConstants()