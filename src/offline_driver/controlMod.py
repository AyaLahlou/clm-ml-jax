"""
Control Module for CLM-ML JAX.

Translated from CTSM's controlMod.F90

This module provides initialization and control for namelist run control variables.
It manages simulation configuration including:
- Tower site selection and data paths
- Start/stop dates and time stepping
- Input/output directory configuration
- CLM physics version selection

Key differences from Fortran:
- No mutable module-level state (Fortran common blocks/module variables)
- Control parameters stored in immutable NamedTuples
- Configuration passed explicitly through function arguments
- No side effects (endrun becomes exception raising)
- Pure functional approach compatible with JAX transformations

Reference: controlMod.F90, lines 1-127
"""

from typing import NamedTuple, Tuple, Optional
import jax.numpy as jnp


# =============================================================================
# Type Definitions
# =============================================================================


class ControlConfig(NamedTuple):
    """Configuration parameters from control namelist.
    
    This is the primary output of the control initialization, containing
    all parameters needed to configure a CLM-ML simulation run.
    
    Attributes:
        ntim: Number of time steps to process
        clm_start_ymd: CLM history file start date (yyyymmdd format)
        clm_start_tod: CLM history file start time-of-day (seconds past 0Z UTC)
        diratm: Tower meteorology file directory path
        dirclm: CLM history file directory path
        dirout: Model output file directory path
        dirin: Model input file directory path for profile data
        start_date_ymd: Run start date in yyyymmdd format
        start_date_tod: Time-of-day (UTC) of start date (seconds past 0Z)
        dtstep: Time step of forcing data (seconds)
        tower_num: Index of tower site in TowerDataMod arrays (1-based)
        clm_phys: CLM physics version ('CLM4_5' or 'CLM5_0')
        
    Note:
        Reference: controlMod.F90, lines 21-127
    """
    ntim: int
    clm_start_ymd: int
    clm_start_tod: int
    diratm: str
    dirclm: str
    dirout: str
    dirin: str
    start_date_ymd: int
    start_date_tod: int
    dtstep: int
    tower_num: int
    clm_phys: str


class NamelistInput(NamedTuple):
    """Namelist input parameters.
    
    These parameters would traditionally be read from a Fortran namelist file.
    In JAX, they are passed as a structured input to the control function.
    
    Attributes:
        tower_name: Flux tower site to process (6 characters)
        start_ymd: Run start date in yyyymmdd format
        start_tod: Time-of-day (UTC) of start date (seconds past 0Z; 0 to 86400)
        stop_option: Character flag to specify run length ('ndays' or 'nsteps')
        stop_n: Length of simulation (days or timesteps)
        clm_start_ymd: CLM history file start date (yyyymmdd format)
        clm_start_tod: CLM history file start time-of-day (seconds past 0Z UTC)
        
    Note:
        Reference: controlMod.F90, lines 30-50 (namelist definition)
    """
    tower_name: str
    start_ymd: int
    start_tod: int
    stop_option: str
    stop_n: int
    clm_start_ymd: int
    clm_start_tod: int


class TowerData(NamedTuple):
    """Tower site data arrays.
    
    Contains metadata for all available tower sites. This data would
    traditionally come from TowerDataMod in the Fortran code.
    
    Attributes:
        ntower: Number of tower sites
        tower_id: Array of tower site identifiers [ntower]
        tower_time: Array of time step intervals in minutes [ntower]
        
    Note:
        Reference: TowerDataMod (imported in controlMod.F90, line 11)
    """
    ntower: int
    tower_id: Tuple[str, ...]
    tower_time: Tuple[int, ...]


# =============================================================================
# Exceptions
# =============================================================================


class ControlError(Exception):
    """Exception raised for control/configuration errors.
    
    This replaces the Fortran endrun() call from abortutils.
    In JAX, we use exceptions rather than calling abort routines,
    which is more compatible with Python error handling and debugging.
    
    Reference: controlMod.F90, line 3 (use abortutils, only: endrun)
    """
    pass


# =============================================================================
# Constants
# =============================================================================


# Precision kind (equivalent to shr_kind_r8 from Fortran)
# Reference: controlMod.F90, line 4
R8_KIND = jnp.float64

# Default directory paths
DEFAULT_DIRATM = '../input_files/tower-forcing/'
DEFAULT_DIROUT = '../output_files/'
DEFAULT_DIRIN = '../output_files/'

# CLM physics version directory paths
CLM4_5_DIR = '../input_files/clm4_5/'
CLM5_0_DIR = '../input_files/clm5_0/'

# Time conversion constants
SECONDS_PER_DAY = 86400
SECONDS_PER_MINUTE = 60

# Valid configuration options
VALID_RUN_TYPES = {'startup', 'restart', 'branch'}
VALID_STOP_OPTIONS = {'ndays', 'nsteps'}
VALID_CLM_PHYSICS = {'CLM4_5', 'CLM5_0'}

# Special tower sites that use CLM5.0 physics
# Reference: controlMod.F90, lines 68-70
CLM5_TOWER_SITES = {'CHATS7', 'UMBSmw'}


# =============================================================================
# Default Configuration
# =============================================================================


DEFAULT_CONTROL_CONFIG = ControlConfig(
    ntim=0,
    clm_start_ymd=0,
    clm_start_tod=0,
    diratm=DEFAULT_DIRATM,
    dirclm=CLM4_5_DIR,
    dirout=DEFAULT_DIROUT,
    dirin=DEFAULT_DIRIN,
    start_date_ymd=0,
    start_date_tod=0,
    dtstep=0,
    tower_num=0,
    clm_phys='CLM4_5',
)


# =============================================================================
# Validation Functions
# =============================================================================


def validate_control_config(config: ControlConfig) -> None:
    """Validate control configuration parameters.
    
    Checks that all configuration parameters are within valid ranges
    and have consistent values.
    
    Args:
        config: Control configuration to validate
        
    Raises:
        ControlError: If configuration is invalid
        
    Note:
        This provides additional validation beyond what's in the original
        Fortran code, following Python best practices.
    """
    # Validate CLM physics version
    if config.clm_phys not in VALID_CLM_PHYSICS:
        raise ControlError(
            f"Invalid clm_phys: {config.clm_phys}. "
            f"Must be one of {VALID_CLM_PHYSICS}"
        )
    
    # Validate time step count
    if config.ntim <= 0:
        raise ControlError(f"ntim must be positive, got {config.ntim}")
    
    # Validate time step size
    if config.dtstep <= 0:
        raise ControlError(f"dtstep must be positive, got {config.dtstep}")
    
    # Validate tower number
    if config.tower_num <= 0:
        raise ControlError(f"tower_num must be positive, got {config.tower_num}")
    
    # Validate time-of-day is within valid range
    if not (0 <= config.start_date_tod <= SECONDS_PER_DAY):
        raise ControlError(
            f"start_date_tod must be in [0, {SECONDS_PER_DAY}], "
            f"got {config.start_date_tod}"
        )
    
    if not (0 <= config.clm_start_tod <= SECONDS_PER_DAY):
        raise ControlError(
            f"clm_start_tod must be in [0, {SECONDS_PER_DAY}], "
            f"got {config.clm_start_tod}"
        )


def validate_namelist_input(namelist: NamelistInput) -> None:
    """Validate namelist input parameters.
    
    Args:
        namelist: Namelist input to validate
        
    Raises:
        ControlError: If namelist input is invalid
    """
    # Validate stop option
    if namelist.stop_option not in VALID_STOP_OPTIONS:
        raise ControlError(
            f"Invalid stop_option: {namelist.stop_option}. "
            f"Must be one of {VALID_STOP_OPTIONS}"
        )
    
    # Validate stop_n is positive
    if namelist.stop_n <= 0:
        raise ControlError(f"stop_n must be positive, got {namelist.stop_n}")
    
    # Validate time-of-day
    if not (0 <= namelist.start_tod <= SECONDS_PER_DAY):
        raise ControlError(
            f"start_tod must be in [0, {SECONDS_PER_DAY}], "
            f"got {namelist.start_tod}"
        )
    
    if not (0 <= namelist.clm_start_tod <= SECONDS_PER_DAY):
        raise ControlError(
            f"clm_start_tod must be in [0, {SECONDS_PER_DAY}], "
            f"got {namelist.clm_start_tod}"
        )
    
    # Validate tower name is not empty
    if not namelist.tower_name or not namelist.tower_name.strip():
        raise ControlError("tower_name cannot be empty")


def validate_tower_data(tower_data: TowerData) -> None:
    """Validate tower data structure.
    
    Args:
        tower_data: Tower data to validate
        
    Raises:
        ControlError: If tower data is invalid
    """
    if tower_data.ntower <= 0:
        raise ControlError(f"ntower must be positive, got {tower_data.ntower}")
    
    if len(tower_data.tower_id) != tower_data.ntower:
        raise ControlError(
            f"tower_id length ({len(tower_data.tower_id)}) "
            f"does not match ntower ({tower_data.ntower})"
        )
    
    if len(tower_data.tower_time) != tower_data.ntower:
        raise ControlError(
            f"tower_time length ({len(tower_data.tower_time)}) "
            f"does not match ntower ({tower_data.ntower})"
        )
    
    # Validate all time intervals are positive
    for i, time_interval in enumerate(tower_data.tower_time):
        if time_interval <= 0:
            raise ControlError(
                f"tower_time[{i}] must be positive, got {time_interval}"
            )


# =============================================================================
# Main Control Function
# =============================================================================


def control(
    namelist: NamelistInput,
    tower_data: TowerData,
) -> ControlConfig:
    """Initialize run control variables from namelist.
    
    Translated from controlMod.F90 lines 21-127.
    
    This function processes namelist input to configure the simulation run,
    including tower site selection, time stepping, and directory paths.
    It is a pure function with no side effects, returning an immutable
    configuration structure.
    
    Args:
        namelist: Namelist input parameters
        tower_data: Tower site data (IDs and time intervals)
        
    Returns:
        ControlConfig with all run control parameters
        
    Raises:
        ControlError: If tower site not found, invalid stop_option,
                     or other configuration errors
        
    Note:
        Original Fortran lines 21-127. Key logic:
        - Lines 64-66: Set calendar variables
        - Lines 68-70: Special CLM5.0 physics for CHATS7 and UMBSmw
        - Lines 72-83: Set directory paths based on CLM physics version
        - Lines 85-95: Match tower name to index
        - Lines 97-105: Calculate time steps based on stop_option
        
    Example:
        >>> namelist = NamelistInput(
        ...     tower_name='US-Ha1',
        ...     start_ymd=20000101,
        ...     start_tod=0,
        ...     stop_option='ndays',
        ...     stop_n=365,
        ...     clm_start_ymd=20000101,
        ...     clm_start_tod=0,
        ... )
        >>> tower_data = TowerData(
        ...     ntower=2,
        ...     tower_id=('US-Ha1', 'US-UMB'),
        ...     tower_time=(30, 30),
        ... )
        >>> config = control(namelist, tower_data)
        >>> config.ntim  # Number of 30-minute steps in 365 days
        17520
    """
    # Validate inputs
    validate_namelist_input(namelist)
    validate_tower_data(tower_data)
    
    # Line 64-66: Set calendar variables
    # These are directly copied from namelist
    start_date_ymd = namelist.start_ymd
    start_date_tod = namelist.start_tod
    
    # Line 68-70: CHATS and UMBSmw use CLM5.0 soils
    # Default to CLM4.5, but certain tower sites require CLM5.0
    clm_phys = 'CLM4_5'  # Default
    if namelist.tower_name in CLM5_TOWER_SITES:
        clm_phys = 'CLM5_0'
    
    # Line 72-83: Specify input and output directories
    # Atmospheric forcing directory is constant
    diratm = DEFAULT_DIRATM
    
    # CLM history file directory depends on physics version
    if clm_phys == 'CLM4_5':
        dirclm = CLM4_5_DIR
    elif clm_phys == 'CLM5_0':
        dirclm = CLM5_0_DIR
    else:
        # Fallback to CLM4.5 (should not reach here due to validation)
        dirclm = CLM4_5_DIR
    
    # Output and input directories
    dirout = DEFAULT_DIROUT
    dirin = DEFAULT_DIRIN
    
    # Line 85-95: Match tower site to correct index for TowerDataMod arrays
    # Search through tower_id array to find matching site
    tower_num = 0
    for i in range(tower_data.ntower):
        if namelist.tower_name == tower_data.tower_id[i]:
            tower_num = i + 1  # Fortran 1-based indexing
            break
    
    # Error if tower site not found
    if tower_num == 0:
        raise ControlError(
            f'control error: tower site = {namelist.tower_name} not found. '
            f'Available sites: {tower_data.tower_id}'
        )
    
    # Line 99: Time step of forcing data (in seconds)
    # Convert from minutes to seconds
    # Note: tower_num is 1-based, so subtract 1 for 0-based Python indexing
    dtstep = tower_data.tower_time[tower_num - 1] * SECONDS_PER_MINUTE
    
    # Line 101-107: Set length of simulation
    # Calculate number of time steps based on stop_option
    if namelist.stop_option == 'nsteps':
        # Direct specification of number of steps
        ntim = namelist.stop_n
    elif namelist.stop_option == 'ndays':
        # Calculate steps from number of days
        steps_per_day = SECONDS_PER_DAY // dtstep  # Integer division
        ntim = steps_per_day * namelist.stop_n
    else:
        # Should not reach here due to validation, but include for safety
        raise ControlError(
            f'control error: invalid stop_option = {namelist.stop_option}. '
            f'Must be one of {VALID_STOP_OPTIONS}'
        )
    
    # Construct configuration
    config = ControlConfig(
        ntim=ntim,
        clm_start_ymd=namelist.clm_start_ymd,
        clm_start_tod=namelist.clm_start_tod,
        diratm=diratm,
        dirclm=dirclm,
        dirout=dirout,
        dirin=dirin,
        start_date_ymd=start_date_ymd,
        start_date_tod=start_date_tod,
        dtstep=dtstep,
        tower_num=tower_num,
        clm_phys=clm_phys,
    )
    
    # Final validation of constructed configuration
    validate_control_config(config)
    
    return config


# =============================================================================
# Utility Functions
# =============================================================================


def get_steps_per_day(dtstep: int) -> int:
    """Calculate number of time steps per day.
    
    Args:
        dtstep: Time step size in seconds
        
    Returns:
        Number of time steps in one day
        
    Raises:
        ControlError: If dtstep does not evenly divide a day
    """
    if SECONDS_PER_DAY % dtstep != 0:
        raise ControlError(
            f"dtstep ({dtstep}s) does not evenly divide a day "
            f"({SECONDS_PER_DAY}s)"
        )
    return SECONDS_PER_DAY // dtstep


def calculate_total_seconds(ntim: int, dtstep: int) -> int:
    """Calculate total simulation time in seconds.
    
    Args:
        ntim: Number of time steps
        dtstep: Time step size in seconds
        
    Returns:
        Total simulation time in seconds
    """
    return ntim * dtstep


def calculate_total_days(ntim: int, dtstep: int) -> float:
    """Calculate total simulation time in days.
    
    Args:
        ntim: Number of time steps
        dtstep: Time step size in seconds
        
    Returns:
        Total simulation time in days (may be fractional)
    """
    total_seconds = calculate_total_seconds(ntim, dtstep)
    return total_seconds / SECONDS_PER_DAY


def format_ymd(ymd: int) -> str:
    """Format yyyymmdd integer as readable date string.
    
    Args:
        ymd: Date in yyyymmdd format
        
    Returns:
        Formatted date string "YYYY-MM-DD"
        
    Example:
        >>> format_ymd(20000101)
        '2000-01-01'
    """
    year = ymd // 10000
    month = (ymd % 10000) // 100
    day = ymd % 100
    return f"{year:04d}-{month:02d}-{day:02d}"


def format_tod(tod: int) -> str:
    """Format time-of-day in seconds as readable time string.
    
    Args:
        tod: Time-of-day in seconds past midnight
        
    Returns:
        Formatted time string "HH:MM:SS"
        
    Example:
        >>> format_tod(3661)
        '01:01:01'
    """
    hours = tod // 3600
    minutes = (tod % 3600) // 60
    seconds = tod % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_config_summary(config: ControlConfig) -> str:
    """Generate human-readable summary of configuration.
    
    Args:
        config: Control configuration
        
    Returns:
        Multi-line string summarizing configuration
        
    Note:
        This is useful for logging and debugging, similar to the
        write statements in the original Fortran code.
    """
    total_days = calculate_total_days(config.ntim, config.dtstep)
    
    summary = f"""
Control Configuration Summary:
==============================
Tower Site: #{config.tower_num}
CLM Physics: {config.clm_phys}

Time Configuration:
  Start Date: {format_ymd(config.start_date_ymd)} {format_tod(config.start_date_tod)}
  CLM Start:  {format_ymd(config.clm_start_ymd)} {format_tod(config.clm_start_tod)}
  Time Step:  {config.dtstep} seconds
  Total Steps: {config.ntim}
  Total Days:  {total_days:.2f}

Directories:
  Atmospheric: {config.diratm}
  CLM History: {config.dirclm}
  Output:      {config.dirout}
  Input:       {config.dirin}
"""
    return summary.strip()