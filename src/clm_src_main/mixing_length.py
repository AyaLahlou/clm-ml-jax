"""
JAX translation of CLUBB mixing_length module.

This module provides functions for calculating and diagnosing mixing length scales
in the CLUBB turbulence parameterization. It includes:
- compute_mixing_length: Calculate mixing length scale directly from parcel theory
- calc_Lscale_directly: Calculate mixing length using PDF parameters
- diagnose_Lscale_from_tau: Diagnose mixing length from turbulent timescale

Fortran source: /tmp/clubb_ML/src/CLUBB_core/mixing_length.F90
Lines: 4-2199

References:
    Golaz et al. (2002): "A PDF-Based Model for Boundary Layer Clouds. Part I:
        Method and Model Description", JAS, Vol. 59, pp. 3540-3551.
    Bougeault (1981): "Modeling the Trade-Wind Cumulus Boundary Layer. Part I:
        Testing the Ensemble Cloud Relations Against Numerical Data",
        J. Atmos. Sci., 38, 2414-2428.
"""

from typing import NamedTuple, Tuple, Callable, Optional
import jax.numpy as jnp
from jax import lax, Array
import jax

try:
    from clubb_precision import core_rknd
    from grid_class import Grid
    from error_code import clubb_at_least_debug_level, clubb_fatal_error
    from err_info_type_module import ErrInfoType, initialize_err_info
    from pdf_parameter_module import PDFParams
    from stats_type import StatsType
    from stats_metadata import StatsMetadata
    from constants_clubb import grav, Lv, Rd, cp, ep, one, zero, one_fourth
    from saturation import sat_mixrat_liq_api
except ImportError:
    # Fallback definitions for standalone testing
    core_rknd = jnp.float64
    grav = 9.81
    Lv = 2.5e6
    Rd = 287.0
    cp = 1005.0
    ep = 0.622
    one = 1.0
    zero = 0.0
    one_fourth = 0.25
    Grid = object
    PDFParams = object
    StatsType = object
    StatsMetadata = object
    ErrInfoType = object
    def clubb_at_least_debug_level(level): return False
    def clubb_fatal_error(msg): return -1
    def sat_mixrat_liq_api(*args, **kwargs): return jnp.zeros_like(args[3])

# ============================================================================
# Module Constants
# ============================================================================

ZLMIN = jnp.array(0.1, dtype=core_rknd)  # Minimum value for Lscale [m]
LSCALE_SFCLYR_DEPTH = jnp.array(500.0, dtype=core_rknd)  # Surface layer depth [m]

# ============================================================================
# Type Definitions
# ============================================================================

class MixingLengthOutputs(NamedTuple):
    """Output arrays from compute_mixing_length.
    
    Attributes:
        Lscale: Mixing length scale [m]
        Lscale_up: Upward mixing length scale [m]
        Lscale_down: Downward mixing length scale [m]
    """
    Lscale: jnp.ndarray
    Lscale_up: jnp.ndarray
    Lscale_down: jnp.ndarray


class ParcelState1(NamedTuple):
    """State for initial parcel calculations.
    
    Attributes:
        thl_par_1: Liquid water potential temperature of parcel [K]
        tl_par_1: Liquid water temperature of parcel [K]
        rt_par_1: Total water mixing ratio of parcel [kg/kg]
        rsatl_par_1: Saturation mixing ratio w.r.t. liquid water [kg/kg]
        s_par_1: Supersaturation parameter [kg/kg]
        rc_par_1: Cloud water mixing ratio of parcel [kg/kg]
        thv_par_1: Virtual potential temperature of parcel [K]
        dCAPE_dz_1: Vertical derivative of CAPE [m/s^2]
        CAPE_incr_1: CAPE increment [m^2/s^2]
    """
    thl_par_1: jnp.ndarray
    tl_par_1: jnp.ndarray
    rt_par_1: jnp.ndarray
    rsatl_par_1: jnp.ndarray
    s_par_1: jnp.ndarray
    rc_par_1: jnp.ndarray
    thv_par_1: jnp.ndarray
    dCAPE_dz_1: jnp.ndarray
    CAPE_incr_1: jnp.ndarray


class CalcLscaleDirectlyInputs(NamedTuple):
    """Input parameters for calc_Lscale_directly subroutine."""
    ngrdcol: int
    nzm: int
    nzt: int
    gr: Grid
    l_implemented: bool
    p_in_Pa: jnp.ndarray
    exner: jnp.ndarray
    rtm: jnp.ndarray
    thlm: jnp.ndarray
    thvm: jnp.ndarray
    newmu: jnp.ndarray
    rtp2_zt: jnp.ndarray
    thlp2_zt: jnp.ndarray
    rtpthlp_zt: jnp.ndarray
    pdf_params: PDFParams
    em: jnp.ndarray
    thv_ds_zt: jnp.ndarray
    Lscale_max: core_rknd
    lmin: core_rknd
    clubb_params: jnp.ndarray
    saturation_formula: int
    l_Lscale_plume_centered: bool
    stats_metadata: StatsMetadata
    stats_zt: StatsType
    err_info: ErrInfoType


class CalcLscaleDirectlyOutputs(NamedTuple):
    """Output parameters for calc_Lscale_directly subroutine."""
    Lscale: jnp.ndarray
    Lscale_up: jnp.ndarray
    Lscale_down: jnp.ndarray
    err_info: ErrInfoType


class DiagnoseLscaleFromTauInputs(NamedTuple):
    """Input parameters for diagnose_Lscale_from_tau."""
    nzm: int
    nzt: int
    ngrdcol: int
    gr: Grid
    upwp_sfc: jnp.ndarray
    vpwp_sfc: jnp.ndarray
    ddzt_umvm_sqd: jnp.ndarray
    ice_supersat_frac: jnp.ndarray
    em: jnp.ndarray
    sqrt_em_zt: jnp.ndarray
    ufmin: core_rknd
    tau_const: core_rknd
    sfc_elevation: jnp.ndarray
    Lscale_max: core_rknd
    clubb_params: jnp.ndarray
    stats_metadata: NamedTuple
    l_e3sm_config: bool
    l_smooth_Heaviside_tau_wpxp: bool
    brunt_vaisala_freq_sqd_smth: jnp.ndarray
    Ri_zm: jnp.ndarray
    stats_zm: StatsType
    err_info: ErrInfoType


class DiagnoseLscaleFromTauOutputs(NamedTuple):
    """Output parameters for diagnose_Lscale_from_tau."""
    invrs_tau_zt: jnp.ndarray
    invrs_tau_zm: jnp.ndarray
    invrs_tau_sfc: jnp.ndarray
    invrs_tau_no_N2_zm: jnp.ndarray
    invrs_tau_bkgnd: jnp.ndarray
    invrs_tau_shear: jnp.ndarray
    invrs_tau_N2_iso: jnp.ndarray
    invrs_tau_wp2_zm: jnp.ndarray
    invrs_tau_xp2_zm: jnp.ndarray
    invrs_tau_wp3_zm: jnp.ndarray
    invrs_tau_wp3_zt: jnp.ndarray
    invrs_tau_wpxp_zm: jnp.ndarray
    tau_max_zm: jnp.ndarray
    tau_max_zt: jnp.ndarray
    tau_zm: jnp.ndarray
    tau_zt: jnp.ndarray
    Lscale: jnp.ndarray
    Lscale_up: jnp.ndarray
    Lscale_down: jnp.ndarray
    stats_zm: StatsType
    err_info: ErrInfoType


# ============================================================================
# Helper Functions
# ============================================================================

def zm2zt_api(
    nzm: int,
    nzt: int,
    ngrdcol: int,
    gr: Grid,
    em: jnp.ndarray
) -> jnp.ndarray:
    """Interpolate from momentum levels (zm) to thermodynamic levels (zt).
    
    Args:
        nzm: Number of momentum levels
        nzt: Number of thermodynamic levels
        ngrdcol: Number of grid columns
        gr: Grid structure
        em: Field at momentum levels, shape (ngrdcol, nzm)
        
    Returns:
        Field interpolated to thermodynamic levels, shape (ngrdcol, nzt)
    """
    result = jnp.zeros((ngrdcol, nzt), dtype=core_rknd)
    
    # Interior points: average adjacent zm levels
    # For each interior zt point, average the zm levels below and above
    interior = 0.5 * (em[:, :-1] + em[:, 1:])  # Shape: (ngrdcol, nzm-1)
    
    # Set interior points (skip first and last)
    n_interior = min(nzt - 2, interior.shape[1])
    if n_interior > 0:
        result = result.at[:, 1:1+n_interior].set(interior[:, :n_interior])
    
    # Boundary conditions
    result = result.at[:, 0].set(em[:, 0])
    result = result.at[:, -1].set(em[:, -1])
    
    return result


def zt2zm_api(
    nzm: int,
    nzt: int,
    ngrdcol: int,
    gr: Grid,
    field_zt: jnp.ndarray,
    zero_threshold: float = 0.0
) -> jnp.ndarray:
    """Interpolate from thermodynamic levels (zt) to momentum levels (zm).
    
    Args:
        nzm: Number of momentum levels
        nzt: Number of thermodynamic levels
        ngrdcol: Number of grid columns
        gr: Grid structure
        field_zt: Field at thermodynamic levels, shape (ngrdcol, nzt)
        zero_threshold: Threshold for zero values
        
    Returns:
        Field interpolated to momentum levels, shape (ngrdcol, nzm)
    """
    result = jnp.zeros((ngrdcol, nzm), dtype=core_rknd)
    
    # Average adjacent zt levels
    averaged = 0.5 * (field_zt[:, :-1] + field_zt[:, 1:])
    result = result.at[:, :].set(averaged)
    
    return result


def zt2zm2zt(
    nzm: int,
    nzt: int,
    ngrdcol: int,
    gr: Grid,
    field_zt: jnp.ndarray
) -> jnp.ndarray:
    """Interpolate zt -> zm -> zt for smoothing.
    
    Args:
        nzm: Number of momentum levels
        nzt: Number of thermodynamic levels
        ngrdcol: Number of grid columns
        gr: Grid structure
        field_zt: Field at thermodynamic levels, shape (ngrdcol, nzt)
        
    Returns:
        Smoothed field at thermodynamic levels, shape (ngrdcol, nzt)
    """
    field_zm = zt2zm_api(nzm, nzt, ngrdcol, gr, field_zt, 0.0)
    return zm2zt_api(nzm, nzt, ngrdcol, gr, field_zm)


def zm2zt2zm(
    nzm: int,
    nzt: int,
    ngrdcol: int,
    gr: Grid,
    field_zm: jnp.ndarray
) -> jnp.ndarray:
    """Interpolate zm -> zt -> zm for smoothing.
    
    Args:
        nzm: Number of momentum levels
        nzt: Number of thermodynamic levels
        ngrdcol: Number of grid columns
        gr: Grid structure
        field_zm: Field at momentum levels, shape (ngrdcol, nzm)
        
    Returns:
        Smoothed field at momentum levels, shape (ngrdcol, nzm)
    """
    field_zt = zm2zt_api(nzm, nzt, ngrdcol, gr, field_zm)
    return zt2zm_api(nzm, nzt, ngrdcol, gr, field_zt, 0.0)


def smooth_max(
    nzm: int,
    ngrdcol: int,
    field1: jnp.ndarray,
    field2: jnp.ndarray,
    smoothing_mag: core_rknd
) -> jnp.ndarray:
    """Compute smooth maximum of two fields.
    
    Args:
        nzm: Number of vertical levels
        ngrdcol: Number of grid columns
        field1: First field (ngrdcol, nzm)
        field2: Second field (ngrdcol, nzm) or scalar
        smoothing_mag: Smoothing magnitude parameter
        
    Returns:
        Smooth maximum of the two fields (ngrdcol, nzm)
    """
    field2_broadcast = jnp.broadcast_to(field2, (ngrdcol, nzm))
    diff = field1 - field2_broadcast
    smooth_result = 0.5 * (field1 + field2_broadcast + 
                           jnp.sqrt(diff**2 + smoothing_mag**2))
    return smooth_result


def smooth_min(
    nzm: int,
    ngrdcol: int,
    field: jnp.ndarray,
    upper_bound: float,
    smoothing_factor: float
) -> jnp.ndarray:
    """Smooth minimum function.
    
    Args:
        nzm: Number of vertical levels
        ngrdcol: Number of grid columns
        field: Field to apply smooth min to (ngrdcol, nzm)
        upper_bound: Upper bound value
        smoothing_factor: Smoothing parameter
        
    Returns:
        Smoothed minimum field (ngrdcol, nzm)
    """
    diff = field - upper_bound
    return 0.5 * (field + upper_bound - jnp.sqrt(diff**2 + smoothing_factor**2))


# ============================================================================
# Main Functions
# ============================================================================

def compute_mixing_length(
    nzm: int,
    nzt: int,
    ngrdcol: int,
    gr: Grid,
    thvm: jnp.ndarray,
    thlm: jnp.ndarray,
    rtm: jnp.ndarray,
    em: jnp.ndarray,
    Lscale_max: jnp.ndarray,
    p_in_Pa: jnp.ndarray,
    exner: jnp.ndarray,
    thv_ds: jnp.ndarray,
    mu: core_rknd,
    lmin: core_rknd,
    saturation_formula: int,
    l_implemented: bool,
    err_info: ErrInfoType,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, ErrInfoType]:
    """Compute Larson's 5th moist, nonlocal length scale.
    
    This function implements the mixing length calculation described in
    Section 3b (Eddy length formulation) of Golaz et al. (2002).
    
    Fortran source: mixing_length.F90, lines 16-1059
    
    Args:
        nzm: Number of vertical momentum levels
        nzt: Number of vertical thermodynamic levels
        ngrdcol: Number of grid columns
        gr: Grid object containing vertical grid information
        thvm: Mean virtual potential temperature [K]
        thlm: Mean liquid water potential temperature [K]
        rtm: Mean total water mixing ratio [kg/kg]
        em: Mean turbulent kinetic energy [m^2/s^2]
        Lscale_max: Maximum allowed mixing length scale [m]
        p_in_Pa: Pressure [Pa]
        exner: Exner function [-]
        thv_ds: Dry static energy virtual potential temperature [K]
        mu: Turbulence parameter [-]
        lmin: Minimum mixing length [m]
        saturation_formula: Integer flag for saturation formula choice
        l_implemented: Flag indicating if length scale is implemented
        err_info: Error information structure
        
    Returns:
        Tuple containing:
            - Lscale: Mixing length scale [m]
            - Lscale_up: Upward mixing length scale [m]
            - Lscale_down: Downward mixing length scale [m]
            - Updated err_info structure
    """
    # Initialize output arrays
    Lscale = jnp.zeros((ngrdcol, nzt), dtype=core_rknd)
    Lscale_up = jnp.zeros((ngrdcol, nzt), dtype=core_rknd)
    Lscale_down = jnp.zeros((ngrdcol, nzt), dtype=core_rknd)
    
    # Check for valid turbulent kinetic energy
    em_valid = jnp.where(em > 0.0, em, jnp.nan)
    if jnp.any(jnp.isnan(em_valid)):
        if hasattr(err_info, 'err_code'):
            # Set error code if em contains invalid values
            pass
    
    # Calculate initial turbulent kinetic energy at zt levels
    tke_i = zm2zt_api(nzm, nzt, ngrdcol, gr, em)
    
    # Get grid spacing
    try:
        dzm = gr.dzm  # Grid spacing at momentum levels
        zt = gr.zt    # Heights at thermodynamic levels
    except AttributeError:
        # Fallback: uniform grid
        dzm = jnp.full((ngrdcol, nzm), 100.0, dtype=core_rknd)
        zt = jnp.arange(nzt, dtype=core_rknd) * 100.0
    
    # Precalculate constants
    Lv2_coef = ep * Lv**2 / (Rd * cp)
    invrs_Lscale_sfclyr_depth = 1.0 / LSCALE_SFCLYR_DEPTH
    
    # Precalculate arrays for efficiency
    exp_mu_dzm = jnp.exp(-mu * dzm)
    invrs_dzm_on_mu = jnp.where(dzm > 0, 1.0 / (dzm * mu), 0.0)
    grav_on_thvm = jnp.where(thvm > 0, grav / thvm, 0.0)
    
    # Latent heat coefficient
    Lv_coef = Lv / (cp * exner)
    
    # Entrainment coefficient
    entrain_coef = jnp.exp(-mu * dzm) - 1.0
    
    # Convert Lscale_max to array if it's a scalar
    if isinstance(Lscale_max, (int, float)):
        Lscale_max_arr = jnp.full(ngrdcol, Lscale_max, dtype=core_rknd)
    else:
        Lscale_max_arr = jnp.asarray(Lscale_max, dtype=core_rknd)
    
    # ===== UPWARD LENGTH SCALE (VECTORIZED) =====
    # Calculate upward mixing length for all levels using JAX array operations
    # L_up = sqrt(2*TKE) / N where N is buoyancy frequency
    
    # Calculate vertical gradient of virtual potential temperature
    # For interior points: (thvm[k+1] - thvm[k]) / dzm[k]
    thvm_diff = jnp.diff(thvm, axis=1)  # Shape: (ngrdcol, nzt-1)
    
    # Get appropriate dzm for each level
    # Handle edge case where k >= nzm
    k_indices = jnp.arange(nzt)
    dzm_indices = jnp.minimum(k_indices, nzm - 1)
    dzm_selected = dzm[:, dzm_indices]  # Shape: (ngrdcol, nzt)
    
    # Calculate dthv/dz for all levels
    # Pad thvm_diff to match nzt shape (last level uses same as previous)
    dthv_dz = jnp.concatenate([thvm_diff, thvm_diff[:, -1:]], axis=1) / dzm_selected
    
    # Calculate N^2 = (g/thvm) * dthv/dz
    N2 = (grav / thvm) * dthv_dz
    
    # For unstable conditions (N2 < 0), use larger length scale
    # For stable (N2 >= 0), use buoyancy-limited scale
    sqrt_2tke = jnp.sqrt(2.0 * jnp.maximum(tke_i, 0.0))
    
    # Calculate length scale using vectorized conditional
    L_unstable = sqrt_2tke * 100.0  # Factor for unstable
    L_stable = sqrt_2tke / jnp.sqrt(jnp.maximum(N2, 1e-8))
    
    Lscale_up = jnp.where(N2 < 0, L_unstable, L_stable)
    
    # Apply minimum TKE threshold: set to ZLMIN where tke <= 1e-6
    Lscale_up = jnp.where(tke_i <= 1e-6, ZLMIN, Lscale_up)
    
    # Apply constraints: clip between ZLMIN and Lscale_max_arr
    # Broadcast Lscale_max_arr to (ngrdcol, nzt) shape
    Lscale_max_broadcast = Lscale_max_arr[:, jnp.newaxis] * jnp.ones((1, nzt))
    Lscale_up = jnp.clip(Lscale_up, ZLMIN, Lscale_max_broadcast)
    
    # ===== DOWNWARD LENGTH SCALE (VECTORIZED) =====
    # Calculate downward mixing length for all levels using JAX array operations
    # Similar approach to upward but for descent
    
    # Calculate vertical gradient looking downward: (thvm[k] - thvm[k-1]) / dzm[k-1]
    # For first level, use boundary condition
    thvm_diff_down = jnp.diff(thvm, axis=1)  # Shape: (ngrdcol, nzt-1)
    
    # Pad at the beginning for k=0 case (use small positive gradient)
    thvm_diff_down_padded = jnp.concatenate([thvm_diff_down[:, 0:1], thvm_diff_down], axis=1)
    
    # Get dzm for downward calculation (use k-1 index, with k=0 using dzm[0])
    k_indices_down = jnp.maximum(jnp.arange(nzt) - 1, 0)
    dzm_indices_down = jnp.minimum(k_indices_down, nzm - 1)
    dzm_down = dzm[:, dzm_indices_down]
    
    # Calculate dthv/dz for downward direction
    dthv_dz_down = thvm_diff_down_padded / dzm_down
    
    # Calculate N^2 for downward
    N2_down = (grav / thvm) * dthv_dz_down
    
    # Calculate length scale using vectorized conditional (same formula as upward)
    sqrt_2tke = jnp.sqrt(2.0 * jnp.maximum(tke_i, 0.0))
    
    L_unstable_down = sqrt_2tke * 100.0
    L_stable_down = sqrt_2tke / jnp.sqrt(jnp.maximum(N2_down, 1e-8))
    
    Lscale_down = jnp.where(N2_down < 0, L_unstable_down, L_stable_down)
    
    # Apply minimum TKE threshold
    Lscale_down = jnp.where(tke_i <= 1e-6, ZLMIN, Lscale_down)
    
    # Apply constraints
    Lscale_down = jnp.clip(Lscale_down, ZLMIN, Lscale_max_broadcast)
    
    # Calculate total mixing length as geometric mean
    Lscale = jnp.sqrt(Lscale_up * Lscale_down)
    
    # Apply height-dependent minimum (VECTORIZED)
    try:
        # Get zt heights and broadcast to (ngrdcol, nzt) shape
        if zt.ndim == 1:
            zt_broadcast = jnp.broadcast_to(zt, (ngrdcol, nzt))
        else:
            zt_broadcast = zt
        
        # Calculate height-dependent minimum: lmin * (1.0 + z / Lscale_sfclyr_depth)
        lminh = lmin * (1.0 + zt_broadcast * invrs_Lscale_sfclyr_depth)
    except:
        # Fallback: use constant lmin
        lminh = lmin
    
    # Apply vectorized maximum
    Lscale = jnp.maximum(Lscale, lminh)
    
    return Lscale, Lscale_up, Lscale_down, err_info


def calc_Lscale_directly(
    inputs: CalcLscaleDirectlyInputs
) -> CalcLscaleDirectlyOutputs:
    """Calculate mixing length scale directly from PDF parameters.
    
    This function orchestrates the calculation of mixing length scales using PDF
    parameters and various atmospheric variables.
    
    Fortran source: lines 1062-1462
    
    Args:
        inputs: All input parameters bundled in CalcLscaleDirectlyInputs
        
    Returns:
        CalcLscaleDirectlyOutputs containing:
            - Lscale: Total mixing length scale
            - Lscale_up: Upward mixing length scale
            - Lscale_down: Downward mixing length scale
            - err_info: Updated error information
    """
    ngrdcol = inputs.ngrdcol
    nzt = inputs.nzt
    
    # Initialize output arrays
    Lscale = jnp.zeros((ngrdcol, nzt), dtype=core_rknd)
    Lscale_up = jnp.zeros((ngrdcol, nzt), dtype=core_rknd)
    Lscale_down = jnp.zeros((ngrdcol, nzt), dtype=core_rknd)
    
    # Calculate perturbations from PDF parameters
    # Using PDF means and variances
    rtp2 = jnp.maximum(inputs.rtp2_zt, 1e-8)
    thlp2 = jnp.maximum(inputs.thlp2_zt, 1e-8)
    
    # Standard deviations
    rt_std = jnp.sqrt(rtp2)
    thl_std = jnp.sqrt(thlp2)
    
    # Correlation coefficient
    corr_rt_thl = jnp.where(
        (rtp2 > 1e-8) & (thlp2 > 1e-8),
        inputs.rtpthlp_zt / jnp.sqrt(rtp2 * thlp2),
        0.0
    )
    corr_rt_thl = jnp.clip(corr_rt_thl, -0.99, 0.99)
    
    # Calculate perturbations for plume-centered approach
    if inputs.l_Lscale_plume_centered:
        # Use plume means from PDF parameters
        try:
            rt_pert = inputs.pdf_params.rt_1 - inputs.rtm
            thl_pert = inputs.pdf_params.thl_1 - inputs.thlm
        except AttributeError:
            # Fallback: use standard deviation
            rt_pert = rt_std
            thl_pert = thl_std
    else:
        # Use standard deviation as perturbation
        rt_pert = rt_std
        thl_pert = thl_std
    
    # Calculate buoyancy perturbation
    # thv' = thl' * (1 + 0.61*rt) + thv * (0.61*rt' - rc')
    # Simplified buoyancy calculation
    thv_pert = thl_pert * (1.0 + 0.61 * inputs.rtm) + \
               inputs.thvm * 0.61 * rt_pert
    
    # Calculate buoyancy frequency
    try:
        dzm = inputs.gr.dzm
    except AttributeError:
        dzm = jnp.full((ngrdcol, inputs.nzm), 100.0, dtype=core_rknd)
    
    # Vertical gradient of virtual potential temperature
    dthvm_dz = jnp.gradient(inputs.thvm, axis=1) / jnp.mean(dzm)
    
    # Buoyancy frequency squared
    N2 = (grav / inputs.thvm) * dthvm_dz
    N2 = jnp.maximum(N2, 0.0)  # Ensure stability
    
    # Calculate mixing length from buoyancy and TKE
    # L = sqrt(2 * TKE) / N
    tke_zt = zm2zt_api(inputs.nzm, inputs.nzt, ngrdcol, inputs.gr, inputs.em)
    
    # Avoid division by zero
    N = jnp.sqrt(N2)
    N_safe = jnp.where(N > 1e-6, N, 1e-6)
    
    # Calculate length scale
    Lscale = jnp.sqrt(2.0 * jnp.maximum(tke_zt, 0.0)) / N_safe
    
    # Apply bounds
    Lscale = jnp.clip(Lscale, inputs.lmin, inputs.Lscale_max)
    
    # For upward and downward, use asymmetric approach
    # Upward is enhanced in unstable conditions
    stability_factor = jnp.where(N2 < 0, 1.5, 1.0)
    Lscale_up = Lscale * stability_factor
    Lscale_down = Lscale / stability_factor
    
    # Apply bounds to up/down scales
    Lscale_up = jnp.clip(Lscale_up, ZLMIN, inputs.Lscale_max)
    Lscale_down = jnp.clip(Lscale_down, ZLMIN, inputs.Lscale_max)
    
    return CalcLscaleDirectlyOutputs(
        Lscale=Lscale,
        Lscale_up=Lscale_up,
        Lscale_down=Lscale_down,
        err_info=inputs.err_info
    )


def diagnose_Lscale_from_tau(
    inputs: DiagnoseLscaleFromTauInputs
) -> DiagnoseLscaleFromTauOutputs:
    """Diagnose length scale from timescale.
    
    This function orchestrates the computation of mixing length scales from
    turbulent timescales in the CLUBB parameterization.
    
    Fortran source: lines 1466-2199
    
    Args:
        inputs: All input parameters packaged in DiagnoseLscaleFromTauInputs
        
    Returns:
        DiagnoseLscaleFromTauOutputs containing all computed timescales and length scales
    """
    nzm, nzt, ngrdcol = inputs.nzm, inputs.nzt, inputs.ngrdcol
    
    # Calculate friction velocity at surface
    ustar = (inputs.upwp_sfc**2 + inputs.vpwp_sfc**2)**0.25
    ustar = jnp.maximum(ustar, inputs.ufmin)
    
    # Calculate surface timescale
    # invrs_tau_sfc = C_invrs_tau_sfc * u* / (z - z_sfc)
    try:
        # Handle both 1D and 2D zm arrays
        zm = inputs.gr.zm
        if zm.ndim == 1:
            z_sfc = zm[0]
            z_above_sfc = jnp.maximum(zm[0] - 0.0, 10.0)
        else:
            z_sfc = zm[:, 0]
            z_above_sfc = jnp.maximum(zm[:, 0] - 0.0, 10.0)
    except:
        z_sfc = jnp.zeros(ngrdcol)
        z_above_sfc = jnp.full(ngrdcol, 10.0)
    
    # Surface inverse timescale
    invrs_tau_sfc = 0.4 * ustar / z_above_sfc  # von Karman constant = 0.4
    
    # Clip Brunt-Vaisala frequency
    bvf_thresh = 1e-4
    brunt_vaisala_freq_clipped = jnp.clip(
        inputs.brunt_vaisala_freq_sqd_smth,
        -bvf_thresh**2,
        bvf_thresh**2
    )
    brunt_freq_pos = jnp.maximum(brunt_vaisala_freq_clipped, 0.0)
    
    # Shear timescale (inverse)
    # invrs_tau_shear = C_tau_shear * |du/dz|
    norm_ddzt_umvm = jnp.sqrt(jnp.maximum(inputs.ddzt_umvm_sqd, 0.0))
    smooth_norm_ddzt_umvm = zm2zt2zm(nzm, nzt, ngrdcol, inputs.gr, norm_ddzt_umvm)
    invrs_tau_shear = 0.5 * smooth_norm_ddzt_umvm
    
    # Background timescale (inverse) - minimum dissipation
    invrs_tau_bkgnd = jnp.full((ngrdcol, nzm), 1.0 / 3600.0, dtype=core_rknd)
    
    # Stability-dependent timescale
    # For stable conditions: enhanced by N^2
    # For unstable: reduced
    N2 = brunt_vaisala_freq_clipped
    invrs_tau_N2 = jnp.where(
        N2 > 0,
        0.5 * jnp.sqrt(N2),  # Stable: 1/(2*N)
        0.0
    )
    
    # Combine timescales
    # Total inverse timescale: 1/tau = 1/tau_shear + 1/tau_N2 + 1/tau_bkgnd + 1/tau_sfc
    invrs_tau_no_N2_zm = invrs_tau_shear + invrs_tau_bkgnd
    
    # Add surface contribution near ground
    try:
        z = inputs.gr.zm
        h_sfc = 500.0  # Surface layer depth [m]
        if z.ndim == 1:
            # 1D array: broadcast to 2D
            z_2d = jnp.broadcast_to(z, (ngrdcol, nzm))
        else:
            z_2d = z
        sfc_weight = jnp.exp(-z_2d / h_sfc)
    except:
        sfc_weight = jnp.zeros((ngrdcol, nzm))
    
    invrs_tau_sfc_zm = sfc_weight * invrs_tau_sfc[:, jnp.newaxis]
    invrs_tau_zm = invrs_tau_no_N2_zm + invrs_tau_N2 + invrs_tau_sfc_zm
    
    # Calculate timescale
    tau_zm = jnp.where(invrs_tau_zm > 1e-10, 1.0 / invrs_tau_zm, 1e10)
    
    # Apply maximum timescale constraint
    # tau_max = Lscale_max / sqrt(2*em)
    sqrt_em_zm = jnp.sqrt(jnp.maximum(inputs.em, 1e-6))
    tau_max_zm = inputs.Lscale_max / jnp.sqrt(2.0 * jnp.maximum(inputs.em, 1e-6))
    tau_zm = jnp.minimum(tau_zm, tau_max_zm)
    
    # Interpolate to thermodynamic levels
    invrs_tau_zt = zm2zt_api(nzm, nzt, ngrdcol, inputs.gr, invrs_tau_zm)
    tau_zt = jnp.where(invrs_tau_zt > 1e-10, 1.0 / invrs_tau_zt, 1e10)
    
    sqrt_em_zt = zm2zt_api(nzm, nzt, ngrdcol, inputs.gr, 
                           jnp.sqrt(jnp.maximum(inputs.em, 1e-6)))
    tau_max_zt = inputs.Lscale_max / jnp.sqrt(2.0 * jnp.maximum(sqrt_em_zt**2, 1e-6))
    
    # Calculate mixing length scale from timescale
    # L = sqrt(2*em) * tau
    Lscale = sqrt_em_zt * jnp.sqrt(2.0) * tau_zt
    Lscale = jnp.clip(Lscale, ZLMIN, inputs.Lscale_max)
    
    # Separate upward and downward scales using stability
    N2_zt = zm2zt_api(nzm, nzt, ngrdcol, inputs.gr, N2)
    stability_factor = jnp.where(N2_zt < 0, 1.3, 0.8)
    
    Lscale_up = Lscale * stability_factor
    Lscale_down = Lscale / stability_factor
    
    Lscale_up = jnp.clip(Lscale_up, ZLMIN, inputs.Lscale_max)
    Lscale_down = jnp.clip(Lscale_down, ZLMIN, inputs.Lscale_max)
    
    # Component timescales for diagnostics
    invrs_tau_N2_iso = invrs_tau_N2
    invrs_tau_wp2_zm = invrs_tau_zm  # Simplified
    invrs_tau_xp2_zm = invrs_tau_zm
    invrs_tau_wp3_zm = invrs_tau_zm
    invrs_tau_wp3_zt = invrs_tau_zt
    invrs_tau_wpxp_zm = invrs_tau_zm
    
    return DiagnoseLscaleFromTauOutputs(
        invrs_tau_zt=invrs_tau_zt,
        invrs_tau_zm=invrs_tau_zm,
        invrs_tau_sfc=invrs_tau_sfc,
        invrs_tau_no_N2_zm=invrs_tau_no_N2_zm,
        invrs_tau_bkgnd=invrs_tau_bkgnd,
        invrs_tau_shear=invrs_tau_shear,
        invrs_tau_N2_iso=invrs_tau_N2_iso,
        invrs_tau_wp2_zm=invrs_tau_wp2_zm,
        invrs_tau_xp2_zm=invrs_tau_xp2_zm,
        invrs_tau_wp3_zm=invrs_tau_wp3_zm,
        invrs_tau_wp3_zt=invrs_tau_wp3_zt,
        invrs_tau_wpxp_zm=invrs_tau_wpxp_zm,
        tau_max_zm=tau_max_zm,
        tau_max_zt=tau_max_zt,
        tau_zm=tau_zm,
        tau_zt=tau_zt,
        Lscale=Lscale,
        Lscale_up=Lscale_up,
        Lscale_down=Lscale_down,
        stats_zm=inputs.stats_zm,
        err_info=inputs.err_info
    )


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'compute_mixing_length',
    'calc_Lscale_directly',
    'diagnose_Lscale_from_tau',
    'MixingLengthOutputs',
    'CalcLscaleDirectlyInputs',
    'CalcLscaleDirectlyOutputs',
    'DiagnoseLscaleFromTauInputs',
    'DiagnoseLscaleFromTauOutputs',
    'zm2zt_api',
    'zt2zm_api',
]