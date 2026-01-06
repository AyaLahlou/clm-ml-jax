"""
Longwave Radiation Transfer Through Canopy.

Translated from CTSM's MLLongwaveRadiationMod.F90

This module calculates longwave radiation transfer through a multi-layer
canopy using radiative transfer theory. The main approach follows Norman's
two-stream approximation for thermal radiation.

Key physics:
    - Longwave radiation emitted by leaves, ground, and sky
    - Absorption and emission by canopy layers
    - Multiple scattering between layers
    - Temperature-dependent emission (Stefan-Boltzmann law)

The Norman (1979) method treats the canopy as a series of horizontal layers,
each with specified leaf area and temperature. Longwave radiation is exchanged
between layers and with the ground surface, accounting for emission and
absorption by leaves.

Key equations:
    Leaf scattering coefficient: omega = 1 - emleaf
    Emitted radiation: sigma * emleaf * T^4
    Layer source: weighted by sunlit fraction and interception (1 - td)
    
    Tridiagonal system for upward/downward fluxes:
        refld = (1 - td) * rho  (diffuse reflectance)
        trand = (1 - td) * tau + td  (diffuse transmittance)
        
    Conservation: absorbed = incoming - outgoing

Reference:
    Norman, J.M. (1979). Modeling the complete crop canopy.
    In: Modification of the Aerial Environment of Crops.
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp


# ============================================================================
# Type Definitions
# ============================================================================

class BoundsType(NamedTuple):
    """Spatial domain bounds.
    
    Attributes:
        begp: Beginning patch index
        endp: Ending patch index
    """
    begp: int
    endp: int


class MLCanopyType(NamedTuple):
    """Multi-layer canopy state and fluxes.
    
    Attributes:
        ncan: Number of aboveground layers [-] [n_patches]
        ntop: Index for top leaf layer (0-based) [-] [n_patches]
        nbot: Index for bottom leaf layer (0-based) [-] [n_patches]
        tleaf_sun: Sunlit leaf temperature [K] [n_patches, nlevcan]
        tleaf_sha: Shaded leaf temperature [K] [n_patches, nlevcan]
        fracsun: Sunlit fraction of canopy layer [-] [n_patches, nlevcan]
        td: Canopy layer transmittance of diffuse radiation [-] [n_patches, nlevcan]
        dpai: Layer plant area index [m2/m2] [n_patches, nlevcan]
        tg: Ground temperature [K] [n_patches]
        lwsky: Atmospheric longwave down [W/m2] [n_patches]
        itype: PFT type index [n_patches]
        lwup_layer: Upward longwave flux [W/m2] [n_patches, nlevcan+1]
        lwdn_layer: Downward longwave flux [W/m2] [n_patches, nlevcan+1]
        lwleaf_sun: Sunlit leaf absorbed longwave [W/m2 leaf] [n_patches, nlevcan]
        lwleaf_sha: Shaded leaf absorbed longwave [W/m2 leaf] [n_patches, nlevcan]
        lwsoi: Absorbed longwave by soil [W/m2] [n_patches]
        lwveg: Absorbed longwave by vegetation [W/m2] [n_patches]
        lwup: Upward longwave at canopy top [W/m2] [n_patches]
    """
    ncan: jnp.ndarray
    ntop: jnp.ndarray
    nbot: jnp.ndarray
    tleaf_sun: jnp.ndarray
    tleaf_sha: jnp.ndarray
    fracsun: jnp.ndarray
    td: jnp.ndarray
    dpai: jnp.ndarray
    tg: jnp.ndarray
    lwsky: jnp.ndarray
    itype: jnp.ndarray
    lwup_layer: jnp.ndarray
    lwdn_layer: jnp.ndarray
    lwleaf_sun: jnp.ndarray
    lwleaf_sha: jnp.ndarray
    lwsoi: jnp.ndarray
    lwveg: jnp.ndarray
    lwup: jnp.ndarray


class LongwaveRadiationParams(NamedTuple):
    """Parameters for longwave radiation calculations.
    
    Attributes:
        sb: Stefan-Boltzmann constant [W/m2/K4]
        emg: Ground (soil) emissivity [-]
        emleaf: Leaf emissivity by PFT [-] [n_pfts]
        nlevcan: Maximum number of canopy layers
    """
    sb: float
    emg: float
    emleaf: jnp.ndarray
    nlevcan: int


class LongwaveInitState(NamedTuple):
    """State after initialization of longwave radiation arrays.
    
    Attributes:
        lwup_layer: Upward longwave radiation by layer [W/m2] [n_patches, nlevcan+1]
        lwdn_layer: Downward longwave radiation by layer [W/m2] [n_patches, nlevcan+1]
        lwleaf_sun: Sunlit leaf absorbed longwave [W/m2 leaf] [n_patches, nlevcan]
        lwleaf_sha: Shaded leaf absorbed longwave [W/m2 leaf] [n_patches, nlevcan]
        lw_source: Emitted longwave by layer [W/m2] [n_patches, nlevcan]
        omega: Leaf scattering coefficient [-] [n_patches]
        rho: Reflection coefficient [-] [n_patches]
        tau: Transmission coefficient [-] [n_patches]
    """
    lwup_layer: jnp.ndarray
    lwdn_layer: jnp.ndarray
    lwleaf_sun: jnp.ndarray
    lwleaf_sha: jnp.ndarray
    lw_source: jnp.ndarray
    omega: jnp.ndarray
    rho: jnp.ndarray
    tau: jnp.ndarray


class TridiagonalCoefficients(NamedTuple):
    """Coefficients for tridiagonal system of longwave flux equations.
    
    The system is: atri * x[i-1] + btri * x[i] + ctri * x[i+1] = dtri
    where x alternates between upward and downward fluxes.
    
    Attributes:
        atri: Lower diagonal coefficients [n_patches, n_equations]
        btri: Main diagonal coefficients [n_patches, n_equations]
        ctri: Upper diagonal coefficients [n_patches, n_equations]
        dtri: Right-hand side [n_patches, n_equations]
        n_equations: Number of equations per patch
    """
    atri: jnp.ndarray
    btri: jnp.ndarray
    ctri: jnp.ndarray
    dtri: jnp.ndarray
    n_equations: int


class NormanPart3Output(NamedTuple):
    """Output from Norman method part 3.
    
    Attributes:
        lwup_layer: Upward longwave flux above each layer [W/m2] [n_patches, n_layers+1]
        lwdn_layer: Downward longwave flux onto each layer [W/m2] [n_patches, n_layers+1]
        lwsoi: Absorbed longwave radiation by soil [W/m2] [n_patches]
        lwveg: Absorbed longwave radiation by vegetation (partial) [W/m2] [n_patches]
    """
    lwup_layer: jnp.ndarray
    lwdn_layer: jnp.ndarray
    lwsoi: jnp.ndarray
    lwveg: jnp.ndarray


class NormanPart4Output(NamedTuple):
    """Output from Norman longwave radiation part 4.
    
    Attributes:
        lwleaf_sun: Absorbed longwave by sunlit leaves [W/m2 leaf] [n_patches, n_layers]
        lwleaf_shade: Absorbed longwave by shaded leaves [W/m2 leaf] [n_patches, n_layers]
        lwveg: Total absorbed longwave by vegetation [W/m2 ground] [n_patches]
        lwup: Upward longwave at canopy top [W/m2] [n_patches]
    """
    lwleaf_sun: jnp.ndarray
    lwleaf_shade: jnp.ndarray
    lwveg: jnp.ndarray
    lwup: jnp.ndarray


# ============================================================================
# Helper Functions
# ============================================================================

def tridiag_solve(
    atri: jnp.ndarray,
    btri: jnp.ndarray,
    ctri: jnp.ndarray,
    dtri: jnp.ndarray,
    n: int,
) -> jnp.ndarray:
    """Solve tridiagonal system of equations.
    
    Solves: atri[i] * x[i-1] + btri[i] * x[i] + ctri[i] * x[i+1] = dtri[i]
    for equations indexed from 0 to n (inclusive), using 0-based indexing.
    
    Args:
        atri: Lower diagonal [n_patches, max_equations]
        btri: Main diagonal [n_patches, max_equations]
        ctri: Upper diagonal [n_patches, max_equations]
        dtri: Right-hand side [n_patches, max_equations]
        n: Index of last equation (0-based). System has equations 0, 1, 2, ..., n
           for a total of (n+1) equations.
        
    Returns:
        Solution vector [n_patches, max_equations]
        
    Note:
        Arrays are sized to hold max_equations but only indices 0 through n are used.
        The Thomas algorithm is used: forward elimination followed by back substitution.
    """
    n_patches = atri.shape[0]
    
    # Forward elimination: Process equations 1 through n
    # Eliminate lower diagonal by modifying main diagonal and RHS
    def forward_step(i, carry):
        btri_mod, dtri_mod = carry
        
        # Modify diagonal and RHS to eliminate atri[i] coefficient
        denom = btri_mod[:, i] - atri[:, i] * ctri[:, i-1] / btri_mod[:, i-1]
        btri_new = btri_mod.at[:, i].set(denom)
        
        dtri_new = dtri_mod.at[:, i].set(
            (dtri_mod[:, i] - atri[:, i] * dtri_mod[:, i-1] / btri_mod[:, i-1])
        )
        
        return (btri_new, dtri_new)
    
    # Process equations at indices 1, 2, ..., n
    btri_mod, dtri_mod = jax.lax.fori_loop(
        1, n + 1, forward_step, (btri, dtri)
    )
    
    # Back substitution: Solve for x from equation n down to equation 0
    x = jnp.zeros_like(dtri)
    # Start with last equation (index n): btri[n] * x[n] = dtri[n]
    x = x.at[:, n].set(dtri_mod[:, n] / btri_mod[:, n])
    
    def backward_step(k, x_carry):
        # Map k ∈ [0, n) to equation indices in descending order: n-1, n-2, ..., 0
        # When k=0: i=n-1, when k=n-1: i=0
        i = n - 1 - k
        # Solve: btri[i] * x[i] + ctri[i] * x[i+1] = dtri[i]
        # where x[i+1] is already known from previous iteration
        x_new = x_carry.at[:, i].set(
            (dtri_mod[:, i] - ctri[:, i] * x_carry[:, i+1]) / btri_mod[:, i]
        )
        return x_new
    
    # Process equations at indices n-1, n-2, ..., 0
    x = jax.lax.fori_loop(0, n, backward_step, x)
    
    return x


# ============================================================================
# Core Physics Functions
# ============================================================================

def initialize_longwave_arrays_and_sources(
    emleaf: jnp.ndarray,
    ncan: jnp.ndarray,
    ntop: jnp.ndarray,
    nbot: jnp.ndarray,
    tleaf_sun: jnp.ndarray,
    tleaf_sha: jnp.ndarray,
    fracsun: jnp.ndarray,
    td: jnp.ndarray,
    sb: float,
    nlevcan: int,
) -> LongwaveInitState:
    """Initialize longwave radiation arrays and calculate emission source terms.
    
    This function performs the initialization phase of the Norman longwave
    radiation calculation (Fortran lines 103-152). It:
    1. Zeros out all radiative flux arrays
    2. Calculates leaf scattering coefficient (omega)
    3. Sets reflection (rho) and transmission (tau) coefficients
    4. Computes emitted longwave radiation from each canopy layer
    
    Args:
        emleaf: Leaf emissivity [-] [n_patches]
        ncan: Number of aboveground layers [-] [n_patches]
        ntop: Index for top leaf layer (0-based) [-] [n_patches]
        nbot: Index for bottom leaf layer (0-based) [-] [n_patches]
        tleaf_sun: Sunlit leaf temperature [K] [n_patches, nlevcan]
        tleaf_sha: Shaded leaf temperature [K] [n_patches, nlevcan]
        fracsun: Sunlit fraction of canopy layer [-] [n_patches, nlevcan]
        td: Canopy layer transmittance of diffuse radiation [-] [n_patches, nlevcan]
        sb: Stefan-Boltzmann constant [W/m2/K4]
        nlevcan: Maximum number of canopy layers
        
    Returns:
        LongwaveInitState containing initialized arrays and source terms
        
    Note:
        Fortran lines 103-152
        - Arrays are initialized to zero for all layers
        - Leaf scattering coefficient omega = 1 - emleaf
        - Intercepted radiation is reflected: rho = omega, tau = 0
        - Emitted radiation weighted by sunlit/shaded fractions
    """
    n_patches = emleaf.shape[0]
    
    # Initialize all radiative flux arrays to zero (lines 123-132)
    lwup_layer = jnp.zeros((n_patches, nlevcan + 1))
    lwdn_layer = jnp.zeros((n_patches, nlevcan + 1))
    lwleaf_sun = jnp.zeros((n_patches, nlevcan))
    lwleaf_sha = jnp.zeros((n_patches, nlevcan))
    
    # Leaf scattering coefficient (line 136)
    omega = 1.0 - emleaf
    
    # Terms for longwave radiation reflected and transmitted by a layer (lines 140-141)
    rho = omega
    tau = jnp.zeros_like(omega)
    
    # Calculate emitted longwave radiation for each layer (lines 145-150)
    lw_source_sun = emleaf[:, None] * sb * tleaf_sun**4
    lw_source_sha = emleaf[:, None] * sb * tleaf_sha**4
    
    # Weight by sunlit fraction and interception factor (1 - td)
    lw_source = (
        (lw_source_sun * fracsun + lw_source_sha * (1.0 - fracsun))
        * (1.0 - td)
    )
    
    # Only compute for layers between nbot and ntop (inclusive)
    layer_indices = jnp.arange(nlevcan)
    active_mask = (
        (layer_indices[None, :] >= nbot[:, None]) &
        (layer_indices[None, :] <= ntop[:, None])
    )
    lw_source = jnp.where(active_mask, lw_source, 0.0)
    
    return LongwaveInitState(
        lwup_layer=lwup_layer,
        lwdn_layer=lwdn_layer,
        lwleaf_sun=lwleaf_sun,
        lwleaf_sha=lwleaf_sha,
        lw_source=lw_source,
        omega=omega,
        rho=rho,
        tau=tau,
    )


def setup_tridiagonal_system(
    td: jnp.ndarray,
    rho: jnp.ndarray,
    tau: jnp.ndarray,
    emg: float,
    sb: float,
    tg: jnp.ndarray,
    lw_source: jnp.ndarray,
    nbot: jnp.ndarray,
    ntop: jnp.ndarray,
    nlevcan: int,
) -> TridiagonalCoefficients:
    """Set up tridiagonal system for longwave radiation transfer.
    
    Constructs the coefficient matrices for the tridiagonal system that
    solves for upward and downward longwave fluxes through the canopy.
    
    Args:
        td: Exponential transmittance of diffuse radiation [n_patches, n_layers]
        rho: Leaf reflectance for diffuse radiation [n_patches]
        tau: Leaf transmittance for diffuse radiation [n_patches]
        emg: Ground (soil) emissivity [scalar]
        sb: Stefan-Boltzmann constant [W/m2/K4]
        tg: Ground temperature [K] [n_patches]
        lw_source: Longwave emission source term for each layer [W/m2] [n_patches, n_layers]
        nbot: Index of bottom canopy layer (0-based) [n_patches]
        ntop: Index of top canopy layer (0-based) [n_patches]
        nlevcan: Maximum number of canopy layers
        
    Returns:
        TridiagonalCoefficients containing the system matrices
        
    Note:
        Fortran lines 153-212
        - The system has 2 equations per layer plus 2 for soil
        - Equation indices alternate: upward flux (odd m), downward flux (even m)
    """
    n_patches = td.shape[0]
    
    # Calculate maximum number of equations needed
    max_n_leaf_layers = nlevcan
    max_n_equations = 2 * max_n_leaf_layers + 2
    
    # Initialize coefficient arrays
    atri = jnp.zeros((n_patches, max_n_equations))
    btri = jnp.zeros((n_patches, max_n_equations))
    ctri = jnp.zeros((n_patches, max_n_equations))
    dtri = jnp.zeros((n_patches, max_n_equations))
    
    # Soil: upward flux (m=0, Fortran lines 161-165)
    m = 0
    atri = atri.at[:, m].set(0.0)
    btri = btri.at[:, m].set(1.0)
    ctri = ctri.at[:, m].set(-(1.0 - emg))
    dtri = dtri.at[:, m].set(emg * sb * tg**4)
    
    # Soil: downward flux (m=1, Fortran lines 167-176)
    m = 1
    # Use proper per-patch indexing: select td[i, nbot[i]] for each patch i
    patch_indices = jnp.arange(n_patches)
    td_at_nbot = td[patch_indices, nbot]
    refld = (1.0 - td_at_nbot) * rho
    trand = (1.0 - td_at_nbot) * tau + td_at_nbot
    aic = refld - trand * trand / refld
    bic = trand / refld
    
    atri = atri.at[:, m].set(-aic)
    btri = btri.at[:, m].set(1.0)
    ctri = ctri.at[:, m].set(-bic)
    lw_source_at_nbot = lw_source[patch_indices, nbot]
    dtri = dtri.at[:, m].set((1.0 - bic) * lw_source_at_nbot)
    
    # Leaf layers, excluding top layer (Fortran lines 178-210)
    def process_layer(ic, carry):
        atri_c, btri_c, ctri_c, dtri_c, m_c = carry
        
        # Upward flux (lines 182-191)
        refld_up = (1.0 - td[:, ic]) * rho
        trand_up = (1.0 - td[:, ic]) * tau + td[:, ic]
        fic = refld_up - trand_up * trand_up / refld_up
        eic = trand_up / refld_up
        
        atri_c = atri_c.at[:, m_c].set(-eic)
        btri_c = btri_c.at[:, m_c].set(1.0)
        ctri_c = ctri_c.at[:, m_c].set(-fic)
        dtri_c = dtri_c.at[:, m_c].set((1.0 - eic) * lw_source[:, ic])
        m_c += 1
        
        # Downward flux (lines 193-202)
        refld_dn = (1.0 - td[:, ic + 1]) * rho
        trand_dn = (1.0 - td[:, ic + 1]) * tau + td[:, ic + 1]
        aic_dn = refld_dn - trand_dn * trand_dn / refld_dn
        bic_dn = trand_dn / refld_dn
        
        atri_c = atri_c.at[:, m_c].set(-aic_dn)
        btri_c = btri_c.at[:, m_c].set(1.0)
        ctri_c = ctri_c.at[:, m_c].set(-bic_dn)
        dtri_c = dtri_c.at[:, m_c].set((1.0 - bic_dn) * lw_source[:, ic + 1])
        m_c += 1
        
        return (atri_c, btri_c, ctri_c, dtri_c, m_c)
    
    # Process layers from nbot to ntop-1
    m = 2
    for ic in range(nlevcan - 1):
        # Only process if layer is active
        is_active = (ic >= nbot) & (ic < ntop)
        
        if jnp.any(is_active):
            atri, btri, ctri, dtri, m = process_layer(
                ic, (atri, btri, ctri, dtri, m)
            )
    
    n_equations = m
    
    return TridiagonalCoefficients(
        atri=atri,
        btri=btri,
        ctri=ctri,
        dtri=dtri,
        n_equations=n_equations,
    )


def solve_tridiagonal_and_assign_fluxes(
    atri: jnp.ndarray,
    btri: jnp.ndarray,
    ctri: jnp.ndarray,
    dtri: jnp.ndarray,
    m: int,
    td: jnp.ndarray,
    rho: jnp.ndarray,
    tau: jnp.ndarray,
    lw_source: jnp.ndarray,
    lwsky: jnp.ndarray,
    nbot: jnp.ndarray,
    ntop: jnp.ndarray,
    nlevcan: int,
) -> NormanPart3Output:
    """Complete tridiagonal setup, solve system, and assign fluxes.
    
    This function completes the Norman (1979) longwave radiation calculation by:
    1. Adding top canopy layer equations to tridiagonal system
    2. Solving the tridiagonal system
    3. Assigning solution to flux arrays
    4. Calculating absorbed radiation
    
    Args:
        atri: Lower diagonal of tridiagonal matrix [n_patches, max_eqs]
        btri: Main diagonal of tridiagonal matrix [n_patches, max_eqs]
        ctri: Upper diagonal of tridiagonal matrix [n_patches, max_eqs]
        dtri: Right-hand side vector [n_patches, max_eqs]
        m: Current equation index (before top layer) [scalar]
        td: Diffuse transmittance [n_patches, n_layers]
        rho: Leaf reflectance for longwave [n_patches]
        tau: Leaf transmittance for longwave [n_patches]
        lw_source: Longwave source term for each layer [n_patches, n_layers]
        lwsky: Downward atmospheric longwave radiation [W/m2] [n_patches]
        nbot: Bottom canopy layer index [n_patches]
        ntop: Top canopy layer index [n_patches]
        nlevcan: Number of canopy layers [scalar]
        
    Returns:
        NormanPart3Output containing flux arrays and absorbed radiation
        
    Reference:
        Fortran lines 213-281
    """
    n_patches = atri.shape[0]
    batch_idx = jnp.arange(n_patches)
    
    # Top canopy layer: upward flux (lines 217-223)
    td_top = td[batch_idx, ntop]
    lw_source_top = lw_source[batch_idx, ntop]
    
    refld = (1.0 - td_top) * rho
    trand = (1.0 - td_top) * tau + td_top
    fic = refld - trand * trand / refld
    eic = trand / refld
    
    # Add upward flux equation
    m_up = m
    atri = atri.at[batch_idx, m_up].set(-eic)
    btri = btri.at[batch_idx, m_up].set(1.0)
    ctri = ctri.at[batch_idx, m_up].set(-fic)
    dtri = dtri.at[batch_idx, m_up].set((1.0 - eic) * lw_source_top)
    
    # Top canopy layer: downward flux (lines 230-234)
    m_dn = m_up + 1
    atri = atri.at[batch_idx, m_dn].set(0.0)
    btri = btri.at[batch_idx, m_dn].set(1.0)
    ctri = ctri.at[batch_idx, m_dn].set(0.0)
    dtri = dtri.at[batch_idx, m_dn].set(lwsky)
    
    # Solve tridiagonal system (line 236)
    utri = tridiag_solve(atri, btri, ctri, dtri, m_dn)
    
    # Initialize output arrays
    lwup_layer = jnp.zeros((n_patches, nlevcan + 1))
    lwdn_layer = jnp.zeros((n_patches, nlevcan + 1))
    
    # Soil fluxes (lines 247-250)
    lwup_layer = lwup_layer.at[:, 0].set(utri[:, 0])
    lwdn_layer = lwdn_layer.at[:, 0].set(utri[:, 1])
    
    # Leaf layer fluxes (lines 252-256)
    for ic in range(nlevcan):
        m_layer_up = 2 + 2 * ic
        m_layer_dn = 3 + 2 * ic
        
        layer_active = (ic >= nbot) & (ic <= ntop)
        
        lwup_layer = lwup_layer.at[:, ic + 1].set(
            jnp.where(layer_active, utri[:, m_layer_up], 0.0)
        )
        lwdn_layer = lwdn_layer.at[:, ic + 1].set(
            jnp.where(layer_active, utri[:, m_layer_dn], 0.0)
        )
    
    # Absorbed longwave radiation for ground (line 260)
    lwsoi = lwdn_layer[:, 0] - lwup_layer[:, 0]
    
    # Initialize vegetation absorbed radiation
    lwveg = jnp.zeros(n_patches)
    
    return NormanPart3Output(
        lwup_layer=lwup_layer,
        lwdn_layer=lwdn_layer,
        lwsoi=lwsoi,
        lwveg=lwveg,
    )


def calculate_final_absorption(
    lwdn_layer: jnp.ndarray,
    lwup_layer: jnp.ndarray,
    lw_source: jnp.ndarray,
    td: jnp.ndarray,
    dpai: jnp.ndarray,
    emleaf: jnp.ndarray,
    lwsoi: jnp.ndarray,
    lwsky: jnp.ndarray,
    ntop: jnp.ndarray,
    nbot: jnp.ndarray,
    itype: jnp.ndarray,
    nlevcan: int,
) -> NormanPart4Output:
    """Calculate final longwave absorption and perform conservation check.
    
    This function completes the Norman (1979) longwave radiation transfer
    calculation by computing absorbed radiation in each canopy layer and
    verifying energy conservation.
    
    Args:
        lwdn_layer: Downward longwave at layer top [W/m2] [n_patches, n_layers+1]
        lwup_layer: Upward longwave at layer bottom [W/m2] [n_patches, n_layers+1]
        lw_source: Longwave source term [W/m2] [n_patches, n_layers]
        td: Diffuse transmittance [0-1] [n_patches, n_layers]
        dpai: Layer plant area index [m2/m2] [n_patches, n_layers]
        emleaf: Leaf emissivity [0-1] [n_pfts]
        lwsoi: Absorbed longwave by soil [W/m2] [n_patches]
        lwsky: Incoming longwave from sky [W/m2] [n_patches]
        ntop: Top canopy layer index [n_patches]
        nbot: Bottom canopy layer index [n_patches]
        itype: PFT type index [n_patches]
        nlevcan: Maximum number of canopy layers
        
    Returns:
        NormanPart4Output containing absorbed radiation and upward flux
        
    Note:
        Fortran lines 282-310
        The conservation check ensures absorbed = incoming - outgoing
    """
    n_patches = lwdn_layer.shape[0]
    batch_idx = jnp.arange(n_patches)
    
    # Initialize output arrays
    lwleaf_sun = jnp.zeros((n_patches, nlevcan))
    lwleaf_shade = jnp.zeros((n_patches, nlevcan))
    lwveg = jnp.zeros(n_patches)
    
    # Process each canopy layer (lines 282-295)
    for ic in range(nlevcan - 1, -1, -1):  # Top to bottom
        # Get emissivity for each patch's PFT type
        em = emleaf[itype]
        
        # Determine layer below
        icm1 = jnp.where(ic == nbot, ic, ic - 1)
        
        # Get upward longwave from layer below
        lwup_below = lwup_layer[batch_idx, icm1]
        
        # Calculate absorbed longwave (lines 285-286)
        lwabs = (
            em * (lwdn_layer[:, ic] + lwup_below) * (1.0 - td[:, ic])
            - 2.0 * lw_source[:, ic]
        )
        
        # Absorbed per unit leaf area (lines 287-288)
        lwleaf_val = jnp.where(
            dpai[:, ic] > 0.0,
            lwabs / dpai[:, ic],
            0.0
        )
        
        # Only update if layer is active
        is_active = (ntop >= ic) & (nbot <= ic)
        
        lwleaf_sun = lwleaf_sun.at[:, ic].set(
            jnp.where(is_active, lwleaf_val, 0.0)
        )
        lwleaf_shade = lwleaf_shade.at[:, ic].set(
            jnp.where(is_active, lwleaf_val, 0.0)
        )
        
        # Accumulate total vegetation absorption (line 292)
        lwveg = jnp.where(is_active, lwveg + lwabs, lwveg)
    
    # Canopy emitted longwave radiation (line 297)
    lwup = lwup_layer[batch_idx, ntop]
    
    # Conservation check (lines 300-304)
    sumabs = lwsky - lwup
    err = sumabs - (lwveg + lwsoi)
    max_err = jnp.max(jnp.abs(err))
    
    # Print warning if conservation error is large
    jax.lax.cond(
        max_err > 1.e-3,
        lambda: jax.debug.print(
            "WARNING: Norman longwave conservation error: {}", max_err
        ),
        lambda: None,
    )
    
    return NormanPart4Output(
        lwleaf_sun=lwleaf_sun,
        lwleaf_shade=lwleaf_shade,
        lwveg=lwveg,
        lwup=lwup,
    )


def norman_longwave(
    bounds: BoundsType,
    num_filter: int,
    filter_indices: jnp.ndarray,
    mlcanopy_inst: MLCanopyType,
    params: LongwaveRadiationParams,
) -> MLCanopyType:
    """Calculate longwave radiation transfer through canopy using Norman (1979).
    
    This is the main entry point for the Norman longwave radiation scheme.
    It orchestrates the calculation of:
    1. Leaf emissivities and absorptivities
    2. Tridiagonal matrix coefficients for radiation transfer
    3. Solution of the tridiagonal system for upward/downward fluxes
    4. Net radiation at each canopy layer and ground surface
    
    Args:
        bounds: Spatial domain bounds (begp, endp for patches)
        num_filter: Number of patches to process [scalar]
        filter_indices: Indices of patches to process [n_filter]
        mlcanopy_inst: Multi-layer canopy state
        params: Longwave radiation parameters
        
    Returns:
        Updated mlcanopy_inst with calculated longwave fluxes
        
    Reference:
        Fortran lines 53-310
    """
    # Extract parameters
    sb = params.sb
    emg = params.emg
    emleaf = params.emleaf
    nlevcan = params.nlevcan
    
    # Get emissivity for each patch based on PFT type
    emleaf_patch = emleaf[mlcanopy_inst.itype]
    
    # Part 1: Initialize arrays and calculate source terms (lines 103-152)
    init_state = initialize_longwave_arrays_and_sources(
        emleaf=emleaf_patch,
        ncan=mlcanopy_inst.ncan,
        ntop=mlcanopy_inst.ntop,
        nbot=mlcanopy_inst.nbot,
        tleaf_sun=mlcanopy_inst.tleaf_sun,
        tleaf_sha=mlcanopy_inst.tleaf_sha,
        fracsun=mlcanopy_inst.fracsun,
        td=mlcanopy_inst.td,
        sb=sb,
        nlevcan=nlevcan,
    )
    
    # Part 2: Set up tridiagonal system (lines 153-212)
    tridiag_coeffs = setup_tridiagonal_system(
        td=mlcanopy_inst.td,
        rho=init_state.rho,
        tau=init_state.tau,
        emg=emg,
        sb=sb,
        tg=mlcanopy_inst.tg,
        lw_source=init_state.lw_source,
        nbot=mlcanopy_inst.nbot,
        ntop=mlcanopy_inst.ntop,
        nlevcan=nlevcan,
    )
    
    # Part 3: Solve system and assign fluxes (lines 213-281)
    part3_output = solve_tridiagonal_and_assign_fluxes(
        atri=tridiag_coeffs.atri,
        btri=tridiag_coeffs.btri,
        ctri=tridiag_coeffs.ctri,
        dtri=tridiag_coeffs.dtri,
        m=tridiag_coeffs.n_equations - 2,  # Before top layer equations
        td=mlcanopy_inst.td,
        rho=init_state.rho,
        tau=init_state.tau,
        lw_source=init_state.lw_source,
        lwsky=mlcanopy_inst.lwsky,
        nbot=mlcanopy_inst.nbot,
        ntop=mlcanopy_inst.ntop,
        nlevcan=nlevcan,
    )
    
    # Part 4: Calculate final absorption (lines 282-310)
    part4_output = calculate_final_absorption(
        lwdn_layer=part3_output.lwdn_layer,
        lwup_layer=part3_output.lwup_layer,
        lw_source=init_state.lw_source,
        td=mlcanopy_inst.td,
        dpai=mlcanopy_inst.dpai,
        emleaf=emleaf,
        lwsoi=part3_output.lwsoi,
        lwsky=mlcanopy_inst.lwsky,
        ntop=mlcanopy_inst.ntop,
        nbot=mlcanopy_inst.nbot,
        itype=mlcanopy_inst.itype,
        nlevcan=nlevcan,
    )
    
    # Update mlcanopy_inst with results
    mlcanopy_inst = mlcanopy_inst._replace(
        lwup_layer=part3_output.lwup_layer,
        lwdn_layer=part3_output.lwdn_layer,
        lwleaf_sun=part4_output.lwleaf_sun,
        lwleaf_sha=part4_output.lwleaf_shade,
        lwsoi=part3_output.lwsoi,
        lwveg=part4_output.lwveg,
        lwup=part4_output.lwup,
    )
    
    return mlcanopy_inst


def longwave_radiation(
    bounds: BoundsType,
    num_filter: int,
    filter_indices: jnp.ndarray,
    mlcanopy_inst: MLCanopyType,
    params: LongwaveRadiationParams,
    longwave_type: int = 1,
) -> MLCanopyType:
    """Calculate longwave radiation transfer through canopy.
    
    Dispatcher function that calls the appropriate longwave radiation
    scheme based on configuration. Currently only supports Norman (1979)
    two-stream approximation.
    
    Args:
        bounds: Bounds defining spatial domain
        num_filter: Number of patches in filter [scalar]
        filter_indices: Patch indices to process [n_patches]
        mlcanopy_inst: Multi-layer canopy state and fluxes
        params: Longwave radiation parameters
        longwave_type: Longwave radiation scheme selector [scalar]
            1 = Norman (1979) two-stream approximation
            
    Returns:
        Updated mlcanopy_inst with longwave radiation fluxes
        
    Raises:
        ValueError: If longwave_type is not valid (not equal to 1)
        
    Note:
        Fortran lines 27-50
    """
    # Validate longwave_type
    if longwave_type != 1:
        raise ValueError(
            f"ERROR: LongwaveRadiation: longwave_type={longwave_type} not valid. "
            f"Only longwave_type=1 (Norman) is currently supported."
        )
    
    # Dispatch to Norman scheme
    mlcanopy_inst = norman_longwave(
        bounds=bounds,
        num_filter=num_filter,
        filter_indices=filter_indices,
        mlcanopy_inst=mlcanopy_inst,
        params=params,
    )
    
    return mlcanopy_inst