"""
PFT constants module for CLM.

This module contains vegetation constants and methods to read and initialize
vegetation (Plant Functional Type) constants for CLM physiological and
biophysical calculations.

Translation of CLM-ml_v1/clm_src_main/pftconMod.F90 to Python/JAX.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Dict, Any, List
import logging
from dataclasses import dataclass, field
import numpy as np

# Import related modules
try:
    from .clm_varpar import mxpft, numrad, ivis, inir
except ImportError:
    # Provide fallback values
    mxpft = 78
    numrad = 2
    ivis = 0  # Visible radiation band index
    inir = 1  # Near-infrared radiation band index

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class pftcon_type:
    """
    PFT constants data type.
    
    This class contains all vegetation constants and parameters for CLM
    Plant Functional Types, including physiological, biophysical, and
    optical properties.
    
    Attributes:
        # CLM pft parameters
        dleaf: Characteristic leaf dimension (m)
        c3psn: Photosynthetic pathway: 0. = C4, 1. = C3
        xl: Leaf/stem orientation index
        rhol: Leaf reflectance: [pft, band] where band=0(vis), 1(nir)
        rhos: Stem reflectance: [pft, band]
        taul: Leaf transmittance: [pft, band]
        taus: Stem transmittance: [pft, band]
        roota_par: Zeng2001 rooting distribution parameter (1/m)
        rootb_par: Zeng2001 rooting distribution parameter (1/m)
        rootprof_beta: Jackson1996 rooting distribution parameter (-)
        slatop: Specific leaf area at top of canopy (m2/gC)
        
        # pft parameters for CLMml
        vcmaxpft: Maximum carboxylation rate at 25C (umol/m2/s)
        gplant_SPA: Stem (xylem-to-leaf) hydraulic conductance (mmol H2O/m2 leaf area/s/Mpa)
        capac_SPA: Plant capacitance (mmol H2O/m2 leaf area/MPa)
        iota_SPA: Stomatal water-use efficiency (umol CO2/ mol H2O)
        root_radius_SPA: Fine root radius (m)
        root_density_SPA: Fine root density (g biomass / m3 root)
        root_resist_SPA: Hydraulic resistivity of root tissue (MPa.s.g/mmol H2O)
        gsmin_SPA: Minimum stomatal conductance (mol H2O/m2/s)
        g0_BB: Ball-Berry minimum leaf conductance (mol H2O/m2/s)
        g1_BB: Ball-Berry slope of conductance-photosynthesis relationship
        g0_MED: Medlyn minimum leaf conductance (mol H2O/m2/s)
        g1_MED: Medlyn slope of conductance-photosynthesis relationship
        psi50_gs: Leaf water potential at which 50% of stomatal conductance is lost (MPa)
        shape_gs: Shape parameter for stomatal conductance in relation to leaf water potential (-)
        emleaf: Leaf emissivity (-)
        clump_fac: Foliage clumping index (-)
        pbeta_lai: Parameter for the leaf area density beta distribution (-)
        qbeta_lai: Parameter for the leaf area density beta distribution (-)
        pbeta_sai: Parameter for the stem area density beta distribution (-)
        qbeta_sai: Parameter for the stem area density beta distribution (-)
        
        # Metadata
        is_initialized: Whether the constants have been initialized
        metadata: Additional metadata and configuration info
    """
    # CLM pft parameters
    dleaf: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    c3psn: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    xl: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    rhol: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    rhos: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    taul: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    taus: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    roota_par: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    rootb_par: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    rootprof_beta: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    slatop: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    
    # pft parameters for CLMml
    vcmaxpft: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    gplant_SPA: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    capac_SPA: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    iota_SPA: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    root_radius_SPA: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    root_density_SPA: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    root_resist_SPA: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    gsmin_SPA: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    g0_BB: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    g1_BB: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    g0_MED: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    g1_MED: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    psi50_gs: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    shape_gs: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    emleaf: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    clump_fac: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    pbeta_lai: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    qbeta_lai: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    pbeta_sai: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    qbeta_sai: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    
    # Metadata
    is_initialized: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def Init(self) -> None:
        """
        Initialize PFT constants.
        
        This method allocates memory and reads/initializes all vegetation
        constants following the original Fortran implementation.
        """
        try:
            logger.info("Initializing PFT constants")
            
            self.InitAllocate()
            self.InitRead()
            
            self.is_initialized = True
            self.metadata['initialization_complete'] = True
            
            logger.info("PFT constants initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PFT constants: {e}")
            raise ValueError(f"PFT constants initialization failed: {e}") from e
    
    def InitAllocate(self) -> None:
        """
        Allocate memory for PFT data structure.
        
        Creates JAX arrays for all PFT parameters with proper dimensions.
        """
        try:
            logger.debug("Allocating memory for PFT data structure")
            
            # Calculate array size (0 to mxpft inclusive)
            array_size = mxpft + 1
            
            # Initialize with -999 (missing value indicator)
            missing_val = -999.0
            
            # CLM pft parameters
            self.dleaf = jnp.full(array_size, missing_val)
            self.c3psn = jnp.full(array_size, missing_val)
            self.xl = jnp.full(array_size, missing_val)
            self.rhol = jnp.full((array_size, numrad), missing_val)
            self.rhos = jnp.full((array_size, numrad), missing_val)
            self.taul = jnp.full((array_size, numrad), missing_val)
            self.taus = jnp.full((array_size, numrad), missing_val)
            self.roota_par = jnp.full(array_size, missing_val)
            self.rootb_par = jnp.full(array_size, missing_val)
            self.rootprof_beta = jnp.full(array_size, missing_val)
            self.slatop = jnp.full(array_size, missing_val)
            
            # pft parameters for CLMml
            self.vcmaxpft = jnp.full(array_size, missing_val)
            self.gplant_SPA = jnp.full(array_size, missing_val)
            self.capac_SPA = jnp.full(array_size, missing_val)
            self.iota_SPA = jnp.full(array_size, missing_val)
            self.root_radius_SPA = jnp.full(array_size, missing_val)
            self.root_density_SPA = jnp.full(array_size, missing_val)
            self.root_resist_SPA = jnp.full(array_size, missing_val)
            self.gsmin_SPA = jnp.full(array_size, missing_val)
            self.g0_BB = jnp.full(array_size, missing_val)
            self.g1_BB = jnp.full(array_size, missing_val)
            self.g0_MED = jnp.full(array_size, missing_val)
            self.g1_MED = jnp.full(array_size, missing_val)
            self.psi50_gs = jnp.full(array_size, missing_val)
            self.shape_gs = jnp.full(array_size, missing_val)
            self.emleaf = jnp.full(array_size, missing_val)
            self.clump_fac = jnp.full(array_size, missing_val)
            self.pbeta_lai = jnp.full(array_size, missing_val)
            self.qbeta_lai = jnp.full(array_size, missing_val)
            self.pbeta_sai = jnp.full(array_size, missing_val)
            self.qbeta_sai = jnp.full(array_size, missing_val)
            
            logger.debug(f"Allocated PFT arrays for {array_size} PFTs with {numrad} radiation bands")
            
        except Exception as e:
            logger.error(f"Failed to allocate PFT data structure: {e}")
            raise ValueError(f"PFT allocation failed: {e}") from e
    
    def InitRead(self) -> None:
        """
        Read and initialize vegetation (PFT) constants.
        
        This method sets all PFT parameter values following the exact
        values from the original Fortran implementation.
        """
        try:
            logger.debug("Reading and initializing PFT constants")
            
            # Leaf dimension (m)
            self.dleaf = self.dleaf.at[1:17].set(0.04)
            
            # Photosynthetic pathway: 1. = C3 plant and 0. = C4 plant
            self.c3psn = self.c3psn.at[1:14].set(1.0)
            self.c3psn = self.c3psn.at[14].set(0.0)  # C4 grass
            self.c3psn = self.c3psn.at[15:17].set(1.0)
            
            # Leaf angle
            self.xl = self.xl.at[1:4].set(0.01)
            self.xl = self.xl.at[4:6].set(0.10)
            self.xl = self.xl.at[6].set(0.01)
            self.xl = self.xl.at[7:9].set(0.25)
            self.xl = self.xl.at[9].set(0.01)
            self.xl = self.xl.at[10:12].set(0.25)
            self.xl = self.xl.at[12:17].set(-0.30)
            
            # Leaf reflectance: visible and near-infrared
            self.rhol = self.rhol.at[1:4, ivis].set(0.07)
            self.rhol = self.rhol.at[4:9, ivis].set(0.10)
            self.rhol = self.rhol.at[9, ivis].set(0.07)
            self.rhol = self.rhol.at[10:12, ivis].set(0.10)
            self.rhol = self.rhol.at[12:17, ivis].set(0.11)
            
            self.rhol = self.rhol.at[1:4, inir].set(0.35)
            self.rhol = self.rhol.at[4:9, inir].set(0.45)
            self.rhol = self.rhol.at[9, inir].set(0.35)
            self.rhol = self.rhol.at[10:12, inir].set(0.45)
            self.rhol = self.rhol.at[12:17, inir].set(0.35)
            
            # Stem reflectance: visible and near-infrared
            self.rhos = self.rhos.at[1:12, ivis].set(0.16)
            self.rhos = self.rhos.at[12:17, ivis].set(0.31)
            
            self.rhos = self.rhos.at[1:12, inir].set(0.39)
            self.rhos = self.rhos.at[12:17, inir].set(0.53)
            
            # Leaf transmittance: visible and near-infrared
            self.taul = self.taul.at[1:17, ivis].set(0.05)
            
            self.taul = self.taul.at[1:4, inir].set(0.10)
            self.taul = self.taul.at[4:9, inir].set(0.25)
            self.taul = self.taul.at[9, inir].set(0.10)
            self.taul = self.taul.at[10:12, inir].set(0.25)
            self.taul = self.taul.at[12:17, inir].set(0.34)
            
            # Stem transmittance: visible and near-infrared
            self.taus = self.taus.at[1:12, ivis].set(0.001)
            self.taus = self.taus.at[12:17, ivis].set(0.12)
            
            self.taus = self.taus.at[1:12, inir].set(0.001)
            self.taus = self.taus.at[12:17, inir].set(0.25)
            
            # Zeng2001 rooting distribution parameters (1/m)
            self.roota_par = self.roota_par.at[1:6].set(7.0)
            self.roota_par = self.roota_par.at[6:9].set(6.0)
            self.roota_par = self.roota_par.at[9:12].set(7.0)
            self.roota_par = self.roota_par.at[12:15].set(11.0)
            self.roota_par = self.roota_par.at[15:17].set(6.0)
            
            self.rootb_par = self.rootb_par.at[1:4].set(2.0)
            self.rootb_par = self.rootb_par.at[4:6].set(1.0)
            self.rootb_par = self.rootb_par.at[6:9].set(2.0)
            self.rootb_par = self.rootb_par.at[9:12].set(1.5)
            self.rootb_par = self.rootb_par.at[12:15].set(2.0)
            self.rootb_par = self.rootb_par.at[15:17].set(3.0)
            
            # Jackson1996 rooting distribution parameters (-)
            self.rootprof_beta = self.rootprof_beta.at[1].set(0.976)
            self.rootprof_beta = self.rootprof_beta.at[2:4].set(0.943)
            self.rootprof_beta = self.rootprof_beta.at[4].set(0.993)
            self.rootprof_beta = self.rootprof_beta.at[5].set(0.966)
            self.rootprof_beta = self.rootprof_beta.at[6].set(0.993)
            self.rootprof_beta = self.rootprof_beta.at[7].set(0.966)
            self.rootprof_beta = self.rootprof_beta.at[8].set(0.943)
            self.rootprof_beta = self.rootprof_beta.at[9:11].set(0.964)
            self.rootprof_beta = self.rootprof_beta.at[11:13].set(0.914)
            self.rootprof_beta = self.rootprof_beta.at[13:17].set(0.943)
            
            # Specific leaf area at top of canopy (m2/gC)
            sla_values = jnp.array([0.010, 0.008, 0.024, 0.012, 0.012, 0.030, 0.030, 0.030,
                                  0.012, 0.030, 0.030, 0.030, 0.030, 0.030, 0.030, 0.030])
            self.slatop = self.slatop.at[1:17].set(sla_values)
            
            # ========================
            # pft parameters for CLMml
            # ========================
            
            # vcmax (umol/m2/s)
            vcmax_values = jnp.array([62.5, 62.5, 39.1, 41.0, 61.4, 41.0, 57.7, 57.7,
                                    61.7, 54.0, 54.0, 78.2, 78.2, 51.6, 100.7, 100.7])
            self.vcmaxpft = self.vcmaxpft.at[1:17].set(vcmax_values)
            
            # Plant hydraulics
            self.gplant_SPA = self.gplant_SPA.at[1:17].set(4.0)
            
            self.capac_SPA = self.capac_SPA.at[1:12].set(2500.0)
            self.capac_SPA = self.capac_SPA.at[12:17].set(500.0)
            
            # Stomatal optimization
            self.iota_SPA = self.iota_SPA.at[1].set(750.0)
            self.iota_SPA = self.iota_SPA.at[2:4].set(1500.0)
            self.iota_SPA = self.iota_SPA.at[4].set(500.0)
            self.iota_SPA = self.iota_SPA.at[5:17].set(750.0)
            
            # Root hydraulics
            self.root_radius_SPA = self.root_radius_SPA.at[1:17].set(0.29e-3)
            self.root_density_SPA = self.root_density_SPA.at[1:17].set(0.31e6)
            self.root_resist_SPA = self.root_resist_SPA.at[1:17].set(25.0)
            
            # Minimum stomatal conductance
            self.gsmin_SPA = self.gsmin_SPA.at[1:17].set(0.002)
            
            # Ball-Berry stomatal conductance parameters
            self.g0_BB = self.g0_BB.at[1:14].set(0.01)
            self.g0_BB = self.g0_BB.at[14].set(0.04)
            self.g0_BB = self.g0_BB.at[15:17].set(0.01)
            
            self.g1_BB = self.g1_BB.at[1:14].set(9.0)
            self.g1_BB = self.g1_BB.at[14].set(4.0)
            self.g1_BB = self.g1_BB.at[15:17].set(9.0)
            
            # Medlyn stomatal conductance parameters
            self.g0_MED = self.g0_MED.at[1:17].set(0.0001)
            
            g1_med_values = jnp.array([2.35, 2.35, 2.35, 4.12, 4.12, 4.45, 4.45, 4.45,
                                     4.70, 4.70, 4.70, 2.22, 5.25, 1.62, 5.79, 5.79])
            self.g1_MED = self.g1_MED.at[1:17].set(g1_med_values)
            
            # Leaf water potential at which 50% of stomatal conductance is lost
            self.psi50_gs = self.psi50_gs.at[1:17].set(-2.3)
            
            # Shape parameter for stomatal conductance
            self.shape_gs = self.shape_gs.at[1:17].set(40.0)
            
            # Leaf emissivity
            self.emleaf = self.emleaf.at[1:17].set(0.98)
            
            # Foliage clumping index
            self.clump_fac = self.clump_fac.at[1:17].set(1.0)
            
            # Parameters for leaf/stem area density beta distribution
            # PFT 1 (needleleaf evergreen temperate)
            self.pbeta_lai = self.pbeta_lai.at[1].set(11.5)
            self.qbeta_lai = self.qbeta_lai.at[1].set(3.5)
            
            # PFTs 2-3 (needleleaf evergreen/deciduous boreal)
            self.pbeta_lai = self.pbeta_lai.at[2:4].set(3.5)
            self.qbeta_lai = self.qbeta_lai.at[2:4].set(2.0)
            
            # PFTs 4-5 (broadleaf evergreen)
            self.pbeta_lai = self.pbeta_lai.at[4:6].set(3.5)
            self.qbeta_lai = self.qbeta_lai.at[4:6].set(2.0)
            
            # PFTs 6-8 (broadleaf deciduous)
            self.pbeta_lai = self.pbeta_lai.at[6:9].set(3.5)
            self.qbeta_lai = self.qbeta_lai.at[6:9].set(2.0)
            
            # PFTs 9-11 (shrubs)
            self.pbeta_lai = self.pbeta_lai.at[9:12].set(3.5)
            self.qbeta_lai = self.qbeta_lai.at[9:12].set(2.0)
            
            # PFTs 12-16 (grasses and crops)
            self.pbeta_lai = self.pbeta_lai.at[12:17].set(2.5)
            self.qbeta_lai = self.qbeta_lai.at[12:17].set(2.5)
            
            # Stem area parameters same as leaf area
            self.pbeta_sai = self.pbeta_lai
            self.qbeta_sai = self.qbeta_lai
            
            logger.debug("PFT constants read and initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to read PFT constants: {e}")
            raise ValueError(f"PFT constants reading failed: {e}") from e
    
    def is_valid(self) -> bool:
        """Check if PFT constants are properly initialized."""
        if not self.is_initialized:
            return False
        
        # Check that arrays have expected size
        expected_size = mxpft + 1
        
        # Check 1D arrays
        arrays_1d = [self.dleaf, self.c3psn, self.xl, self.slatop, self.vcmaxpft]
        for arr in arrays_1d:
            if len(arr) != expected_size:
                return False
        
        # Check 2D arrays
        arrays_2d = [self.rhol, self.rhos, self.taul, self.taus]
        for arr in arrays_2d:
            if arr.shape != (expected_size, numrad):
                return False
        
        return True
    
    def get_pft_parameters(self, pft_idx: int) -> Dict[str, Any]:
        """
        Get all parameters for a specific PFT.
        
        Args:
            pft_idx: PFT index (0-78)
            
        Returns:
            Dictionary with all PFT parameters
            
        Raises:
            ValueError: If PFT index is invalid
        """
        if pft_idx < 0 or pft_idx > mxpft:
            raise ValueError(f"Invalid PFT index: {pft_idx}. Must be 0-{mxpft}")
        
        return {
            # Basic properties
            'pft_index': pft_idx,
            'dleaf': float(self.dleaf[pft_idx]),
            'c3psn': float(self.c3psn[pft_idx]),
            'xl': float(self.xl[pft_idx]),
            
            # Optical properties
            'rhol_vis': float(self.rhol[pft_idx, ivis]),
            'rhol_nir': float(self.rhol[pft_idx, inir]),
            'rhos_vis': float(self.rhos[pft_idx, ivis]),
            'rhos_nir': float(self.rhos[pft_idx, inir]),
            'taul_vis': float(self.taul[pft_idx, ivis]),
            'taul_nir': float(self.taul[pft_idx, inir]),
            'taus_vis': float(self.taus[pft_idx, ivis]),
            'taus_nir': float(self.taus[pft_idx, inir]),
            
            # Root properties
            'roota_par': float(self.roota_par[pft_idx]),
            'rootb_par': float(self.rootb_par[pft_idx]),
            'rootprof_beta': float(self.rootprof_beta[pft_idx]),
            
            # Leaf properties
            'slatop': float(self.slatop[pft_idx]),
            'emleaf': float(self.emleaf[pft_idx]),
            
            # Physiological properties
            'vcmaxpft': float(self.vcmaxpft[pft_idx]),
            'gsmin_SPA': float(self.gsmin_SPA[pft_idx]),
            'g0_BB': float(self.g0_BB[pft_idx]),
            'g1_BB': float(self.g1_BB[pft_idx]),
            'g0_MED': float(self.g0_MED[pft_idx]),
            'g1_MED': float(self.g1_MED[pft_idx]),
            
            # Plant hydraulics
            'gplant_SPA': float(self.gplant_SPA[pft_idx]),
            'capac_SPA': float(self.capac_SPA[pft_idx]),
            'root_radius_SPA': float(self.root_radius_SPA[pft_idx]),
            'root_density_SPA': float(self.root_density_SPA[pft_idx]),
            'root_resist_SPA': float(self.root_resist_SPA[pft_idx]),
            
            # Stress response
            'psi50_gs': float(self.psi50_gs[pft_idx]),
            'shape_gs': float(self.shape_gs[pft_idx]),
            
            # Canopy structure
            'clump_fac': float(self.clump_fac[pft_idx]),
            'pbeta_lai': float(self.pbeta_lai[pft_idx]),
            'qbeta_lai': float(self.qbeta_lai[pft_idx]),
            'pbeta_sai': float(self.pbeta_sai[pft_idx]),
            'qbeta_sai': float(self.qbeta_sai[pft_idx]),
        }


# Global pftcon instance (for Fortran compatibility)
pftcon = pftcon_type()


# Utility functions
@jax.jit
def get_photosynthesis_pathway(pft_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Get photosynthesis pathway for PFT indices.
    
    Args:
        pft_indices: Array of PFT indices
        
    Returns:
        Array of photosynthesis pathways (0=C4, 1=C3)
    """
    global pftcon
    return pftcon.c3psn[pft_indices]


@jax.jit
def get_leaf_reflectance(pft_indices: jnp.ndarray, band: int) -> jnp.ndarray:
    """
    Get leaf reflectance for PFT indices and radiation band.
    
    Args:
        pft_indices: Array of PFT indices
        band: Radiation band (0=visible, 1=near-infrared)
        
    Returns:
        Array of leaf reflectance values
    """
    global pftcon
    return pftcon.rhol[pft_indices, band]


@jax.jit
def get_vcmax(pft_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Get maximum carboxylation rate for PFT indices.
    
    Args:
        pft_indices: Array of PFT indices
        
    Returns:
        Array of Vcmax values (umol/m2/s)
    """
    global pftcon
    return pftcon.vcmaxpft[pft_indices]


def reset_pftcon() -> None:
    """Reset the global pftcon instance."""
    global pftcon
    pftcon = pftcon_type()
    logger.info("Global pftcon instance reset")


def validate_pftcon() -> Tuple[bool, str]:
    """
    Validate PFT constants structure and values.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    global pftcon
    
    if not pftcon.is_valid():
        return False, "PFT constants basic validation failed"
    
    # Check for reasonable parameter ranges
    if not jnp.all((pftcon.c3psn == 0) | (pftcon.c3psn == 1) | (pftcon.c3psn == -999)):
        return False, "Invalid c3psn values (must be 0, 1, or -999)"
    
    # Check reflectance values are between 0 and 1 (where not missing)
    valid_rhol = (pftcon.rhol >= 0) & (pftcon.rhol <= 1) | (pftcon.rhol == -999)
    if not jnp.all(valid_rhol):
        return False, "Invalid leaf reflectance values (must be 0-1 or -999)"
    
    return True, "PFT constants are valid"


def print_pftcon_summary() -> None:
    """Print a summary of PFT constants."""
    global pftcon
    
    print("\n=== PFT Constants Summary ===")
    print(f"Initialized: {pftcon.is_initialized}")
    print(f"Valid: {pftcon.is_valid()}")
    print(f"Max PFT index: {mxpft}")
    print(f"Radiation bands: {numrad}")
    
    if pftcon.is_initialized:
        # Show some example PFT parameters
        example_pfts = [1, 4, 9, 12, 14, 15]  # Representative PFTs
        print(f"\nExample PFT Parameters:")
        for pft in example_pfts:
            if pft <= mxpft:
                c3 = "C3" if pftcon.c3psn[pft] == 1 else "C4" if pftcon.c3psn[pft] == 0 else "N/A"
                vcmax = pftcon.vcmaxpft[pft]
                print(f"  PFT {pft}: {c3}, Vcmax={vcmax:.1f}, dleaf={pftcon.dleaf[pft]:.3f}")
    
    if pftcon.metadata:
        print(f"\nMetadata:")
        for key, value in pftcon.metadata.items():
            print(f"  {key}: {value}")
    
    print("=" * 30)


def create_pftcon_subset(pft_indices: List[int]) -> Dict[str, jnp.ndarray]:
    """
    Create a subset of PFT constants for specified PFTs.
    
    Args:
        pft_indices: List of PFT indices to include
        
    Returns:
        Dictionary with PFT constant arrays for the subset
    """
    global pftcon
    
    if not pftcon.is_initialized:
        raise ValueError("PFT constants must be initialized first")
    
    subset = {}
    
    # 1D parameters
    for param_name in ['dleaf', 'c3psn', 'xl', 'slatop', 'vcmaxpft', 'gsmin_SPA']:
        param_array = getattr(pftcon, param_name)
        subset[param_name] = param_array[pft_indices]
    
    # 2D parameters
    for param_name in ['rhol', 'rhos', 'taul', 'taus']:
        param_array = getattr(pftcon, param_name)
        subset[param_name] = param_array[pft_indices, :]
    
    return subset


# Export interface
__all__ = [
    'pftcon_type',
    'pftcon',  # Global instance
    'get_photosynthesis_pathway',
    'get_leaf_reflectance', 
    'get_vcmax',
    'reset_pftcon',
    'validate_pftcon',
    'print_pftcon_summary',
    'create_pftcon_subset'
]