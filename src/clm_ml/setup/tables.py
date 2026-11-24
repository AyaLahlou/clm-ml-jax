"""Data tables of the setup: soil texture classes, soil colour albedos, PFT
default canopy heights and leaf/stem area profiles.

Values copied from the CLM-ml v2 sources.
"""

from __future__ import annotations

#: Soil texture classes (Cosby et al. 1984; Clapp and Hornberger 1978)
#: Fortran reference: SoilTexMod.F90
SOIL_TEXTURES = (
    "sand", "loamy sand", "sandy loam", "silty loam", "loam", "sandy clay loam",
    "silty clay loam", "clay loam", "sandy clay", "silty clay", "clay",
)  # fmt: skip
SAND_TEX = (0.92, 0.82, 0.58, 0.17, 0.43, 0.58, 0.10, 0.32, 0.52, 0.06, 0.22)
CLAY_TEX = (0.03, 0.06, 0.10, 0.13, 0.18, 0.27, 0.34, 0.34, 0.42, 0.47, 0.58)
WATSAT_TEX = (0.395, 0.410, 0.435, 0.485, 0.451, 0.420, 0.477, 0.476, 0.426, 0.492, 0.482)
SMPSAT_TEX = (-121.0, -90.0, -218.0, -786.0, -478.0, -299.0, -356.0, -630.0, -153.0, -490.0,
              -405.0)  # fmt: skip
HKSAT_TEX = (10.560, 9.380, 2.080, 0.432, 0.417, 0.378, 0.102, 0.147, 0.130, 0.062, 0.077)
BSW_TEX = (4.05, 4.38, 4.90, 5.30, 5.39, 7.12, 7.75, 8.52, 10.40, 10.40, 11.40)

#: Saturated and dry soil albedo of the 20 CLM5 soil colour classes, (visible, near-infrared)
#: Fortran reference: SurfaceAlbedoMod.F90::SurfaceAlbedoInitTimeConst
ALBSAT = (
    (0.25, 0.23, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09,
     0.08, 0.07, 0.06, 0.05, 0.04),
    (0.50, 0.46, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18,
     0.16, 0.14, 0.12, 0.10, 0.08),
)  # fmt: skip
ALBDRY = (
    (0.36, 0.34, 0.32, 0.31, 0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.20, 0.18,
     0.16, 0.14, 0.12, 0.10, 0.08),
    (0.61, 0.57, 0.53, 0.51, 0.49, 0.48, 0.45, 0.43, 0.41, 0.39, 0.37, 0.35, 0.33, 0.31, 0.29,
     0.27, 0.25, 0.23, 0.21, 0.16),
)  # fmt: skip


def default_canopy_height(pft: int) -> float:
    """CLM's canopy top height of PFT ``pft`` (m): the canopy height of a patch without one.

    Fortran reference: CLMml_driver.F90 (``htop_pft``, data statement)."""
    if 1 <= pft <= 3:
        return (17.0, 17.0, 14.0)[pft - 1]  # needleleaf trees
    if 4 <= pft <= 8:
        return (35.0, 35.0, 18.0, 20.0, 20.0)[pft - 4]  # broadleaf trees
    if 9 <= pft <= 16:
        return 0.5  # shrubs, grasses, crops
    return 0.0  # not vegetated, other crops


def default_area_profile(pft: int) -> tuple:
    """The PFT default beta-distribution parameters (p, q) of the leaf and of the stem area
    profile (the same for both); (-999, -999) for PFTs without one.

    Fortran reference: MLinitVerticalMod.F90::getPADparameters."""
    if pft == 1:
        return (11.5, 3.5)
    if 2 <= pft <= 11:
        return (3.5, 2.0)
    if 12 <= pft <= 16:
        return (2.5, 2.5)
    return (-999.0, -999.0)
