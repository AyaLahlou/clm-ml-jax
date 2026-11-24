"""The soil columns at setup: layer structure, bedrock, hydraulic and thermal
properties, roots, soil colour albedos and the initial soil temperature, as 0-based arrays
(soil levels from the top down; one column per patch).

The arithmetic follows the Fortran operation by operation; ``tests/regression/test_setup.py``
checks the values bit for bit against a recorded setup.
"""

from __future__ import annotations

import numpy as np

from clm_ml.io import netcdf
from clm_ml.setup import tables

#: Hydrologically active and all ground levels of the CLM soil layer structure (CLM5's, the
#: only one supported). Fortran reference: clm_varpar.F90::clm_varpar_init (clm_phys CLM5_0)
NLEVSOI = 20
NLEVGRND = 25

ZMIN_BEDROCK = 0.4  # minimum depth to bedrock, m (clm_varcon)


def soil_layers() -> tuple:
    """``(z, dz, zi)`` of the CLM soil layer structure, 0-based: node depth and thickness of
    each level (``nlevgrnd``,), and the interface depths from the surface (``nlevgrnd + 1``,;
    ``zi[0] = 0``).

    Fortran reference: initVerticalMod.F90::initVertical (clm_phys CLM5_0)."""
    import jax.numpy as jnp

    nlevsoi, n = NLEVSOI, NLEVGRND
    z = jnp.zeros(n + 1, dtype=jnp.float64)  # 1-based work arrays, as in Fortran
    dz = jnp.zeros(n + 1, dtype=jnp.float64)
    zi = jnp.zeros(n + 1, dtype=jnp.float64)
    for j in range(1, 5):
        dz = dz.at[j].set(j * 0.02)
    for j in range(5, 14):
        dz = dz.at[j].set(dz[4] + (j - 4) * 0.04)
    for j in range(14, nlevsoi + 1):
        dz = dz.at[j].set(dz[13] + (j - 13) * 0.10)
    for j in range(nlevsoi + 1, n + 1):
        dz = dz.at[j].set(dz[nlevsoi] + (((j - nlevsoi) * 25.0) ** 1.5) / 100.0)
    zi = zi.at[0].set(0.0)
    for j in range(1, n + 1):
        zi = zi.at[j].set(float(jnp.sum(dz[1 : j + 1])))
    for j in range(1, n + 1):
        z = z.at[j].set(0.5 * (zi[j - 1] + zi[j]))
    return np.asarray(z[1:]), np.asarray(dz[1:]), np.asarray(zi)


def bedrock_index(zbed, zi, nlevsoi: int):
    """The 1-based index of the deepest soil level above bedrock of each column (``zbed``
    (C,) depth to bedrock, m, NaN: none; ``zi`` (nlevgrnd + 1,) interface depths).

    Fortran reference: initVerticalMod.F90::initVertical."""
    out = np.empty(len(zbed), dtype=int)
    for c in range(len(zbed)):
        jmin = 3
        for j in range(3, nlevsoi + 1):
            if zi[j - 1] < ZMIN_BEDROCK <= zi[j]:
                jmin = j
        nb = nlevsoi
        zb = float(zbed[c])
        for j in range(jmin, nlevsoi + 1):
            if zi[j - 1] < zb <= zi[j]:  # NaN: never
                nb = j
        out[c] = nb
    return out


def texture_index(texture: str) -> int:
    """The 1-based class of soil texture ``texture`` (``tables.SOIL_TEXTURES``)."""
    for m, name in enumerate(tables.SOIL_TEXTURES):
        if texture == name:
            return m + 1
    raise ValueError(
        f"soil texture class {texture!r} not found ({', '.join(tables.SOIL_TEXTURES)})"
    )


def soil_properties(sand, clay, organic, texture, z, nlevsoi: int):
    """The hydraulic and thermal properties of each column, ``{name: (C, nlevgrnd)}``
    (``watsat``, ``sucsat`` mm, ``hksat`` mm/s, ``bsw``, ``tkmg``, ``tkdry``, ``csol``), from
    its sand and clay (percent; NaN: from the texture class ``texture``) and organic matter
    (kg m-3); ``z`` (nlevgrnd,) node depths. A texture class's values are the row of its
    class in the tables (the class number is 1-based, the tables' rows 0-based).

    Fortran reference: SoilStateInitTimeConstMod.F90::SoilStateInitTimeConst."""
    csol_bedrock = 2.0e6  # clm_varcon
    organic_max, zsapric, pcalpha, pcbeta = 130.0, 0.5, 0.5, 0.139
    n = len(z)
    names = ("watsat", "sucsat", "hksat", "bsw", "tkmg", "tkdry", "csol")
    out = {k: np.empty((len(sand), n)) for k in names}
    for c in range(len(sand)):
        om_frac = float(organic[c]) / organic_max
        if not (np.isnan(sand[c]) or np.isnan(clay[c])):
            tex, cl, s = 0, float(clay[c]), float(sand[c])
        else:
            tex = texture_index(texture[c])  # 1-based, as Fortran; the tables' row is tex - 1
            cl = float(tables.CLAY_TEX[tex - 1]) * 100.0
            s = float(tables.SAND_TEX[tex - 1]) * 100.0
        for j in range(1, n + 1):
            zj = float(z[j - 1])
            if zj > 0.5:
                om_frac = 0.0
            if tex == 0:
                watsat = 0.489 - 0.00126 * s
                sucsat = 10.0 * (10.0 ** (1.88 - 0.0131 * s))
                hksat = 0.0070556 * (10.0 ** (-0.884 + 0.0153 * s))
                bsw = 2.91 + 0.159 * cl
            else:
                watsat = float(tables.WATSAT_TEX[tex - 1])
                sucsat = -float(tables.SMPSAT_TEX[tex - 1])
                hksat = float(tables.HKSAT_TEX[tex - 1]) / 60.0
                bsw = float(tables.BSW_TEX[tex - 1])
            z_ratio = zj / zsapric
            om_watsat = max(0.93 - 0.1 * z_ratio, 0.83)
            om_sucsat = min(10.3 - 0.2 * z_ratio, 10.1)
            om_hksat = max(0.28 - 0.2799 * z_ratio, hksat)
            om_b = min(2.7 + 9.3 * z_ratio, 12.0)
            watsat = (1.0 - om_frac) * watsat + om_watsat * om_frac
            sucsat = (1.0 - om_frac) * sucsat + om_sucsat * om_frac
            bsw = (1.0 - om_frac) * bsw + om_frac * om_b
            if om_frac > pcalpha:
                perc_norm = (1.0 - pcalpha) ** (-pcbeta)
                perc_frac = perc_norm * (om_frac - pcalpha) ** pcbeta
            else:
                perc_frac = 0.0
            uncon_frac = (1.0 - om_frac) + (1.0 - perc_frac) * om_frac
            if om_frac < 1.0:
                uncon_hksat = uncon_frac / (
                    (1.0 - om_frac) / hksat + ((1.0 - perc_frac) * om_frac) / om_hksat
                )
            else:
                uncon_hksat = 0.0
            hksat = uncon_frac * uncon_hksat + (perc_frac * om_frac) * om_hksat
            om_frac_therm = om_frac
            if j <= nlevsoi:
                bulk = 2700.0 * (1.0 - watsat)
                tkdry_min = (0.135 * bulk + 64.7) / (2700.0 - 0.947 * bulk)
                tkdry = (1.0 - om_frac_therm) * tkdry_min + om_frac_therm * 0.05
                tksol_min = (8.80 * s + 2.92 * cl) / (s + cl)
                tkm = (1.0 - om_frac_therm) * tksol_min + om_frac_therm * 0.25
                tkmg = tkm ** (1.0 - watsat)
            else:
                bulk = 2700.0
                tkdry = (0.135 * bulk + 64.7) / (2700.0 - 0.947 * bulk)
                tkmg = 3.0
            cvsol = ((2.128 * s + 2.385 * cl) / (s + cl)) * 1.0e6 if tex == 0 else 1.926e6
            if j <= nlevsoi:
                csol = (1.0 - om_frac_therm) * cvsol + om_frac_therm * 2.5e6
            else:
                csol = csol_bedrock
            for k, v in zip(names, (watsat, sucsat, hksat, bsw, tkmg, tkdry, csol)):
                out[k][c, j - 1] = v
    return out


def root_fractions(beta, zi, nbedrock, nlevsoi: int):
    """The Jackson et al. (1996) root profile of each patch, ``(P, nlevsoi)``: ``beta`` (P,)
    its PFT's ``root_profile_beta``, ``zi`` (nlevgrnd + 1,) the interface depths; roots below
    bedrock moved evenly to the levels above it.

    Fortran reference: SoilStateInitTimeConstMod.F90::SoilStateInitTimeConst."""
    m_to_cm = 1.0e2
    out = np.zeros((len(beta), nlevsoi))
    for p in range(len(beta)):
        b = float(beta[p])
        if not 0.0 < b < 1.0:
            raise ValueError(f"invalid root profile beta {b} (patch {p})")
        r = [0.0] * (nlevsoi + 1)  # 1-based
        for j in range(1, nlevsoi + 1):
            r[j] = b ** (float(zi[j - 1]) * m_to_cm) - b ** (float(zi[j]) * m_to_cm)
        nb = int(nbedrock[p])
        surplus = sum(float(r[j]) for j in range(nb + 1, nlevsoi + 1))
        for j in range(1, nb + 1):
            r[j] = float(r[j]) + surplus / float(nb)
        for j in range(nb + 1, nlevsoi + 1):
            r[j] = 0.0
        out[p] = r[1:]
    return out


def soil_albedos(colour) -> tuple:
    """``(albsat, albdry)``, each (C, band): the saturated and dry soil albedo of each
    column's soil colour class (1-20).

    Fortran reference: SurfaceAlbedoMod.F90::SurfaceAlbedoInitTimeConst."""
    k = np.asarray(colour, dtype=int) - 1
    if (k < 0).any() or (k >= len(tables.ALBSAT[0])).any():
        raise ValueError("soil colour classes must be 1-20")
    sat = np.stack([np.asarray(b)[k] for b in tables.ALBSAT], axis=-1)
    dry = np.stack([np.asarray(b)[k] for b in tables.ALBDRY], axis=-1)
    return sat, dry


def initial_soil_temperature(path, time_index: int, nlevgrnd: int) -> np.ndarray:
    """The soil temperature (nlevgrnd,) of a CLM history file (``TSOI``) at 1-based time slice
    ``time_index``.

    Fortran reference: CLMml_driver.F90::SoilInit."""

    with netcdf.dataset(path) as ds:
        if "TSOI" not in ds.variables:
            raise ValueError(f"{path}: no TSOI")
        var = ds.variables["TSOI"]
        if var.shape[1] != nlevgrnd:
            raise ValueError(f"{path}: TSOI has {var.shape[1]} levels; the run needs {nlevgrnd}")
        if not 1 <= time_index <= var.shape[0]:
            raise ValueError(f"{path}: time slice {time_index} outside 1..{var.shape[0]}")
        t = var[time_index - 1, :nlevgrnd, 0]
    return np.array([float(v) for v in t])
