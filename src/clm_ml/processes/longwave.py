"""Longwave radiation: Norman (1979) radiative transfer through the canopy layers, with leaf
thermal emission.

One patch. Layers are 0-based slots on axis 0 (``clm_ml.state.PatchLayers``; leaves in the
leafy layers ``first_leafy .. n_within - 1``); levels are the ground (0) and the layers;
leaves 0 sunlit, 1 shaded. The scheme is chosen by name (``LONGWAVE_RADIATION``, model.yaml
``processes.radiation.longwave``): ``norman`` (Fortran ``longwave_type = 1``, the only one).

Fortran reference: CLM-ml v2 ``MLLongwaveRadiationMod.F90::LongwaveRadiation``, ``Norman``.
Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, chapter 14.

Reads and writes are the arguments and results; ``Reads``/``Writes`` in the docstrings name
the CLM-ml v2 (``mlcanopy_type``) fields they correspond to.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from clm_ml.processes.constants import emg, sb
from clm_ml.processes.math import tridiag
from clm_ml.state import PatchLayers


class LongwaveFluxes(NamedTuple):
    """Longwave fluxes (W/m2); profile values 0 outside the leafy layers."""

    lwupw: Array  # (level,) upward flux from the ground (0) and each layer
    lwdwn: Array  # (level,) downward flux onto the ground (0) and each layer
    lwleaf: Array  # (layer, leaf) absorbed per unit leaf area (same for sun and shade)
    lwsoi: Array  # () absorbed by the ground
    lwveg: Array  # () absorbed by the vegetation
    lwup: Array  # () upward flux above the canopy
    error: Array  # () energy conservation error (Fortran stops above 1e-3)


def norman(
    tleaf: Array,
    fracsun: Array,
    td: Array,
    dpai: Array,
    lwsky,
    tg,
    layers: PatchLayers,
    leaf_emissivity,
    unroll: bool = False,
) -> LongwaveFluxes:
    """Norman (1979) longwave radiative transfer for any layer counts.

    Leaves scatter ``omega = 1 - leaf_emissivity`` of the intercepted radiation, all of it
    reflected (``rho = omega``, ``tau = 0``); each layer emits from its sunlit and shaded
    leaves (``tleaf``, K). The upward and downward fluxes at the ground and every leafy layer
    are one tridiagonal system of ``2 * max_layers + 2`` rows: the real rows in Fortran order
    (ground, then two rows per leafy layer), the sky row right after them, then identity rows
    (x = 0) for the unused leafy slots. The sky row has no upper coupling, so the identity
    rows do not change the solution. ``unroll``: the solve's loops over rows written out
    (``math.tridiag``).

    Fortran reference: MLLongwaveRadiationMod.F90::Norman.
    Reads: ``tleaf_leaf``, ``fracsun_profile``, ``td_profile``, ``dpai_profile``,
    ``lwsky_forcing``, ``tg_soil``, the PFT's leaf emissivity. Writes: ``lwupw_profile``,
    ``lwdwn_profile``, ``lwleaf_leaf``, ``lwsoi_soil``, ``lwveg_canopy``, ``lwup_canopy``; the
    longwave check.
    """
    n_slots = dpai.shape[0]
    first, top = layers.first_leafy, layers.n_within - 1
    n_leafy = layers.n_within - layers.first_leafy
    rho = 1.0 - leaf_emissivity
    tau = 0.0

    # Leafy slot j = 0..max_layers-1 is layer first + j (valid for j < n_leafy); unused
    # slots repeat the top leafy layer's values (finite), and their results are discarded.
    j = jnp.arange(n_slots)
    valid = j < n_leafy
    ics = jnp.minimum(first + j, top)

    # Layer emission
    lw_sun = leaf_emissivity * sb * tleaf[ics, 0] ** 4
    lw_sha = leaf_emissivity * sb * tleaf[ics, 1] ** 4
    fs = fracsun[ics]
    td_v = td[ics]
    src = (lw_sun * fs + lw_sha * (1.0 - fs)) * (1.0 - td_v)

    # Row coefficients per leafy layer: "a" rows (-aic, 1, -bic) and "f" rows (-eic, 1, -fic)
    refld = (1.0 - td_v) * rho
    trand = (1.0 - td_v) * tau + td_v
    aic = refld - trand * trand / refld
    bic = trand / refld
    fic = refld - trand * trand / refld
    eic = trand / refld

    m_max = 2 * n_slots + 2
    sky_row = 2 * n_leafy + 2  # 1-based
    # Rows as arrays (0-based index = row - 1): ground row, then layer j's "a" row (row 2 + 2j)
    # and "f" row (row 3 + 2j); unused slots are identity rows (a = c = d = 0, b = 1)
    z = jnp.zeros(n_slots)
    atri = jnp.zeros(m_max).at[1 : 2 * n_slots + 1 : 2].set(jnp.where(valid, -aic, z))
    atri = atri.at[2 : 2 * n_slots + 2 : 2].set(jnp.where(valid, -eic, z))
    ctri = jnp.zeros(m_max).at[0].set(-(1.0 - emg))
    ctri = ctri.at[1 : 2 * n_slots + 1 : 2].set(jnp.where(valid, -bic, z))
    ctri = ctri.at[2 : 2 * n_slots + 2 : 2].set(jnp.where(valid, -fic, z))
    dtri = jnp.zeros(m_max).at[0].set(emg * sb * tg**4)
    dtri = dtri.at[1 : 2 * n_slots + 1 : 2].set(jnp.where(valid, (1.0 - bic) * src, z))
    dtri = dtri.at[2 : 2 * n_slots + 2 : 2].set(jnp.where(valid, (1.0 - eic) * src, z))
    btri = jnp.ones(m_max)
    # Sky row right after the last real row (overrides that slot's identity row)
    is_sky = jnp.arange(1, m_max + 1) == sky_row
    atri = jnp.where(is_sky, 0.0, atri)
    ctri = jnp.where(is_sky, 0.0, ctri)
    dtri = jnp.where(is_sky, lwsky, dtri)

    u = tridiag(atri, btri, ctri, dtri, unroll=unroll)

    # Unknowns: up(ground), dn(ground), then up, dn of each leafy layer (level = slot + 1)
    lev_out = jnp.where(valid, ics + 1, n_slots + 1)  # unused slots write a dropped entry
    lwupw = jnp.zeros(n_slots + 2).at[0].set(u[0]).at[lev_out].set(u[2::2][:n_slots])
    lwdwn = jnp.zeros(n_slots + 2).at[0].set(u[1]).at[lev_out].set(u[3::2][:n_slots])

    lwsoi = lwdwn[0] - lwupw[0]

    # Absorbed by each leafy layer: from the flux above it and the flux below it (the ground
    # for the lowest leafy layer)
    lev_below = jnp.where(ics == first, 0, ics)
    lwabs = leaf_emissivity * (lwdwn[ics + 1] + lwupw[lev_below]) * (1.0 - td_v) - 2.0 * src
    lwabs = jnp.where(valid, lwabs, 0.0)
    dpai_ics = jnp.where(valid, dpai[ics], 1.0)
    lwveg = jnp.sum(lwabs)
    lwleaf = jnp.zeros(n_slots + 2).at[lev_out].set(lwabs / dpai_ics)[1 : n_slots + 1]

    lwup = lwupw[top + 1]

    # Conservation check (Fortran stops; recorded and checked on the host)
    error = (lwsky - lwup) - (lwveg + lwsoi)

    return LongwaveFluxes(
        lwupw=lwupw[: n_slots + 1],
        lwdwn=lwdwn[: n_slots + 1],
        lwleaf=jnp.stack([lwleaf, lwleaf], axis=1),
        lwsoi=lwsoi,
        lwveg=lwveg,
        lwup=lwup,
        error=error,
    )


#: Longwave radiative transfer variants by name (model.yaml ``processes.radiation.longwave``).
LONGWAVE_RADIATION = {"norman": norman}
