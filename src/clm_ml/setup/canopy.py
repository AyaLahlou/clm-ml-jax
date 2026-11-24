"""The canopy at setup: leaf and stem area profile parameters, the layer grid
and the leaf/stem area fractions of the layers, and the initial canopy-air and leaf state,
as 0-based arrays (layer slots ``0 .. max_layers - 1``; slots above the top layer in use
hold ``spval``).

The arithmetic follows the Fortran operation by operation; ``tests/regression/test_setup.py``
checks the values bit for bit against a recorded setup.
"""

from __future__ import annotations

import numpy as np

from clm_ml.setup import tables

SPVAL = 1.0e36  # clm_varcon.spval: the fill value of slots not in use

# ---------------------------------------------------------------------------------------
# Beta distribution (clm_ml.processes.math), vectorized
# ---------------------------------------------------------------------------------------


def _log_gamma(x):
    """``clm_ml.processes.math.log_gamma_function`` on arrays (same operations)."""
    import jax.numpy as jnp

    coef = (
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    )
    x = np.asarray(x, dtype=float)
    y = x.copy()
    tmp = x + 5.5
    tmp = np.asarray((x + 0.5) * jnp.log(tmp) - tmp)
    ser = np.full_like(x, 1.000000000190015)
    for c in coef:
        y = y + 1.0
        ser = ser + c / y
    return np.asarray(tmp + jnp.log(2.5066282746310005 * ser / x))


def _beta_cf(a, b, x):
    """``clm_ml.processes.math._beta_function_incomplete_cf`` on arrays: each element stops at its
    own iteration (modified Lentz, maxit 100, eps 3e-7, fpmin 1e-30)."""
    a, b, x = (np.asarray(v, dtype=float) for v in np.broadcast_arrays(a, b, x))
    fpmin, eps = 1.0e-30, 3.0e-7
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = np.ones_like(x)
    d = 1.0 - qab * x / qap
    d = np.where(np.abs(d) < fpmin, fpmin, d)
    d = 1.0 / d
    h = d.copy()
    done = np.zeros(x.shape, dtype=bool)
    for m in range(1, 101):
        m2 = 2 * m
        aa = float(m) * (b - float(m)) * x / ((qam + float(m2)) * (a + float(m2)))
        d1 = 1.0 + aa * d
        d1 = np.where(np.abs(d1) < fpmin, fpmin, d1)
        c1 = 1.0 + aa / c
        c1 = np.where(np.abs(c1) < fpmin, fpmin, c1)
        d1 = 1.0 / d1
        h1 = h * d1 * c1
        aa = -(a + float(m)) * (qab + float(m)) * x / ((qap + float(m2)) * (a + float(m2)))
        d2 = 1.0 + aa * d1
        d2 = np.where(np.abs(d2) < fpmin, fpmin, d2)
        c2 = 1.0 + aa / c1
        c2 = np.where(np.abs(c2) < fpmin, fpmin, c2)
        d2 = 1.0 / d2
        delta = d2 * c2
        h2 = h1 * delta
        h, c, d = np.where(done, h, h2), np.where(done, c, c2), np.where(done, d, d2)
        done = done | (np.abs(delta - 1.0) < eps)
        if done.all():
            return h
    raise RuntimeError("beta_function_incomplete_cf error (no convergence)")


def beta_cdf(a, b, x):
    """``clm_ml.processes.math.beta_distribution_cdf`` on arrays (same operations and branches)."""
    import jax.numpy as jnp

    a, b, x = (np.asarray(v, dtype=float) for v in np.broadcast_arrays(a, b, x))
    edge = (x == 0.0) | (x == 1.0)
    xs = np.where(edge, 0.5, x)  # placeholder where bt = 0 (log of 0 never taken)
    bt = np.asarray(
        jnp.exp(
            _log_gamma(a + b)
            - _log_gamma(a)
            - _log_gamma(b)
            + a * np.asarray(jnp.log(xs))
            + b * np.asarray(jnp.log(1.0 - xs))
        )
    )
    bt = np.where(edge, 0.0, bt)
    direct = x < (a + 1.0) / (a + b + 2.0)
    out = np.empty_like(x)
    d, r = direct, ~direct  # each element's own branch only (the other need not converge)
    if d.any():
        out[d] = np.asarray(bt[d] * _beta_cf(a[d], b[d], x[d]) / a[d])
    if r.any():
        out[r] = np.asarray(1.0 - bt[r] * _beta_cf(b[r], a[r], 1.0 - x[r]) / b[r])
    return out


# ---------------------------------------------------------------------------------------
# Soil column (initVertical, SoilStateInitTimeConst)
# ---------------------------------------------------------------------------------------


def area_profile_parameters(leaf_beta, stem_beta, pft) -> tuple:
    """The beta-distribution parameters (p, q) of the leaf and stem area profiles of each
    patch, ``(P, 2)`` each: the patch's own when all four are positive (not NaN), otherwise
    its PFT's default.

    Fortran reference: CLMml_driver.F90::TowerVeg, MLinitVerticalMod.F90::getPADparameters."""
    lai, sai = np.array(leaf_beta, dtype=float), np.array(stem_beta, dtype=float)
    for p in range(len(pft)):
        if not (lai[p] > 0.0).all() or not (sai[p] > 0.0).all():  # NaN: not positive
            d = tables.default_area_profile(int(pft[p]))
            lai[p], sai[p] = d, d
    return lai, sai


def canopy_heights(canopy_height, pft) -> np.ndarray:
    """Each patch's canopy height (m): its own, or its PFT's default (NaN or not positive).

    Fortran reference: CLMml_driver.F90::TowerVeg."""
    h = np.array(canopy_height, dtype=float)
    return np.array(
        [h[p] if h[p] > 0.0 else tables.default_canopy_height(int(pft[p])) for p in range(len(h))]
    )


def vertical_structure(zref, htop, pbeta_lai, pbeta_sai, n_within, n_above, max_layers: int,
                       dz_tall: float, dz_short: float, dz_param: float, dpai_min: float) -> dict:  # fmt: skip
    """The layer grid of each patch: ``{name: (P, ...)}`` with ``n_layers``, ``n_within``,
    ``first_leafy`` (0-based, int32), ``dz``, ``zs`` (P, layer), ``zw`` (P, level),
    ``dlai_frac``, ``dsai_frac`` (P, layer) and ``zbot`` (P,) (m). ``n_within``/``n_above``:
    explicit layer counts (0: from the canopy height with ``dz_tall``/``dz_short``).

    Fortran reference: MLinitVerticalMod.F90::initVerticalStructure."""
    n_p, nlev = len(zref), max_layers
    out = {
        "zw": np.full((n_p, nlev + 1), SPVAL),
        "dz": np.full((n_p, nlev), SPVAL),
        "zs": np.full((n_p, nlev), SPVAL),
        "dlai_frac": np.full((n_p, nlev), SPVAL),
        "dsai_frac": np.full((n_p, nlev), SPVAL),
        "n_layers": np.zeros(n_p, dtype=np.int32),
        "n_within": np.zeros(n_p, dtype=np.int32),
        "first_leafy": np.zeros(n_p, dtype=np.int32),
        "zbot": np.zeros(n_p),
    }
    geo = []
    for p in range(n_p):
        zr, zt = float(zref[p]), float(htop[p])
        ztop_to_zref = zr - zt
        if n_within[p] > 0 and n_above[p] > 0:
            ntop = int(n_within[p])
            dz_within = zt / float(ntop)
            nabove = int(n_above[p])
            ncan = ntop + nabove
            dz_above = ztop_to_zref / float(nabove)
        else:
            dz_within = dz_tall if zt > dz_param else dz_short
            ntop = round(zt / dz_within)
            dz_within = zt / float(ntop)
            dz_above = dz_within
            nabove = round(ztop_to_zref / dz_above)
            ncan = ntop + nabove
            dz_above = ztop_to_zref / float(nabove)
        if ncan > nlev:
            raise ValueError(f"patch {p} needs {ncan} canopy layers; max_layers is {nlev}")
        zw = [0.0] * (ncan + 1)
        zw[ntop] = zt
        for ic in range(ntop - 1, -1, -1):
            zw[ic] = zw[ic + 1] - dz_within
        if abs(zw[0]) > 1.0e-10:
            raise ValueError(f"patch {p}: zw(0) improperly defined")
        zw[0] = min(max(zw[0], 0.0), 0.0)
        zw[ncan] = zr
        for ic in range(ncan - 1, ntop, -1):
            zw[ic] = zw[ic + 1] - dz_above
        for ic in range(0, ncan + 1):
            out["zw"][p, ic] = zw[ic]
        for ic in range(1, ncan + 1):
            out["dz"][p, ic - 1] = zw[ic] - zw[ic - 1]
            out["zs"][p, ic - 1] = 0.5 * (zw[ic] + zw[ic - 1])
        out["n_within"][p], out["n_layers"][p] = ntop, ncan
        geo.append((ntop, ncan, zt, zw))
    # beta CDFs at every layer bound of every patch, in one call
    pts = [(p, ic) for p, (ntop, _, _, _) in enumerate(geo) for ic in range(1, ntop + 1)]
    zb = np.array([min(geo[p][3][ic - 1] / geo[p][2], 1.0) for p, ic in pts])
    zt_ = np.array([min(geo[p][3][ic] / geo[p][2], 1.0) for p, ic in pts])
    a_l = np.array([pbeta_lai[p][0] for p, _ in pts])
    b_l = np.array([pbeta_lai[p][1] for p, _ in pts])
    a_s = np.array([pbeta_sai[p][0] for p, _ in pts])
    b_s = np.array([pbeta_sai[p][1] for p, _ in pts])
    dlai_all = beta_cdf(a_l, b_l, zt_) - beta_cdf(a_l, b_l, zb)
    dsai_all = beta_cdf(a_s, b_s, zt_) - beta_cdf(a_s, b_s, zb)
    k = 0
    for p, (ntop, ncan, zt, zw) in enumerate(geo):
        dlai = [0.0] + [float(v) * 1.0 for v in dlai_all[k : k + ntop]]
        dsai = [0.0] + [float(v) * 1.0 for v in dsai_all[k : k + ntop]]
        k += ntop
        if abs(sum(dlai[ic] + dsai[ic] for ic in range(1, ntop + 1)) - 2.0) > 1.0e-6:
            raise ValueError(f"patch {p}: plant area profile does not sum to canopy total")
        lai_miss = sai_miss = 0.0
        for ic in range(1, ntop + 1):
            if dlai[ic] + dsai[ic] < dpai_min:
                lai_miss += dlai[ic]
                sai_miss += dsai[ic]
                dlai[ic] = dsai[ic] = 0.0
        if lai_miss > 0.0:
            lai_sum = sum(dlai[ic] for ic in range(1, ntop + 1))
            for ic in range(1, ntop + 1):
                dlai[ic] = dlai[ic] + lai_miss * (dlai[ic] / lai_sum)
        if sai_miss > 0.0:
            sai_sum = sum(dsai[ic] for ic in range(1, ntop + 1))
            for ic in range(1, ntop + 1):
                dsai[ic] = dsai[ic] + sai_miss * (dsai[ic] / sai_sum)
        nbot = 0
        for ic in range(ntop, 0, -1):
            if dlai[ic] + dsai[ic] > 0.0:
                nbot = ic
        if nbot == 0:
            raise ValueError(f"patch {p}: nbot not defined")
        out["first_leafy"][p] = nbot - 1
        out["zbot"][p] = zw[nbot - 1]
        dlai += [0.0] * (ncan - ntop)
        dsai += [0.0] * (ncan - ntop)
        for ic in range(1, ncan + 1):
            out["dlai_frac"][p, ic - 1] = dlai[ic] / 1.0
            out["dsai_frac"][p, ic - 1] = dsai[ic] / 1.0
    return out


def initial_canopy_state(n_layers, max_layers: int, u, v, t, q, pbot, pco2) -> dict:
    """The canopy-air and leaf state of each patch before its first step, from the first
    step's forcing (each (P,)): wind ``u``, ``v`` (m/s), air temperature ``t`` (K), specific
    humidity ``q`` (kg/kg), pressure ``pbot`` (Pa), CO2 partial pressure ``pco2`` (Pa).
    ``{name: (P, ...)}`` with the :class:`clm_ml.state.CanopyState` names (leaf axis 0
    sunlit, 1 shaded) plus ``wind`` (P, layer): the layers in use set, the others ``spval``.

    Fortran reference: MLinitVerticalMod.F90::initVerticalProfiles."""
    import jax.numpy as jnp

    from clm_ml.processes.constants import mmdry, mmh2o

    n_p = len(n_layers)
    out = {
        "wind": np.full((n_p, max_layers), SPVAL),
        "air_temperature": np.full((n_p, max_layers), SPVAL),
        "vapor_pressure": np.full((n_p, max_layers), SPVAL),
        "co2": np.full((n_p, max_layers), SPVAL),
        "leaf_temperature": np.full((n_p, max_layers, 2), SPVAL),
        "leaf_water_potential": np.full((n_p, max_layers, 2), SPVAL),
        "intercepted_water": np.full((n_p, max_layers), SPVAL),
        "ground_temperature": np.zeros(n_p),
    }
    for p in range(n_p):
        up, vp = float(u[p]), float(v[p])
        tp, qp = float(t[p]), float(q[p])
        pb, co2 = float(pbot[p]), float(pco2[p])
        wind = float(jnp.sqrt(up**2 + vp**2))
        e = qp * pb / (mmh2o / mmdry + (1.0 - mmh2o / mmdry) * qp)
        n = int(n_layers[p])
        out["wind"][p, :n] = wind
        out["air_temperature"][p, :n] = tp
        out["vapor_pressure"][p, :n] = e
        out["co2"][p, :n] = co2 / pb * 1.0e6
        out["leaf_temperature"][p, :n] = tp
        out["leaf_water_potential"][p, :n] = -0.1
        out["intercepted_water"][p, :n] = 0.0
        out["ground_temperature"][p] = tp
    return out
