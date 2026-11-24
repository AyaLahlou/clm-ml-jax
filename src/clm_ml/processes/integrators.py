"""Time integration of the canopy state over a sub-step: explicit Runge-Kutta methods.

Each stage runs the physics from the state at the start of the sub-step (``bef``) plus a
weighted sum of the earlier stages' increments; the increment of stage ``j`` is the change
the physics made, ``k_j = x - x_bef``, and after the last stage
``x = x_bef + sum_j b_j k_j``. The methods are Butcher tableaus ``(a, b, c)``, registered by
name in ``TABLEAUS`` (``numerics.integrator`` in model.yaml). With ``euler`` there is one
stage and no update: the physics' own result is the new state.

The state integrated (``Integrated``): air temperature, vapor pressure and intercepted water
of each layer, leaf temperature and leaf water potential of each layer's leaves, and the
ground temperature. Layers are slots on axis 0 (0-based); only the layers in use change.

Fortran reference: CLM-ml v2 ``MLRungeKuttaMod.F90``.

References: Kutta (1901), Z. Math. Phys. 46, 435-453; Butcher (1996), Appl. Numer. Math. 20,
247-260; Ralston (1962), Math. Comp. 16, 431-437.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from clm_ml.processes.layers import keep
from clm_ml.state import PatchLayers

#: Fill value of the unused (upper-triangle) entries of ``a``, as Fortran's ``spval``
_UNUSED = 1.0e36


class Tableau(NamedTuple):
    """Butcher tableau of an ``n``-stage explicit method (numpy constants)."""

    a: np.ndarray  # (n, n) weights of earlier stages' increments; upper triangle unused
    b: np.ndarray  # (n,) weights of the final update
    c: np.ndarray  # (n,) stage times as fractions of the sub-step (not used by the update)


def _tableau(rows: dict, b, c) -> Tableau:
    n = len(b)
    a = np.full((n, n), _UNUSED)
    for (i, j), v in rows.items():
        a[i, j] = v
    return Tableau(a, np.asarray(b, dtype=float), np.asarray(c, dtype=float))


#: Integrators by name (``numerics.integrator``) -> tableau; Fortran ``runge_kutta_type``
#: codes in the comments. Entries computed as Fortran writes them (e.g. ``2/6``).
TABLEAUS = {
    # Euler (10): one stage. The canopy sub-step keeps the processes' own result and never
    # calls the update (Fortran never calls RungeKuttaUpdate); b = [1] is still the correct
    # weight, so an update would give bef + k = cur, not reset the state
    "euler": Tableau(np.full((1, 1), _UNUSED), np.ones(1), np.zeros(1)),
    # 2nd-order trapezoidal rule (21; Butcher 1996)
    "rk2_trapezoid": _tableau({(1, 0): 1.0}, [1.0 / 2.0, 1.0 / 2.0], [0.0, 1.0]),
    # 2nd-order midpoint rule (22; Butcher 1996)
    "rk2_midpoint": _tableau({(1, 0): 1.0 / 2.0}, [0.0, 1.0], [0.0, 1.0 / 2.0]),
    # 2nd-order Ralston (23; Ralston 1962)
    "rk2_ralston": _tableau({(1, 0): 2.0 / 3.0}, [1.0 / 4.0, 3.0 / 4.0], [0.0, 2.0 / 3.0]),
    # 3rd-order Heun (31; Butcher 1996)
    "rk3_heun": _tableau(
        {(1, 0): 1.0 / 3.0, (2, 0): 0.0, (2, 1): 2.0 / 3.0},
        [1.0 / 4.0, 0.0, 3.0 / 4.0],
        [0.0, 1.0 / 3.0, 2.0 / 3.0],
    ),
    # 3rd-order Ralston (32; Ralston 1962)
    "rk3_ralston": _tableau(
        {(1, 0): 1.0 / 2.0, (2, 0): 0.0, (2, 1): 3.0 / 4.0},
        [2.0 / 9.0, 3.0 / 9.0, 4.0 / 9.0],
        [0.0, 1.0 / 2.0, 3.0 / 4.0],
    ),
    # 3rd-order Kutta (33; Kutta 1901)
    "rk3_kutta": _tableau(
        {(1, 0): 1.0 / 2.0, (2, 0): -1.0, (2, 1): 2.0},
        [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0],
        [0.0, 1.0 / 2.0, 1.0],
    ),
    # 4th-order classical Kutta (41; Butcher 1996, Kutta 1901)
    "rk4": _tableau(
        {(1, 0): 1.0 / 2.0, (2, 0): 0.0, (2, 1): 1.0 / 2.0, (3, 0): 0.0, (3, 1): 0.0, (3, 2): 1.0},
        [1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0],
        [0.0, 1.0 / 2.0, 1.0 / 2.0, 1.0],
    ),
}


class Integrated(NamedTuple):
    """The state the integrator advances (or, for increments, a trailing stage axis)."""

    tair: Array  # (layer,) air temperature, K
    eair: Array  # (layer,) vapor pressure, Pa
    h2ocan: Array  # (layer,) intercepted water, kg H2O/m2
    tleaf: Array  # (layer, leaf) leaf temperature, K
    lwp: Array  # (layer, leaf) leaf water potential, MPa
    tg: Array  # () ground temperature, K


def runge_kutta_update(
    stage,
    tableau: Tableau,
    layers: PatchLayers,
    bef: Integrated,
    cur: Integrated,
    increments: Integrated,
) -> tuple[Integrated, Integrated, Array]:
    """The state after Runge-Kutta stage ``stage`` (1-based, as Fortran; a Python int or a
    traced integer when the stages run as a ``lax.scan``).

    ``cur`` is the state the physics produced in this stage from the stage's starting
    state; its increment is ``cur - bef``. Before the last stage the result is the next
    stage's starting state ``bef + sum_{j<=stage} a[stage, j] k_j``; at the last stage the
    final ``bef + sum_j b_j k_j``; after it (the physics-only pass) ``cur`` unchanged.
    ``increments`` hold the earlier stages' increments on a trailing stage axis (which may
    be longer than the tableau). Earlier stages are summed first, then this stage's term
    is added (Fortran adds them one by one). Layers not in use keep
    their values; inputs there are zeroed so no fill values enter the arithmetic.

    Fortran reference: MLRungeKuttaMod.F90::RungeKuttaUpdate.
    Reads: ``tair_profile``, ``eair_profile``, ``h2ocan_profile``, ``tleaf_leaf``,
    ``lwp_leaf``, ``tg_soil``, their ``*_bef`` copies and the increments ``dtair_profile``,
    ``deair_profile``, ``dh2ocan_profile``, ``dtleaf_leaf``, ``dlwp_leaf``, ``dtg_soil``.
    Writes: the state and stage ``k`` of the increments.
    Returns ``(state, this stage's increments, k)``, ``k`` the 0-based stage axis index to
    write the increments to (the last stage's index after the last stage, where the
    increments returned are the ones already there).
    """
    a, b = jnp.asarray(tableau.a), jnp.asarray(tableau.b)
    nrk = b.shape[0]
    irk0 = jnp.asarray(stage) - 1
    do_update = irk0 < nrk
    k = jnp.minimum(irk0, nrk - 1)  # stage axis index written (clamped after the last stage)
    weights = jnp.where(irk0 < nrk - 1, a[jnp.minimum(irk0 + 1, nrk - 1)], b)
    w_curr = weights[k]

    def _advance(x_bef, x_cur, dx_prev):
        dx_new = x_cur - x_bef
        m = dx_prev.shape[-1]
        j = jnp.arange(m)
        # earlier stages j < k only (the upper triangle of a is unused)
        w_prev = jnp.zeros(m).at[:nrk].set(jnp.where(j[:nrk] < k, weights, 0.0))
        x_new = x_bef + jnp.where(j < k, dx_prev * w_prev, 0.0).sum(axis=-1) + w_curr * dx_new
        return jnp.where(do_update, x_new, x_cur), dx_new

    new, increment = {}, {}
    for name in Integrated._fields:
        x_bef, x_cur, dx = getattr(bef, name), getattr(cur, name), getattr(increments, name)
        dx_col = dx[..., k]
        if name == "tg":
            x_new, dx_new = _advance(x_bef, x_cur, dx)
            new[name] = x_new
            increment[name] = jnp.where(do_update, dx_new, dx_col)
            continue
        x_bef_in, x_cur_in, dx_in = (
            keep(layers.in_use, v, jnp.zeros_like(v)) for v in (x_bef, x_cur, dx)
        )
        x_new, dx_new = _advance(x_bef_in, x_cur_in, dx_in)
        new[name] = keep(layers.in_use, x_new, x_cur)
        increment[name] = keep(layers.in_use, jnp.where(do_update, dx_new, dx_col), dx_col)
    return Integrated(**new), Integrated(**increment), k
