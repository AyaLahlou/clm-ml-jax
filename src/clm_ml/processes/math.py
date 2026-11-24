"""Mathematical tools: root finding, quadratic and tridiagonal solvers, beta distribution.

Pure functions of their arguments (no model state). Used by the canopy processes
(turbulence, photosynthesis, radiation, flux-profile solution, soil temperature) and by the
setup (leaf and stem area profiles from beta distributions).

Fortran reference: CLM-ml v2 ``MLMathToolsMod.F90``. Fortran's root finders that call a
function threading the canopy container (``hybrid``, ``zbrent``, ``bisection``) are not
translated here: the time step uses traceable solvers written for each problem (Obukhov length,
intercellular CO2, stomatal conductance) and :func:`bisection_traceable`.
"""

from __future__ import annotations

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Public: tracing helper
# ---------------------------------------------------------------------------


def is_concrete(*values) -> bool:
    """True if none of ``values`` is a JAX tracer.

    Runtime checks that stop the model (Fortran ``endrun`` on conservation or
    convergence errors) can only run on concrete values. Under ``jit``/``grad``
    tracing they are skipped.
    """
    return not any(isinstance(v, jax.core.Tracer) for v in values)


# ---------------------------------------------------------------------------
# Public: bisection root finder (traceable)
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("func", "tol", "itmax", "unroll"))
def bisection_traceable(
    func, xa, xb, tol: float, itmax: int, c_init=None, args=(), unroll: bool = False
):
    """
    JAX-traceable bisection with the Fortran stopping rule.

    Fortran reference: MLMathToolsMod.F90::bisection.

    Same iterations and result as Fortran ``bisection`` (MLMathToolsMod.F90,
    lines 249-310) and :func:`bisection_scalar`: halve while ``|b - a| > tol``
    (at most ``itmax`` times), keeping the half where ``f(a) * f(c) < 0``
    puts the root in ``[a, c]``, and return the **last midpoint** ``c``.

    ``lax.while_loop`` cannot be reverse-differentiated, so this is a
    ``fori_loop`` of ``itmax`` iterations whose updates stop once the
    tolerance is met. The caller checks that the root is bracketed (Fortran
    stops with an error otherwise).

    Compiled once per ``func``: pass a module-level function and its
    parameters through ``args`` (a closure would be a new function at every
    call and recompile every time).

    Args:
        func: Traceable function ``func(x, *args)`` of one scalar ``x``;
            a module-level function (static argument of ``jax.jit``).
        xa, xb: Ends of the bracket.
        tol: Stop when the bracket is no wider than this.
        itmax: Maximum number of halvings.
        c_init: Value returned if no halving is done (Fortran leaves ``c``
            undefined; defaults to ``xa`` as in :func:`bisection_scalar`).
        args: Further arguments of ``func`` (traced).
        unroll: The halvings written out instead of a compiled loop (the same halvings;
            model.yaml ``numerics.solver_loops``; faster on a GPU).

    Returns:
        The last midpoint.
    """
    a = jnp.asarray(xa)
    b = jnp.asarray(xb)
    fa = func(a, *args)
    c0 = a if c_init is None else jnp.asarray(c_init, dtype=a.dtype)

    def body(_, carry):
        a, b, fa, c, done = carry
        active = jnp.logical_and(jnp.logical_not(done), jnp.abs(b - a) > tol)
        c_new = (a + b) / 2.0
        fc = func(c_new, *args)
        root_in_left = fa * fc < 0.0  # Fortran: if (fa * fc < 0) then b = c else a = c
        b = jnp.where(active & root_in_left, c_new, b)
        a = jnp.where(active & ~root_in_left, c_new, a)
        fa = jnp.where(active & ~root_in_left, fc, fa)
        c = jnp.where(active, c_new, c)
        return a, b, fa, c, jnp.logical_not(active)

    init = (a, b, fa, c0, jnp.asarray(False))
    _, _, _, c, _ = jax.lax.fori_loop(0, itmax, body, init, unroll=unroll or None)
    return c


# ---------------------------------------------------------------------------
# Public: quadratic solver
# ---------------------------------------------------------------------------


def quadratic(a: float, b: float, c: float) -> Tuple[float, float]:
    """
    Solve the quadratic equation ``a*x^2 + b*x + c = 0`` for its
    two roots using a numerically stable form.

    Fortran reference: MLMathToolsMod.F90::quadratic.

    Uses the sign of ``b`` to choose the branch that avoids
    catastrophic cancellation:

    .. code-block:: none

        q  = -0.5*(b + sqrt(b^2 - 4ac))   if b >= 0
        q  = -0.5*(b - sqrt(b^2 - 4ac))   if b < 0
        r1 = q/a
        r2 = c/q   (or 1e36 if q == 0)

    Args:
        a: Quadratic coefficient (must be non-zero).
        b: Linear coefficient.
        c: Constant term.

    Returns:
        Tuple ``(r1, r2)`` of the two roots.
    """
    # Safe sqrt: clamp discriminant to ≥0 to avoid NaN when JAX traces
    # both branches unconditionally (e.g. inside jnp.where callers).
    discriminant = jnp.sqrt(jnp.maximum(b * b - 4.0 * a * c, 1.0e-30))

    # Numerically stable branch selection via jnp.where — avoids
    # Python `if` on potentially JAX-traced `b`.
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug.
    # All callers pass a > 0 (curvature parameters, conductances); using
    # maximum is numerically equivalent for all physically valid inputs.
    a_safe = jnp.maximum(a, 1.0e-30)
    q = jnp.where(
        b >= 0.0,
        -0.5 * (b + discriminant),
        -0.5 * (b - discriminant),
    )
    # q can be signed; guard against exact 0 by adding a tiny offset
    # when q == 0 (avoids select-as-denominator pattern).
    q_safe = q + jnp.asarray(q == 0.0, dtype=q.dtype)  # adds 1.0 when q == 0
    r1 = q / a_safe
    r2 = jnp.where(q != 0.0, c / q_safe, jnp.asarray(1.0e36))

    return r1, r2


# ---------------------------------------------------------------------------
# Public: tridiagonal solver
# ---------------------------------------------------------------------------


def tridiag(
    a: Array,
    b: Array,
    c: Array,
    r: Array,
    min_pivot: float | None = None,
    unroll: bool = False,
) -> Array:
    """
    Solve a tridiagonal system with the Thomas algorithm, as ``lax.scan`` loops.

    Fortran reference: MLMathToolsMod.F90::tridiag.

    Row ``j`` (0-based) reads ``a[j] u[j-1] + b[j] u[j] + c[j] u[j+1] = r[j]``; ``a[0]`` and
    ``c[-1]`` are not used. The arithmetic per row is Fortran's; the loops are scans, so the
    compiled program does not grow with the number of rows (unless ``unroll``).

    Args:
        a, b, c: Lower, main and upper diagonals, shape ``(n,)``.
        r: Right-hand side, shape ``(n,)``.
        min_pivot: If given, every pivot ``bet`` is replaced by ``max(|bet|, min_pivot)``
            (a guard not in Fortran, used by the solar radiation and soil temperature
            solves). ``None``: Fortran's unguarded pivots.
        unroll: The rows written out one after another instead of compiled loops (the same
            operations in the same order; faster on a GPU, where every loop iteration is
            a separate command; model.yaml ``numerics.solver_loops``).

    Returns:
        The solution ``u``, shape ``(n,)``.
    """
    a, b, c, r = (jnp.asarray(x) for x in (a, b, c, r))

    def _pivot(bet):
        return bet if min_pivot is None else jnp.maximum(jnp.abs(bet), min_pivot)

    def _forward(carry, row):
        piv, u_prev, c_prev = carry
        a_j, b_j, c_j, r_j = row
        gam = c_prev / piv
        piv = _pivot(b_j - a_j * gam)
        u = (r_j - a_j * u_prev) / piv
        return (piv, u, c_j), (gam, u)

    one, zero = jnp.ones((), dtype=b.dtype), jnp.zeros((), dtype=b.dtype)
    # First row: gam = 0 / 1 = 0, bet = b[0] - a[0] * 0, u = (r[0] - a[0] * 0) / bet
    rows = (a.at[0].set(0.0), b, c, r)
    _, (gam, u) = jax.lax.scan(_forward, (one, zero, zero), rows, unroll=unroll or 1)

    def _back(u_next, row):
        u_j, gam_next = row
        u_j = u_j - gam_next * u_next
        return u_j, u_j

    _, u_head = jax.lax.scan(_back, u[-1], (u[:-1], gam[1:]), reverse=True, unroll=unroll or 1)
    return jnp.concatenate([u_head, u[-1:]])


# ---------------------------------------------------------------------------
# Public: coupled 2-equation tridiagonal solver
# ---------------------------------------------------------------------------


def tridiag_2eq(
    a1: list[float],
    b11: list[float],
    b12: list[float],
    c1: list[float],
    d1: list[float],
    a2: list[float],
    b21: list[float],
    b22: list[float],
    c2: list[float],
    d2: list[float],
    n: int,
    unroll: bool = False,
) -> Tuple[Array, Array]:
    """
    Solve the coupled tridiagonal system for air temperature ``t`` and
    water vapour mole fraction ``q`` at each canopy layer.

    Fortran reference: MLMathToolsMod.F90::tridiag_2eq.

    The system at each layer ``i = 0, ..., n-1`` (0-based) is:

    .. code-block:: none

        a1(i)*T(i-1) + b11(i)*T(i) + b12(i)*q(i) + c1(i)*T(i+1) = d1(i)
        a2(i)*q(i-1) + b21(i)*T(i) + b22(i)*q(i) + c2(i)*q(i+1) = d2(i)

    Solved by forward elimination to express each layer in terms of the
    layer above, followed by back substitution from the top layer downward
    (:func:`_tridiag_2eq_solve`, Fortran's arithmetic).

    Derivatives come from ``jax.lax.custom_linear_solve``,
    not from differentiating the elimination, whose intermediate ``e``-coefficients can be
    huge and give NaN gradients. For ``A x = d``, JAX applies the implicit function theorem:
    forward mode solves ``A dx = dd - dA x`` with the same elimination; reverse mode solves
    the transpose system ``A^T z = g`` (:func:`_tridiag_2eq_transpose_solve`, the same
    algorithm on the transposed blocks) and forms ``dL/dd = z`` and ``dL/dA[i, j] = -z[i]
    x[j]`` through the operator :func:`_tridiag_2eq_matvec`. Both modes therefore work
    (``jax.grad``, ``jax.jvp``, ``jax.jacfwd``), and the forward value is the elimination's.

    All arrays are **0-based** (indices 0..n-1), matching the calling
    convention in ``ImplicitFluxProfileSolution``.

    Args:
        a1, b11, b12, c1, d1: Coefficients for the temperature equation,
            each length at least ``n``, 0-indexed.
        a2, b21, b22, c2, d2: Coefficients for the water vapour equation,
            each length at least ``n``, 0-indexed.
        n: Number of layers.
        unroll: The rows written out instead of compiled loops (as :func:`tridiag`).

    Returns:
        Tuple ``(t, q)`` of solution vectors, each of length ``n``, 0-indexed.
    """
    a1, b11, b12, c1, d1, a2, b21, b22, c2, d2 = (
        _as_array(x) for x in (a1, b11, b12, c1, d1, a2, b21, b22, c2, d2)
    )
    coefficients = (a1, b11, b12, c1, a2, b21, b22, c2)
    return jax.lax.custom_linear_solve(
        functools.partial(_tridiag_2eq_matvec, coefficients),
        (d1, d2),
        solve=lambda _, d: _tridiag_2eq_solve(
            a1, b11, b12, c1, d[0], a2, b21, b22, c2, d[1], n, unroll
        ),
        transpose_solve=lambda _, g: _tridiag_2eq_transpose_solve(coefficients, g, unroll),
    )


def _shift_down(v):
    """``v[i-1]`` at row ``i``, 0 in row 0 (no row below the ground)."""
    return jnp.concatenate([jnp.zeros(1, dtype=v.dtype), v[:-1]])


def _shift_up(v):
    """``v[i+1]`` at row ``i``, 0 in the top row."""
    return jnp.concatenate([v[1:], jnp.zeros(1, dtype=v.dtype)])


def _tridiag_2eq_matvec(coefficients, x):
    """The operator ``A x`` of :func:`tridiag_2eq` (left-hand sides of its two equations),
    for ``x = (t, q)``; JAX differentiates it for the derivatives with respect to the
    coefficients. ``a1[0]``, ``a2[0]`` and ``c1[-1]``, ``c2[-1]`` multiply 0."""
    a1, b11, b12, c1, a2, b21, b22, c2 = coefficients
    t, q = x
    return (
        a1 * _shift_down(t) + b11 * t + b12 * q + c1 * _shift_up(t),
        a2 * _shift_down(q) + b21 * t + b22 * q + c2 * _shift_up(q),
    )


def _tridiag_2eq_transpose_solve(coefficients, g, unroll: bool = False):
    """Solve ``A^T z = g`` for the reverse-mode derivative of :func:`tridiag_2eq`, with
    the same block Thomas algorithm.

    ``A`` is block tridiagonal with 2x2 blocks per row ``i`` (0-based): lower
    ``diag(a1[i], a2[i])``, main ``[[b11, b12], [b21, b22]][i]``, upper
    ``diag(c1[i], c2[i])``. ``A^T`` has the same structure with lower
    ``diag(c1[i-1], c2[i-1])`` (0 in row 0), main ``[[b11, b21], [b12, b22]][i]`` and upper
    ``diag(a1[i+1], a2[i+1])`` (0 in the top row).
    """
    a1, b11, b12, c1, a2, b21, b22, c2 = coefficients
    a1T, a2T = _shift_down(c1), _shift_down(c2)
    c1T, c2T = _shift_up(a1), _shift_up(a2)
    return _tridiag_2eq_solve(
        a1T, b11, b21, c1T, _as_array(g[0]), a2T, b12, b22, c2T, _as_array(g[1]), None, unroll
    )


def _as_array(x):
    """A list of scalars or an array -> a 1-D array."""
    return jnp.stack(x) if isinstance(x, (list, tuple)) else jnp.asarray(x)


def _tridiag_2eq_solve(
    a1,
    b11,
    b12,
    c1,
    d1,
    a2,
    b21,
    b22,
    c2,
    d2,
    n,
    unroll: bool = False,
):
    """Core Thomas algorithm for the block-2x2 tridiagonal system.

    Accepts lists of scalars or 1-D arrays (0-based rows); returns ``(t, q)`` arrays. The
    forward elimination and the back substitution are ``lax.scan`` loops (the per-row
    arithmetic as before), so the compiled program does not grow with the number of rows;
    ``unroll`` writes the rows out instead (as :func:`tridiag`).
    """
    rows = tuple(_as_array(x) for x in (a1, b11, b12, c1, d1, a2, b21, b22, c2, d2))
    _eps_det = jnp.asarray(1.0e-10)

    def _eliminate(prev, row):
        e11_prev, e12_prev, e21_prev, e22_prev, f1_prev, f2_prev = prev
        a1_i, b11_i, b12_i, c1_i, d1_i, a2_i, b21_i, b22_i, c2_i, d2_i = row
        ainv = b11_i - a1_i * e11_prev
        binv = b12_i - a1_i * e12_prev
        cinv = b21_i - a2_i * e21_prev
        dinv = b22_i - a2_i * e22_prev
        det = ainv * dinv - binv * cinv
        _abs_det = jnp.abs(det)
        det_safe = jnp.where(_abs_det > _eps_det, det, _eps_det)

        e11 = dinv * c1_i / det_safe
        e12 = -binv * c2_i / det_safe
        e21 = -cinv * c1_i / det_safe
        e22 = ainv * c2_i / det_safe

        f1 = (dinv * (d1_i - a1_i * f1_prev) - binv * (d2_i - a2_i * f2_prev)) / det_safe
        f2 = (-cinv * (d1_i - a1_i * f1_prev) + ainv * (d2_i - a2_i * f2_prev)) / det_safe
        out = (e11, e12, e21, e22, f1, f2)
        return out, out

    zero = jnp.zeros((), dtype=rows[0].dtype)
    _, (e11, e12, e21, e22, f1, f2) = jax.lax.scan(
        _eliminate, (zero,) * 6, rows, unroll=unroll or 1
    )

    # Back substitution from the top row: t = f1 - e11 t_next - e12 q_next (the top row has
    # no row above: t = f1, q = f2)
    def _substitute(nxt, row):
        t_next, q_next = nxt
        e11_i, e12_i, e21_i, e22_i, f1_i, f2_i = row
        t = f1_i - e11_i * t_next - e12_i * q_next
        q = f2_i - e21_i * t_next - e22_i * q_next
        return (t, q), (t, q)

    top = (f1[-1], f2[-1])
    _, (t, q) = jax.lax.scan(
        _substitute,
        top,
        (e11[:-1], e12[:-1], e21[:-1], e22[:-1], f1[:-1], f2[:-1]),
        reverse=True,
        unroll=unroll or 1,
    )
    return jnp.concatenate([t, f1[-1:]]), jnp.concatenate([q, f2[-1:]])


# ---------------------------------------------------------------------------
# Public: log-gamma function
# ---------------------------------------------------------------------------


def log_gamma_function(x: float) -> float:
    """
    Return the natural logarithm of the gamma function, ``ln(Γ(x))``,
    using Lanczos's approximation.

    Fortran reference: MLMathToolsMod.F90::log_gamma_function.

    Coefficients follow the Numerical Recipes implementation
    (Fortran lines 377-379).

    Args:
        x: Input argument (x > 0).

    Returns:
        ``ln(Γ(x))``.
    """
    # Fortran: parameter coef(6), stp
    coef = (
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    )
    stp: float = 2.5066282746310005

    y = x
    tmp = x + 5.5
    tmp = (x + 0.5) * jnp.log(tmp) - tmp
    ser = 1.000000000190015
    for j in range(6):  # Fortran: do j = 1, 6; y = y + 1; ser += coef(j)/y
        y += 1.0
        ser += coef[j] / y
    return tmp + jnp.log(stp * ser / x)


# ---------------------------------------------------------------------------
# Public: beta function
# ---------------------------------------------------------------------------


def beta_function(a: float, b: float) -> float:
    """
    Return the beta function ``B(a, b) = Γ(a)Γ(b)/Γ(a+b)``.

    Fortran reference: MLMathToolsMod.F90::beta_function.

    Args:
        a: First parameter (a > 0).
        b: Second parameter (b > 0).

    Returns:
        ``B(a, b)``.
    """
    return jnp.exp(log_gamma_function(a) + log_gamma_function(b) - log_gamma_function(a + b))


# ---------------------------------------------------------------------------
# Public: beta distribution PDF
# ---------------------------------------------------------------------------


def beta_distribution_pdf(a: float, b: float, x: float) -> float:
    """
    Return the beta distribution probability density function,
    ``f(x; a, b)``.

    Fortran reference: MLMathToolsMod.F90::beta_distribution_pdf.

    .. code-block:: none

        f(x; a, b) = x^(a-1) * (1-x)^(b-1) / B(a, b)

    Args:
        a: First shape parameter (a > 0).
        b: Second shape parameter (b > 0).
        x: Evaluation point (0 ≤ x ≤ 1).

    Returns:
        ``f(x; a, b)``.
    """
    return (1.0 / beta_function(a, b)) * x ** (a - 1.0) * (1.0 - x) ** (b - 1.0)


# ---------------------------------------------------------------------------
# Public: beta distribution CDF
# ---------------------------------------------------------------------------


def beta_distribution_cdf(a: float, b: float, x: float) -> float:
    """
    Return the beta distribution cumulative distribution function,
    ``F(x; a, b) = I_x(a, b)``, using the incomplete beta function.

    Fortran reference: MLMathToolsMod.F90::beta_distribution_cdf.

    Uses the continued fraction representation via
    :func:`_beta_function_incomplete_cf`.  When
    ``x < (a+1)/(a+b+2)`` the direct form is used; otherwise the
    symmetry relation ``I_x(a,b) = 1 - I_{1-x}(b,a)`` is applied
    (Fortran lines 446-450).

    Args:
        a: First shape parameter.
        b: Second shape parameter.
        x: Evaluation point (0 ≤ x ≤ 1).

    Returns:
        ``F(x; a, b)``.
    """
    # Fortran lines 440-444
    if x == 0.0 or x == 1.0:
        bt = 0.0
    else:
        bt = jnp.exp(
            log_gamma_function(a + b)
            - log_gamma_function(a)
            - log_gamma_function(b)
            + a * jnp.log(x)
            + b * jnp.log(1.0 - x)
        )

    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _beta_function_incomplete_cf(a, b, x) / a
    else:
        return 1.0 - bt * _beta_function_incomplete_cf(b, a, 1.0 - x) / b


# ---------------------------------------------------------------------------
# Private: incomplete beta continued fraction
# ---------------------------------------------------------------------------


def _beta_function_incomplete_cf(a: float, b: float, x: float) -> float:
    """
    Evaluate the continued fraction representation of the incomplete
    beta function.

    Fortran reference: MLMathToolsMod.F90::beta_function_incomplete_cf.

    Uses the modified Lentz algorithm with ``maxit = 100``,
    ``eps = 3e-7``, and ``fpmin = 1e-30``.

    Args:
        a: First shape parameter.
        b: Second shape parameter.
        x: Evaluation point.

    Returns:
        The continued fraction value ``betacf``.
    """
    maxit: int = 100  # Fortran: parameter maxit = 100
    eps: float = 3.0e-7  # Fortran: parameter eps = 3.e-07_r8
    fpmin: float = 1.0e-30  # Fortran: parameter fpmin = 1.e-30_r8

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, maxit + 1):  # Fortran: do m = 1, maxit
        m2 = 2 * m
        # Even step — Fortran lines 487-494
        aa = float(m) * (b - float(m)) * x / ((qam + float(m2)) * (a + float(m2)))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h = h * d * c
        # Odd step — Fortran lines 495-503
        aa = -(a + float(m)) * (qab + float(m)) * x / ((qap + float(m2)) * (a + float(m2)))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h = h * delta
        if abs(delta - 1.0) < eps:
            return h

    # Fortran stops the model (endrun)
    raise RuntimeError("beta_function_incomplete_cf: no convergence in maxit iterations")
