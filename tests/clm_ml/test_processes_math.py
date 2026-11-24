"""clm_ml.processes.math: the tridiagonal solvers as scans.

The Fortran golden tests check every routine against Fortran (unguarded pivots); these check
the guarded variant used by the solar radiation and soil temperature solves, and the
derivatives of the coupled solver ``tridiag_2eq`` in both modes.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from clm_ml.processes.math import tridiag, tridiag_2eq  # noqa: E402


def _system(n, seed):
    rng = np.random.default_rng(seed)
    a, c, r = rng.uniform(-1, 0, n), rng.uniform(-1, 0, n), rng.normal(size=n)
    b = 2.5 + rng.uniform(size=n)  # diagonally dominant: every pivot > 1
    a[0], c[-1] = 0.0, 0.0
    return a, b, c, r


def _dense(a, b, c):
    return np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)


def test_tridiag_solves_the_system():
    for n in (1, 2, 7, 103):
        a, b, c, r = _system(n, n)
        expected = np.linalg.solve(_dense(a, b, c), r)
        for min_pivot in (None, 1.0e-10):
            u = np.asarray(tridiag(a, b, c, r, min_pivot=min_pivot))
            np.testing.assert_allclose(u, expected, rtol=1e-12, atol=1e-14)


def test_guard_changes_nothing_when_pivots_exceed_it():
    a, b, c, r = _system(50, 3)
    a[0] = 123.0  # a[0] is not used
    guarded = np.asarray(tridiag(a, b, c, r, min_pivot=1.0e-10))
    a[0] = 0.0
    np.testing.assert_array_equal(guarded, np.asarray(tridiag(a, b, c, r)))


def test_guard_keeps_a_zero_pivot_finite():
    a, b, c, r = (np.zeros(3), np.array([0.0, 1.0, 1.0]), np.zeros(3), np.ones(3))
    assert np.all(np.isfinite(np.asarray(tridiag(a, b, c, r, min_pivot=1.0e-10))))


# -- tridiag_2eq: derivatives -------------------------------------------------------------

N2 = 9


def _system_2eq(seed):
    """Coefficients (a1, b11, b12, c1, d1, a2, b21, b22, c2, d2) of a diagonally dominant
    coupled system of N2 rows."""
    rng = np.random.default_rng(seed)
    co = [rng.uniform(-1, 1, N2) for _ in range(10)]
    co[1] += 5.0  # b11
    co[7] += 5.0  # b22
    return tuple(jnp.asarray(x) for x in co)


def _dense_2eq(co):
    """``(t, q)`` from a dense solve of the same system (rows t_0..t_n-1, q_0..q_n-1)."""
    a1, b11, b12, c1, d1, a2, b21, b22, c2, d2 = co
    n, i = N2, jnp.arange(N2)
    A = jnp.zeros((2 * n, 2 * n))
    A = A.at[i, i].set(b11).at[i, n + i].set(b12).at[n + i, i].set(b21).at[n + i, n + i].set(b22)
    A = A.at[i[1:], i[:-1]].set(a1[1:]).at[n + i[1:], n + i[:-1]].set(a2[1:])
    A = A.at[i[:-1], i[1:]].set(c1[:-1]).at[n + i[:-1], n + i[1:]].set(c2[:-1])
    x = jnp.linalg.solve(A, jnp.concatenate([d1, d2]))
    return x[:n], x[n:]


def _solve_2eq(*co):
    return tridiag_2eq(*co, N2)


def test_tridiag_2eq_forward_mode_matches_dense_solve():
    co = _system_2eq(1)
    tangents = _system_2eq(2)
    primal, tangent = jax.jvp(_solve_2eq, co, tangents)
    primal_ref, tangent_ref = jax.jvp(lambda *c: _dense_2eq(c), co, tangents)
    for got, ref in zip(primal + tangent, primal_ref + tangent_ref):
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


def test_tridiag_2eq_forward_and_reverse_mode_agree():
    """Dot-product test: <J v, u> = <v, J^T u> for random v and u."""
    co, v = _system_2eq(3), _system_2eq(4)
    rng = np.random.default_rng(5)
    u = (jnp.asarray(rng.normal(size=N2)), jnp.asarray(rng.normal(size=N2)))
    _, jv = jax.jvp(_solve_2eq, co, v)
    _, vjp = jax.vjp(_solve_2eq, *co)
    jtu = vjp(u)
    lhs = sum(float(jnp.dot(a, b)) for a, b in zip(jv, u))
    rhs = sum(float(jnp.dot(a, b)) for a, b in zip(v, jtu))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


def test_tridiag_2eq_reverse_mode_matches_implicit_function_formulas():
    """The reverse-mode derivatives are those of the custom VJP rule they replaced:
    dL/dd = z with A^T z = g, and dL/dA[i, j] = -z[i] x[j] (ghost values 0)."""
    co = _system_2eq(6)
    rng = np.random.default_rng(7)
    g = (jnp.asarray(rng.normal(size=N2)), jnp.asarray(rng.normal(size=N2)))
    (t, q), vjp = jax.vjp(_solve_2eq, *co)
    grads = vjp(g)
    a1, b11, b12, c1, d1, a2, b21, b22, c2, d2 = co
    n, i = N2, jnp.arange(N2)
    A = jnp.zeros((2 * n, 2 * n))
    A = A.at[i, i].set(b11).at[i, n + i].set(b12).at[n + i, i].set(b21).at[n + i, n + i].set(b22)
    A = A.at[i[1:], i[:-1]].set(a1[1:]).at[n + i[1:], n + i[:-1]].set(a2[1:])
    A = A.at[i[:-1], i[1:]].set(c1[:-1]).at[n + i[:-1], n + i[1:]].set(c2[:-1])
    z = jnp.linalg.solve(A.T, jnp.concatenate(g))
    z1, z2 = z[:n], z[n:]
    zero = jnp.zeros(1)
    prev = lambda v: jnp.concatenate([zero, v[:-1]])  # noqa: E731
    nxt = lambda v: jnp.concatenate([v[1:], zero])  # noqa: E731
    expected = (
        -z1 * prev(t), -z1 * t, -z1 * q, -z1 * nxt(t), z1,
        -z2 * prev(q), -z2 * t, -z2 * q, -z2 * nxt(q), z2,
    )  # fmt: skip
    for got, ref in zip(grads, expected):
        np.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-14)


def test_tridiag_2eq_batches_and_compiles():
    co = _system_2eq(8)
    d1 = jnp.stack([co[4], 2.0 * co[4], -co[4]])
    t, q = jax.jit(jax.vmap(lambda d: tridiag_2eq(*co[:4], d, *co[5:], N2)))(d1)
    assert t.shape == q.shape == (3, N2)
    np.testing.assert_allclose(t[1], 2.0 * t[0] - _solve_2eq(*co[:4], 0 * co[4], *co[5:])[0],
                               rtol=1e-12)  # fmt: skip


def test_unrolled_solves_match_the_loops():
    """The solves with their rows written out (numerics.solver_loops) do the
    loops' operations in the loops' order."""
    a, b, c, r = _system(43, 5)
    for min_pivot in (None, 1.0e-10):
        loop = tridiag(a, b, c, r, min_pivot=min_pivot)
        np.testing.assert_allclose(
            tridiag(a, b, c, r, min_pivot=min_pivot, unroll=True), loop, rtol=1e-14
        )
    co = _system_2eq(1)
    for got, ref in zip(tridiag_2eq(*co, N2, unroll=True), tridiag_2eq(*co, N2)):
        np.testing.assert_allclose(got, ref, rtol=1e-14)
    grads = [
        jax.grad(lambda c: jnp.sum(tridiag_2eq(*c, N2, unroll=u)[0]))(co) for u in (False, True)
    ]
    for got, ref in zip(grads[1], grads[0]):
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)
