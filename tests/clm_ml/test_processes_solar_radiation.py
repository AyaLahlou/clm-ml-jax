"""clm_ml.processes.solar_radiation, checked against what it must
conserve."""

from collections import namedtuple

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.processes import solar_radiation as solar  # noqa: E402
from clm_ml.state import PatchLayers  # noqa: E402

N_SLOTS = 6  # layer slots; the profiles below have N_SLOTS + 1 entries (index 0 the ground)
NCAN, NTOP, NBOT = 5, 3, 2  # Fortran-style counts: leafy layers 2..3, padding slot 6
Params = namedtuple(
    "Params",
    "leaf_reflectance leaf_transmittance stem_reflectance stem_transmittance "
    "leaf_angle_index clumping_index",
)
PARAMS = Params(
    jnp.array([0.0, 0.06, 0.42]), jnp.array([0.0, 0.04, 0.43]), jnp.array([0.0, 0.21, 0.49]),
    jnp.array([0.0, 0.001, 0.001]), 0.53, 0.85,
)  # fmt: skip


def _profiles():
    """Leaf and stem area of the ground and layer slots (leafy layers only)."""
    dlai = np.zeros(N_SLOTS + 1)
    dsai = np.zeros(N_SLOTS + 1)
    dlai[NBOT : NTOP + 1] = [0.8, 1.2]
    dsai[NBOT : NTOP + 1] = [0.4, 0.3]
    return dlai, dsai


def _physics(leaf_angle_index=PARAMS.leaf_angle_index):
    dlai, dsai = _profiles()
    layers = PatchLayers(
        n_layers=jnp.asarray(NCAN), n_within=jnp.asarray(NTOP),
        first_leafy=jnp.asarray(NBOT - 1), dz=jnp.ones(N_SLOTS), zs=jnp.ones(N_SLOTS),
        zw=jnp.ones(N_SLOTS + 1),
    )  # fmt: skip
    dpai = jnp.asarray((dlai + dsai)[1:])
    optics = solar.canopy_optics(
        0.6, jnp.asarray(dlai[1:]), jnp.asarray(dsai[1:]), dpai, layers,
        *(x[1:] for x in PARAMS[:4]), leaf_angle_index, PARAMS.clumping_index,
    )  # fmt: skip
    sky_b, sky_d, alb = jnp.array([300.0, 280.0]), jnp.array([60.0, 40.0]), jnp.array([0.1, 0.2])
    fluxes = solar.two_stream(optics, dpai, sky_b, sky_d, alb, alb, layers)
    return optics, fluxes, dpai, sky_b, sky_d


# planophile (trees) .. erectophile (grasses and crops, -0.3); 0.0 is moved to 0.01 and
# values beyond [-0.4, 0.6] to the limits, as in CLM
@pytest.mark.parametrize("chil", [-0.5, -0.4, -0.3, 0.0, 0.01, 0.25, 0.53, 0.6])
def test_avmu_is_the_fortran_formula_for_any_leaf_angle_index(chil):
    """The average inverse diffuse optical depth with phi2 as it is, negative for a negative
    leaf angle index (a guard max(phi2, 1e-10) would give 1 / (2 phi1), 0.757 instead of
    1.077 at -0.3, with amplified rounding noise), and energy is conserved."""
    optics, fluxes, _, _, _ = _physics(chil)
    x = min(max(chil, -0.4), 0.6)
    x = 0.01 if abs(x) <= 0.01 else x
    phi1 = 0.5 - 0.633 * x - 0.330 * x * x
    phi2 = 0.877 * (1.0 - 2.0 * phi1)
    assert (phi2 < 0) == (x < 0)
    expected = (1.0 - phi1 / phi2 * np.log((phi1 + phi2) / phi1)) / phi2
    np.testing.assert_allclose(np.asarray(optics.avmu)[NBOT - 1 : NTOP], expected, rtol=1e-12)
    assert np.all(np.abs(np.asarray(fluxes.error)) < 1e-9)


def test_avmu_is_smooth_in_a_negative_leaf_angle_index():
    """No rounding noise: second differences of avmu over steps of 1e-6 are at round-off."""
    avmu = [float(_physics(-0.3 + k * 1e-6)[0].avmu[NBOT - 1]) for k in range(-2, 3)]
    assert np.abs(np.diff(avmu, 2)).max() < 1e-11
    assert avmu[2] == pytest.approx(1.0773, abs=1e-4)


def test_two_stream_conserves_energy():
    optics, f, dpai, sky_b, sky_d = _physics()
    assert float(optics.sunlit_fraction_error) == 0.0
    incoming = sky_b + sky_d
    np.testing.assert_allclose(f.swveg + f.swsoi + f.albcan * incoming, incoming, rtol=1e-12)
    np.testing.assert_allclose(f.swvegsun + f.swvegsha, f.swveg, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(f.error), 0.0, atol=1e-9)


def test_direct_beam_onto_the_ground_is_beer_law():
    optics, f, dpai, sky_b, _ = _physics()
    k = float(optics.kb[NBOT - 1])  # same in every leafy layer
    expected = np.exp(-k * PARAMS.clumping_index * float(dpai.sum()))
    np.testing.assert_allclose(optics.tbi_ground, expected, rtol=1e-12)
    np.testing.assert_allclose(f.swbeam[0], sky_b * expected, rtol=1e-12)


def test_layers_without_leaves_get_zero():
    optics, f, *_ = _physics()
    leafy = np.zeros(N_SLOTS, bool)
    leafy[NBOT - 1 : NTOP] = True
    for v in (optics.kb, optics.fracsun, optics.tbi, f.swleaf, f.swbeam[1:]):
        assert np.all(np.asarray(v)[~leafy] == 0.0)
        assert np.all(np.asarray(v)[leafy] != 0.0)
    np.testing.assert_allclose(solar.absorbed_par(f.swleaf), f.swleaf[..., 0] * 4.6, rtol=0)


def test_norman_is_registered_but_not_supported():
    assert set(solar.SOLAR_RADIATION) == {"two_stream", "norman"}
    with pytest.raises(NotImplementedError, match="norman is not supported"):
        solar.SOLAR_RADIATION["norman"]()
