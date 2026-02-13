"""Debug level 3 descent."""
import sys
from pathlib import Path
import xarray as xr
import numpy as np
import jax.numpy as jnp
import jax
from jax import lax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from clubb_core.mixing_length import compute_mixing_length
from clubb_core.grid_class import Grid
from clubb_core.saturation import sat_mixrat_liq
from clubb_core.clubb_constants import core_rknd


# Load test data
tests_dir = Path(__file__).parent / 'tests/mixing_length_test'
path = tests_dir / "sample_data" / "bomex_mixing_length_calculation_samples.nc"
data = xr.open_dataset(path)
sample = data.isel(samples=0).squeeze().expand_dims(dim="column", axis=0)

# Extract needed values
nzt = sample.attrs["nzt"]
i = 0  # column index
k = 3  # level to debug

zt = jnp.array(sample.zt.expand_dims(dim="column", axis=0).values)
dzm = jnp.array(sample.dzm.values)
tke_i = jnp.array(sample.em.values)  # TKE
thlm = jnp.array(sample.thlm.values)
rtm = jnp.array(sample.rtm.values)
exner = jnp.array(sample.exner.values)
p_in_Pa = jnp.array(sample.p_in_Pa.values)
thvm = jnp.array(sample.thvm.values)
thv_ds = jnp.array(sample.thv_ds.values)

# Constants
mu = float(sample.mu.values.item())
saturation_formula = int(sample.saturation_formula.values.item())
zero = jnp.array(0.0, dtype=core_rknd)
one = jnp.array(1.0, dtype=core_rknd)
one_half = jnp.array(0.5, dtype=core_rknd)
ep1 = jnp.array(0.608, dtype=core_rknd)
grav = jnp.array(9.81, dtype=core_rknd)
Lv = jnp.array(2.5e6, dtype=core_rknd)
Cp = jnp.array(1005.0, dtype=core_rknd)
zero_threshold = jnp.array(1.0e-30, dtype=core_rknd)

# Compute derived quantities
exp_mu_dzm = jnp.exp(-mu * dzm)
entrain_coef = (one - exp_mu_dzm) / mu
grav_on_thvm = grav / thvm
Lv_coef = Lv / (Cp * exner)
Lv2_coef = (Lv / Cp)**2

# Precalculate for downward motion
thl_par_j_precalc_down = jnp.zeros((1, nzt), dtype=core_rknd)
rt_par_j_precalc_down = jnp.zeros((1, nzt), dtype=core_rknd)

for j in range(0, nzt-1):
    j_zm = j + 1
    thl_par_j_precalc_down = thl_par_j_precalc_down.at[:, j].set(
        thlm[:, j] - thlm[:, j+1] * exp_mu_dzm[:, j_zm]
        - (thlm[:, j] - thlm[:, j+1]) * entrain_coef[:, j_zm]
    )
    rt_par_j_precalc_down = rt_par_j_precalc_down.at[:, j].set(
        rtm[:, j] - rtm[:, j+1] * exp_mu_dzm[:, j_zm]
        - (rtm[:, j] - rtm[:, j+1]) * entrain_coef[:, j_zm]
    )

# Calculate initial parcel properties for downward motion
thl_par_1_down = jnp.zeros((1, nzt), dtype=core_rknd)
rt_par_1_down = jnp.zeros((1, nzt), dtype=core_rknd)
dCAPE_dz_1_down = jnp.zeros((1, nzt), dtype=core_rknd)
CAPE_incr_1_down = jnp.zeros((1, nzt), dtype=core_rknd)

for j in range(0, nzt-1):
    j_zm = j + 1
    thl_par_1_down = thl_par_1_down.at[:, j].set(
        thlm[:, j] - (thlm[:, j] - thlm[:, j+1]) * entrain_coef[:, j_zm]
    )
    rt_par_1_down = rt_par_1_down.at[:, j].set(
        rtm[:, j] - (rtm[:, j] - rtm[:, j+1]) * entrain_coef[:, j_zm]
    )
    tl_down = thl_par_1_down[:, j] * exner[:, j]
    rsatl_down = sat_mixrat_liq(tl_down, p_in_Pa[:, j], saturation_formula)
    tl_sqd = tl_down**2
    s_par_down = ((rt_par_1_down[:, j] - rsatl_down) * tl_sqd
                  / (tl_sqd + Lv2_coef * rsatl_down))
    rc_par_down = jnp.maximum(s_par_down, zero_threshold)
    thv_par_down = (thl_par_1_down[:, j] + ep1 * thv_ds[:, j] * rt_par_1_down[:, j]
                    + Lv_coef[:, j] * rc_par_down)
    dCAPE_dz_1_down = dCAPE_dz_1_down.at[:, j].set(
        grav_on_thvm[:, j] * (thv_par_down - thvm[:, j])
    )
    CAPE_incr_1_down = CAPE_incr_1_down.at[:, j].set(
        one_half * dCAPE_dz_1_down[:, j] * dzm[:, j_zm]
    )

# Manual descent for level k=3
k_minus_1 = max(k - 1, 0)
init_tke = tke_i[i, k] - CAPE_incr_1_down[i, k_minus_1]
print(f"\nLevel k={k} (zt={float(zt[i,k])}m)")
print(f"Initial TKE: {float(tke_i[i,k]):.6f}")
print(f"CAPE_incr_1_down[{k_minus_1}]: {float(CAPE_incr_1_down[i,k_minus_1]):.6f}")
print(f"TKE after first step: {float(init_tke):.6f}")
print(f"\nStarting descent from j={k-2}...")

# Initialize parcel
j = k - 2
thl_par = thl_par_1_down[i, k]
rt_par = rt_par_1_down[i, k]
tke_curr = init_tke
tke_prev = tke_i[i, k]
dCAPE_prev = dCAPE_dz_1_down[i, k_minus_1]

iteration = 0
while tke_curr > 0 and j >= 0 and iteration < 10:
    print(f"\nIteration {iteration}: j={j}")
    j_zm = j

    # Update parcel properties
    thl_par = thl_par_j_precalc_down[i, j] + thl_par * exp_mu_dzm[i, j_zm]
    rt_par = rt_par_j_precalc_down[i, j] + rt_par * exp_mu_dzm[i, j_zm]

    # Calculate buoyancy
    tl_par = thl_par * exner[i, j]
    rsatl_par = sat_mixrat_liq(tl_par, p_in_Pa[i, j], saturation_formula)
    tl_par_sqd = tl_par**2
    s_par = ((rt_par - rsatl_par) * tl_par_sqd / (tl_par_sqd + Lv2_coef * rsatl_par))
    rc_par = jnp.maximum(s_par, zero_threshold)
    thv_par = (thl_par + ep1 * thv_ds[i, j] * rt_par + Lv_coef[i, j] * rc_par)
    dCAPE_dz = grav_on_thvm[i, j] * (thv_par - thvm[i, j])

    # CAPE increment
    CAPE_incr = one_half * (dCAPE_dz + dCAPE_prev) * dzm[i, j_zm]

    # Update TKE
    tke_new = tke_curr - CAPE_incr

    print(f"  dCAPE/dz = {float(dCAPE_dz):.6f}")
    print(f"  CAPE_incr = {float(CAPE_incr):.6f}")
    print(f"  TKE: {float(tke_curr):.6f} -> {float(tke_new):.6f}")

    if tke_new <= 0:
        print(f"  TKE exhausted! Stopping at j={j}")
        break

    # Continue to next level
    tke_prev = tke_curr
    tke_curr = tke_new
    dCAPE_prev = dCAPE_dz
    j = j - 1
    iteration += 1

j_final = j if tke_curr <= 0 else (j - 1 if j >= 0 else j)
j_reached = np.clip(j_final + 1, 0, k)
Lscale_computed = 0.1 + float(zt[i, k] - zt[i, j_reached])

print(f"\nFinal state:")
print(f"  j_final = {j_final}")
print(f"  j_reached = {j_reached} (zt={float(zt[i,j_reached])}m)")
print(f"  Lscale = {Lscale_computed:.2f}m")
print(f"  Fortran Lscale = 120.10m")
