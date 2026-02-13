"""Test specific levels to debug downward descent."""
import sys
from pathlib import Path
import xarray as xr
import numpy as np
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from clubb_core.mixing_length import compute_mixing_length
from clubb_core.grid_class import Grid


def sample_dataset_to_arguments(sample_data: xr.Dataset):
    """Convert sample data to function arguments."""
    nzm = sample_data.attrs["nzm"]
    nzt = sample_data.attrs["nzt"]
    ngrdcol = 1

    zm = jnp.array(sample_data["zm"].expand_dims(dim="column", axis=0).to_numpy())
    zt = jnp.array(sample_data["zt"].expand_dims(dim="column", axis=0).to_numpy())
    dzm = jnp.array(sample_data["dzm"].to_numpy())
    dzt = jnp.array(sample_data["dzt"].to_numpy())
    invrs_dzm = jnp.array(sample_data["invrs_dzm"].to_numpy())
    invrs_dzt = jnp.array(sample_data["invrs_dzt"].to_numpy())

    gr = Grid(
        nzm=nzm, nzt=nzt, ngrdcol=ngrdcol,
        zm=zm, zt=zt, dzm=dzm, dzt=dzt,
        invrs_dzm=invrs_dzm, invrs_dzt=invrs_dzt,
        weights_zt2zm=jnp.zeros((ngrdcol, nzm, 2)),
        weights_zm2zt=jnp.zeros((ngrdcol, nzt, 2)),
        k_lb_zm=0, k_ub_zm=nzm-1,
        k_lb_zt=0, k_ub_zt=nzt-1,
        grid_dir_indx=1, grid_dir=1.0,
    )

    class ErrInfo:
        pass
    err_info = ErrInfo()

    kwargs = {
        "nzm": nzm, "nzt": nzt, "ngrdcol": ngrdcol, "gr": gr,
        "thvm": jnp.array(sample_data["thvm"].to_numpy()),
        "thlm": jnp.array(sample_data["thlm"].to_numpy()),
        "rtm": jnp.array(sample_data["rtm"].to_numpy()),
        "em": jnp.array(sample_data["em"].to_numpy()),
        "Lscale_max": jnp.array(sample_data["Lscale_max"].to_numpy()),
        "p_in_Pa": jnp.array(sample_data["p_in_Pa"].to_numpy()),
        "exner": jnp.array(sample_data["exner"].to_numpy()),
        "thv_ds": jnp.array(sample_data["thv_ds"].to_numpy()),
        "mu": float(sample_data["mu"].to_numpy().item()),
        "lmin": float(sample_data["lmin"].to_numpy().item()),
        "saturation_formula": int(sample_data["saturation_formula"].to_numpy().item()),
        "l_implemented": bool(sample_data["l_implemented"].to_numpy().item()),
        "err_info": err_info,
    }
    return kwargs


# Load test data
tests_dir = Path(__file__).parent / 'tests/mixing_length_test'
path = tests_dir / "sample_data" / "bomex_mixing_length_calculation_samples.nc"
data = xr.open_dataset(path)

# Get sample
sample_data = data.isel(samples=0).squeeze().expand_dims(dim="column", axis=0)

# Run JAX implementation
print("Running JAX implementation...")
kwargs = sample_dataset_to_arguments(sample_data)
Lscale_jax, Lscale_up_jax, Lscale_down_jax, _ = compute_mixing_length(**kwargs)

# Get Fortran reference
Lscale_down_fort = sample_data["Lscale_down"].values[0, :]
zt = sample_data["zt"].values

# Convert JAX arrays to numpy
Lscale_down_jax_np = np.array(Lscale_down_jax[0, :])

print("\n" + "="*70)
print("COMPARISON OF DOWNWARD MIXING LENGTH")
print("="*70)
print(f"{'k':>3} {'zt[m]':>8} {'Fort[m]':>10} {'JAX[m]':>10} {'Diff[m]':>10} {'alt_Fort':>10} {'alt_JAX':>10}")
print("-"*70)
for k in [0, 1, 2, 3, 4, 5, 10, 15, 20, 40, 60, 80, 86]:
    if k < len(zt):
        zt_k = zt[k]
        fort = Lscale_down_fort[k]
        jax_val = Lscale_down_jax_np[k]
        diff = jax_val - fort
        alt_fort = zt_k - fort
        alt_jax = zt_k - jax_val
        print(f"{k:3d} {zt_k:8.1f} {fort:10.2f} {jax_val:10.2f} {diff:10.2f} {alt_fort:10.1f} {alt_jax:10.1f}")
