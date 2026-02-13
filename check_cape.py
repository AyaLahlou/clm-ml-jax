"""Check CAPE_incr_1_down values."""
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
sample_data = data.isel(samples=0).squeeze().expand_dims(dim="column", axis=0)

# Add instrumentation to check CAPE_incr_1_down
import clubb_core.mixing_length as ml_module

# Monkey-patch to capture CAPE_incr_1_down
original_compute = ml_module.compute_mixing_length

captured_cape = {}

def instrumented_compute(*args, **kwargs):
    result = original_compute(*args, **kwargs)
    return result

# Just run normally and extract from source
kwargs = sample_dataset_to_arguments(sample_data)
tke = kwargs['em']

print("TKE at higher levels:")
print("k    zt[m]    TKE[i,k]")
for k in [10, 11, 12, 13, 14, 15, 20]:
    print(f"{k:2d}  {float(kwargs['gr'].zt[0,k]):6.1f}  {float(tke[0,k]):.6e}")
