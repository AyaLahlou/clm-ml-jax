"""
Tests that match the Python implementation of `compute_mixing_length` vs. Fortran.

Refer to `bomex_mixing_length` fixture for details on how the reference data was
collected.
"""
import sys
from pathlib import Path
import xarray as xr
import numpy as np
import jax.numpy as jnp
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from clubb_core.mixing_length import compute_mixing_length
from clubb_core.grid_class import Grid


def sample_dataset_to_arguments(sample_data: xr.Dataset) -> dict[str, Any]:
    """
    Translate single sample data to argument list.

    We do it manually since we expect the argument names on Python side to change
    and we need to provide a translation. Also we probably don't want to provide
    to invocation more arguments that it accepts!

    Also we convert the arguments to their expected types.
    """
    nzm = sample_data.attrs["nzm"]
    nzt = sample_data.attrs["nzt"]
    ngrdcol = 1  # Data was sampled in one column problem
    
    # Convert grid arrays to JAX arrays and ensure proper shape
    zm = jnp.array(sample_data["zm"].expand_dims(dim="column", axis=0).to_numpy())
    zt = jnp.array(sample_data["zt"].expand_dims(dim="column", axis=0).to_numpy())
    dzm = jnp.array(sample_data["dzm"].to_numpy())
    dzt = jnp.array(sample_data["dzt"].to_numpy())
    invrs_dzm = jnp.array(sample_data["invrs_dzm"].to_numpy())
    invrs_dzt = jnp.array(sample_data["invrs_dzt"].to_numpy())
    
    # Create Grid object with dummy interpolation weights (not used in mixing_length)
    gr = Grid(
        nzm=nzm,
        nzt=nzt,
        ngrdcol=ngrdcol,
        zm=zm,
        zt=zt,
        dzm=dzm,
        dzt=dzt,
        invrs_dzm=invrs_dzm,
        invrs_dzt=invrs_dzt,
        weights_zt2zm=jnp.zeros((ngrdcol, nzm, 2)),  # Dummy weights
        weights_zm2zt=jnp.zeros((ngrdcol, nzt, 2)),  # Dummy weights
        k_lb_zm=0,
        k_ub_zm=nzm-1,
        k_lb_zt=0,
        k_ub_zt=nzt-1,
        grid_dir_indx=1,
        grid_dir=1.0,
    )
    
    # Create a minimal err_info object
    class ErrInfo:
        pass
    err_info = ErrInfo()
    
    kwargs = {
        "nzm": nzm,
        "nzt": nzt,
        "ngrdcol": ngrdcol,
        "gr": gr,
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


def check_single_case(sample_data):
    """Run comparison for a single `compute_mixing_length` Fortran sample."""
    # Pull all attributes and
    kwargs = sample_dataset_to_arguments(sample_data)
    Lscale, Lscale_up, Lscale_down, err_info = compute_mixing_length(**kwargs)

    # The maximum error in the test set is about 800 ULPs (~ 1.e-13 relative)
    # but we allow for extra margin (e.g. for future samples)
    tol = 1e-10
    np.testing.assert_allclose(np.array(Lscale), sample_data["Lscale"], rtol=tol)
    np.testing.assert_allclose(np.array(Lscale_up), sample_data["Lscale_up"], rtol=tol)
    np.testing.assert_allclose(np.array(Lscale_down), sample_data["Lscale_down"], rtol=tol)


def test_mixing_length(bomex_mixing_length):
    """
    Test all `compute_mixing_length` samples.

    We need to remove degenerate spatial dimensions (x & y) and create
    dummy `column` dimension to match the Python interface.
    """
    for _, sample_data in bomex_mixing_length.groupby("samples"):
        sample_data = sample_data.squeeze().expand_dims(dim="column", axis=0)
        check_single_case(sample_data)
