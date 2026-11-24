"""
Named outputs and output files of the shared regression run.

The run itself is compared with the reference in ``test_end_to_end_regression.py``; this
checks that the named outputs (stream ``step``) agree with the raw step fields (validation
stream ``fields``) and that the netCDF and legacy ASCII files are written.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression


def test_named_outputs_match_step_fields(current):
    _, streams, _ = current
    outputs, raw = streams["step"], streams["fields"].raw
    np.testing.assert_array_equal(
        outputs.latent_heat_flux, raw["diagnostics.lhflx_canopy"].astype(float)
    )
    np.testing.assert_array_equal(outputs.gpp, raw["diagnostics.gppveg_canopy"].astype(float))
    ncan = int(raw["layers.n_layers"][0, 0])
    profile = outputs.air_temperature[0, 0]
    np.testing.assert_array_equal(profile[:ncan], raw["state.air_temperature"][0, 0, :ncan])
    assert np.isnan(profile[ncan:]).all()  # padding above the 46 real layers


def test_output_files_written(current):
    import xarray as xr

    run, streams, out_dir = current
    outputs = streams["step"]
    legacy = sorted(p.name for p in out_dir.glob("*.out"))
    assert len(legacy) == 6, legacy
    with xr.open_dataset(out_dir / "CHATS7_step_2007-05-01_patch.nc") as ds:
        assert ds["latent_heat_flux"].attrs["units"] == "W m-2"
        assert ds.sizes == {
            "time": 50, "patch": 1, "layer": 50, "leaf_class": 2, "leaf_group": 3, "band": 2,
            "soil_layer": outputs.soil_temperature.shape[-1],
        }  # fmt: skip
        np.testing.assert_array_equal(ds["latent_heat_flux"].values, outputs.latent_heat_flux)
    with xr.open_dataset(out_dir / "CHATS7_step_2007-05-01_gridcell.nc") as ds:
        # One patch with weight 1: the gridcell mean equals the patch value.
        np.testing.assert_array_equal(ds["latent_heat_flux"].values, outputs.latent_heat_flux)
