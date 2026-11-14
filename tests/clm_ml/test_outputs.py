"""Named outputs and netCDF writing, built from the reference's step fields (no model run)."""

import numpy as np
import pytest

from clm_ml.experiment import load_experiment
from clm_ml.outputs import VARIABLES, Outputs

from .conftest import PROJECT_ROOT

REFERENCE = PROJECT_ROOT / "tests" / "regression" / "reference" / "CHATS7_50steps.npz"


@pytest.fixture
def outputs(chats_run_dir):
    """The named variables of the reference's step fields (as a run's step stream)."""
    import datetime as dt

    from clm_ml.outputs import _Raw, domain_meta

    with np.load(REFERENCE) as data:
        raw = {k: data[k] for k in data.files if not k.startswith("meta.")}
    run = load_experiment(chats_run_dir)
    r = _Raw(raw, run.config.canopy_layers.max_layers)
    n_steps = raw["layers.n_layers"].shape[0]
    step = dt.timedelta(seconds=run.step_seconds)
    records = {v.name: np.asarray(v.compute(r), dtype=float) for v in VARIABLES}
    records["time"] = np.array([run.start + (k + 1) * step for k in range(n_steps)],
                               "datetime64[ns]")  # fmt: skip
    records["soil_temperature"] = raw["state.soil_temperature"]
    meta = dict(domain_meta(run.domain), soil_depth=np.linspace(0, 10, 25)[None, :])
    return Outputs.from_records(records, meta)


def test_every_variable_has_expected_shape(outputs):
    sizes = {"layer": 50, "leaf_class": 2, "leaf_group": 3, "band": 2}
    for spec in VARIABLES:
        expected = (50, 1) + tuple(sizes[d] for d in spec.dims)
        assert outputs.data[spec.name].shape == expected, spec.name


def test_profiles_are_padded_with_nan(outputs):
    t = outputs.air_temperature[10, 0]
    assert np.isfinite(t[:46]).all() and np.isnan(t[46:]).all()
    leaf_t = outputs.leaf_temperature[10, 0]  # leaves only within the canopy (20 layers)
    assert np.isnan(leaf_t[20:]).all()


def test_gridcell_mean_with_one_patch_equals_patch(outputs):
    means = outputs.gridcell_mean()
    np.testing.assert_array_equal(means["latent_heat_flux"], outputs.latent_heat_flux)
    assert "air_temperature" not in means  # profiles are per patch only


def test_netcdf_round_trip(outputs, tmp_path):
    import xarray as xr

    written = outputs.to_netcdf(tmp_path / "run.nc")
    assert [p.name for p in written] == ["run_patch.nc", "run_gridcell.nc"]
    with xr.open_dataset(written[0]) as ds:
        assert ds["gpp"].attrs["units"] == "umol CO2 m-2 s-1"
        assert ds["pft"].values.tolist() == ["broadleaf_deciduous_temperate_tree"]
        np.testing.assert_array_equal(ds["latent_heat_flux"].values, outputs.latent_heat_flux)
        assert str(ds["time"].values[0]).startswith("2007-05-01T00:30")


def test_select(outputs):
    small = outputs.select(["gpp", "latent_heat_flux"])
    assert set(small.data) == {"gpp", "latent_heat_flux"}
    with pytest.raises(KeyError, match="unknown output variable"):
        outputs.select(["nope"])


def _grid_outputs(n_time=2):
    """Three gridcells (four patches) of a grid run on a 3 x 4 latitude-longitude map."""
    from clm_ml.domain import LatLonMap

    rng = np.random.default_rng(0)
    gridcell_map = LatLonMap(
        lat=np.array([-10.0, 0.0, 10.0]),
        lon=np.array([-5.0, 0.0, 5.0, 10.0]),
        index=np.array([[0, 3], [2, 0], [1, 1]]),
    )
    meta = {"patch_gridcell": [0, 1, 1, 2], "patch_weight": [1.0, 0.25, 0.75, 1.0],
            "patch_pft": ["c3_grass"] * 4, "gridcell_names": ["a", "b", "c"],
            "soil_depth": np.zeros((4, 1)), "gridcell_map": gridcell_map,
            "gridcell_area": {"area": np.array([1e4, 2e4, 3e4]),
                              "land_fraction": np.array([1.0, 0.5, 0.9]),
                              "vegetated_fraction": np.array([0.8, 0.4, 1.0])}}  # fmt: skip
    records = {
        "time": np.array(["2000-01-02", "2000-01-03"][:n_time], "datetime64[ns]"),
        "gpp": rng.random((n_time, 4)),
        "gpp_by_leaf_group": rng.random((n_time, 4, 3)),
        "air_temperature": rng.random((n_time, 4, 5)),  # a profile: patch level only
    }
    return Outputs.from_records(records, meta)


def test_grid_gridcell_level_is_on_the_latlon_map():
    """Decision 107: a grid run's gridcell values at their latitude and longitude, NaN where
    no gridcell was run, extra dimensions before (lat, lon)."""
    out = _grid_outputs()
    ds = out.to_xarray("gridcell")
    means = out.gridcell_mean()
    assert ds["gpp"].dims == ("time", "lat", "lon")
    assert ds["gpp_by_leaf_group"].dims == ("time", "leaf_group", "lat", "lon")
    assert "air_temperature" not in ds and "gridcell" not in ds.dims
    for g, (i, j) in enumerate(out.gridcell_map.index):
        np.testing.assert_array_equal(ds["gpp"].values[:, i, j], means["gpp"][:, g])
        np.testing.assert_array_equal(
            ds["gpp_by_leaf_group"].values[:, :, i, j], means["gpp_by_leaf_group"][:, g]
        )
    assert int(ds["run_mask"].sum()) == 3
    for field, name in (("area", "area"), ("land_fraction", "landfrac"),
                        ("vegetated_fraction", "vegetated_fraction")):  # fmt: skip
        assert ds[name].dims == ("lat", "lon")
        np.testing.assert_array_equal(
            ds[name].values[out.gridcell_map.index[:, 0], out.gridcell_map.index[:, 1]],
            out.gridcell_area[field],
        )
        assert np.isnan(ds[name].values[ds["run_mask"].values == 0]).all()
    assert np.isnan(ds["gpp"].values[:, ds["run_mask"].values == 0]).all()
    assert ds["lat"].attrs["units"] == "degrees_north"
    np.testing.assert_array_equal(ds["lon"].values, [-5.0, 0.0, 5.0, 10.0])
    assert out.to_xarray("patch")["gpp"].dims == ("time", "patch")  # patches stay per patch


def test_grid_gridcell_file_continues_after_restart(tmp_path):
    import xarray as xr

    out = _grid_outputs()
    first, second = out.take(np.array([True, False])), out.take(np.array([False, True]))
    first.to_netcdf(tmp_path / "run.nc", levels=("gridcell",))
    (path,) = second.to_netcdf(tmp_path / "run.nc", levels=("gridcell",), append=True)
    with xr.open_dataset(path) as ds:
        assert ds["gpp"].shape == (2, 3, 4) and ds["run_mask"].dims == ("lat", "lon")
        np.testing.assert_array_equal(ds["gpp"].values, out.to_xarray("gridcell")["gpp"].values)


def test_latlon_map_of_a_region():
    """The map is the region's box on the grid, longitudes in the region's convention."""
    from types import SimpleNamespace

    from clm_ml.grid import latlon_map

    lat1, lon1 = np.arange(-90.0, 91.0, 10.0), np.arange(0.0, 360.0, 10.0)
    lat2, lon2 = np.meshgrid(lat1, lon1, indexing="ij")
    rows, cols = np.array([13, 14, 14]), np.array([35, 0, 2])  # 40N 350E, 50N 0E, 50N 20E
    whole = latlon_map(lat2, lon2, rows, cols, "global")
    assert whole.lat.size == 19 and whole.lon.size == 36
    np.testing.assert_array_equal(whole.index, np.stack([rows, cols], 1))
    box = latlon_map(lat2, lon2, rows, cols, SimpleNamespace(lat=(35, 55), lon=(-20, 20)))
    np.testing.assert_array_equal(box.lat, [40.0, 50.0])
    np.testing.assert_array_equal(box.lon, [-20.0, -10.0, 0.0, 10.0, 20.0])
    np.testing.assert_array_equal(box.index, [[0, 1], [1, 2], [1, 4]])
    assert latlon_map(lat2 + lon2 / 1000, lon2, rows, cols, "global") is None  # not regular
