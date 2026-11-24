"""Restart files as netCDF (clm_ml.io.restart): every value comes back exactly,
the file follows the documented schema, and a broken file falls back to the previous one."""

import datetime as dt

import numpy as np
import pytest

from clm_ml.io.restart import FORMAT_VERSION, RestartError, read_restart, write_restart
from clm_ml.state import CanopyState, SoilState, State


def _state(n_patch=3, n_layer=4, n_soil=5, seed=0):
    rng = np.random.default_rng(seed)

    def r(*shape):
        return rng.normal(size=(n_patch,) + shape)

    canopy = CanopyState(
        air_temperature=r(n_layer), vapor_pressure=r(n_layer), co2=r(n_layer),
        leaf_temperature=r(n_layer, 2), leaf_water_potential=r(n_layer, 2),
        intercepted_water=r(n_layer), ground_temperature=r(), acclimation_temperature=r(),
        acclimation_steps=np.full(n_patch, 7.0),
    )  # fmt: skip
    canopy = canopy._replace(air_temperature=canopy.air_temperature.copy())
    canopy.air_temperature[0, -1] = 1e36  # padding slots hold the fill value
    return State(canopy=canopy, soil=SoilState(temperature=r(n_soil)))


def _content(state):
    mean = {"sum": {"latent_heat_flux": np.array([[1.5, 2.5, np.nan]]),
                    "gridcell:latent_heat_flux": np.array([[4.0]])},
            "count": np.array([3.0])}  # fmt: skip
    inst = {"value": {"air_temperature": np.array([[[290.0, np.inf], [np.nan, -np.inf]]])}}
    return {
        "version": FORMAT_VERSION,
        "time": dt.datetime(2007, 5, 2, 0, 30),
        "state": state,
        "collectors": [
            (dt.date(2007, 5, 1), dt.datetime(2007, 5, 2), mean),
            None,
            (46, dt.datetime(2007, 5, 1, 23, 30), inst),
            ((2007, 5), dt.datetime(2007, 6, 1), mean),
            (dt.datetime(2007, 5, 1, 23), dt.datetime(2007, 5, 2), mean),
        ],
        "streams": [
            ("d", "mean"),
            ("x", "max"),
            ("s", "instantaneous"),
            ("m", "mean"),
            ("h", "mean"),
        ],
        "alive": np.array([True, False, True]),
        "layout": (2, 4, 3, 1),
        "extra": {"writers": [{"label": "2007-05-01", "started": ["2007-05-01"]}, None]},
    }


def _same(a, b):
    if isinstance(a, dict):
        assert sorted(a) == sorted(b)
        for k in a:
            _same(a[k], b[k])
    else:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert np.asarray(a).dtype == np.asarray(b).dtype


def test_round_trip_is_exact(tmp_path):
    content = _content(_state())
    write_restart(tmp_path / "r.nc", content, calendar="NOLEAP")
    back = read_restart(tmp_path / "r.nc")
    assert back["time"] == content["time"] and back["calendar"] == "NOLEAP"
    assert back["layout"] == content["layout"]
    assert back["streams"] == content["streams"]
    assert back["extra"] == content["extra"]
    np.testing.assert_array_equal(back["alive"], content["alive"])
    import jax

    for a, b in zip(jax.tree.leaves(back["state"]), jax.tree.leaves(content["state"])):
        np.testing.assert_array_equal(a, b)
        assert a.dtype == np.float64
    for got, want in zip(back["collectors"], content["collectors"]):
        if want is None:
            assert got is None
            continue
        assert got[0] == want[0] and type(got[0]) is type(want[0])
        assert got[1] == want[1]
        _same(got[2], want[2])


def test_schema(tmp_path):
    import netCDF4 as nc

    write_restart(tmp_path / "r.nc", _content(_state()))
    with nc.Dataset(tmp_path / "r.nc") as ds:
        assert ds.format_version == FORMAT_VERSION
        assert dict((k, len(v)) for k, v in ds.dimensions.items()) == {
            "patch": 3, "layer": 4, "leaf": 2, "soil_level": 5}  # fmt: skip
        temp = ds.groups["state"].variables["canopy_leaf_temperature"]
        assert temp.dimensions == ("patch", "layer", "leaf") and temp.units == "K"
        assert ds.groups["state"].variables["soil_temperature"].dimensions == (
            "patch", "soil_level")  # fmt: skip
        assert ds.variables["alive"].dimensions == ("patch",)
        assert sorted(ds.groups["streams"].groups) == [f"stream_{i}" for i in range(5)]
        s0 = ds.groups["streams"].groups["stream_0"]
        assert (s0.getncattr("name"), s0.statistic) == ("d", "mean")
        var = next(iter(s0.groups["g0"].variables.values()))  # sum of latent_heat_flux
        assert var.dimensions == ("record", "patch") and var.units == "W m-2"


def test_pickle_restart_is_refused_clearly(tmp_path):
    import pickle

    (tmp_path / "r.nc").write_bytes(pickle.dumps({"version": 5}))
    with pytest.raises(RestartError, match="a pickle restart"):
        read_restart(tmp_path / "r.nc")


def test_previous_file_kept_and_used_when_the_file_is_broken(tmp_path):
    path = tmp_path / "r.nc"
    first, second = _content(_state(seed=1)), _content(_state(seed=2))
    second["time"] = dt.datetime(2007, 5, 3)
    write_restart(path, first)
    write_restart(path, second)
    assert read_restart(path)["time"] == second["time"]
    assert read_restart(tmp_path / "r.nc.prev")["time"] == first["time"]
    path.write_bytes(b"truncated")  # e.g. killed while writing
    assert read_restart(path)["time"] == first["time"]  # falls back to .prev
    (tmp_path / "r.nc.prev").unlink()
    with pytest.raises(RestartError, match="no readable restart"):
        read_restart(path)
