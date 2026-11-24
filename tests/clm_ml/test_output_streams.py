"""Output streams: records, accumulation in a compiled scan, and records spanning chunks."""

import datetime as dt

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clm_ml import output_streams as S

jax.config.update("jax_enable_x64", True)

STEP = 1800
START = dt.datetime(2007, 5, 1, 0, 0)


def _starts(n, first=0):
    return [START + dt.timedelta(seconds=STEP * (first + k)) for k in range(n)]


def _run_chunk(spec, first, n, value_of):
    """Scan over steps first..first+n-1 accumulating x = value_of(step); returns buffers."""
    rec = S.chunk_records(_starts(n, first), first, STEP, spec.frequency)
    shapes = {"x": (2,)}
    buffers = S.init_buffers(spec.statistic, len(rec.keys), shapes, jnp)

    def body(buf, inp):
        k, slot = inp
        return S.accumulate(spec.statistic, buf, slot, {"x": value_of(k)}), None

    steps = jnp.arange(first, first + n, dtype=float)
    buffers, _ = jax.jit(lambda b: jax.lax.scan(body, b, (steps, jnp.asarray(rec.index))))(buffers)
    return rec, buffers


def _collect(spec, chunks, value_of):
    collector = S.StreamCollector(spec)
    out = []
    for first, n in chunks:
        out += collector.add_chunk(*_run_chunk(spec, first, n, value_of))
    return out + collector.close()


def value_of(k):
    return jnp.stack([k, -2.0 * k])


@pytest.mark.parametrize("statistic", S.STATISTICS)
def test_daily_records_match_numpy_and_do_not_depend_on_chunks(statistic):
    spec = S.StreamSpec("d", "daily", statistic, ("x",))
    one = _collect(spec, [(0, 144)], value_of)  # 3 days in one chunk
    split = _collect(spec, [(0, 30), (30, 70), (100, 44)], value_of)  # chunks split mid-day
    assert [r[0] for r in one] == [dt.date(2007, 5, d) for d in (1, 2, 3)]
    for (k1, e1, v1), (k2, e2, v2) in zip(one, split):
        assert k1 == k2 and e1 == e2
        np.testing.assert_allclose(v1["x"], v2["x"], rtol=1e-15)
    k = np.arange(144, dtype=float).reshape(3, 48)
    x = np.stack([k, -2.0 * k], axis=-1)  # (day, step, component)
    reduce = {
        "mean": lambda a: a.mean(1),
        "min": lambda a: a.min(1),
        "max": lambda a: a.max(1),
        "instantaneous": lambda a: a[:, -1],
    }[statistic]
    np.testing.assert_allclose(np.stack([v["x"] for _, _, v in one]), reduce(x))
    assert one[0][1] == dt.datetime(2007, 5, 2, 0, 0)  # record time = end of the interval


def test_step_and_hourly_records():
    hourly = _collect(S.StreamSpec("h", "hourly", "mean", ("x",)), [(0, 5)], value_of)
    assert [v["x"][0] for _, _, v in hourly] == [0.5, 2.5, 4.0]
    step = _collect(S.StreamSpec("s", "step", "instantaneous", ("x",)), [(0, 3)], value_of)
    assert [v["x"][0] for _, _, v in step] == [0.0, 1.0, 2.0]


def test_bad_spec():
    with pytest.raises(ValueError):
        S.StreamSpec("bad", "weekly", "mean", ("x",))


def test_file_writer_writes_each_month_when_complete(tmp_path):
    from clm_ml.outputs import StreamFileWriter

    class Opts:
        name, level, file_per = "d", ("patch",), "month"

    meta = {"patch_gridcell": [0], "patch_weight": [1.0], "patch_pft": ["x"],
            "gridcell_names": ["g"], "soil_depth": np.zeros((1, 1))}  # fmt: skip
    writer = StreamFileWriter(Opts, ("gpp",), meta, tmp_path, "T", dt.datetime(2007, 5, 30))
    day = dt.timedelta(days=1)

    def record(d):
        end = dt.datetime(2007, 5, 30) + d * day
        return (end.date(), end, {"gpp": np.array([float(d)])})

    writer.add([record(1), record(2)])  # records ending 31 May 00:00 and 1 June 00:00
    assert not writer.written  # May may still continue
    writer.add([record(3)])  # a June record completes May
    assert [p.name for p in writer.written] == ["T_d_2007-05_patch.nc"]
    writer.close()
    assert [p.name for p in writer.written][-1] == "T_d_2007-06_patch.nc"
    import xarray as xr

    with xr.open_dataset(tmp_path / "T_d_2007-05_patch.nc") as ds:
        assert ds.sizes["time"] == 2  # the record ending at 1 June 00:00 belongs to May


def test_file_writer_continues_a_file_after_a_restart(tmp_path):
    """A restart holds no output records: the writer writes them to their file
    first, and the run resumed from the restart appends to it; the files equal those of an
    uninterrupted run."""
    import xarray as xr

    from clm_ml.outputs import StreamFileWriter

    class Opts:
        name, level, file_per = "d", ("patch",), "month"

    meta = {"patch_gridcell": [0], "patch_weight": [1.0], "patch_pft": ["x"],
            "gridcell_names": ["g"], "soil_depth": np.zeros((1, 1))}  # fmt: skip
    start = dt.datetime(2007, 5, 1)

    def record(d):
        end = start + d * dt.timedelta(days=1)
        return (end.date(), end, {"gpp": np.array([float(d)])})

    records = [record(d) for d in range(1, 40)]
    full = StreamFileWriter(Opts, ("gpp",), meta, tmp_path / "full", "T", start)
    full.add(records)
    full.close()

    first = StreamFileWriter(Opts, ("gpp",), meta, tmp_path / "split", "T", start)
    first.add(records[:10])
    saved = first.state()  # restart after 10 May
    assert "records" not in saved
    first.add(records[10:20])  # the interrupted run goes on (and is lost)
    resumed = StreamFileWriter(Opts, ("gpp",), meta, tmp_path / "split", "T", start)
    resumed.restore(saved)
    resumed.add(records[10:])
    resumed.close()
    for month in ("2007-05", "2007-06"):
        name = f"T_d_{month}_patch.nc"
        with (
            xr.open_dataset(tmp_path / "full" / name) as a,
            xr.open_dataset(tmp_path / "split" / name) as b,
        ):
            xr.testing.assert_identical(a.load(), b.load())


def test_file_writer_restart_at_the_end_of_a_file_period(tmp_path):
    """A restart written when the records held are the last of their month: the next
    record opens the next month's file with nothing left to write to the old one. Both the
    run that goes on and the run resumed from the restart write the files of an
    uninterrupted run."""
    import xarray as xr

    from clm_ml.outputs import StreamFileWriter

    class Opts:
        name, level, file_per = "d", ("patch",), "month"

    meta = {"patch_gridcell": [0], "patch_weight": [1.0], "patch_pft": ["x"],
            "gridcell_names": ["g"], "soil_depth": np.zeros((1, 1))}  # fmt: skip
    start = dt.datetime(2007, 5, 1)

    def record(d):
        end = start + d * dt.timedelta(days=1)
        return (end.date(), end, {"gpp": np.array([float(d)])})

    records = [record(d) for d in range(1, 40)]
    full = StreamFileWriter(Opts, ("gpp",), meta, tmp_path / "full", "T", start)
    full.add(records)
    full.close()

    on = StreamFileWriter(Opts, ("gpp",), meta, tmp_path / "on", "T", start)
    on.add(records[:31])  # the record of 31 May ends the month
    saved = on.state()
    on.add(records[31:])
    on.close()
    resumed = StreamFileWriter(Opts, ("gpp",), meta, tmp_path / "resumed", "T", start)
    resumed.add(records[:31])
    resumed.state()
    again = StreamFileWriter(Opts, ("gpp",), meta, tmp_path / "resumed", "T", start)
    again.restore(saved)
    again.add(records[31:])
    again.close()
    for other in ("on", "resumed"):
        for month in ("2007-05", "2007-06"):
            name = f"T_d_{month}_patch.nc"
            with (
                xr.open_dataset(tmp_path / "full" / name) as a,
                xr.open_dataset(tmp_path / other / name) as b,
            ):
                xr.testing.assert_identical(a.load(), b.load())
