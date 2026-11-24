"""Chunk schedules: calendar months (``execution.chunk: 1 month``), a chunk is the steps
starting in one month; fixed-length chunks (the default)."""

import datetime as dt
import types

import numpy as np
import pytest

from clm_ml.driver import Simulation, chunk_schedule, max_chunk_records, month_chunks, pad_steps


def _run(start_ymd, ntimes, dtime=1800):
    """The clock of a simulation (the only part these functions use)."""
    start = dt.datetime(start_ymd // 10000, start_ymd // 100 % 100, start_ymd % 100)
    sim = types.SimpleNamespace(start=start, step_seconds=dtime, n_steps=ntimes,
                                calendar="GREGORIAN")  # fmt: skip
    sim.step_starts = lambda first, n: Simulation.step_starts(sim, first, n)
    return sim


def test_split_at_month_start():
    # 30 April 00:00, 3 days of half-hourly steps: steps 0-47 start in April.
    run = _run(20070430, 3 * 48)
    assert month_chunks(run, 1) == [(1, 47), (48, 96)]


def test_run_within_one_month_is_one_chunk():
    run = _run(20070501, 50)
    assert month_chunks(run, 1) == [(1, 49)]


def test_step_ending_at_midnight_belongs_to_the_month_it_starts_in():
    # The step from 31 May 23:30 to 1 June 00:00 starts in May.
    run = _run(20070531, 49)
    assert month_chunks(run, 0) == [(0, 48), (48, 1)]


# Fixed-length chunks, counted from the first step


def test_fixed_chunks_pad_the_last_one_to_the_chunk_length():
    run = _run(20000101, 2 * 480 + 95)
    chunks, length = chunk_schedule(run, 0, 480)
    assert chunks == [(0, 480), (480, 480), (960, 95)] and length == 480


def test_run_shorter_than_a_chunk_is_one_unpadded_chunk():
    run = _run(20070501, 50)
    assert chunk_schedule(run, 0, 480) == ([(0, 50)], 50)


def test_resumed_run_keeps_the_chunks_and_rejects_another_chunk_length():
    run = _run(20000101, 3 * 480)
    assert chunk_schedule(run, 480, 480) == ([(480, 480), (960, 480)], 480)
    with pytest.raises(ValueError, match="not the start"):
        chunk_schedule(run, 500, 480)


def test_month_schedule_is_unchanged():
    run = _run(20070430, 3 * 48)
    assert chunk_schedule(run, 0, None) == ([(0, 48), (48, 96)], None)


def test_record_slots_cover_every_chunk_start():
    for frequency, period in (("hourly", 2), ("daily", 48)):
        for length in (1, 47, 48, 480):
            worst = max(len({(k + s) // period for k in range(length)}) for s in range(period))
            assert max_chunk_records(frequency, length, 1800) >= worst
    assert max_chunk_records("daily", 480, 1800) == 11
    assert max_chunk_records("step", 480, 1800) == 480
    assert max_chunk_records("monthly", 480, 1800) == 2
    assert max_chunk_records("yearly", 1, 1800) == 1


def test_pad_steps_repeats_the_last_step():
    a = np.arange(6.0).reshape(3, 2)
    out = pad_steps({"a": a}, 5)["a"]
    assert isinstance(out, np.ndarray) and out.shape == (5, 2)
    np.testing.assert_array_equal(out[3:], [[4.0, 5.0], [4.0, 5.0]])
