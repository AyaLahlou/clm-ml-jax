"""
Output streams: statistics accumulated inside the compiled chunk.

A stream is a set of variables written at one frequency with one statistic (mean,
instantaneous, min, max). Following jax-gcm, outputs are accumulated during the scan, so a
chunk keeps one value per output *record*, not per model step: memory scales with the number
of records.

How it works:

- **Host:** :func:`chunk_records` assigns each model step of a chunk to an output record of
  each stream. A step belongs to the interval that contains its *start* time, so an
  instantaneous record is the value at the end of its interval.
- **Device (in the scan):** :func:`init_buffers` makes one slot per record;
  :func:`accumulate` adds a step's values into its record's slot (sum and count for means,
  running min/max, last value for instantaneous). This is differentiable.
- **Host, after the chunk:** :func:`finish` turns the buffers into record values. A record
  that spans chunks (e.g. a yearly mean over monthly chunks) is carried to the next chunk with
  :func:`merge` before it is finished.

Padded canopy layers are NaN and stay NaN. A layer without leaves at any step of a record
makes that record's leaf value NaN for means (the variable is undefined for part of the
interval).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Mapping, Sequence

import numpy as np

FREQUENCIES = ("step", "hourly", "daily", "monthly", "yearly")
STATISTICS = ("mean", "instantaneous", "min", "max")


@dataclasses.dataclass(frozen=True)
class StreamSpec:
    name: str
    frequency: str  # one of FREQUENCIES
    statistic: str  # one of STATISTICS
    variables: tuple[str, ...]

    def __post_init__(self):
        if self.frequency not in FREQUENCIES:
            raise ValueError(f"stream {self.name}: frequency must be one of {FREQUENCIES}")
        if self.statistic not in STATISTICS:
            raise ValueError(f"stream {self.name}: statistic must be one of {STATISTICS}")


def record_key(step_start: dt.datetime, step_index: int, frequency: str):
    """The output record a step belongs to (by its start time)."""
    if frequency == "step":
        return step_index
    if frequency == "hourly":
        return step_start.replace(minute=0, second=0, microsecond=0)
    if frequency == "daily":
        return step_start.date()
    if frequency == "monthly":
        return (step_start.year, step_start.month)
    return step_start.year


@dataclasses.dataclass(frozen=True)
class ChunkRecords:
    """The records of one stream within one chunk."""

    keys: tuple  # record keys, in time order
    index: np.ndarray  # (n_steps,) int32: record slot of each step
    last_step_end: tuple  # end time of the last step of each record (in this chunk)


def chunk_records(
    step_starts: Sequence[dt.datetime],
    first_step_index: int,
    step_seconds: int,
    frequency: str,
    calendar: str = "GREGORIAN",
) -> ChunkRecords:
    """The records the steps starting at ``step_starts`` belong to (``calendar``: the model
    calendar, ``clm_ml.calendar``)."""
    from clm_ml.calendar import add_seconds

    keys, index, ends = [], [], []
    for k, start in enumerate(step_starts):
        key = record_key(start, first_step_index + k, frequency)
        if not keys or keys[-1] != key:
            keys.append(key)
            ends.append(None)
        index.append(len(keys) - 1)
        ends[-1] = add_seconds(start, step_seconds, calendar)  # model calendar
    return ChunkRecords(tuple(keys), np.asarray(index, dtype=np.int32), tuple(ends))


def init_buffers(statistic: str, n_records: int, shapes: Mapping[str, tuple], xp):
    """Per-variable buffers with one slot per record."""

    def full(shape, value):
        return xp.full((n_records,) + tuple(shape), value, dtype=float)

    if statistic == "mean":
        return {
            "sum": {n: full(s, 0.0) for n, s in shapes.items()},
            "count": xp.zeros((n_records,), dtype=float),
        }
    fill = {"min": np.inf, "max": -np.inf, "instantaneous": np.nan}[statistic]
    return {"value": {n: full(s, fill) for n, s in shapes.items()}}


def accumulate(statistic: str, buffers, slot, values: Mapping):
    """Add one step's ``values`` into record ``slot`` (JAX arrays; ``slot`` may be traced)."""
    if statistic == "mean":
        return {
            "sum": {n: b.at[slot].add(values[n]) for n, b in buffers["sum"].items()},
            "count": buffers["count"].at[slot].add(1.0),
        }
    op = {"min": "min", "max": "max", "instantaneous": "set"}[statistic]
    return {"value": {n: getattr(b.at[slot], op)(values[n]) for n, b in buffers["value"].items()}}


def merge(statistic: str, earlier, later):
    """Combine the partial buffers of one record from two chunks (numpy, one slot each)."""
    if statistic == "mean":
        return {
            "sum": {n: earlier["sum"][n] + later["sum"][n] for n in earlier["sum"]},
            "count": earlier["count"] + later["count"],
        }
    if statistic == "instantaneous":
        return later
    f = np.minimum if statistic == "min" else np.maximum
    return {"value": {n: f(earlier["value"][n], later["value"][n]) for n in earlier["value"]}}


def finish(statistic: str, buffers) -> dict:
    """Record values from buffers (numpy): ``{variable: (n_records, ...)}``."""
    if statistic == "mean":
        count = np.asarray(buffers["count"])
        return {
            n: np.asarray(s) / count.reshape((-1,) + (1,) * (np.ndim(s) - 1))
            for n, s in buffers["sum"].items()
        }
    return {n: np.asarray(v) for n, v in buffers["value"].items()}


def slot(buffers, i: int):
    """Buffers of record slot ``i`` only (numpy), for :func:`merge`."""
    if "sum" in buffers:
        return {
            "sum": {n: np.asarray(b)[i : i + 1] for n, b in buffers["sum"].items()},
            "count": np.asarray(buffers["count"])[i : i + 1],
        }
    return {"value": {n: np.asarray(b)[i : i + 1] for n, b in buffers["value"].items()}}


def drop_last(buffers):
    """Buffers without their last slot (numpy)."""
    if "sum" in buffers:
        return {
            "sum": {n: np.asarray(b)[:-1] for n, b in buffers["sum"].items()},
            "count": np.asarray(buffers["count"])[:-1],
        }
    return {"value": {n: np.asarray(b)[:-1] for n, b in buffers["value"].items()}}


def concat_first(head, buffers):
    """``head`` (one slot) in place of the first slot of ``buffers`` (numpy)."""
    if "sum" in buffers:
        return {
            "sum": {
                n: np.concatenate([head["sum"][n], np.asarray(b)[1:]])
                for n, b in buffers["sum"].items()
            },
            "count": np.concatenate([head["count"], np.asarray(buffers["count"])[1:]]),
        }
    return {
        "value": {
            n: np.concatenate([head["value"][n], np.asarray(b)[1:]])
            for n, b in buffers["value"].items()
        }
    }


class StreamCollector:
    """Host side of one stream: turns each chunk's buffers into completed records.

    The last record of a chunk may continue in the next chunk (same key), so it is held as
    pending and merged; :meth:`close` emits it at the end of the run."""

    def __init__(self, spec: StreamSpec):
        self.spec = spec
        self._pending = None  # (key, end, one-slot buffers)

    def add_chunk(self, records: ChunkRecords, buffers) -> list:
        """Completed records ``[(key, end_time, {variable: value}), ...]`` of this chunk."""
        stat = self.spec.statistic
        keys, ends = list(records.keys), list(records.last_step_end)
        out = []
        if self._pending is not None:
            key, _, head = self._pending
            if keys[0] == key:  # the record continues: merge its two parts
                buffers = concat_first(merge(stat, head, slot(buffers, 0)), buffers)
            else:
                out.extend(self._emit(self._pending))
        self._pending = (keys[-1], ends[-1], slot(buffers, len(keys) - 1))
        if len(keys) > 1:
            values = finish(stat, drop_last(buffers))
            for i, (key, end) in enumerate(zip(keys[:-1], ends[:-1])):
                out.append((key, end, {n: v[i] for n, v in values.items()}))
        return out

    def close(self) -> list:
        out = self._emit(self._pending) if self._pending is not None else []
        self._pending = None
        return out

    def _emit(self, pending) -> list:
        key, end, buffers = pending
        values = finish(self.spec.statistic, buffers)
        return [(key, end, {n: v[0] for n, v in values.items()})]
