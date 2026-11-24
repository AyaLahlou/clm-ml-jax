"""One lock for every netCDF read and write in this process.

The forcing for the next chunk is read on a background thread while the output files of the
current one are written on the main thread (``clm_ml.driver.run_streams``). The HDF5 library
under netCDF4 is built without thread safety here (its build settings say "Threadsafety: no")
and netCDF4-python releases the GIL inside library calls, so two threads could be inside HDF5
at once. Every netCDF file is therefore opened through :func:`dataset` (or written under
:data:`LOCK`): reads and writes never overlap, while the device still runs the chunk during a
read.
"""

from __future__ import annotations

import contextlib
import threading

#: Held while any netCDF file is open (re-entrant: a reader may open a second file inside)
LOCK = threading.RLock()


@contextlib.contextmanager
def dataset(path, mode: str = "r", **kwargs):
    """``netCDF4.Dataset(path, mode)`` opened and used under :data:`LOCK`."""
    import netCDF4

    with LOCK:
        with netCDF4.Dataset(path, mode, **kwargs) as ds:
            yield ds
