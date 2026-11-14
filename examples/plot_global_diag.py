"""
Maps of the time-mean state of a CLM-ML-JAX grid run.

Reads the gridcell-level output files (``*_gridcell.nc``, on the latitude-longitude map) in a
run's output directory, averages every variable over all the records of each output stream,
and saves one map per variable and an overview of all of them in OUT_DIR/plots:

    python examples/plot_global_diag.py $SCRATCH/global_1mon_test/output
    python examples/plot_global_diag.py OUT_DIR --variables gpp latent_heat_flux
    python examples/plot_global_diag.py OUT_DIR --stream daily --plot-dir my_plots

Values are per unit vegetated area of a gridcell (the weighted mean of its patches). The
global mean in each title weights gridcells by their vegetated area (area x landfrac x
vegetated_fraction). Grey: no gridcell run (ocean, ice, unvegetated land, outside the region).
Variables with dimensions other than time, lat and lon (by leaf class or band) are skipped.
The output stream needs `level: [gridcell]` (or both levels) in run.yaml.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import xarray as xr

# {tower}_{stream}_{YYYY | YYYY-MM | YYYY-MM-DD}_gridcell.nc (clm_ml.outputs)
FILE_NAME = re.compile(r"^(?P<stream>.+)_\d{4}(-\d{2}){0,2}_gridcell\.nc$")
AREA = ("area", "landfrac", "vegetated_fraction")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("out_dir", type=Path, help="the run's output directory")
    parser.add_argument("--stream", help="only this output stream (e.g. daily)")
    parser.add_argument("--variables", nargs="+", help="only these variables (default: all)")
    parser.add_argument("--plot-dir", type=Path, help="where to save (default: OUT_DIR/plots)")
    args = parser.parse_args()

    streams = find_streams(args.out_dir)
    if args.stream:
        streams = {k: v for k, v in streams.items() if k.split("_", 1)[-1] == args.stream}
    if not streams:
        raise SystemExit(f"no *_gridcell.nc files in {args.out_dir}")

    plot_dir = args.plot_dir or args.out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for stream, paths in streams.items():
        result = time_mean(paths, args.variables)
        if result is None:
            print(f"{stream}: not on a latitude-longitude map (a tower run?), skipped")
            continue
        means, area, period, skipped = result
        print(f"{stream}: {len(paths)} file(s), {period}")
        if skipped:
            print(f"  skipped (extra dimensions): {', '.join(skipped)}")
        for path in plot_stream(means, area, f"{stream}, mean {period}", plot_dir, stream):
            print(f"  {path}")


def find_streams(out_dir: Path) -> dict[str, list[Path]]:
    """``{tower_stream: [files in time order]}`` from the gridcell-level output files."""
    streams: dict[str, list[Path]] = {}
    for path in sorted(out_dir.glob("*_gridcell.nc")):
        match = FILE_NAME.match(path.name)
        if match:
            streams.setdefault(match["stream"], []).append(path)
    return streams


def time_mean(paths: list[Path], variables=None):
    """The mean over every record of a stream, read one file at a time (memory does not grow
    with the run's length). Returns ``(means, area, period, skipped)``, or None when the files
    are not on a lat-lon map."""
    sums: dict[str, xr.DataArray] = {}
    counts: dict[str, xr.DataArray] = {}
    attrs, skipped, times, area = {}, set(), [], None
    for path in paths:
        with xr.open_dataset(path) as ds:
            if "lat" not in ds.dims:
                return None
            for name, v in ds.data_vars.items():
                if "time" not in v.dims or (variables and name not in variables):
                    continue
                if v.dims != ("time", "lat", "lon"):
                    skipped.add(name)
                    continue
                v = v.load()
                total, n = v.sum("time"), v.notnull().sum("time")  # NaN: no gridcell run
                sums[name] = total + sums[name] if name in sums else total
                counts[name] = n + counts[name] if name in counts else n
                attrs[name] = v.attrs
            if area is None:
                area = ds[[a for a in AREA if a in ds]].load()
            if len(ds.time) > 1 and not times:  # a record's time is the end of its interval
                times.append(ds.time.values[0] - (ds.time.values[1] - ds.time.values[0]))
            times += [ds.time.values[0], ds.time.values[-1]]
    means = xr.Dataset(
        {n: (sums[n] / counts[n].where(counts[n] > 0)).assign_attrs(attrs[n]) for n in sums}
    )
    first, last = (
        np.datetime_as_string(t, "m").replace("T", " ") for t in (min(times), max(times))
    )
    period = f"{first} to {last}"
    return means, area, period, sorted(skipped)


def vegetated_mean(field: xr.DataArray, area: xr.Dataset) -> float:
    """Mean over the gridcells run, weighted by vegetated area (NaN without the area fields)."""
    if not all(a in area for a in AREA):
        return float("nan")
    w = area.area * area.landfrac * area.vegetated_fraction
    ok = np.isfinite(field) & np.isfinite(w)
    return float((field * w).where(ok).sum() / w.where(ok).sum())


def plot_stream(means, area, title, plot_dir: Path, stream: str) -> list[Path]:
    """One map per variable and an overview of all of them; returns the files saved."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Longitudes 0-360 -> -180-180, so the map is centred on Greenwich.
    means = means.assign_coords(lon=((means.lon + 180) % 360) - 180).sortby("lon")
    area = area.assign_coords(lon=((area.lon + 180) % 360) - 180).sortby("lon")
    names = list(means.data_vars)
    saved = []

    for name in names:
        fig, ax = plt.subplots(figsize=(10, 5))
        draw(fig, ax, means[name], area)
        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        path = plot_dir / f"{stream}_{name}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        saved.append(path)

    ncols = 2 if len(names) > 1 else 1
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.4 * nrows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        draw(fig, ax, means[name], area, fontsize=8)
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = plot_dir / f"{stream}_overview.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    saved.append(path)
    return saved


def draw(fig, ax, field: xr.DataArray, area: xr.Dataset, fontsize: int = 9) -> None:
    """A map of one time-mean field: a diverging colour scale centred on 0 when the field
    takes both signs, a sequential one otherwise (limits: 2nd and 98th percentiles)."""
    units = field.attrs.get("units", "")
    if units == "K":
        field, units = field - 273.15, "°C"
    elif units == "1":  # dimensionless
        units = ""
    values = field.values
    if not np.isfinite(values).any():
        ax.set_title(f"{field.name}: no values", fontsize=fontsize)
        return
    lo, hi = np.nanpercentile(values, [2, 98])
    if lo < 0 < hi and min(-lo, hi) > 0.1 * max(-lo, hi):
        limit = max(-lo, hi)
        cmap, vmin, vmax = "RdBu_r", -limit, limit
    else:
        cmap, vmin, vmax = "viridis", lo, hi
    mesh = ax.pcolormesh(
        field.lon, field.lat, values, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest"
    )
    ax.set_facecolor("0.85")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=fontsize - 1)
    fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02).set_label(units, fontsize=fontsize)
    long_name = field.attrs.get("long_name", field.name)
    mean = vegetated_mean(field, area)
    ax.set_title(f"{long_name}\n(vegetated-area mean {mean:.3g} {units})", fontsize=fontsize)


if __name__ == "__main__":
    main()
