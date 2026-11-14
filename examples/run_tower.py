"""
Run CLM-ML-JAX at one flux tower (default: the CHATS walnut orchard, 1 May 2007).

The first time, the script writes an experiment folder (run.yaml, domain.yaml, model.yaml,
params.yaml) from a compset, then runs it, prints the day's mean fluxes and saves a figure
(fluxes through the day, and canopy air temperature profiles by day and by night) in the
folder's output directory.

    python examples/run_tower.py                                   # CHATS7, one day
    python examples/run_tower.py --length "7 days"
    python examples/run_tower.py --set model.processes.stomatal_conductance.scheme=medlyn
    python examples/run_tower.py --set params.broadleaf_deciduous_temperate_tree.vcmax25=100

Your own tower: run once to get a folder (or `clm-ml new my_tower --from chats7_1day`), then
edit its domain.yaml (location, forcing height, soil, patches) and run.yaml (dates, forcing
and CLM history files), and run the script again on that folder:

    python examples/run_tower.py --folder my_tower

The same run from the command line: `clm-ml run my_tower`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import clm_ml

FLUXES = ("net_radiation", "sensible_heat_flux", "latent_heat_flux", "ground_heat_flux")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--folder", default="chats7_example", help="experiment folder")
    parser.add_argument("--compset", default="chats7_1day", help="compset for a new folder")
    parser.add_argument("--length", help='run length, e.g. "7 days" (default: run.yaml)')
    parser.add_argument(
        "--set", action="append", default=[], metavar="FILE.KEY=VALUE",
        help="change a setting without editing the files (repeatable), as `clm-ml run --set`",
    )  # fmt: skip
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        clm_ml.new_experiment(folder, args.compset)
        print(f"wrote {folder}/ from the compset {args.compset}")

    overrides = list(args.set)
    if args.length:
        overrides.append(f"run.time.length={args.length}")
    exp = clm_ml.load_experiment(folder, overrides)  # the four files, validated
    print(exp.summary())

    results = exp.execute()  # {output stream name: Outputs}; netCDF files in output/
    name, out = next(iter(results.items()))  # the first output stream in run.yaml
    ds = out.to_xarray()  # dims (time, patch, layer, ...), with units and long names
    print(f"\nwrote {exp.output_directory}/")

    print(f"\nmeans over the run ({name} stream, patch 0):")
    for var in FLUXES + ("gpp",):
        if var in ds:
            v = ds[var].isel(patch=0)
            print(f"  {var:22s} {float(v.mean()):8.1f} {v.attrs['units']}")

    if all(v in ds for v in FLUXES + ("air_temperature", "layer_height", "solar_zenith_angle")):
        figure = plot(ds, exp.output_directory / f"{exp.name}_{name}.png")
        print(f"\nfigure: {figure}")


def plot(ds, path: Path) -> Path:
    """Fluxes through the run, and the canopy air temperature profile when the sun is highest
    and lowest (layers above the top layer in use are NaN)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = ds.isel(patch=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), width_ratios=(2, 1))
    for var in FLUXES:
        ax1.plot(p.time, p[var], label=var.replace("_", " "))
    ax1.set_ylabel("W m-2")
    ax1.set_xlabel("time (UTC, end of each output interval)")
    ax1.legend(fontsize=8)
    ax1.set_title(f"{ds.attrs['title']}: patch 0 ({p.pft.item()})", fontsize=9)
    fig.autofmt_xdate()

    zenith = p.solar_zenith_angle.values
    for i, label in ((int(np.nanargmin(zenith)), "noon"), (int(np.nanargmax(zenith)), "night")):
        t = p.isel(time=i)
        stamp = np.datetime_as_string(t.time.values, unit="m")
        ax2.plot(t.air_temperature - 273.15, t.layer_height, ".-", label=f"{label} ({stamp})")
    ax2.set_xlabel("canopy air temperature (°C)")
    ax2.set_ylabel("height (m)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


if __name__ == "__main__":
    main()
