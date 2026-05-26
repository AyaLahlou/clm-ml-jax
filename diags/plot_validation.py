"""
Diagnostic figure: JAX vs Fortran validation comparison.

Generates a multi-panel figure comparing JAX model output against the
Fortran reference run for CHATS7 May 2007.

Output files:
  diags/figures/validation_flux.png      — surface energy/water fluxes
  diags/figures/validation_profiles.png  — vertical canopy profiles (noon)

Usage (from project root):
    python diags/plot_validation.py [--jax-dir PATH] [--ref-dir PATH]

Columns in *_flux.out (18 cols, no header):
  0  calday
  1  Rn (net radiation W/m2)
  2  H  (sensible heat W/m2)
  3  LE (latent heat W/m2)
  4  Rabs (absorbed shortwave W/m2)
  5  LWup (upwelling LW W/m2)
  6  ET (evapotranspiration mm/s)
  7  GPP (gross primary production umol/m2/s)
  8  SWdown (incoming shortwave W/m2)
  9  Tair (air temperature K)
  10 Tveg (vegetation temperature K)
  11 Tsoil (soil surface temperature K)
  12 ustar (m/s)
  13 CO2 (ppm)
  14 Rnet_veg (vegetation net radiation W/m2)
  15 Rnet_soil (soil net radiation W/m2)
  16 beta (-)
  17 Obu (Obukhov length m)

Columns in *_profile.out (28 cols, no header):
  0  calday
  1  height (m)
  2-5  PAD components (m2/m2)
  6-23 various fluxes (see Fortran output())
  24 wind (m/s)
  25 Tair (K)
  26 q (g/kg)
  27 CO2 (ppm)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# -----------------------------------------------------------------------
# Default paths
# -----------------------------------------------------------------------
DEFAULT_JAX_DIR = SRC_DIR / "output_files" / "JAX_outputs_05_2007_31days"
DEFAULT_REF_DIR = SRC_DIR / "output_files" / "validation_files" / "validation_files_05_2007_31days"
SITE = "CHATS7"
PERIOD = "2007-05"
FIGURES_DIR = Path(__file__).parent / "figures"


def load_flux(directory: Path) -> np.ndarray:
    fpath = directory / f"{SITE}_{PERIOD}_flux.out"
    return np.loadtxt(fpath)


def load_profile(directory: Path) -> np.ndarray:
    fpath = directory / f"{SITE}_{PERIOD}_profile.out"
    return np.loadtxt(fpath)


def calday_to_hour(calday: np.ndarray) -> np.ndarray:
    """Convert fractional calendar day-of-year to hours since start."""
    return (calday - calday[0]) * 24.0


def plot_flux_comparison(jax_dir: Path, ref_dir: Path, out_dir: Path) -> None:
    """Panel plot of surface flux time series: JAX vs Fortran."""
    jax = load_flux(jax_dir)
    ref = load_flux(ref_dir)

    # Use all available timesteps (both files are 31-day runs)
    n = min(len(jax), len(ref))
    jax = jax[:n]
    ref_day = ref[:n]

    t_jax = jax[:, 0]   # calday (fractional day-of-year)
    t_ref = ref_day[:, 0]

    panels = [
        (1,  "Net radiation (W m$^{-2}$)",  "Rn"),
        (2,  "Sensible heat (W m$^{-2}$)",  "H"),
        (3,  "Latent heat (W m$^{-2}$)",    "LE"),
        (6,  "ET (mm s$^{-1}$)",            "ET"),
        (7,  "GPP (µmol m$^{-2}$ s$^{-1}$)","GPP"),
        (12, "u* (m s$^{-1}$)",             "u*"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for ax, (col_idx, ylabel, label) in zip(axes, panels):
        ax.plot(t_ref, ref_day[:, col_idx], "k-",  lw=1.5, label="Fortran ref")
        ax.plot(t_jax, jax[:, col_idx],     "r--", lw=1.5, label="JAX")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes[-3:]:
        ax.set_xlabel("Day of year (calday)", fontsize=9)

    fig.suptitle(f"CHATS7 May 2007 (31 days) — Surface fluxes: JAX vs Fortran", fontsize=12)
    fig.tight_layout()

    out_path = out_dir / "validation_flux.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print RMSE summary
    print("\nRMSE summary (JAX vs Fortran, first day):")
    print(f"  {'Variable':<10}  {'RMSE':>10}  {'Max abs err':>12}")
    for col_idx, ylabel, label in panels:
        nref = min(len(jax), len(ref_day))
        diff = jax[:nref, col_idx] - ref_day[:nref, col_idx]
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae  = float(np.max(np.abs(diff)))
        print(f"  {label:<10}  {rmse:>10.4f}  {mae:>12.4f}")


def plot_profile_comparison(jax_dir: Path, ref_dir: Path, out_dir: Path) -> None:
    """Compare vertical profile at the first midday timestep (calday ~121.5)."""
    jax = load_profile(jax_dir)
    ref = load_profile(ref_dir)

    # Find midday timestep: calday fractional hour ~= 12:00 local
    # CHATS7 longitude ~-121.5° → UTC offset ~-8h → calday noon = calday + (12+8)/24
    target_frac = 0.833  # ~20:00 UTC = noon local for CHATS7
    day0 = int(jax[0, 0])
    target_calday = day0 + target_frac

    # Heights from profile file
    jax_heights = np.unique(jax[:, 1])[::-1]  # descending

    def extract_profile_at_time(data, calday_target):
        """Extract a vertical profile at the timestep closest to calday_target."""
        ts = np.unique(data[:, 0])
        nearest = ts[np.argmin(np.abs(ts - calday_target))]
        rows = data[data[:, 0] == nearest]
        # Sort by height descending
        rows = rows[np.argsort(rows[:, 1])[::-1]]
        return rows

    jax_noon = extract_profile_at_time(jax, target_calday)
    ref_noon = extract_profile_at_time(ref, target_calday)

    # Profile variable columns (after col 0=calday, 1=height)
    profile_vars = [
        (25, "(a) Air temperature (K)", "Tair"),
        (24, "(b) Wind speed (m s$^{-1}$)", "Wind"),
        (26, "(c) Specific humidity (g kg$^{-1}$)", "q"),
        (27, "(d) CO$_2$ (ppm)", "CO2"),
    ]

    fig, axes = plt.subplots(1, len(profile_vars), figsize=(14, 4), sharey=True)

    for ax, (col_idx, xlabel, label) in zip(axes, profile_vars):
        h_jax = jax_noon[:, 1]
        h_ref = ref_noon[:, 1]
        ax.plot(ref_noon[:, col_idx], h_ref, "-",color="navy",  lw=3, label="Fortran")
        ax.plot(jax_noon[:, col_idx], h_jax, "--",color="deepskyblue", lw=3, label="JAX")
        #ax.set_xlabel(xlabel, fontsize=9)
        #ax.set_title(xlabel, fontsize=10, fontweight="bold")
        
        ax.text(0.10, 0.90,
        f"{xlabel}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, ec="none"))

        ax.grid(alpha=0.3)
        if col_idx ==profile_vars[0][0]:
            ax.legend(fontsize=10, loc='lower left', prop={'size': 12})
    
    axes[0].set_ylabel("Height (m)", fontsize=9)
    #fig.suptitle(
    #    f"CHATS7 May 1, 2007 — Canopy profiles at ~noon: JAX vs Fortran\n"
    #    f"(calday ≈ {target_calday:.3f})",
    #    fontsize=12,
    #)
    fig.tight_layout()

    out_path = out_dir / "validation_profiles.png"
    pdf_path = out_dir / "validation_profiles.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path,  bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_scatter_comparison(jax_dir: Path, ref_dir: Path, out_dir: Path) -> None:
    """Scatter plot: JAX vs Fortran for all flux variables (31 days)."""
    jax = load_flux(jax_dir)
    ref = load_flux(ref_dir)

    nref = min(len(jax), len(ref))
    jax = jax[:nref]
    ref_day = ref[:nref]

    # (col_index, short name, unit string)
    var_info = [
        (1,  "(a) Rn",        "W m$^{-2}$"),
        (2,  "(b) H",         "W m$^{-2}$"),
        (3,  "(c) LE",        "W m$^{-2}$"),
        (4,  "(d) Rabs",      "W m$^{-2}$"),
        (5,  "(e) LWup",      "W m$^{-2}$"),
        (6,  "(f) ET",        "mm s$^{-1}$"),
        (7,  "(g) GPP",       "µmol m$^{-2}$ s$^{-1}$"),
        (8,  "(h) SWdn",      "W m$^{-2}$"),
        (9,  "(i) Tair",      "K"),
        (10, "(j) Tveg",      "K"),
        (11, "(k) Tsoil",     "K"),
        (12, "(l) u*",        "m s$^{-1}$"),
        (13, "(m) CO$_2$",    "ppm"),
        (14, "(n) Rnet_veg",  "W m$^{-2}$"),
        (15, "(o) Rnet_soil", "W m$^{-2}$"),
        (17, "(p) Obu",       "m"),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(14, 9))
    axes = axes.flatten()
    

    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
        ], N=256)
    # Pass 2: draw
    last_sc = None
    for ax, (col_idx, name, unit) in zip(axes, var_info):
        x = ref_day[:nref, col_idx]
        y = jax[:nref, col_idx]
        vmin = min(x.min(), y.min())
        vmax = max(x.max(), y.max())

        #last_sc = ax.hist2d(x, y, (35, 35), cmap="coolwarm", cmin=1)#color='darkcyan', alpha=0.3, s=20)
        #'mediumspringgreen'
        ax.scatter(x, y, color='cyan', alpha=0.09, s=30, edgecolors='purple', linewidths=0.9)
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1)

        r2   = float(np.corrcoef(x, y)[0, 1] ** 2) if np.std(x) > 0 else float("nan")
        rmse = float(np.sqrt(np.mean((y - x) ** 2)))
        bias = float(np.mean(y - x))
        denom = np.sum((x - np.mean(x)) ** 2)
        nse  = float(1.0 - np.sum((y - x) ** 2) / denom) if denom > 0 else float("nan")

        ax.text(0.60, 0.05,
                f"Bias : {bias:.3f}\nCorr : {r2:.3f}\nNSE : {nse:.3f}\n",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, ec="none"))

        ax.text(0.05, 0.80, f"{name} ({unit})", transform=ax.transAxes, ha="left", va="bottom", fontsize=12,fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, ec="none"))
        
        if ax.get_gridspec().get_geometry()[0] - 1 == ax.get_subplotspec().rowspan.start:
            ax.set_xlabel("Fortran CLM-ml", fontsize=8)

        if ax.get_gridspec().get_geometry()[1] -4 == ax.get_subplotspec().colspan.start  :
            ax.set_ylabel("JAX CLM-ml", fontsize=8)
            
        ax.tick_params(labelsize=6)

    # Hide unused panels
    for ax in axes[len(var_info):]:
        ax.set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 1])



    out_path = out_dir / "validation_scatter.png"
    pdf_path = out_dir / "validation_scatter.pdf"

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches='tight')

    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot JAX vs Fortran validation figures")
    parser.add_argument("--jax-dir", type=Path, default=DEFAULT_JAX_DIR)
    parser.add_argument("--ref-dir", type=Path, default=DEFAULT_REF_DIR)
    args = parser.parse_args()

    jax_dir = args.jax_dir
    ref_dir = args.ref_dir

    if not jax_dir.exists():
        print(f"ERROR: JAX output dir not found: {jax_dir}", file=sys.stderr)
        print("Run the model first:  python -m offline_executable.main input_files/nl.CHATS7.1day")
        sys.exit(1)
    if not ref_dir.exists():
        print(f"ERROR: Reference dir not found: {ref_dir}", file=sys.stderr)
        sys.exit(1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_flux_comparison(jax_dir, ref_dir, FIGURES_DIR)
    plot_profile_comparison(jax_dir, ref_dir, FIGURES_DIR)
    plot_scatter_comparison(jax_dir, ref_dir, FIGURES_DIR)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
