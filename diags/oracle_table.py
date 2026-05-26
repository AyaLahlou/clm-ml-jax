"""
Experiment 1: Oracle validation table (Table 1 in paper).

Computes per-variable max absolute error, RMSE, and max relative error
between CLM-ml-jax outputs and the Fortran reference for the first day
(48 half-hourly timesteps) at CHATS7, May 1 2007.

Usage (from project root):
    python diags/oracle_table.py

Output:
  - Console: LaTeX-ready table rows
  - diags/figures/oracle_table.csv: raw numbers
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
FIGURES_DIR  = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

JAX_DIR = SRC_DIR / "output_files" / "JAX_outputs_05_2007_31days"
REF_DIR = SRC_DIR / "output_files" / "validation_files" / "validation_files_05_2007_31days"
SITE    = "CHATS7"
PERIOD  = "2007-05"


def load_flux(directory: Path) -> np.ndarray:
    return np.loadtxt(directory / f"{SITE}_{PERIOD}_flux.out")


def load_profile(directory: Path) -> np.ndarray:
    return np.loadtxt(directory / f"{SITE}_{PERIOD}_profile.out")


jax_flux = load_flux(JAX_DIR)
ref_flux  = load_flux(REF_DIR)

# Use all timesteps (31 days × 48 half-hourly = 1488 steps)
M = min(len(jax_flux), len(ref_flux))
jax_f = jax_flux[:M]
ref_f = ref_flux[:M]
jax_f = jax_f[:M]
ref_f = ref_f[:M]

# ── Flux variables ────────────────────────────────────────────────────────────
flux_vars = [
    (1,  "Net radiation",   "Rn",   "W m$^{-2}$"),
    (2,  "Sensible heat",   "H",    "W m$^{-2}$"),
    (3,  "Latent heat",     "LE",   "W m$^{-2}$"),
    (7,  "GPP",             "GPP",  "μmol m$^{-2}$ s$^{-1}$"),
    (10, "Vegetation temp", "Tveg", "K"),
    (12, "Friction vel.",   "u*",   "m s$^{-1}$"),
]

n_days = M // 48
print(f"\nOracle validation: CHATS7 May 2007 ({M} timesteps = {n_days} days)\n")
print(f"{'Variable':<22}  {'Unit':<22}  {'Max|err|':>10}  {'RMSE':>10}  {'Max rtol':>10}")
print("-" * 80)

rows = []
for col, long_name, short, unit in flux_vars:
    diff  = jax_f[:, col] - ref_f[:, col]
    mae   = float(np.max(np.abs(diff)))
    rmse  = float(np.sqrt(np.mean(diff**2)))
    denom = np.abs(ref_f[:, col])
    rtol  = float(np.max(np.abs(diff) / (denom + 1e-30)))
    rows.append((long_name, unit, mae, rmse, rtol))
    print(f"  {long_name:<20}  {unit:<22}  {mae:>10.4f}  {rmse:>10.4f}  {rtol:>10.2e}")

# ── Profile variables (noon timestep) ─────────────────────────────────────────
jax_prof = load_profile(JAX_DIR)
ref_prof  = load_profile(REF_DIR)

target_frac  = 0.833
day0         = int(jax_prof[0, 0])
target_calday = day0 + target_frac

def nearest_profile(data, calday_target):
    ts      = np.unique(data[:, 0])
    nearest = ts[np.argmin(np.abs(ts - calday_target))]
    rows    = data[data[:, 0] == nearest]
    return rows[np.argsort(rows[:, 1])[::-1]]  # sorted by height desc

jax_noon = nearest_profile(jax_prof, target_calday)
ref_noon  = nearest_profile(ref_prof, target_calday)
Np = min(len(jax_noon), len(ref_noon))

profile_vars = [
    (25, "Air temp. profile",  "Tair_prof",  "K"),
    (24, "Wind profile",       "u_prof",     "m s$^{-1}$"),
    (26, "Humidity profile",   "q_prof",     "g kg$^{-1}$"),
    (27, "CO2 profile",        "CO2_prof",   "ppm"),
]

print(f"\n  (Profile comparison at noon, {Np} canopy layers)")
for col, long_name, short, unit in profile_vars:
    x    = ref_noon[:Np, col]
    y    = jax_noon[:Np, col]
    diff = y - x
    mae  = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    rtol = float(np.max(np.abs(diff) / (np.abs(x) + 1e-30)))
    rows.append((long_name, unit, mae, rmse, rtol))
    print(f"  {long_name:<20}  {unit:<22}  {mae:>10.4f}  {rmse:>10.4f}  {rtol:>10.2e}")

# ── LaTeX table ───────────────────────────────────────────────────────────────
print("\n\n% ── LaTeX Table 1 (paste into paper) ─────────────────────────────────")
print(r"\begin{table}[t]")
print(r"\caption{Oracle validation: clm-ml-jax vs. Fortran reference, CHATS7 May 2007 (1488 half-hourly timesteps = 31 days).}")
print(r"\label{tab:oracle}")
print(r"\centering")
print(r"\begin{tabular}{llrrr}")
print(r"\toprule")
print(r"Variable & Unit & Max$|\Delta|$ & RMSE & Max rtol \\")
print(r"\midrule")
for long_name, unit, mae, rmse, rtol in rows:
    print(f"{long_name} & {unit} & {mae:.4f} & {rmse:.4f} & {rtol:.2e} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

# ── CSV ───────────────────────────────────────────────────────────────────────
csv_path = FIGURES_DIR / "oracle_table.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Variable", "Unit", "Max|err|", "RMSE", "Max rtol"])
    for row in rows:
        writer.writerow(row)
print(f"\nCSV saved: {csv_path}")
