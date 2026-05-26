"""
Load and quality-filter flux observations for parameter optimization.

Reads observed GPP and LE from an AmeriFlux-format observations file
and aligns them to model timesteps.  Returns masked numpy arrays
suitable for use as optimization targets.

Expected input format
---------------------
CSV file with at minimum the following columns:
  TIMESTAMP_START : yyyyMMddHHmm  (30-minute resolution)
  GPP_DT_VUT_REF  : Gross Primary Production (gC m-2 day-1)  or
  GPP_NT_VUT_REF  : GPP from nighttime partitioning
  LE              : Latent heat flux (W m-2)
  LE_QC           : Quality flag (0=measured, 1=good quality gap-fill, 2=medium, ...)
  SW_IN_F         : Incoming shortwave (W m-2) — for daytime filtering
  USTAR           : Friction velocity (m s-1) — for turbulence quality filtering
  NEE_VUT_REF_QC  : NEE quality flag (0=measured, 1=good)

This matches the FLUXNET2015 FULLSET product format available from
https://fluxnet.org/data/fluxnet2015-dataset/

Preprocessing applied
---------------------
1. Align observation timestamps to model forcing timesteps (±1 minute tolerance)
2. Daytime filter: remove SW_IN_F < 5 W m-2
3. Turbulence filter: remove USTAR < ustar_min (default 0.2 m s-1)
4. Quality filter: remove LE_QC > max_qc (default 1 = good gap-fill OK)
5. Convert GPP from gC m-2 day-1 to μmol CO₂ m-2 s-1

Usage
-----
    from diags.expt_load_obs import load_obs
    obs = load_obs("path/to/CHATS7_obs.csv", forcing_nc="input_files/tower-forcing/CHATS7/2007-05.nc")
    obs_gpp, obs_le, masks = obs["gpp"], obs["le"], obs["mask"]
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Unit conversion: gC m-2 day-1 → μmol CO₂ m-2 s-1
# 1 gC = 83.26 μmol CO₂ (using 12 g/mol)
# 1 day = 86400 s
_GC_PER_DAY_TO_UMOL_PER_SEC = 1e6 / 12.011 / 86400.0   # = 0.9645 μmol/gC/s


@dataclass
class FluxObs:
    """Container for quality-filtered flux observations aligned to model timesteps.

    All arrays have shape (n_timesteps,).  Invalid/masked entries are NaN
    and mask[t] = False.

    Attributes:
        gpp  : GPP (μmol CO₂ m-2 s-1), NaN where masked
        le   : Latent heat flux (W m-2), NaN where masked
        mask : Boolean array, True where both GPP and LE are valid
        timestamps : Unix timestamps for each model timestep (seconds since epoch)
        n_valid : Number of valid (unmasked) timesteps
    """
    gpp:        np.ndarray   # (n_timesteps,) μmol CO₂ m-2 s-1
    le:         np.ndarray   # (n_timesteps,) W m-2
    mask:       np.ndarray   # (n_timesteps,) bool
    timestamps: np.ndarray   # (n_timesteps,) int64 seconds
    n_valid:    int


def load_obs(
    obs_csv: str | Path,
    forcing_nc: str | Path,
    gpp_col: str = "GPP_DT_VUT_REF",
    le_col: str = "LE",
    le_qc_col: str = "LE_QC",
    sw_col: str = "SW_IN_F",
    ustar_col: str = "USTAR",
    sw_min: float = 5.0,
    ustar_min: float = 0.2,
    max_qc: int = 1,
    timestamp_col: str = "TIMESTAMP_START",
    timestamp_fmt: str = "%Y%m%d%H%M",
    gpp_unit: str = "gC_m2_day",
) -> FluxObs:
    """Load and quality-filter flux observations.

    Parameters
    ----------
    obs_csv : path to AmeriFlux FULLSET CSV file
    forcing_nc : path to model forcing NetCDF (used to get model timestamps)
    gpp_col : column name for GPP in obs_csv
    le_col : column name for LE in obs_csv
    le_qc_col : quality flag column for LE (0=measured, 1=good gap-fill)
    sw_col : column name for incoming shortwave
    ustar_col : friction velocity column
    sw_min : minimum SW for daytime (W m-2)
    ustar_min : minimum u* for turbulence quality (m s-1)
    max_qc : maximum acceptable LE quality flag (0=meas only, 1=good fill)
    timestamp_col : column with timestamps
    timestamp_fmt : strptime format for timestamp column
    gpp_unit : "gC_m2_day" (FLUXNET2015 default) or "umol_m2_s"

    Returns
    -------
    FluxObs with aligned, filtered observations
    """
    import pandas as pd
    import netCDF4 as nc

    obs_csv = Path(obs_csv)
    if not obs_csv.exists():
        raise FileNotFoundError(
            f"Observations file not found: {obs_csv}\n"
            "Download FLUXNET2015 FULLSET for your site from https://fluxnet.org"
        )

    # ── Load observations ─────────────────────────────────────────────────────
    df = pd.read_csv(obs_csv, na_values=["-9999", "-9999.0"])
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in {obs_csv}")

    df["_datetime"] = pd.to_datetime(df[timestamp_col], format=timestamp_fmt)
    df = df.set_index("_datetime").sort_index()

    # ── Load model forcing timestamps ─────────────────────────────────────────
    forcing_nc = Path(forcing_nc)
    if not forcing_nc.exists():
        raise FileNotFoundError(f"Forcing file not found: {forcing_nc}")

    with nc.Dataset(str(forcing_nc), "r") as ds:
        time_var  = ds.variables["time"]
        # time_var.units is typically "days since YYYY-MM-DD HH:MM:SS"
        import netCDF4 as nc4
        times_nc  = nc4.num2date(time_var[:], time_var.units, calendar="standard")

    model_datetimes = pd.DatetimeIndex(
        [pd.Timestamp(t.year, t.month, t.day, t.hour, t.minute, t.second)
         for t in times_nc]
    )
    n_model = len(model_datetimes)

    # ── Align obs to model timesteps (nearest match within 1 minute) ──────────
    gpp_aligned  = np.full(n_model, np.nan)
    le_aligned   = np.full(n_model, np.nan)
    sw_aligned   = np.full(n_model, np.nan)
    ustar_aligned = np.full(n_model, np.nan)
    qc_aligned   = np.full(n_model, np.nan)

    for i, dt in enumerate(model_datetimes):
        # Find closest observation within ±90 seconds
        if gpp_col in df.columns:
            try:
                idx = df.index.get_indexer([dt], method="nearest", tolerance="90s")
                if idx[0] >= 0:
                    row = df.iloc[idx[0]]
                    gpp_aligned[i]   = row.get(gpp_col, np.nan)
                    le_aligned[i]    = row.get(le_col, np.nan)
                    sw_aligned[i]    = row.get(sw_col, np.nan)
                    ustar_aligned[i] = row.get(ustar_col, np.nan)
                    qc_aligned[i]    = row.get(le_qc_col, np.nan)
            except Exception:
                pass

    # ── Unit conversion for GPP ───────────────────────────────────────────────
    if gpp_unit == "gC_m2_day":
        gpp_aligned = gpp_aligned * _GC_PER_DAY_TO_UMOL_PER_SEC
    elif gpp_unit != "umol_m2_s":
        raise ValueError(f"Unknown gpp_unit='{gpp_unit}', expected 'gC_m2_day' or 'umol_m2_s'")

    # ── Quality masks ─────────────────────────────────────────────────────────
    # 1. Not NaN
    valid_gpp   = np.isfinite(gpp_aligned)
    valid_le    = np.isfinite(le_aligned)
    # 2. Daytime
    valid_day   = sw_aligned >= sw_min
    # 3. Turbulence quality
    valid_ustar = ustar_aligned >= ustar_min
    # 4. LE quality flag
    valid_qc    = qc_aligned <= max_qc

    mask = valid_gpp & valid_le & valid_day & valid_ustar & valid_qc
    n_valid = int(np.sum(mask))

    gpp_out = np.where(mask, gpp_aligned, np.nan)
    le_out  = np.where(mask, le_aligned,  np.nan)

    print(f"load_obs: {n_valid}/{n_model} timesteps valid ({100*n_valid/n_model:.1f}%)")
    print(f"  GPP range: {np.nanmin(gpp_out):.2f}–{np.nanmax(gpp_out):.2f} μmol CO₂ m⁻² s⁻¹")
    print(f"  LE range:  {np.nanmin(le_out):.1f}–{np.nanmax(le_out):.1f} W m⁻²")

    timestamps = np.array([int(dt.timestamp()) for dt in model_datetimes], dtype=np.int64)

    return FluxObs(
        gpp=gpp_out,
        le=le_out,
        mask=mask,
        timestamps=timestamps,
        n_valid=n_valid,
    )


def load_obs_synthetic(
    forcing_nc: str | Path,
    vcmax25_true: float = 125.0,
    iota_true: float = 375.0,
) -> FluxObs:
    """Generate synthetic observations by running the model with 'true' parameters.

    Used for the identifiability test (Section 7.3 of parameter_optimization_experiment.md):
    generate observations with known CHATS7 parameters, then verify that optimization
    starting from CLM defaults recovers the true values.

    Parameters
    ----------
    forcing_nc : forcing NetCDF for timestamp extraction
    vcmax25_true : true vcmax25 value (default: CHATS7 Rosati et al. 2006)
    iota_true : true iota_SPA value (default: CHATS7 value)

    Returns
    -------
    FluxObs with synthetic GPP and LE (mask = True everywhere, no noise)
    """
    warnings.warn(
        "load_obs_synthetic: this generates perfect synthetic observations "
        "for identifiability testing only. Do not use for real calibration.",
        UserWarning, stacklevel=2,
    )
    raise NotImplementedError(
        "Synthetic obs generation requires running the full forward model. "
        "Implement in optimize_params.py using the 'true' parameter set."
    )


# ── CHATS7-specific convenience wrapper ───────────────────────────────────────
def load_obs_chats7(
    obs_csv: str | Path,
    year: int = 2007,
    month: int = 5,
) -> FluxObs:
    """Load CHATS7 May 2007 observations.

    Expects an AmeriFlux FULLSET CSV for the CHATS walnut orchard site
    (US-Bo? / CHATS AmeriFlux ID).

    The CHATS data are described in:
      Patton et al. (2011) "The Canopy Horizontal Array Turbulence Study"
      Bull. Am. Meteorol. Soc., doi:10.1175/2010BAMS2614.1

    Flux partitioning follows the Reichstein nighttime and Lasslop daytime
    methods, both available in the FLUXNET2015 FULLSET product.
    """
    forcing_nc = (
        Path(__file__).resolve().parent.parent
        / "src" / "input_files" / "tower-forcing" / "CHATS7"
        / f"{year:04d}-{month:02d}.nc"
    )
    return load_obs(
        obs_csv=obs_csv,
        forcing_nc=forcing_nc,
        gpp_col="GPP_DT_VUT_REF",   # daytime-partitioned GPP
        le_col="LE_F_MDS",          # gap-filled LE
        le_qc_col="LE_F_MDS_QC",
        sw_col="SW_IN_F",
        ustar_col="USTAR",
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python expt_load_obs.py path/to/observations.csv")
        print("Example: python expt_load_obs.py data/CHATS7_2007.csv")
        sys.exit(1)
    obs = load_obs_chats7(obs_csv=sys.argv[1])
    print(f"\nLoaded {obs.n_valid} valid flux observations for CHATS7")
    print(f"GPP mean (daytime): {np.nanmean(obs.gpp):.2f} μmol CO₂ m⁻² s⁻¹")
    print(f"LE mean (daytime):  {np.nanmean(obs.le):.1f} W m⁻²")
