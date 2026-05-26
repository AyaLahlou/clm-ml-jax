"""
plot_grad_check.py — Publication-quality gradient correctness figure for CLM-ML-JAX.

Two-panel figure:
  A. Relative error heatmap  (3 outputs × 7 parameters, log10 color scale)
     GPP: 7/7 verified PASS; LE/H: 4/4 original params verified, 3 new marked "—".
  B. AD vs FD agreement scatter (log-log, 1:1 line, ±1% band)
     15 verified checks span 4 decades of gradient magnitude.

Data sources:
  - Original 4 params (sw, tref, iota, vcmax): hardcoded from Sessions 36-37.
  - New 3 params (q, pbot, u): check_10param_grads.csv (Session 38-39).
    alpha_pbot NaN fixed by jnp.maximum guards in MLLeafPhotosynthesisMod.py.
Stomatal model: WUE (gs_type=2). epsilon=1e-4 central FD.

Usage:
  python diags/plot_grad_check.py
  python diags/plot_grad_check.py --out path/to/grad_check.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

_HERE = Path(__file__).resolve().parent
_FIGURES_DIR = _HERE / "figures"

# ── Typography & style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "text.usetex":       False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Data ──────────────────────────────────────────────────────────────────────
# 7 parameters under WUE (gs_type=2); g1_MED removed (inactive under WUE).
# New params q, pbot, u added from check_10param_grads.csv (Session 38-39).
# alpha_pbot NaN fixed by jnp.maximum guards in MLLeafPhotosynthesisMod.py.
PARAMS  = [r"$\alpha_\mathrm{sw}$", r"$\alpha_\mathrm{tref}$",
           r"$\alpha_{\iota}$",      r"$\alpha_\mathrm{vcmax}$",
           r"$\alpha_{q}$",          r"$\alpha_{P}$",
           r"$\alpha_{u}$"]
OUTPUTS = ["GPP", "LE", "H"]

_NA = np.nan  # not verified (LE/H for new params pending; no cell is INACT)

# rel_err[output_row][param_col] — NaN = not yet verified
REL_ERR = np.array([
    # α_sw    α_tref   α_ι      α_vcmax  α_q      α_P      α_u
    [3.7e-7,  1.3e-4,  1.1e-6,  1.8e-8,  8.3e-5,  4.4e-6,  4.2e-5],  # GPP (all verified)
    [1.0e-6,  1.1e-4,  1.3e-6,  8.5e-8,  _NA,     _NA,     _NA   ],  # LE  (new 3: TBD)
    [1.4e-6,  1.1e-4,  1.2e-6,  8.4e-8,  _NA,     _NA,     _NA   ],  # H   (new 3: TBD)
])

# Active (output, param, |AD|, |FD|) pairs for scatter
# New GPP pairs from check_10param_grads.csv; LE/H only for verified params.
GRAD_PAIRS = [
    # GPP — original 4
    ("GPP", r"$\alpha_\mathrm{sw}$",     10.70,   10.70),
    ("GPP", r"$\alpha_\mathrm{tref}$",   48.69,   48.69),
    ("GPP", r"$\alpha_{\iota}$",          2.136,   2.136),
    ("GPP", r"$\alpha_\mathrm{vcmax}$",  14.14,   14.14),
    # GPP — new 3 (from check_10param_grads.csv; pbot NaN fixed in Session 38)
    ("GPP", r"$\alpha_{q}$",              0.1536,  0.1536),
    ("GPP", r"$\alpha_{P}$",              4.698,   4.698),
    ("GPP", r"$\alpha_{u}$",              0.875,   0.875),
    # LE — original 4
    ("LE",  r"$\alpha_\mathrm{sw}$",    200.0,   200.0),
    ("LE",  r"$\alpha_\mathrm{tref}$",  9238.0,  9237.0),
    ("LE",  r"$\alpha_{\iota}$",         77.93,   77.93),
    ("LE",  r"$\alpha_\mathrm{vcmax}$",  60.42,   60.42),
    # H — original 4
    ("H",   r"$\alpha_\mathrm{sw}$",    117.5,   117.5),
    ("H",   r"$\alpha_\mathrm{tref}$",  8998.0,  8997.0),
    ("H",   r"$\alpha_{\iota}$",         69.13,   69.13),
    ("H",   r"$\alpha_\mathrm{vcmax}$",  54.69,   54.69),
]

# Per-output visual style
_STYLE = {
    "GPP": dict(color="#16A34A", marker="o"),
    "LE":  dict(color="#2563EB", marker="s"),
    "H":   dict(color="#DC2626", marker="^"),
}

# Manual label offsets in log-decades (dx, dy) relative to point
_OFFSET = {
    ("GPP", r"$\alpha_{\iota}$"):          (-0.50,  0.12),
    ("GPP", r"$\alpha_\mathrm{sw}$"):      (-0.30, -0.22),
    ("GPP", r"$\alpha_\mathrm{vcmax}$"):   ( 0.06, -0.22),
    ("GPP", r"$\alpha_\mathrm{tref}$"):    (-0.55,  0.12),
    ("GPP", r"$\alpha_{q}$"):              (-0.55, -0.22),
    ("GPP", r"$\alpha_{P}$"):              ( 0.06,  0.12),
    ("GPP", r"$\alpha_{u}$"):              ( 0.06, -0.22),
    ("LE",  r"$\alpha_\mathrm{sw}$"):      (-0.55, -0.20),
    ("LE",  r"$\alpha_\mathrm{tref}$"):    (-0.62,  0.12),   # push left (H is to right)
    ("LE",  r"$\alpha_{\iota}$"):          (-0.56,  0.12),
    ("LE",  r"$\alpha_\mathrm{vcmax}$"):   ( 0.06, -0.22),
    ("H",   r"$\alpha_\mathrm{sw}$"):      ( 0.06,  0.12),
    ("H",   r"$\alpha_\mathrm{tref}$"):    ( 0.06,  0.10),   # push right (LE is to left)
    ("H",   r"$\alpha_{\iota}$"):          ( 0.06,  0.12),
    ("H",   r"$\alpha_\mathrm{vcmax}$"):   (-0.60, -0.22),
}


def _fmt_sci(v: float) -> str:
    """Format as compact scientific notation: m.n×10^e."""
    exp = int(np.floor(np.log10(v)))
    mant = v / 10**exp
    return rf"${mant:.1f}{{\times}}10^{{{exp}}}$"


def plot(out_stem: Path) -> None:
    fig = plt.figure(figsize=(11, 4.2))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1],
                  left=0.07, right=0.97, bottom=0.14, top=0.93, wspace=0.40)

    ax_heat = fig.add_subplot(gs[0])
    ax_scat = fig.add_subplot(gs[1])

    # ── Panel A: Relative error heatmap ───────────────────────────────────────
    # Colormap: green (small error) → red (large error); gray for NaN/INACT
    norm = LogNorm(vmin=1e-9, vmax=1e-2)
    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color="#D1D5DB")

    im = ax_heat.imshow(REL_ERR, norm=norm, cmap=cmap,
                        aspect="auto", interpolation="nearest",
                        origin="upper")

    # Cell annotations
    for r in range(len(OUTPUTS)):
        for c in range(len(PARAMS)):
            v = REL_ERR[r, c]
            if np.isnan(v):
                rect = mpatches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    linewidth=0, facecolor="none",
                    edgecolor="#6B7280", hatch="////", zorder=3,
                )
                ax_heat.add_patch(rect)
                ax_heat.text(c, r, "—", ha="center", va="center",
                             fontsize=9, color="#4B5563",
                             zorder=4)
            else:
                log_v = np.log10(v)
                # Dark background → white text; light background → black text
                txt_color = "white" if log_v < -5.0 else "black"
                ax_heat.text(c, r, _fmt_sci(v), ha="center", va="center",
                             fontsize=7.5, color=txt_color, zorder=4)

    # Ticks and labels — origin='upper': row 0 (GPP) at top
    ax_heat.set_xticks(range(len(PARAMS)))
    ax_heat.set_xticklabels(PARAMS, fontsize=8.5)
    ax_heat.set_yticks(range(len(OUTPUTS)))
    ax_heat.set_yticklabels(OUTPUTS, fontsize=9, fontstyle="italic",
                            fontweight="bold")
    ax_heat.tick_params(bottom=False, left=False)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    # White grid between cells
    for x in np.arange(-0.5, len(PARAMS), 1):
        ax_heat.axvline(x, color="white", lw=1.5, zorder=2)
    for y in np.arange(-0.5, len(OUTPUTS), 1):
        ax_heat.axhline(y, color="white", lw=1.5, zorder=2)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical",
                        fraction=0.046, pad=0.03, shrink=0.90)
    cbar.set_label("Relative error  |AD−FD| / |FD|", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    # Annotate 1% threshold on colorbar (it's at the very top = vmax)
    cbar.ax.axhline(1.0, color="#B91C1C", lw=1.8, ls="--")
    cbar.ax.text(2.4, 0.96, "1%\nlimit", va="top", ha="left",
                 fontsize=7, color="#B91C1C",
                 transform=cbar.ax.transAxes, clip_on=False)

    ax_heat.set_title("(a)  Relative error: AD vs. FD  (— = LE/H not yet verified)",
                      loc="left", fontsize=9, fontweight="bold", pad=6)

    # ── Panel B: AD vs FD log-log scatter ─────────────────────────────────────
    lim_lo, lim_hi = 0.05, 2.5e4
    x_ref = np.logspace(np.log10(lim_lo), np.log10(lim_hi), 300)

    # ±1% shaded band around identity
    ax_scat.fill_between(x_ref, x_ref * 0.99, x_ref * 1.01,
                         color="#94A3B8", alpha=0.3, lw=0, zorder=1,
                         label="_nolegend_")
    # 1:1 identity
    ax_scat.plot(x_ref, x_ref, color="#1E293B", lw=1.2, ls="-",
                 zorder=2, label="1:1 identity")

    # Scatter points and labels
    for out in ["GPP", "LE", "H"]:
        pairs = [(p, ad, fd) for (o, p, ad, fd) in GRAD_PAIRS if o == out]
        xs = np.array([fd  for _, _, fd in pairs])
        ys = np.array([ad  for _, ad, _  in pairs])
        st = _STYLE[out]
        ax_scat.scatter(xs, ys, marker=st["marker"], color=st["color"],
                        s=60, zorder=5, label=out,
                        edgecolors="white", linewidths=0.4)
        for (param, ad, fd) in pairs:
            dx, dy = _OFFSET.get((out, param), (0.06, 0.12))
            ax_scat.text(fd * 10**dx, ad * 10**dy, param,
                         fontsize=7, color=st["color"],
                         ha="center", va="center", zorder=6)

    ax_scat.set_xscale("log")
    ax_scat.set_yscale("log")
    ax_scat.set_xlim(lim_lo, lim_hi)
    ax_scat.set_ylim(lim_lo, lim_hi)
    ax_scat.set_xlabel(r"|gradient| — central FD ($\varepsilon = 10^{-4}$)",
                       fontsize=8.5)
    ax_scat.set_ylabel(r"|gradient| — jax.grad",
                       fontsize=8.5)

    # Legend
    output_handles = [
        mpatches.Patch(color=_STYLE[o]["color"], label=o)
        for o in ["GPP", "LE", "H"]
    ]
    ref_handles = [
        plt.Line2D([0], [0], color="#1E293B", lw=1.2, label="1:1 identity"),
        mpatches.Patch(color="#94A3B8", alpha=0.55, label=r"$\pm$1% band"),
    ]
    ax_scat.legend(handles=output_handles + ref_handles,
                   fontsize=7.5, framealpha=0.9, loc="upper left",
                   handlelength=1.2, borderpad=0.5, labelspacing=0.3)

    ax_scat.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax_scat.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    ax_scat.set_title("(b)  AD vs. FD agreement across 4+ decades",
                      loc="left", fontsize=9, fontweight="bold", pad=6)

    # Pass-rate badge
    ax_scat.text(0.97, 0.04,
                 "15/15 verified checks PASS\n(7 GPP · 4 LE · 4 H, all rel. err $<$ 1%)",
                 transform=ax_scat.transAxes,
                 ha="right", va="bottom", fontsize=7.5, color="#166534",
                 bbox=dict(boxstyle="round,pad=0.35", fc="#DCFCE7",
                           ec="#16A34A", lw=0.9))

    # ── Save (pdf + png) ──────────────────────────────────────────────────────
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = out_stem.with_suffix("")          # drop extension if given
    for ext in ("pdf", "png"):
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=300)
        print(f"Saved: {p}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=_FIGURES_DIR / "grad_check.pdf",
                    help="Output path (extension auto-added for pdf+png pair)")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
