"""
Create comparison plots between JAX and Fortran implementations of mixing_length.
"""
import sys
from pathlib import Path
import xarray as xr
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from clubb_core.mixing_length import compute_mixing_length
from clubb_core.grid_class import Grid


def sample_dataset_to_arguments(sample_data: xr.Dataset):
    """Convert sample data to function arguments."""
    nzm = sample_data.attrs["nzm"]
    nzt = sample_data.attrs["nzt"]
    ngrdcol = 1

    zm = jnp.array(sample_data["zm"].expand_dims(dim="column", axis=0).to_numpy())
    zt = jnp.array(sample_data["zt"].expand_dims(dim="column", axis=0).to_numpy())
    dzm = jnp.array(sample_data["dzm"].to_numpy())
    dzt = jnp.array(sample_data["dzt"].to_numpy())
    invrs_dzm = jnp.array(sample_data["invrs_dzm"].to_numpy())
    invrs_dzt = jnp.array(sample_data["invrs_dzt"].to_numpy())

    gr = Grid(
        nzm=nzm, nzt=nzt, ngrdcol=ngrdcol,
        zm=zm, zt=zt, dzm=dzm, dzt=dzt,
        invrs_dzm=invrs_dzm, invrs_dzt=invrs_dzt,
        weights_zt2zm=jnp.zeros((ngrdcol, nzm, 2)),
        weights_zm2zt=jnp.zeros((ngrdcol, nzt, 2)),
        k_lb_zm=0, k_ub_zm=nzm-1,
        k_lb_zt=0, k_ub_zt=nzt-1,
        grid_dir_indx=1, grid_dir=1.0,
    )

    class ErrInfo:
        pass
    err_info = ErrInfo()

    kwargs = {
        "nzm": nzm, "nzt": nzt, "ngrdcol": ngrdcol, "gr": gr,
        "thvm": jnp.array(sample_data["thvm"].to_numpy()),
        "thlm": jnp.array(sample_data["thlm"].to_numpy()),
        "rtm": jnp.array(sample_data["rtm"].to_numpy()),
        "em": jnp.array(sample_data["em"].to_numpy()),
        "Lscale_max": jnp.array(sample_data["Lscale_max"].to_numpy()),
        "p_in_Pa": jnp.array(sample_data["p_in_Pa"].to_numpy()),
        "exner": jnp.array(sample_data["exner"].to_numpy()),
        "thv_ds": jnp.array(sample_data["thv_ds"].to_numpy()),
        "mu": float(sample_data["mu"].to_numpy().item()),
        "lmin": float(sample_data["lmin"].to_numpy().item()),
        "saturation_formula": int(sample_data["saturation_formula"].to_numpy().item()),
        "l_implemented": bool(sample_data["l_implemented"].to_numpy().item()),
        "err_info": err_info,
    }
    return kwargs


def create_comparison_plots(sample_idx=0):
    """Create comprehensive comparison plots."""
    # Load test data
    tests_dir = Path(__file__).parent / 'tests/mixing_length_test'
    path = tests_dir / "sample_data" / "bomex_mixing_length_calculation_samples.nc"
    data = xr.open_dataset(path)

    # Get sample
    sample_data = data.isel(samples=sample_idx).squeeze().expand_dims(dim="column", axis=0)

    # Run JAX implementation
    print(f"Running JAX implementation for sample {sample_idx}...")
    kwargs = sample_dataset_to_arguments(sample_data)
    Lscale_jax, Lscale_up_jax, Lscale_down_jax, _ = compute_mixing_length(**kwargs)

    # Get Fortran reference
    Lscale_fort = sample_data["Lscale"].values[0, :]
    Lscale_up_fort = sample_data["Lscale_up"].values[0, :]
    Lscale_down_fort = sample_data["Lscale_down"].values[0, :]

    # Convert JAX arrays to numpy
    Lscale_jax = np.array(Lscale_jax[0, :])
    Lscale_up_jax = np.array(Lscale_up_jax[0, :])
    Lscale_down_jax = np.array(Lscale_down_jax[0, :])

    # Get heights
    zt = sample_data["zt"].values

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # === Row 1: Lscale_up ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(Lscale_up_fort, zt, 'b-', linewidth=2, label='Fortran', alpha=0.7)
    ax1.plot(Lscale_up_jax, zt, 'r--', linewidth=2, label='JAX', alpha=0.7)
    ax1.set_xlabel('Lscale_up [m]', fontsize=11)
    ax1.set_ylabel('Height [m]', fontsize=11)
    ax1.set_title('Upward Mixing Length', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    diff_up = Lscale_up_jax - Lscale_up_fort
    ax2.plot(diff_up, zt, 'g-', linewidth=2)
    ax2.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Difference [m]', fontsize=11)
    ax2.set_ylabel('Height [m]', fontsize=11)
    ax2.set_title('Lscale_up: JAX - Fortran', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    rel_err_up = np.where(Lscale_up_fort > 0,
                          100 * np.abs(diff_up) / Lscale_up_fort, 0)
    ax3.plot(rel_err_up, zt, 'orange', linewidth=2)
    ax3.set_xlabel('Relative Error [%]', fontsize=11)
    ax3.set_ylabel('Height [m]', fontsize=11)
    ax3.set_title('Lscale_up: Relative Error', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)

    # === Row 2: Lscale_down ===
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(Lscale_down_fort, zt, 'b-', linewidth=2, label='Fortran', alpha=0.7)
    ax4.plot(Lscale_down_jax, zt, 'r--', linewidth=2, label='JAX', alpha=0.7)
    ax4.set_xlabel('Lscale_down [m]', fontsize=11)
    ax4.set_ylabel('Height [m]', fontsize=11)
    ax4.set_title('Downward Mixing Length', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    diff_down = Lscale_down_jax - Lscale_down_fort
    ax5.plot(diff_down, zt, 'g-', linewidth=2)
    ax5.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Difference [m]', fontsize=11)
    ax5.set_ylabel('Height [m]', fontsize=11)
    ax5.set_title('Lscale_down: JAX - Fortran', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    rel_err_down = np.where(Lscale_down_fort > 0,
                            100 * np.abs(diff_down) / Lscale_down_fort, 0)
    ax6.plot(rel_err_down, zt, 'orange', linewidth=2)
    ax6.set_xlabel('Relative Error [%]', fontsize=11)
    ax6.set_ylabel('Height [m]', fontsize=11)
    ax6.set_title('Lscale_down: Relative Error', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(left=0)

    # === Row 3: Lscale (total) ===
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(Lscale_fort, zt, 'b-', linewidth=2, label='Fortran', alpha=0.7)
    ax7.plot(Lscale_jax, zt, 'r--', linewidth=2, label='JAX', alpha=0.7)
    ax7.set_xlabel('Lscale [m]', fontsize=11)
    ax7.set_ylabel('Height [m]', fontsize=11)
    ax7.set_title('Total Mixing Length (√(up × down))', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 1])
    diff_total = Lscale_jax - Lscale_fort
    ax8.plot(diff_total, zt, 'g-', linewidth=2)
    ax8.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Difference [m]', fontsize=11)
    ax8.set_ylabel('Height [m]', fontsize=11)
    ax8.set_title('Lscale: JAX - Fortran', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    ax9 = fig.add_subplot(gs[2, 2])
    rel_err_total = np.where(Lscale_fort > 0,
                             100 * np.abs(diff_total) / Lscale_fort, 0)
    ax9.plot(rel_err_total, zt, 'orange', linewidth=2)
    ax9.set_xlabel('Relative Error [%]', fontsize=11)
    ax9.set_ylabel('Height [m]', fontsize=11)
    ax9.set_title('Lscale: Relative Error', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim(left=0)

    # Overall title
    fig.suptitle(f'Mixing Length Comparison: JAX vs Fortran (Sample {sample_idx})',
                 fontsize=14, fontweight='bold', y=0.995)

    # Save figure
    output_path = Path(__file__).parent / f'mixing_length_comparison_sample{sample_idx}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")

    # Print statistics
    print("\n" + "="*70)
    print("COMPARISON STATISTICS")
    print("="*70)

    print("\nLscale_up:")
    print(f"  Max absolute error: {np.max(np.abs(diff_up)):.6e} m")
    print(f"  Mean absolute error: {np.mean(np.abs(diff_up)):.6e} m")
    print(f"  Max relative error: {np.max(rel_err_up):.6f} %")
    print(f"  RMS error: {np.sqrt(np.mean(diff_up**2)):.6e} m")

    print("\nLscale_down:")
    print(f"  Max absolute error: {np.max(np.abs(diff_down)):.6e} m")
    print(f"  Mean absolute error: {np.mean(np.abs(diff_down)):.6e} m")
    print(f"  Max relative error: {np.max(rel_err_down):.6f} %")
    print(f"  RMS error: {np.sqrt(np.mean(diff_down**2)):.6e} m")

    print("\nLscale (total):")
    print(f"  Max absolute error: {np.max(np.abs(diff_total)):.6e} m")
    print(f"  Mean absolute error: {np.mean(np.abs(diff_total)):.6e} m")
    print(f"  Max relative error: {np.max(rel_err_total):.6f} %")
    print(f"  RMS error: {np.sqrt(np.mean(diff_total**2)):.6e} m")

    # Print some specific level comparisons
    print("\n" + "="*70)
    print("SAMPLE VALUES AT SPECIFIC LEVELS")
    print("="*70)
    levels = [0, 1, 10, 20, 40, 60, 80, 86]
    print(f"\n{'Level':>5} {'Height':>8} {'Fort_up':>10} {'JAX_up':>10} {'Fort_down':>10} {'JAX_down':>10}")
    print("-"*70)
    for k in levels:
        if k < len(zt):
            print(f"{k:5d} {zt[k]:8.1f} {Lscale_up_fort[k]:10.2f} {Lscale_up_jax[k]:10.2f} "
                  f"{Lscale_down_fort[k]:10.2f} {Lscale_down_jax[k]:10.2f}")

    return fig


if __name__ == "__main__":
    print("Creating mixing length comparison plots...")
    print("(Note: Debug output from upward parcel tracking will appear)\n")
    fig = create_comparison_plots(sample_idx=0)
    print("\nPlot creation complete!")
