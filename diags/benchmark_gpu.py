"""
GPU performance benchmark for CLM-ML-JAX — MLCanopyFluxes baseline.

Runs multiple CLM timesteps (via ModelAdvance, which internally calls
MLCanopyFluxes) and measures:
  - JIT compile + first-run time (first ModelAdvance call)
  - Steady-state wall-clock time (subsequent ModelAdvance calls, averaged)
  - Per-sub-step timing estimate
  - Device memory stats

Uses jax.block_until_ready on mlcanopy_inst.tleaf_leaf to ensure all GPU
work is complete before stopping the timer.

NOTE: Direct MLCanopyFluxes calls after ModelAdvance OOM with CUDA graph
instantiation limits (~14 alive graphs). Benchmarking via sequential
ModelAdvance avoids this because ModelAdvance manages its own CUDA graph
lifecycle end-to-end.

Usage (from project root):
    python diags/benchmark_gpu.py

Output: diags/figures/benchmark_baseline.txt
"""
import sys
import time
import statistics
from pathlib import Path

# ---------------------------------------------------------------------------
# JAX must be configured for 64-bit BEFORE any other imports that touch JAX
# ---------------------------------------------------------------------------
import jax
jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Path setup — benchmark lives in diags/, source in src/
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT  = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC_ROOT))

# ---------------------------------------------------------------------------
# Early model imports (safe before InitializeRealize — no array creation)
# ---------------------------------------------------------------------------
import offline_executable.main as _main_mod
from offline_driver import controlMod, TowerDataMod, clmSoilOptionMod
from multilayer_canopy import MLclm_varctl
from clm_src_utils import clm_time_manager
from clm_src_utils.clm_time_manager import get_curr_calday
from clm_src_cpl.lnd_comp_nuopc import InitializeRealize, ModelAdvance
from clm_src_main import filterMod as _filterMod
from clm_share.shr_orb_mod import shr_orb_params
from clm_src_utils import clm_varorb

# ---------------------------------------------------------------------------
# Locate the namelist (nl.CHATS7.1day lives in src/offline_executable/)
# ---------------------------------------------------------------------------
namelist_file = str(_SRC_ROOT / "offline_executable" / "nl.CHATS7.1day")
sys.argv = ["benchmark_gpu.py", namelist_file]

nml    = _main_mod.read_namelist(namelist_file)
params = nml.get("clmML_inparm", nml.get("clm_inparm", nml.get("clm_input", {})))

tower_name      = str(params.get("tower_name", "CHATS7"))
start_ymd       = int(params.get("start_ymd", 20070501))
iyear           = start_ymd // 10000
imonth          = (start_ymd // 100) % 100

fin_tower       = _main_mod._resolve_path(str(params.get("fin_tower", "")))
fin_clm         = _main_mod._resolve_path(str(params.get("fin_clm", "")))
fin_soil_adjust = _main_mod._resolve_path(str(params.get("fin_soil_adjust", "")))
dirout_raw      = str(params.get("dirout", "")).strip()
dirout          = _main_mod._resolve_path(dirout_raw) if dirout_raw else ""

fin1, fin2 = fin_tower, fin_clm

# ---------------------------------------------------------------------------
# Configure model globals from namelist
# ---------------------------------------------------------------------------
controlMod.tower_site  = tower_name
controlMod.iyear       = iyear
controlMod.imonth      = imonth

tower_num = 0
for i in range(1, int(TowerDataMod.ntower) + 1):
    if tower_name == str(TowerDataMod.tower_id[i]):
        tower_num = i
        break
TowerDataMod.tower_num = tower_num

clmSoilOptionMod.clm_phys          = str(params.get("clm_phys", clmSoilOptionMod.clm_phys))
clmSoilOptionMod.nlev_soil_adjust  = int(params.get("nlev_soil_adjust", 0))
clmSoilOptionMod.fin_soil_adjust   = fin_soil_adjust
MLclm_varctl.met_type  = int(params.get("met_type",   MLclm_varctl.met_type))
MLclm_varctl.dpai_min  = float(params.get("dpai_min", MLclm_varctl.dpai_min))
MLclm_varctl.pftcon_val = int(params.get("pftcon_val", MLclm_varctl.pftcon_val))

clm_time_manager.start_date_ymd = start_ymd
clm_time_manager.start_date_tod = int(params.get("start_tod", 0))
dtstep_sec = int(TowerDataMod.tower_time[tower_num]) * 60
clm_time_manager.dtstep = dtstep_sec

bounds = _main_mod.build_bounds(nml)
ntimes_met = 2   # number of met records pre-loaded

# ---------------------------------------------------------------------------
# InitializeRealize — sets nlevgrnd/nlevsno, allocates all state containers.
# CRITICAL: all imports that pull in clm_instMod must happen AFTER this call.
# ---------------------------------------------------------------------------
print("Initializing model (phase 1)...", flush=True)
InitializeRealize(bounds)

# ---------------------------------------------------------------------------
# Post-init imports (safe now that nlevgrnd/nlevsno are set)
# ---------------------------------------------------------------------------
from clm_src_main import clm_instMod
import clm_src_main.clm_driver as _clm_driver_mod
from offline_driver.CLMml_driver import init_acclim, TowerVeg, SoilInit
from offline_driver.TowerMetMod import TowerMetCurr, TowerMetNext

_new_filter = _filterMod.setFilters(_filterMod.filter)
_filterMod.filter       = _new_filter
_clm_driver_mod.filter  = _new_filter

# ---------------------------------------------------------------------------
# Orbital parameters
# ---------------------------------------------------------------------------
_eccen, _obliq, _mvelp, _obliqr, _lambm0, _mvelpp = shr_orb_params(iyear)
clm_varorb.eccen  = _eccen
clm_varorb.obliqr = _obliqr
clm_varorb.lambm0 = _lambm0
clm_varorb.mvelpp = _mvelpp

# ---------------------------------------------------------------------------
# Acclimation, vegetation, and soil initialization
# ---------------------------------------------------------------------------
print("Initializing model (phase 2: acclim + veg + soil)...", flush=True)

(clm_instMod.atm2lnd_inst,
 clm_instMod.wateratm2lndbulk_inst,
 clm_instMod.temperature_inst,
 clm_instMod.frictionvel_inst,
 clm_instMod.mlcanopy_inst) = init_acclim(
    fin1, tower_num, ntimes_met,
    bounds.begp, bounds.endp,
    clm_instMod.atm2lnd_inst,
    clm_instMod.wateratm2lndbulk_inst,
    clm_instMod.temperature_inst,
    clm_instMod.frictionvel_inst,
    clm_instMod.mlcanopy_inst,
)

(clm_instMod.canopystate_inst,
 clm_instMod.mlcanopy_inst) = TowerVeg(
    tower_num, bounds.begp, bounds.endp,
    clm_instMod.canopystate_inst,
    clm_instMod.mlcanopy_inst,
)

clm_start_ymd = int(params.get("clm_start_ymd", start_ymd))
clm_start_tod = int(params.get("clm_start_tod", 0))
run_start_ymd = clm_time_manager.start_date_ymd
run_start_tod = clm_time_manager.start_date_tod
clm_time_manager.start_date_ymd = clm_start_ymd
clm_time_manager.start_date_tod = clm_start_tod
clm_time_manager.itim = 1
start_calday_clm = get_curr_calday(offset=0)
clm_time_manager.start_date_ymd = run_start_ymd
clm_time_manager.start_date_tod = run_start_tod
clm_time_manager.itim = 1
curr_calday = get_curr_calday(offset=0)
soil_init_time_indx = round((curr_calday - start_calday_clm) * 86400.0 / float(dtstep_sec)) + 1

(clm_instMod.waterstatebulk_inst,
 clm_instMod.temperature_inst) = SoilInit(
    fin2, soil_init_time_indx,
    bounds.begc, bounds.endc,
    clm_instMod.soilstate_inst,
    clm_instMod.waterstatebulk_inst,
    clm_instMod.temperature_inst,
)

# Load met forcing for timestep 1
met_type = MLclm_varctl.met_type
clm_time_manager.itim = 1

(clm_instMod.atm2lnd_inst,
 clm_instMod.wateratm2lndbulk_inst,
 clm_instMod.frictionvel_inst) = TowerMetCurr(
    fin1, 1, tower_num,
    bounds.begp, bounds.endp,
    clm_instMod.atm2lnd_inst,
    clm_instMod.wateratm2lndbulk_inst,
    clm_instMod.frictionvel_inst,
)

if met_type == 3:
    clm_instMod.mlcanopy_inst = TowerMetNext(
        fin1, min(2, ntimes_met),
        bounds.begp, bounds.endp,
        clm_instMod.mlcanopy_inst,
    )

print("Initialization complete.", flush=True)

from clm_src_utils.clm_time_manager import get_curr_date

num_ml_steps = int(dtstep_sec / MLclm_varctl.dtime_ml)
nrk_inner    = MLclm_varctl.runge_kutta_type // 10 + 1


def _advance_one(time_indx: int) -> float:
    """Run one ModelAdvance call and block until GPU work is done. Returns elapsed time."""
    t0 = time.perf_counter()
    ModelAdvance(bounds, time_indx, fin1, fin2)
    jax.block_until_ready(clm_instMod.mlcanopy_inst.tleaf_leaf)
    return time.perf_counter() - t0


def _get_time_indx() -> int:
    """Compute the current time index for ModelAdvance."""
    curr_calday_now = get_curr_calday(offset=0)
    return round((curr_calday_now - start_calday_clm) * 86400.0 / float(dtstep_sec)) + 1


# ---------------------------------------------------------------------------
# Timestep 1: JIT compile + first physics call
# ---------------------------------------------------------------------------
print(f"\n[1/2] Compile + first run (timestep 1)...", flush=True)
time_indx = _get_time_indx()
t_compile = _advance_one(time_indx)
print(f"      Compile + first-run: {t_compile:.3f} s", flush=True)

# Check filter was populated
num_evp = _clm_driver_mod.filter.num_exposedvegp
print(f"      num_exposedvegp after first advance: {num_evp}", flush=True)

# ---------------------------------------------------------------------------
# Timesteps 2–6: steady-state timing (5 calls post-compile)
# ---------------------------------------------------------------------------
N_STEADY = 5
print(f"\n[2/2] Steady-state timing ({N_STEADY} more timesteps)...", flush=True)
times_steady = []
for k in range(N_STEADY):
    clm_time_manager.itim += 1
    time_indx = _get_time_indx()
    dt = _advance_one(time_indx)
    times_steady.append(dt)
    print(f"      timestep {k+2}/{N_STEADY+1}: {dt:.4f} s", flush=True)

mean_steady = statistics.mean(times_steady)
std_steady  = statistics.stdev(times_steady) if len(times_steady) > 1 else 0.0
min_steady  = min(times_steady)
max_steady  = max(times_steady)
time_per_substep = mean_steady / num_ml_steps if num_ml_steps > 0 else float("nan")

# ---------------------------------------------------------------------------
# Device and memory info
# ---------------------------------------------------------------------------
devices    = jax.devices()
device_str = str(devices[0])
backend    = jax.default_backend()

mem_info_str = "not available"
try:
    stats = devices[0].memory_stats()
    if stats:
        used_mb = stats.get("bytes_in_use", 0) / 1024**2
        peak_mb = stats.get("peak_bytes_in_use", 0) / 1024**2
        mem_info_str = f"{used_mb:.1f} MB in use, peak {peak_mb:.1f} MB"
except Exception as exc:
    mem_info_str = f"error: {exc}"

# ---------------------------------------------------------------------------
# Report to stdout
# ---------------------------------------------------------------------------
SEP = "=" * 70
print()
print(SEP)
print("  CLM-ML-JAX  |  GPU Benchmark Baseline  |  CHATS7 site")
print(SEP)
print(f"  JAX backend:           {backend.upper()}  —  {device_str}")
print(f"  JAX version:           {jax.__version__}")
print(f"  Site / forcing:        CHATS7  (2007-05.nc)")
print(f"  dtstep:                {dtstep_sec} s")
print(f"  num_ml_steps:          {num_ml_steps}  (dtime_ml = {MLclm_varctl.dtime_ml} s)")
print(f"  RK inner steps:        {nrk_inner}  (runge_kutta_type = {MLclm_varctl.runge_kutta_type})")
print(f"  num_exposedvegp:       {num_evp}")
print()
print(f"  Compile + first run (timestep 1):    {t_compile:.3f} s")
print()
print(f"  Steady-state ({N_STEADY} timesteps, jax.block_until_ready):")
print(f"    mean:                {mean_steady:.4f} s")
print(f"    std:                 {std_steady:.4f} s")
print(f"    min:                 {min_steady:.4f} s")
print(f"    max:                 {max_steady:.4f} s")
print(f"    per-ml-substep est:  {time_per_substep*1000:.2f} ms")
print()
print(f"  Device memory:         {mem_info_str}")
print(SEP)
print()
print("  NOTE: Per-module breakdown requires manual instrumentation.")
print("  Non-diff mode (grid=None) — standard forward pass, no gradient tracking.")
print("  Timing via ModelAdvance (includes met I/O, output writing).")
print("  Direct MLCanopyFluxes calls OOM on CUDA graph instantiation after")
print("  ModelAdvance has already populated all CUDA graphs (14-graph limit).")
print(SEP)

# ---------------------------------------------------------------------------
# Save to file
# ---------------------------------------------------------------------------
out_dir  = _REPO_ROOT / "diags" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "benchmark_baseline.txt"

with open(out_path, "w") as f:
    f.write("CLM-ML-JAX  |  GPU Benchmark Baseline  |  CHATS7 site\n")
    f.write(SEP + "\n")
    f.write(f"JAX backend:           {backend.upper()}  —  {device_str}\n")
    f.write(f"JAX version:           {jax.__version__}\n")
    f.write(f"Site / forcing:        CHATS7  (2007-05.nc)\n")
    f.write(f"dtstep:                {dtstep_sec} s\n")
    f.write(f"num_ml_steps:          {num_ml_steps}  (dtime_ml = {MLclm_varctl.dtime_ml} s)\n")
    f.write(f"RK inner steps:        {nrk_inner}  (runge_kutta_type = {MLclm_varctl.runge_kutta_type})\n")
    f.write(f"num_exposedvegp:       {num_evp}\n")
    f.write("\n")
    f.write(f"Compile + first run (timestep 1):    {t_compile:.3f} s\n")
    f.write("\n")
    f.write(f"Steady-state ({N_STEADY} timesteps, jax.block_until_ready):\n")
    f.write(f"  mean:                {mean_steady:.4f} s\n")
    f.write(f"  std:                 {std_steady:.4f} s\n")
    f.write(f"  min:                 {min_steady:.4f} s\n")
    f.write(f"  max:                 {max_steady:.4f} s\n")
    f.write(f"  per-ml-substep est:  {time_per_substep*1000:.2f} ms\n")
    f.write("\n")
    f.write(f"Device memory:         {mem_info_str}\n")
    f.write(SEP + "\n")
    f.write("\nNOTE: Per-module breakdown requires manual instrumentation.\n")
    f.write("Non-diff mode (grid=None) — standard forward pass, no gradient tracking.\n")
    f.write("Timing via ModelAdvance (includes met I/O, output writing).\n")
    f.write("Direct MLCanopyFluxes calls OOM on CUDA graph instantiation after\n")
    f.write("ModelAdvance has already populated all CUDA graphs (14-graph limit).\n")

print(f"\nResults saved to: {out_path}")
