#!/usr/bin/env python3
"""Diagnose which JAX functions are recompiling across CLM timesteps.

Run with:
    python diags/debug_recompile.py 2>&1 | grep -i "compil" | head -200

JAX_LOG_COMPILES=1 must be set before any jax import.
"""
import os
os.environ["JAX_LOG_COMPILES"] = "1"   # must be set before importing jax

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT  = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC_ROOT))

# ---------------------------------------------------------------------------
# Model imports (copy pattern from benchmark_gpu.py)
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
# Locate the namelist
# ---------------------------------------------------------------------------
namelist_file = str(_SRC_ROOT / "offline_executable" / "nl.CHATS7.1day")
sys.argv = ["debug_recompile.py", namelist_file]

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
# Configure model globals
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
ntimes_met = 2

# ---------------------------------------------------------------------------
# InitializeRealize
# ---------------------------------------------------------------------------
print("=== INIT START ===", flush=True)
InitializeRealize(bounds)

# ---------------------------------------------------------------------------
# Post-init imports
# ---------------------------------------------------------------------------
from clm_src_main import clm_instMod
import clm_src_main.clm_driver as _clm_driver_mod
from offline_driver.CLMml_driver import init_acclim, TowerVeg, SoilInit
from offline_driver.TowerMetMod import TowerMetCurr, TowerMetNext

_new_filter = _filterMod.setFilters(_filterMod.filter)
_filterMod.filter       = _new_filter
_clm_driver_mod.filter  = _new_filter

# Orbital parameters
_eccen, _obliq, _mvelp, _obliqr, _lambm0, _mvelpp = shr_orb_params(iyear)
clm_varorb.eccen  = _eccen
clm_varorb.obliqr = _obliqr
clm_varorb.lambm0 = _lambm0
clm_varorb.mvelpp = _mvelpp

# Acclimation, vegetation, and soil initialization
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

print("=== INIT DONE ===", flush=True)

from clm_src_utils.clm_time_manager import get_curr_date


def _get_time_indx() -> int:
    curr_calday_now = get_curr_calday(offset=0)
    return round((curr_calday_now - start_calday_clm) * 86400.0 / float(dtstep_sec)) + 1


# ---------------------------------------------------------------------------
# Run 3 timesteps with logging, print markers between them
# ---------------------------------------------------------------------------
N_STEPS = 3
for k in range(N_STEPS):
    print(f"\n=== TIMESTEP {k+1} START ===", flush=True)
    t0 = time.perf_counter()
    time_indx = _get_time_indx()
    ModelAdvance(bounds, time_indx, fin1, fin2)
    jax.block_until_ready(clm_instMod.mlcanopy_inst.tleaf_leaf)
    elapsed = time.perf_counter() - t0
    print(f"=== TIMESTEP {k+1} END  (elapsed={elapsed:.3f}s) ===", flush=True)
    if k < N_STEPS - 1:
        clm_time_manager.itim += 1

print("\nDone.", flush=True)
