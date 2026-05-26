"""
Debug script to check tridiag_2eq det values during actual computation.
Run with:
    cd /burg-archive/home/al4385/clm-ml-jax
    CLM_ML_NO_CHECKPOINT=1 python diags/debug_tridiag.py
"""
import os, sys
os.environ['CLM_ML_NO_CHECKPOINT'] = '1'
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ── Patch tridiag_2eq to print det values ────────────────────────────────────
import multilayer_canopy.MLMathToolsMod as _mm

_orig_tridiag_2eq = _mm.tridiag_2eq
_call_count = [0]

def _debug_tridiag_2eq(a1, b11, b12, c1, d1, a2, b21, b22, c2, d2, n):
    _call_count[0] += 1
    e11_prev = 0.0
    e12_prev = 0.0
    e21_prev = 0.0
    e22_prev = 0.0
    det_vals = []
    e11_vals = []
    for i in range(n):
        _a1i  = float(a1[i]) if hasattr(a1[i], '__float__') else a1[i]
        _b11i = float(b11[i]) if hasattr(b11[i], '__float__') else b11[i]
        _b12i = float(b12[i]) if hasattr(b12[i], '__float__') else b12[i]
        _c1i  = float(c1[i]) if hasattr(c1[i], '__float__') else c1[i]
        _a2i  = float(a2[i]) if hasattr(a2[i], '__float__') else a2[i]
        _b22i = float(b22[i]) if hasattr(b22[i], '__float__') else b22[i]
        _b21i = float(b21[i]) if hasattr(b21[i], '__float__') else b21[i]
        _c2i  = float(c2[i]) if hasattr(c2[i], '__float__') else c2[i]
        ainv  = _b11i - _a1i * e11_prev
        binv  = _b12i - _a1i * e12_prev
        cinv  = _b21i - _a2i * e21_prev
        dinv  = _b22i - _a2i * e22_prev
        det   = ainv * dinv - binv * cinv
        det_vals.append(det)
        det_safe = max(abs(det), 1.0e-30)
        e11_prev  = dinv * _c1i / det_safe
        e12_prev  = -binv * _c2i / det_safe
        e21_prev  = -cinv * _c1i / det_safe
        e22_prev  = ainv * _c2i / det_safe
        e11_vals.append(abs(e11_prev))
    det_arr = np.array(det_vals)
    e11_arr = np.array(e11_vals)
    print(f'  [tridiag call #{_call_count[0]}, n={n}] '
          f'det: min={det_arr.min():.2e}, max={det_arr.max():.2e}, '
          f'#<0: {(det_arr < 0).sum()}, #<1e-10: {(det_arr < 1e-10).sum()}, '
          f'|e11|: min={e11_arr.min():.2e}, max={e11_arr.max():.2e}')
    return _orig_tridiag_2eq(a1, b11, b12, c1, d1, a2, b21, b22, c2, d2, n)

_mm.tridiag_2eq = _debug_tridiag_2eq

# Also patch MLFluxProfileSolutionMod to use our version
import multilayer_canopy.MLFluxProfileSolutionMod as fps_mod
fps_mod.tridiag_2eq = _debug_tridiag_2eq

# ── Full pipeline setup ───────────────────────────────────────────────────────
namelist_file = str(Path(__file__).parent.parent / "src/offline_executable/nl.CHATS7.1day")
sys.argv = ['debug_tridiag.py', namelist_file]

import offline_executable.main as _main_mod
nml    = _main_mod.read_namelist(namelist_file)
params = nml.get("clmML_inparm", nml.get("clm_inparm", nml.get("clm_input", {})))

tower_name = str(params.get("tower_name", "CHATS7"))
start_ymd  = int(params.get("start_ymd", 20070501))
iyear      = start_ymd // 10000
imonth     = (start_ymd // 100) % 100

fin_tower       = _main_mod._resolve_path(str(params.get("fin_tower", "")))
fin_clm         = _main_mod._resolve_path(str(params.get("fin_clm", "")))
fin_soil_adjust = _main_mod._resolve_path(str(params.get("fin_soil_adjust", "")))
dirout_raw      = str(params.get("dirout", "")).strip()
dirout          = _main_mod._resolve_path(dirout_raw) if dirout_raw else ""

from offline_driver import controlMod, TowerDataMod, clmSoilOptionMod
from multilayer_canopy import MLclm_varctl
from clm_src_utils import clm_time_manager
from clm_src_utils.clm_time_manager import get_curr_calday

controlMod.tower_site = tower_name
controlMod.iyear = iyear
controlMod.imonth = imonth

tower_num = 0
for i in range(1, int(TowerDataMod.ntower) + 1):
    if tower_name == str(TowerDataMod.tower_id[i]):
        tower_num = i
        break
TowerDataMod.tower_num = tower_num

clmSoilOptionMod.clm_phys  = str(params.get("clm_phys", clmSoilOptionMod.clm_phys))
clmSoilOptionMod.nlev_soil_adjust = int(params.get("nlev_soil_adjust", 0))
clmSoilOptionMod.fin_soil_adjust  = fin_soil_adjust
MLclm_varctl.met_type   = int(params.get("met_type",   MLclm_varctl.met_type))
MLclm_varctl.dpai_min   = float(params.get("dpai_min", MLclm_varctl.dpai_min))
MLclm_varctl.pftcon_val = int(params.get("pftcon_val", MLclm_varctl.pftcon_val))

clm_time_manager.start_date_ymd = start_ymd
clm_time_manager.start_date_tod = int(params.get("start_tod", 0))
dtstep_sec = int(TowerDataMod.tower_time[tower_num]) * 60
clm_time_manager.dtstep = dtstep_sec

bounds = _main_mod.build_bounds(nml)
ntimes = 2

from clm_src_cpl.lnd_comp_nuopc import InitializeRealize, ModelAdvance
from clm_src_main import filterMod as _filterMod

InitializeRealize(bounds)
from clm_src_main import clm_instMod

_new_filter = _filterMod.setFilters(_filterMod.filter)
_filterMod.filter = _new_filter
import clm_src_main.clm_driver as _clm_driver_mod
_clm_driver_mod.filter = _new_filter

from offline_driver.CLMml_driver import init_acclim, TowerVeg, SoilInit
from offline_driver.TowerMetMod import TowerMetCurr, TowerMetNext
from clm_share.shr_orb_mod import shr_orb_params
from clm_src_utils import clm_varorb

_eccen, _obliq, _mvelp, _obliqr, _lambm0, _mvelpp = shr_orb_params(iyear)
clm_varorb.eccen  = _eccen
clm_varorb.obliqr = _obliqr
clm_varorb.lambm0 = _lambm0
clm_varorb.mvelpp = _mvelpp

(clm_instMod.atm2lnd_inst, clm_instMod.wateratm2lndbulk_inst,
 clm_instMod.temperature_inst, clm_instMod.frictionvel_inst,
 clm_instMod.mlcanopy_inst) = init_acclim(
    fin_tower, tower_num, ntimes, bounds.begp, bounds.endp,
    clm_instMod.atm2lnd_inst, clm_instMod.wateratm2lndbulk_inst,
    clm_instMod.temperature_inst, clm_instMod.frictionvel_inst,
    clm_instMod.mlcanopy_inst,
)
(clm_instMod.canopystate_inst, clm_instMod.mlcanopy_inst) = TowerVeg(
    tower_num, bounds.begp, bounds.endp,
    clm_instMod.canopystate_inst, clm_instMod.mlcanopy_inst,
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

(clm_instMod.waterstatebulk_inst, clm_instMod.temperature_inst) = SoilInit(
    fin_clm, soil_init_time_indx, bounds.begc, bounds.endc,
    clm_instMod.soilstate_inst, clm_instMod.waterstatebulk_inst,
    clm_instMod.temperature_inst,
)

os.makedirs(dirout, exist_ok=True)

met_type = MLclm_varctl.met_type
clm_time_manager.itim = 1
curr_calday_ts1 = get_curr_calday(offset=0)
time_indx = round((curr_calday_ts1 - start_calday_clm) * 86400.0 / float(dtstep_sec)) + 1

(clm_instMod.atm2lnd_inst, clm_instMod.wateratm2lndbulk_inst,
 clm_instMod.frictionvel_inst) = TowerMetCurr(
    fin_tower, 1, tower_num, bounds.begp, bounds.endp,
    clm_instMod.atm2lnd_inst, clm_instMod.wateratm2lndbulk_inst,
    clm_instMod.frictionvel_inst,
)
if met_type == 3:
    clm_instMod.mlcanopy_inst = TowerMetNext(
        fin_tower, min(2, ntimes), bounds.begp, bounds.endp,
        clm_instMod.mlcanopy_inst,
    )

print("Running 1 ModelAdvance to warm up state...")
ModelAdvance(bounds, time_indx, fin_tower, fin_clm)
print(f"tridiag called {_call_count[0]} times during warm-up timestep")
_call_count[0] = 0

# ── Now run 1 step in diff mode (Euler, 1 sub-step) to see det values ──────────
MLclm_varctl.runge_kutta_type = 10
MLclm_varctl.dtime_ml         = float(clm_time_manager.dtstep)

from multilayer_canopy.MLCanopyFluxesMod import make_clm_ml_forward, GridInfo
from clm_src_main import clm_instMod

mlcanopy_inst = clm_instMod.mlcanopy_inst
filt = _clm_driver_mod.filter
num_evp = filt.num_exposedvegp
evp_list = list(int(v) for v in filt.exposedvegp[:num_evp])
p = int(evp_list[0])

print(f"\nRunning 1 forward step (diff mode, Euler)...")
forward_fn = make_clm_ml_forward(
    mlcanopy_inst_template=mlcanopy_inst,
    bounds=bounds,
    num_exposedvegp=num_evp,
    filter_exposedvegp=evp_list,
    atm2lnd_inst=clm_instMod.atm2lnd_inst,
    canopystate_inst=clm_instMod.canopystate_inst,
    soilstate_inst=clm_instMod.soilstate_inst,
    temperature_inst=clm_instMod.temperature_inst,
    waterstatebulk_inst=clm_instMod.waterstatebulk_inst,
    waterfluxbulk_inst=clm_instMod.waterfluxbulk_inst,
    energyflux_inst=clm_instMod.energyflux_inst,
    frictionvel_inst=clm_instMod.frictionvel_inst,
    surfalb_inst=clm_instMod.surfalb_inst,
    solarabs_inst=clm_instMod.solarabs_inst,
    wateratm2lndbulk_inst=clm_instMod.wateratm2lndbulk_inst,
    waterdiagnosticbulk_inst=clm_instMod.waterdiagnosticbulk_inst,
)
loss = forward_fn(mlcanopy_inst)
print(f"Forward loss = {float(loss):.4f}")
print(f"tridiag called {_call_count[0]} times during forward pass")
