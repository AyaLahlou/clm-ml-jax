"""
Grid domains: gridcells and vegetated patches from a CLM surface dataset.

``domain.yaml``::

    type: grid
    min_patch_weight: 0.01        # of the gridcell's vegetated area (then renormalized)
    grid:
      surface_data: .../surfdata_0.9x1.25_hist_2000_16pfts_c251022.nc
      land_mask: .../<case>.clm2.h0a.2000-01.nc   # CLM history file: landmask == 1
      region: {lat: [25, 50], lon: [-125, -65]}   # optional box (degrees north / east)

What the loader takes from the surface data:

- **Gridcells**: the surface-data cells with ``landmask == 1`` in the CLM history file (the
  CLM run whose soil moisture is prescribed) and some vegetation; their latitude and
  longitude (``LATIXY``, ``LONGXY``).
- **Patches**: one per PFT 1-16 with a non-zero area: natural PFTs 1-14 from
  ``PCT_NATVEG * PCT_NAT_PFT`` and the crop types 15-16 from ``PCT_CROP * PCT_CFT``. Bare
  ground (PFT 0) and the other land units are left out (decision 29): the weights are the
  patches' fractions of the gridcell's vegetated area. Patches below ``min_patch_weight``
  are dropped and the rest renormalized (as for towers).
- **Canopy top and bottom** (``MONTHLY_HEIGHT_TOP/BOT``): CLM's satellite-phenology values
  at the run's first step (:func:`clm_monthly_weights`); the layer structure is fixed at
  setup. (In the 2000 surface data the heights do not vary by month.) The bottom is kept in
  the domain but not used: CLM-ml's leaf-area profile runs from the ground to the top, and
  CLM uses the bottom height only for snow burial, which this model does not have (step 5).
- **Forcing height** per patch, CLM's rule: ``forcing_height + z0m + displa`` with
  ``z0m = z0mr * htop`` and ``displa = displar * htop`` (CLM's ``ZengWang2007`` roughness;
  ``forc_hgt_u_patch = forc_hgt_u + z0m + displa``, ``Sa_z = 30 m`` in CLM's data
  atmosphere), with ``z0mr`` and ``displar`` the PFT's ``z0m_to_height_ratio`` and
  ``displacement_to_height_ratio`` in params.yaml (e.g. the built-in ``clm6``).
- **Soil column** (one per patch, ``soil.columns: per_patch``): the top level of
  ``PCT_SAND``, ``PCT_CLAY``, ``ORGANIC``; ``SOIL_COLOR``; ``zbedrock``. Organic matter
  is capped at CLM-ml's ``organic_max`` (130 kg m-3): CLM caps the organic fraction at 1
  (``om_frac = min(organic / organic_max, 1)``), CLM-ml's setup formula has no cap (its
  towers never exceed it), and some peat soils of the surface data reach 318 kg m-3.
- **Root biomass**: not in a CLM surface dataset (CLM has no fine-root biomass in
  satellite-phenology mode); ``grid.root_biomass`` for every patch (default 500 g m-2, the
  value of every CLM-ml forest tower). Leaf and stem area profiles: the PFT defaults.

Fortran reference: CTSM surfrdMod.F90 (surfrd_veg_all), SatellitePhenologyMod.F90
(interpMonthlyVeg), FrictionVelocityMod.F90 / CanopyFluxesMod.F90 (forc_hgt_u_patch).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

from clm_ml.io import netcdf
from clm_ml.io.yaml_schema import ConfigError, option

#: Days per month (CLM's interpMonthlyVeg, a no-leap table also in Gregorian runs)
NDAYPM = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

#: PFT index of each natural-vegetation (PCT_NAT_PFT 0-14) and crop (PCT_CFT 0-1) slot
N_NATPFT = 15  # natpft 0 (bare ground) to 14
CFT_PFTS = (15, 16)  # c3_crop, c3_irrigated

#: kg m-3: CLM-ml's organic_max (SoilStateInitTimeConstMod); organic matter is capped at it
ORGANIC_MAX = 130.0


@dataclasses.dataclass(frozen=True)
class RegionSection:
    """A latitude-longitude box."""

    lat: tuple[float, float] = option((-90.0, 90.0), "Latitudes [south, north], degrees north.")
    lon: tuple[float, float] = option(
        (-180.0, 360.0), "Longitudes [west, east], degrees east (may wrap)."
    )


@dataclasses.dataclass(frozen=True)
class GridSection:
    """``domain.yaml`` ``grid``: the gridcells and patches from a CLM surface dataset."""

    surface_data: str = option(
        doc="CLM surface dataset (natural PFTs, their weights, canopy heights, monthly LAI and "
        "SAI, soil texture and colour).",
        path=True,
    )
    land_mask: str = option(
        doc="A CLM history file with landmask on the surface-data grid.", path=True
    )
    region: Union[Literal["global"], RegionSection] = option(
        "global",
        "The gridcells used: global, or those in a box {lat: [south, north], "
        "lon: [west, east]}.",
    )
    root_biomass: float = option(500.0, "Root biomass of every patch (g m-2).")
    forcing_height: float = option(
        30.0, "Forcing height above the zero-plane displacement plus z0m (m; CLM's Sa_z)."
    )
    # Test setting (hidden): gridcells beyond this many are dropped; null: all
    max_gridcells: Optional[int] = option(None, hidden=True)


def clm_monthly_weights(month: int, day: int) -> tuple[int, int, float, float]:
    """CLM's satellite-phenology time weights (``interpMonthlyVeg``) for a date: the two
    months (1-12) whose monthly values are interpolated and their weights. CLM evaluates
    them for each step at the step's end (:func:`satellite_phenology_date`); the weights
    change once a day, and a month's value is reached at the middle of
    its ``NDAYPM`` days. ``NDAYPM`` has 28 days for February in every calendar, as in CTSM,
    so in a Gregorian run 29 February gives (29 - 0.5)/28 > 1: its weights step slightly
    beyond 1 March's, exactly as CLM's do (the noleap calendar has no 29 February).

    Fortran reference: SatellitePhenologyMod.F90, subroutine interpMonthlyVeg."""
    t = (day - 0.5) / NDAYPM[month - 1]
    it1 = int(t + 0.5)  # Fortran integer assignment (truncation; t + 0.5 > 0)
    it2 = it1 + 1
    m1, m2 = month + it1 - 1, month + it2 - 1
    if m1 < 1:
        m1 = 12
    if m2 > 12:
        m2 = 1
    w1 = (it1 + 0.5) - t
    return m1, m2, w1, 1.0 - w1


def satellite_phenology_date(
    step_start: dt.datetime, step_seconds: int, calendar: str = "GREGORIAN"
) -> dt.datetime:
    """The date of the monthly-vegetation time weights for the step starting at
    ``step_start``: the end of the step, in the model calendar.

    As CTSM: ``interpMonthlyVeg`` (start of step n, ``get_curr_date(offset = dtime)``, the end
    of step n + one step) sets the weights that ``UpdateSatellitePhenologyCanopy`` uses after
    ``CanopyFluxes`` of step n, for the next step's leaf and stem area (``clm_driver.F90``):
    step n + 1 sees the weights at its own end. (Phase 6 review, R9: the end of the step plus
    one step had been used here, one step early.)"""
    from clm_ml.calendar import add_seconds

    return add_seconds(step_start, step_seconds, calendar)


def _in_region(lat, lon, region):
    if region is None or region == "global":
        return np.ones(lat.shape, dtype=bool)
    s, n = region.lat
    w, e = (float(x) % 360.0 for x in region.lon)
    lon360 = np.mod(lon, 360.0)
    if region.lon[1] - region.lon[0] >= 360.0:
        in_lon = np.ones(lat.shape, dtype=bool)
    elif w <= e:
        in_lon = (lon360 >= w) & (lon360 <= e)
    else:  # wraps through 0 E
        in_lon = (lon360 >= w) | (lon360 <= e)
    return (lat >= s) & (lat <= n) & in_lon


def latlon_map(lat2, lon2, rows, cols, region):
    """The latitude-longitude map of the gridcell output (decision 108): the gridcells at
    ``rows``, ``cols`` of the surface data (``LATIXY``, ``LONGXY``) on the grid's rows and
    columns inside the ``region`` (all of them for ``global``). The longitudes of a region
    are in its own convention, increasing eastward from its west edge (e.g. -100 to -95,
    or -10 to 20 through 0 E); a region spanning every longitude keeps the grid's. None if
    the surface data is not a regular latitude-longitude grid (the output stays per
    gridcell)."""
    from clm_ml.domain import LatLonMap

    lat1, lon1 = lat2[:, 0].astype(float), lon2[0, :].astype(float)
    if not (np.allclose(lat2, lat1[:, None]) and np.allclose(lon2, lon1[None, :])):
        return None
    row_axis, col_axis, lon = np.arange(lat1.size), np.arange(lon1.size), lon1
    if region is not None and region != "global":
        row_axis = np.flatnonzero((lat1 >= region.lat[0]) & (lat1 <= region.lat[1]))
        west, east = region.lon
        if east - west < 360.0:
            # the gridcells' own test (_in_region), so every gridcell is on the map
            inside = np.flatnonzero(_in_region(np.full_like(lon1, region.lat[0]), lon1, region))
            east_of_west = np.mod(lon1 - west, 360.0)  # degrees east of the west edge
            col_axis = inside[np.argsort(east_of_west[inside], kind="stable")]
            lon = west + east_of_west
    row_of, col_of = np.full(lat1.size, -1), np.full(lon1.size, -1)
    row_of[row_axis], col_of[col_axis] = np.arange(row_axis.size), np.arange(col_axis.size)
    return LatLonMap(
        lat=lat1[row_axis], lon=lon[col_axis], index=np.stack([row_of[rows], col_of[cols]], 1)
    )


def load_grid_domain(
    section: GridSection,
    *,
    params,
    config,
    start: dt.datetime,
    step_seconds: int,
    min_patch_weight: float = 0.0,
    resolve=lambda p: Path(p),
):
    """The :class:`clm_ml.domain.Domain` of a grid run (module docstring). ``start`` and
    ``step_seconds`` give the first step (for the canopy heights, CLM's time weights);
    ``resolve`` turns the folder's relative paths into absolute paths."""

    from clm_ml.domain import Domain, Gridcells, Patches, SoilColumns

    surf, mask_file = resolve(section.surface_data), resolve(section.land_mask)
    for key, path in (("surface_data", surf), ("land_mask", mask_file)):
        if not Path(path).exists():
            raise ConfigError(f"domain.yaml: grid.{key}: file not found: {path}")

    with netcdf.dataset(mask_file) as ds:
        landmask = np.asarray(ds.variables["landmask"][:]) == 1
        cell_km2 = np.ma.filled(ds.variables["area"][:], np.nan).astype(float)
        landfrac = np.ma.filled(ds.variables["landfrac"][:], np.nan).astype(float)
    with netcdf.dataset(surf) as ds:

        def var(name):
            return np.ma.filled(ds.variables[name][:], np.nan).astype(float)

        lat2, lon2 = var("LATIXY"), var("LONGXY")
        if landmask.shape != lat2.shape:
            raise ConfigError(
                f"{mask_file}: landmask has shape {landmask.shape}, the surface data "
                f"{lat2.shape} (both must be on the same grid)"
            )
        natveg, crop = var("PCT_NATVEG"), var("PCT_CROP")
        nat_pft, cft = var("PCT_NAT_PFT"), var("PCT_CFT")
        if nat_pft.shape[0] != N_NATPFT or cft.shape[0] != len(CFT_PFTS):
            raise ConfigError(
                f"{surf}: expected {N_NATPFT} natural PFTs and {len(CFT_PFTS)} crop types "
                f"(a 16-PFT surface dataset); got {nat_pft.shape[0]} and {cft.shape[0]}"
            )
        # area of each PFT 1-16 in percent of the gridcell, (16, lat, lon)
        area = np.concatenate([natveg[None] * nat_pft[1:] / 100.0, crop[None] * cft / 100.0])
        area = np.where(np.isfinite(area), area, 0.0)
        vegetated = area.sum(axis=0)
        cells = landmask & (vegetated > 0.0) & _in_region(lat2, lon2, section.region)
        rows, cols = np.nonzero(cells)
        if section.max_gridcells is not None:
            rows, cols = rows[: section.max_gridcells], cols[: section.max_gridcells]
        if rows.size == 0:
            raise ConfigError("domain.yaml: grid: no vegetated land gridcell in the domain")

        # CLM's satellite-phenology heights at the first step
        when = satellite_phenology_date(start, step_seconds)
        m1, m2, w1, w2 = clm_monthly_weights(when.month, when.day)

        def at_cells(name, level):  # one level of a (level, lat, lon) field
            return np.ma.filled(ds.variables[name][level], np.nan)[rows, cols].astype(float)

        def monthly(name):  # (16, cells): PFTs 1-16 at the first step
            v = ds.variables[name]
            a = np.ma.filled(v[m1 - 1], np.nan)[1:17][:, rows, cols].astype(float)
            b = np.ma.filled(v[m2 - 1], np.nan)[1:17][:, rows, cols].astype(float)
            return w1 * a + w2 * b

        htop, hbot = monthly("MONTHLY_HEIGHT_TOP"), monthly("MONTHLY_HEIGHT_BOT")
        sand, clay = at_cells("PCT_SAND", 0), at_cells("PCT_CLAY", 0)
        organic = np.minimum(at_cells("ORGANIC", 0), ORGANIC_MAX)  # CLM: om_frac <= 1
        color = np.ma.filled(ds.variables["SOIL_COLOR"][:], 0)[rows, cols].astype(int)
        zbed = np.ma.filled(ds.variables["zbedrock"][:], np.nan)[rows, cols].astype(float)

    cell_area = area[:, rows, cols]  # (16, ng)
    frac = cell_area / cell_area.sum(axis=0, keepdims=True)
    keep = frac > 0.0
    if min_patch_weight > 0.0:
        keep &= frac >= min_patch_weight
        empty = ~keep.any(axis=0)
        if empty.any():  # keep each gridcell's largest patch
            keep[np.argmax(frac[:, empty], axis=0), np.flatnonzero(empty)] = True
    weight = np.where(keep, frac, 0.0)
    weight = weight / weight.sum(axis=0, keepdims=True)
    g_of, k_pft = np.nonzero(keep.T)  # gridcell-major order, then PFT
    pft = k_pft + 1
    n_p = pft.size

    ratios = ("z0m_to_height_ratio", "displacement_to_height_ratio")
    missing = params.missing_for(sorted(set(pft.tolist())), ratios)
    if missing:
        raise ConfigError(
            f"params ({params.source}): no value for {missing[:4]}: grid runs need each PFT's "
            "roughness ratios for the forcing height (CLM's rule), e.g. params: clm6"
        )
    z0mr = np.asarray(params.values[ratios[0]], dtype=float)[pft]
    displar = np.asarray(params.values[ratios[1]], dtype=float)[pft]
    top = htop[k_pft, g_of]
    if not np.all(np.isfinite(top)) or np.any(top <= 0.0):
        bad = int(np.sum(~(top > 0.0)))
        raise ConfigError(f"{surf}: {bad} patches have no canopy height (MONTHLY_HEIGHT_TOP)")
    zref = section.forcing_height + (z0mr + displar) * top

    lat, lon = lat2[rows, cols], lon2[rows, cols]
    names = tuple(f"{la:.2f}N_{lo:.2f}E" for la, lo in zip(lat, lon))
    gridcells = Gridcells(
        names=names,
        latitude=lat.astype(float),
        longitude=lon.astype(float),
        measurement_height=np.full(rows.size, float(section.forcing_height)),
        grid_index=np.stack([rows, cols], axis=1),
        latlon=latlon_map(lat2, lon2, rows, cols, section.region),
        area=cell_km2[rows, cols],
        land_fraction=landfrac[rows, cols],
        vegetated_fraction=vegetated[rows, cols] / 100.0,  # PCT_* are percent of the land
    )
    patches = Patches(
        gridcell_index=g_of.astype(int),
        column_index=np.arange(n_p),
        pft_index=pft.astype(int),
        weight=weight[k_pft, g_of],
        canopy_height=top,
        root_biomass=np.full(n_p, float(section.root_biomass)),
        leaf_area_profile_beta=np.full((n_p, 2), np.nan),
        stem_area_profile_beta=np.full((n_p, 2), np.nan),
        n_layers_within=np.zeros(n_p, dtype=int),
        n_layers_above=np.zeros(n_p, dtype=int),
        canopy_bottom=hbot[k_pft, g_of],
        measurement_height=zref,
    )
    columns = SoilColumns(
        texture_class=(None,) * n_p,
        sand_percent=sand[g_of],
        clay_percent=clay[g_of],
        organic_matter=organic[g_of],
        color_class=color[g_of],
        bedrock_depth=zbed[g_of],
    )
    domain = Domain(gridcells, patches, columns, towers={})
    missing = params.missing_for(sorted(set(pft.tolist())))
    if missing:
        raise ConfigError(f"params: no value for {missing[:5]} (used by the domain)")
    bad_soil = (sand + clay <= 0.0) | ~np.isfinite(sand + clay + organic + zbed)
    if bad_soil.any():
        raise ConfigError(
            f"{surf}: {int(bad_soil.sum())} gridcells have no top-level sand/clay, organic "
            f"matter or bedrock depth, e.g. {names[int(np.flatnonzero(bad_soil)[0])]}"
        )
    c = config.canopy_layers
    if isinstance(c.layer_counts, str):
        raise ConfigError(
            "domain.yaml type grid: set model.yaml canopy_layers.layer_counts to "
            "{within: N, above: M} (constant layer counts, decision 28)"
        )
    if c.layer_counts.within + c.layer_counts.above > c.max_layers:
        raise ConfigError(
            f"model.yaml: canopy_layers: {c.layer_counts.within} + {c.layer_counts.above} "
            f"layers exceed max_layers {c.max_layers}"
        )
    return domain
