"""Fortran's conservation checks: recorded inside the compiled step (the largest
error of each check, ``Diagnostics.balance_error``) and checked on the host after each
chunk.

Fortran reference: CLM-ml v2, the ``endrun`` checks of ``MLLeafFluxesMod``,
``MLLongwaveRadiationMod``, ``MLCanopyNitrogenProfileMod``, ``MLSoilFluxesMod``,
``MLSolarRadiationMod``, ``MLCanopyFluxesMod`` and ``MLSoilTemperatureMod``.
"""

from __future__ import annotations

import numpy as np

#: (name, tolerance) of each check, in the order of ``Diagnostics.balance_error``
BALANCE_CHECKS = (
    ("leaf energy balance (LeafFluxes), W m-2", 1.0e-3),
    ("total longwave conservation (Norman), W m-2", 1.0e-3),
    ("canopy nitrogen integration (CanopyNitrogenProfile)", 1.0e-6),
    ("soil energy balance (SoilFluxes), W m-2", 1.0e-3),
    ("total solar radiation conservation (TwoStream), W m-2", 1.0e-6),
    ("solar radiation conservation (Norman), W m-2", 1.0e-3),
    ("plant area profile sums to LAI + SAI (MLCanopyFluxes), m2 m-2", 1.0e-6),
    ("vegetation energy balance (CanopyFluxesDiagnostics 1), W m-2", 1.0e-3),
    ("net radiation (CanopyFluxesDiagnostics 2), W m-2", 1.0e-3),
    ("canopy energy balance (CanopyFluxesDiagnostics 3), W m-2", 1.0e-2),
    ("radiation at the canopy top (CanopyFluxesDiagnostics 4), W m-2", 1.0e-3),
    ("soil temperature energy conservation (SoilTemperature), W m-2", 1.0e-6),
    ("sunlit and shaded fractions positive (SolarRadiation; 1 = violated)", 0.0),
)
(LEAF_ENERGY, LONGWAVE, NITROGEN, SOIL_ENERGY, SOLAR, NORMAN_SOLAR, PLANT_AREA,
 CANOPY_ENERGY_1, CANOPY_ENERGY_2, CANOPY_ENERGY_3, CANOPY_ENERGY_4, SOIL_TEMPERATURE,
 SUNLIT_FRACTION) = range(len(BALANCE_CHECKS))  # fmt: skip


class ModelError(RuntimeError):
    """The model stopped: a conservation check failed or the state is unhealthy (Fortran's
    ``endrun``)."""


def check_balance(errors, where: str) -> None:
    """Stop the run if any recorded error (``(..., len(BALANCE_CHECKS))``, maximum taken
    over the leading axes) exceeds its tolerance. ``where`` goes into the message."""
    worst = np.max(np.asarray(errors, dtype=float).reshape(-1, len(BALANCE_CHECKS)), axis=0)
    for (name, tol), e in zip(BALANCE_CHECKS, worst):
        if e > tol:
            raise ModelError(f"ERROR: {name}: error {e:.3g} exceeds {tol:g} ({where})")
