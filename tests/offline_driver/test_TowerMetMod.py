"""
Comprehensive pytest suite for TowerMetMod module.

This test suite covers:
- partition_solar_radiation: Solar radiation partitioning into direct/diffuse and vis/NIR bands
- relative_humidity_to_specific_humidity: RH to specific humidity conversion
- specific_humidity_to_vapor_pressure: Specific humidity to vapor pressure conversion
- calculate_longwave_radiation: Longwave radiation calculation from temperature and vapor pressure
- assign_co2_o2_coordinates: CO2/O2 partial pressure and coordinate assignment
- process_tower_met_forcing: Column-level forcing variable processing
- process_tower_met: Complete end-to-end tower meteorology processing
- get_default_solar_params: Default solar radiation parameters
- get_default_constants: Default physical constants

Test coverage includes:
- Nominal cases: Typical atmospheric conditions across climate zones
- Edge cases: Zero values, boundary conditions, extreme but valid scenarios
- Physical realism: Temperature > 0K, valid humidity ranges, realistic radiation values
- Array shapes and data types
"""

import sys
from pathlib import Path
from typing import NamedTuple, Callable

import pytest
import jax.numpy as jnp
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from offline_driver.TowerMetMod import (
    partition_solar_radiation,
    relative_humidity_to_specific_humidity,
    specific_humidity_to_vapor_pressure,
    calculate_longwave_radiation,
    assign_co2_o2_coordinates,
    process_tower_met_forcing,
    process_tower_met,
    get_default_solar_params,
    get_default_constants,
    SolarRadiationParams,
    PhysicalConstants,
    TowerMetRawData,
    TowerMetInputs,
    TowerMetState,
    ColumnForcing,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def default_solar_params():
    """Fixture providing default solar radiation parameters."""
    return SolarRadiationParams(
        a0=0.17639,
        a1=0.0038,
        a2=-9.0039e-06,
        a3=8.1351e-09,
        b0=0.29548,
        b1=0.00504,
        b2=-1.4957e-05,
        b3=1.4881e-08
    )


@pytest.fixture
def default_constants():
    """Fixture providing default physical constants."""
    return PhysicalConstants(
        mmh2o=18.016,
        mmdry=28.966,
        sb=5.67e-08,
        co2_ppm=367.0,
        o2_frac=0.209,
        default_pressure=101325.0,
        default_height=30.0,
        missing_value=-999.0
    )


@pytest.fixture
def mock_qsat_function():
    """
    Mock saturation vapor pressure function for testing.
    
    Uses simplified Clausius-Clapeyron relation:
    e_sat = 611.2 * exp(17.67 * (T - 273.15) / (T - 29.65))
    
    Mimics the signature of sat_vap from MLWaterVaporMod.
    """
    def qsat(temperature):
        """Calculate saturation vapor pressure [Pa] from temperature [K].
        
        Returns:
            Tuple of (esat, degdT) where:
                esat: Saturation vapor pressure [Pa]
                degdT: Temperature derivative [Pa/K] (returned as zero for simplicity)
        """
        t_celsius = temperature - 273.15
        esat = 611.2 * jnp.exp(17.67 * t_celsius / (temperature - 29.65))
        degdT = jnp.zeros_like(esat)  # Derivative not used in tests
        return esat, degdT
    return qsat


# ============================================================================
# Tests for partition_solar_radiation
# ============================================================================

@pytest.mark.parametrize("fsds,expected_sum", [
    (800.0, 800.0),  # Midday clear sky
    (300.0, 300.0),  # Morning conditions
    (0.0, 0.0),      # Nighttime
    (1.0, 1.0),      # Dawn/dusk
    (1400.0, 1400.0), # Extreme high
])
def test_partition_solar_radiation_conservation(fsds, expected_sum, default_solar_params):
    """
    Test that total radiation is conserved in partitioning.
    
    The sum of direct beam and diffuse radiation across both bands
    should equal the input total shortwave radiation.
    """
    forc_solad, forc_solai = partition_solar_radiation(
        fsds=fsds,
        params=default_solar_params,
        ivis=0,
        inir=1
    )
    
    total_output = jnp.sum(forc_solad) + jnp.sum(forc_solai)
    assert jnp.allclose(total_output, expected_sum, atol=1e-6, rtol=1e-6), \
        f"Radiation not conserved: input={fsds}, output={total_output}"


def test_partition_solar_radiation_shapes(default_solar_params):
    """Test that output arrays have correct shape [2] for vis and NIR bands."""
    fsds = 600.0
    forc_solad, forc_solai = partition_solar_radiation(
        fsds=fsds,
        params=default_solar_params,
        ivis=0,
        inir=1
    )
    
    assert forc_solad.shape == (2,), f"Expected shape (2,), got {forc_solad.shape}"
    assert forc_solai.shape == (2,), f"Expected shape (2,), got {forc_solai.shape}"


def test_partition_solar_radiation_non_negative(default_solar_params):
    """Test that all output radiation values are non-negative."""
    test_values = [0.0, 1.0, 300.0, 800.0, 1400.0]
    
    for fsds in test_values:
        forc_solad, forc_solai = partition_solar_radiation(
            fsds=fsds,
            params=default_solar_params,
            ivis=0,
            inir=1
        )
        
        assert jnp.all(forc_solad >= 0), \
            f"Negative direct beam radiation for fsds={fsds}: {forc_solad}"
        assert jnp.all(forc_solai >= 0), \
            f"Negative diffuse radiation for fsds={fsds}: {forc_solai}"


def test_partition_solar_radiation_dtypes(default_solar_params):
    """Test that output arrays have correct JAX float dtype."""
    fsds = 600.0
    forc_solad, forc_solai = partition_solar_radiation(
        fsds=fsds,
        params=default_solar_params,
        ivis=0,
        inir=1
    )
    
    assert jnp.issubdtype(forc_solad.dtype, jnp.floating), \
        f"Expected floating dtype, got {forc_solad.dtype}"
    assert jnp.issubdtype(forc_solai.dtype, jnp.floating), \
        f"Expected floating dtype, got {forc_solai.dtype}"


def test_partition_solar_radiation_zero_input(default_solar_params):
    """Test that zero input radiation produces zero output."""
    forc_solad, forc_solai = partition_solar_radiation(
        fsds=0.0,
        params=default_solar_params,
        ivis=0,
        inir=1
    )
    
    assert jnp.allclose(forc_solad, 0.0, atol=1e-10), \
        f"Expected zero direct beam, got {forc_solad}"
    assert jnp.allclose(forc_solai, 0.0, atol=1e-10), \
        f"Expected zero diffuse, got {forc_solai}"


# ============================================================================
# Tests for relative_humidity_to_specific_humidity
# ============================================================================

@pytest.mark.parametrize("rh,temp,press,esat", [
    (60.0, jnp.array([288.15, 290.15, 285.15]), 
     jnp.array([101325.0, 101325.0, 101325.0]),
     jnp.array([1705.0, 1937.0, 1402.0])),  # Standard conditions
    (30.0, jnp.array([263.15, 268.15, 273.15]),
     jnp.array([95000.0, 96000.0, 97000.0]),
     jnp.array([259.9, 389.8, 611.2])),  # Cold dry
    (85.0, jnp.array([303.15, 305.15, 308.15]),
     jnp.array([101325.0, 101000.0, 100500.0]),
     jnp.array([4242.0, 4758.0, 5622.0])),  # Hot humid
])
def test_relative_humidity_to_specific_humidity_range(rh, temp, press, esat, default_constants):
    """
    Test that specific humidity is in valid range [0, 1] kg/kg.
    
    Specific humidity represents mass fraction of water vapor in air,
    so it must be between 0 (completely dry) and 1 (pure water vapor).
    """
    q = relative_humidity_to_specific_humidity(
        rh=rh,
        temperature=temp,
        pressure=press,
        esat=esat,
        constants=default_constants
    )
    
    assert jnp.all(q >= 0.0), f"Negative specific humidity: {q}"
    assert jnp.all(q <= 1.0), f"Specific humidity > 1: {q}"


def test_relative_humidity_to_specific_humidity_shapes(default_constants):
    """Test that output shape matches input temperature/pressure shape."""
    n_cols = 5
    rh = 60.0
    temp = jnp.full(n_cols, 288.15)
    press = jnp.full(n_cols, 101325.0)
    esat = jnp.full(n_cols, 1705.0)
    
    q = relative_humidity_to_specific_humidity(
        rh=rh,
        temperature=temp,
        pressure=press,
        esat=esat,
        constants=default_constants
    )
    
    assert q.shape == (n_cols,), f"Expected shape ({n_cols},), got {q.shape}"


def test_relative_humidity_to_specific_humidity_zero_rh(default_constants):
    """Test that zero relative humidity produces zero specific humidity."""
    rh = 0.0
    temp = jnp.array([288.15, 290.15])
    press = jnp.array([101325.0, 101325.0])
    esat = jnp.array([1705.0, 1937.0])
    
    q = relative_humidity_to_specific_humidity(
        rh=rh,
        temperature=temp,
        pressure=press,
        esat=esat,
        constants=default_constants
    )
    
    assert jnp.allclose(q, 0.0, atol=1e-10), \
        f"Expected zero specific humidity for RH=0, got {q}"


def test_relative_humidity_to_specific_humidity_saturation(default_constants):
    """Test that 100% RH produces maximum specific humidity for given conditions."""
    rh = 100.0
    temp = jnp.array([288.15, 290.15, 293.15])
    press = jnp.array([101325.0, 101325.0, 101325.0])
    esat = jnp.array([1705.0, 1937.0, 2339.0])
    
    q = relative_humidity_to_specific_humidity(
        rh=rh,
        temperature=temp,
        pressure=press,
        esat=esat,
        constants=default_constants
    )
    
    # At saturation, specific humidity should be relatively high
    assert jnp.all(q > 0.01), f"Expected high specific humidity at saturation, got {q}"
    assert jnp.all(q < 0.03), f"Unrealistically high specific humidity: {q}"


def test_relative_humidity_to_specific_humidity_monotonic(default_constants):
    """Test that specific humidity increases with temperature at constant RH."""
    rh = 60.0
    temps = jnp.array([273.15, 283.15, 293.15, 303.15])  # Increasing temperature
    press = jnp.full(4, 101325.0)
    # Saturation vapor pressure increases with temperature
    esat = jnp.array([611.2, 1228.0, 2339.0, 4242.0])
    
    q = relative_humidity_to_specific_humidity(
        rh=rh,
        temperature=temps,
        pressure=press,
        esat=esat,
        constants=default_constants
    )
    
    # Specific humidity should increase with temperature
    assert jnp.all(q[1:] > q[:-1]), \
        f"Specific humidity not monotonically increasing with temperature: {q}"


# ============================================================================
# Tests for specific_humidity_to_vapor_pressure
# ============================================================================

@pytest.mark.parametrize("q,press", [
    (jnp.array([0.008, 0.01, 0.012]), jnp.array([101325.0, 101325.0, 101325.0])),
    (jnp.array([0.001, 0.002, 0.003]), jnp.array([95000.0, 96000.0, 97000.0])),
    (jnp.array([0.02, 0.022, 0.024, 0.026]), jnp.array([101325.0, 101000.0, 100500.0, 100000.0])),
])
def test_specific_humidity_to_vapor_pressure_non_negative(q, press, default_constants):
    """Test that vapor pressure is always non-negative."""
    e = specific_humidity_to_vapor_pressure(
        q=q,
        pressure=press,
        constants=default_constants
    )
    
    assert jnp.all(e >= 0.0), f"Negative vapor pressure: {e}"


def test_specific_humidity_to_vapor_pressure_shapes(default_constants):
    """Test that output shape matches input shape."""
    n_cols = 7
    q = jnp.full(n_cols, 0.01)
    press = jnp.full(n_cols, 101325.0)
    
    e = specific_humidity_to_vapor_pressure(
        q=q,
        pressure=press,
        constants=default_constants
    )
    
    assert e.shape == (n_cols,), f"Expected shape ({n_cols},), got {e.shape}"


def test_specific_humidity_to_vapor_pressure_zero_humidity(default_constants):
    """Test that zero specific humidity produces zero vapor pressure."""
    q = jnp.array([0.0, 0.0, 0.0])
    press = jnp.array([101325.0, 101325.0, 101325.0])
    
    e = specific_humidity_to_vapor_pressure(
        q=q,
        pressure=press,
        constants=default_constants
    )
    
    assert jnp.allclose(e, 0.0, atol=1e-10), \
        f"Expected zero vapor pressure for q=0, got {e}"


def test_specific_humidity_to_vapor_pressure_proportional(default_constants):
    """Test that vapor pressure is proportional to specific humidity at constant pressure."""
    press = jnp.full(4, 101325.0)
    q_values = jnp.array([0.005, 0.010, 0.015, 0.020])
    
    e = specific_humidity_to_vapor_pressure(
        q=q_values,
        pressure=press,
        constants=default_constants
    )
    
    # Vapor pressure should increase proportionally with specific humidity
    ratios = e[1:] / e[:-1]
    expected_ratio = q_values[1:] / q_values[:-1]
    
    assert jnp.allclose(ratios, expected_ratio, rtol=0.01), \
        f"Vapor pressure not proportional to specific humidity: ratios={ratios}, expected={expected_ratio}"


def test_specific_humidity_to_vapor_pressure_less_than_pressure(default_constants):
    """Test that vapor pressure is always less than total atmospheric pressure."""
    q = jnp.array([0.01, 0.015, 0.02])
    press = jnp.array([101325.0, 95000.0, 90000.0])
    
    e = specific_humidity_to_vapor_pressure(
        q=q,
        pressure=press,
        constants=default_constants
    )
    
    assert jnp.all(e < press), \
        f"Vapor pressure exceeds atmospheric pressure: e={e}, press={press}"


# ============================================================================
# Tests for calculate_longwave_radiation
# ============================================================================

@pytest.mark.parametrize("temp,vp", [
    (jnp.array([288.15, 290.15, 285.15]), jnp.array([1200.0, 1400.0, 1000.0])),
    (jnp.array([253.15, 258.15, 263.15]), jnp.array([100.0, 150.0, 250.0])),
    (jnp.array([303.15, 305.15, 308.15]), jnp.array([3000.0, 3500.0, 4200.0])),
])
def test_calculate_longwave_radiation_non_negative(temp, vp, default_constants):
    """Test that longwave radiation is always non-negative."""
    lwrad = calculate_longwave_radiation(
        temperature=temp,
        vapor_pressure=vp,
        constants=default_constants
    )
    
    assert jnp.all(lwrad >= 0.0), f"Negative longwave radiation: {lwrad}"


def test_calculate_longwave_radiation_shapes(default_constants):
    """Test that output shape matches input shape."""
    n_cols = 6
    temp = jnp.full(n_cols, 288.15)
    vp = jnp.full(n_cols, 1200.0)
    
    lwrad = calculate_longwave_radiation(
        temperature=temp,
        vapor_pressure=vp,
        constants=default_constants
    )
    
    assert lwrad.shape == (n_cols,), f"Expected shape ({n_cols},), got {lwrad.shape}"


def test_calculate_longwave_radiation_increases_with_temperature(default_constants):
    """Test that longwave radiation increases with temperature (Stefan-Boltzmann)."""
    temps = jnp.array([253.15, 273.15, 293.15, 313.15])
    vp = jnp.full(4, 1000.0)  # Constant vapor pressure
    
    lwrad = calculate_longwave_radiation(
        temperature=temps,
        vapor_pressure=vp,
        constants=default_constants
    )
    
    # Longwave radiation should increase with temperature
    assert jnp.all(lwrad[1:] > lwrad[:-1]), \
        f"Longwave radiation not increasing with temperature: {lwrad}"


def test_calculate_longwave_radiation_realistic_range(default_constants):
    """Test that longwave radiation is in realistic range for Earth's atmosphere."""
    # Typical atmospheric conditions
    temp = jnp.array([273.15, 288.15, 303.15])  # 0°C, 15°C, 30°C
    vp = jnp.array([611.2, 1705.0, 4242.0])
    
    lwrad = calculate_longwave_radiation(
        temperature=temp,
        vapor_pressure=vp,
        constants=default_constants
    )
    
    # Typical range for downward longwave: 150-550 W/m²
    assert jnp.all(lwrad >= 150.0), f"Unrealistically low longwave radiation: {lwrad}"
    assert jnp.all(lwrad <= 550.0), f"Unrealistically high longwave radiation: {lwrad}"


def test_calculate_longwave_radiation_zero_vapor_pressure(default_constants):
    """Test longwave radiation with zero vapor pressure (dry atmosphere)."""
    temp = jnp.array([288.15, 290.15])
    vp = jnp.array([0.0, 0.0])
    
    lwrad = calculate_longwave_radiation(
        temperature=temp,
        vapor_pressure=vp,
        constants=default_constants
    )
    
    # Should still produce positive radiation (from temperature alone)
    assert jnp.all(lwrad > 0.0), \
        f"Expected positive longwave radiation even with zero vapor pressure, got {lwrad}"


# ============================================================================
# Tests for assign_co2_o2_coordinates
# ============================================================================

def test_assign_co2_o2_coordinates_shapes(default_constants):
    """Test that output arrays have correct shapes."""
    n_cols = 5
    n_gridcells = 2
    forc_pbot = jnp.full(n_cols, 101325.0)
    col_to_gridcell = jnp.array([0, 0, 0, 1, 1])
    
    forc_pco2, forc_po2, latdeg, londeg = assign_co2_o2_coordinates(
        forc_pbot=forc_pbot,
        tower_lat=45.5,
        tower_lon=-93.2,
        col_to_gridcell=col_to_gridcell,
        n_gridcells=n_gridcells,
        constants=default_constants
    )
    
    assert forc_pco2.shape == (n_gridcells,), f"Expected shape ({n_gridcells},), got {forc_pco2.shape}"
    assert forc_po2.shape == (n_gridcells,), f"Expected shape ({n_gridcells},), got {forc_po2.shape}"
    assert latdeg.shape == (n_gridcells,), f"Expected shape ({n_gridcells},), got {latdeg.shape}"
    assert londeg.shape == (n_gridcells,), f"Expected shape ({n_gridcells},), got {londeg.shape}"


def test_assign_co2_o2_coordinates_non_negative(default_constants):
    """Test that CO2 and O2 partial pressures are non-negative."""
    forc_pbot = jnp.array([101325.0, 99000.0, 95000.0])
    col_to_gridcell = jnp.array([0, 0, 1])
    
    forc_pco2, forc_po2, _, _ = assign_co2_o2_coordinates(
        forc_pbot=forc_pbot,
        tower_lat=40.0,
        tower_lon=-105.5,
        col_to_gridcell=col_to_gridcell,
        n_gridcells=2,
        constants=default_constants
    )
    
    assert jnp.all(forc_pco2 >= 0.0), f"Negative CO2 partial pressure: {forc_pco2}"
    assert jnp.all(forc_po2 >= 0.0), f"Negative O2 partial pressure: {forc_po2}"


def test_assign_co2_o2_coordinates_latitude_range(default_constants):
    """Test that latitude is in valid range [-90, 90]."""
    forc_pbot = jnp.array([101325.0, 101325.0])
    col_to_gridcell = jnp.array([0, 0])
    
    test_lats = [-89.5, -45.0, 0.0, 45.5, 89.5]
    
    for lat in test_lats:
        _, _, latdeg, _ = assign_co2_o2_coordinates(
            forc_pbot=forc_pbot,
            tower_lat=lat,
            tower_lon=0.0,
            col_to_gridcell=col_to_gridcell,
            n_gridcells=1,
            constants=default_constants
        )
        
        assert jnp.all(latdeg >= -90.0), f"Latitude below -90: {latdeg}"
        assert jnp.all(latdeg <= 90.0), f"Latitude above 90: {latdeg}"
        assert jnp.allclose(latdeg, lat, atol=1e-6), \
            f"Latitude mismatch: expected {lat}, got {latdeg}"


def test_assign_co2_o2_coordinates_longitude_range(default_constants):
    """Test that longitude is in valid range [-180, 180]."""
    forc_pbot = jnp.array([101325.0, 101325.0])
    col_to_gridcell = jnp.array([0, 0])
    
    test_lons = [-179.5, -93.2, 0.0, 93.2, 179.5]
    
    for lon in test_lons:
        _, _, _, londeg = assign_co2_o2_coordinates(
            forc_pbot=forc_pbot,
            tower_lat=0.0,
            tower_lon=lon,
            col_to_gridcell=col_to_gridcell,
            n_gridcells=1,
            constants=default_constants
        )
        
        assert jnp.all(londeg >= -180.0), f"Longitude below -180: {londeg}"
        assert jnp.all(londeg <= 180.0), f"Longitude above 180: {londeg}"
        assert jnp.allclose(londeg, lon, atol=1e-6), \
            f"Longitude mismatch: expected {lon}, got {londeg}"


def test_assign_co2_o2_coordinates_pressure_scaling(default_constants):
    """Test that partial pressures scale with atmospheric pressure."""
    # High altitude (low pressure) vs sea level
    forc_pbot_high = jnp.array([70000.0, 72000.0])
    forc_pbot_sea = jnp.array([101325.0, 101325.0])
    col_to_gridcell = jnp.array([0, 0])
    
    pco2_high, po2_high, _, _ = assign_co2_o2_coordinates(
        forc_pbot=forc_pbot_high,
        tower_lat=0.0,
        tower_lon=0.0,
        col_to_gridcell=col_to_gridcell,
        n_gridcells=1,
        constants=default_constants
    )
    
    pco2_sea, po2_sea, _, _ = assign_co2_o2_coordinates(
        forc_pbot=forc_pbot_sea,
        tower_lat=0.0,
        tower_lon=0.0,
        col_to_gridcell=col_to_gridcell,
        n_gridcells=1,
        constants=default_constants
    )
    
    # Partial pressures should be lower at high altitude
    assert jnp.all(pco2_high < pco2_sea), \
        f"CO2 partial pressure not lower at altitude: high={pco2_high}, sea={pco2_sea}"
    assert jnp.all(po2_high < po2_sea), \
        f"O2 partial pressure not lower at altitude: high={po2_high}, sea={po2_sea}"


# ============================================================================
# Tests for process_tower_met_forcing
# ============================================================================

def test_process_tower_met_forcing_shapes(default_constants, mock_qsat_function):
    """Test that output arrays have correct shapes."""
    n_columns = 3
    n_patches = 5
    
    raw_data = TowerMetRawData(
        zbot=30.0,
        tbot=288.15,
        rhbot=60.0,
        qbot=0.008,
        ubot=3.5,
        fsdsbot=600.0,
        fldsbot=350.0,
        pbot=101325.0,
        prect=0.0
    )
    
    column_forcing, forc_hgt_u = process_tower_met_forcing(
        raw_data=raw_data,
        tower_ht=30.0,
        constants=default_constants,
        n_columns=n_columns,
        n_patches=n_patches,
        sat_vap_func=mock_qsat_function
    )
    
    # Check column forcing shapes
    assert column_forcing.forc_t.shape == (n_columns,), \
        f"Expected forc_t shape ({n_columns},), got {column_forcing.forc_t.shape}"
    assert column_forcing.forc_q.shape == (n_columns,), \
        f"Expected forc_q shape ({n_columns},), got {column_forcing.forc_q.shape}"
    assert column_forcing.forc_pbot.shape == (n_columns,), \
        f"Expected forc_pbot shape ({n_columns},), got {column_forcing.forc_pbot.shape}"
    assert column_forcing.forc_lwrad.shape == (n_columns,), \
        f"Expected forc_lwrad shape ({n_columns},), got {column_forcing.forc_lwrad.shape}"
    assert column_forcing.forc_rain.shape == (n_columns,), \
        f"Expected forc_rain shape ({n_columns},), got {column_forcing.forc_rain.shape}"
    assert column_forcing.forc_snow.shape == (n_columns,), \
        f"Expected forc_snow shape ({n_columns},), got {column_forcing.forc_snow.shape}"
    
    # Check forcing height shape
    assert forc_hgt_u.shape == (n_patches,), \
        f"Expected forc_hgt_u shape ({n_patches},), got {forc_hgt_u.shape}"


def test_process_tower_met_forcing_physical_constraints(default_constants, mock_qsat_function):
    """Test that outputs satisfy physical constraints."""
    raw_data = TowerMetRawData(
        zbot=30.0,
        tbot=288.15,
        rhbot=60.0,
        qbot=0.008,
        ubot=3.5,
        fsdsbot=600.0,
        fldsbot=350.0,
        pbot=101325.0,
        prect=0.0
    )
    
    column_forcing, forc_hgt_u = process_tower_met_forcing(
        raw_data=raw_data,
        tower_ht=30.0,
        constants=default_constants,
        n_columns=3,
        n_patches=5,
        sat_vap_func=mock_qsat_function
    )
    
    # Temperature > 0K
    assert jnp.all(column_forcing.forc_t > 0.0), \
        f"Temperature not positive: {column_forcing.forc_t}"
    
    # Specific humidity in [0, 1]
    assert jnp.all(column_forcing.forc_q >= 0.0), \
        f"Negative specific humidity: {column_forcing.forc_q}"
    assert jnp.all(column_forcing.forc_q <= 1.0), \
        f"Specific humidity > 1: {column_forcing.forc_q}"
    
    # Pressure > 0
    assert jnp.all(column_forcing.forc_pbot > 0.0), \
        f"Non-positive pressure: {column_forcing.forc_pbot}"
    
    # Longwave radiation >= 0
    assert jnp.all(column_forcing.forc_lwrad >= 0.0), \
        f"Negative longwave radiation: {column_forcing.forc_lwrad}"
    
    # Precipitation >= 0
    assert jnp.all(column_forcing.forc_rain >= 0.0), \
        f"Negative rain: {column_forcing.forc_rain}"
    assert jnp.all(column_forcing.forc_snow >= 0.0), \
        f"Negative snow: {column_forcing.forc_snow}"
    
    # Forcing height > 0
    assert jnp.all(forc_hgt_u > 0.0), \
        f"Non-positive forcing height: {forc_hgt_u}"


def test_process_tower_met_forcing_nighttime(default_constants, mock_qsat_function):
    """Test processing of nighttime conditions with zero solar radiation."""
    raw_data = TowerMetRawData(
        zbot=30.0,
        tbot=278.15,
        rhbot=85.0,
        qbot=0.006,
        ubot=0.5,
        fsdsbot=0.0,  # Nighttime
        fldsbot=280.0,
        pbot=101325.0,
        prect=0.0
    )
    
    column_forcing, _ = process_tower_met_forcing(
        raw_data=raw_data,
        tower_ht=30.0,
        constants=default_constants,
        n_columns=2,
        n_patches=3,
        sat_vap_func=mock_qsat_function
    )
    
    # Should still produce valid outputs
    assert jnp.all(jnp.isfinite(column_forcing.forc_t)), \
        "Non-finite temperature in nighttime conditions"
    assert jnp.all(jnp.isfinite(column_forcing.forc_lwrad)), \
        "Non-finite longwave radiation in nighttime conditions"


# ============================================================================
# Tests for process_tower_met (end-to-end)
# ============================================================================

def test_process_tower_met_complete_state(default_solar_params, default_constants, mock_qsat_function):
    """Test that complete processing produces all required state variables."""
    raw_data = TowerMetRawData(
        zbot=30.0,
        tbot=288.15,
        rhbot=60.0,
        qbot=0.008,
        ubot=3.5,
        fsdsbot=600.0,
        fldsbot=350.0,
        pbot=101325.0,
        prect=0.0
    )
    
    inputs = TowerMetInputs(
        raw_data=raw_data,
        tower_ht=30.0,
        tower_lat=45.5,
        tower_lon=-93.2,
        n_gridcells=1,
        n_columns=3,
        n_patches=5,
        col_to_gridcell=jnp.array([0, 0, 0]),
        col_to_tower=jnp.array([0, 0, 0])
    )
    
    state = process_tower_met(
        inputs=inputs,
        solar_params=default_solar_params,
        constants=default_constants,
        sat_vap_func=mock_qsat_function
    )
    
    # Check that all fields exist and have correct types
    assert hasattr(state, 'forc_t'), "Missing forc_t"
    assert hasattr(state, 'forc_q'), "Missing forc_q"
    assert hasattr(state, 'forc_pbot'), "Missing forc_pbot"
    assert hasattr(state, 'forc_u'), "Missing forc_u"
    assert hasattr(state, 'forc_v'), "Missing forc_v"
    assert hasattr(state, 'forc_lwrad'), "Missing forc_lwrad"
    assert hasattr(state, 'forc_rain'), "Missing forc_rain"
    assert hasattr(state, 'forc_snow'), "Missing forc_snow"
    assert hasattr(state, 'forc_solad'), "Missing forc_solad"
    assert hasattr(state, 'forc_solai'), "Missing forc_solai"
    assert hasattr(state, 'forc_hgt_u'), "Missing forc_hgt_u"
    assert hasattr(state, 'forc_hgt_t'), "Missing forc_hgt_t"
    assert hasattr(state, 'forc_hgt_q'), "Missing forc_hgt_q"
    assert hasattr(state, 'forc_pco2'), "Missing forc_pco2"
    assert hasattr(state, 'forc_po2'), "Missing forc_po2"


def test_process_tower_met_shapes(default_solar_params, default_constants, mock_qsat_function):
    """Test that all output arrays have correct shapes."""
    n_gridcells = 2
    n_columns = 6
    n_patches = 12
    
    raw_data = TowerMetRawData(
        zbot=50.0,
        tbot=290.15,
        rhbot=55.0,
        qbot=0.009,
        ubot=4.2,
        fsdsbot=750.0,
        fldsbot=360.0,
        pbot=100000.0,
        prect=0.0001
    )
    
    inputs = TowerMetInputs(
        raw_data=raw_data,
        tower_ht=50.0,
        tower_lat=35.0,
        tower_lon=-110.0,
        n_gridcells=n_gridcells,
        n_columns=n_columns,
        n_patches=n_patches,
        col_to_gridcell=jnp.array([0, 0, 0, 1, 1, 1]),
        col_to_tower=jnp.zeros(n_columns, dtype=int)
    )
    
    state = process_tower_met(
        inputs=inputs,
        solar_params=default_solar_params,
        constants=default_constants,
        sat_vap_func=mock_qsat_function
    )
    
    # Column-level variables
    assert state.forc_t.shape == (n_columns,), f"Wrong forc_t shape: {state.forc_t.shape}"
    assert state.forc_q.shape == (n_columns,), f"Wrong forc_q shape: {state.forc_q.shape}"
    assert state.forc_pbot.shape == (n_columns,), f"Wrong forc_pbot shape: {state.forc_pbot.shape}"
    assert state.forc_lwrad.shape == (n_columns,), f"Wrong forc_lwrad shape: {state.forc_lwrad.shape}"
    assert state.forc_rain.shape == (n_columns,), f"Wrong forc_rain shape: {state.forc_rain.shape}"
    assert state.forc_snow.shape == (n_columns,), f"Wrong forc_snow shape: {state.forc_snow.shape}"
    
    # Gridcell-level variables
    assert state.forc_u.shape == (n_gridcells,), f"Wrong forc_u shape: {state.forc_u.shape}"
    assert state.forc_v.shape == (n_gridcells,), f"Wrong forc_v shape: {state.forc_v.shape}"
    assert state.forc_solad.shape == (n_gridcells, 2), f"Wrong forc_solad shape: {state.forc_solad.shape}"
    assert state.forc_solai.shape == (n_gridcells, 2), f"Wrong forc_solai shape: {state.forc_solai.shape}"
    assert state.forc_pco2.shape == (n_gridcells,), f"Wrong forc_pco2 shape: {state.forc_pco2.shape}"
    assert state.forc_po2.shape == (n_gridcells,), f"Wrong forc_po2 shape: {state.forc_po2.shape}"
    
    # Patch-level variables
    assert state.forc_hgt_u.shape == (n_patches,), f"Wrong forc_hgt_u shape: {state.forc_hgt_u.shape}"
    assert state.forc_hgt_t.shape == (n_patches,), f"Wrong forc_hgt_t shape: {state.forc_hgt_t.shape}"
    assert state.forc_hgt_q.shape == (n_patches,), f"Wrong forc_hgt_q shape: {state.forc_hgt_q.shape}"


def test_process_tower_met_physical_realism(default_solar_params, default_constants, mock_qsat_function):
    """Test that complete processing produces physically realistic values."""
    raw_data = TowerMetRawData(
        zbot=30.0,
        tbot=288.15,
        rhbot=60.0,
        qbot=0.008,
        ubot=3.5,
        fsdsbot=600.0,
        fldsbot=350.0,
        pbot=101325.0,
        prect=0.0
    )
    
    inputs = TowerMetInputs(
        raw_data=raw_data,
        tower_ht=30.0,
        tower_lat=45.5,
        tower_lon=-93.2,
        n_gridcells=1,
        n_columns=3,
        n_patches=5,
        col_to_gridcell=jnp.array([0, 0, 0]),
        col_to_tower=jnp.array([0, 0, 0])
    )
    
    state = process_tower_met(
        inputs=inputs,
        solar_params=default_solar_params,
        constants=default_constants,
        sat_vap_func=mock_qsat_function
    )
    
    # Temperature in realistic range (200-350 K)
    assert jnp.all(state.forc_t > 200.0), f"Unrealistically low temperature: {state.forc_t}"
    assert jnp.all(state.forc_t < 350.0), f"Unrealistically high temperature: {state.forc_t}"
    
    # Specific humidity in valid range
    assert jnp.all(state.forc_q >= 0.0), f"Negative specific humidity: {state.forc_q}"
    assert jnp.all(state.forc_q <= 0.05), f"Unrealistically high specific humidity: {state.forc_q}"
    
    # Pressure in realistic range (50000-110000 Pa)
    assert jnp.all(state.forc_pbot > 50000.0), f"Unrealistically low pressure: {state.forc_pbot}"
    assert jnp.all(state.forc_pbot < 110000.0), f"Unrealistically high pressure: {state.forc_pbot}"
    
    # Wind components non-negative (assuming positive input)
    assert jnp.all(state.forc_u >= 0.0), f"Negative u-wind: {state.forc_u}"
    assert jnp.all(state.forc_v >= 0.0), f"Negative v-wind: {state.forc_v}"
    
    # Longwave radiation in realistic range (150-500 W/m²)
    assert jnp.all(state.forc_lwrad > 150.0), f"Unrealistically low longwave: {state.forc_lwrad}"
    assert jnp.all(state.forc_lwrad < 500.0), f"Unrealistically high longwave: {state.forc_lwrad}"
    
    # Solar radiation non-negative
    assert jnp.all(state.forc_solad >= 0.0), f"Negative direct beam: {state.forc_solad}"
    assert jnp.all(state.forc_solai >= 0.0), f"Negative diffuse: {state.forc_solai}"
    
    # Heights positive
    assert jnp.all(state.forc_hgt_u > 0.0), f"Non-positive height: {state.forc_hgt_u}"


def test_process_tower_met_extreme_cold(default_solar_params, default_constants, mock_qsat_function):
    """Test processing of extreme cold (Arctic winter) conditions."""
    raw_data = TowerMetRawData(
        zbot=20.0,
        tbot=253.15,  # -20°C
        rhbot=75.0,
        qbot=0.001,
        ubot=8.0,
        fsdsbot=50.0,
        fldsbot=180.0,
        pbot=101325.0,
        prect=0.0003
    )
    
    inputs = TowerMetInputs(
        raw_data=raw_data,
        tower_ht=20.0,
        tower_lat=70.0,
        tower_lon=25.0,
        n_gridcells=1,
        n_columns=2,
        n_patches=4,
        col_to_gridcell=jnp.array([0, 0]),
        col_to_tower=jnp.array([0, 0])
    )
    
    state = process_tower_met(
        inputs=inputs,
        solar_params=default_solar_params,
        constants=default_constants,
        sat_vap_func=mock_qsat_function
    )
    
    # Should handle extreme cold without errors
    assert jnp.all(jnp.isfinite(state.forc_t)), "Non-finite temperature in extreme cold"
    assert jnp.all(jnp.isfinite(state.forc_q)), "Non-finite humidity in extreme cold"
    assert jnp.all(state.forc_t > 0.0), "Non-positive temperature in extreme cold"


def test_process_tower_met_tropical(default_solar_params, default_constants, mock_qsat_function):
    """Test processing of tropical rainforest conditions."""
    raw_data = TowerMetRawData(
        zbot=40.0,
        tbot=301.15,  # 28°C
        rhbot=90.0,
        qbot=0.02,
        ubot=2.0,
        fsdsbot=500.0,
        fldsbot=420.0,
        pbot=101325.0,
        prect=0.008
    )
    
    inputs = TowerMetInputs(
        raw_data=raw_data,
        tower_ht=40.0,
        tower_lat=-3.0,
        tower_lon=-60.0,
        n_gridcells=1,
        n_columns=4,
        n_patches=8,
        col_to_gridcell=jnp.array([0, 0, 0, 0]),
        col_to_tower=jnp.array([0, 0, 0, 0])
    )
    
    state = process_tower_met(
        inputs=inputs,
        solar_params=default_solar_params,
        constants=default_constants,
        sat_vap_func=mock_qsat_function
    )
    
    # Should handle hot, humid conditions
    assert jnp.all(jnp.isfinite(state.forc_t)), "Non-finite temperature in tropical conditions"
    assert jnp.all(jnp.isfinite(state.forc_q)), "Non-finite humidity in tropical conditions"
    assert jnp.all(state.forc_q > 0.01), "Unrealistically low humidity for tropical conditions"


# ============================================================================
# Tests for get_default_solar_params and get_default_constants
# ============================================================================

def test_get_default_solar_params():
    """Test that default solar parameters are returned correctly."""
    params = get_default_solar_params()
    
    assert isinstance(params, SolarRadiationParams), \
        f"Expected SolarRadiationParams, got {type(params)}"
    
    # Check that all fields exist
    assert hasattr(params, 'a0'), "Missing a0"
    assert hasattr(params, 'a1'), "Missing a1"
    assert hasattr(params, 'a2'), "Missing a2"
    assert hasattr(params, 'a3'), "Missing a3"
    assert hasattr(params, 'b0'), "Missing b0"
    assert hasattr(params, 'b1'), "Missing b1"
    assert hasattr(params, 'b2'), "Missing b2"
    assert hasattr(params, 'b3'), "Missing b3"
    
    # Check expected values
    assert params.a0 == pytest.approx(0.17639, abs=1e-6), f"Unexpected a0: {params.a0}"
    assert params.b0 == pytest.approx(0.29548, abs=1e-6), f"Unexpected b0: {params.b0}"


def test_get_default_constants():
    """Test that default physical constants are returned correctly."""
    constants = get_default_constants()
    
    assert isinstance(constants, PhysicalConstants), \
        f"Expected PhysicalConstants, got {type(constants)}"
    
    # Check that all fields exist
    assert hasattr(constants, 'mmh2o'), "Missing mmh2o"
    assert hasattr(constants, 'mmdry'), "Missing mmdry"
    assert hasattr(constants, 'sb'), "Missing sb"
    assert hasattr(constants, 'co2_ppm'), "Missing co2_ppm"
    assert hasattr(constants, 'o2_frac'), "Missing o2_frac"
    assert hasattr(constants, 'default_pressure'), "Missing default_pressure"
    assert hasattr(constants, 'default_height'), "Missing default_height"
    assert hasattr(constants, 'missing_value'), "Missing missing_value"
    
    # Check expected values
    assert constants.mmh2o == pytest.approx(18.016, abs=1e-3), \
        f"Unexpected mmh2o: {constants.mmh2o}"
    assert constants.mmdry == pytest.approx(28.966, abs=1e-3), \
        f"Unexpected mmdry: {constants.mmdry}"
    assert constants.sb == pytest.approx(5.67e-08, abs=1e-10), \
        f"Unexpected Stefan-Boltzmann constant: {constants.sb}"
    assert constants.co2_ppm == pytest.approx(367.0, abs=1.0), \
        f"Unexpected CO2 concentration: {constants.co2_ppm}"
    assert constants.o2_frac == pytest.approx(0.209, abs=1e-3), \
        f"Unexpected O2 fraction: {constants.o2_frac}"


def test_default_params_consistency():
    """Test that default parameters are consistent between fixture and function."""
    params_func = get_default_solar_params()
    
    # Should match the values in the fixture
    assert params_func.a0 == pytest.approx(0.17639, abs=1e-6)
    assert params_func.a1 == pytest.approx(0.0038, abs=1e-6)
    assert params_func.b0 == pytest.approx(0.29548, abs=1e-6)
    assert params_func.b1 == pytest.approx(0.00504, abs=1e-6)


def test_default_constants_consistency():
    """Test that default constants are consistent between fixture and function."""
    constants_func = get_default_constants()
    
    # Should match the values in the fixture
    assert constants_func.mmh2o == pytest.approx(18.016, abs=1e-3)
    assert constants_func.mmdry == pytest.approx(28.966, abs=1e-3)
    assert constants_func.sb == pytest.approx(5.67e-08, abs=1e-10)
    assert constants_func.co2_ppm == pytest.approx(367.0, abs=1.0)
    assert constants_func.o2_frac == pytest.approx(0.209, abs=1e-3)


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_pipeline_integration(default_solar_params, default_constants, mock_qsat_function):
    """
    Integration test for complete tower meteorology processing pipeline.
    
    Tests the full workflow from raw tower data through all processing steps
    to final state variables, ensuring all components work together correctly.
    """
    # Setup realistic tower data
    raw_data = TowerMetRawData(
        zbot=30.0,
        tbot=288.15,
        rhbot=60.0,
        qbot=0.008,
        ubot=3.5,
        fsdsbot=600.0,
        fldsbot=350.0,
        pbot=101325.0,
        prect=0.0
    )
    
    inputs = TowerMetInputs(
        raw_data=raw_data,
        tower_ht=30.0,
        tower_lat=45.5,
        tower_lon=-93.2,
        n_gridcells=1,
        n_columns=3,
        n_patches=5,
        col_to_gridcell=jnp.array([0, 0, 0]),
        col_to_tower=jnp.array([0, 0, 0])
    )
    
    # Process through complete pipeline
    state = process_tower_met(
        inputs=inputs,
        solar_params=default_solar_params,
        constants=default_constants,
        sat_vap_func=mock_qsat_function
    )
    
    # Verify all outputs are finite
    assert jnp.all(jnp.isfinite(state.forc_t)), "Non-finite temperature"
    assert jnp.all(jnp.isfinite(state.forc_q)), "Non-finite specific humidity"
    assert jnp.all(jnp.isfinite(state.forc_pbot)), "Non-finite pressure"
    assert jnp.all(jnp.isfinite(state.forc_u)), "Non-finite u-wind"
    assert jnp.all(jnp.isfinite(state.forc_v)), "Non-finite v-wind"
    assert jnp.all(jnp.isfinite(state.forc_lwrad)), "Non-finite longwave"
    assert jnp.all(jnp.isfinite(state.forc_rain)), "Non-finite rain"
    assert jnp.all(jnp.isfinite(state.forc_snow)), "Non-finite snow"
    assert jnp.all(jnp.isfinite(state.forc_solad)), "Non-finite direct beam"
    assert jnp.all(jnp.isfinite(state.forc_solai)), "Non-finite diffuse"
    assert jnp.all(jnp.isfinite(state.forc_pco2)), "Non-finite CO2"
    assert jnp.all(jnp.isfinite(state.forc_po2)), "Non-finite O2"
    
    # Verify physical constraints
    assert jnp.all(state.forc_t > 0.0), "Non-positive temperature"
    assert jnp.all(state.forc_q >= 0.0) and jnp.all(state.forc_q <= 1.0), \
        "Specific humidity out of range"
    assert jnp.all(state.forc_pbot > 0.0), "Non-positive pressure"
    assert jnp.all(state.forc_lwrad >= 0.0), "Negative longwave radiation"
    assert jnp.all(state.forc_solad >= 0.0), "Negative direct beam"
    assert jnp.all(state.forc_solai >= 0.0), "Negative diffuse"
    
    # Verify energy conservation in solar partitioning
    total_solar = jnp.sum(state.forc_solad) + jnp.sum(state.forc_solai)
    expected_solar = raw_data.fsdsbot * inputs.n_gridcells
    assert jnp.allclose(total_solar, expected_solar, rtol=0.01), \
        f"Solar radiation not conserved: input={expected_solar}, output={total_solar}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])