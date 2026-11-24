"""Physical constants and fixed model coefficients of the processes in ``clm_ml.processes``
(Python floats, so importing them creates no arrays and needs no JAX setting).

Values from CLM-ml v2.

Fortran reference: CLM-ml v2 ``clm_varcon.F90`` (CLM constants) and ``MLclm_varcon.F90``
(multilayer canopy constants).
"""

# CLM constants (clm_varcon.F90)
cpice = 2117.27  # Specific heat of ice (J/kg/K)
cpliq = 4188.0  # Specific heat of water (J/kg/K)
csol_bedrock = 2000000.0  # Vol. heat capacity of granite/sandstone (J/m3/K)
denh2o = 1000.0  # Density of liquid water (kg/m3)
denice = 917.0  # Density of ice (kg/m3)
grav = 9.80665  # Gravitational acceleration (m/s2)
hsub = 2834700.0  # Latent heat of sublimation (J/kg)
hvap = 2501000.0  # Latent heat of evaporation (J/kg)
rpi = 3.141592654  # pi
sb = 5.67e-08  # Stefan-Boltzmann constant (W/m2/K4)
spval = 1e36  # Special value for real data
tfrz = 273.15  # Freezing point of water (K)
thk_bedrock = 3.0  # Thermal conductivity of saturated granitic rock (W/m/K)
tkice = 2.29  # Thermal conductivity of ice (W/m/K)
tkwat = 0.57  # Thermal conductivity of water (W/m/K)
vkc = 0.4  # von Karman constant

# Multilayer canopy constants (MLclm_varcon.F90)
J_to_umol = 4.6  # PAR conversion: W/m2 → umol/m2/s (umol/J)
LcL_max = 1.0  # Maximum value for Lc/obu
LcL_min = -2.0  # Minimum value for Lc/obu
Pr0 = 0.5  # Neutral value for Pr (Sc)
Pr1 = 0.3  # Magnitude of variation of Pr (Sc) with stability
Pr2 = 2.0  # Scale of variation of Pr (Sc) with stability
beta_neutral_max = 0.35  # Maximum beta in neutral conditions
c2 = 0.5  # Depth scale multiplier
cd = 0.25  # Drag coefficient for canopy elements (dimensionless)
chil_max = 0.6  # Maximum value for xl leaf angle orientation parameter
chil_min = -0.4  # Minimum value for xl leaf angle orientation parameter
colim_c3a = 0.98  # C3 co-limitation (Ac, Aj)
colim_c4a = 0.8  # C4 co-limitation (Ac, Aj)
colim_c4b = 0.95  # C4 co-limitation (Ap)
cp25 = 42.75  # CO2 compensation point at 25 °C (umol/mol)
cpbio = 1396.0  # Specific heat of dry biomass (J/kg/K)
cpd = 1005.0  # Specific heat of dry air at constant pressure (J/kg/K)
cpha = 37830.0  # Activation energy for cp (J/mol)
cpw = 1846.0  # Specific heat of water vapour at constant pressure (J/kg/K)
cr = 0.3  # Parameter to calculate beta_neutral
dc0 = 1.38e-05  # Molecular diffusivity, CO2,  at 0 °C and 1013.25 hPa (m2/s)
dewmx = 0.1  # Maximum allowed interception (kg H2O / m2 leaf)
dh0 = 1.89e-05  # Molecular diffusivity, heat, at 0 °C and 1013.25 hPa (m2/s)
dh2o_to_dco2 = 1.6  # Diffusivity H2O / Diffusivity CO2
dv0 = 2.18e-05  # Molecular diffusivity, H2O,  at 0 °C and 1013.25 hPa (m2/s)
emg = 0.96  # Ground (soil) emissivity
eta_max = 20.0  # Maximum value for "eta" parameter
fcarbon = 0.5  # Fraction of dry biomass that is carbon (kg C / kg DM)
fwater = 0.7  # Fraction of fresh biomass that is water (kg H2O / kg FM)
fwet_exponent = 0.67  # Exponent for wetted canopy fraction
gb_factor = 1.5  # Empirical correction factor for Nusselt number
gbh_min = 0.2  # Minimum leaf boundary layer conductance for heat (mol/m2/s)
interception_fraction = 1.0  # Fraction of intercepted precipitation
jmax25_to_vcmax25_noacclim = 1.67  # Jmax/Vcmax at 25 °C, no acclimation (umol/umol)
jmaxha_acclim = 50000.0  # Activation energy for jmax, with acclimation (J/mol)
jmaxha_noacclim = 43540.0  # Activation energy for jmax, no acclimation (J/mol)
jmaxhd_acclim = 200000.0  # Deactivation energy for jmax, with acclimation (J/mol)
jmaxhd_noacclim = 150000.0  # Deactivation energy for jmax, no acclimation (J/mol)
jmaxse_noacclim = 490.0  # Entropy term for jmax, no acclimation (J/mol/K)
kb_max = 40.0  # Maximum direct beam extinction coefficient
kc25 = 404.9  # Michaelis-Menten constant for CO2 at 25 °C (umol/mol)
kcha = 79430.0  # Activation energy for kc (J/mol)
ko25 = 278.4  # Michaelis-Menten constant for O2 at 25 °C (mmol/mol)
koha = 36380.0  # Activation energy for ko (J/mol)
kp25_to_vcmax25_c4 = 0.02  # Kp/Vcmax at 25 °C, C4 (mol/umol)
lapse_rate = 0.0098  # Temperature lapse rate (K/m)
maximum_leaf_wetted_fraction = 0.05  # Maximum fraction of leaf that can be wet
mmdry = 0.02897  # Molecular mass of dry air (kg/mol)
mmh2o = 0.01802  # Molecular mass of water vapour (kg/mol)
phi_psII = 0.7  # C3: quantum yield of PS II
qe_c4 = 0.05  # C4: quantum yield (mol CO2 / mol photons)
ra_max = 500.0  # Maximum aerodynamic resistance for a canopy layer (s/m)
rd25_to_vcmax25_c3 = 0.015  # Rd/Vcmax at 25 °C, C3 (umol/umol)
rd25_to_vcmax25_c4 = 0.025  # Rd/Vcmax at 25 °C, C4 (umol/umol)
rdha = 46390.0  # Activation energy for Rd (J/mol)
rdhd = 150000.0  # Deactivation energy for Rd (J/mol)
rdse = 490.0  # Entropy term for Rd (J/mol/K)
rgas = 8.31446  # Universal gas constant (J/K/mol)
rh_min_BB = 0.2  # Minimum RH for Ball-Berry stomatal conductance (fraction)
theta_j = 0.9  # C3: curvature parameter for electron transport rate
vcmaxha_acclim = 72000.0  # Activation energy for vcmax, with acclimation (J/mol)
vcmaxha_noacclim = 65330.0  # Activation energy for vcmax, no acclimation (J/mol)
vcmaxhd_acclim = 200000.0  # Deactivation energy for vcmax, with acclimation (J/mol)
vcmaxhd_noacclim = 150000.0  # Deactivation energy for vcmax, no acclimation (J/mol)
vcmaxse_noacclim = 490.0  # Entropy term for vcmax, no acclimation (J/mol/K)
visc0 = 1.33e-05  # Kinematic viscosity at 0 °C and 1013.25 hPa (m2/s)
vpd_min_MED = 100.0  # Minimum VPD for Medlyn stomatal conductance (Pa)
wind_forc_min = 0.5  # Minimum wind speed at forcing height (m/s)
z0mg = 0.01  # Roughness length of ground (m)
