# Overleaf doc: https://www.overleaf.com/read/ysgbfhgmndrs

# This script requires:
# - jaxlib (`pip install jaxlib`)
# - jax
# - jaxopt
# - sympy
# - scipy

# # Setup

import jax
import jax.numpy as jnp
import sympy
import scipy.constants as const
import numpy as np
# import math
import scipy.special as sp
from jaxopt import Bisection

# from jax.config import config
# We need 64 bit floats so that solving for Te (Egedal et al) remains stable for a wide range
# config.update("jax_enable_x64", True)

# # Units used
# * Temperature: keV
# * Density: m^-3
# * Field: T
# # Fundamental parameters
# ## Sympy variables
# Machine / engineering parameters
[Eb,  # keV, neutral beam energy
 beta,  # Maximum plasma beta
 B_pm,  # Tesla -- plug (and central cell) mirror coil
 B_p,  # Tesla -- plug midplane field
 r_b,  # Bore of mirror throat
 L_p,  # Length of the plug cell
 Ti,  # Placeholder (will be calculated later), keV
 Te,  # Placeholder (will be calculated later), keV
 DT_reac,  # Placeholder for DT reactivity (evaluated after Ti and Te)
 DD_reac,  # Placeholder -- should really be split into nHe3 and pT at some point
 p_aux,  # Auxiliary power used in the Egedal 2022 power balance equation
 T_frac,  # fraction of fuel that is tritium (assuming deuterium)
 Tep_fudge,  # fudge factor to reduce electron temperature when running in a tandem configuration

 # Tandem parameters
 B_cc,  # central cell field
 n_ccr,  # central cell-to-plug density ratio
 Ti_ccr,  # central cell ion temeprature to plug electron temperature ratio, assumes Maxwellian & thermal eq
 Te_ccr,  # same, but for Te
 L_cc,  # central cell length

 # Engineering parameters
 Ef_DEC,  # direct energy converter efficiency
 Ef_TE,  # thermal-to-electric efficiency
 Ef_ECH,  # ECH efficiency
 Ef_NBI,  # NBI efficiency -- shinethrough needs to be calculated separately
 Ef_RF,  # RF heating efficiency (ICRH or HHFW?)
 d_blanket,  # thickness of the blanket
 d_vv,  # thickness of the vacuum vessel
 a_wall_r,  # vessel wall radius to plasma radius ratio
 cost_HTS_kAm,  # Superconducting cable/tape cost
 coil_spacing_cc  # Spacing of central cell coils (which affects ripple --> fast particle confinement)
 ] = sympy.symbols('Eb ' 'beta ' 'B_pm ' 'B_p ' 'r_b ' 'L_p ' 'Ti ' 'Te ' 'DT_reac ' 'DD_reac '
                   'p_aux ' 'T_frac ' 'Tep_fudge ' 'B_cc ' 'n_ccr ' 'Ti_ccr ' 'Te_ccr ' 'L_cc '
                   'Ef_DEC ' 'Ef_TE ' 'Ef_ECH ' 'Ef_NBI mirror_physics_setup.py' 'Ef_RF ' 'd_blanket ' 'd_vv ' 'a_wall_r '
                   'cost_HTS_kAm ' 'coil_spacing_cc')

# Physics parameters -- thsee need to be solved iteratively but for now we leave them fixed
[mu,  # Average species mass in atomic units
 Z_eff,  # Average Z
 I_cooling  # ???
 ] = sympy.symbols('mu ' 'Z_eff ' 'I_cooling ')

f_Ti = sympy.Function('f_Ti ')
f_Te = sympy.Function('f_Te ')
f_DT_reac = sympy.Function('f_DT_reac ')
f_DD_reac = sympy.Function('f_DD_reac ')

e = np.e
ln = np.log
ln_j = jnp.log
gamma_0 = sp.exp1
gamma_0_j = jax.scipy.special.exp1

# ## Cross section / reactivity paramterization
# Accepts: ion temperature in keV. Yields: reactivity in cm^3/s
# Linearly interpolate between reactivity datapoints from the NRL formulary (in log-log space) (cm^3! need to divide by 1e6 to get m^3/s)
linear_reactivity_temps = jnp.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0])
linear_reactivity_DT = jnp.array(
    [5.5e-21, 2.6e-19, 1.3e-17, 1.1e-16, 4.2e-16, 8.7e-16, 8.5e-16, 6.3e-16, 3.7e-16, 2.7e-16]) / 1e6
linear_reactivity_DD = jnp.array(
    [1.5e-22, 5.4e-21, 1.8e-19, 1.2e-18, 5.2e-18, 2.1e-17, 4.5e-17, 8.8e-17, 1.8e-16, 2.2e-16]) / 1e6
linear_reactivity_DHe3 = jnp.array(
    [1e-26, 1.4e-23, 6.7e-21, 2.3e-19, 3.8e-18, 5.4e-17, 1.6e-16, 2.4e-16, 2.3e-16, 1.8e-16]) / 1e6


@jax.jit
def calc_DT_reactivity_linear(Ti):
    return jnp.e ** jnp.interp(jnp.log(Ti), jnp.log(linear_reactivity_temps), jnp.log(linear_reactivity_DT))


@jax.jit
def calc_DD_reactivity_linear(Ti):
    return jnp.e ** jnp.interp(jnp.log(Ti), jnp.log(linear_reactivity_temps), jnp.log(linear_reactivity_DD))


@jax.jit
def calc_DHe3_reactivity_linear(Ti):
    return jnp.e ** jnp.interp(jnp.log(Ti), jnp.log(linear_reactivity_temps), jnp.log(linear_reactivity_DHe3))


# # Plug / simple mirror calculations
# ## Solving for Ti and Te
# Implementing a root finding algorithm in JAX for speed
#    (adapted from https://stackoverflow.com/questions/13088115/finding-the-roots-of-a-large-number-of-functions-with-one-variable)
@jax.jit
def alpha_eq(a, R_in, power_aux_in):
    return ((((e ** (-a) / gamma_0_j(a) - a)) * 2 / 3 + 6 *
             (((e ** (-a) / gamma_0_j(a) - a)) * a ** 2 * ln_j(R_in) ** 2 / (22.4) ** 2 * 3 / 2) ** (1 / 3)) -
            (1 + power_aux_in))


@jax.jit
def find_alpha(R_in, power_aux_in):
    bisec = Bisection(optimality_fun=alpha_eq, lower=0.001, upper=100.0, tol=1e-4, maxiter=20,
                      check_bracket=False, unroll=True, jit=True)  # Check_bracket needs to be False to be JIT'd
    alpha = bisec.run(R_in=R_in, power_aux_in=power_aux_in).params
    return alpha


@jax.jit
def Ti_given_alpha(a):
    return ((e ** (-a) / gamma_0_j(a) - a)) * 2 / 3


@jax.jit
def Te_given_alpha(a, R):
    return (((e ** (-a) / gamma_0_j(a) - a)) * a ** 2 * ln_j(R) ** 2 / (22.4) ** 2 * 2 / 3) ** (1 / 3)


@jax.jit
def find_Ti(B_pm, B_p, p_aux, Eb):
    R_in = B_pm / B_p
    E_beam = Eb
    alpha = find_alpha(R_in, p_aux)
    Ti = Ti_given_alpha(alpha)
    return Ti * E_beam


@jax.jit
def find_Te(B_pm, B_p, p_aux, Eb):
    R_in = B_pm / B_p
    E_beam = Eb
    alpha = find_alpha(R_in, p_aux)
    Te = Te_given_alpha(alpha, R_in)
    return Te * E_beam


@jax.jit
def find_DT_reac(B_pm, B_p, p_aux, Eb):
    return calc_DT_reactivity_linear(find_Ti(B_pm, B_p, p_aux, Eb))


@jax.jit
def find_DD_reac(B_pm, B_p, p_aux, Eb):
    return calc_DD_reactivity_linear(find_Ti(B_pm, B_p, p_aux, Eb))

# @jax.jit
# def simple_Ti(Eb):
#     return 0.66 * Eb

# @jax.jit
# def simple_Te(Eb, B_p, B_pm, Tep_fudge):
#     return 0.089 * Eb * ((sympy.log(R_p / (1)) / sympy.log(10)) ** 0.4) * Tep_fudge

# Mirror ratio
R_p = B_pm / B_p

# ### values of Ti and Te
if flags['simple_temps']:
    values = {**values, **{'DT_reac': calc_DT_reactivity_linear,
                           'DD_reac': calc_DD_reactivity_linear}}
    Te = 0.089 * Eb * ((sympy.log(R_p / (1)) / sympy.log(10)) ** 0.4) * Tep_fudge
    Ti = 0.66 * Eb
else:
    values = {**values, **{'Ti': find_Ti,
                           'Te': find_Te,
                           'DT_reac': find_DT_reac,
                           'DD_reac': find_DD_reac}}
# Radius at midplane
a_plug = r_b * sympy.sqrt(R_p)
# Volume
V_p = L_p * sympy.pi * a_plug ** 2
A_p = L_p * const.pi * a_plug * 2.0
# ## n_plug
# Density at a given beta
const.mu_0 * const.elementary_charge * 2 * 1000 / 1e-20
n_plug = (B_p) ** 2 * beta / (2 * const.mu_0 * const.elementary_charge * 1000 * (Ti + Te))
# Total particle number
N_tot_plug = V_p * n_plug
# ## Confinement times
# ### tau fowler/baldwin
# **General formula:** Particle conefinement time
tau_Fowler_Baldwin = 2.4 * 10 ** 16 * \
    Eb ** (3 / 2) / n_plug * sympy.log(R_p / sympy.sqrt(1 - beta)) / sympy.log(10)
# ### Classical cross-field diffusion confinement time (plug)
# This depends on lambda_ei which needs to have it's own definition for the tandem central cell. This is all from Chen 5.8
# **General formula:** Coulomb logarithms: (T_e in eV)
lambda_ei_plug = 24 - 0.5 * sympy.log(n_plug) + sympy.log(Te * 1e3)
eta_par_plug = 5.2e-5 * Z_eff * lambda_ei_plug * sympy.sqrt(mu) / (1e3 * Te) ** (3 / 2)
D_classical_plug = 2 * eta_par_plug * n_plug * \
    ((1e3 * Te * const.elementary_charge) + (1e3 * Ti * const.elementary_charge)) / B_p ** 2
tau_classical_plug = a_plug ** 2 * n_plug / (2 * D_classical_plug * 3 * n_plug)

# ### Gyro-Bohm confinement time (plug)
# Assuming density gradient is linear from 3x n_cc to 0
D_B_plug = 1 / 16 * Te * 1e3 / B_p
rho_plug = sympy.sqrt(2 * 3.343e-27 * const.elementary_charge * (3 / 2) * Ti * 1e3) / (const.elementary_charge * B_p)
rho_star_plug = rho_plug / a_plug
tau_Bohm_plug = n_plug / (2 * D_B_plug * 3 * n_plug / a_plug)
tau_gyroBohm_plug = n_plug / (2 * D_B_plug * 3 * n_plug / a_plug) / rho_star_plug * 10.0 # arbitrary

tau_tot_plug = 1 / (1 / tau_Fowler_Baldwin + 1 / tau_gyroBohm_plug)
# ## dN/dt
# Particles lost per second
dN_dt_plug = N_tot_plug / tau_tot_plug
# **General formula:** Ion gyroradius at center of the plug
rho_i = 3.22 * 10 ** -3 * sympy.sqrt(mu * Ti) / B_p
# Number of gryoradii in the plasma radius
N_gyro = a_plug / rho_i
# NBI current (A)
I_NBI_plug = const.elementary_charge * dN_dt_plug
# **General formula:** Slowing down times
tau_i_slow_plug = 0.1 * mu * Te ** (3 / 2) / (n_plug / 10 ** 20 * Z_eff ** 2 * lambda_ei_plug)
tau_alpha_plug = 0.1 * 4 * Te ** (3 / 2) / (n_plug / 10 ** 20 * 2 ** 2 * lambda_ei_plug)
# Electron heating by fast ions (MW)
P_e_heating_fastI = 10 ** -3 * I_NBI_plug * Eb / tau_i_slow_plug
# **General formula:** Lorentz factor
gamma = sympy.sqrt(1 - Te / 511)
# ## power losses
# Synchrotron radiation power loss (MW)
P_synch_plug = 6 * 10 ** -3 * V_p * n_plug / 10 ** 20 * Te * gamma ** 2 * B_p ** 2
# Bremsstrahlung radiation power loss (MW)
P_brem_plug = 5.35 * 10 ** -3 * (n_plug / 10 ** 20) ** 2 * Z_eff * sympy.sqrt(Te) * V_p
# Power loss from escaping electrons (MW)
P_e_endloss_plug = 10 ** -3 * (I_NBI_plug + I_cooling) * 7 * Te
# Power loss from escaping fast ions (MW). Technically not I_NBI because that includes classical radial losses
P_i_endloss_plug = 10 ** -3 * (N_tot_plug / tau_Fowler_Baldwin *
                               const.elementary_charge) * (Eb - Te)
# Injected NBI Power (MW)
P_NBI_plug = 10 ** -3 * I_NBI_plug * Eb
# Injected ECH Power (MW)
# * ignore P_e_endloss and P_e_heating_fastI because that's already included in the reduced model power balance (I think); include if using simple temperature model
if flags['simple_temps']:
    P_ECH_plug = P_synch_plug / 20 + P_brem_plug + P_e_endloss_plug - P_e_heating_fastI
else:
    P_ECH_plug = P_synch_plug / 20 + P_brem_plug
P_heating_plug = P_ECH_plug + P_NBI_plug  # + P_RF

# ## alpha radii field regularization
alpha_mass = 6.644e-27

fast_alpha_gyroradius = sympy.sqrt(
    2 * alpha_mass * 3.5e6 * const.elementary_charge) / (2 * const.elementary_charge)  # units of m * T

# The 4 is rather arbitrary
a_min_vv_plug = 4 * fast_alpha_gyroradius / B_p
# A negative value indicates that the reactor vessel is sufficiently large for zero penalty
a_min_plug_diff = a_min_vv_plug - (a_plug * a_wall_r)
B_reg_plug = sympy.Piecewise((1.0, a_min_plug_diff < 0), (sympy.exp(-5.0 *
                                                                    a_min_plug_diff / (fast_alpha_gyroradius / B_p)), a_min_plug_diff >= 0))
# B_reg_plug = (1 - sympy.tanh(a_min_plug_diff / (fast_alpha_gyroradius / B_p))) / 2
if flags['field_reg_alphas']:
    charged_power_reg_coeff_plug = B_reg_plug
else:
    charged_power_reg_coeff_plug = 1.0

# ## fusion power

# Detuerium and tritium densities
n_plug_D = (1 - T_frac) * n_plug
n_plug_T = T_frac * n_plug
# DT fusion reaction rate (#/s). Here we assume a 50-50 DT fuel mixture
# Reminder: DT_reac and DD_reac are functions
# Rx_plug_DT = V_p * n_plug ** 2 / 4 * DT_reactivity  # If using Bosch 1992 parameterization
Rx_plug_DT = V_p * (n_plug_D * n_plug_T) * DT_reac * charged_power_reg_coeff_plug
# DD fusion reaction rate (#/s). Here we assume a 50-50 DT fuel mixture
# Rx_plug_DD = V_p * n_plug ** 2 * (DD_pT_reactivity + DD_nHe_reactivity) / 2  # If using Bosch 1992 parameterization
Rx_plug_DD = V_p * n_plug_D ** 2 * (DD_reac) / 2 * charged_power_reg_coeff_plug
# Fusion power (MW)
# 1e-6 converts W to MW, 1e6 converts from MeV to eV; cancels out
P_fus_plug_DT_charged = 3.5 * const.elementary_charge * Rx_plug_DT
P_fus_plug_DT_neutron = 14.1 * const.elementary_charge * Rx_plug_DT
P_fus_plug_DT = P_fus_plug_DT_charged + P_fus_plug_DT_neutron
# 1e-6 converts W to MW; the 1/2 assumes 50-50 branching ratio (decent approx)
P_fus_plug_DD_charged = (4.02 + 0.82) / 2 * const.elementary_charge * Rx_plug_DD * 1e-6
P_fus_plug_DD_neutron = (2.45) / 2 * const.elementary_charge * Rx_plug_DD
P_fus_plug_DD = P_fus_plug_DD_charged + P_fus_plug_DD_neutron
# P_fus_plug_DD_cat_charged = (3.5 + 18.3)/2 * const.elementary_charge * Rx_plug_DD  # multiplying by Rx_plug_DD because this depends on the DD reaction rate
P_fus_plug_DD_cat_charged = (3.5) / 2 * const.elementary_charge * Rx_plug_DD
P_fus_plug_DD_cat_neutron = (14.1) / 2 * const.elementary_charge * Rx_plug_DD
P_fus_plug_DD_cat = P_fus_plug_DD_cat_charged + P_fus_plug_DD_cat_neutron
# neutrons per m^2, assumes isotropy (not true for spin-polarized fuels)
neutron_flux_plug_DT = Rx_plug_DT / (A_p)
neutron_irradiance_plug_DT = P_fus_plug_DT_neutron / (A_p)  # neutron power, MW / m^2
neutron_flux_plug_DD = (Rx_plug_DD / 2) / (A_p)  # neutrons per m^2
neutron_irradiance_plug_DD = P_fus_plug_DD_neutron / (A_p)  # neutron power, MW / m^2
neutron_flux_plug_DD_cat = (Rx_plug_DD / 2) / (A_p)  # neutrons per m^2
neutron_irradiance_plug_DD_cat = P_fus_plug_DD_cat_neutron / (A_p)  # neutron power, MW / m^2
P_fus_plug_charged = (P_fus_plug_DT_charged + P_fus_plug_DD_charged)
P_fus_plug_neutron = (P_fus_plug_DT_neutron + P_fus_plug_DD_neutron)
P_fus_plug = P_fus_plug_DT + P_fus_plug_DD
neutron_flux_plug = neutron_flux_plug_DT + neutron_flux_plug_DD
neutron_irradiance_plug = neutron_irradiance_plug_DT + neutron_irradiance_plug_DD
if flags['DD_cat']:
    P_fus_plug_charged = P_fus_plug_charged + P_fus_plug_DD_cat_charged
    P_fus_plug_neutron = P_fus_plug_neutron + P_fus_plug_DD_cat_neutron
    P_fus_plug = P_fus_plug + P_fus_plug_DD_cat

    neutron_flux_plug = neutron_flux_plug + neutron_flux_plug_DD_cat
    neutron_irradiance_plug = neutron_irradiance_plug + neutron_irradiance_plug_DD_cat

# Lawson Triple Product
triple_product_plug = tau_Fowler_Baldwin * n_plug * Ti
# Burnup fraction, DT
frac_burnup_plug = (Rx_plug_DT + Rx_plug_DD) / dN_dt_plug

# ## Q_plug (power in MW)
Q_plug = P_fus_plug / (P_NBI_plug + P_ECH_plug)

# ## Triple product, energy confinement time, etc...
# Energy confinement time (power balance) (volume is overestimated — a linear falloff would cut total particle count by a third):
# plug_in_values((V_p * n_plug * 1000 * const.elementary_charge *
#                 (Ti + Te) * 3 / 2) / (1e6 * P_NBI_plug), values)

# ### alpha particles
n_alpha_plug = tau_alpha_plug * (Rx_plug_DT + 0.5 * Rx_plug_DD) / V_p
Z_eff_plug = (n_plug + 4 * n_alpha_plug) / (n_plug + 2 * n_alpha_plug)

# # Tandem mirror calculations
a_cc = r_b * sympy.sqrt(B_pm / B_cc)
R_cc = B_pm / B_cc
V_cc = const.pi * a_cc ** 2 * L_cc
A_cc = const.pi * a_cc * 2.0 * L_cc
n_cc = n_ccr * n_plug
T_ic = Ti_ccr * Te
T_ec = Te_ccr * Te
N_tot_cc = V_cc * n_cc
beta_cc = 4.0267e-25 * n_cc * (T_ic + T_ec) * 1000 / B_cc ** 2
beta_cc_limited = sympy.Piecewise((0.9, beta_cc > 0.9), (beta_cc, beta_cc <= 0.9))
Pastukhov = sympy.log(2 * B_pm / B_cc * 1 / sympy.sqrt(1 - beta_cc_limited) + 1) * sympy.log(n_plug / n_cc) * (n_plug / n_cc) ** (Te / T_ic)
tau_ii_cc = 1.25e16 * T_ic ** (3 / 2) * sympy.sqrt(mu) / n_cc / (Z_eff ** 4)
chi_ETG = 0.1 * T_ec ** (3 / 2) / B_cc
tau_ETG = a_cc ** 2 / chi_ETG
lambda_ei_cc = 24 - 0.5 * sympy.log(n_cc) + sympy.log(T_ec * 1e3)

# ## Confinement times (tandem)
# ### Classical cross-field diffusion confinement time (tandem)
# This depends on lambda_ei which needs to have it's own definition for the tandem central cell. This is all from Chen 5.8
eta_par_cc = 5.2e-5 * Z_eff * lambda_ei_cc * sympy.sqrt(mu) / (1e3 * T_ec) ** (3 / 2)
D_classical_cc = 2 * eta_par_cc * n_cc * \
    ((1e3 * T_ec * const.elementary_charge) + (1e3 * T_ic * const.elementary_charge)) / B_cc ** 2
tau_classical_cc = a_cc ** 2 * n_cc / (2 * D_classical_cc * 3 * n_cc)

# ### Gyro-Bohm confinement time (tandem)
# Assuming density gradient is linear from 3x n_cc to 0
D_B_cc = 1 / 16 * T_ec * 1e3 / B_cc
rho_cc = sympy.sqrt(2 * 3.343e-27 * const.elementary_charge * (3 / 2) * T_ic * 1e3) / (const.elementary_charge * B_cc)
rho_star_cc = rho_cc / a_cc
tau_Bohm_cc = n_cc / (2 * D_B_cc * 3 * n_cc / a_cc)
tau_gyroBohm_cc = n_cc / (2 * D_B_cc * 3 * n_cc / a_cc) / rho_star_cc * 10.0  # arbitrary
# ## Scaling-law cross-field (tandem)
# tau_L97_cc = 0.01 * B_cc ** 0.99 * L_cc ** 0.93 * a_cc ** 1.86 * (n_cc/1e20) ** 0.4 * (P_ECH_cc) ** -0.73; plug_in_values(tau_L97_cc, {**values, 'B_cc': 5})
# tau_ETG_cc = 0.025 * L_cc ** 0.33 * a_cc ** 2.66 * (n_cc/1e20) ** 1 * (P_ECH_cc) ** -0.33; plug_in_values(tau_ETG_cc, values)
# ### Pastukhov tandem confinement times
tau_E_cc = Pastukhov * tau_ii_cc
tau_tot_cc = 1 / (1 / tau_gyroBohm_cc + 1 / tau_E_cc)
dN_dt_cc = N_tot_cc / tau_tot_cc
I_fuel_cc = dN_dt_cc * const.elementary_charge
# Use tau_E_cc because we're concerend with only losses out the end
P_i_endloss_cc = 1e-3 * 3 / 2 * const.elementary_charge * const.pi * a_cc ** 2 * L_cc * n_cc * (T_ic + T_ec) / tau_E_cc
# Includes ion, electron, and classical radial losses
P_loss_cc = 1e-3 * 3 / 2 * const.elementary_charge * const.pi * a_cc ** 2 * L_cc * n_cc * (T_ic + T_ec) / tau_tot_cc
# P_NBI_cc = I_NBI_cc * T_ic * 1e-3 + P_loss_cc
# P_NBI_cc = 3/2 * T_ic * I_NBI_cc * 1e-3 - P_i_endloss - P_e_endloss + P_loss_cc;  # 1e-3 to convert to MW

# ## alpha radii field regularization (tandem)
a_min_vv_cc = 4 * fast_alpha_gyroradius / B_cc
a_min_cc_diff = a_min_vv_cc - (a_cc * a_wall_r)
B_reg_cc = sympy.Piecewise((1.0, a_min_cc_diff < 0), (sympy.exp(-5.0 * a_min_cc_diff / (fast_alpha_gyroradius / B_cc)), a_min_cc_diff >= 0))
# B_reg_cc = (1 - sympy.tanh(a_min_cc_diff / (fast_alpha_gyroradius / B_cc))) / 2
if flags['field_reg_alphas']:
    charged_power_reg_coeff_cc = B_reg_cc
else:
    charged_power_reg_coeff_cc = 1.0

# ## fusion power (tandem)
# Detuerium and tritium densities
n_cc_D = (1 - T_frac) * n_cc
n_cc_T = T_frac * n_cc
# DT fusion reaction rate (#/s)
# Rx_plug_DT = V_p * n_plug ** 2 / 4 * DT_reactivity  # If using Bosch 1992 parameterization
Rx_cc_DT = V_cc * (n_cc_D * n_cc_T) * DT_reac * charged_power_reg_coeff_cc
# DD fusion reaction rate (#/s)
Rx_cc_DD = V_cc * n_cc_D ** 2 * (DD_reac) / 2 * charged_power_reg_coeff_cc
# Fusion power (MW):
# 1e-6 converts W to MW, 1e6 converts from MeV to eV; cancels out
P_fus_cc_DT_charged = 3.5 * const.elementary_charge * Rx_cc_DT
P_fus_cc_DT_neutron = 14.1 * const.elementary_charge * Rx_cc_DT
P_fus_cc_DT = P_fus_cc_DT_charged + P_fus_cc_DT_neutron
# 1e-6 converts W to MW; the 1/2 assumes 50-50 branching ratio (decent approx)
P_fus_cc_DD_charged = (4.02 + 0.82) / 2 * const.elementary_charge * Rx_cc_DD * 1e-6
P_fus_cc_DD_neutron = (2.45) / 2 * const.elementary_charge * Rx_cc_DD
P_fus_cc_DD = P_fus_cc_DD_charged + P_fus_cc_DD_neutron
# P_fus_cc_DD_cat_charged = (3.5 + 18.3)/2 * const.elementary_charge * Rx_cc_DD  # multiplying by Rx_cc_DD because this depends on the DD reaction rate
P_fus_cc_DD_cat_charged = (3.5) / 2 * const.elementary_charge * Rx_cc_DD
P_fus_cc_DD_cat_neutron = (14.1) / 2 * const.elementary_charge * Rx_cc_DD
P_fus_cc_DD_cat = P_fus_cc_DD_cat_charged + P_fus_cc_DD_cat_neutron
# neutrons per m^2, assumes isotropy (not true for spin-polarized fuels)
neutron_flux_cc_DT = Rx_cc_DT / (A_cc)
neutron_irradiance_cc_DT = P_fus_cc_DT_neutron / (A_cc)  # neutron power, MW / m^2
neutron_flux_cc_DD = (Rx_cc_DD / 2) / (A_cc)  # neutrons per m^2
neutron_irradiance_cc_DD = P_fus_cc_DD_neutron / (A_cc)  # neutron power, MW / m^2
neutron_flux_cc_DD_cat = (Rx_cc_DD / 2) / (A_cc)  # neutrons per m^2
neutron_irradiance_cc_DD_cat = P_fus_cc_DD_cat_neutron / (A_cc)  # neutron power, MW / m^2
P_fus_cc_charged = (P_fus_cc_DT_charged + P_fus_cc_DD_charged)
P_fus_cc_neutron = (P_fus_cc_DT_neutron + P_fus_cc_DD_neutron)
P_fus_cc = P_fus_cc_DT + P_fus_cc_DD
neutron_flux_cc = neutron_flux_cc_DT + neutron_flux_cc_DD
neutron_irradiance_cc = neutron_irradiance_cc_DT + neutron_irradiance_cc_DD
if flags['DD_cat']:
    P_fus_cc_charged = P_fus_cc_charged + P_fus_cc_DD_cat_charged
    P_fus_cc_neutron = P_fus_cc_neutron + P_fus_cc_DD_cat_neutron
    P_fus_cc = P_fus_cc + P_fus_cc_DD_cat
    neutron_flux_cc = neutron_flux_cc + neutron_flux_cc_DD_cat
    neutron_irradiance_cc = neutron_irradiance_cc + neutron_irradiance_cc_DD_cat

# Keep in mind this is total fusion power, not fusion power per meter as defined in Cary's spreadsheet
P_fus_total = P_fus_cc + 2 * P_fus_plug
# P_HHFW aka P_RF
# L_breakeven: lenght of the central cell for breakeven (including fusion power from the endplugs). Perhaps a useful tandem mirror metric:
L_breakeven_cc = 2 * (P_heating_plug - P_fus_plug) * L_cc / P_fus_cc
L_breakeven_cc_only = 2 * (P_heating_plug) * L_cc / P_fus_cc

tau_alpha_cc = 0.1 * 4 * Te ** (3 / 2) / (n_cc / 10 ** 20 * 2 ** 2 * lambda_ei_cc)
n_alpha_cc = tau_alpha_cc * (Rx_cc_DT + 0.5 * Rx_cc_DD) / V_cc
Z_eff_cc = (n_cc + 4 * n_alpha_cc) / (n_cc + 2 * n_alpha_cc)

# ## power balance and Q (tandem)
P_synch_cc = 6 * 10 ** -3 * V_cc * n_cc / 10 ** 20 * Te * gamma ** 2 * B_cc ** 2
P_brem_cc = 5.35 * 10 ** -3 * (n_cc / 10 ** 20) ** 2 * Z_eff * sympy.sqrt(Te) * V_cc
if flags['cc_aux_ECH']:
    P_ECH_cc = P_synch_cc / 20 + P_brem_cc + P_loss_cc
else:
    P_ECH_cc = P_synch_cc / 20 + P_brem_cc
Q_tandem = P_fus_total / (2 * P_heating_plug + P_ECH_cc)
P_electric_in = Ef_ECH * (2 * P_ECH_plug + P_ECH_cc) + Ef_NBI * (2 * P_NBI_plug)
P_recirculating = Ef_DEC * (P_fus_cc_charged + 2 * P_fus_plug_charged +
                            P_i_endloss_cc + 2 * P_i_endloss_plug)
P_thermal = P_fus_cc_neutron + 2 * P_fus_plug_neutron + (1 - Ef_DEC) * P_recirculating / Ef_DEC
P_thermal_electric = Ef_TE * P_thermal
P_net_electric = P_thermal_electric + P_recirculating - P_electric_in

# # Magnet costs

# Mirror coils (plug, mirror). Will need four of them (one pair for each plug)
a_B_pm = r_b * a_wall_r + d_vv + d_blanket
I_B_pm = 2 / const.mu_0 * B_pm * a_B_pm / 1e3
S_B_pm = 2 * const.pi * a_B_pm * I_B_pm
cost_B_pm = S_B_pm * cost_HTS_kAm

# Plug field coils
a_B_p = a_plug * a_wall_r + d_vv + d_blanket
I_B_p = 2 / const.mu_0 * B_pm * a_B_p / 1e3
S_B_p = 2 * const.pi * a_B_p * I_B_p
cost_B_p = S_B_p * cost_HTS_kAm

# Central cell
a_B_cc = a_cc * a_wall_r + d_vv + d_blanket
S_B_cc = 2 * const.pi * a_B_cc * B_cc / const.mu_0 / 1e3
cost_B_cc = S_B_cc * L_cc * cost_HTS_kAm * 1 / coil_spacing_cc

# Power costs
P_ECH_total = P_ECH_cc + P_ECH_plug
cost_ECH_perMW = 10.
cost_ECH_total = P_ECH_total * cost_ECH_perMW

P_NBI_total = P_NBI_plug
cost_NBI_perMW = 5.
cost_NBI_total = P_NBI_total * cost_NBI_perMW

cost_heat_total = cost_ECH_total + cost_NBI_total
