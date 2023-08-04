import matplotlib.pyplot as plt
from tqdm import tqdm
# ## Values / JAX variables to optimize for
values = {
    'Eb': 1000.,
    'beta': 0.8,
    'B_pm': 30.,
    'B_p': 6.,
    'r_b': 0.25,
    'L_p': 4.,
    'p_aux': 0.0,  # this is in addition to synchotron losses (which are calculated separately)
    'T_frac': 0.5,
    'Tep_fudge': 0.5,  # this should remain constant

    'B_cc': 1.5,
    'n_ccr': 0.25,
    'Ti_ccr': 1.0,
    'Te_ccr': 1.0,
    'L_cc': 20.0,

    # efficiencies should remain constant
    'Ef_DEC': 0.7,
    'Ef_TE': 0.5,
    'Ef_ECH': 0.6,
    'Ef_NBI': 0.6,
    'Ef_RF': 0.9,

    # engineering parameters remaining constant (for now)
    'd_blanket': 0.6,
    'd_vv': 0.2,
    'a_wall_r': 1.1,
    'cost_HTS_kAm': 1e-4,  # in megadollars (M$)
    'coil_spacing_cc': 1.0,  # in meters

    # physical quantities that are remaining constant for simplicity
    'mu': 2.5,
    'Z_eff': 1.13,
    'I_cooling': 0
}


# ## Flags
flags = {
    # assumes Ti_plug = 2/3 E_inj and Te_plug = 0.09 log(Rm/(sqrt(1-beta))^0.4)
    'simple_temps': False,
    'DD_cat': True,  # assumes DD fusion products are burned instantly. Is there any reason to turn this off?
    'field_reg_alphas': False,  # Regularize midplane field strengths via an alpha particle gyroradii heuristic
    'field_reg_Bohm': False,  # Regularize midplane field via Bohm diffusion heuristic
    'cc_aux_ECH': True,  # Use ECH to replace tandem mirror axial losses
}

exec(open('mirror_physics_setup.py').read())

# These constants are valid for 0.2-100 keV (Maxwellian) ion temperatures. Verified correct to 4 decimal places for 0.2, 1, and 10 keV.
DT_BG = 34.3827  # keV^0.5
DT_mrc2 = 1124656  # keV
DT_C1 = 1.17302e-9
DT_C2 = 1.51361e-2
DT_C3 = 7.51886e-2
DT_C4 = 4.60643e-3
DT_C5 = 1.35e-2
DT_C6 = -1.06750e-4
DT_C7 = 1.366e-5

DD_pT_BG = 31.3970  # keV^0.5
DD_pT_mrc2 = 937814  # keV
DD_pT_C1 = 5.65718e-12
DD_pT_C2 = 3.41267e-3
DD_pT_C3 = 1.99167e-3
DD_pT_C4 = 0
DD_pT_C5 = 1.05060e-5
DD_pT_C6 = 0
DD_pT_C7 = 0

DD_nHe_BG = 31.3970  # keV^0.5
DD_nHe_mrc2 = 937814  # keV
DD_nHe_C1 = 5.43360e-12
DD_nHe_C2 = 5.85778e-3
DD_nHe_C3 = 7.68222e-3
DD_nHe_C4 = 0
DD_nHe_C5 = -2.96400e-6
DD_nHe_C6 = 0
DD_nHe_C7 = 0

DT_theta = Ti / (1 -  Ti * (DT_C2 + Ti * (DT_C4 + Ti * DT_C6)) / (1 + Ti * (DT_C3 + Ti * (DT_C5 + Ti * DT_C7))))
DT_xi = (DT_BG ** 2 / (4 * DT_theta)) ** (1/3)
DT_reactivity = DT_C1 * DT_theta * (DT_xi / (DT_mrc2 * Ti ** 3)) ** 0.5 * sympy.E ** (-3 * DT_xi) / 1e6

DD_pT_theta = Ti / (1 -  Ti * (DD_pT_C2 + Ti * (DD_pT_C4 + Ti * DD_pT_C6)) / (1 + Ti * (DD_pT_C3 + Ti * (DD_pT_C5 + Ti * DD_pT_C7))))
DD_pT_xi = (DD_pT_BG ** 2 / (4 * DD_pT_theta)) ** (1/3)
DD_pT_reactivity = DD_pT_C1 * DD_pT_theta * (DD_pT_xi / (DD_pT_mrc2 * Ti ** 3)) ** 0.5 * sympy.E ** (-3 * DD_pT_xi) / 1e6

DD_nHe_theta = Ti / (1 -  Ti * (DD_nHe_C2 + Ti * (DD_nHe_C4 + Ti * DD_nHe_C6)) / (1 + Ti * (DD_nHe_C3 + Ti * (DD_nHe_C5 + Ti * DD_nHe_C7))))
DD_nHe_xi = (DD_nHe_BG ** 2 / (4 * DD_nHe_theta)) ** (1/3)
DD_nHe_reactivity = DD_nHe_C1 * DD_nHe_theta * (DD_nHe_xi / (DD_nHe_mrc2 * Ti ** 3)) ** 0.5 * sympy.E ** (-3 * DD_nHe_xi) / 1e6

test_DD_nHe_theta = Ti / (1 -  Ti * (DD_nHe_C2 + Ti * (DD_nHe_C4 + Ti * DD_nHe_C6)) / (1 + Ti * (DD_nHe_C3 + Ti * (0.0 + Ti * DD_nHe_C7))))
test_DD_nHe_xi = (DD_nHe_BG ** 2 / (4 * test_DD_nHe_theta)) ** (1/3)
test_DD_nHe_reactivity = DD_nHe_C1 * test_DD_nHe_theta * (test_DD_nHe_xi / (DD_nHe_mrc2 * Ti ** 3)) ** 0.5 * sympy.E ** (-3 * test_DD_nHe_xi) / 1e6

jax_DT_reactivity = sympy.lambdify(list(DT_reactivity.free_symbols), DT_reactivity, jax.numpy)
jax_DD_pT_reactivity = sympy.lambdify(list(DD_pT_reactivity.free_symbols), DD_pT_reactivity, jax.numpy)
jax_DD_nHe_reactivity = sympy.lambdify(list(DD_nHe_reactivity.free_symbols), DD_nHe_reactivity, jax.numpy)

# ## Cross section via interpolation from NRL


# polyfit (isn't good) -- not used
def calc_lin_reactivity(coefficients, x):
    num_c = coefficients.shape[-1]
    num_x = x.shape[0]
    x_mat = 1 - jnp.triu(jnp.ones((num_c, num_c)))[jnp.newaxis, :, :].repeat(num_x, axis=0)
    x_mat = x_mat * x.reshape(num_x, 1, 1)
    x_mat = jnp.tri(num_c).transpose() + x_mat
    x = jnp.product(x_mat, axis=-2)
    return jnp.sum(x * coefficients, axis=1)


# ## Cross section comparison
temp_plot = 10 ** jnp.linspace(0, 3, 100)
plt.figure()
plt.plot(temp_plot, jax_DT_reactivity(temp_plot), label='DT (Bosch 1992)')
# plt.plot(temp_plot, jax_DD_nHe_reactivity(temp_plot) + jax_DD_pT_reactivity(temp_plot), label='pT+nHe3 (Bosch 1992)')
# plt.plot(temp_plot, jax_DD_nHe_reactivity(temp_plot), label='nHe3 (Bosch 1992)')
# plt.plot(temp_plot, jax_DD_pT_reactivity(temp_plot), label='pT (Bosch 1992)')
plt.plot(temp_plot, calc_DT_reactivity_linear(temp_plot), label='DT, linear interpolation')
# plt.plot(temp_plot, calc_DD_reactivity_linear(temp_plot), label='log-space linear interpolation')
# plt.plot(temp_plot, calc_lin_reactivity(lin_c, temp_plot), label='DD, poly')
plt.scatter(linear_reactivity_temps, linear_reactivity_DT)
# plt.scatter(linear_reactivity_temps, linear_reactivity_DD, label="NRL datapoints")
# plt.plot(temp_plot, calc_DHe3_reactivity_linear(temp_plot), label='DHe3, linear')
plt.yscale('log')
plt.xscale('log')
plt.title('DT fusion reactivity')
plt.ylabel('Reactivity ($m^3/s$)')
plt.xlabel("Temperature (keV)")
plt.legend()
# plt.savefig('DT_reactivity.pdf')

temp_plot = 10 ** jnp.linspace(0, 3, 100)
plt.figure()
# plt.plot(temp_plot, jax_DT_reactivity(temp_plot), label='DT')
plt.plot(temp_plot, jax_DD_nHe_reactivity(temp_plot) +
         jax_DD_pT_reactivity(temp_plot), label='pT+nHe3 (Bosch 1992)')
plt.plot(temp_plot, jax_DD_nHe_reactivity(temp_plot), label='nHe3 (Bosch 1992)')
plt.plot(temp_plot, jax_DD_pT_reactivity(temp_plot), label='pT (Bosch 1992)')
# plt.plot(temp_plot, test_jax_DD_nHe_reactivity(temp_plot) + jax_DD_pT_reactivity(temp_plot), label='Test _ DD (pT+nHe3)')
# plt.plot(temp_plot, calc_DT_reactivity_linear(temp_plot), label='DT, linear')
plt.plot(temp_plot, calc_DD_reactivity_linear(temp_plot), label='log-space linear interpolation')
# plt.plot(temp_plot, calc_lin_reactivity(lin_c, temp_plot), label='DD, poly')
# plt.scatter(linear_reactivity_temps, linear_reactivity_DT)
plt.scatter(linear_reactivity_temps, linear_reactivity_DD, label="NRL datapoints")
# plt.plot(temp_plot, calc_DHe3_reactivity_linear(temp_plot), label='DHe3, linear')
plt.yscale('log')
plt.xscale('log')
plt.title('DD fusion reactivity')
plt.ylabel('Reactivity ($m^3/s$)')
plt.xlabel("Temperature (keV)")
plt.legend()
# plt.savefig('DD_reactivity.pdf')

temp_plot = 10 ** jnp.linspace(0, 3, 100)
plt.figure()
plt.plot(temp_plot, calc_DT_reactivity_linear(temp_plot), label='DT')
plt.plot(temp_plot, calc_DD_reactivity_linear(temp_plot), label='DD')
plt.plot(temp_plot, calc_DHe3_reactivity_linear(temp_plot), label='D-He3')
# plt.scatter(linear_reactivity_temps, linear_reactivity_DD, label="NRL datapoints")
plt.yscale('log')
plt.xscale('log')
plt.title('DT, DD, and D-He3 fusion reactivity (log-space interpolation of NRL data)')
plt.ylabel('Reactivity ($m^3/s$)')
plt.xlabel("Temperature (keV)")
plt.legend()
plt.show()
# plt.savefig('compare_reactivity.pdf')
