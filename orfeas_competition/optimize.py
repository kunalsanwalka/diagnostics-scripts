import mirror_physics

# # Optimization constraints
# ## see: field regularization via alphas
# # Optimizing Q_plug
values_to_opt = values.copy()

# for key in values_to_opt:
#     values_to_opt[key] = jnp.ones(10) * values_to_opt[key]
values_to_opt['L_p'] = jnp.ones(10) * values_to_opt['L_p']
values_to_opt['T_frac'] = jnp.ones(10) * values_to_opt['T_frac']
values_to_opt['Tep_fudge'] = jnp.ones(10) * values_to_opt['Tep_fudge']
values_to_opt['Z_eff'] = jnp.ones(10) * values_to_opt['Z_eff']
values_to_opt['mu'] = jnp.ones(10) * values_to_opt['mu']
values_to_opt['beta'] = jnp.ones(10) * values_to_opt['beta']
values_to_opt['r_b'] = jnp.ones(10) * values_to_opt['r_b']
values_to_opt['Eb'] = jnp.ones(10) * values_to_opt['Eb']

values_to_opt['B_p'] = jnp.ones(10) * 6.0
values_to_opt['B_pm'] = jnp.logspace(0.1, 2, 10)
values_to_opt['p_aux'] = jnp.zeros(10)


# Substitute in temperature and reactivities functions into the sympy equation and then lambdify
Q_plug_func_sympy, Q_plug_function, Q_arg_tuple, Q_arg_idx = sub_and_lambdify(
    Q_plug - 1 / B_p * 1, values_to_opt)


# Build gradient function (only need to do this once!)
symbol_list = list(Q_plug_func_sympy.free_symbols)
grad_Q_func = jax.jit(jax.vmap(jax.grad(Q_plug_function, jnp.arange(len(symbol_list)))))


# Optimize for:
# * B_pm
# * B_p
# * r_b
# * L_p
# * T_frac
# * Eb
opt_symbol_list = [B_pm, B_p, r_b, L_p, T_frac, Eb]
# opt_symbol_list = [T_frac]


# Create mask so that parameters we don't want to optimize remain constant
symbol_list_names = [sym.name for sym in symbol_list]
# print(symbol_list)
opt_idx = [symbol_list.index(sym) for sym in opt_symbol_list]
opt_idx_mask = np.zeros(len(symbol_list))
opt_idx_mask[opt_idx] = 1
for _ in zip(symbol_list, opt_idx_mask):
    print(_[0].name + ": {}".format(int(_[1])))


# ## Optimization loop
num_samples = 8  # len(Q_arg_tuple[0])
loop_values = []
loop_Q_grad = []
# print(Q_arg_tuple)
loop_values.append(jnp.array(Q_arg_tuple)[:, 0:num_samples])
loop_Q_grad.append(jnp.zeros((len(symbol_list), num_samples)))

for i in tqdm(range(1000)):
    grad_Q = -jnp.array(grad_Q_func(*tuple(loop_values[i]))) * \
        jnp.tile(opt_idx_mask[:, jnp.newaxis], [1, num_samples]) * 1e-3
#     if i > 1000:
#     new_value = loop_values[-1] - (grad_Q + 0.99 * loop_Q_grad[-1])
#     else:
    new_value = loop_values[-1] - grad_Q
    loop_values.append(new_value)
    loop_Q_grad.append(grad_Q)


# ## Plot Q and gradient norm
fig, axQ = plt.subplots()
axQ.plot(np.arange(0, 10001, 50), [Q_plug_function(*tuple(_))
                                   for _ in tqdm(loop_values[::50])], label='Q', color="#1f77b4")
axQ.tick_params(colors="#1f77b4", axis='y')
# axQ.plot(Q_plug_function(*tuple([np.array(loop_values).transpose()[_, :] for _ in tqdm(range(8))])), label='Q')
axGrad = axQ.twinx()
axGrad.plot(np.sqrt(np.sum(np.array(loop_Q_grad) ** 2, axis=1)),
            color='#ff7f0e', label='grad l2 norm')
# axGrad.plot(np.array(loop_Q_grad)[:, 0], color='#ff7f0e', label='grad l2 norm')
axGrad.tick_params(colors="#ff7f0e", axis='y')
axQ.set_xlabel('iteration')
fig.legend()

# fig.savefig('Q_opt.pdf')


# ## Other plots
Te_subbed, Te_jaxed, Te_arg_tuple, Te_arg_idx = sub_and_lambdify(
    1.0 * Te, values_to_opt, symbol_list_names)
Ti_subbed, Ti_jaxed, Ti_arg_tuple, Ti_arg_idx = sub_and_lambdify(
    1.0 * Ti, values_to_opt, symbol_list_names)

Te_plot = [Te_jaxed(*tuple(loop_values[i][Te_arg_idx])) for i in tqdm(range(0, 5001, 25))]
Ti_plot = [Ti_jaxed(*tuple(loop_values[i][Ti_arg_idx])) for i in tqdm(range(0, 5001, 25))]

figT, axTi = plt.subplots()

axTi.plot(np.arange(0, 5001, 25), Ti_plot, color='blue', label='<Ei>')
axTi.tick_params(colors='blue', axis='y')
axTe = axTi.twinx()
axTe.plot(np.arange(0, 5001, 25), Te_plot, color='red', label='Te')
axTe.tick_params(colors='red', axis='y')
axTi.set_xlabel('iteration')
figT.legend()

# figT.savefig('temp_change.pdf')

print("symbol \t value_i \t value_f \t RMS grad")
for i in range(len(symbol_list)):
    print(symbol_list[i].name + ":\t {:.4e}".format(loop_values[0][i]) + ":\t {:.4e}".format(
        loop_values[-1][i]), "\t {:.4e}".format(np.sqrt(np.sum(np.array(loop_Q_grad[0:-1]) ** 2, axis=0))[i]))


# # Optimizing simple tandem model
values_to_opt = values.copy()


# Substitute in temperature and reactivities functions into the sympy equation and then lambdify

# 60*60 converts MWh, $300 per MWh. Cost = electric power revenue per year - magnet cost averaged over 30 years
tandem_net_cost = ((P_recirculating + P_thermal_electric - P_electric_in) *
                   (60 * 60) * 300 * 8760 / 1e6) - (cost_B_cc + cost_B_p + cost_B_pm) / 30

tandem_func_sympy, tandem_function, tandem_arg_tuple, tandem_arg_idx = sub_and_lambdify(
    tandem_net_cost, values_to_opt)


# Build gradient function (only need to do this once!)
jnp.arange(len(tandem_func_sympy.free_symbols))

tandem_symbol_list = list(tandem_func_sympy.free_symbols)
tandem_grad_func = jax.jit(jax.grad(tandem_function, jnp.arange(len(tandem_symbol_list))))

tandem_symbol_list


# Optimize for:
# * B_pm
# * B_p
# * r_b
# * L_p
# * T_frac
# * Eb
# * L_cc
# * B_cc
tandem_opt_symbol_list = [B_pm, B_p, r_b, L_p, T_frac, Eb, L_cc, B_cc]
# opt_symbol_list = [T_frac]


# Create mask so that parameters we don't want to optimize remain constant
tandem_symbol_list_names = [sym.name for sym in tandem_symbol_list]
print(tandem_symbol_list)
opt_idx = [tandem_symbol_list.index(sym) for sym in tandem_opt_symbol_list]
opt_idx_mask = np.zeros(len(tandem_symbol_list))
opt_idx_mask[opt_idx] = 1
print(opt_idx_mask)


# ## Optimization loop
loop_values = []
loop_cost_grad = []
loop_values.append(jnp.array(tandem_arg_tuple))
loop_cost_grad.append(jnp.zeros(len(tandem_symbol_list)))

for i in tqdm(range(10000)):
    grad_cost = -jnp.array(tandem_grad_func(*tuple(loop_values[i]))) * opt_idx_mask * 3e-11
#     if i > 1000:
#         new_value = loop_values[-1] - (grad_cost + 0.5 * loop_cost_grad[-1])
#     else:
    new_value = loop_values[-1] - grad_cost
    loop_values.append(new_value)
    loop_cost_grad.append(grad_cost)


# ## Plot Q and gradient norm
fig, axQ = plt.subplots()
axQ.plot(np.arange(0, 10001, 50), [tandem_function(*tuple(_))
                                   for _ in tqdm(loop_values[::50])], label='Opt function', color="#1f77b4")
axQ.tick_params(colors="#1f77b4", axis='y')
# axQ.plot(tandem_function(*tuple([np.array(loop_values).transpose()[_, :] for _ in tqdm(range(8))])), label='Q')
axGrad = axQ.twinx()
axGrad.plot(np.arange(0, 10001, 1), np.sqrt(
    np.sum(np.array(loop_cost_grad) ** 2, axis=1)), color='#ff7f0e', label='grad l2 norm')
# axGrad.plot(np.array(loop_Q_grad)[:, 0], color='#ff7f0e', label='grad l2 norm')
# axGrad.tick_params(colors="#ff7f0e", axis='y')
axQ.set_xlabel('iteration')
fig.legend()
# plt.savefig('tandem_opt.pdf')

# fig.savefig('Q_opt.pdf')


# ## Other plots
num_plot = len(tandem_symbol_list)
fig, axes = plt.subplots(np.sum(opt_idx_mask).astype(int), figsize=(6, 10), sharex=True)
# print("symbol \t value_i \t value_f \t RMS grad")
j = 0
for i in range(num_plot):
    #     print(tandem_symbol_list[i].name + ":\t {:.4e}".format(loop_values[0][i]) + ":\t {:.4e}".format(loop_values[-1][i]), "\t {:.4e}".format(np.sqrt(np.sum(np.array(loop_cost_grad[0:-1]) ** 2, axis=0))[i]))
    if opt_idx_mask[i] == 0:
        continue
    else:
        axes[j].plot(np.array(loop_values)[:, i], label=tandem_symbol_list[i].name)
        axes[j].legend()
#         axes[j].set_title(tandem_symbol_list[i].name)
        j += 1
plt.legend()
plt.tight_layout()
# plt.savefig('tandem_opt_params.pdf')

last_iter = 6000
print("symbol \t value_i \t value_f \t RMS grad")
for i in range(len(tandem_symbol_list)):
    print(tandem_symbol_list[i].name + ":\t\t {:.4e}".format(loop_values[0][i]) + ":\t {:.4e}".format(loop_values[last_iter][i]),
          "\t {:.4e}".format(np.sqrt(np.sum(np.array(loop_cost_grad[0:last_iter]) ** 2, axis=0))[i]))
