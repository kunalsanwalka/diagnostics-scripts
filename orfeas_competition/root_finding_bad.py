def find_roots_vectorized_nojax(R_in, power_aux_in, max_iter=5, max_error = 1e-6):
    
    x0 = jnp.ones(R_in.shape[0]) * 1e-4
    x1 = jnp.ones(R_in.shape[0]) * 1e2
    curr_iter = 0
    
    for i in tqdm(range(max_iter)):
        curr_iter = curr_iter + 1
        x_mid = (x0 + x1) / 2.0
        F_mid = alpha_eq(x_mid, R_in, power_aux_in)

        F0 = alpha_eq(x0, R_in, power_aux_in)
        F1 = alpha_eq(x1, R_in, power_aux_in)

        x0 = np.where(np.sign(F_mid) == np.sign(F0), x_mid, x0)
        x1 = np.where(np.sign(F_mid) == np.sign(F1), x_mid, x1)


    return alpha_eq((x0 + x1) / 2.0, R_in, power_aux_in)


# func: function to find the roots for
# max_iter: maximum number of iterations
# x0, x1: left and right initial guesses
# max_error: maximum error tolerated before stopping
# @jax.jit
@jax.profiler.annotate_function
def find_roots_vectorized(R_in, power_aux_in, max_iter=2, max_error = 1e-6):
    @jax.profiler.annotate_function
    def cond(vals):
        curr_iter, x0, x1 = vals
        error = jnp.amax(jnp.abs(x1 - x0))
        return (curr_iter < max_iter)
    @jax.profiler.annotate_function
    def body(i, vals):
        iters, x0, x1 = vals
        curr_iter = iters + 1
        x_mid = (x0 + x1) / 2.0
        
        F_mid = alpha_eq(x_mid, R_in, power_aux_in)
    
#         mask0 = 
        
        F0 = alpha_eq(x0, R_in, power_aux_in)
        F1 = alpha_eq(x1, R_in, power_aux_in)
        
        x0 = jnp.where(jnp.sign(F_mid) == jnp.sign(F0), x_mid, x0)
        x1 = jnp.where(jnp.sign(F_mid) == jnp.sign(F1), x_mid, x1)
        return curr_iter, x0, x1
    
    x0_init = jnp.ones(R_in.shape[0]) * 1e-4
    x1_init = jnp.ones(R_in.shape[0]) * 1e2
    init_vals = (0, x0_init, x1_init)
#     iters, x0, x1 = jax.lax.while_loop(cond, body, init_vals)
    
    iters, x0, x1 = jax.lax.fori_loop(0, max_iter, body, init_vals)
#     print(jnp.amin(jnp.abs(x1 - x0)))
    return alpha_eq((x0 + x1) / 2.0, R_in, power_aux_in)