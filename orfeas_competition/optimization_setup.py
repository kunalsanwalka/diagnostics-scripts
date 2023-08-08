import jax
import jax.numpy as jnp
import numpy as np
import sympy
from sympy.printing.numpy import NumPyPrinter
from tqdm import tqdm
from jax.config import config
from functools import partial
# We need 64 bit floats so that solving for Te (Egedal et al) remains stable for a wide range
config.update("jax_enable_x64", True)


# Use the numpy printer for jax for piecewise functions, stitched together from ComPWA and numpy docs/source:
# jax: https://compwa-org.readthedocs.io/report/001.html
# numpy: https://github.com/sympy/sympy/blob/master/sympy/printing/pycode.py#L206
class JaxPrinter(NumPyPrinter):
    _module = "jax"

    def _print_Piecewise(self, expr):
        result = []
        i = 0
        for arg in expr.args:
            e = arg.expr
            c = arg.cond
            if i == 0:
                result.append('(')
            result.append('(')
            result.append(self._print(e))
            result.append(')')
            result.append(' if ')
            result.append(self._print(c))
            result.append(' else ')
            i += 1
        result = result[:-1]
        if result[-1] == 'True':
            result = result[:-2]
            result.append(')')
        else:
            result.append(' else None)')
        return ''.join(result)


# ## plug_in_values
def sub_and_lambdify(equation, values_in, flags, symbol_list_names=None):
    eq_in = equation.copy()  # Copy so as not to modify the original function
    # Subsitute sympy function for jax code
    if flags['simple_temps']:
        custom_sub_dict = {}
        eq_in = eq_in.subs(DD_reac, f_DD_reac(Ti))
        eq_in = eq_in.subs(DT_reac, f_DT_reac(Ti))
    else:
        # Substitute Ti variable with a sympy function
        eq_in = eq_in.subs(Ti, f_Ti(B_pm, B_p, p_aux, Eb))
        eq_in = eq_in.subs(Te, f_Te(B_pm, B_p, p_aux, Eb) * Tep_fudge)
        eq_in = eq_in.subs(DD_reac, f_DD_reac(B_pm, B_p, p_aux, Eb))
        eq_in = eq_in.subs(DT_reac, f_DT_reac(B_pm, B_p, p_aux, Eb))
        custom_sub_dict = {'f_Ti': values_in['Ti'],
                           'f_Te': values_in['Te']}
    custom_sub_dict = {**custom_sub_dict,
                       'f_DD_reac': values_in['DD_reac'],
                       'f_DT_reac': values_in['DT_reac']}
    jax_equation = (sympy.lambdify(list(eq_in.free_symbols), eq_in, [jnp, custom_sub_dict, "jax"]))
    # If we want to quickly solve for other quantities given Ti and Te, we just remove the dictionary above that calls the functions
    #   for calculating Ti and Te (and the reactivities) and put the numbers into the values dictionary. Functions are substituted above
    #   so we can calculate gradients all the way through the Ti and Te calculations for JAX

    # Need to find position of arguments because it's not consistent run-to-run
    arg_tuple = tuple(values_in[_.name] for _ in list(eq_in.free_symbols))
    arg_order = [_.name for _ in list(eq_in.free_symbols)]

    # Return the indicies so we can easily pass in values to the jax function when we're evaluting from a larger set of values.
    #   Useful for calculating other quantities (e.g., temperatures) when optimizing for another one.
    if symbol_list_names is not None:
        arg_idx = jnp.array([symbol_list_names.index(arg_order[_]) for _ in range(len(arg_order))])
    else:
        arg_idx = None

    return eq_in, jax_equation, arg_tuple, arg_idx


def evaluate_lambdified_func(sympy_eq_in, jax_eq_in, arg_tuple):
    return jax_eq_in(*arg_tuple)


def plug_in_values(equation, values_in, flags):
    sympy_equation, jax_equation, arg_tuple, arg_idx = sub_and_lambdify(equation, values_in, flags)
    result = evaluate_lambdified_func(sympy_equation, jax_equation, arg_tuple)
    return result


# Optimize using gradient descent by computing the derivatives of sympy_function with respect to
#    opt_symbol_list (list of sympy symbols) and starting at initial_values (dictionary).
# Initial values must be a dictionary of arrays, where the each array is the number of samples.
#   Broadcasting is not done automatically, unfortunately.
class Optimizer():

    def __init__(self, sympy_func, opt_symbol_list, initial_values, flags):
        self.steps_executed = 0
        self.opt_symbol_list = opt_symbol_list
        self.initial_values = initial_values
        self.flags = flags
        # Substitute in temperature and reactivities functions into the sympy equation and then lambdify
        # func_subbed and _jax are the respective functions. arg_tuple is the inital_values in the
        #   correct order for subsitution into the jax equation, and arg_idx are the respective indices
        self.func_subbed, self.func_jax, self.arg_tuple, self.arg_idx = sub_and_lambdify(sympy_func, self.initial_values, self.flags)
        # Find the fundamental free parameters of the function to be optimized
        self.symbol_list = list(self.func_subbed.free_symbols)
        # Compute the gradient of the equation with respect to the free parameters, then map the jax
        #   function to allow multiple inputs at one time (enables SIMD)
        self.grad_func_jax = jax.jit(jax.vmap(jax.grad(self.func_jax, jnp.arange(len(self.symbol_list)))))
        # vmap the jax function (of the cost function) to allow multiple inputs
        self.func_jax_vmap = jax.jit(jax.vmap(self.func_jax))

        # Find the number of samples that will be optimized
        self.num_samples = jnp.array(self.arg_tuple).shape[1]  # has shape [num arguments, num samples]

        # Get the names of the symbols for use later
        self.symbol_list_names = [sym.name for sym in self.symbol_list]
        # Find the index of the free parameters in the sympy function that will be optimized.
        # The order of the arguments in the function is not guaranteed to be the same each time!
        self.opt_idx = [self.symbol_list.index(sym) for sym in self.opt_symbol_list]
        # Create the gradient index mask so that we only change the parameters we want to optimize
        #   and leave the remaining parameters unchanged.
        # Tile the mask so that it is applied to every sample ()
        self.opt_idx_mask = np.zeros(len(self.symbol_list))  # Needs to be numpy array so we can assign indices values
        self.opt_idx_mask[self.opt_idx] = 1
        self.opt_idx_mask = jnp.tile(self.opt_idx_mask[:, jnp.newaxis], [1, self.num_samples])

        # Create the arrays used to hold the values and gradients at each optimization step
        self.loop_values = []
        self.loop_gradients = []
        # Set the first values to the initial values. If you did jnp.array(loop_values), the indices
        #   would be [step number, num arguments, num samples].
        self.loop_values.append(jnp.array(self.arg_tuple))
        # Set the inital gradients to zero (important if using momentum-based gradient descent)
        # self.loop_gradients.append(jnp.zeros((len(self.symbol_list), self.num_samples)))
        # Compile functions by executing them on one sample
        print("Compiling JAX functions...")
        self.func_jax_vmap(*tuple((self.loop_values[0][:, :])))
        self.grad_func_jax(*tuple((self.loop_values[0][:, :])))

    # num_iterations is the number of optimization steps and momentum controls whether to use previous
    #    gradient information as well as the current gradient.
    def run(self, num_iterations=1000, momentum=False, step_size=1e-4):
        for i in tqdm(range(num_iterations)):
            grad = (jnp.array(self.grad_func_jax(*tuple(self.loop_values[self.steps_executed]))) *
                    self.opt_idx_mask *
                    step_size)
    #     if i > 1000:
    #     new_value = loop_values[-1] - (grad_Q + 0.99 * loop_gradients[-1])
    #     else:
            new_value = self.loop_values[-1] - grad
            self.loop_values.append(new_value)
            self.loop_gradients.append(grad)
            self.steps_executed += 1

        return self.loop_values, self.loop_gradients

    # Reset the optimizer but still retain the compiled functions so we can quickly run again
    def reset_runs(self):
        self.loop_values = []
        self.loop_values.append(jnp.array(self.arg_tuple))
        self.loop_gradients = []
        self.steps_executed = 0

    # Translate a tuple (used for jax functions) into a dict (used with the plug_in_values function)
    def tuple_to_dict(self, value_tuple, value_dict):
        for i in range(len(self.symbol_list)):
            value_dict[self.symbol_list[i].name] = value_tuple[i]
        return value_dict

    # Evaluate the cost function for a selected iteration range
    def evaluate(self, iter_start=-1, iter_end=None, iter_step=None):
        if iter_end is None:
            return self.func_jax_vmap(*tuple(self.loop_values[iter_start]))
        else:
            return [self.func_jax_vmap(*tuple(self.loop_values[i]))
                    for i in tqdm(range(iter_start, iter_end, iter_step))]

    # Evaluate a different sympy function using the values calculated
    def evaluate_sympy_func(self, sympy_func,
                            iter_start=-1, iter_end=None, iter_step=None):
        func_subbed, func_jax, arg_tuple, arg_idx = sub_and_lambdify(sympy_func, self.initial_values,
                                                                     self.flags, self.symbol_list_names)
        func_jax_vmap = jax.jit(jax.vmap(func_jax))
        if iter_end is None:
            return func_jax_vmap(*tuple(self.loop_values[iter_start][arg_idx]))
        else:
            return [func_jax_vmap(*tuple(self.loop_values[i][arg_idx]))
                    for i in tqdm(range(iter_start, iter_end, iter_step))]

    def get_loop_value_by_name(self, symbol_name):
        sym_index = [sym.name for sym in self.symbol_list].index(symbol_name)
        # Order is [optimization iteration, symbol index, number of samples]
        return jnp.array(self.loop_values)[:, sym_index, :]
