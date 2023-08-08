# 0D mirror (simple + tandem) optimization
## File overview
### Python scripts
* `mirror_physics_setup.py`: sets up the sympy graph for computing all values from Cary's spreadsheet.
* `mirror_physics_setup-gyroBohm.py`: same, but assumes gyroBohm transport (this should be controlled with a flag instead of a separate file!)
* `optimization_setup.py`: define the class for optimizing based on some sympy expression. Also includes a bunch of helper functions such as evaluating any quantity using parameters from an optimization step.

### Notebooks
* `optimize-simple-mirror.ipynb`: optimize a simple mirror (interactively) using `mirror_physics_setup.py` and `optimization_setup.py`.
* `optimize-tandem-mirroy.ipynb`: optimize a tandem mirror (interactively) using `mirror_physics_setup-gyroBohm.py` and `optimization_setup.py`
* `reactor-optimization_phil.ipynb`: old notebook for building out/prototyping the physics and optimization code. No longer update
* `reactor-optimization_kunal`: Kunal's notebook (rather old and no longer updated)

## Getting started

`optimize-simple-mirror.ipynb` is a good place to start. We first define the fundamental values used for the optimization in the `values` dictionary. We then modify this dictionary with a jax (numpy-like) array that contains the range of values that will be the starting point for the optimization. Every value needed for the given cost function needs to have the same shape (this requirement will hopefully be relaxed in the future). We then create the `Optimizer` class using a defined cost function. We then optimize using `some_optimizer_instance.run(...)`. 

## Performing scans over a parameter range
Yet to be implemented -- I'll be doing this hopefully later today. 

## Implementation overview
The equations are implemented in [sympy](https://www.sympy.org/en/index.html): a symbolic math / computer algebra system for python. This module allows us to write equations analytically for evaluation later. It will also write out a quantity in LaTeX as a function of the fundamental quantities ('symbols') that are defined at the very beginning. I thought this would be a nice thing to have, but many interesting quantities — like total fusion power — end up looking like a total mess when all the symbols are substituted in (it may be worth gutting sympy from this and going for a pure jax implementation).

This sympy code/graph can be translated to python code ('lambdify') using your numerical module of choice. In this case we use [jax](https://jax.readthedocs.io/en/latest/), which is a high-performance array computing library for python that creates, compiles (to XLA), and executes a computation graph. This graph allows us to compute gradients with respect to the input symbols very quickly which is required for optimization of multivariable equations.
