import jax.numpy as jnp
import numpy as np

from jax import grad, jit, vmap, pmap

#JAX's low level API
from jax import lax

from jax import random
from jax import make_jaxpr
from jax import device_put

import matplotlib.pyplot as plt

def simple_plot():
    x = np.linspace(0,10, 1000)
    y = 2 * np.sin(x) * np.cos(x)
    plt.plot(x,y)
    plt.show()

def simple_plot_jax():
    x = jnp.linspace(0, 10, 1000)
    y = 2 * jnp.sin(x) * jnp.cos(x)
    plt.plot(x,y)
    plt.show()

def immutablity():
    "JAX arrays are immutable! (embrace the functional programming paradigm)"
    size = 10
    index = 0
    value = 23

    x = np.arange(size)
    print(x)
    x[index] = value
    print(x)

    x = jnp.arange(size)
    print(x)
    #x[index] = value JAX arrays are immutable and do not support in-place item assignment
    x = x.at[index].set(value)
    print(x)

if __name__ == "__main__":
    immutablity()