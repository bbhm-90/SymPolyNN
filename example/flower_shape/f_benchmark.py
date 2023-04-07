import numpy as np

# Material properties
E       = 200e3 # Young's modulus
sigma_y = 250.0 # Initial yield stress [MPa]
k       = 3.0   # no. of petals
amp     = 0.325 # amplitude

# Benchmark
def f_benchmark(p, rho, theta, lamda):

    phi = np.sqrt(3/2)*rho*(1 + amp*np.sin(k*theta))

    return phi - sigma_y