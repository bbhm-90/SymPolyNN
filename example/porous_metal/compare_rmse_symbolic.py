import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import elementwise_grad as egrad

# Set number of inputs & range
Ninput = 500

# >> p: (1/3)*tr(sigma) [MPa]
min_p = 0
max_p = 1.8 

# >> theta: Lode's angle [rad]
min_theta = 0
max_theta = 2*np.pi

# >> v: void fraction
min_v = 0.063
max_v = 0.065


# Generate random inputs
p     = (max_p - min_p)*np.random.rand(Ninput) + min_p
theta = (max_theta - min_theta)*np.random.rand(Ninput) + min_theta
v     = (max_v - min_v)*np.random.rand(Ninput) + min_v


# Solve for rho that satisfies f = 0
from example.porous_metal.f_benchmark import *
get_dfdrho = egrad(f_benchmark, 1)

rho = np.zeros_like(p)

for i in range(np.shape(p)[0]):

    x = 1/p[i]

    print(">> Point", i, "------------------------------------")

    for ii in range(10):
        res = f_benchmark(p[i], x, theta[i], v[i])
        jac = get_dfdrho(p[i], x, theta[i], v[i])

        dx = -res / jac
        x = x + dx

        err = np.linalg.norm(dx)

        print(" Newton iter.",ii, ": err =", err)

        if err < 1e-11:
            rho[i] = x
            break


# Compute RMSE
from example.porous_metal.f_symbolic import *
from example.porous_metal.f_symbolic_multivariate import *

f_ground_truth = np.zeros_like(p)
f_symb_out = np.zeros_like(p)
f_symb_mv_out = np.zeros_like(p)

for i in range(np.shape(p)[0]):
    f_ground_truth[i] = f_benchmark(p[i], rho[i], theta[i], v[i])
    f_symb_out[i] = f_symbolic(p[i], rho[i], theta[i], v[i])
    f_symb_mv_out[i] = f_symbolic_multivariate(p[i], rho[i], theta[i], v[i])

RMSE_symb = np.sqrt(np.sum((f_symb_out-f_ground_truth)**2) / np.shape(p)[0])
RMSE_symb_mv = np.sqrt(np.sum((f_symb_mv_out-f_ground_truth)**2) / np.shape(p)[0])

print(" ")
print("=======================================================================")
print("RMSE (QNM-based symbolic regression):", RMSE_symb)
print("RMSE (Direct multivariate symbolic regression):", RMSE_symb_mv)
print("=======================================================================")