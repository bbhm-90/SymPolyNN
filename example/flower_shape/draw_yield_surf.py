import autograd.numpy as np
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt

# Data points
N_points = 100
theta = np.linspace(0, 2*np.pi, N_points)


# Newton-Raphson details
maxiter = 10
tol = 1e-11


# Benchmark yield function
from example.flower_shape.f_benchmark import *
get_dfdrho = egrad(f_benchmark, 1)


# NAM yield function
from example.flower_shape.f_NAM import *


# NAM-symbolic yield function
from example.flower_shape.f_symbolic import *


# -----------------------------------------------------------------
# Return mapping for benchmark yield function
rho = np.zeros_like(theta)

for i in range(np.shape(theta)[0]):

    x = 300.0

    print(">> Point", i, "------------------------------------")

    for ii in range(maxiter):
        res = f_benchmark(0.0, x, theta[i], 0.0)
        jac = get_dfdrho(0.0, x, theta[i], 0.0)

        dx = -res / jac
        x = x + dx

        err = np.linalg.norm(dx)

        print(" Newton iter.",ii, ": err =", err)

        if err < tol:
            rho[i] = x
            break
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# Return mapping for NAM yield function
rho_NAM = np.zeros_like(theta)

for i in range(np.shape(theta)[0]):

    x = 300.0

    print(">> Point", i, "------------------------------------")

    for ii in range(maxiter):
        res = f_NAM(0.0, x, theta[i], 0.0)
        jac = 1 # just used constant
        
        dx = -res / jac
        x = x + dx

        err = np.linalg.norm(dx)

        print(" Newton iter.",ii, ": err =", err)

        if err < tol or ii == maxiter-1:
          rho_NAM[i] = x
          break
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# Return mapping for NAM-symbolic yield function
rho_symb = np.zeros_like(theta)

for i in range(np.shape(theta)[0]):

    x = 300.0

    print(">> Point", i, "------------------------------------")

    for ii in range(maxiter):
        res = f_symbolic(0.0, x, theta[i], 0.0)
        jac = 1 # just used constant
        
        dx = -res / jac
        x = x + dx

        err = np.linalg.norm(dx)

        print(" Newton iter.",ii, ": err =", err)

        if err < tol or ii == maxiter-1:
          rho_symb[i] = x
          break
# -----------------------------------------------------------------


# Plot results
fig = plt.figure(0,figsize=(7,7))
ax = fig.add_subplot(111, projection='polar')
ax.plot(theta, rho, 'k-', linewidth=2.0, label='Analytical')
ax.plot(theta, rho_NAM, 'r--', linewidth=1.5, label='NAM')
ax.plot(theta, rho_symb, 'b:', linewidth=1.0, label='Symbolic regression')
ax.legend()
plt.show()