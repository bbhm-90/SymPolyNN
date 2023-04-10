import pandas as pd
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import elementwise_grad as egrad

# Specify p & v
# p_spec = 0.5
# v_spec = 0.0635 
# fn = "1"

# p_spec = 1.0
# v_spec = 0.065
# fn = "2"

# p_spec = 1.25
# v_spec = 0.063
# fn = "3"

# p_spec = 1.5
# v_spec = 0.064
# fn = "4"

# p_spec = 0.47368
# v_spec = 0.064111
# fn = "tr1"

# p_spec = 0.66315
# v_spec = 0.063666
# fn = "tr2"

# p_spec = 1.23157
# v_spec = 0.064555
# fn = "tr3"

p_spec = 1.42105
v_spec = 0.063888
fn = "tr4"


# Data points
N_points = 100
theta = np.linspace(0, 2*np.pi, N_points)


# Newton-Raphson details
maxiter = 10
tol = 1e-11


# Training data
data = pd.read_csv("data/augmented_data_Bomarito_66k_noisy_4.csv")

# >> stress point cloud at f = 0
p_tr     = data["p"].values
rho_tr   = data["rho"].values
theta_tr = data["theta"].values
v_tr     = data["v"].values
f_tr     = data["f"].values

mask1 = np.abs(p_tr - p_spec) < 0.001

p_tr = p_tr[mask1]
rho_tr = rho_tr[mask1]
theta_tr = theta_tr[mask1]
v_tr = v_tr[mask1]
f_tr = f_tr[mask1]

mask2 = np.abs(v_tr - v_spec) < 0.0001

p_tr = p_tr[mask2]
rho_tr = rho_tr[mask2]
theta_tr = theta_tr[mask2]
v_tr = v_tr[mask2]
f_tr = f_tr[mask2]

mask3 = np.abs(f_tr) < 0.00001

p_tr = p_tr[mask3]
rho_tr = rho_tr[mask3]
theta_tr = theta_tr[mask3]
v_tr = v_tr[mask3]
f_tr = f_tr[mask3]


# Benchmark yield function
from example.porous_metal.f_benchmark import *
get_dfdrho = egrad(f_benchmark, 1)


# NAM yield function
from example.porous_metal.f_NAM import *


# QNM yield function
from example.porous_metal.f_QNM import *


# QNM-symbolic yield function
from example.porous_metal.f_symbolic import *


# -----------------------------------------------------------------
# Return mapping for benchmark yield function
rho = np.zeros_like(theta)
for i in range(np.shape(theta)[0]):

    x = p_spec

    print(">> Point", i, "------------------------------------")

    for ii in range(maxiter):
        res = f_benchmark(p_spec, x, theta[i], v_spec)
        jac = get_dfdrho(p_spec, x, theta[i], v_spec)

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

    x = p_spec

    print(">> Point", i, "------------------------------------")

    for ii in range(maxiter):
        res = f_NAM(p_spec, x, theta[i], v_spec)
        jac = 1
        
        dx = -res / jac
        x = x + dx

        err = np.linalg.norm(dx)

        print(" Newton iter.",ii, ": err =", err)

        if err < tol or ii == maxiter-1:
            rho_NAM[i] = x
            break
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# Return mapping for QNM yield function
rho_QNM = np.zeros_like(theta)
for i in range(np.shape(theta)[0]):

    x = p_spec

    print(">> Point", i, "------------------------------------")

    for ii in range(maxiter):
        res = f_QNM(p_spec, x, theta[i], v_spec)
        jac = 1
        
        dx = -res / jac
        x = x + dx

        err = np.linalg.norm(dx)

        print(" Newton iter.",ii, ": err =", err)

        if err < tol or ii == maxiter-1:
            rho_QNM[i] = x
            break
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# Return mapping for QNM-symbolic yield function
rho_symb = np.zeros_like(theta)
for i in range(np.shape(theta)[0]):

    x = p_spec

    print(">> Point", i, "------------------------------------")

    for ii in range(maxiter):
        res = f_symbolic(p_spec, x, theta[i], v_spec)
        jac = 1
        
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

if fn[0]== "t":
    ax.plot(theta_tr, rho_tr, 'ro', label='training data')
    ax.plot(theta, rho_NAM, 'r-', label='NAM')
    ax.plot(theta, rho_QNM, 'b:', label='QNM')
    ax.plot(theta, rho, 'k-', label='Analytical solution')
    ax.plot(theta, rho_symb, 'g--', label='Symbolic regression (without sparsity)')
else:
    ax.plot(theta, rho, 'k-', label='Analytical solution')
    ax.plot(theta, rho_NAM, 'r-', markersize=4, label='NAM')
    ax.plot(theta, rho_QNM, 'b:', label='QNM')
    ax.plot(theta, rho_symb, 'g--', label='Symbolic regression')

ax.legend()
ax.set_ylim(0,1)
plt.title("p="+str(p_spec)+"; v="+str(v_spec))
plt.show()