import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import elementwise_grad as egrad

# Specify p
p_spec1 = 50.0
p_spec2 = 100.0
p_spec3 = 150.0


# Data points
N_points = 100
theta = np.linspace(0, 2*np.pi, N_points)


# Newton-Raphson details
maxiter = 20
tol = 1e-11


# Benchmark yield function
from example.MatsuokaNakai.f_benchmark import *
get_dfdrho = egrad(f_benchmark, 1)


# NAM yield function
from example.MatsuokaNakai.f_NAM import *


# NAM-symbolic yield function
from example.MatsuokaNakai.f_symbolic import *


# -----------------------------------------------------------------
# Return mapping for benchmark yield function
rho1 = np.zeros_like(theta)
for i in range(np.shape(theta)[0]):

  x = p_spec1

  print(">> Point", i, "------------------------------------")

  for ii in range(maxiter):
    res = f_benchmark(p_spec1, x, theta[i])
    jac = get_dfdrho(p_spec1, x, theta[i])

    dx = -res / jac
    x = x + dx

    err = np.linalg.norm(dx)

    print(" Newton iter.",ii, ": err =", err)

    if err < tol:
      rho1[i] = x
      break

rho2 = np.zeros_like(theta)
for i in range(np.shape(theta)[0]):

  x = p_spec2

  print(">> Point", i, "------------------------------------")

  for ii in range(maxiter):
    res = f_benchmark(p_spec2, x, theta[i])
    jac = get_dfdrho(p_spec2, x, theta[i])

    dx = -res / jac
    x = x + dx

    err = np.linalg.norm(dx)

    print(" Newton iter.",ii, ": err =", err)

    if err < tol:
      rho2[i] = x
      break

rho3 = np.zeros_like(theta)
for i in range(np.shape(theta)[0]):

  x = p_spec3

  print(">> Point", i, "------------------------------------")

  for ii in range(maxiter):
    res = f_benchmark(p_spec3, x, theta[i])
    jac = get_dfdrho(p_spec3, x, theta[i])

    dx = -res / jac
    x = x + dx

    err = np.linalg.norm(dx)

    print(" Newton iter.",ii, ": err =", err)

    if err < tol:
      rho3[i] = x
      break
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# Return mapping for NAM yield function
rho_NAM1 = np.zeros_like(theta)
for i in range(np.shape(theta)[0]):

  x = p_spec1

  print(">> Point", i, "------------------------------------")

  for ii in range(maxiter):
    res = f_NAM(p_spec1, x, theta[i])
    jac = 1
    
    dx = -res / jac
    x = x + dx

    err = np.linalg.norm(dx)

    print(" Newton iter.",ii, ": err =", err)

    if err < tol or ii == maxiter-1:
      rho_NAM1[i] = x
      break

rho_NAM2 = np.zeros_like(theta)
for i in range(np.shape(theta)[0]):

  x = p_spec2

  print(">> Point", i, "------------------------------------")

  for ii in range(maxiter):
    res = f_NAM(p_spec2, x, theta[i])
    jac = 1
    
    dx = -res / jac
    x = x + dx

    err = np.linalg.norm(dx)

    print(" Newton iter.",ii, ": err =", err)

    if err < tol or ii == maxiter-1:
      rho_NAM2[i] = x
      break

rho_NAM3 = np.zeros_like(theta)
for i in range(np.shape(theta)[0]):

  x = p_spec3

  print(">> Point", i, "------------------------------------")

  for ii in range(maxiter):
    res = f_NAM(p_spec3, x, theta[i])
    jac = 1
    
    dx = -res / jac
    x = x + dx

    err = np.linalg.norm(dx)

    print(" Newton iter.",ii, ": err =", err)

    if err < tol or ii == maxiter-1:
      rho_NAM3[i] = x
      break
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# Return mapping for NAM-symbolic yield function
rho_symb1 = np.zeros_like(theta)
for i in range(np.shape(theta)[0]):

  x = p_spec1

  print(">> Point", i, "------------------------------------")

  for ii in range(maxiter):
    res = f_symbolic(p_spec1, x, theta[i])
    jac = 1
    
    dx = -res / jac
    x = x + dx

    err = np.linalg.norm(dx)

    print(" Newton iter.",ii, ": err =", err)

    if err < tol or ii == maxiter-1:
      rho_symb1[i] = x
      break

rho_symb2 = np.zeros_like(theta)
for i in range(np.shape(theta)[0]):

  x = p_spec2

  print(">> Point", i, "------------------------------------")

  for ii in range(maxiter):
    res = f_symbolic(p_spec2, x, theta[i])
    jac = 1
    
    dx = -res / jac
    x = x + dx

    err = np.linalg.norm(dx)

    print(" Newton iter.",ii, ": err =", err)

    if err < tol or ii == maxiter-1:
      rho_symb2[i] = x
      break

rho_symb3 = np.zeros_like(theta)
for i in range(np.shape(theta)[0]):

  x = p_spec3

  print(">> Point", i, "------------------------------------")

  for ii in range(maxiter):
    res = f_symbolic(p_spec3, x, theta[i])
    jac = 1
    
    dx = -res / jac
    x = x + dx

    err = np.linalg.norm(dx)

    print(" Newton iter.",ii, ": err =", err)

    if err < tol or ii == maxiter-1:
      rho_symb3[i] = x
      break
# -----------------------------------------------------------------


# Plot results
fig = plt.figure(0,figsize=(7,7))
ax = fig.add_subplot(111, projection='polar')
ax.plot(theta, rho1, 'k-', linewidth=2.0, label='Analytical')
ax.plot(theta, rho2, 'k-', linewidth=2.0)
ax.plot(theta, rho3, 'k-', linewidth=2.0)
ax.plot(theta, rho_NAM1, 'r--', linewidth=1.5, label='NAM')
ax.plot(theta, rho_NAM2, 'r--', linewidth=1.5)
ax.plot(theta, rho_NAM3, 'r--', linewidth=1.5)
ax.plot(theta, rho_symb1, 'b:', linewidth=1.0, label='Symbolic regression')
ax.plot(theta, rho_symb2, 'b:', linewidth=1.0)
ax.plot(theta, rho_symb3, 'b:', linewidth=1.0)
ax.legend()
plt.show()