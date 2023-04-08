import autograd.numpy as np
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt

# INPUT --------------------------------------------------------
# Material properties
E       = 200e3 # Young's modulus
nu      = 0.3   # Poisson ratio

# Loading steps & increment
Nstep   = 275    # loading steps
eps_inc = 1.0e-4 # strain increment

deps = np.array([[eps_inc, 0.0,         0.0],
                 [0.0,     -nu*eps_inc, 0.0],
                 [0.0,     0.0,         -nu*eps_inc]])

# Newton-Raphson parameters
tol     = 1e-9
maxiter = 10

# Option
optn = "benchmark"
# --------------------------------------------------------------


# Load model
if optn == "benchmark":
    from example.flower_shape.f_benchmark import *
elif optn == "MLP":
    import sys
    sys.path.append('example/flower_shape/MLP')
    from example.flower_shape.f_MLP import *
elif optn == "symbolic":
    from example.flower_shape.f_symbolic import *
 

# Define yield function
def f_yield(sigma1, sigma2, sigma3, optn):

    sigma = np.array([sigma1, sigma2, sigma3])

    Rinv = np.array([[ np.sqrt(2)/2,             0, -np.sqrt(2)/2],
                     [-np.sqrt(6)/6,  np.sqrt(6)/3, -np.sqrt(6)/6],
                     [ np.sqrt(3)/3,  np.sqrt(3)/3,  np.sqrt(3)/3]])

    sigma_pp = np.dot(Rinv, sigma)

    sigma1_pp = sigma_pp[0]
    sigma2_pp = sigma_pp[1]
    sigma3_pp = sigma_pp[2]

    rho   = np.sqrt(sigma1_pp**2 + sigma2_pp**2)
    theta = np.arctan2(sigma2_pp, sigma1_pp)
    p     = (1/np.sqrt(3))*sigma3_pp

    if theta < 0:
        theta = theta + 2*np.pi

    if optn == "benchmark":
        return f_benchmark(p, rho, theta)
    elif optn == "MLP":
        return f_MLP(p, rho, theta)
    elif optn == "symbolic":
        return f_symbolic(p, rho, theta)


# Define first gradient
def df_yield(sigma1, sigma2, sigma3, optn):

    dist = 1e-3
    f = f_yield(sigma1, sigma2, sigma3, optn)
    f_s1dist = f_yield(sigma1+dist, sigma2, sigma3, optn)
    f_s2dist = f_yield(sigma1, sigma2+dist, sigma3, optn)
    f_s3dist = f_yield(sigma1, sigma2, sigma3+dist, optn)

    dfdsig1 = (f_s1dist - f) / dist
    dfdsig2 = (f_s2dist - f) / dist
    dfdsig3 = (f_s3dist - f) / dist

    return dfdsig1, dfdsig2, dfdsig3


# Define second gradient
def df2_yield(sigma1, sigma2, sigma3, optn):

    dist = 1e-3
    dfdsig1, dfdsig2, dfdsig3 = df_yield(sigma1, sigma2, sigma3, optn)
    dfdsig1_s1dist, dfdsig2_s1dist, dfdsig3_s1dist = df_yield(sigma1+dist, sigma2, sigma3, optn)
    dfdsig1_s2dist, dfdsig2_s2dist, dfdsig3_s2dist = df_yield(sigma1, sigma2+dist, sigma3, optn)
    dfdsig1_s3dist, dfdsig2_s3dist, dfdsig3_s3dist = df_yield(sigma1, sigma2, sigma3+dist, optn)

    d2fdsig1dsig1 = (dfdsig1_s1dist - dfdsig1) / dist
    d2fdsig2dsig2 = (dfdsig2_s2dist - dfdsig2) / dist
    d2fdsig3dsig3 = (dfdsig3_s3dist - dfdsig3) / dist

    d2fdsig1dsig2 = (dfdsig1_s2dist - dfdsig1) / dist
    d2fdsig2dsig3 = (dfdsig2_s3dist - dfdsig2) / dist
    d2fdsig3dsig1 = (dfdsig3_s1dist - dfdsig3) / dist

    return d2fdsig1dsig1, d2fdsig2dsig2, d2fdsig3dsig3, d2fdsig1dsig2, d2fdsig2dsig3, d2fdsig3dsig1


# Define elasticity model
# >> elasticity tensor
K  = E / (3*(1-2*nu))
mu = E / (2*(1+nu))

a = K + (4/3)*mu
b = K - (2/3)*mu

Ce_principal = np.array([[a, b, b],
                         [b, a, b],
                         [b, b, a]])

# >> derivatives
dsig1depse1 = a
dsig1depse2 = b
dsig1depse3 = b
dsig2depse1 = b
dsig2depse2 = a
dsig2depse3 = b
dsig3depse1 = b
dsig3depse2 = b
dsig3depse3 = a


# Define necessary variables
I = np.eye(3) # identity tensor

sigma = np.zeros((3,3)) # stress

eps_e = np.zeros((3,3)) # elastic strain 
eps_p = np.zeros((3,3)) # plastic strain
eps   = eps_e + eps_p   # strain

lamda = 0 # plastic multiplier


# Perform material point simulation 
print(":: Stress-point simulation ::")

sigma11_out = np.zeros(Nstep+1)
eps11_out   = np.zeros(Nstep+1)

for i in range(Nstep):

    print("Loading step [",i+1,"] ---------------------------------------")

    if i == 70:
        deps = -deps
    elif i == 95:
        deps = -deps
    elif i == 175:
        deps = -deps
    elif i == 200:
        deps = -deps

    # [1] Compute trial strain
    eps_e_tr = eps_e + deps

    eps_e_tr_principal_mag, eps_e_tr_principal_vec = np.linalg.eig(eps_e_tr)

    eps_e_tr1 = eps_e_tr_principal_mag[0]
    eps_e_tr2 = eps_e_tr_principal_mag[1]
    eps_e_tr3 = eps_e_tr_principal_mag[2]

    n1 = eps_e_tr_principal_vec[:,0]
    n2 = eps_e_tr_principal_vec[:,1]
    n3 = eps_e_tr_principal_vec[:,2]

    # [2] Compute trial stress
    sigma_tr_principal_mag = np.inner(Ce_principal, eps_e_tr_principal_mag)

    sigma_tr1 = sigma_tr_principal_mag[0]
    sigma_tr2 = sigma_tr_principal_mag[1]
    sigma_tr3 = sigma_tr_principal_mag[2]

    sigma_tr = sigma_tr1*np.tensordot(n1,n1,axes=0) + sigma_tr2*np.tensordot(n2,n2,axes=0) + sigma_tr3*np.tensordot(n3,n3,axes=0)

    # [3] Check yielding
    f = f_yield(sigma_tr1, sigma_tr2, sigma_tr3, optn)

    # [3.1] If f <= 0, elastic.
    if f <= 0:
        print(">> Elastic!")

        # Update stress & strain
        sigma = sigma_tr

        eps_e = eps_e_tr
        eps   = eps_e + eps_p

    # [3.2] If f > 0, plastic.
    else:
        print(">> Plastic!")

        # Initialize variables
        eps_e_principal_mag, eps_e_principal_vec = np.linalg.eig(eps_e)

        eps_e1 = eps_e_principal_mag[0]
        eps_e2 = eps_e_principal_mag[1]
        eps_e3 = eps_e_principal_mag[2]
        dlamda  = 0

        x = np.zeros(4) # target variables
        x[0] = eps_e1
        x[1] = eps_e2
        x[2] = eps_e3
        x[3] = dlamda

        # Newton-Raphson iteration (return mapping)
        for ii in range(maxiter):

            # Initialize residual and jacobian
            res = np.zeros(4)
            jac = np.zeros((4,4))

            # Current strain
            eps_e1_current = x[0]
            eps_e2_current = x[1]
            eps_e3_current = x[2]

            # Current stress
            sigma1_current = a*eps_e1_current + b*eps_e2_current + b*eps_e3_current
            sigma2_current = b*eps_e1_current + a*eps_e2_current + b*eps_e3_current
            sigma3_current = b*eps_e1_current + b*eps_e2_current + a*eps_e3_current

            sigma1_current = sigma1_current 
            sigma2_current = sigma2_current 
            sigma3_current = sigma3_current 

            # Current lamda
            lamda_current = lamda + x[3]

            # Update derivatives
            # >> First order derivatives
            dfdsig1, dfdsig2, dfdsig3 \
                = df_yield(sigma1_current, sigma2_current, sigma3_current, optn)

            # >> Second order derivatives
            d2fdsig1dsig1, d2fdsig2dsig2, d2fdsig3dsig3, d2fdsig1dsig2, d2fdsig2dsig3, d2fdsig3dsig1 \
                = df2_yield(sigma1_current, sigma2_current, sigma3_current, optn)

            # Update residual
            res[0] = x[0] - eps_e_tr1 + x[3]*dfdsig1
            res[1] = x[1] - eps_e_tr2 + x[3]*dfdsig2
            res[2] = x[2] - eps_e_tr3 + x[3]*dfdsig3
            res[3] = f_yield(sigma1_current, sigma2_current, sigma3_current, optn)

            # Update Jacobian ***
            jac[0,0] = 1 + x[3]*(d2fdsig1dsig1*dsig1depse1 + d2fdsig1dsig2*dsig2depse1 + d2fdsig3dsig1*dsig3depse1)
            jac[0,1] =     x[3]*(d2fdsig1dsig1*dsig1depse2 + d2fdsig1dsig2*dsig2depse2 + d2fdsig3dsig1*dsig3depse2)
            jac[0,2] =     x[3]*(d2fdsig1dsig1*dsig1depse3 + d2fdsig1dsig2*dsig2depse3 + d2fdsig3dsig1*dsig3depse3)
            jac[0,3] = dfdsig1

            jac[1,0] =     x[3]*(d2fdsig1dsig2*dsig1depse1 + d2fdsig2dsig2*dsig2depse1 + d2fdsig2dsig3*dsig3depse1)
            jac[1,1] = 1 + x[3]*(d2fdsig1dsig2*dsig1depse2 + d2fdsig2dsig2*dsig2depse2 + d2fdsig2dsig3*dsig3depse2)
            jac[1,2] =     x[3]*(d2fdsig1dsig2*dsig1depse3 + d2fdsig2dsig2*dsig2depse3 + d2fdsig2dsig3*dsig3depse3)
            jac[1,3] = dfdsig2

            jac[2,0] =     x[3]*(d2fdsig3dsig1*dsig1depse1 + d2fdsig2dsig3*dsig2depse1 + d2fdsig3dsig3*dsig3depse1)
            jac[2,1] =     x[3]*(d2fdsig3dsig1*dsig1depse2 + d2fdsig2dsig3*dsig2depse2 + d2fdsig3dsig3*dsig3depse2)
            jac[2,2] = 1 + x[3]*(d2fdsig3dsig1*dsig1depse3 + d2fdsig2dsig3*dsig2depse3 + d2fdsig3dsig3*dsig3depse3)
            jac[2,3] = dfdsig3

            jac[3,0] = dfdsig1*dsig1depse1 + dfdsig2*dsig2depse1 + dfdsig3*dsig3depse1
            jac[3,1] = dfdsig1*dsig1depse2 + dfdsig2*dsig2depse2 + dfdsig3*dsig3depse2
            jac[3,2] = dfdsig1*dsig1depse3 + dfdsig2*dsig2depse3 + dfdsig3*dsig3depse3
            jac[3,3] = 0

            # Solve system of equations
            dx = np.linalg.solve(jac, -res) # increment of target variables

            # Update x
            x = x + dx

            # Compute error
            err = np.linalg.norm(dx)

            print(" Newton iter.",ii, ": err =", err)

            if err < tol:
                break
        
        # Update strain
        eps   = eps + deps
        eps_e = x[0]*np.tensordot(n1,n1,axes=0) + x[1]*np.tensordot(n2,n2,axes=0) + x[2]*np.tensordot(n3,n3,axes=0)
        eps_p = eps - eps_e
        lamda = lamda + x[3]

        # Update stress
        sigma1 = a*x[0] + b*x[1] + b*x[2] 
        sigma2 = b*x[0] + a*x[1] + b*x[2] 
        sigma3 = b*x[0] + b*x[1] + a*x[2] 
        sigma  = sigma1*np.tensordot(n1,n1,axes=0) + sigma2*np.tensordot(n2,n2,axes=0) + sigma3*np.tensordot(n3,n3,axes=0)


    # [4] Record stress and strain
    sigma11_out[i+1] = sigma[0,0]
    eps11_out[i+1]   = eps[0,0]


# Plot stress-strain curve
plt.plot(eps11_out, sigma11_out, 'k-', linewidth=1.0, label=optn)
plt.xlabel(r'$\epsilon_{11}$', fontsize=15)
plt.ylabel(r'$\sigma_{11}$ [MPa]', fontsize=15)
plt.axhline(0, color = 'k',alpha = 0.5)
plt.axvline(0, color = 'k',alpha = 0.5)
plt.legend(loc="upper left")
plt.show()