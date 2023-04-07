import autograd.numpy as np

# Material properties
c1  = 7.47
c2  = -33.75
c3  = 843.0
c4  = 20.0
c5  = -5.24e9
c6  = 2.04e7
c7  = -2.23e11
c8  = 2.62e9
c9  = 1.02e7
c10 = 8.83e3
c11 = -8.67e8

# Benchmark
def f_benchmark(p, rho, theta, v):

    sigma1pp = rho*np.cos(theta)
    sigma2pp = rho*np.sin(theta)
    sigma3pp = np.sqrt(3)*p

    sigma_pp = np.array([sigma1pp, sigma2pp, sigma3pp])

    R = np.array([[ np.sqrt(2)/2, -np.sqrt(6)/6, np.sqrt(3)/3],
                    [            0,  np.sqrt(6)/3, np.sqrt(3)/3],
                    [-np.sqrt(2)/2, -np.sqrt(6)/6, np.sqrt(3)/3]])

    sigma = np.dot(R, sigma_pp)

    sigma1 = sigma[0]
    sigma2 = sigma[1]
    sigma3 = sigma[2]

    I1 = sigma1 + sigma2 + sigma3
    J2 = (1./6.)*((sigma1-sigma2)**2 + (sigma2-sigma3)**2 + (sigma3-sigma1)**2)
    
    sigma_h  = I1/3.
    sigma_vm = np.sqrt(3.*J2) 
    
    J3 = (sigma1 - sigma_h)*(sigma2 - sigma_h)*(sigma3 - sigma_h)

    L = (3.*np.sqrt(3)/2.)*(J3/(J2**1.5))

    f = (sigma_vm*(4.*sigma_h**2 + (2.*v - L + c1)*sigma_h + c2) - sigma_h*(c3*v**2 + c4))*(c5*v**2 + 2.*v + c6) \
        + c7*v**2 - sigma_vm*(L**2)*(c8*v**2 - c9) + c10 - c11

    return f