import numpy as np

# Material properties
phi = np.radians(30)

# Benchmark
def f_benchmark(p, rho, theta, lamda):

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
    I2 = sigma1*sigma2 + sigma2*sigma3 + sigma3*sigma1
    I3 = sigma1*sigma2*sigma3

    beta = (9 - np.sin(phi)**2)/(1 - np.sin(phi)**2)

    f = -(I1*I2)**(1/3) + (beta*I3)**(1/3)

    return f