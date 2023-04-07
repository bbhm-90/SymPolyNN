import torch
import joblib
import numpy as np

model_NAM = torch.jit.load("example/porous_metal/step_1_nn_training/results/Bomarito/PolynomialHO/model.ptjit")
f_INPUT_scaler  = joblib.load("example/porous_metal/step_1_nn_training/results/Bomarito/PolynomialHO/xscaler.joblib")
f_OUTPUT_scaler = joblib.load("example/porous_metal/step_1_nn_training/results/Bomarito/PolynomialHO/yscaler.joblib")

def f_NAM(p, rho, theta, v):

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

    RT = np.array([sigma_h, sigma_vm, L, v]).reshape(1,4)
    RT = f_INPUT_scaler.transform(RT)
    RT = torch.tensor(RT, dtype=torch.float)
    
    f = model_NAM(RT)
    f = f[0]
    f_numpy = f.detach().numpy()[0,0]
    f_numpy = f_OUTPUT_scaler.inverse_transform(f_numpy.reshape(-1,1))

    return f_numpy[0,0]