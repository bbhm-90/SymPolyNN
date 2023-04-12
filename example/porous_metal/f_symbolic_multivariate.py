import numpy as np
from src.symbolic.equation_assembler import SymbolicComponent

config0 = {
      'input_dims':[0, 1, 2, 3],
      'xscaler_path':"example/porous_metal/step_2_symbolic_regression_training/multi_var/results/tmpxf9jbwt_/xscaler.joblib",
      'yscaler_path':"example/porous_metal/step_2_symbolic_regression_training/multi_var/results/tmpxf9jbwt_/yscaler.joblib",
      'equation':"(sin((sin((x1 + cos(sin(exp(-0.78243595)))) / (0.35087702 / (0.35087702 + -0.22634321))) * x1) + ((((x3 + sin(x2)) / (((0.419321 / 0.12345698) + x3) / 0.419321)) + (x1 + (x0 + (sin(-1.5365558 + cos(cos(sin(-1.0325121)) * x2)) * 0.30654994)))) / cos(cos(sin(cos(0.90577376 + x0)))))) / sin(0.58552516))" # complexity = 56
}

model_symb_mv = SymbolicComponent(config0)

def f_symbolic_multivariate(p, rho, theta, v):
  
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
    f = model_symb_mv.predict(RT).item()

    return f