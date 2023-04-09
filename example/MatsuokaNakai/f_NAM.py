import torch
import joblib
import numpy as np

model_NAM = torch.jit.load("example/MatsuokaNakai/step_1_nn_training/results_20/cylindrical/PolynomialHO/PosEnc_0/model.ptjit")
f_INPUT_scaler  = joblib.load("example/MatsuokaNakai/step_1_nn_training/results_20/cylindrical/PolynomialHO/PosEnc_0/xscaler.joblib")
f_OUTPUT_scaler = joblib.load("example/MatsuokaNakai/step_1_nn_training/results_20/cylindrical/PolynomialHO/PosEnc_0/yscaler.joblib")

def f_NAM(p, rho, theta, lamda):

    RT = np.array([p, rho, theta]).reshape(1,3)
    RT = f_INPUT_scaler.transform(RT)
    RT = torch.tensor(RT, dtype=torch.float)
    
    f = model_NAM(RT)
    f = f[0]
    f_numpy = f.detach().numpy()[0,0]
    f_numpy = f_OUTPUT_scaler.inverse_transform(f_numpy.reshape(-1,1))

    return f_numpy[0,0]